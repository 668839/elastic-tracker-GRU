#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models.py

GRU网络架构定义，实现公式推导文档v3.1中的：
1. Bi-GRU编码器 (公式21-23)
2. 自回归GRU解码器 (公式24)
3. Cholesky输出层 (公式25-30)
4. 变分Dropout (RNN专用，同序列同mask)

参考：
- Gal & Ghahramani (2016): Dropout as Bayesian Approximation
- Gal & Ghahramani (2016): Theoretically Grounded Application of Dropout in RNNs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math


class LockedDropout(nn.Module):
    """
    变分Dropout：同一序列的所有时间步使用相同的mask
    
    实现公式A2: h_t = f(x_t ⊙ d_x, h_{t-1} ⊙ d_h)
    其中 d_x, d_h 在整个序列中保持不变
    """
    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch, seq_len, features) 或 (batch, features)
        """
        if not self.training or self.p == 0:
            return x
        
        if x.dim() == 3:
            # (batch, seq_len, features) -> mask shape (batch, 1, features)
            mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p)
        else:
            # (batch, features) -> mask shape (batch, features)
            mask = x.new_empty(x.size(0), x.size(1)).bernoulli_(1 - self.p)
        
        mask = mask / (1 - self.p)  # 缩放以保持期望值不变
        return x * mask


class BiGRUEncoder(nn.Module):
    """
    双向GRU编码器
    
    实现公式21-23:
    - 前向: h_k^→ = GRU^→(s_k ⊙ d_x, h_{k-1}^→ ⊙ d_h)
    - 后向: h_k^← = GRU^←(s_k ⊙ d_x, h_{k+1}^← ⊙ d_h)
    - 上下文: c = [h_t^→; h_{t-T_obs+1}^←]
    """
    def __init__(self, 
                 input_dim: int = 6,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout
        
        # 输入Dropout
        self.input_dropout = LockedDropout(dropout)
        
        # 双向GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 隐藏状态Dropout (层间)
        self.hidden_dropout = LockedDropout(dropout)
        
    def forward(self, x: torch.Tensor, 
                apply_dropout: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入序列, shape (batch, seq_len, input_dim)
            apply_dropout: 是否应用dropout (推理时可关闭)
            
        Returns:
            context: 上下文向量, shape (batch, 2 * hidden_dim)
            outputs: 所有时间步的输出, shape (batch, seq_len, 2 * hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # 输入Dropout
        if apply_dropout:
            x = self.input_dropout(x)
        
        # 双向GRU
        outputs, hidden = self.gru(x)
        # outputs: (batch, seq_len, 2 * hidden_dim)
        # hidden: (2 * num_layers, batch, hidden_dim)
        
        # 构造上下文向量 (公式23)
        # 前向最后一个时间步: outputs[:, -1, :hidden_dim]
        # 后向第一个时间步: outputs[:, 0, hidden_dim:]
        forward_final = outputs[:, -1, :self.hidden_dim]
        backward_final = outputs[:, 0, self.hidden_dim:]
        context = torch.cat([forward_final, backward_final], dim=1)
        # context: (batch, 2 * hidden_dim)
        
        return context, outputs


class AutoregressiveDecoder(nn.Module):
    """
    自回归GRU解码器
    
    实现公式24:
    h_k^dec = GRU^dec([x̂_{k-1}; c], h_{k-1}^dec)
    
    输出层 (公式25-30):
    - 位置均值: x̂_k = W_μ h_k^dec + b_μ
    - Cholesky因子: [z1, z2, z3] = W_L h_k^dec + b_L
    """
    def __init__(self,
                 context_dim: int = 512,  # 2 * encoder_hidden_dim
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 output_dim: int = 2,
                 dropout: float = 0.2,
                 output_uncertainty: bool = True):
        super().__init__()
        
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout_p = dropout
        self.output_uncertainty = output_uncertainty
        
        # 解码器输入: 上一步预测位置 + 上下文
        decoder_input_dim = output_dim + context_dim
        
        # 隐藏状态初始化层 (公式24a)
        self.init_hidden = nn.Linear(context_dim, hidden_dim * num_layers)
        
        # 输入Dropout
        self.input_dropout = LockedDropout(dropout)
        
        # GRU解码器
        self.gru = nn.GRU(
            input_size=decoder_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 位置输出层 (公式25)
        self.pos_output = nn.Linear(hidden_dim, output_dim)
        
        # Cholesky因子输出层 (公式26)
        if output_uncertainty:
            self.chol_output = nn.Linear(hidden_dim, 3)  # l11, l21, l22
        
        # 数值稳定性常数
        self.eps = 1e-6
        
    def init_decoder_hidden(self, context: torch.Tensor) -> torch.Tensor:
        """
        初始化解码器隐藏状态 (公式24a)
        h_0^dec = tanh(W_init c + b_init)
        """
        batch_size = context.size(0)
        hidden = torch.tanh(self.init_hidden(context))
        # Reshape to (num_layers, batch, hidden_dim)
        hidden = hidden.view(batch_size, self.num_layers, self.hidden_dim)
        hidden = hidden.permute(1, 0, 2).contiguous()
        return hidden
    
    def compute_cholesky(self, z: torch.Tensor) -> torch.Tensor:
        """
        从网络输出构造Cholesky下三角矩阵 (公式27-29)
        
        L = [[l11, 0  ],
             [l21, l22]]
        
        其中 l11 = softplus(z1) + ε, l21 = z2, l22 = softplus(z3) + ε
        """
        l11 = F.softplus(z[:, 0]) + self.eps
        l21 = z[:, 1]
        l22 = F.softplus(z[:, 2]) + self.eps
        
        # 构造矩阵
        batch_size = z.size(0)
        L = torch.zeros(batch_size, 2, 2, device=z.device, dtype=z.dtype)
        L[:, 0, 0] = l11
        L[:, 1, 0] = l21
        L[:, 1, 1] = l22
        
        return L
    
    def forward(self, 
                context: torch.Tensor,
                pred_len: int,
                target_seq: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 0.0,
                apply_dropout: bool = True,
                reference_trajectory: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        自回归解码
        
        Args:
            context: 编码器输出的上下文, shape (batch, context_dim)
            pred_len: 预测序列长度
            target_seq: 真实目标序列 (用于teacher forcing), shape (batch, pred_len, 2)
            teacher_forcing_ratio: teacher forcing比例
            apply_dropout: 是否应用dropout
            reference_trajectory: MC采样时使用的参考轨迹, shape (batch, pred_len, 2)
            
        Returns:
            predictions: 预测位置序列, shape (batch, pred_len, 2)
            cholesky_factors: Cholesky因子序列 (如果output_uncertainty), shape (batch, pred_len, 2, 2)
        """
        batch_size = context.size(0)
        device = context.device
        
        # 初始化隐藏状态
        hidden = self.init_decoder_hidden(context)
        
        # 初始输入: 原点 [0, 0] (航向对齐坐标系下当前位置)
        current_pos = torch.zeros(batch_size, self.output_dim, device=device)
        
        predictions = []
        cholesky_factors = [] if self.output_uncertainty else None
        
        for t in range(pred_len):
            # 构造解码器输入: [上一步位置; 上下文]
            decoder_input = torch.cat([current_pos, context], dim=1)
            decoder_input = decoder_input.unsqueeze(1)  # (batch, 1, input_dim)
            
            # 应用Dropout
            if apply_dropout:
                decoder_input = self.input_dropout(decoder_input)
            
            # GRU前向传播
            output, hidden = self.gru(decoder_input, hidden)
            output = output.squeeze(1)  # (batch, hidden_dim)
            
            # 预测位置 (公式25)
            pred_pos = self.pos_output(output)  # (batch, 2)
            predictions.append(pred_pos)
            
            # Cholesky因子 (公式26-29)
            if self.output_uncertainty:
                chol_params = self.chol_output(output)  # (batch, 3)
                L = self.compute_cholesky(chol_params)  # (batch, 2, 2)
                cholesky_factors.append(L)
            
            # 决定下一步的输入
            if reference_trajectory is not None:
                # MC采样模式：使用参考轨迹
                current_pos = reference_trajectory[:, t, :]
            elif target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Teacher forcing: 使用真实值
                current_pos = target_seq[:, t, :]
            else:
                # 自回归: 使用预测值
                current_pos = pred_pos
        
        predictions = torch.stack(predictions, dim=1)  # (batch, pred_len, 2)
        
        if self.output_uncertainty:
            cholesky_factors = torch.stack(cholesky_factors, dim=1)  # (batch, pred_len, 2, 2)
        
        return predictions, cholesky_factors


class TrajectoryPredictor(nn.Module):
    """
    完整的轨迹预测模型
    
    组合编码器和解码器，支持：
    - Teacher模型 (大网络)
    - Student模型 (小网络)
    - MC Dropout推理
    """
    def __init__(self,
                 input_dim: int = 6,
                 encoder_hidden: int = 256,
                 encoder_layers: int = 3,
                 decoder_hidden: int = 256,
                 decoder_layers: int = 2,
                 output_dim: int = 2,
                 dropout: float = 0.2,
                 output_uncertainty: bool = True):
        super().__init__()
        
        self.encoder = BiGRUEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_hidden,
            num_layers=encoder_layers,
            dropout=dropout
        )
        
        context_dim = 2 * encoder_hidden  # 双向
        
        self.decoder = AutoregressiveDecoder(
            context_dim=context_dim,
            hidden_dim=decoder_hidden,
            num_layers=decoder_layers,
            output_dim=output_dim,
            dropout=dropout,
            output_uncertainty=output_uncertainty
        )
        
        self.output_uncertainty = output_uncertainty
        
    def forward(self,
                obs_seq: torch.Tensor,
                pred_len: int,
                target_seq: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 0.0) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            obs_seq: 观测序列, shape (batch, obs_len, input_dim)
            pred_len: 预测长度
            target_seq: 目标序列 (训练时使用)
            teacher_forcing_ratio: teacher forcing比例
            
        Returns:
            predictions: shape (batch, pred_len, 2)
            cholesky_factors: shape (batch, pred_len, 2, 2) 或 None
        """
        # 编码
        context, _ = self.encoder(obs_seq, apply_dropout=True)
        
        # 解码
        predictions, cholesky_factors = self.decoder(
            context, pred_len, target_seq, teacher_forcing_ratio, apply_dropout=True
        )
        
        return predictions, cholesky_factors
    
    def predict_deterministic(self, obs_seq: torch.Tensor, pred_len: int) -> torch.Tensor:
        """
        确定性预测 (关闭Dropout)
        """
        self.eval()
        with torch.no_grad():
            context, _ = self.encoder(obs_seq, apply_dropout=False)
            predictions, _ = self.decoder(
                context, pred_len, apply_dropout=False
            )
        return predictions
    
    def predict_with_uncertainty(self,
                                 obs_seq: torch.Tensor,
                                 pred_len: int,
                                 num_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MC Dropout推理，估计不确定性
        
        实现共享参考轨迹策略 (文档3.2.3和3.4.1)
        
        Args:
            obs_seq: shape (batch, obs_len, input_dim)
            pred_len: 预测长度
            num_samples: MC采样次数
            
        Returns:
            mean_pred: 均值预测, shape (batch, pred_len, 2)
            epistemic_cov: 认知不确定性, shape (batch, pred_len, 2, 2)
            aleatoric_cov: 偶然不确定性, shape (batch, pred_len, 2, 2)
        """
        batch_size = obs_seq.size(0)
        device = obs_seq.device
        
        # Phase 1: 生成参考轨迹 (无Dropout)
        self.eval()
        with torch.no_grad():
            context_ref, _ = self.encoder(obs_seq, apply_dropout=False)
            ref_trajectory, _ = self.decoder(
                context_ref, pred_len, apply_dropout=False
            )
        
        # Phase 2: MC采样 (有Dropout)
        self.train()  # 开启Dropout
        
        all_predictions = []
        all_cholesky = []
        
        for _ in range(num_samples):
            with torch.no_grad():
                # 编码器使用Dropout
                context, _ = self.encoder(obs_seq, apply_dropout=True)
                
                # 解码器使用Dropout，但自回归输入使用参考轨迹
                predictions, cholesky = self.decoder(
                    context, pred_len, 
                    apply_dropout=True,
                    reference_trajectory=ref_trajectory  # 关键：使用参考轨迹
                )
                
                all_predictions.append(predictions)
                if cholesky is not None:
                    all_cholesky.append(cholesky)
        
        self.eval()
        
        # 堆叠采样结果
        all_predictions = torch.stack(all_predictions, dim=0)  # (M, batch, pred_len, 2)
        
        # 计算均值 (公式44)
        mean_pred = all_predictions.mean(dim=0)  # (batch, pred_len, 2)
        
        # 计算认知不确定性 (公式45)
        # Σ_epi = (1/M) Σ (x - μ)(x - μ)^T
        diff = all_predictions - mean_pred.unsqueeze(0)  # (M, batch, pred_len, 2)
        epistemic_cov = torch.einsum('mbti,mbtj->btij', diff, diff) / num_samples
        # epistemic_cov: (batch, pred_len, 2, 2)
        
        # 计算偶然不确定性 (公式42-43)
        if all_cholesky:
            all_cholesky = torch.stack(all_cholesky, dim=0)  # (M, batch, pred_len, 2, 2)
            # Σ_ale = L @ L^T
            aleatoric_samples = torch.matmul(all_cholesky, all_cholesky.transpose(-1, -2))
            aleatoric_cov = aleatoric_samples.mean(dim=0)  # (batch, pred_len, 2, 2)
        else:
            aleatoric_cov = torch.zeros_like(epistemic_cov)
        
        # 使用参考轨迹作为最终预测
        mean_pred = ref_trajectory
        
        return mean_pred, epistemic_cov, aleatoric_cov


def create_teacher_model(dropout: float = 0.2, output_uncertainty: bool = False) -> TrajectoryPredictor:
    """
    创建Teacher模型
    
    配置 (表3.2.6):
    - 编码器: 3层Bi-GRU, 256隐藏单元
    - 解码器: 2层GRU, 256隐藏单元
    - 参数量: ~2.1M
    """
    return TrajectoryPredictor(
        input_dim=6,
        encoder_hidden=256,
        encoder_layers=3,
        decoder_hidden=256,
        decoder_layers=2,
        output_dim=2,
        dropout=dropout,
        output_uncertainty=output_uncertainty
    )


def create_student_model(dropout: float = 0.2, output_uncertainty: bool = True) -> TrajectoryPredictor:
    """
    创建Student模型
    
    配置 (表3.2.6):
    - 编码器: 2层Bi-GRU, 64隐藏单元
    - 解码器: 1层GRU, 64隐藏单元
    - 参数量: ~0.4M
    """
    return TrajectoryPredictor(
        input_dim=6,
        encoder_hidden=64,
        encoder_layers=2,
        decoder_hidden=64,
        decoder_layers=1,
        output_dim=2,
        dropout=dropout,
        output_uncertainty=output_uncertainty
    )


def count_parameters(model: nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 测试模型
    print("Testing Teacher Model:")
    teacher = create_teacher_model()
    print(f"  Parameters: {count_parameters(teacher):,}")
    
    # 测试前向传播
    batch_size, obs_len, pred_len = 32, 30, 60
    x = torch.randn(batch_size, obs_len, 6)
    target = torch.randn(batch_size, pred_len, 2)
    
    pred, chol = teacher(x, pred_len, target, teacher_forcing_ratio=0.5)
    print(f"  Input shape: {x.shape}")
    print(f"  Prediction shape: {pred.shape}")
    print(f"  Cholesky shape: {chol.shape if chol is not None else 'None'}")
    
    print("\nTesting Student Model:")
    student = create_student_model()
    print(f"  Parameters: {count_parameters(student):,}")
    
    pred, chol = student(x, pred_len, target, teacher_forcing_ratio=0.5)
    print(f"  Prediction shape: {pred.shape}")
    print(f"  Cholesky shape: {chol.shape if chol is not None else 'None'}")
    
    print("\nTesting MC Dropout Inference:")
    mean_pred, epi_cov, ale_cov = student.predict_with_uncertainty(x, pred_len, num_samples=10)
    print(f"  Mean prediction shape: {mean_pred.shape}")
    print(f"  Epistemic covariance shape: {epi_cov.shape}")
    print(f"  Aleatoric covariance shape: {ale_cov.shape}")
