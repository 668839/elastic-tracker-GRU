#!/usr/bin/env python3
"""
Student GRU模型（轻量化）
通过知识蒸馏从Teacher学习
"""

import torch
import torch.nn as nn

class StudentGRU(nn.Module):
    """轻量化学生模型"""
    
    def __init__(self, 
                 input_dim=4,
                 hidden_dim=64,       # 1/4 of Teacher
                 num_layers=2,        # 更浅
                 output_dim=2,
                 pred_horizon=60,
                 dropout=0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.pred_horizon = pred_horizon
        self.dropout_p = dropout
        
        # 编码器：2层GRU
        self.encoder = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # MC Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 解码器（更简单）
        self.decoder_gru = nn.GRU(
            output_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # 输出层（轻量化）
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        x: (batch, obs_len=30, input_dim=4)
        返回: (batch, pred_len=60, output_dim=2)
        """
        # 编码
        enc_out, hidden = self.encoder(x)
        hidden = self.dropout(hidden)
        
        # 自回归解码
        decoder_input = x[:, -1, :2].unsqueeze(1)
        predictions = []
        
        decoder_hidden = hidden[-1:, :, :]  # 取最后1层
        
        for t in range(self.pred_horizon):
            dec_out, decoder_hidden = self.decoder_gru(decoder_input, decoder_hidden)
            dec_out = self.dropout(dec_out)
            pred_t = self.fc_out(dec_out)
            predictions.append(pred_t)
            decoder_input = pred_t
        
        predictions = torch.cat(predictions, dim=1)
        return predictions
    
    def enable_dropout(self):
        """强制启用Dropout用于MC采样"""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
    
    def predict_with_uncertainty(self, x, n_samples=3):
        """MC Dropout不确定性估计"""
        self.eval()
        self.enable_dropout()
        
        with torch.no_grad():
            samples = []
            for _ in range(n_samples):
                pred = self.forward(x)
                samples.append(pred.cpu().numpy())
            
            import numpy as np
            samples = np.array(samples)
            mean_pred = samples.mean(axis=0)
            std_pred = samples.std(axis=0)
            uncertainty = std_pred.mean(axis=-1)
        
        return mean_pred, uncertainty

# 测试
if __name__ == '__main__':
    model = StudentGRU()
    x = torch.randn(4, 30, 4)
    pred = model(x)
    print(f"Student GRU - Input: {x.shape}, Output: {pred.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试MC Dropout
    mean, uncert = model.predict_with_uncertainty(x, n_samples=3)
    print(f"Mean shape: {mean.shape}, Uncertainty: {uncert.shape}")