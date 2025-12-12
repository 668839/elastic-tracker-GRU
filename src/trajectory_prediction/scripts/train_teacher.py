#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_teacher_v2.py

改进版Teacher模型训练脚本

特点：
1. 增量预测（Delta预测）- 学习"如何移动"而非"位置在哪"
2. 可选模型大小（tiny/small/medium/large）
3. 完整的防过拟合机制（数据增强、dropout、weight_decay、早停）
4. 生成teacher_errors.npz用于知识蒸馏
5. 定期保存checkpoint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
from datetime import datetime
from tqdm import tqdm
import json
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional


# ============================================================
# 模型定义
# ============================================================

class GRUEncoder(nn.Module):
    """GRU编码器（可选双向）"""
    
    def __init__(self, 
                 input_dim: int = 6,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 bidirectional: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.dropout(x)
        outputs, hidden = self.gru(x)
        
        if self.bidirectional:
            forward_final = outputs[:, -1, :self.hidden_dim]
            backward_final = outputs[:, 0, self.hidden_dim:]
            context = torch.cat([forward_final, backward_final], dim=1)
        else:
            context = outputs[:, -1, :]
        
        return context, outputs


class DeltaDecoder(nn.Module):
    """
    增量预测解码器
    预测位移增量 Δx 而非绝对位置
    x_t = x_{t-1} + Δx_t
    """
    
    def __init__(self,
                 context_dim: int = 256,
                 hidden_dim: int = 128,
                 num_layers: int = 1,
                 output_dim: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        decoder_input_dim = output_dim + context_dim
        
        self.init_hidden = nn.Sequential(
            nn.Linear(context_dim, hidden_dim * num_layers),
            nn.Tanh()
        )
        
        self.gru = nn.GRU(
            input_size=decoder_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.delta_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                context: torch.Tensor,
                pred_len: int,
                last_velocity: torch.Tensor,
                target_seq: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 0.0) -> torch.Tensor:
        batch_size = context.size(0)
        device = context.device
        
        hidden = self.init_hidden(context)
        hidden = hidden.view(batch_size, self.num_layers, self.hidden_dim)
        hidden = hidden.permute(1, 0, 2).contiguous()
        
        dt = 1.0 / 30.0
        current_delta = last_velocity * dt
        current_pos = torch.zeros(batch_size, self.output_dim, device=device)
        
        if target_seq is not None:
            target_deltas = torch.zeros_like(target_seq)
            target_deltas[:, 0, :] = target_seq[:, 0, :]
            target_deltas[:, 1:, :] = target_seq[:, 1:, :] - target_seq[:, :-1, :]
        
        positions = []
        
        for t in range(pred_len):
            decoder_input = torch.cat([current_delta, context], dim=1)
            decoder_input = self.dropout(decoder_input)
            decoder_input = decoder_input.unsqueeze(1)
            
            output, hidden = self.gru(decoder_input, hidden)
            output = output.squeeze(1)
            
            pred_delta = self.delta_output(output)
            current_pos = current_pos + pred_delta
            positions.append(current_pos)
            
            if target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                current_delta = target_deltas[:, t, :]
            else:
                current_delta = pred_delta
        
        positions = torch.stack(positions, dim=1)
        return positions


class TrajectoryPredictorV2(nn.Module):
    """改进版轨迹预测模型"""
    
    def __init__(self,
                 input_dim: int = 6,
                 encoder_hidden: int = 128,
                 encoder_layers: int = 2,
                 decoder_hidden: int = 128,
                 decoder_layers: int = 1,
                 output_dim: int = 2,
                 dropout: float = 0.2,
                 bidirectional: bool = True):
        super().__init__()
        
        self.encoder = GRUEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_hidden,
            num_layers=encoder_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        context_dim = encoder_hidden * (2 if bidirectional else 1)
        
        self.decoder = DeltaDecoder(
            context_dim=context_dim,
            hidden_dim=decoder_hidden,
            num_layers=decoder_layers,
            output_dim=output_dim,
            dropout=dropout
        )
        
    def forward(self,
                obs_seq: torch.Tensor,
                pred_len: int,
                target_seq: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 0.0) -> Tuple[torch.Tensor, None]:
        context, _ = self.encoder(obs_seq)
        last_velocity = obs_seq[:, -1, 2:4]
        predictions = self.decoder(context, pred_len, last_velocity, target_seq, teacher_forcing_ratio)
        return predictions, None
    
    def predict_deterministic(self, obs_seq: torch.Tensor, pred_len: int) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            predictions, _ = self.forward(obs_seq, pred_len, target_seq=None, teacher_forcing_ratio=0)
        return predictions


def create_model_v2(size: str = 'medium', dropout: float = 0.2) -> TrajectoryPredictorV2:
    """创建不同大小的模型"""
    configs = {
        'tiny': {'encoder_hidden': 32, 'encoder_layers': 1, 'decoder_hidden': 32, 'decoder_layers': 1},
        'small': {'encoder_hidden': 64, 'encoder_layers': 1, 'decoder_hidden': 64, 'decoder_layers': 1},
        'medium': {'encoder_hidden': 128, 'encoder_layers': 2, 'decoder_hidden': 128, 'decoder_layers': 1},
        'large': {'encoder_hidden': 256, 'encoder_layers': 2, 'decoder_hidden': 256, 'decoder_layers': 2},
    }
    
    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")
    
    return TrajectoryPredictorV2(input_dim=6, output_dim=2, dropout=dropout, bidirectional=True, **configs[size])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# 训练工具函数
# ============================================================

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_ade(pred, target):
    return torch.norm(pred - target, p=2, dim=-1).mean(dim=-1)


def compute_fde(pred, target):
    return torch.norm(pred[:, -1, :] - target[:, -1, :], p=2, dim=-1)


def augment_trajectory(obs: torch.Tensor, pred: torch.Tensor,
                       noise_std: float = 0.03,
                       scale_range: tuple = (0.85, 1.15),
                       rotation: bool = True) -> tuple:
    """数据增强：缩放、旋转、噪声"""
    batch_size = obs.size(0)
    device = obs.device
    
    obs_aug = obs.clone()
    pred_aug = pred.clone()
    
    # 随机缩放
    scale = torch.FloatTensor(batch_size, 1, 1).uniform_(*scale_range).to(device)
    obs_aug[:, :, 0:2] *= scale
    obs_aug[:, :, 2:4] *= scale
    obs_aug[:, :, 4:6] *= scale
    pred_aug *= scale[:, :, :2]
    
    # 随机旋转
    if rotation:
        angles = torch.FloatTensor(batch_size).uniform_(-0.3, 0.3).to(device)
        cos_a = torch.cos(angles).view(-1, 1, 1)
        sin_a = torch.sin(angles).view(-1, 1, 1)
        
        # 旋转位置
        x, y = obs_aug[:, :, 0:1], obs_aug[:, :, 1:2]
        obs_aug[:, :, 0:1] = x * cos_a - y * sin_a
        obs_aug[:, :, 1:2] = x * sin_a + y * cos_a
        
        # 旋转速度
        vx, vy = obs_aug[:, :, 2:3], obs_aug[:, :, 3:4]
        obs_aug[:, :, 2:3] = vx * cos_a - vy * sin_a
        obs_aug[:, :, 3:4] = vx * sin_a + vy * cos_a
        
        # 旋转加速度
        ax, ay = obs_aug[:, :, 4:5], obs_aug[:, :, 5:6]
        obs_aug[:, :, 4:5] = ax * cos_a - ay * sin_a
        obs_aug[:, :, 5:6] = ax * sin_a + ay * cos_a
        
        # 旋转预测
        px, py = pred_aug[:, :, 0:1], pred_aug[:, :, 1:2]
        pred_aug[:, :, 0:1] = px * cos_a - py * sin_a
        pred_aug[:, :, 1:2] = px * sin_a + py * cos_a
    
    # 添加噪声
    obs_aug[:, :, 0:2] += torch.randn_like(obs_aug[:, :, 0:2]) * noise_std
    
    return obs_aug, pred_aug


def train_epoch(model, loader, optimizer, criterion, device, tf_ratio, pred_len, use_augmentation=False):
    model.train()
    loss_m, ade_m, fde_m = AverageMeter(), AverageMeter(), AverageMeter()
    
    pbar = tqdm(loader, desc='Training')
    for obs, target in pbar:
        obs, target = obs.to(device), target.to(device)
        
        if use_augmentation:
            obs, target = augment_trajectory(obs, target)
        
        optimizer.zero_grad()
        pred, _ = model(obs, pred_len, target, tf_ratio)
        loss = criterion(pred, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        with torch.no_grad():
            ade = compute_ade(pred, target).mean()
            fde = compute_fde(pred, target).mean()
        
        bs = obs.size(0)
        loss_m.update(loss.item(), bs)
        ade_m.update(ade.item(), bs)
        fde_m.update(fde.item(), bs)
        
        pbar.set_postfix(loss=f'{loss_m.avg:.4f}', ADE=f'{ade_m.avg:.4f}', FDE=f'{fde_m.avg:.4f}')
    
    return {'loss': loss_m.avg, 'ade': ade_m.avg, 'fde': fde_m.avg}


def evaluate(model, loader, criterion, device, pred_len):
    model.eval()
    loss_m, ade_m, fde_m = AverageMeter(), AverageMeter(), AverageMeter()
    
    with torch.no_grad():
        for obs, target in tqdm(loader, desc='Evaluating'):
            obs, target = obs.to(device), target.to(device)
            pred = model.predict_deterministic(obs, pred_len)
            
            loss = criterion(pred, target)
            ade = compute_ade(pred, target).mean()
            fde = compute_fde(pred, target).mean()
            
            bs = obs.size(0)
            loss_m.update(loss.item(), bs)
            ade_m.update(ade.item(), bs)
            fde_m.update(fde.item(), bs)
    
    return {'loss': loss_m.avg, 'ade': ade_m.avg, 'fde': fde_m.avg}


def compute_teacher_errors(model, loader, device, pred_len):
    """计算Teacher在每个样本、每个时间步的预测误差（用于知识蒸馏）"""
    model.eval()
    all_errors = []
    
    with torch.no_grad():
        for obs, target in tqdm(loader, desc='Computing teacher errors'):
            obs, target = obs.to(device), target.to(device)
            pred = model.predict_deterministic(obs, pred_len)
            
            # 每个时间步的平方误差
            error_sq = torch.sum((pred - target) ** 2, dim=-1)  # (batch, pred_len)
            all_errors.append(error_sq.cpu().numpy())
    
    return np.concatenate(all_errors, axis=0)


# ============================================================
# 主训练函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Train Teacher Model V2')
    parser.add_argument('--data_path', type=str, default='data/processed_data.npz')
    parser.add_argument('--save_dir', type=str, default='checkpoints/teacher')
    parser.add_argument('--version', type=str, default=None, help='Version name (default: timestamp)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing directory')
    parser.add_argument('--model_size', type=str, default='medium', choices=['tiny', 'small', 'medium', 'large'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--use_augmentation', action='store_true')
    parser.add_argument('--checkpoint_freq', type=int, default=20, help='Save checkpoint every N epochs')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 确定保存目录
    if args.overwrite:
        save_dir = args.save_dir
    elif args.version:
        save_dir = os.path.join(args.save_dir, args.version)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(args.save_dir, timestamp)
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to: {save_dir}")
    
    # 加载数据
    print(f"Loading data from {args.data_path}...")
    data = np.load(args.data_path)
    
    train_obs = torch.from_numpy(data['train_obs'])
    train_pred = torch.from_numpy(data['train_pred'])
    val_obs = torch.from_numpy(data['val_obs'])
    val_pred = torch.from_numpy(data['val_pred'])
    
    obs_len = int(data['obs_len'])
    pred_len = int(data['pred_len'])
    
    print(f"Train samples: {len(train_obs)}")
    print(f"Val samples: {len(val_obs)}")
    print(f"Observation length: {obs_len}")
    print(f"Prediction length: {pred_len}")
    
    train_loader = DataLoader(TensorDataset(train_obs, train_pred), 
                              batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(TensorDataset(val_obs, val_pred), 
                            batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 创建模型
    model = create_model_v2(size=args.model_size, dropout=args.dropout)
    model = model.to(device)
    
    num_params = count_parameters(model)
    ratio = len(train_obs) / num_params
    
    print(f"\nTeacher Model V2 ({args.model_size}):")
    print(f"  Parameters: {num_params:,}")
    print(f"  Samples/Parameters ratio: {ratio:.2f}")
    if ratio < 1:
        print(f"  ⚠️  High overfitting risk! Consider more data or smaller model.")
    
    # 训练设置
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    
    # 保存配置
    config = {
        'data_path': args.data_path,
        'model_size': args.model_size,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'use_augmentation': args.use_augmentation,
        'num_params': num_params,
        'train_samples': len(train_obs),
        'val_samples': len(val_obs),
        'obs_len': obs_len,
        'pred_len': pred_len,
        'start_time': datetime.now().isoformat()
    }
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 训练记录
    history = {
        'train_loss': [], 'train_ade': [], 'train_fde': [],
        'val_loss': [], 'val_ade': [], 'val_fde': [], 'lr': []
    }
    best_val_ade = float('inf')
    patience_counter = 0
    warmup_epochs = 5  # 前5个epoch不保存最佳模型（避免TF=1时的虚假最佳）
    
    print(f"\nStarting training...")
    print(f"Using augmentation: {args.use_augmentation}")
    print(f"Early stopping patience: {args.patience}")
    print(f"Checkpoint frequency: every {args.checkpoint_freq} epochs")
    
    for epoch in range(args.epochs):
        # Teacher forcing逐渐减少
        # 小模型需要更慢的TF衰减，大模型可以更快
        if args.model_size in ['tiny', 'small']:
            # 小模型：在90%训练时才完全关闭TF，且最低保持0.1
            tf_ratio = max(0.1, 1.0 - epoch / (args.epochs * 0.9))
        else:
            # 中/大模型：在70%训练时完全关闭TF
            tf_ratio = max(0.0, 1.0 - epoch / (args.epochs * 0.7))
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Teacher forcing ratio: {tf_ratio:.2f}")
        
        # 训练
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, 
                                    device, tf_ratio, pred_len, args.use_augmentation)
        
        # 验证
        val_metrics = evaluate(model, val_loader, criterion, device, pred_len)
        
        # 学习率调度
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录
        history['train_loss'].append(train_metrics['loss'])
        history['train_ade'].append(train_metrics['ade'])
        history['train_fde'].append(train_metrics['fde'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_ade'].append(val_metrics['ade'])
        history['val_fde'].append(val_metrics['fde'])
        history['lr'].append(current_lr)
        
        # 过拟合监控
        overfit_ratio = val_metrics['ade'] / train_metrics['ade'] if train_metrics['ade'] > 0 else float('inf')
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, ADE: {train_metrics['ade']:.4f}, FDE: {train_metrics['fde']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, ADE: {val_metrics['ade']:.4f}, FDE: {val_metrics['fde']:.4f}")
        print(f"LR: {current_lr:.6f}, Overfit ratio: {overfit_ratio:.2f}x")
        
        # 保存最佳模型（跳过warmup期，避免TF=1时的虚假最佳）
        if epoch >= warmup_epochs and val_metrics['ade'] < best_val_ade:
            best_val_ade = val_metrics['ade']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ade': val_metrics['ade'],
                'val_fde': val_metrics['fde'],
                'model_size': args.model_size,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"✓ Saved best model (ADE: {best_val_ade:.4f})")
        elif epoch >= warmup_epochs:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
        else:
            print(f"Warmup epoch {epoch+1}/{warmup_epochs}, not saving best yet")
        
        # 定期保存checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ade': val_metrics['ade'],
            }, checkpoint_path)
            print(f"✓ Saved checkpoint: checkpoint_epoch{epoch+1}.pth")
        
        # 早停
        if patience_counter >= args.patience:
            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
            break
    
    # 保存最终模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'model_size': args.model_size,
    }, os.path.join(save_dir, 'final_model.pth'))
    
    # 保存训练历史
    history['best_val_ade'] = best_val_ade
    history['end_time'] = datetime.now().isoformat()
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # ============================================================
    # 计算Teacher误差（用于知识蒸馏）
    # ============================================================
    print("\nComputing teacher errors on training set...")
    
    # 加载最佳模型
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 顺序加载训练数据（不shuffle）
    train_loader_ordered = DataLoader(TensorDataset(train_obs, train_pred), 
                                      batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    teacher_errors = compute_teacher_errors(model, train_loader_ordered, device, pred_len)
    
    # 统计信息
    min_error = teacher_errors.min()
    max_error = teacher_errors.max()
    eta = max_error - min_error if max_error > min_error else max_error
    mean_error = teacher_errors.mean()
    
    print(f"Teacher error statistics:")
    print(f"  Min error: {min_error:.6f}")
    print(f"  Max error: {max_error:.6f}")
    print(f"  η (normalization factor): {eta:.6f}")
    print(f"  Mean error: {mean_error:.6f}")
    
    # 保存teacher_errors.npz
    np.savez(os.path.join(save_dir, 'teacher_errors.npz'),
             errors=teacher_errors,
             min_error=min_error,
             max_error=max_error,
             eta=eta,
             mean_error=mean_error)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}")
    print(f"Best validation ADE: {best_val_ade:.4f}")
    print(f"Results saved to: {save_dir}")
    print(f"\nGenerated files:")
    print(f"  - best_model.pth")
    print(f"  - final_model.pth")
    print(f"  - teacher_errors.npz")
    print(f"  - config.json")
    print(f"  - history.json")
    print(f"  - checkpoint_epoch*.pth (every {args.checkpoint_freq} epochs)")


if __name__ == '__main__':
    main()
