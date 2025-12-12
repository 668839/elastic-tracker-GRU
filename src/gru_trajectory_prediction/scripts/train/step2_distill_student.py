#!/usr/bin/env python3
"""
步骤2:知识蒸馏 - 从Teacher训练Student
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.teacher_gru import TeacherGRU
from models.student_gru import StudentGRU
from models.dataset import TrajectoryDataset
import argparse
from tqdm import tqdm

class DistillationLoss(nn.Module):
    """蒸馏损失"""
    
    def __init__(self, alpha=0.7, temperature=3.0):
        """
        alpha: Student损失权重
        temperature: 蒸馏温度
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.mse_loss = nn.MSELoss()
        
    def forward(self, student_pred, teacher_pred, ground_truth):
        """
        计算蒸馏损失 = α*hard_loss + (1-α)*soft_loss
        
        hard_loss: Student vs Ground Truth
        soft_loss: Student vs Teacher (温度软化)
        """
        # Hard loss（学生直接学习真实标签）
        hard_loss = self.mse_loss(student_pred, ground_truth)
        
        # Soft loss（学生模仿教师的输出分布）
        # 对于回归任务，可以用MSE + temperature缩放
        soft_loss = self.mse_loss(
            student_pred / self.temperature,
            teacher_pred.detach() / self.temperature
        )
        
        # 加权组合
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        return total_loss, hard_loss, soft_loss

def train_epoch(student, teacher, loader, optimizer, criterion, device):
    student.train()
    teacher.eval()
    
    total_loss = 0
    total_hard = 0
    total_soft = 0
    
    with torch.no_grad():
        teacher.eval()  # 确保Teacher不更新
    
    for obs, future, _ in tqdm(loader, desc='Distilling'):
        obs, future = obs.to(device), future.to(device)
        
        # Teacher预测（不需要梯度）
        with torch.no_grad():
            teacher_pred = teacher(obs)
        
        # Student预测
        student_pred = student(obs)
        
        # 蒸馏损失
        loss, hard_loss, soft_loss = criterion(student_pred, teacher_pred, future)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_hard += hard_loss.item()
        total_soft += soft_loss.item()
    
    return {
        'total': total_loss / len(loader),
        'hard': total_hard / len(loader),
        'soft': total_soft / len(loader)
    }

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    ade_errors = []
    fde_errors = []
    
    with torch.no_grad():
        for obs, future, _ in tqdm(loader, desc='Validating'):
            obs, future = obs.to(device), future.to(device)
            pred = model(obs)
            
            loss = criterion(pred, future)
            total_loss += loss.item()
            
            displacement = torch.norm(pred - future, dim=2)
            ade = displacement.mean(dim=1).cpu().numpy()
            fde = displacement[:, -1].cpu().numpy()
            
            ade_errors.extend(ade)
            fde_errors.extend(fde)
    
    import numpy as np
    return {
        'loss': total_loss / len(loader),
        'ade': np.mean(ade_errors),
        'fde': np.mean(fde_errors)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./processed_data')
    parser.add_argument('--teacher_path', required=True, help='Path to trained teacher model')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=0.7, help='Hard loss weight')
    parser.add_argument('--temperature', type=float, default=3.0)
    parser.add_argument('--save_dir', default='./checkpoints/student')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载Teacher
    print("Loading Teacher model...")
    teacher = TeacherGRU().to(device)
    checkpoint = torch.load(args.teacher_path, map_location=device)
    teacher.load_state_dict(checkpoint['model_state_dict'])
    teacher.eval()
    print(f"Teacher ADE: {checkpoint['ade']:.4f}")
    
    # 数据集
    full_dataset = TrajectoryDataset(args.data_dir)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4)
    
    # Student模型
    student = StudentGRU().to(device)
    print(f"Student parameters: {sum(p.numel() for p in student.parameters()):,}")
    print(f"Compression ratio: {sum(p.numel() for p in teacher.parameters()) / sum(p.numel() for p in student.parameters()):.2f}x")
    
    optimizer = optim.Adam(student.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    distill_criterion = DistillationLoss(alpha=args.alpha, temperature=args.temperature)
    mse_criterion = nn.MSELoss()
    
    # 训练
    best_ade = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        losses = train_epoch(student, teacher, train_loader, 
                            optimizer, distill_criterion, device)
        val_metrics = validate(student, val_loader, mse_criterion, device)
        scheduler.step(val_metrics['ade'])
        
        print(f"Train - Total: {losses['total']:.4f}, "
              f"Hard: {losses['hard']:.4f}, Soft: {losses['soft']:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, "
              f"ADE: {val_metrics['ade']:.4f}, FDE: {val_metrics['fde']:.4f}")
        
        # 保存最佳模型
        if val_metrics['ade'] < best_ade:
            best_ade = val_metrics['ade']
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ade': best_ade
            }, f"{args.save_dir}/best_student.pth")
            print(f"✓ Saved best student (ADE: {best_ade:.4f})")
    
    print(f"\nStudent training complete! Best ADE: {best_ade:.4f}")
    print(f"Teacher ADE: {checkpoint['ade']:.4f}")
    print(f"Performance gap: {best_ade - checkpoint['ade']:.4f}")

if __name__ == '__main__':
    main()