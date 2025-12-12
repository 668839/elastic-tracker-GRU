#!/usr/bin/env python3
"""
步骤1:训练大容量Teacher模型
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.teacher_gru import TeacherGRU
from models.dataset import TrajectoryDataset
import argparse
from tqdm import tqdm
import json

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for obs, future, _ in tqdm(loader, desc='Training'):
        obs, future = obs.to(device), future.to(device)
        
        pred = model(obs)
        loss = criterion(pred, future)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

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
            
            # 计算ADE/FDE
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
    parser.add_argument('--data_dir', default='./processed_data', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', default='./checkpoints/teacher')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    # 模型
    model = TeacherGRU().to(device)
    print(f"Teacher parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    criterion = nn.MSELoss()
    
    # 训练
    best_ade = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step(val_metrics['ade'])
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, "
              f"ADE: {val_metrics['ade']:.4f}, FDE: {val_metrics['fde']:.4f}")
        
        # 保存最佳模型
        if val_metrics['ade'] < best_ade:
            best_ade = val_metrics['ade']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ade': best_ade
            }, f"{args.save_dir}/best_teacher.pth")
            print(f"✓ Saved best model (ADE: {best_ade:.4f})")
    
    print(f"\nTeacher training complete! Best ADE: {best_ade:.4f}")

if __name__ == '__main__':
    main()