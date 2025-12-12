#!/usr/bin/env python3
"""
步骤3：模型剪枝 - 移除不重要的权重
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from models.student_gru import StudentGRU
from models.dataset import TrajectoryDataset
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import copy

def compute_accuracy(model, loader, device):
    """评估模型准确度"""
    model.eval()
    ade_errors = []
    
    with torch.no_grad():
        for obs, future, _ in loader:
            obs, future = obs.to(device), future.to(device)
            pred = model(obs)
            displacement = torch.norm(pred - future, dim=2)
            ade = displacement.mean(dim=1).cpu().numpy()
            ade_errors.extend(ade)
    
    import numpy as np
    return np.mean(ade_errors)

def prune_model(model, amount=0.3):
    """
    结构化剪枝
    
    amount: 剪枝比例（0.3 = 移除30%的权重）
    """
    print(f"\nApplying {amount*100:.0f}% pruning...")
    
    # 对所有Linear层进行L1非结构化剪枝
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # 永久移除
            print(f"Pruned layer: {name}")
        
        # 对GRU层剪枝（更复杂）
        elif isinstance(module, nn.GRU):
            # 剪枝输入到隐藏层的权重
            prune.l1_unstructured(module, name='weight_ih_l0', amount=amount)
            prune.remove(module, 'weight_ih_l0')
            print(f"Pruned GRU layer: {name}")
    
    return model

def fine_tune(model, loader, optimizer, criterion, device, epochs=10):
    """剪枝后微调"""
    print("\nFine-tuning pruned model...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for obs, future, _ in tqdm(loader, desc=f'Fine-tune Epoch {epoch+1}'):
            obs, future = obs.to(device), future.to(device)
            
            pred = model(obs)
            loss = criterion(pred, future)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./processed_data')
    parser.add_argument('--student_path', required=True, help='Path to trained student')
    parser.add_argument('--prune_amount', type=float, default=0.3, help='Pruning ratio')
    parser.add_argument('--finetune_epochs', type=int, default=10)
    parser.add_argument('--save_dir', default='./checkpoints/pruned')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载Student模型
    print("Loading Student model...")
    model = StudentGRU().to(device)
    checkpoint = torch.load(args.student_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 评估原始模型
    dataset = TrajectoryDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    original_ade = compute_accuracy(model, loader, device)
    print(f"Original Student ADE: {original_ade:.4f}")
    
    # 剪枝
    pruned_model = prune_model(copy.deepcopy(model), amount=args.prune_amount)
    
    # 评估剪枝后性能
    pruned_ade = compute_accuracy(pruned_model, loader, device)
    print(f"Pruned Model ADE: {pruned_ade:.4f} (drop: {pruned_ade - original_ade:.4f})")
    
    # 微调
    optimizer = torch.optim.Adam(pruned_model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    
    finetuned_model = fine_tune(pruned_model, train_loader, optimizer, 
                               criterion, device, args.finetune_epochs)
    
    # 最终评估
    final_ade = compute_accuracy(finetuned_model, loader, device)
    print(f"\nFinal ADE after fine-tuning: {final_ade:.4f}")
    print(f"Performance recovery: {pruned_ade - final_ade:.4f}")
    
    # 保存
    torch.save({
        'model_state_dict': finetuned_model.state_dict(),
        'ade': final_ade,
        'prune_amount': args.prune_amount
    }, f"{args.save_dir}/pruned_student.pth")
    
    print(f"\n✓ Saved pruned model to {args.save_dir}")
    
    # 统计
    original_params = sum(p.numel() for p in model.parameters())
    pruned_params = sum((p != 0).sum() for p in finetuned_model.parameters() if p.requires_grad)
    print(f"\nCompression:")
    print(f"  Original: {original_params:,} params")
    print(f"  Pruned: {pruned_params:,} params")
    print(f"  Ratio: {original_params/pruned_params:.2f}x")

if __name__ == '__main__':
    main()