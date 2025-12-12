#!/usr/bin/env python3
"""
步骤4:量化 - FP32 -> INT8
进一步压缩模型并加速推理
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.quantization
from models.student_gru import StudentGRU
from models.dataset import TrajectoryDataset
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

def calibrate(model, loader, device, num_batches=100):
    """
    校准模型（确定量化参数）
    """
    print("Calibrating quantization...")
    model.eval()
    
    with torch.no_grad():
        for i, (obs, _, _) in enumerate(tqdm(loader, total=num_batches)):
            if i >= num_batches:
                break
            obs = obs.to(device)
            _ = model(obs)

def evaluate_model(model, loader, device):
    """评估模型"""
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./processed_data')
    parser.add_argument('--pruned_path', required=True, help='Path to pruned model')
    parser.add_argument('--save_dir', default='./checkpoints/quantized')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cpu')  # 量化需要在CPU上
    
    # 加载剪枝后的模型
    print("Loading pruned model...")
    model = StudentGRU()
    checkpoint = torch.load(args.pruned_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 数据
    dataset = TrajectoryDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 评估FP32模型
    fp32_ade = evaluate_model(model, loader, device)
    print(f"FP32 Model ADE: {fp32_ade:.4f}")
    
    # 配置量化
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # 准备量化（插入观察器）
    torch.quantization.prepare(model, inplace=True)
    
    # 校准
    calibrate(model, loader, device, num_batches=100)
    
    # 转换为量化模型
    torch.quantization.convert(model, inplace=True)
    
    # 评估INT8模型
    int8_ade = evaluate_model(model, loader, device)
    print(f"INT8 Model ADE: {int8_ade:.4f}")
    print(f"Accuracy drop: {int8_ade - fp32_ade:.4f}")
    
    # 保存量化模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'ade': int8_ade
    }, f"{args.save_dir}/quantized_student.pth")
    
    print(f"\n✓ Saved quantized model to {args.save_dir}")
    
    # 模型大小对比
    import os
    fp32_size = os.path.getsize(args.pruned_path) / 1024 / 1024
    int8_size = os.path.getsize(f"{args.save_dir}/quantized_student.pth") / 1024 / 1024
    print(f"\nModel size:")
    print(f"  FP32: {fp32_size:.2f} MB")
    print(f"  INT8: {int8_size:.2f} MB")
    print(f"  Compression: {fp32_size/int8_size:.2f}x")

if __name__ == '__main__':
    main()