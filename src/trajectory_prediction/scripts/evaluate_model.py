#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_model.py

模型评估脚本，计算以下指标：
1. ADE (Average Displacement Error) - 平均位移误差
2. FDE (Final Displacement Error) - 最终位移误差
3. Miss Rate - 超过阈值的预测比例
4. 不同时间范围的ADE/FDE (如1s, 1.5s, 2s)

输出：
- 数值指标
- 可视化图表
"""

import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

from models import create_teacher_model, create_student_model


def compute_ade(pred: np.ndarray, target: np.ndarray) -> float:
    """计算ADE"""
    displacement = np.sqrt(np.sum((pred - target) ** 2, axis=-1))
    return np.mean(displacement)


def compute_fde(pred: np.ndarray, target: np.ndarray) -> float:
    """计算FDE"""
    return np.sqrt(np.sum((pred[:, -1, :] - target[:, -1, :]) ** 2, axis=-1)).mean()


def compute_ade_at_timestep(pred: np.ndarray, target: np.ndarray, timestep: int) -> float:
    """计算特定时间步的ADE"""
    displacement = np.sqrt(np.sum((pred[:, :timestep+1, :] - target[:, :timestep+1, :]) ** 2, axis=-1))
    return np.mean(displacement)


def compute_miss_rate(pred: np.ndarray, target: np.ndarray, threshold: float = 2.0) -> float:
    """
    计算Miss Rate
    
    如果FDE > threshold，则算作miss
    """
    fde_per_sample = np.sqrt(np.sum((pred[:, -1, :] - target[:, -1, :]) ** 2, axis=-1))
    return np.mean(fde_per_sample > threshold)


def evaluate_model(model: nn.Module,
                   dataloader: DataLoader,
                   device: torch.device,
                   pred_len: int,
                   dt: float = 1/30) -> dict:
    """
    评估模型
    
    Args:
        model: 待评估的模型
        dataloader: 测试数据加载器
        device: 计算设备
        pred_len: 预测长度
        dt: 时间步长
        
    Returns:
        metrics: 评估指标字典
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for obs_seq, pred_target in tqdm(dataloader, desc='Evaluating'):
            obs_seq = obs_seq.to(device)
            
            predictions = model.predict_deterministic(obs_seq, pred_len)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(pred_target.numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    print(f"Evaluation samples: {len(predictions)}")
    
    # 计算指标
    metrics = {}
    
    # 总体ADE/FDE
    metrics['ADE'] = compute_ade(predictions, targets)
    metrics['FDE'] = compute_fde(predictions, targets)
    
    # 不同时间范围的ADE
    # 假设30Hz，30帧=1s，45帧=1.5s，60帧=2s
    time_horizons = {
        '1.0s': 30,
        '1.5s': 45,
        '2.0s': 60
    }
    
    for name, timestep in time_horizons.items():
        if timestep <= pred_len:
            metrics[f'ADE@{name}'] = compute_ade_at_timestep(predictions, targets, timestep-1)
            # FDE at that timestep
            metrics[f'FDE@{name}'] = np.sqrt(
                np.sum((predictions[:, timestep-1, :] - targets[:, timestep-1, :]) ** 2, axis=-1)
            ).mean()
    
    # Miss Rate
    for threshold in [1.0, 2.0, 3.0]:
        metrics[f'MissRate@{threshold}m'] = compute_miss_rate(predictions, targets, threshold)
    
    # 每个时间步的误差 (用于绘图)
    timestep_errors = np.sqrt(np.sum((predictions - targets) ** 2, axis=-1)).mean(axis=0)
    metrics['timestep_errors'] = timestep_errors.tolist()
    
    return metrics


def plot_results(metrics: dict, save_path: str, pred_len: int, dt: float = 1/30):
    """绘制评估结果"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 随时间变化的误差
    ax1 = axes[0, 0]
    timestep_errors = np.array(metrics['timestep_errors'])
    time_axis = np.arange(1, len(timestep_errors) + 1) * dt
    ax1.plot(time_axis, timestep_errors, 'b-', linewidth=2)
    ax1.set_xlabel('Prediction Horizon (s)')
    ax1.set_ylabel('Displacement Error (m)')
    ax1.set_title('Prediction Error vs Time')
    ax1.grid(True)
    
    # 2. ADE/FDE柱状图
    ax2 = axes[0, 1]
    ade_fde_keys = ['ADE', 'FDE']
    ade_fde_values = [metrics[k] for k in ade_fde_keys]
    ax2.bar(ade_fde_keys, ade_fde_values, color=['blue', 'orange'])
    ax2.set_ylabel('Error (m)')
    ax2.set_title('Overall ADE and FDE')
    for i, v in enumerate(ade_fde_values):
        ax2.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # 3. 不同时间范围的ADE
    ax3 = axes[1, 0]
    horizon_keys = [k for k in metrics.keys() if k.startswith('ADE@') and 's' in k]
    horizon_values = [metrics[k] for k in horizon_keys]
    horizon_labels = [k.replace('ADE@', '') for k in horizon_keys]
    ax3.bar(horizon_labels, horizon_values, color='green')
    ax3.set_xlabel('Prediction Horizon')
    ax3.set_ylabel('ADE (m)')
    ax3.set_title('ADE at Different Horizons')
    for i, v in enumerate(horizon_values):
        ax3.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # 4. Miss Rate
    ax4 = axes[1, 1]
    miss_keys = [k for k in metrics.keys() if k.startswith('MissRate')]
    miss_values = [metrics[k] * 100 for k in miss_keys]  # 转换为百分比
    miss_labels = [k.replace('MissRate@', '') for k in miss_keys]
    ax4.bar(miss_labels, miss_values, color='red')
    ax4.set_xlabel('Threshold')
    ax4.set_ylabel('Miss Rate (%)')
    ax4.set_title('Miss Rate at Different Thresholds')
    for i, v in enumerate(miss_values):
        ax4.text(i, v + 0.5, f'{v:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Results plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trajectory prediction model')
    parser.add_argument('--data_path', type=str, default='data/processed_data.npz',
                        help='Path to processed data')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='teacher',
                        choices=['teacher', 'student'],
                        help='Model type')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Data split to evaluate')
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    print(f"Loading data from {args.data_path}...")
    data = np.load(args.data_path)
    
    obs_data = torch.from_numpy(data[f'{args.split}_obs'])
    pred_data = torch.from_numpy(data[f'{args.split}_pred'])
    
    obs_len = int(data['obs_len'])
    pred_len = int(data['pred_len'])
    dt = float(data['dt'])
    
    print(f"Evaluating on {args.split} set: {len(obs_data)} samples")
    
    # 创建DataLoader
    dataset = TensorDataset(obs_data, pred_data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 创建模型
    if args.model_type == 'teacher':
        model = create_teacher_model(output_uncertainty=False)
    else:
        model = create_student_model(output_uncertainty=True)
    
    # 加载权重
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # 评估
    metrics = evaluate_model(model, dataloader, device, pred_len, dt)
    
    # 打印结果
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"ADE: {metrics['ADE']:.4f} m")
    print(f"FDE: {metrics['FDE']:.4f} m")
    print("-"*50)
    
    for key in sorted(metrics.keys()):
        if key.startswith('ADE@') or key.startswith('FDE@'):
            print(f"{key}: {metrics[key]:.4f} m")
    
    print("-"*50)
    for key in sorted(metrics.keys()):
        if key.startswith('MissRate'):
            print(f"{key}: {metrics[key]*100:.2f}%")
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存为JSON (不包含timestep_errors数组)
    metrics_json = {k: v for k, v in metrics.items() if k != 'timestep_errors'}
    with open(os.path.join(args.output_dir, f'{args.model_type}_{args.split}_metrics.json'), 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    # 绘图
    plot_path = os.path.join(args.output_dir, f'{args.model_type}_{args.split}_results.png')
    plot_results(metrics, plot_path, pred_len, dt)
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()
