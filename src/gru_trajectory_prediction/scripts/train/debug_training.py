#!/usr/bin/env python3
"""
诊断训练问题
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import glob
import pandas as pd
from models.dataset import TrajectoryDataset
from models.teacher_gru import TeacherGRU

def check_data_loading(data_dir):
    """检查数据加载"""
    print("="*60)
    print("1. Checking Data Loading")
    print("="*60)
    
    # 检查路径
    abs_path = os.path.abspath(data_dir)
    print(f"Data directory: {abs_path}")
    
    if not os.path.exists(data_dir):
        print(f"❌ ERROR: Directory does not exist!")
        return False
    
    # 检查CSV文件
    csv_files = glob.glob(f"{data_dir}/traj_*.csv")
    print(f"Found {len(csv_files)} trajectory files")
    
    if len(csv_files) == 0:
        print(f"❌ ERROR: No trajectory files found!")
        print(f"   Looking for: {data_dir}/traj_*.csv")
        return False
    
    # 检查第一个文件
    print(f"\nChecking first file: {csv_files[0]}")
    try:
        df = pd.read_csv(csv_files[0])
        print(f"✓ File loaded, shape: {df.shape}")
        print(f"✓ Columns: {df.columns.tolist()}")
        print(f"\nFirst 3 rows:")
        print(df.head(3))
        
        # 检查数据范围
        print(f"\nData ranges:")
        print(f"  x: {df.x.min():.2f} ~ {df.x.max():.2f}")
        print(f"  y: {df.y.min():.2f} ~ {df.y.max():.2f}")
        print(f"  vx: {df.vx.min():.2f} ~ {df.vx.max():.2f}")
        print(f"  vy: {df.vy.min():.2f} ~ {df.vy.max():.2f}")
        
        # 检查单位（如果位置>100，可能是厘米）
        if abs(df.x.max()) > 100 or abs(df.y.max()) > 100:
            print(f"\n⚠️  WARNING: Position values are very large (>{100})")
            print(f"   This might be in centimeters instead of meters!")
            print(f"   Consider dividing by 100")
        
        return True
    except Exception as e:
        print(f"❌ ERROR reading file: {e}")
        return False

def check_dataset(data_dir):
    """检查Dataset类"""
    print("\n" + "="*60)
    print("2. Checking Dataset Class")
    print("="*60)
    
    try:
        dataset = TrajectoryDataset(
            data_dir=data_dir,
            obs_len=30,
            pred_len=60,
            skip=1
        )
        
        print(f"✓ Dataset created")
        print(f"  Total samples: {len(dataset)}")
        
        if len(dataset) == 0:
            print(f"❌ ERROR: Dataset is empty!")
            return False
        
        # 测试获取一个样本
        obs, pred = dataset[0]
        print(f"\n✓ Sample loaded")
        print(f"  Observation shape: {obs.shape} (should be [30, 4])")
        print(f"  Prediction shape: {pred.shape} (should be [60, 2])")
        
        # 检查数据范围
        print(f"\n  Observation range:")
        print(f"    x: {obs[:, 0].min():.2f} ~ {obs[:, 0].max():.2f}")
        print(f"    y: {obs[:, 1].min():.2f} ~ {obs[:, 1].max():.2f}")
        print(f"    vx: {obs[:, 2].min():.2f} ~ {obs[:, 2].max():.2f}")
        print(f"    vy: {obs[:, 3].min():.2f} ~ {obs[:, 3].max():.2f}")
        
        print(f"\n  Prediction range:")
        print(f"    x: {pred[:, 0].min():.2f} ~ {pred[:, 0].max():.2f}")
        print(f"    y: {pred[:, 1].min():.2f} ~ {pred[:, 1].max():.2f}")
        
        # 检查是否全是零
        if np.abs(obs).sum() < 0.001:
            print(f"\n⚠️  WARNING: Observation data is all zeros!")
        if np.abs(pred).sum() < 0.001:
            print(f"\n⚠️  WARNING: Prediction data is all zeros!")
        
        return True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_model():
    """检查模型"""
    print("\n" + "="*60)
    print("3. Checking Model")
    print("="*60)
    
    try:
        model = TeacherGRU(
            input_dim=4,
            hidden_dim=256,
            num_layers=3,
            pred_horizon=60
        )
        print(f"✓ Model created")
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # 测试前向传播
        x = torch.randn(2, 30, 4)  # batch=2, seq=30, feat=4
        output = model(x)
        print(f"\n✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape} (should be [2, 60, 2])")
        
        # 检查输出范围
        print(f"  Output range: {output.min():.2f} ~ {output.max():.2f}")
        
        return True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_training_loop(data_dir):
    """检查训练循环"""
    print("\n" + "="*60)
    print("4. Testing Training Loop (1 batch)")
    print("="*60)
    
    try:
        # 创建数据集
        dataset = TrajectoryDataset(data_dir=data_dir, obs_len=30, pred_len=60)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
        
        # 创建模型
        model = TeacherGRU(input_dim=4, hidden_dim=256, num_layers=3, pred_horizon=60)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        # 训练一个batch
        obs, pred = next(iter(loader))
        
        print(f"Batch loaded:")
        print(f"  obs: {obs.shape}, range: [{obs.min():.2f}, {obs.max():.2f}]")
        print(f"  pred: {pred.shape}, range: [{pred.min():.2f}, {pred.max():.2f}]")
        
        # 前向传播
        output = model(obs)
        loss = criterion(output, pred)
        
        print(f"\nBefore training:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Output range: [{output.min():.2f}, {output.max():.2f}]")
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 再次前向传播
        output2 = model(obs)
        loss2 = criterion(output2, pred)
        
        print(f"\nAfter 1 step:")
        print(f"  Loss: {loss2.item():.4f}")
        print(f"  Loss change: {loss2.item() - loss.item():.4f}")
        
        if loss2.item() < loss.item():
            print(f"  ✓ Loss decreased (model is learning)")
        else:
            print(f"  ⚠️  Loss increased (might need to adjust learning rate)")
        
        # 计算ADE
        ade = torch.mean(torch.sqrt(torch.sum((output2 - pred)**2, dim=-1)))
        print(f"\n  ADE: {ade.item():.4f} meters")
        
        if ade.item() > 10:
            print(f"  ❌ ADE is very high! Check data preprocessing")
        
        return True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    args = parser.parse_args()
    
    print("\n" + "�� Training Diagnostics")
    print("="*60)
    
    # 运行所有检查
    checks = [
        check_data_loading(args.data_dir),
        check_dataset(args.data_dir),
        check_model(),
        check_training_loop(args.data_dir)
    ]
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Data loading: {'✓' if checks[0] else '❌'}")
    print(f"Dataset class: {'✓' if checks[1] else '❌'}")
    print(f"Model creation: {'✓' if checks[2] else '❌'}")
    print(f"Training loop: {'✓' if checks[3] else '❌'}")
    
    if all(checks):
        print("\n✓ All checks passed! Training should work.")
    else:
        print("\n❌ Some checks failed. Fix the issues above.")