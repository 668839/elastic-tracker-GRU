#!/usr/bin/env python3
"""
轨迹数据集加载器 - 修复版
✅ 只返回2个值: (obs, pred)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
import glob

class TrajectoryDataset(Dataset):
    """轨迹数据集"""
    
    def __init__(self, data_dir, obs_len=30, pred_len=60, stride=10):
        """
        Args:
            data_dir: CSV文件目录
            obs_len: 观测长度（30步 = 1秒@30Hz）
            pred_len: 预测长度（60步 = 2秒@30Hz）
            stride: 滑动窗口步长
        """
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.stride = stride
        
        self.samples = []
        
        # ✅ 修复：支持子目录（如果你用batch分类的话）
        data_files = list(Path(data_dir).glob('**/*.csv'))
        # 过滤掉报告文件
        data_files = [f for f in data_files if 'report' not in f.name]
        
        print(f"Loading {len(data_files)} trajectory files...")
        
        if len(data_files) == 0:
            print(f"WARNING: No trajectory files found in {data_dir}")
            print(f"Make sure you have traj_*.csv files")
        
        for file in sorted(data_files):
            try:
                df = pd.read_csv(file)
                self._extract_samples(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        print(f"Total samples: {len(self.samples)}")
        
        if len(self.samples) == 0:
            print("ERROR: No valid samples! Check your data.")
    
    def _extract_samples(self, df):
        """从轨迹提取样本（滑动窗口）"""
        total_len = self.obs_len + self.pred_len
        
        # 检查轨迹是否足够长
        if len(df) < total_len:
            return
        
        for start_idx in range(0, len(df) - total_len + 1, self.stride):
            end_idx = start_idx + total_len
            segment = df.iloc[start_idx:end_idx]
            
            # 观测：[x, y, vx, vy]
            obs = segment[['x', 'y', 'vx', 'vy']].values[:self.obs_len]
            # 真实未来：[x, y]
            future = segment[['x', 'y']].values[self.obs_len:]
            
            # 转换为相对坐标（基于最后观测帧）
            last_pos = obs[-1, :2].copy()
            obs[:, :2] -= last_pos
            future -= last_pos
            
            self.samples.append({
                'obs': obs.astype(np.float32),
                'future': future.astype(np.float32)
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # ✅ 修复：只返回2个值
        return (
            torch.FloatTensor(sample['obs']),     # [30, 4]: 观测
            torch.FloatTensor(sample['future'])   # [60, 2]: 预测目标
        )

# 测试代码
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = './gazebo_data'
    
    print(f"Testing dataset with: {data_dir}")
    
    dataset = TrajectoryDataset(data_dir)
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        obs, future = dataset[0]
        print(f"Observation shape: {obs.shape}")  # [30, 4]
        print(f"Future shape: {future.shape}")    # [60, 2]
        print(f"\nSample observation (first 3 steps):")
        print(obs[:3])
        print(f"\nSample future (first 3 steps):")
        print(future[:3])
    else:
        print("ERROR: Dataset is empty!")