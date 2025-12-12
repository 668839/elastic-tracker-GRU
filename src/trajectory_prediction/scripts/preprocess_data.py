#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_data.py

数据预处理脚本，实现公式推导文档v3.1中的：
1. Savitzky-Golay滤波计算速度和加速度 (公式7-10)
2. 航向对齐坐标系变换 (公式11-17)
3. 生成训练样本 (观测序列 -> 预测序列)

输入: raw_trajectories.csv (timestamp, x, y, z, vx, vy, vz, trajectory_id)
输出: processed_data.npz (包含训练/验证/测试集)
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from typing import List, Tuple, Dict
import os
import argparse
from tqdm import tqdm


class TrajectoryPreprocessor:
    """轨迹数据预处理器"""
    
    def __init__(self, 
                 obs_len: int = 30,      # 观测序列长度 (1秒 @ 30Hz)
                 pred_len: int = 60,     # 预测序列长度 (2秒 @ 30Hz)
                 sample_rate: float = 30.0,  # 采样率 Hz
                 sg_window: int = 5,     # SG滤波窗口大小
                 sg_polyorder: int = 2,  # SG滤波多项式阶数
                 min_velocity: float = 0.1):  # 航向计算的最小速度阈值
        
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.total_len = obs_len + pred_len
        self.dt = 1.0 / sample_rate
        self.sg_window = sg_window
        self.sg_polyorder = sg_polyorder
        self.min_velocity = min_velocity
        
        # SG滤波系数 (公式9): [-3, 12, 17, 12, -3] / 35
        # 这里使用scipy的savgol_filter，它内部会计算这些系数
        
    def compute_velocity_sg(self, positions: np.ndarray) -> np.ndarray:
        """
        使用SG滤波计算速度
        
        按照公式7-8的顺序：先差分再平滑
        
        Args:
            positions: shape (N, 2), 世界坐标系下的位置 [x, y]
            
        Returns:
            velocities: shape (N, 2), 平滑后的速度 [vx, vy]
        """
        N = positions.shape[0]
        
        # Step 1: 有限差分计算原始速度 (公式7)
        raw_velocity = np.zeros_like(positions)
        raw_velocity[1:] = (positions[1:] - positions[:-1]) / self.dt
        raw_velocity[0] = raw_velocity[1]  # 边界处理
        
        # Step 2: SG滤波平滑 (公式8)
        if N >= self.sg_window:
            velocity = np.zeros_like(raw_velocity)
            for i in range(2):  # x和y分别滤波
                velocity[:, i] = savgol_filter(raw_velocity[:, i], 
                                               self.sg_window, 
                                               self.sg_polyorder,
                                               mode='nearest')
        else:
            velocity = raw_velocity
            
        return velocity
    
    def compute_acceleration(self, velocities: np.ndarray) -> np.ndarray:
        """
        计算加速度 (公式10)
        
        Args:
            velocities: shape (N, 2), 速度 [vx, vy]
            
        Returns:
            accelerations: shape (N, 2), 加速度 [ax, ay]
        """
        N = velocities.shape[0]
        
        # 差分计算加速度
        raw_acc = np.zeros_like(velocities)
        raw_acc[1:] = (velocities[1:] - velocities[:-1]) / self.dt
        raw_acc[0] = raw_acc[1]
        
        # 可选：对加速度也做SG平滑
        if N >= self.sg_window:
            acceleration = np.zeros_like(raw_acc)
            for i in range(2):
                acceleration[:, i] = savgol_filter(raw_acc[:, i],
                                                   self.sg_window,
                                                   self.sg_polyorder,
                                                   mode='nearest')
        else:
            acceleration = raw_acc
            
        return acceleration
    
    def compute_heading_angle(self, vx: float, vy: float, 
                              last_heading: float = 0.0) -> float:
        """
        计算航向角 (公式11)
        
        Args:
            vx, vy: 当前速度分量
            last_heading: 上一有效航向角 (用于低速情况)
            
        Returns:
            heading: 航向角 (弧度, 范围 [-π, π])
        """
        velocity_magnitude = np.sqrt(vx**2 + vy**2)
        
        # 边界情况：速度太小时使用上一有效航向或0
        if velocity_magnitude < self.min_velocity:
            return last_heading
        
        return np.arctan2(vy, vx)
    
    def get_rotation_matrix(self, theta: float) -> np.ndarray:
        """
        构造航向对齐旋转矩阵 (公式12)
        
        R_align = [[cos(θ), sin(θ)], [-sin(θ), cos(θ)]]
        
        注意：这是从世界坐标系到航向对齐坐标系的变换
        """
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, s], [-s, c]])
    
    def transform_to_heading_aligned(self, 
                                     positions: np.ndarray,
                                     velocities: np.ndarray,
                                     accelerations: np.ndarray,
                                     anchor_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        将序列变换到航向对齐坐标系 (公式13-15)
        
        Args:
            positions: shape (N, 2), 世界坐标系位置
            velocities: shape (N, 2), 世界坐标系速度
            accelerations: shape (N, 2), 世界坐标系加速度
            anchor_idx: 锚点索引 (通常是观测序列的最后一个点)
            
        Returns:
            pos_aligned: 航向对齐坐标系下的相对位置
            vel_aligned: 航向对齐坐标系下的速度
            acc_aligned: 航向对齐坐标系下的加速度
            heading: 航向角
        """
        # 锚点：当前位置和速度
        anchor_pos = positions[anchor_idx]
        anchor_vel = velocities[anchor_idx]
        
        # 计算航向角 (公式11)
        heading = self.compute_heading_angle(anchor_vel[0], anchor_vel[1])
        
        # 旋转矩阵 (公式12)
        R = self.get_rotation_matrix(heading)
        
        # 位置变换：相对于锚点，然后旋转 (公式13)
        relative_pos = positions - anchor_pos
        pos_aligned = (R @ relative_pos.T).T
        
        # 速度变换：只旋转 (公式14)
        vel_aligned = (R @ velocities.T).T
        
        # 加速度变换：只旋转 (公式15)
        acc_aligned = (R @ accelerations.T).T
        
        return pos_aligned, vel_aligned, acc_aligned, heading
    
    def process_single_trajectory(self, 
                                  timestamps: np.ndarray,
                                  positions_3d: np.ndarray) -> List[Dict]:
        """
        处理单条轨迹，生成多个训练样本
        
        Args:
            timestamps: shape (N,), 时间戳
            positions_3d: shape (N, 3), [x, y, z]
            
        Returns:
            samples: 训练样本列表，每个样本包含：
                - obs_seq: 观测序列 (obs_len, 6) [x, y, vx, vy, ax, ay]
                - pred_seq: 预测序列 (pred_len, 2) [x, y]
                - heading: 航向角
                - anchor_pos: 锚点位置 (世界坐标系)
        """
        samples = []
        N = len(timestamps)
        
        if N < self.total_len:
            return samples
        
        # 只使用x, y (假设地面目标)
        positions = positions_3d[:, :2]
        
        # 计算速度和加速度
        velocities = self.compute_velocity_sg(positions)
        accelerations = self.compute_acceleration(velocities)
        
        # 滑动窗口生成样本
        stride = 5  # 步长，可调整以控制样本数量
        for start_idx in range(0, N - self.total_len + 1, stride):
            end_obs_idx = start_idx + self.obs_len
            end_pred_idx = start_idx + self.total_len
            
            # 提取原始序列
            pos_seq = positions[start_idx:end_pred_idx]
            vel_seq = velocities[start_idx:end_pred_idx]
            acc_seq = accelerations[start_idx:end_pred_idx]
            
            # 锚点是观测序列的最后一个点
            anchor_idx = self.obs_len - 1
            
            # 变换到航向对齐坐标系
            pos_aligned, vel_aligned, acc_aligned, heading = \
                self.transform_to_heading_aligned(pos_seq, vel_seq, acc_seq, anchor_idx)
            
            # 构造6维输入特征 (公式19): [x, y, vx, vy, ax, ay]
            obs_features = np.concatenate([
                pos_aligned[:self.obs_len],      # (obs_len, 2)
                vel_aligned[:self.obs_len],      # (obs_len, 2)
                acc_aligned[:self.obs_len]       # (obs_len, 2)
            ], axis=1)  # -> (obs_len, 6)
            
            # 预测目标：航向对齐坐标系下的位置
            pred_targets = pos_aligned[self.obs_len:self.total_len]  # (pred_len, 2)
            
            # 保存锚点位置 (用于后续还原到世界坐标系)
            anchor_pos = positions[start_idx + anchor_idx]
            
            samples.append({
                'obs_seq': obs_features.astype(np.float32),
                'pred_seq': pred_targets.astype(np.float32),
                'heading': heading,
                'anchor_pos': anchor_pos.astype(np.float32)
            })
        
        return samples
    
    def process_dataset(self, csv_path: str, output_path: str,
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       test_ratio: float = 0.15,
                       seed: int = 42):
        """
        处理整个数据集
        
        Args:
            csv_path: 原始CSV文件路径
            output_path: 输出NPZ文件路径
            train_ratio, val_ratio, test_ratio: 数据集划分比例
            seed: 随机种子
        """
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        print(f"Total rows: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Trajectory IDs: {df['trajectory_id'].unique()}")
        
        all_samples = []
        
        # 按轨迹ID分组处理
        trajectory_ids = df['trajectory_id'].unique()
        
        for traj_id in tqdm(trajectory_ids, desc="Processing trajectories"):
            traj_df = df[df['trajectory_id'] == traj_id].sort_values('timestamp')
            
            timestamps = traj_df['timestamp'].values
            positions = traj_df[['x', 'y', 'z']].values
            
            # 检查采样率是否一致
            if len(timestamps) > 1:
                dt_mean = np.mean(np.diff(timestamps))
                if abs(dt_mean - self.dt) > 0.01:
                    print(f"Warning: Trajectory {traj_id} has inconsistent sample rate: "
                          f"expected {1/self.dt:.1f}Hz, got {1/dt_mean:.1f}Hz")
            
            samples = self.process_single_trajectory(timestamps, positions)
            all_samples.extend(samples)
        
        print(f"Total samples generated: {len(all_samples)}")
        
        if len(all_samples) == 0:
            print("Error: No samples generated!")
            return
        
        # 随机打乱并划分数据集
        np.random.seed(seed)
        indices = np.random.permutation(len(all_samples))
        
        n_train = int(len(all_samples) * train_ratio)
        n_val = int(len(all_samples) * val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # 提取数据
        def extract_split(split_indices):
            obs = np.array([all_samples[i]['obs_seq'] for i in split_indices])
            pred = np.array([all_samples[i]['pred_seq'] for i in split_indices])
            headings = np.array([all_samples[i]['heading'] for i in split_indices])
            anchors = np.array([all_samples[i]['anchor_pos'] for i in split_indices])
            return obs, pred, headings, anchors
        
        train_obs, train_pred, train_headings, train_anchors = extract_split(train_indices)
        val_obs, val_pred, val_headings, val_anchors = extract_split(val_indices)
        test_obs, test_pred, test_headings, test_anchors = extract_split(test_indices)
        
        # 保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        np.savez(output_path,
                 # 训练集
                 train_obs=train_obs,
                 train_pred=train_pred,
                 train_headings=train_headings,
                 train_anchors=train_anchors,
                 # 验证集
                 val_obs=val_obs,
                 val_pred=val_pred,
                 val_headings=val_headings,
                 val_anchors=val_anchors,
                 # 测试集
                 test_obs=test_obs,
                 test_pred=test_pred,
                 test_headings=test_headings,
                 test_anchors=test_anchors,
                 # 元数据
                 obs_len=self.obs_len,
                 pred_len=self.pred_len,
                 dt=self.dt)
        
        print(f"\nDataset saved to {output_path}")
        print(f"Train: {len(train_obs)} samples")
        print(f"Val: {len(val_obs)} samples")
        print(f"Test: {len(test_obs)} samples")
        print(f"\nData shapes:")
        print(f"  obs_seq: {train_obs.shape} (samples, obs_len, features)")
        print(f"  pred_seq: {train_pred.shape} (samples, pred_len, 2)")


def main():
    parser = argparse.ArgumentParser(description='Preprocess trajectory data')
    parser.add_argument('--input', type=str, 
                        default='data/raw_trajectories.csv',
                        help='Input CSV file path')
    parser.add_argument('--output', type=str,
                        default='data/processed_data.npz',
                        help='Output NPZ file path')
    parser.add_argument('--obs_len', type=int, default=30,
                        help='Observation sequence length')
    parser.add_argument('--pred_len', type=int, default=60,
                        help='Prediction sequence length')
    parser.add_argument('--sample_rate', type=float, default=30.0,
                        help='Sample rate in Hz')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    preprocessor = TrajectoryPreprocessor(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        sample_rate=args.sample_rate
    )
    
    preprocessor.process_dataset(
        csv_path=args.input,
        output_path=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
