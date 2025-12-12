#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_gazebo_data.py

预处理从Gazebo收集的actor轨迹数据
输入: collect_gazebo_actor.py 生成的多个 traj_XXXX.csv 文件
输出: processed_data.npz (用于训练)

处理流程:
1. 读取目录下所有 traj_*.csv 文件
2. 重采样到固定频率 (30Hz)，包括位置和速度
3. 使用Savitzky-Golay滤波计算加速度
4. 航向对齐坐标变换
5. 生成训练样本 (滑动窗口)
6. 划分训练/验证/测试集

注意: 速度直接使用Gazebo提供的数据（插值），不重新计算
      只有加速度使用SG滤波从速度计算

使用方法:
    python3 preprocess_gazebo_data.py \
        --input_dir data/gazebo_raw \
        --output data/processed_data.npz
"""

import numpy as np
import pandas as pd
import argparse
import os
import glob
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from tqdm import tqdm


def load_trajectory_files(input_dir):
    """
    加载目录下所有轨迹文件
    
    Args:
        input_dir: 包含 traj_*.csv 文件的目录
    
    Returns:
        trajectories: list of DataFrames
        filenames: list of filenames
    """
    pattern = os.path.join(input_dir, 'traj_*.csv')
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No trajectory files found in {input_dir}")
    
    trajectories = []
    filenames = []
    
    for f in files:
        try:
            df = pd.read_csv(f)
            # 检查必要的列
            required_cols = ['time', 'x', 'y', 'z', 'vx', 'vy', 'vz']
            if all(col in df.columns for col in required_cols):
                trajectories.append(df)
                filenames.append(os.path.basename(f))
            else:
                print(f"Warning: Skipping {f} - missing required columns")
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
    
    return trajectories, filenames


def resample_trajectory(df, target_freq=30.0):
    """
    重采样轨迹到固定频率（包括位置和速度）
    
    Args:
        df: 单条轨迹的DataFrame (columns: time, x, y, z, vx, vy, vz)
        target_freq: 目标频率(Hz)
    
    Returns:
        重采样后的DataFrame，或None如果轨迹太短
    """
    if len(df) < 2:
        return None
    
    # 时间范围
    t_start = df['time'].iloc[0]
    t_end = df['time'].iloc[-1]
    duration = t_end - t_start
    
    if duration < 0.1:  # 太短
        return None
    
    # 新的时间点 (严格等间隔)
    dt = 1.0 / target_freq
    new_times = np.arange(t_start, t_end, dt)
    
    if len(new_times) < 10:
        return None
    
    # 插值各个字段（位置和速度都插值）
    result = {'time': new_times}
    
    for col in ['x', 'y', 'z', 'vx', 'vy', 'vz']:
        interp_func = interp1d(
            df['time'].values, 
            df[col].values,
            kind='linear',
            fill_value='extrapolate'
        )
        result[col] = interp_func(new_times)
    
    return pd.DataFrame(result)


def compute_acceleration_sg(velocity, dt, window=5, polyorder=2):
    """
    使用Savitzky-Golay滤波计算加速度 (公式10)
    
    先差分再平滑:
    a_raw = diff(v) / dt
    a_smooth = SG_filter(a_raw)
    
    Args:
        velocity: 速度序列 (N,)
        dt: 时间步长
        window: SG滤波窗口大小 (必须是奇数)
        polyorder: 多项式阶数
    
    Returns:
        加速度序列 (N,)
    """
    # 确保window是奇数
    if window % 2 == 0:
        window += 1
    
    if len(velocity) < window:
        # 数据太短，直接用差分
        acceleration = np.gradient(velocity, dt)
        return acceleration
    
    # 先差分
    accel_raw = np.gradient(velocity, dt)
    
    # 再SG平滑
    accel_smooth = savgol_filter(accel_raw, window, polyorder, mode='interp')
    
    return accel_smooth


def heading_aligned_transform(x, y, vx, vy, ax, ay, anchor_idx=-1):
    """
    航向对齐坐标变换 (公式11-17)
    
    将世界坐标系转换到航向对齐坐标系:
    - 原点: 观测序列最后一个点 (anchor)
    - X轴: 沿当前速度方向
    - Y轴: 垂直于速度方向(左侧)
    
    Args:
        x, y: 位置序列
        vx, vy: 速度序列
        ax, ay: 加速度序列
        anchor_idx: 锚点索引 (默认-1表示最后一个观测点)
    
    Returns:
        x_aligned, y_aligned: 对齐后的位置
        vx_aligned, vy_aligned: 对齐后的速度
        ax_aligned, ay_aligned: 对齐后的加速度
        heading: 航向角
        anchor: 锚点位置
    """
    # 锚点位置
    anchor = np.array([x[anchor_idx], y[anchor_idx]])
    
    # 航向角 (公式12): θ = atan2(vy, vx)
    heading = np.arctan2(vy[anchor_idx], vx[anchor_idx])
    
    # 旋转矩阵 (公式14): R_align
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    # R_align = [[cos θ, sin θ], [-sin θ, cos θ]]
    # 这样变换后，X轴指向航向方向
    
    # 位置变换 (公式15): 先平移到锚点，再旋转
    x_centered = x - anchor[0]
    y_centered = y - anchor[1]
    
    x_aligned = cos_h * x_centered + sin_h * y_centered
    y_aligned = -sin_h * x_centered + cos_h * y_centered
    
    # 速度变换 (公式16): 只旋转，不平移
    vx_aligned = cos_h * vx + sin_h * vy
    vy_aligned = -sin_h * vx + cos_h * vy
    
    # 加速度变换 (公式17): 只旋转，不平移
    ax_aligned = cos_h * ax + sin_h * ay
    ay_aligned = -sin_h * ax + cos_h * ay
    
    return (x_aligned, y_aligned, vx_aligned, vy_aligned, 
            ax_aligned, ay_aligned, heading, anchor)


def process_single_trajectory(df, dt, sg_window=5, sg_polyorder=2):
    """
    处理单条轨迹
    
    注意: 速度直接使用Gazebo提供的数据（已插值）
          只有加速度使用SG滤波计算
    
    Args:
        df: 轨迹DataFrame (columns: time, x, y, z, vx, vy, vz)
        dt: 时间步长
        sg_window: SG滤波窗口 (用于加速度计算)
        sg_polyorder: SG滤波阶数
    
    Returns:
        processed: dict with processed data
    """
    x = df['x'].values
    y = df['y'].values
    
    # 直接使用Gazebo提供的速度（已经插值到30Hz）
    vx = df['vx'].values
    vy = df['vy'].values
    
    # 只有加速度需要用SG滤波计算（从速度差分+平滑）
    ax = compute_acceleration_sg(vx, dt, sg_window, sg_polyorder)
    ay = compute_acceleration_sg(vy, dt, sg_window, sg_polyorder)
    
    return {
        'x': x, 'y': y,
        'vx': vx, 'vy': vy,
        'ax': ax, 'ay': ay,
        'time': df['time'].values
    }


def generate_samples(processed_traj, obs_len=30, pred_len=60, stride=5):
    """
    使用滑动窗口生成训练样本
    
    Args:
        processed_traj: 处理后的轨迹数据
        obs_len: 观测序列长度 (30 = 1秒 @ 30Hz)
        pred_len: 预测序列长度 (60 = 2秒 @ 30Hz)
        stride: 滑动步长
    
    Returns:
        samples: list of (obs_seq, pred_seq, heading, anchor)
    """
    total_len = obs_len + pred_len
    traj_len = len(processed_traj['x'])
    
    if traj_len < total_len:
        return []
    
    samples = []
    
    for start_idx in range(0, traj_len - total_len + 1, stride):
        obs_end = start_idx + obs_len
        pred_end = obs_end + pred_len
        
        # 提取观测和预测段
        x = processed_traj['x'][start_idx:pred_end]
        y = processed_traj['y'][start_idx:pred_end]
        vx = processed_traj['vx'][start_idx:pred_end]
        vy = processed_traj['vy'][start_idx:pred_end]
        ax = processed_traj['ax'][start_idx:pred_end]
        ay = processed_traj['ay'][start_idx:pred_end]
        
        # 航向对齐变换 (以观测序列最后一点为锚点)
        (x_al, y_al, vx_al, vy_al, ax_al, ay_al, 
         heading, anchor) = heading_aligned_transform(
            x, y, vx, vy, ax, ay, anchor_idx=obs_len-1
        )
        
        # 构建观测序列: (obs_len, 6) = [x, y, vx, vy, ax, ay]
        obs_seq = np.stack([
            x_al[:obs_len],
            y_al[:obs_len],
            vx_al[:obs_len],
            vy_al[:obs_len],
            ax_al[:obs_len],
            ay_al[:obs_len]
        ], axis=1)
        
        # 构建预测序列: (pred_len, 2) = [x, y]
        pred_seq = np.stack([
            x_al[obs_len:],
            y_al[obs_len:]
        ], axis=1)
        
        samples.append({
            'obs': obs_seq,
            'pred': pred_seq,
            'heading': heading,
            'anchor': anchor
        })
    
    return samples


def split_data(samples, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    划分训练/验证/测试集
    
    Args:
        samples: 所有样本
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子
    
    Returns:
        train_samples, val_samples, test_samples
    """
    np.random.seed(seed)
    
    n_samples = len(samples)
    indices = np.random.permutation(n_samples)
    
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    test_samples = [samples[i] for i in test_indices]
    
    return train_samples, val_samples, test_samples


def samples_to_arrays(samples):
    """
    将样本列表转换为numpy数组
    
    Args:
        samples: list of sample dicts
    
    Returns:
        obs_array: (N, obs_len, 6)
        pred_array: (N, pred_len, 2)
        headings: (N,)
        anchors: (N, 2)
    """
    if not samples:
        return None, None, None, None
    
    obs_array = np.stack([s['obs'] for s in samples], axis=0)
    pred_array = np.stack([s['pred'] for s in samples], axis=0)
    headings = np.array([s['heading'] for s in samples])
    anchors = np.stack([s['anchor'] for s in samples], axis=0)
    
    return obs_array, pred_array, headings, anchors


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess Gazebo trajectory data (multiple traj_*.csv files)'
    )
    parser.add_argument('--input_dir', type=str, 
                        default='data/gazebo_raw',
                        help='Directory containing traj_*.csv files')
    parser.add_argument('--output', type=str,
                        default='data/processed_data.npz',
                        help='Output NPZ file for training')
    parser.add_argument('--obs_len', type=int, default=30,
                        help='Observation sequence length (default: 30 = 1s @ 30Hz)')
    parser.add_argument('--pred_len', type=int, default=60,
                        help='Prediction sequence length (default: 60 = 2s @ 30Hz)')
    parser.add_argument('--sample_rate', type=float, default=30.0,
                        help='Target sample rate in Hz')
    parser.add_argument('--stride', type=int, default=5,
                        help='Sliding window stride')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--sg_window', type=int, default=5,
                        help='Savitzky-Golay filter window size (for acceleration)')
    parser.add_argument('--sg_polyorder', type=int, default=2,
                        help='Savitzky-Golay filter polynomial order')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Gazebo Data Preprocessing")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output file: {args.output}")
    print(f"Observation length: {args.obs_len} frames ({args.obs_len/args.sample_rate:.2f}s)")
    print(f"Prediction length: {args.pred_len} frames ({args.pred_len/args.sample_rate:.2f}s)")
    print(f"Sample rate: {args.sample_rate} Hz")
    print(f"Sliding window stride: {args.stride}")
    print(f"")
    print(f"NOTE: Using Gazebo-provided velocity directly (interpolated)")
    print(f"      Only acceleration is computed via SG filter")
    print(f"      SG filter: window={args.sg_window}, polyorder={args.sg_polyorder}")
    print("=" * 60)
    
    # 加载轨迹文件
    print("\nLoading trajectory files...")
    try:
        trajectories, filenames = load_trajectory_files(args.input_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    print(f"Found {len(trajectories)} trajectory files")
    
    # 统计原始数据
    total_points = sum(len(df) for df in trajectories)
    total_duration = sum(df['time'].max() for df in trajectories)
    print(f"Total data points: {total_points}")
    print(f"Total duration: {total_duration:.1f}s")
    
    dt = 1.0 / args.sample_rate
    all_samples = []
    valid_trajs = 0
    skipped_trajs = 0
    
    print("\nProcessing trajectories...")
    for df, filename in tqdm(zip(trajectories, filenames), total=len(trajectories), desc="Processing"):
        # 重采样到固定频率（位置和速度都插值）
        resampled = resample_trajectory(df, args.sample_rate)
        if resampled is None:
            skipped_trajs += 1
            continue
        
        # 检查长度是否足够
        min_len = args.obs_len + args.pred_len
        if len(resampled) < min_len:
            skipped_trajs += 1
            continue
        
        # 处理轨迹（直接用Gazebo速度，只计算加速度）
        processed = process_single_trajectory(
            resampled, dt, args.sg_window, args.sg_polyorder
        )
        
        # 生成样本
        samples = generate_samples(
            processed, args.obs_len, args.pred_len, args.stride
        )
        
        if samples:
            all_samples.extend(samples)
            valid_trajs += 1
    
    print(f"\nTrajectories processed: {valid_trajs}")
    print(f"Trajectories skipped (too short): {skipped_trajs}")
    print(f"Total samples generated: {len(all_samples)}")
    
    if not all_samples:
        print("Error: No samples generated!")
        print("Check that your trajectories are long enough:")
        print(f"  Required minimum length: {args.obs_len + args.pred_len} frames = {(args.obs_len + args.pred_len)/args.sample_rate:.2f}s")
        return
    
    # 划分数据集
    print("\nSplitting data...")
    train_samples, val_samples, test_samples = split_data(
        all_samples, args.train_ratio, args.val_ratio, args.seed
    )
    
    print(f"  Training:   {len(train_samples)} samples ({100*len(train_samples)/len(all_samples):.1f}%)")
    print(f"  Validation: {len(val_samples)} samples ({100*len(val_samples)/len(all_samples):.1f}%)")
    print(f"  Testing:    {len(test_samples)} samples ({100*len(test_samples)/len(all_samples):.1f}%)")
    
    # 转换为数组
    train_obs, train_pred, train_headings, train_anchors = samples_to_arrays(train_samples)
    val_obs, val_pred, val_headings, val_anchors = samples_to_arrays(val_samples)
    test_obs, test_pred, test_headings, test_anchors = samples_to_arrays(test_samples)
    
    # 保存
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    np.savez(
        args.output,
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
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        sample_rate=args.sample_rate,
        input_dim=6,  # [x, y, vx, vy, ax, ay]
        output_dim=2  # [x, y]
    )
    
    print(f"\n✓ Saved to {args.output}")
    
    # 打印数据统计
    print("\n" + "=" * 60)
    print("Data Statistics")
    print("=" * 60)
    print(f"Observation shape: {train_obs.shape}")
    print(f"Prediction shape: {train_pred.shape}")
    
    # 位置范围
    all_obs = np.concatenate([train_obs, val_obs, test_obs], axis=0)
    all_pred = np.concatenate([train_pred, val_pred, test_pred], axis=0)
    
    print(f"\nObservation position range (heading-aligned):")
    print(f"  X: [{all_obs[:,:,0].min():.2f}, {all_obs[:,:,0].max():.2f}]")
    print(f"  Y: [{all_obs[:,:,1].min():.2f}, {all_obs[:,:,1].max():.2f}]")
    
    print(f"\nPrediction position range (heading-aligned):")
    print(f"  X: [{all_pred[:,:,0].min():.2f}, {all_pred[:,:,0].max():.2f}]")
    print(f"  Y: [{all_pred[:,:,1].min():.2f}, {all_pred[:,:,1].max():.2f}]")
    
    # 速度范围
    print(f"\nVelocity range (from Gazebo):")
    print(f"  Vx: [{all_obs[:,:,2].min():.2f}, {all_obs[:,:,2].max():.2f}] m/s")
    print(f"  Vy: [{all_obs[:,:,3].min():.2f}, {all_obs[:,:,3].max():.2f}] m/s")
    
    # 加速度范围
    print(f"\nAcceleration range (SG filtered):")
    print(f"  Ax: [{all_obs[:,:,4].min():.2f}, {all_obs[:,:,4].max():.2f}] m/s²")
    print(f"  Ay: [{all_obs[:,:,5].min():.2f}, {all_obs[:,:,5].max():.2f}] m/s²")
    
    print("=" * 60)
    print("Preprocessing complete!")
    print(f"\nNext step: Train the model with:")
    print(f"  python3 train_teacher.py --data_path {args.output}")


if __name__ == '__main__':
    main()
