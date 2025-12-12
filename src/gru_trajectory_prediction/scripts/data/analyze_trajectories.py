#!/usr/bin/env python3
"""
分析轨迹类型
"""

import pandas as pd
import numpy as np
import glob
import os

def classify_trajectory(csv_file):
    """分类轨迹"""
    df = pd.read_csv(csv_file)
    
    # 计算位移
    displacement = np.sqrt(
        (df['x'].iloc[-1] - df['x'].iloc[0])**2 + 
        (df['y'].iloc[-1] - df['y'].iloc[0])**2
    )
    
    # 计算轨迹长度（实际路径）
    dx = np.diff(df['x'])
    dy = np.diff(df['y'])
    path_length = np.sum(np.sqrt(dx**2 + dy**2))
    
    # 直线度：位移/路径长度，越接近1越直
    straightness = displacement / (path_length + 1e-6)
    
    # 平均速度
    speed = np.sqrt(df['vx']**2 + df['vy']**2).mean()
    
    # 角速度（检测转弯）
    angles = np.arctan2(df['vy'], df['vx'])
    angular_velocity = np.abs(np.diff(angles)).mean()
    
    # 分类
    if straightness > 0.9:
        if speed < 0.8:
            return "straight_slow", straightness, speed, angular_velocity
        elif speed < 1.5:
            return "straight_medium", straightness, speed, angular_velocity
        else:
            return "straight_fast", straightness, speed, angular_velocity
    elif angular_velocity > 0.1:
        return "sharp_turn", straightness, speed, angular_velocity
    elif angular_velocity > 0.05:
        return "smooth_turn", straightness, speed, angular_velocity
    else:
        return "complex", straightness, speed, angular_velocity

def analyze_all(data_dir):
    """分析所有轨迹"""
    files = sorted(glob.glob(f"{data_dir}/traj_*.csv"))
    
    stats = {
        'straight_slow': 0,
        'straight_medium': 0,
        'straight_fast': 0,
        'sharp_turn': 0,
        'smooth_turn': 0,
        'complex': 0
    }
    
    print("="*60)
    print("Trajectory Classification")
    print("="*60)
    
    for f in files:
        trajectory_type, straightness, speed, angular_vel = classify_trajectory(f)
        stats[trajectory_type] += 1
        
        basename = os.path.basename(f)
        print(f"{basename}: {trajectory_type:15s} "
              f"(straight={straightness:.2f}, speed={speed:.2f}m/s, "
              f"ang_vel={angular_vel:.3f})")
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    total = sum(stats.values())
    for key, count in stats.items():
        pct = count / total * 100 if total > 0 else 0
        print(f"{key:20s}: {count:3d} ({pct:5.1f}%)")
    
    print(f"\nTotal: {total}")
    print("="*60)
    
    # 检查分布是否平衡
    if total > 50:
        max_count = max(stats.values())
        min_count = min([v for v in stats.values() if v > 0])
        imbalance = max_count / (min_count + 1e-6)
        
        if imbalance > 3:
            print(f"\n⚠️  WARNING: Data imbalance (ratio: {imbalance:.1f}x)")
            print(f"Collect more of the underrepresented scenarios")

if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else './gazebo_data'
    analyze_all(data_dir)