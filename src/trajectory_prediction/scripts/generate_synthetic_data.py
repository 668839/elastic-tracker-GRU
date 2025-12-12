#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_synthetic_data_v2.py

改进版模拟轨迹数据生成器
- 更多轨迹类型
- 更大的参数随机范围
- 组合轨迹（多段拼接）
- 更真实的地面目标运动模式

输出格式: timestamp, x, y, z, vx, vy, vz, trajectory_id
"""

import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm


def generate_linear(duration, dt, speed, direction, start_pos):
    """直线运动"""
    n_steps = int(duration / dt)
    t = np.arange(n_steps) * dt
    
    vx = speed * np.cos(direction)
    vy = speed * np.sin(direction)
    
    x = start_pos[0] + vx * t
    y = start_pos[1] + vy * t
    
    return t, x, y, np.full(n_steps, vx), np.full(n_steps, vy)


def generate_circular(duration, dt, radius, omega, center, start_angle):
    """圆弧运动"""
    n_steps = int(duration / dt)
    t = np.arange(n_steps) * dt
    
    angles = start_angle + omega * t
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    
    vx = -radius * omega * np.sin(angles)
    vy = radius * omega * np.cos(angles)
    
    return t, x, y, vx, vy


def generate_sinusoidal(duration, dt, forward_speed, amplitude, frequency, direction, start_pos):
    """S形曲线"""
    n_steps = int(duration / dt)
    t = np.arange(n_steps) * dt
    
    # 沿主方向前进 + 横向摆动
    forward = forward_speed * t
    lateral = amplitude * np.sin(2 * np.pi * frequency * t)
    
    cos_d, sin_d = np.cos(direction), np.sin(direction)
    x = start_pos[0] + forward * cos_d - lateral * sin_d
    y = start_pos[1] + forward * sin_d + lateral * cos_d
    
    # 速度
    v_forward = forward_speed
    v_lateral = amplitude * 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * t)
    
    vx = v_forward * cos_d - v_lateral * sin_d
    vy = v_forward * sin_d + v_lateral * cos_d
    
    return t, x, y, vx, vy


def generate_acceleration(duration, dt, initial_speed, acceleration, direction, start_pos):
    """加速/减速运动"""
    n_steps = int(duration / dt)
    t = np.arange(n_steps) * dt
    
    # v = v0 + at, s = v0*t + 0.5*a*t^2
    speed = initial_speed + acceleration * t
    speed = np.maximum(speed, 0.1)  # 最小速度
    
    displacement = initial_speed * t + 0.5 * acceleration * t**2
    
    x = start_pos[0] + displacement * np.cos(direction)
    y = start_pos[1] + displacement * np.sin(direction)
    
    vx = speed * np.cos(direction)
    vy = speed * np.sin(direction)
    
    return t, x, y, vx, vy


def generate_random_walk(duration, dt, base_speed, turn_rate, start_pos):
    """随机游走（平滑转向）"""
    n_steps = int(duration / dt)
    t = np.arange(n_steps) * dt
    
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    vx = np.zeros(n_steps)
    vy = np.zeros(n_steps)
    
    x[0], y[0] = start_pos
    direction = np.random.uniform(0, 2*np.pi)
    speed = base_speed
    
    for i in range(1, n_steps):
        # 平滑转向
        direction += np.random.normal(0, turn_rate * dt)
        speed = base_speed * (1 + 0.3 * np.random.randn())
        speed = np.clip(speed, 0.2, base_speed * 2)
        
        vx[i-1] = speed * np.cos(direction)
        vy[i-1] = speed * np.sin(direction)
        
        x[i] = x[i-1] + vx[i-1] * dt
        y[i] = y[i-1] + vy[i-1] * dt
    
    vx[-1], vy[-1] = vx[-2], vy[-2]
    
    return t, x, y, vx, vy


def generate_sudden_turn(duration, dt, speed, initial_dir, turn_angle, turn_time, start_pos):
    """突然转向"""
    n_steps = int(duration / dt)
    turn_step = int(turn_time / dt)
    t = np.arange(n_steps) * dt
    
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    vx = np.zeros(n_steps)
    vy = np.zeros(n_steps)
    
    x[0], y[0] = start_pos
    
    for i in range(1, n_steps):
        if i < turn_step:
            direction = initial_dir
        else:
            # 平滑过渡
            progress = min(1.0, (i - turn_step) / 10)
            direction = initial_dir + turn_angle * progress
        
        vx[i-1] = speed * np.cos(direction)
        vy[i-1] = speed * np.sin(direction)
        x[i] = x[i-1] + vx[i-1] * dt
        y[i] = y[i-1] + vy[i-1] * dt
    
    vx[-1], vy[-1] = vx[-2], vy[-2]
    
    return t, x, y, vx, vy


def generate_stop_and_go(duration, dt, speed, direction, stop_times, stop_duration, start_pos):
    """走走停停"""
    n_steps = int(duration / dt)
    t = np.arange(n_steps) * dt
    
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    vx = np.zeros(n_steps)
    vy = np.zeros(n_steps)
    
    x[0], y[0] = start_pos
    
    for i in range(1, n_steps):
        current_time = i * dt
        
        # 检查是否在停止期间
        is_stopped = False
        for stop_time in stop_times:
            if stop_time <= current_time < stop_time + stop_duration:
                is_stopped = True
                break
        
        if is_stopped:
            vx[i-1], vy[i-1] = 0, 0
        else:
            vx[i-1] = speed * np.cos(direction)
            vy[i-1] = speed * np.sin(direction)
        
        x[i] = x[i-1] + vx[i-1] * dt
        y[i] = y[i-1] + vy[i-1] * dt
    
    vx[-1], vy[-1] = vx[-2], vy[-2]
    
    return t, x, y, vx, vy


def generate_spiral(duration, dt, initial_radius, expansion_rate, omega, center):
    """螺旋线（逐渐扩大/缩小的圆）"""
    n_steps = int(duration / dt)
    t = np.arange(n_steps) * dt
    
    radius = initial_radius + expansion_rate * t
    radius = np.maximum(radius, 0.5)
    
    angles = omega * t
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    
    # 速度（近似）
    vx = -radius * omega * np.sin(angles) + expansion_rate * np.cos(angles)
    vy = radius * omega * np.cos(angles) + expansion_rate * np.sin(angles)
    
    return t, x, y, vx, vy


def generate_figure_eight(duration, dt, size, omega, center):
    """8字形轨迹"""
    n_steps = int(duration / dt)
    t = np.arange(n_steps) * dt
    
    # 参数方程: x = sin(t), y = sin(2t)/2
    x = center[0] + size * np.sin(omega * t)
    y = center[1] + size * np.sin(2 * omega * t) / 2
    
    vx = size * omega * np.cos(omega * t)
    vy = size * omega * np.cos(2 * omega * t)
    
    return t, x, y, vx, vy


def generate_composite_trajectory(duration, dt, start_pos):
    """组合轨迹：随机拼接2-3种运动模式"""
    n_segments = np.random.randint(2, 4)
    segment_duration = duration / n_segments
    
    all_t, all_x, all_y, all_vx, all_vy = [], [], [], [], []
    current_pos = start_pos.copy()
    current_time = 0
    
    for seg in range(n_segments):
        # 随机选择运动类型
        motion_type = np.random.choice(['linear', 'circular', 'sinusoidal', 'acceleration'])
        
        if motion_type == 'linear':
            t, x, y, vx, vy = generate_linear(
                segment_duration, dt,
                speed=np.random.uniform(0.5, 2.5),
                direction=np.random.uniform(0, 2*np.pi),
                start_pos=current_pos
            )
        elif motion_type == 'circular':
            t, x, y, vx, vy = generate_circular(
                segment_duration, dt,
                radius=np.random.uniform(1, 4),
                omega=np.random.uniform(0.3, 1.0) * np.random.choice([-1, 1]),
                center=current_pos,
                start_angle=np.random.uniform(0, 2*np.pi)
            )
        elif motion_type == 'sinusoidal':
            t, x, y, vx, vy = generate_sinusoidal(
                segment_duration, dt,
                forward_speed=np.random.uniform(0.5, 2.0),
                amplitude=np.random.uniform(0.5, 2.0),
                frequency=np.random.uniform(0.1, 0.4),
                direction=np.random.uniform(0, 2*np.pi),
                start_pos=current_pos
            )
        else:  # acceleration
            t, x, y, vx, vy = generate_acceleration(
                segment_duration, dt,
                initial_speed=np.random.uniform(0.5, 1.5),
                acceleration=np.random.uniform(-0.3, 0.5),
                direction=np.random.uniform(0, 2*np.pi),
                start_pos=current_pos
            )
        
        # 调整时间
        t = t + current_time
        current_time = t[-1] + dt
        current_pos = np.array([x[-1], y[-1]])
        
        all_t.append(t)
        all_x.append(x)
        all_y.append(y)
        all_vx.append(vx)
        all_vy.append(vy)
    
    return (np.concatenate(all_t), np.concatenate(all_x), np.concatenate(all_y),
            np.concatenate(all_vx), np.concatenate(all_vy))


def add_noise(x, y, vx, vy, pos_noise=0.02, vel_noise=0.05):
    """添加测量噪声"""
    x = x + np.random.normal(0, pos_noise, len(x))
    y = y + np.random.normal(0, pos_noise, len(y))
    vx = vx + np.random.normal(0, vel_noise, len(vx))
    vy = vy + np.random.normal(0, vel_noise, len(vy))
    return x, y, vx, vy


def main():
    parser = argparse.ArgumentParser(description='Generate diverse synthetic trajectory data')
    parser.add_argument('--output', type=str, default='data/raw_trajectories.csv')
    parser.add_argument('--num_trajectories', type=int, default=1000)
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--sample_rate', type=float, default=30.0)
    parser.add_argument('--noise', type=float, default=0.02)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    dt = 1.0 / args.sample_rate
    
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # 轨迹类型及其权重（增加多样性）
    trajectory_types = [
        ('linear', 0.15),
        ('circular', 0.12),
        ('sinusoidal', 0.12),
        ('acceleration', 0.10),
        ('random_walk', 0.15),
        ('sudden_turn', 0.10),
        ('stop_and_go', 0.08),
        ('spiral', 0.06),
        ('figure_eight', 0.04),
        ('composite', 0.08),
    ]
    types, weights = zip(*trajectory_types)
    weights = np.array(weights)
    weights /= weights.sum()
    
    all_data = []
    trajectory_id = 0
    base_timestamp = 0.0
    
    # 统计每种类型生成的数量
    type_counts = {t: 0 for t in types}
    
    print(f"Generating {args.num_trajectories} diverse trajectories...")
    print(f"Types: {types}")
    
    for i in tqdm(range(args.num_trajectories)):
        # 随机起始位置（更大范围）
        start_pos = np.random.uniform(-10, 10, size=2)
        
        # 随机选择轨迹类型
        traj_type = np.random.choice(types, p=weights)
        type_counts[traj_type] += 1
        
        # 随机参数范围更大
        if traj_type == 'linear':
            t, x, y, vx, vy = generate_linear(
                args.duration, dt,
                speed=np.random.uniform(0.3, 3.0),  # 更大速度范围
                direction=np.random.uniform(0, 2*np.pi),
                start_pos=start_pos
            )
        elif traj_type == 'circular':
            t, x, y, vx, vy = generate_circular(
                args.duration, dt,
                radius=np.random.uniform(1, 6),  # 更大半径范围
                omega=np.random.uniform(0.2, 1.2) * np.random.choice([-1, 1]),
                center=start_pos,
                start_angle=np.random.uniform(0, 2*np.pi)
            )
        elif traj_type == 'sinusoidal':
            t, x, y, vx, vy = generate_sinusoidal(
                args.duration, dt,
                forward_speed=np.random.uniform(0.3, 2.5),
                amplitude=np.random.uniform(0.5, 3.0),
                frequency=np.random.uniform(0.08, 0.5),
                direction=np.random.uniform(0, 2*np.pi),
                start_pos=start_pos
            )
        elif traj_type == 'acceleration':
            t, x, y, vx, vy = generate_acceleration(
                args.duration, dt,
                initial_speed=np.random.uniform(0.3, 2.0),
                acceleration=np.random.uniform(-0.5, 0.8),
                direction=np.random.uniform(0, 2*np.pi),
                start_pos=start_pos
            )
        elif traj_type == 'random_walk':
            t, x, y, vx, vy = generate_random_walk(
                args.duration, dt,
                base_speed=np.random.uniform(0.5, 2.0),
                turn_rate=np.random.uniform(0.5, 2.0),
                start_pos=start_pos
            )
        elif traj_type == 'sudden_turn':
            t, x, y, vx, vy = generate_sudden_turn(
                args.duration, dt,
                speed=np.random.uniform(0.5, 2.0),
                initial_dir=np.random.uniform(0, 2*np.pi),
                turn_angle=np.random.uniform(np.pi/6, 5*np.pi/6) * np.random.choice([-1, 1]),
                turn_time=np.random.uniform(2, 7),
                start_pos=start_pos
            )
        elif traj_type == 'stop_and_go':
            num_stops = np.random.randint(1, 4)
            stop_times = sorted(np.random.uniform(1, args.duration-2, num_stops))
            t, x, y, vx, vy = generate_stop_and_go(
                args.duration, dt,
                speed=np.random.uniform(0.5, 2.0),
                direction=np.random.uniform(0, 2*np.pi),
                stop_times=stop_times,
                stop_duration=np.random.uniform(0.5, 1.5),
                start_pos=start_pos
            )
        elif traj_type == 'spiral':
            t, x, y, vx, vy = generate_spiral(
                args.duration, dt,
                initial_radius=np.random.uniform(1, 3),
                expansion_rate=np.random.uniform(-0.3, 0.5),
                omega=np.random.uniform(0.3, 1.0) * np.random.choice([-1, 1]),
                center=start_pos
            )
        elif traj_type == 'figure_eight':
            t, x, y, vx, vy = generate_figure_eight(
                args.duration, dt,
                size=np.random.uniform(2, 5),
                omega=np.random.uniform(0.3, 0.8),
                center=start_pos
            )
        else:  # composite
            t, x, y, vx, vy = generate_composite_trajectory(
                args.duration, dt,
                start_pos=start_pos
            )
        
        # 添加噪声
        x, y, vx, vy = add_noise(x, y, vx, vy, args.noise, args.noise * 2)
        
        # 调整时间戳
        t = t + base_timestamp
        
        # 组合数据
        n_points = len(t)
        for j in range(n_points):
            all_data.append({
                'timestamp': t[j],
                'x': x[j],
                'y': y[j],
                'z': 0.0,
                'vx': vx[j],
                'vy': vy[j],
                'vz': 0.0,
                'trajectory_id': trajectory_id
            })
        
        trajectory_id += 1
        base_timestamp = t[-1] + 1.0
    
    # 保存
    df = pd.DataFrame(all_data)
    df.to_csv(args.output, index=False)
    
    # 统计
    print(f"\n{'='*60}")
    print(f"Generated {trajectory_id} trajectories, {len(all_data)} total points")
    print(f"Saved to {args.output}")
    
    print(f"\nTrajectory type distribution:")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count} ({100*count/trajectory_id:.1f}%)")
    
    print(f"\nData statistics:")
    print(f"  X range: [{df['x'].min():.2f}, {df['x'].max():.2f}]")
    print(f"  Y range: [{df['y'].min():.2f}, {df['y'].max():.2f}]")
    speed = np.sqrt(df['vx']**2 + df['vy']**2)
    print(f"  Speed range: [{speed.min():.2f}, {speed.max():.2f}] m/s")
    print(f"  Mean speed: {speed.mean():.2f} m/s")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
