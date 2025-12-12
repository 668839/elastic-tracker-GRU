#!/usr/bin/env python3
"""
从Gazebo仿真收集轨迹数据 - 修复版
✅ 自动续接编号，不会覆盖已有数据
"""

import rospy
import numpy as np
import pandas as pd
from nav_msgs.msg import Odometry
import os
from datetime import datetime
import glob

class GazeboDataCollector:
    def __init__(self, output_dir='./gazebo_data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.trajectories = []
        self.current_traj = {
            'time': [], 'x': [], 'y': [], 'z': [],
            'vx': [], 'vy': [], 'vz': []
        }
        self.last_time = None
        
        # ✅ 修复：自动找到下一个可用的ID
        self.scenario_id = self._get_next_scenario_id()
        self.start_id = self.scenario_id  # 记录本次运行的起始ID
        
        rospy.init_node('gazebo_data_collector', anonymous=True)
        self.target_sub = rospy.Subscriber(
            '/target_ekf_node/target_odom',
            Odometry,
            self.target_callback
        )
        
        rospy.loginfo(f"Data collector initialized")
        rospy.loginfo(f"Output directory: {self.output_dir}")
        rospy.loginfo(f"Starting from trajectory ID: {self.scenario_id}")
    
    def _get_next_scenario_id(self):
        """✅ 自动找到下一个可用的ID（避免覆盖）"""
        existing_files = glob.glob(f"{self.output_dir}/traj_*.csv")
        
        if not existing_files:
            rospy.loginfo("No existing trajectories, starting from 0")
            return 0
        
        # 提取所有文件编号
        ids = []
        for f in existing_files:
            try:
                basename = os.path.basename(f)
                id_str = basename.replace('traj_', '').replace('.csv', '')
                ids.append(int(id_str))
            except ValueError:
                continue
        
        if ids:
            max_id = max(ids)
            next_id = max_id + 1
            rospy.loginfo(f"Found {len(ids)} existing trajectories (max ID: {max_id})")
            rospy.loginfo(f"Will continue from ID: {next_id}")
            return next_id
        else:
            return 0
    
    def target_callback(self, msg):
        current_time = msg.header.stamp.to_sec()
        
        # 检测场景切换（停止超过2秒 = 新轨迹）
        if self.last_time is not None:
            dt = current_time - self.last_time
            if dt > 2.0:
                rospy.loginfo(f"Gap detected ({dt:.2f}s), saving trajectory...")
                self.save_current_trajectory()
                self.current_traj = {
                    'time': [], 'x': [], 'y': [], 'z': [],
                    'vx': [], 'vy': [], 'vz': []
                }
        
        # 记录数据
        self.current_traj['time'].append(current_time)
        self.current_traj['x'].append(msg.pose.pose.position.x)
        self.current_traj['y'].append(msg.pose.pose.position.y)
        self.current_traj['z'].append(msg.pose.pose.position.z)
        self.current_traj['vx'].append(msg.twist.twist.linear.x)
        self.current_traj['vy'].append(msg.twist.twist.linear.y)
        self.current_traj['vz'].append(msg.twist.twist.linear.z)
        
        self.last_time = current_time
    
    def save_current_trajectory(self):
        if len(self.current_traj['time']) < 90:  # 至少3秒@30Hz
            rospy.logwarn(f"Trajectory too short ({len(self.current_traj['time'])} points), skipping")
            return
        
        df = pd.DataFrame(self.current_traj)
        
        # 时间归零
        df['time'] = df['time'] - df['time'].iloc[0]
        
        # 计算统计信息
        duration = df['time'].max()
        speed = np.sqrt(df['vx']**2 + df['vy']**2).mean()
        displacement = np.sqrt(
            (df['x'].iloc[-1] - df['x'].iloc[0])**2 + 
            (df['y'].iloc[-1] - df['y'].iloc[0])**2
        )
        
        # 保存
        filename = f"{self.output_dir}/traj_{self.scenario_id:04d}.csv"
        df.to_csv(filename, index=False)
        
        rospy.loginfo(
            f"✓ Saved trajectory {self.scenario_id} | "
            f"Points: {len(df)} | "
            f"Duration: {duration:.2f}s | "
            f"Speed: {speed:.2f} m/s | "
            f"Distance: {displacement:.2f}m"
        )
        
        self.scenario_id += 1
        self.trajectories.append(filename)
    
    def run(self, duration_minutes=60):
        rospy.loginfo("="*60)
        rospy.loginfo(f"Starting data collection for {duration_minutes} minutes")
        rospy.loginfo("="*60)
        rospy.loginfo("Instructions:")
        rospy.loginfo("  1. Move target in Gazebo (teleop_twist_keyboard)")
        rospy.loginfo("  2. Let it move 5-10 seconds")
        rospy.loginfo("  3. STOP for 2+ seconds (auto-save)")
        rospy.loginfo("  4. Repeat with different patterns:")
        rospy.loginfo("     - Straight lines (slow/medium/fast)")
        rospy.loginfo("     - Sharp turns")
        rospy.loginfo("     - Smooth curves")
        rospy.loginfo("     - Acceleration/deceleration")
        rospy.loginfo("     - S-curves, circles, random")
        rospy.loginfo("="*60)
        
        end_time = rospy.Time.now() + rospy.Duration(duration_minutes * 60)
        rate = rospy.Rate(30)
        
        try:
            while not rospy.is_shutdown() and rospy.Time.now() < end_time:
                rate.sleep()
        except KeyboardInterrupt:
            rospy.loginfo("Collection interrupted by user")
        
        # 保存最后一条轨迹
        self.save_current_trajectory()
        self.generate_report()
    
    def generate_report(self):
        report_path = f"{self.output_dir}/collection_report.txt"
        
        # 统计所有轨迹（包括之前的）
        all_files = sorted(glob.glob(f"{self.output_dir}/traj_*.csv"))
        this_session = self.scenario_id - self.start_id
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("Data Collection Report\n")
            f.write("="*60 + "\n")
            f.write(f"Collection time: {datetime.now()}\n")
            f.write(f"\n")
            f.write(f"Total trajectories in directory: {len(all_files)}\n")
            f.write(f"Collected this session: {this_session}\n")
            f.write(f"ID range this session: {self.start_id} - {self.scenario_id-1}\n")
            f.write(f"\n")
            
            # 数据质量统计
            if this_session > 0:
                durations = []
                speeds = []
                for i in range(self.start_id, self.scenario_id):
                    filename = f"{self.output_dir}/traj_{i:04d}.csv"
                    if os.path.exists(filename):
                        df = pd.read_csv(filename)
                        durations.append(df['time'].max())
                        speed = np.sqrt(df['vx']**2 + df['vy']**2).mean()
                        speeds.append(speed)
                
                f.write("Data Quality:\n")
                f.write(f"  Average duration: {np.mean(durations):.2f}s\n")
                f.write(f"  Average speed: {np.mean(speeds):.2f} m/s\n")
                f.write(f"  Speed range: {np.min(speeds):.2f} - {np.max(speeds):.2f} m/s\n")
                f.write(f"\n")
            
            f.write("All trajectory files:\n")
            for traj_file in all_files:
                f.write(f"  {traj_file}\n")
        
        rospy.loginfo("="*60)
        rospy.loginfo("✓ Collection complete!")
        rospy.loginfo(f"✓ Total trajectories in directory: {len(all_files)}")
        rospy.loginfo(f"✓ Collected this session: {this_session}")
        rospy.loginfo(f"✓ Report saved: {report_path}")
        rospy.loginfo("="*60)

if __name__ == '__main__':
    collector = GazeboDataCollector(output_dir='./gazebo_data')
    collector.run(duration_minutes=60)