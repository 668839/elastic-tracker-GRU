#!/usr/bin/env python3
"""
Gazebo数据收集 - 速度判断版本
✅ 改用速度判断静止状态
✅ 不会误判缓慢移动
"""

import rospy
import numpy as np
import pandas as pd
from gazebo_msgs.msg import ModelStates
import os
from datetime import datetime
import glob

class GazeboDataCollector:
    def __init__(self, output_dir='./gazebo_data', target_name='actor1'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.target_name = target_name
        self.target_confirmed = False
        self.target_idx = None
        
        self.trajectories = []
        self.current_traj = {
            'time': [], 'x': [], 'y': [], 'z': [],
            'vx': [], 'vy': [], 'vz': []
        }
        
        # ═══════════════════════════════════════════════════════════
        # ✅ 修改：改用速度阈值判断静止
        # ═══════════════════════════════════════════════════════════
        self.STATIC_SPEED_THRESHOLD = 0.05  # 速度<5cm/s算静止
        self.STATIC_DURATION = 5.0
        self.MIN_POINTS = 150
        self.MIN_DISPLACEMENT = 2.0
        self.MIN_DURATION = 5.0
        self.MAX_SPEED = 8.0
        # ═══════════════════════════════════════════════════════════
        
        self.last_time = None
        self.last_pos = None
        self.static_start_time = None
        self.is_static = False
        self.last_saved_time = None
        self.min_interval_between_saves = 3.0
        self.skip_recording_until = None
        
        self.last_pose_for_velocity = None
        self.last_time_for_velocity = None
        
        self.scenario_id = self._get_next_scenario_id()
        self.start_id = self.scenario_id
        
        rospy.init_node('gazebo_data_collector', anonymous=True)
        
        self.model_sub = rospy.Subscriber(
            '/gazebo/model_states',
            ModelStates,
            self.model_callback
        )
        
        rospy.loginfo("="*60)
        rospy.loginfo("Gazebo Data Collector - Speed-based Detection")
        rospy.loginfo("="*60)
        rospy.loginfo(f"Target: {self.target_name}")
        rospy.loginfo(f"Output: {self.output_dir}")
        rospy.loginfo(f"Requirements:")
        rospy.loginfo(f"  Move at least {self.MIN_DURATION}s")
        rospy.loginfo(f"  Move at least {self.MIN_DISPLACEMENT}m")
        rospy.loginfo(f"  Stop (speed < {self.STATIC_SPEED_THRESHOLD}m/s) for {self.STATIC_DURATION}s")
        rospy.loginfo(f"Waiting for {self.target_name}...")
        
        rate = rospy.Rate(10)
        timeout = rospy.Time.now() + rospy.Duration(10)
        
        while not self.target_confirmed and not rospy.is_shutdown():
            if rospy.Time.now() > timeout:
                rospy.logerr(f"TIMEOUT: {self.target_name} not found!")
                rospy.signal_shutdown("Target not found")
                return
            rate.sleep()
        
        rospy.loginfo(f"✅ {self.target_name} confirmed at index {self.target_idx}")
        rospy.loginfo(f"Starting from ID: {self.scenario_id}")
        rospy.loginfo("="*60)
    
    def _get_next_scenario_id(self):
        existing_files = glob.glob(f"{self.output_dir}/traj_*.csv")
        if not existing_files:
            return 0
        ids = []
        for f in existing_files:
            try:
                basename = os.path.basename(f)
                id_str = basename.replace('traj_', '').replace('.csv', '')
                ids.append(int(id_str))
            except ValueError:
                continue
        return max(ids) + 1 if ids else 0
    
    def model_callback(self, msg):
        if not self.target_confirmed:
            try:
                self.target_idx = msg.name.index(self.target_name)
                self.target_confirmed = True
                pose = msg.pose[self.target_idx]
                rospy.loginfo(f"Found {self.target_name} at index {self.target_idx}")
                rospy.loginfo(f"Initial position: ({pose.position.x:.2f}, {pose.position.y:.2f}, {pose.position.z:.2f})")
            except ValueError:
                return
            return
        
        current_time = rospy.Time.now().to_sec()
        
        try:
            pose = msg.pose[self.target_idx]
        except IndexError:
            return
        
        x = pose.position.x
        y = pose.position.y
        z = pose.position.z
        
        # 跳过记录逻辑
        if self.skip_recording_until is not None:
            if current_time < self.skip_recording_until:
                self.last_time = current_time
                self.last_pos = (x, y, z)
                self.last_pose_for_velocity = (x, y, z)
                self.last_time_for_velocity = current_time
                return
            else:
                rospy.loginfo(">>> Starting new trajectory <<<")
                self.skip_recording_until = None
        
        # 计算速度
        if self.last_pose_for_velocity is not None and self.last_time_for_velocity is not None:
            dt = current_time - self.last_time_for_velocity
            if dt > 0:
                vx = (x - self.last_pose_for_velocity[0]) / dt
                vy = (y - self.last_pose_for_velocity[1]) / dt
                vz = (z - self.last_pose_for_velocity[2]) / dt
            else:
                vx, vy, vz = 0.0, 0.0, 0.0
        else:
            vx, vy, vz = 0.0, 0.0, 0.0
        
        self.last_pose_for_velocity = (x, y, z)
        self.last_time_for_velocity = current_time
        
        # ═══════════════════════════════════════════════════════════
        # ✅ 核心修改：改用速度判断静止
        # ═══════════════════════════════════════════════════════════
        
        # 计算当前速度（水平方向）
        current_speed = np.sqrt(vx**2 + vy**2)
        
        if current_speed < self.STATIC_SPEED_THRESHOLD:
            # 速度很低，判定为静止
            if not self.is_static:
                self.static_start_time = current_time
                self.is_static = True
                rospy.loginfo(
                    f"Target stopped (speed: {current_speed:.3f}m/s, "
                    f"traj points: {len(self.current_traj['time'])})"
                )
            else:
                static_duration = current_time - self.static_start_time
                
                can_save = (
                    static_duration > self.STATIC_DURATION and
                    len(self.current_traj['time']) >= self.MIN_POINTS
                )
                
                if can_save and self.last_saved_time is not None:
                    time_since_last_save = current_time - self.last_saved_time
                    if time_since_last_save < self.min_interval_between_saves:
                        can_save = False
                
                if can_save:
                    save_success = self.save_current_trajectory()
                    
                    if save_success:
                        rospy.loginfo("✓ Trajectory saved successfully")
                        self.current_traj = {
                            'time': [], 'x': [], 'y': [], 'z': [],
                            'vx': [], 'vy': [], 'vz': []
                        }
                        self.last_saved_time = current_time
                        self.skip_recording_until = current_time + 1.0
                        self.is_static = False
                        self.static_start_time = None
                    else:
                        rospy.logwarn("✗ Trajectory doesn't meet requirements. Clearing.")
                        self.current_traj = {
                            'time': [], 'x': [], 'y': [], 'z': [],
                            'vx': [], 'vy': [], 'vz': []
                        }
                        self.is_static = False
                        self.static_start_time = None
        else:
            # 速度正常，判定为移动中
            if self.is_static:
                rospy.loginfo(
                    f"Target moving (speed: {current_speed:.3f}m/s, "
                    f"traj points: {len(self.current_traj['time'])})"
                )
            
            self.is_static = False
            self.static_start_time = None
        
        # ═══════════════════════════════════════════════════════════
        
        # 记录数据
        self.current_traj['time'].append(current_time)
        self.current_traj['x'].append(x)
        self.current_traj['y'].append(y)
        self.current_traj['z'].append(z)
        self.current_traj['vx'].append(vx)
        self.current_traj['vy'].append(vy)
        self.current_traj['vz'].append(vz)
        
        self.last_time = current_time
        self.last_pos = (x, y, z)
    
    def save_current_trajectory(self):
        if len(self.current_traj['time']) < self.MIN_POINTS:
            rospy.logwarn(f"Too short: {len(self.current_traj['time'])} pts < {self.MIN_POINTS}")
            return False
        
        df = pd.DataFrame(self.current_traj)
        df['time'] = df['time'] - df['time'].iloc[0]
        
        duration = df['time'].max()
        if duration < self.MIN_DURATION:
            rospy.logwarn(f"Duration too short: {duration:.1f}s < {self.MIN_DURATION}s")
            return False
        
        speed = np.sqrt(df['vx']**2 + df['vy']**2).mean()
        displacement = np.sqrt(
            (df['x'].iloc[-1] - df['x'].iloc[0])**2 + 
            (df['y'].iloc[-1] - df['y'].iloc[0])**2
        )
        
        max_speed = np.sqrt(df['vx']**2 + df['vy']**2).max()
        
        if displacement < self.MIN_DISPLACEMENT:
            rospy.logwarn(f"Displacement too small: {displacement:.2f}m < {self.MIN_DISPLACEMENT}m")
            return False
        
        if max_speed > self.MAX_SPEED:
            rospy.logwarn(f"Max speed too high: {max_speed:.2f}m/s")
            return False
        
        # 计算线性度
        path_length = 0.0
        for i in range(1, len(df)):
            dx = df['x'].iloc[i] - df['x'].iloc[i-1]
            dy = df['y'].iloc[i] - df['y'].iloc[i-1]
            path_length += np.sqrt(dx**2 + dy**2)
        
        straightness = displacement / path_length if path_length > 0 else 0.0
        
        if straightness > 0.95:
            traj_type = "straight"
        elif straightness > 0.7:
            traj_type = "slight_curve"
        elif straightness > 0.5:
            traj_type = "curve"
        else:
            traj_type = "complex"
        
        filename = f"{self.output_dir}/traj_{self.scenario_id:04d}.csv"
        df.to_csv(filename, index=False)
        
        rospy.loginfo(
            f"✓ Saved traj_{self.scenario_id:04d} | "
            f"Pts: {len(df)} | "
            f"Dur: {duration:.1f}s | "
            f"Spd: {speed:.2f}m/s | "
            f"Dist: {displacement:.1f}m | "
            f"Straight: {straightness:.2f} ({traj_type})"
        )
        
        self.scenario_id += 1
        self.trajectories.append(filename)
        
        return True
    
    def run(self, duration_minutes=60):
        if not self.target_confirmed:
            rospy.logerr("Cannot start: target not confirmed")
            return
        
        rospy.loginfo("Collection started! Control actor with keyboard.")
        rospy.loginfo(">>> Starting new trajectory <<<")
        
        end_time = rospy.Time.now() + rospy.Duration(duration_minutes * 60)
        rate = rospy.Rate(30)
        
        try:
            while not rospy.is_shutdown() and rospy.Time.now() < end_time:
                rate.sleep()
        except KeyboardInterrupt:
            rospy.loginfo("Interrupted by user")
        
        if len(self.current_traj['time']) >= self.MIN_POINTS:
            rospy.loginfo("Saving final trajectory...")
            self.save_current_trajectory()
        
        self.generate_report()
    
    def generate_report(self):
        report_path = f"{self.output_dir}/collection_report.txt"
        all_files = sorted(glob.glob(f"{self.output_dir}/traj_*.csv"))
        this_session = self.scenario_id - self.start_id
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("Data Collection Report\n")
            f.write("="*60 + "\n")
            f.write(f"Collection time: {datetime.now()}\n")
            f.write(f"Target: {self.target_name}\n")
            f.write(f"\nTotal trajectories: {len(all_files)}\n")
            f.write(f"This session: {this_session}\n")
            
            if this_session > 0:
                durations = []
                speeds = []
                displacements = []
                straightnesses = []
                
                for i in range(self.start_id, self.scenario_id):
                    filename = f"{self.output_dir}/traj_{i:04d}.csv"
                    if os.path.exists(filename):
                        df = pd.read_csv(filename)
                        durations.append(df['time'].max())
                        speed = np.sqrt(df['vx']**2 + df['vy']**2).mean()
                        speeds.append(speed)
                        disp = np.sqrt(
                            (df['x'].iloc[-1] - df['x'].iloc[0])**2 + 
                            (df['y'].iloc[-1] - df['y'].iloc[0])**2
                        )
                        displacements.append(disp)
                        
                        path_length = 0.0
                        for j in range(1, len(df)):
                            dx = df['x'].iloc[j] - df['x'].iloc[j-1]
                            dy = df['y'].iloc[j] - df['y'].iloc[j-1]
                            path_length += np.sqrt(dx**2 + dy**2)
                        straightness = disp / path_length if path_length > 0 else 0
                        straightnesses.append(straightness)
                
                f.write(f"\nData Quality:\n")
                f.write(f"  Avg duration: {np.mean(durations):.2f}s\n")
                f.write(f"  Avg speed: {np.mean(speeds):.2f} m/s\n")
                f.write(f"  Avg displacement: {np.mean(displacements):.2f}m\n")
                f.write(f"  Avg straightness: {np.mean(straightnesses):.2f}\n")
                f.write(f"\nTrajectory types:\n")
                f.write(f"  Straight (>0.95): {sum(1 for s in straightnesses if s > 0.95)}\n")
                f.write(f"  Slight curve (0.7-0.95): {sum(1 for s in straightnesses if 0.7 < s <= 0.95)}\n")
                f.write(f"  Curve (0.5-0.7): {sum(1 for s in straightnesses if 0.5 < s <= 0.7)}\n")
                f.write(f"  Complex (<0.5): {sum(1 for s in straightnesses if s <= 0.5)}\n")
        
        rospy.loginfo("="*60)
        rospy.loginfo("✓ Collection complete!")
        rospy.loginfo(f"✓ Total: {len(all_files)} trajectories")
        rospy.loginfo(f"✓ This session: {this_session}")
        rospy.loginfo("="*60)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Collect trajectory data from Gazebo'
    )
    parser.add_argument('--output_dir', default='./gazebo_data')
    parser.add_argument('--target_name', default='actor1')
    parser.add_argument('--duration', type=int, default=60)
    args = parser.parse_args()
    
    try:
        collector = GazeboDataCollector(
            output_dir=args.output_dir,
            target_name=args.target_name
        )
        collector.run(duration_minutes=args.duration)
    except rospy.ROSInterruptException:
        pass

