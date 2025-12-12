#!/usr/bin/env python3
"""
对比原方法(A*)和新方法(GRU)的性能
"""

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray

class PerformanceComparison:
    def __init__(self):
        rospy.init_node('performance_comparison')
        
        # 记录真实轨迹
        self.ground_truth = []
        rospy.Subscriber('/target_ekf_node/target_odom', Odometry, self.gt_callback)
        
        # 记录GRU预测
        self.gru_predictions = []
        rospy.Subscriber('/gru_prediction', PoseArray, self.gru_callback)
        
    def gt_callback(self, msg):
        pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        self.ground_truth.append(pos)
        
    def gru_callback(self, msg):
        pred = np.array([[p.position.x, p.position.y] for p in msg.poses])
        self.gru_predictions.append(pred)
        
    def evaluate(self):
        if len(self.gru_predictions) < 10:
            rospy.logwarn("Not enough data")
            return
        
        # 计算ADE/FDE
        ade_errors = []
        fde_errors = []
        
        for i in range(min(len(self.gru_predictions), len(self.ground_truth) - 60)):
            pred = self.gru_predictions[i]
            gt = np.array(self.ground_truth[i+1:i+61])  # 未来60步
            
            if len(gt) != 60:
                continue
            
            # 计算误差
            errors = np.linalg.norm(pred - gt, axis=1)
            ade_errors.append(errors.mean())
            fde_errors.append(errors[-1])
        
        print(f"\n{'='*50}")
        print(f"Performance Evaluation")
        print(f"{'='*50}")
        print(f"ADE (Average Displacement Error): {np.mean(ade_errors):.4f} m")
        print(f"FDE (Final Displacement Error): {np.mean(fde_errors):.4f} m")
        print(f"{'='*50}\n")

if __name__ == '__main__':
    comp = PerformanceComparison()
    rospy.sleep(30)  # 收集30秒数据
    comp.evaluate()