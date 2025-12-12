#!/usr/bin/env python3
"""
卡尔曼融合节点 - 融合GRU预测和Kalman预测
"""

import rospy
import numpy as np
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float32MultiArray
from filterpy.kalman import KalmanFilter

class KalmanFusionNode:
    def __init__(self):
        rospy.init_node('kalman_fusion_node', anonymous=False)
        
        # 参数
        self.dt = 1.0 / 30.0  # 30Hz
        self.base_R = rospy.get_param('~base_measurement_noise', 0.3)
        self.alpha = rospy.get_param('~fusion_alpha', 0.7)
        self.beta = rospy.get_param('~fusion_beta', 0.5)
        
        # 卡尔曼滤波器
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.setup_kalman()
        
        # 缓存
        self.gru_pred = None
        self.uncertainty = None
        
        # 订阅
        rospy.Subscriber('/gru_prediction', PoseArray, self.gru_callback)
        rospy.Subscriber('/gru_uncertainty', Float32MultiArray, self.uncertainty_callback)
        
        # 发布
        self.fused_pub = rospy.Publisher('/fused_prediction', PoseArray, queue_size=1)
        
        rospy.loginfo("Kalman Fusion Node ready")
    
    def setup_kalman(self):
        """初始化卡尔曼滤波器"""
        dt = self.dt
        
        # 状态转移矩阵（恒速模型）
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # 观测矩阵
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # 过程噪声
        q = 0.1 ** 2
        self.kf.Q = np.array([
            [q*dt**4/4, 0, q*dt**3/2, 0],
            [0, q*dt**4/4, 0, q*dt**3/2],
            [q*dt**3/2, 0, q*dt**2, 0],
            [0, q*dt**3/2, 0, q*dt**2]
        ])
        
        # 测量噪声（会动态调整）
        self.kf.R = np.eye(2) * self.base_R ** 2
        self.kf.P *= 10
    
    def gru_callback(self, msg):
        """接收GRU预测"""
        self.gru_pred = np.array([[p.position.x, p.position.y] for p in msg.poses])
    
    def uncertainty_callback(self, msg):
        """接收不确定性"""
        self.uncertainty = np.array(msg.data)
        
        # 当收到新的不确定性时，执行融合
        if self.gru_pred is not None:
            self.fuse_and_publish()
    
    def fuse_and_publish(self):
        """融合GRU和Kalman预测"""
        T = len(self.gru_pred)
        fused_trajectory = np.zeros((T, 2))
        
        # 初始化卡尔曼状态
        self.kf.x = np.array([self.gru_pred[0, 0], self.gru_pred[0, 1], 0, 0])
        
        for t in range(T):
            # 卡尔曼预测
            self.kf.predict()
            kalman_pred = self.kf.x[:2]
            
            # 动态融合权重
            w_gru = self.alpha * np.exp(-self.beta * self.uncertainty[t])
            w_kalman = 1 - w_gru
            
            # 融合
            fused_pred = w_gru * self.gru_pred[t] + w_kalman * kalman_pred
            
            # 动态调整测量噪声
            adaptive_R = self.base_R ** 2 * (1 + 2 * self.uncertainty[t])
            self.kf.R = np.eye(2) * adaptive_R
            
            # 卡尔曼更新
            self.kf.update(fused_pred)
            
            fused_trajectory[t] = fused_pred
        
        # 发布
        msg = PoseArray()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        
        for pos in fused_trajectory:
            from geometry_msgs.msg import Pose
            pose = Pose()
            pose.position.x = pos[0]
            pose.position.y = pos[1]
            pose.position.z = 0.0
            pose.orientation.w = 1.0
            msg.poses.append(pose)
        
        self.fused_pub.publish(msg)

if __name__ == '__main__':
    try:
        node = KalmanFusionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass