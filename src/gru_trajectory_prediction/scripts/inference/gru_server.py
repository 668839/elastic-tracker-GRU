#!/usr/bin/env python3
"""
GRU推理服务 - Python ROS节点
提供轨迹预测服务给C++节点调用
"""

import rospy
import torch
import numpy as np
from collections import deque
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Float32MultiArray
import sys
import os

# 添加模型路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.student_gru import StudentGRU

class GRUPredictionServer:
    def __init__(self):
        rospy.init_node('gru_prediction_server', anonymous=False)
        
        # 加载配置
        self.obs_len = rospy.get_param('~obs_len', 30)
        self.pred_len = rospy.get_param('~pred_len', 60)
        self.mc_samples = rospy.get_param('~mc_samples', 3)
        model_path = rospy.get_param('~model_path')
        
        # 加载模型
        rospy.loginfo(f"Loading model from {model_path}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = StudentGRU(
            input_dim=4,
            hidden_dim=64,
            num_layers=2,
            pred_horizon=self.pred_len
        ).to(self.device)
        
        # 加载TorchScript模型
        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            rospy.loginfo("Loaded TorchScript model")
        except:
            # 如果不是TorchScript，加载普通checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            rospy.loginfo("Loaded PyTorch checkpoint")
        
        self.model.eval()
        
        # 历史轨迹缓存
        self.history_buffer = deque(maxlen=self.obs_len)
        self.last_pos = np.array([0.0, 0.0])
        
        # ROS通信
        self.target_sub = rospy.Subscriber(
            '/target_ekf_node/target_odom',
            Odometry,
            self.target_callback,
            queue_size=1
        )
        
        self.pred_pub = rospy.Publisher(
            '/gru_prediction',
            PoseArray,
            queue_size=1
        )
        
        self.uncertainty_pub = rospy.Publisher(
            '/gru_uncertainty',
            Float32MultiArray,
            queue_size=1
        )
        
        rospy.loginfo("GRU Prediction Server ready")
    
    def target_callback(self, msg):
        """接收目标状态并预测"""
        # 提取状态
        state = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y
        ])
        
        # 更新缓存
        self.history_buffer.append(state)
        
        if len(self.history_buffer) < self.obs_len:
            rospy.logwarn_throttle(1.0, f"Collecting history: {len(self.history_buffer)}/{self.obs_len}")
            return
        
        # 预测
        try:
            predictions, uncertainty = self.predict()
            self.publish_prediction(predictions, uncertainty)
        except Exception as e:
            rospy.logerr(f"Prediction failed: {e}")
    
    def predict(self):
        """执行GRU预测 + MC Dropout"""
        # 准备输入
        history = np.array(list(self.history_buffer))  # (30, 4)
        
        # 转换为相对坐标
        self.last_pos = history[-1, :2].copy()
        history[:, :2] -= self.last_pos
        
        # 转为Tensor
        x = torch.FloatTensor(history).unsqueeze(0).to(self.device)  # (1, 30, 4)
        
        # MC Dropout采样
        self.model.eval()
        if hasattr(self.model, 'enable_dropout'):
            self.model.enable_dropout()
        
        samples = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                pred = self.model(x)  # (1, 60, 2)
                samples.append(pred.cpu().numpy())
        
        samples = np.array(samples)  # (n_samples, 1, 60, 2)
        
        # 计算均值和不确定性
        mean_pred = samples.mean(axis=0)[0]  # (60, 2)
        std_pred = samples.std(axis=0)[0]    # (60, 2)
        uncertainty = std_pred.mean(axis=-1)  # (60,)
        
        # 转回绝对坐标
        mean_pred += self.last_pos
        
        return mean_pred, uncertainty
    
    def publish_prediction(self, predictions, uncertainty):
        """发布预测结果"""
        # 位置预测
        msg = PoseArray()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        
        for pred in predictions:
            pose = Pose()
            pose.position.x = pred[0]
            pose.position.y = pred[1]
            pose.position.z = 0.0
            pose.orientation.w = 1.0
            msg.poses.append(pose)
        
        self.pred_pub.publish(msg)
        
        # 不确定性
        uncert_msg = Float32MultiArray()
        uncert_msg.data = uncertainty.tolist()
        self.uncertainty_pub.publish(uncert_msg)

if __name__ == '__main__':
    try:
        server = GRUPredictionServer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass