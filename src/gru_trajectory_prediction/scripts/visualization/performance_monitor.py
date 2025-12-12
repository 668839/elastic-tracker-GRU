#!/usr/bin/env python3
"""
实时性能监控 - 在RViz中显示性能指标
"""

import rospy
import numpy as np
from collections import deque
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
import time

class PerformanceMonitor:
    def __init__(self):
        rospy.init_node('performance_monitor', anonymous=False)
        
        # 性能指标
        self.ade_window = deque(maxlen=100)  # 最近100个ADE
        self.fde_window = deque(maxlen=100)
        self.latency_window = deque(maxlen=100)
        
        # 数据缓存
        self.gt_buffer = deque(maxlen=200)
        self.pred_buffer = deque(maxlen=10)
        self.pred_timestamp = None
        
        # 订阅
        rospy.Subscriber('/target_ekf_node/target_odom', Odometry, self.gt_callback)
        rospy.Subscriber('/fused_prediction', PoseArray, self.pred_callback)
        
        # 发布
        self.text_pub = rospy.Publisher('/performance_text', Marker, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0), self.publish_metrics)
        
        rospy.loginfo("Performance Monitor started")
    
    def gt_callback(self, msg):
        pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        timestamp = msg.header.stamp.to_sec()
        self.gt_buffer.append((timestamp, pos))
    
    def pred_callback(self, msg):
        pred = np.array([[p.position.x, p.position.y] for p in msg.poses])
        self.pred_buffer.append(pred)
        self.pred_timestamp = rospy.Time.now().to_sec()
        
        # 计算延迟
        if self.pred_timestamp and len(self.gt_buffer) > 0:
            latency = rospy.Time.now().to_sec() - self.gt_buffer[-1][0]
            self.latency_window.append(latency * 1000)  # 转为ms
    
    def compute_metrics(self):
        """计算ADE和FDE"""
        if len(self.pred_buffer) == 0 or len(self.gt_buffer) < 60:
            return None, None
        
        # 获取最新预测
        pred = self.pred_buffer[-1]  # (60, 2)
        
        # 获取对应的未来真实轨迹
        gt_positions = [pos for _, pos in self.gt_buffer]
        if len(gt_positions) < 60:
            return None, None
        
        gt_future = np.array(gt_positions[-60:])  # (60, 2)
        
        # 计算误差
        errors = np.linalg.norm(pred - gt_future, axis=1)
        ade = errors.mean()
        fde = errors[-1]
        
        return ade, fde
    
    def publish_metrics(self, event):
        """发布性能指标文本"""
        ade, fde = self.compute_metrics()
        
        if ade is not None:
            self.ade_window.append(ade)
            self.fde_window.append(fde)
        
        # 计算统计量
        avg_ade = np.mean(self.ade_window) if self.ade_window else 0.0
        avg_fde = np.mean(self.fde_window) if self.fde_window else 0.0
        avg_latency = np.mean(self.latency_window) if self.latency_window else 0.0
        
        # 创建文本marker
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "performance"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # 位置（固定在视野左上角）
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 3.0
        marker.pose.orientation.w = 1.0
        
        marker.scale.z = 0.3  # 字体大小
        
        # 根据性能好坏设置颜色
        if avg_ade < 0.3:
            marker.color = ColorRGBA(0.0, 1.0, 0.0, 1.0)  # 绿色 - 优秀
        elif avg_ade < 0.6:
            marker.color = ColorRGBA(1.0, 1.0, 0.0, 1.0)  # 黄色 - 良好
        else:
            marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)  # 红色 - 需改进
        
        # 文本内容
        marker.text = f"""
╔═══════════════════════════╗
║   PREDICTION METRICS      ║
╠═══════════════════════════╣
║ ADE:     {avg_ade:.3f} m         ║
║ FDE:     {avg_fde:.3f} m         ║
║ Latency: {avg_latency:.1f} ms        ║
║ Samples: {len(self.ade_window)}              ║
╚═══════════════════════════╝
        """
        
        marker.lifetime = rospy.Duration(2.0)
        
        self.text_pub.publish(marker)

if __name__ == '__main__':
    try:
        monitor = PerformanceMonitor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass