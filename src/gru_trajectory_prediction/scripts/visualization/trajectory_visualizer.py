#!/usr/bin/env python3
"""
轨迹可视化节点 - 发布美观的可视化标记
包括:GRU预测、Kalman预测、真实轨迹、不确定性区域
"""

import rospy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray, Point
from std_msgs.msg import Float32MultiArray, ColorRGBA
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from collections import deque

class TrajectoryVisualizer:
    def __init__(self):
        rospy.init_node('trajectory_visualizer', anonymous=False)
        
        # 颜色配置
        self.colors = {
            'gru': ColorRGBA(0.0, 1.0, 0.0, 0.8),      # 绿色 - GRU预测
            'kalman': ColorRGBA(0.0, 0.5, 1.0, 0.8),   # 蓝色 - Kalman预测
            'fused': ColorRGBA(1.0, 0.5, 0.0, 1.0),    # 橙色 - 融合预测
            'ground_truth': ColorRGBA(1.0, 0.0, 0.0, 1.0),  # 红色 - 真实轨迹
            'uncertainty': ColorRGBA(0.5, 0.5, 0.5, 0.3),   # 灰色半透明 - 不确定性
        }
        
        # 历史轨迹缓存
        self.gt_history = deque(maxlen=200)  # 真实轨迹历史
        self.fused_history = deque(maxlen=200)  # 融合预测历史
        
        # 当前预测
        self.gru_pred = None
        self.kalman_pred = None
        self.fused_pred = None
        self.uncertainty = None
        
        # 订阅
        rospy.Subscriber('/target_ekf_node/target_odom', Odometry, self.gt_callback)
        rospy.Subscriber('/gru_prediction', PoseArray, self.gru_callback)
        rospy.Subscriber('/fused_prediction', PoseArray, self.fused_callback)
        rospy.Subscriber('/gru_uncertainty', Float32MultiArray, self.uncertainty_callback)
        
        # 发布
        self.marker_pub = rospy.Publisher('/trajectory_markers', MarkerArray, queue_size=1)
        self.gt_path_pub = rospy.Publisher('/ground_truth_path', Path, queue_size=1)
        
        # 定时发布
        self.timer = rospy.Timer(rospy.Duration(0.1), self.publish_markers)
        
        rospy.loginfo("Trajectory Visualizer started")
    
    def gt_callback(self, msg):
        """真实轨迹"""
        point = Point()
        point.x = msg.pose.pose.position.x
        point.y = msg.pose.pose.position.y
        point.z = msg.pose.pose.position.z
        self.gt_history.append(point)
    
    def gru_callback(self, msg):
        """GRU原始预测"""
        self.gru_pred = [Point(p.position.x, p.position.y, p.position.z) 
                        for p in msg.poses]
    
    def fused_callback(self, msg):
        """融合后的预测"""
        self.fused_pred = [Point(p.position.x, p.position.y, p.position.z) 
                          for p in msg.poses]
        if self.fused_pred:
            self.fused_history.append(self.fused_pred[0])
    
    def uncertainty_callback(self, msg):
        """不确定性"""
        self.uncertainty = np.array(msg.data)
    
    def create_line_marker(self, points, color, marker_id, ns, line_width=0.05):
        """创建线条marker"""
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = line_width
        marker.color = color
        marker.points = points
        marker.lifetime = rospy.Duration(0.5)
        return marker
    
    def create_sphere_list_marker(self, points, color, marker_id, ns, sphere_scale=0.1):
        """创建球列表marker"""
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = sphere_scale
        marker.scale.y = sphere_scale
        marker.scale.z = sphere_scale
        marker.color = color
        marker.points = points
        marker.lifetime = rospy.Duration(0.5)
        return marker
    
    def create_uncertainty_tubes(self, points, uncertainties, marker_id_start):
        """创建不确定性管道（圆柱体）"""
        markers = []
        
        for i, (point, uncert) in enumerate(zip(points, uncertainties)):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "uncertainty"
            marker.id = marker_id_start + i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            marker.pose.position = point
            marker.pose.orientation.w = 1.0
            
            # 半径基于不确定性
            radius = max(0.05, uncert * 0.5)  # 最小半径0.05m
            marker.scale.x = radius * 2
            marker.scale.y = radius * 2
            marker.scale.z = 0.05  # 薄圆盘
            
            marker.color = self.colors['uncertainty']
            marker.lifetime = rospy.Duration(0.5)
            
            markers.append(marker)
        
        return markers
    
    def create_text_marker(self, position, text, marker_id, color):
        """创建文本marker"""
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "text"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position = position
        marker.scale.z = 0.2  # 文字大小
        marker.color = color
        marker.text = text
        marker.lifetime = rospy.Duration(0.5)
        return marker
    
    def publish_markers(self, event):
        """发布所有可视化标记"""
        markers = MarkerArray()
        
        # 1. 真实轨迹历史（红色粗线）
        if len(self.gt_history) > 1:
            gt_marker = self.create_line_marker(
                list(self.gt_history),
                self.colors['ground_truth'],
                marker_id=0,
                ns="ground_truth",
                line_width=0.08
            )
            markers.markers.append(gt_marker)
            
            # 当前位置（大红球）
            current_pos = self.gt_history[-1]
            current_marker = Marker()
            current_marker.header.frame_id = "world"
            current_marker.header.stamp = rospy.Time.now()
            current_marker.ns = "current_position"
            current_marker.id = 100
            current_marker.type = Marker.SPHERE
            current_marker.action = Marker.ADD
            current_marker.pose.position = current_pos
            current_marker.pose.orientation.w = 1.0
            current_marker.scale.x = 0.3
            current_marker.scale.y = 0.3
            current_marker.scale.z = 0.3
            current_marker.color = self.colors['ground_truth']
            current_marker.lifetime = rospy.Duration(0.5)
            markers.markers.append(current_marker)
        
        # 2. GRU原始预测（绿色细线 + 点）
        if self.gru_pred:
            gru_line = self.create_line_marker(
                self.gru_pred,
                self.colors['gru'],
                marker_id=1,
                ns="gru_prediction",
                line_width=0.03
            )
            markers.markers.append(gru_line)
            
            # 每10个点标记一个
            sample_points = self.gru_pred[::10]
            gru_points = self.create_sphere_list_marker(
                sample_points,
                self.colors['gru'],
                marker_id=2,
                ns="gru_points",
                sphere_scale=0.08
            )
            markers.markers.append(gru_points)
        
        # 3. 融合预测（橙色粗线）
        if self.fused_pred:
            fused_line = self.create_line_marker(
                self.fused_pred,
                self.colors['fused'],
                marker_id=3,
                ns="fused_prediction",
                line_width=0.06
            )
            markers.markers.append(fused_line)
            
            # 终点标记
            end_point = self.fused_pred[-1]
            end_marker = Marker()
            end_marker.header.frame_id = "world"
            end_marker.header.stamp = rospy.Time.now()
            end_marker.ns = "prediction_end"
            end_marker.id = 101
            end_marker.type = Marker.ARROW
            end_marker.action = Marker.ADD
            end_marker.pose.position = end_point
            end_marker.pose.orientation.w = 1.0
            end_marker.scale.x = 0.3
            end_marker.scale.y = 0.1
            end_marker.scale.z = 0.1
            end_marker.color = self.colors['fused']
            end_marker.lifetime = rospy.Duration(0.5)
            markers.markers.append(end_marker)
        
        # 4. 不确定性管道
        if self.fused_pred and self.uncertainty is not None:
            uncert_markers = self.create_uncertainty_tubes(
                self.fused_pred[::5],  # 每5个点一个管道
                self.uncertainty[::5],
                marker_id_start=1000
            )
            markers.markers.extend(uncert_markers)
        
        # 5. 文本标签
        if len(self.gt_history) > 0:
            label_pos = Point()
            label_pos.x = self.gt_history[-1].x
            label_pos.y = self.gt_history[-1].y
            label_pos.z = self.gt_history[-1].z + 0.5
            
            text_marker = self.create_text_marker(
                label_pos,
                "Target",
                marker_id=200,
                color=self.colors['ground_truth']
            )
            markers.markers.append(text_marker)
        
        # 6. 预测时间线标注
        if self.fused_pred:
            for i, t in enumerate([0.5, 1.0, 1.5, 2.0]):  # 0.5s, 1s, 1.5s, 2s
                idx = int(t * 30)  # 30Hz
                if idx < len(self.fused_pred):
                    time_label_pos = Point()
                    time_label_pos.x = self.fused_pred[idx].x
                    time_label_pos.y = self.fused_pred[idx].y
                    time_label_pos.z = self.fused_pred[idx].z + 0.3
                    
                    time_text = self.create_text_marker(
                        time_label_pos,
                        f"t+{t:.1f}s",
                        marker_id=300 + i,
                        color=ColorRGBA(1.0, 1.0, 1.0, 0.8)
                    )
                    markers.markers.append(time_text)
        
        # 发布
        self.marker_pub.publish(markers)
        
        # 发布Path消息（用于RViz的Path显示）
        if len(self.gt_history) > 1:
            path_msg = Path()
            path_msg.header.frame_id = "world"
            path_msg.header.stamp = rospy.Time.now()
            
            for point in self.gt_history:
                pose = PoseStamped()
                pose.header.frame_id = "world"
                pose.pose.position = point
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
            
            self.gt_path_pub.publish(path_msg)

if __name__ == '__main__':
    try:
        visualizer = TrajectoryVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass