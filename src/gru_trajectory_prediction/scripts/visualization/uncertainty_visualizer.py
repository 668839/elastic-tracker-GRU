#!/usr/bin/env python3
"""
不确定性热力图可视化
显示预测不确定性的空间分布
"""

import rospy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray, Point
from std_msgs.msg import Float32MultiArray, ColorRGBA
import matplotlib.cm as cm

class UncertaintyVisualizer:
    def __init__(self):
        rospy.init_node('uncertainty_visualizer', anonymous=False)
        
        self.prediction = None
        self.uncertainty = None
        
        rospy.Subscriber('/fused_prediction', PoseArray, self.pred_callback)
        rospy.Subscriber('/gru_uncertainty', Float32MultiArray, self.uncert_callback)
        
        self.marker_pub = rospy.Publisher('/uncertainty_heatmap', MarkerArray, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(0.2), self.publish_heatmap)
        
        # 使用jet colormap
        self.colormap = cm.get_cmap('jet')
        
        rospy.loginfo("Uncertainty Visualizer started")
    
    def pred_callback(self, msg):
        self.prediction = [(p.position.x, p.position.y, p.position.z) for p in msg.poses]
    
    def uncert_callback(self, msg):
        self.uncertainty = np.array(msg.data)
    
    def uncertainty_to_color(self, uncert_value, max_uncert=1.0):
        """将不确定性值映射到颜色"""
        normalized = min(uncert_value / max_uncert, 1.0)
        rgba = self.colormap(normalized)
        return ColorRGBA(rgba[0], rgba[1], rgba[2], 0.7)
    
    def publish_heatmap(self, event):
        if self.prediction is None or self.uncertainty is None:
            return
        
        if len(self.prediction) != len(self.uncertainty):
            return
        
        markers = MarkerArray()
        
        # 归一化不确定性
        max_uncert = max(self.uncertainty) if len(self.uncertainty) > 0 else 1.0
        
        for i, (pos, uncert) in enumerate(zip(self.prediction, self.uncertainty)):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "uncertainty_heatmap"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position.x = pos[0]
            marker.pose.position.y = pos[1]
            marker.pose.position.z = pos[2]
            marker.pose.orientation.w = 1.0
            
            # 球的大小反映不确定性
            size = 0.1 + uncert * 0.3
            marker.scale.x = size
            marker.scale.y = size
            marker.scale.z = size
            
            # 颜色反映不确定性大小
            marker.color = self.uncertainty_to_color(uncert, max_uncert)
            marker.lifetime = rospy.Duration(0.5)
            
            markers.markers.append(marker)
        
        # 添加colorbar说明
        colorbar_marker = Marker()
        colorbar_marker.header.frame_id = "world"
        colorbar_marker.header.stamp = rospy.Time.now()
        colorbar_marker.ns = "colorbar"
        colorbar_marker.id = 9999
        colorbar_marker.type = Marker.TEXT_VIEW_FACING
        colorbar_marker.action = Marker.ADD
        
        if self.prediction:
            colorbar_marker.pose.position.x = self.prediction[0][0] + 1.0
            colorbar_marker.pose.position.y = self.prediction[0][1]
            colorbar_marker.pose.position.z = self.prediction[0][2] + 1.0
        
        colorbar_marker.scale.z = 0.2
        colorbar_marker.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
        colorbar_marker.text = f"Uncertainty\nMax: {max_uncert:.3f}"
        colorbar_marker.lifetime = rospy.Duration(0.5)
        
        markers.markers.append(colorbar_marker)
        
        self.marker_pub.publish(markers)

if __name__ == '__main__':
    try:
        visualizer = UncertaintyVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass