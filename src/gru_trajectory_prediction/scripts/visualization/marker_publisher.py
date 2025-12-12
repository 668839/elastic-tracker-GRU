#!/usr/bin/env python3
"""
额外的marker发布器 - 发布碰撞检测结果等
"""

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray, Point
from std_msgs.msg import ColorRGBA

class CollisionMarkerPublisher:
    def __init__(self):
        rospy.init_node('collision_marker_publisher', anonymous=False)
        
        self.collision_indices = []
        self.prediction = None
        
        # 订阅碰撞检测结果（假设有这个话题）
        # rospy.Subscriber('/collision_indices', ..., self.collision_callback)
        rospy.Subscriber('/fused_prediction', PoseArray, self.pred_callback)
        
        self.marker_pub = rospy.Publisher('/collision_markers', MarkerArray, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(0.2), self.publish_markers)
        
        rospy.loginfo("Collision Marker Publisher started")
    
    def pred_callback(self, msg):
        self.prediction = [Point(p.position.x, p.position.y, p.position.z) 
                          for p in msg.poses]
    
    def publish_markers(self, event):
        """发布碰撞点标记"""
        if not self.prediction or not self.collision_indices:
            return
        
        markers = MarkerArray()
        
        for i, idx in enumerate(self.collision_indices):
            if idx >= len(self.prediction):
                continue
            
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "collision"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            marker.pose.position = self.prediction[idx]
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            
            # 红色闪烁
            alpha = 0.5 + 0.5 * (rospy.Time.now().to_sec() % 1.0)
            marker.color = ColorRGBA(1.0, 0.0, 0.0, alpha)
            
            marker.lifetime = rospy.Duration(0.5)
            markers.markers.append(marker)
        
        self.marker_pub.publish(markers)

if __name__ == '__main__':
    try:
        publisher = CollisionMarkerPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass