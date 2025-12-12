// src/collision_checker.cpp

#include "gru_trajectory_prediction/collision_checker.hpp"

namespace gru_prediction {

CollisionChecker::CollisionChecker(ros::NodeHandle& nh) 
    : map_received_(false) {
    
    map_ptr_ = std::make_shared<mapping::OccGridMap>();
    
    map_sub_ = nh.subscribe<quadrotor_msgs::OccMap3d>(
        "/gridmap_inflate", 1,
        &CollisionChecker::mapCallback, this
    );
    
    ROS_INFO("CollisionChecker initialized");
}

void CollisionChecker::mapCallback(const quadrotor_msgs::OccMap3dConstPtr& msg) {
    map_ptr_->from_msg(*msg);
    if (!map_received_) {
        ROS_INFO("First map received");
        map_received_ = true;
    }
}

bool CollisionChecker::isCollision(const Eigen::Vector3d& point) const {
    if (!map_received_) {
        return false;  // 地图未准备好，假设无碰撞
    }
    return map_ptr_->isOccupied(point);
}

std::vector<int> CollisionChecker::findCollisions(
    const std::vector<Eigen::Vector3d>& trajectory) const {
    
    std::vector<int> collision_indices;
    
    for (size_t i = 0; i < trajectory.size(); ++i) {
        if (isCollision(trajectory[i])) {
            collision_indices.push_back(i);
        }
    }
    
    return collision_indices;
}

double CollisionChecker::getDistanceToObstacle(const Eigen::Vector3d& point) const {
    if (!map_received_) {
        return 10.0;  // 默认安全距离
    }
    
    const double resolution = 0.15;
    const double max_range = 5.0;
    
    // 简单的放射状采样
    for (double r = 0.0; r < max_range; r += resolution) {
        for (double theta = 0; theta < 2*M_PI; theta += M_PI/8) {
            Eigen::Vector3d test_point = point;
            test_point.x() += r * cos(theta);
            test_point.y() += r * sin(theta);
            
            if (map_ptr_->isOccupied(test_point)) {
                return r;
            }
        }
    }
    
    return max_range;
}

} // namespace gru_prediction