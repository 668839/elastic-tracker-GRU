// include/gru_trajectory_prediction/collision_checker.hpp

#ifndef COLLISION_CHECKER_HPP
#define COLLISION_CHECKER_HPP

#include <ros/ros.h>
#include <mapping/mapping.h>
#include <quadrotor_msgs/OccMap3d.h>
#include <Eigen/Core>
#include <memory>

namespace gru_prediction {

class CollisionChecker {
public:
    CollisionChecker(ros::NodeHandle& nh);
    
    // 检查单点是否碰撞
    bool isCollision(const Eigen::Vector3d& point) const;
    
    // 检查轨迹是否有碰撞
    std::vector<int> findCollisions(const std::vector<Eigen::Vector3d>& trajectory) const;
    
    // 获取到最近障碍物的距离
    double getDistanceToObstacle(const Eigen::Vector3d& point) const;
    
    // 地图是否已加载
    bool isMapReady() const { return map_received_; }

private:
    void mapCallback(const quadrotor_msgs::OccMap3dConstPtr& msg);
    
    std::shared_ptr<mapping::OccGridMap> map_ptr_;
    ros::Subscriber map_sub_;
    bool map_received_;
};

} // namespace gru_prediction

#endif