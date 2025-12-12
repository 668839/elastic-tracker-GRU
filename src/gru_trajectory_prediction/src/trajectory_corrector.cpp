// src/trajectory_corrector.cpp

#include "gru_trajectory_prediction/trajectory_corrector.hpp"

namespace gru_prediction {

TrajectoryCorrector::TrajectoryCorrector(ros::NodeHandle& nh) {
    collision_checker_ = std::make_shared<CollisionChecker>(nh);
    
    nh.param("collision/check_interval", check_interval_, 3);
    nh.param("collision/search_radius_base", base_search_radius_, 0.5);
    nh.param("collision/uncertainty_scale", uncertainty_scale_, 2.0);
    
    ROS_INFO("TrajectoryCorrector initialized (check_interval=%d)", check_interval_);
}

std::vector<Eigen::Vector3d> TrajectoryCorrector::correctTrajectory(
    const std::vector<Eigen::Vector3d>& raw_trajectory,
    const std::vector<double>& uncertainty) {
    
    std::vector<Eigen::Vector3d> corrected_trajectory;
    corrected_trajectory.reserve(raw_trajectory.size());
    
    int collision_count = 0;
    
    for (size_t i = 0; i < raw_trajectory.size(); ++i) {
        Eigen::Vector3d point = raw_trajectory[i];
        
        // 只检查部分点以提高效率
        if (i % check_interval_ == 0) {
            if (collision_checker_->isCollision(point)) {
                // 计算搜索半径（基于不确定性）
                double search_radius = base_search_radius_ + 
                                      uncertainty_scale_ * uncertainty[i];
                
                // 修正为最近的自由点
                point = findNearestFreePoint(point, search_radius);
                collision_count++;
            }
        }
        
        corrected_trajectory.push_back(point);
    }
    
    if (collision_count > 0) {
        ROS_WARN("Corrected %d collision points", collision_count);
    }
    
    return corrected_trajectory;
}

Eigen::Vector3d TrajectoryCorrector::findNearestFreePoint(
    const Eigen::Vector3d& collision_point,
    double search_radius) {
    
    const int num_samples = 16;
    double best_dist = std::numeric_limits<double>::max();
    Eigen::Vector3d best_point = collision_point;
    
    // 在圆周上采样
    for (int i = 0; i < num_samples; ++i) {
        double theta = 2.0 * M_PI * i / num_samples;
        
        Eigen::Vector3d test_point = collision_point;
        test_point.x() += search_radius * cos(theta);
        test_point.y() += search_radius * sin(theta);
        
        if (!collision_checker_->isCollision(test_point)) {
            double dist = (test_point - collision_point).norm();
            if (dist < best_dist) {
                best_dist = dist;
                best_point = test_point;
            }
        }
    }
    
    // 如果没找到，扩大搜索范围
    if (best_dist == std::numeric_limits<double>::max()) {
        return findNearestFreePoint(collision_point, search_radius * 1.5);
    }
    
    return best_point;
}

} // namespace gru_prediction