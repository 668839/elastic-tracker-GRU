// include/gru_trajectory_prediction/trajectory_corrector.hpp

#ifndef TRAJECTORY_CORRECTOR_HPP
#define TRAJECTORY_CORRECTOR_HPP

#include "gru_trajectory_prediction/collision_checker.hpp"
#include <vector>
#include <Eigen/Core>

namespace gru_prediction {

class TrajectoryCorrector {
public:
    TrajectoryCorrector(ros::NodeHandle& nh);
    
    // 修正整条轨迹
    std::vector<Eigen::Vector3d> correctTrajectory(
        const std::vector<Eigen::Vector3d>& raw_trajectory,
        const std::vector<double>& uncertainty
    );
    
    // 找到最近的自由点
    Eigen::Vector3d findNearestFreePoint(
        const Eigen::Vector3d& collision_point,
        double search_radius
    );

private:
    std::shared_ptr<CollisionChecker> collision_checker_;
    int check_interval_;          // 每隔几个点检测一次
    double base_search_radius_;   // 基础搜索半径
    double uncertainty_scale_;    // 不确定性缩放因子
};

} // namespace gru_prediction

#endif