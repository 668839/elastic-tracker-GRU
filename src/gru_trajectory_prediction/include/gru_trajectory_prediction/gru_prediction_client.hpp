// include/gru_trajectory_prediction/gru_prediction_client.hpp

#ifndef GRU_PREDICTION_CLIENT_HPP
#define GRU_PREDICTION_CLIENT_HPP

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseArray.h>
#include <std_msgs/Float32MultiArray.h>
#include <Eigen/Core>
#include <deque>
#include <vector>

#include "gru_trajectory_prediction/collision_checker.hpp"
#include "gru_trajectory_prediction/trajectory_corrector.hpp"

namespace gru_prediction {

class GRUPredictionClient {
public:
    GRUPredictionClient(ros::NodeHandle& nh);
    
    // 获取最新预测结果
    bool getLatestPrediction(
        std::vector<Eigen::Vector3d>& prediction,
        std::vector<double>& uncertainty
    );
    
    // 检查预测是否可用
    bool isPredictionReady() const { return prediction_received_; }

private:
    void predictionCallback(const geometry_msgs::PoseArrayConstPtr& msg);
    void uncertaintyCallback(const std_msgs::Float32MultiArrayConstPtr& msg);
    
    ros::Subscriber pred_sub_;
    ros::Subscriber uncertainty_sub_;
    
    std::vector<Eigen::Vector3d> latest_prediction_;
    std::vector<double> latest_uncertainty_;
    bool prediction_received_;
    
    std::shared_ptr<TrajectoryCorrector> corrector_;
};

} // namespace gru_prediction

#endif