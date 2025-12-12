// src/gru_prediction_client.cpp

#include "gru_trajectory_prediction/gru_prediction_client.hpp"

namespace gru_prediction {

GRUPredictionClient::GRUPredictionClient(ros::NodeHandle& nh)
    : prediction_received_(false) {
    
    // 初始化轨迹修正器
    corrector_ = std::make_shared<TrajectoryCorrector>(nh);
    
    // 订阅Python服务的预测结果
    pred_sub_ = nh.subscribe<geometry_msgs::PoseArray>(
        "/gru_prediction", 1,
        &GRUPredictionClient::predictionCallback, this
    );
    
    uncertainty_sub_ = nh.subscribe<std_msgs::Float32MultiArray>(
        "/gru_uncertainty", 1,
        &GRUPredictionClient::uncertaintyCallback, this
    );
    
    ROS_INFO("GRU Prediction Client initialized");
}

void GRUPredictionClient::predictionCallback(
    const geometry_msgs::PoseArrayConstPtr& msg) {
    
    latest_prediction_.clear();
    for (const auto& pose : msg->poses) {
        Eigen::Vector3d point(
            pose.position.x,
            pose.position.y,
            pose.position.z
        );
        latest_prediction_.push_back(point);
    }
}

void GRUPredictionClient::uncertaintyCallback(
    const std_msgs::Float32MultiArrayConstPtr& msg) {
    
    latest_uncertainty_.clear();
    for (float val : msg->data) {
        latest_uncertainty_.push_back(static_cast<double>(val));
    }
    
    prediction_received_ = true;
}

bool GRUPredictionClient::getLatestPrediction(
    std::vector<Eigen::Vector3d>& prediction,
    std::vector<double>& uncertainty) {
    
    if (!prediction_received_) {
        return false;
    }
    
    // 应用碰撞检测和修正
    prediction = corrector_->correctTrajectory(
        latest_prediction_,
        latest_uncertainty_
    );
    
    uncertainty = latest_uncertainty_;
    
    return true;
}

} // namespace gru_prediction

// 测试main函数
int main(int argc, char** argv) {
    ros::init(argc, argv, "gru_prediction_client_node");
    ros::NodeHandle nh("~");
    
    gru_prediction::GRUPredictionClient client(nh);
    
    ros::Rate rate(10);  // 10Hz
    while (ros::ok()) {
        std::vector<Eigen::Vector3d> prediction;
        std::vector<double> uncertainty;
        
        if (client.getLatestPrediction(prediction, uncertainty)) {
            ROS_INFO_THROTTLE(1.0, "Got prediction: %lu points", prediction.size());
        }
        
        ros::spinOnce();
        rate.sleep();
    }
    
    return 0;
}