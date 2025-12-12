/**
 * @file gazebo_target_tracker_node.cpp
 * @brief 从Gazebo获取目标模型位置，转换到VINS坐标系后发布
 * 
 * 该节点从/gazebo/model_states获取指定目标的位置，
 * 通过与VINS里程计对比计算坐标系偏移，
 * 将目标位置转换到VINS坐标系后发布。
 */

#include <ros/ros.h>
#include <gazebo_msgs/ModelStates.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Empty.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <mutex>
#include <atomic>

class GazeboTargetTracker {
private:
    ros::NodeHandle nh_;
    
    // 订阅者
    ros::Subscriber gazebo_sub_;
    ros::Subscriber odom_sub_;
    
    // 发布者
    ros::Publisher target_odom_pub_;
    ros::Publisher target_pose_pub_;
    
    // 定时器
    ros::Timer publish_timer_;
    
    // 参数
    std::string target_model_name_;
    std::string drone_model_name_;
    double publish_rate_;
    
    // 状态标志
    std::atomic<bool> odom_received_{false};
    std::atomic<bool> target_found_{false};
    std::atomic<bool> drone_found_{false};
    std::atomic<bool> coord_initialized_{false};
    
    // 互斥锁
    std::mutex odom_mutex_;
    std::mutex gazebo_mutex_;
    
    // 无人机状态 (VINS坐标系)
    Eigen::Vector3d drone_pos_vins_;
    Eigen::Vector3d drone_vel_vins_;
    Eigen::Quaterniond drone_quat_vins_;
    
    // 无人机状态 (Gazebo坐标系)
    Eigen::Vector3d drone_pos_gazebo_;
    
    // 目标状态 (Gazebo坐标系)
    Eigen::Vector3d target_pos_gazebo_;
    Eigen::Vector3d target_vel_gazebo_;
    Eigen::Quaterniond target_quat_gazebo_;
    
    // 坐标系转换参数
    // target_pos_vins = target_pos_gazebo - coord_offset_
    // 其中 coord_offset_ = drone_pos_gazebo - drone_pos_vins
    Eigen::Vector3d coord_offset_;
    
    // 用于计算偏移量的采样
    std::vector<Eigen::Vector3d> offset_samples_;
    int max_offset_samples_;
    
public:
    GazeboTargetTracker() : nh_("~") {
        // 读取参数
        nh_.param<std::string>("target_model_name", target_model_name_, "hatchback_red");
        nh_.param<std::string>("drone_model_name", drone_model_name_, "iris");
        nh_.param<double>("publish_rate", publish_rate_, 30.0);
        nh_.param<int>("max_offset_samples", max_offset_samples_, 50);
        
        // 初始化
        coord_offset_.setZero();
        drone_pos_vins_.setZero();
        drone_vel_vins_.setZero();
        drone_quat_vins_.setIdentity();
        drone_pos_gazebo_.setZero();
        target_pos_gazebo_.setZero();
        target_vel_gazebo_.setZero();
        target_quat_gazebo_.setIdentity();
        
        // 订阅者
        gazebo_sub_ = nh_.subscribe("/gazebo/model_states", 1, 
                                    &GazeboTargetTracker::gazeboCallback, this);
        odom_sub_ = nh_.subscribe("/vins_fusion/imu_propagate", 10, 
                                  &GazeboTargetTracker::odomCallback, this, 
                                  ros::TransportHints().tcpNoDelay());
        
        // 发布者
        target_odom_pub_ = nh_.advertise<nav_msgs::Odometry>("target_odom", 10);
        target_pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("target_pose", 10);
        
        // 定时发布
        publish_timer_ = nh_.createTimer(ros::Duration(1.0 / publish_rate_), 
                                         &GazeboTargetTracker::publishCallback, this);
        
        ROS_INFO("[GazeboTargetTracker] Initialized.");
        ROS_INFO("[GazeboTargetTracker] Target model: %s", target_model_name_.c_str());
        ROS_INFO("[GazeboTargetTracker] Drone model: %s", drone_model_name_.c_str());
    }
    
    /**
     * @brief Gazebo模型状态回调
     */
    void gazeboCallback(const gazebo_msgs::ModelStates::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(gazebo_mutex_);
        
        for (size_t i = 0; i < msg->name.size(); ++i) {
            // 查找目标模型
            if (msg->name[i] == target_model_name_) {
                target_pos_gazebo_.x() = msg->pose[i].position.x;
                target_pos_gazebo_.y() = msg->pose[i].position.y;
                target_pos_gazebo_.z() = msg->pose[i].position.z;
                
                target_quat_gazebo_.w() = msg->pose[i].orientation.w;
                target_quat_gazebo_.x() = msg->pose[i].orientation.x;
                target_quat_gazebo_.y() = msg->pose[i].orientation.y;
                target_quat_gazebo_.z() = msg->pose[i].orientation.z;
                target_quat_gazebo_.normalize();
                
                target_vel_gazebo_.x() = msg->twist[i].linear.x;
                target_vel_gazebo_.y() = msg->twist[i].linear.y;
                target_vel_gazebo_.z() = msg->twist[i].linear.z;
                
                target_found_ = true;
            }
            
            // 查找无人机模型
            if (msg->name[i] == drone_model_name_) {
                drone_pos_gazebo_.x() = msg->pose[i].position.x;
                drone_pos_gazebo_.y() = msg->pose[i].position.y;
                drone_pos_gazebo_.z() = msg->pose[i].position.z;
                
                drone_found_ = true;
            }
        }
        
        // 初始化坐标系偏移量
        if (odom_received_ && drone_found_ && !coord_initialized_) {
            updateCoordinateOffset();
        }
    }
    
    /**
     * @brief VINS里程计回调
     */
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(odom_mutex_);
        
        drone_pos_vins_.x() = msg->pose.pose.position.x;
        drone_pos_vins_.y() = msg->pose.pose.position.y;
        drone_pos_vins_.z() = msg->pose.pose.position.z;
        
        drone_vel_vins_.x() = msg->twist.twist.linear.x;
        drone_vel_vins_.y() = msg->twist.twist.linear.y;
        drone_vel_vins_.z() = msg->twist.twist.linear.z;
        
        drone_quat_vins_.w() = msg->pose.pose.orientation.w;
        drone_quat_vins_.x() = msg->pose.pose.orientation.x;
        drone_quat_vins_.y() = msg->pose.pose.orientation.y;
        drone_quat_vins_.z() = msg->pose.pose.orientation.z;
        drone_quat_vins_.normalize();
        
        odom_received_ = true;
    }
    
    /**
     * @brief 更新坐标系偏移量
     * 使用多次采样取平均以提高精度
     */
    void updateCoordinateOffset() {
        std::lock_guard<std::mutex> lock(odom_mutex_);
        
        Eigen::Vector3d current_offset = drone_pos_gazebo_ - drone_pos_vins_;
        offset_samples_.push_back(current_offset);
        
        if ((int)offset_samples_.size() >= max_offset_samples_) {
            // 计算平均偏移量
            Eigen::Vector3d avg_offset = Eigen::Vector3d::Zero();
            for (const auto& sample : offset_samples_) {
                avg_offset += sample;
            }
            coord_offset_ = avg_offset / offset_samples_.size();
            coord_initialized_ = true;
            
            ROS_INFO("[GazeboTargetTracker] Coordinate offset initialized:");
            ROS_INFO("  offset = [%.3f, %.3f, %.3f]", 
                     coord_offset_.x(), coord_offset_.y(), coord_offset_.z());
            ROS_INFO("  drone_gazebo = [%.3f, %.3f, %.3f]",
                     drone_pos_gazebo_.x(), drone_pos_gazebo_.y(), drone_pos_gazebo_.z());
            ROS_INFO("  drone_vins = [%.3f, %.3f, %.3f]",
                     drone_pos_vins_.x(), drone_pos_vins_.y(), drone_pos_vins_.z());
        }
    }
    
    /**
     * @brief 定时发布目标位置
     */
    void publishCallback(const ros::TimerEvent& e) {
        // 检查状态
        if (!odom_received_) {
            ROS_WARN_THROTTLE(2.0, "[GazeboTargetTracker] Waiting for VINS odometry...");
            return;
        }
        if (!target_found_) {
            ROS_WARN_THROTTLE(2.0, "[GazeboTargetTracker] Target model '%s' not found in Gazebo!", 
                             target_model_name_.c_str());
            return;
        }
        if (!drone_found_) {
            ROS_WARN_THROTTLE(2.0, "[GazeboTargetTracker] Drone model '%s' not found in Gazebo!",
                             drone_model_name_.c_str());
            return;
        }
        if (!coord_initialized_) {
            ROS_INFO_THROTTLE(1.0, "[GazeboTargetTracker] Initializing coordinate offset... (%lu/%d samples)",
                             offset_samples_.size(), max_offset_samples_);
            return;
        }
        
        // 获取目标位置 (Gazebo坐标系)
        Eigen::Vector3d target_pos_gz, target_vel_gz;
        Eigen::Quaterniond target_quat_gz;
        {
            std::lock_guard<std::mutex> lock(gazebo_mutex_);
            target_pos_gz = target_pos_gazebo_;
            target_vel_gz = target_vel_gazebo_;
            target_quat_gz = target_quat_gazebo_;
        }
        
        // 转换到VINS坐标系
        // target_pos_vins = target_pos_gazebo - coord_offset_
        Eigen::Vector3d target_pos_vins = target_pos_gz - coord_offset_;
        Eigen::Vector3d target_vel_vins = target_vel_gz;  // 速度不需要偏移
        
        // 发布Odometry消息
        nav_msgs::Odometry odom_msg;
        odom_msg.header.stamp = ros::Time::now();
        odom_msg.header.frame_id = "world";
        odom_msg.child_frame_id = "target";
        
        odom_msg.pose.pose.position.x = target_pos_vins.x();
        odom_msg.pose.pose.position.y = target_pos_vins.y();
        odom_msg.pose.pose.position.z = target_pos_vins.z();
        
        odom_msg.pose.pose.orientation.w = target_quat_gz.w();
        odom_msg.pose.pose.orientation.x = target_quat_gz.x();
        odom_msg.pose.pose.orientation.y = target_quat_gz.y();
        odom_msg.pose.pose.orientation.z = target_quat_gz.z();
        
        odom_msg.twist.twist.linear.x = target_vel_vins.x();
        odom_msg.twist.twist.linear.y = target_vel_vins.y();
        odom_msg.twist.twist.linear.z = target_vel_vins.z();
        
        target_odom_pub_.publish(odom_msg);
        
        // 发布PoseStamped消息
        geometry_msgs::PoseStamped pose_msg;
        pose_msg.header = odom_msg.header;
        pose_msg.pose = odom_msg.pose.pose;
        target_pose_pub_.publish(pose_msg);
        
        ROS_INFO_THROTTLE(1.0, "[GazeboTargetTracker] Target (VINS): [%.2f, %.2f, %.2f], vel: [%.2f, %.2f, %.2f]",
                         target_pos_vins.x(), target_pos_vins.y(), target_pos_vins.z(),
                         target_vel_vins.x(), target_vel_vins.y(), target_vel_vins.z());
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "gazebo_target_tracker_node");
    
    GazeboTargetTracker tracker;
    
    ros::spin();
    return 0;
}
