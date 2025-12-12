/**
 * @file simple_follower_node.cpp
 * @brief 简单的目标跟随控制器
 * 
 * 该节点订阅目标位置和无人机位置，
 * 计算跟随点并发送位置控制指令给PX4。
 * 无人机会跟在目标后方指定距离处。
 */

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/PositionTarget.h>
#include <std_msgs/Empty.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <mutex>
#include <atomic>
#include <cmath>

class SimpleFollower {
private:
    ros::NodeHandle nh_;
    
    // 订阅者
    ros::Subscriber odom_sub_;
    ros::Subscriber target_sub_;
    ros::Subscriber px4_state_sub_;
    
    // 发布者
    ros::Publisher cmd_pub_;
    ros::Publisher heartbeat_pub_;
    
    // 定时器
    ros::Timer control_timer_;
    
    // 参数
    double tracking_distance_;      // 跟随距离
    double tracking_height_;        // 相对目标的高度
    double control_rate_;           // 控制频率
    double max_velocity_;           // 最大速度
    double position_gain_;          // 位置控制增益
    double yaw_rate_limit_;         // 偏航角速率限制
    
    // 状态标志
    std::atomic<bool> odom_received_{false};
    std::atomic<bool> target_received_{false};
    std::atomic<bool> is_offboard_{false};
    std::atomic<bool> is_armed_{false};
    
    // 互斥锁
    std::mutex odom_mutex_;
    std::mutex target_mutex_;
    
    // 无人机状态
    Eigen::Vector3d drone_pos_;
    Eigen::Vector3d drone_vel_;
    Eigen::Quaterniond drone_quat_;
    double drone_yaw_;
    
    // 目标状态
    Eigen::Vector3d target_pos_;
    Eigen::Vector3d target_vel_;
    Eigen::Quaterniond target_quat_;
    
    // 控制状态
    double last_yaw_;
    Eigen::Vector3d last_cmd_pos_;
    ros::Time last_target_time_;
    
public:
    SimpleFollower() : nh_("~") {
        // 读取参数
        nh_.param<double>("tracking_distance", tracking_distance_, 1.5);
        nh_.param<double>("tracking_height", tracking_height_, 1.2);
        nh_.param<double>("control_rate", control_rate_, 30.0);
        nh_.param<double>("max_velocity", max_velocity_, 1.5);
        nh_.param<double>("position_gain", position_gain_, 1.0);
        nh_.param<double>("yaw_rate_limit", yaw_rate_limit_, 0.5);
        
        // 初始化
        drone_pos_.setZero();
        drone_vel_.setZero();
        drone_quat_.setIdentity();
        drone_yaw_ = 0.0;
        
        target_pos_.setZero();
        target_vel_.setZero();
        target_quat_.setIdentity();
        
        last_yaw_ = 0.0;
        last_cmd_pos_.setZero();
        last_target_time_ = ros::Time::now();
        
        // 订阅者
        odom_sub_ = nh_.subscribe("/vins_fusion/imu_propagate", 10,
                                  &SimpleFollower::odomCallback, this,
                                  ros::TransportHints().tcpNoDelay());
        target_sub_ = nh_.subscribe("target_odom", 10,
                                    &SimpleFollower::targetCallback, this,
                                    ros::TransportHints().tcpNoDelay());
        px4_state_sub_ = nh_.subscribe("/mavros/state", 1,
                                       &SimpleFollower::px4StateCallback, this,
                                       ros::TransportHints().tcpNoDelay());
        
        // 发布者
        cmd_pub_ = nh_.advertise<mavros_msgs::PositionTarget>(
            "/mavros/setpoint_raw/local", 10);
        heartbeat_pub_ = nh_.advertise<std_msgs::Empty>("heartbeat", 10);
        
        // 控制定时器
        control_timer_ = nh_.createTimer(ros::Duration(1.0 / control_rate_),
                                         &SimpleFollower::controlCallback, this);
        
        ROS_INFO("[SimpleFollower] Initialized.");
        ROS_INFO("[SimpleFollower] Tracking distance: %.2f m", tracking_distance_);
        ROS_INFO("[SimpleFollower] Tracking height: %.2f m", tracking_height_);
    }
    
    /**
     * @brief VINS里程计回调
     */
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(odom_mutex_);
        
        drone_pos_.x() = msg->pose.pose.position.x;
        drone_pos_.y() = msg->pose.pose.position.y;
        drone_pos_.z() = msg->pose.pose.position.z;
        
        drone_vel_.x() = msg->twist.twist.linear.x;
        drone_vel_.y() = msg->twist.twist.linear.y;
        drone_vel_.z() = msg->twist.twist.linear.z;
        
        drone_quat_.w() = msg->pose.pose.orientation.w;
        drone_quat_.x() = msg->pose.pose.orientation.x;
        drone_quat_.y() = msg->pose.pose.orientation.y;
        drone_quat_.z() = msg->pose.pose.orientation.z;
        drone_quat_.normalize();
        
        // 计算偏航角
        Eigen::Matrix3d R = drone_quat_.toRotationMatrix();
        drone_yaw_ = std::atan2(R(1, 0), R(0, 0));
        
        if (!odom_received_) {
            last_cmd_pos_ = drone_pos_;
            last_yaw_ = drone_yaw_;
        }
        
        odom_received_ = true;
    }
    
    /**
     * @brief 目标位置回调
     */
    void targetCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(target_mutex_);
        
        target_pos_.x() = msg->pose.pose.position.x;
        target_pos_.y() = msg->pose.pose.position.y;
        target_pos_.z() = msg->pose.pose.position.z;
        
        target_vel_.x() = msg->twist.twist.linear.x;
        target_vel_.y() = msg->twist.twist.linear.y;
        target_vel_.z() = msg->twist.twist.linear.z;
        
        target_quat_.w() = msg->pose.pose.orientation.w;
        target_quat_.x() = msg->pose.pose.orientation.x;
        target_quat_.y() = msg->pose.pose.orientation.y;
        target_quat_.z() = msg->pose.pose.orientation.z;
        target_quat_.normalize();
        
        last_target_time_ = ros::Time::now();
        target_received_ = true;
    }
    
    /**
     * @brief PX4状态回调
     */
    void px4StateCallback(const mavros_msgs::State::ConstPtr& msg) {
        is_armed_ = msg->armed;
        is_offboard_ = (msg->mode == "OFFBOARD");
    }
    
    /**
     * @brief 计算跟随点位置
     * 在目标后方tracking_distance_距离处
     */
    Eigen::Vector3d computeFollowPoint() {
        std::lock_guard<std::mutex> lock(target_mutex_);
        
        // 获取目标朝向
        Eigen::Matrix3d target_rot = target_quat_.toRotationMatrix();
        Eigen::Vector3d target_forward = target_rot.col(0);  // X轴是前进方向
        
        // 只取水平分量
        Eigen::Vector3d forward_horizontal(target_forward.x(), target_forward.y(), 0);
        if (forward_horizontal.norm() < 0.01) {
            // 如果目标几乎朝上或朝下，使用从无人机到目标的方向
            std::lock_guard<std::mutex> odom_lock(odom_mutex_);
            forward_horizontal = target_pos_ - drone_pos_;
            forward_horizontal.z() = 0;
        }
        
        if (forward_horizontal.norm() > 0.01) {
            forward_horizontal.normalize();
        } else {
            forward_horizontal << 1, 0, 0;  // 默认方向
        }
        
        // 跟随点在目标后方
        Eigen::Vector3d follow_point = target_pos_ - tracking_distance_ * forward_horizontal;
        
        // 设置高度
        follow_point.z() = target_pos_.z() + tracking_height_;
        
        return follow_point;
    }
    
    /**
     * @brief 计算期望偏航角（朝向目标）
     */
    double computeDesiredYaw() {
        std::lock_guard<std::mutex> odom_lock(odom_mutex_);
        std::lock_guard<std::mutex> target_lock(target_mutex_);
        
        Eigen::Vector3d dir = target_pos_ - drone_pos_;
        return std::atan2(dir.y(), dir.x());
    }
    
    /**
     * @brief 限制偏航角变化率
     */
    double limitYawRate(double desired_yaw, double dt) {
        double d_yaw = desired_yaw - last_yaw_;
        
        // 处理角度跨越±π的情况
        while (d_yaw > M_PI) d_yaw -= 2 * M_PI;
        while (d_yaw < -M_PI) d_yaw += 2 * M_PI;
        
        // 限制变化率
        double max_d_yaw = yaw_rate_limit_ * dt;
        if (std::fabs(d_yaw) > max_d_yaw) {
            d_yaw = (d_yaw > 0) ? max_d_yaw : -max_d_yaw;
        }
        
        return last_yaw_ + d_yaw;
    }
    
    /**
     * @brief 发布位置控制指令
     */
    void publishCommand(const Eigen::Vector3d& position, 
                       const Eigen::Vector3d& velocity,
                       double yaw) {
        mavros_msgs::PositionTarget cmd;
        cmd.header.stamp = ros::Time::now();
        cmd.header.frame_id = "world";
        cmd.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
        
        // 位置
        cmd.position.x = position.x();
        cmd.position.y = position.y();
        cmd.position.z = position.z();
        
        // 速度前馈
        cmd.velocity.x = velocity.x();
        cmd.velocity.y = velocity.y();
        cmd.velocity.z = velocity.z();
        
        // 偏航角
        cmd.yaw = yaw;
        cmd.yaw_rate = 0;
        
        // 设置type_mask，指定使用位置+速度+偏航角控制
        cmd.type_mask = mavros_msgs::PositionTarget::IGNORE_AFX |
                       mavros_msgs::PositionTarget::IGNORE_AFY |
                       mavros_msgs::PositionTarget::IGNORE_AFZ |
                       mavros_msgs::PositionTarget::IGNORE_YAW_RATE;
        
        cmd_pub_.publish(cmd);
    }
    
    /**
     * @brief 发布悬停指令
     */
    void publishHover() {
        std::lock_guard<std::mutex> lock(odom_mutex_);
        publishCommand(last_cmd_pos_, Eigen::Vector3d::Zero(), last_yaw_);
    }
    
    /**
     * @brief 控制循环
     */
    void controlCallback(const ros::TimerEvent& e) {
        // 发送心跳
        heartbeat_pub_.publish(std_msgs::Empty());
        
        double dt = 1.0 / control_rate_;
        
        // 检查里程计
        if (!odom_received_) {
            ROS_WARN_THROTTLE(2.0, "[SimpleFollower] Waiting for odometry...");
            return;
        }
        
        // 检查是否在Offboard模式
        if (!is_offboard_) {
            // 不在Offboard模式时，持续发送当前位置作为setpoint
            std::lock_guard<std::mutex> lock(odom_mutex_);
            last_cmd_pos_ = drone_pos_;
            last_yaw_ = drone_yaw_;
            publishCommand(drone_pos_, Eigen::Vector3d::Zero(), drone_yaw_);
            ROS_INFO_THROTTLE(2.0, "[SimpleFollower] Waiting for OFFBOARD mode...");
            return;
        }
        
        // 检查目标
        if (!target_received_) {
            ROS_WARN_THROTTLE(2.0, "[SimpleFollower] Waiting for target...");
            publishHover();
            return;
        }
        
        // 检查目标数据是否超时
        double target_age = (ros::Time::now() - last_target_time_).toSec();
        if (target_age > 1.0) {
            ROS_WARN_THROTTLE(1.0, "[SimpleFollower] Target data timeout (%.2f s), hovering...", 
                             target_age);
            publishHover();
            return;
        }
        
        // 计算跟随点
        Eigen::Vector3d follow_point = computeFollowPoint();
        
        // 计算期望偏航角
        double desired_yaw = computeDesiredYaw();
        double cmd_yaw = limitYawRate(desired_yaw, dt);
        
        // 获取当前位置和目标速度
        Eigen::Vector3d current_pos, target_vel_local;
        {
            std::lock_guard<std::mutex> odom_lock(odom_mutex_);
            current_pos = drone_pos_;
        }
        {
            std::lock_guard<std::mutex> target_lock(target_mutex_);
            target_vel_local = target_vel_;
        }
        
        // 计算速度指令（前馈 + 位置误差反馈）
        Eigen::Vector3d pos_error = follow_point - current_pos;
        Eigen::Vector3d cmd_vel = target_vel_local + position_gain_ * pos_error;
        
        // 限制速度
        double vel_norm = cmd_vel.head<2>().norm();  // 水平速度
        if (vel_norm > max_velocity_) {
            cmd_vel.head<2>() *= max_velocity_ / vel_norm;
        }
        cmd_vel.z() = std::max(-1.0, std::min(1.0, cmd_vel.z()));  // 垂直速度限制
        
        // 发布指令
        publishCommand(follow_point, cmd_vel, cmd_yaw);
        
        // 更新状态
        last_cmd_pos_ = follow_point;
        last_yaw_ = cmd_yaw;
        
        // 调试信息
        ROS_INFO_THROTTLE(1.0, "[SimpleFollower] Following: target=[%.2f,%.2f,%.2f], "
                         "follow_pt=[%.2f,%.2f,%.2f], drone=[%.2f,%.2f,%.2f]",
                         target_pos_.x(), target_pos_.y(), target_pos_.z(),
                         follow_point.x(), follow_point.y(), follow_point.z(),
                         current_pos.x(), current_pos.y(), current_pos.z());
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "simple_follower_node");
    
    SimpleFollower follower;
    
    ros::spin();
    return 0;
}
