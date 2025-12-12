/**
 * simple_tracker_node.cpp (V4.1 安全加固版)
 * 功能：
 * 1. 使用标准 ENU 坐标 + frame_id="world"，由 MAVROS 负责底层转换，解决乱飞/坠机问题。
 * 2. 增加[安全刹车]机制：如果距离误差过大，自动降速或悬停，防止坐标错误时无人机飞丢。
 */

#include <ros/ros.h>
#include <gazebo_msgs/ModelStates.h>
#include <nav_msgs/Odometry.h>
#include <mavros_msgs/PositionTarget.h>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <string>

class SimpleTracker {
public:
    SimpleTracker(ros::NodeHandle& nh) {
        // 1. 订阅 Gazebo 真值
        gazebo_sub_ = nh.subscribe("/gazebo/model_states", 1, &SimpleTracker::gazeboCallback, this);

        // 2. 订阅 VINS (用于获取当前无人机位置，做安全检查)
        vins_sub_ = nh.subscribe("/vins_fusion/imu_propagate", 1, &SimpleTracker::vinsCallback, this);

        // 3. 发布控制指令
        cmd_pub_ = nh.advertise<mavros_msgs::PositionTarget>("/mavros/setpoint_raw/local", 1);

        target_name_ = "hatchback_red";
        
        // --- 跟踪参数 ---
        follow_distance_ = 1.0;  // 目标后方 1米
        follow_height_   = 1.5;  // 目标上方 1.5米
        
        // --- 安全参数 ---
        max_position_error_ = 3.0; // 如果误差超过3米，认为异常，取消速度前馈
        emergency_stop_dist_ = 8.0; // 如果误差超过8米，强制悬停
        
        ROS_INFO("Simple Tracker V4.1 (Safe Mode) Started. Target: %s", target_name_.c_str());
    }

    void vinsCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        drone_pos_ = Eigen::Vector3d(msg->pose.pose.position.x,
                                     msg->pose.pose.position.y,
                                     msg->pose.pose.position.z);
        has_odom_ = true;
    }

    void gazeboCallback(const gazebo_msgs::ModelStates::ConstPtr& msg) {
        if (!has_odom_) return; // 等待 VINS 初始化

        int idx = -1;
        for (size_t i = 0; i < msg->name.size(); ++i) {
            if (msg->name[i] == target_name_) {
                idx = i;
                break;
            }
        }

        if (idx == -1) return;

        // 1. 获取目标 ENU 状态
        Eigen::Vector3d car_pos(msg->pose[idx].position.x, 
                                msg->pose[idx].position.y, 
                                msg->pose[idx].position.z);
        
        Eigen::Vector3d car_vel(msg->twist[idx].linear.x, 
                                msg->twist[idx].linear.y, 
                                msg->twist[idx].linear.z);

        Eigen::Quaterniond car_quat(msg->pose[idx].orientation.w,
                                    msg->pose[idx].orientation.x,
                                    msg->pose[idx].orientation.y,
                                    msg->pose[idx].orientation.z);

        // 2. 计算期望位置 (ENU)
        // 提取 Yaw 角 (忽略车身倾斜)
        Eigen::Vector3d euler = car_quat.toRotationMatrix().eulerAngles(2, 1, 0); 
        double car_yaw = euler[0]; 

        // 计算车后方偏移
        Eigen::AngleAxisd yaw_rot(car_yaw, Eigen::Vector3d::UnitZ());
        Eigen::Vector3d offset_body(-follow_distance_, 0, 0); // 车后方
        Eigen::Vector3d target_pos_final = car_pos + yaw_rot * offset_body;
        target_pos_final.z() = car_pos.z() + follow_height_; // 叠加高度

        // 3. 安全检查 (防止飞丢)
        double dist_err = (Eigen::Vector2d(target_pos_final.x(), target_pos_final.y()) - 
                           Eigen::Vector2d(drone_pos_.x(), drone_pos_.y())).norm();

        mavros_msgs::PositionTarget cmd_msg;
        cmd_msg.header.stamp = ros::Time::now();
        cmd_msg.header.frame_id = "world"; // 关键：告诉MAVROS这是ENU坐标，请自动转换
        cmd_msg.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
        
        // 默认掩码：启用 位置 + 速度 + 偏航
        cmd_msg.type_mask = mavros_msgs::PositionTarget::IGNORE_AFX |
                            mavros_msgs::PositionTarget::IGNORE_AFY |
                            mavros_msgs::PositionTarget::IGNORE_AFZ |
                            mavros_msgs::PositionTarget::IGNORE_YAW_RATE;

        // --- 安全策略逻辑 ---
        if (dist_err > emergency_stop_dist_) {
            // [紧急情况] 距离太远(>8m)，可能是坐标系错了，强制原地悬停，防止炸机
            ROS_WARN_THROTTLE(1.0, "EMERGENCY: Distance %.1fm > Limit! Hovering.", dist_err);
            cmd_msg.position.x = drone_pos_.x();
            cmd_msg.position.y = drone_pos_.y();
            cmd_msg.position.z = drone_pos_.z();
            cmd_msg.velocity.x = 0;
            cmd_msg.velocity.y = 0;
            cmd_msg.velocity.z = 0;
        } 
        else if (dist_err > max_position_error_) {
            // [警告情况] 距离较远(>3m)，只发位置指令，不发速度前馈 (防止正反馈震荡)
            // 让无人机慢慢飞过去
            cmd_msg.type_mask |= mavros_msgs::PositionTarget::IGNORE_VX |
                                 mavros_msgs::PositionTarget::IGNORE_VY |
                                 mavros_msgs::PositionTarget::IGNORE_VZ;
            
            cmd_msg.position.x = target_pos_final.x();
            cmd_msg.position.y = target_pos_final.y();
            cmd_msg.position.z = std::max(0.5, target_pos_final.z()); // 最低高度保护
            cmd_msg.yaw = car_yaw;
            
            ROS_WARN_THROTTLE(1.0, "Gap Large (%.1fm). Disabling feedforward.", dist_err);
        }
        else {
            // [正常情况] 距离接近，开启全功能跟踪 (位置 + 速度前馈)
            cmd_msg.position.x = target_pos_final.x();
            cmd_msg.position.y = target_pos_final.y();
            cmd_msg.position.z = std::max(0.5, target_pos_final.z());
            
            cmd_msg.velocity.x = car_vel.x();
            cmd_msg.velocity.y = car_vel.y();
            cmd_msg.velocity.z = car_vel.z();
            
            cmd_msg.yaw = car_yaw;
        }

        cmd_pub_.publish(cmd_msg);

        // 4. 详细调试日志
        ROS_INFO_THROTTLE(1.0, "ENU | Car(%.1f, %.1f) | Drone(%.1f, %.1f) | Gap: %.1fm",
            car_pos.x(), car_pos.y(),
            drone_pos_.x(), drone_pos_.y(),
            dist_err);
    }

private:
    ros::Subscriber gazebo_sub_;
    ros::Subscriber vins_sub_;
    ros::Publisher cmd_pub_;
    
    std::string target_name_;
    double follow_distance_;
    double follow_height_;
    double max_position_error_;
    double emergency_stop_dist_;
    
    Eigen::Vector3d drone_pos_;
    bool has_odom_ = false;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "simple_tracker_node");
    ros::NodeHandle nh("~");
    SimpleTracker tracker(nh);
    ros::spin();
    return 0;
}