/**
 * @file gazebo_target_node.cpp
 * @brief 从Gazebo获取目标位置并转换到VINS坐标系 (V7 修正版)
 * 
 * 核心方法：通过body坐标系作为中间桥梁进行坐标变换
 * 
 * 变换链：
 *   Gazebo世界 → Body坐标系 → VINS世界
 * 
 * 公式：
 *   relative_body = q_gazebo^(-1) * (target_gazebo - drone_gazebo)
 *   target_vins = drone_vins + q_vins * relative_body
 */

#include <ros/ros.h>
#include <gazebo_msgs/ModelStates.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Bool.h>
#include <string>
#include <Eigen/Geometry>
#include <mutex>
#include <deque>

/**
 * @brief EKF滤波器
 */
struct Ekf {
    double dt;
    Eigen::MatrixXd A, B, C;
    Eigen::MatrixXd Qt, Rt;
    Eigen::MatrixXd Sigma, K;
    Eigen::VectorXd x;
    
    Ekf(double _dt) : dt(_dt) {
        A.setIdentity(6, 6);
        Sigma.setZero(6, 6);
        B.setZero(6, 3);
        C.setZero(3, 6);
        A(0, 3) = dt; A(1, 4) = dt; A(2, 5) = dt;
        double t2 = dt * dt / 2;
        B(0, 0) = t2; B(1, 1) = t2; B(2, 2) = t2;
        B(3, 0) = dt; B(4, 1) = dt; B(5, 2) = dt;
        C(0, 0) = 1; C(1, 1) = 1; C(2, 2) = 1;
        K = C;
        Qt.setIdentity(3, 3);
        Rt.setIdentity(3, 3);
        Qt(0, 0) = 4; Qt(1, 1) = 4; Qt(2, 2) = 1;
        Rt(0, 0) = 0.1; Rt(1, 1) = 0.1; Rt(2, 2) = 0.1;
        x.setZero(6);
    }
    
    inline void predict() {
        x = A * x;
        Sigma = A * Sigma * A.transpose() + B * Qt * B.transpose();
    }
    
    inline void reset(const Eigen::Vector3d& z) {
        x.head(3) = z;
        x.tail(3).setZero();
        Sigma.setZero();
    }
    
    inline bool checkValid(const Eigen::Vector3d& z) const {
        Eigen::MatrixXd K_tmp = Sigma * C.transpose() * (C * Sigma * C.transpose() + Rt).inverse();
        Eigen::VectorXd x_tmp = x + K_tmp * (z - C * x);
        const double vmax = 10;
        return x_tmp.tail(3).norm() <= vmax;
    }
    
    inline void update(const Eigen::Vector3d& z) {
        K = Sigma * C.transpose() * (C * Sigma * C.transpose() + Rt).inverse();
        x = x + K * (z - C * x);
        Sigma = Sigma - K * C * Sigma;
    }
    
    inline const Eigen::Vector3d pos() const { return x.head(3); }
    inline const Eigen::Vector3d vel() const { return x.tail(3); }
};


class GazeboTargetPub {
private:
    ros::NodeHandle nh_;
    ros::Publisher target_odom_pub_;
    ros::Publisher raw_odom_pub_;
    ros::Subscriber model_states_sub_;
    ros::Subscriber vins_odom_sub_;
    ros::Subscriber enable_sub_;
    ros::Timer ekf_timer_;
    
    std::string target_model_name_;
    std::string drone_model_name_;
    
    // 启用控制
    bool enabled_ = false;
    
    // Gazebo数据
    Eigen::Vector3d drone_p_gazebo_;
    Eigen::Quaterniond drone_q_gazebo_;
    Eigen::Vector3d target_p_gazebo_;
    Eigen::Vector3d target_v_gazebo_;
    std::mutex gazebo_mutex_;
    bool target_found_ = false;
    bool drone_found_ = false;
    
    // VINS数据
    Eigen::Vector3d drone_p_vins_;
    Eigen::Quaterniond drone_q_vins_;
    std::mutex vins_mutex_;
    bool vins_odom_received_ = false;
    
    // VINS稳定性检测
    std::deque<Eigen::Vector3d> vins_history_;
    static const int STABILITY_WINDOW = 30;
    double stability_threshold_ = 0.3;
    
    // EKF
    bool use_ekf_ = true;
    int ekf_rate_ = 30;
    std::shared_ptr<Ekf> ekfPtr_;
    ros::Time last_update_stamp_;
    
    // 距离限制
    double max_tracking_dist_ = 15.0;
    double min_tracking_dist_ = 0.3;
    
public:
    GazeboTargetPub(ros::NodeHandle& nh) : nh_(nh) {
        nh_.param<std::string>("target_model_name", target_model_name_, "hatchback_red");
        nh_.param<std::string>("drone_model_name", drone_model_name_, "iris");
        nh_.param<bool>("use_ekf", use_ekf_, true);
        nh_.param<int>("ekf_rate", ekf_rate_, 30);
        nh_.param<double>("stability_threshold", stability_threshold_, 0.3);
        nh_.param<double>("max_tracking_dist", max_tracking_dist_, 15.0);
        
        ekfPtr_ = std::make_shared<Ekf>(1.0 / ekf_rate_);
        last_update_stamp_ = ros::Time::now();
        
        // 发布者
        target_odom_pub_ = nh_.advertise<nav_msgs::Odometry>("target_odom", 10);
        raw_odom_pub_ = nh_.advertise<nav_msgs::Odometry>("raw_odom", 10);
        
        // 订阅者
        model_states_sub_ = nh_.subscribe("/gazebo/model_states", 10, 
                                          &GazeboTargetPub::modelStatesCallback, this);
        vins_odom_sub_ = nh_.subscribe("odom", 10, 
                                        &GazeboTargetPub::vinsOdomCallback, this);
        enable_sub_ = nh_.subscribe("enable", 10, 
                                     &GazeboTargetPub::enableCallback, this);
        
        // EKF预测定时器
        if (use_ekf_) {
            ekf_timer_ = nh_.createTimer(ros::Duration(1.0 / ekf_rate_), 
                                         &GazeboTargetPub::ekfPredictCallback, this);
        }
        
        ROS_INFO("================================================");
        ROS_INFO("[gazebo_target] V7 Fixed - Body Frame Transform");
        ROS_INFO("================================================");
        ROS_INFO("[gazebo_target] Target: %s", target_model_name_.c_str());
        ROS_INFO("[gazebo_target] Drone: %s", drone_model_name_.c_str());
        ROS_WARN("[gazebo_target] Tracking DISABLED by default");
        ROS_WARN("[gazebo_target] Wait for VINS stable, then enable:");
        ROS_WARN("[gazebo_target]   rostopic pub /gazebo_target_node/enable std_msgs/Bool \"data: true\" -1");
        ROS_INFO("================================================");
    }
    
private:
    bool checkVinsStability() {
        if (vins_history_.size() < STABILITY_WINDOW) return false;
        Eigen::Vector3d min_p = vins_history_[0], max_p = vins_history_[0];
        for (const auto& p : vins_history_) {
            for (int i = 0; i < 3; ++i) {
                min_p[i] = std::min(min_p[i], p[i]);
                max_p[i] = std::max(max_p[i], p[i]);
            }
        }
        return (max_p - min_p).norm() < stability_threshold_;
    }
    
    /**
     * @brief 核心函数：计算目标在VINS坐标系中的位置
     */
    Eigen::Vector3d computeTargetPositionVins() {
        Eigen::Vector3d drone_p_gazebo_copy, target_p_gazebo_copy;
        Eigen::Quaterniond drone_q_gazebo_copy;
        Eigen::Vector3d drone_p_vins_copy;
        Eigen::Quaterniond drone_q_vins_copy;
        
        {
            std::lock_guard<std::mutex> lock(gazebo_mutex_);
            drone_p_gazebo_copy = drone_p_gazebo_;
            target_p_gazebo_copy = target_p_gazebo_;
            drone_q_gazebo_copy = drone_q_gazebo_;
        }
        {
            std::lock_guard<std::mutex> lock(vins_mutex_);
            drone_p_vins_copy = drone_p_vins_;
            drone_q_vins_copy = drone_q_vins_;
        }
        
        // 步骤1: Gazebo世界坐标系中的相对向量
        Eigen::Vector3d relative_gazebo = target_p_gazebo_copy - drone_p_gazebo_copy;
        
        // 步骤2: 转换到body坐标系
        Eigen::Vector3d relative_body = drone_q_gazebo_copy.inverse() * relative_gazebo;
        
        // 步骤3: 转换到VINS世界坐标系
        Eigen::Vector3d relative_vins = drone_q_vins_copy * relative_body;
        
        // 步骤4: 加上无人机在VINS中的位置
        Eigen::Vector3d target_p_vins = drone_p_vins_copy + relative_vins;
        
        return target_p_vins;
    }
    
    Eigen::Vector3d computeTargetVelocityVins() {
        Eigen::Vector3d target_v_gazebo_copy;
        Eigen::Quaterniond drone_q_gazebo_copy, drone_q_vins_copy;
        
        {
            std::lock_guard<std::mutex> lock(gazebo_mutex_);
            target_v_gazebo_copy = target_v_gazebo_;
            drone_q_gazebo_copy = drone_q_gazebo_;
        }
        {
            std::lock_guard<std::mutex> lock(vins_mutex_);
            drone_q_vins_copy = drone_q_vins_;
        }
        
        Eigen::Vector3d v_body = drone_q_gazebo_copy.inverse() * target_v_gazebo_copy;
        Eigen::Vector3d v_vins = drone_q_vins_copy * v_body;
        
        return v_vins;
    }
    
    void enableCallback(const std_msgs::Bool::ConstPtr& msg) {
        if (msg->data && !enabled_) {
            // 检查条件
            if (!vins_odom_received_) {
                ROS_WARN("[gazebo_target] Cannot enable: VINS not received!");
                return;
            }
            if (!target_found_ || !drone_found_) {
                ROS_WARN("[gazebo_target] Cannot enable: Models not found!");
                return;
            }
            if (!checkVinsStability()) {
                ROS_WARN("[gazebo_target] Cannot enable: VINS not stable! Wait longer.");
                return;
            }
            
            // 重置EKF
            Eigen::Vector3d target_p_vins = computeTargetPositionVins();
            ekfPtr_->reset(target_p_vins);
            last_update_stamp_ = ros::Time::now();
            
            enabled_ = true;
            
            double dist = (target_p_gazebo_ - drone_p_gazebo_).norm();
            ROS_INFO("================================================");
            ROS_INFO("[gazebo_target] *** TRACKING ENABLED ***");
            ROS_INFO("[gazebo_target] Distance: %.2f m", dist);
            ROS_INFO("[gazebo_target] Target VINS: (%.2f, %.2f, %.2f)",
                     target_p_vins.x(), target_p_vins.y(), target_p_vins.z());
            ROS_INFO("================================================");
        } 
        else if (!msg->data && enabled_) {
            enabled_ = false;
            ROS_INFO("[gazebo_target] *** TRACKING DISABLED ***");
        }
    }
    
    void vinsOdomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(vins_mutex_);
        
        drone_p_vins_ = Eigen::Vector3d(
            msg->pose.pose.position.x,
            msg->pose.pose.position.y,
            msg->pose.pose.position.z
        );
        drone_q_vins_ = Eigen::Quaterniond(
            msg->pose.pose.orientation.w,
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z
        );
        drone_q_vins_.normalize();
        
        vins_history_.push_back(drone_p_vins_);
        if (vins_history_.size() > STABILITY_WINDOW) {
            vins_history_.pop_front();
        }
        
        if (!vins_odom_received_) {
            vins_odom_received_ = true;
            ROS_INFO("[gazebo_target] VINS odometry received");
        }
    }
    
    void ekfPredictCallback(const ros::TimerEvent& event) {
        if (!enabled_) return;
        
        double dt = (ros::Time::now() - last_update_stamp_).toSec();
        if (dt < 2.0) {
            ekfPtr_->predict();
        } else {
            return;
        }
        
        // 发布EKF滤波后的位置
        nav_msgs::Odometry target_odom;
        target_odom.header.stamp = ros::Time::now();
        target_odom.header.frame_id = "world";
        target_odom.pose.pose.position.x = ekfPtr_->pos().x();
        target_odom.pose.pose.position.y = ekfPtr_->pos().y();
        target_odom.pose.pose.position.z = ekfPtr_->pos().z();
        target_odom.twist.twist.linear.x = ekfPtr_->vel().x();
        target_odom.twist.twist.linear.y = ekfPtr_->vel().y();
        target_odom.twist.twist.linear.z = ekfPtr_->vel().z();
        target_odom.pose.pose.orientation.w = 1.0;
        target_odom_pub_.publish(target_odom);
    }
    
    void modelStatesCallback(const gazebo_msgs::ModelStates::ConstPtr& msg) {
        int target_idx = -1, drone_idx = -1;
        for (size_t i = 0; i < msg->name.size(); ++i) {
            if (msg->name[i] == target_model_name_) target_idx = i;
            if (msg->name[i] == drone_model_name_) drone_idx = i;
        }
        
        if (target_idx < 0 || drone_idx < 0) {
            ROS_WARN_THROTTLE(5.0, "[gazebo_target] Models not found!");
            return;
        }
        
        if (!target_found_) { target_found_ = true; ROS_INFO("[gazebo_target] Target found"); }
        if (!drone_found_) { drone_found_ = true; ROS_INFO("[gazebo_target] Drone found"); }
        
        // 更新Gazebo数据
        {
            std::lock_guard<std::mutex> lock(gazebo_mutex_);
            drone_p_gazebo_ = Eigen::Vector3d(
                msg->pose[drone_idx].position.x,
                msg->pose[drone_idx].position.y,
                msg->pose[drone_idx].position.z
            );
            drone_q_gazebo_ = Eigen::Quaterniond(
                msg->pose[drone_idx].orientation.w,
                msg->pose[drone_idx].orientation.x,
                msg->pose[drone_idx].orientation.y,
                msg->pose[drone_idx].orientation.z
            );
            drone_q_gazebo_.normalize();
            
            target_p_gazebo_ = Eigen::Vector3d(
                msg->pose[target_idx].position.x,
                msg->pose[target_idx].position.y,
                msg->pose[target_idx].position.z
            );
            target_v_gazebo_ = Eigen::Vector3d(
                msg->twist[target_idx].linear.x,
                msg->twist[target_idx].linear.y,
                msg->twist[target_idx].linear.z
            );
        }
        
        // 禁用状态：输出信息
        if (!enabled_) {
            static int count = 0;
            if (++count % 60 == 0) {
                double dist = (target_p_gazebo_ - drone_p_gazebo_).norm();
                bool stable = checkVinsStability();
                ROS_INFO("[gazebo_target] DISABLED | dist=%.1fm | VINS %s",
                         dist, stable ? "STABLE - ready" : "unstable");
            }
            return;
        }
        
        // === 启用状态 ===
        
        double distance_gazebo = (target_p_gazebo_ - drone_p_gazebo_).norm();
        
        if (distance_gazebo > max_tracking_dist_) {
            ROS_WARN_THROTTLE(1.0, "[gazebo_target] Target too far: %.1f m (max: %.1f m)", 
                             distance_gazebo, max_tracking_dist_);
            return;
        }
        
        // 计算目标位置和速度
        Eigen::Vector3d target_p_vins = computeTargetPositionVins();
        Eigen::Vector3d target_v_vins = computeTargetVelocityVins();
        
        // 调试输出
        static int debug_count = 0;
        if (++debug_count % 30 == 0) {
            ROS_INFO("[gazebo_target] ENABLED | dist=%.2fm | target=(%.2f,%.2f,%.2f)",
                     distance_gazebo,
                     target_p_vins.x(), target_p_vins.y(), target_p_vins.z());
        }
        
        // 发布原始位置
        nav_msgs::Odometry raw_odom;
        raw_odom.header.stamp = ros::Time::now();
        raw_odom.header.frame_id = "world";
        raw_odom.pose.pose.position.x = target_p_vins.x();
        raw_odom.pose.pose.position.y = target_p_vins.y();
        raw_odom.pose.pose.position.z = target_p_vins.z();
        raw_odom.twist.twist.linear.x = target_v_vins.x();
        raw_odom.twist.twist.linear.y = target_v_vins.y();
        raw_odom.twist.twist.linear.z = target_v_vins.z();
        raw_odom.pose.pose.orientation.w = 1.0;
        raw_odom_pub_.publish(raw_odom);
        
        // EKF更新
        if (use_ekf_) {
            double dt = (ros::Time::now() - last_update_stamp_).toSec();
            if (dt > 3.0) {
                ekfPtr_->reset(target_p_vins);
                ROS_WARN("[gazebo_target] EKF reset (timeout)");
            } else if (ekfPtr_->checkValid(target_p_vins)) {
                ekfPtr_->update(target_p_vins);
            } else {
                ROS_WARN_THROTTLE(1.0, "[gazebo_target] EKF update skipped");
                return;
            }
            last_update_stamp_ = ros::Time::now();
        } else {
            target_odom_pub_.publish(raw_odom);
        }
    }
};


int main(int argc, char** argv) {
    ros::init(argc, argv, "gazebo_target_node");
    ros::NodeHandle nh("~");
    GazeboTargetPub node(nh);
    ros::spin();
    return 0;
}