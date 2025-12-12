/**
 * data_collector_node.cpp
 * 
 * 数据收集节点：订阅目标位置，保存轨迹数据用于训练
 * 
 * 功能：
 * 1. 订阅 /target_ekf_node/target_odom 获取目标位置和速度
 * 2. 维护滑动窗口存储历史轨迹
 * 3. 保存完整轨迹到CSV文件，用于离线训练
 * 
 * 数据格式 (每行):
 * timestamp, x, y, z, vx, vy, vz, trajectory_id
 */

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Bool.h>

#include <fstream>
#include <sstream>
#include <deque>
#include <mutex>
#include <cmath>
#include <iomanip>

class DataCollector {
private:
    ros::NodeHandle nh_;
    ros::Subscriber target_sub_;
    ros::Subscriber recording_trigger_sub_;
    
    // 数据存储
    std::ofstream data_file_;
    std::string save_path_;
    
    // 轨迹管理
    int trajectory_id_;
    bool is_recording_;
    double last_timestamp_;
    
    // 参数
    double min_velocity_threshold_;  // 最小速度阈值，过滤静止数据
    double max_position_jump_;       // 最大位置跳变，过滤异常数据
    double sample_rate_;             // 期望采样率 (Hz)
    
    // 上一个位置，用于检测跳变
    double last_x_, last_y_, last_z_;
    bool has_last_position_;
    
    // 统计
    int total_samples_;
    int valid_samples_;
    
    std::mutex data_mutex_;

public:
    DataCollector() : nh_("~"), trajectory_id_(0), is_recording_(false), 
                      has_last_position_(false), total_samples_(0), valid_samples_(0) {
        
        // 加载参数
        nh_.param<std::string>("save_path", save_path_, 
            "/home/claude/trajectory_prediction/data/raw_trajectories.csv");
        nh_.param<double>("min_velocity_threshold", min_velocity_threshold_, 0.05);  // m/s
        nh_.param<double>("max_position_jump", max_position_jump_, 1.0);  // m
        nh_.param<double>("sample_rate", sample_rate_, 30.0);  // Hz
        
        // 创建数据目录
        std::string dir = save_path_.substr(0, save_path_.find_last_of('/'));
        std::string cmd = "mkdir -p " + dir;
        system(cmd.c_str());
        
        // 打开文件，写入表头
        data_file_.open(save_path_, std::ios::out | std::ios::app);
        if (!data_file_.is_open()) {
            ROS_ERROR("Failed to open data file: %s", save_path_.c_str());
            ros::shutdown();
            return;
        }
        
        // 检查文件是否为空，如果为空则写入表头
        data_file_.seekp(0, std::ios::end);
        if (data_file_.tellp() == 0) {
            data_file_ << "timestamp,x,y,z,vx,vy,vz,trajectory_id" << std::endl;
        }
        
        // 订阅目标里程计
        target_sub_ = nh_.subscribe("/target_ekf_node/target_odom", 10, 
                                     &DataCollector::targetCallback, this);
        
        // 订阅录制触发器 (可选，用于手动控制录制)
        recording_trigger_sub_ = nh_.subscribe("recording_trigger", 1,
                                                &DataCollector::triggerCallback, this);
        
        // 默认开始录制
        is_recording_ = true;
        last_timestamp_ = ros::Time::now().toSec();
        
        ROS_INFO("[DataCollector] Initialized. Saving to: %s", save_path_.c_str());
        ROS_INFO("[DataCollector] Parameters: min_vel=%.2f m/s, max_jump=%.2f m, rate=%.1f Hz",
                 min_velocity_threshold_, max_position_jump_, sample_rate_);
    }
    
    ~DataCollector() {
        if (data_file_.is_open()) {
            data_file_.close();
        }
        ROS_INFO("[DataCollector] Closed. Total samples: %d, Valid samples: %d (%.1f%%)",
                 total_samples_, valid_samples_, 
                 total_samples_ > 0 ? 100.0 * valid_samples_ / total_samples_ : 0.0);
    }
    
    void triggerCallback(const std_msgs::Bool::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        if (msg->data && !is_recording_) {
            // 开始新轨迹
            trajectory_id_++;
            is_recording_ = true;
            has_last_position_ = false;
            ROS_INFO("[DataCollector] Started recording trajectory %d", trajectory_id_);
        } else if (!msg->data && is_recording_) {
            // 停止录制
            is_recording_ = false;
            ROS_INFO("[DataCollector] Stopped recording trajectory %d", trajectory_id_);
        }
    }
    
    void targetCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        if (!is_recording_) return;
        
        std::lock_guard<std::mutex> lock(data_mutex_);
        total_samples_++;
        
        double timestamp = msg->header.stamp.toSec();
        double x = msg->pose.pose.position.x;
        double y = msg->pose.pose.position.y;
        double z = msg->pose.pose.position.z;
        double vx = msg->twist.twist.linear.x;
        double vy = msg->twist.twist.linear.y;
        double vz = msg->twist.twist.linear.z;
        
        // 检查时间间隔
        double dt = timestamp - last_timestamp_;
        if (dt < 0.5 / sample_rate_) {
            // 采样过快，跳过
            return;
        }
        
        // 检查速度阈值 (过滤静止目标)
        double velocity = std::sqrt(vx*vx + vy*vy);
        // 注意：这里不过滤静止目标，因为静止也是有效的运动模式
        // if (velocity < min_velocity_threshold_) {
        //     return;
        // }
        
        // 检查位置跳变 (过滤异常数据)
        if (has_last_position_) {
            double jump = std::sqrt(std::pow(x - last_x_, 2) + 
                                   std::pow(y - last_y_, 2) +
                                   std::pow(z - last_z_, 2));
            if (jump > max_position_jump_) {
                ROS_WARN_THROTTLE(1.0, "[DataCollector] Position jump detected: %.2f m, starting new trajectory", jump);
                trajectory_id_++;
                has_last_position_ = false;
            }
        }
        
        // 检查NaN
        if (std::isnan(x) || std::isnan(y) || std::isnan(z) ||
            std::isnan(vx) || std::isnan(vy) || std::isnan(vz)) {
            ROS_WARN_THROTTLE(1.0, "[DataCollector] NaN detected, skipping sample");
            return;
        }
        
        // 写入数据
        data_file_ << std::fixed << std::setprecision(6)
                   << timestamp << ","
                   << x << "," << y << "," << z << ","
                   << vx << "," << vy << "," << vz << ","
                   << trajectory_id_ << std::endl;
        
        // 更新状态
        last_x_ = x;
        last_y_ = y;
        last_z_ = z;
        last_timestamp_ = timestamp;
        has_last_position_ = true;
        valid_samples_++;
        
        // 定期刷新文件
        if (valid_samples_ % 100 == 0) {
            data_file_.flush();
            ROS_INFO_THROTTLE(10.0, "[DataCollector] Recorded %d valid samples, trajectory %d",
                             valid_samples_, trajectory_id_);
        }
    }
    
    void spin() {
        ros::spin();
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "data_collector_node");
    
    DataCollector collector;
    collector.spin();
    
    return 0;
}
