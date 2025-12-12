#include <ros/ros.h>
#include <gazebo_msgs/ModelStates.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <string>

// 这个节点读取Gazebo真值，欺骗PX4/MAVROS说这是视觉定位数据
// 这是开启OFFBOARD模式的前提：必须有Local Position

std::string drone_name;
ros::Publisher vision_pub;
ros::Publisher odom_pub;

void gazeboCallback(const gazebo_msgs::ModelStates::ConstPtr& msg)
{
    int index = -1;
    for(size_t i = 0; i < msg->name.size(); ++i){
        if(msg->name[i] == drone_name){
            index = i;
            break;
        }
    }

    if(index == -1) return;

    geometry_msgs::Pose raw_pose = msg->pose[index];
    geometry_msgs::Twist raw_twist = msg->twist[index];
    ros::Time current_time = ros::Time::now();

    // 1. 发送 Vision Pose 给 MAVROS
    // PX4 EKF2 收到这个包后，会认为无人机有了位置锁定 (Local Position Valid)
    // 这是在 QGC 点击起飞或切换 OFFBOARD 的必要条件
    geometry_msgs::PoseStamped vision_msg;
    vision_msg.header.stamp = current_time;
    vision_msg.header.frame_id = "map"; // 对应PX4的本地坐标系
    vision_msg.pose = raw_pose;
    vision_pub.publish(vision_msg);

    // 2. 发送 Odom (用于Rviz显示，或者作为其他节点的里程计输入)
    nav_msgs::Odometry odom_msg;
    odom_msg.header.stamp = current_time;
    odom_msg.header.frame_id = "world";
    odom_msg.child_frame_id = "base_link";
    odom_msg.pose.pose = raw_pose;
    odom_msg.twist.twist = raw_twist;
    odom_pub.publish(odom_msg);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "gazebo_fake_localization");
    ros::NodeHandle nh("~");

    // 默认无人机名字为 iris，如果你的仿真环境里叫 drone 或 uav，请在 launch 文件修改
    nh.param<std::string>("drone_name", drone_name, "iris");

    ros::Subscriber sub = nh.subscribe("/gazebo/model_states", 1, gazeboCallback);

    // 关键话题：发送给 MAVROS 的视觉定位接口
    vision_pub = nh.advertise<geometry_msgs::PoseStamped>("/mavros/vision_pose/pose", 10);
    odom_pub = nh.advertise<nav_msgs::Odometry>("/vins_estimator/odometry", 10);

    ros::spin();
    return 0;
}