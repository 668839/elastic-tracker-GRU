#include <ros/ros.h>
#include <gazebo_msgs/ModelStates.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <mavros_msgs/State.h>
#include <cmath>

std::string target_name;
std::string drone_name;
ros::Publisher local_pos_pub;
mavros_msgs::State current_state;

void state_cb(const mavros_msgs::State::ConstPtr& msg){
    current_state = *msg;
}

void gazeboCallback(const gazebo_msgs::ModelStates::ConstPtr& msg)
{
    int target_idx = -1;
    
    // 查找目标车辆
    for(size_t i = 0; i < msg->name.size(); ++i){
        if(msg->name[i] == target_name) target_idx = i;
    }

    // 如果找不到车，就不发控制指令，这在OFFBOARD模式下会让无人机悬停或降落（取决于Failsafe）
    if(target_idx == -1) return; 

    // 获取小车当前位置
    geometry_msgs::Pose car_pose = msg->pose[target_idx];

    // --- 坐标计算：车后方1米，上方1.5米 ---
    
    // 1. 提取车的朝向 (Yaw)
    tf2::Quaternion q(
        car_pose.orientation.x,
        car_pose.orientation.y,
        car_pose.orientation.z,
        car_pose.orientation.w
    );
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    /*
    // 2. 几何计算
    double follow_dist = 1.0;  // 距离车中心1米
    double height_cmd = 2.0;   // 期望高度1.5米 (相对于地图原点)

    // 如果车是有高度的（比如在山上），我们应该基于车的高度加偏移
    // 但为了防止数据噪声导致无人机不停上下跳，这里我们取车高度 + 1.5
    double target_z = std::max(1.0, car_pose.position.z + height_cmd); 

    // 计算 X 和 Y (车屁股后面)
    // cos(yaw) 是车头方向，所以减去它是车尾方向
    double target_x = car_pose.position.x - follow_dist * cos(yaw);
    double target_y = car_pose.position.y - follow_dist * sin(yaw);

    // 3. 构建发送给飞控的指令
    geometry_msgs::PoseStamped pose;
    pose.header.stamp = ros::Time::now();
    pose.header.frame_id = "map"; 

    pose.pose.position.x = target_x;
    pose.pose.position.y = target_y;
    pose.pose.position.z = target_z;

    // 让无人机机头也朝向车头方向
    pose.pose.orientation = car_pose.orientation;

    */
    
    // =========== 【修改开始】 ===========
    
    // 修正角度：针对 actor1 人物模型，给 Yaw 增加 90 度 (1.57弧度)
    // 如果修改后无人机飞到了人物的左边，请把这里的 + 号改为 - 号
    yaw -= 1.57; 

    // 2. 几何计算 (使用修正后的 yaw)
    double follow_dist = 2.0; 
    double height_cmd = 1.0; // 之前改过的 2.0 米高度

    // 计算目标位置
    double target_x = car_pose.position.x - follow_dist * cos(yaw);
    double target_y = car_pose.position.y - follow_dist * sin(yaw);
    double target_z = std::max(1.0, car_pose.position.z + height_cmd);

    // 3. 构建指令
    geometry_msgs::PoseStamped pose;
    pose.header.stamp = ros::Time::now();
    pose.header.frame_id = "map";

    pose.pose.position.x = target_x;
    pose.pose.position.y = target_y;
    pose.pose.position.z = target_z;

    // 【关键修改】让无人机的机头朝向也跟随修正后的 Yaw
    // 这样无人机才会正对着人物的后背，而不是侧着飞
    tf2::Quaternion q_new;
    q_new.setRPY(0, 0, yaw); // 只保留偏航角，忽略人物晃动的Roll/Pitch
    
    pose.pose.orientation.x = q_new.x();
    pose.pose.orientation.y = q_new.y();
    pose.pose.orientation.z = q_new.z();
    pose.pose.orientation.w = q_new.w();

    // =========== 【修改结束】 ===========

    // 4. 发布 Setpoint
    // 即使不在 OFFBOARD 模式，这个话题也需要持续发布，
    // 这样当你切换模式的一瞬间，飞控才不会因为没有目标点而拒绝切换
    local_pos_pub.publish(pose);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "simple_car_tracker");
    ros::NodeHandle nh("~");

    nh.param<std::string>("target_name", target_name, "hatchback_red");
    nh.param<std::string>("drone_name", drone_name, "iris");

    // 订阅状态，确认连接
    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>("mavros/state", 10, state_cb);
    
    // 订阅 Gazebo 真值
    ros::Subscriber sub = nh.subscribe("/gazebo/model_states", 1, gazeboCallback);
    
    // 发布位置设定点
    local_pos_pub = nh.advertise<geometry_msgs::PoseStamped>("/mavros/setpoint_position/local", 10);

    ros::spin();
    return 0;
}