// include/onboard_detector/single_target_selector.h
#ifndef SINGLE_TARGET_SELECTOR_H
#define SINGLE_TARGET_SELECTOR_H

#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <object_detection_msgs/BoundingBoxes.h>
#include <object_detection_msgs/BoundingBox.h>
#include <nav_msgs/Odometry.h>
#include <Eigen/Dense>
#include <string>
#include <deque>
#include <algorithm>
#include <cmath>
#include <limits>

namespace onboard_detector {

/**
 * @brief 3D障碍物信息结构体（对应onboard_detector的box3D）
 */
struct Obstacle3D {
    int id;                       // 障碍物ID
    Eigen::Vector3d position;     // 世界坐标 (x, y, z)
    Eigen::Vector3d size;         // 尺寸 (width, height, depth)
    Eigen::Vector2d velocity;     // 速度 (vx, vy)
    bool is_human;                // 是否为人体
    bool is_dynamic;              // 是否为动态障碍物
    std::string class_name;       // 类别名称
    double confidence;            // 置信度
    ros::Time stamp;              // 时间戳
    
    Obstacle3D() : id(-1), is_human(false), is_dynamic(false), 
                   class_name("unknown"), confidence(0.0) {
        position.setZero();
        size.setZero();
        velocity.setZero();
    }
};

/**
 * @brief 单目标选择器类
 * 功能：从onboard_detector检测的多个动态障碍物中选择单个目标并持续跟踪
 */
class SingleTargetSelector {
public:
    /**
     * @brief 选择策略枚举
     */
    enum SelectionStrategy {
        HUMAN_FIRST,      // 优先选择人体
        NEAREST,          // 选择最近的
        LARGEST,          // 选择最大的
        HUMAN_NEAREST,    // 人体优先，人体中选最近的（默认）
        DYNAMIC_FIRST     // 优先选择动态障碍物
    };

    /**
     * @brief 构造函数
     */
    SingleTargetSelector(ros::NodeHandle& nh);

    /**
     * @brief 析构函数
     */
    ~SingleTargetSelector() = default;

private:
    // ==================== ROS相关 ====================
    ros::NodeHandle nh_;
    ros::Subscriber marker_sub_;       // 订阅onboard_detector的dynamic_bboxes
    ros::Subscriber odom_sub_;         // 订阅无人机里程计
    ros::Subscriber extrinsic_sub_;    // 订阅外参
    ros::Publisher bbox_pub_;          // 发布选中目标的bbox（兼容opencv_target）
    ros::Publisher marker_pub_;        // 发布可视化marker
    
    // ==================== 相机参数 ====================
    double fx_, fy_, cx_, cy_;         // 相机内参
    int img_width_, img_height_;       // 图像尺寸
    Eigen::Matrix4d body_to_cam_;      // body到相机的变换
    bool params_received_;             // 参数是否接收
    
    // ==================== 无人机位姿 ====================
    Eigen::Vector3d drone_position_;
    Eigen::Quaterniond drone_orientation_;
    bool odom_received_;
    
    // ==================== 跟踪状态 ====================
    bool is_tracking_;                 // 是否正在跟踪
    int tracked_id_;                   // 当前跟踪的目标ID
    Obstacle3D last_target_;           // 上一帧的目标
    ros::Time last_valid_time_;        // 最后一次有效跟踪时间
    std::deque<Obstacle3D> target_history_;  // 目标历史（用于速度估计）
    
    // ==================== 参数配置 ====================
    SelectionStrategy strategy_;       // 选择策略
    double distance_threshold_;        // 位置相似度阈值 (m)
    double velocity_threshold_;        // 速度相似度阈值 (m/s)
    double timeout_;                   // 跟踪超时时间 (s)
    double min_confidence_;            // 最小置信度阈值
    bool require_human_;               // 是否必须选择人体
    bool use_velocity_prediction_;     // 是否使用速度预测
    int history_size_;                 // 历史队列大小
    
    // 相似度计算权重（可配置）
    double weight_position_;           // 位置权重
    double weight_velocity_;           // 速度权重
    double weight_size_;               // 尺寸权重
    
    // 深度范围限制（可配置）
    double min_depth_;                 // 最小深度 (m)
    double max_depth_;                 // 最大深度 (m)
    
    // 目标丢失重试机制
    int lost_count_;                   // 连续丢失计数
    int max_lost_count_;               // 最大允许丢失次数
    
    // 常量
    static constexpr double EPSILON = 1e-6;  // 浮点数比较容差
    
    // ==================== 回调函数 ====================
    void markerCallback(const visualization_msgs::MarkerArrayConstPtr& markers);
    void odomCallback(const nav_msgs::OdometryConstPtr& odom);
    void extrinsicCallback(const nav_msgs::OdometryConstPtr& extrinsic);
    
    // ==================== 核心功能函数 ====================
    /**
     * @brief 从marker解析障碍物信息
     */
    Obstacle3D parseMarker(const visualization_msgs::Marker& marker);
    
    /**
     * @brief 初始目标选择（根据策略）
     */
    int selectInitialTarget(const std::vector<Obstacle3D>& obstacles);
    
    /**
     * @brief 跟踪模式下的目标选择（基于相似度）
     */
    int selectTrackingTarget(const std::vector<Obstacle3D>& obstacles);
    
    /**
     * @brief 计算两个障碍物的相似度
     */
    double computeSimilarity(const Obstacle3D& curr, const Obstacle3D& ref);
    
    // ==================== 各策略的独立选择函数 ====================
    int selectByNearest(const std::vector<Obstacle3D>& obstacles);
    int selectByLargest(const std::vector<Obstacle3D>& obstacles);
    int selectByHumanFirst(const std::vector<Obstacle3D>& obstacles);
    int selectByDynamicFirst(const std::vector<Obstacle3D>& obstacles);
    
    // ==================== 坐标转换函数 ====================
    /**
     * @brief 世界坐标转换到相机坐标系
     */
    Eigen::Vector3d worldToCameraFrame(const Eigen::Vector3d& world_pos);
    
    /**
     * @brief 相机坐标投影到像素坐标
     */
    void projectToPixel(const Eigen::Vector3d& cam_pos, double& px, double& py, double& depth);
    
    // ==================== 发布函数 ====================
    /**
     * @brief 发布BoundingBox（兼容opencv_target格式）
     */
    void publishBoundingBox(const Obstacle3D& obstacle);
    
    /**
     * @brief 发布可视化marker
     */
    void publishVisualization(const Obstacle3D& obstacle);
    
    // ==================== 工具函数 ====================
    /**
     * @brief 加载参数
     */
    void loadParameters();
    
    /**
     * @brief 加载相机参数
     */
    void loadCameraParams();
    
    /**
     * @brief 重置跟踪状态
     */
    void resetTracking();
    
    /**
     * @brief 检查是否超时
     */
    bool isTimeout();
    
    /**
     * @brief 检查点是否在相机FOV内
     */
    bool isInCameraFOV(const Eigen::Vector3d& world_pos);
    
    /**
     * @brief 验证相机参数有效性
     */
    bool validateCameraParams() const;
    
    /**
     * @brief 验证障碍物有效性
     */
    bool validateObstacle(const Obstacle3D& obstacle) const;
    
    /**
     * @brief 从marker的text字段解析速度
     */
    bool parseVelocityFromText(const std::string& text, double& vx, double& vy);
};

} // namespace onboard_detector

#endif // SINGLE_TARGET_SELECTOR_H