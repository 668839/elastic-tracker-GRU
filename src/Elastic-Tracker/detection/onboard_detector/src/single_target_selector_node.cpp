// src/single_target_selector_node.cpp
#include <onboard_detector/single_target_selector.h>
#include <object_detection_msgs/BoundingBox.h>
#include <object_detection_msgs/BoundingBoxes.h>

namespace onboard_detector {

// ==================== 构造函数 ====================
SingleTargetSelector::SingleTargetSelector(ros::NodeHandle& nh) 
    : nh_(nh), 
      params_received_(false),
      odom_received_(false),
      is_tracking_(false),
      tracked_id_(-1),
      lost_count_(0),
      weight_position_(0.5),
      weight_velocity_(0.3),
      weight_size_(0.2),
      min_depth_(0.2),
      max_depth_(10.0),
      max_lost_count_(5)
{
    // 加载参数
    loadParameters();
    loadCameraParams();
    
    // 初始化订阅者
    marker_sub_ = nh_.subscribe("/onboard_detector/dynamic_bboxes", 10, 
                                &SingleTargetSelector::markerCallback, this);
    odom_sub_ = nh_.subscribe("/vins_fusion/imu_propagate", 10,
                              &SingleTargetSelector::odomCallback, this);
    extrinsic_sub_ = nh_.subscribe("/vins_fusion/extrinsic", 10,
                                   &SingleTargetSelector::extrinsicCallback, this);
    
    // 初始化发布者
    bbox_pub_ = nh_.advertise<object_detection_msgs::BoundingBoxes>("/target/bbox", 10);
    marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/selected_target_marker", 10);
    
    // 初始化body到相机的变换（单位矩阵，等待外参更新）
    body_to_cam_ = Eigen::Matrix4d::Identity();
    
    ROS_INFO("[SingleTargetSelector] Initialized successfully");
    ROS_INFO("[SingleTargetSelector] Strategy: %d, Require Human: %s", 
             strategy_, require_human_ ? "true" : "false");
}

// ==================== 参数加载 ====================
void SingleTargetSelector::loadParameters() {
    // 选择策略
    std::string strategy_str;
    nh_.param<std::string>("selection_strategy", strategy_str, "human_nearest");
    
    if (strategy_str == "human_first") {
        strategy_ = HUMAN_FIRST;
    } else if (strategy_str == "nearest") {
        strategy_ = NEAREST;
    } else if (strategy_str == "largest") {
        strategy_ = LARGEST;
    } else if (strategy_str == "dynamic_first") {
        strategy_ = DYNAMIC_FIRST;
    } else {
        strategy_ = HUMAN_NEAREST;
    }
    
    // 基础参数
    nh_.param<bool>("require_human", require_human_, false);  // ✅ 默认改为false
    nh_.param<double>("distance_threshold", distance_threshold_, 1.5);
    nh_.param<double>("velocity_threshold", velocity_threshold_, 0.5);
    nh_.param<double>("timeout", timeout_, 3.0);
    nh_.param<double>("min_confidence", min_confidence_, 0.5);
    nh_.param<bool>("use_velocity_prediction", use_velocity_prediction_, true);
    nh_.param<int>("history_size", history_size_, 10);
    
    // ✅ 相似度权重参数（可选，有默认值）
    nh_.param<double>("similarity/weight_position", weight_position_, 0.5);
    nh_.param<double>("similarity/weight_velocity", weight_velocity_, 0.3);
    nh_.param<double>("similarity/weight_size", weight_size_, 0.2);
    
    // ✅ 深度范围参数（可选，有默认值）
    nh_.param<double>("depth_range/min", min_depth_, 0.2);
    nh_.param<double>("depth_range/max", max_depth_, 10.0);
    
    // ✅ 丢失重试参数（可选，有默认值）
    nh_.param<int>("max_lost_count", max_lost_count_, 5);
    
    // ✅ 参数合理性检查（使用EPSILON）
    double weight_sum = weight_position_ + weight_velocity_ + weight_size_;
    if (std::abs(weight_sum - 1.0) > EPSILON) {
        ROS_WARN("[SingleTargetSelector] Similarity weights sum to %.3f, normalizing...", weight_sum);
        weight_position_ /= weight_sum;
        weight_velocity_ /= weight_sum;
        weight_size_ /= weight_sum;
    }
    
    if (min_depth_ >= max_depth_) {
        ROS_ERROR("[SingleTargetSelector] Invalid depth range! min_depth >= max_depth");
        min_depth_ = 0.2;
        max_depth_ = 10.0;
    }
    
    ROS_INFO("[SingleTargetSelector] Loaded parameters:");
    ROS_INFO("  - Strategy: %s", strategy_str.c_str());
    ROS_INFO("  - Distance threshold: %.2f m", distance_threshold_);
    ROS_INFO("  - Velocity threshold: %.2f m/s", velocity_threshold_);
    ROS_INFO("  - Timeout: %.2f s", timeout_);
    ROS_INFO("  - Depth range: [%.2f, %.2f] m", min_depth_, max_depth_);
    ROS_INFO("  - Similarity weights: [%.2f, %.2f, %.2f]", 
             weight_position_, weight_velocity_, weight_size_);
}

// ==================== 相机参数加载 ====================
void SingleTargetSelector::loadCameraParams() {
    // ✅ 增加默认值和错误处理
    bool params_ok = true;
    
    // ✅ 新增：从onboard_detector命名空间读取图像尺寸
    nh_.param("/onboard_detector/image_cols", img_width_, 640);
    nh_.param("/onboard_detector/image_rows", img_height_, 480);
    
    if (!nh_.getParam("fx", fx_)) {
        ROS_WARN("[SingleTargetSelector] Failed to load fx! Using default 608.0");
        fx_ = 608.0;
        params_ok = false;
    }
    if (!nh_.getParam("fy", fy_)) {
        ROS_WARN("[SingleTargetSelector] Failed to load fy! Using default 608.0");
        fy_ = 608.0;
        params_ok = false;
    }
    if (!nh_.getParam("cx", cx_)) {
        ROS_WARN("[SingleTargetSelector] Failed to load cx! Using default 320.0");
        cx_ = 320.0;
        params_ok = false;
    }
    if (!nh_.getParam("cy", cy_)) {
        ROS_WARN("[SingleTargetSelector] Failed to load cy! Using default 240.0");
        cy_ = 240.0;
        params_ok = false;
    }
    
    if (params_ok) {
        ROS_INFO("[SingleTargetSelector] Camera intrinsics: fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f, size=%dx%d",
                 fx_, fy_, cx_, cy_, img_width_, img_height_);
        params_received_ = validateCameraParams();
    } else {
        ROS_WARN("[SingleTargetSelector] Using default camera parameters");
        params_received_ = false;
    }
}

// ✅ 相机参数验证
bool SingleTargetSelector::validateCameraParams() const {
    if (fx_ <= 0 || fy_ <= 0) {
        ROS_ERROR("[SingleTargetSelector] Invalid focal length: fx=%.1f, fy=%.1f", fx_, fy_);
        return false;
    }
    if (cx_ < 0 || cx_ >= img_width_ || cy_ < 0 || cy_ >= img_height_) {
        ROS_ERROR("[SingleTargetSelector] Principal point out of bounds: cx=%.1f, cy=%.1f (image: %dx%d)", 
                  cx_, cy_, img_width_, img_height_);
        return false;
    }
    return true;
}

// ✅ 障碍物验证
bool SingleTargetSelector::validateObstacle(const Obstacle3D& obstacle) const {
    // 检查置信度
    if (obstacle.confidence < min_confidence_) {
        return false;
    }
    
    // 检查尺寸（避免异常值）
    if (obstacle.size.x() <= 0 || obstacle.size.y() <= 0 || obstacle.size.z() <= 0) {
        return false;
    }
    if (obstacle.size.x() > 10.0 || obstacle.size.y() > 10.0 || obstacle.size.z() > 10.0) {
        ROS_WARN_THROTTLE(1.0, "[SingleTargetSelector] Abnormal obstacle size: [%.2f, %.2f, %.2f]",
                          obstacle.size.x(), obstacle.size.y(), obstacle.size.z());
        return false;
    }
    
    // 检查位置（避免NaN）
    if (std::isnan(obstacle.position.x()) || std::isnan(obstacle.position.y()) || 
        std::isnan(obstacle.position.z())) {
        return false;
    }
    
    // ✅ 检查深度范围
    double depth = (obstacle.position - drone_position_).norm();
    if (depth < min_depth_ || depth > max_depth_) {
        return false;
    }
    
    // ✅ 如果要求人体但不是人体
    if (require_human_ && !obstacle.is_human) {
        return false;
    }
    
    return true;
}

// ==================== 回调函数 ====================

// 里程计回调
void SingleTargetSelector::odomCallback(const nav_msgs::OdometryConstPtr& odom) {
    drone_position_ = Eigen::Vector3d(
        odom->pose.pose.position.x,
        odom->pose.pose.position.y,
        odom->pose.pose.position.z
    );
    
    drone_orientation_ = Eigen::Quaterniond(
        odom->pose.pose.orientation.w,
        odom->pose.pose.orientation.x,
        odom->pose.pose.orientation.y,
        odom->pose.pose.orientation.z
    );
    
    if (!odom_received_) {
        ROS_INFO("[SingleTargetSelector] Odometry received");
        odom_received_ = true;
    }
}

// 外参回调
void SingleTargetSelector::extrinsicCallback(const nav_msgs::OdometryConstPtr& extrinsic) {
    // 提取旋转矩阵
    Eigen::Quaterniond q(
        extrinsic->pose.pose.orientation.w,
        extrinsic->pose.pose.orientation.x,
        extrinsic->pose.pose.orientation.y,
        extrinsic->pose.pose.orientation.z
    );
    Eigen::Matrix3d rot = q.toRotationMatrix();
    
    // 提取平移向量
    Eigen::Vector3d trans(
        extrinsic->pose.pose.position.x,
        extrinsic->pose.pose.position.y,
        extrinsic->pose.pose.position.z
    );
    
    // 构建4x4变换矩阵
    body_to_cam_.block<3, 3>(0, 0) = rot;
    body_to_cam_.block<3, 1>(0, 3) = trans;
    body_to_cam_(3, 3) = 1.0;
    
    if (!params_received_) {
        ROS_INFO("[SingleTargetSelector] Extrinsic parameters received");
        params_received_ = true;
    }
}

// ==================== Marker回调 - 主要逻辑 ====================
void SingleTargetSelector::markerCallback(const visualization_msgs::MarkerArrayConstPtr& markers) {
    // 检查前提条件
    if (!odom_received_) {
        ROS_WARN_THROTTLE(2.0, "[SingleTargetSelector] Waiting for odometry...");
        return;
    }
    
    if (!params_received_) {
        ROS_WARN_THROTTLE(2.0, "[SingleTargetSelector] Waiting for camera parameters...");
        return;
    }
    
    // 检查超时
    if (isTimeout()) {
        ROS_WARN("[SingleTargetSelector] Target tracking timeout, resetting...");
        resetTracking();
    }
    
    // 解析所有障碍物
    std::vector<Obstacle3D> obstacles;
    for (const auto& marker : markers->markers) {
        if (marker.action == visualization_msgs::Marker::DELETE) {
            continue;  // 跳过删除标记
        }
        
        Obstacle3D obs = parseMarker(marker);
        
        // ✅ 验证障碍物有效性
        if (validateObstacle(obs)) {
            obstacles.push_back(obs);
        }
    }
    
    // 如果没有有效障碍物
    if (obstacles.empty()) {
        lost_count_++;
        
        if (lost_count_ > max_lost_count_) {
            ROS_WARN("[SingleTargetSelector] No valid obstacles for %d frames, resetting tracking",
                     lost_count_);
            resetTracking();
        }
        return;
    }
    
    // 选择目标
    int selected_idx = -1;
    
    if (!is_tracking_) {
        // 初始选择
        selected_idx = selectInitialTarget(obstacles);
        
        if (selected_idx >= 0) {
            is_tracking_ = true;
            tracked_id_ = obstacles[selected_idx].id;
            last_target_ = obstacles[selected_idx];
            last_valid_time_ = ros::Time::now();
            lost_count_ = 0;
            
            ROS_INFO("[SingleTargetSelector] ✓ Started tracking obstacle ID=%d at (%.2f, %.2f, %.2f)",
                     tracked_id_,
                     obstacles[selected_idx].position.x(),
                     obstacles[selected_idx].position.y(),
                     obstacles[selected_idx].position.z());
        } else {
            ROS_WARN_THROTTLE(1.0, "[SingleTargetSelector] No suitable initial target found");
        }
    } else {
        // 跟踪模式
        selected_idx = selectTrackingTarget(obstacles);
        
        if (selected_idx >= 0) {
            // 成功匹配
            last_target_ = obstacles[selected_idx];
            tracked_id_ = obstacles[selected_idx].id;
            last_valid_time_ = ros::Time::now();
            lost_count_ = 0;
            
            ROS_DEBUG("[SingleTargetSelector] ✓ Tracking ID=%d at (%.2f, %.2f, %.2f)",
                      tracked_id_,
                      obstacles[selected_idx].position.x(),
                      obstacles[selected_idx].position.y(),
                      obstacles[selected_idx].position.z());
        } else {
            // 跟踪丢失
            lost_count_++;
            
            ROS_WARN("[SingleTargetSelector] ✗ Lost target (count: %d/%d)",
                     lost_count_, max_lost_count_);
            
            if (lost_count_ > max_lost_count_) {
                ROS_WARN("[SingleTargetSelector] Target lost for too long, resetting...");
                resetTracking();
                return;
            }
            
            // ✅ 使用速度预测维持跟踪
            if (use_velocity_prediction_ && !target_history_.empty()) {
                Obstacle3D predicted = last_target_;
                double dt = (ros::Time::now() - last_valid_time_).toSec();
                predicted.position.x() += predicted.velocity.x() * dt;
                predicted.position.y() += predicted.velocity.y() * dt;
                
                ROS_INFO("[SingleTargetSelector] Using velocity prediction: v=(%.2f, %.2f) m/s",
                         predicted.velocity.x(), predicted.velocity.y());
                
                publishBoundingBox(predicted);
                publishVisualization(predicted);
            }
            return;
        }
    }
    
    // 发布选中的目标
    if (selected_idx >= 0) {
        // 更新历史
        target_history_.push_front(obstacles[selected_idx]);
        if (target_history_.size() > static_cast<size_t>(history_size_)) {
            target_history_.pop_back();
        }
        
        // 发布消息
        publishBoundingBox(obstacles[selected_idx]);
        publishVisualization(obstacles[selected_idx]);
    }
}

// ==================== 解析Marker ====================
Obstacle3D SingleTargetSelector::parseMarker(const visualization_msgs::Marker& marker) {
    Obstacle3D obs;
    obs.id = marker.id;
    obs.stamp = marker.header.stamp;
    
    // 位置（从marker的pose）
    obs.position << marker.pose.position.x,
                    marker.pose.position.y,
                    marker.pose.position.z;
    
    // 尺寸（从LINE_LIST的点或scale）
    if (!marker.points.empty()) {
        // 计算bounding box的尺寸（从LINE_LIST的点）
        double x_min = std::numeric_limits<double>::max();
        double x_max = std::numeric_limits<double>::lowest();
        double y_min = std::numeric_limits<double>::max();
        double y_max = std::numeric_limits<double>::lowest();
        double z_min = std::numeric_limits<double>::max();
        double z_max = std::numeric_limits<double>::lowest();
        
        for (const auto& pt : marker.points) {
            x_min = std::min(x_min, pt.x);
            x_max = std::max(x_max, pt.x);
            y_min = std::min(y_min, pt.y);
            y_max = std::max(y_max, pt.y);
            z_min = std::min(z_min, pt.z);
            z_max = std::max(z_max, pt.z);
        }
        
        obs.size << (x_max - x_min), (y_max - y_min), (z_max - z_min);
    } else {
        obs.size << marker.scale.x, marker.scale.y, marker.scale.z;
    }
    
    // ✅ 改进：更健壮的速度解析
    obs.velocity.setZero();
    if (!marker.text.empty()) {
        double vx, vy;
        if (parseVelocityFromText(marker.text, vx, vy)) {
            obs.velocity << vx, vy;
        } else {
            ROS_DEBUG_THROTTLE(5.0, "[SingleTargetSelector] Failed to parse velocity from: %s", 
                              marker.text.c_str());
        }
    }
    
    // 从marker颜色推断类型（onboard_detector用蓝色标记动态障碍物）
    obs.is_dynamic = (marker.color.b > 0.9 && marker.color.r < 0.1 && marker.color.g < 0.1);
    
    // 从marker命名空间或其他元数据推断是否为人体
    obs.is_human = (marker.ns.find("human") != std::string::npos || 
                    marker.ns.find("person") != std::string::npos);
    
    obs.confidence = 1.0; // onboard_detector没有置信度，默认为1
    obs.class_name = obs.is_human ? "person" : (obs.is_dynamic ? "dynamic" : "obstacle");
    
    return obs;
}

// ✅ 改进：增强的速度解析函数
bool SingleTargetSelector::parseVelocityFromText(const std::string& text, double& vx, double& vy) {
    try {
        size_t vx_pos = text.find("Vx");
        size_t vy_pos = text.find("Vy");
        
        if (vx_pos == std::string::npos || vy_pos == std::string::npos) {
            return false;
        }
        
        // 安全解析：查找数字起始位置
        size_t vx_start = text.find_first_of("-0123456789.", vx_pos + 2);
        if (vx_start == std::string::npos) return false;
        
        size_t vx_end = text.find_first_not_of("-0123456789.", vx_start);
        if (vx_end == std::string::npos) vx_end = text.length();
        
        size_t vy_start = text.find_first_of("-0123456789.", vy_pos + 2);
        if (vy_start == std::string::npos) return false;
        
        size_t vy_end = text.find_first_not_of("-0123456789.", vy_start);
        if (vy_end == std::string::npos) vy_end = text.length();
        
        std::string vx_str = text.substr(vx_start, vx_end - vx_start);
        std::string vy_str = text.substr(vy_start, vy_end - vy_start);
        
        vx = std::stod(vx_str);
        vy = std::stod(vy_str);
        return true;
        
    } catch (const std::exception& e) {
        ROS_WARN_THROTTLE(5.0, "[SingleTargetSelector] Failed to parse velocity: %s", e.what());
        return false;
    }
}

// ==================== 初始目标选择 ====================
int SingleTargetSelector::selectInitialTarget(const std::vector<Obstacle3D>& obstacles) {
    if (obstacles.empty()) return -1;
    
    switch (strategy_) {
        case HUMAN_FIRST:
            return selectByHumanFirst(obstacles);
            
        case NEAREST:
            return selectByNearest(obstacles);
            
        case LARGEST:
            return selectByLargest(obstacles);
            
        case HUMAN_NEAREST: {
            // 先尝试找人体
            int human_idx = selectByHumanFirst(obstacles);
            if (human_idx >= 0) return human_idx;
            // 找不到人体则选最近的
            return selectByNearest(obstacles);
        }
        
        case DYNAMIC_FIRST:
            return selectByDynamicFirst(obstacles);
            
        default:
            return selectByNearest(obstacles);
    }
}

// ✅ 新增：独立的选择函数（避免递归）
int SingleTargetSelector::selectByNearest(const std::vector<Obstacle3D>& obstacles) {
    int nearest_idx = -1;
    double min_dist = std::numeric_limits<double>::max();
    
    for (size_t i = 0; i < obstacles.size(); ++i) {
        double dist = (obstacles[i].position - drone_position_).norm();
        if (dist < min_dist) {
            min_dist = dist;
            nearest_idx = i;
        }
    }
    return nearest_idx;
}

int SingleTargetSelector::selectByLargest(const std::vector<Obstacle3D>& obstacles) {
    int largest_idx = -1;
    double max_volume = 0.0;
    
    for (size_t i = 0; i < obstacles.size(); ++i) {
        double volume = obstacles[i].size.prod();
        if (volume > max_volume) {
            max_volume = volume;
            largest_idx = i;
        }
    }
    return largest_idx;
}

int SingleTargetSelector::selectByHumanFirst(const std::vector<Obstacle3D>& obstacles) {
    // 优先返回第一个人体
    for (size_t i = 0; i < obstacles.size(); ++i) {
        if (obstacles[i].is_human) {
            return i;
        }
    }
    // ✅ 修复：找不到人体时返回最近的，而非递归
    ROS_WARN("[SingleTargetSelector] No human found, falling back to nearest");
    return selectByNearest(obstacles);
}

int SingleTargetSelector::selectByDynamicFirst(const std::vector<Obstacle3D>& obstacles) {
    // 优先选择动态障碍物
    for (size_t i = 0; i < obstacles.size(); ++i) {
        if (obstacles[i].is_dynamic) {
            return i;
        }
    }
    // ✅ 找不到动态障碍物时选最近的
    ROS_WARN("[SingleTargetSelector] No dynamic obstacle found, falling back to nearest");
    return selectByNearest(obstacles);
}

// ==================== 跟踪目标选择 ====================
int SingleTargetSelector::selectTrackingTarget(const std::vector<Obstacle3D>& obstacles) {
    if (obstacles.empty()) return -1;
    
    // 计算每个障碍物与上次目标的相似度
    std::vector<double> similarities;
    int max_idx = -1;
    double max_sim = -1.0;
    
    for (size_t i = 0; i < obstacles.size(); ++i) {
        double sim = computeSimilarity(obstacles[i], last_target_);
        similarities.push_back(sim);
        
        if (sim > max_sim) {
            max_sim = sim;
            max_idx = i;
        }
    }
    
    // ✅ 改进：自适应阈值（根据目标速度动态调整）
    double adaptive_threshold = 0.5;
    
    if (!target_history_.empty()) {
        double velocity_norm = last_target_.velocity.norm();
        
        if (velocity_norm > 1.0) {
            // 快速移动的目标，降低阈值
            adaptive_threshold = 0.3;
        } else if (velocity_norm < 0.2) {
            // 静止目标，提高阈值
            adaptive_threshold = 0.7;
        }
    }
    
    // 检查相似度是否超过阈值
    if (max_sim < adaptive_threshold) {
        ROS_DEBUG("[SingleTargetSelector] Max similarity %.2f < threshold %.2f", 
                  max_sim, adaptive_threshold);
        return -1;
    }
    
    // ✅ 额外的ID匹配检查（如果ID没变化，更可靠）
    for (size_t i = 0; i < obstacles.size(); ++i) {
        if (obstacles[i].id == tracked_id_ && similarities[i] > 0.3) {
            ROS_DEBUG("[SingleTargetSelector] Matched by ID: %d (similarity: %.2f)", 
                      tracked_id_, similarities[i]);
            return i;
        }
    }
    
    return max_idx;
}

// ==================== 相似度计算 ====================
double SingleTargetSelector::computeSimilarity(const Obstacle3D& curr, const Obstacle3D& ref) {
    // 1. 位置相似度（高斯核）
    double pos_diff = (curr.position - ref.position).norm();
    double pos_sim = std::exp(-pos_diff * pos_diff / (2.0 * distance_threshold_ * distance_threshold_));
    
    // 2. 速度相似度（高斯核）
    double vel_diff = (curr.velocity - ref.velocity).norm();
    double vel_sim = std::exp(-vel_diff * vel_diff / (2.0 * velocity_threshold_ * velocity_threshold_));
    
    // 3. 尺寸相似度（比例差异）
    double size_diff_x = std::abs(curr.size.x() - ref.size.x()) / (ref.size.x() + EPSILON);
    double size_diff_y = std::abs(curr.size.y() - ref.size.y()) / (ref.size.y() + EPSILON);
    double size_diff_z = std::abs(curr.size.z() - ref.size.z()) / (ref.size.z() + EPSILON);
    double size_diff = (size_diff_x + size_diff_y + size_diff_z) / 3.0;
    double size_sim = std::exp(-size_diff * size_diff / 0.5);  // σ=0.5
    
    // ✅ 加权综合（使用可配置权重）
    double total_sim = weight_position_ * pos_sim + 
                       weight_velocity_ * vel_sim + 
                       weight_size_ * size_sim;
    
    // ✅ 对同类型目标给予额外加成
    if (curr.is_human == ref.is_human && curr.is_human) {
        total_sim *= 1.2;  // 人体目标给予20%加成
        total_sim = std::min(total_sim, 1.0);  // 限制在[0,1]
    }
    
    return total_sim;
}

// ==================== 坐标转换 ====================

// 世界坐标转相机坐标
Eigen::Vector3d SingleTargetSelector::worldToCameraFrame(const Eigen::Vector3d& world_pos) {
    // ✅ 增加变换有效性检查
    if (!params_received_) {
        ROS_WARN_THROTTLE(2.0, "[SingleTargetSelector] Extrinsic not ready, using identity");
        return world_pos;
    }
    
    // World → Body
    Eigen::Vector3d body_pos = drone_orientation_.inverse() * (world_pos - drone_position_);
    
    // Body → Camera
    Eigen::Vector4d body_pos_homo;
    body_pos_homo << body_pos, 1.0;
    Eigen::Vector4d cam_pos_homo = body_to_cam_ * body_pos_homo;
    
    return cam_pos_homo.head<3>();
}

// 相机坐标投影到像素坐标
void SingleTargetSelector::projectToPixel(const Eigen::Vector3d& cam_pos, 
                                          double& px, double& py, double& depth) {
    depth = cam_pos.z();
    
    // ✅ 深度有效性检查
    if (depth < min_depth_ || depth > max_depth_) {
        ROS_WARN_THROTTLE(1.0, "[SingleTargetSelector] Depth %.2f out of range [%.2f, %.2f]",
                         depth, min_depth_, max_depth_);
        px = py = -1;
        return;
    }
    
    if (depth < EPSILON) {
        px = py = -1;
        return;
    }
    
    // 透视投影
    px = fx_ * cam_pos.x() / depth + cx_;
    py = fy_ * cam_pos.y() / depth + cy_;
    
    // ✅ 使用动态图像尺寸检查
    if (px < 0 || px >= img_width_ || py < 0 || py >= img_height_) {
        ROS_DEBUG("[SingleTargetSelector] Pixel (%.1f, %.1f) out of image bounds (%dx%d)", 
                  px, py, img_width_, img_height_);
        px = py = -1;
    }
}

// ==================== 发布函数 ====================

// 发布BoundingBox
void SingleTargetSelector::publishBoundingBox(const Obstacle3D& obstacle) {
    // 世界坐标 → 相机坐标
    Eigen::Vector3d cam_pos = worldToCameraFrame(obstacle.position);
    
    // 相机坐标 → 像素坐标
    double center_px, center_py, depth;
    projectToPixel(cam_pos, center_px, center_py, depth);
    
    // ✅ 深度有效性检查
    if (center_px < 0 || center_py < 0 || depth < min_depth_ || depth > max_depth_) {
        ROS_WARN_THROTTLE(1.0, "[SingleTargetSelector] Invalid projection, skipping bbox publish");
        return;
    }
    
    // 估算像素尺寸（粗略估计）
    double width_px = (obstacle.size.x() / depth) * fx_;
    double height_px = (obstacle.size.y() / depth) * fy_;
    
    // ✅ 像素尺寸合理性检查
    width_px = std::max(10.0, std::min(width_px, 200.0));   // 限制在[10, 200]像素
    height_px = std::max(10.0, std::min(height_px, 200.0));
    
    // 构建BoundingBox消息（兼容opencv_target格式）
    object_detection_msgs::BoundingBox bbox;
    bbox.xmin = static_cast<int>(center_px - width_px / 2.0);
    bbox.xmax = static_cast<int>(center_px + width_px / 2.0);
    bbox.ymin = static_cast<int>(center_py - height_px / 2.0);
    bbox.ymax = static_cast<int>(center_py + height_px / 2.0);
    bbox.depth = depth;
    bbox.probability = obstacle.confidence;
    bbox.id = obstacle.id;
    bbox.Class = "red";  // 兼容opencv_target
    
    // 发布
    object_detection_msgs::BoundingBoxes bboxes_msg;
    bboxes_msg.bounding_boxes.push_back(bbox);
    bboxes_msg.header.stamp = ros::Time::now();
    bboxes_msg.header.frame_id = "camera";
    
    bbox_pub_.publish(bboxes_msg);
    
    ROS_DEBUG("[SingleTargetSelector] Published bbox: center=(%.1f, %.1f), depth=%.2f, size=(%.1fx%.1f)",
              center_px, center_py, depth, width_px, height_px);
}

// 可视化发布
void SingleTargetSelector::publishVisualization(const Obstacle3D& obstacle) {
    visualization_msgs::MarkerArray marker_array;
    visualization_msgs::Marker marker;
    
    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();
    marker.ns = "selected_target";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;
    
    marker.pose.position.x = obstacle.position.x();
    marker.pose.position.y = obstacle.position.y();
    marker.pose.position.z = obstacle.position.z();
    marker.pose.orientation.w = 1.0;
    
    marker.scale.x = obstacle.size.x();
    marker.scale.y = obstacle.size.y();
    marker.scale.z = obstacle.size.z();
    
    // 绿色表示选中的目标
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    marker.color.a = 0.7;
    
    marker.lifetime = ros::Duration(0.2);
    
    marker_array.markers.push_back(marker);
    marker_pub_.publish(marker_array);
}

// ==================== 工具函数 ====================

// 重置跟踪状态
void SingleTargetSelector::resetTracking() {
    is_tracking_ = false;
    tracked_id_ = -1;
    lost_count_ = 0;
    target_history_.clear();
    ROS_INFO("[SingleTargetSelector] Tracking reset");
}

// 检查是否超时
bool SingleTargetSelector::isTimeout() {
    if (!is_tracking_) {
        return false;
    }
    
    double elapsed = (ros::Time::now() - last_valid_time_).toSec();
    bool timeout = elapsed > timeout_;
    
    if (timeout) {
        ROS_WARN("[SingleTargetSelector] Timeout: %.2fs > %.2fs", elapsed, timeout_);
    }
    
    return timeout;
}

// ✅ FOV检查（参考dynamicDetector的isInFov函数）
bool SingleTargetSelector::isInCameraFOV(const Eigen::Vector3d& world_pos) {
    // 转换到相机坐标系
    Eigen::Vector3d cam_pos = worldToCameraFrame(world_pos);
    
    // 深度检查
    if (cam_pos.z() < min_depth_ || cam_pos.z() > max_depth_) {
        return false;
    }
    
    // 水平/垂直视场角检查 (D435i: H=87°, V=58°)
    double h_angle = std::atan2(std::abs(cam_pos.x()), cam_pos.z());
    double v_angle = std::atan2(std::abs(cam_pos.y()), cam_pos.z());
    
    const double H_FOV = 87.0 * M_PI / 180.0;  // 弧度
    const double V_FOV = 58.0 * M_PI / 180.0;
    
    return (h_angle < H_FOV / 2.0) && (v_angle < V_FOV / 2.0);
}

} // namespace onboard_detector

// ==================== 主函数 ====================
int main(int argc, char** argv) {
    ros::init(argc, argv, "single_target_selector_node");
    ros::NodeHandle nh("~");
    
    onboard_detector::SingleTargetSelector selector(nh);
    
    ROS_INFO("[SingleTargetSelector] Node started, spinning...");
    ros::spin();
    
    return 0;
}