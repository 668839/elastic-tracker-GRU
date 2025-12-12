#include <opencv_target/opencv_target.h>
#include <opencv_target/depth_processing.h>

// 1.图像发布函数：将处理后的图像发布到ROS话题
void TargetTracker::pubProcessedImage(cv::Mat& bgr_image)
{
    sensor_msgs::ImagePtr msg;
    // 图像数据转换;cv_bridge构造函数​：创建cv_bridge中间对象，连接OpenCV和ROS图像格式
    msg = cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::BGR8, bgr_image).toImageMsg();
    // 消息发布
    TargetTracker::image_pub_.publish(msg);

    return;
}

/* 2.特征提取函数：extractFeature函数就像是一个​​智能的特征记录员​​：
                ​​输入​​：一张图片 + 目标位置框
                ​​处理​​：增强特征、统计分析、标准化
                ​​输出​​：目标的"特征身份证"（颜色分布 + 距离信息）*/
TargetTracker::TargetFeature TargetTracker::extractFeature(const cv::Mat& bgr, const cv::Mat& depth, const cv::Rect& rect)
{
    // 初始化特征结构
    TargetFeature feat;
    feat.rect = rect;

    // 提取颜色直方图（HSV空间H+S通道）
    cv::Rect safe_rect = rect & cv::Rect(0, 0, bgr.cols, bgr.rows);
    cv::Mat roi = bgr(safe_rect);
    cv::Mat hsv_roi;
    cv::cvtColor(roi, hsv_roi, cv::COLOR_BGR2HSV);
    
    // 直方图均衡化增强鲁棒性
    std::vector<cv::Mat> channels;
    cv::split(hsv_roi, channels);
    clahe_->apply(channels[0], channels[0]);
    clahe_->apply(channels[1], channels[1]);
    
    // 计算2D直方图（H:0-180, S:0-256）
    int histSize[] = {30, 32};    // 降低维度提升速度
    float hRanges[] = {0, 180};
    float sRanges[] = {0, 256};
    const float* ranges[] = {hRanges, sRanges};
    int channels_num[] = {0, 1};
    
    cv::calcHist(&hsv_roi, 1, channels_num, cv::Mat(), feat.color_hist, 2, histSize, ranges);
    cv::normalize(feat.color_hist, feat.color_hist);

    // 计算深度
    feat.avg_depth = computeMedianDepth(depth, rect);

    return feat;
}

/*3.特征相似度计算函数
            决策逻辑​​:
            即使衣服颜色完全一样，但距离差2米 → 可能不是同一个人
            即使距离很近，但颜色完全不同 → 也可能不是目标
*/
double TargetTracker::compareFeatures(const TargetFeature& curr, const TargetFeature& ref) {
    // 颜色相似度（直方图交集）
    double color_sim = cv::compareHist(curr.color_hist, 
                                      ref.color_hist, 
                                      cv::HISTCMP_INTERSECT);
    
    // 深度相似度（高斯加权）
    double depth_diff = fabs(curr.avg_depth - ref.avg_depth);
    double depth_sim = exp(-pow(depth_diff, 2) / (2 * pow(0.5, 2))); // σ=0.5m
    
    // 综合评分（加权求和）
    return depth_weight_ * depth_sim + color_weight_ * color_sim;
}


/* 4.RGB与深度图像同步订阅 （处理深度图像给pubProcessedImage函数）
帧1（初始化）​​：检测到3个红色物体：小红点（噪声）、中等红色物体、大红色气球;选择：大红色气球（面积最大）;记录特征：颜色分布、距离5米
帧2（跟踪）​​：检测到2个红色物体：大红色气球（稍微移动）、另一个红色物体;比较特征：物体1相似度0.9，物体2相似度0.3;选择：物体1（相似度最高）;更新位置：气球现在在5.1米外
帧3（目标丢失）​​：检测到红色物体，但相似度只有0.4（低于0.5阈值）;处理：标记为"目标丢失"，等待5秒后重置
*/
void TargetTracker::syncCallback(const sensor_msgs::ImageConstPtr& rgb_msg, const sensor_msgs::ImageConstPtr& depth_msg)
{
    // 4.1 状态检查与重置:状态管理​​: 如果目标丢失超过5秒，重置跟踪状态;防止误判​​: 避免短暂遮挡导致跟踪失败
    ros::Time current_time = ros::Time::now();
    
    // 丢失目标超过一定时间，重置
    if (lost_target_ && 
       (current_time - last_valid_time_).toSec() > 5.0) {
        is_tracking_ = false;
        lost_target_ = false;
        ROS_WARN("Reset tracking state after timeout");
    }
    
    /* -------- 订阅话题传入：数据传入 -------- */

    // 4.2 图像数据转换

    // 传入RGB数据，并转为BGR格式
    cv_bridge::CvImagePtr bgr_cv_ptr;       
    try{
        bgr_cv_ptr = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e){
        ROS_ERROR("Image conversion error: %s", e.what());
        return;
    }
    
    // 传入深度数据，默认为16UC1类型
    cv_bridge::CvImagePtr depth_cv_ptr;     
    try{
        depth_cv_ptr = cv_bridge::toCvCopy(depth_msg);
    } 
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Depth image conversion error: %s", e.what());
        return;
    }
    
    /* -------- 图像处理 -------- */
    // 4.3 红色目标检测

    // HSV颜色检测
    cv::Mat bgr_image = bgr_cv_ptr->image;
    cv::Mat hsv, mask;
    cv::cvtColor(bgr_image, hsv, cv::COLOR_BGR2HSV);
     
    /*
    // 红色阈值范围
    cv::Scalar lower_red1(0, 120, 70);
    cv::Scalar upper_red1(10, 255, 255);
    cv::Scalar lower_red2(170, 120, 70);
    cv::Scalar upper_red2(180, 255, 255);
    */

    
    // 替换原来的红色阈值范围
    cv::Scalar lower_red1(0, 150, 45);    // 专门针对暗红色
    cv::Scalar upper_red1(10, 220, 85);   
    cv::Scalar lower_red2(170, 150, 45);  
    cv::Scalar upper_red2(180, 220, 85);
    

	// 创建红色掩膜
    cv::Mat mask1, mask2;
    cv::inRange(hsv, lower_red1, upper_red1, mask1);
    cv::inRange(hsv, lower_red2, upper_red2, mask2);
    cv::bitwise_or(mask1, mask2, mask);

    // 形态学操作
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // 4.4 轮廓检测与过滤。RETR_EXTERNAL​​: 只检测最外层轮廓;​​CHAIN_APPROX_SIMPLE​​: 压缩轮廓点，减少内存占用

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 如果追踪到目标，处理检测结果并发送
    if (contours.size() > 0)
	{
        std::vector<cv::Rect> valid_rects;
        cv::Mat depth_image = depth_cv_ptr->image;

        // 筛选有效轮廓
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area < 100) continue;  // 面积阈值过滤

            cv::Rect rect = cv::boundingRect(contour);
            valid_rects.push_back(rect);
        }

        // 4.5 初始化变量和特征提取。features向量​​: 为每个有效矩形框创建特征描述;extractFeature调用​​: 对每个候选目标提取颜色直方图和深度信息

        // 如果具有有效的目标
        if (!valid_rects.empty()) {
            int selected_idx = 0;

            std::vector<TargetFeature> features;
            std::vector<double> scores;
            
            // 提取所有候选特征
            for (const auto& rect : valid_rects) {
                features.push_back(extractFeature(bgr_image, depth_image, rect));
            }
            
            // 4.6 目标选择逻辑：选择策略​​: 面积最大 + 深度有效;跟踪策略​​: 综合颜色 + 深度相似度评分

            // 首次选择：最大面积+深度合法值
            if (!is_tracking_) {
                auto it = std::max_element(features.begin(), features.end(),
                    [](const TargetFeature& a, const TargetFeature& b) {
                        return a.rect.area() < b.rect.area() || 
                               std::isnan(a.avg_depth);
                    });
                last_feature_ = *it;
                is_tracking_ = true;
            } 
            // 跟踪状态：综合评分选择
            else {
                // 计算每个候选与上一次特征的相似度
                for (const auto& feat : features) {
                    scores.push_back(compareFeatures(feat, last_feature_));
                }
            
                // 选择最高分且超过阈值的目标
                auto max_it = std::max_element(scores.begin(), scores.end());
                if (*max_it > 0.5) { // 相似度阈值
                    selected_idx = std::distance(scores.begin(), max_it);
                    last_feature_ = features[selected_idx];
                } else {
                // 无相似目标，认为找不到目标
                    ROS_WARN("No similar target, keep last feature");
                    lost_target_ = true;
                    pubProcessedImage(bgr_image);
                    return;
                }
            }

            // 更新最后已知位置
            cv::Rect rect = valid_rects[selected_idx];
            double depth = features[selected_idx].avg_depth;
            depth += depth_fix_;

            // 4.7 结果发布
            // 绘制目标框
        	cv::rectangle(bgr_image, rect, cv::Scalar(0,255,0), 2);
            
            // 深度值合法则允许发布
            if (std::isnan(depth) || depth <= 0.0) {
                ROS_WARN_STREAM_THROTTLE(1.0,"Invalid depth value: " << depth);
            }
            else{
                ROS_INFO_THROTTLE(1.0,"Target at Pixel: (%.1f, %.1f), depth = %.2f", 
                                  rect.x + rect.width/2.0 , rect.y + rect.height/2.0, depth);
                // 赋值bbox
                object_detection_msgs::BoundingBox opencv_bbox_;
                opencv_bbox_.xmin = rect.x;
                opencv_bbox_.xmax = rect.x + rect.width;
                opencv_bbox_.ymin = rect.y;
                opencv_bbox_.ymax = rect.y + rect.height;
                opencv_bbox_.depth = depth;
                opencv_bbox_.probability = 0.9;
                opencv_bbox_.id = 0;
                opencv_bbox_.Class = "red";
                // 发布消息
                object_detection_msgs::BoundingBoxes opencv_bboxes_;
                opencv_bboxes_.bounding_boxes.push_back(opencv_bbox_);
                opencv_bboxes_.header.stamp = ros::Time::now();
                bbox_pub_.publish(opencv_bboxes_);
            }
		}
        // 所有目标都是无效目标
        else {
            is_tracking_ = false;  // 重置跟踪状态
            ROS_INFO_THROTTLE(1.0,"NO Valid Target.");
        }     
    }
    // 未追踪到目标
    else
    {
        is_tracking_ = false;  // 重置跟踪状态
        ROS_INFO_THROTTLE(1.0,"NO Target.");
    }
    
    // 4.8 最终处理
    pubProcessedImage(bgr_image);
    last_valid_time_ = current_time;
    return;
}

// 主函数
int main(int argc, char** argv)
{
    ros::init(argc, argv, "opencv_target");
    ros::NodeHandle nh;

    TargetTracker opencv_target(nh);
    ros::spin();

    return 0;
}
