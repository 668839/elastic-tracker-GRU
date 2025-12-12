# TrajectoryPrediction.msg
# 轨迹预测结果消息

std_msgs/Header header

# 预测的未来位置序列 (世界坐标系)
geometry_msgs/Point[] predicted_positions

# 预测时间步数
uint32 prediction_horizon

# 预测起始时间戳
time prediction_start_time

# 采样间隔 (秒)
float64 dt
