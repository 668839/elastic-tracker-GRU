# PredictionWithUncertainty.msg
# 带不确定性估计的轨迹预测结果

std_msgs/Header header

# 预测的未来位置序列 (世界坐标系)
geometry_msgs/Point[] predicted_positions

# 每个预测位置的不确定性协方差 (展平的2x2矩阵: [σ_xx, σ_xy, σ_xy, σ_yy])
float64[] covariances

# 认知不确定性 (epistemic)
float64[] epistemic_uncertainty

# 偶然不确定性 (aleatoric)  
float64[] aleatoric_uncertainty

# 预测时间步数
uint32 prediction_horizon

# 采样间隔 (秒)
float64 dt

# 当前航向角 (用于坐标还原)
float64 heading_angle
