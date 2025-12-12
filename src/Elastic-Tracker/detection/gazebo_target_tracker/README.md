# Gazebo 目标跟踪器使用指南

## 概述

这是一个简单的目标跟踪功能包，用于让无人机直接跟随 Gazebo 仿真中的 `hatchback_red` 目标。

**关键特性：**
- 直接从 `/gazebo/model_states` 获取目标位置
- 自动计算 Gazebo 坐标系与 VINS 坐标系之间的偏移
- 无人机跟在目标后方指定距离处
- 无需复杂的路径规划，简单直接

## 文件结构

```
gazebo_target_tracker/
├── CMakeLists.txt
├── package.xml
├── launch/
│   ├── simple_tracker.launch           # 基础启动文件
│   └── run_simple_tracker_gazebo.launch # 完整仿真启动
└── src/
    ├── gazebo_target_tracker_node.cpp  # 目标位置获取与坐标转换
    └── simple_follower_node.cpp         # 简单跟随控制器
```

## 安装步骤

### 1. 复制功能包到工作空间

```bash
# 进入你的工作空间 src 目录
cd ~/Target-Tracking-Drone-250/src/Elastic-Tracker/detection

# 复制整个功能包
cp -r gazebo_target_tracker ./

# 或者如果你从别处获取了这个包
# mv /path/to/gazebo_target_tracker ./
```

### 2. 编译

```bash
# 返回工作空间根目录
cd ~/Target-Tracking-Drone-250

# 编译
catkin_make -DCMAKE_BUILD_TYPE=Release

# 刷新环境
source devel/setup.bash
```

## 使用方法

### 方法一：分步启动

```bash
# 终端1：启动 Gazebo 仿真
roslaunch px4 mavros_posix_sitl.launch

# 终端2：启动 VINS 定位
cd ~/Target-Tracking-Drone-250
source devel/setup.bash
roslaunch vins Drone_gazebo.launch

# 终端3：启动简单跟踪器
roslaunch gazebo_target_tracker simple_tracker.launch
```

### 方法二：使用QGC飞行

1. 打开 QGroundControl
2. 等待 VINS 初始化完成（看到 "Initialization finish!"）
3. 等待坐标系偏移初始化完成（看到 "Coordinate offset initialized"）
4. 在 QGC 中切换到 Position 模式起飞
5. 悬停稳定后切换到 Offboard 模式
6. 无人机将自动开始跟随目标

## 参数说明

### 在 launch 文件中可配置的参数：

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `target_name` | `hatchback_red` | Gazebo 中目标模型的名称 |
| `drone_name` | `iris` | Gazebo 中无人机模型的名称 |
| `tracking_distance` | `1.5` | 跟随距离（米） |
| `tracking_height` | `1.2` | 相对目标的飞行高度（米） |
| `max_velocity` | `1.5` | 最大飞行速度（米/秒） |

### 修改参数示例：

```bash
# 跟踪距离改为2米，高度改为1.5米
roslaunch gazebo_target_tracker simple_tracker.launch \
    tracking_distance:=2.0 \
    tracking_height:=1.5
```

## 坐标系转换说明

### 问题背景
- VINS 定位以初始化位置为原点
- Gazebo 使用固定的世界坐标系
- 两者之间存在位置偏移

### 解决方案
节点启动时会同时读取：
1. 无人机在 VINS 坐标系下的位置
2. 无人机在 Gazebo 坐标系下的位置

计算偏移量：
```
coord_offset = drone_pos_gazebo - drone_pos_vins
```

转换目标位置：
```
target_pos_vins = target_pos_gazebo - coord_offset
```

### 注意事项
- 坐标系偏移需要在 VINS 初始化完成后才能计算
- 节点会采集多个样本取平均以提高精度
- 如果看到无人机飞向错误方向，检查 VINS 是否正确初始化

## 话题说明

### 订阅的话题
| 话题名 | 消息类型 | 说明 |
|--------|----------|------|
| `/gazebo/model_states` | `gazebo_msgs/ModelStates` | Gazebo 模型状态 |
| `/vins_fusion/imu_propagate` | `nav_msgs/Odometry` | VINS 里程计 |
| `/mavros/state` | `mavros_msgs/State` | PX4 状态 |

### 发布的话题
| 话题名 | 消息类型 | 说明 |
|--------|----------|------|
| `/mavros/setpoint_raw/local` | `mavros_msgs/PositionTarget` | 位置控制指令 |
| `gazebo_target_tracker_node/target_odom` | `nav_msgs/Odometry` | 目标位置（VINS坐标系） |

## 故障排查

### 1. 无人机不动
- 检查是否在 Offboard 模式
- 检查 VINS 是否初始化
- 检查目标是否被正确识别

### 2. 无人机飞向错误方向
- 检查 VINS 坐标系是否正确
- 检查坐标偏移是否正确计算
- 尝试重新初始化（重启节点）

### 3. 找不到目标
- 确认 Gazebo 中目标模型名称是否为 `hatchback_red`
- 使用 `rostopic echo /gazebo/model_states` 查看所有模型名称

### 4. 跟踪不稳定
- 降低 `max_velocity` 参数
- 降低 `position_gain` 参数
- 检查 VINS 定位是否稳定

## 与原项目的对比

| 特性 | 原 Elastic-Tracker | 本简单跟踪器 |
|------|-------------------|-------------|
| 目标获取 | OpenCV 颜色识别 | 直接从 Gazebo 获取 |
| 路径规划 | 多项式轨迹优化 | 直接位置控制 |
| 避障 | 支持 | 不支持 |
| 复杂度 | 高 | 低 |
| 适用场景 | 实机/复杂环境 | 仿真/简单验证 |

## 扩展开发

如需添加更多功能，可以参考以下方向：

1. **添加简单避障**：在 `simple_follower_node` 中添加距离检测
2. **添加速度平滑**：使用低通滤波器平滑速度指令
3. **添加轨迹预测**：根据目标历史位置预测未来位置
4. **支持多目标**：修改 `gazebo_target_tracker_node` 支持跟踪多个目标
