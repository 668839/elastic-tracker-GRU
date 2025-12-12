# Trajectory Prediction for UAV Target Tracking

基于GRU + MC Dropout + 自适应Kalman融合的地面目标轨迹预测系统。

## 概述

本项目实现了论文中描述的轨迹预测方法，包括：
- **Phase 1**: 数据收集 + Teacher模型
- **Phase 2**: 知识蒸馏训练Student模型
- **Phase 3**: MC Dropout不确定性估计
- **Phase 4**: 自适应Kalman融合 + ROS部署

## 目录结构

```
trajectory_prediction/
├── CMakeLists.txt
├── package.xml
├── README.md
│
├── data/                           # 数据目录
│   ├── gazebo_raw/                 # Gazebo仿真原始数据
│   │   └── raw_trajectories.csv
│   └── processed_data.npz          # 预处理后的训练数据
│
├── checkpoints/                    # 模型检查点
│   └── teacher/
│       ├── best_model.pth
│       ├── teacher_errors.npz
│       └── history.json
│
├── results/                        # 评估结果
│
├── launch/
│   ├── data_collection.launch      # 真实环境数据收集 (订阅EKF)
│   └── collect_gazebo_data.launch  # Gazebo仿真数据收集
│
├── msg/
│   ├── TrajectoryPrediction.msg
│   └── PredictionWithUncertainty.msg
│
├── scripts/
│   ├── collect_gazebo_actor.py     # [仿真] Gazebo数据收集
│   ├── preprocess_gazebo_data.py   # [仿真] Gazebo数据预处理
│   ├── preprocess_data.py          # [真实] EKF数据预处理
│   ├── models.py                   # 模型定义
│   ├── train_teacher.py            # Teacher训练
│   ├── evaluate_model.py           # 模型评估
│   └── generate_synthetic_data.py  # 生成模拟数据(测试用)
│
└── src/
    └── data_collector_node.cpp     # [真实] C++数据收集节点 (订阅EKF)
```

## 数据收集方式

本项目支持两种数据收集方式：

| 场景 | 数据收集脚本 | 预处理脚本 | 订阅话题 |
|------|-------------|-----------|---------|
| **Gazebo仿真** | `collect_gazebo_actor.py` | `preprocess_gazebo_data.py` | `/gazebo/model_states` |
| **真实环境** | `data_collector_node.cpp` | `preprocess_data.py` | `/target_ekf_node/target_odom` |

---

## 方式一：Gazebo仿真数据收集 (推荐先用此方式测试)

### Step 1: 启动Gazebo仿真

```bash
# 启动你的Gazebo世界 (确保有actor1)
roslaunch your_package your_gazebo_world.launch
```

### Step 2: 收集数据

**方法A: 使用launch文件**
```bash
roslaunch trajectory_prediction collect_gazebo_data.launch \
    target_name:=actor1 \
    duration:=30
```

**方法B: 直接运行Python脚本**
```bash
cd ~/Elastic-Tracker-Drone/src/trajectory_prediction
python3 scripts/collect_gazebo_actor.py \
    --output_dir data/gazebo_raw \
    --target_name actor1 \
    --duration 30
```

在Gazebo中控制actor1移动，数据会自动保存。当actor静止超过3秒，会自动分割为新轨迹。

### Step 3: 预处理数据

```bash
python3 scripts/preprocess_gazebo_data.py \
    --input data/gazebo_raw/raw_trajectories.csv \
    --output data/processed_data.npz \
    --obs_len 30 \
    --pred_len 60
```

### Step 4: 训练Teacher模型

```bash
python3 scripts/train_teacher.py \
    --data_path data/processed_data.npz \
    --save_dir checkpoints/teacher \
    --epochs 100
```

### Step 5: 评估模型

```bash
python3 scripts/evaluate_model.py \
    --data_path data/processed_data.npz \
    --checkpoint checkpoints/teacher/best_model.pth \
    --model_type teacher
```

---

## 方式二：真实环境数据收集

### Step 1: 启动跟踪系统

```bash
# 启动Elastic-Tracker
roslaunch planning run_in_gazebo.launch  # 或真实环境的launch文件
```

确保 `/target_ekf_node/target_odom` 话题正在发布。

### Step 2: 收集数据

```bash
roslaunch trajectory_prediction data_collection.launch
```

数据会保存到 `data/raw_trajectories.csv`。

### Step 3: 预处理数据

```bash
python3 scripts/preprocess_data.py \
    --input data/raw_trajectories.csv \
    --output data/processed_data.npz
```

### Step 4-5: 训练和评估

与Gazebo方式相同。

---

## 快速测试 (使用模拟数据)

如果还没有收集真实数据，可以先用模拟数据测试pipeline：

```bash
cd ~/Elastic-Tracker-Drone/src/trajectory_prediction

# 生成模拟数据
python3 scripts/generate_synthetic_data.py \
    --num_trajectories 200 \
    --output data/raw_trajectories.csv

# 预处理
python3 scripts/preprocess_data.py \
    --input data/raw_trajectories.csv \
    --output data/processed_data.npz

# 训练
python3 scripts/train_teacher.py \
    --data_path data/processed_data.npz \
    --save_dir checkpoints/teacher \
    --epochs 50
```

---

## 模型架构

### Teacher模型

| 组件 | 参数 |
|------|------|
| 编码器 | 3层 Bi-GRU, 256隐藏单元 |
| 解码器 | 2层 自回归GRU, 256隐藏单元 |
| Dropout | 0.2 |
| 参数量 | ~2.1M |

### Student模型 (Phase 2)

| 组件 | 参数 |
|------|------|
| 编码器 | 2层 Bi-GRU, 64隐藏单元 |
| 解码器 | 1层 自回归GRU, 64隐藏单元 |
| Dropout | 0.2 |
| 参数量 | ~0.4M |

---

## 数据格式

### 原始数据 (CSV)

```csv
timestamp,x,y,z,vx,vy,vz,trajectory_id
0.000000,1.234,5.678,0.0,0.5,0.3,0.0,0
0.033333,1.251,5.688,0.0,0.5,0.3,0.0,0
...
```

### 预处理后数据 (NPZ)

```python
data = np.load('data/processed_data.npz')

# 训练集
train_obs = data['train_obs']      # (N, 30, 6) 观测序列 [x,y,vx,vy,ax,ay]
train_pred = data['train_pred']    # (N, 60, 2) 预测目标 [x,y]
train_headings = data['train_headings']  # (N,) 航向角
train_anchors = data['train_anchors']    # (N, 2) 锚点位置

# 验证集、测试集同理
```

---

## 评估指标

- **ADE (Average Displacement Error)**: 所有预测时间步的平均位移误差
- **FDE (Final Displacement Error)**: 最终时间步的位移误差
- **ADE@1s, ADE@1.5s, ADE@2s**: 不同预测时长的ADE
- **Miss Rate@Xm**: FDE超过X米的样本比例

---

## 依赖安装

### Python依赖

```bash
pip3 install torch numpy pandas scipy tqdm matplotlib
```

### ROS依赖

```bash
# 在catkin工作空间编译
cd ~/Elastic-Tracker-Drone
catkin_make
source devel/setup.bash
```

---

## 文件说明

### 数据收集

| 文件 | 用途 | 订阅话题 |
|------|------|---------|
| `collect_gazebo_actor.py` | Gazebo仿真数据收集 | `/gazebo/model_states` |
| `data_collector_node.cpp` | 真实环境数据收集 | `/target_ekf_node/target_odom` |

### 数据预处理

| 文件 | 用途 | 输入 |
|------|------|------|
| `preprocess_gazebo_data.py` | 处理Gazebo收集的数据 | `gazebo_raw/raw_trajectories.csv` |
| `preprocess_data.py` | 处理EKF收集的数据 | `raw_trajectories.csv` |

两个预处理脚本功能相同：
1. SG滤波计算速度/加速度
2. 航向对齐坐标变换
3. 滑动窗口生成样本
4. 划分训练/验证/测试集

### Launch文件

| 文件 | 用途 |
|------|------|
| `collect_gazebo_data.launch` | Gazebo仿真数据收集 |
| `data_collection.launch` | 真实环境数据收集 |

---

## Phase 1 输出

训练完成后生成：

1. **`checkpoints/teacher/best_model.pth`**: 最佳Teacher模型
2. **`checkpoints/teacher/teacher_errors.npz`**: 误差统计 (用于Phase 2蒸馏)
3. **`results/teacher_test_metrics.json`**: 测试集评估指标
4. **`results/teacher_test_results.png`**: 可视化结果

---

## 下一步

完成Phase 1后，继续执行Phase 2进行知识蒸馏：

```bash
# Phase 2: 知识蒸馏 (待实现)
python3 scripts/train_student.py \
    --teacher_path checkpoints/teacher/best_model.pth \
    --data_path data/processed_data.npz \
    --save_dir checkpoints/student
```

---

## 常见问题

### Q: Gazebo中找不到actor1?

确保你的Gazebo世界中有名为`actor1`的模型。可以用以下命令检查：
```bash
rostopic echo /gazebo/model_states -n 1 | grep name
```

### Q: 数据收集后样本数太少?

- 增加轨迹数量和持续时间
- 降低`--stride`参数增加样本重叠
- 检查轨迹是否满足最小长度要求

### Q: 训练loss不下降?

- 检查数据预处理是否正确
- 降低学习率
- 增加数据量

---

## 参考文献

- Gal & Ghahramani (2016). "Dropout as a Bayesian Approximation"
- Saputra et al. (2019). "Distilling Knowledge from a Deep Pose Regressor Network"
- Alahi et al. (2016). "Social LSTM: Human Trajectory Prediction"
