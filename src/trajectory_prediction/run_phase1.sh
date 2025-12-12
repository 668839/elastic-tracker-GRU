#!/bin/bash
# run_phase1.sh
# Phase 1: 数据收集 + Teacher模型训练
# 
# 使用方法:
#   ./run_phase1.sh                    # 使用模拟数据 (快速测试)
#   ./run_phase1.sh --gazebo           # 使用已收集的Gazebo数据
#   ./run_phase1.sh --real             # 使用已收集的真实环境数据
#
# 数据收集需要单独运行:
#   Gazebo仿真: roslaunch trajectory_prediction collect_gazebo_data.launch
#   真实环境:   roslaunch trajectory_prediction data_collection.launch

set -e  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Phase 1: Data Collection + Teacher Model${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查Python依赖
echo -e "\n${YELLOW}Checking Python dependencies...${NC}"
python3 -c "import torch; import numpy; import pandas; import scipy; import tqdm" 2>/dev/null || {
    echo -e "${RED}Missing Python dependencies. Please install:${NC}"
    echo "pip3 install torch numpy pandas scipy tqdm matplotlib"
    exit 1
}

# 创建必要的目录
mkdir -p data/gazebo_raw
mkdir -p checkpoints/teacher
mkdir -p results

# 解析参数
DATA_MODE="synthetic"
if [ "$1" == "--gazebo" ]; then
    DATA_MODE="gazebo"
elif [ "$1" == "--real" ]; then
    DATA_MODE="real"
fi

echo -e "\n${BLUE}Data mode: ${DATA_MODE}${NC}"

# ============================================================
# Step 1: 数据准备
# ============================================================
echo -e "\n${GREEN}Step 1: Data Preparation${NC}"
echo "----------------------------------------"

case $DATA_MODE in
    "synthetic")
        echo -e "${YELLOW}Generating SYNTHETIC data for testing...${NC}"
        python3 scripts/generate_synthetic_data.py \
            --num_trajectories 300 \
            --duration 10.0 \
            --sample_rate 30.0 \
            --output data/raw_trajectories.csv \
            --noise 0.02 \
            --seed 42
        
        RAW_DATA="data/raw_trajectories.csv"
        PREPROCESS_SCRIPT="scripts/preprocess_data.py"
        ;;
        
    "gazebo")
        echo -e "${YELLOW}Using GAZEBO simulation data${NC}"
        RAW_DATA="data/gazebo_raw/raw_trajectories.csv"
        PREPROCESS_SCRIPT="scripts/preprocess_gazebo_data.py"
        
        if [ ! -f "$RAW_DATA" ]; then
            echo -e "${RED}Error: $RAW_DATA not found!${NC}"
            echo ""
            echo "Please collect data first using:"
            echo "  roslaunch trajectory_prediction collect_gazebo_data.launch"
            echo ""
            echo "Or run directly:"
            echo "  python3 scripts/collect_gazebo_actor.py --output_dir data/gazebo_raw"
            exit 1
        fi
        echo -e "${GREEN}✓ Found Gazebo data: $RAW_DATA${NC}"
        ;;
        
    "real")
        echo -e "${YELLOW}Using REAL environment data (EKF)${NC}"
        RAW_DATA="data/raw_trajectories.csv"
        PREPROCESS_SCRIPT="scripts/preprocess_data.py"
        
        if [ ! -f "$RAW_DATA" ]; then
            echo -e "${RED}Error: $RAW_DATA not found!${NC}"
            echo ""
            echo "Please collect data first using:"
            echo "  roslaunch trajectory_prediction data_collection.launch"
            exit 1
        fi
        echo -e "${GREEN}✓ Found real data: $RAW_DATA${NC}"
        ;;
esac

# 显示数据统计
echo -e "\nData file: $RAW_DATA"
echo "Lines: $(wc -l < $RAW_DATA)"
echo "Preview:"
head -3 $RAW_DATA

# ============================================================
# Step 2: 数据预处理
# ============================================================
echo -e "\n${GREEN}Step 2: Data Preprocessing${NC}"
echo "----------------------------------------"
echo "Using: $PREPROCESS_SCRIPT"
echo "Applying SG filter, heading-aligned transform..."

if [ "$DATA_MODE" == "gazebo" ]; then
    python3 $PREPROCESS_SCRIPT \
        --input $RAW_DATA \
        --output data/processed_data.npz \
        --obs_len 30 \
        --pred_len 60 \
        --sample_rate 30.0 \
        --train_ratio 0.7 \
        --val_ratio 0.15 \
        --seed 42
else
    python3 $PREPROCESS_SCRIPT \
        --input $RAW_DATA \
        --output data/processed_data.npz \
        --obs_len 30 \
        --pred_len 60 \
        --sample_rate 30.0 \
        --train_ratio 0.7 \
        --val_ratio 0.15 \
        --seed 42
fi

# ============================================================
# Step 3: 训练Teacher模型
# ============================================================
echo -e "\n${GREEN}Step 3: Training Teacher Model${NC}"
echo "----------------------------------------"
echo "Architecture: 3-layer Bi-GRU encoder (256) + 2-layer decoder (256)"
echo "Expected parameters: ~2.1M"

# 检测GPU
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    DEVICE="cuda"
    echo -e "${GREEN}✓ CUDA available, using GPU${NC}"
else
    DEVICE="cpu"
    echo -e "${YELLOW}! CUDA not available, using CPU (slower)${NC}"
fi

python3 scripts/train_teacher.py \
    --data_path data/processed_data.npz \
    --save_dir checkpoints/teacher \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --dropout 0.2 \
    --device $DEVICE \
    --seed 42

# ============================================================
# Step 4: 评估模型
# ============================================================
echo -e "\n${GREEN}Step 4: Evaluating Teacher Model${NC}"
echo "----------------------------------------"

python3 scripts/evaluate_model.py \
    --data_path data/processed_data.npz \
    --checkpoint checkpoints/teacher/best_model.pth \
    --model_type teacher \
    --batch_size 64 \
    --device $DEVICE \
    --output_dir results \
    --split test

# ============================================================
# 完成
# ============================================================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Phase 1 Completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Data mode: $DATA_MODE"
echo ""
echo "Outputs:"
echo "  - Raw data:        $RAW_DATA"
echo "  - Processed data:  data/processed_data.npz"
echo "  - Teacher model:   checkpoints/teacher/best_model.pth"
echo "  - Teacher errors:  checkpoints/teacher/teacher_errors.npz"
echo "  - Results:         results/teacher_test_metrics.json"
echo "  - Plots:           results/teacher_test_results.png"
echo ""
echo "Next step: Run Phase 2 (Knowledge Distillation)"
echo "  python3 scripts/train_student.py (coming in Phase 2)"
