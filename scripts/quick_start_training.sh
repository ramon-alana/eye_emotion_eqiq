#!/bin/bash
# 快速开始训练脚本 - 使用 FER2013 数据集

set -e

cd /code/sa2va_wzx/eye_emotion_iq

echo "=========================================="
echo "快速开始训练流程"
echo "=========================================="
echo ""

# 激活虚拟环境
source .venv/bin/activate

# 检查数据集是否存在
if [ ! -d "data/raw/fer2013" ] || [ -z "$(ls -A data/raw/fer2013 2>/dev/null)" ]; then
    echo "❌ FER2013 数据集未找到"
    echo ""
    echo "请先下载数据集："
    echo "  方式 1: bash scripts/download_fer2013.sh"
    echo "  方式 2: 手动从 https://www.kaggle.com/datasets/msambare/fer2013 下载"
    echo ""
    exit 1
fi

echo "✓ 数据集已找到"
echo ""

# 步骤 1: 预处理数据
echo "步骤 1: 预处理数据（提取眼部区域）..."
echo ""

read -p "要处理多少样本？(直接回车使用全部，或输入数字，如 1000): " num_samples

if [ -z "$num_samples" ]; then
    PREPROCESS_CMD="python -m src.data.preprocess \
      --raw-dir data/raw/fer2013 \
      --output-dir data/processed/fer2013 \
      --format custom"
else
    PREPROCESS_CMD="python -m src.data.preprocess \
      --raw-dir data/raw/fer2013 \
      --output-dir data/processed/fer2013 \
      --format custom \
      --max-samples $num_samples"
fi

echo "运行: $PREPROCESS_CMD"
eval $PREPROCESS_CMD

if [ ! -f "data/processed/fer2013/metadata.csv" ]; then
    echo "❌ 预处理失败"
    exit 1
fi

echo ""
echo "✓ 预处理完成"
echo ""

# 步骤 2: 检查数据
num_images=$(ls data/processed/fer2013/images/ 2>/dev/null | wc -l)
echo "处理后的图像数量: $num_images"
echo ""

# 步骤 3: 开始训练
echo "步骤 2: 开始训练..."
echo ""

read -p "训练轮数 (默认 30): " epochs
epochs=${epochs:-30}

read -p "批次大小 (默认 32): " batch_size
batch_size=${batch_size:-32}

read -p "学习率 (默认 2e-4): " lr
lr=${lr:-2e-4}

echo ""
echo "开始训练..."
echo ""

python src/train.py \
  --metadata data/processed/fer2013/metadata.csv \
  --image-root data/processed/fer2013/images \
  --epochs $epochs \
  --batch-size $batch_size \
  --lr $lr

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "模型检查点保存在: checkpoints/"
echo ""
echo "使用训练好的模型进行推理："
echo "  python -m src.demo \\"
echo "    --image <图片路径> \\"
echo "    --checkpoint checkpoints/epoch_${epochs}.pt"
echo ""


