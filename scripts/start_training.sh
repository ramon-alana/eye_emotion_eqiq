#!/bin/bash
# 一键启动训练（自动处理数据并启动 tmux 训练）

set -e

cd /code/sa2va_wzx/eye_emotion_iq

SESSION_NAME="eye_iq_train"

echo "=========================================="
echo "准备训练数据并启动训练"
echo "=========================================="
echo ""

source .venv/bin/activate

# 检查原始数据
if [ ! -f "data/raw/fer2013_processed/metadata.csv" ]; then
    echo "❌ 错误: 找不到数据文件"
    exit 1
fi

# 检查是否需要预处理
if [ ! -d "data/processed/fer2013/images" ] || [ -z "$(ls -A data/processed/fer2013/images 2>/dev/null)" ]; then
    echo "正在预处理数据（提取眼部区域）..."
    echo "这可能需要一些时间..."
    echo ""
    
    python -m src.data.preprocess \
        --raw-dir data/raw/fer2013_processed/images \
        --output-dir data/processed/fer2013 \
        --format custom \
        --max-samples 10000
    
    if [ ! -f "data/processed/fer2013/metadata.csv" ]; then
        echo "❌ 预处理失败，使用原始图像"
        METADATA="data/raw/fer2013_processed/metadata.csv"
        IMAGE_ROOT="data/raw/fer2013_processed"
    else
        echo "✓ 预处理完成"
        METADATA="data/processed/fer2013/metadata.csv"
        IMAGE_ROOT="data/processed/fer2013"
    fi
else
    echo "✓ 使用已预处理的数据"
    METADATA="data/processed/fer2013/metadata.csv"
    IMAGE_ROOT="data/processed/fer2013"
fi

echo ""
echo "数据文件: $METADATA"
echo "图像目录: $IMAGE_ROOT"
echo ""

# 启动 tmux 训练
bash scripts/train_tmux.sh

