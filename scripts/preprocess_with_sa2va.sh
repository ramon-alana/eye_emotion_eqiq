#!/bin/bash
# 使用 Sa2VA 进行数据预处理的脚本

set -e

cd /code/sa2va_wzx/eye_emotion_iq

# 激活虚拟环境
source .venv/bin/activate

# 检查参数
if [ -z "$1" ]; then
    echo "使用方法: $0 <原始数据目录> [输出目录] [Sa2VA模型路径]"
    echo ""
    echo "示例:"
    echo "  $0 data/raw/my_dataset"
    echo "  $0 data/raw/my_dataset data/processed"
    echo "  $0 data/raw/my_dataset data/processed OMG-Research/Sa2VA-4B"
    exit 1
fi

RAW_DIR="$1"
OUTPUT_DIR="${2:-data/processed}"
SA2VA_MODEL="${3:-OMG-Research/Sa2VA-4B}"

# 检查原始数据目录
if [ ! -d "$RAW_DIR" ]; then
    echo "错误: 原始数据目录不存在: $RAW_DIR"
    exit 1
fi

echo "开始使用 Sa2VA 预处理数据..."
echo "原始数据目录: $RAW_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "Sa2VA 模型: $SA2VA_MODEL"

# 运行预处理
python -m src.data.preprocess \
    --raw-dir "$RAW_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --format custom \
    --use-sa2va \
    --sa2va-model "$SA2VA_MODEL"

echo ""
echo "预处理完成！"
echo "生成的 metadata.csv 位于: $OUTPUT_DIR/metadata.csv"
echo "处理后的图像位于: $OUTPUT_DIR/images/"


