#!/bin/bash
# Demo 运行脚本

set -e

cd /code/sa2va_wzx/eye_emotion_iq

# 激活虚拟环境
source .venv/bin/activate

# 检查参数
if [ -z "$1" ]; then
    echo "使用方法: $0 <图片路径> [模型检查点路径]"
    echo ""
    echo "示例:"
    echo "  $0 /path/to/image.jpg"
    echo "  $0 /path/to/image.jpg checkpoints/best_model.pth"
    echo ""
    echo "如果没有提供模型检查点，将使用未训练的模型（结果仅供参考）"
    exit 1
fi

IMAGE_PATH="$1"
CHECKPOINT="${2:-}"

# 检查图片是否存在
if [ ! -f "$IMAGE_PATH" ]; then
    echo "错误: 图片文件不存在: $IMAGE_PATH"
    exit 1
fi

# 创建输出目录
mkdir -p data/demo_output

# 运行 demo
echo "正在运行 Demo..."
python -m src.demo \
    --image "$IMAGE_PATH" \
    --output-eye "data/demo_output/extracted_eye.jpg" \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
    --format pretty

echo ""
echo "Demo 完成！提取的眼部图像保存在: data/demo_output/extracted_eye.jpg"


