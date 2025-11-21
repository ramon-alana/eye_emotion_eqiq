#!/bin/bash
# 数据预处理脚本示例

set -e

cd /code/sa2va_wzx/eye_emotion_iq

# 激活虚拟环境
source .venv/bin/activate

# 安装 mediapipe（如果还没安装）
pip install mediapipe>=0.10.0

# 示例 1: 处理自定义格式的数据集（仅图像文件）
# 将你的原始图像放在 data/raw/my_dataset/ 目录下
python -m src.data.preprocess \
  --raw-dir data/raw/my_dataset \
  --output-dir data/processed \
  --format custom \
  --max-samples 1000  # 可选：限制处理数量用于测试

# 示例 2: 处理 AffectNet 格式的数据集（如果有标注文件）
# python -m src.data.preprocess \
#   --raw-dir data/raw/affectnet \
#   --output-dir data/processed \
#   --format affectnet \
#   --annotation-file data/raw/affectnet/annotations.csv

echo "数据预处理完成！"
echo "生成的 metadata.csv 位于: data/processed/metadata.csv"
echo "处理后的图像位于: data/processed/images/"


