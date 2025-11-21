#!/bin/bash
# 自动下载 FER2013 数据集的脚本

set -e

cd /code/sa2va_wzx/eye_emotion_iq

echo "=========================================="
echo "FER2013 数据集下载脚本"
echo "=========================================="
echo ""

# 检查是否安装了 kaggle
if ! command -v kaggle &> /dev/null; then
    echo "Kaggle CLI 未安装，正在安装..."
    pip install kaggle
    echo ""
    echo "⚠️  请配置 Kaggle API："
    echo "1. 访问 https://www.kaggle.com/account"
    echo "2. 创建 API Token（下载 kaggle.json）"
    echo "3. 将 kaggle.json 放在 ~/.kaggle/ 目录下"
    echo "4. 运行: chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    echo "配置完成后，重新运行此脚本"
    exit 1
fi

# 检查 Kaggle 配置
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ 未找到 Kaggle API 配置文件"
    echo "请按照上面的说明配置 Kaggle API"
    exit 1
fi

echo "✓ Kaggle CLI 已安装"
echo ""

# 创建目录
mkdir -p data/raw/fer2013

echo "正在下载 FER2013 数据集..."
echo "这可能需要一些时间，请耐心等待..."
echo ""

# 下载数据集
kaggle datasets download -d msambare/fer2013 -p data/raw/fer2013

echo ""
echo "正在解压..."
cd data/raw/fer2013
unzip -q fer2013.zip || unzip -q *.zip

echo ""
echo "=========================================="
echo "下载完成！"
echo "=========================================="
echo ""
echo "数据集位置: data/raw/fer2013/"
echo ""
echo "下一步：运行预处理"
echo "  python -m src.data.preprocess \\"
echo "    --raw-dir data/raw/fer2013 \\"
echo "    --output-dir data/processed/fer2013 \\"
echo "    --format custom"
echo ""


