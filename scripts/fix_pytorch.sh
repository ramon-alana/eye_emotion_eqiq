#!/bin/bash
# 修复 PyTorch 安装问题的脚本

set -e

cd /code/sa2va_wzx/eye_emotion_iq

echo "正在激活虚拟环境..."
source .venv/bin/activate

echo "正在卸载现有的 PyTorch 相关包..."
pip uninstall -y torch torchvision torchaudio torchmetrics || true

echo "正在清理 pip 缓存..."
pip cache purge || true

echo "正在重新安装 PyTorch 和相关依赖..."
# 使用官方推荐的安装方式，根据 CUDA 版本选择
# 如果有 CUDA，使用 CUDA 版本；否则使用 CPU 版本
if command -v nvidia-smi &> /dev/null; then
    echo "检测到 NVIDIA GPU，安装 CUDA 版本的 PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "未检测到 GPU，安装 CPU 版本的 PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo "正在安装其他依赖..."
pip install torchmetrics>=1.4
pip install -r requirements.txt

echo "验证安装..."
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}')"

echo "修复完成！"


