#!/bin/bash
# 使用 tmux 在后台训练模型的脚本

set -e

cd /code/sa2va_wzx/eye_emotion_iq

SESSION_NAME="eye_iq_train"

echo "=========================================="
echo "启动训练任务（tmux 会话: $SESSION_NAME）"
echo "=========================================="
echo ""

# 检查数据
if [ ! -f "data/raw/fer2013_processed/metadata.csv" ]; then
    echo "❌ 错误: 找不到数据文件"
    echo "请先运行: python scripts/prepare_fer2013_format.py data/raw/dataset data/raw/fer2013_processed"
    exit 1
fi

# 检查是否已经有预处理好的眼部图像
if [ ! -d "data/processed/fer2013/images" ] || [ -z "$(ls -A data/processed/fer2013/images 2>/dev/null)" ]; then
    echo "⚠️  警告: 未找到预处理好的眼部图像"
    echo "将使用原始图像（FER2013 已经是面部图像）"
    echo ""
    METADATA="data/raw/fer2013_processed/metadata.csv"
    IMAGE_ROOT="data/raw/fer2013_processed"
else
    echo "✓ 使用预处理好的眼部图像"
    METADATA="data/processed/fer2013/metadata.csv"
    IMAGE_ROOT="data/processed/fer2013"
fi

# 检查是否已有同名会话
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠️  警告: tmux 会话 '$SESSION_NAME' 已存在"
    read -p "是否要终止现有会话并创建新的？(y/n): " answer
    if [ "$answer" = "y" ]; then
        tmux kill-session -t "$SESSION_NAME"
    else
        echo "使用现有会话，运行: tmux attach -t $SESSION_NAME"
        exit 0
    fi
fi

# 激活虚拟环境并启动训练
echo "正在创建 tmux 会话..."
tmux new-session -d -s "$SESSION_NAME" -c "$(pwd)" bash -c "
    source .venv/bin/activate && \
    echo '==========================================' && \
    echo '开始训练模型' && \
    echo '==========================================' && \
    echo '' && \
    echo '数据文件: $METADATA' && \
    echo '图像目录: $IMAGE_ROOT' && \
    echo '' && \
    python src/train.py \
        --metadata $METADATA \
        --image-root $IMAGE_ROOT \
        --epochs 30 \
        --batch-size 32 \
        --lr 2e-4 \
        --num-emotions 7 \
        --num-workers 4 \
        --checkpoint-dir checkpoints && \
    echo '' && \
    echo '==========================================' && \
    echo '训练完成！' && \
    echo '==========================================' && \
    echo '' && \
    echo '模型检查点保存在: checkpoints/' && \
    echo '' && \
    read -p '按 Enter 键退出...'
"

echo "✓ tmux 会话已创建: $SESSION_NAME"
echo ""
echo "查看训练进度："
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "分离会话（不停止训练）："
echo "  按 Ctrl+B，然后按 D"
echo ""
echo "停止训练："
echo "  tmux kill-session -t $SESSION_NAME"
echo ""
echo "查看所有会话："
echo "  tmux ls"
echo ""

# 询问是否立即查看
read -p "是否立即查看训练进度？(y/n): " view_now
if [ "$view_now" = "y" ]; then
    tmux attach -t "$SESSION_NAME"
fi

