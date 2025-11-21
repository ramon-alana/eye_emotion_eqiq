#!/bin/bash
# 完整的重新训练流程：生成标签 -> 训练模型

set -e  # 遇到错误立即退出

PROJECT_ROOT="/code/sa2va_wzx/eye_emotion_iq"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "EQ/IQ标签生成与模型重新训练流程"
echo "=========================================="

# 配置
METADATA_INPUT="data/processed/fer2013/metadata.csv"
METADATA_OUTPUT="data/processed/fer2013/metadata_with_labels.csv"
IMAGE_ROOT="data/processed/fer2013"
CHECKPOINT_DIR="checkpoints"
EPOCHS=30
BATCH_SIZE=32
LR=2e-4

echo ""
echo "步骤 1: 生成EQ/IQ代理标签..."
echo "----------------------------------------"
python scripts/generate_eq_iq_labels.py \
    --metadata "$METADATA_INPUT" \
    --image-root "$IMAGE_ROOT" \
    --output "$METADATA_OUTPUT"

if [ ! -f "$METADATA_OUTPUT" ]; then
    echo "错误: 标签生成失败"
    exit 1
fi

echo ""
echo "步骤 2: 验证生成的标签..."
echo "----------------------------------------"
python -c "
import pandas as pd
df = pd.read_csv('$METADATA_OUTPUT')
print(f'总记录数: {len(df)}')
print(f'IQ标签非空: {df[\"iq_proxy\"].notna().sum()} ({df[\"iq_proxy\"].notna().sum()/len(df)*100:.1f}%)')
print(f'EQ标签非空: {df[\"eq_proxy\"].notna().sum()} ({df[\"eq_proxy\"].notna().sum()/len(df)*100:.1f}%)')
if df['iq_proxy'].notna().sum() > 0:
    print(f'IQ均值: {df[\"iq_proxy\"].mean():.3f}, 标准差: {df[\"iq_proxy\"].std():.3f}')
if df['eq_proxy'].notna().sum() > 0:
    print(f'EQ均值: {df[\"eq_proxy\"].mean():.3f}, 标准差: {df[\"eq_proxy\"].std():.3f}')
"

echo ""
echo "步骤 3: 开始训练模型..."
echo "----------------------------------------"
python src/train.py \
    --metadata "$METADATA_OUTPUT" \
    --image-root "$IMAGE_ROOT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --num-emotions 7 \
    --device cuda

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "生成的标签文件: $METADATA_OUTPUT"
echo "模型检查点保存在: $CHECKPOINT_DIR/"
echo ""
echo "可以使用以下命令测试新模型:"
echo "  python src/demo.py --image data/demo_images/ben.jpg --checkpoint $CHECKPOINT_DIR/epoch_${EPOCHS}.pt --skip-extraction"

