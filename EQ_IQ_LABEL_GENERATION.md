# EQ/IQ 标签生成与模型重新训练指南

## 📋 概述

由于真实的EQ/IQ标签难以获得，本方案使用**代理指标**来生成标签，然后重新训练模型。

## 🔬 标签生成方法

### EQ (情绪智力) 代理指标

基于以下假设和特征：

1. **情绪识别能力 (30%)**
   - 能识别多种情绪类型的人通常EQ较高
   - 使用情绪分布的熵来衡量情绪识别多样性

2. **情绪调节能力 (40%)**
   - 情绪稳定性：效价和唤醒度在合理范围内
   - 情绪平衡：效价和唤醒度的平衡程度

3. **积极情绪倾向 (30%)**
   - 积极情绪倾向与EQ正相关（适度）

### IQ (认知能力) 代理指标

基于以下假设和特征：

1. **图像质量指标 (40%)**
   - 清晰度：使用拉普拉斯算子计算图像清晰度
   - 对比度：图像对比度反映视觉质量

2. **情绪与认知的关系 (30%)**
   - 中性情绪通常与更好的认知表现相关
   - 积极情绪（快乐、惊讶）也有助于认知

3. **情绪稳定性 (30%)**
   - 中等唤醒度与最佳认知表现相关
   - 中性效价与认知能力相关

## 🚀 使用方法

### 方法1: 使用自动化脚本（推荐）

```bash
cd /code/sa2va_wzx/eye_emotion_iq
bash scripts/retrain_with_labels.sh
```

这个脚本会自动：
1. 生成EQ/IQ标签
2. 验证标签
3. 重新训练模型

### 方法2: 分步执行

#### 步骤1: 生成标签

```bash
python scripts/generate_eq_iq_labels.py \
    --metadata data/processed/fer2013/metadata.csv \
    --image-root data/processed/fer2013 \
    --output data/processed/fer2013/metadata_with_labels.csv
```

#### 步骤2: 验证标签

```bash
python scripts/check_training_data.py
```

#### 步骤3: 训练模型

```bash
python src/train.py \
    --metadata data/processed/fer2013/metadata_with_labels.csv \
    --image-root data/processed/fer2013 \
    --epochs 30 \
    --batch-size 32 \
    --lr 2e-4 \
    --checkpoint-dir checkpoints \
    --num-emotions 7 \
    --device cuda
```

## 📊 标签格式

生成的标签是**原始值**（未归一化），范围大约在 [-2, 2]：

- **IQ标签**: `iq_proxy` 列，浮点数
- **EQ标签**: `eq_proxy` 列，浮点数

训练时，模型会学习将这些值映射到合适的范围。推理时，会使用sigmoid函数归一化到[0,1]，然后缩放到[0,100]。

## ⚠️ 重要说明

### 代理标签的局限性

1. **不是真实标签**: 这些标签是基于启发式方法生成的代理指标
2. **假设可能不准确**: 基于的假设可能不完全符合实际情况
3. **需要验证**: 建议在实际应用中验证模型效果

### 改进方向

1. **收集真实标签**: 如果可能，收集真实的EQ/IQ测试分数
2. **使用其他数据集**: 寻找包含EQ/IQ标签的公开数据集
3. **多任务学习**: 结合其他相关任务来改进标签质量
4. **半监督学习**: 使用少量真实标签 + 大量代理标签

## 📈 预期效果

使用代理标签训练后：

1. **模型会有学习**: EQ/IQ输出头会学习到一些模式
2. **输出更稳定**: 相比随机初始化，输出会更稳定和有意义
3. **仍需谨慎**: 由于标签是代理的，实际应用仍需谨慎

## 🔍 验证方法

训练后，可以通过以下方式验证：

1. **检查损失**: EQ/IQ的MAE损失应该下降
2. **检查分布**: 输出值的分布应该更合理
3. **实际测试**: 在测试集上验证效果

## 📝 相关文件

- 标签生成脚本: `scripts/generate_eq_iq_labels.py`
- 训练脚本: `scripts/retrain_with_labels.sh`
- 数据检查: `scripts/check_training_data.py`
- 训练代码: `src/train.py`

---

*最后更新: 2025-11-21*

