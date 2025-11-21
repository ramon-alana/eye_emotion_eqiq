# 真实数据集下载与使用指南

本指南将帮助你下载和准备用于训练的真实数据集。

## 推荐数据集

### 1. FER2013（推荐，易于获取）

**特点：**
- 约 35,000 张面部图像
- 7 种情绪类别：愤怒、厌恶、恐惧、快乐、中性、悲伤、惊讶
- 图像尺寸：48×48（较小，但适合快速测试）
- 可直接从 Kaggle 下载

**下载步骤：**

1. **从 Kaggle 下载（需要 Kaggle 账号）**
   ```bash
   # 安装 kaggle CLI（如果还没有）
   pip install kaggle
   
   # 配置 Kaggle API（需要从 kaggle.com 获取 API token）
   # 将 kaggle.json 放在 ~/.kaggle/ 目录下
   
   # 下载数据集
   kaggle datasets download -d msambare/fer2013
   unzip fer2013.zip -d data/raw/fer2013
   ```

2. **手动下载**
   - 访问：https://www.kaggle.com/datasets/msambare/fer2013
   - 点击 "Download" 按钮
   - 解压到 `data/raw/fer2013/` 目录

**预处理：**
```bash
# FER2013 已经是裁剪好的面部图像，但需要提取眼部区域
python -m src.data.preprocess \
  --raw-dir data/raw/fer2013 \
  --output-dir data/processed/fer2013 \
  --format custom
```

---

### 2. RAF-DB（Real-world Affective Faces Database）

**特点：**
- 约 30,000 张高分辨率面部图像
- 7 种基本情绪 + 12 种复合情绪
- 图像质量高，适合训练

**下载步骤：**

1. **访问官网**
   - 网址：http://www.whdeng.cn/raf/model1.html
   - 需要注册并申请访问权限

2. **下载后处理**
   ```bash
   # 将下载的文件解压到 data/raw/rafdb/
   # 通常包含：
   #   - images/ (图像文件夹)
   #   - EmoLabel/list_patition_label.txt (标注文件)
   
   # 需要转换标注格式为 CSV
   python scripts/convert_rafdb_labels.py
   ```

---

### 3. AffectNet（大规模数据集）

**特点：**
- 超过 100 万张图像
- 7 种情绪类别
- 包含效价（valence）和唤醒度（arousal）标注
- 需要申请访问权限

**下载步骤：**

1. **访问官网**
   - 网址：http://mohammadmahoor.com/affectnet/
   - 需要注册并填写申请表单

2. **下载后处理**
   ```bash
   # 将下载的数据放在 data/raw/affectnet/
   # 通常包含：
   #   - images/ (图像文件夹)
   #   - annotations.csv (标注文件)
   
   # 使用 AffectNet 格式预处理
   python -m src.data.preprocess \
     --raw-dir data/raw/affectnet \
     --output-dir data/processed/affectnet \
     --format affectnet \
     --annotation-file data/raw/affectnet/annotations.csv
   ```

---

### 4. CK+ (Extended Cohn-Kanade Dataset)

**特点：**
- 约 1,000 个视频序列
- 7 种情绪类别
- 图像质量高，但数据量较小

**下载步骤：**

1. **访问官网**
   - 网址：https://www.kaggle.com/datasets/shawon10/ckplus
   - 或：http://www.pitt.edu/~emotion/ck-spread.htm

2. **下载后处理**
   ```bash
   # 从视频中提取帧
   python scripts/extract_frames_from_videos.py \
     --video-dir data/raw/ckplus/videos \
     --output-dir data/raw/ckplus/images
   
   # 预处理
   python -m src.data.preprocess \
     --raw-dir data/raw/ckplus/images \
     --output-dir data/processed/ckplus \
     --format custom
   ```

---

## 快速开始（使用 FER2013）

### 步骤 1：下载数据集

```bash
cd /code/sa2va_wzx/eye_emotion_iq

# 方式 1: 使用 Kaggle CLI（推荐）
pip install kaggle
# 配置 Kaggle API token（从 kaggle.com 获取）
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d data/raw/fer2013

# 方式 2: 手动下载
# 访问 https://www.kaggle.com/datasets/msambare/fer2013
# 下载后解压到 data/raw/fer2013/
```

### 步骤 2：预处理数据

```bash
source .venv/bin/activate

# 提取眼部区域
python -m src.data.preprocess \
  --raw-dir data/raw/fer2013 \
  --output-dir data/processed/fer2013 \
  --format custom \
  --max-samples 1000  # 先用少量数据测试
```

### 步骤 3：检查数据

```bash
# 查看生成的 metadata.csv
head -10 data/processed/fer2013/metadata.csv

# 查看图像数量
ls data/processed/fer2013/images/ | wc -l
```

### 步骤 4：开始训练

```bash
python src/train.py \
  --metadata data/processed/fer2013/metadata.csv \
  --image-root data/processed/fer2013/images \
  --epochs 30 \
  --batch-size 32 \
  --lr 2e-4
```

---

## 数据格式要求

### 输入格式

预处理脚本支持以下目录结构：

```
data/raw/<dataset_name>/
  ├── images/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  └── annotations.csv  (可选)
```

### 标注文件格式（CSV）

如果使用 AffectNet 格式，CSV 应包含：

```csv
subDirectory_filePath,expression,valence,arousal
images/001.jpg,3,0.5,0.3
images/002.jpg,4,-0.2,0.1
...
```

- `subDirectory_filePath`: 图像路径
- `expression`: 情绪标签（0-6）
- `valence`: 效价（-1 到 1）
- `arousal`: 唤醒度（-1 到 1）

---

## 数据增强建议

对于小数据集，建议在训练时使用数据增强：

```python
# 在 dataset.py 中已包含数据增强
# - 水平翻转
# - 颜色抖动
# - 归一化
```

---

## 常见问题

### Q: 数据集太大，下载很慢怎么办？

A: 可以先下载部分数据测试：
```bash
# 只处理前 1000 张图像
python -m src.data.preprocess \
  --raw-dir data/raw/fer2013 \
  --max-samples 1000
```

### Q: 如何合并多个数据集？

A: 可以分别预处理，然后合并 metadata.csv：
```python
import pandas as pd

df1 = pd.read_csv('data/processed/dataset1/metadata.csv')
df2 = pd.read_csv('data/processed/dataset2/metadata.csv')
df_combined = pd.concat([df1, df2], ignore_index=True)
df_combined.to_csv('data/processed/combined/metadata.csv', index=False)
```

### Q: 内存不足怎么办？

A: 
1. 减小 batch_size
2. 使用 `--max-samples` 限制数据量
3. 分批处理数据

---

## 下一步

下载并预处理数据后，参考 [训练指南](../README.md#训练) 开始训练模型。


