# 手动下载数据集指南（无需 Kaggle CLI）

如果网络无法访问 Kaggle 或 pip 安装失败，可以使用以下方法手动下载数据集。

## 方法 1: 手动下载 FER2013

### 步骤 1: 下载数据集

1. **访问 Kaggle 网站**
   - 打开浏览器访问：https://www.kaggle.com/datasets/msambare/fer2013
   - 需要登录 Kaggle 账号（如果没有，先注册）

2. **下载数据集**
   - 点击页面上的 "Download" 按钮
   - 下载完成后会得到一个 zip 文件（通常名为 `fer2013.zip`）

3. **上传到服务器**
   ```bash
   # 使用 scp 或其他方式将文件上传到服务器
   # 例如：
   scp fer2013.zip user@server:/code/sa2va_wzx/eye_emotion_iq/data/raw/
   ```

4. **解压数据集**
   ```bash
   cd /code/sa2va_wzx/eye_emotion_iq
   mkdir -p data/raw/fer2013
   unzip data/raw/fer2013.zip -d data/raw/fer2013/
   ```

### 步骤 2: 预处理数据

```bash
source .venv/bin/activate

python -m src.data.preprocess \
  --raw-dir data/raw/fer2013 \
  --output-dir data/processed/fer2013 \
  --format custom
```

---

## 方法 2: 使用其他公开数据集

### CK+ 数据集（无需注册）

1. **下载地址**
   - Kaggle: https://www.kaggle.com/datasets/shawon10/ckplus
   - 或直接搜索 "CK+ dataset"

2. **处理步骤**
   ```bash
   # 解压后
   python -m src.data.preprocess \
     --raw-dir data/raw/ckplus \
     --output-dir data/processed/ckplus \
     --format custom
   ```

### JAFFE 数据集（日本女性面部表情）

1. **下载地址**
   - 搜索 "JAFFE dataset" 或访问相关学术网站
   - 通常可以直接下载

2. **处理步骤**
   ```bash
   python -m src.data.preprocess \
     --raw-dir data/raw/jaffe \
     --output-dir data/processed/jaffe \
     --format custom
   ```

---

## 方法 3: 使用现有图片创建小数据集

如果你已经有一些图片，可以直接使用：

```bash
# 将图片放在 data/raw/my_images/ 目录
python -m src.data.preprocess \
  --raw-dir data/raw/my_images \
  --output-dir data/processed/my_images \
  --format custom
```

---

## 方法 4: 使用合成数据集进行测试

如果只是想测试训练流程，可以使用合成数据集：

```bash
python scripts/create_synthetic_dataset.py 500

# 然后训练
python src/train.py \
  --metadata data/processed_synthetic/metadata.csv \
  --image-root data/processed_synthetic/images \
  --epochs 10 \
  --batch-size 8
```

---

## 网络问题解决方案

### 如果 pip 安装失败

1. **使用国内镜像源**
   ```bash
   pip install kaggle -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

2. **手动下载 wheel 文件**
   - 访问：https://pypi.org/project/kaggle/#files
   - 下载对应的 wheel 文件
   - 使用 `pip install kaggle-*.whl` 安装

3. **使用 conda（如果有）**
   ```bash
   conda install -c conda-forge kaggle
   ```

---

## 快速检查数据集结构

下载后，检查数据集结构：

```bash
# 查看目录结构
ls -la data/raw/fer2013/

# 应该看到类似：
# - train/ (训练图像)
# - test/ (测试图像)
# - 或其他图像文件夹
```

如果结构不同，可能需要调整预处理命令。

---

## 下一步

数据集准备好后，运行：

```bash
# 预处理
python -m src.data.preprocess \
  --raw-dir data/raw/<dataset_name> \
  --output-dir data/processed/<dataset_name> \
  --format custom

# 训练
python src/train.py \
  --metadata data/processed/<dataset_name>/metadata.csv \
  --image-root data/processed/<dataset_name>/images \
  --epochs 30 \
  --batch-size 32
```


