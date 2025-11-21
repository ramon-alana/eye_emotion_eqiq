# Sa2VA 集成说明

本项目已完全集成 Sa2VA 模型用于眼部区域提取，可以在数据预处理、训练和推理的各个环节使用。

## 功能特性

1. **统一接口**：通过 `UnifiedEyeExtractor` 自动选择最佳后端
2. **多后端支持**：支持 MediaPipe 和 Sa2VA 两种提取方法
3. **自动回退**：如果 Sa2VA 不可用，自动回退到 MediaPipe
4. **全流程集成**：在预处理、训练和推理中都可以使用

## 使用方法

### 1. 数据预处理

#### 使用 Sa2VA 预处理数据

```bash
# 方式 1: 使用脚本
bash scripts/preprocess_with_sa2va.sh data/raw/my_dataset

# 方式 2: 直接使用 Python 模块
python -m src.data.preprocess \
  --raw-dir data/raw/my_dataset \
  --output-dir data/processed \
  --format custom \
  --use-sa2va \
  --sa2va-model OMG-Research/Sa2VA-4B
```

#### 使用 MediaPipe（默认）

```bash
python -m src.data.preprocess \
  --raw-dir data/raw/my_dataset \
  --output-dir data/processed \
  --format custom
```

### 2. Demo 推理

#### 使用 Sa2VA 进行推理

```bash
python -m src.demo \
  --image data/raw/test_image.jpg \
  --use-sa2va \
  --sa2va-model OMG-Research/Sa2VA-4B \
  --output-eye data/demo_output/extracted_eye.jpg
```

#### 使用自动选择（推荐）

```bash
python -m src.demo \
  --image data/raw/test_image.jpg \
  --output-eye data/demo_output/extracted_eye.jpg
```

系统会自动选择可用的最佳后端。

### 3. 在代码中使用

```python
from src.data.unified_extractor import create_eye_extractor
import cv2

# 自动选择后端（优先 Sa2VA）
extractor = create_eye_extractor(backend="auto")

# 或明确指定后端
extractor = create_eye_extractor(backend="sa2va", sa2va_model="OMG-Research/Sa2VA-4B")
# 或
extractor = create_eye_extractor(backend="mediapipe")

# 提取眼部区域
image = cv2.imread("path/to/image.jpg")
eye_region = extractor.extract_eye_region(image, eye_side="both")

# 查看使用的后端
print(f"当前后端: {extractor.get_backend()}")
```

## 后端对比

| 特性 | MediaPipe | Sa2VA |
|------|-----------|-------|
| 速度 | 快 | 较慢 |
| 精度 | 基于关键点 | 语义分割，更精确 |
| 资源需求 | 低 | 高（需要 GPU） |
| 适用场景 | 实时处理、批量处理 | 高精度需求 |

## 配置要求

### Sa2VA 配置

1. 确保 `vlm` 目录存在且可访问
2. Sa2VA 模型路径正确（默认：`OMG-Research/Sa2VA-4B`）
3. 有足够的 GPU 内存（建议 8GB+）

### MediaPipe 配置

1. 安装 `mediapipe` 包：`pip install mediapipe>=0.10.0`
2. 无需额外配置

## 故障排除

### Sa2VA 不可用

如果 Sa2VA 初始化失败，系统会自动回退到 MediaPipe。常见原因：

1. **模型路径错误**：检查 `--sa2va-model` 参数
2. **GPU 内存不足**：尝试使用较小的模型或 CPU 模式
3. **依赖缺失**：确保 `vlm` 目录配置正确

### 性能优化

1. **批量处理**：对于大量数据，建议使用 MediaPipe 以提高速度
2. **混合使用**：对关键样本使用 Sa2VA，其他使用 MediaPipe
3. **缓存结果**：预处理后的图像可以重复使用，无需重复提取

## 示例

完整的数据处理流程：

```bash
# 1. 使用 Sa2VA 预处理数据
python -m src.data.preprocess \
  --raw-dir data/raw/my_dataset \
  --output-dir data/processed \
  --format custom \
  --use-sa2va

# 2. 训练模型
python src/train.py \
  --metadata data/processed/metadata.csv \
  --image-root data/processed/images \
  --epochs 30 \
  --batch-size 64

# 3. 使用 Sa2VA 进行推理
python -m src.demo \
  --image data/raw/test_image.jpg \
  --use-sa2va \
  --checkpoint checkpoints/best_model.pth
```

## 注意事项

1. Sa2VA 模型较大，首次加载需要时间
2. 使用 Sa2VA 时建议在 GPU 上运行
3. 对于实时应用，MediaPipe 可能更合适
4. 预处理阶段使用 Sa2VA 可以提高数据质量，但会增加处理时间


