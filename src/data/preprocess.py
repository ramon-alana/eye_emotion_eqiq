"""
数据预处理脚本：从面部图像中提取眼部区域并生成 metadata.csv
支持多种数据集格式和面部检测方法（MediaPipe 和 Sa2VA）
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

# 尝试导入 Sa2VA 提取器
try:
    from .sa2va_eye_extractor import Sa2VAEyeExtractor, create_sa2va_extractor
    SA2VA_AVAILABLE = True
except ImportError:
    SA2VA_AVAILABLE = False


class EyeExtractor:
    """使用 MediaPipe 从面部图像中提取眼部区域"""

    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        # MediaPipe 面部网格关键点索引（眼部区域）
        # 左眼：33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
        # 右眼：362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
        self.left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    def extract_eye_region(self, image: np.ndarray, eye_side: str = "both") -> Optional[np.ndarray]:
        """
        提取眼部区域
        Args:
            image: 输入图像 (BGR 格式)
            eye_side: "left", "right", 或 "both"（合并双眼）
        Returns:
            裁剪并调整大小后的眼部图像，如果检测失败返回 None
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]

        if eye_side == "both":
            # 提取双眼区域并合并
            left_eye = self._extract_single_eye(image, face_landmarks, self.left_eye_indices, h, w)
            right_eye = self._extract_single_eye(image, face_landmarks, self.right_eye_indices, h, w)
            
            if left_eye is None or right_eye is None:
                return None
            
            # 调整尺寸使左右眼高度一致（取较大的高度）
            max_height = max(left_eye.shape[0], right_eye.shape[0])
            if left_eye.shape[0] != max_height:
                left_eye = cv2.resize(left_eye, (int(left_eye.shape[1] * max_height / left_eye.shape[0]), max_height))
            if right_eye.shape[0] != max_height:
                right_eye = cv2.resize(right_eye, (int(right_eye.shape[1] * max_height / right_eye.shape[0]), max_height))
            
            # 水平拼接双眼
            combined = np.hstack([left_eye, right_eye])
            return cv2.resize(combined, (self.image_size, self.image_size))
        elif eye_side == "left":
            eye = self._extract_single_eye(image, face_landmarks, self.left_eye_indices, h, w)
        elif eye_side == "right":
            eye = self._extract_single_eye(image, face_landmarks, self.right_eye_indices, h, w)
        else:
            raise ValueError(f"Invalid eye_side: {eye_side}")

        if eye is None:
            return None

        return cv2.resize(eye, (self.image_size, self.image_size))

    def _extract_single_eye(
        self, image: np.ndarray, face_landmarks, eye_indices: List[int], img_h: int, img_w: int
    ) -> Optional[np.ndarray]:
        """提取单只眼睛区域"""
        points = []
        for idx in eye_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * img_w)
            y = int(landmark.y * img_h)
            points.append([x, y])

        points = np.array(points)
        if len(points) == 0:
            return None

        # 计算边界框，添加一些边距
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        # 添加边距（20%）
        margin_x = int((x_max - x_min) * 0.2)
        margin_y = int((y_max - y_min) * 0.2)
        
        x_min = max(0, x_min - margin_x)
        y_min = max(0, y_min - margin_y)
        x_max = min(img_w, x_max + margin_x)
        y_max = min(img_h, y_max + margin_y)

        # 确保是正方形区域
        width = x_max - x_min
        height = y_max - y_min
        size = max(width, height)
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        x_min = max(0, center_x - size // 2)
        y_min = max(0, center_y - size // 2)
        x_max = min(img_w, x_min + size)
        y_max = min(img_h, y_min + size)

        # 裁剪图像
        eye_region = None
        try:
            eye_region = image[y_min:y_max, x_min:x_max]
            if eye_region.size == 0:
                return None
        except:
            return None

        return eye_region


def process_affectnet_format(
    raw_dir: Path,
    output_dir: Path,
    annotation_file: Optional[Path] = None,
    max_samples: Optional[int] = None,
    use_sa2va: bool = False,
    sa2va_model: str = "OMG-Research/Sa2VA-4B",
) -> pd.DataFrame:
    """
    处理 AffectNet 格式的数据集
    假设目录结构：
    raw_dir/
        images/
            *.jpg
        annotations.csv (可选)
    
    Args:
        use_sa2va: 是否使用 Sa2VA 模型提取眼部区域
        sa2va_model: Sa2VA 模型路径
    """
    # 选择提取器
    if use_sa2va and SA2VA_AVAILABLE:
        print("使用 Sa2VA 模型提取眼部区域...")
        extractor = create_sa2va_extractor(model_path=sa2va_model, image_size=224)
        if extractor is None:
            print("警告: Sa2VA 提取器创建失败，改用 MediaPipe")
            extractor = EyeExtractor(image_size=224)
    else:
        if use_sa2va:
            print("警告: Sa2VA 不可用，使用 MediaPipe")
        extractor = EyeExtractor(image_size=224)
    images_dir = raw_dir / "images"
    if not images_dir.exists():
        images_dir = raw_dir  # 如果没有 images 子目录，直接使用 raw_dir

    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    if max_samples:
        image_files = image_files[:max_samples]

    records = []
    output_images_dir = output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    # 加载标注文件（如果存在）
    annotations = {}
    if annotation_file and annotation_file.exists():
        df_ann = pd.read_csv(annotation_file)
        for _, row in df_ann.iterrows():
            img_name = Path(row.get("subDirectory_filePath", "")).name
            annotations[img_name] = {
                "emotion": row.get("expression", None),
                "valence": row.get("valence", None),
                "arousal": row.get("arousal", None),
            }

    print(f"正在处理 {len(image_files)} 张图像...")
    for img_path in tqdm(image_files):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        eye_region = extractor.extract_eye_region(image, eye_side="both")
        if eye_region is None:
            continue

        # 保存处理后的眼部图像
        output_img_name = f"{img_path.stem}_eye.jpg"
        output_img_path = output_images_dir / output_img_name
        cv2.imwrite(str(output_img_path), eye_region)

        # 获取标注信息
        img_name = img_path.name
        ann = annotations.get(img_name, {})
        
        records.append({
            "image_path": f"images/{output_img_name}",
            "emotion_label": ann.get("emotion", None),
            "valence": ann.get("valence", None),
            "arousal": ann.get("arousal", None),
            "iq_proxy": None,
            "eq_proxy": None,
        })

    return pd.DataFrame(records)


def process_custom_format(
    raw_dir: Path,
    output_dir: Path,
    image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    max_samples: Optional[int] = None,
    use_sa2va: bool = False,
    sa2va_model: str = "OMG-Research/Sa2VA-4B",
) -> pd.DataFrame:
    """
    处理自定义格式的数据集（仅图像文件，无标注）
    
    Args:
        use_sa2va: 是否使用 Sa2VA 模型提取眼部区域
        sa2va_model: Sa2VA 模型路径
    """
    # 选择提取器
    if use_sa2va and SA2VA_AVAILABLE:
        print("使用 Sa2VA 模型提取眼部区域...")
        extractor = create_sa2va_extractor(model_path=sa2va_model, image_size=224)
        if extractor is None:
            print("警告: Sa2VA 提取器创建失败，改用 MediaPipe")
            extractor = EyeExtractor(image_size=224)
    else:
        if use_sa2va:
            print("警告: Sa2VA 不可用，使用 MediaPipe")
        extractor = EyeExtractor(image_size=224)
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(raw_dir.glob(f"*{ext}")))
        image_files.extend(list(raw_dir.glob(f"*{ext.upper()}")))

    if max_samples:
        image_files = image_files[:max_samples]

    records = []
    output_images_dir = output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    print(f"正在处理 {len(image_files)} 张图像...")
    for img_path in tqdm(image_files):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        eye_region = extractor.extract_eye_region(image, eye_side="both")
        if eye_region is None:
            continue

        # 保存处理后的眼部图像
        output_img_name = f"{img_path.stem}_eye.jpg"
        output_img_path = output_images_dir / output_img_name
        cv2.imwrite(str(output_img_path), eye_region)

        records.append({
            "image_path": f"images/{output_img_name}",
            "emotion_label": None,
            "valence": None,
            "arousal": None,
            "iq_proxy": None,
            "eq_proxy": None,
        })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="预处理眼部图像数据并生成 metadata.csv")
    parser.add_argument(
        "--raw-dir",
        type=str,
        required=True,
        help="原始数据目录路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="输出目录路径（默认: data/processed）",
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
        default=None,
        help="标注文件路径（CSV 格式，可选）",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["affectnet", "custom"],
        default="custom",
        help="数据集格式（默认: custom）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大处理样本数（用于测试，默认: 全部）",
    )
    parser.add_argument(
        "--use-sa2va",
        action="store_true",
        help="使用 Sa2VA 模型提取眼部区域（需要配置 Sa2VA 模型）",
    )
    parser.add_argument(
        "--sa2va-model",
        type=str,
        default="OMG-Research/Sa2VA-4B",
        help="Sa2VA 模型路径（默认: OMG-Research/Sa2VA-4B）",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raise ValueError(f"原始数据目录不存在: {raw_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotation_file = Path(args.annotation_file) if args.annotation_file else None

    print(f"开始处理数据...")
    print(f"输入目录: {raw_dir}")
    print(f"输出目录: {output_dir}")

    if args.format == "affectnet":
        df = process_affectnet_format(
            raw_dir, output_dir, annotation_file, args.max_samples,
            use_sa2va=args.use_sa2va, sa2va_model=args.sa2va_model
        )
    else:
        df = process_custom_format(
            raw_dir, output_dir, max_samples=args.max_samples,
            use_sa2va=args.use_sa2va, sa2va_model=args.sa2va_model
        )

    # 保存 metadata.csv
    metadata_path = output_dir / "metadata.csv"
    df.to_csv(metadata_path, index=False)
    print(f"\n处理完成！")
    print(f"共处理 {len(df)} 张有效图像")
    print(f"metadata.csv 已保存到: {metadata_path}")
    print(f"处理后的图像保存在: {output_dir / 'images'}")


if __name__ == "__main__":
    main()

