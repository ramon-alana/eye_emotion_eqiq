#!/usr/bin/env python3
"""
创建合成数据集用于测试训练流程
生成一些简单的测试图像和标注
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import random

def create_synthetic_face_image(emotion_label: int, output_path: Path):
    """
    创建合成的面部图像（简单的测试用）
    
    Args:
        emotion_label: 情绪标签 (0-6: 愤怒, 厌恶, 恐惧, 快乐, 中性, 悲伤, 惊讶)
        output_path: 输出路径
    """
    # 创建基础图像
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # 绘制"脸"（椭圆）
    cv2.ellipse(img, (320, 240), (200, 250), 0, 0, 360, (200, 180, 160), -1)
    
    # 根据情绪调整眼睛和嘴巴的形状
    eye_y = 200
    mouth_y = 300
    
    if emotion_label == 0:  # 愤怒
        # 眉毛向下，眼睛眯起
        cv2.ellipse(img, (280, 180), (30, 15), 0, 0, 180, (50, 50, 50), 3)
        cv2.ellipse(img, (360, 180), (30, 15), 0, 0, 180, (50, 50, 50), 3)
        cv2.circle(img, (280, eye_y), 15, (50, 50, 50), -1)
        cv2.circle(img, (360, eye_y), 15, (50, 50, 50), -1)
        cv2.ellipse(img, (320, mouth_y), (60, 30), 0, 0, 180, (100, 50, 50), 3)
    
    elif emotion_label == 1:  # 厌恶
        cv2.circle(img, (280, eye_y), 18, (50, 50, 50), -1)
        cv2.circle(img, (360, eye_y), 18, (50, 50, 50), -1)
        cv2.ellipse(img, (320, mouth_y), (50, 20), 0, 180, 360, (100, 50, 50), 3)
    
    elif emotion_label == 2:  # 恐惧
        cv2.circle(img, (280, eye_y), 25, (50, 50, 50), -1)
        cv2.circle(img, (360, eye_y), 25, (50, 50, 50), -1)
        cv2.ellipse(img, (320, mouth_y), (40, 60), 0, 0, 360, (100, 50, 50), 3)
    
    elif emotion_label == 3:  # 快乐
        cv2.ellipse(img, (280, eye_y), (20, 10), 0, 0, 180, (50, 50, 50), 3)
        cv2.ellipse(img, (360, eye_y), (20, 10), 0, 0, 180, (50, 50, 50), 3)
        cv2.ellipse(img, (320, mouth_y), (80, 40), 0, 0, 180, (100, 50, 50), 3)
    
    elif emotion_label == 4:  # 中性
        cv2.circle(img, (280, eye_y), 20, (50, 50, 50), -1)
        cv2.circle(img, (360, eye_y), 20, (50, 50, 50), -1)
        cv2.line(img, (270, mouth_y), (370, mouth_y), (100, 50, 50), 3)
    
    elif emotion_label == 5:  # 悲伤
        cv2.ellipse(img, (280, eye_y), (20, 10), 0, 180, 360, (50, 50, 50), 3)
        cv2.ellipse(img, (360, eye_y), (20, 10), 0, 180, 360, (50, 50, 50), 3)
        cv2.ellipse(img, (320, mouth_y), (80, 40), 0, 0, 180, (100, 50, 50), 3)
    
    elif emotion_label == 6:  # 惊讶
        cv2.circle(img, (280, eye_y), 30, (50, 50, 50), -1)
        cv2.circle(img, (360, eye_y), 30, (50, 50, 50), -1)
        cv2.ellipse(img, (320, mouth_y), (30, 50), 0, 0, 360, (100, 50, 50), 3)
    
    cv2.imwrite(str(output_path), img)

def create_synthetic_dataset(
    output_dir: Path,
    num_samples: int = 100,
    emotions: list = None
):
    """
    创建合成数据集
    
    Args:
        output_dir: 输出目录
        num_samples: 样本数量
        emotions: 情绪标签列表（如果为 None，则随机生成）
    """
    if emotions is None:
        emotions = [random.randint(0, 6) for _ in range(num_samples)]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    emotion_names = ["愤怒", "厌恶", "恐惧", "快乐", "中性", "悲伤", "惊讶"]
    
    print(f"正在创建 {num_samples} 个合成样本...")
    for i, emotion_label in enumerate(emotions):
        img_name = f"synthetic_{i:04d}.jpg"
        img_path = images_dir / img_name
        create_synthetic_face_image(emotion_label, img_path)
        
        # 生成随机的标注值
        records.append({
            "image_path": f"images/{img_name}",
            "emotion_label": emotion_label,
            "valence": random.uniform(-1.0, 1.0),
            "arousal": random.uniform(-1.0, 1.0),
            "iq_proxy": random.uniform(0.0, 1.0),
            "eq_proxy": random.uniform(0.0, 1.0),
        })
    
    # 保存 metadata.csv
    df = pd.DataFrame(records)
    metadata_path = output_dir / "metadata.csv"
    df.to_csv(metadata_path, index=False)
    
    print(f"\n合成数据集创建完成！")
    print(f"图像目录: {images_dir}")
    print(f"元数据文件: {metadata_path}")
    print(f"\n情绪分布:")
    emotion_counts = df['emotion_label'].value_counts().sort_index()
    for label, count in emotion_counts.items():
        print(f"  {emotion_names[label]}: {count}")

if __name__ == "__main__":
    import sys
    
    output_dir = Path("data/processed_synthetic")
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    
    create_synthetic_dataset(output_dir, num_samples)


