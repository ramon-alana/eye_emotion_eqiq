#!/usr/bin/env python3
"""
处理 FER2013 格式的数据集
将按情绪分类的图像整理并生成 metadata.csv
"""

import pandas as pd
from pathlib import Path
import shutil
import sys

# 情绪标签映射
EMOTION_MAP = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprise': 6,
}

EMOTION_NAMES = ["愤怒", "厌恶", "恐惧", "快乐", "中性", "悲伤", "惊讶"]

def prepare_fer2013_format(
    dataset_dir: Path,
    output_dir: Path,
    use_train: bool = True,
    use_test: bool = True,
):
    """
    处理 FER2013 格式的数据集
    
    Args:
        dataset_dir: 数据集根目录（包含 train/ 和 test/ 目录）
        output_dir: 输出目录
        use_train: 是否使用训练集
        use_test: 是否使用测试集
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建统一的图像目录
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    
    # 处理训练集
    if use_train and (dataset_dir / "train").exists():
        print("正在处理训练集...")
        train_dir = dataset_dir / "train"
        for emotion_folder in train_dir.iterdir():
            if not emotion_folder.is_dir():
                continue
            
            emotion_name = emotion_folder.name.lower()
            if emotion_name not in EMOTION_MAP:
                print(f"警告: 未知的情绪文件夹: {emotion_folder.name}")
                continue
            
            emotion_label = EMOTION_MAP[emotion_name]
            print(f"  处理 {emotion_name} ({EMOTION_NAMES[emotion_label]})...")
            
            for img_file in emotion_folder.glob("*.jpg"):
                # 复制图像到统一目录
                new_name = f"train_{emotion_label}_{img_file.stem}.jpg"
                dest_path = images_dir / new_name
                shutil.copy2(img_file, dest_path)
                
                records.append({
                    "image_path": f"images/{new_name}",
                    "emotion_label": emotion_label,
                    "valence": None,
                    "arousal": None,
                    "iq_proxy": None,
                    "eq_proxy": None,
                })
    
    # 处理测试集
    if use_test and (dataset_dir / "test").exists():
        print("正在处理测试集...")
        test_dir = dataset_dir / "test"
        for emotion_folder in test_dir.iterdir():
            if not emotion_folder.is_dir():
                continue
            
            emotion_name = emotion_folder.name.lower()
            if emotion_name not in EMOTION_MAP:
                print(f"警告: 未知的情绪文件夹: {emotion_folder.name}")
                continue
            
            emotion_label = EMOTION_MAP[emotion_name]
            print(f"  处理 {emotion_name} ({EMOTION_NAMES[emotion_label]})...")
            
            for img_file in emotion_folder.glob("*.jpg"):
                # 复制图像到统一目录
                new_name = f"test_{emotion_label}_{img_file.stem}.jpg"
                dest_path = images_dir / new_name
                shutil.copy2(img_file, dest_path)
                
                records.append({
                    "image_path": f"images/{new_name}",
                    "emotion_label": emotion_label,
                    "valence": None,
                    "arousal": None,
                    "iq_proxy": None,
                    "eq_proxy": None,
                })
    
    # 保存 metadata.csv
    df = pd.DataFrame(records)
    metadata_path = output_dir / "metadata.csv"
    df.to_csv(metadata_path, index=False)
    
    print(f"\n处理完成！")
    print(f"总图像数: {len(df)}")
    print(f"图像目录: {images_dir}")
    print(f"元数据文件: {metadata_path}")
    print(f"\n情绪分布:")
    emotion_counts = df['emotion_label'].value_counts().sort_index()
    for label, count in emotion_counts.items():
        if label < len(EMOTION_NAMES):
            print(f"  {EMOTION_NAMES[label]}: {count}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python prepare_fer2013_format.py <数据集目录> [输出目录]")
        print("")
        print("示例:")
        print("  python prepare_fer2013_format.py data/raw/dataset data/raw/fer2013_processed")
        sys.exit(1)
    
    dataset_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else dataset_dir.parent / "fer2013_processed"
    
    if not dataset_dir.exists():
        print(f"错误: 数据集目录不存在: {dataset_dir}")
        sys.exit(1)
    
    prepare_fer2013_format(dataset_dir, output_dir)

