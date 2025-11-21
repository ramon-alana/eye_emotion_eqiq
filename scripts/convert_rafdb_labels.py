#!/usr/bin/env python3
"""
转换 RAF-DB 标注格式为 CSV
RAF-DB 的标注格式通常是：
  image_name emotion_label
"""

import pandas as pd
from pathlib import Path
import sys

def convert_rafdb_labels(
    label_file: str,
    output_csv: str,
    image_dir: str = None
):
    """
    转换 RAF-DB 标注文件为 CSV 格式
    
    Args:
        label_file: RAF-DB 标注文件路径（通常是 list_patition_label.txt）
        output_csv: 输出 CSV 文件路径
        image_dir: 图像目录（可选，用于验证图像存在）
    """
    label_path = Path(label_file)
    if not label_path.exists():
        print(f"错误: 标注文件不存在: {label_file}")
        return
    
    records = []
    
    print(f"正在读取标注文件: {label_file}")
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
            
            image_name = parts[0]
            emotion_label = int(parts[1]) - 1  # RAF-DB 使用 1-7，转换为 0-6
            
            # 检查图像是否存在（如果提供了图像目录）
            if image_dir:
                image_path = Path(image_dir) / image_name
                if not image_path.exists():
                    print(f"警告: 图像不存在: {image_path}")
                    continue
            
            records.append({
                "image_path": image_name,
                "emotion_label": emotion_label,
                "valence": None,  # RAF-DB 不提供这些
                "arousal": None,
                "iq_proxy": None,
                "eq_proxy": None,
            })
    
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    
    print(f"\n转换完成！")
    print(f"共 {len(df)} 条记录")
    print(f"输出文件: {output_csv}")
    print(f"\n情绪分布:")
    emotion_counts = df['emotion_label'].value_counts().sort_index()
    emotion_names = ["愤怒", "厌恶", "恐惧", "快乐", "中性", "悲伤", "惊讶"]
    for label, count in emotion_counts.items():
        if label < len(emotion_names):
            print(f"  {emotion_names[label]}: {count}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("使用方法: python convert_rafdb_labels.py <标注文件> <输出CSV> [图像目录]")
        print("")
        print("示例:")
        print("  python convert_rafdb_labels.py \\")
        print("    data/raw/rafdb/EmoLabel/list_patition_label.txt \\")
        print("    data/raw/rafdb/annotations.csv \\")
        print("    data/raw/rafdb/images")
        sys.exit(1)
    
    label_file = sys.argv[1]
    output_csv = sys.argv[2]
    image_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    convert_rafdb_labels(label_file, output_csv, image_dir)


