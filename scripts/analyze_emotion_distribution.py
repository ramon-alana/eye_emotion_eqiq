#!/usr/bin/env python3
"""
分析训练数据中情绪标签的分布，检查类别不平衡问题
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse

# 情绪标签映射
EMOTION_LABELS = {
    0: "愤怒",
    1: "厌恶", 
    2: "恐惧",
    3: "快乐",
    4: "中性",
    5: "悲伤",
    6: "惊讶"
}


def analyze_distribution(metadata_csv: Path):
    """分析情绪标签分布"""
    print(f"\n{'='*60}")
    print(f"分析文件: {metadata_csv}")
    print(f"{'='*60}\n")
    
    if not metadata_csv.exists():
        print(f"❌ 文件不存在: {metadata_csv}")
        return None
    
    df = pd.read_csv(metadata_csv)
    print(f"总样本数: {len(df)}")
    
    if 'emotion_label' not in df.columns:
        print("⚠️  未找到 'emotion_label' 列")
        return None
    
    # 统计情绪分布
    emotion_counts = df['emotion_label'].dropna().astype(int).value_counts().sort_index()
    total_with_labels = emotion_counts.sum()
    
    print(f"\n有情绪标签的样本: {total_with_labels} ({total_with_labels/len(df)*100:.1f}%)")
    print(f"\n{'='*60}")
    print("情绪标签分布:")
    print(f"{'='*60}")
    print(f"{'情绪':<8} {'类别':<6} {'数量':<10} {'百分比':<10} {'状态'}")
    print("-" * 60)
    
    results = {}
    for idx in range(7):
        count = emotion_counts.get(idx, 0)
        percentage = (count / total_with_labels * 100) if total_with_labels > 0 else 0
        emotion_name = EMOTION_LABELS.get(idx, f"类别{idx}")
        
        # 判断是否平衡（理想情况下每个类别应该占14.3%左右）
        ideal_percentage = 100 / 7
        if percentage < ideal_percentage * 0.5:
            status = "⚠️  过少"
        elif percentage > ideal_percentage * 2:
            status = "⚠️  过多"
        else:
            status = "✓ 正常"
        
        results[idx] = {
            'name': emotion_name,
            'count': count,
            'percentage': percentage,
            'status': status
        }
        
        print(f"{emotion_name:<8} {idx:<6} {count:<10} {percentage:>6.2f}%   {status}")
    
    # 分析不平衡程度
    print(f"\n{'='*60}")
    print("类别不平衡分析:")
    print(f"{'='*60}")
    
    if len(emotion_counts) > 0:
        max_count = emotion_counts.max()
        min_count = emotion_counts.min()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"最多样本数: {max_count} ({EMOTION_LABELS[emotion_counts.idxmax()]})")
        print(f"最少样本数: {min_count} ({EMOTION_LABELS[emotion_counts.idxmin()]})")
        print(f"不平衡比例: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 3:
            print("\n⚠️  警告: 数据严重不平衡！建议进行数据平衡处理")
        elif imbalance_ratio > 2:
            print("\n⚠️  注意: 数据存在不平衡，建议进行数据平衡处理")
        else:
            print("\n✓ 数据分布相对平衡")
    
    return results


def suggest_solutions(results: dict):
    """提供解决方案建议"""
    print(f"\n{'='*60}")
    print("解决方案建议:")
    print(f"{'='*60}\n")
    
    # 找出问题类别
    problematic = []
    for idx, data in results.items():
        if "过多" in data['status'] or "过少" in data['status']:
            problematic.append((idx, data))
    
    if not problematic:
        print("✓ 数据分布良好，无需特殊处理")
        return
    
    print("1. 数据增强 (Data Augmentation)")
    print("   - 对样本少的类别进行数据增强")
    print("   - 使用旋转、翻转、颜色调整等方法")
    print("   - 脚本: scripts/augment_imbalanced_data.py\n")
    
    print("2. 类别权重 (Class Weighting)")
    print("   - 在训练时使用加权损失函数")
    print("   - 给样本少的类别更高权重")
    print("   - 修改 train.py 添加 class_weight 参数\n")
    
    print("3. 过采样/欠采样 (Oversampling/Undersampling)")
    print("   - 对少数类进行过采样")
    print("   - 对多数类进行欠采样")
    print("   - 使用 SMOTE 等方法\n")
    
    print("4. 焦点损失 (Focal Loss)")
    print("   - 使用 Focal Loss 替代交叉熵损失")
    print("   - 自动处理类别不平衡问题\n")
    
    print("5. 收集更多数据")
    print("   - 针对不平衡的类别收集更多样本")
    print("   - 使用数据标注工具\n")


def main():
    parser = argparse.ArgumentParser(description="分析情绪标签分布")
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/processed/fer2013/metadata.csv",
        help="metadata.csv文件路径"
    )
    
    args = parser.parse_args()
    
    metadata_path = Path(args.metadata)
    results = analyze_distribution(metadata_path)
    
    if results:
        suggest_solutions(results)
    
    print(f"\n{'='*60}")
    print("下一步操作:")
    print(f"{'='*60}")
    print("1. 运行数据平衡脚本")
    print("2. 使用加权损失函数重新训练")
    print("3. 收集更多不平衡类别的数据")


if __name__ == "__main__":
    main()

