#!/usr/bin/env python3
"""检查训练数据中EQ和IQ标签的情况"""

import pandas as pd
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
metadata_path = project_root / "data" / "processed" / "fer2013" / "metadata.csv"

if not metadata_path.exists():
    print(f"错误: 找不到metadata文件: {metadata_path}")
    sys.exit(1)

df = pd.read_csv(metadata_path)

print("=" * 60)
print("训练数据标签检查报告")
print("=" * 60)
print(f"\n数据文件: {metadata_path}")
print(f"总样本数: {len(df)}")
print(f"\n列名: {df.columns.tolist()}")

print("\n" + "-" * 60)
print("各标签的缺失情况:")
print("-" * 60)

for col in ['emotion_label', 'valence', 'arousal', 'iq_proxy', 'eq_proxy']:
    if col in df.columns:
        non_null = df[col].notna().sum()
        null_count = df[col].isna().sum()
        print(f"{col:15s}: 有值={non_null:6d} ({non_null/len(df)*100:5.1f}%), 缺失={null_count:6d} ({null_count/len(df)*100:5.1f}%)")
        if non_null > 0:
            print(f"               范围: [{df[col].min():.3f}, {df[col].max():.3f}], 均值: {df[col].mean():.3f}")
    else:
        print(f"{col:15s}: 列不存在")

print("\n" + "-" * 60)
print("结论:")
print("-" * 60)

if 'eq_proxy' in df.columns:
    eq_count = df['eq_proxy'].notna().sum()
    if eq_count == 0:
        print("❌ EQ标签: 数据集中没有EQ标签，所有eq_proxy值都是空的")
        print("   这意味着模型在训练时没有使用真实的EQ标签进行监督学习")
    else:
        print(f"✓ EQ标签: 有 {eq_count} 个样本包含EQ标签")

if 'iq_proxy' in df.columns:
    iq_count = df['iq_proxy'].notna().sum()
    if iq_count == 0:
        print("❌ IQ标签: 数据集中没有IQ标签，所有iq_proxy值都是空的")
        print("   这意味着模型在训练时没有使用真实的IQ标签进行监督学习")
    else:
        print(f"✓ IQ标签: 有 {iq_count} 个样本包含IQ标签")

print("\n" + "=" * 60)

