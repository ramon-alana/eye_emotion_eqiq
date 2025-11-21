#!/usr/bin/env python3
"""
为现有数据集生成EQ/IQ代理标签
基于情绪识别、效价、唤醒度等特征来推断EQ/IQ分数

EQ (情绪智力) 代理指标:
- 基于情绪识别能力（能识别更多情绪 = 高EQ）
- 基于情绪调节能力（效价和唤醒度的平衡）
- 基于情绪表达的丰富度

IQ (认知能力) 代理指标:
- 基于眼部特征的复杂度
- 基于注意力集中度（眼部清晰度）
- 基于视觉处理能力（眼部对称性等）
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import cv2
from tqdm import tqdm


def calculate_eq_proxy(
    emotion_label: Optional[int],
    valence: Optional[float],
    arousal: Optional[float],
    emotion_probs: Optional[dict] = None,
    image_path: Optional[Path] = None,
) -> float:
    """
    计算EQ代理分数 (0-1范围，后续会映射到0-100)
    
    基于以下假设:
    1. 情绪识别能力: 能识别多种情绪的人通常EQ较高
    2. 情绪调节能力: 效价和唤醒度的平衡反映情绪调节能力
    3. 情绪表达: 情绪表达的丰富度
    4. 图像特征: 眼部表情的丰富度和复杂度
    """
    eq_score = 0.5  # 基础分数
    has_emotion_data = False
    
    # 1. 情绪识别能力 (30%)
    if emotion_probs is not None:
        # 情绪分布的熵：能识别更多情绪类型 = 高EQ
        probs = np.array(list(emotion_probs.values()))
        probs = probs[probs > 0.01]  # 过滤极小值
        if len(probs) > 0:
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(len(emotion_probs))
            emotion_diversity = entropy / max_entropy if max_entropy > 0 else 0
            eq_score += emotion_diversity * 0.15
            has_emotion_data = True
    elif emotion_label is not None:
        # 如果有情绪标签，给予基础分数
        eq_score += 0.1
        has_emotion_data = True
    
    # 2. 情绪调节能力 (40%)
    if valence is not None and arousal is not None:
        # 情绪稳定性：效价和唤醒度在合理范围内
        valence_stability = 1.0 - abs(valence)  # 接近0表示中性，更稳定
        arousal_stability = 1.0 - abs(arousal)  # 接近0表示中等唤醒，更稳定
        
        # 情绪平衡：效价和唤醒度的平衡
        balance = 1.0 - abs(valence - arousal) / 2.0
        
        eq_score += (valence_stability * 0.15 + arousal_stability * 0.15 + balance * 0.1)
        has_emotion_data = True
    
    # 3. 积极情绪倾向 (20%)
    if valence is not None:
        # 积极情绪倾向与EQ正相关（适度）
        positive_bias = max(0, valence) * 0.2
        eq_score += positive_bias
        has_emotion_data = True
    
    # 4. 如果没有情绪数据，使用图像特征 (100%)
    if not has_emotion_data and image_path is not None:
        try:
            if image_path.exists():
                img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # 使用图像复杂度作为EQ代理
                    # 更丰富的眼部表情 = 更高的情绪表达能力 = 更高的EQ
                    
                    # 边缘复杂度
                    edges = cv2.Canny(img, 50, 150)
                    edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
                    
                    # 纹理复杂度
                    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
                    texture_score = min(laplacian_var / 500.0, 1.0)
                    
                    # 对比度
                    contrast = img.std() / 50.0
                    contrast_score = min(contrast, 1.0)
                    
                    # 综合评分
                    complexity = (edge_density * 0.4 + texture_score * 0.3 + contrast_score * 0.3)
                    eq_score = 0.4 + complexity * 0.4  # 映射到[0.4, 0.8]范围
        except:
            pass
    
    # 确保在[0, 1]范围内
    eq_score = np.clip(eq_score, 0.0, 1.0)
    
    # 映射到更合理的分布（使用sigmoid的逆函数，使分布更均匀）
    # 将[0,1]映射到大约[-2, 2]范围，这样sigmoid后分布更合理
    eq_raw = (eq_score - 0.5) * 4.0  # 映射到[-2, 2]
    
    return eq_raw


def calculate_iq_proxy(
    image_path: Path,
    emotion_label: Optional[int],
    valence: Optional[float],
    arousal: Optional[float],
) -> float:
    """
    计算IQ代理分数 (0-1范围，后续会映射到0-100)
    
    基于以下假设:
    1. 视觉复杂度: 眼部特征的复杂度
    2. 注意力集中度: 眼部清晰度和对称性
    3. 认知负荷: 情绪状态与认知能力的关系
    """
    iq_score = 0.5  # 基础分数
    
    # 1. 图像质量指标 (40%)
    try:
        if image_path.exists():
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # 清晰度：使用拉普拉斯算子
                laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
                clarity_score = min(laplacian_var / 1000.0, 1.0)  # 归一化
                iq_score += clarity_score * 0.2
                
                # 对比度
                contrast = img.std()
                contrast_score = min(contrast / 50.0, 1.0)  # 归一化
                iq_score += contrast_score * 0.2
    except:
        pass
    
    # 2. 情绪与认知的关系 (30%)
    if emotion_label is not None:
        # 中性情绪通常与更好的认知表现相关
        if emotion_label == 4:  # 中性
            iq_score += 0.15
        elif emotion_label in [3, 6]:  # 快乐、惊讶（积极情绪）
            iq_score += 0.1
        elif emotion_label in [0, 1, 2]:  # 负面情绪
            iq_score -= 0.05
    
    # 3. 情绪稳定性 (30%)
    if valence is not None and arousal is not None:
        # 中等唤醒度与最佳认知表现相关
        optimal_arousal = 0.0  # 中等唤醒
        arousal_score = 1.0 - abs(arousal - optimal_arousal)
        iq_score += arousal_score * 0.15
        
        # 中性效价与认知能力相关
        valence_score = 1.0 - abs(valence)
        iq_score += valence_score * 0.15
    
    # 确保在[0, 1]范围内
    iq_score = np.clip(iq_score, 0.0, 1.0)
    
    # 映射到更合理的分布
    iq_raw = (iq_score - 0.5) * 4.0  # 映射到[-2, 2]
    
    return iq_raw


def generate_labels(
    metadata_csv: Path,
    image_root: Path,
    output_csv: Path,
    use_emotion_probs: bool = False,
) -> None:
    """
    为metadata.csv生成EQ/IQ标签
    """
    print(f"正在读取metadata: {metadata_csv}")
    df = pd.read_csv(metadata_csv)
    
    print(f"原始数据: {len(df)} 条记录")
    print(f"列名: {df.columns.tolist()}")
    
    # 检查必要的列
    if "image_path" not in df.columns:
        raise ValueError("metadata必须包含 'image_path' 列")
    
    # 初始化新列
    if "iq_proxy" not in df.columns:
        df["iq_proxy"] = None
    if "eq_proxy" not in df.columns:
        df["eq_proxy"] = None
    
    # 统计信息
    stats = {
        "total": len(df),
        "with_emotion": 0,
        "with_valence": 0,
        "with_arousal": 0,
        "generated_iq": 0,
        "generated_eq": 0,
    }
    
    print("\n正在生成EQ/IQ标签...")
    
    # 如果有情绪概率列，尝试解析
    emotion_prob_cols = [col for col in df.columns if "emotion_prob" in col.lower() or "prob_" in col.lower()]
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理中"):
        image_path = Path(image_root) / row["image_path"]
        
        # 获取现有标签
        emotion_label = row.get("emotion_label", None)
        if pd.notna(emotion_label):
            emotion_label = int(emotion_label)
            stats["with_emotion"] += 1
        else:
            emotion_label = None
        
        valence = row.get("valence", None)
        if pd.notna(valence):
            valence = float(valence)
            stats["with_valence"] += 1
        else:
            valence = None
        
        arousal = row.get("arousal", None)
        if pd.notna(arousal):
            arousal = float(arousal)
            stats["with_arousal"] += 1
        else:
            arousal = None
        
        # 获取情绪概率（如果有）
        emotion_probs = None
        if use_emotion_probs and emotion_prob_cols:
            emotion_probs = {}
            for col in emotion_prob_cols:
                if pd.notna(row[col]):
                    emotion_name = col.replace("prob_", "").replace("emotion_prob_", "")
                    emotion_probs[emotion_name] = float(row[col])
        
        # 生成IQ标签
        try:
            iq_raw = calculate_iq_proxy(image_path, emotion_label, valence, arousal)
            df.at[idx, "iq_proxy"] = iq_raw
            stats["generated_iq"] += 1
        except Exception as e:
            print(f"\n警告: 生成IQ标签失败 (行{idx}): {e}")
        
        # 生成EQ标签
        try:
            eq_raw = calculate_eq_proxy(emotion_label, valence, arousal, emotion_probs, image_path)
            df.at[idx, "eq_proxy"] = eq_raw
            stats["generated_eq"] += 1
        except Exception as e:
            print(f"\n警告: 生成EQ标签失败 (行{idx}): {e}")
    
    # 保存结果
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("标签生成统计")
    print("=" * 60)
    print(f"总记录数: {stats['total']}")
    print(f"有情绪标签: {stats['with_emotion']} ({stats['with_emotion']/stats['total']*100:.1f}%)")
    print(f"有效价标签: {stats['with_valence']} ({stats['with_valence']/stats['total']*100:.1f}%)")
    print(f"有唤醒度标签: {stats['with_arousal']} ({stats['with_arousal']/stats['total']*100:.1f}%)")
    print(f"生成IQ标签: {stats['generated_iq']} ({stats['generated_iq']/stats['total']*100:.1f}%)")
    print(f"生成EQ标签: {stats['generated_eq']} ({stats['generated_eq']/stats['total']*100:.1f}%)")
    
    # 打印标签分布
    if stats['generated_iq'] > 0:
        iq_values = df["iq_proxy"].dropna()
        print(f"\nIQ标签分布:")
        print(f"  均值: {iq_values.mean():.3f}")
        print(f"  标准差: {iq_values.std():.3f}")
        print(f"  范围: [{iq_values.min():.3f}, {iq_values.max():.3f}]")
    
    if stats['generated_eq'] > 0:
        eq_values = df["eq_proxy"].dropna()
        print(f"\nEQ标签分布:")
        print(f"  均值: {eq_values.mean():.3f}")
        print(f"  标准差: {eq_values.std():.3f}")
        print(f"  范围: [{eq_values.min():.3f}, {eq_values.max():.3f}]")
    
    print(f"\n✓ 已保存到: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="为数据集生成EQ/IQ代理标签")
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="输入metadata.csv路径",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        required=True,
        help="图像根目录",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出metadata.csv路径",
    )
    parser.add_argument(
        "--use-emotion-probs",
        action="store_true",
        help="使用情绪概率列（如果存在）",
    )
    
    args = parser.parse_args()
    
    generate_labels(
        metadata_csv=Path(args.metadata),
        image_root=Path(args.image_root),
        output_csv=Path(args.output),
        use_emotion_probs=args.use_emotion_probs,
    )


if __name__ == "__main__":
    main()

