#!/usr/bin/env python3
"""
为每张图片生成详细的个人分析报告
"""

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.demo import load_image, preprocess_eye_image, predict
from src.data.unified_extractor import create_eye_extractor
from src.models.eye_iq_net import EyeIQNet
import cv2
import torch


def generate_report(image_path: Path, checkpoint_path: Path, output_dir: Path, 
                   skip_extraction: bool = False, device: str = "cuda"):
    """为单张图片生成详细报告"""
    
    # 加载图像
    print(f"正在分析: {image_path.name}")
    image = load_image(image_path)
    
    # 提取眼部区域
    if skip_extraction:
        eye_image = image
        if len(eye_image.shape) == 3 and eye_image.shape[2] == 3:
            eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)
            eye_image = cv2.cvtColor(eye_image, cv2.COLOR_RGB2BGR)
    else:
        try:
            extractor = create_eye_extractor(backend="auto", image_size=224, device=device)
            eye_image = extractor.extract_eye_region(image, eye_side="both")
            if eye_image is None:
                print(f"警告: 无法提取眼部区域，使用原图")
                eye_image = image
        except:
            eye_image = image
    
    # 预处理
    image_tensor = preprocess_eye_image(eye_image)
    
    # 加载模型
    device_obj = torch.device(device)
    model = EyeIQNet(num_emotions=7, pretrained=True)
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device_obj)
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
    
    model.to(device_obj)
    
    # 预测
    emotion_labels = ["愤怒", "厌恶", "恐惧", "快乐", "中性", "悲伤", "惊讶"]
    result = predict(model, image_tensor, device_obj, emotion_labels)
    
    # 生成报告
    report = generate_markdown_report(image_path, result, eye_image.shape)
    
    # 保存报告
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{image_path.stem}_report.md"
    report_path.write_text(report, encoding='utf-8')
    
    # 保存JSON格式
    json_path = output_dir / f"{image_path.stem}_report.json"
    json_data = {
        "image_name": image_path.name,
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results": result
    }
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding='utf-8')
    
    print(f"✓ 报告已生成: {report_path}")
    return report_path, json_path


def generate_markdown_report(image_path: Path, result: dict, image_shape: tuple) -> str:
    """生成Markdown格式的报告"""
    
    emotion_data = result['情绪预测']
    dimension_data = result['情绪维度']
    ability_data = result['能力评估']
    raw_data = result['原始输出']
    
    # 情绪概率排序
    emotion_probs = []
    for emotion, prob_str in emotion_data['所有情绪概率'].items():
        prob = float(prob_str.replace('%', ''))
        emotion_probs.append((emotion, prob))
    emotion_probs.sort(key=lambda x: x[1], reverse=True)
    
    # 生成情绪条形图（文本）
    emotion_bars = ""
    for emotion, prob in emotion_probs:
        bar_length = int(prob / 2)  # 每2%一个字符
        bar = "█" * bar_length + "░" * (50 - bar_length)
        emotion_bars += f"{emotion:6s} │{bar}│ {prob:5.2f}%\n"
    
    # 能力评估解读
    iq_score = float(ability_data['IQ 代理分数'].split('/')[0])
    eq_score = float(ability_data['EQ 代理分数'].split('/')[0])
    
    iq_level = get_score_level(iq_score)
    eq_level = get_score_level(eq_score)
    
    # 情绪维度解读
    valence = float(dimension_data['效价 (Valence)'])
    arousal = float(dimension_data['唤醒度 (Arousal)'])
    
    valence_desc = "积极" if valence > 0.3 else "中性" if valence > -0.3 else "消极"
    arousal_desc = "高唤醒" if arousal > 0.3 else "中等唤醒" if arousal > -0.3 else "低唤醒"
    
    report = f"""# 眼部情绪与能力分析报告

**图片名称**: {image_path.name}  
**分析时间**: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}  
**图片尺寸**: {image_shape[1]} × {image_shape[0]} 像素

---

## 📊 情绪分析

### 主要情绪识别

**检测到的情绪**: **{emotion_data['主要情绪']}**  
**置信度**: {emotion_data['置信度']}

### 详细情绪分布

```
{emotion_bars}
```

### 情绪维度分析

- **效价 (Valence)**: {dimension_data['效价 (Valence)']}
  - 解读: {valence_desc}（范围: -1 到 1，正值表示积极情绪）
  
- **唤醒度 (Arousal)**: {dimension_data['唤醒度 (Arousal)']}
  - 解读: {arousal_desc}（范围: -1 到 1，正值表示高唤醒状态）

### 情绪特征总结

根据分析结果，该眼部图像表现出以下情绪特征：
- 主要情绪为 **{emotion_data['主要情绪']}**，置信度达到 {emotion_data['置信度']}
- 情绪效价偏向 **{valence_desc}**，唤醒度处于 **{arousal_desc}** 水平
- 情绪分布较为{'集中' if float(emotion_data['置信度'].replace('%', '')) > 80 else '分散'}，主要情绪占主导地位

---

## 🧠 能力评估

### IQ 代理分数

**分数**: {ability_data['IQ 代理分数']}  
**等级**: {iq_level['level']} ({iq_level['desc']})

### EQ 代理分数

**分数**: {ability_data['EQ 代理分数']}  
**等级**: {eq_level['level']} ({eq_level['desc']})

### 能力评估说明

- **IQ 代理分数**: 基于眼部特征的认知能力代理指标
- **EQ 代理分数**: 基于眼部特征的情绪智力代理指标
- 分数范围: 0-100分
- ⚠️ **重要提示**: 这些分数仅基于眼部图像特征，未经科学验证，仅供参考，不应作为实际能力评估的依据

---

## 📈 数据详情

### 原始输出值

- 效价 (Valence): {raw_data['valence']:.4f}
- 唤醒度 (Arousal): {raw_data['arousal']:.4f}
- IQ 代理原始值: {raw_data['iq_proxy']:.4f}
- EQ 代理原始值: {raw_data['eq_proxy']:.4f}

---

## 📝 分析说明

本报告基于深度学习模型对眼部图像的分析结果生成。分析内容包括：

1. **情绪识别**: 识别7种基本情绪（愤怒、厌恶、恐惧、快乐、中性、悲伤、惊讶）
2. **情绪维度**: 评估情绪的效价（积极/消极）和唤醒度（高/低）
3. **能力评估**: 提供IQ和EQ的代理分数（仅供参考）

**免责声明**: 
- 本分析结果仅基于眼部图像特征，不构成医学或心理学诊断
- IQ/EQ分数为代理指标，未经科学验证，仅供参考
- 实际能力评估应通过专业的标准化测试进行

---

*报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    
    return report


def get_score_level(score: float) -> dict:
    """根据分数返回等级描述"""
    if score >= 80:
        return {"level": "优秀", "desc": "表现优异"}
    elif score >= 70:
        return {"level": "良好", "desc": "表现良好"}
    elif score >= 60:
        return {"level": "中等", "desc": "表现中等"}
    elif score >= 50:
        return {"level": "一般", "desc": "表现一般"}
    else:
        return {"level": "待提升", "desc": "有提升空间"}


def main():
    parser = argparse.ArgumentParser(description="生成图片分析报告")
    parser.add_argument("--images", type=str, nargs="+", help="图片路径列表")
    parser.add_argument("--image-dir", type=str, help="图片目录（批量处理）")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/epoch_30.pt", help="模型检查点路径")
    parser.add_argument("--output-dir", type=str, default="data/reports", help="报告输出目录")
    parser.add_argument("--skip-extraction", action="store_true", help="跳过眼部提取")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    
    # 收集图片列表
    image_paths = []
    if args.images:
        image_paths = [Path(p) for p in args.images]
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    else:
        # 默认使用demo_images目录
        image_dir = project_root / "data" / "demo_images"
        image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    if not image_paths:
        print("错误: 未找到图片文件")
        return
    
    print(f"找到 {len(image_paths)} 张图片，开始生成报告...\n")
    
    # 生成报告
    reports = []
    for img_path in image_paths:
        try:
            report_path, json_path = generate_report(
                img_path, checkpoint_path, output_dir, 
                args.skip_extraction, args.device
            )
            reports.append((img_path.name, report_path, json_path))
        except Exception as e:
            print(f"✗ 处理 {img_path.name} 时出错: {e}")
    
    # 生成汇总报告
    if reports:
        summary_path = output_dir / "summary.md"
        generate_summary(reports, summary_path)
        print(f"\n✓ 汇总报告已生成: {summary_path}")
    
    print(f"\n完成！共生成 {len(reports)} 份报告")


def generate_summary(reports: list, output_path: Path):
    """生成汇总报告"""
    summary = f"""# 批量分析汇总报告

**生成时间**: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}  
**分析图片数量**: {len(reports)}

---

## 📋 报告列表

"""
    
    for i, (img_name, report_path, json_path) in enumerate(reports, 1):
        summary += f"{i}. **{img_name}**\n"
        summary += f"   - Markdown报告: `{report_path.name}`\n"
        summary += f"   - JSON数据: `{json_path.name}`\n\n"
    
    summary += "---\n\n*所有报告保存在同一目录下*\n"
    
    output_path.write_text(summary, encoding='utf-8')


if __name__ == "__main__":
    main()

