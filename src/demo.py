"""
单张图片推理 Demo 脚本
可以直接处理单张图片，提取眼部区域并进行情绪、IQ、EQ 预测
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.unified_extractor import UnifiedEyeExtractor, create_eye_extractor
from src.models.eye_iq_net import EyeIQNet


def load_image(image_path: Path) -> np.ndarray:
    """加载图像"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    return img


def preprocess_eye_image(eye_image: np.ndarray, image_size: int = 224) -> torch.Tensor:
    """预处理眼部图像为模型输入格式"""
    # 转换为 RGB
    if len(eye_image.shape) == 3 and eye_image.shape[2] == 3:
        eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)
    
    # 调整大小
    eye_image = cv2.resize(eye_image, (image_size, image_size))
    
    # 归一化到 [0, 1] 然后标准化
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    tensor = transform(eye_image)
    return tensor.unsqueeze(0)  # 添加 batch 维度


def predict(
    model: EyeIQNet,
    image_tensor: torch.Tensor,
    device: torch.device,
    emotion_labels: list[str] | None = None,
) -> dict:
    """进行预测"""
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # 处理情绪预测
    emotion_probs = torch.softmax(outputs["emotion_logits"], dim=1).cpu().numpy()[0]
    emotion_idx = int(np.argmax(emotion_probs))
    
    if emotion_labels is None:
        emotion_labels = ["愤怒", "厌恶", "恐惧", "快乐", "中性", "悲伤", "惊讶"]
    
    # 归一化 IQ/EQ 到 0-100 范围
    iq_score = float(outputs["iq_proxy"].cpu().item())
    eq_score = float(outputs["eq_proxy"].cpu().item())
    
    # 使用 sigmoid 函数将任意值映射到 [0, 1]，然后缩放到 [0, 100]
    # 这样可以处理超出 [-1, 1] 范围的输出值
    # 对原始分数应用 sigmoid，然后缩放到 0-100
    iq_sigmoid = torch.sigmoid(torch.tensor(iq_score)).item()
    eq_sigmoid = torch.sigmoid(torch.tensor(eq_score)).item()
    
    iq_normalized = iq_sigmoid * 100
    
    # EQ 评分调整：使用线性变换使平均分在60左右
    # 原始方法：eq_normalized = eq_sigmoid * 100 (平均约47)
    # 调整后：使用线性变换 eq_normalized = eq_sigmoid * scale + offset
    # 目标：使平均分在60左右，同时保持分数在合理范围内
    eq_normalized = eq_sigmoid * 128 + 0  # 初步调整
    # 如果平均分仍不够，可以进一步调整偏移量
    # 基于测试数据，当前平均sigmoid约0.468，目标平均分60
    # 计算：0.468 * scale + offset = 60，同时保持合理范围
    # 使用：scale=128, offset=0 时，平均分约60
    # 如果还需要微调，可以调整offset
    
    # 确保结果在 [0, 100] 范围内（双重保险）
    iq_normalized = max(0.0, min(100.0, iq_normalized))
    eq_normalized = max(0.0, min(100.0, eq_normalized))
    
    result = {
        "情绪预测": {
            "主要情绪": emotion_labels[emotion_idx] if emotion_idx < len(emotion_labels) else f"类别{emotion_idx}",
            "置信度": f"{emotion_probs[emotion_idx] * 100:.2f}%",
            "所有情绪概率": {
                label: f"{prob * 100:.2f}%" 
                for label, prob in zip(emotion_labels[:len(emotion_probs)], emotion_probs)
            }
        },
        "情绪维度": {
            "效价 (Valence)": f"{float(outputs['valence'].cpu().item()):.3f}",
            "唤醒度 (Arousal)": f"{float(outputs['arousal'].cpu().item()):.3f}",
        },
        "能力评估": {
            "IQ 代理分数": f"{iq_normalized:.1f}/100",
            "EQ 代理分数": f"{eq_normalized:.1f}/100",
            "警告": "IQ/EQ输出头未使用真实标签训练，输出值无实际意义，仅供参考",
        },
        "原始输出": {
            "valence": float(outputs["valence"].cpu().item()),
            "arousal": float(outputs["arousal"].cpu().item()),
            "iq_proxy": float(outputs["iq_proxy"].cpu().item()),
            "eq_proxy": float(outputs["eq_proxy"].cpu().item()),
        }
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="单张图片情绪与 IQ/EQ 预测 Demo")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="输入图片路径",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="训练好的模型检查点路径（可选，如果不提供则使用未训练的模型）",
    )
    parser.add_argument(
        "--output-eye",
        type=str,
        default=None,
        help="保存提取的眼部区域图像路径（可选）",
    )
    parser.add_argument(
        "--num-emotions",
        type=int,
        default=7,
        help="情绪类别数量（默认: 7）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备（默认: 自动选择）",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["pretty", "json"],
        default="pretty",
        help="输出格式（默认: pretty）",
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
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="跳过眼部提取，直接使用输入图片（如果输入已经是眼部区域）",
    )
    
    args = parser.parse_args()
    
    # 加载图像
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"错误: 图像文件不存在: {image_path}")
        return
    
    print(f"正在加载图像: {image_path}")
    image = load_image(image_path)
    print(f"图像尺寸: {image.shape}")
    
    # 提取眼部区域（或跳过）
    if args.skip_extraction:
        print("跳过眼部提取，直接使用输入图片...")
        eye_image = image
        # 确保是RGB格式
        if len(eye_image.shape) == 3 and eye_image.shape[2] == 3:
            eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)
            eye_image = cv2.cvtColor(eye_image, cv2.COLOR_RGB2BGR)  # 转回BGR用于保存
    else:
        print("正在提取眼部区域...")
        # 使用统一的提取器接口
        backend = "sa2va" if args.use_sa2va else "auto"
        try:
            extractor = create_eye_extractor(
                backend=backend,
                image_size=224,
                sa2va_model=args.sa2va_model,
                device=args.device
            )
            print(f"使用后端: {extractor.get_backend()}")
            eye_image = extractor.extract_eye_region(image, eye_side="both")
        except Exception as e:
            print(f"提取器初始化失败: {e}")
            # 如果失败，尝试使用 MediaPipe
            try:
                extractor = create_eye_extractor(backend="mediapipe", image_size=224)
                print("回退到 MediaPipe 后端")
                eye_image = extractor.extract_eye_region(image, eye_side="both")
            except Exception as e2:
                print(f"所有提取器都失败: {e2}")
                print("提示: 如果输入图片已经是眼部区域，请使用 --skip-extraction 参数")
                return
        
        if eye_image is None:
            print("错误: 无法从图像中检测到人脸或眼部区域")
            print("提示: 请确保图像中包含清晰的人脸，或使用 --skip-extraction 参数直接使用输入图片")
            return
    
    print(f"眼部区域提取成功，尺寸: {eye_image.shape}")
    
    # 保存提取的眼部图像（如果指定）
    if args.output_eye:
        output_path = Path(args.output_eye)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), eye_image)
        print(f"眼部区域已保存到: {output_path}")
    
    # 预处理图像
    image_tensor = preprocess_eye_image(eye_image)
    
    # 加载模型
    print(f"正在加载模型（设备: {args.device}）...")
    device = torch.device(args.device)
    model = EyeIQNet(num_emotions=args.num_emotions, pretrained=True)
    
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"正在加载检查点: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
        print("模型检查点加载成功")
    else:
        print("警告: 未提供训练好的模型，使用预训练 ResNet18 的随机初始化头部")
        print("（预测结果仅供参考，建议先训练模型）")
    
    model.to(device)
    
    # 进行预测
    print("\n正在进行预测...")
    result = predict(model, image_tensor, device)
    
    # 输出结果
    print("\n" + "="*60)
    print("预测结果")
    print("="*60)
    
    if args.format == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # 美化输出
        print(f"\n【情绪预测】")
        print(f"  主要情绪: {result['情绪预测']['主要情绪']}")
        print(f"  置信度: {result['情绪预测']['置信度']}")
        print(f"\n  所有情绪概率:")
        for emotion, prob in result['情绪预测']['所有情绪概率'].items():
            print(f"    - {emotion}: {prob}")
        
        print(f"\n【情绪维度】")
        print(f"  效价 (Valence): {result['情绪维度']['效价 (Valence)']} (范围: -1 到 1，正值表示积极)")
        print(f"  唤醒度 (Arousal): {result['情绪维度']['唤醒度 (Arousal)']} (范围: -1 到 1，正值表示高唤醒)")
        
        print(f"\n【能力评估】")
        print(f"  IQ 代理分数: {result['能力评估']['IQ 代理分数']}")
        print(f"  EQ 代理分数: {result['能力评估']['EQ 代理分数']}")
        
        print(f"\n⚠️  重要警告:")
        print(f"  - IQ/EQ 输出头在训练时没有使用真实标签，输出值来自随机初始化的权重")
        print(f"  - 这些分数没有实际意义，不能代表真实的IQ/EQ水平")
        print(f"  - 仅供参考，不应作为任何能力评估的依据")
    
    print("="*60)


if __name__ == "__main__":
    main()

