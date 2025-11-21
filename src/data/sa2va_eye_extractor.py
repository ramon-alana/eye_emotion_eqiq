"""
使用 Sa2VA 模型提取眼部轮廓的接口
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# 添加 vlm 目录到路径
vlm_path = Path(__file__).parent.parent.parent.parent / "vlm"
if str(vlm_path) not in sys.path:
    sys.path.insert(0, str(vlm_path))

try:
    from sa2va_eval.vlmeval.vlm.sa2va_chat import Sa2VAChat
    SA2VA_AVAILABLE = True
except ImportError:
    SA2VA_AVAILABLE = False
    print("警告: 无法导入 Sa2VA，请确保 vlm 目录存在且配置正确")


class Sa2VAEyeExtractor:
    """使用 Sa2VA 模型从面部图像中提取眼部区域"""

    def __init__(
        self,
        model_path: str = "OMG-Research/Sa2VA-4B",
        image_size: int = 224,
        device: str = "cuda",
    ):
        """
        初始化 Sa2VA 眼部提取器
        
        Args:
            model_path: Sa2VA 模型路径
            image_size: 输出图像尺寸
            device: 计算设备
        """
        if not SA2VA_AVAILABLE:
            raise ImportError("Sa2VA 不可用，请检查 vlm 目录配置")
        
        self.image_size = image_size
        self.device = device
        
        print(f"正在加载 Sa2VA 模型: {model_path}")
        try:
            self.model = Sa2VAChat(model_path=model_path, load_in_8bit=False)
            print("Sa2VA 模型加载成功")
        except Exception as e:
            print(f"加载 Sa2VA 模型失败: {e}")
            raise

    def extract_eye_region(
        self,
        image: np.ndarray,
        eye_side: str = "both",
        prompt: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        使用 Sa2VA 提取眼部区域
        
        Args:
            image: 输入图像 (BGR 格式)
            eye_side: "left", "right", 或 "both"（合并双眼）
            prompt: 自定义提示词（可选）
        
        Returns:
            裁剪并调整大小后的眼部图像，如果提取失败返回 None
        """
        # 转换为 RGB PIL 图像
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # 构建提示词，使用 [SEG] 标记来触发分割
        if prompt is None:
            if eye_side == "both":
                prompt = "请分割出图像中的双眼区域，包括左眼和右眼。[SEG]"
            elif eye_side == "left":
                prompt = "请分割出图像中的左眼区域。[SEG]"
            elif eye_side == "right":
                prompt = "请分割出图像中的右眼区域。[SEG]"
            else:
                raise ValueError(f"Invalid eye_side: {eye_side}")
        
        try:
            # 直接调用 predict_forward 获取掩码
            input_dict = {
                'image': pil_image,
                'text': prompt,
                'past_text': '',
                'mask_prompts': None,
                'tokenizer': self.model.tokenizer,
            }
            
            result = self.model.model.predict_forward(**input_dict)
            
            # 获取预测的掩码
            prediction_masks = result.get('prediction_masks', [])
            if not prediction_masks or len(prediction_masks) == 0:
                print("Sa2VA 未返回掩码")
                return None
            
            # 使用第一个掩码（如果有多个，合并它们）
            mask = prediction_masks[0]
            if len(mask.shape) == 3:
                # 如果是多个掩码，合并或选择最大的
                mask = mask[0] if mask.shape[0] == 1 else np.max(mask, axis=0)
            
            # 确保掩码是二值的
            mask = (mask > 0.5).astype(np.uint8) * 255
            
            # 使用掩码提取眼部区域
            # 找到掩码的边界框
            coords = np.column_stack(np.where(mask > 0))
            if len(coords) == 0:
                print("掩码为空")
                return None
            
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # 添加一些边距
            margin = 10
            h, w = image.shape[:2]
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)
            
            # 裁剪眼部区域
            eye_region = image[y_min:y_max, x_min:x_max]
            
            if eye_region.size == 0:
                return None
            
            # 调整大小
            eye_region = cv2.resize(eye_region, (self.image_size, self.image_size))
            
            return eye_region
            
        except Exception as e:
            print(f"Sa2VA 提取失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def extract_eye_mask(
        self,
        image: np.ndarray,
        prompt: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        提取眼部区域的掩码
        
        Args:
            image: 输入图像 (BGR 格式)
            prompt: 自定义提示词（可选）
        
        Returns:
            眼部区域的二值掩码，如果提取失败返回 None
        """
        # 转换为 RGB PIL 图像
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        if prompt is None:
            prompt = "请分割出图像中的眼部区域，包括左眼和右眼。[SEG]"
        
        try:
            input_dict = {
                'image': pil_image,
                'text': prompt,
                'past_text': '',
                'mask_prompts': None,
                'tokenizer': self.model.tokenizer,
            }
            
            result = self.model.model.predict_forward(**input_dict)
            prediction_masks = result.get('prediction_masks', [])
            
            if not prediction_masks or len(prediction_masks) == 0:
                return None
            
            mask = prediction_masks[0]
            if len(mask.shape) == 3:
                mask = mask[0] if mask.shape[0] == 1 else np.max(mask, axis=0)
            
            # 转换为二值掩码
            mask = (mask > 0.5).astype(np.uint8) * 255
            
            return mask
            
        except Exception as e:
            print(f"Sa2VA 掩码提取失败: {e}")
            import traceback
            traceback.print_exc()
            return None


def create_sa2va_extractor(
    model_path: str = "OMG-Research/Sa2VA-4B",
    image_size: int = 224,
    device: str = "cuda",
) -> Optional[Sa2VAEyeExtractor]:
    """
    创建 Sa2VA 眼部提取器的便捷函数
    
    Returns:
        Sa2VAEyeExtractor 实例，如果 Sa2VA 不可用则返回 None
    """
    if not SA2VA_AVAILABLE:
        print("警告: Sa2VA 不可用，无法创建提取器")
        return None
    
    try:
        return Sa2VAEyeExtractor(model_path=model_path, image_size=image_size, device=device)
    except Exception as e:
        print(f"创建 Sa2VA 提取器失败: {e}")
        return None

