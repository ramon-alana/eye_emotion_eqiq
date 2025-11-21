"""
统一的眼部提取器接口，支持多种后端（MediaPipe 和 Sa2VA）
"""

from __future__ import annotations

from typing import Optional
from abc import ABC, abstractmethod

import numpy as np

try:
    from .preprocess import EyeExtractor as MediaPipeExtractor
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    from .sa2va_eye_extractor import Sa2VAEyeExtractor, create_sa2va_extractor
    SA2VA_AVAILABLE = True
except ImportError:
    SA2VA_AVAILABLE = False


class BaseEyeExtractor(ABC):
    """眼部提取器基类"""
    
    @abstractmethod
    def extract_eye_region(
        self,
        image: np.ndarray,
        eye_side: str = "both",
    ) -> Optional[np.ndarray]:
        """提取眼部区域"""
        pass


class UnifiedEyeExtractor(BaseEyeExtractor):
    """
    统一的眼部提取器，自动选择最佳后端
    支持 MediaPipe 和 Sa2VA
    """
    
    def __init__(
        self,
        backend: str = "auto",
        image_size: int = 224,
        sa2va_model: str = "OMG-Research/Sa2VA-4B",
        device: str = "cuda",
    ):
        """
        初始化统一提取器
        
        Args:
            backend: 后端选择 ("auto", "mediapipe", "sa2va")
            image_size: 输出图像尺寸
            sa2va_model: Sa2VA 模型路径（如果使用 Sa2VA）
            device: 计算设备
        """
        self.image_size = image_size
        self.backend_name = backend
        self.extractor = None
        
        if backend == "auto":
            # 自动选择：优先使用 Sa2VA，如果不可用则使用 MediaPipe
            if SA2VA_AVAILABLE:
                try:
                    self.extractor = create_sa2va_extractor(
                        model_path=sa2va_model,
                        image_size=image_size,
                        device=device
                    )
                    if self.extractor is not None:
                        self.backend_name = "sa2va"
                        print("使用 Sa2VA 后端提取眼部区域")
                    else:
                        raise ValueError("Sa2VA 提取器创建失败")
                except Exception as e:
                    print(f"Sa2VA 初始化失败: {e}，改用 MediaPipe")
                    self.extractor = None
            
            if self.extractor is None:
                if MEDIAPIPE_AVAILABLE:
                    self.extractor = MediaPipeExtractor(image_size=image_size)
                    self.backend_name = "mediapipe"
                    print("使用 MediaPipe 后端提取眼部区域")
                else:
                    raise ImportError("没有可用的眼部提取后端（MediaPipe 或 Sa2VA）")
        
        elif backend == "sa2va":
            if not SA2VA_AVAILABLE:
                raise ImportError("Sa2VA 不可用")
            self.extractor = create_sa2va_extractor(
                model_path=sa2va_model,
                image_size=image_size,
                device=device
            )
            if self.extractor is None:
                raise ValueError("Sa2VA 提取器创建失败")
            self.backend_name = "sa2va"
            print("使用 Sa2VA 后端提取眼部区域")
        
        elif backend == "mediapipe":
            if not MEDIAPIPE_AVAILABLE:
                raise ImportError("MediaPipe 不可用")
            self.extractor = MediaPipeExtractor(image_size=image_size)
            self.backend_name = "mediapipe"
            print("使用 MediaPipe 后端提取眼部区域")
        
        else:
            raise ValueError(f"不支持的后端: {backend}")
    
    def extract_eye_region(
        self,
        image: np.ndarray,
        eye_side: str = "both",
    ) -> Optional[np.ndarray]:
        """
        提取眼部区域
        
        Args:
            image: 输入图像 (BGR 格式)
            eye_side: "left", "right", 或 "both"（合并双眼）
        
        Returns:
            裁剪并调整大小后的眼部图像，如果提取失败返回 None
        """
        if self.extractor is None:
            raise RuntimeError("提取器未初始化")
        
        return self.extractor.extract_eye_region(image, eye_side=eye_side)
    
    def get_backend(self) -> str:
        """获取当前使用的后端名称"""
        return self.backend_name


def create_eye_extractor(
    backend: str = "auto",
    image_size: int = 224,
    sa2va_model: str = "OMG-Research/Sa2VA-4B",
    device: str = "cuda",
) -> UnifiedEyeExtractor:
    """
    创建统一眼部提取器的便捷函数
    
    Args:
        backend: 后端选择 ("auto", "mediapipe", "sa2va")
        image_size: 输出图像尺寸
        sa2va_model: Sa2VA 模型路径
        device: 计算设备
    
    Returns:
        UnifiedEyeExtractor 实例
    """
    return UnifiedEyeExtractor(
        backend=backend,
        image_size=image_size,
        sa2va_model=sa2va_model,
        device=device
    )


