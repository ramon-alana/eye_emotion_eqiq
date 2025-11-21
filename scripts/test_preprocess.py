#!/usr/bin/env python3
"""
快速测试预处理功能
如果没有真实数据集，这个脚本会创建一些测试图像来验证预处理流程
"""

import cv2
import numpy as np
from pathlib import Path

def create_test_images(output_dir: Path, num_images: int = 10):
    """创建一些包含人脸的测试图像（简单的合成图像）"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"正在创建 {num_images} 张测试图像...")
    for i in range(num_images):
        # 创建一个简单的"人脸"图像（白色背景，黑色椭圆代表脸，两个小圆代表眼睛）
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # 绘制"脸"（椭圆）
        cv2.ellipse(img, (320, 240), (200, 250), 0, 0, 360, (200, 180, 160), -1)
        
        # 绘制"眼睛"（两个圆）
        cv2.circle(img, (280, 200), 20, (50, 50, 50), -1)  # 左眼
        cv2.circle(img, (360, 200), 20, (50, 50, 50), -1)  # 右眼
        
        # 绘制"嘴巴"（椭圆）
        cv2.ellipse(img, (320, 300), (80, 40), 0, 0, 180, (100, 50, 50), 3)
        
        # 保存图像
        img_path = output_dir / f"test_face_{i:03d}.jpg"
        cv2.imwrite(str(img_path), img)
    
    print(f"测试图像已保存到: {output_dir}")

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    test_dir = project_root / "data" / "raw" / "test_dataset"
    
    num_images = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    create_test_images(test_dir, num_images)
    
    print("\n现在可以运行预处理脚本：")
    print(f"python -m src.data.preprocess --raw-dir {test_dir} --output-dir data/processed --format custom")


