#!/usr/bin/env python3
"""
帮助将图片保存到指定路径的脚本
"""

import shutil
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("使用方法: python save_image.py <源图片路径> [目标路径]")
        print("")
        print("示例:")
        print("  python save_image.py /path/to/image.jpg")
        print("  python save_image.py /path/to/image.jpg data/raw/user_demo_image.jpg")
        sys.exit(1)
    
    source_path = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        dest_path = Path(sys.argv[2])
    else:
        # 默认保存到 data/raw/user_demo_image.jpg
        project_root = Path(__file__).parent.parent
        dest_path = project_root / "data" / "raw" / "user_demo_image.jpg"
    
    if not source_path.exists():
        print(f"错误: 源图片不存在: {source_path}")
        sys.exit(1)
    
    # 创建目标目录
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 复制文件
    shutil.copy2(source_path, dest_path)
    print(f"✓ 图片已保存到: {dest_path}")
    print(f"  文件大小: {dest_path.stat().st_size / 1024:.2f} KB")
    
    return str(dest_path)

if __name__ == "__main__":
    main()


