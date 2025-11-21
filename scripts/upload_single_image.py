#!/usr/bin/env python3
"""
上传单张图片到demo_images目录
支持从base64、URL或本地路径上传
"""

import argparse
import base64
import shutil
from pathlib import Path
from urllib.request import urlretrieve
from urllib.parse import urlparse

DEMO_DIR = Path(__file__).parent.parent / "data" / "demo_images"
DEMO_DIR.mkdir(parents=True, exist_ok=True)


def upload_from_path(source_path: str, target_name: str = None) -> Path:
    """从本地路径复制图片"""
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"源文件不存在: {source_path}")
    
    if target_name is None:
        target_name = source.name
    
    target = DEMO_DIR / target_name
    shutil.copy2(source, target)
    print(f"✓ 已复制: {source.name} -> {target}")
    return target


def upload_from_url(url: str, target_name: str = None) -> Path:
    """从 URL 下载图片"""
    if target_name is None:
        parsed = urlparse(url)
        target_name = Path(parsed.path).name or "downloaded_image.jpg"
    
    target = DEMO_DIR / target_name
    print(f"正在从 URL 下载: {url}")
    urlretrieve(url, target)
    print(f"✓ 已下载: {target_name} -> {target}")
    return target


def upload_from_base64(base64_str: str, target_name: str, file_format: str = "jpg") -> Path:
    """从 base64 编码保存图片"""
    target = DEMO_DIR / f"{target_name}.{file_format}"
    
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    
    image_data = base64.b64decode(base64_str)
    target.write_bytes(image_data)
    print(f"✓ 已保存: {target_name}.{file_format} -> {target}")
    return target


def main():
    parser = argparse.ArgumentParser(description="上传图片到 demo_images 目录")
    parser.add_argument("--path", type=str, help="本地图片路径")
    parser.add_argument("--url", type=str, help="图片 URL")
    parser.add_argument("--base64", type=str, help="base64 编码的图片数据")
    parser.add_argument("--name", type=str, help="目标文件名（可选）")
    parser.add_argument("--format", type=str, default="jpg", help="文件格式（用于 base64，默认: jpg）")
    
    args = parser.parse_args()
    
    if args.path:
        upload_from_path(args.path, args.name)
    elif args.url:
        upload_from_url(args.url, args.name)
    elif args.base64:
        if not args.name:
            args.name = "uploaded_image"
        upload_from_base64(args.base64, args.name, args.format)
    else:
        print("请指定 --path, --url 或 --base64")
        return
    
    print(f"\n图片已保存到: {DEMO_DIR}")
    print(f"可以使用以下命令运行demo:")
    print(f"  python src/demo.py --image data/demo_images/{args.name or 'image.jpg'} --checkpoint checkpoints/epoch_30.pt --skip-extraction")


if __name__ == "__main__":
    main()


