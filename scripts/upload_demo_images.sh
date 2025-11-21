#!/bin/bash
# 上传 demo 图片的辅助脚本

DEMO_DIR="/code/sa2va_wzx/eye_emotion_iq/data/demo_images"

echo "Demo 图片上传辅助脚本"
echo "======================"
echo ""
echo "请将图片文件路径粘贴到下方（支持多个文件，用空格分隔）："
echo "或者直接按 Enter 查看当前目录的图片文件"
echo ""

read -p "图片路径: " image_paths

if [ -z "$image_paths" ]; then
    echo "当前 demo_images 目录中的图片："
    ls -lh "$DEMO_DIR"/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null || echo "目录为空"
    exit 0
fi

for img_path in $image_paths; do
    if [ -f "$img_path" ]; then
        filename=$(basename "$img_path")
        cp "$img_path" "$DEMO_DIR/$filename"
        echo "✓ 已复制: $filename -> $DEMO_DIR/"
    else
        echo "✗ 文件不存在: $img_path"
    fi
done

echo ""
echo "完成！当前 demo_images 目录内容："
ls -lh "$DEMO_DIR"

