#!/bin/bash
# 从 Windows 路径上传图片到 demo_images 目录

# Windows 路径（用户需要修改）
WINDOWS_PATH="C:/Users/15701/Desktop/新建文件夹"

# Linux 中可能的挂载点
MOUNT_POINTS=(
    "/mnt/c/Users/15701/Desktop/新建文件夹"
    "/c/Users/15701/Desktop/新建文件夹"
    "/media/c/Users/15701/Desktop/新建文件夹"
)

DEMO_DIR="/code/sa2va_wzx/eye_emotion_iq/data/demo_images"

echo "正在查找 Windows 路径..."
echo "Windows 路径: $WINDOWS_PATH"
echo ""

# 尝试找到正确的挂载点
SOURCE_DIR=""
for mount_point in "${MOUNT_POINTS[@]}"; do
    if [ -d "$mount_point" ]; then
        SOURCE_DIR="$mount_point"
        echo "✓ 找到路径: $mount_point"
        break
    fi
done

if [ -z "$SOURCE_DIR" ]; then
    echo "❌ 未找到 Windows 路径"
    echo ""
    echo "请尝试以下方法："
    echo ""
    echo "方法 1: 如果使用 WSL，请确认 Windows 驱动器已挂载"
    echo "   运行: ls /mnt/c/Users/15701/Desktop/"
    echo ""
    echo "方法 2: 如果文件在远程 Windows 机器，请使用 scp 传输："
    echo "   scp -r C:/Users/15701/Desktop/新建文件夹/* user@server:/code/sa2va_wzx/eye_emotion_iq/data/demo_images/"
    echo ""
    echo "方法 3: 如果文件已在 Linux 系统中，请直接指定路径："
    echo "   bash upload_from_windows.sh /actual/path/to/images"
    echo ""
    exit 1
fi

# 如果提供了命令行参数，使用该路径
if [ -n "$1" ]; then
    SOURCE_DIR="$1"
    echo "使用指定路径: $SOURCE_DIR"
fi

# 检查源目录中的图片文件
echo ""
echo "正在查找图片文件..."
images=$(find "$SOURCE_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) 2>/dev/null)

if [ -z "$images" ]; then
    echo "❌ 在 $SOURCE_DIR 中未找到图片文件（jpg, jpeg, png）"
    exit 1
fi

# 统计图片数量
image_count=$(echo "$images" | wc -l)
echo "找到 $image_count 张图片"
echo ""

# 复制图片
copied=0
failed=0

while IFS= read -r img; do
    if [ -f "$img" ]; then
        filename=$(basename "$img")
        if cp "$img" "$DEMO_DIR/$filename" 2>/dev/null; then
            echo "✓ 已复制: $filename"
            ((copied++))
        else
            echo "✗ 复制失败: $filename"
            ((failed++))
        fi
    fi
done <<< "$images"

echo ""
echo "=" | head -c 60
echo ""
echo "完成！"
echo "  成功: $copied 张"
echo "  失败: $failed 张"
echo ""
echo "图片已保存到: $DEMO_DIR"
echo ""
echo "现在可以运行 demo："
echo "  cd /code/sa2va_wzx/eye_emotion_iq"
echo "  python src/demo.py --image data/demo_images/图片名.jpg"

