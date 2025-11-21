#!/bin/bash
# 启动HTTP上传服务器，方便从Windows上传图片

cd /code/sa2va_wzx/eye_emotion_iq

echo "=========================================="
echo "启动图片上传服务器"
echo "=========================================="
echo ""
echo "服务器启动后，在浏览器中访问显示的地址"
echo "然后可以拖拽图片上传"
echo ""
echo "按 Ctrl+C 停止服务器"
echo ""

python scripts/simple_upload_server.py


