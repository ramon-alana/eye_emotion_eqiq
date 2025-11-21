#!/bin/bash
# 准备GitHub上传的脚本

set -e

cd "$(dirname "$0")/.."

echo "=========================================="
echo "准备GitHub上传"
echo "=========================================="

# 1. 检查git状态
echo ""
echo "1. 检查Git状态..."
git status --short

# 2. 添加所有更改
echo ""
echo "2. 添加文件到暂存区..."
git add .gitignore
git add src/demo.py
git add docs/
git add scripts/
git add data/demo_images/.gitkeep
git add data/reports/.gitkeep
git add data/demo_images/README.md

# 3. 显示将要提交的文件
echo ""
echo "3. 将要提交的文件:"
git status --short

# 4. 提示用户
echo ""
echo "=========================================="
echo "准备完成！"
echo "=========================================="
echo ""
echo "下一步操作："
echo ""
echo "1. 检查远程仓库URL（如果需要修改）:"
echo "   git remote set-url origin https://github.com/YOUR_USERNAME/eye-emotion-iq.git"
echo ""
echo "2. 提交更改:"
echo "   git commit -m 'Add EQ/IQ label generation and training scripts'"
echo ""
echo "3. 推送到GitHub:"
echo "   git push origin main"
echo ""
echo "或者使用以下命令一次性完成:"
echo "   git commit -m 'Add EQ/IQ label generation, training scripts, and documentation' && git push origin main"
echo ""


