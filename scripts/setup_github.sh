#!/bin/bash
# 设置GitHub远程仓库的脚本

echo "=========================================="
echo "GitHub 远程仓库设置"
echo "=========================================="
echo ""

# 显示当前远程仓库
echo "当前远程仓库配置:"
git remote -v
echo ""

# 提示用户输入
read -p "请输入你的GitHub用户名: " GITHUB_USERNAME
read -p "请输入仓库名称 (默认: eye-emotion-iq): " REPO_NAME
REPO_NAME=${REPO_NAME:-eye-emotion-iq}

# 设置远程仓库URL
echo ""
echo "设置远程仓库URL..."
git remote set-url origin "https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"

echo ""
echo "✓ 远程仓库已更新为:"
git remote -v
echo ""

# 测试连接
echo "测试GitHub连接..."
if git ls-remote --heads origin main &>/dev/null; then
    echo "✓ 连接成功！仓库存在。"
    echo ""
    echo "现在可以推送代码:"
    echo "  git push origin main"
else
    echo "⚠️  警告: 无法连接到仓库"
    echo ""
    echo "可能的原因:"
    echo "1. 仓库不存在，请先在GitHub上创建仓库: https://github.com/new"
    echo "2. 网络连接问题"
    echo "3. 需要配置GitHub认证（SSH密钥或Personal Access Token）"
    echo ""
    echo "如果使用SSH，可以改用:"
    echo "  git remote set-url origin git@github.com:${GITHUB_USERNAME}/${REPO_NAME}.git"
fi


