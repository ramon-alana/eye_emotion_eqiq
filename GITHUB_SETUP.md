# GitHub 发布指南

本指南将帮助你将项目发布到 GitHub。

## 步骤 1: 准备项目

项目已经配置好了必要的文件：
- ✅ `.gitignore` - 忽略不需要版本控制的文件
- ✅ `LICENSE` - MIT 许可证
- ✅ `README.md` - 项目说明文档
- ✅ `CONTRIBUTING.md` - 贡献指南

## 步骤 2: 初始化 Git 仓库

```bash
cd /code/sa2va_wzx/eye_emotion_iq

# 初始化 Git 仓库
git init

# 添加所有文件
git add .

# 创建初始提交
git commit -m "Initial commit: Eye Emotion IQ project"
```

## 步骤 3: 在 GitHub 上创建仓库

1. **登录 GitHub**
   - 访问 https://github.com
   - 登录你的账号

2. **创建新仓库**
   - 点击右上角的 "+" 号
   - 选择 "New repository"
   - 填写仓库信息：
     - **Repository name**: `eye-emotion-iq` (或你喜欢的名字)
     - **Description**: `基于眼部的情绪与 IQ/EQ 评分系统 - Deep learning system for emotion, IQ and EQ assessment from eye images`
     - **Visibility**: 选择 Public（公开）或 Private（私有）
     - **不要**勾选 "Initialize this repository with a README"（我们已经有了）
   - 点击 "Create repository"

## 步骤 4: 连接本地仓库到 GitHub

GitHub 会显示连接命令，类似这样：

```bash
# 添加远程仓库（替换 YOUR_USERNAME 为你的 GitHub 用户名）
git remote add origin https://github.com/YOUR_USERNAME/eye-emotion-iq.git

# 或者使用 SSH（如果你配置了 SSH key）
# git remote add origin git@github.com:YOUR_USERNAME/eye-emotion-iq.git

# 重命名主分支为 main（如果 GitHub 使用 main）
git branch -M main

# 推送代码到 GitHub
git push -u origin main
```

## 步骤 5: 验证上传

1. 刷新 GitHub 仓库页面
2. 你应该能看到所有文件
3. README.md 会自动显示在仓库首页

## 步骤 6: 添加仓库描述和主题

在 GitHub 仓库页面：
1. 点击 "Settings"（设置）
2. 在 "Topics" 中添加标签，例如：
   - `deep-learning`
   - `pytorch`
   - `emotion-recognition`
   - `computer-vision`
   - `eye-tracking`
   - `python`

## 步骤 7: 创建 Release（可选）

如果你想发布一个版本：

```bash
# 创建标签
git tag -a v1.0.0 -m "First release: Eye Emotion IQ v1.0.0"

# 推送标签
git push origin v1.0.0
```

然后在 GitHub 上：
1. 进入 "Releases"
2. 点击 "Create a new release"
3. 选择标签 v1.0.0
4. 填写发布说明
5. 点击 "Publish release"

## 常见问题

### Q: 如何更新代码到 GitHub？

```bash
# 添加更改
git add .

# 提交更改
git commit -m "描述你的更改"

# 推送到 GitHub
git push origin main
```

### Q: 如何忽略大文件？

大文件（如数据集、模型检查点）已经在 `.gitignore` 中被忽略了。如果之前已经提交了这些文件：

```bash
# 从 Git 中移除但保留本地文件
git rm --cached data/raw/*.jpg
git rm --cached checkpoints/*.pt

# 提交更改
git commit -m "Remove large files from git"
git push origin main
```

### Q: 如何添加协作者？

1. 进入仓库的 "Settings"
2. 点击 "Collaborators"
3. 点击 "Add people"
4. 输入协作者的 GitHub 用户名或邮箱

## 下一步

- 📖 完善 README.md（已完成）
- 🏷️ 添加 GitHub Topics
- 📝 创建 Issues 模板
- 🔄 设置 GitHub Actions（CI/CD）
- 📊 添加项目徽章

## 有用的 Git 命令

```bash
# 查看状态
git status

# 查看提交历史
git log --oneline

# 查看远程仓库
git remote -v

# 拉取最新更改
git pull origin main

# 创建新分支
git checkout -b feature/new-feature

# 切换分支
git checkout main
```

祝你发布顺利！🎉

