# Windows用户 - GitHub上传指南

## 📋 你的情况
- ✅ Windows系统
- ❌ 无法直接访问服务器
- ❌ 没有安装Git

## 🎯 解决方案

### 方案1：请管理员帮忙下载文件（最简单）

**联系服务器管理员，请他帮你：**

1. 下载这两个文件到你的电脑：
   - `/tmp/eye_emotion_eqiq_code_only.tar.gz` (51KB)
   - 或者 `/tmp/eye_emotion_eqiq.bundle` (333KB)

2. 可以通过以下方式传输：
   - 通过邮件发送
   - 上传到网盘（百度网盘、OneDrive等）
   - 通过聊天工具发送

---

### 方案2：直接在GitHub网页上创建文件

如果无法获取服务器文件，可以手动在GitHub上创建主要文件：

#### 步骤1：访问GitHub仓库
打开：https://github.com/ramon-alana/eye_emotion_eqiq

#### 步骤2：创建主要文件

**创建README.md**（如果还没有）：
1. 点击 "Add file" → "Create new file"
2. 文件名：`README.md`
3. 复制项目中的README.md内容

**创建主要脚本文件**：
按照以下顺序创建重要文件：

1. `scripts/generate_eq_iq_labels.py`
2. `scripts/generate_reports.py`
3. `scripts/check_training_data.py`
4. `docs/EQ_IQ_TRAINING_ANALYSIS.md`
5. `docs/EQ_IQ_LABEL_GENERATION.md`

---

### 方案3：使用GitHub Desktop（Windows图形界面工具）

1. **下载安装GitHub Desktop**：
   - 访问：https://desktop.github.com/
   - 下载并安装

2. **登录GitHub账号**

3. **克隆仓库**：
   - File → Clone repository
   - 选择 `ramon-alana/eye_emotion_eqiq`

4. **请管理员帮忙**：
   - 让管理员把代码文件打包发给你
   - 解压后复制到GitHub Desktop的仓库文件夹
   - 在GitHub Desktop中提交并推送

---

## 📝 重要文件列表

如果手动创建，这些是最重要的新文件：

### 文档文件
- `docs/EQ_IQ_TRAINING_ANALYSIS.md` - EQ/IQ训练分析
- `docs/EQ_IQ_LABEL_GENERATION.md` - 标签生成指南

### 脚本文件
- `scripts/generate_eq_iq_labels.py` - 生成EQ/IQ标签
- `scripts/generate_reports.py` - 生成分析报告
- `scripts/check_training_data.py` - 检查训练数据
- `scripts/retrain_with_labels.sh` - 重新训练脚本

### 修改的文件
- `src/demo.py` - 更新了警告信息
- `.gitignore` - 更新了忽略规则

---

## 💡 推荐操作

**最佳方案**：联系服务器管理员，请他：
1. 下载 `/tmp/eye_emotion_eqiq_code_only.tar.gz` 文件
2. 通过邮件或其他方式发给你
3. 你解压后在GitHub网页上传

**或者**：如果管理员可以访问GitHub，让他直接在服务器上推送：
```bash
cd /code/sa2va_wzx/eye_emotion_iq
git push -u origin main
```

---

## ❓ 需要帮助？

告诉我：
- 你能联系到服务器管理员吗？
- 你有其他方式访问服务器上的文件吗？

我可以提供更具体的帮助。


