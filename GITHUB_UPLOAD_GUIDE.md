# GitHub上传操作指南

## 📍 什么是"本地机器"？

- **服务器（远程）**：代码现在所在的机器 `HGX-033`，无法直接连接GitHub
- **本地机器**：你自己的电脑（Windows/Mac/Linux），可以访问GitHub

## 🚀 最简单的操作方法

### 方案A：使用GitHub网页上传（最简单，推荐）

如果服务器无法连接GitHub，最简单的方法是：

1. **在服务器上打包代码**
2. **下载到你的电脑**
3. **在GitHub网页上上传**

#### 步骤1：在服务器上打包代码

```bash
# 在服务器上执行（已经准备好了）
cd /code/sa2va_wzx/eye_emotion_iq
tar -czf /tmp/eye_emotion_eqiq.tar.gz --exclude='.git' --exclude='checkpoints' --exclude='data/processed/images' .
```

#### 步骤2：下载到你的电脑

**Windows用户**：
- 使用WinSCP、FileZilla等工具
- 或使用PowerShell：`scp root@服务器IP:/tmp/eye_emotion_eqiq.tar.gz .`

**Mac/Linux用户**：
```bash
scp root@服务器IP:/tmp/eye_emotion_eqiq.tar.gz ~/Downloads/
```

#### 步骤3：在GitHub网页上传

1. 访问：https://github.com/ramon-alana/eye_emotion_eqiq
2. 点击 "uploading an existing file"
3. 解压下载的tar.gz文件
4. 拖拽所有文件到GitHub网页
5. 填写提交信息："Add EQ/IQ label generation and training improvements"
6. 点击 "Commit changes"

---

### 方案B：使用Git命令（需要本地有Git）

#### 步骤1：在你的电脑上打开终端/命令行

**Windows**: 打开 PowerShell 或 Git Bash  
**Mac**: 打开 Terminal  
**Linux**: 打开 Terminal

#### 步骤2：下载代码包

```bash
# 替换 服务器IP 为实际IP地址
scp root@服务器IP:/tmp/eye_emotion_eqiq.bundle ~/Downloads/
```

#### 步骤3：克隆GitHub仓库

```bash
cd ~/Downloads
git clone git@github.com:ramon-alana/eye_emotion_eqiq.git
cd eye_emotion_eqiq
```

#### 步骤4：应用代码并推送

```bash
git pull ~/Downloads/eye_emotion_eqiq.bundle main
git push -u origin main
```

---

### 方案C：直接在服务器上操作（如果网络问题解决）

如果服务器可以连接GitHub了，直接在服务器上执行：

```bash
cd /code/sa2va_wzx/eye_emotion_iq
git push -u origin main
```

---

## ❓ 常见问题

**Q: 我不知道服务器IP地址？**  
A: 在服务器上运行 `hostname -I` 或 `ip addr` 查看

**Q: 我没有SSH密钥？**  
A: 使用方案A（网页上传）最简单，不需要SSH

**Q: 我无法访问服务器？**  
A: 联系服务器管理员，或者使用其他方式获取代码

---

## 📝 当前代码状态

- ✅ 代码已提交到本地Git仓库
- ✅ 提交ID: `e11123d`
- ✅ 包含16个新文件/修改
- ⚠️ 需要推送到GitHub


