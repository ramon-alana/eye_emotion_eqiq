# GitHub 推送说明

由于服务器无法直接连接GitHub，请按照以下步骤在本地机器上推送代码。

## 方法1: 使用Bundle文件（推荐）

### 步骤1: 下载Bundle文件

Bundle文件已创建在服务器上：`/tmp/eye_emotion_eqiq.bundle`

下载到本地：
```bash
# 使用scp从服务器下载
scp user@server:/tmp/eye_emotion_eqiq.bundle ./
```

### 步骤2: 在本地克隆并应用Bundle

```bash
# 在本地机器上
cd /path/to/your/workspace

# 克隆空仓库
git clone git@github.com:ramon-alana/eye_emotion_eqiq.git
cd eye_emotion_eqiq

# 从bundle中拉取代码
git pull ../eye_emotion_eqiq.bundle main

# 推送到GitHub
git push -u origin main
```

## 方法2: 使用补丁文件

### 步骤1: 在服务器上创建补丁

```bash
cd /code/sa2va_wzx/eye_emotion_iq
git format-patch -1 HEAD -o /tmp/
```

### 步骤2: 下载补丁并在本地应用

```bash
# 下载补丁
scp user@server:/tmp/0001-*.patch ./

# 在本地仓库
cd eye_emotion_eqiq
git am 0001-*.patch
git push origin main
```

## 方法3: 直接克隆并手动同步

如果本地已有仓库：

```bash
# 在本地
cd eye_emotion_eqiq
git remote add server /path/to/server/repo  # 或使用其他方式访问服务器代码
git fetch server
git merge server/main
git push origin main
```

## 当前提交信息

- **提交ID**: `e11123d`
- **提交信息**: "Add EQ/IQ label generation and training improvements"
- **文件数**: 16个文件更改
- **新增行数**: 1747行
- **删除行数**: 27行

## 包含的新功能

- ✅ EQ/IQ标签生成脚本
- ✅ 训练数据分析脚本
- ✅ 自动报告生成
- ✅ 重新训练脚本
- ✅ 图片上传工具
- ✅ 详细文档

---

*如果遇到问题，请检查：*
1. GitHub仓库是否已创建
2. SSH密钥是否已配置
3. 网络连接是否正常


