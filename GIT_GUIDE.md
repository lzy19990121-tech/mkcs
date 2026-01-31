# Git仓库推送指南

## 当前状态

- ✅ 本地Git仓库: `/home/neal/mkcs/.git/`
- ❌ 远程仓库: 未配置

## 推送到GitHub

### 方法1: 通过GitHub网页创建

1. **创建新仓库**
   - 访问 https://github.com/new
   - 仓库名: `mkcs` (或自定义)
   - 描述: `自动交易Agent系统`
   - **不要**勾选 "Initialize with README"
   - 点击 "Create repository"

2. **推送代码**
   ```bash
   cd /home/neal/mkcs
   git remote add origin https://github.com/你的用户名/mkcs.git
   git branch -M main
   git push -u origin main
   ```

### 方法2: 通过GitHub CLI

```bash
# 安装gh CLI (如果未安装)
sudo dnf install gh

# 登录
gh auth login

# 创建仓库并推送
 cd /home/neal/mkcs
gh repo create mkcs --public --source=. --push
```

### 方法3: 推送到GitLab

```bash
 cd /home/neal/mkcs
git remote add origin https://gitlab.com/你的用户名/mkcs.git
git push -u origin main
```

## 本地Git信息

```bash
# 查看提交历史
git log --oneline --all

# 查看当前状态
git status

# 查看配置
git config --list
```

## 当前提交历史

使用以下命令查看：
```bash
git log --oneline --all
```

## 推荐设置

配置用户信息（如果尚未配置）：
```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

## 注意事项

⚠️ **重要**: 推送前请检查是否有敏感信息（如API密钥、密码等）

当前项目包含：
- ✅ 源代码
- ✅ 文档
- ✅ 配置文件（config.yaml）
- ⚠️ 日志文件（logs/）建议添加到.gitignore
