# GitHub Actions 自动化部署指南

## 一、创建 GitHub 仓库并推送代码

```bash
cd /path/to/stock          # 进入你的项目目录

git init
git add .
git commit -m "Initial commit: A股量化分析系统"

# 在 GitHub 网页上创建一个新仓库 (建议 Private)
# 然后:
git remote add origin git@github.com:你的用户名/仓库名.git
git branch -M main
git push -u origin main
```

> 注意: `.gitignore` 已经配好，`data/stocks.db` 和 JSON 文件不会被提交到 git。

## 二、上传初始数据库

第一次需要手动上传你本地的数据库，后续 Action 会自动维护。

```bash
# 压缩数据库
gzip -9 -k data/stocks.db

# 用 GitHub CLI 创建初始 Release
gh release create db-latest \
  --title "DB Snapshot (初始)" \
  --notes "手动上传的初始数据库" \
  --latest=false \
  data/stocks.db.gz

# 清理
rm data/stocks.db.gz
```

如果没装 `gh` CLI，也可以在 GitHub 网页上操作:
1. 进入仓库 → Releases → Create a new release
2. Tag 填 `db-latest`
3. 把 `stocks.db.gz` 拖到附件区域
4. 发布

## 三、确认 Action 权限

进入仓库 → Settings → Actions → General → Workflow permissions:
- 选择 **Read and write permissions**
- 勾选 "Allow GitHub Actions to create and approve pull requests"
- 保存

这一步是必须的，否则 Action 无法上传 Release。

## 四、运行方式

### 自动运行
每个交易日北京时间 16:00 自动触发。内置了交易日判断（含节假日），非交易日会自动跳过。

### 手动触发
进入仓库 → Actions → "A股每日数据更新" → Run workflow

可以调整参数:
- `batch_size`: 每轮拉取股票数（默认 50）
- `interval`: 每只间隔秒数（默认 3）
- `skip_moneyflow`: 勾选则只跑 K线+基本面，不跑资金流向

## 五、数据库同步到本地

Action 跑完后，数据库会上传到 Release `db-latest`。想在本地用最新数据：

```bash
# 下载最新数据库
gh release download db-latest --pattern "stocks.db.gz" --dir data/ --clobber
gunzip -f data/stocks.db.gz
```

## 六、费用说明

- GitHub Actions 对 **公开仓库完全免费**，不限分钟数
- **私有仓库** 每月 2000 分钟免费额度（Linux runner）
- 本 workflow 每次运行约 2-4 小时，每月约 60-120 小时
- 私有仓库可能不够用，建议设为公开仓库，或者购买额外分钟数
- GitHub Release 存储免费，单文件限制 2GB

## 七、常见问题

**Q: 节假日补班日（周六交易）能识别吗？**
A: 能。`_http_utils.py` 里维护了 2025-2027 年的补班交易日历表。

**Q: 跑到一半超时了怎么办？**
A: 脚本支持断点续传，下次运行会自动从上次停的地方继续。timeout 设的 6 小时。

**Q: 数据库越来越大怎么办？**
A: Release 每次会覆盖 `db-latest`，不会累积。数据库本身随时间增长是正常的。

**Q: 怎么查看运行日志？**
A: 仓库 → Actions → 点击具体的运行记录 → 展开各 step 查看输出。

**Q: 需要更新节假日日历怎么办？**
A: 每年底证监会公布下一年休市安排后，更新 `_http_utils.py` 中的 `_CN_HOLIDAYS` 和 `_CN_EXTRA_TRADE_DAYS`。
