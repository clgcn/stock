# A 股量化分析系统

本项目是一套以本地 SQLite 为核心的 A 股研究工具。

当前数据流已经固定为：

- `stocks`：只保存股票宇宙元信息，来自本地 JSON 导入
- `stock_fundamentals`：保存每日估值快照，在线来源为腾讯
- `stock_history`：保存日 K 历史，在线来源为东方财富

## 项目结构

```text
stock/
├── slow_fetcher.py         # 数据更新主入口
├── import_json.py          # 从本地 JSON 导入 stocks
├── stock_screener.py       # 选股器
├── stock_tool.py           # 底层行情/财务/K线工具
├── quant_engine.py         # 量化诊断引擎
├── backtest_engine.py      # 回测引擎
├── risk_manager.py         # 风险评估
├── news_analyzer.py        # 市场新闻 / 个股新闻
├── stock_mcp_server.py     # MCP 服务
├── data/
│   └── stocks.db           # SQLite 数据库
└── charts/                 # 图表输出目录
```

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install mcp matplotlib pandas numpy curl_cffi
```

## 当前数据库结构

### `stocks`

只保留股票主数据：

- `code`
- `name`
- `suspended`

### `stock_fundamentals`

只保留腾讯估值快照：

- `code`
- `trade_date`
- `pe_ttm`
- `pb`
- `total_mv`
- `float_mv`
- `updated_at`
- `source`
- `batch_id`

### `stock_history`

保存日 K 历史：

- `code`
- `date`
- `open`
- `close`
- `high`
- `low`
- `volume`
- `amount`
- `amplitude`
- `pct_chg`
- `change`
- `turnover`

## 日常使用流程

### 1. 导入股票名单

先把你手动保存的 JSON 导入到 `stocks`：

```bash
/Users/chenglanguo/stock/.venv/bin/python /Users/chenglanguo/stock/import_json.py
```

说明：

- 只写入 `stocks`
- 会按 `code` 去重
- 重复导入不会产生重复股票

### 2. 盘中更新估值快照

盘中只刷新 `stock_fundamentals`，不动 `stock_history`：

```bash
/Users/chenglanguo/stock/.venv/bin/python /Users/chenglanguo/stock/slow_fetcher.py --intraday-update --batch 200 --interval 10
```

说明：

- 数据源：腾讯 `qt.gtimg.cn`
- 默认按批次刷新，`--batch 200` 表示每批 200 只股票
- `--interval 10` 表示批次间隔 10 秒
- 现在已支持同一天断点续跑
- 如果当天已经完整刷过，会提示无需重复刷新

### 3. 收盘后正式更新

收盘后统一更新：

- `stock_fundamentals`
- `stock_history`

命令：

```bash
/Users/chenglanguo/stock/.venv/bin/python /Users/chenglanguo/stock/slow_fetcher.py --daily-close-update --batch 100 --interval 10
```

说明：

- 先刷新当天 `stock_fundamentals`
- 再把 `stock_history` 增量追到目标交易日
- 已有历史的股票只补缺失日期，不会每次全量重拉

### 4. 指定某个交易日更新

如果你要补某个指定日期：

```bash
/Users/chenglanguo/stock/.venv/bin/python /Users/chenglanguo/stock/slow_fetcher.py --daily-close-update --trade-date 2026-03-23 --batch 100 --interval 10
```

这条命令适合：

- 补跑某个交易日
- 只把所有股票追到那一天

## 只更新 `stock_history`

如果你只想补 K 线历史：

```bash
/Users/chenglanguo/stock/.venv/bin/python /Users/chenglanguo/stock/slow_fetcher.py --history --batch 100 --interval 10 --auto
```

说明：

- 从 `stocks` 表读取未停牌股票
- 按股票逐只增量补日 K
- 不是每次全量重拉，而是从每只股票最后一条记录继续补

## 只补今天这一根日 K

最推荐的做法仍然是用：

```bash
/Users/chenglanguo/stock/.venv/bin/python /Users/chenglanguo/stock/slow_fetcher.py --daily-close-update --trade-date 2026-03-23 --batch 100 --interval 10
```

因为这会把所有股票增量追到指定日期，不会把已有历史整段重拉。

## 查看进度

```bash
/Users/chenglanguo/stock/.venv/bin/python /Users/chenglanguo/stock/slow_fetcher.py --status
```

会显示：

- `stocks` 数量
- `stock_history` 已覆盖股票数
- `stock_history` 总条数
- fundamentals / history 的最近更新时间

## 重置进度

### 重置通用进度

```bash
/Users/chenglanguo/stock/.venv/bin/python /Users/chenglanguo/stock/slow_fetcher.py --reset
```

说明：

- 会清除元进度信息
- 包括 fundamentals 的断点续跑位置
- 不会删除已写入的数据表内容

### 重置 K 线进度

```bash
/Users/chenglanguo/stock/.venv/bin/python /Users/chenglanguo/stock/slow_fetcher.py --reset-history
```

## 导出快照

```bash
/Users/chenglanguo/stock/.venv/bin/python /Users/chenglanguo/stock/slow_fetcher.py --export snapshot.csv
```

## 单股分析

### CLI

```bash
/Users/chenglanguo/stock/.venv/bin/python /Users/chenglanguo/stock/slow_fetcher.py --analyze 000066
/Users/chenglanguo/stock/.venv/bin/python /Users/chenglanguo/stock/slow_fetcher.py --analyze 600519 --analyze-days 200
```

输出会包含：

- 市场环境
- 事件因子
- 买 / 卖 / 观望
- 入场区间
- 止损位 / 目标位
- Kelly 仓位 / VaR
- 支持理由 / 反对理由

### Python

```python
from slow_fetcher import analyze_stock, load_stock_history, load_stocks_from_db

result = analyze_stock("000066")
print(result["decision"]["conclusion"])

df = load_stock_history("000066")
snapshot = load_stocks_from_db()
```

## 选股

```python
from stock_screener import screen_stocks

result = screen_stocks("value", top_n=10)
print(result["summary"])

result = screen_stocks("momentum", top_n=10)
print(result["summary"])

result = screen_stocks("oversold", top_n=10)
print(result["summary"])

result = screen_stocks("potential", top_n=10)
print(result["summary"])
```

当前选股流程包含：

- 本地市场快照读取
- `stock_history` 历史画像评分
- 策略初筛
- 深度量化诊断
- 事件复核
- 最终交易决策排序

## MCP

当前 MCP 服务入口：

```bash
/Users/chenglanguo/stock/.venv/bin/python /Users/chenglanguo/stock/stock_mcp_server.py
```

推荐的高层工具：

- `resolve_stock`
- `full_stock_analysis`
- `stock_screener`
- `stock_news`
- `market_news`

单股完整分析会自动串联：

- 标的解析
- 市场新闻
- 个股新闻
- 公告事件
- 财报分析
- 财务数据
- 量化诊断
- 量化资金分析
- 风险评估

## 重要说明

### 关于 `--intraday-update`

它不是更新 `stock_history`，只更新：

- `stock_fundamentals`

并且现在支持同一天断点续跑。

### 关于 `--daily-close-update`

它会做两件事：

1. 刷新当天 `stock_fundamentals`
2. 增量更新 `stock_history`

### 关于 `stock_history`

`stock_history` 当前仍然是逐只股票拉日 K。

也就是说：

- 不能一次批量拉几百只股票的完整日 K
- 现在采用的是“逐只增量补到目标交易日”

## 当前推荐的每日命令

### 盘中

```bash
/Users/chenglanguo/stock/.venv/bin/python /Users/chenglanguo/stock/slow_fetcher.py --intraday-update --batch 200 --interval 10
```

### 收盘后

```bash
/Users/chenglanguo/stock/.venv/bin/python /Users/chenglanguo/stock/slow_fetcher.py --daily-close-update --batch 100 --interval 10
```

### 补指定交易日

```bash
/Users/chenglanguo/stock/.venv/bin/python /Users/chenglanguo/stock/slow_fetcher.py --daily-close-update --trade-date 2026-03-23 --batch 100 --interval 10
```

## 免责声明

本项目仅用于量化研究与流程辅助，不构成任何投资建议。投资有风险，入市需谨慎。
