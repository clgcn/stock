# A 股量化分析系统

全离线 A 股量化分析工具。通过慢速增量拉取策略绕过东方财富 API 频率限制，将数据存入本地 SQLite，后续分析完全脱离网络依赖。

## 项目结构

```
stock/
├── slow_fetcher.py        # 数据采集 + 存储 + 实时分析 (日常使用入口)
├── stock_screener.py      # 选股器 (快筛 + 深度量化诊断)
├── stock_tool.py           # 底层 API 封装 (K线/行情/财务/画图)
├── quant_engine.py        # 7 维量化分析引擎 (纯数学计算)
├── backtest_engine.py     # 策略回测引擎
├── risk_manager.py        # 风险管理 (VaR/凯利公式/止损)
├── stock_mcp_server.py    # MCP 服务器 (Cursor/Claude Desktop 集成)
├── setup.sh               # 一键安装脚本
├── data/
│   └── stocks.db          # SQLite 数据库 (股票列表 + K线历史)
└── charts/                # 图表输出目录
```

## 快速开始

```bash
# 安装依赖
bash setup.sh

# 或手动安装
python3 -m venv .venv
source .venv/bin/activate
pip install mcp requests pandas numpy matplotlib curl_cffi
```

---

## 定时任务 (Cron)

数据需要通过定时任务慢慢积累，以下是需要配置的 cron 任务。

### 1. 拉取股票列表 (基本面快照)

每小时拉 1 页 (100 只)，约 35 小时拉完全部 ~3500 只主板 A 股。

```bash
# 每小时整点执行
0 * * * * cd ~/stock && .venv/bin/python slow_fetcher.py >> data/fetcher.log 2>&1
```

拉取的数据：代码、名称、现价、涨跌幅、PE、PB、市值、换手率、60日涨跌幅等。

### 2. 拉取 K 线历史

K 线接口限制较松，可以更频繁。每 10 分钟拉 3 只股票的 1 年日 K 线。

```bash
# 每 10 分钟执行
*/10 * * * * cd ~/stock && .venv/bin/python slow_fetcher.py --history --batch 3 >> data/history.log 2>&1
```

拉取的数据：每只股票 243 个交易日的 OHLCV（开高低收量）、涨跌幅、换手率。

### 3. 定期刷新 (数据拉完后)

数据全部拉完后，定期重置并刷新：

```bash
# 每周一凌晨 1 点重置并重新拉
0 1 * * 1 cd ~/stock && .venv/bin/python slow_fetcher.py --reset && .venv/bin/python slow_fetcher.py --reset-history
```

### 查看拉取进度

```bash
python slow_fetcher.py --status
```

输出示例：

```
╔══════════════════════════════════════════════════════╗
║  [股票列表]                                          ║
║    库中股票:   3472/3472 (100.0%)                     ║
║    状态:       ✓ 已完成                               ║
║                                                      ║
║  [K线历史]                                           ║
║    已拉股票:   3472/3472 只                           ║
║    K线总条数:  844,296 条                             ║
║    状态:       ✓ 已完成                               ║
╚══════════════════════════════════════════════════════╝
```

---

## 分析工具

数据库填好后，以下命令均为**离线运行**（仅实时行情需 1 次轻量 API 调用）。

### 分析单只股票

```bash
python slow_fetcher.py --analyze 600519
python slow_fetcher.py --analyze 000858 --analyze-days 200
```

输出：现价、PE/PB/市值、量化评分 (-100~+100)、买卖信号、趋势方向、蒙特卡洛模拟概率、风险提示。

工作原理：历史 K 线从本地 SQLite 读取，今日实时行情从新浪拉取（1 次轻量请求），拼接后跑完整 7 维量化诊断。

### 选股筛选

```bash
# 在 Python 中调用
from stock_screener import screen_stocks

# 价值股策略 (低PE + 低PB)
result = screen_stocks("value", top_n=10)

# 动量策略 (强趋势 + 放量)
result = screen_stocks("momentum", top_n=10)

# 超跌反弹策略
result = screen_stocks("oversold", top_n=10)

# 潜力股策略 (60日深跌 + 基本面完好)
result = screen_stocks("potential", top_n=10)

print(result["summary"])
```

### 导出数据

```bash
# 导出股票列表到 CSV
python slow_fetcher.py --export snapshot.csv
```

### Python API

```python
from slow_fetcher import (
    analyze_stock,           # 混合分析单只股票
    load_stocks_from_db,     # 读取全部股票基本面
    load_stock_history,      # 读取单只股票 K 线历史
    load_all_history,        # 读取全部 K 线历史
)

# 分析单只股票 (本地历史 + 线上实时)
result = analyze_stock("600519")
print(result["signal"])          # "buy" / "hold" / "sell"
print(result["total_score"])     # -100 ~ +100
print(result["prob_up"])         # 蒙特卡洛上涨概率

# 读取本地数据
stocks = load_stocks_from_db()   # DataFrame: 全部主板股票
df = load_stock_history("600519") # DataFrame: 1 年日 K 线
```

---

## MCP 服务器 (Cursor / Claude Desktop)

安装后 Cursor 和 Claude Desktop 可以直接调用以下工具：

| 工具 | 功能 |
|------|------|
| `realtime_quote` | 实时行情 (单只/多只) |
| `kline_data` | K 线数据 (日/周/月/分钟级) |
| `financial_data` | 财务指标 (PE/PB/ROE/市值) |
| `kline_chart` | K 线图 (蜡烛图 + MACD/RSI/KDJ) |
| `batch_quote` | 批量行情对比 |
| `stock_diagnosis` | 7 维量化诊断 (趋势/动量/波动/均值回归/量价/蒙特卡洛) |
| `strategy_backtest` | 策略回测 (均线交叉/MACD/RSI/布林带/多因子) |
| `risk_assessment` | 风险评估 (VaR/CVaR/凯利公式/止损建议) |
| `stock_screener` | 全市场选股 (价值/动量/超跌/潜力) |

直接对 Cursor/Claude 说「帮我分析一下 600519」或「帮我找几只低估值的股票」即可。

---

## 数据源

全部来自**东方财富**和**新浪财经**的公开 API，免费、无需注册、无需 API Key。

| 数据 | 来源 | 接口 |
|------|------|------|
| 股票列表 + 基本面 | 东方财富 | `push2.eastmoney.com` clist API |
| K 线历史 | 东方财富 | `push2his.eastmoney.com` kline API |
| 实时行情 | 新浪财经 | `hq.sinajs.cn` |
| 财务指标 | 东方财富 | `push2.eastmoney.com` stock API |

---

## 数据库结构 (data/stocks.db)

```
stocks 表 — 全 A 股基本面快照
├── code (主键), name, current, pct_chg, change
├── volume, amount, amplitude, high, low, open, prev_close
├── volume_ratio, turnover_rate, pe_ttm, pb
├── total_mv (亿), float_mv (亿), chg_60d, chg_ytd
├── raw_json (原始 API 返回), updated_at, batch_id
│
stock_history 表 — 个股日 K 线历史
├── code + date (联合主键)
├── open, close, high, low, volume, amount
├── amplitude, pct_chg, change, turnover
│
meta 表 — 拉取进度追踪
├── next_page, total_expected, batch_id
├── history_done, history_last_fetch, ...
```

## 免责声明

本工具所有分析结果基于数学模型和历史数据，不构成任何投资建议。投资有风险，入市需谨慎。
