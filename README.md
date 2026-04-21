# A 股量化分析系统

一套基于 PostgreSQL + MCP 的 A 股量化研究与决策支持系统，提供五面体系分析（技术面/资金面/催化剂面/基本面/风控面）、多因子选股、回测引擎、机器学习预测和组合优化。

## 项目架构

```
stock/
├── stock_mcp_server.py       # MCP 服务入口（27 个工具）
├── prompt.md                 # AI 系统提示词（五面体系 + 工具使用规范）
│
├── 数据层
│   ├── db.py                 # PostgreSQL 连接 + 线程安全 + 上下文管理
│   ├── slow_fetcher.py       # 数据更新主入口 + 本地 DB 管理
│   ├── import_json.py        # 从本地 JSON 导入 stocks
│   ├── data_fetcher.py       # K 线/实时行情/技术指标（东方财富/腾讯）
│   └── _http_utils.py        # 共享 HTTP 工具
│
├── 基本面/事件
│   ├── financial.py          # PE/PB/ROE/毛利率/杜邦/PEG（CAGR 几何均值）
│   ├── announcements.py      # 公告分类 + 业绩变动提取 + 重大合同/增发/立案识别
│   ├── news_analyzer.py      # 市场新闻/个股新闻情绪分析
│   └── market_quote.py       # 外围市场行情（美/欧/亚/大宗）
│
├── 资金/机构
│   ├── capital_flow.py       # 北向资金/主力资金/融资融券
│   └── institutional.py      # 机构持仓评分（季报+时间衰减）+ 近实时龙虎榜/大宗
│
├── 量化引擎
│   ├── quant_engine.py       # 7 维诊断/EWMA/线性回归/蒙特卡洛
│   ├── factor_model.py       # 多因子模型（MAD z-score/Spearman IC/动量/质量）
│   ├── ml_predictor.py       # GBM + 决策树 + Hurst 指数（对数 R/S）
│   ├── backtest_engine.py    # 回测引擎（T+1 挂单/涨跌停/停牌处理）
│   ├── risk_manager.py       # VaR/CVaR/Kelly/ATR（多日持仓 VaR 缩放）
│   ├── portfolio_optimizer.py# 组合优化（max_sharpe/min_variance/risk_parity）
│   └── sector_rotation.py    # 行业轮动 + 周期阶段识别
│
├── 五面体系
│   ├── module_a_environment.py  # 环境层（外围+新闻+北向 → H/M/L 评级）
│   ├── module_c_scorecard.py    # 评分卡聚合（短线/长线加权 → A+/A/B+/B/C/D）
│   ├── module_d_riskcontrol.py  # 风控层（ATR 止损/Kelly 仓位/VaR 约束）
│   └── quant_detector.py        # 量化资金活跃度检测
│
├── 选股
│   └── stock_screener.py     # 两阶段选股（DB 初筛 → 全维度复核）
│
└── requirements.txt
```

## 环境要求

- Python 3.10+
- PostgreSQL 14+（DATABASE_URL 环境变量）

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` 主要依赖（含版本约束）：
```
pandas>=2.0,<3.0
numpy>=1.24,<3.0
matplotlib>=3.7
requests>=2.28
curl_cffi>=0.6
mcp>=1.0
psycopg2-binary>=2.9,<3.0
python-dotenv>=1.0
akshare>=1.12
```

## 数据库配置

系统使用 PostgreSQL，通过 `DATABASE_URL` 环境变量连接：

```bash
export DATABASE_URL="postgresql://user:pass@host:port/dbname?sslmode=require"
```

或在项目根目录创建 `.env` 文件：
```
DATABASE_URL=postgresql://user:pass@host:port/dbname?sslmode=require
```

### 主要数据表

| 表名 | 内容 |
|------|------|
| `stocks` | 股票宇宙元信息（code/name/suspended） |
| `stock_fundamentals` | 每日估值快照（PE/PB/总市值/流通市值） |
| `stock_history` | 日 K 历史（OHLCV + 涨跌幅/换手率） |
| `stock_scores` | 量化评分历史（用于选股初筛） |
| `institution_top_holders` | 十大流通股东季报 |
| `fund_holdings` | 基金持仓数据 |

## MCP 服务

### 启动

```bash
.venv/bin/python stock_mcp_server.py
```

### 27 个工具总览

**入口级（优先调用）**
| 工具 | 说明 |
|------|------|
| `resolve_stock` | 股票名称/代码标准化解析 |
| `full_stock_analysis` | 单股五面体系全维度分析 + 3×3 综合决策 |
| `full_stock_selection` | 两阶段选股（DB 初筛 + 全维度复核） |

**市场/宏观**
| 工具 | 说明 |
|------|------|
| `market_news` | 外围行情 + 市场新闻基线 |
| `northbound_flow` | 北向资金成交活跃度（外资参与度） |
| `sector_analysis` | 行业板块轮动排名 + 周期阶段 + 超配/低配建议 |

**个股基本面**
| 工具 | 说明 |
|------|------|
| `stock_news` | 个股新闻情绪与催化线索 |
| `stock_announcements` | 公告/财报窗口检查 |
| `earnings_analysis` | 财报质量与市场反应分析 |
| `financial_data` | PE/PB/ROE/毛利率/净利率/市值 |
| `valuation_quality` | PEG + 杜邦分析（估值质量深度） |
| `balance_sheet` | 资产负债率 + 商誉预警 |
| `dividend_history` | 分红历史（价值型/长线标的） |
| `institutional_holdings` | 季报机构持仓（十大股东+基金+机构共识评分） |

**资金面**
| 工具 | 说明 |
|------|------|
| `moneyflow` | 主力大单净流入/流出（短线核心） |
| `margin_trading` | 融资融券余额变化（多空力量对比） |
| `institutional_realtime` | 近实时机构活动（主力资金+龙虎榜+大宗交易） |

**技术面/量化**
| 工具 | 说明 |
|------|------|
| `realtime_quote` | 单股/多股实时价格快照 |
| `batch_quote` | 批量行情概览（多股横向对比） |
| `kline_data` | K 线原始数据 + MA/MACD/RSI/BOLL/KDJ |
| `stock_diagnosis` | 7 维量化诊断 + K 线形态 + 蒙特卡洛 |
| `quant_activity` | 量化资金参与度（仅交易时段有效） |
| `factor_analysis` | 多因子分析（单股因子暴露/全市场截面排名） |

**风控/回测**
| 工具 | 说明 |
|------|------|
| `risk_assessment` | VaR/CVaR/Kelly/ATR 止损止盈/仓位建议 |
| `strategy_backtest` | 历史策略回测（T+1/涨跌停/停牌合规） |
| `portfolio_optimize` | 多股组合优化（权重+建议份额+风险分解） |

**底层**
| 工具 | 说明 |
|------|------|
| `stock_screener` | 仅用户明确要求纯技术面粗筛时才单独调用 |

### MCP 客户端配置示例（Claude Desktop）

```json
{
  "mcpServers": {
    "stock": {
      "command": "/path/to/stock/.venv/bin/python",
      "args": ["/path/to/stock/stock_mcp_server.py"],
      "env": {
        "DATABASE_URL": "postgresql://user:pass@host:port/dbname"
      }
    }
  }
}
```

## 数据更新流程

### 导入股票名单

```bash
.venv/bin/python import_json.py
```

### 盘中更新（仅 fundamentals）

```bash
.venv/bin/python slow_fetcher.py --intraday-update --batch 200 --interval 10
```

### 收盘后完整更新

```bash
.venv/bin/python slow_fetcher.py --daily-close-update --batch 100 --interval 10
```

### 补指定交易日

```bash
.venv/bin/python slow_fetcher.py --daily-close-update --trade-date 2026-04-21 --batch 100 --interval 10
```

### 仅补 K 线历史

```bash
.venv/bin/python slow_fetcher.py --history --batch 100 --interval 10 --auto
```

### 查看进度

```bash
.venv/bin/python slow_fetcher.py --status
```

### 重置进度

```bash
.venv/bin/python slow_fetcher.py --reset           # 重置 fundamentals 断点
.venv/bin/python slow_fetcher.py --reset-history   # 重置 K 线进度
```

### 导出快照

```bash
.venv/bin/python slow_fetcher.py --export snapshot.csv
```

## 核心算法说明

### 多因子模型（factor_model.py）

- **标准化**：MAD 鲁棒 z-score，`(x - median) / (MAD / 0.6745)`，Winsorize ±3σ
- **IC 计算**：Spearman 等级相关（Pearson-on-ranks，O(n log n)）
- **动量因子**：`mom_3m = close[-6] / close[-70] - 1`（跳过最近 5 日避免短期反转）
- **下行风险**：半方差 `E[min(r, 0)²]`（全样本计算，非截断样本）

### 回测引擎（backtest_engine.py）

- **T+1 合规**：信号在 T 日收盘触发 → 次日开盘价执行（挂单机制）
- **涨跌停处理**：买入遇涨停取消；卖出遇跌停延至下一可交易日
- **停牌处理**：成交量为 0 视为停牌，跳过所有订单
- **Sortino**：全样本半方差 `√E[min(r-rf, 0)²] × √252`

### 风险管理（risk_manager.py）

- **多日 VaR**：持仓 N 日时用重叠对数收益率滑窗估计，避免简单 `σ × √N` 低估尾部
- **VaR 方法**：历史模拟法（HS），置信度 95%/99%

### ML 预测（ml_predictor.py）

- **Hurst 指数**：R/S 分析，`H = log(R/S) / log(n)`（对数比，范围 0~1）
- **特征标准化**：仅用训练集均值/标准差归一化，防止前向偏差
- **GBM**：自实现梯度提升分类器，walk-forward 验证

## 量化体系五面架构

```
环境层(A): market_news + northbound_flow → 大盘评级 H/M/L
     ↓
催化剂面: 个股新闻 + 公告 + 财报 (短线/长线共用)
     ↓
┌────────────────┐    ┌─────────────────────────────────┐
│   短线路径      │    │          长线路径                │
│ 技术面   25%   │    │ 财务质量    30%                  │
│ 资金面   40%   │    │ 估值质量    25%                  │
│ 量化诊断 10%   │    │ 负债/商誉   20%                  │
│ 催化剂面 25%   │    │ 成长        15%                  │
└────────────────┘    │ 分红/治理   10%                  │
                      └─────────────────────────────────┘
     ↓
风控面: VaR/Kelly/ATR → 仓位约束（独立层，不参与加权）
     ↓
3×3 综合决策矩阵（短线强度 × 长线档位）
```

## 免责声明

本项目仅用于量化研究与流程辅助，不构成任何投资建议。投资有风险，入市需谨慎。
