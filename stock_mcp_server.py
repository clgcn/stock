#!/usr/bin/env python3
"""
A-Share Stock Data MCP Server
================================
Enables Claude Desktop to directly call A-share real-time quotes,
K-line data, financial indicators, and technical analysis tools.

Installation:
    pip install mcp

Register in Claude Desktop config (see README or setup instructions).
"""

import json
import re
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# -- Add this directory to path for module imports --
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print(
        "ERROR: mcp package not found. Please run: pip install mcp\n"
        "   or run: bash setup.sh",
        file=sys.stderr,
    )
    sys.exit(1)

import stock_tool as st
import quant_engine as qe
import backtest_engine as bt
import risk_manager as rm
import stock_screener as sc
import news_analyzer as na
import quant_detector as qd
import slow_fetcher as sf

# -- Initialize MCP server --
mcp = FastMCP(
    name="a-share-stock",
    instructions=(
        "你现在是A股量化投资研究员，目标不是随意给观点，而是按严格流程把市场、新闻、财报、估值、技术、量化资金、风险控制全部纳入，再给出明确的买/卖/观望结论。\n\n"

        "【工具定位】\n"
        "1. resolve_stock       — 股票名称/代码标准化解析，任何名称输入都先过这一步\n"
        "2. full_stock_analysis — 单股完整流程入口，会自动串联市场新闻、公告、财报、估值、量化、风控\n"
        "3. market_news         — 市场环境/外围市场/新闻面总开关，任何选股或单股分析前优先调用\n"
        "4. stock_screener      — 全市场候选池筛选，基于本地 stocks + stock_history + stock_fundamentals\n"
        "5. stock_news          — 个股新闻面，补充公司/行业相关新闻情绪与催化线索\n"
        "6. stock_announcements — 个股公告/财报窗口检查，单股分析和候选复核时必调\n"
        "7. earnings_analysis   — 命中财报/业绩公告时必调，用于判断财报质量和市场反应\n"
        "8. financial_data      — 估值和财务质量复核\n"
        "9. kline_data / kline_chart — 技术结构和图形复核\n"
        "10. quant_activity      — 交易时段内的量化资金参与度分析，短线/盘中结论时强烈建议调用\n"
        "11. stock_diagnosis     — 核心量化诊断，输出综合分、概率和交易结论\n"
        "12. risk_assessment     — 风险、止损、仓位、Kelly、VaR/CVaR\n"
        "13. realtime_quote / batch_quote — 实时价格确认和批量行情概览\n\n"

        "【硬规则】\n"
        "1. 用户给股票名称、别名或名称+代码混合输入时，必须先调用 resolve_stock，不能凭模型记忆猜代码。\n"
        "2. 用户给股票代码时，也应优先用 resolve_stock 反查名称并确认标的是同一只股票。\n"
        "3. 如果名称和代码冲突，必须指出冲突并以本地 stocks 表解析结果为准，不能直接进入分析。\n"
        "4. 没有先看 market_news，不允许直接给买入建议。\n"
        "5. 没有检查 stock_announcements，不允许直接给单股最终结论。\n"
        "6. 命中财报/业绩预告/业绩快报时，没有调用 earnings_analysis，不允许直接给买入建议。\n"
        "7. 没有做 risk_assessment，不允许给最终仓位建议。\n"
        "8. 交易时段内，如果要给短线/盘中交易结论，应补 quant_activity；若无法获取，要明确说明缺失。\n"
        "9. 若市场面、新闻面、财报面、技术面互相冲突，默认降级为观望，不强行给买入。\n"
        "10. 允许大量输出观望；不要为了给答案而勉强买卖。\n\n"

        "【新闻面与时效性】\n"
        "market_news 提供的是市场与新闻环境基线，必须把它视为'新闻面步骤'的一部分。\n"
        "若涉及极强时效事件（突发政策、地缘政治、重大科技发布、会议/展会进展），在调用 market_news 后仍应额外用 web search 补最新信息，不能把旧RSS内容当成实时事实。\n\n"

        "【展示要求】\n"
        "默认使用结构化、仪表板式的输出风格，优先分区、表格、评分摘要、结论卡片化表达。\n"
        "除非用户明确要求纯文本简写，否则单股和选股结果都应尽量按仪表盘格式组织，而不是写成长段散文。\n"
        "仪表盘格式应优先使用清晰分区标题、关键指标摘要、表格/短列表、结论卡片式段落，让用户一眼看到最重要信息。\n"
        "如果当前任务适合可视化，且工具能力允许，应尽量补充图形输出而不是只给纯文本。\n"
        "可视化优先级：kline_chart > kline_data 表格 > 纯文字描述。\n"
        "若无法出图，要明确说明原因，不要假装已经展示图表。\n"
        "无论是否出图，量化资金参与度分析都必须在最终结果中单独成段出现，不能只隐含在其他结论里。\n\n"

        "【仪表盘输出模板】\n"
        "单股分析默认按以下区块顺序输出：\n"
        "1. 顶部摘要：股票名称/代码、最新价、结论、强度、置信度。\n"
        "2. 市场环境：market_news 结论与市场风险偏好。\n"
        "3. 个股新闻面：stock_news 结论与新闻情绪微调。\n"
        "4. 公告/财报面：stock_announcements + earnings_analysis 结论。\n"
        "5. 估值与财务面：PE/PB/ROE/毛利率/净利率/市值等。\n"
        "6. 技术与量化诊断：趋势、动量、波动、支撑阻力、概率模拟。\n"
        "7. 量化资金面：quant_activity 结论，明确写出是否量化主导及交易含义。\n"
        "8. 风险面：VaR/CVaR/Kelly/止损止盈/风险收益比。\n"
        "9. 操作建议：入场区间、止损位、目标位、仓位建议、持有周期、失效条件。\n"
        "10. 支持理由 / 反对理由：分别单列。\n"
        "选股结果默认按以下区块顺序输出：\n"
        "1. 顶部市场结论。\n"
        "2. 推荐名单表格。\n"
        "3. 每只股票的小卡片：新闻面、财报面、估值面、技术面、量化资金面、风险面、操作建议。\n\n"

        "【单股分析标准流程】\n"
        "用户问'这只股票能不能买/值不值得看/帮我分析'时，按下面顺序：\n"
        "优先直接调用 full_stock_analysis；只有用户明确要求拆步骤时，才手工逐个工具展开。\n"
        "Step 0. resolve_stock\n"
        "        若用户给的是股票名称、简称或名称+代码混合输入，先解析成标准 code + name；若存在歧义，先指出再继续。\n"
        "Step 1. market_news\n"
        "        先判断当前市场是否允许进攻，给出市场风险偏好与 news_sentiment 基线。\n"
        "Step 2. stock_news\n"
        "        补充公司/行业相关新闻、情绪方向和潜在催化，不要只看大盘新闻。\n"
        "Step 3. stock_announcements\n"
        "        检查公告、财报、业绩预告、监管事项、回购减持等事件。\n"
        "Step 4. earnings_analysis\n"
        "        仅当 Step 3 命中财报/业绩类公告时必须调用；根据结果修正 news_sentiment。\n"
        "Step 5. financial_data\n"
        "        查看 PE/PB/ROE/毛利率/净利率/市值等估值和财务质量因素。\n"
        "Step 6. realtime_quote + kline_data\n"
        "        确认当前价格位置、趋势、动量、支撑阻力、技术结构。\n"
        "Step 7. quant_activity\n"
        "        交易时段内分析量化资金参与度；非交易时段可跳过，但要说明。\n"
        "Step 8. stock_diagnosis\n"
        "        使用前面得到的 news_sentiment 和上下文做最终量化诊断。\n"
        "Step 9. risk_assessment\n"
        "        给出VaR/CVaR/Kelly/ATR止损止盈/仓位风险。\n"
        "Step 10. 最终结论\n"
        "        必须综合以下维度后输出：市场环境、新闻面、公告事件、财报质量、估值、技术趋势、量化资金、风险收益比、仓位建议。\n\n"

        "【单股输出要求】\n"
        "最终回答必须明确包含：\n"
        "1. 结论：买入 / 卖出 / 观望\n"
        "2. 结论强度与置信度\n"
        "3. 市场环境结论\n"
        "4. 新闻面/事件面结论\n"
        "5. 财报面结论（若有）\n"
        "6. 估值面结论\n"
        "7. 技术面结论\n"
        "8. 量化资金结论（必须明确写出：量化主导 / 活跃 / 参与 / 偏少，以及对操作的含义）\n"
        "9. 如工具可用且场景合适，补充图表或说明为何未出图\n"
        "10. 入场区间、止损位、目标位、仓位建议、持有周期\n"
        "11. 支持理由与反对理由\n"
        "12. 失效条件\n\n"

        "【选股标准流程】\n"
        "用户问'有哪些股票值得买/帮我选几只/找最适合入场的票'时，按下面顺序：\n"
        "Step 0. resolve_stock\n"
        "        若用户指定了行业龙头、公司名称、已有自选股等名称输入，先解析为标准股票代码；若只是全市场选股，可跳过。\n"
        "Step 1. market_news\n"
        "        先判断今天/当前阶段属于 risk_on、neutral 还是 risk_off。\n"
        "Step 2. stock_screener\n"
        "        运行筛选器，形成候选池。必要时可分别跑 value、momentum、oversold、potential 四种策略并合并去重。\n"
        "Step 3. 候选复核\n"
        "        对前排候选逐只调用 stock_news、stock_announcements；若命中财报则必调 earnings_analysis。\n"
        "Step 4. 估值与技术复核\n"
        "        对前排候选逐只调用 financial_data、kline_data、stock_diagnosis；如适合展示图表，可补 kline_chart。\n"
        "Step 5. 量化资金与风险复核\n"
        "        交易时段内优先补 quant_activity，并调用 risk_assessment。\n"
        "Step 6. 综合排序\n"
        "        结合市场环境、新闻面、财报面、估值面、技术面、量化资金、风险面后，输出最终推荐名单。\n\n"

        "【选股输出要求】\n"
        "不要只给股票名单。每只候选至少要说明：\n"
        "1. 为什么入选\n"
        "2. 主要风险点\n"
        "3. 量化资金是否活跃，以及这对追涨/低吸/分批下单意味着什么\n"
        "4. 当前是否适合买入，还是只适合观察\n"
        "5. 入场区间、止损位、目标位、仓位建议\n\n"

        "【决策原则】\n"
        "市场环境决定能不能做，财报与新闻决定有没有逻辑，技术与量化资金决定现在能不能进，风险评估决定该不该下手以及仓位大小。\n"
        "如果任何一层明显不成立，默认降级为观望。\n\n"

        "免责声明：所有分析仅用于量化研究与流程辅助，不构成投资建议。"
    ),
)

# =====================================================
# Internal helpers
# =====================================================

def _resolve_stock_candidates(query: str, limit: int = 10):
    q = (query or "").strip()
    if not q:
        return []

    conn = sf._get_db()
    try:
        if q.isdigit() and len(q) == 6:
            return conn.execute(
                """
                SELECT code, name, suspended
                FROM stocks
                WHERE code = ?
                ORDER BY suspended ASC, code ASC
                LIMIT ?
                """,
                (q, max(int(limit), 1)),
            ).fetchall()

        exact_rows = conn.execute(
            """
            SELECT code, name, suspended
            FROM stocks
            WHERE name = ?
            ORDER BY suspended ASC, code ASC
            LIMIT ?
            """,
            (q, max(int(limit), 1)),
        ).fetchall()
        like_rows = conn.execute(
            """
            SELECT code, name, suspended
            FROM stocks
            WHERE name LIKE ?
            ORDER BY
                CASE WHEN name = ? THEN 0
                     WHEN name LIKE ? THEN 1
                     ELSE 2 END,
                suspended ASC,
                code ASC
            LIMIT ?
            """,
            (f"%{q}%", q, f"{q}%", max(int(limit), 1)),
        ).fetchall()

        seen = set()
        merged = []
        for row in list(exact_rows) + list(like_rows):
            key = row[0]
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
        return merged
    finally:
        conn.close()


def _resolve_stock_unique(query: str):
    rows = _resolve_stock_candidates(query, limit=10)
    if len(rows) == 1:
        code, name, suspended = rows[0]
        return {"code": code, "name": name, "suspended": bool(suspended), "rows": rows}
    return {"code": None, "name": None, "suspended": None, "rows": rows}


# =====================================================
# Tool 1: Resolve Stock
# =====================================================

@mcp.tool()
def resolve_stock(query: str, limit: int = 10) -> str:
    """
    Resolve a stock name/code/alias-like input into canonical A-share code + name
    using the local stocks universe table.

    Parameters
    ----------
    query : str
        Stock code, exact name, or partial name. Example: "000001", "平安银行", "中国长城".
    limit : int
        Maximum number of matches to return, default 10.

    Returns
    -------
    A plain-text resolution result. When uniquely matched, returns the canonical
    stock code and name. When ambiguous, returns the top candidates and asks the
    caller to continue with the confirmed code.
    """
    q = (query or "").strip()
    if not q:
        return "ERROR: query is empty."

    try:
        rows = _resolve_stock_candidates(q, limit=limit)
    except Exception as e:
        return f"ERROR: Failed to resolve stock from local database: {e}"

    if not rows:
        return (
            f"No stock matched '{q}' in local stocks table.\n"
            "Please update stocks first or provide a more precise code/name."
        )

    if len(rows) == 1:
        code, name, suspended = rows[0]
        suspended_text = "Yes" if suspended else "No"
        return (
            "Resolved Stock\n"
            "--------------\n"
            f"Query: {q}\n"
            f"Code: {code}\n"
            f"Name: {name}\n"
            f"Suspended: {suspended_text}\n"
            "Use this canonical code/name pair for all subsequent analysis."
        )

    lines = [
        "Ambiguous Stock Match",
        "----------------------",
        f"Query: {q}",
        f"Found {len(rows)} candidates. Use the exact code for the next step:",
    ]
    for code, name, suspended in rows[:max(int(limit), 1)]:
        tag = "停牌" if suspended else "正常"
        lines.append(f"- {code} {name} ({tag})")
    return "\n".join(lines)


# =====================================================
# Tool 2: Full Stock Analysis
# =====================================================

@mcp.tool()
def full_stock_analysis(
    stock_query: str,
    analysis_days: int = 250,
    monte_carlo_days: int = 20,
    include_quant_activity: bool = True,
    include_market_news: bool = True,
) -> str:
    """
    Run the full single-stock analysis workflow in one call.

    This tool is the preferred entry point when the user asks:
    - "分析这只股票"
    - "这只股票能不能买"
    - "帮我看看某只票"

    Workflow:
      resolve_stock -> market_news -> stock_announcements -> earnings_analysis(if needed)
      -> financial_data -> stock_diagnosis -> quant_activity(optional) -> risk_assessment

    Parameters
    ----------
    stock_query : str
        Stock code or stock name.
    analysis_days : int
        Historical days for diagnosis/risk analysis.
    monte_carlo_days : int
        Forward days for Monte Carlo diagnosis.
    include_quant_activity : bool
        Whether to include quant activity analysis.
    include_market_news : bool
        Whether to include market news baseline.

    Returns
    -------
    A fully assembled stock research report including market/news, announcements,
    earnings, valuation, quant diagnosis, quant-fund participation and risk review.
    """
    resolved = _resolve_stock_unique(stock_query)
    rows = resolved["rows"]
    if not rows:
        return (
            f"ERROR: No stock matched '{stock_query}' in local stocks table.\n"
            "Please update stocks first or provide a more precise code/name."
        )
    if len(rows) != 1:
        lines = [
            "ERROR: Stock query is ambiguous.",
            f"Query: {stock_query}",
            "Candidates:",
        ]
        for code, name, suspended in rows[:10]:
            tag = "停牌" if suspended else "正常"
            lines.append(f"- {code} {name} ({tag})")
        lines.append("Please retry with the exact code.")
        return "\n".join(lines)

    stock_code = resolved["code"]
    stock_name = resolved["name"]

    market_report = market_news() if include_market_news else "Skipped market_news by request."
    stock_news_report = stock_news(stock_query=stock_query, max_items=8)
    ann_report = stock_announcements(stock_code)

    earnings_report = None
    earnings_hit_keywords = ("年报", "半年报", "季报", "一季报", "三季报", "业绩预告", "业绩快报", "业绩修正")
    if any(keyword in ann_report for keyword in earnings_hit_keywords):
        earnings_report = earnings_analysis(stock_code)

    market_sentiment = 0.0
    stock_sentiment = 0.0
    earnings_sentiment = 0.0
    market_match = re.search(r"综合建议 news_sentiment 参考值：([+-]?\d+(?:\.\d+)?)", market_report)
    if market_match:
        market_sentiment = float(market_match.group(1))
    stock_match = re.search(r"个股新闻情绪微调：\s*([+-]?\d+(?:\.\d+)?)", stock_news_report)
    if stock_match:
        stock_sentiment = float(stock_match.group(1))
    if earnings_report:
        if "news_sentiment 建议在技术分析基础上 +0.2 ~ +0.3" in earnings_report:
            earnings_sentiment = 0.25
        elif "news_sentiment 建议在技术分析基础上 -0.2 ~ -0.3" in earnings_report:
            earnings_sentiment = -0.25
    combined_news_sentiment = max(-1.0, min(1.0, market_sentiment + stock_sentiment + earnings_sentiment))

    financial_report = financial_data(stock_code)
    diagnosis_report = stock_diagnosis(
        stock_code=stock_code,
        analysis_days=analysis_days,
        news_sentiment=combined_news_sentiment,
        monte_carlo_days=monte_carlo_days,
    )
    quant_report = quant_activity(stock_code) if include_quant_activity else "Skipped quant_activity by request."
    risk_report = risk_assessment(stock_code=stock_code, analysis_days=analysis_days)

    sections = [
        f"Full Stock Analysis  {stock_name} ({stock_code})",
        "=" * 72,
        "",
        "[1] Stock Resolution",
        resolve_stock(stock_query, limit=10),
        "",
        "[2] Market News / Macro Context",
        market_report,
        "",
        "[3] Stock-specific News",
        stock_news_report,
        "",
        "[4] Announcements / Event Review",
        ann_report,
    ]

    if earnings_report:
        sections.extend([
            "",
            "[5] Earnings Analysis",
            earnings_report,
        ])

    sections.extend([
        "",
        "[6] Combined News Sentiment",
        (
            f"Market {market_sentiment:+.2f} + Stock {stock_sentiment:+.2f}"
            f" + Earnings {earnings_sentiment:+.2f} = {combined_news_sentiment:+.2f}"
        ),
        "",
        "[7] Financial Data",
        financial_report,
        "",
        "[8] Stock Diagnosis",
        diagnosis_report,
        "",
        "[9] Quant Activity",
        quant_report,
        "",
        "[10] Risk Assessment",
        risk_report,
    ])
    return "\n".join(sections)


# =====================================================
# Tool 3: Real-time Quote
# =====================================================

@mcp.tool()
def realtime_quote(stock_codes: str) -> str:
    """
    Get real-time quotes for one or more A-share stocks (latest price, change %, volume, etc.).

    Parameters
    ----------
    stock_codes : str
        Single code (e.g. "600519") or comma-separated codes (e.g. "600519,000858,600036").
        Shanghai stocks start with 6, Shenzhen with 0/3. No sh/sz prefix needed.

    Returns
    -------
    Formatted quote text including: stock name, latest price, change %,
    change amount, open, high, low, prev close, volume (lots), turnover (10k CNY), time.
    """
    codes = [c.strip() for c in stock_codes.split(",") if c.strip()]
    try:
        df = st.get_realtime(codes)
    except Exception as e:
        return f"ERROR: Failed to fetch quotes: {e}"

    lines = ["Real-time Quotes", "-" * 50]
    for _, row in df.iterrows():
        arrow = "UP" if row["pct_chg"] >= 0 else "DOWN"
        lines.append(
            f"[{row['name']}] {row['code']}\n"
            f"  Latest: {row['current']:.2f}   "
            f"{arrow} {abs(row['pct_chg']):.2f}%  ({row['change']:+.2f})\n"
            f"  Open: {row['open']:.2f}   High: {row['high']:.2f}   Low: {row['low']:.2f}   Prev Close: {row['prev_close']:.2f}\n"
            f"  Volume: {row['volume']:.0f} lots   Turnover: {row['amount']:.1f} (10k CNY)\n"
            f"  Time: {row['date']} {row['time']}"
        )
    return "\n".join(lines)


# =====================================================
# Tool 2: K-line Data
# =====================================================

@mcp.tool()
def kline_data(
    stock_code: str,
    period: str = "daily",
    start_date: str = "",
    end_date: str = "",
    recent_days: int = 60,
    adjust: str = "qfq",
    with_indicators: bool = True,
) -> str:
    """
    Get A-share historical K-line data with optional technical indicators (MA/MACD/RSI/BOLL/KDJ).

    Parameters
    ----------
    stock_code      : str   Stock code, e.g. "600519" (Kweichow Moutai), "000858" (Wuliangye)
    period          : str   K-line period: daily / weekly / monthly
                            / 1m / 5m / 15m / 30m / 60m (intraday, trading days only)
    start_date      : str   Start date "YYYY-MM-DD", if empty uses recent_days
    end_date        : str   End date "YYYY-MM-DD", if empty uses today
    recent_days     : int   Effective when start_date is empty, fetch recent N days, default 60
    adjust          : str   Adjustment: qfq (forward, default) / hfq (backward) / none
    with_indicators : bool  Whether to compute MA/MACD/RSI/BOLL/KDJ, default True

    Returns
    -------
    Text summary of the most recent 20 K-line bars + indicator values.
    """
    if not start_date and recent_days:
        start_date = (datetime.today() - timedelta(days=recent_days)).strftime("%Y-%m-%d")

    try:
        df = st.get_kline(
            stock_code,
            period=period,
            start=start_date or None,
            end=end_date or None,
            adjust=adjust,
        )
    except Exception as e:
        return f"ERROR: Failed to fetch K-line: {e}"

    name = df.attrs.get("name", stock_code)

    if with_indicators:
        df = st.add_indicators(df)

    last = df.tail(20).copy()
    last["date"] = last["date"].dt.strftime("%Y-%m-%d")

    lines = [
        f"{name} ({stock_code}) {period} K-line   Total {len(df)} bars",
        f"   Range: {df.iloc[0]['date'].strftime('%Y-%m-%d')} -> {df.iloc[-1]['date'].strftime('%Y-%m-%d')}",
        "-" * 70,
    ]

    base_cols = ["date", "open", "high", "low", "close", "pct_chg", "volume"]
    ind_cols  = ["ma5", "ma20", "macd", "rsi12", "kdj_k", "boll_ub", "boll_lb"]
    show_cols = base_cols + ([c for c in ind_cols if c in last.columns] if with_indicators else [])

    lines.append(last[show_cols].to_string(index=False))

    latest = df.iloc[-1]
    ma20_val = latest.get("ma20") if with_indicators else None
    rsi_val  = latest.get("rsi12") if with_indicators else None
    macd_val = latest.get("macd")  if with_indicators else None

    lines.append("-" * 70)
    lines.append(
        f"Latest Close: {latest['close']:.2f}  "
        f"Change: {latest['pct_chg']:+.2f}%"
    )
    if ma20_val:
        rel = "above" if latest["close"] > ma20_val else "below"
        lines.append(f"MA20: {ma20_val:.2f}  -> Price {rel} MA")
    if rsi_val is not None:
        zone = "overbought (>70)" if rsi_val > 70 else ("oversold (<30)" if rsi_val < 30 else "neutral")
        lines.append(f"RSI12: {rsi_val:.1f}  -> {zone}")
    if macd_val is not None:
        trend = "bullish" if macd_val > 0 else "bearish"
        lines.append(f"MACD Hist: {macd_val:.3f}  -> {trend} energy")

    return "\n".join(lines)


# =====================================================
# Tool 3: Financial Data
# =====================================================

@mcp.tool()
def financial_data(stock_code: str) -> str:
    """
    Get key financial and valuation indicators for an A-share stock (PE, PB, ROE, market cap, etc.).

    Parameters
    ----------
    stock_code : str   Stock code, e.g. "600519", "000858"

    Returns
    -------
    Includes PE TTM, PB, PS, total market cap, float market cap,
    ROE, gross margin, net margin, turnover rate, volume ratio, etc.
    """
    try:
        d = st.get_financial(stock_code)
    except Exception as e:
        return f"ERROR: Failed to fetch financial data: {e}"

    def fmt(v, suffix="", fmt=".2f"):
        if v is None:
            return "N/A"
        try:
            return f"{v:{fmt}}{suffix}"
        except Exception:
            return str(v)

    lines = [
        f"Financial Indicators  {d.get('name', stock_code)} ({stock_code})",
        "-" * 40,
        f"  Latest Price    : {fmt(d.get('current'))}",
        "",
        "  -- Valuation --",
        f"  PE TTM          : {fmt(d.get('pe_ttm'))}",
        f"  PE Static       : {fmt(d.get('pe_static'))}",
        f"  PB              : {fmt(d.get('pb'))}",
        f"  PS TTM          : {fmt(d.get('ps_ttm'))}",
        "",
        "  -- Market Cap --",
        f"  Total Mkt Cap   : {fmt(d.get('total_mv'), ' (100M CNY)')}",
        f"  Float Mkt Cap   : {fmt(d.get('float_mv'), ' (100M CNY)')}",
        "",
        "  -- Profitability --",
        f"  ROE             : {fmt(d.get('roe'), '%')}",
        f"  Gross Margin    : {fmt(d.get('gross_margin'), '%')}",
        f"  Net Margin      : {fmt(d.get('net_margin'), '%')}",
        "",
        "  -- Trading --",
        f"  Turnover Rate   : {fmt(d.get('turnover_rate'), '%')}",
        f"  Volume Ratio    : {fmt(d.get('volume_ratio'))}",
    ]
    return "\n".join(lines)


# =====================================================
# Tool 4: K-line Chart (generate image)
# =====================================================

@mcp.tool()
def kline_chart(
    stock_code: str,
    period: str = "daily",
    recent_days: int = 120,
    indicator: str = "macd",
    adjust: str = "qfq",
) -> str:
    """
    Generate an A-share candlestick chart (dark theme, with MA lines, volume, indicator panel)
    and save as PNG image.

    Parameters
    ----------
    stock_code  : str  Stock code, e.g. "600519"
    period      : str  K-line period: daily / weekly / monthly / 1m / 5m / 15m / 30m / 60m
    recent_days : int  Show K-lines within the last N calendar days, default 120
    indicator   : str  Bottom indicator panel: macd (default) / rsi / kdj / boll
    adjust      : str  Adjustment: qfq (forward, default) / hfq / none

    Returns
    -------
    File path of the saved chart, can be opened directly.
    """
    start = (datetime.today() - timedelta(days=recent_days)).strftime("%Y-%m-%d")

    try:
        df = st.get_kline(stock_code, period=period, start=start, adjust=adjust)
    except Exception as e:
        return f"ERROR: Failed to fetch K-line: {e}"

    df = st.add_indicators(df)

    out_dir = _HERE / "charts"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = str(out_dir / f"{stock_code}_{period}_{indicator}_{ts}.png")

    try:
        path = st.plot_kline(df, indicator=indicator, save_path=save_path)
    except Exception as e:
        return f"ERROR: Chart generation failed: {e}"

    name = df.attrs.get("name", stock_code)
    latest = df.iloc[-1]
    return (
        f"K-line chart generated\n"
        f"   Stock: {name} ({stock_code})\n"
        f"   Period: {period}   Indicator: {indicator.upper()}   Total {len(df)} bars\n"
        f"   Latest Close: {latest['close']:.2f}   Change: {latest['pct_chg']:+.2f}%\n"
        f"   File: {path}"
    )


# =====================================================
# Tool 5: Batch Quote Overview
# =====================================================

@mcp.tool()
def batch_quote(stock_list: str) -> str:
    """
    Query multiple stocks at once for a concise overview, suitable for quick comparison.

    Parameters
    ----------
    stock_list : str
        Comma-separated stock codes, e.g. "600519,000858,600036,000001,300750"

    Returns
    -------
    Table-format multi-stock comparison sorted by change % descending.
    """
    codes = [c.strip() for c in stock_list.split(",") if c.strip()]
    if not codes:
        return "ERROR: Please enter at least one stock code, separated by commas"

    try:
        df = st.get_realtime(codes)
    except Exception as e:
        return f"ERROR: Failed to fetch quotes: {e}"

    df = df.sort_values("pct_chg", ascending=False).reset_index(drop=True)

    lines = ["Batch Quote Overview", f"  Total {len(df)} stocks   {df.iloc[0]['date']} {df.iloc[0]['time']}",
             "-" * 65,
             f"  {'Name':<8} {'Code':<8} {'Latest':>8} {'Chg%':>8} {'Change':>7} {'Vol(lots)':>12} {'Turnover(10k)':>12}",
             "-" * 65]

    for _, row in df.iterrows():
        sign = "+" if row["pct_chg"] >= 0 else "-"
        lines.append(
            f"  {row['name']:<8} {row['code']:<8} "
            f"{row['current']:>8.2f} "
            f"{sign}{abs(row['pct_chg']):>6.2f}% "
            f"{row['change']:>+7.2f} "
            f"{row['volume']:>12,.0f} "
            f"{row['amount']:>12,.1f}"
        )

    lines.append("-" * 65)
    up   = (df["pct_chg"] > 0).sum()
    down = (df["pct_chg"] < 0).sum()
    flat = len(df) - up - down
    lines.append(f"  Up {up} / Flat {flat} / Down {down}")
    return "\n".join(lines)


# =====================================================
# Tool 6: Stock Comprehensive Diagnosis (Core Tool)
# =====================================================

@mcp.tool()
def stock_diagnosis(
    stock_code: str,
    analysis_days: int = 250,
    news_sentiment: float = 0.0,
    monte_carlo_days: int = 20,
) -> str:
    """
    Perform full-dimensional quantitative diagnosis on an A-share stock (core tool).

    Runs 7 mathematical models simultaneously:
      Trend analysis (linear regression + ADX + MA alignment)
      Momentum analysis (MACD + RSI + ROC + Williams %R)
      Volatility analysis (ATR + Bollinger + historical volatility)
      Mean reversion (Z-Score multi-period deviation)
      Volume-price analysis (OBV + volume-price divergence)
      Support/Resistance (local extrema + Fibonacci retracement)
      Statistical features (Hurst exponent + skewness + kurtosis + Sharpe ratio)

    Outputs a composite score from -100 to +100 via weighted multi-factor scoring,
    along with a clear buy/hold/sell signal and confidence level.

    Also runs Monte Carlo simulation (5000 GBM paths) to provide
    future N-day up/down probabilities and price ranges.

    Parameters
    ----------
    stock_code       : str    Stock code, e.g. "600519"
    analysis_days    : int    Historical data days for analysis (default 250 ~ 1 year)
    news_sentiment   : float  News sentiment score assessed by you (Claude),
                              range -1.0 (extremely bearish) to +1.0 (extremely bullish),
                              0 means no news info or neutral
    monte_carlo_days : int    Monte Carlo simulation future days (default 20 trading days ~ 1 month)

    Returns
    -------
    Detailed diagnosis report with composite score, per-dimension scores,
    Monte Carlo probabilities, support/resistance levels, and risk warnings.
    """
    start = (datetime.today() - timedelta(days=analysis_days)).strftime("%Y-%m-%d")

    data_source = "online"
    try:
        df = st.get_kline(stock_code, period="daily", start=start, adjust="qfq")
    except Exception as e:
        try:
            df = sf.load_stock_history(stock_code)
            if not df.empty and "date" in df.columns:
                df = df[df["date"] >= start].copy()
            if df.empty or len(df) < 30:
                return f"ERROR: Failed to fetch K-line data: {e}"
            data_source = "local_db"
        except Exception:
            return f"ERROR: Failed to fetch K-line data: {e}"

    if len(df) < 30:
        return f"ERROR: Insufficient data: only {len(df)} bars, at least 30 required"

    try:
        result = qe.comprehensive_diagnosis(
            df,
            news_sentiment=news_sentiment,
            run_monte_carlo=True,
            mc_days=monte_carlo_days,
        )
    except Exception as e:
        return f"ERROR: Diagnosis analysis failed: {e}"

    market_regime = sf.assess_market_regime()
    event_risk = sf.assess_event_risk(stock_code)
    decision = sf._build_trade_decision(
        df,
        result,
        float(df.iloc[-1]["close"]),
        market_regime,
        event_risk,
    )

    # Build detailed output
    lines = [result["summary"], f"Data Source: {data_source}"]
    lines.extend([
        "",
        "=" * 55,
        "Trade Decision:",
        f"  Conclusion      : {decision['conclusion']} ({decision['strength']}, confidence {decision['confidence']})",
        f"  Market Regime   : {market_regime['score']:+.1f}  {market_regime['label']} / {market_regime['action_bias']}",
        f"  Event Factor    : {event_risk['score']:+.1f}  {event_risk['label']}",
        f"  Position Advice : {decision['position_advice']}",
        f"  Holding Period  : {decision['holding_period']}",
        f"  Entry Style     : {decision['entry_style']}",
        f"  Entry Zone      : {decision['entry_low']:.2f} ~ {decision['entry_high']:.2f}",
        f"  Stop Loss       : {decision['stop_loss']:.2f}" if decision.get("stop_loss") is not None else "  Stop Loss       : N/A",
        f"  Take Profit     : {decision['take_profit']:.2f}" if decision.get("take_profit") is not None else "  Take Profit     : N/A",
        f"  Risk/Reward     : 1:{decision['risk_reward']:.2f}" if decision.get("risk_reward") is not None else "  Risk/Reward     : N/A",
        f"  Kelly Position  : {decision['kelly_pct']:.1f}%" if decision.get("kelly_pct") is not None else "  Kelly Position  : N/A",
        f"  VaR(95)         : {decision['var_95_pct']:.2f}%" if decision.get("var_95_pct") is not None else "  VaR(95)         : N/A",
        f"  Event Summary   : {event_risk['summary']}",
    ])

    if decision.get("reasons_for"):
        lines.append("  Why this conclusion:")
        for item in decision["reasons_for"]:
            lines.append(f"    + {item}")

    if decision.get("reasons_against"):
        lines.append("  What is still missing / risky:")
        for item in decision["reasons_against"]:
            lines.append(f"    - {item}")

    if event_risk.get("positive_flags") or event_risk.get("negative_flags"):
        lines.append("  Event details:")
        for item in event_risk.get("positive_flags", [])[:3]:
            lines.append(f"    + {item}")
        for item in event_risk.get("negative_flags", [])[:3]:
            lines.append(f"    - {item}")

    lines.extend([
        f"  Invalidation    : {decision['invalidation']}",
        f"  Why not opposite: {decision['why_not_opposite']}",
    ])

    # Per-dimension details
    lines.extend([
        "",
        "=" * 55,
        "Per-Dimension Scores:",
    ])
    for key, factor in result["factors"].items():
        bar_len = int(abs(factor["score"]) / 5)
        bar = "#" * bar_len
        sign = "+" if factor["score"] >= 0 else ""
        indicator = "+" if factor["score"] > 0 else ("-" if factor["score"] < 0 else "=")
        lines.append(
            f"  {factor['label']:<18} : {sign}{factor['score']:>6.1f}  "
            f"(weight {factor['weight']:.0%})  {indicator} {bar}"
        )

    # Hurst exponent interpretation
    stats_detail = result.get("statistics_detail", {}).get("details", {})
    if "hurst" in stats_detail:
        lines.append(f"\n  Hurst Exponent: {stats_detail['hurst']:.3f} -> {stats_detail.get('hurst_interp', '')}")

    return "\n".join(lines)


# =====================================================
# Tool 7: Strategy Backtest
# =====================================================

@mcp.tool()
def strategy_backtest(
    stock_code: str,
    strategy_name: str = "ma_cross",
    backtest_days: int = 500,
    initial_capital: float = 100000,
    stop_loss_pct: float = 0.0,
    take_profit_pct: float = 0.0,
) -> str:
    """
    Run historical strategy backtest on an A-share stock, simulating real trading
    (with commission, stamp tax, slippage).

    Parameters
    ----------
    stock_code      : str    Stock code, e.g. "600519"
    strategy_name   : str    Strategy name, options:
                             - "ma_cross"    MA Crossover (MA5/MA20 golden/death cross)
                             - "macd"        MACD golden/death cross
                             - "rsi"         RSI overbought/oversold reversal
                             - "bollinger"   Bollinger Band mean reversion
                             - "multifactor" Multi-factor composite scoring (slower)
    backtest_days   : int    Historical days for backtest (default 500 ~ 2 years)
    initial_capital : float  Starting capital (default 100,000 CNY)
    stop_loss_pct   : float  Stop-loss percentage (e.g. 0.05 = stop at 5% loss, 0 = disabled)
    take_profit_pct : float  Take-profit percentage (e.g. 0.10 = take at 10% gain, 0 = disabled)

    Returns
    -------
    Full backtest report: CAGR, Sharpe ratio, max drawdown, win rate, profit factor, trade log, etc.
    """
    start = (datetime.today() - timedelta(days=backtest_days)).strftime("%Y-%m-%d")

    try:
        df = st.get_kline(stock_code, period="daily", start=start, adjust="qfq")
    except Exception as e:
        return f"ERROR: Failed to fetch K-line data: {e}"

    if len(df) < 60:
        return f"ERROR: Insufficient data: only {len(df)} bars, backtest requires at least 60"

    available = list(bt.STRATEGIES.keys())
    if strategy_name not in available:
        return f"ERROR: Unknown strategy: {strategy_name}. Available: {available}"

    try:
        result = bt.backtest(
            df,
            strategy_name=strategy_name,
            initial_capital=initial_capital,
            stop_loss=stop_loss_pct,
            take_profit=take_profit_pct,
        )
    except Exception as e:
        return f"ERROR: Backtest execution failed: {e}"

    return bt.format_backtest_result(result)


# =====================================================
# Tool 8: Risk Assessment
# =====================================================

@mcp.tool()
def risk_assessment(
    stock_code: str,
    position_value: float = 100000,
    analysis_days: int = 250,
) -> str:
    """
    Perform comprehensive risk assessment on an A-share stock.

    Analysis includes:
      VaR (Value at Risk) -- 95% and 99% confidence
      CVaR (Conditional VaR) -- expected loss in extreme scenarios
      Kelly Criterion -- optimal position size recommendation
      ATR Dynamic Stops -- volatility-based stop-loss/take-profit levels
      Stress Test -- historical extreme replay + Monte Carlo extreme paths

    Parameters
    ----------
    stock_code     : str    Stock code, e.g. "600519"
    position_value : float  Current or planned position value (default 100,000 CNY)
    analysis_days  : int    Historical data days for analysis (default 250 ~ 1 year)

    Returns
    -------
    Complete risk report with VaR/CVaR, Kelly position, stop/take-profit levels,
    and stress test scenarios.
    """
    start = (datetime.today() - timedelta(days=analysis_days)).strftime("%Y-%m-%d")

    try:
        df = st.get_kline(stock_code, period="daily", start=start, adjust="qfq")
    except Exception as e:
        return f"ERROR: Failed to fetch K-line data: {e}"

    if len(df) < 30:
        return f"ERROR: Insufficient data: only {len(df)} bars"

    try:
        report = rm.comprehensive_risk_assessment(df, position_value=position_value)
    except Exception as e:
        return f"ERROR: Risk assessment failed: {e}"

    return report


# =====================================================
# Tool 9: Stock Screener
# =====================================================

@mcp.tool()
def stock_screener(
    strategy: str = "value",
    top_n: int = 20,
    deep_scan: bool = True,
    pe_max: float = 0,
    pe_min: float = 0,
    pb_max: float = 0,
    pb_min: float = 0,
    mv_min: float = 0,
    pct_chg_min: float = 0,
    pct_chg_max: float = 0,
    price_min: float = 0,
    price_max: float = 0,
) -> str:
    """
    Scan all ~5000 A-share stocks to find the most promising candidates.

    Two-stage screening process:
      Stage 1 (Fast): Fetch entire market from Eastmoney API, apply basic metric filters.
      Stage 2 (Deep): Run full 7-dimension quant analysis + Monte Carlo on shortlisted stocks.

    Parameters
    ----------
    strategy    : str   Screening strategy:
                        - "value"     Low PE + Low PB + decent market cap (classic value investing)
                        - "momentum"  Strong price trend + volume expansion (trend following)
                        - "oversold"  Deeply oversold stocks near support (mean reversion bounce)
                        - "potential" Fallen 10%+ in 60 days but fundamentals intact — find
                                      low-position stocks with reversal potential (bottom-fishing)
                        - "custom"    User-defined conditions (use the filter parameters below)
    top_n       : int   Number of top results to return (default 20)
    deep_scan   : bool  Whether to run Stage 2 deep quant analysis (default True).
                        Set to False for a faster scan using only basic metrics.
    pe_max      : float [custom] Max PE TTM (0 = no limit)
    pe_min      : float [custom] Min PE TTM (0 = no limit)
    pb_max      : float [custom] Max PB ratio (0 = no limit)
    pb_min      : float [custom] Min PB ratio (0 = no limit)
    mv_min      : float [custom] Min total market cap in 100M CNY (0 = no limit)
    pct_chg_min : float [custom] Min daily change % (0 = no limit)
    pct_chg_max : float [custom] Max daily change % (0 = no limit)
    price_min   : float [custom] Min stock price in CNY (0 = no limit)
    price_max   : float [custom] Max stock price in CNY (0 = no limit)

    Returns
    -------
    Screening report with ranked stock list, scores, signals, and key metrics.
    For deep scan: includes quant scores, trend direction, Monte Carlo probabilities.
    For fast scan: includes PE, PB, market cap, and fast filter scores.
    """
    # Build custom filter kwargs (only pass non-zero values)
    filter_kwargs = {}
    if strategy == "custom":
        if pe_max > 0:
            filter_kwargs["pe_max"] = pe_max
        if pe_min > 0:
            filter_kwargs["pe_min"] = pe_min
        if pb_max > 0:
            filter_kwargs["pb_max"] = pb_max
        if pb_min > 0:
            filter_kwargs["pb_min"] = pb_min
        if mv_min > 0:
            filter_kwargs["mv_min"] = mv_min
        if pct_chg_min != 0:
            filter_kwargs["pct_chg_min"] = pct_chg_min
        if pct_chg_max != 0:
            filter_kwargs["pct_chg_max"] = pct_chg_max
        if price_min > 0:
            filter_kwargs["price_min"] = price_min
        if price_max > 0:
            filter_kwargs["price_max"] = price_max

    try:
        result = sc.screen_stocks(
            strategy=strategy,
            top_n=top_n,
            deep_scan_enabled=deep_scan,
            stage1_limit=min(top_n * 4, 80),
            **filter_kwargs,
        )
    except Exception as e:
        return f"ERROR: Stock screening failed: {e}"

    if "error" in result:
        return f"ERROR: {result['error']}"

    if not deep_scan:
        return result["summary"]

    lines = [result["summary"]]
    market_regime = result.get("market_regime")
    picks = result.get("results", [])

    if market_regime:
        lines.extend([
            "",
            "=" * 55,
            "Screening Decision Context:",
            f"  Market Regime : {market_regime['score']:+.1f}  {market_regime['label']} / {market_regime['action_bias']}",
            f"  Breadth       : Up {market_regime.get('up_ratio', 'N/A')}% / Down {market_regime.get('down_ratio', 'N/A')}%",
            f"  Avg Change    : {market_regime.get('avg_pct_chg', 'N/A')}%",
            f"  Interpretation: {market_regime['summary']}",
        ])

    if picks:
        lines.extend([
            "",
            "Top Actionable Picks:",
        ])
        for idx, item in enumerate(picks[:min(5, len(picks))], 1):
            decision = item.get("decision", {}) or {}
            lines.extend([
                f"{idx}. {item['name']} ({item['code']})  {decision.get('conclusion', item.get('signal', 'N/A'))}",
                f"   Price {item['current']:.2f}   EQS {item.get('entry_quality_score', 0):+.1f}   Quant {item.get('total_score', 0):+.1f}",
                f"   Position {decision.get('position_advice', 'N/A')}   Holding {decision.get('holding_period', 'N/A')}",
                f"   Entry {item.get('entry_low', 'N/A')} ~ {item.get('entry_high', 'N/A')}   Stop {item.get('stop_loss', 'N/A')}   TP {item.get('take_profit', 'N/A')}",
                f"   P(up) {item.get('prob_up', 'N/A')}   Risk/Reward {item.get('risk_reward', 'N/A')}   Event {item.get('event_score', 'N/A')}",
            ])
            for reason in decision.get("reasons_for", [])[:2]:
                lines.append(f"   + {reason}")
            for reason in decision.get("reasons_against", [])[:2]:
                lines.append(f"   - {reason}")
            event_risk = item.get("event_risk", {}) or {}
            for reason in event_risk.get("negative_flags", [])[:1]:
                lines.append(f"   - Event: {reason}")

    return "\n".join(lines)


# =====================================================
# Tool 10: 个股新闻面（新增）
# =====================================================

@mcp.tool()
def stock_news(stock_query: str, max_items: int = 8) -> str:
    """
    Analyze stock-specific news flow beyond market-wide headlines.

    This tool focuses on company-level and stock-specific public news, and should
    be used together with stock_announcements:
      - stock_news: company / media / industry-related news flow
      - stock_announcements: exchange announcements / filings / earnings windows

    Parameters
    ----------
    stock_query : str
        Stock code or stock name, e.g. "000066" or "中国长城"
    max_items : int
        Max number of stock-specific news headlines to show.

    Returns
    -------
    A Chinese report containing stock-specific headlines and a small
    news_sentiment delta for downstream diagnosis.
    """
    resolved = _resolve_stock_unique(stock_query)
    rows = resolved["rows"]
    if not rows:
        return (
            f"ERROR: No stock matched '{stock_query}' in local stocks table.\n"
            "Please update stocks first or provide a more precise code/name."
        )
    if len(rows) != 1:
        lines = [
            "ERROR: Stock query is ambiguous.",
            f"Query: {stock_query}",
            "Candidates:",
        ]
        for code, name, suspended in rows[:10]:
            tag = "停牌" if suspended else "正常"
            lines.append(f"- {code} {name} ({tag})")
        lines.append("Please retry with the exact code.")
        return "\n".join(lines)

    try:
        return na.get_stock_news_report(
            stock_name=resolved["name"],
            stock_code=resolved["code"],
            max_items=max_items,
        )
    except Exception as e:
        return f"ERROR: 个股新闻面获取失败: {e}"


# =====================================================
# Tool 11: 财报深度分析（新增）
# =====================================================

@mcp.tool()
def earnings_analysis(
    stock_code: str,
    periods: int = 8,
) -> str:
    """
    对某只A股进行财报深度分析，结合历史多期财务数据和历次财报公告的市场反应。

    【何时调用】
      - 当 stock_announcements 发现近期有年报/季报/业绩预告时，必须紧接着调用此工具
      - 当用户明确关心财报质量、业绩趋势、或财报是否超预期时

    分析内容：
      ① 历史财务趋势（最近 N 期）
           营业收入 / 同比增长率
           归母净利润 / 同比增长率
           毛利率、ROE 趋势
      ② 历次财报公告的市场反应
           公告当日涨跌幅 + 成交量倍数
           公告前5日 vs 后3日累计涨跌
           市场反应定性（强烈正面/正面/中性/负面/强烈负面）
      ③ 对当前分析的影响
           财报超预期 → news_sentiment +0.2~+0.3
           符合预期   → news_sentiment 不调整
           低于预期   → news_sentiment -0.2~-0.3

    Parameters
    ----------
    stock_code : str   股票代码，如 "000423"（东阿阿胶）
    periods    : int   分析历史期数，默认 8 期（约 2 年）

    Returns
    -------
    完整中文财报分析报告，含财务趋势表、历次市场反应、news_sentiment 调整建议。
    """
    # 获取股票名称
    try:
        rt = st.get_realtime([stock_code])
        name = rt.iloc[0]["name"] if not rt.empty else stock_code
    except Exception:
        name = stock_code

    # ── 历史财务数据 ──
    try:
        history = st.get_financial_history(stock_code, periods=periods)
    except Exception as e:
        return f"ERROR: 历史财务数据获取失败: {e}"

    if not history:
        return "ERROR: 未获取到历史财务数据，请检查股票代码或稍后重试"

    # ── 公告列表，找财报类公告 ──
    try:
        anns = st.get_announcements(stock_code, page_size=30)
        earnings_anns = [a for a in anns if a["is_earnings"]][:6]
    except Exception:
        earnings_anns = []

    # ── 拉足够长的K线（一次请求复用于所有反应分析） ──
    kline_df = None
    if earnings_anns:
        try:
            from datetime import datetime, timedelta
            oldest = earnings_anns[-1]["date"] if earnings_anns else ""
            start_dt = (
                (datetime.strptime(oldest, "%Y-%m-%d") - timedelta(days=40)).strftime("%Y-%m-%d")
                if oldest else
                (datetime.today() - timedelta(days=600)).strftime("%Y-%m-%d")
            )
            kline_df = st.get_kline(stock_code, period="daily",
                                    start=start_dt, adjust="none")
        except Exception:
            kline_df = None

    # ── 逐条分析财报市场反应 ──
    reactions = []
    for ann in earnings_anns:
        if ann["date"]:
            r = st.analyze_earnings_reaction(stock_code, ann["date"], kline_df)
            r["title"] = ann["title"]
            reactions.append(r)

    return st.format_earnings_analysis(stock_code, name, history, reactions)


# =====================================================
# Tool 12: 个股公告查询（新增）
# =====================================================

@mcp.tool()
def stock_announcements(
    stock_code: str,
    count: int = 20,
) -> str:
    """
    查询某只A股最近的公告列表，自动识别并高亮财报/业绩相关公告。

    【重要用途】在做股票分析之前，先调用此工具确认：
      - 是否已披露最新年报/季报/业绩预告
      - 财报披露日期（避免用旧财务数据做估值判断）
      - 是否有重大公告影响当前走势

    智能识别以下财报类公告并给出提示：
      年报 / 半年报 / 季报（一季报/三季报）/ 业绩预告 / 业绩快报 / 业绩修正

    Parameters
    ----------
    stock_code : str   股票代码，如 "000423"（东阿阿胶）
    count      : int   返回最近 N 条公告，默认 20

    Returns
    -------
    中文公告列表，包含：
      - 财报状态提示（是否已披露、披露日期）
      - 全部公告列表（财报类用★标注）
      - 财报数据是否可能尚未更新的风险提示
    """
    try:
        anns = st.get_announcements(stock_code, page_size=count)
    except Exception as e:
        return f"ERROR: 公告获取失败: {e}"

    # 尝试获取股票名称
    try:
        rt = st.get_realtime([stock_code])
        name = rt.iloc[0]["name"] if not rt.empty else ""
    except Exception:
        name = ""

    return st.format_announcements(stock_code, anns, name=name)


# =====================================================
# Tool 13: 市场消息面环境日报（新增）
# =====================================================

@mcp.tool()
def market_news() -> str:
    """
    获取当日市场消息面环境分析报告（中文）。

    【建议在分析任何个股之前优先调用此工具】，以了解当日市场背景。

    内容涵盖：
      外围市场行情 —— 美股三大指数（标普/纳指/道指）+ 恐慌指数VIX、
                      欧股三大（富时/DAX/CAC）、亚太（恒生/日经）、黄金/原油
      国际重要新闻 —— 路透社 RSS 实时头条（地缘政治、科技事件、宏观经济等）
      国内官方媒体 —— 新华社、人民日报、央视财经最新消息
                      （注：国内只看政府官方媒体，不看民间财经号）
      综合情绪研判 —— 基于外围行情+新闻关键词给出 news_sentiment 建议参考值

    Returns
    -------
    完整中文市场环境日报。
    报告末尾会给出"综合建议 news_sentiment 参考值"，
    可直接或微调后填入 stock_diagnosis 的 news_sentiment 参数。

    注意：本工具需要抓取网络数据，执行时间约 20-30 秒，请耐心等待。
    """
    try:
        return na.get_market_news_report()
    except Exception as e:
        return f"ERROR: 消息面分析获取失败: {e}"


# =====================================================
# Tool 14: 量化资金活跃度分析（新增）
# =====================================================

@mcp.tool()
def quant_activity(
    stock_code: str,
    tick_n: int = 1500,
) -> str:
    """
    分析某只A股的量化资金参与程度（需在交易时段调用）。

    通过三个维度的实时公开数据推断量化资金活跃度:
      ① 均单手数     — 量化大量拆单 → 单笔成交手数极小（数据: 东方财富逐笔成交）
      ② 微单笔数占比 — 量化倾向≤5手微单 → 微单笔数占全部笔数的比例
      ③ 分钟成交均匀度 — 量化VWAP策略 → 全天成交量趋于均匀（CV低）

    综合输出 0-100 的量化活跃度得分:
      80-100: 量化主导  — 算法/高频资金为主要参与方
      62-79:  量化活跃  — 量化与散户/游资混合博弈
      42-61:  量化参与  — 有量化参与但非主导
      0-41:   量化偏少  — 散户/游资情绪交易为主

    并根据得分给出针对性的操作建议（止损策略/追涨策略/下单方式等）。

    Parameters
    ----------
    stock_code : str   股票代码，如 "600519"
    tick_n     : int   分析最近 N 笔逐笔成交，默认 1500（越大越准确，但耗时稍长）

    Returns
    -------
    中文分析报告，包含三维度得分、买卖力道分析、针对性操作建议。

    注意: 逐笔数据仅在交易时段（09:30-15:00）有效，非交易时段调用会提示无数据。
    """
    try:
        return qd.get_quant_activity_report(stock_code, tick_n=tick_n)
    except Exception as e:
        return f"ERROR: 量化活跃度分析失败: {e}"


# =====================================================
# Start
# =====================================================

if __name__ == "__main__":
    mcp.run(transport="stdio")
