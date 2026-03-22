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

# -- Initialize MCP server --
mcp = FastMCP(
    name="a-share-stock",
    instructions=(
        "你现在拥有完整的A股量化分析能力，工具清单：\n"
        "0.  market_news          — 【每日必先调用】外围市场行情+路透社RSS新闻+国内官方媒体，输出 news_sentiment 参考值\n"
        "1.  realtime_quote       — 获取A股实时行情\n"
        "2.  kline_data           — 获取K线数据与技术指标\n"
        "3.  financial_data       — 获取财务/估值指标\n"
        "4.  kline_chart          — 生成K线图（PNG）\n"
        "5.  batch_quote          — 批量行情概览\n"
        "6.  stock_diagnosis      — 全维度量化诊断（核心工具）\n"
        "7.  strategy_backtest    — 策略回测\n"
        "8.  risk_assessment      — 风险评估\n"
        "9.  stock_screener       — 全市场选股扫描\n"
        "10. earnings_analysis    — 【有财报时必调】历史多期财务趋势 + 历次财报市场反应分析\n"
        "11. stock_announcements  — 【个股分析必调】查询公告列表，识别财报披露状态\n"
        "12. market_news          — 消息面环境日报（RSS数据，可能有数小时延迟）\n"
        "13. quant_activity       — 【交易时段专用】量化资金活跃度分析，输出0-100得分+操作建议\n\n"

        "⚠️⚠️⚠️ 【关于新闻时效性——极其重要】\n"
        "market_news 工具依赖 RSS 订阅源，存在数小时甚至更长的延迟，不代表最新实时信息。\n"
        "以下情况，你必须在调用 market_news 之后，额外主动用 web search 搜索最新信息：\n"
        "  1. 涉及行业会议/展会: 如 OFC光纤通信大会、CES、MWC、AI大会等，\n"
        "     须搜索'[会议名称] [当前年月] 结果 最新'确认会议是否已结束、有无重大发布\n"
        "  2. 涉及地缘政治/突发事件: 战争、制裁、关税等实时变化快，必须搜索最新进展\n"
        "  3. 涉及科技圈重大发布: 芯片新品、AI模型发布、禁令等，须搜索当天最新动态\n"
        "  4. 涉及A股重大政策: 降准/降息/证监会表态等，须搜索当日最新官方表述\n"
        "  5. 当用户明确提到某个近期事件: 必须搜索确认其最新状态（已结束/进行中/有新进展）\n"
        "  ⛔ 不允许用 RSS 旧数据代替实时搜索，否则会给出过期判断（例如把已结束的会议当成利好催化剂）\n\n"

        "【完整分析流程】用户询问'要不要买某只股票'时，按以下顺序调用：\n"
        "  第一步: market_news         → 获取外围市场行情和 RSS 新闻，得到 news_sentiment 基础值\n"
        "             ↳ 立即用 web search 补充搜索：当日最新宏观/行业/事件动态，修正 RSS 的时效盲区\n"
        "  第二步: stock_announcements → ⚠️ 必查公告！确认是否近期有财报/业绩预告披露\n"
        "             ↳ 若有财报: 调用 earnings_analysis，根据结果调整 news_sentiment\n"
        "             ↳ 若有重大公告(回购/重组/定增等): 用 web search 搜索市场反应，纳入判断\n"
        "  第三步: realtime_quote      → 查看当前价格\n"
        "  第四步: quant_activity      → 判断量化资金参与程度（决定操作策略方向）\n"
        "  第五步: kline_data          → 获取K线与技术指标\n"
        "  第六步: financial_data      → 获取财务指标（财报刚出时可能滞后，以earnings_analysis为准）\n"
        "  第七步: stock_diagnosis     → 综合诊断（news_sentiment = 外围情绪 + 财报调整 + 实时搜索修正）\n"
        "  第八步: risk_assessment     → 风险评估\n\n"

        "【web search 搜索规范】\n"
        "  - 搜索语言: 国际事件用英文搜索（结果更全），国内政策用中文搜索\n"
        "  - 时间限定: 搜索词加上当前年月，如'OFC 2026 March results'避免拿到旧文章\n"
        "  - 核实来源: 优先采信路透社/彭博/官方网站，不采信股评类自媒体\n"
        "  - 国内政策: 只看新华社/人民日报/证监会/央行官网，不看民间财经号\n\n"

        "【earnings_analysis 财报影响说明】\n"
        "  - 强烈正面/正面: news_sentiment 在外围基础上 +0.2~+0.3\n"
        "  - 中性: news_sentiment 不调整\n"
        "  - 负面/强烈负面: news_sentiment 在外围基础上 -0.2~-0.3，等待充分定价再入场\n\n"

        "【quant_activity 等级说明】\n"
        "  - 量化主导(80+): 顺势持仓、日线操作、不追涨停、市价单成交\n"
        "  - 量化活跃(62+): 混合博弈，区间操作，不追板\n"
        "  - 量化参与(42+): 常规技术/消息策略有效\n"
        "  - 量化偏少(<42): 散户情绪主导，题材/消息驱动明显\n\n"

        "【选股流程 — 完整三阶段流程，必须严格执行】\n"
        "用户询问'有哪些股票值得买/下周选股/推荐几支'时，按以下三阶段执行：\n\n"

        "▌第一阶段：市场环境评估\n"
        "  Step 1: market_news → 获取外围行情+RSS新闻，得到 news_sentiment 基础值\n"
        "          ↳ 立即 web search 补充搜索当日最新宏观/政策/行业动态，修正 RSS 时效盲区\n\n"

        "▌第二阶段：多策略海选（并行跑全部4个策略，缺一不可）\n"
        "  Step 2a: stock_screener(strategy='value')     → 低PE低PB价值股候选\n"
        "  Step 2b: stock_screener(strategy='momentum')  → 强趋势量能扩张动量股候选\n"
        "  Step 2c: stock_screener(strategy='oversold')  → 深度超跌均值回归候选\n"
        "  Step 2d: stock_screener(strategy='potential') → 跌深但基本面完好的底部潜力候选\n"
        "  → 四份结果合并去重，综合排名取前10~15支候选股\n\n"

        "▌第三阶段：逐股深度多维分析（对每支候选股依次执行）\n"
        "  Step 3a: stock_announcements → 确认近期是否有财报/业绩预告/重大公告\n"
        "           ↳ 有财报/业绩预告 → 必须调用 earnings_analysis，根据结果调整 news_sentiment\n"
        "           ↳ 有回购/重组/定增等重大公告 → web search 搜索最新市场反应，纳入判断\n"
        "  Step 3b: financial_data → 估值维度：PE/PB/ROE/毛利率/净利率/市值\n"
        "  Step 3c: kline_data     → 技术维度：MA/MACD/RSI/BOLL/KDJ趋势判断\n"
        "  Step 3d: quant_activity → 量化资金维度：0-100分，判断量化主导/活跃/参与程度\n"
        "           ↳ 非交易时段此步骤跳过\n"
        "  Step 3e: stock_diagnosis(news_sentiment=最终调整值) → 7维量化综合评分(-100~+100)\n"
        "           含：趋势/动量/波动率/均值回归/量价关系/支撑阻力/统计特征 + 蒙特卡洛概率\n"
        "  Step 3f: risk_assessment → 风险维度：VaR/CVaR/Kelly仓位/ATR动态止损位\n\n"

        "▌第四阶段：综合排名输出\n"
        "  综合以下6个维度加权打分，输出最终推荐Top 5：\n"
        "    ① stock_diagnosis综合分（量化技术面，权重35%）\n"
        "    ② 估值吸引力：financial_data PE/PB/ROE综合（权重20%）\n"
        "    ③ 财报质量：earnings_analysis趋势与市场反应（权重20%）\n"
        "    ④ 风险度：risk_assessment VaR/Kelly比值（权重15%）\n"
        "    ⑤ 市场情绪：news_sentiment最终值（权重10%）\n"
        "  每支股票输出：推荐理由(覆盖上述维度)、建议入场区间、止损位、风险提示\n\n"
        "news_sentiment 最终值 = RSS外围情绪得分 + 实时搜索修正 + 财报调整，范围 -1.0 ~ +1.0\n"
        "数据来源: 东方财富/新浪财经（免费，无需API密钥）\n"
        "免责声明: 所有分析基于模型计算，不构成投资建议。"
    ),
)

# =====================================================
# Tool 1: Real-time Quote
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

    try:
        df = st.get_kline(stock_code, period="daily", start=start, adjust="qfq")
    except Exception as e:
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

    # Build detailed output
    lines = [result["summary"]]

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

    return result["summary"]


# =====================================================
# Tool 10: 财报深度分析（新增）
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
# Tool 11: 个股公告查询（新增）
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
# Tool 11: 市场消息面环境日报（新增）
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
# Tool 11: 量化资金活跃度分析（新增）
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
