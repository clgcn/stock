"""
模块 B1 — 短线分析 Pipeline
============================
架构位置: 第2层（情绪+资金） + 第3层（技术诊断）

短线路径调用链:
  1. realtime_quote / batch_quote  → 实时行情快照
  2. stock_news                    → 催化剂确认
  3. moneyflow + margin_trading    → 主力资金 + 融资融券
  4. kline_data (MA/MACD/KDJ)      → 技术形态
  5. stock_diagnosis (7维诊断)     → 量化诊断
  6. quant_activity (交易时段)     → 量化资金活跃度

重点: 量价 + 情绪 + 形态三者共振
总调用: 6-7个工具，当日完成

输出: ShortTermResult dict，包含各维度原始数据 + 用于评分卡的结构化信号
"""

import re
from typing import TypedDict, Optional


class ShortTermSignals(TypedDict):
    """短线各维度的结构化信号，供评分卡使用。"""
    # 催化剂强度 (权重25%)
    catalyst_type: str          # "strong" / "medium" / "weak" / "negative"
    catalyst_detail: str

    # 主力资金 (权重30%)
    net_inflow_ratio: Optional[float]   # 大单净流入比 = 净流入/成交额
    consecutive_outflow_days: int       # 连续净流出天数
    margin_trend: str                   # "up" / "flat" / "down" / "unknown"

    # 技术形态 (权重25%)
    ma_alignment: str           # "bullish" / "bearish" / "mixed"
    macd_signal: str            # "golden_cross" / "death_cross" / "expanding" / "contracting" / "neutral"
    rsi_value: Optional[float]
    kdj_j_value: Optional[float]
    kdj_cross_50: bool          # J值是否上穿50

    # 量化诊断 (权重10%)
    diagnosis_score: Optional[float]    # 0-100 综合评分
    monte_carlo_up_prob: Optional[float]
    quant_share_pct: Optional[float]    # 量化资金占比
    hurst: Optional[float]              # Hurst指数 (≥0.55趋势市 / ≤0.45均值回归)

    # 融资融券情绪 (权重10%)
    margin_balance_change_pct: Optional[float]  # 融资余额周环比变化
    northbound_same_direction: bool              # 北向当日净买入是否同向


class ShortTermResult(TypedDict):
    """短线分析完整结果。"""
    # 原始报告
    realtime_report: str
    stock_news_report: str
    moneyflow_report: str
    margin_report: Optional[str]
    kline_report: str
    diagnosis_report: str
    quant_report: str

    # 结构化信号
    signals: ShortTermSignals

    # 个股情绪
    stock_sentiment: float
    earnings_sentiment: float


def run_shortterm_pipeline(
    stock_code: str,
    stock_name: str,
    combined_base_sentiment: float,
    northbound_adj: float,
    # MCP tool functions
    realtime_quote_fn=None,
    stock_news_fn=None,
    stock_announcements_fn=None,
    earnings_analysis_fn=None,
    moneyflow_fn=None,
    margin_trading_fn=None,
    kline_data_fn=None,
    stock_diagnosis_fn=None,
    quant_activity_fn=None,
    # Options
    analysis_days: int = 250,
    monte_carlo_days: int = 20,
    include_margin_trading: bool = False,
    include_quant_activity: bool = True,
) -> ShortTermResult:
    """
    执行短线分析完整 pipeline。

    Parameters
    ----------
    stock_code : str              股票代码
    stock_name : str              股票名称
    combined_base_sentiment : float 模块A输出的合并基线情绪
    northbound_adj : float        北向资金修正值
    *_fn : callable               各MCP工具函数
    """
    def _safe(fn, *a, label="", **kw):
        if fn is None:
            return f"⏸ {label} 未提供"
        try:
            return fn(*a, **kw)
        except Exception as e:
            return f"{label} 获取失败: {e}"

    # ── 1. 实时行情 ──
    rt_report = _safe(realtime_quote_fn, stock_code, label="realtime_quote")

    # ── 2. 个股新闻（催化剂确认）──
    news_report = _safe(stock_news_fn, stock_query=stock_code, max_items=8, label="stock_news")
    stock_sentiment = _extract_stock_sentiment(news_report)

    # ── 2b. 公告面（用于触发 earnings_analysis）──
    ann_report = _safe(stock_announcements_fn, stock_code, label="stock_announcements")
    earnings_report, earnings_sentiment = _check_earnings_trigger(
        ann_report, stock_code, earnings_analysis_fn
    )

    # ── 3. 主力资金 + 融资融券 ──
    mf_report = _safe(moneyflow_fn, stock_code=stock_code, days=10, label="moneyflow")
    margin_report = None
    if include_margin_trading:
        margin_report = _safe(margin_trading_fn, stock_code=stock_code, days=20, label="margin_trading")

    # ── 4. K线数据（MA/MACD/RSI/BOLL/KDJ）──
    kline_report = _safe(
        kline_data_fn,
        stock_code=stock_code,
        period="daily",
        recent_days=60,
        with_indicators=True,
        label="kline_data",
    )

    # ── 5. 量化诊断（7维 + KDJ + K线形态 + 蒙特卡洛）──
    total_sentiment = max(-1.0, min(1.0,
        combined_base_sentiment + stock_sentiment + earnings_sentiment
    ))
    diagnosis_report = _safe(
        stock_diagnosis_fn,
        stock_code=stock_code,
        analysis_days=analysis_days,
        news_sentiment=total_sentiment,
        monte_carlo_days=monte_carlo_days,
        label="stock_diagnosis",
    )

    # ── 6. 量化资金活跃度 ──
    quant_report = "⏸ quant_activity 已跳过（非交易时段或用户未启用）"
    if include_quant_activity:
        quant_report = _safe(quant_activity_fn, stock_code, label="quant_activity")

    # ── 提取结构化信号 ──
    signals = _extract_signals(
        news_report=news_report,
        mf_report=mf_report,
        margin_report=margin_report,
        kline_report=kline_report,
        diagnosis_report=diagnosis_report,
        quant_report=quant_report,
        northbound_adj=northbound_adj,
    )

    return ShortTermResult(
        realtime_report=rt_report,
        stock_news_report=news_report,
        moneyflow_report=mf_report,
        margin_report=margin_report,
        kline_report=kline_report,
        diagnosis_report=diagnosis_report,
        quant_report=quant_report,
        signals=signals,
        stock_sentiment=stock_sentiment,
        earnings_sentiment=earnings_sentiment,
    )


# ══════════════════════════════════════════════════════
# 内部辅助函数
# ══════════════════════════════════════════════════════

def _extract_stock_sentiment(news_report: str) -> float:
    """从 stock_news 报告中提取个股情绪微调值。"""
    m = re.search(r"个股新闻情绪微调：\s*([+-]?\d+(?:\.\d+)?)", news_report)
    return float(m.group(1)) if m else 0.0


def _check_earnings_trigger(ann_report: str, stock_code: str, earnings_fn) -> tuple:
    """检查公告中是否命中财报关键词，触发 earnings_analysis。"""
    keywords = ("年报", "半年报", "季报", "一季报", "三季报", "业绩预告", "业绩快报", "业绩修正")
    if not any(kw in ann_report for kw in keywords):
        return None, 0.0

    if earnings_fn is None:
        return None, 0.0

    try:
        report = earnings_fn(stock_code)
        sentiment = 0.0
        if "news_sentiment 建议在技术分析基础上 +0.2 ~ +0.3" in report:
            sentiment = 0.25
        elif "news_sentiment 建议在技术分析基础上 -0.2 ~ -0.3" in report:
            sentiment = -0.25
        return report, sentiment
    except Exception:
        return None, 0.0


def _extract_signals(
    news_report: str,
    mf_report: str,
    margin_report: Optional[str],
    kline_report: str,
    diagnosis_report: str,
    quant_report: str,
    northbound_adj: float,
) -> ShortTermSignals:
    """从各工具报告中提取结构化信号，供评分卡使用。"""

    # ── 催化剂强度 ──
    catalyst_type = "weak"
    catalyst_detail = ""
    if "涨停" in news_report or "重大利好" in news_report:
        catalyst_type = "strong"
        catalyst_detail = "当日公告/涨停催化"
    elif "政策" in news_report or "行业利好" in news_report:
        catalyst_type = "medium"
        catalyst_detail = "行业政策/预期改善"
    elif "利空" in news_report or "负面" in news_report:
        catalyst_type = "negative"
        catalyst_detail = "利空公告/负面消息"
    else:
        catalyst_detail = "无明显催化剂"

    # ── 主力资金信号 ──
    net_inflow_ratio = _extract_float(mf_report, r"大单净流入比[：:]\s*([+-]?\d+(?:\.\d+)?)%")
    consec_out = 0
    m_out = re.search(r"连续(\d+)日净流出", mf_report)
    if m_out:
        consec_out = int(m_out.group(1))

    margin_trend = "unknown"
    if margin_report:
        if "融资余额上升" in margin_report or "融资余额增加" in margin_report:
            margin_trend = "up"
        elif "融资余额下降" in margin_report or "融资余额减少" in margin_report:
            margin_trend = "down"
        else:
            margin_trend = "flat"

    # ── 技术形态信号 ──
    ma_alignment = "mixed"
    if "多头排列" in kline_report or "bullish" in kline_report.lower():
        ma_alignment = "bullish"
    elif "空头排列" in kline_report or "bearish" in kline_report.lower():
        ma_alignment = "bearish"

    macd_signal = "neutral"
    if "MACD金叉" in kline_report or "golden cross" in kline_report.lower():
        macd_signal = "golden_cross"
    elif "MACD死叉" in kline_report or "death cross" in kline_report.lower():
        macd_signal = "death_cross"

    rsi_val = _extract_float(kline_report, r"RSI12?[：:]\s*(\d+(?:\.\d+)?)")
    if rsi_val is None:
        rsi_val = _extract_float(diagnosis_report, r"RSI[：:]\s*(\d+(?:\.\d+)?)")

    kdj_j = _extract_float(diagnosis_report, r"J值[：:]\s*(\d+(?:\.\d+)?)")
    kdj_cross = "J值上穿50" in diagnosis_report

    # ── 量化诊断信号 ──
    diag_score = _extract_float(diagnosis_report, r"综合(?:评)?分[：:]\s*([+-]?\d+(?:\.\d+)?)")
    # 将 -100~+100 映射到 0~100
    if diag_score is not None:
        diag_score = max(0, min(100, (diag_score + 100) / 2))

    mc_up = _extract_float(diagnosis_report, r"上涨概率[：:]\s*(\d+(?:\.\d+)?)%")

    quant_pct = _extract_float(quant_report, r"量化资金占比[：:]\s*(\d+(?:\.\d+)?)%")

    # Hurst指数从诊断报告中提取，格式: "H=0.600" 或 "Hurst Exponent: 0.600"
    hurst_val = _extract_float(diagnosis_report, r"H=(\d+\.\d+)")
    if hurst_val is None:
        hurst_val = _extract_float(diagnosis_report, r"[Hh]urst[^:：]*[：:]\s*(\d+\.\d+)")

    # ── 融资融券情绪 ──
    margin_change = None
    if margin_report:
        margin_change = _extract_float(margin_report, r"周环比[：:]\s*([+-]?\d+(?:\.\d+)?)%")

    nb_same_dir = northbound_adj > 0  # 北向当日净买入同向

    return ShortTermSignals(
        catalyst_type=catalyst_type,
        catalyst_detail=catalyst_detail,
        net_inflow_ratio=net_inflow_ratio,
        consecutive_outflow_days=consec_out,
        margin_trend=margin_trend,
        ma_alignment=ma_alignment,
        macd_signal=macd_signal,
        rsi_value=rsi_val,
        kdj_j_value=kdj_j,
        kdj_cross_50=kdj_cross,
        diagnosis_score=diag_score,
        monte_carlo_up_prob=mc_up,
        quant_share_pct=quant_pct,
        margin_balance_change_pct=margin_change,
        northbound_same_direction=nb_same_dir,
        hurst=hurst_val,
    )


def _extract_float(text: str, pattern: str) -> Optional[float]:
    """从文本中用正则提取浮点数。"""
    m = re.search(pattern, text)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, IndexError):
            pass
    return None


def format_shortterm_report(result: ShortTermResult, stock_name: str, stock_code: str) -> str:
    """格式化短线分析报告。"""
    sections = [
        "",
        "=" * 60,
        f"【模块 B1】短线分析 — {stock_name}（{stock_code}）",
        "=" * 60,
        "",
        "▌ 实时行情", result["realtime_report"],
        "",
        "▌ 催化剂/新闻面", result["stock_news_report"],
        "",
        "▌ 主力资金流向", result["moneyflow_report"],
    ]
    if result["margin_report"]:
        sections.extend(["", "▌ 融资融券", result["margin_report"]])
    sections.extend([
        "",
        "▌ K线技术面（MA/MACD/RSI/BOLL/KDJ）", result["kline_report"],
        "",
        "▌ 量化诊断（7维+KDJ+形态+蒙特卡洛）", result["diagnosis_report"],
        "",
        "▌ 量化资金活跃度", result["quant_report"],
    ])
    return "\n".join(sections)
