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
from common_utils import (
    _extract_float, _extract_float_warn,
    _has_keyword_unaffirmed, _module_header,
)


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

    # 量价配合
    volume_breakout: Optional[bool]  # True=放量确认突破, False=缩量警告, None=无信息

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
        # 弹性正则：匹配 "+0.2 ~ +0.3" / "+0.2~+0.3" / "+0.20 ～ +0.30" 等变体
        m_pos = re.search(
            r'建议[^+\-\d]{0,15}\+\s*(\d+\.\d+)\s*[~～]\s*\+\s*(\d+\.\d+)', report
        )
        m_neg = re.search(
            r'建议[^+\-\d]{0,15}-\s*(\d+\.\d+)\s*[~～]\s*-\s*(\d+\.\d+)', report
        )
        if m_pos:
            sentiment = (float(m_pos.group(1)) + float(m_pos.group(2))) / 2
        elif m_neg:
            sentiment = -((float(m_neg.group(1)) + float(m_neg.group(2))) / 2)
        return report, round(sentiment, 3)
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

    # ── 催化剂强度（否定语义感知）──
    # 负面优先：正负混报时保守处理；每个关键词均检查前8字是否有否定词
    _NEG_WORDS  = ("利空", "负面", "风险提示", "警示", "问询函", "诉讼", "立案调查")
    _STR_WORDS  = ("涨停", "重大利好", "大幅超预期", "重组", "并购")
    _MED_WORDS  = ("政策", "行业利好", "预期改善", "受益")

    catalyst_type = "weak"
    catalyst_detail = "无明显催化剂"
    if _has_keyword_unaffirmed(news_report, _NEG_WORDS):
        catalyst_type = "negative"
        catalyst_detail = "利空公告/负面消息"
    elif _has_keyword_unaffirmed(news_report, _STR_WORDS):
        catalyst_type = "strong"
        catalyst_detail = "当日公告/涨停催化"
    elif _has_keyword_unaffirmed(news_report, _MED_WORDS):
        catalyst_type = "medium"
        catalyst_detail = "行业政策/预期改善"

    # ── 主力资金信号（带失败监控）──
    net_inflow_ratio = _extract_float_warn(
        mf_report, r"大单净流入比[：:]\s*([+-]?\d+(?:\.\d+)?)%", label="net_inflow_ratio"
    )
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
    if _has_keyword_unaffirmed(kline_report, ("多头排列",)) or "bullish" in kline_report.lower():
        ma_alignment = "bullish"
    elif _has_keyword_unaffirmed(kline_report, ("空头排列",)) or "bearish" in kline_report.lower():
        ma_alignment = "bearish"

    # MACD 信号：两者同时出现时死叉优先（保守）
    macd_signal = "neutral"
    _kr_lower = kline_report.lower()
    _has_golden = "MACD金叉" in kline_report or "golden cross" in _kr_lower
    _has_death  = "MACD死叉" in kline_report or "death cross" in _kr_lower
    if _has_death:
        macd_signal = "death_cross"
    elif _has_golden:
        macd_signal = "golden_cross"

    # RSI：优先取12日，依次回退到6日，再回退诊断报告
    # 修复原模式 RSI12?（只匹配 RSI1/RSI12，漏掉 RSI6/RSI24）
    rsi_val = _extract_float_warn(
        kline_report, r"RSI\s*1[26]\s*[：:]\s*(\d+(?:\.\d+)?)", label="rsi_kline_12"
    )
    if rsi_val is None:
        rsi_val = _extract_float(kline_report, r"RSI\s*\d*\s*[：:]\s*(\d+(?:\.\d+)?)")
    if rsi_val is None:
        rsi_val = _extract_float_warn(
            diagnosis_report, r"RSI\s*\d*\s*[：:]\s*(\d+(?:\.\d+)?)", label="rsi_diagnosis"
        )

    kdj_j    = _extract_float_warn(diagnosis_report, r"J值[：:]\s*(\d+(?:\.\d+)?)", label="kdj_j")
    kdj_cross = "J值上穿50" in diagnosis_report

    # ── 量价配合信号（从 kline_report 文本提取）──
    # 放量确认突破时 True，缩量警告时 False，无信息时 None
    _VOL_UP   = ("放量", "量能配合", "成交量放大", "量增价涨", "有效放量")
    _VOL_DOWN = ("缩量", "量能萎缩", "成交量萎缩", "无量上涨", "无量突破")
    if _has_keyword_unaffirmed(kline_report, _VOL_UP):
        volume_breakout: Optional[bool] = True
    elif _has_keyword_unaffirmed(kline_report, _VOL_DOWN):
        volume_breakout = False
    else:
        volume_breakout = None

    # ── 量化诊断信号（带失败监控）──
    diag_score = _extract_float_warn(
        diagnosis_report, r"综合(?:评)?分[：:]\s*([+-]?\d+(?:\.\d+)?)", label="diag_score"
    )
    if diag_score is not None:
        # -100~+100 → 0~100
        diag_score = max(0, min(100, (diag_score + 100) / 2))

    mc_up     = _extract_float(diagnosis_report, r"上涨概率[：:]\s*(\d+(?:\.\d+)?)%")
    quant_pct = _extract_float(quant_report,     r"量化资金占比[：:]\s*(\d+(?:\.\d+)?)%")

    hurst_val = _extract_float(diagnosis_report, r"H=(\d+\.\d+)")
    if hurst_val is None:
        hurst_val = _extract_float(diagnosis_report, r"[Hh]urst[^:：]*[：:]\s*(\d+\.\d+)")

    # ── 融资融券情绪（带失败监控）──
    margin_change = None
    if margin_report:
        margin_change = _extract_float_warn(
            margin_report, r"周环比[：:]\s*([+-]?\d+(?:\.\d+)?)%", label="margin_change"
        )

    nb_same_dir = northbound_adj > 0

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
        volume_breakout=volume_breakout,
        diagnosis_score=diag_score,
        monte_carlo_up_prob=mc_up,
        quant_share_pct=quant_pct,
        margin_balance_change_pct=margin_change,
        northbound_same_direction=nb_same_dir,
        hurst=hurst_val,
    )



def format_shortterm_report(result: ShortTermResult, stock_name: str, stock_code: str) -> str:
    """格式化短线分析报告。"""
    sections = [
        *_module_header("【模块 B1】短线分析", stock_name, stock_code),
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
