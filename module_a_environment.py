"""
模块 A — 环境感知层（每次必跑）
==============================
架构位置: 第0层 — 任何分析前必调（全局基线）

职责:
  1. 调用 market_quote → 获取外围行情 + 计算市场情绪评分
  2. 调用 news_analyzer → 新闻面情绪微调
  3. 调用 capital_flow → 北向资金情绪修正
  4. 综合判定大盘情绪评级: H (乐观) / M (中性偏谨慎) / L (悲观)
  5. 若评级为 L，短线分析自动降权 30%

输出:
  EnvironmentResult dict:
    - market_data: dict          外围行情原始数据
    - market_score: float        外围行情情绪评分 [-1, +1]
    - news_delta: float          新闻面情绪微调 [-0.3, +0.3]
    - northbound_report: str     北向资金原始报告
    - northbound_adj: float      北向资金情绪修正值
    - combined_sentiment: float  合并后的情绪值
    - market_rating: str         "H" / "M" / "L"
    - short_term_weight: float   短线降权系数 (L=0.7, 其他=1.0)
    - rating_reason: str         评级理由
    - summary_text: str          格式化摘要文本
    - news_report: str           新闻标题列表（格式化文本）
"""

import re
from typing import TypedDict

import market_quote as mq
import news_analyzer as na
import capital_flow as cf


class EnvironmentResult(TypedDict):
    market_data: dict
    market_score: float
    news_delta: float
    northbound_report: str
    northbound_adj: float
    combined_sentiment: float
    market_rating: str        # "H" / "M" / "L"
    short_term_weight: float  # L=0.7, M/H=1.0
    rating_reason: str
    summary_text: str
    news_report: str


def assess_environment(
    include_market_quote: bool = True,
    include_news: bool = True,
    include_northbound: bool = True,
    northbound_days: int = 10,
) -> EnvironmentResult:
    """
    执行环境感知层全部逻辑，直接调用底层模块获取结构化数据。

    Parameters
    ----------
    include_market_quote : bool  是否获取外围行情
    include_news : bool          是否获取新闻面
    include_northbound : bool    是否获取北向资金
    northbound_days : int        北向资金查询天数
    """
    # ── 1. 外围市场行情 ──
    market_data = {}
    market_score = 0.0
    if include_market_quote:
        try:
            market_data = mq.get_foreign_markets()
            market_score = mq.compute_market_score(market_data)
        except Exception:
            pass  # 行情获取失败 → 评分保持 0

    # ── 2. 新闻面微调 ──
    news_delta = 0.0
    news_report = ""
    if include_news:
        try:
            intl, domestic = na.get_news_feeds()
            news_delta = na.news_sentiment_delta(intl, domestic)
            # 同时格式化新闻标题列表，供选股报告输出
            parts = []
            intl_block = na._fmt_news_block(intl, "【国际/全球财经新闻】")
            domestic_block = na._fmt_news_block(domestic, "【国内A股/财经新闻】")
            parts.append(intl_block)
            parts.append("")
            parts.append(domestic_block)
            total = len(intl) + len(domestic)
            parts.append("")
            parts.append(f"  有效新闻: {total} 条  情绪微调值: {news_delta:+.2f}")
            news_report = "\n".join(parts)
        except Exception:
            news_report = "  （新闻获取失败）"

    # ── 3. 北向资金修正（基于成交额趋势，2024年5月后无净流入数据）──
    northbound_report = ""
    northbound_adj = 0.0
    consecutive_outflow = 0  # 保留字段兼容，改为"连续缩量天数"
    if include_northbound:
        try:
            flow = cf.get_northbound_flow(days=northbound_days)
            northbound_report = cf.format_northbound_report(flow)

            # 基于成交额趋势计算情绪修正
            if flow:
                amts = [d.get("total_deal_amt", 0) for d in flow]
                recent5 = amts[:5]
                early5 = amts[-5:] if len(amts) >= 10 else amts[:5]
                avg_recent = sum(recent5) / len(recent5) if recent5 else 0
                avg_early = sum(early5) / len(early5) if early5 else 0
                trend_pct = (avg_recent / avg_early - 1) * 100 if avg_early > 0 else 0

                # 成交额趋势 → 情绪修正
                if trend_pct > 20:
                    northbound_adj = 0.10   # 大幅放量，外资关注度显著提升
                elif trend_pct > 5:
                    northbound_adj = 0.05   # 温和放量
                elif trend_pct > -5:
                    northbound_adj = 0.0    # 持平
                elif trend_pct > -20:
                    northbound_adj = -0.05  # 温和缩量
                else:
                    northbound_adj = -0.10  # 大幅缩量，外资兴趣明显减退

                # 连续缩量天数（替代旧的连续流出天数）
                for i in range(len(amts) - 1):
                    if amts[i] < amts[i + 1]:
                        consecutive_outflow += 1
                    else:
                        break
        except Exception:
            northbound_report = "北向资金获取失败"

    # ── 4. 合并情绪 ──
    # 外围行情(权重60%) + 新闻微调(权重20%) + 北向修正(权重20%)
    combined = market_score + news_delta * 0.4 + northbound_adj
    combined = max(-1.0, min(1.0, round(combined, 2)))

    # ── 5. 大盘情绪评级 H/M/L ──
    rating, reason, weight = _compute_rating(
        combined, northbound_adj, consecutive_outflow, market_data
    )

    # ── 格式化摘要 ──
    summary = _format_summary(
        market_data, market_score, news_delta,
        northbound_adj, combined, rating, reason, weight
    )

    return EnvironmentResult(
        market_data=market_data,
        market_score=market_score,
        news_delta=news_delta,
        northbound_report=northbound_report,
        northbound_adj=northbound_adj,
        combined_sentiment=combined,
        market_rating=rating,
        short_term_weight=weight,
        rating_reason=reason,
        summary_text=summary,
        news_report=news_report,
    )


def _compute_rating(
    combined: float,
    northbound_adj: float,
    consecutive_outflow: int,
    market_data: dict,
) -> tuple:
    """
    综合判定 H/M/L 评级。

    H (乐观): combined >= 0.2 且北向非持续缩量
    L (悲观): combined <= -0.2 或北向连续5日以上缩量
    M (中性偏谨慎): 其他
    """
    reasons = []

    # 强烈悲观信号
    if consecutive_outflow >= 5:
        reasons.append(f"北向资金连续{consecutive_outflow}日成交缩量")
    if combined <= -0.3:
        reasons.append(f"综合情绪极低({combined:+.2f})")

    # 检查 VIX 恐慌
    us_items = market_data.get("美股", [])
    vix = next((i for i in us_items if "VIX" in i.get("name", "")), None)
    if vix and vix.get("price") and vix["price"] > 30:
        reasons.append(f"VIX恐慌指数={vix['price']:.1f}(>30)")

    if reasons or combined <= -0.2:
        if not reasons:
            reasons.append(f"综合情绪偏低({combined:+.2f})")
        return "L", "；".join(reasons), 0.7

    # 乐观信号
    if combined >= 0.2 and northbound_adj >= 0.05:
        return "H", f"综合情绪积极({combined:+.2f})，北向资金成交放量", 1.0
    if combined >= 0.2:
        return "H", f"综合情绪积极({combined:+.2f})", 1.0

    # 中性
    return "M", f"综合情绪中性({combined:+.2f})", 1.0


def _format_summary(
    market_data, market_score, news_delta,
    northbound_adj, combined, rating, reason, weight
) -> str:
    """格式化环境感知摘要。"""
    rating_labels = {"H": "乐观", "M": "中性偏谨慎", "L": "悲观"}
    sep = "=" * 56

    lines = [
        sep,
        "【环境感知层】",
        sep,
    ]

    # 外围行情概览
    if market_data:
        lines.append(mq.format_market_block(market_data))
        lines.append("")
        lines.append(mq.build_market_interpretation(market_data, market_score))
        lines.append("")

    lines.extend([
        f"  外围行情评分: {market_score:+.2f}  ({mq.score_to_label(market_score)})",
        f"  新闻面微调:   {news_delta:+.2f}",
        f"  北向资金修正: {northbound_adj:+.2f}",
        f"  ─────────────────────────",
        f"  合并情绪值:   {combined:+.2f}",
        "",
        f"  大盘情绪评级: {rating} ({rating_labels.get(rating, '未知')})",
        f"  评级理由: {reason}",
    ])
    if rating == "L":
        lines.append(f"  ⚠️ 悲观环境: 短线分析自动降权 {(1-weight)*100:.0f}%")

    lines.append(sep)
    return "\n".join(lines)


# ── 向后兼容 ──
def format_environment_summary(env: EnvironmentResult) -> str:
    """向后兼容接口，直接返回预生成的摘要。"""
    return env.get("summary_text", "")
