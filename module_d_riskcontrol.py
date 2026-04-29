"""
模块 D — 风控参数建议（每笔交易必做）
=====================================
架构位置: 第4层 — 最终风控（两条路径最终汇合）

职责:
  1. 调用 risk_assessment → VaR/CVaR/Kelly/ATR止损止盈
  2. 结合评分卡结论，输出风控参数建议
  3. 支持短线 (ATR止损) 和长线 (VaR/Kelly) 两种风控模式

输出:
  RiskControlResult dict:
    - risk_report: str           risk_assessment 原始报告
    - stop_loss_pct: float       止损百分比（负数）
    - take_profit_pct: float     止盈百分比（正数）
    - kelly_position_pct: float  Kelly仓位上限百分比
    - var_95_pct: float          VaR(95%)日损失百分比
    - risk_reward_ratio: float   风险收益比
    - position_advice: str       仓位建议文本
"""

import re
from typing import TypedDict, Optional
from common_utils import _extract_float, _module_header


class RiskControlResult(TypedDict):
    risk_report: str
    stop_loss_pct: Optional[float]
    take_profit_pct: Optional[float]
    kelly_position_pct: Optional[float]
    var_95_pct: Optional[float]
    risk_reward_ratio: Optional[float]
    position_advice: str


def run_risk_control(
    stock_code: str,
    risk_assessment_fn=None,
    analysis_days: int = 250,
    scorecard: dict = None,
) -> RiskControlResult:
    """
    执行风控参数计算。

    Parameters
    ----------
    stock_code : str             股票代码
    risk_assessment_fn : callable risk_assessment MCP tool function
    analysis_days : int          分析历史天数
    scorecard : ScorecardResult  评分卡结果，用于调整仓位建议
    """
    # ── 调用 risk_assessment ──
    risk_report = ""
    if risk_assessment_fn:
        try:
            risk_report = risk_assessment_fn(stock_code=stock_code, analysis_days=analysis_days)
        except Exception as e:
            risk_report = f"risk_assessment 获取失败: {e}"

    # ── 从报告中提取参数 ──
    stop_loss = _extract_float(risk_report, r"(?:止损|Stop Loss)[^%\d]*(-?\d+(?:\.\d+)?)%")
    take_profit = _extract_float(risk_report, r"(?:止盈|Take Profit)[^%\d]*\+?(\d+(?:\.\d+)?)%")
    kelly = _extract_float(risk_report, r"Kelly[^%\d]*(\d+(?:\.\d+)?)%")
    var95 = _extract_float(risk_report, r"VaR\s*\(?95%?\)?[^%\d]*(-?\d+(?:\.\d+)?)%")
    rr = _extract_float(risk_report, r"(?:风险收益比|Risk.*?Reward)[^:\d]*1[：:](\d+(?:\.\d+)?)")

    # 如果 stop_loss 是正数，转成负数
    if stop_loss is not None and stop_loss > 0:
        stop_loss = -stop_loss

    # 止损安全检查：提取失败时基于评分卡收紧或放宽默认值
    # 高分标的（>70）容忍宽止损，低分标的（<40）用窄止损
    if stop_loss is None or stop_loss == 0 or not (-50.0 < stop_loss < 0):
        if scorecard is not None:
            avg_score = (scorecard.get("short_term_score", 50) + scorecard.get("long_term_score", 50)) / 2
            if avg_score >= 70:
                stop_loss = -10.0   # 高质量标的，允许更大波动空间
            elif avg_score >= 40:
                stop_loss = -8.0    # 中性，A股平均ATR基准
            else:
                stop_loss = -6.0    # 低分标的，收窄止损控制尾部风险
        else:
            stop_loss = -8.0
    # 止盈安全检查：与止损保持至少 1.5:1 风险收益比
    if take_profit is None or take_profit <= 0:
        take_profit = round(abs(stop_loss) * 1.5, 1)

    # ── 基于评分卡调整仓位建议 ──
    position_advice = _compute_position_advice(scorecard, kelly, var95)

    return RiskControlResult(
        risk_report=risk_report,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit,
        kelly_position_pct=kelly,
        var_95_pct=var95,
        risk_reward_ratio=rr,
        position_advice=position_advice,
    )


def _compute_position_advice(
    scorecard: Optional[dict],
    kelly: Optional[float],
    var95: Optional[float],
) -> str:
    """基于评分卡 + Kelly + VaR 综合给出仓位建议。"""
    if scorecard is None:
        if kelly is not None:
            return f"Kelly建议仓位上限: {kelly:.1f}%"
        return "仓位建议需结合评分卡结果"

    short_score = scorecard.get("short_term_score", 50)
    long_score = scorecard.get("long_term_score", 50)
    rating = scorecard.get("market_rating", "M")
    vetoed = scorecard.get("short_vetoed", False) or scorecard.get("long_vetoed", False)

    # 基础仓位 (取 Kelly 或基于评分计算)
    if kelly is not None:
        base_pct = kelly
    else:
        avg_score = (short_score + long_score) / 2
        # 低于40分的标的建议空仓，不设5%下限以避免强推弱势股
        if avg_score < 40:
            base_pct = 0.0
        else:
            base_pct = min(30, (avg_score - 40) / 60 * 25 + 5)

    # 环境修正
    if rating == "L":
        base_pct *= 0.5
        env_note = "，悲观环境减半"
    elif rating == "H":
        base_pct *= 1.2
        env_note = "，乐观环境上浮20%"
    else:
        env_note = ""

    # 一票否决修正
    if vetoed:
        base_pct = min(base_pct, 5)
        env_note += "，存在一票否决项"

    # VaR(95%)>8% 极端波动 → 仓位上限×0.7（prompt 规则落地）
    var_note = ""
    if var95 is not None and abs(var95) > 8:
        base_pct *= 0.7
        var_note = "，VaR>8%极端波动×0.7"

    base_pct = min(30, max(0, base_pct))

    parts = [f"建议仓位上限: {base_pct:.1f}%{env_note}{var_note}"]

    if short_score >= 70 and long_score >= 70:
        parts.append("短线+长线均偏强，可考虑分批建仓")
    elif short_score >= 70 and long_score < 50:
        parts.append("短线可关注但长线偏弱，适合短线快进快出")
    elif short_score < 50 and long_score >= 70:
        parts.append("长线优质但短线时机未到，可等待回调再介入")
    elif short_score < 40 and long_score < 40:
        parts.append("短线长线均不佳，建议观望")

    return "；".join(parts)




def format_risk_control(rc: RiskControlResult, stock_name: str = "", stock_code: str = "") -> str:
    """格式化风控参数报告。"""
    def _fmt(v, suffix="%", default="N/A"):
        if v is None:
            return default
        return f"{v:.1f}{suffix}"

    # 风险收益比：优先使用工具返回值，回退到止损/止盈倒推
    rr = rc["risk_reward_ratio"]
    sl = rc["stop_loss_pct"]
    tp = rc["take_profit_pct"]
    if rr is None and sl and tp and sl < 0:
        rr = round(tp / abs(sl), 2)
    rr_str = f"1:{rr:.2f}" if rr is not None else "N/A"

    lines = [
        *_module_header("【模块 D】风控参数建议", stock_name, stock_code),
        f"  止损:       {_fmt(rc['stop_loss_pct'])}",
        f"  止盈目标:   +{_fmt(rc['take_profit_pct'])}",
        f"  风险收益比: {rr_str}",
        f"  Kelly仓位:  {_fmt(rc['kelly_position_pct'])}",
        f"  VaR(95%):   {_fmt(rc['var_95_pct'])}",
        "",
        f"  {rc['position_advice']}",
    ]
    return "\n".join(lines)
