"""
评分卡 (Scorecard)
==================
汇总各面评分 → 加权总分 → 一票否决 → 仓位建议

短线权重:
  技术形态共振  25%
  主力资金净入  30%
  融资融券情绪  10%
  量化诊断      10%
  催化剂强度    25%
  (风控面独立约束，不参与加权)

长线权重:
  财务质量      30%
  估值安全边际  25%
  负债与商誉    20%
  成长质量      15%
  分红与治理    10%
  (风控面独立约束，不参与加权)

环境层修正:
  市场评级 H → 无修正
  市场评级 M → 短线×0.85
  市场评级 L → 短线×0.70
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from .face_technical import TechnicalFace, TechnicalSignals
from .face_capital import CapitalFace, CapitalSignals
from .face_catalyst import CatalystFace, CatalystSignals
from .face_fundamental import FundamentalFace, FundamentalSignals
from .face_risk import RiskFace, RiskSignals


# ══════════════════════════════════════════════════════
# 结果结构
# ══════════════════════════════════════════════════════

@dataclass
class DimensionScore:
    """单维度评分。"""
    name: str
    score: float       # 0~10
    weight: float      # 0~1
    max: float = 10.0


@dataclass
class VetoItem:
    """一票否决项。"""
    triggered: bool
    reason: str
    cap_score: Optional[float] = None   # 触发后该维度上限; None=整体惩罚


@dataclass
class ScorecardResult:
    """评分卡最终结果。"""
    mode: str                                   # "short" / "long"
    raw_total: float = 0.0                      # 加权原始总分 (0~100)
    env_adjusted_total: float = 0.0             # 环境修正后总分
    final_total: float = 0.0                    # 一票否决后最终得分
    market_rating: str = "M"                    # "H" / "M" / "L"
    dimensions: List[Dict] = field(default_factory=list)
    vetos: List[Dict] = field(default_factory=list)
    position_advice: Dict = field(default_factory=dict)
    grade: str = ""                             # "A+" / "A" / "B+" / "B" / "C" / "D"


# ══════════════════════════════════════════════════════
# 评分卡类
# ══════════════════════════════════════════════════════

class Scorecard:
    """评分卡汇总模块。"""

    # 环境修正系数
    ENV_MULTIPLIER = {"H": 1.0, "M": 0.85, "L": 0.70}

    @staticmethod
    def compute_short(
        tech_signals: TechnicalSignals,
        capital_signals: CapitalSignals,
        catalyst_signals: CatalystSignals,
        risk_signals: RiskSignals,
        market_rating: str = "M",
    ) -> ScorecardResult:
        """
        短线评分卡。

        各面调用自己的 score() 方法产生子维度评分，
        然后在这里汇总加权 + 一票否决。
        """
        result = ScorecardResult(mode="short", market_rating=market_rating)

        # ── 1. 收集各面评分 ──
        dims = []
        dims.extend(TechnicalFace.score(tech_signals))      # 技术形态共振25%, 量化诊断10%
        dims.extend(CapitalFace.score(capital_signals))      # 主力资金30%, 融资融券10%
        dims.extend(CatalystFace.score(catalyst_signals))    # 催化剂强度25%
        # 风控面评分记录但不加权 (weight=0)
        risk_dims = RiskFace.score(risk_signals)
        dims.extend(risk_dims)

        result.dimensions = dims

        # ── 2. 收集一票否决 ──
        all_vetos = []
        all_vetos.extend(TechnicalFace.check_veto(tech_signals))
        all_vetos.extend(CapitalFace.check_veto(capital_signals))
        all_vetos.extend(CatalystFace.check_veto(catalyst_signals))
        all_vetos.extend(RiskFace.check_veto(risk_signals))

        result.vetos = [
            {"triggered": v[0], "reason": v[1], "cap_score": v[2]}
            for v in all_vetos if v[0]
        ]

        # ── 3. 加权总分 (0~100) ──
        raw = _weighted_sum(dims)
        result.raw_total = round(raw, 1)

        # ── 4. 环境修正 ──
        mult = Scorecard.ENV_MULTIPLIER.get(market_rating, 0.85)
        adjusted = raw * mult
        result.env_adjusted_total = round(adjusted, 1)

        # ── 5. 一票否决修正 ──
        final = _apply_vetos(adjusted, dims, result.vetos)
        result.final_total = round(final, 1)

        # ── 6. 等级 ──
        result.grade = _grade(result.final_total)

        # ── 7. 仓位建议 ──
        result.position_advice = RiskFace.position_advice(risk_signals, result.final_total)

        return result

    @staticmethod
    def compute_long(
        fundamental_signals: FundamentalSignals,
        catalyst_signals: CatalystSignals,
        risk_signals: RiskSignals,
        market_rating: str = "M",
    ) -> ScorecardResult:
        """
        长线评分卡。

        长线不受环境层短线修正影响 (环境修正仅调L→×0.9)。
        """
        result = ScorecardResult(mode="long", market_rating=market_rating)

        # ── 1. 收集各面评分 ──
        dims = []
        dims.extend(FundamentalFace.score(fundamental_signals))  # 5个子维度
        # 催化剂面 — 长线仅作参考，降权到5%
        cat_dims = CatalystFace.score(catalyst_signals)
        for d in cat_dims:
            d["weight"] = 0.05   # 长线催化剂降权
        dims.extend(cat_dims)
        # 风控面
        risk_dims = RiskFace.score(risk_signals)
        dims.extend(risk_dims)

        # 重新归一化权重 (基本面5项总95% + 催化剂5% = 100%)
        _renormalize_weights(dims)

        result.dimensions = dims

        # ── 2. 一票否决 ──
        all_vetos = []
        all_vetos.extend(FundamentalFace.check_veto(fundamental_signals))
        all_vetos.extend(CatalystFace.check_veto(catalyst_signals))
        all_vetos.extend(RiskFace.check_veto(risk_signals))

        result.vetos = [
            {"triggered": v[0], "reason": v[1], "cap_score": v[2]}
            for v in all_vetos if v[0]
        ]

        # ── 3. 加权总分 ──
        raw = _weighted_sum(dims)
        result.raw_total = round(raw, 1)

        # ── 4. 环境修正 (长线轻微) ──
        long_mult = {"H": 1.0, "M": 1.0, "L": 0.90}
        mult = long_mult.get(market_rating, 1.0)
        adjusted = raw * mult
        result.env_adjusted_total = round(adjusted, 1)

        # ── 5. 一票否决 ──
        final = _apply_vetos(adjusted, dims, result.vetos)
        result.final_total = round(final, 1)

        # ── 6. 等级 ──
        result.grade = _grade(result.final_total)

        # ── 7. 仓位建议 ──
        result.position_advice = RiskFace.position_advice(risk_signals, result.final_total)

        return result

    @staticmethod
    def format_report(result: ScorecardResult) -> str:
        """格式化评分卡文本报告。"""
        mode_label = "短线" if result.mode == "short" else "长线"
        lines = [
            f"{'='*50}",
            f"  {mode_label}评分卡   总分: {result.final_total}/100   等级: {result.grade}",
            f"  市场评级: {result.market_rating}   环境修正后: {result.env_adjusted_total}",
            f"{'='*50}",
            "",
            "维度评分:",
        ]

        for d in result.dimensions:
            w = d.get("weight", 0)
            if w == 0:
                tag = "(独立约束)"
            else:
                tag = f"(×{w:.0%})"
            lines.append(
                f"  {d['name']:　<10} {d['score']:.1f}/{d['max']}  {tag}"
            )

        if result.vetos:
            lines.extend(["", "一票否决:"])
            for v in result.vetos:
                cap = f"→ 上限{v['cap_score']}" if v.get("cap_score") is not None else "→ 整体惩罚"
                lines.append(f"  ⚠️ {v['reason']}  {cap}")

        pa = result.position_advice
        if pa:
            lines.extend([
                "",
                "仓位建议:",
                f"  建议仓位: {pa.get('position_pct', 0)}%",
                f"  止损: {pa.get('stop_loss', 'N/A')}   止盈: {pa.get('take_profit', 'N/A')}",
                f"  {pa.get('rationale', '')}",
            ])

        return "\n".join(lines)


# ══════════════════════════════════════════════════════
# 内部函数
# ══════════════════════════════════════════════════════

def _weighted_sum(dims: list) -> float:
    """计算加权总分 (0~100)。"""
    total = 0.0
    for d in dims:
        w = d.get("weight", 0)
        s = d.get("score", 0)
        mx = d.get("max", 10)
        # 归一化到 0~100 的贡献
        total += (s / mx) * w * 100
    return total


def _renormalize_weights(dims: list):
    """重新归一化权重使有效权重之和=1.0。"""
    effective = [d for d in dims if d.get("weight", 0) > 0]
    total_w = sum(d["weight"] for d in effective)
    if total_w > 0 and abs(total_w - 1.0) > 0.01:
        for d in effective:
            d["weight"] = d["weight"] / total_w


def _apply_vetos(adjusted_total: float, dims: list, vetos: list) -> float:
    """应用一票否决，返回最终得分。"""
    final = adjusted_total

    for v in vetos:
        cap = v.get("cap_score")
        if cap is not None:
            # 特定维度封顶 → 等效扣分
            # cap 是该维度得分上限 (0~10)
            # 简化处理: 如果cap很低，对总分施加惩罚
            if cap == 0:
                final *= 0.5
            elif cap <= 2:
                final *= 0.7
            elif cap <= 3:
                final *= 0.8
        else:
            # 整体惩罚 (通常×0.7~0.8)
            final *= 0.75

    return max(0, min(100, final))


def _grade(score: float) -> str:
    """总分映射等级。"""
    if score >= 85:
        return "A+"
    elif score >= 75:
        return "A"
    elif score >= 65:
        return "B+"
    elif score >= 55:
        return "B"
    elif score >= 40:
        return "C"
    else:
        return "D"


# ══════════════════════════════════════════════════════
# 综合决策矩阵（短线进场 + 长线安全垫）
# ══════════════════════════════════════════════════════

@dataclass
class CombinedDecision:
    """两层决策结构的输出。"""
    # 长线安全垫
    long_score: float = 0.0
    long_tier: str = "C"                        # "A" / "B" / "C"
    long_tier_label: str = "无托底"

    # 短线信号
    short_score: float = 0.0
    short_level: str = "weak"                    # "strong" / "medium" / "weak"
    short_level_label: str = "短线弱"

    # 3×3 矩阵输出
    matrix_action: str = "排除"                  # 积极进场 / 可进场 / 观察池 / ...
    matrix_detail: str = ""

    # 仓位控制（按档位调整后）
    kelly_raw_pct: float = 0.0                   # 原始Kelly
    tier_multiplier: float = 0.3                 # A=1.0, B=0.6, C=0.3
    position_pct: float = 0.0                    # 调整后仓位

    # 止损策略（按档位）
    stop_strategy: str = ""                      # "ATR×1.5浮动止损" / "固定-5%硬止损" / ...
    stop_loss_price: Optional[float] = None
    stop_loss_pct: float = 0.0

    # 持仓行为规则
    if_trapped: str = ""                         # 被套行为
    if_profit: str = ""                          # 盈利行为
    if_dip: str = ""                             # 下跌补仓规则


# ── 长线安全垫分档 ──

def _classify_long_tier(long_score: float) -> tuple:
    """长线得分 → A/B/C 档。"""
    if long_score >= 75:
        return "A", "优质托底"
    elif long_score >= 55:
        return "B", "一般托底"
    else:
        return "C", "无托底"


# ── 短线信号强度 ──

def _classify_short_level(short_score: float) -> tuple:
    """短线得分 → 强/中/弱。"""
    if short_score >= 75:
        return "strong", "短线强（≥75）"
    elif short_score >= 65:
        return "medium", "短线中（65-74）"
    else:
        return "weak", "短线弱（<65）"


# ── 3×3 决策矩阵 ──

_DECISION_MATRIX = {
    # (short_level, long_tier): (action, detail)
    ("strong", "A"): ("积极进场", "短线强+优质托底，可持至目标位，Kelly上限仓位"),
    ("strong", "B"): ("进场+严格止损", "短线强但托底一般，进场后严格止损，短线目标即离场"),
    ("strong", "C"): ("轻仓短炒", "短线强但无托底，极轻仓不拿过夜，当天完成"),
    ("medium", "A"): ("可进场", "短线中+优质托底，短线止盈后可留底仓等长线兑现"),
    ("medium", "B"): ("谨慎进场", "短线中+一般托底，仓位减半，严格纪律"),
    ("medium", "C"): ("观望", "短线中但无托底，风险收益不合适，不参与"),
    ("weak",   "A"): ("加入观察池", "短线弱但基本面优质，等短线信号转强后再评估"),
    ("weak",   "B"): ("不操作", "短线弱+一般托底，性价比太低"),
    ("weak",   "C"): ("排除", "双弱不碰，无任何安全边际"),
}


# ── 仓位 × 档位乘数 ──

_TIER_KELLY_MULT = {"A": 1.0, "B": 0.6, "C": 0.3}
_TIER_KELLY_RANGE = {
    "A": "Kelly上限（约10-15%）",
    "B": "Kelly×0.6（约6-9%）",
    "C": "Kelly×0.3（约3-5%）",
}


# ── 止损策略 × 档位 ──

def _tier_stop_strategy(tier: str, current_price: float,
                        atr_stop: Optional[float],
                        risk_signals=None) -> tuple:
    """
    按档位选择止损策略。
    Returns: (strategy_label, stop_loss_price, stop_loss_pct)
    """
    if tier == "A":
        if atr_stop is not None and current_price > atr_stop:
            # A档: ATR×1.5 浮动止损（比普通ATR止损更宽松）
            # atr_stop 是 risk_manager 的 ATR 止损价，通常是 close - 2*ATR
            # 我们用更宽的 close - 1.5*ATR 近似：取距离 × 0.75
            gap = current_price - atr_stop
            adjusted_stop = current_price - gap * 0.75  # 宽松 25%
            pct = (current_price - adjusted_stop) / current_price * 100
            return "ATR×1.5 浮动止损", round(adjusted_stop, 2), round(pct, 2)
        else:
            # ATR 不可用时 fallback: A档用较宽的 -8%
            stop = current_price * 0.92
            return "ATR×1.5 浮动止损（ATR不可用，按-8%）", round(stop, 2), 8.0
    elif tier == "B":
        stop = current_price * 0.95
        return "固定 -5% 硬止损", round(stop, 2), 5.0
    else:  # C
        stop = current_price * 0.97
        return "固定 -3% 极严止损", round(stop, 2), 3.0


# ── 持仓行为规则 × 档位 ──

_HOLDING_RULES = {
    "A": {
        "if_trapped":  "转长线持仓，等基本面催化（优质托底，可以等待修复）",
        "if_profit":   "止盈一半 + 留底仓，升止损至成本线",
        "if_dip":      "可在关键支撑补仓（长线质量支持）",
    },
    "B": {
        "if_trapped":  "止损后重新评估，不盲目补仓",
        "if_profit":   "全部止盈离场，不恋战",
        "if_dip":      "原则上不补仓",
    },
    "C": {
        "if_trapped":  "触止损立即离场，绝不扛单",
        "if_profit":   "全部止盈离场，当天内完成",
        "if_dip":      "严禁补仓",
    },
}


def compute_combined_decision(
    sc_short: ScorecardResult,
    sc_long: ScorecardResult,
    risk_signals: 'RiskSignals' = None,
    current_price: float = 0.0,
) -> CombinedDecision:
    """
    两层决策结构：短线进场触发 × 长线安全垫评级 → 综合决策。

    第一层: 短线评分 → strong(≥75) / medium(65-74) / weak(<65)
    第二层: 长线评分 → A(≥75) / B(55-74) / C(<55)
    交叉查 3×3 矩阵 → 最终决策 + 仓位 + 止损 + 持仓行为
    """
    cd = CombinedDecision()

    # ── 分档 ──
    cd.long_score = sc_long.final_total
    cd.long_tier, cd.long_tier_label = _classify_long_tier(sc_long.final_total)

    cd.short_score = sc_short.final_total
    cd.short_level, cd.short_level_label = _classify_short_level(sc_short.final_total)

    # ── 3×3 矩阵查表 ──
    key = (cd.short_level, cd.long_tier)
    action, detail = _DECISION_MATRIX.get(key, ("排除", "未知组合"))
    cd.matrix_action = action
    cd.matrix_detail = detail

    # ── 仓位调整 ──
    kelly_raw = 0.0
    atr_stop = None
    if risk_signals is not None:
        kelly_raw = risk_signals.kelly_position_pct or 0.0
        atr_stop = risk_signals.stop_loss_price
    cd.kelly_raw_pct = kelly_raw

    cd.tier_multiplier = _TIER_KELLY_MULT.get(cd.long_tier, 0.3)
    cd.position_pct = round(max(0, kelly_raw * cd.tier_multiplier), 1)

    # 极端风险硬覆盖
    if risk_signals and risk_signals.risk_level == "extreme":
        cd.position_pct = 0.0
    elif risk_signals and risk_signals.risk_level == "high":
        cd.position_pct = min(cd.position_pct, 10.0)

    # 非进场决策，仓位清零
    if cd.matrix_action in ("观望", "不操作", "排除"):
        cd.position_pct = 0.0

    # ── 止损策略 ──
    if current_price > 0:
        cd.stop_strategy, cd.stop_loss_price, cd.stop_loss_pct = _tier_stop_strategy(
            cd.long_tier, current_price, atr_stop, risk_signals,
        )
    else:
        cd.stop_strategy = _TIER_KELLY_RANGE.get(cd.long_tier, "")

    # ── 持仓行为 ──
    rules = _HOLDING_RULES.get(cd.long_tier, _HOLDING_RULES["C"])
    cd.if_trapped = rules["if_trapped"]
    cd.if_profit = rules["if_profit"]
    cd.if_dip = rules["if_dip"]

    return cd


def format_combined_decision(cd: CombinedDecision) -> str:
    """格式化综合决策为文本报告。"""
    lines = [
        f"{'─'*50}",
        f"  综合决策（短线进场 × 长线安全垫）",
        f"{'─'*50}",
        f"  短线评分: {cd.short_score:.0f}/100  {cd.short_level_label}",
        f"  长线评分: {cd.long_score:.0f}/100  {cd.long_tier}档（{cd.long_tier_label}）",
        f"",
        f"  ▶ 决策: {cd.matrix_action}",
        f"    {cd.matrix_detail}",
        f"",
        f"  仓位: {cd.position_pct:.1f}%   "
        f"（Kelly {cd.kelly_raw_pct:.1f}% × {cd.long_tier}档系数{cd.tier_multiplier}）",
        f"  止损: {cd.stop_strategy}",
    ]
    if cd.stop_loss_price:
        lines.append(f"    止损价 {cd.stop_loss_price}（-{cd.stop_loss_pct:.1f}%）")

    lines.extend([
        f"",
        f"  持仓规则（{cd.long_tier}档）:",
        f"    被套: {cd.if_trapped}",
        f"    盈利: {cd.if_profit}",
        f"    补仓: {cd.if_dip}",
    ])

    return "\n".join(lines)
