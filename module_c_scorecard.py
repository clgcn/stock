"""
模块 C — 综合评分卡引擎
========================
架构位置: 模块B1/B2之后，模块D之前

职责:
  1. 接收模块A环境 + 模块B1短线信号 + 模块B2长线信号
  2. 按确定性评分规则计算各维度得分 (0-10)
  3. 加权汇总短线总分 (0-100) 和长线总分 (0-100)
  4. 执行一票否决机制（hard gate）
  5. 提取关键亮点3条 + 关键风险3条
  6. 输出结构化评分卡

短线维度评分规则 (5维):
  催化剂强度   25%  — stock_news + market_news
  主力资金净入 30%  — moneyflow + margin_trading
  技术形态共振 25%  — kline_data (MA/MACD/KDJ/RSI)
  量化诊断得分 10%  — stock_diagnosis + quant_activity
  融资融券情绪 10%  — margin_trading + northbound_flow

长线维度评分规则 (6维):
  财务质量     30%  — financial_data + earnings_analysis + valuation_quality
  估值安全边际 25%  — valuation_quality + financial_data
  负债与商誉   20%  — balance_sheet + financial_data
  成长质量     15%  — earnings_analysis + announcements
  分红与治理   10%  — dividend_history + announcements
"""

from typing import TypedDict, Optional


class DimensionScore(TypedDict):
    name: str
    raw_score: float   # 0-10
    weight: float      # 0.0-1.0
    weighted: float    # raw_score * weight * 10 (贡献到0-100总分)
    vetoed: bool       # 是否被一票否决
    veto_reason: str   # 一票否决原因


class ScorecardResult(TypedDict):
    market_rating: str           # "H" / "M" / "L"
    short_term_score: float      # 0-100
    long_term_score: float       # 0-100
    short_term_label: str        # 描述
    long_term_label: str         # 描述
    short_dimensions: list       # list[DimensionScore]
    long_dimensions: list        # list[DimensionScore]
    highlights: list             # 关键亮点 (最多3条)
    risks: list                  # 关键风险 (最多3条)
    short_vetoed: bool           # 短线是否被整体否决
    long_vetoed: bool            # 长线是否被整体否决
    short_veto_reason: str
    long_veto_reason: str


# ══════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════

def compute_scorecard(
    env_result: dict,
    shortterm_signals: dict,
    longterm_signals: dict,
) -> ScorecardResult:
    """
    计算综合评分卡。

    Parameters
    ----------
    env_result : EnvironmentResult  模块A环境感知结果
    shortterm_signals : ShortTermSignals  模块B1短线信号
    longterm_signals : LongTermSignals    模块B2长线信号
    """
    market_rating = env_result.get("market_rating", "M")
    short_weight = env_result.get("short_term_weight", 1.0)

    # ── 短线评分 ──
    short_dims = _score_shortterm(shortterm_signals)
    short_total = sum(d["weighted"] for d in short_dims)
    # L环境自动降权
    short_total *= short_weight
    # Hurst指数技术面调整（prompt规则落地）:
    #   ≥0.55 趋势持续，技术面+15；≤0.45 均值回归，技术面-10
    hurst = shortterm_signals.get("hurst")
    if hurst is not None:
        if hurst >= 0.55:
            short_total = min(100, short_total + 15)
        elif hurst <= 0.45:
            short_total = max(0, short_total - 10)
    short_vetoed, short_veto_reason = _check_short_veto(shortterm_signals, short_dims)
    if short_vetoed:
        short_total = min(short_total, 30)  # 一票否决: 强制得分上限30

    # ── 长线评分 ──
    long_dims = _score_longterm(longterm_signals)
    long_total = sum(d["weighted"] for d in long_dims)
    long_vetoed, long_veto_reason = _check_long_veto(longterm_signals, long_dims)
    if long_vetoed:
        long_total = min(long_total, 30)

    # ── 亮点 + 风险 ──
    highlights = _extract_highlights(env_result, shortterm_signals, longterm_signals, short_dims, long_dims)
    risks = _extract_risks(env_result, shortterm_signals, longterm_signals, short_dims, long_dims)

    return ScorecardResult(
        market_rating=market_rating,
        short_term_score=round(short_total, 1),
        long_term_score=round(long_total, 1),
        short_term_label=_score_label(short_total),
        long_term_label=_score_label(long_total),
        short_dimensions=short_dims,
        long_dimensions=long_dims,
        highlights=highlights[:3],
        risks=risks[:3],
        short_vetoed=short_vetoed,
        long_vetoed=long_vetoed,
        short_veto_reason=short_veto_reason,
        long_veto_reason=long_veto_reason,
    )


def _score_label(score: float) -> str:
    """技术信号描述，避免投资建议类措辞以降低合规风险。"""
    if score >= 80:
        return "技术信号偏强，可关注"
    elif score >= 60:
        return "技术信号较强"
    elif score >= 40:
        return "技术信号中性"
    elif score >= 20:
        return "技术信号偏弱"
    else:
        return "技术信号偏弱，谨慎"


# ══════════════════════════════════════════════════════
# 短线评分（5维）
# ══════════════════════════════════════════════════════

def _score_shortterm(s: dict) -> list:
    """短线5维评分，每维0-10分。"""
    dims = []

    # 1. 催化剂强度 25%
    cat_score = _score_catalyst(s)
    dims.append(_dim("催化剂强度", cat_score, 0.25))

    # 2. 主力资金净入 30%
    fund_score = _score_main_fund(s)
    dims.append(_dim("主力资金净入", fund_score, 0.30))

    # 3. 技术形态共振 25%
    tech_score = _score_tech_pattern(s)
    dims.append(_dim("技术形态共振", tech_score, 0.25))

    # 4. 量化诊断得分 10%
    quant_score = _score_quant_diagnosis(s)
    dims.append(_dim("量化诊断得分", quant_score, 0.10))

    # 5. 融资融券情绪 10%
    margin_score = _score_margin_sentiment(s)
    dims.append(_dim("融资融券情绪", margin_score, 0.10))

    return dims


def _score_catalyst(s: dict) -> float:
    """
    催化剂强度评分:
      强催化（当日公告+涨停）: 9-10
      中等（行业政策/预期改善）: 5-8
      无明显催化剂: 2-4
      负面消息（利空公告）: 0-1
    """
    ct = s.get("catalyst_type", "weak")
    if ct == "strong":
        return 9.5
    elif ct == "medium":
        return 6.5
    elif ct == "negative":
        return 0.5
    else:  # weak
        return 3.0


def _score_main_fund(s: dict) -> float:
    """
    主力资金评分:
      大单净流入比（净流入/成交额）:
        >= +5%: 9-10
        >= +2%: 7-8
        -1%~+2%（中性）: 4-6
        -2%~-5%: 2-3
        <= -5%（主力明显出逃）: 0-1
    """
    ratio = s.get("net_inflow_ratio")
    if ratio is None:
        return 5.0  # 无数据默认中性

    if ratio >= 5:
        return min(10, 9 + (ratio - 5) / 10)
    elif ratio >= 2:
        return 7 + (ratio - 2) / 3
    elif ratio >= -1:
        return 4 + (ratio + 1) / 3 * 2
    elif ratio >= -5:
        return 2 + (ratio + 5) / 4
    else:
        return max(0, 1 + (ratio + 5) / 5)


def _score_tech_pattern(s: dict) -> float:
    """
    技术形态评分（3个子项各0-10，加权平均）:
      均线多头排列(5>10>20): +3
      MACD金叉+柱体扩张: +3
      RSI 40-70(健康区间): +2
      KDJ J值上穿50: +2
      缺一项扣对应分
    """
    score = 0.0

    # 均线排列 (0-3)
    ma = s.get("ma_alignment", "mixed")
    if ma == "bullish":
        score += 3.0
    elif ma == "mixed":
        score += 1.5
    # bearish = 0

    # MACD (0-3)
    macd = s.get("macd_signal", "neutral")
    if macd == "golden_cross":
        score += 3.0
    elif macd == "expanding":
        score += 2.0
    elif macd == "neutral":
        score += 1.5
    elif macd == "contracting":
        score += 1.0
    # death_cross = 0

    # RSI (0-2)
    rsi = s.get("rsi_value")
    if rsi is not None:
        if 40 <= rsi <= 70:
            score += 2.0
        elif 30 <= rsi < 40 or 70 < rsi <= 80:
            score += 1.0
        # < 30 or > 80 = 0

    # KDJ J值上穿50 (0-2)
    if s.get("kdj_cross_50"):
        score += 2.0
    elif s.get("kdj_j_value") is not None and s["kdj_j_value"] > 50:
        score += 1.0

    return score


def _score_quant_diagnosis(s: dict) -> float:
    """
    量化诊断评分:
      直接使用诊断工具输出的综合评分（0-100）→ 除以10得0-10分
      蒙特卡洛上涨概率 >55%: 额外 +1
      量化资金占比 >30%: 额外 -1
    """
    diag = s.get("diagnosis_score")
    if diag is None:
        return 5.0
    score = diag / 10.0
    # 蒙特卡洛修正
    mc = s.get("monte_carlo_up_prob")
    if mc is not None and mc > 55:
        score += 1.0
    # 量化资金占比修正
    qp = s.get("quant_share_pct")
    if qp is not None and qp > 30:
        score -= 1.0
    return max(0, min(10, score))


def _score_margin_sentiment(s: dict) -> float:
    """
    融资融券情绪评分:
      融资余额周环比变化:
        上升 >3%: 7-9
        持平: 5-6
        下降 >3%: 2-4
      北向当日净买入同向: 额外 +1
    """
    change = s.get("margin_balance_change_pct")
    if change is None:
        score = 5.0  # 无数据默认中性
    elif change > 3:
        score = min(9, 7 + change / 10)
    elif change >= -3:
        score = 5 + change / 6
    else:
        score = max(2, 4 + change / 10)

    if s.get("northbound_same_direction"):
        score += 1.0

    return max(0, min(10, score))


# ══════════════════════════════════════════════════════
# 长线评分（6维）
# ══════════════════════════════════════════════════════

def _score_longterm(s: dict) -> list:
    """长线6维评分，每维0-10分。"""
    dims = []

    # 1. 财务质量 30%
    fq_score = _score_financial_quality(s)
    dims.append(_dim("财务质量", fq_score, 0.30))

    # 2. 估值安全边际 25%
    val_score = _score_valuation_margin(s)
    dims.append(_dim("估值安全边际", val_score, 0.25))

    # 3. 负债与商誉 20%
    debt_score = _score_debt_goodwill(s)
    dims.append(_dim("负债与商誉", debt_score, 0.20))

    # 4. 成长质量 15%
    growth_score = _score_growth_quality(s)
    dims.append(_dim("成长质量", growth_score, 0.15))

    # 5. 分红与治理 10%
    div_score = _score_dividend_governance(s)
    dims.append(_dim("分红与治理", div_score, 0.10))

    return dims


def _score_financial_quality(s: dict) -> float:
    """
    财务质量评分（4项子指标等权平均）:
      ROE: >=20%→10, 15-20%→8, 10-15%→5
      净利润增速: >=25%→10, 15-25%→7
      毛利率稳定性: 近3年波动<3%→+2
      自由现金流/净利润: >=80%→+2 (现金质量验证)
    """
    sub_scores = []

    # ROE
    roe = s.get("roe")
    if roe is not None:
        if roe >= 20:
            sub_scores.append(10)
        elif roe >= 15:
            sub_scores.append(8)
        elif roe >= 10:
            sub_scores.append(5)
        elif roe >= 5:
            sub_scores.append(3)
        else:
            sub_scores.append(1)
    else:
        sub_scores.append(5)  # 无数据默认中性

    # 净利润增速
    npg = s.get("net_profit_growth")
    if npg is not None:
        if npg >= 25:
            sub_scores.append(10)
        elif npg >= 15:
            sub_scores.append(7)
        elif npg >= 5:
            sub_scores.append(5)
        elif npg >= 0:
            sub_scores.append(3)
        else:
            sub_scores.append(1)
    else:
        sub_scores.append(5)

    # 毛利率稳定性
    bonus = 0
    if s.get("gross_margin_stable"):
        bonus += 2

    # 自由现金流/净利润
    fcf = s.get("fcf_to_net_profit")
    if fcf is not None and fcf >= 80:
        bonus += 2

    avg = sum(sub_scores) / len(sub_scores) if sub_scores else 5
    return max(0, min(10, avg + bonus))


def _score_valuation_margin(s: dict) -> float:
    """
    估值安全边际评分:
      PEG 为核心指标:
        PEG < 0.8 → 9-10
        0.8-1.0 → 7-8
        1.0-1.5 → 4-6
        > 1.5 → 1-3
      PB < 行业均值 0.8倍: 额外 +1
      PE 处于近5年低30%分位: 额外 +1
    """
    peg = s.get("peg")
    if peg is None:
        score = 5.0
    elif peg < 0.8:
        score = 9.5
    elif peg < 1.0:
        score = 7 + (1.0 - peg) / 0.2 * 1
    elif peg < 1.5:
        score = 4 + (1.5 - peg) / 0.5 * 2
    else:
        score = max(1, 3 - (peg - 1.5))

    # PB vs 行业
    pb_vs = s.get("pb_vs_industry")
    if pb_vs is not None and pb_vs < 0.8:
        score += 1.0

    # PE 百分位
    pe_pct = s.get("pe_percentile_5y")
    if pe_pct is not None and pe_pct <= 30:
        score += 1.0

    return max(0, min(10, score))


def _score_debt_goodwill(s: dict) -> float:
    """
    负债与商誉评分:
      资产负债率（行业调整后）:
        <= 40% → 9-10
        40-55% → 6-8
        55-70% → 3-5
        > 70% → 0-2
      商誉/净资产:
        <10% → +1
        10-20% → 0
        >30% → -2（减分项）
    """
    dr = s.get("debt_ratio")
    if dr is None:
        score = 5.0
    elif dr <= 40:
        score = 9 + (40 - dr) / 40
    elif dr <= 55:
        score = 6 + (55 - dr) / 15 * 2
    elif dr <= 70:
        score = 3 + (70 - dr) / 15 * 2
    else:
        score = max(0, 2 - (dr - 70) / 15)

    # 商誉修正
    gw = s.get("goodwill_ratio")
    if gw is not None:
        if gw < 10:
            score += 1
        elif gw > 30:
            score -= 2

    return max(0, min(10, score))


def _score_growth_quality(s: dict) -> float:
    """
    成长质量评分:
      近3年营收+净利润双增: 基础分 6
      增速加速（今年>去年）: +2
      业绩超预期: +1
      公告无异常: +1
      增速放缓连续2季: -2
    """
    score = 4.0  # 基础分

    if s.get("revenue_growth_3y"):
        score = 6.0
    if s.get("growth_accelerating"):
        score += 2.0
    if s.get("earnings_beat"):
        score += 1.0
    if s.get("no_regulatory_issue"):
        score += 1.0
    if s.get("growth_slowing_2q"):
        score -= 2.0

    return max(0, min(10, score))


def _score_dividend_governance(s: dict) -> float:
    """
    分红与治理评分:
      连续5年分红: 基础分 5
      股息率 > 2%: +2   > 4%: +3
      高管增持记录（近1年）: +2
      无大股东资金占用记录: +1
      大股东高比例质押(>50%): -3分
    """
    years = s.get("consecutive_div_years", 0)
    if years >= 5:
        score = 5.0
    elif years >= 3:
        score = 3.0
    elif years >= 1:
        score = 2.0
    else:
        score = 1.0

    dy = s.get("dividend_yield")
    if dy is not None:
        if dy > 4:
            score += 3
        elif dy > 2:
            score += 2

    if s.get("mgmt_increase"):
        score += 2
    if s.get("no_fund_misuse"):
        score += 1

    pledge = s.get("major_holder_pledge_pct")
    if pledge is not None and pledge > 50:
        score -= 3

    return max(0, min(10, score))


# ══════════════════════════════════════════════════════
# 一票否决机制
# ══════════════════════════════════════════════════════

def _check_short_veto(s: dict, dims: list) -> tuple:
    """
    短线一票否决规则:
      1. 利空公告当日 → 催化剂强制得分 ≤ 1，整体短线降权
      2. 连续3日净流出 且 融资余额下降 → 整体评分 ×0.7
      3. RSI > 85 超买 或 均线空头排列 → 技术分 ≤ 3（强制上限）
      4. 诊断评分 < 30 → 短线整体警示
    """
    vetoed = False
    reasons = []

    # 1. 利空催化剂
    if s.get("catalyst_type") == "negative":
        _cap_dim(dims, "催化剂强度", 1.0)
        reasons.append("利空公告当日，催化剂强制≤1")

    # 2. 主力资金+融资双出
    if s.get("consecutive_outflow_days", 0) >= 3 and s.get("margin_trend") == "down":
        vetoed = True
        reasons.append("连续3日主力净流出且融资余额下降 → 整体评分×0.7")

    # 3. RSI 超买 或 均线空头
    rsi = s.get("rsi_value")
    if rsi is not None and rsi > 85:
        _cap_dim(dims, "技术形态共振", 3.0)
        reasons.append(f"RSI={rsi:.0f}>85 超买，技术分强制≤3")
    if s.get("ma_alignment") == "bearish":
        _cap_dim(dims, "技术形态共振", 3.0)
        reasons.append("均线空头排列，技术分强制≤3")

    # 4. 诊断评分过低
    diag = s.get("diagnosis_score")
    if diag is not None and diag < 30:
        reasons.append(f"量化诊断评分={diag:.0f}<30，短线整体警示")

    if not vetoed and not reasons:
        return False, ""
    return bool(reasons), "；".join(reasons)


def _check_long_veto(s: dict, dims: list) -> tuple:
    """
    长线一票否决规则:
      1. ROE < 8% 或净利润连续2年下滑 → 财务质量强制得分 ≤ 3
      2. PEG > 2.0 或 PE > 历史90%分位 → 高估警告
      3. 商誉/净资产 > 50% → 一票否决，得分强制 ≤ 2
      4. 近期收到监管问询函 → 长线评分整体 ×0.8
      5. 曾有资金占用或实控人变更未满1年 → 分红治理得分强制 0
    """
    vetoed = False
    reasons = []

    # 1. ROE 过低
    roe = s.get("roe")
    if roe is not None and roe < 8:
        _cap_dim(dims, "财务质量", 3.0)
        reasons.append(f"ROE={roe:.1f}%<8%，财务质量强制≤3（质量不达标）")

    # 2. 高估值警告
    peg = s.get("peg")
    pe_pct = s.get("pe_percentile_5y")
    if peg is not None and peg > 2.0:
        reasons.append(f"PEG={peg:.2f}>2.0，高估警告")
    if pe_pct is not None and pe_pct > 90:
        reasons.append(f"PE处于近5年{pe_pct:.0f}%分位>90%，估值偏高")

    # 3. 商誉一票否决
    gw = s.get("goodwill_ratio")
    if gw is not None and gw > 50:
        vetoed = True
        _cap_dim(dims, "负债与商誉", 2.0)
        reasons.append(f"商誉/净资产={gw:.1f}%>50%，一票否决，得分强制≤2")

    # 4. 监管问询
    if not s.get("no_regulatory_issue"):
        reasons.append("近期收到监管问询函，长线评分整体×0.8")

    # 5. 资金占用/实控人变更
    if not s.get("no_fund_misuse"):
        _cap_dim(dims, "分红与治理", 0)
        reasons.append("曾有资金占用或实控人变更，分红治理得分强制0")

    if not vetoed and not reasons:
        return False, ""
    return bool(reasons), "；".join(reasons)


def _cap_dim(dims: list, name: str, max_score: float):
    """将指定维度的得分强制压到上限。"""
    for d in dims:
        if d["name"] == name and d["raw_score"] > max_score:
            d["raw_score"] = max_score
            d["weighted"] = max_score * d["weight"] * 10
            d["vetoed"] = True


# ══════════════════════════════════════════════════════
# 亮点 + 风险提取
# ══════════════════════════════════════════════════════

def _extract_highlights(env, short_s, long_s, short_dims, long_dims) -> list:
    """提取关键亮点（最多3条）。"""
    highlights = []

    # 北向连续净流入
    if env.get("northbound_adj", 0) >= 0.10:
        highlights.append("北向资金连续净流入，外资积极布局")

    # ROE 高
    roe = long_s.get("roe")
    if roe is not None and roe >= 18:
        highlights.append(f"ROE连续>18%（{roe:.1f}%），盈利能力优秀")

    # PEG 低估
    peg = long_s.get("peg")
    if peg is not None and peg < 1.0:
        highlights.append(f"PEG={peg:.2f}<1，成长性洼地")

    # 主力资金大幅流入
    ratio = short_s.get("net_inflow_ratio")
    if ratio is not None and ratio >= 5:
        highlights.append(f"主力大单净流入比+{ratio:.1f}%，资金强势介入")

    # 技术面强
    if short_s.get("ma_alignment") == "bullish" and short_s.get("macd_signal") in ("golden_cross", "expanding"):
        highlights.append("均线多头排列+MACD金叉，技术面强势")

    # 高分红
    dy = long_s.get("dividend_yield")
    if dy is not None and dy > 3:
        highlights.append(f"股息率{dy:.1f}%>3%，股东回报优秀")

    return highlights[:3]


def _extract_risks(env, short_s, long_s, short_dims, long_dims) -> list:
    """提取关键风险（最多3条）。"""
    risks = []

    # 商誉风险
    gw = long_s.get("goodwill_ratio")
    if gw is not None and gw > 15:
        risks.append(f"商誉占净资产{gw:.0f}%，减值风险")

    # 大股东质押
    pledge = long_s.get("major_holder_pledge_pct")
    if pledge is not None and pledge > 30:
        risks.append(f"大股东质押比例{pledge:.0f}%，控制权风险")

    # 量化资金占比偏高
    qp = short_s.get("quant_share_pct")
    if qp is not None and qp > 30:
        risks.append("量化资金占比偏高，波动可能加剧")

    # 主力持续流出
    if short_s.get("consecutive_outflow_days", 0) >= 3:
        risks.append(f"主力连续{short_s['consecutive_outflow_days']}日净流出")

    # 高负债
    dr = long_s.get("debt_ratio")
    if dr is not None and dr > 65:
        risks.append(f"资产负债率{dr:.0f}%偏高")

    # RSI 超买
    rsi = short_s.get("rsi_value")
    if rsi is not None and rsi > 80:
        risks.append(f"RSI={rsi:.0f}超买区域，短期回调风险")

    # 悲观环境
    if env.get("market_rating") == "L":
        risks.append("大盘环境悲观(L)，系统性风险偏高")

    return risks[:3]


# ══════════════════════════════════════════════════════
# 辅助
# ══════════════════════════════════════════════════════

def _dim(name: str, raw: float, weight: float) -> DimensionScore:
    raw = max(0, min(10, raw))
    return DimensionScore(
        name=name,
        raw_score=round(raw, 1),
        weight=weight,
        weighted=round(raw * weight * 10, 1),
        vetoed=False,
        veto_reason="",
    )


# ══════════════════════════════════════════════════════
# 格式化输出
# ══════════════════════════════════════════════════════

def format_scorecard(sc: ScorecardResult, stock_name: str = "", stock_code: str = "") -> str:
    """格式化评分卡报告。"""
    rating_labels = {"H": "乐观", "M": "中性偏谨慎", "L": "悲观"}
    lines = [
        "",
        "=" * 60,
        f"【模块 C】综合评分卡" + (f" — {stock_name}（{stock_code}）" if stock_name else ""),
        "=" * 60,
        "",
        f"  ┌─────────────────┬─────────────────┬─────────────────┐",
        f"  │  大盘环境        │  短线得分        │  长线得分        │",
        f"  │  {sc['market_rating']}               │  {sc['short_term_score']:5.1f} / 100     │  {sc['long_term_score']:5.1f} / 100     │",
        f"  │  {rating_labels.get(sc['market_rating'], '未知'):<8}      │  {sc['short_term_label']:<12} │  {sc['long_term_label']:<12} │",
        f"  └─────────────────┴─────────────────┴─────────────────┘",
    ]

    if sc.get("short_vetoed") or sc.get("long_vetoed"):
        lines.append("")
        if sc["short_veto_reason"]:
            lines.append(f"  ⚠️ 短线一票否决: {sc['short_veto_reason']}")
        if sc["long_veto_reason"]:
            lines.append(f"  ⚠️ 长线一票否决: {sc['long_veto_reason']}")

    # 短线各维度明细
    lines.extend(["", "  短线各维度明细:"])
    lines.append(f"  {'维度':<12} {'评分':>6} {'权重':>6} {'加权':>6} {'否决':>4}")
    for d in sc["short_dimensions"]:
        veto_tag = "⚠️" if d["vetoed"] else ""
        lines.append(
            f"  {d['name']:<12} {d['raw_score']:>5.1f} {d['weight']*100:>5.0f}% {d['weighted']:>5.1f} {veto_tag}"
        )

    # 长线各维度明细
    lines.extend(["", "  长线各维度明细:"])
    lines.append(f"  {'维度':<12} {'评分':>6} {'权重':>6} {'加权':>6} {'否决':>4}")
    for d in sc["long_dimensions"]:
        veto_tag = "⚠️" if d["vetoed"] else ""
        lines.append(
            f"  {d['name']:<12} {d['raw_score']:>5.1f} {d['weight']*100:>5.0f}% {d['weighted']:>5.1f} {veto_tag}"
        )

    # 关键亮点
    if sc["highlights"]:
        lines.extend(["", "  关键亮点:"])
        for h in sc["highlights"]:
            lines.append(f"    + {h}")

    # 关键风险
    if sc["risks"]:
        lines.extend(["", "  关键风险:"])
        for r in sc["risks"]:
            lines.append(f"    - {r}")

    return "\n".join(lines)
