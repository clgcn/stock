"""
公告与财报分析模块 (announcements) — 催化剂面依赖
=================================================
提供:
  get_announcements()            → 个股公告列表
  format_announcements()         → 公告格式化报告
  get_financial_history()        → (从 financial 模块re-export)
  analyze_earnings_reaction()    → 财报公告市场反应分析
  format_earnings_analysis()     → 财报深度分析报告

常量:
  _EARNINGS_KEYWORDS, _COLUMN_LABEL
"""

from _http_utils import _get, cn_now, cn_today
from datetime import datetime, timedelta


# ──────────────────────────────────────────
# 常量
# ──────────────────────────────────────────

_EARNINGS_KEYWORDS = [
    "年度报告", "年报", "半年度报告", "半年报", "中报",
    "季度报告", "季报", "一季报", "三季报",
    "业绩预告", "业绩快报", "业绩修正",
]

# Extended announcement type keywords
_CONTRACT_KEYWORDS = (
    "中标", "签约", "重大合同", "战略合作", "中选", "框架协议",
    "供应合同", "采购合同", "销售合同", "大客户", "重大订单", "项目合同",
)
# Context words indicating the "重大合同" mention is a risk disclosure, not an actual contract win
_CONTRACT_RISK_CONTEXT = ("风险", "可能终止", "终止风险", "集中度风险", "依赖", "流失")

# Staleness thresholds by announcement category (days)
_STALENESS_DAYS = {
    "earnings": 120,       # Financial reports valid longer
    "contract": 90,        # Contract announcements stay relevant ~3 months
    "investigation": 365,  # Regulatory investigations linger much longer
    "default": 90,
}
_DEFAULT_STALENESS_DAYS = 90  # Default for announcements not in above categories
_EQUITY_INCENTIVE_KEYWORDS = ("股权激励", "限制性股票", "股票期权", "期权授予")
_DILUTION_KEYWORDS = (
    "定向增发", "非公开发行", "配股", "可转债发行",
    "增发预案", "配股预案",
)
_INVESTIGATION_KEYWORDS = (
    "立案调查", "被调查", "违规违法", "监管处罚", "行政处罚",
    "问询函", "年报问询", "关联交易质证",
)
_RESTRUCTURING_KEYWORDS = ("重大资产重组", "资产重组", "借壳", "吸收合并", "分拆上市")
_INSIDER_KEYWORDS = ("控股股东增持", "控股股东减持", "实控人增持", "实控人减持", "大股东减持")

_COLUMN_LABEL = {
    "年度报告":   "年报",
    "半年度报告": "半年报",
    "季度报告":   "季报",
    "业绩预告":   "业绩预告",
    "业绩快报":   "业绩快报",
    "定期报告":   "定期报告",
}


def _extract_profit_change(title: str):
    """从公告标题提取业绩变化幅度（%）。正数为增长，负数为下降，None表示无法提取。"""
    import re as _re
    # Turnaround: "扭亏为盈" → sentinel +999
    if '扭亏为盈' in title:
        return 999.0
    # Range pattern: "增长30%至50%" or "增长约30%-50%" → take upper bound
    m = _re.search(r'(?:预增|增长|增加|提升|同比增).*?(\d+\.?\d*)\s*[%％].*?[至~到~-]\s*(\d+\.?\d*)\s*[%％]', title)
    if m:
        return float(m.group(2))  # upper bound is more informative
    # Range decline: "下降20%至30%" → take upper (worse) bound
    m = _re.search(r'(?:预减|下降|下滑|减少).*?(\d+\.?\d*)\s*[%％].*?[至~到~-]\s*(\d+\.?\d*)\s*[%％]', title)
    if m:
        return -float(m.group(2))
    # Single percentage growth: "增长50%"
    m = _re.search(r'(?:预增|增长|增加|提升|同比增).*?(\d+\.?\d*)\s*[%％]', title)
    if m:
        return float(m.group(1))
    m = _re.search(r'(?:预减|下降|下滑|减少|预亏).*?(\d+\.?\d*)\s*[%％]', title)
    if m:
        return -float(m.group(1))
    # Fold growth: "增长3倍" → +300%
    m = _re.search(r'(?:增长|增加|提升|翻).*?(\d+\.?\d*)\s*倍', title)
    if m:
        return float(m.group(1)) * 100
    # Loss keywords without percentage
    if any(kw in title for kw in ('预亏', '首亏', '续亏', '由盈转亏')):
        return -999.0  # 标记亏损但幅度未知
    return None


# ──────────────────────────────────────────
# 1. 公告列表
# ──────────────────────────────────────────

def get_announcements(code: str, page_size: int = 20, ann_type: str = "ALL") -> list[dict]:
    """
    从东方财富获取个股最新公告列表。

    Parameters
    ----------
    code      : str  股票代码
    page_size : int  返回条数，默认 20
    ann_type  : str  公告类型筛选

    Returns
    -------
    list of dict，每条包含:
      date, title, category, label, is_earnings, url
    """
    url = "https://np-anotice-stock.eastmoney.com/api/security/ann"
    params = {
        "sr":           -1,
        "page_size":    page_size,
        "page_index":   1,
        "ann_type":     ann_type,
        "client_source": "web",
        "stock_list":   code,
    }
    try:
        resp = _get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise ConnectionError(f"公告接口请求失败: {e}")

    items = (data.get("data") or {}).get("list") or []
    if not items:
        return []

    results = []
    for item in items:
        raw_date = str(item.get("notice_date", "") or "")
        # Handle both "YYYY-MM-DD..." and "YYYY年MM月DD日" formats
        if "年" in raw_date:
            import re as _re2
            m = _re2.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', raw_date)
            date_str = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if m else ""
        else:
            date_str = raw_date[:10] if raw_date else ""

        title    = item.get("title", "").strip()
        col      = item.get("column") or {}
        category = col.get("column_name", "")
        art_code = item.get("art_code", "")

        label = _COLUMN_LABEL.get(category, category)
        is_earnings = any(kw in title or kw in category for kw in _EARNINGS_KEYWORDS)
        profit_change_pct = _extract_profit_change(title) if is_earnings else None

        # Extended classification
        # Contract: keyword match AND not in a risk-disclosure context (false positive filter)
        _kw_match = any(kw in title for kw in _CONTRACT_KEYWORDS)
        _risk_ctx = any(rc in title for rc in _CONTRACT_RISK_CONTEXT)
        has_major_contract = _kw_match and not _risk_ctx
        has_equity_incentive = any(kw in title for kw in _EQUITY_INCENTIVE_KEYWORDS)
        has_dilution_risk = any(kw in title for kw in _DILUTION_KEYWORDS)
        under_investigation = any(kw in title for kw in _INVESTIGATION_KEYWORDS)
        insider_buy = any(kw in title for kw in ("控股股东增持", "实控人增持"))
        insider_sell = any(kw in title for kw in ("控股股东减持", "实控人减持", "大股东减持"))

        ann_url = (
            f"https://np-anotice-stock.eastmoney.com/api/security/ann/{art_code}"
            if art_code else ""
        )

        results.append({
            "date":               date_str,
            "title":              title,
            "category":           category,
            "label":              label,
            "is_earnings":        is_earnings,
            "profit_change_pct":  profit_change_pct,
            "has_major_contract": has_major_contract,
            "has_equity_incentive": has_equity_incentive,
            "has_dilution_risk":  has_dilution_risk,
            "under_investigation": under_investigation,
            "insider_buy":        insider_buy,
            "insider_sell":       insider_sell,
            "url":                ann_url,
        })

    return results


# ──────────────────────────────────────────
# 2. 公告格式化
# ──────────────────────────────────────────

def format_announcements(code: str, anns: list[dict], name: str = "") -> str:
    """将公告列表格式化为中文报告字符串。"""
    title_str = f"{name}（{code}）" if name else code
    lines = [
        f"【公告查询】{title_str}",
        f"  查询时间: {cn_now().strftime('%Y-%m-%d %H:%M')}",
        "─" * 50,
    ]

    if not anns:
        lines.append("  暂无近期公告数据")
        return "\n".join(lines)

    today = cn_today()
    recent_days = 30
    earnings_recent = [
        a for a in anns
        if a["is_earnings"] and a["date"]
        and (today - datetime.strptime(a["date"], "%Y-%m-%d").date()).days <= recent_days
    ]
    earnings_all = [a for a in anns if a["is_earnings"]]

    if earnings_recent:
        latest = earnings_recent[0]
        lines.append(f"\n  财报提示: {latest['date']} 已披露【{latest['title'][:30]}】")
        lines.append(f"      财务指标数据可能正在更新，估值分析请以最新年报为准。")
    elif earnings_all:
        latest_e = earnings_all[0]
        lines.append(f"\n  最近财报: {latest_e['date']}  {latest_e['title'][:35]}")
    else:
        month = today.month
        if month in (3, 4):
            lines.append(f"\n  年报披露期（1-4月），建议关注近期公告。")
        elif month in (7, 8):
            lines.append(f"\n  半年报披露期（7-8月），建议关注近期公告。")
        elif month == 10:
            lines.append(f"\n  三季报披露期（10月），建议关注近期公告。")

    lines.append(f"\n  最近 {len(anns)} 条公告：\n")
    for i, ann in enumerate(anns, 1):
        prefix = "  ★" if ann["is_earnings"] else "   "
        label  = f"[{ann['label']}]" if ann["label"] else ""
        lines.append(
            f"{prefix} {ann['date']}  {label}  {ann['title'][:48]}"
        )

    lines.append("\n  ★ = 财报/业绩相关公告")
    lines.append("─" * 50)
    return "\n".join(lines)


# ──────────────────────────────────────────
# 3. 财报市场反应分析
# ──────────────────────────────────────────

def analyze_earnings_reaction(
    code: str,
    ann_date: str,
    kline_df=None,
) -> dict:
    """
    分析财报公告当日及前后的市场价格反应。

    Parameters
    ----------
    code      : str        股票代码
    ann_date  : str        公告日期 "YYYY-MM-DD"
    kline_df  : DataFrame  可选，已有的日K数据

    Returns
    -------
    dict: ann_date, reaction_pct, reaction_vol_x, pre_pct_5d, post_pct_3d, verdict, verdict_reason
    """
    # 延迟导入避免循环依赖
    from data_fetcher import get_kline

    try:
        ann_dt = datetime.strptime(ann_date, "%Y-%m-%d")
    except ValueError:
        return {"ann_date": ann_date, "reaction_pct": None, "verdict": "日期格式错误"}

    start = (ann_dt - timedelta(days=60)).strftime("%Y-%m-%d")
    end   = (ann_dt + timedelta(days=15)).strftime("%Y-%m-%d")

    if kline_df is None:
        try:
            kline_df = get_kline(code, period="daily", start=start, end=end, adjust="none")
        except Exception as e:
            return {"ann_date": ann_date, "reaction_pct": None,
                    "verdict": "数据获取失败", "verdict_reason": str(e)}

    df = kline_df.copy()
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    ann_idx = None
    for offset in range(5):
        check = (ann_dt + timedelta(days=offset)).strftime("%Y-%m-%d")
        match = df[df["date_str"] == check]
        if not match.empty:
            ann_idx = match.index[0]
            break

    if ann_idx is None:
        return {"ann_date": ann_date, "reaction_pct": None,
                "verdict": "中性", "verdict_reason": "未找到公告日附近的交易数据"}

    row = df.loc[ann_idx]
    reaction_pct = float(row["pct_chg"])

    pre_rows = df[df.index < ann_idx].tail(20)
    avg_vol = pre_rows["volume"].mean() if len(pre_rows) >= 5 else None
    vol_x = round(float(row["volume"]) / avg_vol, 2) if avg_vol and avg_vol > 0 else None

    pre5 = df[df.index < ann_idx].tail(5)
    pre_pct_5d = None
    if len(pre5) >= 3:
        pre_pct_5d = round(float(pre5["pct_chg"].sum()), 2)

    post3 = df[df.index > ann_idx].head(3)
    post_pct_3d = round(float(post3["pct_chg"].sum()), 2) if not post3.empty else None

    if reaction_pct >= 7:
        verdict = "强烈正面"
        reason  = f"财报公告当日涨幅 {reaction_pct:+.2f}%，市场反应热烈，大幅超预期"
    elif reaction_pct >= 3:
        verdict = "正面"
        reason  = f"财报公告当日涨幅 {reaction_pct:+.2f}%，市场认可度高，优于预期"
    elif reaction_pct >= -2:
        verdict = "中性"
        reason  = f"财报公告当日涨跌 {reaction_pct:+.2f}%，市场反应平淡，符合预期"
    elif reaction_pct >= -5:
        verdict = "负面"
        reason  = f"财报公告当日下跌 {reaction_pct:+.2f}%，市场对财报有所失望"
    else:
        verdict = "强烈负面"
        reason  = f"财报公告当日大跌 {reaction_pct:+.2f}%，财报明显低于预期"

    if vol_x and vol_x >= 2.0:
        reason += f"，成交量是平日 {vol_x:.1f} 倍（资金高度关注）"

    return {
        "ann_date":      ann_date,
        "reaction_pct":  reaction_pct,
        "reaction_vol_x": vol_x,
        "pre_pct_5d":    pre_pct_5d,
        "post_pct_3d":   post_pct_3d,
        "verdict":       verdict,
        "verdict_reason": reason,
    }


# ──────────────────────────────────────────
# 4. 财报深度分析报告
# ──────────────────────────────────────────

def format_earnings_analysis(
    code: str,
    name: str,
    history: list[dict],
    reactions: list[dict],
) -> str:
    """将历史财务数据 + 历次财报市场反应格式化为中文分析报告。"""
    lines = [
        "═" * 56,
        f"  财报深度分析  {name}（{code}）",
        f"  {cn_now().strftime('%Y-%m-%d %H:%M')}",
        "═" * 56,
    ]

    # 一、历史财务趋势
    lines += ["", "【历史财务数据趋势】（近期优先）", ""]

    header = f"  {'报告期':<12} {'类型':<6} {'营收(亿)':<10} {'营收同比':<10} {'净利润(亿)':<11} {'利润同比':<10} {'毛利率':<8} {'ROE'}"
    lines.append(header)
    lines.append("  " + "─" * 78)

    for h in history:
        rv_yoy = f"{h['revenue_yoy']:+.1f}%" if h['revenue_yoy'] is not None else "N/A"
        py_yoy = f"{h['profit_yoy']:+.1f}%" if h['profit_yoy'] is not None else "N/A"
        rv     = f"{h['revenue']:.2f}" if h['revenue'] is not None else "N/A"
        np_    = f"{h['net_profit']:.2f}" if h['net_profit'] is not None else "N/A"
        gm     = f"{h['gross_margin']:.1f}%" if h['gross_margin'] is not None else "N/A"
        roe    = f"{h['roe']:.1f}%" if h['roe'] is not None else "N/A"

        rv_arrow = "▲" if h['revenue_yoy'] and h['revenue_yoy'] > 0 else ("▼" if h['revenue_yoy'] and h['revenue_yoy'] < 0 else " ")
        py_arrow = "▲" if h['profit_yoy'] and h['profit_yoy'] > 0 else ("▼" if h['profit_yoy'] and h['profit_yoy'] < 0 else " ")

        lines.append(
            f"  {h['report_date']:<12} {h['report_type']:<6} "
            f"{rv:<8} {rv_arrow}{rv_yoy:<9} "
            f"{np_:<9} {py_arrow}{py_yoy:<9} "
            f"{gm:<8} {roe}"
        )

    if len(history) >= 2:
        lines.append("")
        latest = history[0]
        prev   = history[1]
        trends = []
        if latest["profit_yoy"] is not None:
            if latest["profit_yoy"] > 20:
                trends.append(f"净利润同比增长 {latest['profit_yoy']:.1f}%，盈利增速强劲")
            elif latest["profit_yoy"] > 0:
                trends.append(f"净利润同比增长 {latest['profit_yoy']:.1f}%，保持正增长")
            elif latest["profit_yoy"] > -20:
                trends.append(f"净利润同比下滑 {abs(latest['profit_yoy']):.1f}%，盈利承压")
            else:
                trends.append(f"净利润同比大幅下滑 {abs(latest['profit_yoy']):.1f}%，基本面走弱")
        rv_yoy = latest.get("revenue_yoy")
        if rv_yoy is not None:
            if rv_yoy > 30:
                trends.append(f"营收同比增长 {rv_yoy:.1f}%，增速强劲")
            elif rv_yoy > 0:
                trends.append(f"营收同比增长 {rv_yoy:.1f}%，保持正增长")
            elif rv_yoy > -20:
                trends.append(f"营收同比下滑 {abs(rv_yoy):.1f}%，收入承压")
            else:
                trends.append(f"营收同比大幅下滑 {abs(rv_yoy):.1f}%，需关注需求侧")
            # Revenue-profit divergence warning (margin compression signal)
            if latest.get("profit_yoy") is not None:
                gap = rv_yoy - latest["profit_yoy"]
                if gap > 15:
                    trends.append(f"⚠️ 营收增速({rv_yoy:.1f}%)显著高于利润增速({latest['profit_yoy']:.1f}%)，关注费用/毛利率恶化")
        if latest["gross_margin"] is not None and prev["gross_margin"] is not None:
            gm_delta = latest["gross_margin"] - prev["gross_margin"]
            if abs(gm_delta) > 1:
                trends.append(f"毛利率较上期{'提升' if gm_delta > 0 else '下降'} {abs(gm_delta):.1f} pct")
        if trends:
            lines.append("  趋势解读: " + "；".join(trends))

    # 二、历次财报市场反应
    if reactions:
        lines += ["", "─" * 56, "【历次财报公告市场反应】", ""]
        for r in reactions:
            verdict_icon = {
                "强烈正面": "++", "正面": "+", "中性": "=",
                "负面": "-", "强烈负面": "--"
            }.get(r.get("verdict", ""), "")

            lines.append(f"  {verdict_icon} {r['ann_date']}  {r.get('verdict','')}")
            lines.append(f"     {r.get('verdict_reason','')}")
            if r.get("pre_pct_5d") is not None:
                lines.append(f"     公告前5日累涨跌: {r['pre_pct_5d']:+.2f}%"
                             + (f"  |  公告后3日: {r['post_pct_3d']:+.2f}%" if r.get("post_pct_3d") is not None else ""))
            lines.append("")

    # 三、综合建议
    lines += ["─" * 56, "【财报对当前分析的影响】", ""]
    latest_r = reactions[0] if reactions else None
    if latest_r:
        v = latest_r.get("verdict", "")
        if v in ("强烈正面", "正面"):
            lines += [
                "  · 最新财报超预期，股价已做出正面反应",
                "  · 基本面形成支撑，回调可视为买点参考",
                "  · news_sentiment 建议在技术分析基础上 +0.2 ~ +0.3",
            ]
        elif v == "中性":
            lines += [
                "  · 财报符合预期，短期价格波动主要由技术面/资金面驱动",
                "  · news_sentiment 维持原有外围市场判断，无需额外调整",
            ]
        else:
            lines += [
                "  · 财报低于预期，股价已承压，基本面利空尚未完全消化",
                "  · news_sentiment 建议在技术分析基础上 -0.2 ~ -0.3",
                "  · 建议等待财报数据被市场充分定价后再考虑入场",
            ]

    lines += [
        "",
        "  财报分析仅供参考，不构成投资建议。",
        "═" * 56,
    ]
    return "\n".join(lines)
