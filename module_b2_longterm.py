"""
模块 B2 — 长线分析 Pipeline
============================
架构位置: 第2层（基本面质量） + 第3层（估值+治理）

长线路径调用链:
  1. financial_data (PE/PB/ROE)        → 财务质量
  2. valuation_quality (PEG/杜邦)      → 估值安全边际
  3. balance_sheet (负债/商誉)         → 负债与商誉
  4. earnings_analysis (业绩质量)      → 成长质量
  5. stock_announcements (公告)        → 成长质量补充
  6. dividend_history (分红记录)       → 分红与治理

重点: 财务质量 + 估值安全边际
总调用: 7-8个工具，可分多日完成

输出: LongTermResult dict，包含各维度原始数据 + 用于评分卡的结构化信号
"""

import re
from typing import TypedDict, Optional
from common_utils import _extract_float, _extract_float_warn, _module_header


class LongTermSignals(TypedDict):
    """长线各维度的结构化信号，供评分卡使用。"""

    # 财务质量 (权重30%)
    roe: Optional[float]                # ROE (%)
    net_profit_growth: Optional[float]  # 净利润增速 (%)
    gross_margin: Optional[float]       # 毛利率 (%)
    gross_margin_stable: bool           # 近3年毛利率波动<3%
    fcf_to_net_profit: Optional[float]  # 自由现金流/净利润 (%)

    # 估值安全边际 (权重25%)
    peg: Optional[float]
    pe_ttm: Optional[float]
    pb: Optional[float]
    pe_percentile_5y: Optional[float]   # PE处于近5年的百分位
    pb_vs_industry: Optional[float]     # PB vs 行业均值倍数

    # 负债与商誉 (权重20%)
    debt_ratio: Optional[float]         # 资产负债率 (%)
    goodwill_ratio: Optional[float]     # 商誉/净资产 (%)

    # 成长质量 (权重15%)
    revenue_growth_3y: bool             # 近3年营收+净利润双增
    growth_accelerating: bool           # 增速加速（今年>去年）
    earnings_beat: bool                 # 业绩超预期
    no_regulatory_issue: bool           # 公告无异常（无重大诉讼/监管函）
    growth_slowing_2q: bool             # 增速放缓连续2季

    # 分红与治理 (权重10%)
    consecutive_div_years: int          # 连续分红年数
    dividend_yield: Optional[float]     # 股息率 (%)
    mgmt_increase: bool                 # 高管近1年增持记录
    no_fund_misuse: bool                # 无大股东资金占用记录
    major_holder_pledge_pct: Optional[float]  # 大股东质押比例 (%)


class LongTermResult(TypedDict):
    """长线分析完整结果。"""
    # 原始报告
    financial_report: str
    valuation_report: str
    balance_report: str
    earnings_report: Optional[str]
    announcements_report: str
    dividend_report: Optional[str]

    # 结构化信号
    signals: LongTermSignals


def run_longterm_pipeline(
    stock_code: str,
    stock_name: str,
    # MCP tool functions
    financial_data_fn=None,
    valuation_quality_fn=None,
    balance_sheet_fn=None,
    earnings_analysis_fn=None,
    stock_announcements_fn=None,
    dividend_history_fn=None,
    # Options
    include_valuation: bool = True,
    include_balance: bool = True,
    include_dividend: bool = True,
) -> LongTermResult:
    """
    执行长线分析完整 pipeline。

    Parameters
    ----------
    stock_code : str   股票代码
    stock_name : str   股票名称
    *_fn : callable    各MCP工具函数
    """
    def _safe(fn, *a, label="", **kw):
        if fn is None:
            return f"⏸ {label} 未提供"
        try:
            return fn(*a, **kw)
        except Exception as e:
            return f"{label} 获取失败: {e}"

    # ── 1. 财务数据 (PE/PB/ROE) ──
    financial_report = _safe(financial_data_fn, stock_code, label="financial_data")

    # ── 2. PEG + 杜邦分析 ──
    valuation_report = "Skipped valuation_quality by request."
    if include_valuation:
        valuation_report = _safe(valuation_quality_fn, stock_code=stock_code, label="valuation_quality")

    # ── 3. 资产负债 + 商誉预警 ──
    balance_report = "Skipped balance_sheet by request."
    if include_balance:
        balance_report = _safe(balance_sheet_fn, stock_code=stock_code, label="balance_sheet")

    # ── 4. 公告面（先查公告，用于判断成长质量）──
    ann_report = _safe(stock_announcements_fn, stock_code, label="stock_announcements")

    # ── 5. 财报分析（条件触发）──
    earnings_report = None
    keywords = ("年报", "半年报", "季报", "一季报", "三季报", "业绩预告", "业绩快报", "业绩修正")
    if any(kw in ann_report for kw in keywords):
        if earnings_analysis_fn:
            try:
                earnings_report = earnings_analysis_fn(stock_code)
            except Exception:
                pass

    # ── 6. 分红历史 ──
    dividend_report = None
    if include_dividend:
        dividend_report = _safe(dividend_history_fn, stock_code=stock_code, label="dividend_history")

    # ── 提取结构化信号 ──
    signals = _extract_signals(
        financial_report=financial_report,
        valuation_report=valuation_report,
        balance_report=balance_report,
        earnings_report=earnings_report,
        ann_report=ann_report,
        dividend_report=dividend_report,
    )

    return LongTermResult(
        financial_report=financial_report,
        valuation_report=valuation_report,
        balance_report=balance_report,
        earnings_report=earnings_report,
        announcements_report=ann_report,
        dividend_report=dividend_report,
        signals=signals,
    )


# ══════════════════════════════════════════════════════
# 内部辅助函数
# ══════════════════════════════════════════════════════


def _extract_signals(
    financial_report: str,
    valuation_report: str,
    balance_report: str,
    earnings_report: Optional[str],
    ann_report: str,
    dividend_report: Optional[str],
) -> LongTermSignals:
    """从各工具报告中提取结构化信号。"""

    # ── 财务质量 ──
    roe = _extract_float_warn(financial_report, r"ROE\s*[：:]\s*(\d+(?:\.\d+)?)%?", "roe")
    net_profit_growth = _extract_float_warn(
        valuation_report or financial_report,
        r"净利润增[速长率][：:]\s*([+-]?\d+(?:\.\d+)?)%?",
        "net_profit_growth",
    )
    gross_margin = _extract_float_warn(
        financial_report,
        r"(?:毛|Gross).*?(?:率|Margin)\s*[：:]\s*(\d+(?:\.\d+)?)%?",
        "gross_margin",
    )
    gross_margin_stable = "毛利率稳定" in (valuation_report or "") or "波动小" in (valuation_report or "")
    fcf_ratio = _extract_float_warn(
        valuation_report or "", r"自由现金流/净利润[：:]\s*(\d+(?:\.\d+)?)%?", "fcf_ratio"
    )

    # ── 估值安全边际 ──
    peg = _extract_float_warn(valuation_report, r"PEG\s*[：:=]\s*(\d+(?:\.\d+)?)", "peg")
    pe_ttm = _extract_float_warn(financial_report, r"PE\s*(?:TTM)?\s*[：:]\s*(\d+(?:\.\d+)?)", "pe_ttm")
    pb = _extract_float_warn(financial_report, r"PB\s*[：:]\s*(\d+(?:\.\d+)?)", "pb")
    pe_pct = _extract_float_warn(valuation_report, r"PE.*?百分位[：:]\s*(\d+(?:\.\d+)?)%?", "pe_percentile")
    pb_vs = _extract_float_warn(valuation_report, r"PB.*?行业.*?(\d+(?:\.\d+)?)倍", "pb_vs_industry")

    # ── 负债与商誉 ──
    debt_ratio = _extract_float_warn(balance_report, r"资产负债率[：:]\s*(\d+(?:\.\d+)?)%?", "debt_ratio")
    goodwill_ratio = _extract_float_warn(
        balance_report, r"商誉.*?占.*?净资产\s*(\d+(?:\.\d+)?)%?", "goodwill_ratio"
    )
    if goodwill_ratio is None:
        goodwill_ratio = _extract_float_warn(balance_report, r"占净资产\s*(\d+(?:\.\d+)?)%", "goodwill_ratio")

    # ── 成长质量 ──
    revenue_3y = "营收" in (earnings_report or "") and "连续增长" in (earnings_report or "")
    if not revenue_3y:
        revenue_3y = "双增" in (earnings_report or "")
    growth_accel = "加速" in (earnings_report or "")
    beat = "超预期" in (earnings_report or "") or "超出预期" in (earnings_report or "")
    no_reg = not any(kw in ann_report for kw in ("收到监管", "监管问询", "被监管", "问询函", "警示函", "诉讼", "立案"))
    slowing = "放缓" in (earnings_report or "") and "连续" in (earnings_report or "")

    # ── 分红与治理 ──
    div_years = 0
    if dividend_report:
        # 优先匹配"连续X年分红"（精确），回退到"共X次"（次≠年，按次数/2估算）
        m_year = re.search(r"连续\s*(\d+)\s*年", dividend_report)
        m_count = re.search(r"共\s*(\d+)\s*次现金分红", dividend_report)
        if m_year:
            div_years = int(m_year.group(1))
        elif m_count:
            div_years = max(1, int(m_count.group(1)) // 2)  # 假设平均2次/年
    div_yield = _extract_float_warn(dividend_report or "", r"股息率[：:]\s*(\d+(?:\.\d+)?)%?", "dividend_yield")
    mgmt_inc = "增持" in (ann_report or "") and "高管" in (ann_report or "")
    no_misuse = not any(kw in ann_report for kw in ("资金占用", "关联交易", "违规担保"))
    pledge_pct = _extract_float_warn(ann_report, r"质押比例[：:]\s*(\d+(?:\.\d+)?)%?", "pledge_pct")

    return LongTermSignals(
        roe=roe,
        net_profit_growth=net_profit_growth,
        gross_margin=gross_margin,
        gross_margin_stable=gross_margin_stable,
        fcf_to_net_profit=fcf_ratio,
        peg=peg,
        pe_ttm=pe_ttm,
        pb=pb,
        pe_percentile_5y=pe_pct,
        pb_vs_industry=pb_vs,
        debt_ratio=debt_ratio,
        goodwill_ratio=goodwill_ratio,
        revenue_growth_3y=revenue_3y,
        growth_accelerating=growth_accel,
        earnings_beat=beat,
        no_regulatory_issue=no_reg,
        growth_slowing_2q=slowing,
        consecutive_div_years=div_years,
        dividend_yield=div_yield,
        mgmt_increase=mgmt_inc,
        no_fund_misuse=no_misuse,
        major_holder_pledge_pct=pledge_pct,
    )


def format_longterm_report(result: LongTermResult, stock_name: str, stock_code: str) -> str:
    """格式化长线分析报告。"""
    sections = [
        *_module_header("【模块 B2】长线分析", stock_name, stock_code),
        "▌ 财务数据（PE/PB/ROE/毛利率/净利率）", result["financial_report"],
        "",
        "▌ 估值质量（PEG + 杜邦分析）", result["valuation_report"],
        "",
        "▌ 资产负债健康度（负债率 + 商誉预警）", result["balance_report"],
        "",
        "▌ 公告/事件面", result["announcements_report"],
    ]
    if result["earnings_report"]:
        sections.extend(["", "▌ 财报分析（业绩质量）", result["earnings_report"]])
    if result["dividend_report"]:
        sections.extend(["", "▌ 分红历史", result["dividend_report"]])
    return "\n".join(sections)
