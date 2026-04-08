"""
基本面 (FundamentalFace)
========================
长线核心维度 — 财务质量(30%) + 估值(25%) + 负债商誉(20%) + 成长(15%) + 分红治理(10%)

分析内容:
  ① 财务质量 — ROE/毛利率/净利率/现金流
  ② 估值安全边际 — PEG + 杜邦分析
  ③ 负债与商誉 — 资产负债率 + 商誉/净资产
  ④ 成长质量 — 营收/净利润增长趋势
  ⑤ 分红与治理 — 分红历史 + 大股东行为

底层依赖:
  financial.get_financial()            → PE/PB/ROE等
  financial.get_financial_history()    → 历史财务数据
  financial.compute_peg()              → PEG计算
  financial.get_balance_sheet()        → 资产负债表
  financial.get_dividend_history()     → 分红记录
  quant_engine.analyze_dupont()        → 杜邦分析
"""

from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from financial import (
    get_financial, get_financial_history,
    get_balance_sheet, get_dividend_history, compute_peg,
)
import quant_engine as qe


# ══════════════════════════════════════════════════════
# 结构化信号
# ══════════════════════════════════════════════════════

@dataclass
class FundamentalSignals:
    """基本面结构化信号。"""

    # 财务质量 (权重30%)
    roe: Optional[float] = None
    net_profit_growth: Optional[float] = None
    gross_margin: Optional[float] = None
    net_margin: Optional[float] = None
    gross_margin_stable: bool = False       # 近3年毛利率波动<3%
    fcf_to_net_profit: Optional[float] = None

    # 估值安全边际 (权重25%)
    pe_ttm: Optional[float] = None
    pb: Optional[float] = None
    peg: Optional[float] = None
    dupont_score: Optional[float] = None

    # 负债与商誉 (权重20%)
    debt_ratio: Optional[float] = None      # %
    goodwill: Optional[float] = None        # 亿
    goodwill_ratio: Optional[float] = None  # %
    cash: Optional[float] = None            # 亿

    # 成长质量 (权重15%)
    revenue_growth_3y: bool = False
    growth_accelerating: bool = False
    growth_slowing_2q: bool = False

    # 分红与治理 (权重10%)
    consecutive_div_years: int = 0
    dividend_yield: Optional[float] = None
    mgmt_increase: bool = False
    no_fund_misuse: bool = True
    major_holder_pledge_pct: Optional[float] = None


@dataclass
class FundamentalResult:
    """基本面分析完整结果。"""
    signals: FundamentalSignals
    financial_report: str = ""
    valuation_report: str = ""
    balance_report: str = ""
    dividend_report: str = ""


# ══════════════════════════════════════════════════════
# 基本面类
# ══════════════════════════════════════════════════════

class FundamentalFace:
    """
    基本面分析模块。

    公开接口:
      analyze()     → FundamentalResult   执行完整基本面分析
      score()       → list                5个子维度评分
      check_veto()  → list                ROE<8% / PEG>2 / 商誉>50% / 资金占用

    内部方法:
      _fetch_financial()           获取财务数据
      _fetch_valuation()           获取PEG+杜邦+历史数据
      _fetch_balance()             获取资产负债表
      _fetch_dividend()            获取分红历史
      _extract_financial()         提取ROE/PE/PB/利润率
      _extract_balance()           提取负债率/商誉/现金
      _extract_dividend()          提取连续分红年数
      _extract_growth()            提取成长信号
      _score_financial_quality()   财务质量评分
      _score_valuation()           估值评分
      _score_debt_goodwill()       负债商誉评分
      _score_growth()              成长质量评分
      _score_dividend()            分红治理评分
      _format_financial()          格式化财务报告
      _format_valuation()          格式化估值报告
      _format_balance()            格式化负债报告
      _format_dividend()           格式化分红报告
    """

    # ╔══════════════════════════════════════════════════╗
    # ║  公开接口                                        ║
    # ╚══════════════════════════════════════════════════╝

    @staticmethod
    def analyze(
        stock_code: str,
        include_valuation: bool = True,
        include_balance: bool = True,
        include_dividend: bool = True,
        local_only: bool = False,
    ) -> FundamentalResult:
        """
        执行完整基本面分析。

        Parameters
        ----------
        stock_code        : str  股票代码
        include_valuation : bool 是否分析PEG+杜邦
        include_balance   : bool 是否分析资产负债
        include_dividend  : bool 是否分析分红历史
        local_only        : bool True时从本地DB读取基础估值数据，跳过需要API的深度分析
        """
        signals = FundamentalSignals()
        fin_report = ""
        val_report = ""
        bal_report = ""
        div_report = ""

        if local_only:
            # ── 本地模式: 从 stock_fundamentals 快照读取 PE/PB ──
            try:
                import slow_fetcher as sf
                overview = sf.load_stock_overview(stock_code)
                if overview:
                    signals.pe_ttm = overview.get("pe_ttm")
                    signals.pb = overview.get("pb")
                    # ROE 近似 = 1/PE × 1/PB × 100 (仅作粗估)
                    if signals.pe_ttm and signals.pb and signals.pe_ttm > 0:
                        signals.roe = (signals.pb / signals.pe_ttm) * 100
                    fin_report = (
                        f"  [本地DB] PE(TTM): {signals.pe_ttm or 'N/A'}"
                        f"  PB: {signals.pb or 'N/A'}"
                        f"  ROE(估): {signals.roe:.1f}%" if signals.roe else
                        f"  [本地DB] PE(TTM): {signals.pe_ttm or 'N/A'}"
                        f"  PB: {signals.pb or 'N/A'}"
                    )
                else:
                    fin_report = "  [本地DB] 无基本面快照数据"
            except Exception as e:
                fin_report = f"  [本地DB] 基本面读取失败: {e}"
            # local_only 模式跳过 PEG/杜邦/负债/分红（需要API）
            val_report = ""
            bal_report = ""
            div_report = ""
        else:
            # ── 1. 财务数据 (PE/PB/ROE) ──
            fin_data, fin_report = FundamentalFace._fetch_financial(stock_code)
            if fin_data:
                FundamentalFace._extract_financial(signals, fin_data)

            # ── 2. PEG + 杜邦分析 ──
            if include_valuation:
                val_report = FundamentalFace._fetch_valuation(stock_code, signals)

            # ── 3. 资产负债 + 商誉 ──
            if include_balance:
                bal_data, bal_report = FundamentalFace._fetch_balance(stock_code)
                if bal_data:
                    FundamentalFace._extract_balance(signals, bal_data)

            # ── 4. 分红历史 ──
            if include_dividend:
                div_data, div_report = FundamentalFace._fetch_dividend(stock_code)
                if div_data:
                    FundamentalFace._extract_dividend(signals, div_data)

        return FundamentalResult(
            signals=signals,
            financial_report=fin_report,
            valuation_report=val_report,
            balance_report=bal_report,
            dividend_report=div_report,
        )

    @staticmethod
    def score(signals: FundamentalSignals) -> list:
        """
        基本面评分 → 长线5个子维度。

        [财务质量(30%), 估值安全边际(25%), 负债与商誉(20%), 成长质量(15%), 分红与治理(10%)]
        """
        return [
            {"name": "财务质量", "score": FundamentalFace._score_financial_quality(signals), "weight": 0.30, "max": 10},
            {"name": "估值安全边际", "score": FundamentalFace._score_valuation(signals), "weight": 0.25, "max": 10},
            {"name": "负债与商誉", "score": FundamentalFace._score_debt_goodwill(signals), "weight": 0.20, "max": 10},
            {"name": "成长质量", "score": FundamentalFace._score_growth(signals), "weight": 0.15, "max": 10},
            {"name": "分红与治理", "score": FundamentalFace._score_dividend(signals), "weight": 0.10, "max": 10},
        ]

    @staticmethod
    def check_veto(signals: FundamentalSignals) -> list:
        """
        基本面一票否决。

        触发条件:
          · ROE < 8%              → 财务质量≤3
          · PEG > 2.0             → 高估警告
          · 商誉/净资产 > 50%     → 负债商誉≤2
          · 资金占用/实控人变更    → 分红治理=0
        """
        vetos = []
        if signals.roe is not None and signals.roe < 8:
            vetos.append((True, f"ROE={signals.roe:.1f}%<8%", 3.0, "财务质量"))
        if signals.peg is not None and signals.peg > 2.0:
            vetos.append((True, f"PEG={signals.peg:.2f}>2.0 高估警告", None, None))
        if signals.goodwill_ratio is not None and signals.goodwill_ratio > 50:
            vetos.append((True, f"商誉/净资产={signals.goodwill_ratio:.0f}%>50% 一票否决", 2.0, "负债商誉"))
        if not signals.no_fund_misuse:
            vetos.append((True, "存在资金占用或实控人变更", 0, "分红治理"))
        return vetos

    # ╔══════════════════════════════════════════════════╗
    # ║  数据获取                                        ║
    # ╚══════════════════════════════════════════════════╝

    @staticmethod
    def _fetch_financial(stock_code: str):
        """获取财务数据。Returns: (data_dict, report_str)"""
        try:
            data = get_financial(stock_code)
            report = FundamentalFace._format_financial(data, stock_code)
            return data, report
        except Exception as e:
            return None, f"财务数据获取失败: {e}"

    @staticmethod
    def _fetch_valuation(stock_code: str, signals: FundamentalSignals) -> str:
        """获取PEG + 杜邦分析 + 毛利率稳定性 + 成长趋势。Returns: report_str"""
        try:
            history = get_financial_history(stock_code, periods=8)
            pe_ttm = signals.pe_ttm
            peg_result = {}

            # PEG
            if history and pe_ttm:
                peg_result = compute_peg(history, pe_ttm)
                signals.peg = peg_result.get("peg")

            # 杜邦分析
            dupont = {}
            if history:
                dupont = qe.analyze_dupont(history)
                signals.dupont_score = dupont.get("score")

                # 毛利率稳定性
                margins = [h.get("gross_margin") for h in history[:6] if h.get("gross_margin") is not None]
                if len(margins) >= 3:
                    import numpy as np
                    signals.gross_margin_stable = float(np.std(margins)) < 3.0

                # 成长趋势
                FundamentalFace._extract_growth(signals, history)

            return FundamentalFace._format_valuation(signals, peg_result, dupont)
        except Exception as e:
            return f"估值分析失败: {e}"

    @staticmethod
    def _fetch_balance(stock_code: str):
        """获取资产负债表。Returns: (data_dict, report_str)"""
        try:
            data = get_balance_sheet(stock_code)
            report = FundamentalFace._format_balance(data, stock_code)
            return data, report
        except Exception as e:
            return None, f"资产负债表获取失败: {e}"

    @staticmethod
    def _fetch_dividend(stock_code: str):
        """获取分红历史。Returns: (data_list, report_str)"""
        try:
            data = get_dividend_history(stock_code, years=5)
            report = FundamentalFace._format_dividend(data, stock_code)
            return data, report
        except Exception as e:
            return None, f"分红数据获取失败: {e}"

    # ╔══════════════════════════════════════════════════╗
    # ║  信号提取                                        ║
    # ╚══════════════════════════════════════════════════╝

    @staticmethod
    def _extract_financial(signals: FundamentalSignals, data: dict):
        """从财务数据提取: ROE / PE / PB / 利润率。"""
        signals.roe = data.get("roe")
        signals.pe_ttm = data.get("pe_ttm")
        signals.pb = data.get("pb")
        signals.gross_margin = data.get("gross_margin")
        signals.net_margin = data.get("net_margin")

    @staticmethod
    def _extract_balance(signals: FundamentalSignals, data: dict):
        """从资产负债表提取: 负债率 / 商誉 / 现金。"""
        signals.debt_ratio = data.get("debt_ratio")
        signals.goodwill = data.get("goodwill")
        signals.goodwill_ratio = data.get("goodwill_ratio")
        signals.cash = data.get("cash")

    @staticmethod
    def _extract_dividend(signals: FundamentalSignals, data: list):
        """从分红历史提取: 连续分红年数。"""
        if not data:
            return
        divs = [d for d in data if d.get("div_per_share") and d["div_per_share"] > 0]
        signals.consecutive_div_years = len(divs)

    @staticmethod
    def _extract_growth(signals: FundamentalSignals, history: list):
        """从历史财务数据提取: 营收3年增长 / 加速 / 减速。"""
        if len(history) < 3:
            return
        revenues = [h.get("revenue") for h in history[:6] if h.get("revenue") is not None]
        profits = [h.get("net_profit") for h in history[:6] if h.get("net_profit") is not None]
        if len(revenues) >= 3:
            signals.revenue_growth_3y = all(revenues[i] >= revenues[i+1] for i in range(min(3, len(revenues)-1)))
        if len(profits) >= 2:
            growths = [(profits[i] - profits[i+1]) / abs(profits[i+1]) * 100
                       if profits[i+1] != 0 else 0
                       for i in range(min(3, len(profits)-1))]
            if len(growths) >= 2 and growths[0] > growths[1]:
                signals.growth_accelerating = True
            if len(growths) >= 2 and growths[0] < growths[1] and growths[0] < 5:
                signals.growth_slowing_2q = True

    # ╔══════════════════════════════════════════════════╗
    # ║  评分子函数                                       ║
    # ╚══════════════════════════════════════════════════╝

    @staticmethod
    def _score_financial_quality(s: FundamentalSignals) -> float:
        """财务质量评分: ROE + 净利润增速 + 毛利率稳定 + 自由现金流。"""
        subs = []
        roe = s.roe
        if roe is not None:
            subs.append(10 if roe >= 20 else 8 if roe >= 15 else 5 if roe >= 10 else 3 if roe >= 5 else 1)
        else:
            subs.append(5)
        npg = s.net_profit_growth
        if npg is not None:
            subs.append(10 if npg >= 25 else 7 if npg >= 15 else 5 if npg >= 5 else 3 if npg >= 0 else 1)
        else:
            subs.append(5)
        avg = sum(subs) / len(subs)
        bonus = (2 if s.gross_margin_stable else 0) + (2 if s.fcf_to_net_profit and s.fcf_to_net_profit >= 80 else 0)
        return round(max(0, min(10, avg + bonus)), 1)

    @staticmethod
    def _score_valuation(s: FundamentalSignals) -> float:
        """估值评分: PEG<0.8→9.5  0.8~1.0→7~8  1.0~1.5→4~6  >1.5→1~3。"""
        peg = s.peg
        if peg is None:
            score = 5.0
        elif peg < 0.8:
            score = 9.5
        elif peg < 1.0:
            score = 7 + (1.0 - peg) / 0.2
        elif peg < 1.5:
            score = 4 + (1.5 - peg) / 0.5 * 2
        else:
            score = max(1, 3 - (peg - 1.5))
        return round(max(0, min(10, score)), 1)

    @staticmethod
    def _score_debt_goodwill(s: FundamentalSignals) -> float:
        """负债商誉评分: 负债率≤40→9+  55→6~8  70→3~5  >70→0~2  + 商誉修正。"""
        dr = s.debt_ratio
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
        gw = s.goodwill_ratio
        if gw is not None:
            if gw < 10:
                score += 1
            elif gw > 30:
                score -= 2
        return round(max(0, min(10, score)), 1)

    @staticmethod
    def _score_growth(s: FundamentalSignals) -> float:
        """成长质量评分: 3年双增→6  加速→+2  减速→-2。"""
        score = 4.0
        if s.revenue_growth_3y:
            score = 6.0
        if s.growth_accelerating:
            score += 2.0
        if s.growth_slowing_2q:
            score -= 2.0
        return round(max(0, min(10, score)), 1)

    @staticmethod
    def _score_dividend(s: FundamentalSignals) -> float:
        """分红治理评分: 连续分红年数 + 股息率 + 管理层增持 + 无资金占用 - 高质押。"""
        yrs = s.consecutive_div_years
        score = 5.0 if yrs >= 5 else 3.0 if yrs >= 3 else 2.0 if yrs >= 1 else 1.0
        dy = s.dividend_yield
        if dy is not None:
            score += 3 if dy > 4 else 2 if dy > 2 else 0
        if s.mgmt_increase:
            score += 2
        if s.no_fund_misuse:
            score += 1
        pledge = s.major_holder_pledge_pct
        if pledge is not None and pledge > 50:
            score -= 3
        return round(max(0, min(10, score)), 1)

    # ╔══════════════════════════════════════════════════╗
    # ║  格式化                                          ║
    # ╚══════════════════════════════════════════════════╝

    @staticmethod
    def _format_financial(data: dict, code: str) -> str:
        """格式化财务指标报告。"""
        def _f(v, s="", fmt=".2f"):
            if v is None: return "N/A"
            try: return f"{v:{fmt}}{s}"
            except Exception: return str(v)
        return "\n".join([
            f"财务指标 — {data.get('name', code)} ({code})",
            "-" * 40,
            f"  最新价: {_f(data.get('current'))}",
            f"  PE TTM: {_f(data.get('pe_ttm'))}   PB: {_f(data.get('pb'))}",
            f"  ROE: {_f(data.get('roe'), '%')}   毛利率: {_f(data.get('gross_margin'), '%')}   净利率: {_f(data.get('net_margin'), '%')}",
            f"  总市值: {_f(data.get('total_mv'), ' 亿')}   流通市值: {_f(data.get('float_mv'), ' 亿')}",
        ])

    @staticmethod
    def _format_valuation(signals, peg_result: dict, dupont: dict) -> str:
        """格式化估值质量报告。"""
        lines = ["估值质量分析", "-" * 40]
        if signals.peg is not None:
            lines.append(f"  PEG: {signals.peg:.2f}")
        if signals.dupont_score is not None:
            lines.append(f"  杜邦综合评分: {signals.dupont_score:.1f}")
        return "\n".join(lines)

    @staticmethod
    def _format_balance(data: dict, code: str) -> str:
        """格式化资产负债表报告。"""
        dr = data.get("debt_ratio")
        gw_r = data.get("goodwill_ratio")
        lines = [
            f"资产负债表 — {code}",
            f"  资产负债率: {dr:.1f}%" if dr else "  资产负债率: N/A",
            f"  商誉/净资产: {gw_r:.1f}%" if gw_r else "  商誉: N/A",
            f"  货币资金: {data.get('cash', 'N/A')} 亿",
        ]
        if gw_r and gw_r > 30:
            lines.append("  ⚠️ 高商誉警告！减值风险")
        return "\n".join(lines)

    @staticmethod
    def _format_dividend(data: list, code: str) -> str:
        """格式化分红历史报告。"""
        if not data:
            return f"{code} 近5年无分红记录"
        divs = [d for d in data if d.get("div_per_share") and d["div_per_share"] > 0]
        return f"{code} 近5年共 {len(divs)} 次现金分红"
