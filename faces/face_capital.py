"""
资金面 (CapitalFace)
====================
短线核心维度 — 主力资金净入(30%) + 融资融券情绪(10%)

分析内容:
  ① 主力资金净流入 (大单比) — moneyflow
  ② 北向资金 (沪深港通) — northbound_flow (方向从环境层传入)
  ③ 融资融券余额变化 — margin_trading
  ④ 量化资金活跃度 — quant_activity

底层依赖:
  capital_flow.get_moneyflow()               → 主力大单数据
  capital_flow.format_moneyflow_report()     → 格式化报告
  capital_flow.get_margin_trading()          → 融资融券余额
  capital_flow.format_margin_report()        → 格式化报告
  quant_detector.get_quant_activity_report() → 量化活跃度
"""

from __future__ import annotations
import sys
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from capital_flow import (
    get_moneyflow, format_moneyflow_report,
    get_margin_trading, format_margin_report,
)
import quant_detector as qd


# ══════════════════════════════════════════════════════
# 结构化信号
# ══════════════════════════════════════════════════════

@dataclass
class CapitalSignals:
    """资金面结构化信号。"""

    # 主力资金 (权重30%)
    net_inflow_ratio: Optional[float] = None    # 大单净流入比 %
    net_inflow_5d_avg: Optional[float] = None   # 近5日均值 亿
    consecutive_outflow_days: int = 0            # 连续净流出天数
    main_force_trend: str = "neutral"            # "inflow" / "outflow" / "neutral"

    # 北向资金 (从环境层传入方向)
    northbound_5d_avg: Optional[float] = None
    northbound_consecutive_out: int = 0
    northbound_direction: str = "neutral"        # "inflow" / "outflow" / "neutral"

    # 融资融券 (权重10%)
    margin_balance_change_pct: Optional[float] = None  # 融资余额周环比 %
    margin_trend: str = "unknown"                       # "up" / "flat" / "down" / "unknown"
    short_selling_surge: bool = False                   # 融券余额快速放大

    # 量化资金
    quant_score: Optional[float] = None          # 0-100 活跃度
    quant_level: str = "unknown"                 # "dominant" / "active" / "present" / "low" / "unknown"
    quant_share_pct: Optional[float] = None      # 量化资金占比 %


@dataclass
class CapitalResult:
    """资金面分析完整结果。"""
    signals: CapitalSignals
    moneyflow_report: str = ""
    margin_report: str = ""
    quant_report: str = ""


# ══════════════════════════════════════════════════════
# 资金面类
# ══════════════════════════════════════════════════════

class CapitalFace:
    """
    资金面分析模块。

    公开接口:
      analyze()     → CapitalResult   执行完整资金面分析
      score()       → list            主力资金净入(30%) + 融资融券情绪(10%)
      check_veto()  → list            连续流出+融资下降 / 融券激增

    内部方法:
      _fetch_moneyflow()        获取主力资金数据
      _fetch_margin()           获取融资融券数据
      _fetch_quant()            获取量化活跃度
      _extract_moneyflow()      从主力资金数据提取信号
      _extract_margin()         从融资融券数据提取信号
      _extract_quant()          从量化活跃度报告提取信号
    """

    # ╔══════════════════════════════════════════════════╗
    # ║  公开接口                                        ║
    # ╚══════════════════════════════════════════════════╝

    @staticmethod
    def analyze(
        stock_code: str,
        include_margin: bool = False,
        include_quant: bool = True,
        moneyflow_days: int = 10,
        margin_days: int = 20,
        northbound_adj: float = 0.0,
        local_only: bool = False,
    ) -> CapitalResult:
        """
        执行完整资金面分析。

        Parameters
        ----------
        stock_code      : str   股票代码
        include_margin  : bool  是否分析融资融券
        include_quant   : bool  是否分析量化活跃度
        moneyflow_days  : int   主力资金查询天数
        margin_days     : int   融资融券查询天数
        northbound_adj  : float 北向资金修正值 (从环境层传入)
        local_only      : bool  True时跳过 moneyflow/margin API（选股批量模式）
        """
        signals = CapitalSignals()
        mf_report = ""
        mg_report = ""
        qt_report = ""

        # ── 1. 主力资金流向 ──
        if local_only:
            mf_data, mf_report, mf_err = CapitalFace._fetch_moneyflow_local(stock_code, moneyflow_days)
        else:
            mf_data, mf_report, mf_err = CapitalFace._fetch_moneyflow(stock_code, moneyflow_days)
        if not mf_err:
            CapitalFace._extract_moneyflow(signals, mf_data, mf_report)

        # ── 2. 北向资金方向 (从环境层传入) ──
        if northbound_adj > 0.05:
            signals.northbound_direction = "inflow"
        elif northbound_adj < -0.05:
            signals.northbound_direction = "outflow"

        # ── 3. 融资融券 ──
        if include_margin and not local_only:
            mg_data, mg_report, mg_err = CapitalFace._fetch_margin(stock_code, margin_days)
            if not mg_err:
                CapitalFace._extract_margin(signals, mg_data, mg_report)

        # ── 4. 量化资金活跃度 ──
        if include_quant:
            qt_report, qt_err = CapitalFace._fetch_quant(stock_code)
            if not qt_err:
                CapitalFace._extract_quant(signals, qt_report)

        return CapitalResult(
            signals=signals,
            moneyflow_report=mf_report,
            margin_report=mg_report,
            quant_report=qt_report,
        )

    @staticmethod
    def score(signals: CapitalSignals) -> list:
        """
        资金面评分 → [主力资金净入(30%), 融资融券情绪(10%)]。

        主力资金 (0-10):
          净流入比>5%→9+  2~5%→7~8  -1~2%→4~6  <-5%→0~1

        融资融券 (0-10):
          融资余额周环比↑→7+  持平→5  ↓→2~4  北向同向+1
        """
        # ── 主力资金 ──
        ratio = signals.net_inflow_ratio
        if ratio is None:
            fund_score = 5.0
        elif ratio >= 5:
            fund_score = min(10, 9 + (ratio - 5) / 10)
        elif ratio >= 2:
            fund_score = 7 + (ratio - 2) / 3
        elif ratio >= -1:
            fund_score = 4 + (ratio + 1) / 3 * 2
        elif ratio >= -5:
            fund_score = 2 + (ratio + 5) / 4
        else:
            fund_score = max(0, 1 + (ratio + 5) / 5)

        # ── 融资融券 ──
        change = signals.margin_balance_change_pct
        if change is None:
            mg_score = 5.0
        elif change > 3:
            mg_score = min(9, 7 + change / 10)
        elif change >= -3:
            mg_score = 5 + change / 6
        else:
            mg_score = max(2, 4 + change / 10)
        # 北向同向加分
        if signals.northbound_direction == "inflow":
            mg_score = min(10, mg_score + 1.0)

        return [
            {"name": "主力资金净入", "score": round(max(0, min(10, fund_score)), 1), "weight": 0.30, "max": 10},
            {"name": "融资融券情绪", "score": round(max(0, min(10, mg_score)), 1), "weight": 0.10, "max": 10},
        ]

    @staticmethod
    def check_veto(signals: CapitalSignals) -> list:
        """
        资金面一票否决。

        触发条件:
          · 连续3日主力净流出 + 融资余额下降 → 整体×0.7
          · 融券余额快速放大 (做空信号)       → 融资融券维度=0
        """
        vetos = []
        if signals.consecutive_outflow_days >= 3 and signals.margin_trend == "down":
            vetos.append((True, "连续3日主力净流出且融资余额下降", None, None))
        if signals.short_selling_surge:
            vetos.append((True, "融券余额快速放大（做空信号）", 0, "融资融券"))
        return vetos

    # ╔══════════════════════════════════════════════════╗
    # ║  数据获取                                        ║
    # ╚══════════════════════════════════════════════════╝

    @staticmethod
    def _fetch_moneyflow_local(stock_code: str, days: int):
        """从本地 DB 读取资金流向。无数据时返回空（不报错）。"""
        try:
            import slow_fetcher as sf
            data = sf.load_stock_moneyflow(stock_code, days=days)
            if not data:
                return [], "  [本地DB] 暂无资金流向数据（需先运行 --moneyflow 拉取）", "no_local_data"
            report = format_moneyflow_report(data)
            report = report.replace("主力资金流向报告", "主力资金流向报告 [本地DB]")
            return data, report, None
        except Exception as e:
            return [], f"  [本地DB] 资金流向读取失败: {e}", str(e)

    @staticmethod
    def _fetch_moneyflow(stock_code: str, days: int):
        """获取主力资金流向数据（在线API）。Returns: (data, report, error_msg)"""
        try:
            data = get_moneyflow(stock_code, days=days)
            report = format_moneyflow_report(data)
            return data, report, None
        except Exception as e:
            return [], f"主力资金获取失败: {e}", str(e)

    @staticmethod
    def _fetch_margin(stock_code: str, days: int):
        """获取融资融券数据。Returns: (data, report, error_msg)"""
        try:
            data = get_margin_trading(stock_code, days=days)
            report = format_margin_report(data)
            return data, report, None
        except Exception as e:
            return [], f"融资融券获取失败（可能非两融标的）: {e}", str(e)

    @staticmethod
    def _fetch_quant(stock_code: str):
        """获取量化活跃度报告。Returns: (report, error_msg)"""
        try:
            report = qd.get_quant_activity_report(stock_code)
            return report, None
        except Exception as e:
            return f"量化活跃度获取失败（可能非交易时段）: {e}", str(e)

    # ╔══════════════════════════════════════════════════╗
    # ║  信号提取                                        ║
    # ╚══════════════════════════════════════════════════╝

    @staticmethod
    def _extract_moneyflow(signals: CapitalSignals, data: list, report: str):
        """从主力资金数据提取: 净流入比 / 5日均值 / 连续流出天数 / 趋势。"""
        if not data:
            return
        # 近5日大单净流入均值
        recent5 = data[:5]
        if recent5:
            avg = sum(d.get("main_net", 0) for d in recent5) / len(recent5)
            signals.net_inflow_5d_avg = avg
            signals.main_force_trend = "inflow" if avg > 0 else "outflow"
        # 净流入比 (从报告提取)
        m = re.search(r"大单净流入比[：:]\s*([+-]?\d+(?:\.\d+)?)%", report)
        if m:
            signals.net_inflow_ratio = float(m.group(1))
        # 连续流出天数 (从原始数据计算)
        consec = 0
        for d in data:
            if d.get("main_net", 0) < 0:
                consec += 1
            else:
                break
        signals.consecutive_outflow_days = consec

    @staticmethod
    def _extract_margin(signals: CapitalSignals, data: list, report: str):
        """从融资融券数据提取: 融资余额变化 / 趋势 / 融券激增。"""
        if not data:
            return
        # 融资余额趋势
        if len(data) >= 2:
            latest_bal = data[0].get("margin_balance", 0)
            prev_bal = data[-1].get("margin_balance", 0) if len(data) >= 5 else data[1].get("margin_balance", 0)
            if prev_bal > 0:
                change_pct = (latest_bal - prev_bal) / prev_bal * 100
                signals.margin_balance_change_pct = change_pct
                if change_pct > 3:
                    signals.margin_trend = "up"
                elif change_pct < -3:
                    signals.margin_trend = "down"
                else:
                    signals.margin_trend = "flat"
        # 融券余额快速放大
        if "融券余额" in report and ("大幅" in report or "快速" in report):
            signals.short_selling_surge = True

    @staticmethod
    def _extract_quant(signals: CapitalSignals, report: str):
        """从量化活跃度报告提取: 得分 / 等级 / 占比。"""
        # 量化活跃度得分
        m = re.search(r"量化活跃度[得评]分[：:]\s*(\d+(?:\.\d+)?)", report)
        if m:
            signals.quant_score = float(m.group(1))
        # 等级
        if "量化主导" in report:
            signals.quant_level = "dominant"
        elif "量化活跃" in report:
            signals.quant_level = "active"
        elif "量化参与" in report:
            signals.quant_level = "present"
        elif "量化偏少" in report:
            signals.quant_level = "low"
        # 量化资金占比
        m2 = re.search(r"量化资金占比[：:]\s*(\d+(?:\.\d+)?)%", report)
        if m2:
            signals.quant_share_pct = float(m2.group(1))
