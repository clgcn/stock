"""
风控面 (RiskFace)
==================
短线 + 长线共用维度 — 风险评估与仓位建议 (独立约束，不参与加权)

分析内容:
  ① VaR / CVaR — 在险价值
  ② Kelly仓位 — 最优仓位比例
  ③ ATR止损止盈 — 动态止损位
  ④ 压力测试 — 极端场景损失

底层依赖:
  data_fetcher.get_kline()                     → K线数据
  risk_manager.calc_var()                      → VaR/CVaR
  risk_manager.kelly_from_history()            → Kelly仓位
  risk_manager.calc_stop_levels()              → 止损止盈
  risk_manager.stress_test()                   → 压力测试
  risk_manager.comprehensive_risk_assessment() → 综合风险报告
"""

from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_fetcher import get_kline, get_kline_prefer_db
import risk_manager as rm


# ══════════════════════════════════════════════════════
# 结构化信号
# ══════════════════════════════════════════════════════

@dataclass
class RiskSignals:
    """风控面结构化信号。"""

    # VaR
    var_95_pct: Optional[float] = None       # 95%置信度 VaR %
    cvar_95_pct: Optional[float] = None      # 95% CVaR %
    var_99_pct: Optional[float] = None       # 99%置信度 VaR %

    # Kelly仓位
    kelly_position_pct: Optional[float] = None   # 半Kelly建议仓位 %
    win_rate: Optional[float] = None             # 历史胜率 %
    odds_ratio: Optional[float] = None           # 盈亏比

    # ATR止损
    atr_14: Optional[float] = None
    stop_loss_price: Optional[float] = None
    stop_loss_pct: Optional[float] = None        # 止损幅度 %
    take_profit_price: Optional[float] = None
    take_profit_pct: Optional[float] = None      # 止盈幅度 %

    # 压力测试
    max_daily_drop_pct: Optional[float] = None   # 历史最大单日跌幅 %
    max_3day_drop_pct: Optional[float] = None    # 历史最大连续3日跌幅 %
    mc_5d_worst_1pct: Optional[float] = None     # 蒙特卡洛5日1%分位 %

    # 综合风险等级
    risk_level: str = "medium"                   # "low" / "medium" / "high" / "extreme"


@dataclass
class RiskResult:
    """风控面分析完整结果。"""
    signals: RiskSignals
    risk_report: str = ""
    stop_levels: Dict = field(default_factory=dict)
    stress_scenarios: Dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════
# 风控面类
# ══════════════════════════════════════════════════════

class RiskFace:
    """
    风控面分析模块。

    公开接口:
      analyze()          → RiskResult   执行完整风控面分析
      score()            → list         风险管理评分 (独立约束，weight=0)
      check_veto()       → list         Kelly≤0 / VaR>8% / 极端风险
      position_advice()  → dict         基于总分+风控信号给出仓位建议

    内部方法:
      _fetch_kline()             获取K线数据
      _fetch_var()               计算VaR/CVaR
      _fetch_kelly()             计算Kelly仓位
      _fetch_stop_levels()       计算ATR止损止盈
      _fetch_stress_test()       执行压力测试
      _extract_stress()          从压力测试场景提取信号
      _classify_risk_level()     综合判定风险等级
    """

    # ╔══════════════════════════════════════════════════╗
    # ║  公开接口                                        ║
    # ╚══════════════════════════════════════════════════╝

    @staticmethod
    def analyze(
        stock_code: str,
        position_value: float = 100000,
        analysis_days: int = 250,
        kline_df=None,
        local_only: bool = False,
    ) -> RiskResult:
        """
        执行完整风控面分析。

        Parameters
        ----------
        stock_code     : str       股票代码
        position_value : float     持仓市值
        analysis_days  : int       历史分析天数
        kline_df       : DataFrame 可选传入K线 (避免重复请求)
        local_only     : bool      True=纯本地模式（选股用），DB无数据直接跳过
        """
        signals = RiskSignals()
        risk_report = ""
        stop_levels = {}
        stress_scenarios = {}

        # ── 获取K线数据 ──
        df = kline_df
        if df is None:
            df, err = RiskFace._fetch_kline(stock_code, analysis_days, local_only=local_only)
            if err:
                return RiskResult(signals=signals, risk_report=err)
        if df is None or len(df) < 30:
            return RiskResult(signals=signals, risk_report="数据不足: 需≥30条")

        # ── 1. VaR / CVaR ──
        RiskFace._fetch_var(signals, df, position_value)

        # ── 2. Kelly仓位 ──
        RiskFace._fetch_kelly(signals, df)

        # ── 3. ATR止损止盈 ──
        stop_levels = RiskFace._fetch_stop_levels(signals, df)

        # ── 4. 压力测试 ──
        stress_scenarios = RiskFace._fetch_stress_test(signals, df, position_value)

        # ── 5. 综合风险报告 ──
        try:
            risk_report = rm.comprehensive_risk_assessment(df, position_value=position_value)
        except Exception as e:
            risk_report = f"综合风险评估失败: {e}"

        # ── 综合风险等级判定 ──
        RiskFace._classify_risk_level(signals)

        return RiskResult(
            signals=signals,
            risk_report=risk_report,
            stop_levels=stop_levels,
            stress_scenarios=stress_scenarios,
        )

    @staticmethod
    def score(signals: RiskSignals) -> list:
        """
        风控面评分 → [风险管理(weight=0, 独立约束)]。

        评分规则 (0-10, 越高=风险越可控):
          VaR<2%→+3  <4%→+2  <6%→+1  >8%→-2
          Kelly 10~40%→+2  >0→+1  ≤0→-1
          ATR止损<8%→+1.5
          历史极端跌幅>10%→-1
        """
        score = 5.0

        # VaR维度
        var = signals.var_95_pct
        if var is not None:
            if var < 2:
                score += 3.0
            elif var < 4:
                score += 2.0
            elif var < 6:
                score += 1.0
            elif var > 8:
                score -= 2.0

        # Kelly维度
        kelly = signals.kelly_position_pct
        if kelly is not None:
            if 10 <= kelly <= 40:
                score += 2.0
            elif kelly > 0:
                score += 1.0
            else:
                score -= 1.0

        # 止损明确性
        if signals.atr_14 is not None and signals.stop_loss_pct is not None:
            if signals.stop_loss_pct < 8:
                score += 1.5
            else:
                score += 0.5

        # 压力测试惩罚
        if signals.max_daily_drop_pct is not None and abs(signals.max_daily_drop_pct) > 10:
            score -= 1.0

        score = max(0, min(10, score))
        return [
            {"name": "风险管理", "score": round(score, 1), "weight": 0.0, "max": 10},
        ]

    @staticmethod
    def check_veto(signals: RiskSignals) -> list:
        """
        风控面一票否决。

        触发条件:
          · Kelly仓位 ≤ 0 (负期望策略) → 仓位建议=0
          · VaR(95%) > 8%              → 极端波动警告
          · 综合风险等级 = extreme      → 建议空仓
        """
        vetos = []
        if signals.kelly_position_pct is not None and signals.kelly_position_pct <= 0:
            vetos.append((True, "Kelly仓位建议为0（负期望策略）", None))
        if signals.var_95_pct is not None and signals.var_95_pct > 8:
            vetos.append((True, f"VaR(95%)={signals.var_95_pct:.1f}%>8% 极端波动", None))
        if signals.risk_level == "extreme":
            vetos.append((True, "综合风险等级: 极端", None))
        return vetos

    @staticmethod
    def position_advice(signals: RiskSignals, total_score: float) -> dict:
        """
        基于风控信号和评分卡总分给出仓位建议。

        规则:
          总分→基础仓位: ≥80→30%  ≥65→20%  ≥50→10%  <50→5%
          Kelly上限约束
          风险等级修正: extreme→0%  high→≤10%
        """
        advice = {
            "position_pct": 0,
            "stop_loss": signals.stop_loss_price,
            "take_profit": signals.take_profit_price,
            "rationale": "",
        }

        kelly_cap = signals.kelly_position_pct if signals.kelly_position_pct and signals.kelly_position_pct > 0 else 30

        if total_score >= 80:
            base = 30
        elif total_score >= 65:
            base = 20
        elif total_score >= 50:
            base = 10
        else:
            base = 5

        position = min(base, kelly_cap)

        if signals.risk_level == "extreme":
            position = 0
            advice["rationale"] = "极端风险，建议空仓观望"
        elif signals.risk_level == "high":
            position = min(position, 10)
            advice["rationale"] = "高风险，限制仓位≤10%"
        elif signals.risk_level == "low":
            advice["rationale"] = f"风险可控，建议仓位{position}%"
        else:
            advice["rationale"] = f"中等风险，建议仓位{position}%"

        advice["position_pct"] = round(position, 1)
        return advice

    # ╔══════════════════════════════════════════════════╗
    # ║  数据获取                                        ║
    # ╚══════════════════════════════════════════════════╝

    @staticmethod
    def _fetch_kline(stock_code: str, analysis_days: int, local_only: bool = False):
        """获取K线数据。local_only=True 时纯本地，不发 API。Returns: (df, error_msg)"""
        try:
            from datetime import datetime, timedelta
            start = (datetime.today() - timedelta(days=analysis_days)).strftime("%Y-%m-%d")
            df = get_kline_prefer_db(stock_code, period="daily", start=start,
                                      adjust="qfq", local_only=local_only)
            if df is None:
                return None, "本地DB无数据（local_only模式）"
            return df, None
        except Exception as e:
            return None, f"K线数据获取失败: {e}"

    @staticmethod
    def _fetch_var(signals: RiskSignals, df, position_value: float):
        """计算VaR/CVaR并填充信号。"""
        try:
            var95 = rm.calc_var(df, confidence=0.95, position_value=position_value)
            if "error" not in var95:
                signals.var_95_pct = var95["var_pct"]
                signals.cvar_95_pct = var95["cvar_pct"]
            var99 = rm.calc_var(df, confidence=0.99, position_value=position_value)
            if "error" not in var99:
                signals.var_99_pct = var99["var_pct"]
        except Exception:
            pass

    @staticmethod
    def _fetch_kelly(signals: RiskSignals, df):
        """计算Kelly仓位并填充信号。"""
        try:
            kelly = rm.kelly_from_history(df)
            if "error" not in kelly:
                signals.kelly_position_pct = kelly["kelly_adjusted_pct"]
                signals.win_rate = kelly["win_rate"]
                signals.odds_ratio = kelly["odds_ratio"]
        except Exception:
            pass

    @staticmethod
    def _fetch_stop_levels(signals: RiskSignals, df) -> dict:
        """计算ATR止损止盈并填充信号。Returns: stop_levels dict"""
        try:
            stops = rm.calc_stop_levels(df)
            signals.atr_14 = stops.get("atr_14")
            signals.stop_loss_price = stops.get("atr_stop_loss")
            signals.stop_loss_pct = stops.get("atr_stop_loss_pct")
            signals.take_profit_price = stops.get("atr_take_profit")
            signals.take_profit_pct = stops.get("atr_take_profit_pct")
            return stops
        except Exception:
            return {}

    @staticmethod
    def _fetch_stress_test(signals: RiskSignals, df, position_value: float) -> dict:
        """执行压力测试并提取信号。Returns: scenarios dict"""
        try:
            stress = rm.stress_test(df, position_value=position_value)
            if "error" not in stress:
                scenarios = stress.get("scenarios", {})
                RiskFace._extract_stress(signals, scenarios)
                return scenarios
        except Exception:
            pass
        return {}

    # ╔══════════════════════════════════════════════════╗
    # ║  信号提取                                        ║
    # ╚══════════════════════════════════════════════════╝

    @staticmethod
    def _extract_stress(signals: RiskSignals, scenarios: dict):
        """从压力测试场景提取: 最大单日跌幅 / 最大3日跌幅 / 蒙特卡洛5日极端。"""
        for name, data in scenarios.items():
            drop = data.get("drop_pct", 0)
            if "max single-day" in name.lower() or "单日" in name:
                signals.max_daily_drop_pct = drop
            elif "3-day" in name.lower() or "3日" in name:
                signals.max_3day_drop_pct = drop
            elif "5-day extreme" in name.lower() and "1st" in name.lower():
                signals.mc_5d_worst_1pct = drop

    # ╔══════════════════════════════════════════════════╗
    # ║  综合判定                                        ║
    # ╚══════════════════════════════════════════════════╝

    @staticmethod
    def _classify_risk_level(signals: RiskSignals):
        """
        综合判定风险等级: low / medium / high / extreme。

        风险积分:
          VaR>8%→+3  >5%→+2  >3%→+1
          Kelly≤0→+3  <10%→+1
          历史最大日跌>10%→+2
        判定: ≥6→extreme  ≥4→high  ≥2→medium  <2→low
        """
        risk_points = 0
        var = signals.var_95_pct
        if var is not None:
            if var > 8:
                risk_points += 3
            elif var > 5:
                risk_points += 2
            elif var > 3:
                risk_points += 1
        kelly = signals.kelly_position_pct
        if kelly is not None:
            if kelly <= 0:
                risk_points += 3
            elif kelly < 10:
                risk_points += 1
        if signals.max_daily_drop_pct is not None and abs(signals.max_daily_drop_pct) > 10:
            risk_points += 2

        if risk_points >= 6:
            signals.risk_level = "extreme"
        elif risk_points >= 4:
            signals.risk_level = "high"
        elif risk_points >= 2:
            signals.risk_level = "medium"
        else:
            signals.risk_level = "low"
