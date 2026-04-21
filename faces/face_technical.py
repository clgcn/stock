"""
技术面 (TechnicalFace)
======================
短线核心维度 — 技术形态共振(25%) + 量化诊断(10%)

分析内容:
  ① 均线系统 (5/10/20日) — 多头排列、空头排列、缠绕
  ② 量价配合 — 放量突破压力位、缩量回调
  ③ MACD / KDJ / RSI 超买超卖
  ④ K线形态 — 涨停板、缩量回调、锤子线、吞没形态
  ⑤ 布林带位置

底层依赖:
  data_fetcher.get_kline()                 → K线原始数据
  data_fetcher.add_indicators()            → 计算MA/MACD/RSI/BOLL/KDJ
  quant_engine.comprehensive_diagnosis()   → 7维综合诊断 + 蒙特卡洛
  quant_engine.analyze_kdj_patterns()      → KDJ信号 + K线形态识别
  quant_engine.analyze_volume()            → 量价配合分析
"""

from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from _http_utils import cn_now
from data_fetcher import get_kline, get_kline_prefer_db, add_indicators
import quant_engine as qe


# ══════════════════════════════════════════════════════
# 结构化信号
# ══════════════════════════════════════════════════════

@dataclass
class TechnicalSignals:
    """技术面结构化信号，供评分卡使用。"""

    # 均线系统
    ma_alignment: str = "mixed"         # "bullish" / "bearish" / "mixed"
    ma5: Optional[float] = None
    ma10: Optional[float] = None
    ma20: Optional[float] = None
    price_vs_ma20: str = "unknown"      # "above" / "below" / "at"

    # MACD
    macd_signal: str = "neutral"        # "golden_cross" / "death_cross" / "bullish" / "bearish" / "neutral"
    macd_hist: Optional[float] = None
    macd_expanding: bool = False

    # RSI
    rsi: Optional[float] = None
    rsi_zone: str = "neutral"           # "overbought" / "oversold" / "neutral"
    rsi_divergence: str = "none"        # "bullish" / "bearish" / "none"

    # KDJ
    kdj_k: Optional[float] = None
    kdj_d: Optional[float] = None
    kdj_j: Optional[float] = None
    kdj_signal: str = "neutral"         # "golden_cross" / "death_cross" / "overbought" / "oversold" / "neutral"
    kdj_j_cross_50: bool = False

    # 布林带
    boll_position: str = "middle"       # "upper" / "middle" / "lower" / "above_upper" / "below_lower"
    boll_bandwidth: Optional[float] = None

    # 量价配合
    volume_trend: str = "normal"        # "expansion" / "contraction" / "normal"
    volume_price_divergence: str = "none"  # "bullish" / "bearish" / "none"
    obv_trend: str = "neutral"          # "bullish" / "bearish" / "neutral"

    # K线形态
    patterns: list = field(default_factory=list)

    # 7维综合诊断
    diagnosis_total_score: Optional[float] = None  # -100 ~ +100
    diagnosis_signal: str = "hold"      # "buy" / "sell" / "hold"

    # 蒙特卡洛
    mc_up_prob: Optional[float] = None
    mc_expected_range: Optional[tuple] = None

    # 各维度原始得分
    trend_score: Optional[float] = None
    momentum_score: Optional[float] = None
    volatility_score: Optional[float] = None
    volume_score: Optional[float] = None


@dataclass
class TechnicalResult:
    """技术面分析完整结果。"""
    signals: TechnicalSignals
    kline_report: str = ""
    diagnosis_report: str = ""
    raw_factors: Dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════
# 技术面类
# ══════════════════════════════════════════════════════

class TechnicalFace:
    """
    技术面分析模块。

    公开接口:
      analyze()     → TechnicalResult   执行完整技术面分析
      score()       → list              技术形态共振(25%) + 量化诊断(10%)
      check_veto()  → list              RSI超买 / 均线空头 / 诊断极弱

    内部方法:
      _fetch_kline()              获取K线 + 指标
      _extract_indicator_values() 从最新行提取MA/MACD/RSI/KDJ/BOLL
      _extract_diagnosis()        从7维诊断结果提取信号
      _extract_kdj_patterns()     从KDJ分析提取信号+K线形态
      _extract_volume()           从量价分析提取信号
      _format_kline_report()      生成K线文本报告
    """

    # ╔══════════════════════════════════════════════════╗
    # ║  公开接口                                        ║
    # ╚══════════════════════════════════════════════════╝

    @staticmethod
    def analyze(
        stock_code: str,
        analysis_days: int = 250,
        news_sentiment: float = 0.0,
        monte_carlo_days: int = 20,
        kline_recent_days: int = 60,
        local_only: bool = False,
    ) -> TechnicalResult:
        """
        执行完整技术面分析。

        Parameters
        ----------
        stock_code        : str   股票代码
        analysis_days     : int   诊断用历史天数 (默认250)
        news_sentiment    : float 新闻情绪 (由环境层提供)
        monte_carlo_days  : int   蒙特卡洛模拟天数
        kline_recent_days : int   K线展示天数
        local_only        : bool  True=纯本地模式（选股用），DB无数据直接跳过
        """
        signals = TechnicalSignals()
        kline_report = ""
        diagnosis_report = ""
        raw_factors = {}

        # ── 1. 获取K线 + 技术指标 ──
        df, df_ind, err = TechnicalFace._fetch_kline(stock_code, analysis_days, local_only=local_only)
        if err:
            return TechnicalResult(signals=signals, kline_report=err, diagnosis_report=err)

        # ── 2. 提取指标信号 ──
        latest = df_ind.iloc[-1]
        TechnicalFace._extract_indicator_values(signals, latest, df_ind)

        # ── 3. K线文本报告 ──
        kline_report = TechnicalFace._format_kline_report(stock_code, df_ind, kline_recent_days)

        # ── 4. 7维综合诊断 ──
        try:
            result = qe.comprehensive_diagnosis(
                df, news_sentiment=news_sentiment,
                run_monte_carlo=True, mc_days=monte_carlo_days,
            )
            diagnosis_report = result.get("summary", "")
            raw_factors = result.get("factors", {})
            TechnicalFace._extract_diagnosis(signals, result)
        except Exception as e:
            diagnosis_report = f"7维诊断失败: {e}"

        # ── 5. KDJ + K线形态 ──
        try:
            kdj_result = qe.analyze_kdj_patterns(df)
            TechnicalFace._extract_kdj_patterns(signals, kdj_result)
        except Exception:
            pass

        # ── 6. 量价分析 ──
        try:
            vol_result = qe.analyze_volume(df)
            TechnicalFace._extract_volume(signals, vol_result)
        except Exception:
            pass

        return TechnicalResult(
            signals=signals,
            kline_report=kline_report,
            diagnosis_report=diagnosis_report,
            raw_factors=raw_factors,
        )

    @staticmethod
    def score(signals: TechnicalSignals) -> list:
        """
        技术面评分 → [技术形态共振(25%), 量化诊断(10%)]。

        技术形态共振 (0-10):
          均线排列 0-3 + MACD 0-3 + RSI 0-2 + KDJ 0-2

        量化诊断 (0-10):
          诊断总分(-100~+100)映射 + 蒙特卡洛加分
        """
        # ── 技术形态共振 ──
        tech_score = 0.0
        # 均线排列 (0-3)
        if signals.ma_alignment == "bullish":
            tech_score += 3.0
        elif signals.ma_alignment == "mixed":
            tech_score += 1.5
        # MACD (0-3)
        macd_map = {"golden_cross": 3.0, "bullish": 2.0, "neutral": 1.5, "bearish": 0.5, "death_cross": 0}
        tech_score += macd_map.get(signals.macd_signal, 1.5)
        # RSI (0-2)
        if signals.rsi is not None:
            if 40 <= signals.rsi <= 70:
                tech_score += 2.0
            elif 30 <= signals.rsi < 40 or 70 < signals.rsi <= 80:
                tech_score += 1.0
        # KDJ (0-2)
        if signals.kdj_j_cross_50:
            tech_score += 2.0
        elif signals.kdj_j is not None and signals.kdj_j > 50:
            tech_score += 1.0

        tech_score = max(0, min(10, tech_score))

        # ── 量化诊断 ──
        diag_score = 5.0
        if signals.diagnosis_total_score is not None:
            diag_score = (signals.diagnosis_total_score + 100) / 20
        if signals.mc_up_prob is not None and signals.mc_up_prob > 55:
            diag_score += 1.0
        diag_score = max(0, min(10, diag_score))

        return [
            {"name": "技术形态共振", "score": round(tech_score, 1), "weight": 0.25, "max": 10},
            {"name": "量化诊断", "score": round(diag_score, 1), "weight": 0.10, "max": 10},
        ]

    @staticmethod
    def check_veto(signals: TechnicalSignals) -> list:
        """
        技术面一票否决。

        触发条件:
          · RSI > 85        → 技术形态共振≤3
          · 均线空头排列     → 技术形态共振≤3
          · 诊断总分 < -40  → 整体惩罚
        """
        vetos = []
        if signals.rsi is not None and signals.rsi > 85:
            vetos.append((True, f"RSI={signals.rsi:.0f}>85 超买", 3.0, "技术形态共振"))
        if signals.ma_alignment == "bearish":
            vetos.append((True, "均线空头排列", 3.0, "技术形态共振"))
        if signals.diagnosis_total_score is not None and signals.diagnosis_total_score < -40:
            vetos.append((True, f"量化诊断评分={signals.diagnosis_total_score:.0f}，极弱", None, None))
        return vetos

    # ╔══════════════════════════════════════════════════╗
    # ║  数据获取                                        ║
    # ╚══════════════════════════════════════════════════╝

    @staticmethod
    def _fetch_kline(stock_code: str, analysis_days: int, local_only: bool = False):
        """
        获取K线 + 叠加技术指标。
        local_only=True: 纯本地模式，DB 没有数据就返回错误，绝不发 API。
        local_only=False: DB 优先，不足时回退 API（走限流器+缓存）。

        Returns: (df_raw, df_with_indicators, error_msg)
        """
        start = (cn_now() - timedelta(days=analysis_days)).strftime("%Y-%m-%d")
        try:
            df = get_kline_prefer_db(stock_code, period="daily", start=start,
                                      adjust="hfq", local_only=local_only)
        except Exception as e:
            return None, None, f"K线数据获取失败: {e}"
        if df is None or len(df) < 30:
            return None, None, f"数据不足: 仅{len(df) if df is not None else 0}条，需≥30"
        df_ind = add_indicators(df.copy())
        return df, df_ind, None

    # ╔══════════════════════════════════════════════════╗
    # ║  信号提取                                        ║
    # ╚══════════════════════════════════════════════════╝

    @staticmethod
    def _extract_indicator_values(signals: TechnicalSignals, latest, df_ind):
        """从最新K线行提取 MA / MACD / RSI / KDJ / BOLL 指标值。"""
        _sf = TechnicalFace._safe_float

        # MA
        signals.ma5 = _sf(latest, "ma5")
        signals.ma10 = _sf(latest, "ma10")
        signals.ma20 = _sf(latest, "ma20")
        close = float(latest["close"])
        if signals.ma20 is not None:
            signals.price_vs_ma20 = "above" if close > signals.ma20 else "below"
        # 均线排列
        if signals.ma5 and signals.ma10 and signals.ma20:
            if signals.ma5 > signals.ma10 > signals.ma20:
                signals.ma_alignment = "bullish"
            elif signals.ma5 < signals.ma10 < signals.ma20:
                signals.ma_alignment = "bearish"
            else:
                signals.ma_alignment = "mixed"

        # MACD
        signals.macd_hist = _sf(latest, "macd")
        if signals.macd_hist is not None:
            signals.macd_signal = "bullish" if signals.macd_hist > 0 else "bearish"
            if len(df_ind) >= 2:
                prev_hist = _sf(df_ind.iloc[-2], "macd")
                if prev_hist is not None:
                    if prev_hist <= 0 < signals.macd_hist:
                        signals.macd_signal = "golden_cross"
                    elif prev_hist >= 0 > signals.macd_hist:
                        signals.macd_signal = "death_cross"
                    signals.macd_expanding = abs(signals.macd_hist) > abs(prev_hist)

        # RSI
        signals.rsi = _sf(latest, "rsi12") or _sf(latest, "rsi6")
        if signals.rsi is not None:
            if signals.rsi > 70:
                signals.rsi_zone = "overbought"
            elif signals.rsi < 30:
                signals.rsi_zone = "oversold"
            else:
                signals.rsi_zone = "neutral"

        # KDJ
        signals.kdj_k = _sf(latest, "kdj_k")
        signals.kdj_d = _sf(latest, "kdj_d")
        signals.kdj_j = _sf(latest, "kdj_j")

        # 布林带
        boll_ub = _sf(latest, "boll_ub")
        boll_lb = _sf(latest, "boll_lb")
        boll_mid = _sf(latest, "boll_mid") or signals.ma20
        if boll_ub and boll_lb:
            if close > boll_ub:
                signals.boll_position = "above_upper"
            elif close < boll_lb:
                signals.boll_position = "below_lower"
            elif boll_mid and close > boll_mid:
                signals.boll_position = "upper"
            elif boll_mid and close < boll_mid:
                signals.boll_position = "lower"
            else:
                signals.boll_position = "middle"
            if boll_mid and boll_mid > 0:
                signals.boll_bandwidth = (boll_ub - boll_lb) / boll_mid * 100

    @staticmethod
    def _extract_diagnosis(signals: TechnicalSignals, result: dict):
        """从 quant_engine.comprehensive_diagnosis 结果提取诊断分 + 蒙特卡洛。"""
        signals.diagnosis_total_score = result.get("total_score")
        sig = result.get("signal", "hold")
        signals.diagnosis_signal = sig if sig in ("buy", "sell", "hold") else "hold"
        # 蒙特卡洛
        mc = result.get("monte_carlo", {})
        if mc:
            signals.mc_up_prob = mc.get("prob_up")
            low = mc.get("percentile_5")
            high = mc.get("percentile_95")
            if low is not None and high is not None:
                signals.mc_expected_range = (low, high)
        # 各维度得分
        factors = result.get("factors", {})
        for key, factor in factors.items():
            score = factor.get("score", 0)
            if "trend" in key.lower():
                signals.trend_score = score
            elif "momentum" in key.lower():
                signals.momentum_score = score
            elif "volatil" in key.lower():
                signals.volatility_score = score
            elif "volume" in key.lower() or "量价" in factor.get("label", ""):
                signals.volume_score = score

    @staticmethod
    def _extract_kdj_patterns(signals: TechnicalSignals, kdj_result: dict):
        """从 quant_engine.analyze_kdj_patterns 提取 KDJ 信号 + K线形态。"""
        kdj_data = kdj_result.get("kdj", {})
        if kdj_data:
            if kdj_data.get("golden_cross"):
                signals.kdj_signal = "golden_cross"
            elif kdj_data.get("death_cross"):
                signals.kdj_signal = "death_cross"
            elif kdj_data.get("overbought"):
                signals.kdj_signal = "overbought"
            elif kdj_data.get("oversold"):
                signals.kdj_signal = "oversold"
            signals.kdj_j_cross_50 = bool(kdj_data.get("j_cross_50"))
        patterns = kdj_result.get("patterns", [])
        if patterns:
            signals.patterns = [p.get("name", str(p)) if isinstance(p, dict) else str(p) for p in patterns]

    @staticmethod
    def _extract_volume(signals: TechnicalSignals, vol_result: dict):
        """从 quant_engine.analyze_volume 提取 OBV / 量价背离 / 成交量趋势。"""
        details = vol_result.get("details", {})
        # OBV 趋势
        obv_slope = details.get("obv_slope")
        if obv_slope is not None:
            signals.obv_trend = "bullish" if obv_slope > 0 else "bearish"
        # 量价背离
        div = details.get("divergence")
        if div:
            if "bullish" in str(div).lower() or "看多" in str(div):
                signals.volume_price_divergence = "bullish"
            elif "bearish" in str(div).lower() or "看空" in str(div):
                signals.volume_price_divergence = "bearish"
        # 成交量趋势
        vol_ratio = details.get("volume_ratio") or details.get("vol_ratio")
        if vol_ratio is not None:
            if vol_ratio > 1.5:
                signals.volume_trend = "expansion"
            elif vol_ratio < 0.7:
                signals.volume_trend = "contraction"

    # ╔══════════════════════════════════════════════════╗
    # ║  格式化                                          ║
    # ╚══════════════════════════════════════════════════╝

    @staticmethod
    def _format_kline_report(stock_code: str, df_ind, recent_days: int) -> str:
        """生成最近N日K线 + 指标文本报告。"""
        last = df_ind.tail(min(recent_days, 20)).copy()
        last_date = last["date"].copy()
        last["date"] = last_date.dt.strftime("%Y-%m-%d") if hasattr(last_date.iloc[0], "strftime") else last_date
        base_cols = ["date", "open", "high", "low", "close", "pct_chg", "volume"]
        ind_cols = ["ma5", "ma20", "macd", "rsi12", "kdj_k", "boll_ub", "boll_lb"]
        show = base_cols + [c for c in ind_cols if c in last.columns]
        lines = [
            f"{stock_code} daily K-line   Total {len(df_ind)} bars",
            f"   Range: {df_ind.iloc[0]['date'].strftime('%Y-%m-%d')} -> {df_ind.iloc[-1]['date'].strftime('%Y-%m-%d')}",
            "-" * 70,
            last[show].to_string(index=False),
        ]
        return "\n".join(lines)

    # ╔══════════════════════════════════════════════════╗
    # ║  工具函数                                        ║
    # ╚══════════════════════════════════════════════════╝

    @staticmethod
    def _safe_float(row, col) -> Optional[float]:
        """安全类型转换为 float。"""
        try:
            v = row.get(col) if hasattr(row, "get") else row[col]
            if v is not None and str(v) not in ("", "nan", "None"):
                return float(v)
        except (KeyError, TypeError, ValueError, IndexError):
            pass
        return None
