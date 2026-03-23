"""
Stock Screener — Two-Stage A-Share Market Scanner
===================================================
Scans all ~5000 A-share stocks to find the most promising candidates.

Stage 1 (Fast Filter):
  Reads the local SQLite market snapshot built from stocks + stock_history
  + stock_fundamentals, then applies basic metric filters (market cap, PE,
  PB, volume, etc.)
  to narrow down to ~50-100 candidates.

Stage 2 (Deep Scan):
  Runs the full 7-dimension quant_engine diagnosis on each candidate,
  scores and ranks them, returning the top N picks.

Built-in screening strategies:
  1. value       — Low PE + Low PB + High ROE (classic value investing)
  2. momentum    — Strong price trend + volume expansion (trend following)
  3. oversold    — Deeply negative deviation + low RSI (mean reversion)
  4. custom      — User-defined filter conditions
"""

import logging

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from stock_tool import get_kline
import quant_engine as qe

logger = logging.getLogger(__name__)

_HISTORY_PROFILE_CACHE: Dict[tuple, pd.DataFrame] = {}


def _decision_rank(result: Dict) -> tuple:
    """Sort buyable candidates ahead of hold/sell, then by quality."""
    decision = result.get("decision", {}) or {}
    action = decision.get("action", "hold")
    strength = decision.get("strength", "weak")
    action_order = {"buy": 3, "hold": 2, "sell": 1}
    strength_order = {"strong": 3, "medium": 2, "weak": 1}
    return (
        action_order.get(action, 0),
        strength_order.get(strength, 0),
        result.get("entry_quality_score", -999),
        result.get("total_score", -999),
        result.get("prob_up", -999) or -999,
    )


def _historical_rank(result: Dict) -> tuple:
    """Sort Stage 1 candidates by historical composite first, then strategy fit."""
    return (
        result.get("stage1_score", -999),
        result.get("history_score", -999),
        result.get("fast_score", -999),
    )


def _calc_return(closes: pd.Series, window: int) -> float:
    if len(closes) <= window:
        return float("nan")
    base = closes.iloc[-window - 1]
    last = closes.iloc[-1]
    if pd.isna(base) or pd.isna(last) or base == 0:
        return float("nan")
    return (last / base - 1.0) * 100.0


def _calc_max_drawdown(closes: pd.Series) -> float:
    if closes.empty:
        return float("nan")
    running_max = closes.cummax()
    drawdown = closes / running_max - 1.0
    return abs(float(drawdown.min())) * 100.0


def _score_label(score: float) -> str:
    if pd.isna(score):
        return "unknown"
    if score >= 35:
        return "strong"
    if score >= 10:
        return "good"
    if score <= -35:
        return "weak"
    if score <= -10:
        return "fragile"
    return "neutral"


def build_historical_profiles(lookback_days: int = 240,
                              min_history_days: int = 90) -> pd.DataFrame:
    """
    Build a cross-sectional historical profile for all stocks from stock_history.

    This is the true Stage 0 of screening:
      1) read historical OHLCV for all stocks from local DB
      2) compute medium/long-term trend, consistency, volatility, drawdown
      3) convert these into a single history_score for Stage 1 ranking
    """
    cache_key = (lookback_days, min_history_days)
    cached = _HISTORY_PROFILE_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    try:
        from slow_fetcher import load_all_history
        history = load_all_history()
    except Exception as e:
        logger.warning("Historical profile build failed: %s", e)
        return pd.DataFrame()

    if history.empty:
        return pd.DataFrame()

    history = history.sort_values(["code", "date"]).copy()
    grouped = history.groupby("code", sort=False)
    rows = []

    for code, g in grouped:
        g = g.tail(lookback_days).copy()
        if len(g) < max(30, min_history_days // 2):
            continue

        closes = pd.to_numeric(g["close"], errors="coerce").dropna()
        if len(closes) < max(30, min_history_days // 2):
            continue

        pct = pd.to_numeric(g.get("pct_chg"), errors="coerce")
        if pct.isna().all():
            pct = closes.pct_change() * 100.0
        pct = pct.replace([np.inf, -np.inf], np.nan)

        volume = pd.to_numeric(g.get("volume"), errors="coerce")

        ma20 = closes.tail(20).mean() if len(closes) >= 20 else np.nan
        ma60 = closes.tail(60).mean() if len(closes) >= 60 else np.nan
        ma120 = closes.tail(120).mean() if len(closes) >= 120 else np.nan
        last_close = closes.iloc[-1]

        ret_20 = _calc_return(closes, 20)
        ret_60 = _calc_return(closes, 60)
        ret_120 = _calc_return(closes, 120)
        positive_ratio = float((pct > 0).mean()) if len(pct.dropna()) else np.nan
        volatility_20 = float(pct.tail(20).std()) if len(pct.tail(20).dropna()) >= 10 else np.nan
        volatility_60 = float(pct.tail(60).std()) if len(pct.tail(60).dropna()) >= 20 else np.nan
        max_drawdown = _calc_max_drawdown(closes)
        price_vs_ma60 = ((last_close / ma60) - 1.0) * 100.0 if pd.notna(ma60) and ma60 else np.nan
        trend_alignment = 0.0
        if pd.notna(ma20) and pd.notna(ma60):
            trend_alignment += 0.5 if ma20 > ma60 else -0.5
        if pd.notna(ma60) and pd.notna(ma120):
            trend_alignment += 0.5 if ma60 > ma120 else -0.5
        volume_trend = np.nan
        if len(volume.dropna()) >= 60:
            vol20 = volume.tail(20).mean()
            vol60 = volume.tail(60).mean()
            if pd.notna(vol20) and pd.notna(vol60) and vol60 > 0:
                volume_trend = (vol20 / vol60 - 1.0) * 100.0

        rows.append({
            "code": code,
            "history_days": int(len(closes)),
            "hist_ret_20": ret_20,
            "hist_ret_60": ret_60,
            "hist_ret_120": ret_120,
            "hist_positive_ratio": positive_ratio * 100.0 if pd.notna(positive_ratio) else np.nan,
            "hist_volatility_20": volatility_20,
            "hist_volatility_60": volatility_60,
            "hist_max_drawdown": max_drawdown,
            "hist_price_vs_ma60": price_vs_ma60,
            "hist_trend_alignment": trend_alignment,
            "hist_volume_trend": volume_trend,
        })

    profile = pd.DataFrame(rows)
    if profile.empty:
        return profile

    def _rank(series: pd.Series, reverse: bool = False) -> pd.Series:
        rank = series.rank(pct=True)
        if reverse:
            rank = 1.0 - rank
        return rank.fillna(0.5)

    score = (
        (_rank(profile["hist_ret_20"]) - 0.5) * 2 * 8
        + (_rank(profile["hist_ret_60"]) - 0.5) * 2 * 18
        + (_rank(profile["hist_ret_120"]) - 0.5) * 2 * 18
        + (_rank(profile["hist_positive_ratio"]) - 0.5) * 2 * 12
        + (_rank(profile["hist_price_vs_ma60"]) - 0.5) * 2 * 12
        + (_rank(profile["hist_trend_alignment"]) - 0.5) * 2 * 12
        + (_rank(profile["hist_volume_trend"]) - 0.5) * 2 * 8
        + (_rank(profile["hist_volatility_20"], reverse=True) - 0.5) * 2 * 5
        + (_rank(profile["hist_volatility_60"], reverse=True) - 0.5) * 2 * 5
        + (_rank(profile["hist_max_drawdown"], reverse=True) - 0.5) * 2 * 10
    )

    history_eligibility = np.where(profile["history_days"] >= min_history_days, 0.0, -15.0)
    profile["history_score"] = np.clip(score + history_eligibility, -100, 100).round(1)
    profile["history_label"] = profile["history_score"].apply(_score_label)
    profile["history_ready"] = profile["history_days"] >= min_history_days

    _HISTORY_PROFILE_CACHE[cache_key] = profile.copy()
    return profile


def attach_historical_profiles(df: pd.DataFrame,
                               lookback_days: int = 240,
                               min_history_days: int = 90) -> pd.DataFrame:
    """Merge Stage 0 historical profiles into the market snapshot."""
    if df.empty:
        return df
    profile = build_historical_profiles(
        lookback_days=lookback_days,
        min_history_days=min_history_days,
    )
    if profile.empty:
        enriched = df.copy()
        enriched["history_score"] = np.nan
        enriched["history_label"] = "unknown"
        enriched["history_days"] = 0
        enriched["history_ready"] = False
        return enriched
    return df.merge(profile, on="code", how="left")

def fetch_all_stocks(market: str = "all",
                     max_retries: int = 3) -> pd.DataFrame:
    """
    Fetch all A-share main-board stocks from the local SQLite snapshot.

    Parameters:
        market       "all" / "sh" / "sz"
        max_retries  Unused, retained for backward compatibility

    Returns:
        DataFrame (see column list in _parse_records)
    """
    try:
        from slow_fetcher import load_stocks_from_db
        df = load_stocks_from_db()
        if df.empty:
            logger.warning("Local DB snapshot is empty; please import stocks first.")
            return df
        logger.info("Loaded %d stocks from local DB", len(df))
        return df
    except (ImportError, FileNotFoundError) as e:
        logger.warning("Local DB not available: %s", e)
        return pd.DataFrame()


# =====================================================
# 2. Stage 1: Fast Filter Strategies
# =====================================================

def _base_filter(df: pd.DataFrame, exclude_st: bool = True) -> pd.Series:
    """
    Common base filter applied to ALL strategies:
    - Exclude ST stocks (special treatment, high risk)
    - Exclude stocks at or near daily limit up (pct_chg >= 9.5%)
      to avoid chasing rallies (main board limit is 10%)
    - Exclude stocks at daily limit down (pct_chg <= -9.5%)
      because they are likely untradeable (no buyers)
    - Exclude suspended stocks (volume == 0 or current == 0)
    """
    mask = pd.Series(True, index=df.index)

    # Exclude ST / *ST stocks (special treatment, high delisting risk)
    # Match names starting with "ST" or "*ST" (A-share naming convention)
    if exclude_st:
        mask &= ~df["name"].str.match(r"^\*?ST\b", case=False, na=False)

    # Exclude stocks at or near daily limit up/down
    # Main board limit: +/-10%, using 9.5% threshold to catch near-limit stocks
    mask &= df["pct_chg"].notna()
    mask &= df["pct_chg"] < 9.5     # Not at or near limit up (avoid chasing)
    mask &= df["pct_chg"] > -9.5    # Not at or near limit down (untradeable)

    # Exclude suspended / zero-volume stocks
    mask &= df["volume"].notna() & (df["volume"] > 0)
    mask &= df["current"].notna() & (df["current"] > 0)

    return mask


def filter_value(df: pd.DataFrame,
                 pe_max: float = 25,
                 pe_min: float = 0,
                 pb_max: float = 3,
                 pb_min: float = 0,
                 mv_min: float = 50,
                 turnover_min: float = 0.5,
                 exclude_st: bool = True) -> pd.DataFrame:
    """
    Value stock filter.
    Criteria: Reasonable PE, low PB, decent market cap, some liquidity.
    """
    mask = _base_filter(df, exclude_st=exclude_st)
    mask &= df["pe_ttm"].notna() & (df["pe_ttm"] > pe_min) & (df["pe_ttm"] <= pe_max)
    mask &= df["pb"].notna() & (df["pb"] > pb_min) & (df["pb"] <= pb_max)
    mask &= df["total_mv"].notna() & (df["total_mv"] >= mv_min)  # 100M CNY
    mask &= df["turnover_rate"].notna() & (df["turnover_rate"] >= turnover_min)

    result = df[mask].copy()

    # Composite value score: lower PE and PB are better
    if not result.empty:
        pe_rank = result["pe_ttm"].rank(pct=True)       # Lower = better
        pb_rank = result["pb"].rank(pct=True)            # Lower = better
        mv_rank = result["total_mv"].rank(pct=True)      # Higher = more stable
        result["fast_score"] = (1 - pe_rank) * 40 + (1 - pb_rank) * 40 + mv_rank * 20

    return result.sort_values("fast_score", ascending=False) if not result.empty else result


def filter_momentum(df: pd.DataFrame,
                    pct_chg_min: float = 1.0,
                    pct_chg_max: float = 7.0,
                    volume_ratio_min: float = 1.2,
                    mv_min: float = 30,
                    turnover_min: float = 1.0,
                    exclude_st: bool = True) -> pd.DataFrame:
    """
    Growth momentum filter.
    Criteria: Positive price change but NOT near daily limit (avoid chasing),
    above-average volume, decent liquidity.
    pct_chg_max defaults to 7% to avoid stocks that are already near limit up.
    """
    mask = _base_filter(df, exclude_st=exclude_st)
    mask &= df["pct_chg"] >= pct_chg_min
    mask &= df["pct_chg"] <= pct_chg_max   # Cap: avoid chasing limit-up stocks
    mask &= df["volume_ratio"].notna() & (df["volume_ratio"] >= volume_ratio_min)
    mask &= df["total_mv"].notna() & (df["total_mv"] >= mv_min)
    mask &= df["turnover_rate"].notna() & (df["turnover_rate"] >= turnover_min)

    result = df[mask].copy()

    if not result.empty:
        # Momentum score: higher change%, higher volume ratio
        chg_rank = result["pct_chg"].rank(pct=True)
        vol_rank = result["volume_ratio"].rank(pct=True)
        turn_rank = result["turnover_rate"].rank(pct=True)
        result["fast_score"] = chg_rank * 40 + vol_rank * 35 + turn_rank * 25

    return result.sort_values("fast_score", ascending=False) if not result.empty else result


def filter_oversold(df: pd.DataFrame,
                    pct_chg_max: float = -2.0,
                    mv_min: float = 30,
                    turnover_min: float = 0.3,
                    exclude_st: bool = True) -> pd.DataFrame:
    """
    Oversold bounce filter.
    Criteria: Significant recent drop, still has decent market cap and liquidity.
    Not at limit down (untradeable).
    """
    mask = _base_filter(df, exclude_st=exclude_st)
    mask &= df["pct_chg"] <= pct_chg_max
    mask &= df["total_mv"].notna() & (df["total_mv"] >= mv_min)
    mask &= df["turnover_rate"].notna() & (df["turnover_rate"] >= turnover_min)
    # Prefer stocks that are not penny stocks
    mask &= df["current"] >= 3.0

    result = df[mask].copy()

    if not result.empty:
        # Oversold score: bigger drop + decent market cap + some volume = better bounce candidate
        drop_rank = result["pct_chg"].rank(pct=True)  # Lower (more negative) = rank lower
        mv_rank = result["total_mv"].rank(pct=True)
        result["fast_score"] = (1 - drop_rank) * 50 + mv_rank * 30 + \
                               result["turnover_rate"].rank(pct=True) * 20

    return result.sort_values("fast_score", ascending=False) if not result.empty else result


def filter_potential(df: pd.DataFrame,
                     chg_60d_max: float = -10.0,
                     chg_60d_min: float = -50.0,
                     pe_max: float = 40,
                     pe_min: float = 0,
                     pb_max: float = 5,
                     mv_min: float = 50,
                     today_chg_min: float = -3.0,
                     turnover_min: float = 0.3,
                     exclude_st: bool = True) -> pd.DataFrame:
    """
    Potential / bottom-fishing filter.

    Core idea: find stocks that have ALREADY FALLEN significantly over the
    past 60 days (low position), but have solid fundamentals (not junk),
    and are NOT crashing further today (stabilization / early reversal sign).

    This is the opposite of momentum — we look for:
    1) 60-day change significantly negative (stock is at a low position)
    2) Fundamentals still intact (positive PE, reasonable PB, decent market cap)
    3) Today is NOT another big drop (showing stabilization or slight bounce)
    4) Still has some trading activity (not dead/suspended)

    The deep scan (Stage 2) will then evaluate which of these have the
    strongest reversal signals (mean reversion Z-score, support levels,
    volume accumulation patterns, Hurst < 0.5, etc.)

    Parameters:
        chg_60d_max     Max 60-day change %. Default -10% means stock must
                        have dropped at least 10% in past 60 days.
        chg_60d_min     Min 60-day change %. Default -50% to exclude stocks
                        that have crashed too hard (may be fundamentally broken).
        pe_max          Max PE TTM (filter out overvalued or loss-making extremes)
        pe_min          Min PE (>0 ensures the company is profitable)
        pb_max          Max PB
        mv_min          Min market cap in 100M CNY
        today_chg_min   Today's min change %. Default -3% to avoid stocks
                        that are still actively crashing.
        turnover_min    Min turnover rate %
    """
    mask = _base_filter(df, exclude_st=exclude_st)

    # Core: stock must have fallen significantly over past 60 days
    mask &= df["chg_60d"].notna()
    mask &= df["chg_60d"] <= chg_60d_max   # e.g., <= -10%
    mask &= df["chg_60d"] >= chg_60d_min   # e.g., >= -50% (not total collapse)

    # Fundamentals still solid: profitable company with reasonable valuation
    mask &= df["pe_ttm"].notna() & (df["pe_ttm"] > pe_min) & (df["pe_ttm"] <= pe_max)
    mask &= df["pb"].notna() & (df["pb"] > 0) & (df["pb"] <= pb_max)
    mask &= df["total_mv"].notna() & (df["total_mv"] >= mv_min)

    # Today: NOT crashing further (stabilization signal)
    mask &= df["pct_chg"] >= today_chg_min

    # Still actively traded
    mask &= df["turnover_rate"].notna() & (df["turnover_rate"] >= turnover_min)

    result = df[mask].copy()

    if not result.empty:
        # Potential score — higher is better:
        #   - Bigger 60d drop = deeper value (if fundamentals intact)
        #   - Lower PE = cheaper
        #   - Higher market cap = more stable (less likely to be junk)
        #   - Today positive change = early reversal sign (bonus)
        drop_depth = result["chg_60d"].rank(pct=True)       # More negative = lower rank
        pe_rank = result["pe_ttm"].rank(pct=True)           # Lower PE = better
        mv_rank = result["total_mv"].rank(pct=True)         # Bigger = more stable
        today_rank = result["pct_chg"].rank(pct=True)       # Today's bounce = bonus

        result["fast_score"] = (
            (1 - drop_depth) * 30 +   # Deeper drop = higher score
            (1 - pe_rank) * 25 +       # Lower PE = higher score
            mv_rank * 20 +             # Bigger market cap = higher score
            today_rank * 15 +          # Today bouncing = bonus
            result["turnover_rate"].rank(pct=True) * 10  # Active trading = bonus
        )

    return result.sort_values("fast_score", ascending=False) if not result.empty else result


def filter_custom(df: pd.DataFrame,
                  pe_min: float = None, pe_max: float = None,
                  pb_min: float = None, pb_max: float = None,
                  mv_min: float = None, mv_max: float = None,
                  pct_chg_min: float = None, pct_chg_max: float = None,
                  turnover_min: float = None, turnover_max: float = None,
                  volume_ratio_min: float = None,
                  price_min: float = None, price_max: float = None,
                  exclude_st: bool = True) -> pd.DataFrame:
    """
    Custom filter with user-specified conditions.
    Any parameter left as None is not applied.
    Always applies base filter (exclude ST, limit-up/down, suspended).
    """
    mask = _base_filter(df, exclude_st=exclude_st)

    if pe_min is not None:
        mask &= df["pe_ttm"].notna() & (df["pe_ttm"] >= pe_min)
    if pe_max is not None:
        mask &= df["pe_ttm"].notna() & (df["pe_ttm"] <= pe_max)
    if pb_min is not None:
        mask &= df["pb"].notna() & (df["pb"] >= pb_min)
    if pb_max is not None:
        mask &= df["pb"].notna() & (df["pb"] <= pb_max)
    if mv_min is not None:
        mask &= df["total_mv"].notna() & (df["total_mv"] >= mv_min)
    if mv_max is not None:
        mask &= df["total_mv"].notna() & (df["total_mv"] <= mv_max)
    if pct_chg_min is not None:
        mask &= df["pct_chg"].notna() & (df["pct_chg"] >= pct_chg_min)
    if pct_chg_max is not None:
        mask &= df["pct_chg"].notna() & (df["pct_chg"] <= pct_chg_max)
    if turnover_min is not None:
        mask &= df["turnover_rate"].notna() & (df["turnover_rate"] >= turnover_min)
    if turnover_max is not None:
        mask &= df["turnover_rate"].notna() & (df["turnover_rate"] <= turnover_max)
    if volume_ratio_min is not None:
        mask &= df["volume_ratio"].notna() & (df["volume_ratio"] >= volume_ratio_min)
    if price_min is not None:
        mask &= df["current"].notna() & (df["current"] >= price_min)
    if price_max is not None:
        mask &= df["current"].notna() & (df["current"] <= price_max)

    result = df[mask].copy()
    if not result.empty:
        # Generic score based on market cap and turnover
        result["fast_score"] = result["total_mv"].rank(pct=True) * 50 + \
                               result["turnover_rate"].rank(pct=True) * 50
    return result.sort_values("fast_score", ascending=False) if not result.empty else result


# =====================================================
# 3. Stage 2: Deep Quantitative Scan
# =====================================================

def _load_kline(code: str, analysis_days: int = 120) -> Optional[pd.DataFrame]:
    """
    Load K-line data for a single stock.
    Priority: local DB (instant) → online API (fallback).
    """
    start = (datetime.today() - timedelta(days=analysis_days)).strftime("%Y-%m-%d")

    # Try local database first
    try:
        from slow_fetcher import load_stock_history
        df = load_stock_history(code)
        if not df.empty and len(df) >= 30:
            if "date" in df.columns:
                df = df[df["date"] >= start].copy()
            if len(df) >= 30:
                df.attrs["code"] = code
                return df
    except (ImportError, FileNotFoundError):
        pass

    # Fallback to online API
    try:
        return get_kline(code, period="daily", start=start, adjust="qfq")
    except Exception:
        return None


def deep_scan(candidates: pd.DataFrame,
              top_n: int = 20,
              analysis_days: int = 120) -> List[Dict]:
    """
    Run full quant_engine comprehensive_diagnosis on each candidate.

    Data source priority:
      1) Local SQLite stock_history (from slow_fetcher, zero network)
      2) Online Eastmoney kline API (fallback)

    Parameters:
        candidates     DataFrame from Stage 1 (must have 'code' column)
        top_n          Max number of stocks to scan (to limit time)
        analysis_days  Days of history to analyze per stock

    Returns:
        List of dicts sorted by total_score descending, each containing:
        code, name, current, pct_chg, total_score, signal, confidence,
        trend, momentum, risk_level, prob_up, fast_score, and key metrics.
    """
    from slow_fetcher import assess_market_regime, assess_event_risk, _build_trade_decision

    results = []
    scan_list = candidates
    market_regime = assess_market_regime()

    for i, (_, row) in enumerate(scan_list.iterrows()):
        code = row["code"]
        try:
            df = _load_kline(code, analysis_days)
            if df is None or len(df) < 30:
                continue

            diag = qe.comprehensive_diagnosis(
                df,
                run_monte_carlo=True,
                mc_days=20,
            )

            mc = diag.get("monte_carlo") or {}
            trend_detail = diag.get("trend_detail", {})
            vol_detail = diag.get("volatility_detail", {})
            decision = _build_trade_decision(
                df,
                diag,
                float(df.iloc[-1]["close"]),
                market_regime,
            )
            entry_quality_score = (
                diag["total_score"] * 0.35
                + row.get("stage1_score", row.get("fast_score", 0)) * 0.18
                + row.get("history_score", 0) * 0.12
                + (mc.get("prob_up", 0) - 50) * 0.6
                + (mc.get("expected_return", 0) or 0) * 1.8
                + (decision.get("risk_reward", 0) or 0) * 8
                + (decision.get("market_regime_score", 0) or 0) * 0.15
            )
            if decision.get("action") == "buy":
                entry_quality_score += 12
            elif decision.get("action") == "sell":
                entry_quality_score -= 20

            results.append({
                "code": code,
                "name": row.get("name", ""),
                "current": float(df.iloc[-1]["close"]),
                "pct_chg": row.get("pct_chg", 0),
                "pe_ttm": row.get("pe_ttm"),
                "pb": row.get("pb"),
                "total_mv": row.get("total_mv"),
                "total_score": diag["total_score"],
                "signal": diag["signal"],
                "confidence": diag["confidence"],
                "trend_dir": trend_detail.get("direction", "N/A"),
                "trend_strength": trend_detail.get("strength", "N/A"),
                "risk_level": vol_detail.get("risk_level", "N/A"),
                "prob_up": mc.get("prob_up", None),
                "prob_down": mc.get("prob_down", None),
                "expected_return": mc.get("expected_return", None),
                "fast_score": row.get("fast_score", 0),
                "stage1_score": row.get("stage1_score", row.get("fast_score", 0)),
                "history_score": row.get("history_score"),
                "history_label": row.get("history_label"),
                "history_days": row.get("history_days"),
                "decision": decision,
                "decision_action": decision.get("action"),
                "decision_text": decision.get("conclusion"),
                "entry_low": decision.get("entry_low"),
                "entry_high": decision.get("entry_high"),
                "stop_loss": decision.get("stop_loss"),
                "take_profit": decision.get("take_profit"),
                "risk_reward": decision.get("risk_reward"),
                "market_regime_score": decision.get("market_regime_score"),
                "entry_quality_score": round(entry_quality_score, 1),
                "_diag": diag,
                "_df": df,
            })
        except Exception:
            continue

    results.sort(key=_decision_rank, reverse=True)

    # Event-risk recheck only for top candidates to keep screener responsive.
    review_limit = min(max(top_n * 2, 6), len(results), 12)
    for item in results[:review_limit]:
        try:
            event_risk = assess_event_risk(item["code"])
            decision = _build_trade_decision(
                item["_df"],
                item["_diag"],
                item["current"],
                market_regime,
                event_risk,
            )
            item["event_risk"] = event_risk
            item["decision"] = decision
            item["decision_action"] = decision.get("action")
            item["decision_text"] = decision.get("conclusion")
            item["entry_low"] = decision.get("entry_low")
            item["entry_high"] = decision.get("entry_high")
            item["stop_loss"] = decision.get("stop_loss")
            item["take_profit"] = decision.get("take_profit")
            item["risk_reward"] = decision.get("risk_reward")
            item["event_score"] = decision.get("event_score")
            item["entry_quality_score"] = round(
                item.get("entry_quality_score", 0) + (decision.get("event_score", 0) or 0) * 0.6,
                1,
            )
        except Exception:
            continue

    for item in results:
        item.pop("_diag", None)
        item.pop("_df", None)

    results.sort(key=_decision_rank, reverse=True)
    return results[:top_n]


# =====================================================
# 4. Main Screening Function
# =====================================================

STRATEGY_MAP = {
    "value": filter_value,
    "momentum": filter_momentum,
    "oversold": filter_oversold,
    "potential": filter_potential,
    "custom": filter_custom,
}


def screen_stocks(
    strategy: str = "value",
    top_n: int = 20,
    deep_scan_enabled: bool = True,
    stage1_limit: int = 80,
    analysis_days: int = 120,
    history_lookback_days: int = 240,
    history_min_days: int = 90,
    **filter_kwargs,
) -> Dict:
    """
    Main entry point for stock screening.

    Parameters:
        strategy          "value" / "momentum" / "oversold" / "custom"
        top_n             Number of top results to return
        deep_scan_enabled Whether to run Stage 2 deep analysis
        stage1_limit      Max candidates to pass from Stage 1 to Stage 2
        analysis_days     Days of history for deep scan
        history_lookback_days Days of history used for Stage 0 profile
        history_min_days  Minimum history days to treat profile as reliable
        **filter_kwargs   Additional parameters passed to the filter function

    Returns:
        {
            "strategy": str,
            "total_market": int,
            "stage1_count": int,
            "stage2_count": int,
            "results": [ list of scored stock dicts ],
            "summary": str
        }
    """
    try:
        from slow_fetcher import assess_market_regime
        market_regime = assess_market_regime()
    except Exception:
        market_regime = None

    # -- Fetch full market --
    all_stocks = fetch_all_stocks()
    if all_stocks.empty:
        return {"error": "Failed to fetch market data from Eastmoney"}

    total_market = len(all_stocks)

    # -- Stage 0: Historical cross-sectional scoring --
    all_stocks = attach_historical_profiles(
        all_stocks,
        lookback_days=history_lookback_days,
        min_history_days=history_min_days,
    )
    history_profile_count = int(all_stocks["history_score"].notna().sum()) \
        if "history_score" in all_stocks.columns else 0

    # -- Stage 1: Fast filter --
    filter_func = STRATEGY_MAP.get(strategy)
    if filter_func is None:
        return {"error": f"Unknown strategy: {strategy}. Available: {list(STRATEGY_MAP.keys())}"}

    candidates = filter_func(all_stocks, **filter_kwargs)
    if not candidates.empty:
        history_component = candidates.get("history_score", pd.Series(0, index=candidates.index)).fillna(0)
        fast_component = candidates.get("fast_score", pd.Series(0, index=candidates.index)).fillna(0)
        readiness_penalty = np.where(
            candidates.get("history_ready", pd.Series(False, index=candidates.index)).fillna(False),
            0.0,
            -8.0,
        )
        candidates["stage1_score"] = (
            fast_component * 0.55
            + history_component * 0.45
            + readiness_penalty
        ).round(1)
        candidates = candidates.sort_values(
            by=["stage1_score", "history_score", "fast_score"],
            ascending=False,
        )
    stage1_count = len(candidates)

    if stage1_count == 0:
        return {
            "strategy": strategy,
            "total_market": total_market,
            "history_profile_count": history_profile_count,
            "stage1_count": 0,
            "stage2_count": 0,
            "market_regime": market_regime,
            "results": [],
            "summary": f"No stocks passed the {strategy} filter out of {total_market} total stocks. "
                       f"Try relaxing filter conditions.",
        }

    # Limit candidates for deep scan
    candidates = candidates.head(stage1_limit)

    # -- Stage 2: Deep scan (optional) --
    if deep_scan_enabled:
        results = deep_scan(candidates, top_n=min(top_n, len(candidates)),
                            analysis_days=analysis_days)
        stage2_count = len(results)
    else:
        # Return Stage 1 results only
        results = []
        for _, row in candidates.head(top_n).iterrows():
            results.append({
                "code": row["code"],
                "name": row["name"],
                "current": round(row["current"], 2),
                "pct_chg": round(row["pct_chg"], 2),
                "pe_ttm": round(row["pe_ttm"], 2) if pd.notna(row.get("pe_ttm")) else None,
                "pb": round(row["pb"], 2) if pd.notna(row.get("pb")) else None,
                "total_mv": round(row["total_mv"], 1) if pd.notna(row.get("total_mv")) else None,
                "fast_score": round(row.get("fast_score", 0), 1),
                "history_score": round(row.get("history_score", 0), 1) if pd.notna(row.get("history_score")) else None,
                "history_label": row.get("history_label"),
                "history_days": int(row.get("history_days", 0) or 0),
                "stage1_score": round(row.get("stage1_score", row.get("fast_score", 0)), 1),
            })
        stage2_count = 0

    return {
        "strategy": strategy,
        "total_market": total_market,
        "history_profile_count": history_profile_count,
        "stage1_count": stage1_count,
        "stage2_count": stage2_count,
        "market_regime": market_regime,
        "results": results,
        "summary": _format_summary(strategy, total_market, stage1_count,
                                    stage2_count, results, deep_scan_enabled,
                                    history_profile_count=history_profile_count),
    }


# =====================================================
# 5. Formatted Output
# =====================================================

def _format_summary(strategy, total_market, stage1_count,
                    stage2_count, results, deep_enabled,
                    history_profile_count: int = 0) -> str:
    """Format screening results into human-readable text."""
    market_regime = None
    if deep_enabled:
        try:
            from slow_fetcher import assess_market_regime
            market_regime = assess_market_regime()
        except Exception:
            market_regime = None

    lines = [
        f"Stock Screener Report",
        f"Strategy: {strategy.upper()}",
        f"Market: {total_market} total A-share stocks scanned",
        f"Stage 0 (historical profile): {history_profile_count} stocks scored from stock_history",
        f"Stage 1 (fast filter): {stage1_count} candidates passed",
    ]

    if deep_enabled:
        lines.append(f"Stage 2 (deep quant scan): {stage2_count} stocks analyzed")
        if market_regime:
            lines.append(
                "Market Regime: "
                f"{market_regime['score']:+.1f} / {market_regime['label']} / {market_regime['action_bias']}"
            )

    lines.append("=" * 65)

    if not results:
        lines.append("No qualifying stocks found.")
        return "\n".join(lines)

    if deep_enabled and results and "total_score" in results[0]:
        buy_count = sum(1 for r in results if r.get("decision_action") == "buy")
        hold_count = sum(1 for r in results if r.get("decision_action") == "hold")
        sell_count = sum(1 for r in results if r.get("decision_action") == "sell")
        lines.append(
            f"Decision Mix: buy {buy_count} / hold {hold_count} / sell {sell_count}"
        )

        # Deep scan results
        lines.append(
            f"  {'#':<3} {'Code':<8} {'Name':<10} {'Price':>7} {'Chg%':>7} "
            f"{'Hist':>6} {'S1':>6} {'Action':<8} {'EQS':>6} {'Evt':>6} {'P(up)':>6}"
        )
        lines.append("-" * 90)

        for i, r in enumerate(results, 1):
            prob_up = f"{r['prob_up']:.0f}%" if r.get("prob_up") is not None else "N/A"
            evt = f"{r.get('event_score', 0):+.0f}" if r.get("event_score") is not None else "N/A"
            hist = f"{r.get('history_score', 0):+.0f}" if r.get("history_score") is not None else "N/A"
            lines.append(
                f"  {i:<3} {r['code']:<8} {r['name']:<10} "
                f"{r['current']:>7.2f} {r['pct_chg']:>+6.2f}% "
                f"{hist:>6} {r.get('stage1_score', r.get('fast_score', 0)):>+6.1f} "
                f"{r.get('decision_action','hold'):<8} {r.get('entry_quality_score', 0):>+6.1f} "
                f"{evt:>6} {prob_up:>6}"
            )

        lines.append("-" * 90)

        # Top pick summary
        if results:
            top = results[0]
            lines.extend([
                "",
                f"Top Pick: {top['name']} ({top['code']})",
                f"  Decision: {top.get('decision_text', top.get('signal', 'N/A'))}",
                f"  Price: {top['current']:.2f}   Hist Score: {top.get('history_score', 0):+.1f} ({top.get('history_label', 'unknown')})",
                f"  Stage1 Score: {top.get('stage1_score', top.get('fast_score', 0)):+.1f}   EQS: {top.get('entry_quality_score', 0):+.1f}   Quant Score: {top['total_score']:+.1f}",
                f"  Trend: {top.get('trend_dir', 'N/A')} ({top.get('trend_strength', 'N/A')})",
                f"  Risk Level: {top.get('risk_level', 'N/A')}",
                f"  Event Score: {top.get('event_score', 0):+.1f}",
                f"  Entry Zone: {top.get('entry_low', 'N/A')} ~ {top.get('entry_high', 'N/A')}",
                f"  Stop / TP: {top.get('stop_loss', 'N/A')} / {top.get('take_profit', 'N/A')}",
            ])
            if top.get("prob_up") is not None:
                lines.append(
                    f"  Monte Carlo: {top['prob_up']:.0f}% up / {top['prob_down']:.0f}% down   "
                    f"Expected return: {top['expected_return']:+.1f}%"
                )
            top_decision = top.get("decision", {}) or {}
            if top_decision.get("reasons_for"):
                lines.append("  Why selected:")
                for item in top_decision["reasons_for"][:3]:
                    lines.append(f"    + {item}")
            top_event = top.get("event_risk", {}) or {}
            for item in top_event.get("negative_flags", [])[:2]:
                lines.append(f"    - {item}")
    else:
        # Fast filter only results
        lines.append(
            f"  {'#':<3} {'Code':<8} {'Name':<10} {'Price':>7} {'Chg%':>7} "
            f"{'Hist':>6} {'S1':>6} {'PE':>7} {'PB':>6} {'MktCap':>8}"
        )
        lines.append("-" * 86)

        for i, r in enumerate(results, 1):
            pe_str = f"{r['pe_ttm']:.1f}" if r.get("pe_ttm") is not None else "N/A"
            pb_str = f"{r['pb']:.1f}" if r.get("pb") is not None else "N/A"
            mv_str = f"{r['total_mv']:.0f}" if r.get("total_mv") is not None else "N/A"
            hist = f"{r.get('history_score', 0):+.0f}" if r.get("history_score") is not None else "N/A"
            lines.append(
                f"  {i:<3} {r['code']:<8} {r['name']:<10} "
                f"{r['current']:>7.2f} {r['pct_chg']:>+6.2f}% "
                f"{hist:>6} {r.get('stage1_score', r.get('fast_score', 0)):>+6.1f} "
                f"{pe_str:>7} {pb_str:>6} {mv_str:>8}"
            )

    lines.extend([
        "",
        "DISCLAIMER: Screening results are based on mathematical models and market data.",
        "They do not constitute investment advice. Please conduct your own due diligence.",
    ])

    return "\n".join(lines)
