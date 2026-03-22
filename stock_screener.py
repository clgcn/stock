"""
Stock Screener — Two-Stage A-Share Market Scanner
===================================================
Scans all ~5000 A-share stocks to find the most promising candidates.

Stage 1 (Fast Filter):
  Fetches the entire market from Eastmoney in a single API call,
  then applies basic metric filters (market cap, PE, PB, volume, etc.)
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
import time

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from stock_tool import _get, get_kline
import quant_engine as qe

logger = logging.getLogger(__name__)


# =====================================================
# 1. Fetch Full Market Data from Eastmoney
# =====================================================

_EM_FIELDS = (
    "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,"
    "f12,f14,f15,f16,f17,f18,f20,f21,f23,"
    "f24,f25,f34"
)

_EM_URL = "https://82.push2.eastmoney.com/api/qt/clist/get"


def _build_fs(market: str) -> str:
    """Return Eastmoney 'fs' filter string for the requested market scope."""
    if market == "sh":
        return "m:1+t:2"
    elif market == "sz":
        return "m:0+t:6"
    return "m:1+t:2,m:0+t:6"


def _parse_records(records: list) -> list:
    """Parse raw Eastmoney records into row dicts, keeping main-board only."""
    rows = []
    for r in records:
        if r.get("f2") is None or r.get("f2") == "-":
            continue
        try:
            row = {
                "code": str(r.get("f12", "")),
                "name": r.get("f14", ""),
                "current": _safe_float(r.get("f2")),
                "change": _safe_float(r.get("f4")),
                "pct_chg": _safe_float(r.get("f3")),
                "volume": _safe_float(r.get("f5")),
                "amount": _safe_float(r.get("f6")),
                "amplitude": _safe_float(r.get("f7")),
                "high": _safe_float(r.get("f15")),
                "low": _safe_float(r.get("f16")),
                "open": _safe_float(r.get("f17")),
                "prev_close": _safe_float(r.get("f18")),
                "volume_ratio": _safe_float(r.get("f10")),
                "turnover_rate": _safe_float(r.get("f8")),
                "pe_ttm": _safe_float(r.get("f9")),
                "pb": _safe_float(r.get("f23")),
                "total_mv": _safe_float(r.get("f20")),
                "float_mv": _safe_float(r.get("f21")),
                "chg_60d": _safe_float(r.get("f24")),
                "chg_ytd": _safe_float(r.get("f25")),
            }
            code = row["code"]
            sh_main = code[:3] in ("600", "601", "603", "605")
            sz_main = code[:3] in ("000", "001", "002", "003")
            if row["current"] > 0 and code and (sh_main or sz_main):
                rows.append(row)
        except (ValueError, TypeError):
            continue
    return rows


def _fetch_one_page(fs: str, page: int = 1, page_size: int = 6000,
                    timeout: int = 30) -> dict:
    """Fetch a single page from Eastmoney clist API."""
    params = {
        "pn": page,
        "pz": page_size,
        "po": 1,
        "np": 1,
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": 2,
        "invt": 2,
        "fid": "f3",
        "fs": fs,
        "fields": _EM_FIELDS,
    }
    resp = _get(_EM_URL, params=params, timeout=timeout)
    return resp.json()


def _to_dataframe(rows: list) -> pd.DataFrame:
    """Convert parsed rows to final DataFrame with unit conversions."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ["total_mv", "float_mv"]:
        if col in df.columns:
            df[col] = df[col] / 1e8
    if "amount" in df.columns:
        df["amount"] = df["amount"] / 1e4
    return df


def fetch_all_stocks(market: str = "all",
                     max_retries: int = 3) -> pd.DataFrame:
    """
    Fetch all A-share main-board stocks with multi-source fallback:

      0) Local SQLite DB (from slow_fetcher.py — always works, no API needed)
      1) Bulk Eastmoney request (pz=6000)
      2) Paginated Eastmoney fallback (1000/page)
      3) Azure proxy pool

    Parameters:
        market       "all" / "sh" / "sz"
        max_retries  Retry count for each request attempt

    Returns:
        DataFrame (see column list in _parse_records)
    """
    # ── Attempt 0: Local database (slow_fetcher) — instant, no network ──
    try:
        from slow_fetcher import load_stocks_from_db
        df = load_stocks_from_db()
        if len(df) >= 2000:
            logger.info("Loaded %d stocks from local DB", len(df))
            return df
        else:
            logger.info("Local DB has only %d stocks, trying online sources...",
                       len(df))
    except (ImportError, FileNotFoundError) as e:
        logger.info("Local DB not available (%s), trying online sources...", e)

    fs = _build_fs(market)

    # ── Attempt 1: single bulk request with retries ──
    for attempt in range(1, max_retries + 1):
        try:
            data = _fetch_one_page(fs, page=1, page_size=6000, timeout=30)
            if data.get("data") and data["data"].get("diff"):
                rows = _parse_records(data["data"]["diff"])
                if rows:
                    logger.info("Bulk fetch OK: %d stocks (attempt %d)",
                                len(rows), attempt)
                    return _to_dataframe(rows)
        except Exception as e:
            logger.warning("Bulk fetch attempt %d/%d failed: %s",
                           attempt, max_retries, e)
            if attempt < max_retries:
                time.sleep(1.5 * attempt)

    # ── Attempt 2: paginated fallback (smaller requests) ──
    logger.info("Falling back to paginated fetch (1000/page)")
    page_size = 1000
    all_rows = []
    for page in range(1, 8):          # 7 pages × 1000 = 7000, more than enough
        for attempt in range(1, max_retries + 1):
            try:
                data = _fetch_one_page(fs, page=page, page_size=page_size,
                                       timeout=20)
                diff = (data.get("data") or {}).get("diff")
                if not diff:
                    # No more pages
                    logger.info("Paginated fetch done at page %d, "
                                "total %d stocks", page, len(all_rows))
                    return _to_dataframe(all_rows)
                all_rows.extend(_parse_records(diff))
                time.sleep(0.3)
                break                   # page succeeded, move to next
            except Exception as e:
                logger.warning("Page %d attempt %d/%d failed: %s",
                               page, attempt, max_retries, e)
                if attempt < max_retries:
                    time.sleep(1.5 * attempt)
        else:
            # All retries exhausted for this page — return what we have
            logger.error("Page %d failed after %d retries, returning "
                         "partial data (%d stocks)", page, max_retries,
                         len(all_rows))
            break

    if all_rows:
        return _to_dataframe(all_rows)

    return pd.DataFrame()


def _safe_float(val) -> float:
    """Safely convert to float, return NaN for invalid values."""
    if val is None or val == "-" or val == "":
        return float("nan")
    try:
        return float(val)
    except (ValueError, TypeError):
        return float("nan")


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
    results = []
    scan_list = candidates.head(top_n)

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

            results.append({
                "code": code,
                "name": row.get("name", ""),
                "current": row.get("current", 0),
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
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["total_score"], reverse=True)
    return results


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
    # -- Fetch full market --
    all_stocks = fetch_all_stocks()
    if all_stocks.empty:
        return {"error": "Failed to fetch market data from Eastmoney"}

    total_market = len(all_stocks)

    # -- Stage 1: Fast filter --
    filter_func = STRATEGY_MAP.get(strategy)
    if filter_func is None:
        return {"error": f"Unknown strategy: {strategy}. Available: {list(STRATEGY_MAP.keys())}"}

    candidates = filter_func(all_stocks, **filter_kwargs)
    stage1_count = len(candidates)

    if stage1_count == 0:
        return {
            "strategy": strategy,
            "total_market": total_market,
            "stage1_count": 0,
            "stage2_count": 0,
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
            })
        stage2_count = 0

    return {
        "strategy": strategy,
        "total_market": total_market,
        "stage1_count": stage1_count,
        "stage2_count": stage2_count,
        "results": results,
        "summary": _format_summary(strategy, total_market, stage1_count,
                                    stage2_count, results, deep_scan_enabled),
    }


# =====================================================
# 5. Formatted Output
# =====================================================

def _format_summary(strategy, total_market, stage1_count,
                    stage2_count, results, deep_enabled) -> str:
    """Format screening results into human-readable text."""
    lines = [
        f"Stock Screener Report",
        f"Strategy: {strategy.upper()}",
        f"Market: {total_market} total A-share stocks scanned",
        f"Stage 1 (fast filter): {stage1_count} candidates passed",
    ]

    if deep_enabled:
        lines.append(f"Stage 2 (deep quant scan): {stage2_count} stocks analyzed")

    lines.append("=" * 65)

    if not results:
        lines.append("No qualifying stocks found.")
        return "\n".join(lines)

    if deep_enabled and results and "total_score" in results[0]:
        # Deep scan results
        lines.append(
            f"  {'#':<3} {'Code':<8} {'Name':<10} {'Price':>7} {'Chg%':>7} "
            f"{'Score':>6} {'Signal':<12} {'Trend':<12} {'P(up)':>6}"
        )
        lines.append("-" * 75)

        for i, r in enumerate(results, 1):
            prob_up = f"{r['prob_up']:.0f}%" if r.get("prob_up") is not None else "N/A"
            lines.append(
                f"  {i:<3} {r['code']:<8} {r['name']:<10} "
                f"{r['current']:>7.2f} {r['pct_chg']:>+6.2f}% "
                f"{r['total_score']:>+6.1f} {r['signal']:<12} "
                f"{r.get('trend_dir',''):<12} {prob_up:>6}"
            )

        lines.append("-" * 75)

        # Top pick summary
        if results:
            top = results[0]
            lines.extend([
                "",
                f"Top Pick: {top['name']} ({top['code']})",
                f"  Price: {top['current']:.2f}   Score: {top['total_score']:+.1f}   Signal: {top['signal']}",
                f"  Trend: {top.get('trend_dir', 'N/A')} ({top.get('trend_strength', 'N/A')})",
                f"  Risk Level: {top.get('risk_level', 'N/A')}",
            ])
            if top.get("prob_up") is not None:
                lines.append(
                    f"  Monte Carlo: {top['prob_up']:.0f}% up / {top['prob_down']:.0f}% down   "
                    f"Expected return: {top['expected_return']:+.1f}%"
                )
    else:
        # Fast filter only results
        lines.append(
            f"  {'#':<3} {'Code':<8} {'Name':<10} {'Price':>7} {'Chg%':>7} "
            f"{'PE':>7} {'PB':>6} {'MktCap':>8} {'Score':>6}"
        )
        lines.append("-" * 70)

        for i, r in enumerate(results, 1):
            pe_str = f"{r['pe_ttm']:.1f}" if r.get("pe_ttm") is not None else "N/A"
            pb_str = f"{r['pb']:.1f}" if r.get("pb") is not None else "N/A"
            mv_str = f"{r['total_mv']:.0f}" if r.get("total_mv") is not None else "N/A"
            lines.append(
                f"  {i:<3} {r['code']:<8} {r['name']:<10} "
                f"{r['current']:>7.2f} {r['pct_chg']:>+6.2f}% "
                f"{pe_str:>7} {pb_str:>6} {mv_str:>8} {r.get('fast_score', 0):>6.1f}"
            )

    lines.extend([
        "",
        "DISCLAIMER: Screening results are based on mathematical models and market data.",
        "They do not constitute investment advice. Please conduct your own due diligence.",
    ])

    return "\n".join(lines)
