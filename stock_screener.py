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
from datetime import timedelta

from _http_utils import cn_now
from data_fetcher import get_kline
import quant_engine as qe
import db

logger = logging.getLogger(__name__)

_HISTORY_PROFILE_CACHE: Dict[tuple, pd.DataFrame] = {}


# ══════════════════════════════════════════════════════════════════
# Data-health diagnostic
# ══════════════════════════════════════════════════════════════════
# The snapshot pipeline can miss a daily run (weekends, holidays, cron
# failures) or land only half-populated. When pct_chg or updated_at
# is mostly stale, every strategy will return 0 results and the user
# has no way to tell whether the filter is too strict or the data
# itself is the blocker. This diagnostic surfaces it.

def _build_data_health(df: pd.DataFrame) -> Dict:
    """Compute a data-health snapshot. Never raises.

    Returns dict with:
      total_rows, pct_chg_coverage (0-1), last_trade_date (str date),
      max_updated_at (str, fundamentals sync ts — info only),
      staleness_days (natural days from last_trade_date to today),
      is_stale (bool), warnings (list of str).

    ── Staleness semantics ──
    A-share 市场只在交易日产生行情。数据库里 *不存在* 周末/节假日行。
    所以 staleness 必须基于 `MAX(stock_history.date)` = 最近交易日,
    再比自然日。阈值:
      • 间隔 ≤ 3 天:正常(覆盖到周二跑上周五数据的长周末场景)
      • 间隔 > 3 天:真的漏了交易日,才告警
    之前拿 `fundamentals.updated_at` 的墙钟时间戳来做这事是错的 —
    那个时间戳和实际交易日解耦, 会在每个周一误报 stale.
    """
    health: Dict = {"warnings": []}
    try:
        health["total_rows"] = int(len(df))

        if "pct_chg" in df.columns:
            coverage = float(df["pct_chg"].notna().mean()) if len(df) else 0.0
        else:
            coverage = 0.0
        health["pct_chg_coverage"] = round(coverage, 3)

        # ── 1. last_trade_date: 取 stock_history 最大 date (唯一可信来源) ──
        last_trade = None
        if "last_trade_date" in df.columns:
            try:
                s = pd.to_datetime(df["last_trade_date"], errors="coerce")
                if s.notna().any():
                    last_trade = s.max()
            except Exception:
                last_trade = None
        health["last_trade_date"] = (
            last_trade.strftime("%Y-%m-%d") if last_trade is not None else None
        )

        # ── 2. max_updated_at: 旧字段保留,现在只做"基本面同步时间"展示 ──
        last_upd = None
        if "updated_at" in df.columns:
            try:
                s = pd.to_datetime(df["updated_at"], errors="coerce")
                if s.notna().any():
                    last_upd = s.max()
            except Exception:
                last_upd = None
        health["max_updated_at"] = str(last_upd) if last_upd is not None else None

        # ── 3. staleness: 以 last_trade_date 为准, 按自然日差 ──
        # 两边都归一成 naive date 再减, 避免 tz-aware/naive 混用抛异常.
        stale_days = None
        if last_trade is not None:
            try:
                today_d = cn_now().date()
                lt_d = pd.Timestamp(last_trade).date()
                stale_days = max(0, (today_d - lt_d).days)
            except Exception:
                stale_days = None
        health["staleness_days"] = int(stale_days) if stale_days is not None else None

        # ── 4. is_stale 判定:覆盖率太低 OR 间隔 > 3 天 ──
        health["is_stale"] = bool(
            coverage < 0.5
            or (stale_days is not None and stale_days > 3)
        )

        if coverage < 0.5:
            health["warnings"].append(
                f"pct_chg 仅 {coverage*100:.1f}% 股票有值(通常应>95%)— "
                "快照可能未在最新交易日更新. 已放宽 pct_chg.notna() 约束,"
                "但建议重跑 slow_fetcher 拉取最新行情."
            )
        if stale_days is not None and stale_days > 3:
            health["warnings"].append(
                f"最近交易日 {health['last_trade_date']} 距今 {stale_days} 天 — "
                f"疑似漏了一个或多个交易日的同步(长周末最多 3 天),"
                f"建议重跑 slow_fetcher."
            )
    except Exception as e:
        # Never let diagnostic itself break the pipeline
        health["warnings"].append(f"data-health probe failed: {type(e).__name__}: {e}")
    return health


# ══════════════════════════════════════════════════════════════════
# v2 pipeline helpers — cross-sectional standardization
# ══════════════════════════════════════════════════════════════════
# These are the foundation for the P0 fixes:
#   - winsorize extremes so rank/z-score are not distorted by outliers
#   - cross-sectional z-score so factors of different scales can be combined
#   - quantile-based cuts so thresholds are always relative to the current
#     distribution (robust across bull/bear markets)

def _winsorize(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    """Clip extreme values at lo/hi percentiles. Leaves NaN as NaN."""
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return s
    q_lo, q_hi = s.quantile([lo, hi])
    if pd.isna(q_lo) or pd.isna(q_hi):
        return s
    return s.clip(lower=q_lo, upper=q_hi)


def _cs_zscore(s: pd.Series, winsorize: bool = True) -> pd.Series:
    """Cross-sectional z-score. Returns 0 for all-NaN / zero-variance input.

    Rationale: combining factors with different natural scales (e.g. PE in
    [-1e5, +1e5] vs. prob_up in [0, 100]) requires first projecting them
    onto a common scale. z-score (mean=0, sd=1) is the standard choice
    because it preserves relative differences while being scale-invariant.
    """
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return pd.Series(0.0, index=s.index)
    s_clean = _winsorize(s) if winsorize else s
    mu = s_clean.mean(skipna=True)
    sd = s_clean.std(skipna=True)
    if sd is None or pd.isna(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    z = (s_clean - mu) / sd
    return z.fillna(0.0)


def _quantile_cut(results: List[Dict], quantile: float = 0.20,
                  min_keep: int = 0, score_key: str = "entry_quality_score") -> List[Dict]:
    """Keep the top `quantile` fraction of results by `score_key`.

    Args:
        results:   list of result dicts, already scored
        quantile:  fraction to keep (0.20 = top 20%)
        min_keep:  safety floor — always keep at least this many results
                   (use 0 to be strict, which may return 0 results if the
                   pool is tiny; use 3-5 for user-facing pipelines)
        score_key: which field to rank by

    Returns sorted (descending) top-k.
    """
    if not results:
        return results
    n = len(results)
    k = max(int(n * quantile), min_keep)
    k = min(k, n)
    return sorted(results,
                  key=lambda r: (r.get(score_key) or -1e18),
                  reverse=True)[:k]


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
        result.get("entry_quality_score") if result.get("entry_quality_score") is not None else -999,
        result.get("total_score") if result.get("total_score") is not None else -999,
        result.get("prob_up") if result.get("prob_up") is not None else -999,
    )


def _historical_rank(result: Dict) -> tuple:
    """Sort Stage 1 candidates by historical composite first, then strategy fit."""
    return (
        result.get("stage1_score") if result.get("stage1_score") is not None else -999,
        result.get("history_score") if result.get("history_score") is not None else -999,
        result.get("fast_score") if result.get("fast_score") is not None else -999,
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
        # 只拉需要的列和天数，大幅减少从 PostgreSQL 传输的数据量
        history = load_all_history(
            lookback_days=lookback_days,
            columns="code, date, close, pct_chg, volume",
        )
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
        # 确保数值列为 float，避免 SQLite NULL 导致 object dtype
        _NUMERIC_COLS = [
            "current", "pct_chg", "open", "high", "low", "volume",
            "amount", "pe_ttm", "pb", "total_mv", "turnover_rate",
            "volume_ratio", "chg_60d",
        ]
        for col in _NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        logger.info("Loaded %d stocks from local DB", len(df))
        return df
    except (ImportError, FileNotFoundError) as e:
        logger.warning("Local DB not available: %s", e)
        return pd.DataFrame()


# =====================================================
# 2. Stage 1: Fast Filter Strategies
# =====================================================

def _base_filter(df: pd.DataFrame, exclude_st: bool = True,
                 allow_limit_up: bool = False) -> pd.Series:
    """
    Common base filter applied to ALL strategies:
    - Exclude ST stocks (special treatment, high risk)
    - Exclude stocks at daily limit down (pct_chg <= -9.5%) because they
      are likely untradeable (no buyers)
    - Exclude suspended stocks (volume == 0 or current == 0)

    Limit-up policy (configurable):
    - Default (allow_limit_up=False):  exclude stocks near limit up
      (pct_chg >= 9.5%). Suitable for value / mean-reversion / fundamental
      strategies where chasing a rally is undesirable.
    - allow_limit_up=True:             keep limit-up candidates. Required
      for momentum / event-driven strategies, where a sealed limit-up is
      often the STRONGEST signal (earnings surprise, M&A news, sector
      catalyst). Pre-v2 pipelines hard-coded this filter on, which was
      self-defeating for momentum.
    """
    mask = pd.Series(True, index=df.index)

    # Exclude ST / *ST stocks (special treatment, high delisting risk)
    # Match names starting with "ST" or "*ST" (A-share naming convention)
    if exclude_st:
        mask &= ~df["name"].str.match(r"^\*?ST\b|^△", case=False, na=False)
        mask &= ~df["name"].str.contains("退市", na=False)

    # Limit-down check. IMPORTANT: NaN pct_chg must be treated as "unknown",
    # not as "excluded" — on a stale snapshot (e.g. pre-market, post-holiday,
    # or missed cron run), the snapshot job may populate most columns but
    # leave pct_chg NULL, which would wipe out the entire universe if we
    # required notna().
    pct = df["pct_chg"]
    mask &= (pct.isna() | (pct > -9.5))

    # Limit-up: only filter out for non-momentum strategies
    if not allow_limit_up:
        mask &= (pct.isna() | (pct < 9.5))

    # Exclude suspended stocks — check explicit suspended field first,
    # then fall back to volume/price proxy for snapshots where field is missing
    if "suspended" in df.columns:
        mask &= df["suspended"].fillna(0) == 0
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
                 exclude_st: bool = True,
                 allow_limit_up: bool = False) -> pd.DataFrame:
    """
    Value stock filter.
    Criteria: Reasonable PE, low PB, decent market cap, some liquidity.
    """
    mask = _base_filter(df, exclude_st=exclude_st, allow_limit_up=allow_limit_up)
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
                    pct_chg_max: float = 11.0,   # v2: 11.0 admits sealed limit-ups (9.8-10.0%)
                    volume_ratio_min: float = 1.2,
                    mv_min: float = 30,
                    turnover_min: float = 1.0,
                    exclude_st: bool = True,
                    allow_limit_up: bool = True) -> pd.DataFrame:
    """
    Growth momentum filter.
    Criteria: Positive price change, above-average volume, decent liquidity.

    v2 change: allow_limit_up defaults to True and pct_chg_max defaults to
    11.0 (was 7.0). A sealed limit-up is the single strongest momentum
    signal — cutting it off for fear of "chasing" is self-defeating for
    a momentum strategy. Callers that explicitly want the old behavior
    can pass allow_limit_up=False and pct_chg_max=7.0.
    """
    mask = _base_filter(df, exclude_st=exclude_st, allow_limit_up=allow_limit_up)
    mask &= df["pct_chg"].notna() & (df["pct_chg"] >= pct_chg_min)
    mask &= df["pct_chg"].notna() & (df["pct_chg"] <= pct_chg_max)
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
                    exclude_st: bool = True,
                    allow_limit_up: bool = False) -> pd.DataFrame:
    """
    Oversold bounce filter.
    Criteria: Significant recent drop, still has decent market cap and liquidity.
    Not at limit down (untradeable).
    """
    mask = _base_filter(df, exclude_st=exclude_st, allow_limit_up=allow_limit_up)
    mask &= df["pct_chg"].notna() & (df["pct_chg"] <= pct_chg_max)
    mask &= df["total_mv"].notna() & (df["total_mv"] >= mv_min)
    mask &= df["turnover_rate"].notna() & (df["turnover_rate"] >= turnover_min)
    # Prefer stocks that are not penny stocks
    mask &= df["current"].notna() & (df["current"] >= 3.0)

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
                     exclude_st: bool = True,
                     allow_limit_up: bool = False) -> pd.DataFrame:
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
    mask = _base_filter(df, exclude_st=exclude_st, allow_limit_up=allow_limit_up)

    # Core: stock must have fallen significantly over past 60 days
    mask &= df["chg_60d"].notna()
    mask &= df["chg_60d"].notna() & (df["chg_60d"] <= chg_60d_max)   # e.g., <= -10%
    mask &= df["chg_60d"].notna() & (df["chg_60d"] >= chg_60d_min)   # e.g., >= -50% (not total collapse)

    # Fundamentals still solid: profitable company with reasonable valuation
    mask &= df["pe_ttm"].notna() & (df["pe_ttm"] > pe_min) & (df["pe_ttm"] <= pe_max)
    mask &= df["pb"].notna() & (df["pb"] > 0) & (df["pb"] <= pb_max)
    mask &= df["total_mv"].notna() & (df["total_mv"] >= mv_min)

    # Today: NOT crashing further (stabilization signal)
    mask &= df["pct_chg"].notna() & (df["pct_chg"] >= today_chg_min)

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
                  exclude_st: bool = True,
                  allow_limit_up: bool = False) -> pd.DataFrame:
    """
    Custom filter with user-specified conditions.
    Any parameter left as None is not applied.
    Always applies base filter (exclude ST, limit-down, suspended).
    Limit-up gating follows allow_limit_up (default False).
    """
    mask = _base_filter(df, exclude_st=exclude_st, allow_limit_up=allow_limit_up)

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

# ── 批量 K 线预加载缓存（减少逐股查询） ──
_KLINE_BATCH_CACHE: Dict[str, pd.DataFrame] = {}


def _batch_preload_klines(codes: list, analysis_days: int = 120):
    """
    一次性从 PostgreSQL 批量加载多只股票的 K 线数据到内存缓存。
    将 N 次逐股查询合并为 1 次 SQL IN 查询，大幅减少网络往返。
    """
    # 模块级 dict, .clear() 原地修改无需 global 声明
    _KLINE_BATCH_CACHE.clear()

    if not codes:
        return

    start = (cn_now() - timedelta(days=analysis_days)).strftime("%Y-%m-%d")
    try:
        import db as _db
        conn = _db.get_conn()
        try:
            placeholders = ",".join(["%s"] * len(codes))
            sql = (
                f"SELECT * FROM stock_history "
                f"WHERE code IN ({placeholders}) AND close IS NOT NULL AND date >= %s "
                f"ORDER BY code, date"
            )
            df = pd.read_sql_query(sql, conn, params=(*codes, start))
            # 类型转换
            for col in ["open", "close", "high", "low", "volume", "amount",
                         "amplitude", "pct_chg", "change", "turnover"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            # 按 code 分组存入缓存
            for code, group in df.groupby("code"):
                _KLINE_BATCH_CACHE[code] = group.copy()
            logger.info("Batch preloaded klines for %d/%d stocks (%d rows)",
                        len(_KLINE_BATCH_CACHE), len(codes), len(df))
        finally:
            conn.close()
    except Exception as e:
        logger.warning("Batch kline preload failed: %s", e)


def _load_kline(code: str, analysis_days: int = 120,
                local_only: bool = False) -> Optional[pd.DataFrame]:
    """
    Load K-line data for a single stock.

    local_only=True  →  仅读本地 SQLite，不触发任何网络请求（批量筛选默认模式）
    local_only=False →  本地读取失败时回退到在线 API（单股分析模式）

    说明:
      批量筛选（deep_scan）必须 local_only=True，否则对 50-80 只候选股各发一次
      网络请求会导致触发 API 限速，筛选耗时剧增。
      本地 DB 数据由 slow_fetcher 定期同步写入，筛选前应确保 DB 已有数据。
    """
    start = (cn_now() - timedelta(days=analysis_days)).strftime("%Y-%m-%d")

    # ── 优先从批量缓存读取（deep_scan 预加载）──
    if code in _KLINE_BATCH_CACHE:
        df = _KLINE_BATCH_CACHE[code]
        if not df.empty and len(df) >= 30:
            df.attrs["code"] = code
            return df

    # ── 回退：逐股读本地 DB ──
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

    # ── 本地数据不足 ──
    if local_only:
        # 批量模式: 跳过，不发网络请求
        return None

    # 单股模式: 回退在线 API
    try:
        return get_kline(code, period="daily", start=start, adjust="hfq")
    except Exception:
        return None


def deep_scan(candidates: pd.DataFrame,
              top_n: int = 20,
              analysis_days: int = 120,
              local_only: bool = True) -> List[Dict]:
    """
    对候选股批量运行 7 维量化诊断（quant_engine.comprehensive_diagnosis）。

    local_only=True（默认，预筛阶段）:
      只读本地 SQLite stock_history，0 次网络请求。
      没有本地 kline 的股票自动跳过。
      适合对 50~80 只候选股做快速全量打分，选出质量最高的一批进入实时深度分析。

    local_only=False（实时分析阶段）:
      本地 kline 不足时，回退在线 API 拉取最新数据。
      适合对最终候选股（通常 ≤20 只）做高质量重算。
    """
    from slow_fetcher import assess_market_regime, assess_event_risk, _build_trade_decision

    results = []
    scan_list = candidates
    market_regime = assess_market_regime()
    skipped_no_data = 0

    # ── 批量预加载所有候选股 K 线（1次查询代替N次）──
    if local_only:
        all_codes = scan_list["code"].tolist()
        _batch_preload_klines(all_codes, analysis_days)

    for i, (_, row) in enumerate(scan_list.iterrows()):
        code = row["code"]
        try:
            # local_only=True: 批量扫描绝不触发网络请求
            # 无本地历史数据的股票直接跳过（不是筛选质量问题，是数据完整性问题）
            df = _load_kline(code, analysis_days, local_only=local_only)
            if df is None or len(df) < 30:
                skipped_no_data += 1
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
            _s1 = row.get("stage1_score") if pd.notna(row.get("stage1_score")) else (
                row.get("fast_score") if pd.notna(row.get("fast_score")) else 0
            )
            _hist = row.get("history_score") if pd.notna(row.get("history_score")) else 0
            _prob_up = mc.get("prob_up") if mc.get("prob_up") is not None else 50
            _exp_ret = mc.get("expected_return") or 0
            _rr = decision.get("risk_reward") or 0
            _regime = decision.get("market_regime_score") or 0
            entry_quality_score = (
                (diag.get("total_score") or 0) * 0.35
                + _s1 * 0.18
                + _hist * 0.12
                + (_prob_up - 50) * 0.6
                + _exp_ret * 1.8
                + _rr * 8
                + _regime * 0.15
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

    # ── Stage 2.5: Event-risk recheck（仅非 local_only 模式，或 local_only 且限 top_n 只）──
    # local_only=True 时：完全不调用公告 API，事件风险由 Stage-3（单股深度复核）完成
    # local_only=False 时：对最多 min(top_n, 8) 只候选股补充事件风险（单股场景）
    if not local_only:
        review_limit = min(top_n, 8, len(results))
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

    # 记录跳过数量供调用方参考
    logger.info(
        "deep_scan: scanned=%d  skipped_no_local_data=%d  results=%d  local_only=%s",
        len(candidates), skipped_no_data, len(results[:top_n]), local_only,
    )
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
    stage1_limit: int = 60,
    quality_threshold: float = 5.0,
    max_candidates: int = 25,
    analysis_days: int = 120,
    history_lookback_days: int = 240,
    history_min_days: int = 90,
    **filter_kwargs,
) -> Dict:
    """
    A股预筛选主入口（纯本地 DB，0次网络请求）。

    数据流:
      全量 ~5000 只股票
        ↓ [基础过滤] PE/PB/市值/换手/价格（stocks + fundamentals 快照）
        ↓ [历史趋势评分] stock_history 计算近期收益率/MA位置/量能趋势
        ↓ [量化深度扫描] 对 stage1_limit 只候选股跑 7 维技术分析（本地kline）
        ↓ [质量阈值过滤] entry_quality_score >= quality_threshold
        ↓ 输出最多 max_candidates 只，按 entry_quality_score 排序
      结果
        → 调用方对每只结果股票做全面实时分析（新闻/公告/主力资金/实时K线/风险）

    参数说明:
        strategy         筛选策略: value / momentum / oversold / potential / custom
        top_n            deep_scan 内部返回数（供质量阈值过滤前使用）
        deep_scan_enabled 是否运行量化深度扫描（建议始终 True）
        stage1_limit     基础过滤后最多送入深度扫描的数量（默认60，上限80）
        quality_threshold 深度扫描后的质量门槛（entry_quality_score，默认5.0）
                          • 含义: 综合技术+历史+蒙特卡洛风险收益的综合得分
                          • 经验参考: 牛市可调高到10~15; 熊市/震荡市调低到0~3
                          • 若返回结果为0，可降低此值; 若返回过多，可提高此值
        max_candidates   最终输出上限（默认25，保证调用方实时分析的可控性）
        analysis_days    深度扫描使用的历史天数
        history_lookback_days 历史横截面评分回溯天数
        history_min_days 历史数据最少天数要求
        **filter_kwargs  传给策略过滤函数的额外参数

    返回:
        {
            "strategy": str,
            "total_market": int,        # 全市场股票数
            "stage1_count": int,        # 基础过滤通过数
            "stage2_count": int,        # 深度扫描完成数
            "qualified_count": int,     # 通过质量阈值的数量
            "quality_threshold": float, # 使用的质量阈值
            "results": [                # 最终候选股列表（含预评分）
                {
                    "code", "name", "current", "pct_chg",
                    "pe_ttm", "pb", "total_mv",
                    "total_score",          # 7维量化得分 -100~+100
                    "entry_quality_score",  # 综合质量分（排序主键）
                    "history_score",        # 历史横截面得分
                    "signal",               # buy/hold/sell
                    "prob_up",              # 蒙特卡洛上涨概率
                    "trend_dir",
                    ...
                }
            ],
            "summary": str              # 文字摘要
        }
    """
    stage1_limit = min(int(stage1_limit), 80)

    try:
        from slow_fetcher import assess_market_regime
        market_regime = assess_market_regime()
    except Exception:
        market_regime = None

    # ── 从本地 DB 读取全量快照 ──
    all_stocks = fetch_all_stocks()
    if all_stocks.empty:
        return {"error": "本地数据库无股票数据，请先运行 slow_fetcher 同步数据"}

    total_market = len(all_stocks)

    # ── 数据健康体检（提前检测 pct_chg 缺失/快照过期等根因问题）──
    data_health = _build_data_health(all_stocks)
    for w in data_health.get("warnings", []):
        logger.warning("data-health: %s", w)

    # ── 历史横截面评分（stock_history → 趋势/动量/回撤评分）──
    all_stocks = attach_historical_profiles(
        all_stocks,
        lookback_days=history_lookback_days,
        min_history_days=history_min_days,
    )
    history_profile_count = int(all_stocks["history_score"].notna().sum()) \
        if "history_score" in all_stocks.columns else 0

    # ── 基础过滤（PE/PB/市值/换手/价格条件）──
    filter_func = STRATEGY_MAP.get(strategy)
    if filter_func is None:
        return {"error": f"未知策略 '{strategy}'，可用: {list(STRATEGY_MAP.keys())}"}

    try:
        candidates = filter_func(all_stocks, **filter_kwargs)
    except Exception as e:
        logger.error("filter_func(%s) failed: %s", strategy, e)
        return {"error": f"策略 '{strategy}' 过滤失败: {e}"}
    if not candidates.empty:
        history_component = candidates.get(
            "history_score", pd.Series(0, index=candidates.index)).fillna(0)
        fast_component = candidates.get(
            "fast_score", pd.Series(0, index=candidates.index)).fillna(0)
        readiness_penalty = np.where(
            candidates.get(
                "history_ready", pd.Series(False, index=candidates.index)).fillna(False),
            0.0, -8.0,
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
        summary_lines = [
            f"基础过滤后无股票通过（策略={strategy}，全市场{total_market}只）。",
        ]
        if data_health.get("is_stale"):
            summary_lines.append(
                f"⚠️ 数据可能过期 (pct_chg覆盖率={data_health.get('pct_chg_coverage', 0)*100:.1f}%, "
                f"最近交易日={data_health.get('last_trade_date')}, "
                f"距今 {data_health.get('staleness_days', '?')} 天)。"
            )
        summary_lines.append("可考虑放宽过滤条件或重跑 slow_fetcher。")
        return {
            "strategy": strategy,
            "total_market": total_market,
            "history_profile_count": history_profile_count,
            "stage1_count": 0,
            "stage2_count": 0,
            "qualified_count": 0,
            "quality_threshold": quality_threshold,
            "market_regime": market_regime,
            "data_health": data_health,
            "results": [],
            "summary": "".join(summary_lines),
        }

    # ── 量化深度扫描（本地 kline，0 次网络）──
    scan_candidates = candidates.head(stage1_limit)

    if deep_scan_enabled:
        # top_n 设为 stage1_limit，让 deep_scan 返回所有结果供后续质量过滤
        all_scanned = deep_scan(
            scan_candidates,
            top_n=len(scan_candidates),
            analysis_days=analysis_days,
            local_only=True,   # 预筛阶段始终 local_only
        )
        stage2_count = len(all_scanned)

        # ── 质量阈值过滤 ──
        # entry_quality_score 综合了: 7维技术得分×0.35 + 历史得分×0.12 +
        #   蒙特卡洛概率 + 风险收益比 + 市场趋势 + 快速得分×0.18
        # 阈值 5.0 ≈ 正向信号略占优势，通常筛出 10~25% 的扫描结果
        results = [
            r for r in all_scanned
            if (r.get("entry_quality_score") or 0) >= quality_threshold
        ]

        # 若阈值过严导致结果为0，自适应降级：取扫描结果前 min(5, stage2_count) 只
        if not results and all_scanned:
            results = sorted(
                all_scanned,
                key=lambda x: x.get("entry_quality_score") or 0,
                reverse=True,
            )[:min(5, len(all_scanned))]
            logger.warning(
                "quality_threshold=%.1f 过滤后结果为空，自适应降级取前%d只",
                quality_threshold, len(results),
            )

        # 按综合质量分排序，上限 max_candidates
        results = sorted(
            results,
            key=lambda x: x.get("entry_quality_score") or 0,
            reverse=True,
        )[:max_candidates]

    else:
        # 不做量化扫描，只返回基础过滤结果
        all_scanned = []
        stage2_count = 0
        results = []
        for _, row in scan_candidates.head(top_n).iterrows():
            results.append({
                "code": row["code"],
                "name": row["name"],
                "current": round(row["current"], 2),
                "pct_chg": round(row["pct_chg"], 2),
                "pe_ttm": round(row["pe_ttm"], 2) if pd.notna(row.get("pe_ttm")) else None,
                "pb": round(row["pb"], 2) if pd.notna(row.get("pb")) else None,
                "total_mv": round(row["total_mv"], 1) if pd.notna(row.get("total_mv")) else None,
                "fast_score": round(row.get("fast_score", 0), 1),
                "history_score": round(row.get("history_score", 0), 1)
                    if pd.notna(row.get("history_score")) else None,
                "history_label": row.get("history_label"),
                "history_days": int(row.get("history_days", 0) or 0),
                "stage1_score": round(row.get("stage1_score", row.get("fast_score", 0)), 1),
                "entry_quality_score": row.get("stage1_score", 0),
            })

    # ── 机构共识度加分 (本地 DB, 0 次网络) ──
    # 对最终候选股查询 fund_holdings + stock_top_holders 表,
    # 加分映射: institutional_score(0~100) × 0.15 → 最多 +15 分到 entry_quality_score
    try:
        from institutional import institutional_score as _inst_score
        _inst_conn = db.get_conn() if results else None
        for r in results:
            try:
                iscore = _inst_score(r["code"], conn=_inst_conn)
                r["institutional_score"] = iscore["score"]
                r["institutional_detail"] = iscore["detail"]
                r["fund_count"] = iscore["fund_count"]
                r["smart_money_types"] = iscore["smart_money_types"]
                # 加分到综合质量分
                bonus = round(iscore["score"] * 0.15, 2)
                r["entry_quality_score"] = round(
                    (r.get("entry_quality_score") or 0) + bonus, 1)
            except Exception:
                r["institutional_score"] = 0
                r["institutional_detail"] = "无数据"
                r["fund_count"] = 0
                r["smart_money_types"] = []
        if _inst_conn:
            _inst_conn.close()
        # 加分后重新排序
        results = sorted(
            results,
            key=lambda x: x.get("entry_quality_score") or 0,
            reverse=True,
        )
    except ImportError:
        logger.debug("institutional module not available, skipping scoring")

    return {
        "strategy": strategy,
        "total_market": total_market,
        "history_profile_count": history_profile_count,
        "stage1_count": stage1_count,
        "stage2_count": stage2_count,
        "qualified_count": len(results),
        "quality_threshold": quality_threshold,
        "market_regime": market_regime,
        "data_health": data_health,
        "results": results,
        "summary": _format_summary(
            strategy, total_market, stage1_count, stage2_count,
            results, deep_scan_enabled,
            history_profile_count=history_profile_count,
            quality_threshold=quality_threshold,
        ),
    }


# =====================================================
# 5. Formatted Output
# =====================================================

def _format_summary(strategy, total_market, stage1_count,
                    stage2_count, results, deep_enabled,
                    history_profile_count: int = 0,
                    quality_threshold: float = 5.0) -> str:
    """Format screening results into human-readable text."""
    market_regime = None
    if deep_enabled:
        try:
            from slow_fetcher import assess_market_regime
            market_regime = assess_market_regime()
        except Exception:
            market_regime = None

    lines = [
        "Stock Screener Report",
        f"Strategy: {strategy.upper()}",
        f"Market: {total_market} total A-share stocks scanned",
        f"Stage 0 (historical profile): {history_profile_count} stocks scored from stock_history",
        f"Stage 1 (fast filter): {stage1_count} candidates passed",
    ]

    if deep_enabled:
        lines.append(f"Stage 2 (deep quant scan): {stage2_count} stocks analyzed")
        lines.append(
            f"Quality Gate: entry_quality_score ≥ {quality_threshold:.1f}  →  "
            f"{len(results)} 只通过  (候选池输出 {len(results)} 只进入实时深度复核)"
        )
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
        has_source = any(r.get("source_strategy") for r in results)
        hdr = (f"  {'#':<3} {'Code':<8} {'Name':<10} {'Price':>7} {'Chg%':>7} "
               f"{'Hist':>6} {'S1':>6} {'Action':<8} {'EQS':>6} {'Evt':>6} {'P(up)':>6}")
        if has_source:
            hdr += f" {'来源':>8}"
        lines.append(hdr)
        lines.append("-" * (98 if has_source else 90))

        for i, r in enumerate(results, 1):
            prob_up = f"{r['prob_up']:.0f}%" if r.get("prob_up") is not None else "N/A"
            evt = f"{r.get('event_score', 0):+.0f}" if r.get("event_score") is not None else "N/A"
            hist = f"{r.get('history_score', 0):+.0f}" if r.get("history_score") is not None else "N/A"
            row = (f"  {i:<3} {r['code']:<8} {r['name']:<10} "
                   f"{r['current']:>7.2f} {r['pct_chg']:>+6.2f}% "
                   f"{hist:>6} {r.get('stage1_score', r.get('fast_score', 0)):>+6.1f} "
                   f"{r.get('decision_action','hold'):<8} {r.get('entry_quality_score', 0):>+6.1f} "
                   f"{evt:>6} {prob_up:>6}")
            if has_source:
                row += f" {r.get('source_strategy', ''):>8}"
            lines.append(row)

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


# =====================================================
# 6. v2 Scoring Pipeline (P0 fixes)
# =====================================================
# Design goals, in plain language:
#   - Every factor enters the composite on the SAME scale (z-score ≈ ±3),
#     so a 0.10 weight on one factor really is 2x the impact of a 0.05
#     weight on another. No more mixing {-100..+100} with {0..1}.
#   - Weights are EXPLICIT and sum to 1.0 — easy to tune, easy to audit.
#   - Quality gate is a cross-sectional QUANTILE (default top 20%),
#     so the result size and the composition track the current market
#     regime instead of depending on a magic absolute threshold.
#   - Multi-strategy merge uses HIT COUNT as the primary rank, rewarding
#     candidates that survive multiple orthogonal lenses.
#   - Refinement step is a single quantile cut on a z-score composite of
#     (prob_up, total_score, expected_return) — no more 65%→55%→giveup.

# Weight spec for v2 EQS. Each weight is explained inline. Weights must
# sum to 1.0 — unit test at the bottom of this file guards it.
EQS_V2_WEIGHTS = {
    # Quant diagnosis (7-dim technical + structural, already composite)
    "tech":   0.22,   # primary technical signal
    # Stage-1 composite (fast_score + historical)
    "s1":     0.14,
    # Medium/long-term trend
    "hist":   0.10,
    # Monte Carlo 20-day upside probability
    "prob":   0.12,
    # Monte Carlo expected return magnitude
    "exp":    0.12,
    # Risk/reward ratio (entry-to-TP vs entry-to-stop)
    "rr":     0.13,
    # Market regime beta
    "regime": 0.05,
    # Discrete decision bonus (buy/hold/sell, already on [-1, +1])
    "action": 0.12,
}
# Multiplier applied after z-weighted sum, so the output range is
# comparable to the v1 EQS (~ ±30). This is cosmetic — the ordering
# is unaffected.
EQS_V2_SCALE = 10.0


def _compute_eqs_v2(results: List[Dict]) -> List[Dict]:
    """Recompute entry_quality_score for a batch of results using
    cross-sectional z-score standardization.

    This is a PURE FUNCTION over the batch — the output score of stock X
    depends on the whole batch (because it's cross-sectional), so don't
    call this per-stock. Always call it once on the full deep_scan output.

    Mutates each dict in-place by overwriting `entry_quality_score` and
    adding `_eqs_factors` (each component's z-score), `entry_quality_raw_v1`
    (the old score, for A/B comparison).
    """
    if not results:
        return results

    df = pd.DataFrame(results)

    # Preserve the v1 score for side-by-side inspection
    df["_v1"] = pd.to_numeric(df.get("entry_quality_score"), errors="coerce")

    # Extract raw component values. Fill sentinels that map to "neutral"
    # so missing data doesn't penalize unfairly (50 is neutral prob_up,
    # 0 is neutral for everything else).
    df["_raw_tech"]   = pd.to_numeric(df.get("total_score"),          errors="coerce").fillna(0)
    df["_raw_s1"]     = pd.to_numeric(df.get("stage1_score"),         errors="coerce").fillna(0)
    df["_raw_hist"]   = pd.to_numeric(df.get("history_score"),        errors="coerce").fillna(0)
    df["_raw_prob"]   = pd.to_numeric(df.get("prob_up"),              errors="coerce").fillna(50) - 50
    df["_raw_exp"]    = pd.to_numeric(df.get("expected_return"),      errors="coerce").fillna(0)
    df["_raw_rr"]     = pd.to_numeric(df.get("risk_reward"),          errors="coerce").fillna(0)
    df["_raw_regime"] = pd.to_numeric(df.get("market_regime_score"),  errors="coerce").fillna(0)

    # Cross-sectional z-score for continuous factors
    for col in ["tech", "s1", "hist", "prob", "exp", "rr", "regime"]:
        df[f"_z_{col}"] = _cs_zscore(df[f"_raw_{col}"])

    # Action is already on a bounded discrete scale — map directly
    action_map = {"buy": 1.0, "hold": 0.0, "sell": -1.0}
    df["_z_action"] = df.get("decision_action", pd.Series("hold", index=df.index)) \
                        .fillna("hold").map(lambda v: action_map.get(v, 0.0))

    # Weighted sum (each _z_* is mean ~0, sd ~1 so weighted sum is also
    # ~N(0, sum_of_squares_of_weights) ≈ N(0, 0.4) for our weights)
    composite = sum(df[f"_z_{k}"] * w for k, w in EQS_V2_WEIGHTS.items())
    df["_eqs_v2"] = (composite * EQS_V2_SCALE).round(2)

    # Write back
    for i, rec in enumerate(results):
        rec["entry_quality_raw_v1"] = (
            float(df.iloc[i]["_v1"]) if pd.notna(df.iloc[i]["_v1"]) else None
        )
        rec["entry_quality_score"]  = float(df.iloc[i]["_eqs_v2"])
        rec["_eqs_factors"] = {
            k: float(df.iloc[i][f"_z_{k}"]) for k in EQS_V2_WEIGHTS
        }
    return results


def _refine_picks_v2(picks: List[Dict], max_review: int = 15,
                     min_review: int = 5) -> Dict:
    """Relative-rank refinement (replaces 65%/55% cascade).

    Strategy:
      1. z-score prob_up, total_score, expected_return across `picks`
      2. average the three z-scores → refine_score
      3. keep top-k where k = min(max_review, len(picks)), with floor min_review

    Returns {"picks_for_review": list, "refine_scores": dict}.
    """
    if not picks:
        return {"picks_for_review": [], "refine_scores": {}}
    if len(picks) <= min_review:
        return {"picks_for_review": list(picks), "refine_scores": {}}

    df = pd.DataFrame(picks)
    z_prob = _cs_zscore(pd.to_numeric(df.get("prob_up"), errors="coerce").fillna(50) - 50)
    z_tot  = _cs_zscore(pd.to_numeric(df.get("total_score"), errors="coerce").fillna(0))
    z_exp  = _cs_zscore(pd.to_numeric(df.get("expected_return"), errors="coerce").fillna(0))
    df["_refine_score"] = (z_prob + z_tot + z_exp) / 3.0

    df_sorted = df.sort_values("_refine_score", ascending=False).reset_index(drop=True)
    n_keep = min(max_review, len(df_sorted))
    n_keep = max(n_keep, min_review)
    n_keep = min(n_keep, len(df_sorted))

    kept_codes = df_sorted["code"].head(n_keep).tolist()
    code_to_score = dict(zip(df_sorted["code"], df_sorted["_refine_score"]))
    picks_for_review = [p for p in picks if p["code"] in set(kept_codes)]
    # Preserve the quantile ranking in the returned list
    picks_for_review.sort(
        key=lambda p: code_to_score.get(p["code"], -1e18), reverse=True,
    )
    return {
        "picks_for_review": picks_for_review,
        "refine_scores": {c: float(code_to_score.get(c, 0.0)) for c in kept_codes},
    }


def _merge_auto_results_v2(per_strategy: Dict[str, List[Dict]]) -> List[Dict]:
    """Multi-strategy merge for `auto` mode.

    Replaces the v1 "take max EQS" approach with:
      1. Within each strategy, z-score fast_score → comparable scale
      2. For each stock: record which strategies hit it, and its per-strategy z
      3. Final sort: (hit_count DESC, avg_z DESC, EQS DESC)

    This rewards robust picks that survive multiple orthogonal lenses
    (value + momentum + potential + ...) instead of just the single
    highest-scoring strategy, which is the correct behavior for an
    ensemble selection system.
    """
    if not per_strategy:
        return []

    # 1. Per-strategy within-cohort z-score of fast_score
    per_strat_z: Dict[str, Dict[str, float]] = {}
    for strat, results in per_strategy.items():
        if not results:
            per_strat_z[strat] = {}
            continue
        sdf = pd.DataFrame(results)
        if "fast_score" not in sdf.columns or sdf["fast_score"].dropna().empty:
            per_strat_z[strat] = {r["code"]: 0.0 for r in results}
            continue
        z = _cs_zscore(pd.to_numeric(sdf["fast_score"], errors="coerce"))
        per_strat_z[strat] = dict(zip(sdf["code"], z))

    # 2. Merge per-stock, keeping the record with highest EQS as the "base"
    merged: Dict[str, Dict] = {}
    for strat, results in per_strategy.items():
        for r in results:
            code = r["code"]
            fz = per_strat_z[strat].get(code, 0.0)
            if code not in merged:
                base = dict(r)
                base["strategy_hits"] = [strat]
                base["strategy_zs"]   = [float(fz)]
                merged[code] = base
            else:
                merged[code]["strategy_hits"].append(strat)
                merged[code]["strategy_zs"].append(float(fz))
                # Upgrade base record if this strategy's EQS is higher
                cur_eqs = merged[code].get("entry_quality_score") or -1e18
                new_eqs = r.get("entry_quality_score") or -1e18
                if new_eqs > cur_eqs:
                    preserved_hits = merged[code]["strategy_hits"]
                    preserved_zs   = merged[code]["strategy_zs"]
                    merged[code] = dict(r)
                    merged[code]["strategy_hits"] = preserved_hits
                    merged[code]["strategy_zs"]   = preserved_zs

    # 3. Compute summary fields
    for code, r in merged.items():
        r["hit_count"]       = len(r["strategy_hits"])
        r["avg_strategy_z"]  = float(np.mean(r["strategy_zs"])) if r["strategy_zs"] else 0.0
        r["source_strategy"] = "+".join(r["strategy_hits"])

    # 4. Sort: hit_count DESC, avg_z DESC, EQS DESC
    ordered = sorted(
        merged.values(),
        key=lambda r: (
            r.get("hit_count", 0),
            r.get("avg_strategy_z", 0.0),
            r.get("entry_quality_score") or -1e18,
        ),
        reverse=True,
    )
    return ordered


def screen_stocks_v2(
    strategy: str = "value",
    top_n: int = 20,
    stage1_limit: int = 60,
    quality_quantile: float = 0.20,
    min_keep: int = 3,
    max_candidates: int = 25,
    analysis_days: int = 120,
    history_lookback_days: int = 240,
    history_min_days: int = 90,
    **filter_kwargs,
) -> Dict:
    """v2 version of screen_stocks with cross-sectional scoring.

    Differences vs v1:
      - EQS is recomputed via _compute_eqs_v2 (z-score composite)
      - Quality gate is a QUANTILE (default top 20%), not an absolute
        threshold. `min_keep` guarantees a minimum batch size only if
        deep scan actually produced that many scored results.
      - Momentum strategy admits limit-up candidates by default.
      - No "if 0 results, give 5 anyway" crutch.

    Backward-compatibility notes:
      - Returned dict has the same top-level keys as v1 plus
        `quality_quantile` and `pipeline_version: "v2"`.
      - Each result dict still has `entry_quality_score`, but the
        magnitude/distribution is different — consumers that compare
        EQS values across v1 and v2 runs will break.
    """
    stage1_limit = min(int(stage1_limit), 80)

    try:
        from slow_fetcher import assess_market_regime
        market_regime = assess_market_regime()
    except Exception:
        market_regime = None

    all_stocks = fetch_all_stocks()
    if all_stocks.empty:
        return {"error": "本地数据库无股票数据，请先运行 slow_fetcher 同步数据",
                "pipeline_version": "v2"}
    total_market = len(all_stocks)

    data_health = _build_data_health(all_stocks)
    for w in data_health.get("warnings", []):
        logger.warning("data-health: %s", w)

    all_stocks = attach_historical_profiles(
        all_stocks,
        lookback_days=history_lookback_days,
        min_history_days=history_min_days,
    )
    history_profile_count = (int(all_stocks["history_score"].notna().sum())
                             if "history_score" in all_stocks.columns else 0)

    filter_func = STRATEGY_MAP.get(strategy)
    if filter_func is None:
        return {"error": f"未知策略 '{strategy}'", "pipeline_version": "v2"}

    # v2: momentum admits limit-up automatically. Callers can override
    # via filter_kwargs={"allow_limit_up": False}.
    if strategy == "momentum" and "allow_limit_up" not in filter_kwargs:
        filter_kwargs["allow_limit_up"] = True

    try:
        candidates = filter_func(all_stocks, **filter_kwargs)
    except Exception as e:
        logger.error("filter_func(%s) failed: %s", strategy, e)
        return {"error": f"策略 '{strategy}' 过滤失败: {e}", "pipeline_version": "v2"}

    if not candidates.empty:
        history_component = candidates.get(
            "history_score", pd.Series(0, index=candidates.index)).fillna(0)
        fast_component = candidates.get(
            "fast_score", pd.Series(0, index=candidates.index)).fillna(0)
        # v2: readiness is a HARD FILTER (exclude) rather than a -8 soft
        # penalty — this is cleaner; a stock with <90 days of history just
        # isn't eligible, instead of being penalized in the composite.
        ready_mask = candidates.get("history_ready",
                                    pd.Series(False, index=candidates.index)).fillna(False)
        if ready_mask.any():
            candidates = candidates[ready_mask].copy()
        # Re-pull the aligned series after the filter
        history_component = candidates["history_score"].fillna(0)
        fast_component = candidates["fast_score"].fillna(0)
        candidates["stage1_score"] = (
            fast_component * 0.55 + history_component * 0.45
        ).round(1)
        candidates = candidates.sort_values(
            by=["stage1_score", "history_score", "fast_score"],
            ascending=False,
        )
    stage1_count = len(candidates)

    if stage1_count == 0:
        summary = f"基础过滤后无股票通过（策略={strategy}，全市场{total_market}只）。"
        if data_health.get("is_stale"):
            summary += (
                f" ⚠️ 数据可能过期 (pct_chg覆盖率="
                f"{data_health.get('pct_chg_coverage', 0)*100:.1f}%, "
                f"最近交易日={data_health.get('last_trade_date')}, "
                f"距今 {data_health.get('staleness_days', '?')} 天),"
                f"建议重跑 slow_fetcher。"
            )
        return {
            "strategy": strategy,
            "pipeline_version": "v2",
            "total_market": total_market,
            "history_profile_count": history_profile_count,
            "stage1_count": 0,
            "stage2_count": 0,
            "qualified_count": 0,
            "quality_quantile": quality_quantile,
            "market_regime": market_regime,
            "data_health": data_health,
            "results": [],
            "summary": summary,
        }

    scan_candidates = candidates.head(stage1_limit)

    all_scanned = deep_scan(
        scan_candidates,
        top_n=len(scan_candidates),
        analysis_days=analysis_days,
        local_only=True,
    )
    stage2_count = len(all_scanned)

    # v2 core: recompute EQS with z-score composite BEFORE any cut
    all_scanned = _compute_eqs_v2(all_scanned)

    # v2 quality gate: top quantile, with honest floor
    qualified = _quantile_cut(all_scanned, quantile=quality_quantile,
                              min_keep=min_keep)

    # Cap at max_candidates
    results = qualified[:max_candidates]

    # Institutional bonus — computed on z-score composite AFTER the cut
    # so that inst score is additive (not baked into the main cut) but
    # still re-ranks within the kept set. Rationale: inst score is a
    # slow-moving quarterly signal, so it's a tiebreaker, not a primary
    # selector.
    try:
        from institutional import institutional_score as _inst_score
        _inst_conn = db.get_conn() if results else None
        for r in results:
            try:
                iscore = _inst_score(r["code"], conn=_inst_conn)
                r["institutional_score"]  = iscore["score"]
                r["institutional_detail"] = iscore["detail"]
                r["fund_count"]           = iscore["fund_count"]
                r["smart_money_types"]    = iscore["smart_money_types"]
                # v2: inst bonus is also z-scaled implicitly — +10 bonus
                # at max (inst_score=100) maps to roughly +1 z-unit, which
                # is comparable to the other factors in EQS_V2_WEIGHTS
                r["entry_quality_score"] = round(
                    (r.get("entry_quality_score") or 0) + iscore["score"] * 0.10, 2
                )
            except Exception:
                r["institutional_score"] = 0
                r["institutional_detail"] = "无数据"
                r["fund_count"] = 0
                r["smart_money_types"] = []
        if _inst_conn:
            _inst_conn.close()
        results = sorted(results,
                         key=lambda x: x.get("entry_quality_score") or -1e18,
                         reverse=True)
    except ImportError:
        pass

    return {
        "strategy": strategy,
        "pipeline_version": "v2",
        "total_market": total_market,
        "history_profile_count": history_profile_count,
        "stage1_count": stage1_count,
        "stage2_count": stage2_count,
        "qualified_count": len(results),
        "quality_quantile": quality_quantile,
        "min_keep": min_keep,
        "market_regime": market_regime,
        "data_health": data_health,
        "results": results,
        "summary": _format_summary(
            strategy, total_market, stage1_count, stage2_count,
            results, True,
            history_profile_count=history_profile_count,
            quality_threshold=quality_quantile * 100,  # display as pct
        ),
    }


# Quick self-test: weights sum to 1.0 (off-by-0.001 tolerance for float)
assert abs(sum(EQS_V2_WEIGHTS.values()) - 1.0) < 1e-6, \
    f"EQS_V2_WEIGHTS must sum to 1.0, got {sum(EQS_V2_WEIGHTS.values())}"
