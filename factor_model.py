"""
多因子股票选择与分析模型 (Multi-factor Stock Selection and Analysis Model)

用于A股市场的专业级因子模型，支持价值、质量、动量、低波动性、规模和流动性等多个因子维度。
实现包括因子计算、标准化、正交化、IC计算等核心功能。

Requirements:
- pandas >= 1.0
- numpy >= 1.15
"""

import db
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {
    # Value factors (25%) — strong A-share premium post-2019
    "ep": 0.17, "bp": 0.08,
    # Quality factors (30%) — NOTE: roe_stability+gm_stability (23%) currently always NaN
    # because stock_fundamentals DB table lacks roe_history/gross_margin columns.
    # composite_alpha_score() correctly redistributes their weight to other factors,
    # but this silently over-weights momentum. Populate DB columns to activate.
    "roe_stability": 0.18, "gm_stability": 0.05, "accrual": 0.05, "downside_dev": 0.02,
    # Risk/leverage (3%)
    "debt_equity": 0.03,
    # Momentum factors (13%) — reduced from 20%; A-share momentum weaker post-2021
    "mom_1m": 0.03, "mom_3m": 0.04, "mom_6m": 0.03, "mom_12_1m": 0.03,
    # Volatility/risk (11%)
    "vol_60d": 0.07, "beta": 0.04,
    # Size (5%)
    "log_mcap": 0.05,
    # Liquidity (8%)
    "turnover_20d": 0.04, "amihud": 0.04,
    # Short-term reversal (5%)
    "reversal_5d": 0.05,
}  # sum=1.00: Value(25)+Quality(30)+Leverage(3)+Momentum(13)+Vol(11)+Size(5)+Liq(8)+Rev(5)

# Constants for robustness
MAD_SCALE = 0.6745  # Scale factor for MAD to std deviation
WINSORIZE_THRESHOLD = 3  # MADs
MIN_HISTORY_DAYS = 60


def _get_db_connection():
    """Get database connection."""
    return db.get_conn()


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    """
    Compute z-score using median/MAD for robustness against outliers.

    Steps:
    1. Compute median and MAD (median absolute deviation)
    2. Winsorize at ±3 MADs
    3. Compute z-score as (x - median) / (MAD * 0.6745)

    Args:
        values: 1D array of values

    Returns:
        Z-score normalized array, NaN for invalid inputs
    """
    if len(values) == 0:
        return np.array([])

    values = np.asarray(values, dtype=np.float64)
    valid_mask = ~np.isnan(values)

    if valid_mask.sum() < 2:
        return np.full_like(values, np.nan)

    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values[valid_mask] - median))

    if mad < 1e-10:
        return np.zeros_like(values)

    # Winsorize at ±3 MADs
    lower_bound = median - WINSORIZE_THRESHOLD * mad
    upper_bound = median + WINSORIZE_THRESHOLD * mad
    winsorized = np.clip(values, lower_bound, upper_bound)

    # Z-score: divide by (MAD / 0.6745) to convert MAD to approximate std
    zscore = (winsorized - median) / (mad / MAD_SCALE)
    zscore[~valid_mask] = np.nan

    return zscore


def _spearman_rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Spearman rank correlation manually using formula:
    IC = 1 - 6*sum(d^2) / (n*(n^2-1))

    Args:
        x: First array
        y: Second array

    Returns:
        Spearman correlation coefficient
    """
    if len(x) != len(y) or len(x) < 2:
        return np.nan

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Remove NaN pairs
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    if valid_mask.sum() < 2:
        return np.nan

    x = x[valid_mask]
    y = y[valid_mask]
    n = len(x)

    # Compute ranks (handling ties with average rank)
    rank_x = _compute_ranks(x)
    rank_y = _compute_ranks(y)

    # Pearson formula on ranks — correctly handles ties (unlike simplified Spearman)
    rank_x_mean = np.mean(rank_x)
    rank_y_mean = np.mean(rank_y)
    numerator = np.sum((rank_x - rank_x_mean) * (rank_y - rank_y_mean))
    denom = np.sqrt(np.sum((rank_x - rank_x_mean) ** 2) * np.sum((rank_y - rank_y_mean) ** 2))
    rho = numerator / denom if denom > 1e-10 else 0.0

    return rho


def _compute_ranks(values: np.ndarray) -> np.ndarray:
    """Compute ranks with average handling for ties — O(n log n)."""
    n = len(values)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[order[j]] == values[order[j + 1]]:
            j += 1
        if j > i:
            avg = (i + j) / 2.0 + 1.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def compute_single_stock_factors(
    code: str,
    history_df: pd.DataFrame,
    fundamentals: Dict,
    benchmark_close: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute all factors for a single stock.

    Args:
        code: Stock code
        history_df: Historical OHLCV data with columns [date, open, high, low, close, volume, turnover, pct_chg]
        fundamentals: Dict with keys [pe_ttm, pb, total_mv, float_mv, roe, gross_margin, accrual, debt_equity]
        benchmark_close: Optional array of benchmark close prices for beta calculation

    Returns:
        Dict mapping factor name to computed value (raw, not z-scored)
    """
    factors = {}

    if history_df is None or len(history_df) == 0:
        return factors

    history_df = history_df.copy()
    history_df['date'] = pd.to_datetime(history_df['date'])
    history_df = history_df.sort_values('date').reset_index(drop=True)

    close = history_df['close'].values
    volume = history_df['volume'].values
    turnover = history_df['turnover'].values if 'turnover' in history_df else None

    # ============ VALUE FACTORS ============
    if 'pe_ttm' in fundamentals and fundamentals['pe_ttm'] and fundamentals['pe_ttm'] > 0:
        factors['ep'] = 1.0 / fundamentals['pe_ttm']
    else:
        factors['ep'] = np.nan

    if 'pb' in fundamentals and fundamentals['pb'] and fundamentals['pb'] > 0:
        factors['bp'] = 1.0 / fundamentals['pb']
    else:
        factors['bp'] = np.nan

    # Dividend/Price - TODO: connect to actual dividend data source; weight reallocated to ep
    # factors['dp'] = 0.0  # removed from DEFAULT_WEIGHTS

    # Sales/Price - TODO: connect to actual sales data source; weight reallocated to roe_stability
    # factors['sp'] = 0.0  # removed from DEFAULT_WEIGHTS

    # ============ QUALITY FACTORS ============
    if 'roe' in fundamentals and isinstance(fundamentals['roe'], (list, np.ndarray)):
        roe_values = np.array(fundamentals['roe'], dtype=np.float64)
        roe_values = roe_values[~np.isnan(roe_values)]
        if len(roe_values) >= 4:  # require 4+ quarters for reliable std estimate
            base = 1.0 / (np.std(roe_values) + 0.01)
            # Growth-aware adjustment: improving ROE trend should not be penalized as instability.
            # roe_values assumed oldest-first (chronological order).
            median_roe = np.median(roe_values)
            std_roe = np.std(roe_values)
            recent_roe = roe_values[-1]
            if recent_roe > median_roe + 0.5 * std_roe:
                base *= 1.2   # improving quality: reward trend
            elif recent_roe < median_roe - 0.5 * std_roe:
                base *= 0.8   # deteriorating quality: penalize
            factors['roe_stability'] = base
        else:
            factors['roe_stability'] = np.nan
    else:
        factors['roe_stability'] = np.nan

    if 'gross_margin' in fundamentals and isinstance(fundamentals['gross_margin'], (list, np.ndarray)):
        gm_values = np.array(fundamentals['gross_margin'], dtype=np.float64)
        gm_values = gm_values[~np.isnan(gm_values)]
        if len(gm_values) >= 4:  # require 4+ quarters for reliable std estimate
            factors['gm_stability'] = 1.0 / (np.std(gm_values) + 0.01)
        else:
            factors['gm_stability'] = np.nan
    else:
        factors['gm_stability'] = np.nan

    if 'accrual' in fundamentals and fundamentals['accrual'] is not None:
        factors['accrual'] = -fundamentals['accrual']  # Negative accrual is good
    else:
        factors['accrual'] = np.nan

    if 'debt_equity' in fundamentals and fundamentals['debt_equity'] is not None:
        factors['debt_equity'] = -fundamentals['debt_equity']  # Lower is better
    else:
        factors['debt_equity'] = np.nan

    # ============ MOMENTUM FACTORS ============
    # 1-month return (skip last 5 days for reversal)
    if len(close) >= 25:
        ret_1m = (close[-6] / close[-25] - 1.0) if close[-25] > 0 else np.nan
        factors['mom_1m'] = ret_1m
    else:
        factors['mom_1m'] = np.nan

    # 3-month return (skip last 5 days to avoid short-term reversal contamination)
    if len(close) >= 70:
        ret_3m = (close[-6] / close[-70] - 1.0) if close[-70] > 0 else np.nan
        factors['mom_3m'] = ret_3m
    else:
        factors['mom_3m'] = np.nan

    # 6-month return (skip last 5 days)
    if len(close) >= 135:
        ret_6m = (close[-6] / close[-135] - 1.0) if close[-135] > 0 else np.nan
        factors['mom_6m'] = ret_6m
    else:
        factors['mom_6m'] = np.nan

    # 12-1 month return: 252-day lookback, skip most recent 21 days (1-month reversal window)
    # Standard: return from T-252 to T-21, NOT including the last month
    if len(close) >= 273:
        ret_12_1m = (close[-21] / close[-273] - 1.0) if close[-273] > 0 else np.nan
        factors['mom_12_1m'] = ret_12_1m
    else:
        factors['mom_12_1m'] = np.nan

    # 5-day reversal — asymmetric for A-shares:
    # up-reversal decays faster (retail stop-loss), down-reversal persists (accumulation)
    if len(close) >= 5:
        ret_5d = (close[-1] / close[-5] - 1.0) if close[-5] > 0 else np.nan
        if ret_5d is not None and not np.isnan(ret_5d):
            # Up moves: stronger reversal signal (×1.3); Down moves: weaker (×0.8)
            asymmetry = 1.3 if ret_5d > 0 else 0.8
            factors['reversal_5d'] = -ret_5d * asymmetry
        else:
            factors['reversal_5d'] = np.nan
    else:
        factors['reversal_5d'] = np.nan

    # ============ LOW VOLATILITY FACTORS ============
    # 60-day realized volatility
    if len(close) >= 60:
        returns_60d = np.diff(np.log(close[-60:])) * 100  # Convert to percentage
        vol_60d = np.std(returns_60d)
        factors['vol_60d'] = -vol_60d  # Lower volatility is preferred (negative sign)
    else:
        factors['vol_60d'] = np.nan

    # Beta to benchmark
    if benchmark_close is not None and len(benchmark_close) >= len(close):
        if len(close) >= 60:
            stock_returns = np.diff(np.log(close[-60:]))
            bench_returns = np.diff(np.log(benchmark_close[-60:]))

            cov_matrix = np.cov(stock_returns, bench_returns)
            bench_var = np.var(bench_returns)

            if bench_var > 0:
                beta = cov_matrix[0, 1] / bench_var
                factors['beta'] = -beta  # Lower beta is preferred
            else:
                factors['beta'] = np.nan
        else:
            factors['beta'] = np.nan
    else:
        factors['beta'] = np.nan

    # Downside deviation — full-sample semi-variance (MAR=0), avoids NaN when all returns positive
    if len(close) >= 60:
        returns_60d = np.diff(np.log(close[-60:])) * 100
        downside_sq = np.minimum(returns_60d, 0) ** 2
        downside_dev = np.sqrt(np.mean(downside_sq))
        factors['downside_dev'] = -downside_dev if downside_dev > 0 else 0.0
    else:
        factors['downside_dev'] = np.nan

    # ============ SIZE FACTOR ============
    if 'float_mv' in fundamentals and fundamentals['float_mv'] and fundamentals['float_mv'] > 0:
        factors['log_mcap'] = -np.log(fundamentals['float_mv'])  # Negative for small cap premium
    elif 'total_mv' in fundamentals and fundamentals['total_mv'] and fundamentals['total_mv'] > 0:
        factors['log_mcap'] = -np.log(fundamentals['total_mv'])
    else:
        factors['log_mcap'] = np.nan

    # ============ LIQUIDITY FACTORS ============
    # 20-day average turnover — winsorized to avoid trading-halt distortion
    if turnover is not None and len(turnover) >= 20:
        t20 = turnover[-20:]
        valid_t = t20[t20 > 0]
        if len(valid_t) >= 10:
            p10, p90 = np.percentile(valid_t, 10), np.percentile(valid_t, 90)
            t20_clipped = np.clip(t20, p10, p90)
        else:
            t20_clipped = t20
        factors['turnover_20d'] = np.mean(t20_clipped)
    else:
        factors['turnover_20d'] = np.nan

    # Amihud illiquidity ratio — median (not mean) to handle A-share illiquidity spikes
    if len(close) >= 20 and turnover is not None and len(turnover) >= 20:
        returns_20d = np.diff(np.log(close[-20:])) * 100  # percentage, 19 elements
        turnover_20d = turnover[-19:]
        valid_mask = turnover_20d > 0
        if valid_mask.sum() >= 5:
            amihud_values = np.abs(returns_20d[valid_mask]) / turnover_20d[valid_mask]
            amihud = np.median(amihud_values)  # median avoids suspension-day spikes
            factors['amihud'] = -amihud  # higher score = more liquid = preferred
        else:
            factors['amihud'] = np.nan
    else:
        factors['amihud'] = np.nan

    return factors


def compute_cross_sectional_factors(all_stocks: List[Dict]) -> pd.DataFrame:
    """
    Compute z-score normalized factors across all stocks.

    Args:
        all_stocks: List of dicts, each with keys {code, factors_raw}
                   where factors_raw is output from compute_single_stock_factors

    Returns:
        DataFrame with index = stock codes, columns = factor names (z-score normalized)
    """
    if not all_stocks:
        return pd.DataFrame()

    # Extract all factors into a dict of lists
    factor_names = set()
    for stock in all_stocks:
        if 'factors_raw' in stock:
            factor_names.update(stock['factors_raw'].keys())

    # Initialize data structure
    data = {factor: [] for factor in factor_names}
    codes = []

    for stock in all_stocks:
        codes.append(stock['code'])
        factors_raw = stock.get('factors_raw', {})

        for factor in factor_names:
            data[factor].append(factors_raw.get(factor, np.nan))

    # Create DataFrame
    df = pd.DataFrame(data, index=codes)

    # Z-score normalize each factor across stocks
    df_zscore = pd.DataFrame(index=codes)

    for col in df.columns:
        values = df[col].values
        zscore = _robust_zscore(values)
        df_zscore[col] = zscore

    return df_zscore


def factor_correlation_matrix(factor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix between all factors.

    Args:
        factor_df: DataFrame with factors as columns (can be z-scored or raw)

    Returns:
        Correlation matrix (symmetric DataFrame)
    """
    if factor_df.empty:
        return pd.DataFrame()

    return factor_df.corr(method='pearson')


def orthogonalize_factors(
    factor_df: pd.DataFrame,
    method: str = "symmetric"
) -> pd.DataFrame:
    """
    Remove redundant factor exposure using orthogonalization.
    Implements symmetric Gram-Schmidt variant.

    Args:
        factor_df: DataFrame with factors as columns, stocks as index
        method: "symmetric" (default) or "gram_schmidt"

    Returns:
        DataFrame with orthogonalized factors
    """
    if factor_df.empty or factor_df.shape[1] <= 1:
        return factor_df.copy()

    factors = factor_df.values.copy()
    n_stocks, n_factors = factors.shape

    # Handle NaN values with column means
    for j in range(n_factors):
        col_mean = np.nanmean(factors[:, j])
        mask = np.isnan(factors[:, j])
        factors[mask, j] = col_mean if not np.isnan(col_mean) else 0

    orthogonal = np.zeros_like(factors, dtype=np.float64)

    if method == "gram_schmidt":
        # Standard Gram-Schmidt
        for i in range(n_factors):
            vec = factors[:, i].astype(np.float64)

            # Orthogonalize against previous vectors
            for j in range(i):
                proj = np.dot(vec, orthogonal[:, j]) / (np.dot(orthogonal[:, j], orthogonal[:, j]) + 1e-10)
                vec = vec - proj * orthogonal[:, j]

            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                orthogonal[:, i] = vec / norm
            else:
                orthogonal[:, i] = vec

    else:  # symmetric
        # Symmetric orthogonalization (more stable)
        # Step 1: Compute correlation matrix
        corr = np.corrcoef(factors.T)
        corr = np.nan_to_num(corr, nan=0, posinf=0, neginf=0)

        # Step 2: Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, 1e-10)  # Ensure positive

        # Step 3: Compute transformation matrix: Q = V * sqrt(Lambda)^(-1) * V^T
        sqrt_lambda_inv = np.diag(1.0 / np.sqrt(eigvals))
        transform = eigvecs @ sqrt_lambda_inv @ eigvecs.T

        # Step 4: Apply transformation
        orthogonal = factors @ transform

    result_df = pd.DataFrame(
        orthogonal,
        index=factor_df.index,
        columns=factor_df.columns
    )

    return result_df


def composite_alpha_score(
    factor_df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None
) -> pd.Series:
    """
    Compute weighted composite alpha score from orthogonalized factors.

    Args:
        factor_df: DataFrame with z-score normalized factors (typically orthogonalized)
        weights: Dict mapping factor names to weights. If None, uses DEFAULT_WEIGHTS.
                 Weights are automatically normalized to sum to 1.

    Returns:
        Series of alpha scores indexed by stock code
    """
    if factor_df.empty:
        return pd.Series(dtype=np.float64)

    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    # Filter weights to factors present in factor_df
    valid_weights = {k: v for k, v in weights.items() if k in factor_df.columns}

    if not valid_weights:
        return pd.Series(np.zeros(len(factor_df)), index=factor_df.index)

    # Normalize weights
    total_weight = sum(valid_weights.values())
    normalized_weights = {k: v / total_weight for k, v in valid_weights.items()}

    # Compute composite score — redistribute weights to non-NaN factors instead of zero-filling.
    # Zero-filling biases stocks with many missing factors toward zero alpha; redistribution is unbiased.
    alpha_scores = np.zeros(len(factor_df))
    weight_sum = np.zeros(len(factor_df))
    missing_count = np.zeros(len(factor_df))

    for factor, weight in normalized_weights.items():
        factor_values = factor_df[factor].values
        valid_mask = ~np.isnan(factor_values)
        alpha_scores[valid_mask] += factor_values[valid_mask] * weight
        weight_sum[valid_mask] += weight
        missing_count[~valid_mask] += 1

    # Normalize by sum of weights that were actually used (handles partial missingness)
    with np.errstate(invalid='ignore', divide='ignore'):
        alpha_scores = np.where(weight_sum > 0, alpha_scores / weight_sum, np.nan)

    # Mark as NaN if >40% factors missing (require ≥60% coverage for reliable scoring)
    alpha_scores[missing_count > len(normalized_weights) * 0.4] = np.nan

    return pd.Series(alpha_scores, index=factor_df.index)


def factor_exposure_report(
    code: str,
    factor_df: pd.DataFrame
) -> Dict[str, Dict]:
    """
    For a single stock, return its exposure to each factor with percentile rank.

    Args:
        code: Stock code
        factor_df: DataFrame with factors (index = codes, columns = factors)

    Returns:
        Dict mapping factor name to {value, percentile}
    """
    if code not in factor_df.index or factor_df.empty:
        return {}

    report = {}
    stock_row = factor_df.loc[code]

    for factor in factor_df.columns:
        value = stock_row[factor]

        # Compute percentile rank (0-100)
        all_values = factor_df[factor].values
        valid_values = all_values[~np.isnan(all_values)]

        if len(valid_values) > 0 and not np.isnan(value):
            percentile = (np.sum(valid_values <= value) / len(valid_values)) * 100
        else:
            percentile = np.nan

        report[factor] = {
            'value': float(value) if not np.isnan(value) else None,
            'percentile': float(percentile) if not np.isnan(percentile) else None
        }

    return report


def compute_ic(
    factor_df: pd.DataFrame,
    forward_returns: pd.Series,
    holding_days: int = 20,
) -> Dict[str, float]:
    """
    Compute Information Coefficient (rank correlation) between each factor and forward returns.

    Args:
        factor_df: DataFrame with factors (index = codes, columns = factors)
        forward_returns: Series of forward returns (index = codes) at the specified horizon.
            Must be computed at holding_days lag to match the intended rebalance frequency.
            Default 20 days ≈ monthly rebalance. For weekly: 5. For quarterly: 60.
        holding_days: Intended holding period in trading days (documentation only; caller
            is responsible for computing forward_returns at this exact lag).

    Returns:
        Dict mapping factor name to IC value
    """
    if factor_df.empty or forward_returns.empty:
        return {}

    # Align indices
    common_idx = factor_df.index.intersection(forward_returns.index)
    if len(common_idx) < 2:
        return {}

    factor_df_aligned = factor_df.loc[common_idx]
    returns_aligned = forward_returns.loc[common_idx].values

    ic_dict = {}

    for factor in factor_df_aligned.columns:
        factor_values = factor_df_aligned[factor].values
        ic = _spearman_rank_correlation(factor_values, returns_aligned)
        ic_dict[factor] = ic

    return ic_dict


def sector_neutralize(
    factor_df: pd.DataFrame,
    sector_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Remove sector bias from factors by demeaning within each sector group.

    Args:
        factor_df: DataFrame with factors (index = codes, columns = factors)
        sector_map: Dict mapping stock code to sector name

    Returns:
        Sector-neutralized DataFrame (same shape as input)
    """
    if factor_df.empty:
        return factor_df.copy()

    result = factor_df.copy()

    for col in result.columns:
        # Group by sector
        codes = result.index.tolist()
        sectors = [sector_map.get(code, 'unknown') for code in codes]

        # Demean within each sector
        for sector in set(sectors):
            sector_mask = np.array(sectors) == sector
            sector_values = result.loc[sector_mask, col].values
            sector_mean = np.nanmean(sector_values)
            result.loc[sector_mask, col] = sector_values - sector_mean

    return result


# ============ HELPER FUNCTIONS ============

def load_stock_data(code: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """
    Load historical and fundamental data for a stock from database.

    Args:
        code: Stock code

    Returns:
        Tuple of (history_df, fundamentals_dict) or (None, None) if not found
    """
    conn = None
    try:
        conn = _get_db_connection()

        # Load history — 500 days covers all factor lookbacks (mom_12_1m needs ~265 days)
        history_df = db.read_sql(
            """SELECT * FROM (
                SELECT * FROM stock_history WHERE code = %s ORDER BY date DESC LIMIT 500
            ) sub ORDER BY date ASC""",
            conn,
            params=(code,)
        )

        # Load fundamentals
        fundamentals_df = db.read_sql(
            "SELECT * FROM stock_fundamentals WHERE code = %s ORDER BY trade_date DESC LIMIT 1",
            conn,
            params=(code,)
        )

        if history_df.empty or fundamentals_df.empty:
            return None, None

        fundamentals = fundamentals_df.iloc[0].to_dict()

        return history_df, fundamentals

    except Exception as e:
        print(f"Error loading data for {code}: {e}")
        return None, None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def batch_compute_factors(codes: List[str], benchmark_close: Optional[np.ndarray] = None) -> List[Dict]:
    """
    Batch compute factors for multiple stocks.

    Args:
        codes: List of stock codes
        benchmark_close: Optional benchmark close prices for beta calculation

    Returns:
        List of dicts with keys {code, factors_raw}
    """
    results = []

    for code in codes:
        history_df, fundamentals = load_stock_data(code)

        if history_df is not None and fundamentals is not None:
            if len(history_df) >= MIN_HISTORY_DAYS:
                factors_raw = compute_single_stock_factors(
                    code,
                    history_df,
                    fundamentals,
                    benchmark_close
                )
                results.append({'code': code, 'factors_raw': factors_raw})

    return results


if __name__ == "__main__":
    # Example usage
    print("Factor Model Module Loaded")
    print(f"Default factor weights: {len(DEFAULT_WEIGHTS)} factors")
