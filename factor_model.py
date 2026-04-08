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
    "ep": 0.12, "bp": 0.08, "dp": 0.05, "sp": 0.05,
    "roe_stability": 0.08, "gm_stability": 0.05,
    "accrual": 0.04, "debt_equity": 0.03,
    "mom_1m": 0.05, "mom_3m": 0.06, "mom_6m": 0.05,
    "mom_12_1m": 0.04,
    "vol_60d": 0.06, "beta": 0.04, "downside_dev": 0.03,
    "log_mcap": 0.05,
    "turnover_20d": 0.04, "amihud": 0.04,
    "reversal_5d": 0.04,
}

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

    # Z-score
    zscore = (winsorized - median) / (mad * MAD_SCALE)
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

    d_squared = np.sum((rank_x - rank_y) ** 2)
    rho = 1 - (6 * d_squared) / (n * (n ** 2 - 1))

    return rho


def _compute_ranks(values: np.ndarray) -> np.ndarray:
    """Compute ranks with average handling for ties."""
    order = np.argsort(values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order)) + 1

    # Handle ties by assigning average rank
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if values[order[i]] == values[order[j]]:
                # Find all equal values
                start = i
                while start > 0 and values[order[start - 1]] == values[order[i]]:
                    start -= 1
                end = j + 1
                while end < len(values) and values[order[end]] == values[order[i]]:
                    end += 1

                avg_rank = np.mean(np.arange(start + 1, end + 1))
                for k in range(start, end):
                    ranks[order[k]] = avg_rank

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

    # Dividend/Price - approximation: 0 if no dividend data
    factors['dp'] = 0.0

    # Sales/Price - approximation using fundamental data if available
    factors['sp'] = 0.0

    # ============ QUALITY FACTORS ============
    if 'roe' in fundamentals and isinstance(fundamentals['roe'], (list, np.ndarray)):
        roe_values = np.array(fundamentals['roe'], dtype=np.float64)
        roe_values = roe_values[~np.isnan(roe_values)]
        if len(roe_values) >= 2:
            factors['roe_stability'] = 1.0 / (np.std(roe_values) + 0.01)  # Inverse of volatility
        else:
            factors['roe_stability'] = np.nan
    else:
        factors['roe_stability'] = np.nan

    if 'gross_margin' in fundamentals and isinstance(fundamentals['gross_margin'], (list, np.ndarray)):
        gm_values = np.array(fundamentals['gross_margin'], dtype=np.float64)
        gm_values = gm_values[~np.isnan(gm_values)]
        if len(gm_values) >= 2:
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

    # 3-month return
    if len(close) >= 65:
        ret_3m = (close[-1] / close[-65] - 1.0) if close[-65] > 0 else np.nan
        factors['mom_3m'] = ret_3m
    else:
        factors['mom_3m'] = np.nan

    # 6-month return
    if len(close) >= 130:
        ret_6m = (close[-1] / close[-130] - 1.0) if close[-130] > 0 else np.nan
        factors['mom_6m'] = ret_6m
    else:
        factors['mom_6m'] = np.nan

    # 12-1 month return (skip recent month)
    if len(close) >= 260:
        ret_12_1m = (close[-25] / close[-260] - 1.0) if close[-260] > 0 else np.nan
        factors['mom_12_1m'] = ret_12_1m
    else:
        factors['mom_12_1m'] = np.nan

    # 5-day reversal
    if len(close) >= 5:
        ret_5d = (close[-1] / close[-5] - 1.0) if close[-5] > 0 else np.nan
        factors['reversal_5d'] = -ret_5d  # Negative of recent return (reversal signal)
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

    # Downside deviation
    if len(close) >= 60:
        returns_60d = np.diff(np.log(close[-60:])) * 100
        downside_returns = returns_60d[returns_60d < 0]
        if len(downside_returns) > 0:
            downside_dev = np.sqrt(np.mean(downside_returns ** 2))
            factors['downside_dev'] = -downside_dev  # Lower is preferred
        else:
            factors['downside_dev'] = np.nan
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
    # 20-day average turnover
    if turnover is not None and len(turnover) >= 20:
        avg_turnover_20d = np.mean(turnover[-20:])
        factors['turnover_20d'] = avg_turnover_20d
    else:
        factors['turnover_20d'] = np.nan

    # Amihud illiquidity ratio
    if len(close) >= 20 and turnover is not None and len(turnover) >= 20:
        returns_20d = np.diff(np.log(close[-20:])) * 100  # In percentage (19 elements)
        turnover_20d = turnover[-19:]  # Match length of returns

        # Avoid division by zero
        valid_mask = turnover_20d > 0
        if valid_mask.sum() > 0:
            amihud_values = np.abs(returns_20d[valid_mask]) / turnover_20d[valid_mask]
            amihud = np.mean(amihud_values)
            factors['amihud'] = -amihud  # Lower illiquidity is preferred
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

    # Compute composite score
    alpha_scores = np.zeros(len(factor_df))
    missing_count = np.zeros(len(factor_df))

    for factor, weight in normalized_weights.items():
        factor_values = factor_df[factor].values
        alpha_scores += np.nan_to_num(factor_values, nan=0) * weight
        missing_count += np.isnan(factor_values).astype(int)

    # Mark as NaN if >50% factors missing
    alpha_scores[missing_count > len(normalized_weights) * 0.5] = np.nan

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
    forward_returns: pd.Series
) -> Dict[str, float]:
    """
    Compute Information Coefficient (rank correlation) between each factor and forward returns.

    Args:
        factor_df: DataFrame with factors (index = codes, columns = factors)
        forward_returns: Series with forward returns (index = codes)

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
    try:
        conn = _get_db_connection()

        # Load history
        history_df = db.read_sql(
            "SELECT * FROM stock_history WHERE code = %s ORDER BY date",
            conn,
            params=(code,)
        )

        # Load fundamentals
        fundamentals_df = db.read_sql(
            "SELECT * FROM stock_fundamentals WHERE code = %s ORDER BY trade_date DESC LIMIT 1",
            conn,
            params=(code,)
        )

        conn.close()

        if history_df.empty or fundamentals_df.empty:
            return None, None

        fundamentals = fundamentals_df.iloc[0].to_dict()

        return history_df, fundamentals

    except Exception as e:
        print(f"Error loading data for {code}: {e}")
        return None, None


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
