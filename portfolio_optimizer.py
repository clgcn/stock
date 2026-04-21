"""
Portfolio Optimization Module for Chinese A-Shares

Implements portfolio optimization techniques using only numpy and pandas.
Supports various constraints, risk models, and A-share specific requirements.

Key Features:
- Multiple return estimation methods (historical, shrinkage)
- Ledoit-Wolf shrinkage covariance estimation (manual implementation)
- Maximum Sharpe ratio optimization (projected gradient ascent)
- Minimum variance optimization
- Risk parity portfolio
- Black-Litterman model
- Comprehensive portfolio analysis and rebalancing
- A-share specific constraints (lot sizes, transaction costs, limits)
"""

import db
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# Configuration

# A-Share Constants
STAMP_TAX_RATE = 0.001  # 0.1% on sales
COMMISSION_RATE = 0.00025  # 0.025%
MIN_COMMISSION = 5  # CNY
LOT_SIZE = 100  # minimum trading unit
RISK_FREE_RATE = 0.025  # default annual risk-free rate (unified across system)

# Market limits
DAILY_LIMIT_MAIN = 0.10  # 10% for main board
DAILY_LIMIT_CHINEXT = 0.20  # 20% for ChiNext (30x)
DAILY_LIMIT_STAR = 0.20  # 20% for STAR board (688xxx)

# Portfolio constraints (defaults)
DEFAULT_MAX_WEIGHT = 0.15  # max 15% per stock
DEFAULT_MIN_WEIGHT = 0.0  # no minimum (0% allowed)
DEFAULT_MAX_STOCKS = 20
DEFAULT_CONCENTRATION_LIMIT = 0.40  # top 3 stocks <= 40%


# ============================================================================
# Data Loading
# ============================================================================

def _load_histories(codes: List[str], days: int = 250) -> Dict[str, pd.DataFrame]:
    """
    Load historical price data from database.

    Args:
        codes: List of stock codes (e.g., ['600519', '000858'])
        days: Number of trading days to load (default 250 ~ 1 year)

    Returns:
        Dict mapping code -> DataFrame with columns [date, close]
    """
    result = {}
    conn = None
    try:
        conn = db.get_conn()
        for code in codes:
            query = """
                SELECT date, close FROM stock_history
                WHERE code=%s
                ORDER BY date DESC
                LIMIT %s
            """
            df = db.read_sql(query, conn, params=(code, days))
            if not df.empty:
                result[code] = df.sort_values('date').reset_index(drop=True)
    except Exception as e:
        warnings.warn(f"Failed to load histories: {e}")
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    return result


def _validate_histories(history_dict: Dict[str, pd.DataFrame], min_periods: int = 30) -> None:
    """
    Validate that all histories have sufficient data.

    Args:
        history_dict: Dict of code -> DataFrame
        min_periods: Minimum required data points

    Raises:
        ValueError: If any stock has insufficient data
    """
    for code, df in history_dict.items():
        if len(df) < min_periods:
            raise ValueError(
                f"Stock {code} has only {len(df)} data points, need >= {min_periods}"
            )


# ============================================================================
# Return Estimation
# ============================================================================

def estimate_returns(
    history_dict: Dict[str, pd.DataFrame],
    method: str = "shrinkage"
) -> pd.Series:
    """
    Estimate expected returns for each stock.

    Args:
        history_dict: Dict mapping code -> DataFrame with 'close' column
        method: Estimation method
            - 'historical': Simple mean of daily returns, annualized
            - 'shrinkage': James-Stein shrinkage toward grand mean (recommended)

    Returns:
        pd.Series indexed by stock code, annualized expected returns

    Raises:
        ValueError: If method is unknown or data is insufficient
    """
    if not history_dict:
        raise ValueError("history_dict cannot be empty")

    # Calculate daily returns for each stock
    daily_returns = {}
    for code, df in history_dict.items():
        if len(df) < 2:
            raise ValueError(f"Stock {code} has insufficient data for return estimation")
        ret = df['close'].pct_change().dropna()
        daily_returns[code] = ret.values

    if method == "historical":
        # Simple annualized mean return
        returns = {}
        for code, ret_array in daily_returns.items():
            mean_daily = np.mean(ret_array)
            annual_return = (1 + mean_daily) ** 252 - 1
            returns[code] = annual_return
        return pd.Series(returns)

    elif method == "shrinkage":
        # James-Stein shrinkage toward grand mean
        returns = {}
        all_rets = np.concatenate(list(daily_returns.values()))
        grand_mean = np.mean(all_rets)
        grand_annual = (1 + grand_mean) ** 252 - 1

        for code, ret_array in daily_returns.items():
            sample_mean = np.mean(ret_array)
            sample_annual = (1 + sample_mean) ** 252 - 1
            n = len(ret_array)

            # James-Stein shrinkage intensity
            # Shrink toward grand mean
            var_estimate = np.var(ret_array, ddof=1)
            if var_estimate > 1e-8:
                shrink_intensity = (var_estimate + 1e-8) / (n * var_estimate + 1e-8)
                shrink_intensity = min(shrink_intensity, 1.0)
            else:
                shrink_intensity = 0.5

            shrunk_annual = (1 - shrink_intensity) * sample_annual + shrink_intensity * grand_annual
            returns[code] = shrunk_annual

        return pd.Series(returns)

    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================================
# Covariance Estimation
# ============================================================================

def estimate_covariance(
    history_dict: Dict[str, pd.DataFrame],
    method: str = "ledoit_wolf",
    use_ewma: bool = True,
    ewma_span: int = 60,
) -> pd.DataFrame:
    """
    Estimate covariance matrix of returns.

    Args:
        history_dict: Dict mapping code -> DataFrame with 'close' column
        method: Estimation method
            - 'sample': Raw sample covariance (can be unstable)
            - 'ledoit_wolf': Ledoit-Wolf shrinkage toward constant correlation matrix
        use_ewma: If True, apply exponential decay (span=60 days) so recent
            observations get higher weight. Critical for A-share regime changes.
        ewma_span: Half-life parameter for EWMA (default 60 trading days ≈ 3 months)

    Returns:
        pd.DataFrame: Annualized covariance matrix (codes x codes)

    Raises:
        ValueError: If method is unknown or data is insufficient
    """
    if not history_dict:
        raise ValueError("history_dict cannot be empty")

    codes = sorted(history_dict.keys())

    # Calculate daily returns
    daily_returns = {}
    for code in codes:
        df = history_dict[code]
        ret = df['close'].pct_change().dropna()
        daily_returns[code] = ret.values

    # Align data to common length
    min_len = min(len(ret) for ret in daily_returns.values())
    returns_array = np.column_stack([
        daily_returns[code][-min_len:]
        for code in codes
    ])

    # EWMA: exponential weights so recent regime dominates (A-share specific)
    if use_ewma and min_len > ewma_span:
        decay = 2.0 / (ewma_span + 1)
        raw_weights = np.array([(1 - decay) ** i for i in range(min_len - 1, -1, -1)])
        ewma_weights = raw_weights / raw_weights.sum()
    else:
        ewma_weights = None

    def _weighted_cov(arr: np.ndarray, w=None) -> np.ndarray:
        if w is None:
            return np.cov(arr.T)
        mean = np.average(arr, axis=0, weights=w)
        centered = arr - mean
        return (centered.T * w) @ centered / (1 - np.sum(w ** 2))

    if method == "sample":
        cov_matrix = _weighted_cov(returns_array, ewma_weights) * 252
        return pd.DataFrame(cov_matrix, index=codes, columns=codes)

    elif method == "ledoit_wolf":
        # Ledoit-Wolf shrinkage toward constant correlation matrix
        n_assets = len(codes)
        n_obs = returns_array.shape[0]

        # Sample covariance — use EWMA weights if available for regime sensitivity
        sample_cov = _weighted_cov(returns_array, ewma_weights)

        # Target: constant correlation matrix
        # Compute average pairwise correlation
        if ewma_weights is not None:
            mean = np.average(returns_array, axis=0, weights=ewma_weights)
            centered = returns_array - mean
            std = np.sqrt(np.diag(sample_cov))
            corr_matrix = sample_cov / np.outer(std, std)
        else:
            corr_matrix = np.corrcoef(returns_array.T)
        np.fill_diagonal(corr_matrix, 1.0)

        # Average correlation (off-diagonal)
        mask = ~np.eye(n_assets, dtype=bool)
        if mask.sum() > 0:
            avg_corr = np.mean(corr_matrix[mask])
        else:
            avg_corr = 0.0

        # Shrinkage target: constant correlation with sample variances
        std_diag = np.diag(np.sqrt(np.diag(sample_cov)))
        target_corr = np.eye(n_assets) * (1 - avg_corr) + avg_corr
        target_cov = std_diag @ target_corr @ std_diag

        # Compute optimal shrinkage intensity (Ledoit-Wolf formula)
        # beta = ((1 - 2/p) * trace(F^2) + trace(F)^2) / ((n+1-2/p) * (trace(F^2) - trace(F)^2/p))
        # where F = sample_cov - target_cov

        F = sample_cov - target_cov
        trace_F2 = np.trace(F @ F)
        trace_F = np.trace(F)
        p = n_assets

        numerator = (1 - 2.0 / p) * trace_F2 + trace_F ** 2
        denominator = (n_obs + 1 - 2.0 / p) * (trace_F2 - trace_F ** 2 / p)

        if abs(denominator) > 1e-10:
            beta = numerator / denominator
            beta = np.clip(beta, 0.0, 1.0)
        else:
            beta = 0.5

        # Shrunk covariance
        shrunk_cov = (1 - beta) * sample_cov + beta * target_cov

        # Annualize (252 trading days)
        shrunk_cov = shrunk_cov * 252

        return pd.DataFrame(shrunk_cov, index=codes, columns=codes)

    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================================
# Regularization & Matrix Utilities
# ============================================================================

def _regularize_covariance(cov: pd.DataFrame, min_eig_ratio: float = 0.01) -> np.ndarray:
    """
    Regularize covariance matrix to ensure it's well-conditioned.

    Args:
        cov: Covariance matrix
        min_eig_ratio: Minimum eigenvalue as ratio of max eigenvalue

    Returns:
        Regularized covariance matrix (numpy array)
    """
    cov_array = cov.values
    eig_vals, eig_vecs = np.linalg.eigh(cov_array)

    # Ensure no negative eigenvalues
    eig_vals = np.maximum(eig_vals, 1e-8)

    # Enforce minimum eigenvalue ratio
    max_eig = np.max(eig_vals)
    min_eig = max(np.min(eig_vals), min_eig_ratio * max_eig)
    eig_vals = np.maximum(eig_vals, min_eig)

    # Reconstruct covariance
    cov_reg = eig_vecs @ np.diag(eig_vals) @ eig_vecs.T
    return cov_reg


def _project_to_simplex(x: np.ndarray) -> np.ndarray:
    """
    Project vector to probability simplex (sum to 1, non-negative).
    Uses Duchi et al. (2008) algorithm.

    Args:
        x: Vector to project

    Returns:
        Projected vector on simplex
    """
    n = len(x)
    x_sorted = np.sort(x)[::-1]
    cumsum = np.cumsum(x_sorted)
    k = np.arange(1, n + 1)
    rho = np.where(x_sorted + (1 - cumsum) / k > 0)[0][-1]
    theta = (1 - cumsum[rho]) / (rho + 1)

    return np.maximum(x + theta, 0)


def _clip_weights(
    weights: np.ndarray,
    min_weight: float = 0.0,
    max_weight: float = 1.0
) -> np.ndarray:
    """
    Clip weights to [min_weight, max_weight] and renormalize.

    Args:
        weights: Portfolio weights
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset

    Returns:
        Clipped and normalized weights
    """
    weights_clipped = np.clip(weights, min_weight, max_weight)
    weights_sum = np.sum(weights_clipped)
    if weights_sum > 1e-10:
        weights_clipped = weights_clipped / weights_sum
    else:
        weights_clipped = np.ones_like(weights) / len(weights)

    return weights_clipped


# ============================================================================
# Core Optimization: Maximum Sharpe Ratio
# ============================================================================

def optimize_max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free: float = RISK_FREE_RATE,
    constraints: Optional[Dict] = None
) -> Dict:
    """
    Find maximum Sharpe ratio portfolio using projected gradient ascent.

    Uses iterative gradient-based optimization with simplex projection
    to handle the constraint that weights sum to 1.

    Args:
        mu: Expected returns (pd.Series indexed by code)
        cov: Covariance matrix (pd.DataFrame)
        risk_free: Risk-free rate (default 0.02)
        constraints: Dict with optional keys:
            - 'max_weight': Maximum weight per stock (default 0.15)
            - 'min_weight': Minimum weight per stock (default 0.0)
            - 'max_stocks': Maximum number of stocks (default 20)

    Returns:
        Dict with keys:
            - 'weights': Dict {code: weight}
            - 'expected_return': Portfolio expected return
            - 'volatility': Portfolio volatility
            - 'sharpe': Sharpe ratio
            - 'success': Boolean indicating convergence

    Raises:
        ValueError: If mu/cov dimensions don't match
    """
    codes = mu.index.tolist()
    n_assets = len(codes)

    if n_assets < 2:
        raise ValueError("Need at least 2 assets for optimization")
    if n_assets < 5:
        import warnings
        warnings.warn(f"Small universe ({n_assets} assets): optimizer may over-concentrate. Consider adding more stocks or using equal-weight.")

    if cov.shape[0] != n_assets or cov.shape[1] != n_assets:
        raise ValueError("mu and cov dimensions don't match")

    # Parse constraints
    if constraints is None:
        constraints = {}
    max_weight = constraints.get('max_weight', DEFAULT_MAX_WEIGHT)
    min_weight = constraints.get('min_weight', DEFAULT_MIN_WEIGHT)
    max_stocks = constraints.get('max_stocks', DEFAULT_MAX_STOCKS)

    # Regularize covariance
    cov_reg = _regularize_covariance(cov)
    mu_array = mu.values

    # Initialize: equal weight
    w = np.ones(n_assets) / n_assets

    # Gradient ascent with projected simplex constraint
    learning_rate = 0.1
    max_iter = 500
    tol = 1e-6

    for iteration in range(max_iter):
        # Compute portfolio statistics
        port_return = w @ mu_array
        port_vol = np.sqrt(w @ cov_reg @ w)

        if port_vol < 1e-8:
            port_vol = 1e-8

        sharpe = (port_return - risk_free) / port_vol

        # Gradient of Sharpe ratio w.r.t. weights
        # Sharpe = (w'mu - rf) / sqrt(w'Cov*w)
        # dSharpe/dw = (mu*sqrt(w'Cov*w) - (w'mu - rf)*Cov*w/sqrt(w'Cov*w)) / (w'Cov*w)

        term1 = mu_array * port_vol  # ∂/∂w of numerator (w'μ - rf); gradient is μ, not (μ - rf)
        term2 = (port_return - risk_free) * (cov_reg @ w) / port_vol
        gradient = (term1 - term2) / (port_vol ** 2)

        # Update weights
        w_new = w + learning_rate * gradient

        # Project to simplex
        w_new = _project_to_simplex(w_new)

        # Apply weight constraints
        w_new = _clip_weights(w_new, min_weight, max_weight)

        # Check convergence
        w_diff = np.linalg.norm(w_new - w)
        if w_diff < tol:
            w = w_new
            break

        w = w_new

    # Apply max_stocks constraint: keep top k stocks
    if max_stocks < n_assets:
        top_indices = np.argsort(w)[-max_stocks:]
        w_mask = np.zeros(n_assets, dtype=bool)
        w_mask[top_indices] = True
        w = w * w_mask
        w = w / np.sum(w) if np.sum(w) > 0 else np.ones(n_assets) / n_assets

    # Final statistics
    port_return = w @ mu_array
    port_vol = np.sqrt(w @ cov_reg @ w)
    sharpe = (port_return - risk_free) / port_vol if port_vol > 0 else 0

    weight_dict = {code: float(w[i]) for i, code in enumerate(codes)}

    return {
        'weights': weight_dict,
        'expected_return': float(port_return),
        'volatility': float(port_vol),
        'sharpe': float(sharpe),
        'success': True
    }


# ============================================================================
# Minimum Variance Optimization
# ============================================================================

def optimize_min_variance(
    cov: pd.DataFrame,
    constraints: Optional[Dict] = None
) -> Dict:
    """
    Find minimum variance portfolio.

    Solves: minimize w'*Cov*w subject to sum(w)=1, w>=0, and optional box constraints.
    Uses analytical solution when unconstrained, iterative projection for constraints.

    Args:
        cov: Covariance matrix (pd.DataFrame)
        constraints: Dict with optional keys:
            - 'max_weight': Maximum weight per stock (default 0.15)
            - 'min_weight': Minimum weight per stock (default 0.0)
            - 'max_stocks': Maximum number of stocks (default 20)

    Returns:
        Dict with keys:
            - 'weights': Dict {code: weight}
            - 'volatility': Portfolio volatility
            - 'success': Boolean
    """
    codes = cov.index.tolist()
    n_assets = len(codes)

    if n_assets < 2:
        raise ValueError("Need at least 2 assets for optimization")

    # Parse constraints
    if constraints is None:
        constraints = {}
    max_weight = constraints.get('max_weight', DEFAULT_MAX_WEIGHT)
    min_weight = constraints.get('min_weight', DEFAULT_MIN_WEIGHT)
    max_stocks = constraints.get('max_stocks', DEFAULT_MAX_STOCKS)

    # Regularize covariance
    cov_reg = _regularize_covariance(cov)

    # Try analytical solution first (unconstrained)
    try:
        cov_inv = np.linalg.inv(cov_reg)
        ones = np.ones(n_assets)
        w = cov_inv @ ones
        w = w / np.sum(w)
    except np.linalg.LinAlgError:
        # Fall back to equal weight
        w = np.ones(n_assets) / n_assets

    # Apply constraints iteratively
    for _ in range(100):
        w_old = w.copy()

        # Clip to bounds
        w = _clip_weights(w, min_weight, max_weight)

        # Re-optimize over active assets (not at bounds)
        active = (w > min_weight + 1e-8) & (w < max_weight - 1e-8)
        n_active = np.sum(active)

        if n_active >= 2:
            try:
                # Optimize only active weights
                cov_active = cov_reg[np.ix_(active, active)]
                cov_inv_active = np.linalg.inv(cov_active)
                ones_active = np.ones(n_active)
                w_active = cov_inv_active @ ones_active
                w_active = w_active / np.sum(w_active)

                w_new = np.zeros(n_assets)
                w_new[active] = w_active
                w = w_new
            except np.linalg.LinAlgError:
                pass

        if np.linalg.norm(w - w_old) < 1e-6:
            break

    # Apply max_stocks constraint
    if max_stocks < n_assets:
        top_indices = np.argsort(w)[-max_stocks:]
        w_mask = np.zeros(n_assets, dtype=bool)
        w_mask[top_indices] = True
        w = w * w_mask
        w = w / np.sum(w) if np.sum(w) > 0 else np.ones(n_assets) / n_assets

    # Final variance
    port_vol = np.sqrt(w @ cov_reg @ w)

    weight_dict = {code: float(w[i]) for i, code in enumerate(codes)}

    return {
        'weights': weight_dict,
        'volatility': float(port_vol),
        'expected_return': 0.0,  # Not computed for min-var
        'sharpe': 0.0,
        'success': True
    }


# ============================================================================
# Risk Parity Optimization
# ============================================================================

def optimize_risk_parity(
    cov: pd.DataFrame,
    budget: Optional[Dict] = None
) -> Dict:
    """
    Find risk parity portfolio where each asset contributes equally to total risk.

    Minimizes: 0.5 * w'*Cov*w - budget * sum(log(w))
    using Newton's method with iterative refinement.

    Args:
        cov: Covariance matrix (pd.DataFrame)
        budget: Optional risk budget per asset as Dict {code: proportion}.
               Defaults to equal budget (1/n for each asset).

    Returns:
        Dict with keys:
            - 'weights': Dict {code: weight}
            - 'volatility': Portfolio volatility
            - 'risk_contributions': Dict {code: risk_contribution}
    """
    codes = cov.index.tolist()
    n_assets = len(codes)

    if n_assets < 2:
        raise ValueError("Need at least 2 assets for optimization")

    # Parse budget
    if budget is None:
        budget = {code: 1.0 / n_assets for code in codes}
    budget_array = np.array([budget.get(code, 1.0 / n_assets) for code in codes])
    budget_array = budget_array / np.sum(budget_array)  # Normalize

    # Regularize covariance
    cov_reg = _regularize_covariance(cov)

    # Initialize: equal weight
    w = np.ones(n_assets) / n_assets

    # Newton's method
    for iteration in range(100):
        w_old = w.copy()

        # Gradient of objective: Cov*w - budget / w
        grad = cov_reg @ w - budget_array / (w + 1e-10)

        # Hessian: Cov + diag(budget / w^2)
        hessian = cov_reg + np.diag(budget_array / (w ** 2 + 1e-10))

        # Newton step
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            # Fallback to gradient step
            step = 0.01 * grad

        # Update with line search
        alpha = 1.0
        for _ in range(10):
            w_new = w - alpha * step
            # Project to simplex
            w_new = _project_to_simplex(w_new)

            # Check for improvement
            obj_old = 0.5 * w @ cov_reg @ w - np.sum(budget_array * np.log(w + 1e-10))
            obj_new = 0.5 * w_new @ cov_reg @ w_new - np.sum(budget_array * np.log(w_new + 1e-10))

            if obj_new < obj_old:
                w = w_new
                break
            alpha *= 0.5
        else:
            # No improvement, use small step
            w = _project_to_simplex(w - 0.01 * grad)

        # Check convergence
        if np.linalg.norm(w - w_old) < 1e-6:
            break

    # Compute risk contributions
    w_cov = cov_reg @ w
    marginal_risk = w_cov
    total_risk = np.sqrt(w @ w_cov)
    risk_contrib = w * marginal_risk / (total_risk + 1e-10)
    risk_contrib_pct = risk_contrib / np.sum(risk_contrib)

    weight_dict = {code: float(w[i]) for i, code in enumerate(codes)}
    risk_contrib_dict = {code: float(risk_contrib_pct[i]) for i, code in enumerate(codes)}

    return {
        'weights': weight_dict,
        'volatility': float(total_risk),
        'expected_return': 0.0,
        'sharpe': 0.0,
        'risk_contributions': risk_contrib_dict,
        'success': True
    }


# ============================================================================
# Simple Portfolios
# ============================================================================

def optimize_equal_weight(codes: List[str]) -> Dict:
    """
    Create simple 1/N equally-weighted portfolio.

    Args:
        codes: List of stock codes

    Returns:
        Dict with weights and basic metrics
    """
    if not codes:
        raise ValueError("codes list cannot be empty")

    n = len(codes)
    weight = 1.0 / n
    weight_dict = {code: weight for code in codes}

    return {
        'weights': weight_dict,
        'expected_return': 0.0,
        'volatility': 0.0,
        'sharpe': 0.0,
        'success': True
    }


# ============================================================================
# Black-Litterman Model
# ============================================================================

def black_litterman(
    mu_mkt: pd.Series,
    cov: pd.DataFrame,
    views: List[Dict],
    tau: float = 0.05
) -> pd.Series:
    """
    Apply simplified Black-Litterman model to adjust expected returns based on views.

    Args:
        mu_mkt: Market-implied expected returns (from equilibrium or historical)
        cov: Covariance matrix
        views: List of view dicts, each with:
            - 'assets': List of asset codes involved in view
            - 'direction': 1 (bullish) or -1 (bearish)
            - 'magnitude': Expected out/underperformance (e.g., 0.05 = 5%)
            - 'confidence': Confidence level 0-1 (0 = no confidence, 1 = certain)
        tau: Uncertainty parameter (0.05 means 5% uncertainty on market view)

    Returns:
        pd.Series of adjusted expected returns

    Notes:
        Simplified implementation without full covariance of views.
        Just adjusts returns based on view weights and confidence.
    """
    codes = mu_mkt.index.tolist()
    n_assets = len(codes)

    if not views:
        return mu_mkt.copy()

    # Start with market view
    mu_adjusted = mu_mkt.copy()

    for view in views:
        assets = view.get('assets', [])
        direction = view.get('direction', 1)
        magnitude = view.get('magnitude', 0.0)
        confidence = view.get('confidence', 0.8)

        if not assets or confidence < 1e-8:
            continue

        # Find indices of assets in view
        indices = [codes.index(asset) if asset in codes else -1 for asset in assets]
        indices = [i for i in indices if i >= 0]

        if not indices:
            continue

        # Adjust returns for these assets
        adjustment = direction * magnitude * confidence
        for idx in indices:
            mu_adjusted.iloc[idx] += adjustment

    return mu_adjusted


# ============================================================================
# Portfolio Analysis
# ============================================================================

def portfolio_stats(
    weights: Dict[str, float],
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free: float = RISK_FREE_RATE
) -> Dict:
    """
    Compute comprehensive portfolio statistics.

    Args:
        weights: Dict {code: weight}
        mu: Expected returns (pd.Series)
        cov: Covariance matrix
        risk_free: Risk-free rate

    Returns:
        Dict with keys:
            - 'return': Portfolio expected return
            - 'volatility': Portfolio volatility
            - 'sharpe': Sharpe ratio
            - 'n_stocks': Number of stocks
            - 'max_weight': Largest individual weight
            - 'hhi': Herfindahl-Hirschman Index (concentration)
            - 'diversification_ratio': Diversification ratio
    """
    codes = list(weights.keys())
    w = np.array([weights[code] for code in codes])
    mu_array = mu[codes].values
    cov_array = cov.loc[codes, codes].values

    # Basic stats
    port_return = w @ mu_array
    port_var = w @ cov_array @ w
    port_vol = np.sqrt(max(port_var, 0))
    sharpe = (port_return - risk_free) / port_vol if port_vol > 1e-10 else 0

    # Concentration metrics
    n_stocks = np.sum(w > 1e-6)
    max_weight = np.max(w)
    hhi = np.sum(w ** 2)  # Herfindahl-Hirschman Index

    # Diversification ratio: sum of weighted vols / portfolio vol
    individual_vols = np.sqrt(np.diag(cov_array))
    div_ratio = (w @ individual_vols) / port_vol if port_vol > 1e-10 else 1.0

    return {
        'return': float(port_return),
        'volatility': float(port_vol),
        'sharpe': float(sharpe),
        'n_stocks': int(n_stocks),
        'max_weight': float(max_weight),
        'hhi': float(hhi),
        'diversification_ratio': float(div_ratio)
    }


def portfolio_risk_decomposition(
    weights: Dict[str, float],
    cov: pd.DataFrame
) -> Dict[str, Dict]:
    """
    Decompose portfolio risk by asset (marginal and percentage contributions).

    Args:
        weights: Dict {code: weight}
        cov: Covariance matrix

    Returns:
        Dict {code: {'marginal_risk': float, 'risk_pct': float, 'contribution': float}}
    """
    codes = list(weights.keys())
    w = np.array([weights[code] for code in codes])
    cov_array = cov.loc[codes, codes].values

    # Marginal risk contribution: (Cov*w)_i
    w_cov = cov_array @ w

    # Portfolio risk
    port_var = w @ w_cov
    port_risk = np.sqrt(max(port_var, 1e-10))

    # Risk contribution: w_i * (Cov*w)_i / portfolio_risk
    risk_contrib = w * w_cov / port_risk

    # Percentage of total risk
    risk_pct = risk_contrib / np.sum(risk_contrib) if np.sum(risk_contrib) > 1e-10 else np.zeros_like(risk_contrib)

    result = {}
    for i, code in enumerate(codes):
        result[code] = {
            'marginal_risk': float(w_cov[i]),
            'risk_pct': float(risk_pct[i]),
            'contribution': float(risk_contrib[i])
        }

    return result


def efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_points: int = 20,
    constraints: Optional[Dict] = None
) -> List[Dict]:
    """
    Generate efficient frontier by varying target return.

    Args:
        mu: Expected returns
        cov: Covariance matrix
        n_points: Number of points along frontier
        constraints: Optional portfolio constraints

    Returns:
        List of portfolio dicts [{'return': r, 'volatility': v, 'weights': {...}}, ...]
        Sorted by return ascending
    """
    if constraints is None:
        constraints = {}

    min_ret = np.min(mu.values)
    max_ret = np.max(mu.values)

    # Generate target returns
    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier = []

    for target_ret in target_returns:
        # For each target return, find min-variance portfolio
        # This is a simplified approach: use mean variance with penalty

        codes = mu.index.tolist()
        n_assets = len(codes)
        w = np.ones(n_assets) / n_assets  # Start from equal weight

        cov_reg = _regularize_covariance(cov)
        mu_array = mu.values

        # Minimize variance subject to expected return constraint
        # Using gradient descent
        for _ in range(100):
            # Gradient of variance
            grad_var = 2 * cov_reg @ w

            # Return constraint: w'*mu = target_ret
            # Add penalty if constraint violated
            current_ret = w @ mu_array
            constraint_violation = current_ret - target_ret

            # Penalized gradient
            grad_penalty = constraint_violation * mu_array

            grad = grad_var + 10 * grad_penalty

            w_new = w - 0.01 * grad
            w_new = _project_to_simplex(w_new)

            if np.linalg.norm(w_new - w) < 1e-5:
                w = w_new
                break

            w = w_new

        # Get final stats
        port_return = w @ mu_array
        port_vol = np.sqrt(w @ cov_reg @ w)
        weight_dict = {code: float(w[i]) for i, code in enumerate(codes)}

        frontier.append({
            'return': float(port_return),
            'volatility': float(port_vol),
            'weights': weight_dict
        })

    # Sort by return
    frontier.sort(key=lambda x: x['return'])

    return frontier


# ============================================================================
# Position Sizing & Rebalancing
# ============================================================================

def position_sizing(
    weights: Dict[str, float],
    total_capital: float,
    prices: Dict[str, float],
    lot_size: int = LOT_SIZE
) -> Dict[str, Dict]:
    """
    Convert portfolio weights to actual share counts respecting lot size requirements.

    A-shares trade in lots of 100 shares minimum.

    Args:
        weights: Dict {code: weight}
        total_capital: Total capital in CNY
        prices: Dict {code: current_price}
        lot_size: Minimum lot size (default 100)

    Returns:
        Dict {code: {'shares': int, 'value': float, 'actual_weight': float}}
    """
    result = {}
    total_allocated = 0.0
    positions = []

    # First pass: compute ideal shares
    for code, weight in weights.items():
        if weight < 1e-10:
            continue

        target_value = weight * total_capital
        price = prices.get(code, 0)

        if price <= 0:
            continue

        # Round to lot size
        ideal_shares = target_value / price
        shares = int(ideal_shares / lot_size) * lot_size

        if shares > 0:
            value = shares * price
            positions.append((code, shares, value, weight))
            total_allocated += value

    # Second pass: allocate remaining capital to largest positions
    remaining_capital = total_capital - total_allocated

    positions_by_weight = sorted(positions, key=lambda x: x[3], reverse=True)

    for code, shares, value, weight in positions_by_weight:
        if remaining_capital <= 1e-2:
            break

        price = prices[code]
        additional_shares = int(remaining_capital / price / lot_size) * lot_size

        if additional_shares > 0:
            result[code] = {
                'shares': shares + additional_shares,
                'value': (shares + additional_shares) * price,
                'actual_weight': ((shares + additional_shares) * price) / total_capital
            }
            remaining_capital -= additional_shares * price
        else:
            result[code] = {
                'shares': shares,
                'value': value,
                'actual_weight': value / total_capital
            }

    # Add positions with no additional shares
    for code, shares, value, _ in positions:
        if code not in result:
            result[code] = {
                'shares': shares,
                'value': value,
                'actual_weight': value / total_capital
            }

    return result


def rebalance_plan(
    current_holdings: Dict[str, int],
    target_weights: Dict[str, float],
    prices: Dict[str, float],
    total_capital: float
) -> List[Dict]:
    """
    Generate buy/sell orders to rebalance from current holdings to target.

    Includes transaction costs (commission + stamp tax).
    Respects lot size constraint.

    Args:
        current_holdings: Dict {code: current_shares}
        target_weights: Dict {code: target_weight}
        prices: Dict {code: current_price}
        total_capital: Total portfolio capital

    Returns:
        List of orders: [{'code': str, 'action': 'buy'|'sell', 'shares': int, 'value': float}, ...]
    """
    orders = []

    # Get target shares
    target_shares = position_sizing(target_weights, total_capital, prices)

    all_codes = set(current_holdings.keys()) | set(target_shares.keys())

    for code in all_codes:
        current_shares = current_holdings.get(code, 0)
        target = target_shares.get(code, {})
        target_shares_count = target.get('shares', 0)

        diff = target_shares_count - current_shares
        price = prices.get(code, 0)

        if abs(diff) < 1:  # No meaningful change
            continue

        if diff > 0:
            # Buy
            gross_value = diff * price
            commission = max(gross_value * COMMISSION_RATE, MIN_COMMISSION)
            net_value = gross_value + commission

            orders.append({
                'code': code,
                'action': 'buy',
                'shares': diff,
                'value': float(net_value),
                'gross_value': float(gross_value),
                'commission': float(commission)
            })
        else:
            # Sell
            shares_to_sell = -diff
            gross_value = shares_to_sell * price
            commission = max(gross_value * COMMISSION_RATE, MIN_COMMISSION)
            stamp_tax = gross_value * STAMP_TAX_RATE
            net_value = gross_value - commission - stamp_tax

            orders.append({
                'code': code,
                'action': 'sell',
                'shares': shares_to_sell,
                'value': float(net_value),
                'gross_value': float(gross_value),
                'commission': float(commission),
                'stamp_tax': float(stamp_tax)
            })

    return orders


# ============================================================================
# Turnover Management
# ============================================================================

def compute_turnover(
    new_weights: Dict[str, float],
    current_weights: Dict[str, float],
) -> float:
    """
    Compute L1 turnover between two weight dicts.
    Returns a value in [0, 2]: 0 = no change, 1 = full portfolio rotated once.
    Round-trip cost at 0.15% per trade: turnover × 0.075% ≈ trading cost fraction.
    """
    all_codes = set(new_weights) | set(current_weights)
    return sum(
        abs(new_weights.get(c, 0.0) - current_weights.get(c, 0.0))
        for c in all_codes
    )


def apply_turnover_constraint(
    new_weights: Dict[str, float],
    current_weights: Dict[str, float],
    turnover_limit: float = 0.30,
) -> Dict[str, float]:
    """
    Blend new_weights toward current_weights to cap L1 turnover.

    A-share round-trip cost is ~0.15% (commission 0.025%×2 + stamp tax 0.1%).
    Unconstrained optimizers routinely generate 40-60% daily turnover, erasing
    ~6-9bps of alpha per rebalance. Capping at turnover_limit=0.30 limits trading
    friction to ~4.5bps while preserving most of the optimizer signal.

    Args:
        new_weights: Target weights from optimizer {code: weight}
        current_weights: Current holdings weights {code: weight}
        turnover_limit: Max L1 turnover (sum |w_new - w_old|). Default 0.30 = 30%

    Returns:
        Adjusted weights respecting turnover_limit with sum = 1.
        Returns new_weights unchanged when current turnover is already within limit.
    """
    turnover = compute_turnover(new_weights, current_weights)

    if turnover <= turnover_limit:
        return dict(new_weights)

    # Linear blend: alpha × new + (1 - alpha) × current achieves exactly turnover_limit
    alpha = turnover_limit / turnover  # 0 < alpha < 1

    all_codes = set(new_weights) | set(current_weights)
    blended: Dict[str, float] = {}
    for code in all_codes:
        w_new = new_weights.get(code, 0.0)
        w_cur = current_weights.get(code, 0.0)
        blended[code] = alpha * w_new + (1.0 - alpha) * w_cur

    # Drop dust positions (< 0.1%) and renormalize
    blended = {c: w for c, w in blended.items() if w > 0.001}
    total = sum(blended.values())
    if total > 0:
        blended = {c: w / total for c, w in blended.items()}

    return blended


# ============================================================================
# Utilities & Validation
# ============================================================================

def validate_weights(weights: Dict[str, float], tolerance: float = 1e-6) -> bool:
    """
    Check if weights sum to 1 and are non-negative.

    Args:
        weights: Dict {code: weight}
        tolerance: Tolerance for sum check

    Returns:
        Boolean indicating validity
    """
    total = sum(weights.values())
    return (
        all(w >= -1e-10 for w in weights.values()) and
        abs(total - 1.0) < tolerance
    )


def get_portfolio_metadata(
    weights: Dict[str, float],
    mu: Optional[pd.Series] = None,
    cov: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Get metadata about a portfolio (number of stocks, concentration, etc).

    Args:
        weights: Dict {code: weight}
        mu: Optional expected returns for stats
        cov: Optional covariance for stats

    Returns:
        Dict with portfolio metadata
    """
    codes = list(weights.keys())
    w = np.array([weights[code] for code in codes])

    n_stocks = np.sum(w > 1e-8)
    max_weight = np.max(w)
    hhi = np.sum(w ** 2)
    top_3 = np.sum(np.sort(w)[-min(3, len(w)):])

    return {
        'n_stocks': int(n_stocks),
        'max_weight': float(max_weight),
        'concentration_hhi': float(hhi),
        'top_3_concentration': float(top_3),
        'is_valid': validate_weights(weights)
    }
