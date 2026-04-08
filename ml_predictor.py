"""
Machine learning prediction module for Chinese A-shares.
Implements feature engineering, decision trees, and gradient boosting from scratch.
Uses ONLY numpy and pandas (no sklearn, scipy, xgboost, or lightgbm).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Feature names constant (in order)
_FEATURE_NAMES = [
    # Price-based (20 features)
    'return_1d', 'return_5d', 'return_10d', 'return_20d', 'return_60d',
    'ma5_ratio', 'ma10_ratio', 'ma20_ratio', 'ma60_ratio',
    'bollinger_pb_20',
    'rsi_6', 'rsi_14', 'rsi_24',
    'macd_histogram', 'macd_signal',
    'williams_r_14',
    'roc_5d', 'roc_10d', 'roc_20d',
    'atr_pct',
    # Volume-based (10 features)
    'vol_ratio_5', 'vol_ratio_20',
    'obv_slope_5', 'obv_slope_20',
    'vwap_deviation',
    'turnover_zscore',
    'volume_trend',
    'price_vol_corr',
    'up_day_vol_ratio',
    'mfi',
    'ad_line',
    # Volatility-based (8 features)
    'volatility_5d', 'volatility_20d', 'volatility_60d',
    'volatility_ratio',
    'intraday_range_avg',
    'gk_volatility',
    'max_drawdown_20d',
    'skewness_20d',
    # Pattern-based (8 features)
    'consecutive_days_count',
    'gap_count_20d',
    'candle_body_ratio_5d',
    'upper_shadow_ratio_5d',
    'lower_shadow_ratio_5d',
    'distance_to_high',
    'distance_to_low',
    'trend_strength',
    'hurst_exponent',
    # Fundamental (4 features, optional)
    'pe_percentile',
    'pb_percentile',
    'market_cap_log',
    'turnover_rate_20d',
]


def _safe_divide(numerator, denominator, fill_value=0):
    """Safely divide, handling division by zero."""
    result = np.divide(numerator, denominator, where=denominator != 0,
                       out=np.full_like(numerator, fill_value, dtype=float))
    return result


def _safe_log(x, fill_value=0):
    """Safely compute log, handling non-positive values."""
    x = np.asarray(x)
    result = np.full_like(x, fill_value, dtype=float)
    mask = x > 0
    result[mask] = np.log(x[mask])
    return result


def extract_features(df: pd.DataFrame, fundamentals: dict = None) -> pd.DataFrame:
    """
    Extract ML features from a stock's history DataFrame.

    Expected columns in df: open, high, low, close, volume, amount (or turnover)

    Returns DataFrame with one row per trading day, columns = features.
    """
    df = df.copy()

    # Ensure required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Reset index to ensure clean numbering
    df.reset_index(drop=True, inplace=True)

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_ = df['open'].values
    volume = df['volume'].values

    features = pd.DataFrame(index=df.index)

    # ==================== Price-based features (20) ====================

    # Returns
    returns = np.diff(close, prepend=close[0]) / close
    returns[0] = 0

    features['return_1d'] = returns

    for period in [5, 10, 20, 60]:
        ret = np.zeros_like(close)
        for i in range(period, len(close)):
            ret[i] = (close[i] - close[i - period]) / close[i - period]
        features[f'return_{period}d'] = ret

    # Moving averages ratios
    for period in [5, 10, 20, 60]:
        ma = _compute_ma(close, period)
        features[f'ma{period}_ratio'] = close / (ma + 1e-8)

    # Bollinger Bands %B (20d)
    bb_pb = _compute_bollinger_pb(close, period=20)
    features['bollinger_pb_20'] = bb_pb

    # RSI
    for period in [6, 14, 24]:
        rsi = _compute_rsi(close, period)
        features[f'rsi_{period}'] = rsi

    # MACD
    macd_h, macd_s = _compute_macd(close)
    features['macd_histogram'] = macd_h
    features['macd_signal'] = macd_s

    # Williams %R(14)
    williams_r = _compute_williams_r(high, low, close, period=14)
    features['williams_r_14'] = williams_r

    # Rate of Change
    for period in [5, 10, 20]:
        roc = np.zeros_like(close)
        for i in range(period, len(close)):
            roc[i] = (close[i] - close[i - period]) / close[i - period]
        features[f'roc_{period}d'] = roc

    # ATR as % of close
    atr = _compute_atr(high, low, close, period=14)
    features['atr_pct'] = atr / (close + 1e-8)

    # ==================== Volume-based features (10) ====================

    # Volume ratios
    vol_ma5 = _compute_ma(volume, 5)
    vol_ma20 = _compute_ma(volume, 20)
    features['vol_ratio_5'] = volume / (vol_ma5 + 1e-8)
    features['vol_ratio_20'] = volume / (vol_ma20 + 1e-8)

    # OBV slope
    obv = _compute_obv(close, volume)
    obv_slope_5 = _compute_ma_slope(obv, 5)
    obv_slope_20 = _compute_ma_slope(obv, 20)
    features['obv_slope_5'] = obv_slope_5
    features['obv_slope_20'] = obv_slope_20

    # VWAP deviation
    vwap_dev = _compute_vwap_deviation(high, low, close, volume)
    features['vwap_deviation'] = vwap_dev

    # Turnover rate z-score
    if 'amount' in df.columns:
        amount = df['amount'].values
    else:
        amount = close * volume

    turnover = volume / (np.sum(volume) + 1e-8)  # Simple approximation
    turnover_ma20 = _compute_ma(turnover, 20)
    turnover_std = _compute_std(turnover, 20)
    turnover_zscore = (turnover - turnover_ma20) / (turnover_std + 1e-8)
    features['turnover_zscore'] = turnover_zscore

    # Volume trend
    vol_trend = np.zeros_like(volume)
    for i in range(1, len(volume)):
        if volume[i] > vol_ma5[i]:
            vol_trend[i] = 1
        elif volume[i] < vol_ma5[i]:
            vol_trend[i] = -1
    features['volume_trend'] = vol_trend

    # Price-volume correlation (20d)
    price_vol_corr = _compute_correlation(returns, volume, 20)
    features['price_vol_corr'] = price_vol_corr

    # Up-day volume ratio
    up_vol_ratio = _compute_up_day_volume_ratio(close, volume)
    features['up_day_vol_ratio'] = up_vol_ratio

    # MFI (Money Flow Index)
    mfi = _compute_mfi(high, low, close, volume, period=14)
    features['mfi'] = mfi

    # Accumulation/Distribution
    ad = _compute_ad_line(high, low, close, volume)
    features['ad_line'] = ad

    # ==================== Volatility-based features (8) ====================

    for period in [5, 20, 60]:
        vol = _compute_volatility(close, period)
        features[f'volatility_{period}d'] = vol

    vol_5 = _compute_volatility(close, 5)
    vol_60 = _compute_volatility(close, 60)
    vol_ratio = vol_5 / (vol_60 + 1e-8)
    features['volatility_ratio'] = vol_ratio

    # Intraday range average
    intraday_range = (high - low) / (close + 1e-8)
    intraday_range_avg = _compute_ma(intraday_range, 5)
    features['intraday_range_avg'] = intraday_range_avg

    # Garman-Klass volatility
    gk_vol = _compute_gk_volatility(high, low, close)
    features['gk_volatility'] = gk_vol

    # Max drawdown (20d)
    max_dd = _compute_max_drawdown(close, 20)
    features['max_drawdown_20d'] = max_dd

    # Skewness of returns (20d)
    skewness = _compute_skewness(close, 20)
    features['skewness_20d'] = skewness

    # ==================== Pattern-based features (8) ====================

    # Consecutive up/down days
    consecutive = _compute_consecutive_days(close)
    features['consecutive_days_count'] = consecutive

    # Gap count (last 20d)
    gap_count = _compute_gap_count(open_, close, 20)
    features['gap_count_20d'] = gap_count

    # Candlestick patterns (5d avg)
    body_ratio = (np.abs(close - open_)) / (high - low + 1e-8)
    features['candle_body_ratio_5d'] = _compute_ma(body_ratio, 5)

    upper_shadow = (high - np.maximum(open_, close)) / (high - low + 1e-8)
    features['upper_shadow_ratio_5d'] = _compute_ma(upper_shadow, 5)

    lower_shadow = (np.minimum(open_, close) - low) / (high - low + 1e-8)
    features['lower_shadow_ratio_5d'] = _compute_ma(lower_shadow, 5)

    # Distance to 20d high/low
    high_20 = _compute_rolling_max(high, 20)
    low_20 = _compute_rolling_min(low, 20)
    features['distance_to_high'] = (high_20 - close) / (close + 1e-8)
    features['distance_to_low'] = (close - low_20) / (close + 1e-8)

    # Trend strength (R² of 20d linear regression)
    trend_strength = _compute_trend_strength(close, 20)
    features['trend_strength'] = trend_strength

    # Hurst exponent (simplified)
    hurst = _compute_hurst_exponent(close, 60)
    features['hurst_exponent'] = hurst

    # ==================== Fundamental features (4, optional) ====================

    if fundamentals:
        # PE TTM percentile
        if 'pe_ttm' in fundamentals and isinstance(fundamentals['pe_ttm'], (list, np.ndarray)):
            pe_percentile = _compute_percentile(fundamentals['pe_ttm'][-1], fundamentals['pe_ttm'])
        else:
            pe_percentile = 0.5
        features['pe_percentile'] = pe_percentile

        # PB percentile
        if 'pb' in fundamentals and isinstance(fundamentals['pb'], (list, np.ndarray)):
            pb_percentile = _compute_percentile(fundamentals['pb'][-1], fundamentals['pb'])
        else:
            pb_percentile = 0.5
        features['pb_percentile'] = pb_percentile

        # Market cap log
        if 'market_cap' in fundamentals:
            mc_log = _safe_log(fundamentals['market_cap'], fill_value=0)
            features['market_cap_log'] = mc_log
        else:
            features['market_cap_log'] = 0

        # Turnover rate (20d avg)
        if 'turnover_rate' in fundamentals and isinstance(fundamentals['turnover_rate'], (list, np.ndarray)):
            features['turnover_rate_20d'] = np.mean(fundamentals['turnover_rate'][-20:]) if len(fundamentals['turnover_rate']) >= 20 else np.mean(fundamentals['turnover_rate'])
        else:
            features['turnover_rate_20d'] = 0
    else:
        features['pe_percentile'] = 0
        features['pb_percentile'] = 0
        features['market_cap_log'] = 0
        features['turnover_rate_20d'] = 0

    # Fill NaN values with column median
    for col in features.columns:
        if features[col].isna().any():
            features[col].fillna(features[col].median(), inplace=True)

    # Replace inf with 0
    features = features.replace([np.inf, -np.inf], 0)

    return features


def generate_labels(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.02,
                   binary: bool = False) -> pd.Series:
    """
    Generate classification labels based on forward returns.

    Args:
        df: DataFrame with 'close' column
        horizon: days ahead to predict
        threshold: threshold for strong/weak classification
        binary: if True, return binary (0/1); if False, return multi-class (-2 to 2)

    Returns:
        Series of labels
    """
    close = df['close'].values
    labels = np.zeros(len(close), dtype=int)

    for i in range(len(close) - horizon):
        future_return = (close[i + horizon] - close[i]) / close[i]

        if binary:
            labels[i] = 1 if future_return > 0 else 0
        else:
            if future_return > 2 * threshold:
                labels[i] = 2
            elif future_return > threshold:
                labels[i] = 1
            elif future_return < -2 * threshold:
                labels[i] = -2
            elif future_return < -threshold:
                labels[i] = -1
            else:
                labels[i] = 0

    # Last horizon rows get label 0 (can't predict forward from them)
    labels[-horizon:] = 0

    return pd.Series(labels, index=df.index)


# ==================== Helper functions for feature computation ====================

def _compute_ma(arr, period):
    """Compute simple moving average."""
    ma = np.zeros_like(arr)
    cumsum = np.cumsum(arr)
    for i in range(len(arr)):
        if i < period:
            ma[i] = np.mean(arr[:i + 1]) if i > 0 else arr[0]
        else:
            ma[i] = (cumsum[i] - cumsum[i - period]) / period
    return ma


def _compute_std(arr, period):
    """Compute rolling standard deviation."""
    std = np.zeros_like(arr)
    for i in range(len(arr)):
        if i < period:
            std[i] = np.std(arr[:i + 1]) if i > 0 else 0
        else:
            std[i] = np.std(arr[i - period + 1:i + 1])
    return std


def _compute_rsi(close, period=14):
    """Compute Relative Strength Index."""
    rsi = np.zeros_like(close)
    deltas = np.diff(close, prepend=close[0])

    seed = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)

    for i in range(period, len(close)):
        avg_gain[i] = np.mean(seed[i - period + 1:i + 1])
        avg_loss[i] = np.mean(loss[i - period + 1:i + 1])

    rs = _safe_divide(avg_gain, avg_loss, fill_value=0)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def _compute_macd(close, fast=12, slow=26, signal=9):
    """Compute MACD."""
    ma_fast = _compute_ma(close, fast)
    ma_slow = _compute_ma(close, slow)
    macd_line = ma_fast - ma_slow
    macd_signal = _compute_ma(macd_line, signal)
    macd_histogram = macd_line - macd_signal

    return macd_histogram, macd_signal


def _compute_williams_r(high, low, close, period=14):
    """Compute Williams %R."""
    williams_r = np.zeros_like(close)

    for i in range(period, len(close)):
        h_max = np.max(high[max(0, i - period + 1):i + 1])
        l_min = np.min(low[max(0, i - period + 1):i + 1])
        if h_max != l_min:
            williams_r[i] = ((close[i] - h_max) / (h_max - l_min)) * -100

    return williams_r


def _compute_atr(high, low, close, period=14):
    """Compute Average True Range."""
    tr = np.zeros_like(close)
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))

    atr = _compute_ma(tr, period)
    return atr


def _compute_bollinger_pb(close, period=20):
    """Compute Bollinger Bands %B."""
    ma = _compute_ma(close, period)
    std = _compute_std(close, period)

    upper = ma + 2 * std
    lower = ma - 2 * std

    pb = _safe_divide(close - lower, upper - lower, fill_value=0.5)
    return np.clip(pb, 0, 1)


def _compute_obv(close, volume):
    """Compute On-Balance Volume."""
    obv = np.zeros_like(close)
    obv[0] = volume[0]

    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]

    return obv


def _compute_ma_slope(arr, period):
    """Compute slope of moving average."""
    ma = _compute_ma(arr, period)
    slope = np.zeros_like(arr)

    for i in range(period, len(arr)):
        y_vals = ma[i - period + 1:i + 1]
        x_vals = np.arange(period)
        if len(x_vals) > 1:
            slope[i] = np.polyfit(x_vals, y_vals, 1)[0]

    return slope


def _compute_vwap_deviation(high, low, close, volume):
    """Compute deviation from VWAP."""
    typical_price = (high + low + close) / 3
    vwap_cum_numerator = np.cumsum(typical_price * volume)
    vwap_cum_denominator = np.cumsum(volume)
    vwap = _safe_divide(vwap_cum_numerator, vwap_cum_denominator, fill_value=close[0])

    deviation = _safe_divide(close - vwap, vwap, fill_value=0)
    return deviation


def _compute_correlation(x, y, period):
    """Compute rolling correlation."""
    corr = np.zeros_like(x)

    for i in range(period, len(x)):
        x_window = x[i - period + 1:i + 1]
        y_window = y[i - period + 1:i + 1]

        if len(x_window) > 1:
            corr[i] = np.corrcoef(x_window, y_window)[0, 1]
        else:
            corr[i] = 0

    return np.nan_to_num(corr, 0)


def _compute_up_day_volume_ratio(close, volume):
    """Compute ratio of volume on up days vs total volume."""
    up_vol = np.zeros_like(volume)
    total_vol = _compute_ma(volume, 20)

    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            up_vol[i] = volume[i]

    up_vol_ma = _compute_ma(up_vol, 20)
    ratio = _safe_divide(up_vol_ma, total_vol, fill_value=0.5)

    return ratio


def _compute_mfi(high, low, close, volume, period=14):
    """Compute Money Flow Index."""
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume

    positive_flow = np.zeros_like(typical_price)
    negative_flow = np.zeros_like(typical_price)

    for i in range(1, len(typical_price)):
        if typical_price[i] > typical_price[i - 1]:
            positive_flow[i] = raw_money_flow[i]
        else:
            negative_flow[i] = raw_money_flow[i]

    pos_mf = _compute_ma(positive_flow, period)
    neg_mf = _compute_ma(negative_flow, period)

    mfr = _safe_divide(pos_mf, neg_mf, fill_value=1)
    mfi = 100 - (100 / (1 + mfr))

    return mfi


def _compute_ad_line(high, low, close, volume):
    """Compute Accumulation/Distribution Line."""
    ad = np.zeros_like(close)
    ad[0] = 0

    for i in range(1, len(close)):
        clv = _safe_divide(2 * close[i] - high[i] - low[i],
                          high[i] - low[i], fill_value=0)
        ad[i] = ad[i - 1] + clv * volume[i]

    return ad


def _compute_volatility(close, period):
    """Compute realized volatility (std of returns)."""
    returns = np.diff(close) / close[:-1]

    vol = np.zeros(len(close))
    vol[0] = 0

    for i in range(period, len(close)):
        vol[i] = np.std(returns[i - period:i])

    return vol


def _compute_gk_volatility(high, low, close, period=14):
    """Compute Garman-Klass volatility."""
    log_hl = np.log(high / low)
    log_cc = np.log(close / np.roll(close, 1))

    gk = np.zeros_like(close)
    for i in range(period, len(close)):
        hl_part = np.sum(log_hl[i - period:i] ** 2) / (2 * period)
        cc_part = np.sum((2 * np.log(2) - 1) * log_cc[i - period:i] ** 2) / period
        gk[i] = np.sqrt(max(0, hl_part - cc_part))

    return gk


def _compute_max_drawdown(close, period):
    """Compute maximum drawdown over period."""
    drawdown = np.zeros_like(close)

    for i in range(period, len(close)):
        running_max = np.max(close[i - period:i + 1])
        dd = (close[i] - running_max) / running_max
        drawdown[i] = dd

    return drawdown


def _compute_skewness(close, period):
    """Compute skewness of returns."""
    returns = np.diff(close) / close[:-1]

    skewness = np.zeros(len(close))
    skewness[0] = 0

    for i in range(period, len(close)):
        r = returns[i - period:i]
        mean_r = np.mean(r)
        std_r = np.std(r)
        if std_r > 0:
            skewness[i] = np.mean(((r - mean_r) / std_r) ** 3)

    return skewness


def _compute_consecutive_days(close):
    """Compute consecutive up/down days."""
    consecutive = np.zeros_like(close)

    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            consecutive[i] = max(0, consecutive[i - 1]) + 1
        elif close[i] < close[i - 1]:
            consecutive[i] = min(0, consecutive[i - 1]) - 1
        else:
            consecutive[i] = 0

    return consecutive


def _compute_gap_count(open_, close, period):
    """Count price gaps in period."""
    gap_count = np.zeros_like(open_)

    for i in range(period, len(open_)):
        gaps = 0
        for j in range(i - period + 1, i + 1):
            if j > 0:
                gap = abs(open_[j] - close[j - 1]) / close[j - 1]
                if gap > 0.01:  # >1% gap
                    gaps += 1
        gap_count[i] = gaps

    return gap_count


def _compute_rolling_max(arr, period):
    """Compute rolling maximum."""
    rolling_max = np.zeros_like(arr)

    for i in range(len(arr)):
        start = max(0, i - period + 1)
        rolling_max[i] = np.max(arr[start:i + 1])

    return rolling_max


def _compute_rolling_min(arr, period):
    """Compute rolling minimum."""
    rolling_min = np.zeros_like(arr)

    for i in range(len(arr)):
        start = max(0, i - period + 1)
        rolling_min[i] = np.min(arr[start:i + 1])

    return rolling_min


def _compute_trend_strength(close, period):
    """Compute R² of linear regression."""
    trend_strength = np.zeros_like(close)

    for i in range(period, len(close)):
        y = close[i - period + 1:i + 1]
        x = np.arange(period)

        if len(x) > 1:
            coeffs = np.polyfit(x, y, 1)
            y_pred = np.polyval(coeffs, x)

            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)

            r_squared = 1 - (ss_res / (ss_tot + 1e-8))
            trend_strength[i] = max(0, r_squared)

    return trend_strength


def _compute_hurst_exponent(close, period=60):
    """Compute simplified Hurst exponent."""
    returns = np.diff(np.log(close))
    hurst = np.zeros_like(close)

    for i in range(period, len(close)):
        r = returns[i - period + 1:i + 1]
        mean_r = np.mean(r)

        # Compute mean absolute deviation from mean
        Y = np.cumsum(r - mean_r)
        R = np.max(Y) - np.min(Y)
        S = np.std(r, ddof=1)

        if S > 0:
            hurst[i] = R / S
        else:
            hurst[i] = 0.5

    return hurst


def _compute_percentile(value, arr):
    """Compute percentile rank of value in arr."""
    if len(arr) == 0:
        return 0.5
    arr = np.asarray(arr)
    return np.mean(arr <= value)


# ==================== Decision Tree ====================

class Node:
    """Decision tree node."""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None,
                 value=None, samples=None, impurity=None):
        self.feature_idx = feature_idx      # Index of splitting feature
        self.threshold = threshold          # Splitting threshold
        self.left = left                    # Left subtree
        self.right = right                  # Right subtree
        self.value = value                  # Class value if leaf
        self.samples = samples              # Number of samples
        self.impurity = impurity           # Gini impurity or entropy


class DecisionTree:
    """
    Decision tree classifier implemented from scratch with numpy.
    Uses Gini impurity for splitting criterion.
    """

    def __init__(self, max_depth=6, min_samples_split=20, min_samples_leaf=10,
                 max_features='sqrt', random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.tree = None
        self.n_features = None
        self.n_classes = None
        self.classes = None

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        """Build decision tree."""
        X = np.asarray(X)
        y = np.asarray(y)

        self.n_features = X.shape[1]
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)

        if sample_weight is None:
            sample_weight = np.ones(len(y))
        else:
            sample_weight = np.asarray(sample_weight)

        self.tree = self._build_tree(X, y, sample_weight, depth=0)
        return self

    def _build_tree(self, X, y, sample_weight, depth):
        """Recursively build tree."""
        n_samples = len(y)
        n_classes_here = len(np.unique(y))

        # Leaf node conditions
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_classes_here == 1):

            # Compute most common class
            unique_classes, counts = np.unique(y, return_counts=True)
            value = unique_classes[np.argmax(counts)]
            gini = self._compute_gini(y, sample_weight)

            return Node(value=value, samples=n_samples, impurity=gini)

        # Try to split
        best_gain = -1
        best_feature = None
        best_threshold = None

        # Determine max features to try
        if self.max_features == 'sqrt':
            n_try = max(1, int(np.sqrt(self.n_features)))
        elif self.max_features == 'log2':
            n_try = max(1, int(np.log2(self.n_features)))
        else:
            n_try = self.n_features

        feature_indices = np.random.choice(self.n_features, n_try, replace=False)

        for feature_idx in feature_indices:
            X_column = X[:, feature_idx]
            unique_vals = np.unique(X_column)

            # Try each unique value as threshold
            for threshold in unique_vals:
                left_mask = X_column <= threshold
                right_mask = ~left_mask

                # Check min samples in leaves
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                # Compute information gain
                gini_parent = self._compute_gini(y, sample_weight)

                y_left = y[left_mask]
                sw_left = sample_weight[left_mask]
                gini_left = self._compute_gini(y_left, sw_left)

                y_right = y[right_mask]
                sw_right = sample_weight[right_mask]
                gini_right = self._compute_gini(y_right, sw_right)

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                weighted_gini = (n_left * gini_left + n_right * gini_right) / n_samples
                gain = gini_parent - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        # If no split found, create leaf
        if best_feature is None:
            unique_classes, counts = np.unique(y, return_counts=True)
            value = unique_classes[np.argmax(counts)]
            gini = self._compute_gini(y, sample_weight)
            return Node(value=value, samples=n_samples, impurity=gini)

        # Recursively build left and right subtrees
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_tree = self._build_tree(X[left_mask], y[left_mask],
                                     sample_weight[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask],
                                      sample_weight[right_mask], depth + 1)

        gini = self._compute_gini(y, sample_weight)
        return Node(feature_idx=best_feature, threshold=best_threshold,
                   left=left_tree, right=right_tree, samples=n_samples, impurity=gini)

    def _compute_gini(self, y, sample_weight):
        """Compute Gini impurity."""
        unique_classes, counts = np.unique(y, return_counts=True)
        weights_per_class = np.zeros(len(self.classes))

        for i, cls in enumerate(self.classes):
            mask = y == cls
            weights_per_class[i] = np.sum(sample_weight[mask])

        total_weight = np.sum(sample_weight)
        p = weights_per_class / (total_weight + 1e-8)
        gini = 1 - np.sum(p ** 2)

        return gini

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities."""
        X = np.asarray(X)
        proba = np.zeros((X.shape[0], self.n_classes))

        for i in range(X.shape[0]):
            leaf = self._traverse_tree(X[i], self.tree)

            # For each class, compute proportion from tree
            class_idx = np.where(self.classes == leaf.value)[0][0]
            proba[i, class_idx] = 1.0

        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X = np.asarray(X)
        predictions = np.zeros(X.shape[0], dtype=int)

        for i in range(X.shape[0]):
            leaf = self._traverse_tree(X[i], self.tree)
            predictions[i] = leaf.value

        return predictions

    def _traverse_tree(self, x, node):
        """Traverse tree to find leaf."""
        if node.feature_idx is None:
            return node

        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


class RegressionTree:
    """Regression tree for gradient boosting (fits continuous residuals using MSE)."""

    def __init__(self, max_depth=4, min_samples_leaf=10, max_features='sqrt',
                 random_state=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.tree = None
        self.n_features = None
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Build regression tree on continuous targets (residuals)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features = X.shape[1]
        self.tree = self._build(X, y, depth=0)
        return self

    def _build(self, X, y, depth):
        n = len(y)
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf:
            return Node(value=float(np.mean(y)), samples=n)

        best_mse = np.inf
        best_feat = None
        best_thr = None

        if self.max_features == 'sqrt':
            n_try = max(1, int(np.sqrt(self.n_features)))
        elif self.max_features == 'log2':
            n_try = max(1, int(np.log2(self.n_features)))
        else:
            n_try = self.n_features

        feat_indices = np.random.choice(self.n_features, min(n_try, self.n_features), replace=False)

        for fi in feat_indices:
            col = X[:, fi]
            # Use quantile thresholds for efficiency
            thresholds = np.unique(np.percentile(col, np.linspace(10, 90, 9)))
            for thr in thresholds:
                left_mask = col <= thr
                n_left = np.sum(left_mask)
                n_right = n - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                mse_left = np.var(y[left_mask]) * n_left
                mse_right = np.var(y[~left_mask]) * n_right
                mse = (mse_left + mse_right) / n
                if mse < best_mse:
                    best_mse = mse
                    best_feat = fi
                    best_thr = thr

        if best_feat is None:
            return Node(value=float(np.mean(y)), samples=n)

        left_mask = X[:, best_feat] <= best_thr
        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return Node(feature_idx=best_feat, threshold=best_thr, left=left, right=right, samples=n)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            node = self.tree
            while node.feature_idx is not None:
                if X[i, node.feature_idx] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            out[i] = node.value
        return out


# ==================== Gradient Boosting ====================

class GradientBoostingClassifier:
    """
    Gradient boosting classifier for binary/multiclass classification.
    Implemented from scratch using numpy.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=4,
                 min_samples_leaf=20, subsample=0.8, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

        self.trees = []
        self.init_score = None
        self.classes = None
        self.n_classes = None
        self.feature_importance_ = None

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the gradient boosting model."""
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes = np.unique(y)
        self.n_classes = len(self.classes)

        # Encode labels to [0, n_classes-1]
        y_encoded = np.zeros_like(y)
        for i, cls in enumerate(self.classes):
            y_encoded[y == cls] = i
        y = y_encoded

        n_samples, n_features = X.shape

        # Initialize score (log odds for binary, or class probability for multiclass)
        if self.n_classes == 2:
            # Binary classification: use log odds
            positive_samples = np.sum(y == 1)
            self.init_score = np.log((positive_samples + 1) / (n_samples - positive_samples + 1))
            F = np.full(n_samples, self.init_score)
        else:
            # Multiclass: use log of class probabilities
            self.init_score = np.log(1.0 / self.n_classes)
            F = np.full((n_samples, self.n_classes), self.init_score)

        self.feature_importance_ = np.zeros(n_features)

        # Boosting iterations
        for iteration in range(self.n_estimators):
            if self.n_classes == 2:
                # Binary classification
                # Compute residuals (pseudo-residuals)
                proba = self._sigmoid(F)
                residuals = y - proba

                # Subsample
                if self.subsample < 1.0:
                    idx = np.random.choice(n_samples,
                                          int(self.subsample * n_samples),
                                          replace=False)
                else:
                    idx = np.arange(n_samples)

                X_sub = X[idx]
                residuals_sub = residuals[idx]

                # Fit regression tree to continuous residuals
                tree = RegressionTree(max_depth=self.max_depth,
                                      min_samples_leaf=self.min_samples_leaf)
                tree.fit(X_sub, residuals_sub)

                # Predict and update
                pred = tree.predict(X)
                F = F + self.learning_rate * pred

                self.trees.append(tree)

            else:
                # Multiclass classification
                for k in range(self.n_classes):
                    y_k = (y == k).astype(float)

                    # Compute probabilities
                    proba = self._softmax(F)
                    residuals = y_k - proba[:, k]

                    # Subsample
                    if self.subsample < 1.0:
                        idx = np.random.choice(n_samples,
                                              int(self.subsample * n_samples),
                                              replace=False)
                    else:
                        idx = np.arange(n_samples)

                    X_sub = X[idx]
                    residuals_sub = residuals[idx]

                    # Fit regression tree to continuous residuals
                    tree = RegressionTree(max_depth=self.max_depth,
                                          min_samples_leaf=self.min_samples_leaf)
                    tree.fit(X_sub, residuals_sub)

                    # Predict and update
                    pred = tree.predict(X)
                    F[:, k] = F[:, k] + self.learning_rate * pred

                    self.trees.append(tree)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates."""
        X = np.asarray(X)
        n_samples = X.shape[0]

        if self.n_classes == 2:
            F = np.full(n_samples, self.init_score)

            for tree in self.trees:
                pred = tree.predict(X).astype(float)
                F = F + self.learning_rate * pred

            proba = self._sigmoid(F)
            return np.column_stack([1 - proba, proba])

        else:
            F = np.full((n_samples, self.n_classes), self.init_score)

            for i, tree in enumerate(self.trees):
                k = i % self.n_classes
                pred = tree.predict(X).astype(float)
                F[:, k] = F[:, k] + self.learning_rate * pred

            return self._softmax(F)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)

        if self.n_classes == 2:
            predictions = (proba[:, 1] > 0.5).astype(int)
            return self.classes[predictions]
        else:
            class_indices = np.argmax(proba, axis=1)
            return self.classes[class_indices]

    def feature_importance(self) -> np.ndarray:
        """Compute feature importance from tree splits."""
        # Simplified: count number of times each feature is used
        n_features = self.trees[0].n_features if self.trees else 0
        importance = np.zeros(n_features)

        for tree in self.trees:
            self._count_feature_splits(tree.tree, importance)

        # Normalize
        if importance.sum() > 0:
            importance = importance / importance.sum()

        return importance

    def _count_feature_splits(self, node, importance):
        """Recursively count feature splits."""
        if node.feature_idx is not None:
            importance[node.feature_idx] += 1
            if node.left:
                self._count_feature_splits(node.left, importance)
            if node.right:
                self._count_feature_splits(node.right, importance)

    @staticmethod
    def _sigmoid(x):
        """Sigmoid function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def _softmax(x):
        """Softmax function."""
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# ==================== Walk-Forward Validation ====================

def walk_forward_validate(df: pd.DataFrame, features_df: pd.DataFrame,
                         labels: pd.Series, model_class, model_params: dict,
                         train_days: int = 500, test_days: int = 20,
                         n_splits: int = 10) -> dict:
    """
    Walk-forward (expanding window) cross-validation.

    Args:
        df: Original OHLCV DataFrame
        features_df: Extracted features DataFrame
        labels: Classification labels
        model_class: Model class (DecisionTree or GradientBoostingClassifier)
        model_params: Model parameters dict
        train_days: Days for training window
        test_days: Days for testing window
        n_splits: Number of splits

    Returns:
        Dict with validation metrics and per-split results
    """
    n_samples = len(features_df)

    if n_samples < train_days + test_days:
        train_days = max(100, n_samples // 2)
        test_days = min(20, (n_samples - train_days) // 5)

    accuracies = []
    precisions = []
    recalls = []
    aucs = []
    f1_scores = []
    per_split_results = []

    step = max(1, (n_samples - train_days - test_days) // n_splits)

    for split in range(n_splits):
        train_end = train_days + split * step
        test_end = min(train_end + test_days, n_samples)

        if test_end <= train_end:
            break

        X_train = features_df.iloc[:train_end].values
        y_train = labels.iloc[:train_end].values

        X_test = features_df.iloc[train_end:test_end].values
        y_test = labels.iloc[train_end:test_end].values

        # Remove samples with no label
        mask_train = y_train != 0
        mask_test = y_test != 0

        if np.sum(mask_train) < 10 or np.sum(mask_test) < 5:
            continue

        X_train = X_train[mask_train]
        y_train = y_train[mask_train]
        X_test = X_test[mask_test]
        y_test = y_test[mask_test]

        # Standardize features
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        std[std == 0] = 1

        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        # Train model
        model = model_class(**model_params)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # For binary metrics, convert to binary
        y_test_binary = (y_test > 0).astype(int)
        y_pred_binary = (y_pred > 0).astype(int)

        # Accuracy
        accuracy = np.mean(y_test_binary == y_pred_binary)
        accuracies.append(accuracy)

        # Precision and Recall
        tp = np.sum((y_test_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_test_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_test_binary == 1) & (y_pred_binary == 0))
        tn = np.sum((y_test_binary == 0) & (y_pred_binary == 0))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        precisions.append(precision)
        recalls.append(recall)

        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        f1_scores.append(f1)

        # AUC (binary classification)
        try:
            proba = model.predict_proba(X_test)
            if proba.shape[1] == 2:
                proba_positive = proba[:, 1]
            else:
                proba_positive = proba[:, 0] if len(proba.shape) > 1 else proba

            auc = _compute_auc(y_test_binary, proba_positive)
            aucs.append(auc)
        except:
            aucs.append(0.5)

        per_split_results.append({
            'train_end': train_end,
            'test_end': test_end,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': aucs[-1] if aucs else 0.5
        })

    if not accuracies:
        return {
            'accuracy': 0.5,
            'precision': 0.5,
            'recall': 0.5,
            'auc_roc': 0.5,
            'f1_score': 0.5,
            'per_split': []
        }

    return {
        'accuracy': np.mean(accuracies),
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'auc_roc': np.mean(aucs),
        'f1_score': np.mean(f1_scores),
        'per_split': per_split_results
    }


def _compute_auc(y_true, y_score):
    """Compute AUC using trapezoidal rule."""
    # Sort by score
    sorted_indices = np.argsort(-y_score)
    y_sorted = y_true[sorted_indices]

    # Compute TPR and FPR
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)

    tpr = tp / n_pos
    fpr = fp / n_neg

    # Add point (0, 0) and (1, 1)
    tpr = np.concatenate([[0], tpr, [1]])
    fpr = np.concatenate([[0], fpr, [1]])

    # Compute AUC using trapezoidal rule
    auc = 0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2

    return max(0, min(1, auc))


# ==================== Main Prediction Interface ====================

def predict_stock(code: str, history_df: pd.DataFrame,
                 fundamentals: dict = None,
                 horizon: int = 5) -> dict:
    """
    Main prediction function for a single stock.

    Args:
        code: Stock code
        history_df: Historical OHLCV data
        fundamentals: Dict with 'pe_ttm', 'pb', 'market_cap', 'turnover_rate'
        horizon: Days ahead to predict

    Returns:
        Dict with probabilities, class, and other metrics
    """
    if len(history_df) < 100:
        return {
            'prob_up': 0.5,
            'prob_strong_up': 0.25,
            'prob_down': 0.5,
            'prob_strong_down': 0.25,
            'predicted_class': 0,
            'confidence': 0.5,
            'feature_importance': {},
            'validation_accuracy': 0.5,
            'validation_auc': 0.5,
            'n_training_samples': 0,
            'signal': 'neutral',
            'error': 'Insufficient data'
        }

    # Extract features and labels
    features = extract_features(history_df, fundamentals)
    labels = generate_labels(history_df, horizon=horizon, binary=False)

    # Standardize features
    mean = np.nanmean(features.values, axis=0)
    std = np.nanstd(features.values, axis=0)
    std[std == 0] = 1

    X = (features.values - mean) / std
    y = labels.values

    # Train-test split (use last 20% for testing)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train model
    model_params = {
        'n_estimators': 50,
        'learning_rate': 0.1,
        'max_depth': 4,
        'min_samples_leaf': 10,
        'subsample': 0.8,
        'random_state': 42
    }

    model = GradientBoostingClassifier(**model_params)
    model.fit(X_train, y_train)

    # Predictions on test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Compute validation metrics
    accuracy = np.mean(y_pred == y_test)

    # Prediction on latest bar
    X_latest = X[-1:, :]
    pred_class = model.predict(X_latest)[0]
    proba_latest = model.predict_proba(X_latest)[0]

    # Convert probabilities to standard output format
    classes = np.unique(y)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    prob_strong_up = proba_latest[class_to_idx.get(2, 0)] if 2 in class_to_idx else 0
    prob_up = (proba_latest[class_to_idx.get(1, 0)] if 1 in class_to_idx else 0) + prob_strong_up
    prob_strong_down = proba_latest[class_to_idx.get(-2, 0)] if -2 in class_to_idx else 0
    prob_down = (proba_latest[class_to_idx.get(-1, 0)] if -1 in class_to_idx else 0) + prob_strong_down
    prob_neutral = proba_latest[class_to_idx.get(0, 0)] if 0 in class_to_idx else (1 - prob_up - prob_down)

    # Normalize probabilities
    total = prob_up + prob_down + prob_neutral
    if total > 0:
        prob_up /= total
        prob_down /= total
        prob_neutral /= total

    # Confidence
    confidence = np.max(proba_latest)

    # Feature importance
    importances = model.feature_importance()
    top_idx = np.argsort(-importances)[:10]
    feature_importance_dict = {
        _FEATURE_NAMES[idx]: float(importances[idx])
        for idx in top_idx if idx < len(_FEATURE_NAMES)
    }

    # Signal
    if pred_class > 0:
        signal = 'bullish'
    elif pred_class < 0:
        signal = 'bearish'
    else:
        signal = 'neutral'

    # AUC
    try:
        y_test_binary = (y_test > 0).astype(int)
        proba_positive = y_proba[:, 1] if y_proba.shape[1] == 2 else np.max(y_proba, axis=1)
        auc = _compute_auc(y_test_binary, proba_positive)
    except:
        auc = 0.5

    return {
        'prob_up': float(prob_up),
        'prob_strong_up': float(prob_strong_up),
        'prob_down': float(prob_down),
        'prob_strong_down': float(prob_strong_down),
        'predicted_class': int(pred_class),
        'confidence': float(confidence),
        'feature_importance': feature_importance_dict,
        'validation_accuracy': float(accuracy),
        'validation_auc': float(auc),
        'n_training_samples': len(X_train),
        'signal': signal
    }


def batch_predict(codes: list, horizon: int = 5, data_dict: dict = None) -> pd.DataFrame:
    """
    Predict for multiple stocks.

    Args:
        codes: List of stock codes
        horizon: Days ahead to predict
        data_dict: Optional dict mapping code to (history_df, fundamentals)

    Returns:
        DataFrame with predictions for each stock
    """
    results = []

    for code in codes:
        try:
            if data_dict and code in data_dict:
                history_df, fundamentals = data_dict[code]
            else:
                # Would load from DB in real implementation
                continue

            pred = predict_stock(code, history_df, fundamentals, horizon)
            pred['code'] = code
            results.append(pred)
        except Exception as e:
            results.append({
                'code': code,
                'error': str(e),
                'prob_up': np.nan,
                'signal': 'error'
            })

    return pd.DataFrame(results)
