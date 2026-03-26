"""
Quantitative Analysis Engine — Multi-Dimensional Mathematical Prediction System
=================================================================================
Combines trend analysis, momentum, volatility, mean reversion, volume-price
relationships, support/resistance, and statistical features through a weighted
multi-factor scoring system to produce comprehensive buy/hold/sell signals.

All calculations rely only on numpy + pandas; no scipy or sklearn required.

Mathematical Model Inventory:
  +-----------------------------------------------------------+
  | Dimension          Model / Formula                        |
  |-----------------------------------------------------------|
  | Trend              Linear regression slope, R-squared,    |
  |                    price vs multi-MA position, ADX        |
  | Momentum           MACD histogram trend, RSI overbought/  |
  |                    oversold, ROC, Williams %R             |
  | Volatility         ATR, Bollinger Bandwidth, HV,          |
  |                    volatility cone                        |
  | Mean Reversion     Z-Score deviation, Bollinger %B        |
  | Volume-Price       OBV trend, volume-price divergence     |
  |                    relative volume strength               |
  | Pattern            Support/Resistance, trend channel,     |
  |                    Fibonacci retracement                  |
  | Statistics         Hurst exponent, skewness/kurtosis,     |
  |                    return normality test                  |
  | Monte Carlo        GBM future price path simulation,      |
  |                    probability distribution               |
  +-----------------------------------------------------------+
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional


# =====================================================
# 1. Basic Mathematical Utilities
# =====================================================

def linear_regression(y: np.ndarray) -> Tuple[float, float, float]:
    """
    Ordinary least squares linear regression.
    y: 1-D array
    Returns: (slope, intercept, r_squared)
    """
    n = len(y)
    x = np.arange(n, dtype=float)
    x_mean, y_mean = x.mean(), y.mean()
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    ss_yy = np.sum((y - y_mean) ** 2)

    slope = ss_xy / ss_xx if ss_xx != 0 else 0
    intercept = y_mean - slope * x_mean
    r_squared = (ss_xy ** 2) / (ss_xx * ss_yy) if (ss_xx * ss_yy) != 0 else 0
    return slope, intercept, r_squared


def hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    """
    Hurst exponent (R/S analysis).
    H > 0.5 -> trend persistence (momentum)
    H ~ 0.5 -> random walk
    H < 0.5 -> mean reversion
    """
    lags = range(2, min(max_lag + 1, len(series) // 2))
    rs_values = []
    for lag in lags:
        chunks = [series[i:i + lag] for i in range(0, len(series) - lag + 1, lag)]
        rs_list = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean_c = chunk.mean()
            deviate = np.cumsum(chunk - mean_c)
            R = deviate.max() - deviate.min()
            S = chunk.std(ddof=1) if chunk.std(ddof=1) > 0 else 1e-10
            rs_list.append(R / S)
        if rs_list:
            rs_values.append((np.log(lag), np.log(np.mean(rs_list))))

    if len(rs_values) < 3:
        return 0.5
    log_lags = np.array([v[0] for v in rs_values])
    log_rs   = np.array([v[1] for v in rs_values])
    slope, _, _ = linear_regression(log_rs)
    # Regress on log_lags
    n = len(log_lags)
    x_m = log_lags.mean()
    y_m = log_rs.mean()
    slope = np.sum((log_lags - x_m) * (log_rs - y_m)) / np.sum((log_lags - x_m) ** 2)
    return np.clip(slope, 0.0, 1.0)


def ewma(series: np.ndarray, span: int) -> np.ndarray:
    """Exponentially weighted moving average"""
    alpha = 2.0 / (span + 1)
    result = np.zeros_like(series, dtype=float)
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


# =====================================================
# 2. Trend Analysis
# =====================================================

def analyze_trend(df: pd.DataFrame) -> Dict:
    """
    Multi-dimensional trend analysis.

    Models:
    1) Linear regression slope & R-squared (short/mid/long windows)
    2) Price vs MA position
    3) Multi-MA alignment (bullish/bearish)
    4) ADX (Average Directional Index)

    Returns:
    {
      "score": -100 ~ +100,
      "direction": "uptrend/downtrend/sideways",
      "strength": "strong/moderate/weak",
      "details": {...}
    }
    """
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    n = len(close)
    details = {}

    # -- Linear regression (20/60/120 day) --
    reg_scores = []
    for window, weight in [(20, 0.5), (60, 0.3), (min(120, n - 1), 0.2)]:
        if n < window + 1:
            continue
        seg = close[-window:]
        slope, _, r2 = linear_regression(seg)
        # Normalize slope: slope / mean price * 100 -> daily % change
        slope_pct = slope / np.mean(seg) * 100
        # Score: slope direction x R-squared x weight
        score = np.clip(slope_pct * 30, -100, 100) * r2
        reg_scores.append(score * weight)
        details[f"reg_{window}d_slope_pct"] = round(slope_pct, 4)
        details[f"reg_{window}d_r2"] = round(r2, 3)

    reg_total = sum(reg_scores) if reg_scores else 0

    # -- Multi-MA alignment --
    ma_periods = [5, 10, 20, 60]
    mas = {}
    for p in ma_periods:
        if n >= p:
            mas[p] = np.mean(close[-p:])

    ma_score = 0
    if len(mas) >= 4:
        # Perfect bullish alignment: MA5 > MA10 > MA20 > MA60
        vals = [mas[p] for p in sorted(mas.keys())]
        for i in range(len(vals) - 1):
            if vals[i] > vals[i + 1]:
                ma_score += 15    # Each correctly ordered pair +15
            else:
                ma_score -= 15
        # Price above all MAs
        if close[-1] > max(vals):
            ma_score += 15
        elif close[-1] < min(vals):
            ma_score -= 15
    details["ma_alignment_score"] = round(ma_score, 1)

    # -- ADX calculation --
    adx_val = _calc_adx(high, low, close, period=14)
    details["adx_14"] = round(adx_val, 1)
    # ADX > 25 = clear trend, > 40 = strong trend
    trend_clarity = min(adx_val / 40, 1.0)  # 0~1

    # -- Composite trend score --
    raw = reg_total * 0.5 + ma_score * 0.5
    score = np.clip(raw, -100, 100)

    # Direction determination
    if score > 20:
        direction = "uptrend"
    elif score < -20:
        direction = "downtrend"
    else:
        direction = "sideways"

    # Strength determination
    abs_score = abs(score)
    if abs_score > 60:
        strength = "strong"
    elif abs_score > 30:
        strength = "moderate"
    else:
        strength = "weak"

    return {
        "score": round(score, 1),
        "direction": direction,
        "strength": strength,
        "trend_clarity": round(trend_clarity, 2),
        "details": details,
    }


def _calc_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period + 1:
        return 0

    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        h_diff = high[i] - high[i - 1]
        l_diff = low[i - 1] - low[i]
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        plus_dm[i] = h_diff if (h_diff > l_diff and h_diff > 0) else 0
        minus_dm[i] = l_diff if (l_diff > h_diff and l_diff > 0) else 0

    # Wilder smoothing
    atr = ewma(tr[1:], period)
    plus_di = 100 * ewma(plus_dm[1:], period) / np.where(atr > 0, atr, 1)
    minus_di = 100 * ewma(minus_dm[1:], period) / np.where(atr > 0, atr, 1)

    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, plus_di + minus_di, 1)
    adx = ewma(dx, period)
    return adx[-1] if len(adx) > 0 else 0


# =====================================================
# 3. Momentum Analysis
# =====================================================

def analyze_momentum(df: pd.DataFrame) -> Dict:
    """
    Momentum analysis.

    Models:
    1) MACD histogram trend (DIF-DEA delta direction)
    2) RSI overbought/oversold + divergence detection
    3) ROC (Rate of Change) multi-period
    4) Williams %R

    Returns score -100 ~ +100
    """
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    n = len(close)
    details = {}

    # -- MACD --
    if n >= 26:
        ema12 = ewma(close, 12)
        ema26 = ewma(close, 26)
        dif = ema12 - ema26
        dea = ewma(dif, 9)
        macd_hist = (dif - dea) * 2

        # MACD histogram trend (last 5 bars direction)
        recent_hist = macd_hist[-5:]
        hist_slope, _, _ = linear_regression(recent_hist)
        macd_score = np.clip(hist_slope / (abs(close[-1]) * 0.001 + 1e-10) * 50, -30, 30)

        # Golden cross / Death cross
        if len(dif) >= 2:
            if dif[-1] > dea[-1] and dif[-2] <= dea[-2]:
                macd_score += 15  # Golden cross
                details["macd_cross"] = "golden_cross"
            elif dif[-1] < dea[-1] and dif[-2] >= dea[-2]:
                macd_score -= 15  # Death cross
                details["macd_cross"] = "death_cross"

        details["dif"] = round(dif[-1], 3)
        details["dea"] = round(dea[-1], 3)
        details["macd_hist"] = round(macd_hist[-1], 3)
    else:
        macd_score = 0

    # -- RSI --
    if n >= 14:
        rsi = _calc_rsi(close, 14)
        rsi_val = rsi[-1]
        details["rsi_14"] = round(rsi_val, 1)

        if rsi_val > 80:
            rsi_score = -30  # Severely overbought
        elif rsi_val > 70:
            rsi_score = -20  # Overbought
        elif rsi_val < 20:
            rsi_score = 30   # Severely oversold (bounce opportunity)
        elif rsi_val < 30:
            rsi_score = 20   # Oversold
        elif 45 <= rsi_val <= 55:
            rsi_score = 0    # Neutral
        elif rsi_val > 55:
            rsi_score = 10   # Slightly bullish
        else:
            rsi_score = -10  # Slightly bearish

        # RSI divergence detection
        divergence = _detect_rsi_divergence(close, rsi)
        if divergence == "bullish":
            rsi_score += 15
            details["rsi_divergence"] = "bullish_divergence"
        elif divergence == "bearish":
            rsi_score -= 15
            details["rsi_divergence"] = "bearish_divergence"
    else:
        rsi_score = 0

    # -- ROC --
    roc_scores = []
    for period in [5, 10, 20]:
        if n > period:
            roc = (close[-1] / close[-period - 1] - 1) * 100
            details[f"roc_{period}"] = round(roc, 2)
            roc_scores.append(np.clip(roc * 3, -15, 15))
    roc_score = np.mean(roc_scores) if roc_scores else 0

    # -- Williams %R --
    if n >= 14:
        wr = _calc_williams_r(high, low, close, 14)
        details["williams_r"] = round(wr, 1)
        if wr < -80:
            wr_score = 15  # Oversold
        elif wr > -20:
            wr_score = -15  # Overbought
        else:
            wr_score = 0
    else:
        wr_score = 0

    total = np.clip(macd_score * 0.35 + rsi_score * 0.30 + roc_score * 0.20 + wr_score * 0.15,
                    -100, 100)

    return {
        "score": round(total, 1),
        "details": details,
    }


def _calc_rsi(close, period=14):
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    avg_gain = ewma(gains, period)
    avg_loss = ewma(losses, period)
    rs = avg_gain / np.where(avg_loss > 0, avg_loss, 1e-10)
    return 100 - 100 / (1 + rs)


def _calc_williams_r(high, low, close, period=14):
    highest = max(high[-period:])
    lowest  = min(low[-period:])
    if highest == lowest:
        return -50
    return (highest - close[-1]) / (highest - lowest) * -100


def _detect_rsi_divergence(close, rsi, lookback=30):
    """Detect RSI divergence"""
    n = min(len(close), len(rsi), lookback)
    if n < 10:
        return None
    c = close[-n:]
    r = rsi[-n:]
    mid = n // 2

    # Bullish divergence: price makes new low, RSI does not
    if c[-1] < np.min(c[:mid]) and r[-1] > np.min(r[:mid]):
        return "bullish"
    # Bearish divergence: price makes new high, RSI does not
    if c[-1] > np.max(c[:mid]) and r[-1] < np.max(r[:mid]):
        return "bearish"
    return None


# =====================================================
# 4. Volatility Analysis
# =====================================================

def analyze_volatility(df: pd.DataFrame) -> Dict:
    """
    Volatility analysis.

    Models:
    1) ATR (Average True Range) and its trend
    2) Bollinger Bandwidth & %B
    3) Historical Volatility (annualized log-return std)
    4) Volatility percentile (current HV rank in history)

    Returns volatility state and risk level
    """
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    n = len(close)
    details = {}

    # -- ATR --
    if n >= 15:
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i - 1]),
                        abs(low[i] - close[i - 1]))
        atr = ewma(tr[1:], 14)
        atr_val = atr[-1]
        atr_pct = atr_val / close[-1] * 100  # ATR as % of price
        details["atr_14"] = round(atr_val, 3)
        details["atr_pct"] = round(atr_pct, 2)

        # ATR trend (expanding or contracting)
        if len(atr) >= 10:
            atr_slope, _, _ = linear_regression(atr[-10:])
            details["atr_trend"] = "expanding" if atr_slope > 0 else "contracting"
    else:
        atr_pct = 2.0

    # -- Bollinger Bands --
    if n >= 20:
        ma20 = np.mean(close[-20:])
        std20 = np.std(close[-20:], ddof=1)
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        bandwidth = (upper - lower) / ma20 * 100
        pct_b = (close[-1] - lower) / (upper - lower) if (upper - lower) > 0 else 0.5

        details["boll_bandwidth"] = round(bandwidth, 2)
        details["boll_pct_b"] = round(pct_b, 3)
        details["boll_upper"] = round(upper, 2)
        details["boll_lower"] = round(lower, 2)

    # -- Historical Volatility --
    if n >= 22:
        log_ret = np.log(close[1:] / close[:-1])
        hv_20 = np.std(log_ret[-20:], ddof=1) * np.sqrt(252) * 100  # Annualized
        hv_60 = np.std(log_ret[-min(60, len(log_ret)):], ddof=1) * np.sqrt(252) * 100
        details["hv_20d"] = round(hv_20, 1)
        details["hv_60d"] = round(hv_60, 1)

        # Volatility percentile (current 20d HV rank in past year)
        if len(log_ret) >= 120:
            hv_series = pd.Series(log_ret).rolling(20).std() * np.sqrt(252) * 100
            hv_series = hv_series.dropna()
            if len(hv_series) > 0:
                percentile = (hv_series < hv_20).sum() / len(hv_series) * 100
                details["hv_percentile"] = round(percentile, 0)

        current_hv = hv_20
    else:
        current_hv = 30

    # -- Risk level --
    if current_hv > 50 or atr_pct > 4:
        risk_level = "extreme"
        vol_score = -30
    elif current_hv > 35 or atr_pct > 3:
        risk_level = "high"
        vol_score = -15
    elif current_hv > 20 or atr_pct > 2:
        risk_level = "moderate"
        vol_score = 0
    else:
        risk_level = "low"
        vol_score = 10

    return {
        "score": round(vol_score, 1),
        "risk_level": risk_level,
        "current_hv": round(current_hv, 1),
        "details": details,
    }


# =====================================================
# 5. Mean Reversion Analysis
# =====================================================

def analyze_mean_reversion(df: pd.DataFrame) -> Dict:
    """
    Mean reversion analysis.

    Models:
    1) Z-Score: (price - MA) / sigma, measures deviation
    2) Bollinger %B: position within Bollinger Bands
    3) Price deviation: % deviation from multiple MAs

    Z > 2 -> severely overvalued, likely to revert down
    Z < -2 -> severely undervalued, likely to bounce
    """
    close = df["close"].values
    n = len(close)
    details = {}
    scores = []

    for window, weight in [(20, 0.5), (60, 0.3), (min(120, n - 1), 0.2)]:
        if n < window + 1:
            continue
        seg = close[-window:]
        ma = np.mean(seg)
        std = np.std(seg, ddof=1)
        if std > 0:
            z = (close[-1] - ma) / std
        else:
            z = 0

        details[f"z_score_{window}d"] = round(z, 2)
        details[f"deviation_{window}d_pct"] = round((close[-1] / ma - 1) * 100, 2)

        # Z-Score -> score (inverted: high Z = excessive rally = bearish)
        if abs(z) > 3:
            s = -np.sign(z) * 40
        elif abs(z) > 2:
            s = -np.sign(z) * 25
        elif abs(z) > 1:
            s = -np.sign(z) * 10
        else:
            s = 0
        scores.append(s * weight)

    total = np.clip(sum(scores), -100, 100) if scores else 0

    # Signal determination
    if total > 15:
        signal = "oversold_reversion (bullish)"
    elif total < -15:
        signal = "overbought_reversion (bearish)"
    else:
        signal = "near_mean"

    return {
        "score": round(total, 1),
        "signal": signal,
        "details": details,
    }


# =====================================================
# 6. Volume-Price Analysis
# =====================================================

def analyze_volume(df: pd.DataFrame) -> Dict:
    """
    Volume-price relationship analysis.

    Models:
    1) OBV (On-Balance Volume) trend vs price trend
    2) Relative volume strength (current vs historical average)
    3) Volume-price divergence detection
    4) Volume expansion/contraction
    """
    close  = df["close"].values
    volume = df["volume"].values
    n = len(close)
    details = {}

    if n < 10:
        return {"score": 0, "details": {}}

    # -- OBV --
    obv = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]

    # OBV trend vs price trend
    window = min(20, n - 1)
    obv_slope, _, obv_r2 = linear_regression(obv[-window:])
    price_slope, _, price_r2 = linear_regression(close[-window:])

    obv_up = obv_slope > 0
    price_up = price_slope > 0

    if obv_up and price_up:
        obv_score = 15       # Volume and price rising together
        details["obv_signal"] = "price_volume_rising (healthy uptrend)"
    elif obv_up and not price_up:
        obv_score = 20       # Accumulation at bottom
        details["obv_signal"] = "accumulation (bullish signal)"
    elif not obv_up and price_up:
        obv_score = -20      # Distribution at top
        details["obv_signal"] = "distribution (bearish signal)"
    else:
        obv_score = -10      # Volume and price falling together
        details["obv_signal"] = "price_volume_falling (weak)"

    # -- Relative volume strength --
    vol_ma5 = np.mean(volume[-5:])
    vol_ma20 = np.mean(volume[-min(20, n):])
    vol_ratio = vol_ma5 / vol_ma20 if vol_ma20 > 0 else 1
    details["vol_ratio_5_20"] = round(vol_ratio, 2)

    if vol_ratio > 2.0:
        details["vol_status"] = "significant_expansion"
    elif vol_ratio > 1.3:
        details["vol_status"] = "moderate_expansion"
    elif vol_ratio < 0.5:
        details["vol_status"] = "significant_contraction"
    elif vol_ratio < 0.7:
        details["vol_status"] = "moderate_contraction"
    else:
        details["vol_status"] = "stable_volume"

    # -- Volume-price divergence --
    if n >= 30:
        mid = n - 15
        # Price makes new high but volume shrinks -> bearish divergence
        if close[-1] > np.max(close[mid-15:mid]) and vol_ma5 < vol_ma20 * 0.8:
            obv_score -= 10
            details["vol_divergence"] = "bearish_divergence (price up, volume down)"
        # Price makes new low but volume shrinks -> bullish divergence (selling exhaustion)
        elif close[-1] < np.min(close[mid-15:mid]) and vol_ma5 < vol_ma20 * 0.7:
            obv_score += 10
            details["vol_divergence"] = "bullish_divergence (selling exhaustion)"

    return {
        "score": round(np.clip(obv_score, -100, 100), 1),
        "details": details,
    }


# =====================================================
# 7. Support / Resistance Analysis
# =====================================================

def analyze_support_resistance(df: pd.DataFrame) -> Dict:
    """
    Support / resistance level analysis.

    Models:
    1) Local extrema detection (recent highs/lows)
    2) Fibonacci retracement levels (0.236, 0.382, 0.5, 0.618)
    3) Distance from price to support/resistance
    """
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    n = len(close)
    details = {}

    if n < 30:
        return {"score": 0, "details": {}, "supports": [], "resistances": []}

    lookback = min(120, n)
    seg_high = high[-lookback:]
    seg_low  = low[-lookback:]
    seg_close = close[-lookback:]
    current = close[-1]

    # -- Recent highs/lows --
    recent_high = np.max(seg_high)
    recent_low  = np.min(seg_low)
    details["recent_high"] = round(recent_high, 2)
    details["recent_low"]  = round(recent_low, 2)

    # -- Fibonacci retracement levels --
    fib_range = recent_high - recent_low
    fib_levels = {}
    for ratio, name in [(0.236, "23.6%"), (0.382, "38.2%"),
                         (0.5, "50%"), (0.618, "61.8%")]:
        # Retracement from high
        level = recent_high - fib_range * ratio
        fib_levels[name] = round(level, 2)
    details["fibonacci"] = fib_levels

    # -- Key support and resistance --
    supports = []
    resistances = []

    # Local low detection (as support)
    for i in range(2, len(seg_low) - 2):
        if seg_low[i] < seg_low[i-1] and seg_low[i] < seg_low[i-2] \
           and seg_low[i] < seg_low[i+1] and seg_low[i] < seg_low[i+2]:
            if seg_low[i] < current:
                supports.append(round(seg_low[i], 2))
    # Local high detection (as resistance)
    for i in range(2, len(seg_high) - 2):
        if seg_high[i] > seg_high[i-1] and seg_high[i] > seg_high[i-2] \
           and seg_high[i] > seg_high[i+1] and seg_high[i] > seg_high[i+2]:
            if seg_high[i] > current:
                resistances.append(round(seg_high[i], 2))

    # Deduplicate and sort
    supports = sorted(set(supports), reverse=True)[:3]   # Nearest 3 support levels
    resistances = sorted(set(resistances))[:3]             # Nearest 3 resistance levels

    # Add Fibonacci levels to support/resistance
    for name, level in fib_levels.items():
        if level < current and level > recent_low:
            supports.append(level)
        elif level > current and level < recent_high:
            resistances.append(level)

    supports = sorted(set(supports), reverse=True)[:5]
    resistances = sorted(set(resistances))[:5]

    # -- Distance to support/resistance --
    if supports:
        nearest_support = supports[0]
        dist_support = (current - nearest_support) / current * 100
        details["nearest_support"] = nearest_support
        details["dist_to_support_pct"] = round(dist_support, 2)
    if resistances:
        nearest_resistance = resistances[0]
        dist_resistance = (nearest_resistance - current) / current * 100
        details["nearest_resistance"] = nearest_resistance
        details["dist_to_resistance_pct"] = round(dist_resistance, 2)

    # -- Position score --
    position_in_range = (current - recent_low) / (recent_high - recent_low) \
        if (recent_high - recent_low) > 0 else 0.5
    details["position_in_range"] = round(position_in_range, 3)

    # Closer to low = more favorable (mean reversion perspective)
    if position_in_range < 0.2:
        score = 15
    elif position_in_range < 0.4:
        score = 5
    elif position_in_range > 0.8:
        score = -15
    elif position_in_range > 0.6:
        score = -5
    else:
        score = 0

    return {
        "score": round(score, 1),
        "supports": supports,
        "resistances": resistances,
        "details": details,
    }


# =====================================================
# 8. Statistical Feature Analysis
# =====================================================

def analyze_statistics(df: pd.DataFrame) -> Dict:
    """
    Return distribution statistical analysis.

    Models:
    1) Hurst exponent -> trend vs mean reversion
    2) Skewness -> return distribution symmetry
    3) Kurtosis -> tail risk
    4) Annualized return and Sharpe ratio
    """
    close = df["close"].values
    n = len(close)
    details = {}

    if n < 30:
        return {"score": 0, "details": {}}

    log_ret = np.log(close[1:] / close[:-1])

    # -- Hurst exponent --
    H = hurst_exponent(log_ret)
    details["hurst"] = round(H, 3)
    if H > 0.6:
        details["hurst_interp"] = "strong trend persistence -> trend-following strategies effective"
    elif H < 0.4:
        details["hurst_interp"] = "mean reversion characteristic -> contrarian strategies effective"
    else:
        details["hurst_interp"] = "near random walk"

    # -- Skewness --
    mean_r = log_ret.mean()
    std_r  = log_ret.std(ddof=1)
    if std_r > 0:
        skew = np.mean(((log_ret - mean_r) / std_r) ** 3)
        kurt = np.mean(((log_ret - mean_r) / std_r) ** 4) - 3
    else:
        skew, kurt = 0, 0
    details["skewness"] = round(skew, 3)
    details["kurtosis"] = round(kurt, 3)
    if skew > 0.5:
        details["skew_interp"] = "positive skew (right tail) -> occasional large gains"
    elif skew < -0.5:
        details["skew_interp"] = "negative skew (left tail) -> occasional large drops, watch risk"
    if kurt > 1:
        details["kurt_interp"] = "fat-tail distribution -> extreme events more likely than normal"

    # -- Annualized return & Sharpe --
    period_days = min(252, len(log_ret))
    recent_ret = log_ret[-period_days:]
    ann_ret = np.mean(recent_ret) * 252 * 100
    ann_vol = np.std(recent_ret, ddof=1) * np.sqrt(252) * 100
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
    details["ann_return_pct"] = round(ann_ret, 1)
    details["ann_volatility_pct"] = round(ann_vol, 1)
    details["sharpe_ratio"] = round(sharpe, 2)

    # Score
    score = 0
    if sharpe > 1:
        score += 15
    elif sharpe > 0.5:
        score += 5
    elif sharpe < -0.5:
        score -= 15
    elif sharpe < 0:
        score -= 5

    return {
        "score": round(score, 1),
        "details": details,
    }


# =====================================================
# 9. KDJ + K线形态识别（涨停/缩量回调/吞没/锤子线）
# =====================================================

def analyze_kdj_patterns(df: pd.DataFrame) -> Dict:
    """
    KDJ随机指标 + 重要K线形态识别。

    KDJ说明:
      RSV = (Close - Lowest_Low_N) / (Highest_High_N - Lowest_Low_N) * 100
      K   = EWMA(RSV, α=1/3)   (平滑周期9)
      D   = EWMA(K, α=1/3)
      J   = 3K - 2D

    信号:
      K从D下方上穿D → 金叉（做多信号）
      K从D上方下穿D → 死叉（做空信号）
      J < 10 → 严重超卖，K < 20 → 超卖
      J > 90 → 严重超买，K > 80 → 超买

    K线形态:
      涨停板:  当日涨幅 >= 9.5%
      缩量回调: 近3日收跌，但成交量比5日均量低30%以上
      放量上攻: 当日涨幅>2% 且量比 >= 1.5
      锤子线:  下影线 >= 2 * 实体，且当日收盘接近最高价
      吞没形态: 今日阳线完全包含昨日阴线（看涨吞没）
      死亡十字: MA5上穿MA10后又下穿（短期趋势反转）

    Returns dict with score(-100~+100), kdj values, pattern signals
    """
    close  = df["close"].values
    high   = df["high"].values
    low    = df["low"].values
    volume = df["volume"].values if "volume" in df.columns else None
    n = len(close)
    details = {}
    patterns = []

    # ── KDJ ──
    kdj_score = 0
    if n >= 9:
        period = 9
        k_arr  = np.full(n, 50.0)
        d_arr  = np.full(n, 50.0)

        for i in range(period - 1, n):
            hh = max(high[i - period + 1: i + 1])
            ll = min(low[i - period + 1: i + 1])
            rsv = (close[i] - ll) / (hh - ll) * 100 if hh != ll else 50.0
            k_arr[i] = k_arr[i - 1] * (2/3) + rsv * (1/3)
            d_arr[i] = d_arr[i - 1] * (2/3) + k_arr[i] * (1/3)

        j_arr = 3 * k_arr - 2 * d_arr
        k_val, d_val, j_val = k_arr[-1], d_arr[-1], j_arr[-1]
        details["kdj_k"] = round(k_val, 1)
        details["kdj_d"] = round(d_val, 1)
        details["kdj_j"] = round(j_val, 1)

        # 金叉/死叉
        if n >= 10:
            if k_arr[-2] <= d_arr[-2] and k_val > d_val:
                details["kdj_cross"] = "golden_cross"
                kdj_score += 20
                patterns.append("KDJ金叉")
            elif k_arr[-2] >= d_arr[-2] and k_val < d_val:
                details["kdj_cross"] = "death_cross"
                kdj_score -= 20
                patterns.append("KDJ死叉")

        # 超买/超卖
        if j_val < 10:
            kdj_score += 20
            details["kdj_signal"] = "oversold_extreme"
            patterns.append("KDJ严重超卖(J<10)")
        elif k_val < 20:
            kdj_score += 12
            details["kdj_signal"] = "oversold"
        elif j_val > 90:
            kdj_score -= 20
            details["kdj_signal"] = "overbought_extreme"
            patterns.append("KDJ严重超买(J>90)")
        elif k_val > 80:
            kdj_score -= 12
            details["kdj_signal"] = "overbought"

    # ── K线形态 ──
    pattern_score = 0
    if n >= 5:
        o  = df["open"].values  if "open" in df.columns else close
        c  = close
        h  = high
        l  = low

        # 1. 涨停板检测（日涨幅 >= 9.5%）
        if n >= 2:
            prev_close = c[-2]
            today_chg = (c[-1] / prev_close - 1) * 100 if prev_close > 0 else 0
            details["today_pct_chg"] = round(today_chg, 2)
            if today_chg >= 9.5:
                details["limit_up"] = True
                patterns.append("涨停板（+9.5%↑）")
                pattern_score += 15   # 短线动能极强

        # 2. 缩量回调（近3日收跌 + 量 < 5日均量的70%）
        if n >= 8 and volume is not None:
            last3_down = all(c[i] < c[i - 1] for i in range(-3, 0))
            vol5_avg = np.mean(volume[-6:-1])  # 5日均量（排除今日）
            today_vol = volume[-1]
            if last3_down and vol5_avg > 0 and today_vol < vol5_avg * 0.70:
                details["shrink_pullback"] = True
                patterns.append("缩量回调（调整健康）")
                pattern_score += 12

        # 3. 放量上攻（今日涨 > 2% + 量比 >= 1.5）
        if n >= 6 and volume is not None:
            vol5_avg2 = np.mean(volume[-6:-1])
            if vol5_avg2 > 0:
                vol_ratio = volume[-1] / vol5_avg2
                details["vol_ratio_today"] = round(vol_ratio, 2)
                if n >= 2:
                    chg_today = (c[-1] / c[-2] - 1) * 100 if c[-2] > 0 else 0
                    if chg_today > 2.0 and vol_ratio >= 1.5:
                        details["volume_surge"] = True
                        patterns.append(f"放量上攻(量比{vol_ratio:.1f}x)")
                        pattern_score += 15

        # 4. 锤子线（下影线 >= 2 * 实体，且收盘接近最高价）
        if n >= 2:
            body = abs(c[-1] - o[-1])
            lower_shadow = min(o[-1], c[-1]) - l[-1]
            upper_shadow = h[-1] - max(o[-1], c[-1])
            if body > 0 and lower_shadow >= 2 * body and upper_shadow <= 0.5 * body:
                details["hammer"] = True
                patterns.append("锤子线（支撑反弹形态）")
                pattern_score += 10

        # 5. 看涨吞没（今日阳线 > 昨日阴线）
        if n >= 2:
            yesterday_bear = o[-2] > c[-2]       # 昨日阴线
            today_bull     = c[-1] > o[-1]        # 今日阳线
            engulf = (c[-1] > o[-2]) and (o[-1] < c[-2])
            if yesterday_bear and today_bull and engulf:
                details["bullish_engulfing"] = True
                patterns.append("看涨吞没形态")
                pattern_score += 12

        # 6. 高开低走（今日高开>1% 但收盘接近低点，阴线）
        if n >= 2:
            gap_up = (o[-1] / c[-2] - 1) * 100 if c[-2] > 0 else 0
            if gap_up > 1.0 and c[-1] < o[-1]:
                down_ratio = (o[-1] - c[-1]) / (h[-1] - l[-1] + 1e-10)
                if down_ratio > 0.6:
                    details["shooting_star"] = True
                    patterns.append("高开低走/射击之星（警惕回调）")
                    pattern_score -= 10

    details["patterns"] = patterns
    total_score = np.clip(kdj_score * 0.6 + pattern_score * 0.4, -100, 100)

    return {
        "score":    round(total_score, 1),
        "kdj_k":    details.get("kdj_k"),
        "kdj_d":    details.get("kdj_d"),
        "kdj_j":    details.get("kdj_j"),
        "patterns": patterns,
        "details":  details,
    }


def analyze_dupont(financial_history: list) -> Dict:
    """
    杜邦分析 (DuPont Analysis) — 拆解ROE来源。

    ROE = 净利润率 × 资产周转率 × 权益乘数
        = (Net Profit / Revenue) × (Revenue / Total Assets) × (Total Assets / Equity)

    由于东财历史财务只有ROE、毛利率、营收、净利润，
    此处做简化版杜邦：通过多期趋势判断ROE质量。

    Parameters
    ----------
    financial_history : list   来自 get_financial_history() 的历史财务数据
                                需要至少3期（年报优先）

    Returns
    -------
    dict:
        roe_trend     : str   ROE趋势 (improving/stable/declining)
        avg_roe       : float 近3期平均ROE (%)
        roe_quality   : str   ROE质量判断（高毛利率驱动 vs 高杠杆驱动）
        gross_trend   : str   毛利率趋势
        profit_trend  : str   净利润同比增速趋势
        verdict       : str   综合判断
        score         : float 杜邦评分 (-30 ~ +30)
    """
    if not financial_history or len(financial_history) < 2:
        return {"score": 0, "verdict": "数据不足", "roe_trend": "unknown"}

    # 优先使用年报
    annual = [r for r in financial_history if r.get("report_type") == "年报"]
    data = annual[:4] if len(annual) >= 2 else financial_history[:6]

    roes    = [r.get("roe") for r in data if r.get("roe") is not None]
    margins = [r.get("gross_margin") for r in data if r.get("gross_margin") is not None]
    growths = [r.get("profit_yoy") for r in data if r.get("profit_yoy") is not None]

    score = 0
    result = {}

    # ROE分析
    if roes:
        avg_roe = sum(roes[:3]) / min(3, len(roes))
        result["avg_roe"] = round(avg_roe, 2)
        if len(roes) >= 2:
            roe_slope = roes[0] - roes[-1]   # 最新 - 最早（倒序排列）
            if roe_slope > 2:
                result["roe_trend"] = "improving"
                score += 15
            elif roe_slope < -3:
                result["roe_trend"] = "declining"
                score -= 15
            else:
                result["roe_trend"] = "stable"
                score += 5 if avg_roe > 15 else 0
        else:
            result["roe_trend"] = "insufficient_data"

        # ROE绝对水平
        if avg_roe >= 20:
            score += 10
        elif avg_roe >= 15:
            score += 5
        elif avg_roe < 8:
            score -= 10

    # 毛利率趋势
    if margins and len(margins) >= 2:
        margin_slope = margins[0] - margins[-1]
        if margin_slope > 2:
            result["gross_trend"] = "improving"
            score += 5
        elif margin_slope < -3:
            result["gross_trend"] = "declining"
            score -= 8
            result["roe_quality"] = "⚠️ 毛利率下滑，ROE质量下降"
        else:
            result["gross_trend"] = "stable"
        result["avg_gross_margin"] = round(sum(margins[:3]) / min(3, len(margins)), 2)
        if margins[0] > 40:
            result["roe_quality"] = "✅ 高毛利率驱动ROE（质量优秀）"
        elif not result.get("roe_quality"):
            result["roe_quality"] = "高资产周转率或财务杠杆驱动"

    # 净利润增速趋势
    if growths and len(growths) >= 2:
        pos_growth = sum(1 for g in growths[:3] if g > 0)
        if pos_growth >= 3:
            result["profit_trend"] = "consistently_growing"
            score += 8
        elif pos_growth == 0:
            result["profit_trend"] = "declining_all"
            score -= 10
        else:
            result["profit_trend"] = "mixed"

    # 综合判断
    score = np.clip(score, -30, 30)
    if score >= 20:
        verdict = "🟢 优秀：ROE高且持续改善，盈利质量强"
    elif score >= 10:
        verdict = "🟡 良好：基本面稳健"
    elif score >= 0:
        verdict = "⚪ 一般：基本面无明显优势"
    elif score >= -10:
        verdict = "🟠 偏弱：ROE或毛利率有下滑迹象"
    else:
        verdict = "🔴 较差：盈利能力持续恶化"

    result["score"]   = round(score, 1)
    result["verdict"] = verdict
    return result


# =====================================================
# 9. Monte Carlo Simulation
# =====================================================

def monte_carlo_simulation(
    df: pd.DataFrame,
    days: int = 20,
    n_simulations: int = 5000,
) -> Dict:
    """
    Monte Carlo future price path simulation.

    Model: Geometric Brownian Motion (GBM)
        dS = mu * S * dt + sigma * S * dW

    Where:
        mu    = drift rate (based on historical daily return mean)
        sigma = volatility (based on historical daily return std)
        dW    = Wiener process increment ~ N(0, sqrt(dt))

    Returns:
        Price ranges for each probability interval, up/down probabilities,
        expected return intervals, etc.
    """
    close = df["close"].values
    n = len(close)

    if n < 30:
        return {"error": "Insufficient data, at least 30 bars required"}

    log_ret = np.log(close[1:] / close[:-1])
    mu  = np.mean(log_ret)
    sigma = np.std(log_ret, ddof=1)
    s0 = close[-1]

    # Generate simulation paths
    # 使用 Student-t 分布（自由度 df=5）替代正态分布
    # A股日收益率峰度远高于3，t分布更真实地捕捉极端行情概率
    # 标准化使得方差=1：z / sqrt(df/(df-2))
    dt = 1.0
    df_t = 5
    z_raw = np.random.standard_t(df_t, size=(n_simulations, days))
    z = z_raw / np.sqrt(df_t / (df_t - 2))   # 归一化到单位方差
    daily_returns = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    price_paths = np.zeros((n_simulations, days + 1))
    price_paths[:, 0] = s0
    for t in range(days):
        price_paths[:, t + 1] = price_paths[:, t] * daily_returns[:, t]

    final_prices = price_paths[:, -1]
    final_returns = (final_prices / s0 - 1) * 100

    # Statistics
    result = {
        "current_price": round(s0, 2),
        "simulation_days": days,
        "n_simulations": n_simulations,
        "mu_daily": round(mu * 100, 4),
        "sigma_daily": round(sigma * 100, 4),

        # Probability statistics
        "prob_up": round((final_prices > s0).mean() * 100, 1),
        "prob_down": round((final_prices < s0).mean() * 100, 1),
        "prob_up_5pct": round((final_returns > 5).mean() * 100, 1),
        "prob_down_5pct": round((final_returns < -5).mean() * 100, 1),
        "prob_up_10pct": round((final_returns > 10).mean() * 100, 1),
        "prob_down_10pct": round((final_returns < -10).mean() * 100, 1),

        # Price percentiles
        "price_5th": round(np.percentile(final_prices, 5), 2),
        "price_25th": round(np.percentile(final_prices, 25), 2),
        "price_median": round(np.percentile(final_prices, 50), 2),
        "price_75th": round(np.percentile(final_prices, 75), 2),
        "price_95th": round(np.percentile(final_prices, 95), 2),

        # Return percentiles
        "return_5th": round(np.percentile(final_returns, 5), 2),
        "return_25th": round(np.percentile(final_returns, 25), 2),
        "return_median": round(np.percentile(final_returns, 50), 2),
        "return_75th": round(np.percentile(final_returns, 75), 2),
        "return_95th": round(np.percentile(final_returns, 95), 2),

        "expected_return": round(np.mean(final_returns), 2),
        "max_loss_95": round(np.percentile(final_returns, 5), 2),
    }
    return result


# =====================================================
# 10. Comprehensive Diagnosis Scoring System
# =====================================================

# Factor weights (sum = 1.0) — default weights for random-walk regime
FACTOR_WEIGHTS = {
    "trend":              0.25,
    "momentum":           0.20,
    "volatility":         0.10,
    "mean_reversion":     0.15,
    "volume":             0.15,
    "support_resistance": 0.10,
    "statistics":         0.05,
}


def _get_adaptive_weights(hurst: float) -> Dict:
    """
    根据 Hurst 指数动态调整因子权重。

    H > 0.6 → 趋势市：强化趋势/动量权重，削减均值回归
    H < 0.4 → 震荡市：强化均值回归/支撑阻力，削减趋势/动量
    H 0.4~0.6 → 随机游走：使用默认权重

    原理：Hurst > 0.5 表示序列具有记忆性（趋势持续），
          Hurst < 0.5 表示序列有回归均值倾向（震荡反弹）。
          在不同市场状态下，用相同权重会导致错误信号被放大。
    """
    if hurst > 0.62:        # 趋势市
        return {
            "trend":              0.32,
            "momentum":           0.25,
            "volatility":         0.08,
            "mean_reversion":     0.07,
            "volume":             0.16,
            "support_resistance": 0.08,
            "statistics":         0.04,
        }
    elif hurst < 0.40:      # 震荡/均值回归市
        return {
            "trend":              0.10,
            "momentum":           0.12,
            "volatility":         0.10,
            "mean_reversion":     0.32,
            "volume":             0.16,
            "support_resistance": 0.16,
            "statistics":         0.04,
        }
    else:                   # 随机游走 — 使用默认权重
        return dict(FACTOR_WEIGHTS)


def comprehensive_diagnosis(
    df: pd.DataFrame,
    news_sentiment: float = 0.0,
    run_monte_carlo: bool = True,
    mc_days: int = 20,
) -> Dict:
    """
    Comprehensive diagnosis scoring.

    Aggregates all analysis dimensions into a weighted total score
    from -100 to +100, and provides a clear buy/hold/sell recommendation.

    Parameters:
        df               K-line DataFrame with OHLCV
        news_sentiment   News sentiment score (-1.0 ~ +1.0), provided by Claude
        run_monte_carlo  Whether to run Monte Carlo simulation
        mc_days          Monte Carlo simulation days

    Returns:
        {
          "total_score": -100 ~ +100,
          "signal": "strong_buy / buy / hold / sell / strong_sell",
          "confidence": "high/moderate/low",
          "factors": { per-dimension scores },
          "monte_carlo": { MC results },
          "risk_warnings": [ risk warning list ],
          "summary": "human-readable analysis summary"
        }
    """
    # -- Run all analysis modules --
    trend   = analyze_trend(df)
    momentum = analyze_momentum(df)
    vol     = analyze_volatility(df)
    mr      = analyze_mean_reversion(df)
    volume  = analyze_volume(df)
    sr      = analyze_support_resistance(df)
    stats   = analyze_statistics(df)

    # -- 自适应权重：根据 Hurst 指数动态调整各维度权重 --
    hurst_val = stats["details"].get("hurst", 0.5)
    w = _get_adaptive_weights(hurst_val)

    factors = {
        "trend":           {"score": trend["score"],    "weight": w["trend"],              "label": "趋势 Trend"},
        "momentum":        {"score": momentum["score"], "weight": w["momentum"],           "label": "动量 Momentum"},
        "volatility":      {"score": vol["score"],      "weight": w["volatility"],         "label": "波动率 Volatility"},
        "mean_reversion":  {"score": mr["score"],       "weight": w["mean_reversion"],     "label": "均值回归 MeanRev"},
        "volume":          {"score": volume["score"],   "weight": w["volume"],             "label": "量价 Volume-Price"},
        "support_resistance": {"score": sr["score"],   "weight": w["support_resistance"], "label": "支撑阻力 S/R"},
        "statistics":      {"score": stats["score"],   "weight": w["statistics"],         "label": "统计特征 Stats"},
    }

    # -- 加权总分 --
    weighted_sum = sum(f["score"] * f["weight"] for f in factors.values())

    # -- 新闻情绪修正 --
    news_adj = news_sentiment * 10   # -10 ~ +10
    if news_sentiment != 0:
        factors["news"] = {"score": news_adj, "weight": 0.0, "label": "新闻情绪 News(adj)"}

    total = np.clip(weighted_sum + news_adj, -100, 100)

    # ── 多条件买卖核查清单 ──────────────────────────────────────
    rsi_val         = momentum["details"].get("rsi_14", 50)
    macd_hist_val   = momentum["details"].get("macd_hist", 0)
    macd_cross_val  = momentum["details"].get("macd_cross", "")
    vol_ratio_val   = volume["details"].get("vol_ratio_5_20", 1.0)
    z_score_val     = mr["details"].get("z_score_20d", 0.0)
    pos_range_val   = sr.get("details", {}).get("position_in_range", 0.5)
    obv_sig_val     = volume["details"].get("obv_signal", "")
    vol_div_val     = volume["details"].get("vol_divergence", "")

    # 买入条件（7项，满足≥5项才触发买入信号）
    buy_conditions = {
        "趋势未下跌":    (trend["score"] > -10,
                          f"趋势分={trend['score']:.0f} (需>-10)"),
        "MACD未死叉":   (macd_cross_val != "death_cross" and macd_hist_val >= -0.002,
                          f"MACD柱={macd_hist_val:.3f} / {macd_cross_val or '无交叉'}"),
        "量能扩张稳定":  (vol_ratio_val > 0.90,
                          f"量比5/20日={vol_ratio_val:.2f} (需>0.90)"),
        "RSI未超买":    (rsi_val < 75,
                          f"RSI14={rsi_val:.1f} (需<75)"),
        "价格未高估":   (z_score_val < 2.0,
                          f"Z-score20d={z_score_val:.2f} (需<2.0)"),
        "非历史高位":   (pos_range_val < 0.80,
                          f"区间位置={pos_range_val:.2f} (需<0.80)"),
        "情绪未极空":   (news_sentiment > -0.30,
                          f"情绪值={news_sentiment:.2f} (需>-0.30)"),
    }
    buy_met = sum(1 for met, _ in buy_conditions.values() if met)

    # 卖出/风险条件（7项，触发≥3项则卖出信号）
    sell_conditions = {
        "强下跌趋势":   (trend["score"] < -35,
                          f"趋势分={trend['score']:.0f} (<-35触发)"),
        "MACD死叉":    (macd_cross_val == "death_cross",
                          f"MACD={macd_cross_val or '无死叉'}"),
        "RSI严重超买":  (rsi_val > 78,
                          f"RSI14={rsi_val:.1f} (>78触发)"),
        "历史高位":    (pos_range_val > 0.88,
                          f"区间位置={pos_range_val:.2f} (>0.88触发)"),
        "量价背离看空": ("bearish" in obv_sig_val or "bearish" in vol_div_val,
                          f"OBV={obv_sig_val[:25] if obv_sig_val else '正常'}"),
        "价格严重高估": (z_score_val > 2.3,
                          f"Z-score20d={z_score_val:.2f} (>2.3触发)"),
        "情绪极端看空": (news_sentiment < -0.40,
                          f"情绪值={news_sentiment:.2f} (<-0.40触发)"),
    }
    sell_met = sum(1 for met, _ in sell_conditions.values() if met)

    # -- 综合信号判定（多条件 + 分数双保险）--
    # 强买：≥6项买入条件 + 总分>15
    # 买：  ≥5项买入条件 + 总分>0 + 卖出触发<2
    # 强卖：≥5项卖出条件 OR 总分<-45
    # 卖：  ≥3项卖出条件 OR 总分<-15
    # 持有：其他情况
    if sell_met >= 5 or total < -45:
        signal = "strong_sell"
    elif sell_met >= 3 or total < -15:
        signal = "sell"
    elif buy_met >= 6 and total > 15 and sell_met == 0:
        signal = "strong_buy"
    elif buy_met >= 5 and total > 0 and sell_met <= 1:
        signal = "buy"
    else:
        signal = "hold"

    # -- Confidence assessment --
    # Based on: factor agreement + trend clarity + data quantity
    factor_scores = [f["score"] for f in factors.values() if f["weight"] > 0]
    score_signs = [np.sign(s) for s in factor_scores if s != 0]
    if score_signs:
        agreement = abs(sum(score_signs)) / len(score_signs)
    else:
        agreement = 0

    data_quality = min(len(df) / 120, 1.0)  # More data = more reliable
    trend_clear  = trend.get("trend_clarity", 0.5)
    conf_score   = agreement * 0.5 + data_quality * 0.3 + trend_clear * 0.2

    if conf_score > 0.7:
        confidence = "high"
    elif conf_score > 0.4:
        confidence = "moderate"
    else:
        confidence = "low"

    # -- Monte Carlo --
    mc_result = None
    if run_monte_carlo:
        mc_result = monte_carlo_simulation(df, days=mc_days)

    # -- Risk warnings --
    warnings = []
    if vol["risk_level"] in ("extreme", "high"):
        warnings.append(f"WARNING: Volatility {vol['risk_level']} (HV {vol['current_hv']:.0f}%), manage position size")
    if "rsi_divergence" in momentum["details"]:
        warnings.append(f"WARNING: RSI {momentum['details']['rsi_divergence']}")
    if "vol_divergence" in volume["details"]:
        warnings.append(f"WARNING: Volume {volume['details']['vol_divergence']}")
    if "macd_cross" in momentum["details"]:
        cross_type = momentum["details"]["macd_cross"]
        warnings.append(f"NOTE: MACD recent {cross_type} signal")
    if sr.get("details", {}).get("position_in_range", 0.5) > 0.85:
        warnings.append("WARNING: Price near recent highs, dense resistance overhead")
    elif sr.get("details", {}).get("position_in_range", 0.5) < 0.15:
        warnings.append("NOTE: Price near recent lows, strong support below")
    if mc_result and mc_result.get("prob_down_10pct", 0) > 30:
        warnings.append(f"WARNING: Monte Carlo: {mc_result['prob_down_10pct']}% probability of >10% decline in {mc_days} days")
    if stats["details"].get("kurtosis", 0) > 3:
        warnings.append("WARNING: Fat-tail return distribution, extreme events more likely")

    # -- Generate summary --
    name = df.attrs.get("name", "")
    code = df.attrs.get("code", "")
    current = df.iloc[-1]["close"]

    # 信号中文映射
    signal_cn = {
        "strong_buy":  "⭐强买入",
        "buy":         "✅买入",
        "hold":        "⏸持有观望",
        "sell":        "🔻卖出",
        "strong_sell": "🚨强力卖出",
    }
    # Hurst市场状态描述
    if hurst_val > 0.62:
        market_regime = f"趋势市(H={hurst_val:.3f}，权重已向趋势/动量倾斜)"
    elif hurst_val < 0.40:
        market_regime = f"震荡市(H={hurst_val:.3f}，权重已向均值回归倾斜)"
    else:
        market_regime = f"随机游走(H={hurst_val:.3f}，使用默认权重)"

    summary_lines = [
        f"【{name}({code}) 全维度量化诊断】",
        f"当前价格: {current:.2f}",
        f"综合得分: {total:.0f}/100  │  信号: {signal_cn.get(signal, signal)}  │  置信度: {confidence}",
        f"市场状态: {market_regime}",
        "",
        f"趋势: {trend['direction']} ({trend['strength']})  得分={trend['score']:.0f}",
        f"动量: 得分={momentum['score']:.0f}   RSI={momentum['details'].get('rsi_14', 'N/A')}",
        f"波动: {vol['risk_level']}   HV={vol['current_hv']:.0f}%",
        f"均值回归: {mr['signal']}  得分={mr['score']:.0f}",
        f"量价关系: {volume['details'].get('obv_signal', 'N/A')}",
    ]

    if sr["supports"]:
        summary_lines.append(f"Support Levels: {sr['supports'][:3]}")
    if sr["resistances"]:
        summary_lines.append(f"Resistance Levels: {sr['resistances'][:3]}")

    if mc_result:
        summary_lines.extend([
            "",
            f"Monte Carlo ({mc_days} days, {mc_result.get('n_simulations', 5000)} simulations):",
            f"  Up probability {mc_result['prob_up']:.0f}%  Down probability {mc_result['prob_down']:.0f}%",
            f"  Expected return {mc_result['expected_return']:+.1f}%",
            f"  95% confidence interval [{mc_result['return_5th']:+.1f}%, {mc_result['return_95th']:+.1f}%]",
            f"  Price range [{mc_result['price_5th']:.2f}, {mc_result['price_95th']:.2f}]",
        ])

    if warnings:
        summary_lines.append("")
        summary_lines.extend(warnings)

    # ── 买卖条件核查清单 ──────────────────────────────────────────
    buy_icon  = "✅" if buy_met >= 5 else ("⚠️" if buy_met >= 3 else "❌")
    sell_icon = "🚨" if sell_met >= 3 else ("⚠️" if sell_met >= 2 else "✅")

    summary_lines.extend([
        "",
        f"━━━ 买入条件核查 ({buy_met}/{len(buy_conditions)}) {buy_icon} ━━━",
    ])
    for cond_name, (met, desc) in buy_conditions.items():
        mark = "✓" if met else "✗"
        summary_lines.append(f"  {mark} {cond_name}: {desc}")

    summary_lines.extend([
        "",
        f"━━━ 风险/卖出条件核查 ({sell_met}/{len(sell_conditions)}) {sell_icon} ━━━",
    ])
    for cond_name, (met, desc) in sell_conditions.items():
        mark = "⚠" if met else "·"
        summary_lines.append(f"  {mark} {cond_name}: {desc}")

    # 最终决策说明
    summary_lines.extend([
        "",
        f"【最终建议】{signal_cn.get(signal, signal)}",
        f"  买入条件满足 {buy_met}/7，风险条件触发 {sell_met}/7",
        f"  决策规则: 买入需≥5项买入条件+总分>0; 卖出需≥3项风险条件或总分<-15",
        "",
        "免责声明: 以上分析基于数学模型，不构成投资建议，投资有风险，入市须谨慎。",
    ])

    return {
        "total_score":    round(total, 1),
        "signal":         signal,
        "confidence":     confidence,
        "hurst":          round(hurst_val, 3),
        "market_regime":  market_regime,
        "buy_met":        buy_met,
        "sell_met":       sell_met,
        "buy_conditions":  {k: {"met": v, "desc": d} for k, (v, d) in buy_conditions.items()},
        "sell_conditions": {k: {"met": v, "desc": d} for k, (v, d) in sell_conditions.items()},
        "factors":        factors,
        "trend_detail":   trend,
        "momentum_detail":    momentum,
        "volatility_detail":  vol,
        "mean_reversion_detail": mr,
        "volume_detail":      volume,
        "support_resistance_detail": sr,
        "statistics_detail":  stats,
        "monte_carlo":    mc_result,
        "risk_warnings":  warnings,
        "summary":        "\n".join(summary_lines),
    }
