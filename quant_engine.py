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
# 8b. Beta / 相对强弱 / Sortino / 下行风险分析
# =====================================================

def analyze_relative_strength(
    df: pd.DataFrame,
    benchmark_close: Optional[np.ndarray] = None,
) -> Dict:
    """
    相对强弱与风险调整收益分析。

    Parameters
    ----------
    df : pd.DataFrame
        个股日K线 (需含 close 列)
    benchmark_close : np.ndarray, optional
        基准指数收盘价序列 (与 df 等长或更长)。
        None 时跳过 Beta/相对动量计算，仅算 Sortino 等绝对指标。

    Returns
    -------
    dict:
        score      : float  综合评分 (-100 ~ +100)
        details    : dict   所有计算明细
    """
    close = df["close"].values.astype(float)
    n = len(close)
    details: Dict = {}

    if n < 30:
        return {"score": 0, "details": {"error": "数据不足30天"}}

    log_ret = np.log(close[1:] / close[:-1])
    score = 0.0

    # ── 1. Sortino Ratio (下行风险调整收益) ──
    # 只惩罚下行波动，上行波动不算风险
    mean_ret = np.mean(log_ret)
    downside = log_ret[log_ret < 0]
    downside_dev = np.std(downside, ddof=1) * np.sqrt(252) if len(downside) > 1 else 1e-10
    ann_ret = mean_ret * 252
    sortino = ann_ret / downside_dev if downside_dev > 0 else 0
    details["sortino_ratio"] = round(sortino, 3)
    details["downside_deviation_ann"] = round(downside_dev * 100, 2)

    if sortino > 1.5:
        score += 20
        details["sortino_grade"] = "优秀(>1.5)"
    elif sortino > 0.8:
        score += 10
        details["sortino_grade"] = "良好(>0.8)"
    elif sortino > 0:
        score += 0
        details["sortino_grade"] = "一般(>0)"
    else:
        score -= 15
        details["sortino_grade"] = "差(<0)"

    # ── 2. 最大回撤 & 回撤恢复时间 ──
    cum_max = np.maximum.accumulate(close)
    drawdown = (close - cum_max) / cum_max
    max_dd = float(np.min(drawdown))
    max_dd_idx = int(np.argmin(drawdown))
    details["max_drawdown_pct"] = round(max_dd * 100, 2)

    # 回撤恢复时间: 从最大回撤点到恢复创新高需要多少天
    recovery_days = None
    if max_dd_idx < n - 1:
        peak_val = cum_max[max_dd_idx]
        for j in range(max_dd_idx + 1, n):
            if close[j] >= peak_val:
                recovery_days = j - max_dd_idx
                break
    details["recovery_days"] = recovery_days
    details["recovery_interp"] = (
        f"{recovery_days}天恢复" if recovery_days
        else "尚未恢复" if max_dd < -0.05 else "回撤轻微"
    )

    if max_dd > -0.10:
        score += 10
    elif max_dd > -0.20:
        score += 0
    elif max_dd > -0.30:
        score -= 10
    else:
        score -= 20

    # ── 3. Calmar Ratio (年化收益 / 最大回撤) ──
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 0.001 else 0
    details["calmar_ratio"] = round(calmar, 3)

    # ── 4. 相对动量 (1/3/6个月 vs 基准) ──
    if benchmark_close is not None and len(benchmark_close) >= n:
        bm = benchmark_close[-n:].astype(float)
        bm_ret = np.log(bm[1:] / bm[:-1])

        # Beta (60日滚动)
        window_beta = min(60, len(log_ret))
        stock_w = log_ret[-window_beta:]
        bm_w = bm_ret[-window_beta:]
        cov_mat = np.cov(stock_w, bm_w)
        beta = cov_mat[0, 1] / cov_mat[1, 1] if cov_mat[1, 1] > 0 else 1.0
        details["beta_60d"] = round(beta, 3)

        # 特异性波动率 (idiosyncratic risk)
        residual = stock_w - beta * bm_w
        idio_vol = float(np.std(residual, ddof=1)) * np.sqrt(252) * 100
        details["idiosyncratic_vol_ann"] = round(idio_vol, 2)

        # Alpha (Jensen's Alpha)
        rf_daily = 0.02 / 252  # 假设无风险利率2%
        alpha = (np.mean(stock_w) - rf_daily) - beta * (np.mean(bm_w) - rf_daily)
        details["alpha_daily"] = round(alpha * 10000, 2)  # 万分之几
        details["alpha_ann_pct"] = round(alpha * 252 * 100, 2)

        # 相对动量: 不同期限的超额收益
        for label, days in [("1m", 20), ("3m", 60), ("6m", 120)]:
            if n > days:
                stock_period_ret = (close[-1] / close[-days - 1] - 1) * 100
                bm_period_ret = (bm[-1] / bm[-days - 1] - 1) * 100
                excess = stock_period_ret - bm_period_ret
                details[f"relative_momentum_{label}"] = round(excess, 2)

        # 相对强弱评分
        rm_1m = details.get("relative_momentum_1m", 0)
        rm_3m = details.get("relative_momentum_3m", 0)
        if rm_1m > 5 and rm_3m > 10:
            score += 15
            details["relative_strength"] = "强势(短中期均跑赢)"
        elif rm_1m > 0 and rm_3m > 0:
            score += 5
            details["relative_strength"] = "偏强(跑赢大盘)"
        elif rm_1m < -5 and rm_3m < -10:
            score -= 15
            details["relative_strength"] = "弱势(短中期均跑输)"
        else:
            details["relative_strength"] = "中性"

        # Beta评分: 牛市偏好高Beta，熊市偏好低Beta
        # 这里给中性评分，由环境层调整
        if beta > 1.5:
            details["beta_interp"] = "高Beta(市场放大器)"
        elif beta < 0.5:
            details["beta_interp"] = "低Beta(防御性)"
        else:
            details["beta_interp"] = "中性Beta"
    else:
        details["beta_note"] = "无基准数据，跳过相对分析"

    # ── 5. 上行/下行捕获率 ──
    if benchmark_close is not None and len(benchmark_close) >= n:
        bm = benchmark_close[-n:].astype(float)
        bm_ret = np.log(bm[1:] / bm[:-1])
        up_days = bm_ret > 0
        down_days = bm_ret < 0
        if np.sum(up_days) > 5 and np.sum(down_days) > 5:
            up_capture = np.mean(log_ret[up_days]) / np.mean(bm_ret[up_days])
            down_capture = np.mean(log_ret[down_days]) / np.mean(bm_ret[down_days])
            details["upside_capture"] = round(up_capture, 3)
            details["downside_capture"] = round(down_capture, 3)
            # 理想: 上行捕获>1, 下行捕获<1
            if up_capture > 1.0 and down_capture < 1.0:
                score += 10
                details["capture_quality"] = "优质(涨多跌少)"
            elif up_capture < 0.8 and down_capture > 1.2:
                score -= 10
                details["capture_quality"] = "差(涨少跌多)"

    score = float(np.clip(score, -100, 100))
    return {"score": round(score, 1), "details": details}


# =====================================================
# 8c. 行业相对估值 & 板块轮动信号
# =====================================================

def analyze_sector_relative(
    pe_ttm: Optional[float] = None,
    pb: Optional[float] = None,
    sector_pe_median: Optional[float] = None,
    sector_pb_median: Optional[float] = None,
    sector_avg_pct_chg_20d: Optional[float] = None,
    stock_pct_chg_20d: Optional[float] = None,
    sector_name: str = "",
) -> Dict:
    """
    行业相对估值与板块轮动分析。

    Parameters
    ----------
    pe_ttm, pb            : 个股估值
    sector_pe_median, sector_pb_median : 行业中位数估值
    sector_avg_pct_chg_20d : 板块20日平均涨跌幅
    stock_pct_chg_20d      : 个股20日涨跌幅
    sector_name            : 所属板块名称

    Returns
    -------
    dict with score, details
    """
    details: Dict = {"sector_name": sector_name}
    score = 0.0

    # ── 1. PE 相对估值 ──
    if pe_ttm and sector_pe_median and sector_pe_median > 0:
        pe_ratio = pe_ttm / sector_pe_median
        details["pe_vs_sector"] = round(pe_ratio, 3)
        if pe_ratio < 0.7:
            score += 15
            details["pe_relative"] = "显著低估(PE低于行业30%)"
        elif pe_ratio < 0.9:
            score += 5
            details["pe_relative"] = "轻度低估"
        elif pe_ratio > 1.5:
            score -= 15
            details["pe_relative"] = "显著高估(PE高于行业50%)"
        elif pe_ratio > 1.2:
            score -= 5
            details["pe_relative"] = "轻度高估"
        else:
            details["pe_relative"] = "接近行业中位数"

    # ── 2. PB 相对估值 ──
    if pb and sector_pb_median and sector_pb_median > 0:
        pb_ratio = pb / sector_pb_median
        details["pb_vs_sector"] = round(pb_ratio, 3)
        if pb_ratio < 0.6:
            score += 10
            details["pb_relative"] = "PB显著低于行业"
        elif pb_ratio > 1.5:
            score -= 10
            details["pb_relative"] = "PB显著高于行业"

    # ── 3. 板块轮动信号 ──
    if sector_avg_pct_chg_20d is not None and stock_pct_chg_20d is not None:
        # 个股 vs 板块
        excess = stock_pct_chg_20d - sector_avg_pct_chg_20d
        details["stock_vs_sector_20d"] = round(excess, 2)

        if sector_avg_pct_chg_20d > 3:
            details["sector_momentum"] = "板块上行(轮动利好)"
            if excess > 0:
                score += 10
                details["sector_position"] = "板块龙头(跑赢板块)"
            else:
                score += 3
                details["sector_position"] = "板块跟随(跑输板块)"
        elif sector_avg_pct_chg_20d < -3:
            details["sector_momentum"] = "板块下行(轮动利空)"
            if excess > 0:
                score += 5
                details["sector_position"] = "逆势抗跌"
            else:
                score -= 10
                details["sector_position"] = "随板块下跌"
        else:
            details["sector_momentum"] = "板块平稳"

    score = float(np.clip(score, -100, 100))
    return {"score": round(score, 1), "details": details}


# =====================================================
# 8d. 财务质量深度分析 (应计异常/营运效率/ROIC)
# =====================================================

def analyze_financial_quality(
    total_assets: Optional[float] = None,
    net_assets: Optional[float] = None,
    revenue: Optional[float] = None,
    net_profit: Optional[float] = None,
    operating_cashflow: Optional[float] = None,
    accounts_receivable: Optional[float] = None,
    inventory: Optional[float] = None,
    accounts_payable: Optional[float] = None,
    prev_receivable: Optional[float] = None,
    prev_inventory: Optional[float] = None,
    prev_payable: Optional[float] = None,
    total_debt: Optional[float] = None,
    cash_and_equivalents: Optional[float] = None,
    ebit: Optional[float] = None,
    tax_rate: float = 0.25,
) -> Dict:
    """
    财务质量深度分析——超越简单PE/PB的应计质量和营运效率。

    计算因子:
      1. 应计比率 (Accrual Ratio): 检测盈余操纵
      2. 现金利润比 (Cash/Earnings): 利润含金量
      3. 应收账款周转率: 收款效率
      4. 存货周转率: 去库存效率
      5. 现金转换周期 (CCC): 运营资金效率
      6. ROIC (已投资本回报): 真实盈利能力
      7. 净负债率: 杠杆安全
    """
    details: Dict = {}
    score = 0.0

    # ── 1. 应计比率 ──
    # Accrual = (Δ应收 + Δ存货 - Δ应付) / 总资产
    # 越高 → 利润越多来自应计而非现金 → 质量越差
    if (total_assets and total_assets > 0
            and prev_receivable is not None and accounts_receivable is not None
            and prev_inventory is not None and inventory is not None
            and prev_payable is not None and accounts_payable is not None):
        delta_ar = accounts_receivable - prev_receivable
        delta_inv = inventory - prev_inventory
        delta_ap = accounts_payable - prev_payable
        accrual = (delta_ar + delta_inv - delta_ap) / total_assets
        details["accrual_ratio"] = round(accrual * 100, 3)
        if accrual < 0.02:
            score += 10
            details["accrual_quality"] = "优(应计低，利润含金量高)"
        elif accrual < 0.05:
            score += 3
            details["accrual_quality"] = "一般"
        elif accrual > 0.10:
            score -= 15
            details["accrual_quality"] = "差(应计过高，盈余操纵风险)"
        else:
            score -= 5
            details["accrual_quality"] = "偏高"

    # ── 2. 现金利润比 ──
    if operating_cashflow is not None and net_profit and net_profit > 0:
        cash_earnings = operating_cashflow / net_profit
        details["cash_earnings_ratio"] = round(cash_earnings, 3)
        if cash_earnings > 1.0:
            score += 10
            details["cash_quality"] = "优(经营现金流>净利润)"
        elif cash_earnings > 0.7:
            score += 3
            details["cash_quality"] = "良好"
        elif cash_earnings < 0.3:
            score -= 15
            details["cash_quality"] = "差(利润含金量低，警惕)"
        else:
            score -= 5
            details["cash_quality"] = "偏低"

    # ── 3. 应收账款周转率 ──
    if revenue and accounts_receivable and accounts_receivable > 0:
        ar_turnover = revenue / accounts_receivable
        dso = 365 / ar_turnover if ar_turnover > 0 else 999
        details["ar_turnover"] = round(ar_turnover, 2)
        details["dso_days"] = round(dso, 1)
        if dso < 60:
            score += 5
        elif dso > 180:
            score -= 10
            details["dso_warning"] = "应收账款回收过慢(>180天)"

    # ── 4. 存货周转率 ──
    if revenue and inventory and inventory > 0:
        inv_turnover = revenue / inventory
        dio = 365 / inv_turnover if inv_turnover > 0 else 999
        details["inventory_turnover"] = round(inv_turnover, 2)
        details["dio_days"] = round(dio, 1)
        if dio < 60:
            score += 5
        elif dio > 200:
            score -= 5
            details["dio_warning"] = "存货周转过慢(>200天)"

    # ── 5. 现金转换周期 (CCC = DSO + DIO - DPO) ──
    dso_val = details.get("dso_days")
    dio_val = details.get("dio_days")
    if dso_val and dio_val and revenue and accounts_payable and accounts_payable > 0:
        dpo = 365 / (revenue / accounts_payable) if accounts_payable > 0 else 0
        details["dpo_days"] = round(dpo, 1)
        ccc = dso_val + dio_val - dpo
        details["cash_conversion_cycle"] = round(ccc, 1)
        if ccc < 30:
            score += 5
            details["ccc_quality"] = "优秀(资金周转快)"
        elif ccc > 150:
            score -= 10
            details["ccc_quality"] = "差(资金占用严重)"

    # ── 6. ROIC (投入资本回报率) ──
    # ROIC = NOPAT / Invested Capital
    # NOPAT = EBIT × (1 - tax_rate)
    # Invested Capital = Total Debt + Equity - Cash
    if (ebit is not None and net_assets and total_debt is not None
            and cash_and_equivalents is not None):
        nopat = ebit * (1 - tax_rate)
        invested_capital = total_debt + net_assets - cash_and_equivalents
        if invested_capital > 0:
            roic = nopat / invested_capital * 100
            details["roic_pct"] = round(roic, 2)
            if roic > 15:
                score += 15
                details["roic_quality"] = "优秀(>15%，强护城河)"
            elif roic > 10:
                score += 8
                details["roic_quality"] = "良好(>10%)"
            elif roic > 0:
                score += 0
                details["roic_quality"] = "一般"
            else:
                score -= 15
                details["roic_quality"] = "差(ROIC为负)"

    # ── 7. 净负债率 ──
    if total_debt is not None and cash_and_equivalents is not None and net_assets and net_assets > 0:
        net_debt_ratio = (total_debt - cash_and_equivalents) / net_assets * 100
        details["net_debt_ratio_pct"] = round(net_debt_ratio, 2)
        if net_debt_ratio < 0:
            score += 5
            details["leverage"] = "净现金(无净负债)"
        elif net_debt_ratio < 50:
            score += 0
            details["leverage"] = "杠杆适中"
        elif net_debt_ratio > 100:
            score -= 10
            details["leverage"] = "高杠杆(净负债>净资产)"

    score = float(np.clip(score, -100, 100))
    return {"score": round(score, 1), "details": details}


# =====================================================
# 8e. GARCH 波动率预测 & 波动率聚集检测
# =====================================================

def analyze_volatility_regime(df: pd.DataFrame, forecast_days: int = 5) -> Dict:
    """
    基于简化 GARCH(1,1) 的波动率体制分析。

    不依赖 scipy/arch 库，使用手动迭代实现。
    检测: 波动率聚集、体制切换、未来波动率预测。

    Parameters
    ----------
    df : pd.DataFrame  (需含 close 列)
    forecast_days : int  预测天数

    Returns
    -------
    dict with score, details
    """
    close = df["close"].values.astype(float)
    n = len(close)
    details: Dict = {}

    if n < 60:
        return {"score": 0, "details": {"error": "数据不足60天"}}

    log_ret = np.log(close[1:] / close[:-1])
    T = len(log_ret)

    # ── 1. 简化 GARCH(1,1) 参数估计 ──
    # σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    # 使用经验参数 (适合A股日频): ω=0.00001, α=0.08, β=0.90
    omega = 0.00001
    alpha = 0.08
    beta_g = 0.90

    sigma2 = np.zeros(T)
    sigma2[0] = np.var(log_ret[:20]) if T >= 20 else np.var(log_ret)
    for t in range(1, T):
        sigma2[t] = omega + alpha * log_ret[t - 1] ** 2 + beta_g * sigma2[t - 1]

    current_vol = np.sqrt(sigma2[-1]) * np.sqrt(252) * 100  # 年化%
    avg_vol = np.sqrt(np.mean(sigma2)) * np.sqrt(252) * 100
    details["garch_current_vol_ann"] = round(current_vol, 2)
    details["garch_avg_vol_ann"] = round(avg_vol, 2)

    # ── 2. 波动率预测 (未来N天) ──
    long_run_var = omega / (1 - alpha - beta_g) if (1 - alpha - beta_g) > 0 else sigma2[-1]
    forecast_var = sigma2[-1]
    forecast_vols = []
    for _ in range(forecast_days):
        forecast_var = omega + (alpha + beta_g) * forecast_var
        forecast_vols.append(np.sqrt(forecast_var) * np.sqrt(252) * 100)
    details["forecast_vol_5d"] = round(forecast_vols[-1], 2) if forecast_vols else 0
    details["long_run_vol_ann"] = round(np.sqrt(long_run_var) * np.sqrt(252) * 100, 2)

    # ── 3. 波动率体制判断 ──
    vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
    details["vol_regime_ratio"] = round(vol_ratio, 3)
    if vol_ratio > 1.5:
        details["vol_regime"] = "高波动体制(当前>>平均)"
        regime_score = -20
    elif vol_ratio > 1.2:
        details["vol_regime"] = "波动率升高"
        regime_score = -10
    elif vol_ratio < 0.7:
        details["vol_regime"] = "低波动体制(压缩中，可能爆发)"
        regime_score = 5  # 低波动后常有突破
    else:
        details["vol_regime"] = "正常波动"
        regime_score = 0

    # ── 4. 波动率聚集强度 ──
    # 自相关: |ε²_{t}| 与 |ε²_{t-1}| 的相关系数
    abs_ret = np.abs(log_ret)
    if T > 5:
        cluster_corr = np.corrcoef(abs_ret[1:], abs_ret[:-1])[0, 1]
        details["volatility_clustering"] = round(cluster_corr, 3)
        if cluster_corr > 0.3:
            details["clustering_interp"] = "强聚集(大波动后还有大波动)"
        elif cluster_corr > 0.15:
            details["clustering_interp"] = "中等聚集"
        else:
            details["clustering_interp"] = "弱聚集(波动较随机)"

    score = float(np.clip(regime_score, -100, 100))
    return {"score": round(score, 1), "details": details}


# =====================================================
# 8f. 多时间级共振 & 缺口分析
# =====================================================

def analyze_timeframe_harmony(df: pd.DataFrame) -> Dict:
    """
    多时间级共振分析 — 日/周/月线趋势一致性。

    原理: 当短期(5日)、中期(20日)、长期(60日)趋势方向一致时，
    信号可靠度大幅提升。这是通达信/同花顺等专业软件的核心功能。

    同时检测缺口(跳空)形态:
      - 突破缺口(伴随放量): 趋势确认
      - 竭尽缺口(高位放量): 反转警告
      - 普通缺口: 通常被回补

    Parameters
    ----------
    df : pd.DataFrame  (需含 open/high/low/close/volume 列)

    Returns
    -------
    dict with score, details
    """
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    opn = df["open"].values.astype(float)
    volume = df["volume"].values.astype(float)
    n = len(close)
    details: Dict = {}
    score = 0.0

    if n < 60:
        return {"score": 0, "details": {"error": "数据不足60天"}}

    # ── 1. 多时间级趋势判断 ──
    # 用线性回归斜率判断不同周期的趋势方向
    def trend_direction(series, window):
        if len(series) < window:
            return 0, 0
        seg = series[-window:]
        slope, _, r2 = linear_regression(seg)
        daily_pct = slope / np.mean(seg) * 100 if np.mean(seg) > 0 else 0
        return daily_pct, r2

    short_slope, short_r2 = trend_direction(close, 5)    # 周线级
    mid_slope, mid_r2 = trend_direction(close, 20)        # 月线级
    long_slope, long_r2 = trend_direction(close, 60)      # 季线级

    details["trend_5d_slope"] = round(short_slope, 4)
    details["trend_20d_slope"] = round(mid_slope, 4)
    details["trend_60d_slope"] = round(long_slope, 4)

    # 判断方向: >0.03%/日为上行, <-0.03%为下行, 否则横盘
    def classify(slope):
        if slope > 0.03:
            return "up"
        elif slope < -0.03:
            return "down"
        return "flat"

    s_dir = classify(short_slope)
    m_dir = classify(mid_slope)
    l_dir = classify(long_slope)
    details["tf_short"] = s_dir
    details["tf_mid"] = m_dir
    details["tf_long"] = l_dir

    # 共振评分
    directions = [s_dir, m_dir, l_dir]
    if directions.count("up") == 3:
        score += 25
        details["timeframe_harmony"] = "三线共振上行(强多头)"
    elif directions.count("down") == 3:
        score -= 25
        details["timeframe_harmony"] = "三线共振下行(强空头)"
    elif directions.count("up") == 2:
        score += 10
        details["timeframe_harmony"] = "双线上行(偏多)"
    elif directions.count("down") == 2:
        score -= 10
        details["timeframe_harmony"] = "双线下行(偏空)"
    elif s_dir == "up" and l_dir == "down":
        score += 0
        details["timeframe_harmony"] = "短多长空(反弹中)"
    elif s_dir == "down" and l_dir == "up":
        score -= 5
        details["timeframe_harmony"] = "短空长多(回调中)"
    else:
        details["timeframe_harmony"] = "方向混合(观望)"

    # 共振强度 = R²的平均值 (趋势明确度)
    avg_r2 = (short_r2 + mid_r2 + long_r2) / 3
    details["trend_clarity"] = round(avg_r2, 3)

    # ── 2. 缺口分析 (Gap Analysis) ──
    gaps = []
    avg_vol = np.mean(volume[-20:]) if n >= 20 else np.mean(volume)

    for i in range(-min(20, n - 1), 0):
        # 向上跳空: 今日Low > 昨日High
        if low[i] > high[i - 1]:
            gap_pct = (low[i] - high[i - 1]) / high[i - 1] * 100
            vol_surge = volume[i] / avg_vol if avg_vol > 0 else 1
            gap_type = "突破缺口" if vol_surge > 1.5 else "普通缺口"
            gaps.append({"dir": "up", "pct": round(gap_pct, 2),
                         "type": gap_type, "vol_ratio": round(vol_surge, 2)})
        # 向下跳空: 今日High < 昨日Low
        elif high[i] < low[i - 1]:
            gap_pct = (low[i - 1] - high[i]) / low[i - 1] * 100
            vol_surge = volume[i] / avg_vol if avg_vol > 0 else 1
            gap_type = "恐慌缺口" if vol_surge > 1.5 else "普通缺口"
            gaps.append({"dir": "down", "pct": round(gap_pct, 2),
                         "type": gap_type, "vol_ratio": round(vol_surge, 2)})

    details["recent_gaps"] = gaps
    details["gap_count"] = len(gaps)

    # 缺口评分
    for g in gaps:
        if g["dir"] == "up" and g["type"] == "突破缺口":
            score += 8
        elif g["dir"] == "down" and g["type"] == "恐慌缺口":
            score -= 12

    # ── 3. K线实体/影线比分析 ──
    if n >= 10:
        recent_bodies = []
        recent_wicks = []
        for i in range(-10, 0):
            body = abs(close[i] - opn[i])
            upper_wick = high[i] - max(close[i], opn[i])
            lower_wick = min(close[i], opn[i]) - low[i]
            total_range = high[i] - low[i]
            if total_range > 0:
                recent_bodies.append(body / total_range)
                recent_wicks.append((upper_wick + lower_wick) / total_range)

        if recent_bodies:
            avg_body_ratio = np.mean(recent_bodies)
            details["avg_body_ratio"] = round(avg_body_ratio, 3)
            if avg_body_ratio > 0.7:
                details["candle_character"] = "实体大(趋势明确)"
            elif avg_body_ratio < 0.3:
                details["candle_character"] = "影线长(犹豫不决)"
                score -= 5
            else:
                details["candle_character"] = "均衡"

    score = float(np.clip(score, -100, 100))
    return {"score": round(score, 1), "details": details}


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
    benchmark_close: Optional[np.ndarray] = None,
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

    # -- 新增维度: 相对强弱 / 多时间级共振 / GARCH波动率 --
    rel_str = analyze_relative_strength(df, benchmark_close=benchmark_close)
    tf_harm = analyze_timeframe_harmony(df)
    vol_reg = analyze_volatility_regime(df)

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
        "relative_strength": {"score": rel_str["score"], "weight": 0.0, "label": "相对强弱 RS",
                              "details": rel_str.get("details", {})},
        "timeframe_harmony": {"score": tf_harm["score"], "weight": 0.0, "label": "多时间级共振 TF",
                              "details": tf_harm.get("details", {})},
        "volatility_regime": {"score": vol_reg["score"], "weight": 0.0, "label": "波动率体制 GARCH",
                              "details": vol_reg.get("details", {})},
    }

    # -- 加权总分 --
    weighted_sum = sum(f["score"] * f["weight"] for f in factors.values())

    # -- 新闻情绪修正 --
    news_adj = news_sentiment * 10   # -10 ~ +10
    if news_sentiment != 0:
        factors["news"] = {"score": news_adj, "weight": 0.0, "label": "新闻情绪 News(adj)"}

    # -- 新增维度辅助修正 (weight=0, 但作为±调整) --
    # 多时间级共振: 三线同方向时 ±5, 双线 ±2
    tf_adj = tf_harm["score"] * 0.2  # 25*0.2=5, 10*0.2=2
    # 波动率体制: 高波动时惩罚, 低波动时中性
    vol_reg_adj = vol_reg["score"] * 0.15  # -20*0.15=-3
    # 相对强弱: 跑赢/跑输大盘
    rs_adj = rel_str["score"] * 0.10  # 15*0.1=1.5

    total = np.clip(weighted_sum + news_adj + tf_adj + vol_reg_adj + rs_adj, -100, 100)

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
    if vol.get("risk_level") in ("extreme", "high"):
        _hv = vol.get('current_hv', 0)
        warnings.append(f"WARNING: Volatility {vol['risk_level']} (HV {_hv:.0f}%), manage position size")
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
        f"趋势: {trend.get('direction', 'N/A')} ({trend.get('strength', 'N/A')})  得分={trend.get('score', 0):.0f}",
        f"动量: 得分={momentum.get('score', 0):.0f}   RSI={momentum.get('details', {}).get('rsi_14', 'N/A')}",
        f"波动: {vol.get('risk_level', 'N/A')}   HV={vol.get('current_hv', 0):.0f}%",
        f"均值回归: {mr.get('signal', 'N/A')}  得分={mr.get('score', 0):.0f}",
        f"量价关系: {volume['details'].get('obv_signal', 'N/A')}",
    ]

    # 新增维度摘要
    tf_detail = tf_harm.get("details", {})
    vol_reg_detail = vol_reg.get("details", {})
    rs_detail = rel_str.get("details", {})
    summary_lines.extend([
        f"多时间级: {tf_detail.get('timeframe_harmony', 'N/A')}  "
        f"(5d={tf_detail.get('tf_short','?')}/20d={tf_detail.get('tf_mid','?')}/60d={tf_detail.get('tf_long','?')})",
        f"波动体制: {vol_reg_detail.get('vol_regime', 'N/A')}  "
        f"GARCH年化={vol_reg_detail.get('garch_current_vol_ann', 'N/A')}%  "
        f"预测5日={vol_reg_detail.get('forecast_vol_5d', 'N/A')}%",
        f"Sortino: {rs_detail.get('sortino_ratio', 'N/A')}  "
        f"最大回撤: {rs_detail.get('max_drawdown_pct', 'N/A')}%  "
        f"Calmar: {rs_detail.get('calmar_ratio', 'N/A')}",
    ])
    if rs_detail.get("relative_strength"):
        summary_lines.append(
            f"相对强弱: {rs_detail['relative_strength']}  "
            f"Beta={rs_detail.get('beta_60d', 'N/A')}  "
            f"Alpha年化={rs_detail.get('alpha_ann_pct', 'N/A')}%"
        )
    if tf_detail.get("recent_gaps"):
        gaps = tf_detail["recent_gaps"]
        gap_strs = [f"{'↑' if g['dir']=='up' else '↓'}{g['pct']}%({g['type']})" for g in gaps[:3]]
        summary_lines.append(f"近期缺口: {', '.join(gap_strs)}")

    if sr["supports"]:
        summary_lines.append(f"Support Levels: {sr['supports'][:3]}")
    if sr["resistances"]:
        summary_lines.append(f"Resistance Levels: {sr['resistances'][:3]}")

    if mc_result and "error" not in mc_result:
        _mc_prob_up = mc_result.get('prob_up')
        _mc_prob_down = mc_result.get('prob_down')
        _mc_exp_ret = mc_result.get('expected_return')
        _mc_ret_5 = mc_result.get('return_5th')
        _mc_ret_95 = mc_result.get('return_95th')
        _mc_p5 = mc_result.get('price_5th')
        _mc_p95 = mc_result.get('price_95th')
        summary_lines.extend([
            "",
            f"Monte Carlo ({mc_days} days, {mc_result.get('n_simulations', 5000)} simulations):",
            f"  Up probability {_mc_prob_up:.0f}%  Down probability {_mc_prob_down:.0f}%"
            if _mc_prob_up is not None and _mc_prob_down is not None else "  Probability data unavailable",
            f"  Expected return {_mc_exp_ret:+.1f}%"
            if _mc_exp_ret is not None else "  Expected return N/A",
            f"  95% confidence interval [{_mc_ret_5:+.1f}%, {_mc_ret_95:+.1f}%]"
            if _mc_ret_5 is not None and _mc_ret_95 is not None else "  95% CI N/A",
            f"  Price range [{_mc_p5:.2f}, {_mc_p95:.2f}]"
            if _mc_p5 is not None and _mc_p95 is not None else "  Price range N/A",
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
        "relative_strength_detail": rel_str,
        "timeframe_harmony_detail": tf_harm,
        "volatility_regime_detail": vol_reg,
        "monte_carlo":    mc_result,
        "risk_warnings":  warnings,
        "summary":        "\n".join(summary_lines),
    }


# =====================================================
# 11. 低位建仓识别 — 四维信号同时成立
# =====================================================
#
# 维度一: 价格形态 (kline)
#   横盘或缓慢阴跌 / 区间反复震荡3-8周 / 振幅收窄(<3%) /
#   下跌时有支撑(不破MA) / MA20走平或微降
#
# 维度二: 资金特征 (moneyflow)
#   每日净流出小且稳定(散户卖出) / 大单净流出<0.2亿/天 /
#   20日累计净流出-1~-3亿 / 偶发单日大单净流入(试盘) /
#   流出量逐周递减(筹码趋稳)
#
# 维度三: 量能结构 (kline)
#   总体缩量(换手率1-3%) / 下跌日量小,上涨日量稍大 /
#   量价背离:价跌但量不放大 / 偶有温和放量后迅速回落(压价吸筹)
#
# 维度四: 筹码与情绪 (多源)
#   融资余额缓慢回升 / 股价处于近1年低30%分位 /
#   北向资金无大额净卖出 / 市场情绪冷淡(无人关注)
#
# 四维同时满足 → 高置信度建仓信号
# 三维满足     → 加入观察池等待确认
# 任一维度反向 → 排除


def detect_accumulation(
    df: pd.DataFrame,
    moneyflow: Optional[list] = None,
    margin_trend: Optional[str] = None,
    northbound_adj: float = 0.0,
    turnover_col: str = "turnover",
    float_mv: float = 0.0,
) -> Dict:
    """
    低位建仓四维识别。

    Parameters
    ----------
    df : pd.DataFrame
        日K线数据, 必须包含 date/open/high/low/close/volume 列,
        可选 turnover(换手率) 列。按日期升序排列。
    moneyflow : list of dict, optional
        资金流向数据 (来自 DB 或 API)。
        每条: {date, main_net, main_net_pct, super_net, big_net, ...}
        按日期降序（最新在前）。None 时维度二跳过。
    margin_trend : str, optional
        融资余额趋势: "up" / "flat" / "down"。None 时不参与评分。
    northbound_adj : float
        北向资金修正值（正=放量，负=缩量）
    turnover_col : str
        换手率列名
    float_mv : float
        流通市值（亿元），用于资金流向阈值的相对化计算。
        0 或未传时使用默认绝对阈值。

    Returns
    -------
    dict:
        dim1_price    : {score, max, signals, desc}
        dim2_money    : {score, max, signals, desc}
        dim3_volume   : {score, max, signals, desc}
        dim4_sentiment: {score, max, signals, desc}
        total_score   : float   四维总分(0-100)
        total_max     : int     满分
        dimensions_met: int     满足阈值的维度数(0-4)
        conclusion    : str     "HIGH" / "WATCH" / "NONE"
        price_info    : dict    支撑/阻力/入场区间/止损
        summary       : str     中文摘要
    """
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)
    n = len(close)

    # NaN 保护: 剔除尾部连续 NaN，确保核心数据可用
    if n > 0 and (np.isnan(close[-1]) or close[-1] <= 0):
        # 数据质量不足，返回空结果
        return {
            "dim1_price": {"score": 0, "max": 25, "signals": ["数据含NaN"]},
            "dim2_money": {"score": 0, "max": 25, "signals": []},
            "dim3_volume": {"score": 0, "max": 25, "signals": []},
            "dim4_sentiment": {"score": 0, "max": 25, "signals": []},
            "total_score": 0, "total_max": 100, "dimensions_met": 0,
            "conclusion": "NONE", "exclusions": [],
            "price_info": {}, "summary": "数据质量不足，无法进行建仓识别",
        }
    # 将 NaN 替换为前值（前向填充）
    for arr in [close, high, low, volume]:
        mask = np.isnan(arr)
        if mask.any():
            for i in range(1, len(arr)):
                if mask[i]:
                    arr[i] = arr[i - 1]

    has_turnover = turnover_col in df.columns
    turnover = df[turnover_col].values.astype(float) if has_turnover else None
    if turnover is not None:
        np.nan_to_num(turnover, copy=False, nan=0.0)

    # ══════════════════════════════════════════════
    # 维度一: 价格形态 (满分 25)
    # ══════════════════════════════════════════════
    d1_score = 0
    d1_max = 25
    d1_signals = []

    if n >= 30:
        # ── 1a. 横盘/缓慢阴跌 检测 ──
        # 用最近 40 天的线性回归斜率判断（斜率接近0=横盘，微负=缓跌）
        window = min(40, n)
        slope, intercept, r2 = linear_regression(close[-window:])
        # 归一化斜率: 每日变化占价格的百分比
        daily_pct = slope / np.mean(close[-window:]) * 100 if np.mean(close[-window:]) > 0 else 0

        if -0.15 <= daily_pct <= 0.05:
            d1_score += 7
            d1_signals.append(f"横盘整理(斜率{daily_pct:+.3f}%/日)")
        elif -0.30 <= daily_pct < -0.15:
            d1_score += 4
            d1_signals.append(f"缓慢阴跌(斜率{daily_pct:+.3f}%/日)")
        elif daily_pct < -0.30:
            d1_signals.append(f"下跌过快(斜率{daily_pct:+.3f}%/日) ✗")

        # ── 1b. 振幅收窄 ──
        # 最近 20 天的平均日振幅
        if n >= 20:
            daily_range = (high[-20:] - low[-20:]) / close[-20:] * 100
            avg_range = np.mean(daily_range)
            if avg_range < 3.0:
                d1_score += 6
                d1_signals.append(f"振幅收窄({avg_range:.1f}% < 3%)")
            elif avg_range < 4.5:
                d1_score += 3
                d1_signals.append(f"振幅适中({avg_range:.1f}%)")
            else:
                d1_signals.append(f"振幅偏大({avg_range:.1f}%) ✗")

        # ── 1c. MA20 走平或微降 ──
        if n >= 25:
            ma20 = np.convolve(close, np.ones(20)/20, mode="valid")
            if len(ma20) >= 5:
                ma20_recent = ma20[-5:]
                ma20_slope, _, _ = linear_regression(ma20_recent)
                ma20_pct = ma20_slope / ma20_recent[0] * 100 if ma20_recent[0] > 0 else 0
                if -0.10 <= ma20_pct <= 0.05:
                    d1_score += 6
                    d1_signals.append(f"MA20走平({ma20_pct:+.3f}%/日)")
                elif -0.20 <= ma20_pct < -0.10:
                    d1_score += 3
                    d1_signals.append(f"MA20微降({ma20_pct:+.3f}%/日)")
                else:
                    d1_signals.append(f"MA20偏陡({ma20_pct:+.3f}%/日) ✗")

        # ── 1d. 下跌时有支撑 ──
        # 最近 20 天的最低价 vs MA20，不跌破MA20太多
        if n >= 25 and len(ma20) >= 1:
            recent_low = np.min(low[-20:])
            current_ma20 = ma20[-1]
            breach_pct = (recent_low - current_ma20) / current_ma20 * 100 if current_ma20 > 0 else 0
            if breach_pct >= -2.0:
                d1_score += 6
                d1_signals.append(f"跌有支撑(最低距MA20 {breach_pct:+.1f}%)")
            elif breach_pct >= -5.0:
                d1_score += 3
                d1_signals.append(f"支撑尚可(最低距MA20 {breach_pct:+.1f}%)")
            else:
                d1_signals.append(f"破位明显(最低距MA20 {breach_pct:+.1f}%) ✗")

    # ══════════════════════════════════════════════
    # 维度二: 资金特征 (满分 25)
    # ══════════════════════════════════════════════
    d2_score = 0
    d2_max = 25
    d2_signals = []

    if moneyflow and len(moneyflow) >= 5:
        # moneyflow 按日期降序(最新在前)
        mf = moneyflow

        # ── 根据流通市值动态调整阈值 ──
        # float_mv 单位: 亿元。阈值 = 流通市值 × 百分比
        # 小盘(<50亿): daily ~0.1%, big ~0.1%, cum20 ~-1.5%, probe >0.05%
        # 中盘(50~300亿): daily ~0.04%, big ~0.04%, cum20 ~-0.6%, probe >0.02%
        # 大盘(>300亿): daily ~0.01%, big ~0.015%, cum20 ~-0.2%, probe >0.005%
        if float_mv and float_mv > 0:
            _mv = float(float_mv)
            th_daily_lo = -_mv * 0.001     # 日均净流出下限
            th_daily_hi = _mv * 0.0003     # 日均净流入上限
            th_daily_warn = -_mv * 0.003   # 日均流出警告
            th_big_ok = _mv * 0.001        # 大单可控
            th_big_warn = _mv * 0.003      # 大单偏大
            th_cum_lo = -_mv * 0.015       # 20日累计吸筹下限
            th_cum_hi = -_mv * 0.003       # 20日累计吸筹上限
            th_cum_balance = _mv * 0.003   # 接近平衡
            th_cum_warn = -_mv * 0.03      # 流出偏多
            th_probe = _mv * 0.0005        # 试盘阈值
            _mv_tag = f"[相对市值{_mv:.0f}亿]"
        else:
            # 降级为绝对阈值（兼容无市值数据的场景）
            th_daily_lo = -0.2
            th_daily_hi = 0.05
            th_daily_warn = -0.5
            th_big_ok = 0.2
            th_big_warn = 0.5
            th_cum_lo = -3.0
            th_cum_hi = -0.5
            th_cum_balance = 0.5
            th_cum_warn = -5.0
            th_probe = 0.1
            _mv_tag = "[绝对值]"

        # ── 2a. 每日净流出小且稳定 ──
        recent_10 = mf[:min(10, len(mf))]
        daily_nets = [d.get("main_net", 0) or 0 for d in recent_10]
        avg_daily_net = np.mean(daily_nets) if daily_nets else 0
        if th_daily_lo <= avg_daily_net <= th_daily_hi:
            d2_score += 5
            d2_signals.append(f"日均净流{avg_daily_net:+.2f}亿(小额稳定){_mv_tag}")
        elif th_daily_warn <= avg_daily_net < th_daily_lo:
            d2_score += 2
            d2_signals.append(f"日均净流{avg_daily_net:+.2f}亿(流出偏多)")
        else:
            d2_signals.append(f"日均净流{avg_daily_net:+.2f}亿 ✗")

        # ── 2b. 大单控盘度 ──
        big_nets = [abs(d.get("main_net", 0) or 0) for d in recent_10]
        max_big_outflow = max(big_nets) if big_nets else 0
        if max_big_outflow < th_big_ok:
            d2_score += 5
            d2_signals.append(f"大单可控(最大{max_big_outflow:.2f}亿)")
        elif max_big_outflow < th_big_warn:
            d2_score += 2
            d2_signals.append(f"大单偏大(最大{max_big_outflow:.2f}亿)")
        else:
            d2_signals.append(f"大单流出过大({max_big_outflow:.2f}亿) ✗")

        # ── 2c. 20日累计净流 ──
        recent_20 = mf[:min(20, len(mf))]
        cum_20 = sum(d.get("main_net", 0) or 0 for d in recent_20)
        if th_cum_lo <= cum_20 <= th_cum_hi:
            d2_score += 5
            d2_signals.append(f"20日累计{cum_20:+.2f}亿(典型吸筹区间)")
        elif th_cum_hi < cum_20 <= th_cum_balance:
            d2_score += 3
            d2_signals.append(f"20日累计{cum_20:+.2f}亿(接近平衡)")
        elif th_cum_warn <= cum_20 < th_cum_lo:
            d2_score += 1
            d2_signals.append(f"20日累计{cum_20:+.2f}亿(流出偏多)")
        else:
            d2_signals.append(f"20日累计{cum_20:+.2f}亿 ✗")

        # ── 2d. 偶发单日大单净流入(试盘) ──
        if len(mf) >= 10:
            inflow_days = [d for d in recent_10 if (d.get("main_net", 0) or 0) > th_probe]
            if 1 <= len(inflow_days) <= 3:
                d2_score += 5
                d2_signals.append(f"{len(inflow_days)}天出现大单试盘")
            elif len(inflow_days) == 0:
                d2_score += 2
                d2_signals.append("无明显试盘(可能更早期)")
            else:
                d2_signals.append(f"大单流入天数过多({len(inflow_days)}天) ✗")

        # ── 2e. 流出量逐周递减(筹码趋稳) ──
        if len(mf) >= 20:
            week1 = sum(abs(d.get("main_net", 0) or 0) for d in mf[:5])
            week2 = sum(abs(d.get("main_net", 0) or 0) for d in mf[5:10])
            week3 = sum(abs(d.get("main_net", 0) or 0) for d in mf[10:15])
            week4 = sum(abs(d.get("main_net", 0) or 0) for d in mf[15:20])
            # 最近一周 < 前一周 < 更早 → 递减
            if week1 < week2 and week2 < week3:
                d2_score += 5
                d2_signals.append(f"流出量逐周递减({week1:.1f}<{week2:.1f}<{week3:.1f})")
            elif week1 < week2:
                d2_score += 3
                d2_signals.append(f"近两周流出收窄({week1:.1f}<{week2:.1f})")
            else:
                d2_signals.append(f"流出未收窄({week1:.1f}>={week2:.1f}) ✗")
    else:
        d2_signals.append("无资金流向数据(需先运行 --moneyflow 拉取)")

    # ══════════════════════════════════════════════
    # 维度三: 量能结构 (满分 25)
    # ══════════════════════════════════════════════
    d3_score = 0
    d3_max = 25
    d3_signals = []

    if n >= 20:
        # ── 3a. 总体缩量(换手率1-3%) ──
        if has_turnover and turnover is not None:
            avg_turnover = np.mean(turnover[-20:])
            if 1.0 <= avg_turnover <= 3.0:
                d3_score += 7
                d3_signals.append(f"缩量换手{avg_turnover:.1f}%(1-3%区间)")
            elif 0.5 <= avg_turnover < 1.0:
                d3_score += 4
                d3_signals.append(f"极低换手{avg_turnover:.1f}%(偏冷)")
            elif 3.0 < avg_turnover <= 5.0:
                d3_score += 3
                d3_signals.append(f"换手偏高{avg_turnover:.1f}%")
            else:
                d3_signals.append(f"换手率{avg_turnover:.1f}% ✗")
        else:
            # 没有换手率数据时用成交量相对比来粗估
            vol_recent = np.mean(volume[-20:])
            vol_early = np.mean(volume[-60:-20]) if n >= 60 else vol_recent
            ratio = vol_recent / vol_early if vol_early > 0 else 1
            if 0.4 <= ratio <= 0.8:
                d3_score += 5
                d3_signals.append(f"成交量萎缩(近期/远期={ratio:.2f})")
            elif ratio < 0.4:
                d3_score += 3
                d3_signals.append(f"极度缩量(比值{ratio:.2f})")
            else:
                d3_signals.append(f"成交量未明显缩量(比值{ratio:.2f}) ✗")

        # ── 3b. 阴日量小，阳日量稍大 ──
        returns_20 = np.diff(close[-21:]) / close[-21:-1]
        vol_20 = volume[-20:]
        up_mask = returns_20 > 0
        down_mask = returns_20 < 0
        if np.sum(up_mask) > 0 and np.sum(down_mask) > 0:
            avg_up_vol = np.mean(vol_20[up_mask])
            avg_down_vol = np.mean(vol_20[down_mask])
            ratio = avg_up_vol / avg_down_vol if avg_down_vol > 0 else 1
            if ratio > 1.1:
                d3_score += 6
                d3_signals.append(f"阳日量>阴日量(比{ratio:.2f})")
            elif ratio > 0.9:
                d3_score += 3
                d3_signals.append(f"阴阳日量接近(比{ratio:.2f})")
            else:
                d3_signals.append(f"阴日量反而更大(比{ratio:.2f}) ✗")

        # ── 3c. 量价背离: 价跌但量不放大 ──
        if n >= 30:
            price_chg_20 = (close[-1] - close[-20]) / close[-20] * 100
            vol_chg_20 = (np.mean(volume[-5:]) - np.mean(volume[-20:])) / np.mean(volume[-20:]) * 100 if np.mean(volume[-20:]) > 0 else 0
            if price_chg_20 < -2 and vol_chg_20 < 10:
                d3_score += 6
                d3_signals.append(f"量价背离(价{price_chg_20:+.1f}%,量{vol_chg_20:+.1f}%)")
            elif price_chg_20 < 0 and vol_chg_20 < 5:
                d3_score += 3
                d3_signals.append(f"温和背离(价{price_chg_20:+.1f}%,量{vol_chg_20:+.1f}%)")
            else:
                d3_signals.append(f"无明显背离(价{price_chg_20:+.1f}%,量{vol_chg_20:+.1f}%)")

        # ── 3d. 偶有温和放量后迅速回落(压价吸筹) ──
        if n >= 20:
            vol_ma = np.mean(volume[-20:])
            spike_count = 0
            for i in range(-15, -1):
                # 某日放量 > 均量1.5倍，次日缩回 < 均量1.2倍
                if volume[i] > vol_ma * 1.5 and volume[i+1] < vol_ma * 1.2:
                    spike_count += 1
            if 1 <= spike_count <= 3:
                d3_score += 6
                d3_signals.append(f"检测到{spike_count}次放量后缩量(吸筹特征)")
            elif spike_count == 0:
                d3_score += 2
                d3_signals.append("无明显放量回落")
            else:
                d3_signals.append(f"放量过于频繁({spike_count}次) ✗")

    # ══════════════════════════════════════════════
    # 维度四: 筹码与情绪 (满分 25)
    # ══════════════════════════════════════════════
    d4_score = 0
    d4_max = 25
    d4_signals = []

    # ── 4a. 股价处于近1年低30%分位 ──
    if n >= 60:
        lookback_1y = min(250, n)
        high_1y = np.max(high[-lookback_1y:])
        low_1y = np.min(low[-lookback_1y:])
        percentile = (close[-1] - low_1y) / (high_1y - low_1y) * 100 if (high_1y - low_1y) > 0 else 50
        if percentile <= 30:
            d4_score += 8
            d4_signals.append(f"价格分位{percentile:.0f}%(低于30%)")
        elif percentile <= 50:
            d4_score += 4
            d4_signals.append(f"价格分位{percentile:.0f}%(中低位)")
        else:
            d4_signals.append(f"价格分位{percentile:.0f}%(偏高位) ✗")

    # ── 4b. 融资余额缓慢回升 ──
    if margin_trend == "up":
        d4_score += 6
        d4_signals.append("融资余额回升(聪明钱进场)")
    elif margin_trend == "flat":
        d4_score += 3
        d4_signals.append("融资余额持平")
    elif margin_trend == "down":
        d4_signals.append("融资余额下降 ✗")
    else:
        d4_signals.append("无融资融券数据")

    # ── 4c. 北向资金无大额净卖出 ──
    if northbound_adj >= 0:
        d4_score += 5
        d4_signals.append(f"北向资金正常(修正{northbound_adj:+.2f})")
    elif northbound_adj > -0.05:
        d4_score += 3
        d4_signals.append(f"北向资金微缩(修正{northbound_adj:+.2f})")
    else:
        d4_signals.append(f"北向资金明显缩量(修正{northbound_adj:+.2f}) ✗")

    # ── 4d. 市场情绪冷淡(无人关注) — 用换手率极低来代理 ──
    if has_turnover and turnover is not None and n >= 10:
        recent_turnover = np.mean(turnover[-10:])
        if recent_turnover < 2.0:
            d4_score += 6
            d4_signals.append(f"冷门低关注(换手{recent_turnover:.1f}%)")
        elif recent_turnover < 4.0:
            d4_score += 3
            d4_signals.append(f"关注度适中(换手{recent_turnover:.1f}%)")
        else:
            d4_signals.append(f"关注度偏高(换手{recent_turnover:.1f}%) ✗")
    else:
        d4_score += 2  # 无数据时给个中性分
        d4_signals.append("无换手率数据")

    # ══════════════════════════════════════════════
    # 综合评定
    # ══════════════════════════════════════════════
    total_score = d1_score + d2_score + d3_score + d4_score
    total_max = d1_max + d2_max + d3_max + d4_max

    # 每个维度达到 60% 视为"满足"
    dims_met = 0
    if d1_score >= d1_max * 0.6:
        dims_met += 1
    if d2_score >= d2_max * 0.6:
        dims_met += 1
    if d3_score >= d3_max * 0.6:
        dims_met += 1
    if d4_score >= d4_max * 0.6:
        dims_met += 1

    if dims_met >= 4:
        conclusion = "HIGH"
    elif dims_met >= 3:
        conclusion = "WATCH"
    else:
        conclusion = "NONE"

    conclusion_labels = {
        "HIGH": "高置信度建仓信号 — 四维同时满足",
        "WATCH": "加入观察池 — 三维满足，等待确认",
        "NONE": "暂不符合建仓条件",
    }

    # ══════════════════════════════════════════════
    # 计算关键价位（支撑/阻力/建议入场区间）
    # ══════════════════════════════════════════════
    current_price = float(close[-1]) if n > 0 else 0.0

    # 支撑位: 近30日最低点 与 MA20 取较低者
    support_price = 0.0
    if n >= 20:
        ma20_val = float(np.mean(close[-20:]))
        recent_low = float(np.min(low[-30:])) if n >= 30 else float(np.min(low[-20:]))
        support_price = round(min(ma20_val, recent_low), 2)

    # 阻力位: 近30日最高点 与 MA60 取较高者
    resistance_price = 0.0
    if n >= 20:
        recent_high = float(np.max(high[-30:])) if n >= 30 else float(np.max(high[-20:]))
        ma60_val = float(np.mean(close[-60:])) if n >= 60 else float(np.mean(close[-20:]))
        resistance_price = round(max(recent_high, ma60_val), 2)

    # 建议入场区间: 支撑位 ~ 支撑位+幅度的30%
    entry_low = support_price
    entry_high = round(support_price + (resistance_price - support_price) * 0.3, 2) if resistance_price > support_price else support_price

    # 止损位: 支撑位下方3%
    stop_loss = round(support_price * 0.97, 2) if support_price > 0 else 0.0

    price_info = {
        "current_price": current_price,
        "support_price": support_price,
        "resistance_price": resistance_price,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop_loss": stop_loss,
    }

    # ══════════════════════════════════════════════
    # 中文摘要
    # ══════════════════════════════════════════════
    lines = [
        "┌─────────────────────────────────┐",
        "│     低位建仓识别 · 四维分析       │",
        "└─────────────────────────────────┘",
        "",
        f"  维度一 价格形态:  {d1_score:>2}/{d1_max}  {'✓' if d1_score >= d1_max * 0.6 else '✗'}",
    ]
    for s in d1_signals:
        lines.append(f"    · {s}")
    lines.append(f"  维度二 资金特征:  {d2_score:>2}/{d2_max}  {'✓' if d2_score >= d2_max * 0.6 else '✗'}")
    for s in d2_signals:
        lines.append(f"    · {s}")
    lines.append(f"  维度三 量能结构:  {d3_score:>2}/{d3_max}  {'✓' if d3_score >= d3_max * 0.6 else '✗'}")
    for s in d3_signals:
        lines.append(f"    · {s}")
    lines.append(f"  维度四 筹码情绪:  {d4_score:>2}/{d4_max}  {'✓' if d4_score >= d4_max * 0.6 else '✗'}")
    for s in d4_signals:
        lines.append(f"    · {s}")
    lines.extend([
        "",
        f"  总分: {total_score}/{total_max}  满足维度: {dims_met}/4",
        f"  结论: {conclusion_labels.get(conclusion, conclusion)}",
    ])

    # 排除信号（任一维度出现反向信号）
    exclusions = []
    for sig_list in [d1_signals, d2_signals, d3_signals, d4_signals]:
        for s in sig_list:
            if "✗" in s and ("过快" in s or "过大" in s or "破位" in s or "过于频繁" in s):
                exclusions.append(s.replace(" ✗", ""))
    if exclusions:
        lines.append("  ⚠ 排除信号:")
        for e in exclusions:
            lines.append(f"    ✗ {e}")

    # 价位信息
    if support_price > 0:
        lines.extend([
            "",
            "  ── 关键价位 ──",
            f"  当前价: {current_price:.2f}    支撑位: {support_price:.2f}    阻力位: {resistance_price:.2f}",
            f"  建议入场区间: {entry_low:.2f} ~ {entry_high:.2f}",
            f"  建议止损位: {stop_loss:.2f} (支撑下方3%)",
        ])

    return {
        "dim1_price":    {"score": d1_score, "max": d1_max, "signals": d1_signals},
        "dim2_money":    {"score": d2_score, "max": d2_max, "signals": d2_signals},
        "dim3_volume":   {"score": d3_score, "max": d3_max, "signals": d3_signals},
        "dim4_sentiment": {"score": d4_score, "max": d4_max, "signals": d4_signals},
        "total_score":    total_score,
        "total_max":      total_max,
        "dimensions_met": dims_met,
        "conclusion":     conclusion,
        "exclusions":     exclusions,
        "price_info":     price_info,
        "summary":        "\n".join(lines),
    }
