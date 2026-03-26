"""
Risk Management Module — VaR, Position Sizing, Stop-Loss, Stress Testing
==========================================================================

Core Features:
  1. VaR / CVaR (Conditional Value at Risk)
  2. Kelly Criterion optimal position sizing
  3. ATR dynamic stop-loss / take-profit
  4. Stress testing (historical extreme scenarios + Monte Carlo)
  5. Position recommendations
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


# =====================================================
# 1. VaR & CVaR
# =====================================================

def calc_var(
    df: pd.DataFrame,
    confidence: float = 0.95,
    holding_days: int = 1,
    method: str = "historical",
    position_value: float = 100000,
) -> Dict:
    """
    Calculate Value at Risk.

    Methods:
    - historical: Historical simulation (return percentile)
    - parametric: Parametric method (assumes normal distribution, mu - z * sigma)

    Parameters:
        confidence      Confidence level (default 95%)
        holding_days    Holding period in days
        position_value  Position market value

    Returns: VaR / CVaR values and interpretation
    """
    close = df["close"].values
    if len(close) < 30:
        return {"error": "Insufficient data, at least 30 days required"}

    log_ret = np.log(close[1:] / close[:-1])

    # Adjust for holding period returns
    if holding_days > 1:
        # Square root of time rule
        adj_ret = log_ret * np.sqrt(holding_days)
    else:
        adj_ret = log_ret

    if method == "historical":
        var_pct = np.percentile(adj_ret, (1 - confidence) * 100)
        # CVaR = average loss beyond VaR threshold
        cvar_pct = np.mean(adj_ret[adj_ret <= var_pct])
    else:  # parametric
        mu = np.mean(adj_ret)
        sigma = np.std(adj_ret, ddof=1)
        # Normal distribution quantile approximation
        z = _norm_ppf(confidence)
        var_pct = mu - z * sigma
        cvar_pct = mu - sigma * _norm_pdf(z) / (1 - confidence)

    var_amount = abs(var_pct) * position_value
    cvar_amount = abs(cvar_pct) * position_value

    return {
        "method": "historical_simulation" if method == "historical" else "parametric",
        "confidence": f"{confidence * 100:.0f}%",
        "holding_days": holding_days,
        "var_pct": round(abs(var_pct) * 100, 2),
        "cvar_pct": round(abs(cvar_pct) * 100, 2),
        "var_amount": round(var_amount, 0),
        "cvar_amount": round(cvar_amount, 0),
        "position_value": position_value,
        "interpretation": (
            f"At {confidence*100:.0f}% confidence level, "
            f"max expected loss over {holding_days} day(s) is "
            f"{abs(var_pct)*100:.2f}% (CNY {var_amount:,.0f}). "
            f"In extreme scenarios, average loss (CVaR) is "
            f"{abs(cvar_pct)*100:.2f}% (CNY {cvar_amount:,.0f})."
        ),
    }


def _norm_pdf(x):
    """Standard normal probability density function"""
    return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)


def _norm_ppf(p):
    """Standard normal quantile function (Abramowitz & Stegun approximation)"""
    if p <= 0 or p >= 1:
        return 0
    if p < 0.5:
        return -_norm_ppf(1 - p)
    t = np.sqrt(-2 * np.log(1 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3)


# =====================================================
# 2. Kelly Criterion Position Sizing
# =====================================================

def kelly_position(
    win_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    kelly_fraction: float = 0.5,
) -> Dict:
    """
    Kelly Criterion for optimal position sizing.

    Formula: f* = (p * b - q) / b
        p = win rate
        b = win/loss ratio (avg win / avg loss)
        q = 1 - p

    Parameters:
        win_rate        Win rate (0~1)
        avg_win_pct     Average win percentage
        avg_loss_pct    Average loss percentage (positive number)
        kelly_fraction  Kelly discount factor (0.5 = half-Kelly, more conservative)
    """
    if avg_loss_pct <= 0:
        return {"error": "Average loss must be positive"}

    p = win_rate
    q = 1 - p
    b = avg_win_pct / avg_loss_pct  # Win/loss ratio

    kelly_full = (p * b - q) / b if b > 0 else 0
    kelly_adj  = kelly_full * kelly_fraction

    return {
        "win_rate": round(p * 100, 1),
        "odds_ratio": round(b, 2),
        "kelly_full_pct": round(max(kelly_full * 100, 0), 1),
        "kelly_adjusted_pct": round(max(kelly_adj * 100, 0), 1),
        "kelly_fraction": kelly_fraction,
        "recommendation": (
            f"Full Kelly position: {max(kelly_full*100, 0):.1f}%\n"
            f"Recommended position ({kelly_fraction}x Kelly): {max(kelly_adj*100, 0):.1f}%\n"
            + ("WARNING: Negative Kelly value, avoid this strategy" if kelly_full <= 0 else
               f"With CNY 100,000 capital, recommended investment: CNY {max(kelly_adj * 100000, 0):,.0f}")
        ),
    }


def kelly_from_history(df: pd.DataFrame, lookback: int = 60) -> Dict:
    """
    Automatically estimate Kelly position from historical data.
    Uses recent lookback days of up/down statistics to estimate win rate and win/loss ratio.
    """
    close = df["close"].values
    n = len(close)
    period = min(lookback, n - 1)
    rets = np.diff(close[-period - 1:]) / close[-period - 1:-1]

    wins  = rets[rets > 0]
    losses = rets[rets < 0]

    if len(wins) == 0 or len(losses) == 0:
        return {"error": "Incomplete win/loss records in the period"}

    win_rate = len(wins) / len(rets)
    avg_win  = np.mean(wins) * 100
    avg_loss = abs(np.mean(losses)) * 100

    return kelly_position(win_rate, avg_win, avg_loss)


# =====================================================
# 3. Stop-Loss / Take-Profit Calculation
# =====================================================

def calc_stop_levels(
    df: pd.DataFrame,
    entry_price: float = 0,
    method: str = "atr",
    atr_multiplier: float = 2.0,
    risk_reward_ratio: float = 2.0,
) -> Dict:
    """
    Calculate stop-loss / take-profit levels.

    Methods:
    - atr:     ATR-based dynamic stop-loss
    - support: Based on nearest support level
    - percent: Fixed percentage

    Parameters:
        entry_price       Entry price (0 = current price)
        atr_multiplier    ATR multiplier (default 2x)
        risk_reward_ratio Risk-reward ratio (default 1:2)
    """
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    n = len(close)

    if entry_price <= 0:
        entry_price = close[-1]

    result = {"entry_price": round(entry_price, 2)}

    # -- ATR stop-loss --
    if n >= 15:
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i - 1]),
                        abs(low[i] - close[i - 1]))
        from quant_engine import ewma
        atr = ewma(tr[1:], 14)
        atr_val = atr[-1]

        stop_loss = entry_price - atr_multiplier * atr_val
        risk = entry_price - stop_loss
        take_profit = entry_price + risk * risk_reward_ratio

        result.update({
            "atr_14": round(atr_val, 2),
            "atr_stop_loss": round(stop_loss, 2),
            "atr_stop_loss_pct": round((entry_price - stop_loss) / entry_price * 100, 2),
            "atr_take_profit": round(take_profit, 2),
            "atr_take_profit_pct": round((take_profit - entry_price) / entry_price * 100, 2),
            "risk_reward_ratio": risk_reward_ratio,
        })

    # -- Support-based stop-loss --
    if n >= 30:
        lookback = min(60, n)
        seg_low = low[-lookback:]
        # Recent local lows
        local_lows = []
        for i in range(2, len(seg_low) - 2):
            if seg_low[i] < seg_low[i-1] and seg_low[i] < seg_low[i+1]:
                if seg_low[i] < entry_price:
                    local_lows.append(seg_low[i])

        if local_lows:
            nearest_support = max(local_lows)  # Nearest (highest) support
            support_stop = nearest_support * 0.99  # Slightly below support
            s_risk = entry_price - support_stop
            s_tp   = entry_price + s_risk * risk_reward_ratio

            result.update({
                "support_level": round(nearest_support, 2),
                "support_stop_loss": round(support_stop, 2),
                "support_stop_loss_pct": round((entry_price - support_stop) / entry_price * 100, 2),
                "support_take_profit": round(s_tp, 2),
            })

    # -- Percentage stop-loss (fixed) --
    for pct in [3, 5, 8]:
        sl = entry_price * (1 - pct / 100)
        tp = entry_price * (1 + pct * risk_reward_ratio / 100)
        result[f"pct_{pct}_stop_loss"] = round(sl, 2)
        result[f"pct_{pct}_take_profit"] = round(tp, 2)

    return result


# =====================================================
# 4. Stress Testing
# =====================================================

def stress_test(
    df: pd.DataFrame,
    position_value: float = 100000,
) -> Dict:
    """
    Stress test: evaluate potential losses under extreme market conditions.

    Scenarios:
    1) Historical max single-day drop replay
    2) Consecutive 3-day crash
    3) 2x historical max drop (black swan)
    4) Monte Carlo extreme paths (1st percentile)
    """
    close = df["close"].values
    n = len(close)
    if n < 30:
        return {"error": "Insufficient data"}

    daily_ret = np.diff(close) / close[:-1]
    current_price = close[-1]

    # -- Historical extreme drops --
    max_daily_drop = np.min(daily_ret)
    max_3day_drop  = 0
    for i in range(2, len(daily_ret)):
        ret_3d = (1 + daily_ret[i]) * (1 + daily_ret[i-1]) * (1 + daily_ret[i-2]) - 1
        max_3day_drop = min(max_3day_drop, ret_3d)

    scenarios = {
        "Historical max single-day drop": {
            "drop_pct": round(max_daily_drop * 100, 2),
            "loss_amount": round(abs(max_daily_drop) * position_value, 0),
            "price_after": round(current_price * (1 + max_daily_drop), 2),
        },
        "Historical max 3-day consecutive drop": {
            "drop_pct": round(max_3day_drop * 100, 2),
            "loss_amount": round(abs(max_3day_drop) * position_value, 0),
            "price_after": round(current_price * (1 + max_3day_drop), 2),
        },
        "Black swan (2x historical max drop)": {
            "drop_pct": round(max_daily_drop * 200, 2),
            "loss_amount": round(abs(max_daily_drop * 2) * position_value, 0),
            "price_after": round(current_price * (1 + max_daily_drop * 2), 2),
        },
    }

    # -- Monte Carlo extreme paths --
    log_ret = np.log(close[1:] / close[:-1])
    mu = np.mean(log_ret)
    sigma = np.std(log_ret, ddof=1)
    n_sim = 10000

    for days, label in [(5, "5-day"), (20, "20-day")]:
        # 使用 t 分布（df=5）捕捉A股肥尾，与 quant_engine 保持一致
        df_t = 5
        z_raw = np.random.standard_t(df_t, size=(n_sim, days))
        z = z_raw / np.sqrt(df_t / (df_t - 2))   # 归一化到单位方差
        paths = np.exp(np.cumsum((mu - 0.5*sigma**2) + sigma * z, axis=1))
        final_rets = paths[:, -1] - 1
        worst_1pct = np.percentile(final_rets, 1)
        worst_5pct = np.percentile(final_rets, 5)

        scenarios[f"Monte Carlo {label} extreme (1st percentile)"] = {
            "drop_pct": round(worst_1pct * 100, 2),
            "loss_amount": round(abs(worst_1pct) * position_value, 0),
            "price_after": round(current_price * (1 + worst_1pct), 2),
        }
        scenarios[f"Monte Carlo {label} (5th percentile)"] = {
            "drop_pct": round(worst_5pct * 100, 2),
            "loss_amount": round(abs(worst_5pct) * position_value, 0),
            "price_after": round(current_price * (1 + worst_5pct), 2),
        }

    return {
        "current_price": round(current_price, 2),
        "position_value": position_value,
        "scenarios": scenarios,
    }


# =====================================================
# 5. Comprehensive Risk Assessment
# =====================================================

def comprehensive_risk_assessment(
    df: pd.DataFrame,
    position_value: float = 100000,
) -> str:
    """
    Comprehensive risk assessment report, combining all risk metrics.
    """
    name = df.attrs.get("name", "")
    code = df.attrs.get("code", "")
    current = df.iloc[-1]["close"]

    lines = [
        f"Risk Assessment Report  {name} ({code})",
        f"  Current Price: {current:.2f}   Position Value: CNY {position_value:,.0f}",
        "=" * 55,
    ]

    # VaR
    for conf in [0.95, 0.99]:
        var = calc_var(df, confidence=conf, holding_days=1, position_value=position_value)
        if "error" not in var:
            lines.extend([
                "",
                f"-- VaR ({var['confidence']}) --",
                f"  Daily VaR  : {var['var_pct']:.2f}%  (CNY {var['var_amount']:,.0f})",
                f"  Daily CVaR : {var['cvar_pct']:.2f}%  (CNY {var['cvar_amount']:,.0f})",
            ])

    # Kelly
    kelly = kelly_from_history(df)
    if "error" not in kelly:
        lines.extend([
            "",
            "-- Kelly Position Recommendation --",
            f"  Win Rate: {kelly['win_rate']:.0f}%   Win/Loss Ratio: {kelly['odds_ratio']:.2f}",
            f"  Recommended Position: {kelly['kelly_adjusted_pct']:.1f}% (Half-Kelly)",
        ])

    # Stop-loss
    stops = calc_stop_levels(df)
    if "atr_stop_loss" in stops:
        lines.extend([
            "",
            "-- Stop-Loss / Take-Profit (ATR method) --",
            f"  ATR(14)     : {stops['atr_14']:.2f}",
            f"  Stop-Loss   : {stops['atr_stop_loss']:.2f}  (-{stops['atr_stop_loss_pct']:.1f}%)",
            f"  Take-Profit : {stops['atr_take_profit']:.2f}  (+{stops['atr_take_profit_pct']:.1f}%)",
            f"  Risk/Reward : 1:{stops['risk_reward_ratio']:.0f}",
        ])

    # Stress test
    stress = stress_test(df, position_value=position_value)
    if "error" not in stress:
        lines.extend(["", "-- Stress Test --"])
        for scenario, data in stress["scenarios"].items():
            lines.append(
                f"  {scenario}: "
                f"{data['drop_pct']:+.1f}%  ->  CNY {data['loss_amount']:,.0f} loss"
                f"  price to {data['price_after']:.2f}"
            )

    lines.extend([
        "",
        "DISCLAIMER: The above is based on historical data and mathematical models",
        "and cannot fully predict future extreme market conditions.",
        "Strictly follow stop-loss discipline; position size should not exceed Kelly recommendation.",
    ])

    return "\n".join(lines)
