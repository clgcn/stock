"""
Backtest Engine — Strategy Validation & Performance Analysis
=============================================================
Supports multiple built-in strategies and custom strategy backtesting
on historical data, producing comprehensive performance metrics.

Built-in strategies:
  1. MA Cross (Moving Average Crossover)
  2. MACD Golden/Death Cross
  3. RSI Overbought/Oversold
  4. Bollinger Band Breakout
  5. Multi-Factor (calls quant_engine scoring)
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Callable

# =====================================================
# Built-in Strategy Definitions
# =====================================================

def strategy_ma_cross(df: pd.DataFrame, fast: int = 5, slow: int = 20) -> pd.Series:
    """
    Moving average crossover strategy.
    MA_fast > MA_slow -> Buy signal (+1)
    MA_fast < MA_slow -> Sell signal (-1)
    """
    ma_f = df["close"].rolling(fast).mean()
    ma_s = df["close"].rolling(slow).mean()
    signal = pd.Series(0, index=df.index)
    signal[ma_f > ma_s] = 1
    signal[ma_f < ma_s] = -1
    return signal


def strategy_macd(df: pd.DataFrame) -> pd.Series:
    """
    MACD strategy.
    DIF > DEA -> Long
    DIF < DEA -> Flat/Short
    """
    close = df["close"]
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    signal = pd.Series(0, index=df.index)
    signal[dif > dea] = 1
    signal[dif < dea] = -1
    return signal


def strategy_rsi(df: pd.DataFrame, period: int = 14,
                 overbought: float = 70, oversold: float = 30) -> pd.Series:
    """
    RSI strategy.
    RSI < oversold  -> Buy
    RSI > overbought -> Sell
    Hold previous signal in the neutral zone
    """
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - 100 / (1 + rs)

    signal = pd.Series(0, index=df.index)
    position = 0
    for i in range(len(rsi)):
        if pd.isna(rsi.iloc[i]):
            continue
        if rsi.iloc[i] < oversold:
            position = 1
        elif rsi.iloc[i] > overbought:
            position = -1
        signal.iloc[i] = position
    return signal


def strategy_bollinger(df: pd.DataFrame, period: int = 20, num_std: float = 2) -> pd.Series:
    """
    Bollinger Band mean reversion strategy.
    Price < lower band -> Buy (oversold bounce)
    Price > upper band -> Sell (overbought pullback)
    Price returns to middle band -> Close position
    """
    ma  = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    upper = ma + num_std * std
    lower = ma - num_std * std

    signal = pd.Series(0, index=df.index)
    position = 0
    for i in range(len(df)):
        if pd.isna(ma.iloc[i]):
            continue
        price = df["close"].iloc[i]
        if price < lower.iloc[i]:
            position = 1
        elif price > upper.iloc[i]:
            position = -1
        elif position == 1 and price > ma.iloc[i]:
            position = 0
        elif position == -1 and price < ma.iloc[i]:
            position = 0
        signal.iloc[i] = position
    return signal


def strategy_multifactor(df: pd.DataFrame, threshold_buy: float = 15,
                          threshold_sell: float = -15) -> pd.Series:
    """
    Multi-factor composite strategy (calls quant_engine).
    Uses rolling window to compute composite score, generates signals based on thresholds.
    """
    import quant_engine as qe

    signal = pd.Series(0, index=df.index)
    window = 60   # Minimum 60 days of data for analysis

    for i in range(window, len(df)):
        sub = df.iloc[:i + 1].copy()
        sub.attrs = df.attrs.copy()
        try:
            result = qe.comprehensive_diagnosis(sub, run_monte_carlo=False)
            score = result["total_score"]
            if score > threshold_buy:
                signal.iloc[i] = 1
            elif score < threshold_sell:
                signal.iloc[i] = -1
        except Exception:
            pass
    return signal


# Strategy registry
STRATEGIES = {
    "ma_cross":    {"func": strategy_ma_cross,    "name": "MA Cross",     "desc": "Short MA crosses above/below long MA"},
    "macd":        {"func": strategy_macd,        "name": "MACD",         "desc": "DIF/DEA golden cross / death cross"},
    "rsi":         {"func": strategy_rsi,         "name": "RSI",          "desc": "RSI overbought/oversold reversal"},
    "bollinger":   {"func": strategy_bollinger,   "name": "Bollinger",    "desc": "Bollinger Band mean reversion"},
    "multifactor": {"func": strategy_multifactor, "name": "Multi-Factor", "desc": "Comprehensive quantitative scoring system"},
}


# =====================================================
# Backtest Core Engine
# =====================================================

def backtest(
    df: pd.DataFrame,
    strategy_name: str = "ma_cross",
    strategy_params: dict = None,
    initial_capital: float = 100000,
    commission: float = 0.001,     # One-way commission 0.1%
    slippage: float = 0.001,       # Slippage 0.1%
    stamp_tax: float = 0.001,      # Stamp tax (sell side only) 0.1%
    max_position: float = 1.0,     # Maximum position ratio
    stop_loss: float = 0.0,        # Stop-loss ratio (0 = disabled)
    take_profit: float = 0.0,      # Take-profit ratio (0 = disabled)
) -> Dict:
    """
    Execute strategy backtest.

    Parameters:
        df               K-line DataFrame (must contain OHLCV)
        strategy_name    Strategy name (see STRATEGIES)
        strategy_params  Strategy parameter dict (overrides defaults)
        initial_capital  Starting capital
        commission       One-way commission rate
        slippage         Slippage (as percentage)
        stamp_tax        Stamp tax (on sell)
        max_position     Maximum position ratio
        stop_loss        Stop-loss ratio (e.g. 0.05 = stop at 5% loss)
        take_profit      Take-profit ratio

    Returns: Complete backtest result dictionary
    """
    if strategy_name not in STRATEGIES:
        return {"error": f"Unknown strategy: {strategy_name}, available: {list(STRATEGIES.keys())}"}

    strategy_func = STRATEGIES[strategy_name]["func"]
    params = strategy_params or {}

    # -- Generate signals --
    signal = strategy_func(df, **params)

    # -- Simulate trading --
    df_bt = df.copy().reset_index(drop=True)
    n = len(df_bt)
    capital = initial_capital
    shares = 0
    entry_price = 0
    trades = []
    equity_curve = []
    positions = []
    trade_log = []

    for i in range(n):
        price = df_bt.iloc[i]["close"]
        sig = signal.iloc[i]

        # Stop-loss / take-profit check (when holding)
        if shares > 0 and entry_price > 0:
            pnl_pct = (price - entry_price) / entry_price
            if stop_loss > 0 and pnl_pct < -stop_loss:
                sig = -1  # Trigger stop-loss
            if take_profit > 0 and pnl_pct > take_profit:
                sig = -1  # Trigger take-profit

        # Buy
        if sig == 1 and shares == 0:
            buy_price = price * (1 + slippage)
            buy_amount = capital * max_position
            shares = int(buy_amount / buy_price / 100) * 100  # A-share: multiples of 100
            # If insufficient for 1 lot, allow 100 shares for backtest purposes
            if shares == 0 and buy_amount >= buy_price:
                shares = 100
            elif shares == 0 and buy_amount < buy_price:
                shares = 100  # Allow minimum 1 lot for strategy validation
            if shares > 0:
                cost = shares * buy_price * (1 + commission)
                capital -= cost
                entry_price = buy_price
                trade_log.append({
                    "date": df_bt.iloc[i]["date"],
                    "action": "BUY",
                    "price": round(buy_price, 2),
                    "shares": shares,
                    "cost": round(cost, 2),
                })

        # Sell
        elif sig == -1 and shares > 0:
            sell_price = price * (1 - slippage)
            revenue = shares * sell_price * (1 - commission - stamp_tax)
            pnl = revenue - (shares * entry_price * (1 + commission))
            capital += revenue

            trade_log.append({
                "date": df_bt.iloc[i]["date"],
                "action": "SELL",
                "price": round(sell_price, 2),
                "shares": shares,
                "revenue": round(revenue, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl / (shares * entry_price) * 100, 2),
            })
            trades.append(pnl)
            shares = 0
            entry_price = 0

        total_equity = capital + shares * price
        equity_curve.append(total_equity)
        positions.append(1 if shares > 0 else 0)

    # If still holding at the end, calculate at last price
    final_equity = capital + shares * df_bt.iloc[-1]["close"]

    # -- Calculate performance metrics --
    metrics = _calc_metrics(
        equity_curve, trades, trade_log,
        initial_capital, df_bt, positions
    )
    metrics["strategy"] = STRATEGIES[strategy_name]["name"]
    metrics["strategy_desc"] = STRATEGIES[strategy_name]["desc"]
    metrics["params"] = params
    metrics["trade_log"] = trade_log
    metrics["equity_curve"] = equity_curve

    return metrics


def _calc_metrics(equity_curve, trades, trade_log,
                  initial_capital, df, positions) -> Dict:
    """Calculate comprehensive performance metrics"""
    eq = np.array(equity_curve)
    n = len(eq)
    final = eq[-1]

    # -- Basic returns --
    total_return = (final / initial_capital - 1) * 100
    trading_days = n
    years = trading_days / 252 if trading_days > 0 else 1
    cagr = ((final / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    # -- Benchmark (buy and hold) --
    buy_hold_return = (df.iloc[-1]["close"] / df.iloc[0]["close"] - 1) * 100

    # -- Maximum drawdown --
    peak = np.maximum.accumulate(eq)
    drawdown = (eq - peak) / peak * 100
    max_drawdown = abs(np.min(drawdown))
    max_dd_end = np.argmin(drawdown)
    max_dd_start = np.argmax(eq[:max_dd_end + 1]) if max_dd_end > 0 else 0

    # -- Daily returns --
    daily_ret = np.diff(eq) / np.where(eq[:-1] != 0, eq[:-1], 1)
    daily_std = np.std(daily_ret, ddof=1) if len(daily_ret) > 1 else 0
    ann_vol = daily_std * np.sqrt(252) * 100

    # -- Sharpe ratio (risk-free rate 3%) --
    rf_daily = 0.03 / 252
    excess_ret = daily_ret - rf_daily
    excess_std = np.std(excess_ret, ddof=1) if len(excess_ret) > 1 else 0
    sharpe = (np.mean(excess_ret) / excess_std * np.sqrt(252)) if excess_std > 1e-10 else 0

    # -- Sortino ratio --
    downside = daily_ret[daily_ret < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 0
    sortino = (np.mean(daily_ret) * 252 - 0.03) / (downside_std * np.sqrt(252)) \
        if downside_std > 1e-10 else 0

    # -- Calmar ratio --
    calmar = cagr / max_drawdown if max_drawdown > 0 else 0

    # -- Trade statistics --
    n_trades = len(trades)
    if n_trades > 0:
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        win_rate = len(wins) / n_trades * 100
        avg_win  = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 1
        profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
        avg_pnl = np.mean(trades)
        max_win = max(trades)
        max_loss = min(trades)
    else:
        win_rate = 0
        avg_win = avg_loss = profit_factor = avg_pnl = max_win = max_loss = 0

    # -- Holding time --
    pos_arr = np.array(positions)
    holding_pct = np.mean(pos_arr) * 100

    # -- Max consecutive wins/losses --
    max_consec_win, max_consec_loss = 0, 0
    consec_win, consec_loss = 0, 0
    for t in trades:
        if t > 0:
            consec_win += 1
            consec_loss = 0
            max_consec_win = max(max_consec_win, consec_win)
        else:
            consec_loss += 1
            consec_win = 0
            max_consec_loss = max(max_consec_loss, consec_loss)

    return {
        # Returns
        "initial_capital": initial_capital,
        "final_equity": round(final, 2),
        "total_return_pct": round(total_return, 2),
        "cagr_pct": round(cagr, 2),
        "buy_hold_return_pct": round(buy_hold_return, 2),
        "excess_return_pct": round(total_return - buy_hold_return, 2),

        # Risk
        "max_drawdown_pct": round(max_drawdown, 2),
        "ann_volatility_pct": round(ann_vol, 2),

        # Risk-adjusted
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "calmar_ratio": round(calmar, 2),

        # Trades
        "n_trades": n_trades,
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "INF",
        "avg_pnl": round(avg_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "max_win": round(max_win, 2),
        "max_loss": round(max_loss, 2),
        "max_consec_wins": max_consec_win,
        "max_consec_losses": max_consec_loss,

        # Other
        "trading_days": trading_days,
        "holding_pct": round(holding_pct, 1),
    }


# =====================================================
# Formatted Output
# =====================================================

def format_backtest_result(result: Dict) -> str:
    """Format backtest result into human-readable text"""
    if "error" in result:
        return f"ERROR: {result['error']}"

    lines = [
        f"Backtest Report -- {result['strategy']}",
        f"   Strategy: {result['strategy_desc']}",
        f"   Parameters: {result.get('params', {})}",
        "=" * 55,
        "",
        "-- Returns --",
        f"  Initial Capital  : CNY {result['initial_capital']:,.0f}",
        f"  Final Equity     : CNY {result['final_equity']:,.0f}",
        f"  Total Return     : {result['total_return_pct']:+.2f}%",
        f"  CAGR             : {result['cagr_pct']:+.2f}%",
        f"  Buy & Hold Return: {result['buy_hold_return_pct']:+.2f}%",
        f"  Excess Return    : {result['excess_return_pct']:+.2f}%",
        "",
        "-- Risk --",
        f"  Max Drawdown     : {result['max_drawdown_pct']:.2f}%",
        f"  Ann. Volatility  : {result['ann_volatility_pct']:.2f}%",
        "",
        "-- Risk-Adjusted Metrics --",
        f"  Sharpe Ratio     : {result['sharpe_ratio']:.2f}",
        f"  Sortino Ratio    : {result['sortino_ratio']:.2f}",
        f"  Calmar Ratio     : {result['calmar_ratio']:.2f}",
        "",
        "-- Trade Statistics --",
        f"  Total Trades     : {result['n_trades']}",
        f"  Win Rate         : {result['win_rate_pct']:.1f}%",
        f"  Profit Factor    : {result['profit_factor']}",
        f"  Avg Win          : CNY {result['avg_win']:,.2f}",
        f"  Avg Loss         : CNY {result['avg_loss']:,.2f}",
        f"  Max Single Win   : CNY {result['max_win']:,.2f}",
        f"  Max Single Loss  : CNY {result['max_loss']:,.2f}",
        f"  Max Consec. Wins : {result['max_consec_wins']}",
        f"  Max Consec. Losses: {result['max_consec_losses']}",
        "",
        "-- Other --",
        f"  Backtest Days    : {result['trading_days']}",
        f"  Time in Market   : {result['holding_pct']:.1f}%",
        "",
        "DISCLAIMER: Backtest results are based on historical data and do not guarantee future performance.",
    ]

    # Recent trade log
    log = result.get("trade_log", [])
    if log:
        lines.extend(["", "-- Recent Trades --"])
        for t in log[-10:]:
            if t["action"] == "BUY":
                lines.append(f"  {t['date']}  BUY {t['shares']} shares @ {t['price']:.2f}")
            else:
                lines.append(
                    f"  {t['date']}  SELL {t['shares']} shares @ {t['price']:.2f}  "
                    f"P&L CNY {t['pnl']:+,.2f} ({t['pnl_pct']:+.1f}%)"
                )

    return "\n".join(lines)
