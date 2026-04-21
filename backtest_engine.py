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
# A-Share-Specific Helpers
# =====================================================

def _get_price_limit(code: str, name: str = "") -> float:
    """
    Return price limit percentage for the stock.
    - ST stocks: ±5%
    - ChiNext (3xxxx): ±20%
    - STAR Market (688xxx): ±20%
    - Main board (60xxxx, 00xxxx): ±10%
    """
    if "ST" in name.upper():
        return 0.05
    if code.startswith("30") or code.startswith("688"):
        return 0.20
    return 0.10


def _is_limit_up(close: float, prev_close: float, limit_pct: float) -> bool:
    """Check if stock hit limit up (涨停)."""
    return close >= prev_close * (1 + limit_pct - 0.005)


def _is_limit_down(close: float, prev_close: float, limit_pct: float) -> bool:
    """Check if stock hit limit down (跌停)."""
    return close <= prev_close * (1 - limit_pct + 0.005)


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


def strategy_decision_engine(
    df: pd.DataFrame,
    window: int = 90,
    decision_config: dict = None,
) -> pd.Series:
    """
    Decision-engine strategy.
    Replays the current buy/hold/sell logic on a rolling window and
    converts explicit trade decisions into backtest signals.
    """
    import quant_engine as qe
    from slow_fetcher import _build_trade_decision

    signal = pd.Series(0, index=df.index)
    decision_config = decision_config or {}
    use_monte_carlo = decision_config.get("use_monte_carlo", False)
    neutral_market = {
        "score": 0.0,
        "label": "neutral",
        "action_bias": "neutral",
        "summary": "Calibration mode uses neutral market regime.",
    }
    neutral_event = {
        "available": False,
        "score": 0.0,
        "label": "neutral",
        "summary": "Calibration mode ignores event risk.",
        "recent_earnings": False,
        "high_uncertainty": False,
        "positive_flags": [],
        "negative_flags": [],
    }

    for i in range(window, len(df)):
        sub = df.iloc[:i + 1].copy()
        sub.attrs = df.attrs.copy()
        try:
            diag = qe.comprehensive_diagnosis(sub, run_monte_carlo=use_monte_carlo, mc_days=20)
            decision = _build_trade_decision(
                sub,
                diag,
                float(sub.iloc[-1]["close"]),
                neutral_market,
                neutral_event,
                config=decision_config,
            )
            if decision["action"] == "buy":
                signal.iloc[i] = 1
            elif decision["action"] == "sell":
                signal.iloc[i] = -1
        except Exception:
            continue
    return signal


# Strategy registry
STRATEGIES = {
    "ma_cross":    {"func": strategy_ma_cross,    "name": "MA Cross",     "desc": "Short MA crosses above/below long MA"},
    "macd":        {"func": strategy_macd,        "name": "MACD",         "desc": "DIF/DEA golden cross / death cross"},
    "rsi":         {"func": strategy_rsi,         "name": "RSI",          "desc": "RSI overbought/oversold reversal"},
    "bollinger":   {"func": strategy_bollinger,   "name": "Bollinger",    "desc": "Bollinger Band mean reversion"},
    "multifactor": {"func": strategy_multifactor, "name": "Multi-Factor", "desc": "Comprehensive quantitative scoring system"},
    "decision_engine": {"func": strategy_decision_engine, "name": "Decision Engine", "desc": "Replay explicit buy/hold/sell decision rules"},
}


# =====================================================
# Backtest Core Engine
# =====================================================

def backtest(
    df: pd.DataFrame,
    strategy_name: str = "ma_cross",
    strategy_params: dict = None,
    initial_capital: float = 100000,
    commission: float = 0.0003,    # One-way commission 万3 (A-share standard)
    slippage: float = 0.001,       # Slippage 0.1%
    stamp_tax: float = 0.001,      # Stamp tax (sell side only) 0.1%
    max_position: float = 1.0,     # Maximum position ratio
    stop_loss: float = 0.0,        # Stop-loss ratio (0 = disabled)
    take_profit: float = 0.0,      # Take-profit ratio (0 = disabled)
    code: str = "",                # Stock code for price limit determination
    name: str = "",                # Stock name for ST detection
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
        code             Stock code (e.g. "600519") for price limit detection
        name             Stock name for ST detection (涨跌停 handling)

    Returns: Complete backtest result dictionary

    A-Share-Specific Features:
        - 涨跌停 (Daily Price Limit): Enforces ±10% (main), ±20% (ChiNext/STAR), ±5% (ST)
        - T+1 Settlement: Cannot sell on the same day as buy
        - Stock Suspension: Skips trading on days with zero volume
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
    # T+1: orders queued at close of day T execute at open of day T+1
    pending_order = None  # 'buy' or 'sell'
    pending_order_days = 0  # bars the pending order has been waiting
    MAX_PENDING_DAYS = 10  # force-close if stuck (e.g., consecutive limit-down)
    trades = []
    equity_curve = []
    positions = []
    trade_log = []

    # Pre-compute column existence — pandas Series has no .get(); check once before loop
    _has_open_col = "open" in df_bt.columns
    _has_volume_col = "volume" in df_bt.columns

    # Get price limit for this stock
    price_limit_pct = _get_price_limit(code, name)

    for i in range(n):
        close = df_bt.iloc[i]["close"]
        # Use open price for order execution; fall back to close if column absent
        open_price = df_bt.iloc[i]["open"] if _has_open_col else close
        sig = signal.iloc[i]

        # Feature 3: Stock Suspension Handling
        volume = df_bt.iloc[i]["volume"] if _has_volume_col else 0
        is_suspended = volume <= 0 or pd.isna(volume)

        # Feature 1: 涨跌停 (Daily Price Limit) Handling
        prev_close = df_bt.iloc[i - 1]["close"] if i > 0 else close
        is_limit_up_flag = _is_limit_up(close, prev_close, price_limit_pct)
        is_limit_down_flag = _is_limit_down(close, prev_close, price_limit_pct)

        # --- Execute pending orders at today's open (T+1 settlement) ---
        if pending_order == "buy" and shares == 0 and not is_suspended:
            buy_price = open_price * (1 + slippage)
            buy_amount = capital * max_position
            new_shares = int(buy_amount / buy_price / 100) * 100
            if new_shares == 0:
                new_shares = 100  # minimum 1 lot for backtest validation
            cost = new_shares * buy_price * (1 + commission)
            capital -= cost
            shares = new_shares
            entry_price = buy_price
            trade_log.append({
                "date": df_bt.iloc[i]["date"],
                "action": "BUY",
                "price": round(buy_price, 2),
                "shares": shares,
                "cost": round(cost, 2),
            })
            pending_order = None
            pending_order_days = 0

        elif pending_order == "sell" and shares > 0 and not is_suspended:
            force_sell = pending_order_days >= MAX_PENDING_DAYS  # timeout: consecutive limit-down
            if not is_limit_down_flag or force_sell:
                exec_price = open_price  # always execute at open (forced or not); limit-down close is untradeable
                sell_price = exec_price * (1 - slippage)
                revenue = shares * sell_price * (1 - commission - stamp_tax)
                pnl = revenue - (shares * entry_price * (1 + commission))
                capital += revenue
                trade_log.append({
                    "date": df_bt.iloc[i]["date"],
                    "action": "SELL" if not force_sell else "SELL(FORCED)",
                    "price": round(sell_price, 2),
                    "shares": shares,
                    "revenue": round(revenue, 2),
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl / (shares * entry_price) * 100, 2),
                })
                trades.append(pnl)
                shares = 0
                entry_price = 0
                pending_order = None
                pending_order_days = 0
            # else: limit down and not timed out → retry next bar

        # Increment pending counter when order remains unfilled
        if pending_order is not None:
            pending_order_days += 1
        else:
            pending_order_days = 0

        # --- Stop-loss / take-profit check at today's close ---
        if shares > 0 and entry_price > 0 and pending_order is None:
            pnl_pct = (close - entry_price) / entry_price
            if stop_loss > 0 and pnl_pct < -stop_loss:
                pending_order = "sell"
            elif take_profit > 0 and pnl_pct > take_profit:
                pending_order = "sell"

        # --- Queue new orders from strategy signal ---
        # T+1 naturally enforced: buy queued day T executes day T+1 open;
        # earliest sell signal fires day T+1 close → executes day T+2 open.
        if pending_order is None and not is_suspended:
            if sig == 1 and shares == 0 and not is_limit_up_flag:
                pending_order = "buy"
            elif sig == -1 and shares > 0 and not is_limit_down_flag:
                pending_order = "sell"

        total_equity = capital + shares * close
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

    # -- Sortino ratio — full-sample semi-variance (includes all days, not just negative) --
    rf_daily_sortino = 0.03 / 252
    downside_ret = np.minimum(daily_ret - rf_daily_sortino, 0)
    downside_dev = np.sqrt(np.mean(downside_ret ** 2)) * np.sqrt(252)
    sortino = (np.mean(daily_ret) * 252 - 0.03) / downside_dev if downside_dev > 1e-10 else 0

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


def calibrate_decision_engine(
    df: pd.DataFrame,
    initial_capital: float = 100000,
    max_candidates: int = 36,
) -> Dict:
    """
    Grid-search a small set of decision thresholds to find a more stable
    configuration for the explicit buy/sell engine.
    """
    candidates = []
    buy_scores = [5, 10, 15, 18]
    buy_probs = [50, 52, 55]
    risk_rewards = [1.2, 1.5, 1.8]
    trend_buys = [0, 5, 10]
    expected_return_mins = [-1.0, -0.2, 0.0]
    min_buy_checks_list = [5, 6, 7]

    for buy_score in buy_scores:
        for buy_prob in buy_probs:
            for rr in risk_rewards:
                for trend_buy in trend_buys:
                    for ev_min in expected_return_mins:
                        for min_buy_checks in min_buy_checks_list:
                            candidates.append({
                                "buy_score_min": buy_score,
                                "sell_score_max": -buy_score,
                                "buy_prob_min": buy_prob,
                                "sell_prob_min": buy_prob,
                                "min_risk_reward": rr,
                                "trend_buy_min": trend_buy,
                                "trend_sell_max": -trend_buy,
                                "expected_return_min": ev_min,
                                "min_buy_checks": min_buy_checks,
                                "min_sell_checks": 3,
                                "require_probabilistic_edge": False,
                                "use_monte_carlo": False,
                            })
    candidates = candidates[:max_candidates]

    trials = []
    for cfg in candidates:
        result = backtest(
            df,
            strategy_name="decision_engine",
            strategy_params={"window": 90, "decision_config": cfg},
            initial_capital=initial_capital,
            max_position=1.0,
            stop_loss=0.0,
            take_profit=0.0,
        )
        if "error" in result:
            continue
        score = (
            result["total_return_pct"] * 0.5
            + result["sharpe_ratio"] * 12
            - result["max_drawdown_pct"] * 0.4
            + result["win_rate_pct"] * 0.15
            + min(result["n_trades"], 12) * 0.8
        )
        result["calibration_score"] = round(score, 2)
        result["decision_config"] = cfg
        trials.append(result)

    if not trials:
        return {"error": "No valid calibration runs completed"}

    trials.sort(
        key=lambda x: (
            x["calibration_score"],
            x["sharpe_ratio"],
            x["total_return_pct"],
            -x["max_drawdown_pct"],
        ),
        reverse=True,
    )
    return {
        "best": trials[0],
        "top_trials": trials[:5],
        "tested": len(trials),
    }


def calibrate_decision_engine_batch(
    datasets: list[tuple[str, pd.DataFrame]],
    initial_capital: float = 100000,
    max_candidates: int = 24,
) -> Dict:
    """
    Batch calibration across multiple stocks.
    Uses the same parameter grid, scores configurations by aggregate robustness.
    """
    if not datasets:
        return {"error": "No datasets provided for batch calibration"}

    buy_scores = [5, 10, 15, 18]
    buy_probs = [50, 52]
    risk_rewards = [1.2, 1.5, 1.8]
    trend_buys = [0, 5, 10]
    expected_return_mins = [-1.0, -0.2, 0.0]
    min_buy_checks_list = [5, 6]

    candidates = []
    for buy_score in buy_scores:
        for buy_prob in buy_probs:
            for rr in risk_rewards:
                for trend_buy in trend_buys:
                    for ev_min in expected_return_mins:
                        for min_buy_checks in min_buy_checks_list:
                            candidates.append({
                                "buy_score_min": buy_score,
                                "sell_score_max": -buy_score,
                                "buy_prob_min": buy_prob,
                                "sell_prob_min": buy_prob,
                                "min_risk_reward": rr,
                                "trend_buy_min": trend_buy,
                                "trend_sell_max": -trend_buy,
                                "expected_return_min": ev_min,
                                "min_buy_checks": min_buy_checks,
                                "min_sell_checks": 3,
                                "require_probabilistic_edge": False,
                                "use_monte_carlo": False,
                            })
    candidates = candidates[:max_candidates]

    trials = []
    for cfg in candidates:
        per_stock = []
        for code, df in datasets:
            result = backtest(
                df,
                strategy_name="decision_engine",
                strategy_params={"window": 90, "decision_config": cfg},
                initial_capital=initial_capital,
                max_position=1.0,
                stop_loss=0.0,
                take_profit=0.0,
                code=code,  # Pass stock code for A-share-specific features
            )
            if "error" in result:
                continue
            result["code"] = code
            per_stock.append(result)

        if not per_stock:
            continue

        avg_return = float(np.mean([r["total_return_pct"] for r in per_stock]))
        avg_sharpe = float(np.mean([r["sharpe_ratio"] for r in per_stock]))
        avg_dd = float(np.mean([r["max_drawdown_pct"] for r in per_stock]))
        avg_win_rate = float(np.mean([r["win_rate_pct"] for r in per_stock]))
        avg_trades = float(np.mean([r["n_trades"] for r in per_stock]))
        profitable_ratio = float(np.mean([1.0 if r["total_return_pct"] > 0 else 0.0 for r in per_stock])) * 100

        score = (
            avg_return * 0.7
            + avg_sharpe * 10
            - avg_dd * 0.6
            + profitable_ratio * 0.12
            + min(avg_trades, 10) * 0.5
        )
        trials.append({
            "decision_config": cfg,
            "calibration_score": round(score, 2),
            "avg_return_pct": round(avg_return, 2),
            "avg_sharpe_ratio": round(avg_sharpe, 2),
            "avg_drawdown_pct": round(avg_dd, 2),
            "avg_win_rate_pct": round(avg_win_rate, 1),
            "avg_trades": round(avg_trades, 1),
            "profitable_ratio_pct": round(profitable_ratio, 1),
            "sample_size": len(per_stock),
            "per_stock": per_stock,
        })

    if not trials:
        return {"error": "No valid batch calibration runs completed"}

    trials.sort(
        key=lambda x: (
            x["calibration_score"],
            x["avg_sharpe_ratio"],
            x["avg_return_pct"],
            -x["avg_drawdown_pct"],
            x["profitable_ratio_pct"],
        ),
        reverse=True,
    )
    return {
        "best": trials[0],
        "top_trials": trials[:5],
        "tested": len(trials),
        "sample_size": len(datasets),
    }


def format_calibration_result(result: Dict) -> str:
    """Format decision-engine calibration results."""
    if "error" in result:
        return f"ERROR: {result['error']}"

    best = result["best"]
    lines = [
        "Decision Engine Calibration",
        "=" * 55,
        f"Configurations tested: {result['tested']}",
        "",
        "Best Configuration:",
        f"  Params           : {best['decision_config']}",
        f"  Calibration Score: {best['calibration_score']:+.2f}",
        f"  Total Return     : {best['total_return_pct']:+.2f}%",
        f"  CAGR             : {best['cagr_pct']:+.2f}%",
        f"  Max Drawdown     : {best['max_drawdown_pct']:.2f}%",
        f"  Sharpe Ratio     : {best['sharpe_ratio']:.2f}",
        f"  Win Rate         : {best['win_rate_pct']:.1f}%",
        f"  Trades           : {best['n_trades']}",
        "",
        "Top Candidates:",
    ]

    for i, trial in enumerate(result["top_trials"], 1):
        lines.append(
            f"  {i}. score {trial['calibration_score']:+.2f}  "
            f"return {trial['total_return_pct']:+.2f}%  "
            f"DD {trial['max_drawdown_pct']:.2f}%  "
            f"Sharpe {trial['sharpe_ratio']:.2f}  "
            f"params {trial['decision_config']}"
        )

    return "\n".join(lines)


def format_batch_calibration_result(result: Dict) -> str:
    """Format batch calibration results."""
    if "error" in result:
        return f"ERROR: {result['error']}"

    best = result["best"]
    lines = [
        "Decision Engine Batch Calibration",
        "=" * 55,
        f"Configurations tested: {result['tested']}",
        f"Stock samples       : {result['sample_size']}",
        "",
        "Best Configuration:",
        f"  Params            : {best['decision_config']}",
        f"  Calibration Score : {best['calibration_score']:+.2f}",
        f"  Avg Return        : {best['avg_return_pct']:+.2f}%",
        f"  Avg Drawdown      : {best['avg_drawdown_pct']:.2f}%",
        f"  Avg Sharpe        : {best['avg_sharpe_ratio']:.2f}",
        f"  Avg Win Rate      : {best['avg_win_rate_pct']:.1f}%",
        f"  Avg Trades        : {best['avg_trades']:.1f}",
        f"  Profitable Ratio  : {best['profitable_ratio_pct']:.1f}%",
        "",
        "Top Candidates:",
    ]
    for i, trial in enumerate(result["top_trials"], 1):
        lines.append(
            f"  {i}. score {trial['calibration_score']:+.2f}  "
            f"avg_return {trial['avg_return_pct']:+.2f}%  "
            f"avg_DD {trial['avg_drawdown_pct']:.2f}%  "
            f"avg_Sharpe {trial['avg_sharpe_ratio']:.2f}  "
            f"profitable {trial['profitable_ratio_pct']:.1f}%  "
            f"params {trial['decision_config']}"
        )

    lines.extend(["", "Per-stock results for best configuration:"])
    for item in best.get("per_stock", [])[:10]:
        lines.append(
            f"  {item['code']}: return {item['total_return_pct']:+.2f}%  "
            f"DD {item['max_drawdown_pct']:.2f}%  Sharpe {item['sharpe_ratio']:.2f}  "
            f"trades {item['n_trades']}"
        )
    return "\n".join(lines)


def decision_engine_signal_diagnostics(
    df: pd.DataFrame,
    window: int = 90,
    decision_config: dict = None,
) -> Dict:
    """
    Diagnose why the decision engine does or does not trigger.
    """
    import quant_engine as qe
    from collections import Counter
    from slow_fetcher import _build_trade_decision

    decision_config = decision_config or {}
    use_monte_carlo = decision_config.get("use_monte_carlo", False)
    neutral_market = {
        "score": 0.0,
        "label": "neutral",
        "action_bias": "neutral",
        "summary": "Diagnostics mode uses neutral market regime.",
    }
    neutral_event = {
        "available": False,
        "score": 0.0,
        "label": "neutral",
        "summary": "Diagnostics mode ignores event risk.",
        "recent_earnings": False,
        "high_uncertainty": False,
        "positive_flags": [],
        "negative_flags": [],
    }

    action_counter = Counter()
    blocker_counter = Counter()
    support_counter = Counter()
    buy_checks = []

    for i in range(window, len(df)):
        sub = df.iloc[:i + 1].copy()
        try:
            diag = qe.comprehensive_diagnosis(sub, run_monte_carlo=use_monte_carlo, mc_days=20)
            decision = _build_trade_decision(
                sub,
                diag,
                float(sub.iloc[-1]["close"]),
                neutral_market,
                neutral_event,
                config=decision_config,
            )
            action_counter[decision["action"]] += 1
            buy_checks.append(decision.get("buy_checks_passed", 0))
            for reason in decision.get("reasons_against", []):
                blocker_counter[reason] += 1
            for reason in decision.get("reasons_for", []):
                support_counter[reason] += 1
        except Exception:
            continue

    total = sum(action_counter.values())
    return {
        "total_windows": total,
        "action_counts": dict(action_counter),
        "avg_buy_checks_passed": round(float(np.mean(buy_checks)) if buy_checks else 0.0, 2),
        "top_blockers": blocker_counter.most_common(8),
        "top_supports": support_counter.most_common(8),
        "config": decision_config,
    }


def format_signal_diagnostics(result: Dict) -> str:
    """Format signal diagnostics for the decision engine."""
    lines = [
        "Decision Engine Signal Diagnostics",
        "=" * 55,
        f"Windows analyzed      : {result['total_windows']}",
        f"Action counts         : {result['action_counts']}",
        f"Avg buy checks passed : {result['avg_buy_checks_passed']}",
        f"Config                : {result['config']}",
        "",
        "Top blockers:",
    ]
    for text, count in result.get("top_blockers", []):
        lines.append(f"  - {count:>3}x  {text}")
    lines.append("")
    lines.append("Top supports:")
    for text, count in result.get("top_supports", []):
        lines.append(f"  + {count:>3}x  {text}")
    return "\n".join(lines)


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
