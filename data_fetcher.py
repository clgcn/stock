"""
数据获取模块 (data_fetcher) — 技术面依赖
==========================================
提供:
  get_kline()        → K线/蜡烛图数据 (日/周/月/分钟) — 含重试+退避
  get_realtime()     → 新浪实时行情
  add_indicators()   → 技术指标计算 (MA/MACD/RSI/BOLL/KDJ)
  plot_kline()       → K线可视化
  _cli()             → 命令行入口

常量:
  PERIOD_MAP, ADJUST_MAP
"""

from _http_utils import _get, _get_secid, _sina_prefix, eastmoney_throttle, kline_cache

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import re
import time
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path


# ──────────────────────────────────────────
# 常量
# ──────────────────────────────────────────

PERIOD_MAP = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30, "60m": 60,
    "daily": 101, "weekly": 102, "monthly": 103,
}

ADJUST_MAP = {
    "none": 0,   # no adjustment
    "qfq": 1,    # forward adjustment
    "hfq": 2,    # backward adjustment
}


# ──────────────────────────────────────────
# 1. K-line Data (东方财富 + 重试退避)
# ──────────────────────────────────────────

_KLINE_MAX_RETRIES = 3
_KLINE_BACKOFF_BASE = 1.5  # 秒: 1.5, 3.0, 6.0


def get_kline(
    code: str,
    period: str = "daily",
    start: str = None,
    end: str = None,
    adjust: str = "qfq",
    limit: int = 500,
    _skip_cache: bool = False,
) -> pd.DataFrame:
    """
    Fetch K-line (candlestick) data from Eastmoney.

    内置三层保护:
      1) 内存缓存 — 同一进程内相同参数 5 分钟内不重复请求
      2) 全局令牌桶限流 — 控制所有东财请求 ≤ 2 QPS，并发排队
      3) 重试 + 指数退避 — 万一触发限流也能自动恢复

    Args:
        code    Stock code, e.g. "600519", "000858"
        period  Period: daily/weekly/monthly/1m/5m/15m/30m/60m
        start   Start date "YYYY-MM-DD" (if empty, fetch latest `limit` bars)
        end     End date "YYYY-MM-DD"
        adjust  Adjustment: qfq (forward) / hfq (backward) / none
        limit   Max number of bars

    Returns:
        DataFrame: date, open, close, high, low, volume, amount,
                   amplitude, pct_chg, change, turnover
    """
    klt = PERIOD_MAP.get(period, 101)
    fqt = ADJUST_MAP.get(adjust, 1)
    secid = _get_secid(code)

    beg = start.replace("-", "") if start else "19900101"
    ened = end.replace("-", "") if end else datetime.today().strftime("%Y%m%d")

    # ── 缓存检查 ──
    cache_key = (code, period, beg, ened, adjust, limit)
    if not _skip_cache:
        cached = kline_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "lmt": limit, "klt": klt, "fqt": fqt, "secid": secid,
        "beg": beg, "end": ened, "_": int(time.time() * 1000),
    }

    last_err = None
    for attempt in range(_KLINE_MAX_RETRIES):
        try:
            # ── 限流: 排队等令牌 ──
            eastmoney_throttle.acquire()

            resp = _get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            klines = data.get("data", {}) or {}
            raw = klines.get("klines") or []
            if not raw:
                raise ValueError(f"No K-line data for {code} (empty response)")

            cols = ["date", "open", "close", "high", "low",
                    "volume", "amount", "amplitude", "pct_chg", "change", "turnover"]
            rows = [line.split(",") for line in raw]
            df = pd.DataFrame(rows, columns=cols)

            df["date"] = pd.to_datetime(df["date"])
            for c in cols[1:]:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            df = df.sort_values("date").reset_index(drop=True)
            df.attrs["code"] = code
            df.attrs["name"] = klines.get("name", code)
            df.attrs["period"] = period

            # ── 写入缓存 ──
            kline_cache.put(cache_key, df)
            return df

        except Exception as e:
            last_err = e
            if attempt < _KLINE_MAX_RETRIES - 1:
                wait = _KLINE_BACKOFF_BASE * (2 ** attempt)
                time.sleep(wait)
                # 刷新时间戳避免服务端缓存
                params["_"] = int(time.time() * 1000)

    raise ConnectionError(
        f"K-line request failed for {code} after {_KLINE_MAX_RETRIES} retries: {last_err}"
    )


# ──────────────────────────────────────────
# 1b. DB 优先 K-line（减少 API 调用）
# ──────────────────────────────────────────

def get_kline_prefer_db(
    code: str,
    period: str = "daily",
    start: str = None,
    adjust: str = "qfq",
    min_bars: int = 30,
    local_only: bool = False,
) -> pd.DataFrame:
    """优先从本地 SQLite 读取 K 线，数据不足时可回退到在线 API。

    用于 Face 模块等批量分析场景，大幅减少并发 API 请求数。
    本地数据取自 slow_fetcher 定期拉取的 stock_history 表。

    Args:
        code        股票代码
        period      仅 "daily" 支持本地，其他自动走 API
        start       起始日期 "YYYY-MM-DD"
        adjust      复权方式 (本地数据默认前复权)
        min_bars    最少需要多少条数据才算够用
        local_only  True = 纯本地模式，DB 没数据就返回 None，绝不发 API。
                    用于盘前选股等批量场景，确保零网络请求。

    Returns:
        DataFrame（与 get_kline() 格式一致），或 local_only 模式下数据不足时返回 None
    """
    if period != "daily" and not local_only:
        return get_kline(code, period=period, start=start, adjust=adjust)
    if period != "daily" and local_only:
        return None

    # 先尝试本地 DB
    try:
        from slow_fetcher import load_stock_history
        df = load_stock_history(code)
        if not df.empty and "date" in df.columns:
            if start:
                df = df[df["date"] >= start].copy()
            if len(df) >= min_bars:
                df = df.sort_values("date").reset_index(drop=True)
                df.attrs["code"] = code
                df.attrs["period"] = period
                return df
    except (ImportError, FileNotFoundError, Exception):
        pass

    if local_only:
        return None

    # 本地不够，回退到在线 API（会走限流器+缓存）
    return get_kline(code, period=period, start=start, adjust=adjust)


# ──────────────────────────────────────────
# 2. Real-time Quotes
# ──────────────────────────────────────────

def get_realtime(codes) -> pd.DataFrame:
    """
    Fetch real-time quotes from Sina Finance.

    Args:
        codes  Single code string or list, e.g. "600519" or ["600519","000858"]

    Returns:
        DataFrame with latest price, change, volume, etc.
    """
    if isinstance(codes, str):
        codes = [codes]

    prefixes = ",".join(_sina_prefix(c) for c in codes)
    url = f"https://hq.sinajs.cn/list={prefixes}"

    try:
        resp = _get(url, extra_headers={"Referer": "https://finance.sina.com.cn/"})
        resp.encoding = "gbk"
        text = resp.text
    except Exception as e:
        raise ConnectionError(f"Realtime quote request failed: {e}")

    rows = []
    for match in re.finditer(r'hq_str_(\w+)="([^"]*)"', text):
        symbol = match.group(1)
        vals = match.group(2).split(",")
        if len(vals) < 10:
            continue
        code = symbol[2:]
        try:
            current = float(vals[3])
            prev_close = float(vals[2])
            pct_chg = (current - prev_close) / prev_close * 100 if prev_close else 0
            row = {
                "code": code, "name": vals[0], "current": current,
                "open": float(vals[1]), "prev_close": prev_close,
                "high": float(vals[4]), "low": float(vals[5]),
                "volume": float(vals[8]) / 100,
                "amount": float(vals[9]) / 10000,
                "pct_chg": round(pct_chg, 2),
                "change": round(current - prev_close, 2),
                "date": vals[30] if len(vals) > 30 else "",
                "time": vals[31] if len(vals) > 31 else "",
            }
            rows.append(row)
        except (ValueError, IndexError):
            continue

    if not rows:
        raise ValueError("No quote data returned. Please check stock codes.")
    return pd.DataFrame(rows)


# ──────────────────────────────────────────
# 3. Technical Indicators
# ──────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to a K-line DataFrame:
      MA5/10/20/60, MACD(DIF/DEA/Hist), RSI(6/12/24),
      Bollinger Bands, KDJ, Volume MA5/MA10
    """
    df = df.copy()
    c = df["close"]

    for n in [5, 10, 20, 60]:
        df[f"ma{n}"] = c.rolling(n).mean().round(3)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["dif"] = (ema12 - ema26).round(3)
    df["dea"] = df["dif"].ewm(span=9, adjust=False).mean().round(3)
    df["macd"] = ((df["dif"] - df["dea"]) * 2).round(3)

    delta = c.diff()
    for n in [6, 12, 24]:
        gain = delta.clip(lower=0).rolling(n).mean()
        loss = (-delta.clip(upper=0)).rolling(n).mean()
        rs = gain / loss.replace(0, np.nan)
        df[f"rsi{n}"] = (100 - 100 / (1 + rs)).round(2)

    ma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["boll"] = ma20.round(3)
    df["boll_ub"] = (ma20 + 2 * std20).round(3)
    df["boll_lb"] = (ma20 - 2 * std20).round(3)

    low_min = df["low"].rolling(9).min()
    high_max = df["high"].rolling(9).max()
    rsv = (c - low_min) / (high_max - low_min) * 100
    df["kdj_k"] = rsv.ewm(com=2, adjust=False).mean().round(2)
    df["kdj_d"] = df["kdj_k"].ewm(com=2, adjust=False).mean().round(2)
    df["kdj_j"] = (3 * df["kdj_k"] - 2 * df["kdj_d"]).round(2)

    df["vol_ma5"] = df["volume"].rolling(5).mean().round(0)
    df["vol_ma10"] = df["volume"].rolling(10).mean().round(0)
    return df


# ──────────────────────────────────────────
# 4. Visualization
# ──────────────────────────────────────────

def plot_kline(df, indicator="macd", save_path=None, title=None) -> str:
    """Draw candlestick chart with volume and indicator panel. Returns saved file path."""
    df = df.copy().reset_index(drop=True)

    import matplotlib.font_manager as fm
    plt.rcParams["axes.unicode_minus"] = False
    all_fonts = {f.name for f in fm.fontManager.ttflist}
    for font in ["PingFang SC", "Heiti SC", "STHeiti", "Microsoft YaHei",
                  "SimHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC",
                  "Source Han Sans CN", "DejaVu Sans"]:
        if font in all_fonts:
            plt.rcParams["font.sans-serif"] = [font]
            break
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

    fig = plt.figure(figsize=(16, 10), facecolor="#1a1a2e")
    gs = GridSpec(4, 1, figure=fig, hspace=0.04, height_ratios=[3, 1, 0.8, 1.2])
    ax_k   = fig.add_subplot(gs[0])
    ax_vol = fig.add_subplot(gs[1], sharex=ax_k)
    ax_ind = fig.add_subplot(gs[3], sharex=ax_k)

    bg, grid_c, text_c = "#1a1a2e", "#2a2a4e", "#e0e0e0"
    up_c, dn_c = "#ff4444", "#00cc66"

    for ax in [ax_k, ax_vol, ax_ind]:
        ax.set_facecolor(bg); ax.tick_params(colors=text_c, labelsize=8)
        ax.spines[:].set_color(grid_c); ax.yaxis.tick_right()
        ax.grid(color=grid_c, linewidth=0.4, alpha=0.7)

    x = np.arange(len(df))
    for i, row in df.iterrows():
        color = up_c if row["close"] >= row["open"] else dn_c
        ax_k.bar(i, abs(row["close"]-row["open"]), bottom=min(row["open"],row["close"]),
                 width=0.7, color=color, linewidth=0)
        ax_k.plot([i,i], [row["low"],row["high"]], color=color, linewidth=0.8)

    for col, (clr, lw) in {"ma5":("#ffdd00",1),"ma10":("#ff9900",1),"ma20":("#ff44aa",1.2),"ma60":("#44aaff",1.2)}.items():
        if col in df.columns:
            ax_k.plot(x, df[col], color=clr, linewidth=lw, label=col.upper())
    ax_k.legend(loc="upper left", fontsize=7, facecolor=bg, labelcolor=text_c, framealpha=0.8)

    if indicator == "boll" and "boll" in df.columns:
        ax_k.plot(x, df["boll"], color="#aaaaff", linewidth=1, linestyle="--", label="BOLL")
        ax_k.plot(x, df["boll_ub"], color="#ffaaaa", linewidth=0.8, label="UB")
        ax_k.plot(x, df["boll_lb"], color="#aaffaa", linewidth=0.8, label="LB")
        ax_k.fill_between(x, df["boll_ub"], df["boll_lb"], alpha=0.06, color="#8888ff")

    vc = [up_c if r["close"]>=r["open"] else dn_c for _,r in df.iterrows()]
    ax_vol.bar(x, df["volume"]/1e4, color=vc, width=0.7, linewidth=0)
    if "vol_ma5" in df.columns: ax_vol.plot(x, df["vol_ma5"]/1e4, color="#ffdd00", linewidth=0.8)
    if "vol_ma10" in df.columns: ax_vol.plot(x, df["vol_ma10"]/1e4, color="#ff9900", linewidth=0.8)
    ax_vol.set_ylabel("Vol(10k)", color=text_c, fontsize=8)

    if indicator == "macd" and "macd" in df.columns:
        ax_ind.bar(x, df["macd"], color=[up_c if v>=0 else dn_c for v in df["macd"]], width=0.7, linewidth=0, alpha=0.8)
        ax_ind.plot(x, df["dif"], color="#ffdd00", linewidth=0.9, label="DIF")
        ax_ind.plot(x, df["dea"], color="#ff9900", linewidth=0.9, label="DEA")
        ax_ind.axhline(0, color=grid_c, linewidth=0.5)
        ax_ind.legend(loc="upper left", fontsize=7, facecolor=bg, labelcolor=text_c, framealpha=0.8)
        ax_ind.set_ylabel("MACD", color=text_c, fontsize=8)
    elif indicator == "rsi" and "rsi6" in df.columns:
        ax_ind.plot(x, df["rsi6"], color="#ffdd00", linewidth=0.9, label="RSI6")
        ax_ind.plot(x, df["rsi12"], color="#ff9900", linewidth=0.9, label="RSI12")
        ax_ind.plot(x, df["rsi24"], color="#44aaff", linewidth=0.9, label="RSI24")
        ax_ind.axhline(70, color="#ff4444", linewidth=0.6, linestyle="--")
        ax_ind.axhline(30, color="#00cc66", linewidth=0.6, linestyle="--")
        ax_ind.set_ylim(0, 100)
        ax_ind.legend(loc="upper left", fontsize=7, facecolor=bg, labelcolor=text_c, framealpha=0.8)
        ax_ind.set_ylabel("RSI", color=text_c, fontsize=8)
    elif indicator == "kdj" and "kdj_k" in df.columns:
        ax_ind.plot(x, df["kdj_k"], color="#ffdd00", linewidth=0.9, label="K")
        ax_ind.plot(x, df["kdj_d"], color="#ff9900", linewidth=0.9, label="D")
        ax_ind.plot(x, df["kdj_j"], color="#44aaff", linewidth=0.9, label="J")
        ax_ind.axhline(80, color="#ff4444", linewidth=0.6, linestyle="--")
        ax_ind.axhline(20, color="#00cc66", linewidth=0.6, linestyle="--")
        ax_ind.legend(loc="upper left", fontsize=7, facecolor=bg, labelcolor=text_c, framealpha=0.8)
        ax_ind.set_ylabel("KDJ", color=text_c, fontsize=8)

    n = len(df); step = max(n//10, 1); ticks = list(range(0, n, step))
    ax_ind.set_xticks(ticks)
    ax_ind.set_xticklabels([df.iloc[i]["date"].strftime("%Y-%m-%d") for i in ticks], rotation=30, ha="right", fontsize=7)
    plt.setp(ax_k.get_xticklabels(), visible=False); plt.setp(ax_vol.get_xticklabels(), visible=False)

    code = df.attrs.get("code",""); name = df.attrs.get("name",""); period = df.attrs.get("period","")
    ttl = title or f"{name} ({code}) {period} K-line"
    last = df.iloc[-1]; pct = last.get("pct_chg", 0)
    fig.suptitle(f"{ttl}   Close {last['close']:.2f}  {'UP' if pct>=0 else 'DN'} {abs(pct):.2f}%",
                 color=text_c, fontsize=13, y=0.98)

    if save_path is None:
        out_dir = Path(__file__).parent / "charts"; out_dir.mkdir(exist_ok=True)
        save_path = str(out_dir / f"{code}_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=bg); plt.close(fig)
    return save_path


# ──────────────────────────────────────────
# 5. CLI Entry Point
# ──────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description="A-Share Stock Data Tool")
    parser.add_argument("--code", type=str, help="Stock code e.g. 600519")
    parser.add_argument("--codes", nargs="+", help="Multiple codes for batch realtime")
    parser.add_argument("--period", type=str, default="daily", choices=list(PERIOD_MAP.keys()))
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=250, help="Recent N days (default 250)")
    parser.add_argument("--adjust", type=str, default="qfq", choices=["qfq","hfq","none"])
    parser.add_argument("--realtime", action="store_true", help="Fetch realtime quotes")
    parser.add_argument("--financial", action="store_true", help="Fetch financial data")
    parser.add_argument("--plot", action="store_true", help="Generate K-line chart")
    parser.add_argument("--indicator", type=str, default="macd", choices=["macd","rsi","kdj","boll"])
    parser.add_argument("--no-indicator", dest="no_tech", action="store_true")
    parser.add_argument("--save", type=str, help="Chart save path")
    parser.add_argument("--export", type=str, help="Export K-line to CSV")
    args = parser.parse_args()

    if args.realtime:
        codes = args.codes or ([args.code] if args.code else None)
        if not codes: print("Specify --code or --codes"); sys.exit(1)
        df = get_realtime(codes)
        print(df[["code","name","current","pct_chg","change","open","high","low","prev_close","volume","amount","date","time"]].to_string(index=False))
        return
    if args.financial:
        print("Financial data moved to financial.py module")
        return
    if not args.code: parser.print_help(); sys.exit(0)

    start = args.start or (datetime.today() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    df = get_kline(args.code, period=args.period, start=start, end=args.end, adjust=args.adjust)
    print(f"Total {len(df)} bars  Name: {df.attrs.get('name','')}")
    if not args.no_tech: df = add_indicators(df)
    pd.set_option("display.max_columns", None); pd.set_option("display.width", 200)
    print(df.tail(10).to_string(index=False))
    if args.export: df.to_csv(args.export, index=False, encoding="utf-8-sig"); print(f"Exported: {args.export}")
    if args.plot:
        if args.no_tech: df = add_indicators(df)
        print(f"Chart saved: {plot_kline(df, indicator=args.indicator, save_path=args.save)}")


if __name__ == "__main__":
    if len(sys.argv) == 1: print("Usage: python data_fetcher.py --help")
    else: _cli()
