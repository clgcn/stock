"""
Stock Data Tool - A-Share Market Data Fetcher & Analyzer
=========================================================
Data Sources: Eastmoney & Sina Finance (free public APIs, no registration needed)

Features:
  - K-line / candlestick data (daily/weekly/monthly/minute)
  - Real-time quotes
  - Financial fundamentals (PE, PB, ROE, etc.)
  - Technical indicators (MA, MACD, RSI, Bollinger Bands)
  - K-line chart visualization (candlestick + volume + indicators)

CLI Examples:
  python stock_tool.py --code 600519 --period daily --start 2023-01-01
  python stock_tool.py --code 000858 --realtime
  python stock_tool.py --code 600519 --period daily --plot --days 120
"""

try:
    from curl_cffi import requests
except ImportError:
    import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import json
import re
import time
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# ──────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────

_NO_PROXY = {"http": "", "https": ""}


def _headers():
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Referer": "https://finance.eastmoney.com/",
    }


_USE_CURL_CFFI = hasattr(requests, "impersonate")


def _get(url, params=None, extra_headers=None, timeout=15, **kwargs):
    """Unified GET request with browser TLS fingerprint impersonation."""
    headers = _headers()
    if extra_headers:
        headers.update(extra_headers)
    if _USE_CURL_CFFI:
        return requests.get(
            url, params=params, headers=headers,
            timeout=timeout, impersonate="chrome", **kwargs
        )
    return requests.get(
        url, params=params, headers=headers,
        proxies=_NO_PROXY, timeout=timeout, **kwargs
    )


def _get_secid(code: str) -> str:
    """
    Convert stock code to Eastmoney secid format.
    Shanghai (60xxxx, 68xxxx, 51xxxx, 11xxxx) -> 1.xxxxxx
    Shenzhen (00xxxx, 30xxxx, 39xxxx, 15xxxx, 16xxxx) -> 0.xxxxxx
    """
    code = str(code).strip().upper().replace("SH", "").replace("SZ", "")
    if code.startswith(("60", "68", "51", "11")):
        return f"1.{code}"
    elif code.startswith(("00", "30", "39", "15", "16")):
        return f"0.{code}"
    return f"1.{code}"


def _sina_prefix(code: str) -> str:
    """Return Sina quote prefix, e.g. sh600519 / sz000858"""
    code = str(code).strip()
    if code.startswith(("60", "68", "51")):
        return f"sh{code}"
    else:
        return f"sz{code}"


# ──────────────────────────────────────────
# 1. K-line Data
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


def get_kline(
    code: str,
    period: str = "daily",
    start: str = None,
    end: str = None,
    adjust: str = "qfq",
    limit: int = 500,
) -> pd.DataFrame:
    """
    Fetch K-line (candlestick) data from Eastmoney.

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

    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "lmt": limit, "klt": klt, "fqt": fqt, "secid": secid,
        "beg": beg, "end": ened, "_": int(time.time() * 1000),
    }

    try:
        resp = _get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise ConnectionError(f"K-line request failed: {e}")

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
    return df


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
# 3. Financial Data
# ──────────────────────────────────────────

def get_financial(code: str) -> dict:
    """Fetch key financial indicators from Eastmoney (PE, PB, ROE, etc.)."""
    secid = _get_secid(code)
    url = "https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "invt": 2, "fltt": 2,
        "fields": ("f57,f58,f43,f44,f45,f46,f47,f48,f50,f57,f58,"
                    "f100,f116,f117,f162,f163,f164,f167,f168,f169,"
                    "f170,f171,f173,f174,f175,f177,f178"),
        "secid": secid, "_": int(time.time() * 1000),
    }
    try:
        resp = _get(url, params=params)
        data = resp.json()
    except Exception as e:
        raise ConnectionError(f"Financial data request failed: {e}")

    d = data.get("data", {}) or {}

    def safe(key, divisor=1):
        v = d.get(key, "-")
        if v in ["-", None, ""]:
            return None
        try:
            return round(float(v) / divisor, 4)
        except Exception:
            return v

    return {
        "code": code, "name": d.get("f58", code),
        "current": safe("f43"),
        "pe_ttm": safe("f162"), "pe_static": safe("f163"),
        "pb": safe("f167"), "ps_ttm": safe("f164"),
        "total_mv": safe("f116", 1e8), "float_mv": safe("f117", 1e8),
        "roe": safe("f173"), "gross_margin": safe("f174"),
        "net_margin": safe("f175"), "turnover_rate": safe("f168"),
        "volume_ratio": safe("f50"),
    }


# ──────────────────────────────────────────
# 4. Announcements (公告)
# ──────────────────────────────────────────

# 重要公告类型关键词（用于识别财报/业绩相关公告）
_EARNINGS_KEYWORDS = [
    "年度报告", "年报", "半年度报告", "半年报", "中报",
    "季度报告", "季报", "一季报", "三季报",
    "业绩预告", "业绩快报", "业绩修正",
]

# 公告分类名称 -> 简化标签
_COLUMN_LABEL = {
    "年度报告":   "📋 年报",
    "半年度报告": "📋 半年报",
    "季度报告":   "📋 季报",
    "业绩预告":   "📣 业绩预告",
    "业绩快报":   "📣 业绩快报",
    "定期报告":   "📋 定期报告",
}


def get_announcements(code: str, page_size: int = 20, ann_type: str = "ALL") -> list[dict]:
    """
    从东方财富获取个股最新公告列表。

    Parameters
    ----------
    code      : str  股票代码，如 "000423"
    page_size : int  返回条数，默认 20
    ann_type  : str  公告类型筛选，"ALL"=全部 / "A"=定期报告 / "SR"=业绩预告快报

    Returns
    -------
    list of dict，每条包含:
      date     : str   公告日期 "YYYY-MM-DD"
      title    : str   公告标题
      category : str   分类名称（如"年度报告"、"业绩预告"）
      label    : str   简化标签（带 emoji）
      is_earnings : bool  是否为财报/业绩相关公告
      url      : str   公告原文链接（东方财富）
    """
    url = "https://np-anotice-stock.eastmoney.com/api/security/ann"
    params = {
        "sr":           -1,          # 倒序（最新在前）
        "page_size":    page_size,
        "page_index":   1,
        "ann_type":     ann_type,
        "client_source": "web",
        "stock_list":   code,
    }
    try:
        resp = _get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise ConnectionError(f"公告接口请求失败: {e}")

    items = (data.get("data") or {}).get("list") or []
    if not items:
        return []

    results = []
    for item in items:
        raw_date = item.get("notice_date", "")
        date_str = raw_date[:10] if raw_date else ""

        title    = item.get("title", "").strip()
        col      = item.get("column") or {}
        category = col.get("column_name", "")
        art_code = item.get("art_code", "")

        # 简化标签
        label = _COLUMN_LABEL.get(category, category)

        # 判断是否为财报/业绩相关
        is_earnings = any(kw in title or kw in category for kw in _EARNINGS_KEYWORDS)

        # 构造公告原文链接
        ann_url = (
            f"https://np-anotice-stock.eastmoney.com/api/security/ann/{art_code}"
            if art_code else ""
        )

        results.append({
            "date":        date_str,
            "title":       title,
            "category":    category,
            "label":       label,
            "is_earnings": is_earnings,
            "url":         ann_url,
        })

    return results


def format_announcements(code: str, anns: list[dict], name: str = "") -> str:
    """
    将公告列表格式化为中文报告字符串，并在顶部给出财报状态提示。

    Parameters
    ----------
    code : str         股票代码
    anns : list[dict]  get_announcements() 返回值
    name : str         股票名称（可选）

    Returns
    -------
    str  格式化报告
    """
    from datetime import datetime, timedelta

    title_str = f"{name}（{code}）" if name else code
    lines = [
        f"【公告查询】{title_str}",
        f"  查询时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "─" * 50,
    ]

    if not anns:
        lines.append("  暂无近期公告数据")
        return "\n".join(lines)

    # ── 财报状态智能提示 ──
    today = datetime.today().date()
    recent_days = 30
    earnings_recent = [
        a for a in anns
        if a["is_earnings"] and a["date"]
        and (today - datetime.strptime(a["date"], "%Y-%m-%d").date()).days <= recent_days
    ]
    earnings_all = [a for a in anns if a["is_earnings"]]

    if earnings_recent:
        latest = earnings_recent[0]
        lines.append(f"\n  ⚠️  财报提示: {latest['date']} 已披露【{latest['title'][:30]}】")
        lines.append(f"      财务指标数据可能正在更新，估值分析请以最新年报为准。")
    elif earnings_all:
        latest_e = earnings_all[0]
        lines.append(f"\n  📋 最近财报: {latest_e['date']}  {latest_e['title'][:35]}")
    else:
        # 根据当前月份推断是否临近财报披露期
        month = today.month
        if month in (3, 4):
            lines.append(f"\n  📅 年报披露期（1-4月），建议关注近期公告。")
        elif month in (7, 8):
            lines.append(f"\n  📅 半年报披露期（7-8月），建议关注近期公告。")
        elif month == 10:
            lines.append(f"\n  📅 三季报披露期（10月），建议关注近期公告。")

    # ── 公告列表 ──
    lines.append(f"\n  最近 {len(anns)} 条公告：\n")
    for i, ann in enumerate(anns, 1):
        # 财报类公告高亮
        prefix = "  ★" if ann["is_earnings"] else "   "
        label  = f"[{ann['label']}]" if ann["label"] else ""
        lines.append(
            f"{prefix} {ann['date']}  {label}  {ann['title'][:48]}"
        )

    lines.append("\n  ★ = 财报/业绩相关公告")
    lines.append("─" * 50)
    return "\n".join(lines)


# ──────────────────────────────────────────
# 5. Financial History & Earnings Analysis
# ──────────────────────────────────────────

def get_financial_history(code: str, periods: int = 8) -> list[dict]:
    """
    从东方财富获取个股历史多期财务数据（按报告期倒序）。

    Parameters
    ----------
    code    : str  股票代码，如 "000423"
    periods : int  获取期数，默认 8 期（约 2 年）

    Returns
    -------
    list of dict，每条包含:
      report_date : str   报告期 "YYYY-MM-DD"
      report_type : str   报告类型（年报/半年报/一季报/三季报）
      revenue     : float 营业收入（亿元）
      revenue_yoy : float 营收同比增长率 (%)
      net_profit  : float 归母净利润（亿元）
      profit_yoy  : float 净利润同比增长率 (%)
      eps         : float 基本每股收益（元）
      roe         : float 净资产收益率 (%)
      gross_margin: float 毛利率 (%)
    """
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    params = {
        "reportName":  "RPT_LICO_FN_CPD",
        "columns":     "SECURITY_CODE,REPORTDATE,BASIC_EPS,"
                       "TOTAL_OPERATE_INCOME,YSTZ,PARENT_NETPROFIT,"
                       "SJLTZ,XSMLL,WEIGHTAVG_ROE",
        "filter":      f'(SECURITY_CODE="{code}")',
        "pageNumber":  1,
        "pageSize":    periods,
        "sortTypes":   -1,
        "sortColumns": "REPORTDATE",
        "source":      "WEB",
        "client":      "WEB",
    }
    try:
        resp = _get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise ConnectionError(f"历史财务数据请求失败: {e}")

    rows = (data.get("result") or {}).get("data") or []
    if not rows:
        raise ValueError(f"未获取到 {code} 的历史财务数据")

    # 报告期类型映射
    def _report_type(date_str: str) -> str:
        if not date_str:
            return ""
        m = date_str[5:7]
        return {"03": "一季报", "06": "半年报", "09": "三季报", "12": "年报"}.get(m, "")

    def _to_yi(val) -> float | None:
        """转换为亿元，保留2位"""
        try:
            return round(float(val) / 1e8, 2) if val is not None else None
        except Exception:
            return None

    def _pct(val) -> float | None:
        try:
            return round(float(val), 2) if val is not None else None
        except Exception:
            return None

    results = []
    for row in rows:
        rd = (row.get("REPORTDATE") or "")[:10]
        results.append({
            "report_date":  rd,
            "report_type":  _report_type(rd),
            "eps":          _pct(row.get("BASIC_EPS")),
            "revenue":      _to_yi(row.get("TOTAL_OPERATE_INCOME")),
            "revenue_yoy":  _pct(row.get("YSTZ")),
            "net_profit":   _to_yi(row.get("PARENT_NETPROFIT")),
            "profit_yoy":   _pct(row.get("SJLTZ")),
            "gross_margin": _pct(row.get("XSMLL")),
            "roe":          _pct(row.get("WEIGHTAVG_ROE")),
        })
    return results


def analyze_earnings_reaction(
    code: str,
    ann_date: str,
    kline_df=None,
) -> dict:
    """
    分析财报公告当日及前后的市场价格反应。

    Parameters
    ----------
    code      : str        股票代码
    ann_date  : str        公告日期 "YYYY-MM-DD"
    kline_df  : DataFrame  可选，已有的日K数据（避免重复请求）

    Returns
    -------
    dict:
      ann_date       : 公告日期
      reaction_pct   : 公告当日涨跌幅 (%)，None 表示非交易日或数据缺失
      reaction_vol_x : 公告当日成交量 / 20日均量倍数
      pre_pct_5d     : 公告前5日累计涨跌幅 (%)
      post_pct_3d    : 公告后3日累计涨跌幅 (%)（数据可能不足）
      verdict        : 市场反应定性（强烈正面/正面/中性/负面/强烈负面）
      verdict_reason : 简要说明
    """
    from datetime import datetime, timedelta

    # 获取公告前后约 40 个交易日的数据
    try:
        ann_dt = datetime.strptime(ann_date, "%Y-%m-%d")
    except ValueError:
        return {"ann_date": ann_date, "reaction_pct": None, "verdict": "日期格式错误"}

    start = (ann_dt - timedelta(days=60)).strftime("%Y-%m-%d")
    end   = (ann_dt + timedelta(days=15)).strftime("%Y-%m-%d")

    if kline_df is None:
        try:
            kline_df = get_kline(code, period="daily", start=start, end=end, adjust="none")
        except Exception as e:
            return {"ann_date": ann_date, "reaction_pct": None,
                    "verdict": "数据获取失败", "verdict_reason": str(e)}

    df = kline_df.copy()
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    # 找公告日当天（若为非交易日，找最近的下一个交易日）
    ann_idx = None
    for offset in range(5):
        check = (ann_dt + timedelta(days=offset)).strftime("%Y-%m-%d")
        match = df[df["date_str"] == check]
        if not match.empty:
            ann_idx = match.index[0]
            break

    if ann_idx is None:
        return {"ann_date": ann_date, "reaction_pct": None,
                "verdict": "中性", "verdict_reason": "未找到公告日附近的交易数据"}

    row = df.loc[ann_idx]
    reaction_pct = float(row["pct_chg"])

    # 成交量倍数（与前20日均量比较）
    pre_rows = df[df.index < ann_idx].tail(20)
    avg_vol = pre_rows["volume"].mean() if len(pre_rows) >= 5 else None
    vol_x = round(float(row["volume"]) / avg_vol, 2) if avg_vol and avg_vol > 0 else None

    # 公告前5日累计涨跌
    pre5 = df[df.index < ann_idx].tail(5)
    pre_pct_5d = None
    if len(pre5) >= 3:
        pre_pct_5d = round(float(pre5["pct_chg"].sum()), 2)

    # 公告后3日累计涨跌（可能数据不足）
    post3 = df[df.index > ann_idx].head(3)
    post_pct_3d = round(float(post3["pct_chg"].sum()), 2) if not post3.empty else None

    # 定性判断
    if reaction_pct >= 7:
        verdict = "强烈正面"
        reason  = f"财报公告当日涨幅 {reaction_pct:+.2f}%，市场反应热烈，大幅超预期"
    elif reaction_pct >= 3:
        verdict = "正面"
        reason  = f"财报公告当日涨幅 {reaction_pct:+.2f}%，市场认可度高，优于预期"
    elif reaction_pct >= -2:
        verdict = "中性"
        reason  = f"财报公告当日涨跌 {reaction_pct:+.2f}%，市场反应平淡，符合预期"
    elif reaction_pct >= -5:
        verdict = "负面"
        reason  = f"财报公告当日下跌 {reaction_pct:+.2f}%，市场对财报有所失望"
    else:
        verdict = "强烈负面"
        reason  = f"财报公告当日大跌 {reaction_pct:+.2f}%，财报明显低于预期"

    if vol_x and vol_x >= 2.0:
        reason += f"，成交量是平日 {vol_x:.1f} 倍（资金高度关注）"

    return {
        "ann_date":      ann_date,
        "reaction_pct":  reaction_pct,
        "reaction_vol_x": vol_x,
        "pre_pct_5d":    pre_pct_5d,
        "post_pct_3d":   post_pct_3d,
        "verdict":       verdict,
        "verdict_reason": reason,
    }


def format_earnings_analysis(
    code: str,
    name: str,
    history: list[dict],
    reactions: list[dict],
) -> str:
    """
    将历史财务数据 + 历次财报市场反应格式化为中文分析报告。
    """
    from datetime import datetime
    lines = [
        "═" * 56,
        f"  财报深度分析  {name}（{code}）",
        f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "═" * 56,
    ]

    # ── 一、历史财务趋势 ──
    lines += ["", "【历史财务数据趋势】（近期优先）", ""]

    header = f"  {'报告期':<12} {'类型':<6} {'营收(亿)':<10} {'营收同比':<10} {'净利润(亿)':<11} {'利润同比':<10} {'毛利率':<8} {'ROE'}"
    lines.append(header)
    lines.append("  " + "─" * 78)

    for h in history:
        rv_yoy = f"{h['revenue_yoy']:+.1f}%" if h['revenue_yoy'] is not None else "N/A"
        py_yoy = f"{h['profit_yoy']:+.1f}%" if h['profit_yoy'] is not None else "N/A"
        rv     = f"{h['revenue']:.2f}" if h['revenue'] is not None else "N/A"
        np_    = f"{h['net_profit']:.2f}" if h['net_profit'] is not None else "N/A"
        gm     = f"{h['gross_margin']:.1f}%" if h['gross_margin'] is not None else "N/A"
        roe    = f"{h['roe']:.1f}%" if h['roe'] is not None else "N/A"

        # 用箭头直观显示同比方向
        rv_arrow = "▲" if h['revenue_yoy'] and h['revenue_yoy'] > 0 else ("▼" if h['revenue_yoy'] and h['revenue_yoy'] < 0 else " ")
        py_arrow = "▲" if h['profit_yoy'] and h['profit_yoy'] > 0 else ("▼" if h['profit_yoy'] and h['profit_yoy'] < 0 else " ")

        lines.append(
            f"  {h['report_date']:<12} {h['report_type']:<6} "
            f"{rv:<8} {rv_arrow}{rv_yoy:<9} "
            f"{np_:<9} {py_arrow}{py_yoy:<9} "
            f"{gm:<8} {roe}"
        )

    # 趋势判断
    if len(history) >= 2:
        lines.append("")
        latest = history[0]
        prev   = history[1]
        trends = []
        if latest["profit_yoy"] is not None:
            if latest["profit_yoy"] > 20:
                trends.append(f"净利润同比增长 {latest['profit_yoy']:.1f}%，盈利增速强劲")
            elif latest["profit_yoy"] > 0:
                trends.append(f"净利润同比增长 {latest['profit_yoy']:.1f}%，保持正增长")
            elif latest["profit_yoy"] > -20:
                trends.append(f"净利润同比下滑 {abs(latest['profit_yoy']):.1f}%，盈利承压")
            else:
                trends.append(f"净利润同比大幅下滑 {abs(latest['profit_yoy']):.1f}%，基本面走弱")
        if latest["gross_margin"] is not None and prev["gross_margin"] is not None:
            gm_delta = latest["gross_margin"] - prev["gross_margin"]
            if abs(gm_delta) > 1:
                trends.append(f"毛利率较上期{'提升' if gm_delta > 0 else '下降'} {abs(gm_delta):.1f} pct")
        if trends:
            lines.append("  趋势解读: " + "；".join(trends))

    # ── 二、历次财报市场反应 ──
    if reactions:
        lines += ["", "─" * 56, "【历次财报公告市场反应】", ""]
        for r in reactions:
            verdict_icon = {
                "强烈正面": "🟢🟢", "正面": "🟢", "中性": "⚪",
                "负面": "🔴", "强烈负面": "🔴🔴"
            }.get(r.get("verdict", ""), "")

            lines.append(f"  {verdict_icon} {r['ann_date']}  {r.get('verdict','')}")
            lines.append(f"     {r.get('verdict_reason','')}")
            if r.get("pre_pct_5d") is not None:
                lines.append(f"     公告前5日累涨跌: {r['pre_pct_5d']:+.2f}%"
                             + (f"  |  公告后3日: {r['post_pct_3d']:+.2f}%" if r.get("post_pct_3d") is not None else ""))
            lines.append("")

    # ── 三、综合建议 ──
    lines += ["─" * 56, "【财报对当前分析的影响】", ""]
    latest_r = reactions[0] if reactions else None
    if latest_r:
        v = latest_r.get("verdict", "")
        if v in ("强烈正面", "正面"):
            lines += [
                "  · 最新财报超预期，股价已做出正面反应",
                "  · 基本面形成支撑，回调可视为买点参考",
                "  · news_sentiment 建议在技术分析基础上 +0.2 ~ +0.3",
            ]
        elif v == "中性":
            lines += [
                "  · 财报符合预期，短期价格波动主要由技术面/资金面驱动",
                "  · news_sentiment 维持原有外围市场判断，无需额外调整",
            ]
        else:
            lines += [
                "  · 财报低于预期，股价已承压，基本面利空尚未完全消化",
                "  · news_sentiment 建议在技术分析基础上 -0.2 ~ -0.3",
                "  · 建议等待财报数据被市场充分定价后再考虑入场",
            ]

    lines += [
        "",
        "  ⚠️  财报分析仅供参考，不构成投资建议。",
        "═" * 56,
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────
# 6. Technical Indicators
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
# 5. Visualization
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
# 6. CLI Entry Point
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
        if not args.code: print("Specify --code"); sys.exit(1)
        for k, v in get_financial(args.code).items(): print(f"  {k:<16}: {v}")
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
    if len(sys.argv) == 1: print("Usage: python stock_tool.py --help")
    else: _cli()
