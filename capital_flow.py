"""
资金流向模块 (capital_flow) — 资金面依赖
=========================================
提供:
  get_northbound_flow()      → 北向资金（沪深港通）成交活跃度
  format_northbound_report() → 北向资金报告
  get_moneyflow()            → 个股主力资金流向
  format_moneyflow_report()  → 主力资金报告
  get_margin_trading()       → 融资融券余额
  format_margin_report()     → 两融报告

注意: 2024年5月起港交所不再披露北向资金实时净买入数据，
      本模块改为基于"成交额趋势 + 十大活跃股"评估外资活跃度。
"""

from _http_utils import _get, _get_secid, eastmoney_throttle
import time
import logging

_log = logging.getLogger(__name__)


# ──────────────────────────────────────────
# 1. 北向资金（沪深港通）— 成交额 + 活跃股
# ──────────────────────────────────────────

def _fetch_northbound_deal(days: int) -> list:
    """
    从东方财富数据中心获取北向资金每日成交额数据。
    2024年5月后港交所仅披露成交总额，不再披露净买入。
    """
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    params = {
        "reportName":  "RPT_MUTUAL_DEAL_HISTORY",
        "columns":     "ALL",
        "filter":      '(MUTUAL_TYPE in ("001","005"))',
        "pageNumber":  1,
        "pageSize":    days * 2 + 5,
        "sortTypes":   -1,
        "sortColumns": "TRADE_DATE",
        "source":      "WEB",
        "client":      "WEB",
    }
    eastmoney_throttle.acquire()
    resp = _get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    rows = (data.get("result") or {}).get("data") or []
    if not rows:
        raise ValueError("datacenter: empty northbound deal data")

    day_map = {}
    for r in rows:
        date_str = (r.get("TRADE_DATE") or "")[:10]
        if not date_str:
            continue
        mtype = r.get("MUTUAL_TYPE", "")
        deal_amt = float(r.get("DEAL_AMT") or 0) / 1e4  # 万→亿
        deal_num = int(r.get("DEAL_NUM") or 0)

        if date_str not in day_map:
            day_map[date_str] = {
                "date": date_str,
                "sh_deal_amt": 0.0, "sz_deal_amt": 0.0,
                "sh_deal_num": 0, "sz_deal_num": 0,
                "total_deal_amt": 0.0, "total_deal_num": 0,
            }
        if mtype == "001":  # 沪股通
            day_map[date_str]["sh_deal_amt"] = round(deal_amt, 2)
            day_map[date_str]["sh_deal_num"] = deal_num
        elif mtype == "005":  # 深股通
            day_map[date_str]["sz_deal_amt"] = round(deal_amt, 2)
            day_map[date_str]["sz_deal_num"] = deal_num

    result = sorted(day_map.values(), key=lambda x: x["date"], reverse=True)
    for d in result:
        d["total_deal_amt"] = round(d["sh_deal_amt"] + d["sz_deal_amt"], 2)
        d["total_deal_num"] = d["sh_deal_num"] + d["sz_deal_num"]
    return result[:days]


def _fetch_northbound_top10(days: int = 3) -> list:
    """获取北向资金十大活跃股（仅最近几天）。"""
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    params = {
        "reportName":  "RPT_MUTUAL_STOCK_NORTHSTA",
        "columns":     "ALL",
        "pageNumber":  1,
        "pageSize":    days * 20,
        "sortTypes":   -1,
        "sortColumns": "TRADE_DATE",
        "source":      "WEB",
        "client":      "WEB",
    }
    try:
        eastmoney_throttle.acquire()
        resp = _get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        rows = (data.get("result") or {}).get("data") or []
        return rows
    except Exception:
        return []


def get_northbound_flow(days: int = 10) -> list:
    """
    获取北向资金（沪深港通）每日成交活跃度数据。

    ⚠️ 2024年5月起，港交所不再披露北向资金净买入数据。
    本接口改为返回每日成交额和笔数，用于评估外资参与度趋势。

    Parameters
    ----------
    days : int   返回最近N个交易日数据，默认10日

    Returns
    -------
    list of dict, 每条包含:
        date            : str   交易日
        sh_deal_amt     : float 沪股通成交额（亿元）
        sz_deal_amt     : float 深股通成交额（亿元）
        total_deal_amt  : float 北向合计成交额（亿元）
        total_deal_num  : int   合计成交笔数
        total_net       : float 净流入（始终为0，兼容旧接口）
    """
    try:
        result = _fetch_northbound_deal(days)
        # 兼容旧接口：添加 total_net 字段（始终为0）
        for d in result:
            d["total_net"] = 0.0
            d.setdefault("sh_net", 0.0)
            d.setdefault("sz_net", 0.0)
        return result
    except Exception as e:
        raise ConnectionError(f"北向资金成交额获取失败: {e}")


def format_northbound_report(flow_data: list) -> str:
    """
    将北向资金成交活跃度数据格式化为中文报告。
    2024年5月后港交所不再披露净买入，改为展示成交额趋势。
    """
    if not flow_data:
        return "ERROR: 无北向资金数据"

    lines = [
        "北向资金成交活跃度报告（沪深港通）",
        f"  数据区间: {flow_data[-1]['date']} ~ {flow_data[0]['date']}",
        "=" * 65,
        f"  {'日期':<12} {'沪股通成交':>10} {'深股通成交':>10} {'合计成交':>10} {'笔数':>8}",
        "-" * 65,
    ]

    for d in flow_data:
        sh_amt = d.get("sh_deal_amt", 0)
        sz_amt = d.get("sz_deal_amt", 0)
        total_amt = d.get("total_deal_amt", 0)
        total_num = d.get("total_deal_num", 0)
        lines.append(
            f"  {d['date']:<12} {sh_amt:>9.1f}亿 {sz_amt:>9.1f}亿 "
            f"{total_amt:>9.1f}亿 {total_num:>7,}"
        )

    lines.append("-" * 65)

    # 成交额趋势分析 — 过滤零流量日（港股节假日伪影）
    amts_raw = [d.get("total_deal_amt", 0) for d in flow_data]
    amts = [a for a in amts_raw if a > 0]  # skip HK holiday zero-flow days
    if not amts:
        amts = amts_raw  # fallback: no data at all, use raw
    recent5 = amts[:5]
    early5 = amts[-5:] if len(amts) >= 10 else amts[:5]
    avg_recent = sum(recent5) / len(recent5) if recent5 else 0
    avg_early = sum(early5) / len(early5) if early5 else 0
    trend_pct = (avg_recent / avg_early - 1) * 100 if avg_early > 0 else 0

    # 成交额变化方向
    if len(amts) >= 3:
        recent3 = amts[:3]
        increasing = all(recent3[i] >= recent3[i+1] for i in range(len(recent3)-1))
        decreasing = all(recent3[i] <= recent3[i+1] for i in range(len(recent3)-1))
        if increasing:
            trend_dir = "连续放量"
        elif decreasing:
            trend_dir = "连续缩量"
        else:
            trend_dir = "震荡波动"
    else:
        trend_dir = "数据不足"

    lines.extend([
        f"  近5日日均成交额:  {avg_recent:.1f} 亿元",
        f"  早期5日日均成交额: {avg_early:.1f} 亿元",
        f"  成交额变化趋势:  {trend_pct:+.1f}%  ({trend_dir})",
        "",
    ])

    # 活跃度信号判定 — 结合绝对量（>10亿为有意义参与）+ 趋势
    _MEANINGFUL_THRESHOLD = 10.0  # 10亿/日为有意义北向参与水位
    abs_level = "高" if avg_recent >= _MEANINGFUL_THRESHOLD else "低"
    if avg_recent < _MEANINGFUL_THRESHOLD:
        _trend_pfx = f"（日均{avg_recent:.1f}亿<{_MEANINGFUL_THRESHOLD}亿参考阈值，信号参考价值有限）"
    else:
        _trend_pfx = f"（日均{avg_recent:.1f}亿）"

    if trend_pct > 20:
        signal = f"外资成交大幅放量{_trend_pfx}"
    elif trend_pct > 5:
        signal = f"外资成交温和放量{_trend_pfx}"
    elif trend_pct > -5:
        signal = f"外资成交基本持平{_trend_pfx}"
    elif trend_pct > -20:
        signal = f"外资成交温和缩量{_trend_pfx}"
    else:
        signal = f"外资成交大幅缩量{_trend_pfx}"

    lines.extend([
        f"  北向资金活跃度信号: {signal}",
        "",
        "注：2024年5月起港交所不再披露北向资金净买入数据，本报告基于成交额趋势评估外资参与度。",
        "    成交额持续放量表示外资关注度提升；缩量表示兴趣减退。方向需结合市场走势综合判断。",
    ])

    return "\n".join(lines)


# ──────────────────────────────────────────
# 2. 主力资金流向（个股大单净流入）
# ──────────────────────────────────────────

def _tencent_symbol(code: str) -> str:
    code = str(code).strip().upper().replace("SH", "").replace("SZ", "")
    if code.startswith(("60", "68", "51", "11")):
        return f"sh{code}"
    return f"sz{code}"


def _get_moneyflow_eastmoney(code: str, days: int) -> list:
    """从东方财富拉取主力资金流向（含详细分级）。"""
    secid = _get_secid(code)
    url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
    params = {
        "lmt":    days,
        "klt":    101,
        "secid":  secid,
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
        "_":      int(time.time() * 1000),
    }
    eastmoney_throttle.acquire()
    resp = _get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    klines = ((data.get("data") or {}).get("klines") or [])
    if not klines:
        raise ValueError(f"东财未返回 {code} 资金流向数据")

    def _yi(v):
        try: return round(float(v) / 1e8, 4)
        except Exception: return None
    def _pct(v):
        try: return round(float(v), 2)
        except Exception: return None
    def _f(v):
        try: return round(float(v), 2)
        except Exception: return None

    # 东财 fflow daykline 字段实际顺序 (f51..f65):
    #   [0]=date
    #   [1]=main_net  [2]=small_net  [3]=mid_net  [4]=big_net  [5]=super_net   (单位: 元)
    #   [6]=main_pct  [7]=small_pct  [8]=mid_pct  [9]=big_pct  [10]=super_pct  (单位: %)
    #   [11]=close (元)  [12]=pct_chg (%)  [13]=?  [14]=?
    results = []
    for line in klines[-days:]:
        parts = line.split(",")
        if len(parts) < 13:
            continue
        results.append({
            "date":         parts[0],
            "close":        _f(parts[11]),
            "pct_chg":      _pct(parts[12]),
            "main_net":     _yi(parts[1]),
            "small_net":    _yi(parts[2]),
            "mid_net":      _yi(parts[3]),
            "big_net":      _yi(parts[4]),
            "super_net":    _yi(parts[5]),
            "main_net_pct": _pct(parts[6]),
        })
    return results


def _xueqiu_symbol(code: str) -> str:
    """600519 → SH600519, 000001 → SZ000001 (大写前缀, 与雪球一致)"""
    code = str(code).strip().upper().replace("SH", "").replace("SZ", "")
    if code.startswith(("60", "68", "51", "11")):
        return f"SH{code}"
    return f"SZ{code}"


# ── 雪球 cookie 缓存 ──────────────────────────────────────────
# 雪球所有 v5 API 都要求请求带 xq_a_token cookie, 否则返回 400003 鉴权失败。
# token 通过访问 xueqiu.com 首页自动 set-cookie 拿到, 有效期约 24 小时。
_XUEQIU_SESSION = None
_XUEQIU_COOKIE_TS = 0
_XUEQIU_COOKIE_TTL = 3600 * 12   # 12 小时复用


def _get_xueqiu_session():
    """获取已带 cookie 的 requests session, 12 小时复用一次。"""
    # COMPLIANCE WARNING: 此实现使用浏览器指纹模拟(impersonate="chrome")，
    # 可能违反雪球用户协议。TODO: 替换为合规数据源，
    # 如 akshare.stock_individual_fund_flow() 等官方接口。
    global _XUEQIU_SESSION, _XUEQIU_COOKIE_TS
    now = time.time()
    if _XUEQIU_SESSION is not None and now - _XUEQIU_COOKIE_TS < _XUEQIU_COOKIE_TTL:
        return _XUEQIU_SESSION

    try:
        from curl_cffi import requests as _xq_requests
        sess = _xq_requests.Session(impersonate="chrome")
    except ImportError:
        import requests as _xq_requests
        sess = _xq_requests.Session()

    sess.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": "https://xueqiu.com/",
        "Origin": "https://xueqiu.com",
    })
    # 访问首页拿 xq_a_token 等 cookie
    sess.get("https://xueqiu.com/", timeout=15)
    _XUEQIU_SESSION = sess
    _XUEQIU_COOKIE_TS = now
    return sess


def _get_moneyflow_xueqiu(code: str, days: int) -> list:
    """从雪球拉取主力资金流向（fallback 数据源, 字段完整）。

    端点: stock.xueqiu.com/v5/stock/capital/assort.json
    返回字段对齐东财分级:
      main_net_inflows  → main_net (主力 = 超大单 + 大单)
      super_net_inflows → super_net
      large_net_inflows → big_net
      medium_net_inflows → mid_net
      small_net_inflows → small_net
    单位: 元
    """
    sym = _xueqiu_symbol(code)
    sess = _get_xueqiu_session()

    url = "https://stock.xueqiu.com/v5/stock/capital/assort.json"
    params = {"symbol": sym, "count": max(days, 30)}
    try:
        resp = sess.get(url, params=params, timeout=15)
    except Exception as e:
        raise ConnectionError(f"雪球请求失败: {e}")
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("error_code") not in (0, "0", None):
        # 鉴权失败时清缓存重试一次
        global _XUEQIU_SESSION
        _XUEQIU_SESSION = None
        raise ValueError(
            f"雪球返回 error_code={payload.get('error_code')}: {payload.get('error_description')}"
        )

    items = (payload.get("data") or {}).get("items") or []
    if not items:
        raise ValueError(f"雪球未返回 {code} 资金流向数据")

    def _yi(v):
        try: return round(float(v) / 1e8, 4)
        except Exception: return None
    def _f(v):
        try: return round(float(v), 2)
        except Exception: return None
    def _pct(v):
        try: return round(float(v), 2)
        except Exception: return None

    results = []
    # 雪球默认升序, 取最近 days 条
    for it in items[-days:]:
        ts = it.get("timestamp")
        if ts:
            # 毫秒时间戳 → YYYY-MM-DD（北京时间）
            from datetime import datetime as _dt, timezone, timedelta as _td
            date_str = (_dt.fromtimestamp(int(ts) / 1000, tz=timezone.utc)
                          .astimezone(timezone(_td(hours=8)))
                          .strftime("%Y-%m-%d"))
        else:
            date_str = ""
        results.append({
            "date":         date_str,
            "close":        _f(it.get("close")),
            "pct_chg":      _pct(it.get("percent")),
            "main_net":     _yi(it.get("main_net_inflows")),
            "super_net":    _yi(it.get("super_net_inflows")),
            "big_net":      _yi(it.get("large_net_inflows")),
            "mid_net":      _yi(it.get("medium_net_inflows")),
            "small_net":    _yi(it.get("small_net_inflows")),
            "main_net_pct": _pct(it.get("main_inflows_ratio")),
        })
    if not results:
        raise ValueError(f"雪球返回 {code} 数据但解析为空")
    return results


def get_moneyflow(code: str, days: int = 10) -> list:
    """
    获取个股主力资金流向（大单净流入/净卖出）。

    数据源: 雪球优先 → 东财回退 (东财在 GitHub Actions 出口 IP 上偶有封锁)
            雪球字段对齐东财, 完整提供 main/super/big/mid/small 分级

    Parameters
    ----------
    code : str   股票代码，如 "600519"
    days : int   返回最近 N 个交易日，默认 10

    Returns
    -------
    list of dict，每条包含:
        date, main_net, main_net_pct, super_net, big_net, mid_net, small_net, close, pct_chg
    """
    last_err = None
    try:
        return _get_moneyflow_xueqiu(code, days)
    except Exception as e:
        last_err = e
        _log.debug("Xueqiu moneyflow failed for %s: %s, falling back to Eastmoney", code, e)

    try:
        return _get_moneyflow_eastmoney(code, days)
    except Exception as e:
        raise ConnectionError(
            f"主力资金流向请求失败 (雪球: {last_err}; 东财: {e})"
        )


def format_moneyflow_report(flow_data: list) -> str:
    """将主力资金流向数据格式化为中文报告。"""
    if not flow_data:
        return "ERROR: 无主力资金数据"

    lines = [
        "主力资金流向报告",
        f"  数据区间: {flow_data[0]['date']} ~ {flow_data[-1]['date']}",
        "=" * 60,
        f"  {'日期':<12} {'主力净流入':>10} {'占比':>6} {'超大单':>8} {'涨跌幅':>7}",
        "-" * 60,
    ]

    for d in flow_data:
        mn   = d.get("main_net")
        pct  = d.get("main_net_pct")
        sup  = d.get("super_net")
        chg  = d.get("pct_chg")
        flag = "↑" if (mn or 0) > 0 else "↓"
        lines.append(
            f"  {d['date']:<12} {(mn or 0):>+9.2f}亿 {(pct or 0):>+5.1f}% "
            f"{(sup or 0):>+7.2f}亿 {(chg or 0):>+6.2f}%  {flag}"
        )

    lines.append("-" * 60)

    nets = [d.get("main_net") or 0 for d in flow_data]
    avg5 = sum(nets[-5:]) / min(5, len(nets)) if nets else 0
    total = sum(nets)
    inflow_days = sum(1 for n in nets if n > 0)
    outflow_days = len(nets) - inflow_days

    if avg5 > 1.0:
        signal = "主力持续净买入（看多信号）"
    elif avg5 > 0.2:
        signal = "主力小幅流入（温和看多）"
    elif avg5 > -0.2:
        signal = "主力进出平衡（观望）"
    elif avg5 > -1.0:
        signal = "主力小幅流出（谨慎）"
    else:
        signal = "主力持续净卖出（看空信号）"

    lines.extend([
        f"  近5日日均主力净流入: {avg5:+.2f} 亿元",
        f"  统计周期合计:       {total:+.2f} 亿元",
        f"  净流入天数/净流出天数: {inflow_days}天 / {outflow_days}天",
        f"  大单净流入比: {(nets[-1] if nets else 0):+.2f}%",
        f"  主力资金信号: {signal}",
        "",
        "注：主力大单（≥50万/笔）是判断机构建仓/减仓的重要参考。",
    ])
    return "\n".join(lines)


# ──────────────────────────────────────────
# 3. 融资融券
# ──────────────────────────────────────────

def get_margin_trading(code: str, days: int = 20) -> list:
    """
    获取个股融资融券余额历史数据。

    Parameters
    ----------
    code : str   股票代码
    days : int   返回最近 N 条记录，默认 20

    Returns
    -------
    list of dict，每条包含:
        date, fin_bal, fin_buy, fin_repay, sec_bal, sec_sell, net_bal
    """
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    params = {
        "reportName":  "RPT_MUTUAL_RZRQ_LSSJ",
        "columns":     "TRADE_DATE,RZYE,RZMRE,RZCHE,RQYE,RQMCL,RQYL",
        "filter":      f'(SCODE="{code}")',
        "pageNumber":  1,
        "pageSize":    days,
        "sortTypes":   -1,
        "sortColumns": "TRADE_DATE",
        "source":      "WEB",
        "client":      "WEB",
    }
    try:
        eastmoney_throttle.acquire()
        resp = _get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise ConnectionError(f"两融数据请求失败: {e}")

    rows = (data.get("result") or {}).get("data") or []
    if not rows:
        raise ValueError(f"未获取到 {code} 的两融数据（该股可能不在两融标的范围内）")

    def _yi(v):
        try: return round(float(v) / 1e8, 4) if v else None
        except Exception: return None
    def _wan(v):
        try: return round(float(v) / 1e4, 2) if v else None
        except Exception: return None

    results = []
    for r in rows:
        rzye  = _yi(r.get("RZYE"))
        rqye  = _yi(r.get("RQYE"))
        results.append({
            "date":      (r.get("TRADE_DATE") or "")[:10],
            "fin_bal":   rzye,
            "fin_buy":   _yi(r.get("RZMRE")),
            "fin_repay": _yi(r.get("RZCHE")),
            "sec_bal":   rqye,
            "sec_sell":  _wan(r.get("RQMCL")),
            "net_bal":   round((rzye or 0) - (rqye or 0), 4),
        })
    return results


def format_margin_report(margin_data: list) -> str:
    """将两融数据格式化为中文报告。"""
    if not margin_data:
        return "ERROR: 无两融数据（该股可能未纳入两融标的）"

    lines = [
        "融资融券余额报告",
        f"  数据区间: {margin_data[-1]['date']} ~ {margin_data[0]['date']}",
        "=" * 60,
        f"  {'日期':<12} {'融资余额':>9} {'融资买入':>9} {'融券余额':>9} {'净多头':>9}",
        "-" * 60,
    ]

    for d in margin_data:
        fb   = d.get("fin_bal") or 0
        fbuy = d.get("fin_buy") or 0
        sb   = d.get("sec_bal") or 0
        net  = d.get("net_bal") or 0
        lines.append(
            f"  {d['date']:<12} {fb:>8.2f}亿 {fbuy:>8.2f}亿 {sb:>8.2f}亿 {net:>+8.2f}亿"
        )

    lines.append("-" * 60)

    fin_bals = [d.get("fin_bal") or 0 for d in margin_data]
    recent5  = fin_bals[:5]
    early5   = fin_bals[-5:] if len(fin_bals) >= 10 else fin_bals[:5]
    avg_recent = sum(recent5) / len(recent5) if recent5 else 0
    avg_early  = sum(early5) / len(early5) if early5 else 0
    chg_pct = (avg_recent / avg_early - 1) * 100 if avg_early != 0 else 0

    if chg_pct > 5:
        signal = "融资余额上升（场内资金加杠杆做多，看多信号）"
    elif chg_pct > 1:
        signal = "融资余额温和上升（小幅增多头）"
    elif chg_pct > -1:
        signal = "融资余额基本稳定（持仓不变）"
    elif chg_pct > -5:
        signal = "融资余额下降（多头减杠杆）"
    else:
        signal = "融资余额明显下降（去杠杆/平仓，看空信号）"

    lines.extend([
        f"  近5日平均融资余额: {avg_recent:.2f} 亿元",
        f"  与早期5日对比变化: {chg_pct:+.1f}%",
        f"  两融信号: {signal}",
        "",
        "注：融资余额持续上升表明场内资金看多情绪增加；融券余额大幅上升需警惕做空压力。",
    ])
    return "\n".join(lines)
