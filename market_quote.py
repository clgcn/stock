"""
外围市场行情模块 (market_quote) — 环境层依赖
=============================================
职责: 获取全球主要指数/大宗商品实时行情，计算市场情绪评分。
      这是**宏观环境数据**，不是新闻。

提供:
  INDICES                       → 监控指数列表
  get_foreign_markets()         → 拉取全部外围行情
  compute_market_score()        → 行情 → 情绪评分 [-1, +1]
  build_market_interpretation() → 评分 → 中文研判段落
  format_market_block()         → 行情 → 中文表格段落
  score_to_label()              → 评分 → 标签

底层数据源: 东方财富统一行情接口 / Google Finance 备用
"""

import time

try:
    from curl_cffi import requests
    _USE_CURL = True
except ImportError:
    import requests
    _USE_CURL = False

_TIMEOUT = 10
_NO_PROXY = {"http": "", "https": ""}


def _get(url, params=None, timeout=_TIMEOUT, extra_headers=None):
    """统一 GET 请求，支持 curl_cffi TLS 指纹伪装"""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json,text/html,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    if extra_headers:
        headers.update(extra_headers)
    try:
        if _USE_CURL:
            return requests.get(
                url, params=params, headers=headers,
                timeout=timeout, impersonate="chrome"
            )
        else:
            return requests.get(
                url, params=params, headers=headers,
                timeout=timeout, proxies=_NO_PROXY
            )
    except Exception as e:
        raise RuntimeError(f"请求失败 {url}: {e}") from e


# ══════════════════════════════════════════════════════
# 监控指数列表
# ══════════════════════════════════════════════════════

# (主源secid列表, 中文名称, 分类)
# 多个 secid 用 | 分隔，按优先级依次尝试
INDICES = [
    # 美股 (Yahoo Finance → Google Finance)
    ("yf:^GSPC|gf:.INX:INDEXSP",            "标普500",      "美股"),
    ("yf:^IXIC|gf:NDX:INDEXNASDAQ",         "纳斯达克",     "美股"),
    ("yf:^DJI|gf:.DJI:INDEXDJX",            "道琼斯",       "美股"),
    ("yf:^VIX|gf:VIX:INDEXCBOE",            "恐慌指数VIX",  "美股"),
    # 欧股 (Yahoo Finance → Google Finance)
    ("yf:^FTSE|gf:UKX:INDEXFTSE",    "富时100(英)",  "欧股"),
    ("yf:^GDAXI|gf:DAX:INDEXDB",     "DAX(德)",      "欧股"),
    ("yf:^FCHI|gf:PX1:INDEXEURO",    "CAC40(法)",    "欧股"),
    # 亚太 (Yahoo Finance → Google Finance)
    ("yf:^HSI|gf:HSI:INDEXHANGSENG",    "恒生指数",     "亚太"),
    ("yf:^N225|gf:NI225:INDEXNIKKEI",   "日经225",      "亚太"),
    # 大宗商品 (Yahoo Finance → Google Finance)
    ("yf:GC=F|gf:GC%3DF:COMMODITIESEXCHANGE",  "黄金",         "大宗"),
    ("yf:CL=F|gf:CL%3DF:COMMODITIESEXCHANGE",  "原油WTI",      "大宗"),
]

# 分类权重
_CATEGORY_WEIGHT = {
    "美股": 0.40,
    "亚太": 0.30,
    "欧股": 0.15,
    "大宗": 0.15,
}


# ══════════════════════════════════════════════════════
# 数据获取
# ══════════════════════════════════════════════════════

def _fetch_eastmoney_quote(secid: str) -> dict:
    """从东方财富统一行情接口获取全球指数/期货的最新价和涨跌幅。"""
    url = "https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "secid": secid,
        "fields": "f43,f57,f58,f59,f60,f169,f170",
        "ut": "f057cbcbce2a86e2866ab8877db1d059",
        "invt": 2,
        "fltt": 2,
    }
    try:
        r = _get(url, params=params, timeout=10,
                 extra_headers={"Referer": "https://quote.eastmoney.com/"})
        data = r.json().get("data") or {}
        if not data:
            return {"error": "empty response"}

        price_raw = data.get("f43")
        prev_raw = data.get("f60")
        chg_raw = data.get("f169")
        pct_raw = data.get("f170")
        decimal_places = data.get("f59")

        def _norm(v, use_decimal=True):
            if v in (None, "", "-"):
                return None
            try:
                fv = float(v)
            except Exception:
                return None
            if use_decimal and decimal_places is not None:
                try:
                    dp = int(decimal_places)
                    if dp > 0 and fv == int(fv) and abs(fv) > 10 ** dp:
                        fv = fv / (10 ** dp)
                except Exception:
                    pass
            return fv

        price = _norm(price_raw)
        prev = _norm(prev_raw)
        chg = _norm(chg_raw)
        pct = _norm(pct_raw, use_decimal=False)

        if price is None:
            return {"error": "missing price"}
        if pct is None and price is not None and prev not in (None, 0):
            pct = (price - prev) / prev * 100
        if prev is None and price is not None and chg is not None:
            prev = price - chg

        return {
            "price": price,
            "prev_close": prev if prev is not None else price,
            "pct_chg": round(pct if pct is not None else 0.0, 2),
            "source": "eastmoney",
        }
    except Exception as e:
        return {"error": str(e)}


def _fetch_google_finance(symbol: str) -> dict:
    """Google Finance 备用数据源（用于VIX等东方财富覆盖不到的指数）。"""
    try:
        url = f"https://www.google.com/finance/quote/{symbol}"
        r = _get(url, timeout=8, extra_headers={
            "Referer": "https://www.google.com/",
            "Accept": "text/html,*/*",
        })
        import re
        text = r.text
        # data-last-price
        m = re.search(r'data-last-price="([^"]+)"', text)
        if not m:
            return {"error": "no price found"}
        price = float(m.group(1))
        m3 = re.search(r'data-previous-close="([^"]+)"', text)
        prev = float(m3.group(1)) if m3 else None
        pct = ((price - prev) / prev * 100) if prev and prev != 0 else None
        # 尝试从 data-change-percent 获取涨跌幅（备用）
        if pct is None:
            m4 = re.search(r'data-change-percent="([^"]+)"', text)
            if m4:
                try:
                    pct = float(m4.group(1))
                except Exception:
                    pass
        return {
            "price": price,
            "prev_close": prev if prev is not None else price,
            "pct_chg": round(pct, 2) if pct is not None else 0.0,
            "source": "google_finance",
        }
    except Exception as e:
        return {"error": str(e)}


def _fetch_yahoo_finance(symbol: str) -> dict:
    """Yahoo Finance 备用数据源 — 全球覆盖最广。"""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        r = _get(url, params={"range": "1d", "interval": "1d"}, timeout=8,
                 extra_headers={"Referer": "https://finance.yahoo.com/"})
        data = r.json()
        meta = data.get("chart", {}).get("result", [{}])[0].get("meta", {})
        price = meta.get("regularMarketPrice")
        prev = meta.get("chartPreviousClose") or meta.get("previousClose")
        if price is None:
            return {"error": "no price from yahoo"}
        pct = ((price - prev) / prev * 100) if prev and prev != 0 else 0.0
        return {
            "price": round(price, 2),
            "prev_close": round(prev, 2) if prev else price,
            "pct_chg": round(pct, 2),
            "source": "yahoo_finance",
        }
    except Exception as e:
        return {"error": str(e)}


def get_foreign_markets() -> dict:
    """
    串行获取所有监控指数行情，支持备选 secid（用 | 分隔）。
    返回 {category: [{"name": str, "price": float, "pct_chg": float, ...}]}
    """
    results = {}
    for secid_str, name, category in INDICES:
        secid_candidates = [s.strip() for s in secid_str.split("|")]
        q = {"error": "all candidates failed"}
        for sid in secid_candidates:
            if sid.startswith("gf:"):
                q = _fetch_google_finance(sid[3:])
            elif sid.startswith("yf:"):
                q = _fetch_yahoo_finance(sid[3:])
            else:
                q = _fetch_eastmoney_quote(sid)
            if "error" not in q:
                q["secid"] = sid
                break
            time.sleep(0.3)
        else:
            q["secid"] = secid_candidates[0]
        q["name"] = name
        results.setdefault(category, []).append(q)
        time.sleep(0.3)
    return results


# ══════════════════════════════════════════════════════
# 评分与格式化
# ══════════════════════════════════════════════════════

def compute_market_score(market_data: dict) -> float:
    """
    基于外围行情计算市场情绪得分 [-1.0, +1.0]。
    权重: 美股40%, 亚太30%, 欧股15%, 大宗15%
    """
    scores = {}
    for cat, items in market_data.items():
        pcts = [i["pct_chg"] for i in items
                if "pct_chg" in i and i["pct_chg"] is not None and "error" not in i]
        if not pcts:
            scores[cat] = 0.0
            continue

        if cat == "美股":
            vix = next((i for i in items if "VIX" in i.get("name", "")), None)
            index_pcts = [p for p in pcts if not (vix and p == vix.get("pct_chg"))]
            avg_pct = sum(index_pcts) / len(index_pcts) if index_pcts else 0
            raw = avg_pct / 3.0
            if vix and "pct_chg" in vix and vix["pct_chg"] is not None:
                vix_chg = vix["pct_chg"]
                if vix.get("price") and vix["price"] > 30:
                    raw -= 0.2
                elif vix.get("price") and vix["price"] > 25:
                    raw -= 0.1
                if vix_chg > 20:
                    raw -= 0.15
                elif vix_chg > 10:
                    raw -= 0.08
        else:
            avg_pct = sum(pcts) / len(pcts)
            raw = avg_pct / 3.0

        scores[cat] = max(-1.0, min(1.0, raw))

    total = sum(scores.get(c, 0) * _CATEGORY_WEIGHT.get(c, 0)
                for c in _CATEGORY_WEIGHT)
    norm = sum(_CATEGORY_WEIGHT.get(c, 0) for c in scores if c in _CATEGORY_WEIGHT) or 1
    final = total / norm
    return round(max(-1.0, min(1.0, final)), 2)


def score_to_label(score: float) -> str:
    """将情绪得分转换为中文标签"""
    if score >= 0.3:  return "偏多（外围积极）"
    if score >= 0.1:  return "中性偏多"
    if score >= -0.1: return "中性"
    if score >= -0.3: return "中性偏空"
    return "偏空（外围低迷）"


def build_market_interpretation(market_data: dict, score: float) -> str:
    """将外围行情数据 + 评分格式化为中文研判段落。"""
    lines = ["【外围市场综合研判】"]

    us   = market_data.get("美股", [])
    ap   = market_data.get("亚太", [])
    bulk = market_data.get("大宗", [])

    sp = next((i for i in us if "标普" in i.get("name", "")), None)
    nd = next((i for i in us if "纳斯达克" in i.get("name", "")), None)
    hsi = next((i for i in ap if "恒生" in i.get("name", "")), None)
    gold = next((i for i in bulk if "黄金" in i.get("name", "")), None)
    oil  = next((i for i in bulk if "原油" in i.get("name", "")), None)

    sp_pct = sp.get("pct_chg") if sp else None
    nd_pct = nd.get("pct_chg") if nd else None
    hsi_pct = hsi.get("pct_chg") if hsi else None
    gold_pct = gold.get("pct_chg") if gold else None
    oil_pct = oil.get("pct_chg") if oil else None

    if sp_pct is not None:
        d = "上涨" if sp_pct >= 0 else "下跌"
        lines.append(f"  · 标普500 {d} {abs(sp_pct):.2f}%，"
                     f"{'市场情绪稳健' if sp_pct >= 0 else '市场情绪偏谨慎'}")
    if nd_pct is not None:
        d = "上涨" if nd_pct >= 0 else "下跌"
        lines.append(f"  · 纳斯达克 {d} {abs(nd_pct):.2f}%，"
                     f"科技板块{'偏强' if nd_pct >= 0 else '偏弱'}（A股科技联动参考）")
    if hsi_pct is not None:
        d = "上涨" if hsi_pct >= 0 else "下跌"
        lines.append(f"  · 恒生指数 {d} {abs(hsi_pct):.2f}%，"
                     f"对 A 股港股通、南向资金板块参考意义较强")
    if gold_pct is not None:
        hint = "避险情绪上升" if gold_pct > 1 else (
            "避险情绪平稳" if gold_pct >= -1 else "避险情绪下降")
        lines.append(f"  · 黄金 {'上涨' if gold_pct >= 0 else '下跌'} "
                     f"{abs(gold_pct):.2f}%，{hint}")
    if oil_pct is not None:
        lines.append(f"  · 原油 {'上涨' if oil_pct >= 0 else '下跌'} "
                     f"{abs(oil_pct):.2f}%，影响能源/化工/航运板块")

    lines.append(f"\n  综合情绪得分: {score:+.2f}  →  {score_to_label(score)}")
    return "\n".join(lines)


def format_market_block(market_data: dict) -> str:
    """将外围市场数据格式化为中文表格块。"""
    category_order = ["美股", "欧股", "亚太", "大宗"]
    lines = ["【外围市场行情】"]

    for cat in category_order:
        items = market_data.get(cat, [])
        if not items:
            continue
        lines.append(f"\n  ── {cat} ──")
        for item in items:
            name = item.get("name", "?")
            if "error" in item:
                lines.append(f"  {name:<14} 获取失败 ({item['error'][:30]})")
                continue
            price = item.get("price")
            pct   = item.get("pct_chg")
            if pct is None:
                pct = 0.0  # 兜底：有价格时涨跌幅不应为 None
            if price is None:
                lines.append(f"  {name:<14} 数据缺失")
                continue

            arrow = "▲" if pct >= 0 else "▼"
            color_hint = ""
            if pct >= 1.5:
                color_hint = "  ↑↑"
            elif pct <= -1.5:
                color_hint = "  ↓↓"

            pct_str = f"{pct:+.2f}%"
            lines.append(f"  {name:<14} {price:>10,.2f}  {arrow} {pct_str}{color_hint}")

    return "\n".join(lines)
