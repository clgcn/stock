"""
市场消息面分析模块 (news_analyzer.py)
====================================================
功能:
  - 抓取外围市场行情 (美股三大指数+VIX、欧股三大、恒生、日经、黄金、原油)
    via Yahoo Finance 免费 API（无需注册）
  - 抓取国际重要新闻 via Reuters RSS 公开订阅源
  - 抓取国内官方媒体新闻 via 新华社/人民日报/央视财经 RSS
  - 基于外围行情自动计算初步市场情绪得分
  - 综合生成中文市场环境日报，供 stock_diagnosis 的 news_sentiment 参数参考

设计原则:
  - 每个数据源独立容错，任一失败不影响整体输出
  - 请求超时均设 10 秒，防止阻塞 MCP 调用
  - 全部输出为中文，风格简洁专业
"""

import json
import re
import time
import html as _html
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime

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
# 一、外围市场行情（Yahoo Finance 免费 API）
# ══════════════════════════════════════════════════════

# 监控指数列表：(Yahoo代码, 中文名称, 分类)
INDICES = [
    # 美股
    ("^GSPC",  "标普500",      "美股"),
    ("^IXIC",  "纳斯达克",     "美股"),
    ("^DJI",   "道琼斯",       "美股"),
    ("^VIX",   "恐慌指数VIX",  "美股"),
    # 欧股
    ("^FTSE",  "富时100(英)",  "欧股"),
    ("^GDAXI", "DAX(德)",      "欧股"),
    ("^FCHI",  "CAC40(法)",    "欧股"),
    # 亚太（对A股联动最强）
    ("^HSI",   "恒生指数",     "亚太"),
    ("^N225",  "日经225",      "亚太"),
    # 大宗商品
    ("GC=F",   "黄金",         "大宗"),
    ("CL=F",   "原油WTI",      "大宗"),
]

# 分类权重（用于计算综合情绪分）
_CATEGORY_WEIGHT = {
    "美股": 0.40,
    "亚太": 0.30,
    "欧股": 0.15,
    "大宗": 0.15,
}


def _fetch_yahoo_quote(symbol: str) -> dict:
    """
    从 Yahoo Finance v8 API 获取单个指数最新行情。
    返回 dict: {price, prev_close, pct_chg, currency} 或 {error}
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"interval": "1d", "range": "5d"}
    try:
        r = _get(url, params=params, timeout=10)
        data = r.json()
        result = data["chart"]["result"][0]
        meta = result["meta"]
        price = meta.get("regularMarketPrice") or meta.get("previousClose") or 0
        prev  = meta.get("chartPreviousClose") or meta.get("previousClose") or price
        pct   = (price - prev) / prev * 100 if prev and prev != 0 else 0.0
        return {
            "price": price,
            "prev_close": prev,
            "pct_chg": round(pct, 2),
            "currency": meta.get("currency", ""),
        }
    except Exception as e:
        return {"error": str(e)}


def get_foreign_markets() -> dict:
    """
    并发（串行）获取所有监控指数行情。
    返回 {category: [{"name": str, "symbol": str, "price": float, "pct_chg": float, ...}]}
    """
    results = {}
    for symbol, name, category in INDICES:
        q = _fetch_yahoo_quote(symbol)
        q["symbol"] = symbol
        q["name"]   = name
        results.setdefault(category, []).append(q)
        time.sleep(0.2)   # 避免触发限速
    return results


def _fmt_market_block(market_data: dict) -> str:
    """将外围市场数据格式化为中文文字块"""
    category_order = ["美股", "欧股", "亚太", "大宗"]
    lines = ["【外围市场行情】"]

    for cat in category_order:
        items = market_data.get(cat, [])
        if not items:
            continue
        lines.append(f"\n  ▌{cat}：")
        for item in items:
            name = item["name"]
            if "error" in item:
                lines.append(f"    {name:<14} 数据获取失败")
                continue
            price = item["price"]
            pct   = item["pct_chg"]
            arrow = "▲" if pct >= 0 else "▼"
            sign  = "+" if pct >= 0 else ""

            # VIX 特殊标注
            if "VIX" in name:
                hint = "  恐慌升温" if pct > 8 else ("  情绪略紧" if pct > 3 else "  市场平稳")
                lines.append(f"    {name:<14} {price:>8.2f}   {arrow} {sign}{pct:.2f}%{hint}")
            # 黄金 / 原油
            elif cat == "大宗":
                unit = "美元/盎司" if "黄金" in name else "美元/桶"
                lines.append(f"    {name:<14} {price:>8.2f} ({unit})   {arrow} {sign}{pct:.2f}%")
            else:
                lines.append(f"    {name:<14} {price:>10,.2f}   {arrow} {sign}{pct:.2f}%")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════
# 二、情绪得分计算
# ══════════════════════════════════════════════════════

def _compute_market_score(market_data: dict) -> float:
    """
    基于外围市场涨跌幅，计算初步情绪得分 [-1.0, +1.0]。
    逻辑：每个分类取非VIX指数均值，归一化后按权重加权；
          VIX 上涨 > 5% 对美股得分施加额外负向惩罚。
    """
    weighted_scores = []
    total_weight = 0.0

    for cat, items in market_data.items():
        weight = _CATEGORY_WEIGHT.get(cat, 0.1)
        valid = [i for i in items if "pct_chg" in i and "VIX" not in i.get("name", "")]
        if not valid:
            continue
        avg_pct = sum(i["pct_chg"] for i in valid) / len(valid)
        # 归一化：±3% 对应 ±1.0
        score = max(-1.0, min(1.0, avg_pct / 3.0))

        # VIX 惩罚（只影响美股这一栏）
        if cat == "美股":
            vix_items = [i for i in items if "VIX" in i.get("name", "") and "pct_chg" in i]
            if vix_items:
                vix_pct = vix_items[0]["pct_chg"]
                if vix_pct > 10:
                    score -= 0.35
                elif vix_pct > 5:
                    score -= 0.18

        weighted_scores.append(score * weight)
        total_weight += weight

    if not weighted_scores or total_weight == 0:
        return 0.0
    return round(sum(weighted_scores) / total_weight, 2)


def _score_to_label(score: float) -> str:
    if score >= 0.5:  return "显著偏多 📈"
    if score >= 0.2:  return "略偏多"
    if score >= -0.2: return "中性"
    if score >= -0.5: return "略偏空"
    return "显著偏空 📉"


def _build_market_interpretation(market_data: dict, score: float) -> str:
    """生成外围市场综合研判段落"""
    lines = ["【外围市场综合研判】"]

    us   = market_data.get("美股", [])
    ap   = market_data.get("亚太", [])
    bulk = market_data.get("大宗", [])

    # 标普500
    sp = next((i for i in us if "标普" in i.get("name", "")), None)
    nd = next((i for i in us if "纳斯达克" in i.get("name", "")), None)
    hsi = next((i for i in ap if "恒生" in i.get("name", "")), None)
    gold = next((i for i in bulk if "黄金" in i.get("name", "")), None)
    oil  = next((i for i in bulk if "原油" in i.get("name", "")), None)

    if sp and "pct_chg" in sp:
        d = "上涨" if sp["pct_chg"] >= 0 else "下跌"
        lines.append(f"  · 标普500 {d} {abs(sp['pct_chg']):.2f}%，"
                     f"{'市场情绪稳健' if sp['pct_chg'] >= 0 else '市场情绪偏谨慎'}")
    if nd and "pct_chg" in nd:
        d = "上涨" if nd["pct_chg"] >= 0 else "下跌"
        lines.append(f"  · 纳斯达克 {d} {abs(nd['pct_chg']):.2f}%，"
                     f"科技板块{'偏强' if nd['pct_chg'] >= 0 else '偏弱'}（A股科技联动参考）")
    if hsi and "pct_chg" in hsi:
        d = "上涨" if hsi["pct_chg"] >= 0 else "下跌"
        lines.append(f"  · 恒生指数 {d} {abs(hsi['pct_chg']):.2f}%，"
                     f"对 A 股港股通、南向资金板块参考意义较强")
    if gold and "pct_chg" in gold:
        hint = "避险情绪上升" if gold["pct_chg"] > 1 else ("避险情绪平稳" if gold["pct_chg"] >= -1 else "避险情绪下降")
        lines.append(f"  · 黄金 {'上涨' if gold['pct_chg'] >= 0 else '下跌'} {abs(gold['pct_chg']):.2f}%，{hint}")
    if oil and "pct_chg" in oil:
        lines.append(f"  · 原油 {'上涨' if oil['pct_chg'] >= 0 else '下跌'} {abs(oil['pct_chg']):.2f}%，"
                     f"影响能源/化工/航运板块")

    lines.append(f"\n  综合情绪得分: {score:+.2f}  →  {_score_to_label(score)}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════
# 三、新闻 RSS 抓取
# ══════════════════════════════════════════════════════

# 数据源：(显示名称, URL, 类别)
#   国际新闻：路透社（Reuters）公开 RSS
#   国内新闻：只看政府媒体（新华社、人民日报、央视财经）
NEWS_FEEDS = [
    # ── 国际 ──
    ("路透社·头条",    "https://feeds.reuters.com/reuters/topNews",        "国际"),
    ("路透社·财经",    "https://feeds.reuters.com/reuters/businessNews",    "国际"),
    ("路透社·科技",    "https://feeds.reuters.com/reuters/technologyNews",  "国际"),
    # ── 国内官方 ──
    ("新华社",         "http://www.xinhuanet.com/english/rss/chinalatestnews.xml", "国内官方"),
    ("人民日报",       "http://www.people.com.cn/rss/politics.xml",               "国内官方"),
    ("人民日报·财经",  "http://www.people.com.cn/rss/finance.xml",                "国内官方"),
    ("央视财经",       "http://rss.cctv.com/rss/financial.xml",                   "国内官方"),
]


def _parse_rss(xml_text: str, max_items: int = 6) -> list[dict]:
    """
    解析 RSS 2.0 / Atom XML，返回新闻条目列表。
    每条目: {title, pub_date_str}
    """
    items = []
    try:
        xml_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', xml_text)
        root = ET.fromstring(xml_text)

        # RSS 2.0
        channel = root.find("channel")
        if channel is not None:
            for item in channel.findall("item")[:max_items]:
                title = _html.unescape(item.findtext("title", "").strip())
                pub   = item.findtext("pubDate", "").strip()
                if title:
                    items.append({"title": title, "pub_date": pub})
        else:
            # Atom
            ns = "http://www.w3.org/2005/Atom"
            for entry in root.findall(f"{{{ns}}}entry")[:max_items]:
                title_el = entry.find(f"{{{ns}}}title")
                title = _html.unescape(title_el.text.strip() if title_el is not None and title_el.text else "")
                pub_el = entry.find(f"{{{ns}}}published") or entry.find(f"{{{ns}}}updated")
                pub    = pub_el.text.strip() if pub_el is not None and pub_el.text else ""
                if title:
                    items.append({"title": title, "pub_date": pub})
    except Exception:
        pass
    return items


def _friendly_date(pub_date_str: str) -> str:
    """将 RFC2822 / ISO8601 日期转为"N小时前"形式"""
    if not pub_date_str:
        return ""
    try:
        dt = parsedate_to_datetime(pub_date_str)
    except Exception:
        try:
            dt = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
        except Exception:
            return ""
    try:
        now = datetime.now(tz=dt.tzinfo)
        delta = now - dt
        hours = int(delta.total_seconds() / 3600)
        if hours < 1:
            return "刚刚"
        if hours < 24:
            return f"{hours}小时前"
        days = hours // 24
        return f"{days}天前"
    except Exception:
        return ""


def _fetch_rss_feed(name: str, url: str, max_items: int = 5) -> list[dict]:
    """抓取并解析单个 RSS 源，失败时返回空列表"""
    try:
        r = _get(url, timeout=10)
        if r.status_code != 200:
            return []
        raw = _parse_rss(r.text, max_items=max_items)
        for item in raw:
            item["source"] = name
            item["date_hint"] = _friendly_date(item.get("pub_date", ""))
        return raw
    except Exception:
        return []


def get_news_feeds() -> tuple[list[dict], list[dict]]:
    """
    获取全部新闻，返回 (international_list, domestic_list)
    每条: {title, source, date_hint}
    """
    international, domestic = [], []
    seen_titles = set()

    for name, url, category in NEWS_FEEDS:
        items = _fetch_rss_feed(name, url, max_items=5)
        for item in items:
            key = item["title"][:40]
            if not key or key in seen_titles:
                continue
            seen_titles.add(key)
            if category == "国际":
                international.append(item)
            else:
                domestic.append(item)

    return international[:12], domestic[:10]


def _fmt_news_block(items: list[dict], section_title: str, max_show: int = 8) -> str:
    """格式化新闻段落"""
    lines = [section_title]
    if not items:
        lines.append("  （暂无数据，可能受网络限制）")
        return "\n".join(lines)

    for i, item in enumerate(items[:max_show], 1):
        title = item["title"]
        # 截断过长标题
        if len(title) > 85:
            title = title[:82] + "..."
        date_hint = item.get("date_hint", "")
        source    = item.get("source", "")
        suffix = f"  {date_hint}" if date_hint else ""
        lines.append(f"  {i}. [{source}] {title}{suffix}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════
# 四、关键词情绪微调
# ══════════════════════════════════════════════════════

# 利空关键词（出现在标题中，降低情绪得分）
_BEARISH_KEYWORDS = [
    "war", "战争", "制裁", "sanction", "tariff", "关税", "recession", "衰退",
    "crash", "崩盘", "inflation", "通胀", "rate hike", "加息", "default", "违约",
    "layoff", "裁员", "bank run", "挤兑", "crisis", "危机", "downgrade", "降级",
    "geopolit", "地缘", "conflict", "冲突", "军事", "military",
]
# 利多关键词（出现在标题中，提升情绪得分）
_BULLISH_KEYWORDS = [
    "stimulus", "刺激", "rate cut", "降息", "cut rate", "bailout", "救市",
    "record high", "创新高", "breakthrough", "突破", "deal", "协议", "trade deal",
    "growth", "增长", "profit", "盈利", "upgrade", "上调", "利好", "positive",
    "ceasefire", "停火", "peace", "和平", "recovery", "复苏",
]


def _news_sentiment_delta(international: list[dict], domestic: list[dict]) -> float:
    """
    基于新闻标题关键词，给出情绪微调值 [-0.3, +0.3]。
    仅作为补充参考，不做主导判断。
    """
    all_titles = [i["title"].lower() for i in (international + domestic)]
    bear_count = sum(
        1 for t in all_titles for kw in _BEARISH_KEYWORDS if kw in t
    )
    bull_count = sum(
        1 for t in all_titles for kw in _BULLISH_KEYWORDS if kw in t
    )
    raw = (bull_count - bear_count) / max(len(all_titles), 1) * 2
    return round(max(-0.3, min(0.3, raw)), 2)


# ══════════════════════════════════════════════════════
# 五、主报告入口
# ══════════════════════════════════════════════════════

def get_market_news_report() -> str:
    """
    生成完整市场消息面环境日报（中文）。

    内容：
      1. 外围市场行情（美/欧/亚太/大宗商品）
      2. 外围市场综合研判
      3. 国际重要新闻（路透社 RSS）
      4. 国内官方媒体消息（新华社/人民日报/央视财经）
      5. 综合情绪得分 & 使用建议

    Returns
    -------
    str
        完整中文报告字符串
    """
    now = datetime.now()
    sep = "═" * 56

    lines = [
        sep,
        f"  市场消息面环境日报   {now.strftime('%Y-%m-%d  %H:%M')}",
        sep,
        "",
    ]

    # ── 1. 外围市场行情 ──
    lines.append("正在抓取外围市场数据（约需 10 秒）...")
    market_data = get_foreign_markets()
    lines[-1] = _fmt_market_block(market_data)

    # ── 2. 外围情绪得分 ──
    market_score = _compute_market_score(market_data)
    lines.append("")
    lines.append(_build_market_interpretation(market_data, market_score))

    # ── 3 & 4. 新闻 ──
    lines.append("")
    lines.append("正在抓取新闻 RSS（约需 15 秒）...")
    intl_news, domestic_news = get_news_feeds()
    lines[-1] = _fmt_news_block(intl_news, "【国际重要新闻】（路透社 Reuters RSS）")
    lines.append("")
    lines.append(_fmt_news_block(domestic_news, "【国内官方媒体】（新华社 / 人民日报 / 央视财经）"))

    # ── 5. 综合建议 ──
    news_delta  = _news_sentiment_delta(intl_news, domestic_news)
    final_score = round(max(-1.0, min(1.0, market_score + news_delta * 0.4)), 2)

    lines.extend([
        "",
        "【消息面综合结论】",
        f"  外围市场情绪得分：{market_score:+.2f}  ({_score_to_label(market_score)})",
        f"  新闻关键词微调：  {news_delta:+.2f}",
        f"  ─────────────────────────────",
        f"  综合建议 news_sentiment 参考值：{final_score:+.2f}",
        "",
        "  ➡  使用方法：",
        "     调用 stock_diagnosis 时，将此值传入 news_sentiment 参数。",
        "     Claude 可根据新闻实际内容在此基础上上下微调 ±0.2。",
        "     情绪分范围: -1.0（极度偏空） ~ +1.0（极度偏多），0 为中性。",
        "",
        "  ⚠️  本报告仅供参考，不构成投资建议。",
        sep,
    ])

    return "\n".join(lines)


# ══════════════════════════════════════════════════════
# 命令行直接运行（调试用）
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    print(get_market_news_report())
