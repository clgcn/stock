"""
新闻面分析模块 (news_analyzer)
==============================
职责: 纯新闻抓取 + 情绪评分。
      只关注**新鲜**的新闻 — 默认仅保留 2 天内（周末盘前放宽至 3 天）。

数据源 (2025-03 v3):
  市场级 — 新浪财经滚动新闻 API（主力, 已验证可达）
           + 东方财富新闻 API（备选, 部分环境被挡）
  个股级 — 新浪财经个股新闻 + 东方财富搜索（备选）

提供:
  ── 市场级 ──
  get_news_feeds()            → (国际新闻, 国内新闻)
  news_sentiment_delta()      → 市场新闻情绪微调 [-0.3, +0.3]
  get_market_news_report()    → 完整市场新闻报告 (不再含外围行情)
  ── 个股级 ──
  get_stock_news_summary()    → 个股新闻快照 {items, score, label}
  get_stock_news_report()     → 个股新闻中文报告

设计原则:
  - 时效性第一: 过时新闻 = 噪音，默认 2 天窗口
  - 每个数据源独立容错
  - 新浪财经为主力源（国内可达性高）
  - 请求超时 10 秒
"""

import json
import re
import time
from datetime import datetime, timedelta
from urllib.parse import quote_plus

from _http_utils import cn_now, cn_today, cn_str, CN_TZ

try:
    from curl_cffi import requests
    _USE_CURL = True
except ImportError:
    import requests
    _USE_CURL = False

_TIMEOUT = 10
_NO_PROXY = {"http": "", "https": ""}


def _get(url, params=None, timeout=_TIMEOUT, extra_headers=None):
    """统一 GET 请求"""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json,text/html,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://finance.sina.com.cn/",
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
# 时效性控制
# ══════════════════════════════════════════════════════

def _default_max_age_days() -> int:
    """
    工作日: 2 天窗口（昨天+今天）
    周末/周一盘前: 3 天（覆盖周五的消息）
    """
    wd = cn_now().weekday()  # 0=Mon ... 6=Sun
    return 3 if wd in (0, 5, 6) else 2


def _parse_datetime(dt_str: str):
    """解析各种日期格式"""
    if not dt_str:
        return None
    # Unix timestamp (新浪 ctime)
    if str(dt_str).isdigit() and len(str(dt_str)) >= 10:
        try:
            return datetime.fromtimestamp(int(dt_str), tz=CN_TZ)
        except Exception:
            pass
    # 常见字符串格式
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
    ):
        try:
            return datetime.strptime(str(dt_str).strip(), fmt)
        except ValueError:
            continue
    # ISO 8601
    try:
        return datetime.fromisoformat(str(dt_str).replace("Z", "+00:00"))
    except Exception:
        pass
    # RFC 2822
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(str(dt_str))
    except Exception:
        pass
    return None


def _friendly_time(dt_str) -> str:
    """将日期转为 "N小时前" / "N天前" 形式"""
    if not dt_str:
        return ""
    dt = _parse_datetime(str(dt_str))
    if dt is None:
        return ""
    try:
        now = cn_now()
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        now = now.replace(tzinfo=None)
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


def _is_within_window(dt_str, max_age_days: int = None) -> bool:
    """判断新闻是否在时效窗口内"""
    if max_age_days is None:
        max_age_days = _default_max_age_days()
    dt = _parse_datetime(str(dt_str) if dt_str else "")
    if dt is None:
        return True  # 无法判断日期时，保守保留
    now = cn_now()
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    now = now.replace(tzinfo=None)
    delta_days = (now - dt).total_seconds() / 86400
    return 0 <= delta_days <= max_age_days


# ══════════════════════════════════════════════════════
# 新浪财经 滚动新闻 API（主力源, 已验证可达）
# ══════════════════════════════════════════════════════
#
# 接口: https://feed.mix.sina.com.cn/api/roll/get
# 已知可用的 pageid + lid 组合:
#   pageid=153, lid=2509 → 财经联播 (全品类, 已验证)
#   pageid=153, lid=2510 → 国内经济
#   pageid=153, lid=2511 → 国际经济
#   pageid=153, lid=2516 → 产经新闻
#   pageid=153, lid=2968 → 宏观经济
# 注意: 返回的 ctime 是 Unix timestamp

def _fetch_sina_roll(pageid: str, lid: str, label: str,
                      max_items: int = 15) -> list[dict]:
    """
    新浪财经滚动新闻通用抓取函数。
    """
    url = "https://feed.mix.sina.com.cn/api/roll/get"
    params = {
        "pageid": pageid,
        "lid": lid,
        "k": "",
        "num": str(max_items),
        "page": "1",
        "r": str(time.time()),
    }
    try:
        r = _get(url, params=params, timeout=_TIMEOUT)
        if r.status_code != 200:
            return []
        data = r.json()
        raw_list = (data.get("result") or {}).get("data") or []
        items = []
        for item in raw_list:
            title = (item.get("title") or "").strip()
            if not title:
                continue
            ctime = item.get("ctime") or item.get("create_time") or ""
            if str(ctime).isdigit() and len(str(ctime)) >= 10:
                dt = datetime.fromtimestamp(int(ctime), tz=CN_TZ)
                pub_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                pub_time = str(ctime)
            items.append({
                "title": title,
                "pub_date": pub_time,
                "source": f"新浪财经·{label}",
                "date_hint": _friendly_time(pub_time),
            })
        return items
    except Exception:
        return []


def _fetch_sina_search_news(keyword: str, max_items: int = 10) -> list[dict]:
    """
    新浪财经关键词新闻搜索 — 用于个股新闻检索。
    使用已验证可达的 feed.mix.sina.com.cn 滚动 API + k 参数。
    """
    items = []

    # ── 方式1: 滚动 API + k 关键词参数 ──
    url = "https://feed.mix.sina.com.cn/api/roll/get"
    params = {
        "pageid": "153",
        "lid": "2509",
        "k": keyword,
        "num": str(max_items),
        "page": "1",
        "r": str(time.time()),
    }
    try:
        r = _get(url, params=params, timeout=_TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            raw_list = (data.get("result") or {}).get("data") or []
            for item in raw_list:
                title = (item.get("title") or "").strip()
                title = re.sub(r"<[^>]+>", "", title)
                if not title:
                    continue
                ctime = item.get("ctime") or item.get("create_time") or ""
                if str(ctime).isdigit() and len(str(ctime)) >= 10:
                    dt = datetime.fromtimestamp(int(ctime), tz=CN_TZ)
                    pub_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    pub_time = str(ctime)
                items.append({
                    "title": title,
                    "pub_date": pub_time,
                    "source": "新浪财经",
                    "date_hint": _friendly_time(pub_time),
                })
                if len(items) >= max_items:
                    break
    except Exception:
        pass

    # ── 方式2: 股票频道 (lid=2510) ──
    if len(items) < 3:
        params2 = {
            "pageid": "153",
            "lid": "2510",
            "k": keyword,
            "num": str(max_items),
            "page": "1",
            "r": str(time.time()),
        }
        try:
            r2 = _get(url, params=params2, timeout=_TIMEOUT)
            if r2.status_code == 200:
                data2 = r2.json()
                seen = {it["title"][:40] for it in items}
                for item in (data2.get("result") or {}).get("data") or []:
                    title = re.sub(r"<[^>]+>", "", (item.get("title") or "").strip())
                    if not title or title[:40] in seen:
                        continue
                    seen.add(title[:40])
                    ctime = item.get("ctime") or ""
                    if str(ctime).isdigit() and len(str(ctime)) >= 10:
                        dt = datetime.fromtimestamp(int(ctime))
                        pub_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        pub_time = str(ctime)
                    items.append({
                        "title": title,
                        "pub_date": pub_time,
                        "source": "新浪财经",
                        "date_hint": _friendly_time(pub_time),
                    })
                    if len(items) >= max_items:
                        break
        except Exception:
            pass

    # ── 方式3: interface.sina.cn 搜索 (部分环境可用) ──
    if len(items) < 3:
        try:
            url3 = "https://interface.sina.cn/news/wap/fymap2.d.json"
            params3 = {
                "word": keyword,
                "cat": "finance",
                "num": str(max_items),
                "page": "1",
            }
            r3 = _get(url3, params=params3, timeout=8)
            if r3.status_code == 200:
                data3 = r3.json()
                seen = {it["title"][:40] for it in items}
                for item in (data3.get("result") or {}).get("list") or []:
                    title = re.sub(r"<[^>]+>", "", (item.get("title") or "").strip())
                    if not title or title[:40] in seen:
                        continue
                    seen.add(title[:40])
                    pub_time = item.get("datetime") or item.get("ctime") or ""
                    items.append({
                        "title": title,
                        "pub_date": pub_time,
                        "source": "新浪财经",
                        "date_hint": _friendly_time(pub_time),
                    })
                    if len(items) >= max_items:
                        break
        except Exception:
            pass

    return items[:max_items]


# ══════════════════════════════════════════════════════
# 东方财富新闻 API（备选源）
# ══════════════════════════════════════════════════════

def _fetch_eastmoney_headlines(column: str = "350", label: str = "财经要闻",
                                max_items: int = 15) -> list[dict]:
    """
    东方财富新闻列表 — 部分环境下 np-listapi.eastmoney.com 被挡。
    """
    url = "https://np-listapi.eastmoney.com/comm/web/getNewsByColumns"
    params = {
        "column": column,
        "bession": "all",
        "client": "web",
        "pageSize": str(max_items),
        "page": "1",
        "_": str(int(time.time() * 1000)),
    }
    try:
        r = _get(url, params=params, timeout=8,
                 extra_headers={"Referer": "https://finance.eastmoney.com/"})
        if r.status_code != 200:
            return []
        data = r.json()
        raw_list = (data.get("data") or {}).get("list") or []
        items = []
        for item in raw_list:
            title = (item.get("title") or item.get("digest") or "").strip()
            if not title:
                continue
            pub_time = item.get("showtime") or item.get("display_time") or ""
            items.append({
                "title": title,
                "pub_date": pub_time,
                "source": f"东方财富·{label}",
                "date_hint": _friendly_time(pub_time),
            })
        return items
    except Exception:
        return []


def _fetch_eastmoney_stock_news(stock_code: str, max_items: int = 10) -> list[dict]:
    """东方财富个股新闻搜索 — 备选源。"""
    try:
        url = "https://search-api-web.eastmoney.com/search/jsonp"
        cb = f"jQuery_{int(time.time()*1000)}"
        search_param = {
            "uid": "",
            "keyword": stock_code,
            "type": ["cmsArticleWebOld"],
            "client": "web",
            "clientType": "web",
            "clientVersion": "curr",
            "param": {
                "cmsArticleWebOld": {
                    "searchScope": "default",
                    "sort": "default",
                    "pageIndex": 1,
                    "pageSize": max_items,
                    "preTag": "",
                    "postTag": "",
                }
            },
        }
        params = {
            "cb": cb,
            "param": json.dumps(search_param, ensure_ascii=False),
        }
        r = _get(url, params=params, timeout=8,
                 extra_headers={"Referer": "https://so.eastmoney.com/"})
        if r.status_code != 200:
            return []
        text = r.text
        if text.startswith(cb):
            text = text[len(cb) + 1:-2]
        data = json.loads(text)
        article_list = (data.get("result") or {}).get("cmsArticleWebOld") or {}
        items = []
        for art in (article_list.get("list") or []):
            title = re.sub(r"<[^>]+>", "", (art.get("title") or "").strip())
            if not title:
                continue
            pub_time = art.get("date") or art.get("showtime") or ""
            items.append({
                "title": title,
                "pub_date": pub_time,
                "source": "东方财富",
                "date_hint": _friendly_time(pub_time),
            })
            if len(items) >= max_items:
                break
        return items
    except Exception:
        return []


# ══════════════════════════════════════════════════════
# 国际新闻关键词分类
# ══════════════════════════════════════════════════════

_INTL_KEYWORDS = {
    "美国", "美股", "美联储", "华尔街", "纳斯达克", "标普", "道琼斯",
    "欧洲", "欧央行", "欧盟", "英国", "德国", "法国",
    "日本", "日经", "日元", "日央行", "韩国",
    "全球", "海外", "国际", "外资", "外汇",
    "特朗普", "拜登", "白宫", "五角大楼",
    "关税", "制裁", "地缘", "俄罗斯", "乌克兰", "中东",
    "原油", "黄金", "gold", "oil", "fed", "trump",
    "imf", "世界银行", "wto", "opec",
}


def _is_international(title: str) -> bool:
    """判断新闻标题是否属于国际/全球类"""
    t = title.lower()
    return any(kw in t for kw in _INTL_KEYWORDS)


# ══════════════════════════════════════════════════════
# 市场级新闻（对外接口）
# ══════════════════════════════════════════════════════

def get_news_feeds() -> tuple[list[dict], list[dict]]:
    """
    获取市场级新闻，返回 (international, domestic)。
    主力源: 新浪财经（已验证可达）。备选: 东方财富。
    时效性: 仅保留 2 天内 (周末 3 天)。
    """
    max_age = _default_max_age_days()
    all_items = []
    seen_titles = set()

    # ── 第 1 层: 新浪财经（主力源, 多频道） ──
    # 财经联播（综合, 已验证可达, 包含国内外新闻）— 加大量
    domestic_raw = _fetch_sina_roll("153", "2509", "财经", max_items=30)
    all_items.extend(domestic_raw)

    # 国际经济（可能返回空, 不影响）
    intl_raw = _fetch_sina_roll("153", "2511", "国际", max_items=15)
    all_items.extend(intl_raw)

    # 宏观经济（补充）
    macro_raw = _fetch_sina_roll("153", "2968", "宏观", max_items=10)
    all_items.extend(macro_raw)

    # ── 第 2 层: 东方财富（备选, 可能被挡） ──
    if len(all_items) < 5:
        em_items = _fetch_eastmoney_headlines("350", "财经要闻", max_items=15)
        all_items.extend(em_items)
        em_items2 = _fetch_eastmoney_headlines("353", "全球市场", max_items=10)
        all_items.extend(em_items2)

    # ── 分类: 按关键词分为 国际 / 国内 ──
    international, domestic = [], []

    for item in all_items:
        if not _is_within_window(item.get("pub_date", ""), max_age):
            continue
        key = item["title"][:40]
        if not key or key in seen_titles:
            continue
        seen_titles.add(key)

        if _is_international(item["title"]):
            international.append(item)
        else:
            domestic.append(item)

    return international[:12], domestic[:12]


def _fmt_news_block(items: list[dict], section_title: str, max_show: int = 8) -> str:
    """格式化新闻段落"""
    lines = [section_title]
    if not items:
        lines.append("  （暂无最新新闻，可能受网络限制或近期无重大消息）")
        return "\n".join(lines)

    for i, item in enumerate(items[:max_show], 1):
        title = item["title"]
        if len(title) > 85:
            title = title[:82] + "..."
        date_hint = item.get("date_hint", "")
        src = item.get("source", "")
        suffix = f"  ({date_hint})" if date_hint else ""
        prefix = f"[{src}]" if src else ""
        lines.append(f"  {i}. {prefix} {title}{suffix}")
    return "\n".join(lines)


# ── 情绪关键词 ──

_BEARISH_KEYWORDS = [
    "crash", "崩", "暴跌", "recession", "衰退", "default", "违约",
    "crisis", "危机", "sell-off", "抛售", "war", "战争", "sanctions",
    "制裁", "inflation", "通胀", "rate hike", "加息", "shutdown",
    "关停", "plunge", "大跌", "bankruptcy", "破产", "tariff", "关税",
    "layoff", "裁员", "slump", "萧条", "downgrade", "下调",
    "hawk", "鹰派", "bear market", "熊市", "contagion", "蔓延",
    "下行", "回落", "走低", "承压", "利空", "缩量", "恐慌",
]

_BULLISH_KEYWORDS = [
    "rally", "反弹", "surge", "大涨", "record", "创纪录", "boom", "繁荣",
    "stimulus", "刺激", "rate cut", "降息", "cut rate", "bailout", "救市",
    "record high", "创新高", "breakthrough", "突破", "deal", "协议",
    "growth", "增长", "profit", "盈利", "upgrade", "上调", "利好",
    "ceasefire", "停火", "peace", "和平", "recovery", "复苏",
    "上涨", "走高", "放量", "活跃", "回暖", "景气", "提振",
]


def news_sentiment_delta(international: list[dict], domestic: list[dict]) -> float:
    """基于新闻标题关键词，给出情绪微调值 [-0.3, +0.3]。"""
    all_titles = [i["title"].lower() for i in (international + domestic)]
    if not all_titles:
        return 0.0
    bear_count = sum(1 for t in all_titles for kw in _BEARISH_KEYWORDS if kw in t)
    bull_count = sum(1 for t in all_titles for kw in _BULLISH_KEYWORDS if kw in t)
    raw = (bull_count - bear_count) / max(len(all_titles), 1) * 2
    return round(max(-0.3, min(0.3, raw)), 2)


def _score_to_label(score: float) -> str:
    if score >= 0.3:  return "偏多（消息面利好）"
    if score >= 0.1:  return "中性偏多"
    if score >= -0.1: return "中性"
    if score >= -0.3: return "中性偏空"
    return "偏空（消息面利空）"


# ══════════════════════════════════════════════════════
# 市场新闻报告（不再含外围行情）
# ══════════════════════════════════════════════════════

def get_market_news_report() -> str:
    """
    生成市场新闻面报告（纯新闻，不含外围行情）。

    内容:
      1. 国际/全球财经新闻（新浪财经 + 东方财富备选）
      2. 国内A股/财经新闻
      3. 新闻情绪微调值
    """
    now = cn_now()
    sep = "═" * 56
    max_age = _default_max_age_days()

    lines = [
        sep,
        f"  市场新闻面报告   {now.strftime('%Y-%m-%d  %H:%M')}",
        f"  (时效窗口: {max_age}天内)",
        sep,
        "",
    ]

    # ── 新闻 ──
    intl_news, domestic_news = get_news_feeds()
    lines.append(_fmt_news_block(intl_news, "【国际/全球财经新闻】"))
    lines.append("")
    lines.append(_fmt_news_block(domestic_news, "【国内A股/财经新闻】"))

    # ── 情绪 ──
    delta = news_sentiment_delta(intl_news, domestic_news)
    total_count = len(intl_news) + len(domestic_news)

    lines.extend([
        "",
        "【新闻面情绪结论】",
        f"  有效新闻数量: {total_count} 条（{max_age}天内）",
        f"  新闻情绪微调值: {delta:+.2f}  ({_score_to_label(delta)})",
        "",
        "  ➡  使用方法:",
        "     此值仅作为消息面补充参考，外围行情评分见 market_quote 模块。",
        "     Claude 可根据新闻实际内容在此基础上微调 ±0.1。",
        "",
        "  注: 仅保留近期新闻，过时消息已被市场消化，不纳入评估。",
        sep,
    ])

    return "\n".join(lines)


# ══════════════════════════════════════════════════════
# 个股级新闻
# ══════════════════════════════════════════════════════

_STOCK_BEARISH_KEYWORDS = [
    "减持", "处罚", "问询", "诉讼", "亏损", "下滑", "暴跌", "违约", "风险", "终止",
    "裁员", "冻结", "停产", "退市", "调查", "失败", "质押", "解禁", "商誉减值",
    "warning", "lawsuit", "probe", "investigation", "loss", "default", "risk",
    "下行", "承压", "利空", "被罚", "警示",
]

_STOCK_BULLISH_KEYWORDS = [
    "回购", "增持", "中标", "签约", "订单", "突破", "增长", "扭亏", "预增", "涨停",
    "并购", "收购", "落地", "投产", "扩产", "合作", "分红", "重组", "利好", "创新高",
    "buyback", "profit", "growth", "contract", "order", "breakthrough", "deal",
    "景气", "回暖", "提振", "获批", "中选",
]


def _stock_news_delta(items: list[dict]) -> float:
    """个股新闻情绪微调 [-0.4, +0.4]"""
    titles = [i.get("title", "").lower() for i in items]
    if not titles:
        return 0.0
    bear_count = sum(1 for t in titles for kw in _STOCK_BEARISH_KEYWORDS if kw in t)
    bull_count = sum(1 for t in titles for kw in _STOCK_BULLISH_KEYWORDS if kw in t)
    raw = (bull_count - bear_count) / max(len(titles), 1) * 2
    return round(max(-0.4, min(0.4, raw)), 2)


def _fetch_stock_news_all(stock_name: str, stock_code: str,
                           max_items: int = 10) -> list[dict]:
    """
    个股新闻聚合抓取：新浪搜索(主) + 东方财富搜索(备)。
    """
    items = []
    seen = set()

    # ── 主力: 新浪财经搜索 ──
    sina_items = _fetch_sina_search_news(
        f"{stock_name} {stock_code}", max_items=max_items
    )
    for it in sina_items:
        key = it["title"][:40]
        if key not in seen:
            seen.add(key)
            items.append(it)

    # ── 备选: 东方财富搜索 ──
    if len(items) < 3:
        em_items = _fetch_eastmoney_stock_news(stock_code, max_items=max_items)
        for it in em_items:
            key = it["title"][:40]
            if key not in seen:
                seen.add(key)
                items.append(it)

    # ── 再备选: 新浪搜索只用代码 ──
    if len(items) < 3:
        sina2 = _fetch_sina_search_news(stock_code, max_items=max_items)
        for it in sina2:
            key = it["title"][:40]
            if key not in seen:
                seen.add(key)
                items.append(it)

    return items[:max_items]


def get_stock_news_summary(stock_name: str, stock_code: str,
                            max_items: int = 8) -> dict:
    """
    个股新闻快照。

    数据源: 新浪财经搜索(主) + 东方财富搜索(备)。

    Returns
    -------
    dict: {items, score, label, query}
    """
    max_age = _default_max_age_days()
    items = _fetch_stock_news_all(stock_name, stock_code, max_items=max_items * 2)

    # 时效过滤
    filtered = []
    seen = set()
    for item in items:
        if not _is_within_window(item.get("pub_date", ""), max_age):
            continue
        key = item["title"][:40]
        if key in seen:
            continue
        seen.add(key)
        filtered.append(item)
        if len(filtered) >= max_items:
            break

    score = _stock_news_delta(filtered)
    label = _score_to_label(score)
    return {
        "items": filtered,
        "score": score,
        "label": label,
        "query": f"{stock_name}({stock_code})",
    }


def get_stock_news_report(stock_name: str, stock_code: str,
                           max_items: int = 8) -> str:
    """个股新闻中文报告"""
    snapshot = get_stock_news_summary(stock_name, stock_code, max_items=max_items)
    items = snapshot["items"]
    score = snapshot["score"]
    label = snapshot["label"]
    query = snapshot["query"]
    max_age = _default_max_age_days()

    lines = [
        "【个股新闻面】",
        f"  标的: {stock_name} ({stock_code})",
        f"  检索关键词: {query}",
        f"  时效窗口: {max_age}天内",
    ]

    if not items:
        lines.extend([
            "  （暂无最新个股新闻，可能受网络限制或近期无公开报道）",
            "  个股新闻情绪微调: +0.00",
            "  个股新闻结论: 中性（数据不足）",
        ])
        return "\n".join(lines)

    lines.append("")
    for idx, item in enumerate(items[:max_items], 1):
        title = item.get("title", "")
        if len(title) > 80:
            title = title[:77] + "..."
        date_hint = item.get("date_hint", "")
        suffix = f"  ({date_hint})" if date_hint else ""
        lines.append(f"  {idx}. {title}{suffix}")

    lines.extend([
        "",
        f"  有效新闻: {len(items)} 条",
        f"  个股新闻情绪微调: {score:+.2f}",
        f"  个股新闻结论: {label}",
        "",
        "  注: 仅包含近期新闻，旧消息已被市场定价。",
    ])
    return "\n".join(lines)


# ══════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    print(get_market_news_report())
    print()
    print("=" * 56)
    print("个股新闻测试: 贵州茅台 600519")
    print("=" * 56)
    print(get_stock_news_report("贵州茅台", "600519"))
