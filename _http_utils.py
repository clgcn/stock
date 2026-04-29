"""
HTTP 工具函数 — 所有底层模块的共用网络请求基础设施
=================================================
提供:
  _headers()      → 浏览器UA头
  _get()          → 带TLS指纹伪装的GET请求
  _get_secid()    → 股票代码→东方财富secid
  _sina_prefix()  → 股票代码→新浪前缀
  eastmoney_throttle  → 全局东财 API 限流器 (令牌桶)
  kline_cache     → K线内存缓存（同一进程内同一股票不重复请求）
"""

try:
    from curl_cffi import requests
except ImportError:
    import requests
import time
import threading
import logging
from functools import lru_cache

_log = logging.getLogger(__name__)

_NO_PROXY = {"http": "", "https": ""}
_USE_CURL_CFFI = hasattr(requests, "impersonate")


# ─── 全局东财 API 令牌桶限流器 ─────────────────────────────
# 所有对 eastmoney 的请求都应先调用 eastmoney_throttle.acquire()
# 确保并发场景（如 full_stock_selection 8线程并行）也不会触发限流

class _TokenBucket:
    """线程安全的令牌桶限流器。

    rate: 每秒放入令牌数（即最大 QPS）
    burst: 桶容量（允许的突发数量）

    用法:
        eastmoney_throttle.acquire()   # 阻塞直到拿到令牌
        resp = requests.get(...)
    """

    def __init__(self, rate: float = 2.0, burst: int = 3):
        self.rate = rate          # 每秒补充令牌数
        self.burst = burst        # 桶容量上限
        self._tokens = float(burst)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 30.0) -> bool:
        """阻塞等待直到获取一个令牌。返回 True 表示成功。"""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                now = time.monotonic()
                # 按时间补充令牌
                elapsed = now - self._last
                self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
                self._last = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True

                # 计算需要等多久才有 1 个令牌
                wait = (1.0 - self._tokens) / self.rate

            if time.monotonic() + wait > deadline:
                _log.warning("Throttle timeout after %.1fs", timeout)
                return False  # 超时也放行，避免死锁

            time.sleep(min(wait, 0.5))  # 每次最多 sleep 0.5s，避免持锁太久


# 默认: 2 QPS，突发3。15只股票各3次请求 = 45次，45/2 = ~22秒完成，
# 比起被限流后重试要快且稳定得多。
eastmoney_throttle = _TokenBucket(rate=2.0, burst=3)


# ─── K 线内存缓存 ──────────────────────────────────────────
# 同一进程内，相同 (code, period, start, end, adjust) 不重复请求。
# TTL: 缓存 5 分钟后过期（盘中数据需要刷新）。

class _KlineCache:
    """线程安全的带 TTL 的 K 线缓存。"""

    def __init__(self, ttl: int = 300):
        self._cache: dict = {}  # key → (timestamp, value)
        self._ttl = ttl
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            ts, val = entry
            if time.monotonic() - ts > self._ttl:
                del self._cache[key]
                return None
            return val

    def put(self, key, value):
        with self._lock:
            self._cache[key] = (time.monotonic(), value)

    def clear(self):
        with self._lock:
            self._cache.clear()


kline_cache = _KlineCache(ttl=300)


def _headers():
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Referer": "https://finance.eastmoney.com/",
    }


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


def tencent_symbol(code: str) -> str:
    """Return Tencent quote symbol, e.g. sh600519 / sz000858"""
    code = str(code).strip().upper().replace("SH", "").replace("SZ", "")
    if code.startswith(("60", "68", "51", "11")):
        return f"sh{code}"
    return f"sz{code}"


# ─── 中国时区工具 ────────────────────────────────────────
# A股分析所有日期/时间必须基于 Asia/Shanghai 时区。
# 所有模块统一使用这里导出的工具函数，不再直接调用
# datetime.now() / datetime.today()。

from datetime import datetime as _dt, date as _date, timedelta as _timedelta
from zoneinfo import ZoneInfo

CN_TZ = ZoneInfo("Asia/Shanghai")


def cn_now() -> _dt:
    """返回当前中国时间（带时区信息）。"""
    return _dt.now(CN_TZ)


def cn_today() -> _date:
    """返回当前中国日期。"""
    return _dt.now(CN_TZ).date()


def cn_str(fmt: str = "%Y-%m-%d") -> str:
    """返回当前中国日期的字符串。"""
    return _dt.now(CN_TZ).strftime(fmt)


# ─── A股休市日历 ─────────────────────────────────────
# 包含法定节假日休市 + 周末补班交易日。
# 数据来源: 中国证监会/上交所/深交所每年底公布的休市安排。
# 每年末需根据官方公告更新下一年度数据。

# 节假日休市（工作日但不开盘）
_CN_HOLIDAYS_2025: set = {
    _date(2025, 1, 1),                                          # 元旦
    _date(2025, 1, 28), _date(2025, 1, 29), _date(2025, 1, 30),
    _date(2025, 1, 31), _date(2025, 2, 1), _date(2025, 2, 2),
    _date(2025, 2, 3), _date(2025, 2, 4),                      # 春节
    _date(2025, 4, 4),                                          # 清明
    _date(2025, 5, 1), _date(2025, 5, 2), _date(2025, 5, 5),   # 劳动节
    _date(2025, 5, 31), _date(2025, 6, 1), _date(2025, 6, 2),  # 端午
    _date(2025, 10, 1), _date(2025, 10, 2), _date(2025, 10, 3),
    _date(2025, 10, 6), _date(2025, 10, 7), _date(2025, 10, 8), # 国庆
}

_CN_HOLIDAYS_2026: set = {
    _date(2026, 1, 1), _date(2026, 1, 2),                      # 元旦
    _date(2026, 2, 16), _date(2026, 2, 17), _date(2026, 2, 18),
    _date(2026, 2, 19), _date(2026, 2, 20), _date(2026, 2, 23),
    _date(2026, 2, 24),                                         # 春节
    _date(2026, 4, 5), _date(2026, 4, 6),                      # 清明
    _date(2026, 5, 1), _date(2026, 5, 4), _date(2026, 5, 5),   # 劳动节
    _date(2026, 6, 19),                                         # 端午
    _date(2026, 10, 1), _date(2026, 10, 2), _date(2026, 10, 5),
    _date(2026, 10, 6), _date(2026, 10, 7), _date(2026, 10, 8), # 国庆
}

# 2027 (预估，需官方确认后更新)
_CN_HOLIDAYS_2027: set = {
    _date(2027, 1, 1),                                          # 元旦
    _date(2027, 2, 5), _date(2027, 2, 8), _date(2027, 2, 9),
    _date(2027, 2, 10), _date(2027, 2, 11), _date(2027, 2, 12), # 春节
    _date(2027, 4, 5),                                          # 清明
    _date(2027, 5, 3), _date(2027, 5, 4), _date(2027, 5, 5),   # 劳动节
    _date(2027, 6, 14),                                         # 端午
    _date(2027, 10, 1), _date(2027, 10, 4), _date(2027, 10, 5),
    _date(2027, 10, 6), _date(2027, 10, 7), _date(2027, 10, 8), # 国庆
}

_CN_HOLIDAYS: set = _CN_HOLIDAYS_2025 | _CN_HOLIDAYS_2026 | _CN_HOLIDAYS_2027

# 周末补班交易日（周六/日但开盘）
_CN_EXTRA_TRADE_DAYS_2025: set = {
    _date(2025, 1, 26),   # 春节调休
    _date(2025, 2, 8),    # 春节调休
    _date(2025, 4, 27),   # 劳动节调休
    _date(2025, 9, 28),   # 国庆调休
    _date(2025, 10, 11),  # 国庆调休
}

_CN_EXTRA_TRADE_DAYS_2026: set = {
    _date(2026, 2, 14),   # 春节调休
    _date(2026, 2, 28),   # 春节调休
    _date(2026, 4, 26),   # 劳动节调休
    _date(2026, 5, 9),    # 劳动节调休
    _date(2026, 9, 27),   # 国庆调休
    _date(2026, 10, 10),  # 国庆调休
}

_CN_EXTRA_TRADE_DAYS: set = _CN_EXTRA_TRADE_DAYS_2025 | _CN_EXTRA_TRADE_DAYS_2026


def is_trade_day(d: _date) -> bool:
    """判断某天是否为A股交易日。"""
    # 周末补班日
    if d in _CN_EXTRA_TRADE_DAYS:
        return True
    # 周末
    if d.weekday() >= 5:
        return False
    # 法定假日
    if d in _CN_HOLIDAYS:
        return False
    return True


def next_trade_day(ref: _date = None) -> _date:
    """返回下一个A股交易日。

    ref 默认为中国当天日期。
    支持法定节假日和周末补班。
    """
    d = ref or cn_today()
    d += _timedelta(days=1)
    while not is_trade_day(d):
        d += _timedelta(days=1)
    return d


def last_trade_day(ref: _date = None) -> _date:
    """返回最近一个已收盘的A股交易日。

    ref 默认为中国当天日期。当天交易日收盘时间15:00为界：
      - 15:00前 → 返回上一个交易日
      - 15:00后 → 返回当天
    若传入 ref 为历史日期，直接向前找最近的交易日（不考虑当日收盘状态）。
    """
    today = cn_today()
    d = ref if ref is not None else today

    if d < today:
        # 历史日期：向前找最近交易日（不受当日收盘时间约束）
        while not is_trade_day(d):
            d -= _timedelta(days=1)
        return d

    # d == today
    if is_trade_day(d) and _dt.now(CN_TZ).hour >= 15:
        return d

    # 今日未收盘或非交易日，向前找
    d -= _timedelta(days=1)
    while not is_trade_day(d):
        d -= _timedelta(days=1)
    return d
