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
