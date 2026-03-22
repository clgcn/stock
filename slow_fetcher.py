"""
慢速增量拉取器 — 绕过东财 API 频率限制
==========================================
核心思路：不对抗限制，在限制内慢慢积累数据。

两种数据，全部来自东方财富：

  1) 股票列表 + 基本面 (push2.eastmoney.com clist API)
     每次 1 页 100 条，~35 次拉完全 A 股。

  2) 个股历史 K 线 (push2his.eastmoney.com kline API)
     每次拉 N 只股票各 1 年日 K，增量存入同一个 SQLite。

数据库:  data/stocks.db
  - stocks        表: 全 A 股基本面快照 (PE/PB/市值/换手率...)
  - stock_history 表: 个股日 K 线历史 (OHLCV/涨跌幅/换手率...)
  - meta          表: 拉取进度追踪

用法:
  # ── 股票列表拉取 ──
  python slow_fetcher.py                          # 拉 1 页 (100条)
  python slow_fetcher.py --pages 5 --interval 60  # 拉 5 页，每页间隔 60s
  python slow_fetcher.py --auto --interval 3600   # 自动模式，每小时拉 1 页

  # ── K 线历史拉取 ──
  python slow_fetcher.py --history                    # 拉 5 只股票的 1 年日 K
  python slow_fetcher.py --history --batch 10         # 每次拉 10 只
  python slow_fetcher.py --history --batch 10 --interval 5  # 每只间隔 5 秒
  python slow_fetcher.py --history --auto --interval 3  # 自动模式，拉完为止
  python slow_fetcher.py --history --days 365         # 拉 N 天历史

  # ── 通用 ──
  python slow_fetcher.py --status                 # 查看所有进度
  python slow_fetcher.py --reset                  # 重置列表拉取进度
  python slow_fetcher.py --reset-history          # 重置 K 线拉取进度
  python slow_fetcher.py --export out.csv         # 导出股票列表到 CSV

数据来源: 东方财富 (push2 + push2his API)

Cron 示例:
  # 每小时拉 1 页股票列表
  0 * * * * cd /path/to/stock && .venv/bin/python slow_fetcher.py
  # 每 10 分钟拉 3 只股票的 K 线历史
  */10 * * * * cd /path/to/stock && .venv/bin/python slow_fetcher.py --history --batch 3
"""

import argparse
import json
import logging
import random
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 配置 ──────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "stocks.db"

EM_URL = "https://82.push2.eastmoney.com/api/qt/clist/get"
EM_FIELDS = (
    "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,"
    "f12,f14,f15,f16,f17,f18,f20,f21,f23,"
    "f24,f25,f34"
)
PAGE_SIZE = 100

SH_MAIN = {"600", "601", "603", "605"}
SZ_MAIN = {"000", "001", "002", "003"}

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]


# ── 数据库 ────────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stocks (
            code          TEXT PRIMARY KEY,
            name          TEXT,
            current       REAL,
            pct_chg       REAL,
            change        REAL,
            volume        REAL,
            amount        REAL,
            amplitude     REAL,
            high          REAL,
            low           REAL,
            open          REAL,
            prev_close    REAL,
            volume_ratio  REAL,
            turnover_rate REAL,
            pe_ttm        REAL,
            pb            REAL,
            total_mv      REAL,
            float_mv      REAL,
            chg_60d       REAL,
            chg_ytd       REAL,
            raw_json      TEXT,
            updated_at    TEXT,
            batch_id      TEXT,
            suspended     INTEGER DEFAULT 0
        )
    """)
    # 兼容旧库：如果 suspended 列不存在则添加
    try:
        conn.execute("ALTER TABLE stocks ADD COLUMN suspended INTEGER DEFAULT 0")
        conn.commit()
    except Exception:
        pass
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stock_history (
            code     TEXT NOT NULL,
            date     TEXT NOT NULL,
            open     REAL,
            close    REAL,
            high     REAL,
            low      REAL,
            volume   REAL,
            amount   REAL,
            amplitude REAL,
            pct_chg  REAL,
            change   REAL,
            turnover REAL,
            PRIMARY KEY (code, date)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_history_code
        ON stock_history(code)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()
    return conn


def _meta_get(conn: sqlite3.Connection, key: str, default: str = "") -> str:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def _meta_set(conn: sqlite3.Connection, key: str, value: str):
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        (key, str(value)),
    )
    conn.commit()


# ── 东财 API 请求 ─────────────────────────────────────────────

def _safe_float(v):
    if v is None or v == "-" or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _fetch_page(page: int) -> dict:
    """请求东财 clist API 的单页数据。"""
    try:
        from curl_cffi import requests
        use_cffi = True
    except ImportError:
        import requests
        use_cffi = False

    params = {
        "pn": page,
        "pz": PAGE_SIZE,
        "po": 1,
        "np": 1,
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": 2,
        "invt": 2,
        "fid": "f12",   # 按股票代码排序（稳定，不会随交易变化），避免分页漂移
        "fs": "m:0+t:6,m:1+t:2",
        "fields": EM_FIELDS,
        "_": str(int(time.time() * 1000)),
    }
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Referer": "https://quote.eastmoney.com/center/gridlist.html",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    NO_PROXY = {"http": None, "https": None}

    if use_cffi:
        resp = requests.get(
            EM_URL, params=params, headers=headers,
            timeout=20, impersonate="chrome",
        )
    else:
        resp = requests.get(
            EM_URL, params=params, headers=headers,
            timeout=20, proxies=NO_PROXY,
        )

    resp.raise_for_status()
    return resp.json()


def _parse_and_store(conn: sqlite3.Connection, records, batch_id: str) -> int:
    """解析东财返回的 diff 数据，存入数据库。返回入库条数。"""
    if isinstance(records, dict):
        records = list(records.values())

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    count = 0

    for r in records:
        if r.get("f2") is None or r.get("f2") == "-":
            continue

        code = str(r.get("f12", ""))
        if code[:3] not in SH_MAIN and code[:3] not in SZ_MAIN:
            continue

        current = _safe_float(r.get("f2"))
        if not current or current <= 0:
            continue

        total_mv = _safe_float(r.get("f20"))
        float_mv = _safe_float(r.get("f21"))
        amount = _safe_float(r.get("f6"))

        conn.execute("""
            INSERT OR REPLACE INTO stocks
            (code, name, current, pct_chg, change, volume, amount,
             amplitude, high, low, open, prev_close, volume_ratio,
             turnover_rate, pe_ttm, pb, total_mv, float_mv,
             chg_60d, chg_ytd, raw_json, updated_at, batch_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            code,
            r.get("f14", ""),
            current,
            _safe_float(r.get("f3")),
            _safe_float(r.get("f4")),
            _safe_float(r.get("f5")),
            amount / 1e4 if amount else None,
            _safe_float(r.get("f7")),
            _safe_float(r.get("f15")),
            _safe_float(r.get("f16")),
            _safe_float(r.get("f17")),
            _safe_float(r.get("f18")),
            _safe_float(r.get("f10")),
            _safe_float(r.get("f8")),
            _safe_float(r.get("f9")),
            _safe_float(r.get("f23")),
            total_mv / 1e8 if total_mv else None,
            float_mv / 1e8 if float_mv else None,
            _safe_float(r.get("f24")),
            _safe_float(r.get("f25")),
            json.dumps(r, ensure_ascii=False),
            now,
            batch_id,
        ))
        count += 1

    conn.commit()
    return count


# ── 主逻辑 ────────────────────────────────────────────────────

def fetch_next_page(conn: sqlite3.Connection = None) -> dict:
    """
    拉取下一页数据并存入数据库。

    Returns:
        {"page": int, "stored": int, "total": int, "done": bool, "error": str|None}
    """
    own_conn = conn is None
    if own_conn:
        conn = _get_db()

    try:
        batch_id = _meta_get(conn, "batch_id", "")
        if not batch_id:
            batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            _meta_set(conn, "batch_id", batch_id)

        page = int(_meta_get(conn, "next_page", "1"))
        total_expected = int(_meta_get(conn, "total_expected", "0"))

        log.info("拉取第 %d 页 (每页 %d 条)...", page, PAGE_SIZE)

        data = _fetch_page(page)
        api_data = data.get("data") or {}
        diff = api_data.get("diff")

        if total_expected == 0:
            total_expected = api_data.get("total", 0)
            _meta_set(conn, "total_expected", str(total_expected))
            log.info("东财报告总数: %d 条", total_expected)

        if not diff:
            _meta_set(conn, "done", "1")
            _meta_set(conn, "done_at", datetime.now().isoformat())
            return {"page": page, "stored": 0, "total": total_expected,
                    "done": True, "error": None}

        stored = _parse_and_store(conn, diff, batch_id)

        _meta_set(conn, "next_page", str(page + 1))
        _meta_set(conn, "last_fetch_at", datetime.now().isoformat())
        _meta_set(conn, "last_page_stored", str(stored))

        db_count = conn.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]
        total_pages = (total_expected + PAGE_SIZE - 1) // PAGE_SIZE if total_expected else "?"
        done = page >= total_pages if isinstance(total_pages, int) else False

        if done:
            _meta_set(conn, "done", "1")
            _meta_set(conn, "done_at", datetime.now().isoformat())

        log.info(
            "第 %d/%s 页  +%d 条  数据库累计: %d/%d  %s",
            page, total_pages, stored, db_count, total_expected,
            "✓ 全部完成!" if done else "",
        )

        return {"page": page, "stored": stored, "total": total_expected,
                "done": done, "error": None}

    except Exception as e:
        log.error("拉取失败: %s", e)
        return {"page": 0, "stored": 0, "total": 0,
                "done": False, "error": str(e)}
    finally:
        if own_conn:
            conn.close()


def get_status() -> dict:
    """获取拉取进度摘要。"""
    conn = _get_db()
    try:
        db_count = conn.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]
        total_expected_str = _meta_get(conn, "total_expected", "?")
        meta_done = _meta_get(conn, "done", "0") == "1"
        done_at = _meta_get(conn, "done_at", "")
        # 只要库存 >= 预期总数，也算完成（防止手动中断后状态卡住）
        if not meta_done and total_expected_str not in ("?", "0"):
            if db_count >= int(total_expected_str):
                meta_done = True
                if not done_at:
                    done_at = _meta_get(conn, "last_fetch_at", "")
        return {
            "db_path": str(DB_PATH),
            "stocks_in_db": db_count,
            "next_page": _meta_get(conn, "next_page", "1"),
            "total_expected": total_expected_str,
            "batch_id": _meta_get(conn, "batch_id", "N/A"),
            "last_fetch_at": _meta_get(conn, "last_fetch_at", "never"),
            "done": meta_done,
            "done_at": done_at,
        }
    finally:
        conn.close()


def reset_progress():
    """重置拉取进度 (不删已有数据，只是从第1页重新拉)。"""
    conn = _get_db()
    try:
        for key in ["next_page", "batch_id", "total_expected",
                     "done", "done_at", "last_fetch_at"]:
            conn.execute("DELETE FROM meta WHERE key=?", (key,))
        conn.commit()
        log.info("进度已重置，下次将从第 1 页开始")
    finally:
        conn.close()


def clear_all():
    """清空数据库中所有股票数据和进度。"""
    conn = _get_db()
    try:
        conn.execute("DELETE FROM stocks")
        conn.execute("DELETE FROM meta")
        conn.commit()
        log.info("数据库已清空")
    finally:
        conn.close()


# ── 读取接口 (供 screener / 其他模块使用) ─────────────────────

def load_stocks_from_db() -> pd.DataFrame:
    """
    从本地数据库读取全部股票数据，返回与 stock_screener.fetch_all_stocks()
    格式兼容的 DataFrame。
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"数据库不存在: {DB_PATH}，请先运行 slow_fetcher.py")

    conn = sqlite3.connect(str(DB_PATH))
    try:
        df = pd.read_sql_query(
            """SELECT code, name, current, pct_chg, change, volume, amount,
                      amplitude, high, low, open, prev_close, volume_ratio,
                      turnover_rate, pe_ttm, pb, total_mv, float_mv,
                      chg_60d, chg_ytd, updated_at
               FROM stocks
               WHERE current > 0
               ORDER BY code""",
            conn,
        )
        df.attrs["source"] = "local_db"
        df.attrs["db_path"] = str(DB_PATH)
        return df
    finally:
        conn.close()


def export_to_csv(path: str):
    """导出数据库到 CSV。"""
    df = load_stocks_from_db()
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log.info("导出 %d 条 → %s", len(df), path)


# ══════════════════════════════════════════════════════════════
# K 线历史拉取 (东财 push2his API)
# ══════════════════════════════════════════════════════════════

KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"


def _get_secid(code: str) -> str:
    code = str(code).strip()
    if code.startswith(("60", "68", "51", "11")):
        return f"1.{code}"
    return f"0.{code}"


def _fetch_kline_raw(code: str, days: int = 365, beg: str = None) -> list:
    """从东财拉取单只股票的日 K 线，返回原始行列表。
    beg: 起始日期字符串 YYYYMMDD，优先于 days 参数。"""
    try:
        from curl_cffi import requests
        use_cffi = True
    except ImportError:
        import requests
        use_cffi = False

    secid = _get_secid(code)
    if beg is None:
        beg = (datetime.today() - timedelta(days=days)).strftime("%Y%m%d")
    end = datetime.today().strftime("%Y%m%d")

    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "lmt": 500, "klt": 101, "fqt": 1,
        "secid": secid, "beg": beg, "end": end,
        "_": int(time.time() * 1000),
    }
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "application/json, text/plain, */*",
    }

    NO_PROXY = {"http": None, "https": None}

    if use_cffi:
        resp = requests.get(
            KLINE_URL, params=params, headers=headers,
            timeout=20, impersonate="chrome",
        )
    else:
        resp = requests.get(
            KLINE_URL, params=params, headers=headers,
            timeout=20, proxies=NO_PROXY,
        )

    resp.raise_for_status()
    data = resp.json()
    klines_data = data.get("data") or {}
    return klines_data.get("klines") or []


def _store_kline(conn: sqlite3.Connection, code: str, raw_lines: list) -> int:
    """解析 K 线并存入 stock_history 表，返回入库条数。"""
    count = 0
    for line in raw_lines:
        parts = line.split(",")
        if len(parts) < 11:
            continue
        try:
            conn.execute("""
                INSERT OR REPLACE INTO stock_history
                (code, date, open, close, high, low, volume, amount,
                 amplitude, pct_chg, change, turnover)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                code,
                parts[0],                     # date
                _safe_float(parts[1]),         # open
                _safe_float(parts[2]),         # close
                _safe_float(parts[3]),         # high
                _safe_float(parts[4]),         # low
                _safe_float(parts[5]),         # volume
                _safe_float(parts[6]),         # amount
                _safe_float(parts[7]),         # amplitude
                _safe_float(parts[8]),         # pct_chg
                _safe_float(parts[9]),         # change
                _safe_float(parts[10]),        # turnover
            ))
            count += 1
        except (ValueError, IndexError):
            continue
    conn.commit()
    return count


def fetch_history_batch(
    batch_size: int = 5,
    days: int = 365,
    interval: float = 2,
    conn: sqlite3.Connection = None,
) -> dict:
    """
    从 stocks 表读取下一批股票，拉取它们的日 K 线历史。

    Returns:
        {"fetched": int, "remaining": int, "done": bool, "errors": list}
    """
    own_conn = conn is None
    if own_conn:
        conn = _get_db()

    try:
        # 所有已入库且未标记停牌的股票代码
        all_codes = [r[0] for r in conn.execute(
            "SELECT code FROM stocks WHERE suspended = 0 ORDER BY code"
        ).fetchall()]

        if not all_codes:
            log.warning("stocks 表为空，请先拉取股票列表")
            return {"fetched": 0, "remaining": 0, "done": False,
                    "errors": ["stocks 表为空"]}

        # 每只股票的最新真实 K 线日期（close IS NOT NULL = 真实数据，排除占位）
        latest_date_map = {
            r[0]: r[1] for r in conn.execute(
                "SELECT code, MAX(date) FROM stock_history "
                "WHERE close IS NOT NULL GROUP BY code"
            ).fetchall()
        }

        # 每只股票最近一次"检查"日期（含占位记录）
        latest_checked_map = {
            r[0]: r[1] for r in conn.execute(
                "SELECT code, MAX(date) FROM stock_history GROUP BY code"
            ).fetchall()
        }

        # 用库中已有的最新交易日作为"已是最新"的判断标准
        latest_trading_day = conn.execute(
            "SELECT MAX(date) FROM stock_history WHERE close IS NOT NULL"
        ).fetchone()[0] or "0000-00-00"

        log.info("库中最新交易日: %s", latest_trading_day)

        # 需要更新的股票：
        #   1. 从未检查过 → 全量拉取
        #   2. 有真实 K 线但数据过期，且本交易日尚未确认过无数据 → 增量拉取
        #   3. 已确认无数据（占位日期 >= latest_trading_day）→ 跳过
        remaining = [
            c for c in all_codes
            if latest_checked_map.get(c, "0000-00-00") < latest_trading_day
            and (c not in latest_date_map                                  # 从未有真实数据
                 or latest_date_map[c] < latest_trading_day)               # 真实数据过期
        ]

        if not remaining:
            _meta_set(conn, "history_done", "1")
            _meta_set(conn, "history_done_at", datetime.now().isoformat())
            return {"fetched": 0, "remaining": 0, "done": True, "errors": []}

        batch = remaining[:batch_size]
        fetched = 0
        errors = []

        for i, code in enumerate(batch):
            name = conn.execute(
                "SELECT name FROM stocks WHERE code=?", (code,)
            ).fetchone()
            name = name[0] if name else code

            # 确定增量起点：有历史则从最新日期拉，否则全量拉 days 天
            last_date = latest_date_map.get(code)
            if last_date:
                # 从该股最新日期的下一天开始，避免重复拉已有数据
                next_day = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y%m%d")
                beg = next_day
                mode_str = f"增量 from {last_date}"
            else:
                beg = None
                mode_str = f"全量 {days}天"

            try:
                raw = _fetch_kline_raw(code, days=days, beg=beg)
                if raw:
                    stored = _store_kline(conn, code, raw)
                    fetched += 1
                    log.info(
                        "  [%d/%d] %s %s  +%d 条日K  %s  (剩余 %d 只)",
                        i + 1, len(batch), code, name,
                        stored, mode_str, len(remaining) - i - 1,
                    )
                else:
                    log.warning("  [%d/%d] %s %s  无数据，标记为停牌",
                               i + 1, len(batch), code, name)
                    conn.execute(
                        "UPDATE stocks SET suspended = 1 WHERE code = ?",
                        (code,),
                    )
                    conn.commit()
            except Exception as e:
                log.error("  [%d/%d] %s %s  失败: %s",
                         i + 1, len(batch), code, name, e)
                errors.append(f"{code}: {e}")

            if i < len(batch) - 1:
                time.sleep(interval)

        _meta_set(conn, "history_last_fetch", datetime.now().isoformat())

        new_remaining = len(remaining) - len(batch)
        done = new_remaining <= 0

        if done:
            _meta_set(conn, "history_done", "1")
            _meta_set(conn, "history_done_at", datetime.now().isoformat())

        return {
            "fetched": fetched,
            "remaining": max(new_remaining, 0),
            "done": done,
            "errors": errors,
        }

    finally:
        if own_conn:
            conn.close()


def reset_history_progress():
    """清空 K 线历史数据，重新拉取。"""
    conn = _get_db()
    try:
        conn.execute("DELETE FROM stock_history")
        for key in ["history_done", "history_done_at", "history_last_fetch"]:
            conn.execute("DELETE FROM meta WHERE key=?", (key,))
        conn.commit()
        log.info("K 线历史进度已重置")
    finally:
        conn.close()


# ── K 线历史读取接口 ──────────────────────────────────────────

def load_stock_history(code: str) -> pd.DataFrame:
    """读取单只股票的日 K 线历史。"""
    conn = sqlite3.connect(str(DB_PATH))
    try:
        df = pd.read_sql_query(
            "SELECT * FROM stock_history WHERE code=? AND close IS NOT NULL ORDER BY date",
            conn, params=(code,),
        )
        for col in ["open", "close", "high", "low", "volume", "amount",
                     "amplitude", "pct_chg", "change", "turnover"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    finally:
        conn.close()


def load_all_history() -> pd.DataFrame:
    """读取全部股票的日 K 线历史 (大表)。"""
    conn = sqlite3.connect(str(DB_PATH))
    try:
        df = pd.read_sql_query(
            "SELECT * FROM stock_history WHERE close IS NOT NULL ORDER BY code, date",
            conn,
        )
        for col in ["open", "close", "high", "low", "volume", "amount",
                     "amplitude", "pct_chg", "change", "turnover"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════
# 实时分析: 本地历史 + 线上今日行情 → 量化诊断
# ══════════════════════════════════════════════════════════════

def analyze_stock(code: str, days: int = 120) -> dict:
    """
    混合分析单只股票:
      - 历史 K 线 → 从本地 DB 读 (0 网络)
      - 今日实时行情 → 新浪 1 次 API 调用 (极轻量)
      - 基本面快照 → 从本地 DB 读 (0 网络)
      - 拼接后跑完整量化诊断

    Returns:
        {
          "code", "name", "current", "pct_chg",
          "pe_ttm", "pb", "total_mv",
          "total_score", "signal", "confidence",
          "trend", "momentum", "risk_level",
          "monte_carlo": {...},
          "kline_source": "local+realtime" / "local_only",
          "history_bars": int,
          "summary": str,
        }
    """
    import quant_engine as qe

    code = str(code).strip()

    # ── 1) 本地历史 K 线 ──
    history = load_stock_history(code)
    if history.empty:
        raise ValueError(f"{code} 在本地数据库中没有 K 线历史，请先运行: "
                        f"python slow_fetcher.py --history")

    cutoff = pd.Timestamp.today() - pd.Timedelta(days=days)
    history = history[history["date"] >= cutoff].copy()

    if len(history) < 20:
        raise ValueError(f"{code} 历史数据不足 (仅 {len(history)} 条)，需至少 20 条")

    # ── 2) 线上实时行情 (新浪, 1 次轻量调用) ──
    kline_source = "local_only"
    realtime_info = {}
    try:
        from stock_tool import get_realtime
        rt = get_realtime(code)
        if not rt.empty:
            r = rt.iloc[0]
            realtime_info = r.to_dict()

            today_str = r.get("date", "")
            last_hist_date = history["date"].max()

            # 如果今天的数据还没在历史里，拼接进去
            if (today_str and r.get("current", 0) > 0
                    and (pd.Timestamp(today_str) > last_hist_date)):
                today_bar = pd.DataFrame([{
                    "code": code,
                    "date": pd.Timestamp(today_str),
                    "open": r.get("open"),
                    "close": r.get("current"),
                    "high": r.get("high"),
                    "low": r.get("low"),
                    "volume": r.get("volume"),
                    "amount": r.get("amount"),
                    "pct_chg": r.get("pct_chg"),
                    "change": r.get("change"),
                    "turnover": None,
                    "amplitude": None,
                }])
                history = pd.concat([history, today_bar], ignore_index=True)
                kline_source = "local+realtime"
                log.info("拼接今日实时行情: %s %.2f (%+.2f%%)",
                        today_str, r["current"], r["pct_chg"])
    except Exception as e:
        log.warning("实时行情获取失败 (仅用本地历史): %s", e)

    # ── 3) 本地基本面快照 ──
    fundamentals = {}
    try:
        conn = sqlite3.connect(str(DB_PATH))
        row = conn.execute(
            "SELECT name, pe_ttm, pb, total_mv, float_mv, turnover_rate, "
            "volume_ratio, chg_60d, chg_ytd FROM stocks WHERE code=?",
            (code,),
        ).fetchone()
        conn.close()
        if row:
            fundamentals = {
                "name": row[0], "pe_ttm": row[1], "pb": row[2],
                "total_mv": row[3], "float_mv": row[4],
                "turnover_rate": row[5], "volume_ratio": row[6],
                "chg_60d": row[7], "chg_ytd": row[8],
            }
    except Exception:
        pass

    name = realtime_info.get("name") or fundamentals.get("name") or code
    current = realtime_info.get("current") or float(history["close"].iloc[-1])
    pct_chg = realtime_info.get("pct_chg", 0)

    # ── 4) 量化诊断 ──
    diag = qe.comprehensive_diagnosis(history, run_monte_carlo=True, mc_days=20)

    mc = diag.get("monte_carlo") or {}
    td = diag.get("trend_detail", {})
    vd = diag.get("volatility_detail", {})
    md = diag.get("momentum_detail", {})
    mrd = diag.get("mean_reversion_detail", {})

    return {
        "code": code,
        "name": name,
        "current": current,
        "pct_chg": pct_chg,
        # 基本面
        "pe_ttm": fundamentals.get("pe_ttm"),
        "pb": fundamentals.get("pb"),
        "total_mv": fundamentals.get("total_mv"),
        "float_mv": fundamentals.get("float_mv"),
        "chg_60d": fundamentals.get("chg_60d"),
        "chg_ytd": fundamentals.get("chg_ytd"),
        # 量化诊断
        "total_score": diag["total_score"],
        "signal": diag["signal"],
        "confidence": diag["confidence"],
        "trend_dir": td.get("direction"),
        "trend_strength": td.get("strength"),
        "momentum": md.get("rsi_signal"),
        "mean_reversion": mrd.get("zscore"),
        "risk_level": vd.get("risk_level"),
        # 蒙特卡洛
        "prob_up": mc.get("prob_up"),
        "prob_down": mc.get("prob_down"),
        "expected_return": mc.get("expected_return"),
        # 元信息
        "kline_source": kline_source,
        "history_bars": len(history),
        "risk_warnings": diag.get("risk_warnings", []),
        "summary": diag.get("summary", ""),
    }


def print_analysis(code: str, days: int = 120):
    """打印单只股票的完整分析报告。"""
    r = analyze_stock(code, days=days)

    signal_colors = {
        "strong_buy": "🟢🟢", "buy": "🟢", "hold": "🟡",
        "sell": "🔴", "strong_sell": "🔴🔴",
    }
    sig = signal_colors.get(r["signal"], "⚪")

    print(f"""
{'='*60}
  {r['name']} ({r['code']})   {sig} {r['signal'].upper()}
{'='*60}

  现价:     {r['current']:.2f}    涨跌: {r['pct_chg']:+.2f}%
  PE(TTM):  {r['pe_ttm'] or 'N/A'}
  PB:       {r['pb'] or 'N/A'}
  市值:     {f"{r['total_mv']:.0f}亿" if r['total_mv'] else 'N/A'}

  ── 量化诊断 ──
  总分:     {r['total_score']:+.1f} / 100
  信号:     {r['signal']}  (置信度: {r['confidence']})
  趋势:     {r['trend_dir']} ({r['trend_strength']})
  动量:     {r['momentum'] or 'N/A'}
  风险:     {r['risk_level']}
""")

    if r.get("prob_up") is not None:
        print(f"""  ── 蒙特卡洛模拟 (20日) ──
  上涨概率: {r['prob_up']:.0f}%
  下跌概率: {r['prob_down']:.0f}%
  预期收益: {r['expected_return']:+.1f}%
""")

    if r.get("risk_warnings"):
        print("  ── 风险提示 ──")
        for w in r["risk_warnings"]:
            print(f"  ⚠ {w}")
        print()

    print(f"  数据: {r['history_bars']}条K线 ({r['kline_source']})")
    print(f"{'='*60}")


# ── CLI ───────────────────────────────────────────────────────

def _print_status():
    s = get_status()
    total = s["total_expected"]
    total_pages = (
        (int(total) + PAGE_SIZE - 1) // PAGE_SIZE
        if total != "?" and total != "0"
        else "?"
    )
    # 已完成时直接显示满页，避免因手动中断导致页码偏低
    if s["done"] and isinstance(total_pages, int):
        pages_done = total_pages
    else:
        pages_done = int(s["next_page"]) - 1
    pct = (
        f"{s['stocks_in_db']}/{total} ({s['stocks_in_db']/int(total)*100:.1f}%)"
        if total not in ("?", "0")
        else f"{s['stocks_in_db']}/?"
    )
    eta = ""
    if not s["done"] and isinstance(total_pages, int) and pages_done < total_pages:
        remaining = total_pages - pages_done
        eta = f"  预计还需 ~{remaining} 次拉取"

    # K 线历史统计
    conn = _get_db()
    try:
        # 非停牌股中有 K 线数据的数量
        history_stocks = conn.execute(
            """SELECT COUNT(DISTINCT h.code) FROM stock_history h
               JOIN stocks s ON h.code = s.code
               WHERE h.close IS NOT NULL AND s.suspended = 0"""
        ).fetchone()[0]
        history_rows = conn.execute(
            "SELECT COUNT(*) FROM stock_history WHERE close IS NOT NULL"
        ).fetchone()[0]
        # 非停牌股中还没有 K 线数据的数量
        history_remaining = conn.execute(
            """SELECT COUNT(*) FROM stocks s
               WHERE s.suspended = 0
               AND s.code NOT IN (
                   SELECT DISTINCT code FROM stock_history WHERE close IS NOT NULL
               )"""
        ).fetchone()[0]
        suspended_count = conn.execute(
            "SELECT COUNT(*) FROM stocks WHERE suspended = 1"
        ).fetchone()[0]
        history_done = _meta_get(conn, "history_done", "0") == "1"
        history_last = _meta_get(conn, "history_last_fetch", "never")
        history_done_at = _meta_get(conn, "history_done_at", "")
    finally:
        conn.close()

    stocks_in_db = s["stocks_in_db"]
    active_stocks = max(stocks_in_db - suspended_count, 0)

    print(f"""
╔══════════════════════════════════════════════════════╗
║              慢速拉取进度总览                         ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  [股票列表] 东财 clist API                           ║
║    数据库:     {s['db_path']}
║    库中股票:   {pct}
║    页码进度:   第 {pages_done}/{total_pages} 页
║    上次拉取:   {s['last_fetch_at']}
║    状态:       {'✓ 已完成 (' + s['done_at'] + ')' if s['done'] else '进行中' + eta}
║                                                      ║
║  [K线历史] 东财 kline API                            ║
║    已拉股票:   {history_stocks}/{active_stocks} 只 (停牌/无数据: {suspended_count} 只)
║    K线总条数:  {history_rows:,} 条
║    剩余:       {history_remaining} 只待拉
║    上次拉取:   {history_last}
║    状态:       {'✓ 已完成 (' + history_done_at + ')' if history_done else '进行中'}
║                                                      ║
╚══════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(
        description="慢速增量拉取器 — 东财数据存 SQLite",
    )
    # 通用
    parser.add_argument("--interval", type=int, default=5,
                        help="请求间隔秒数 (默认5)")
    parser.add_argument("--auto", action="store_true",
                        help="自动模式: 持续拉取直到完成")
    parser.add_argument("--status", action="store_true",
                        help="显示拉取进度")
    parser.add_argument("--export", type=str,
                        help="导出股票列表到 CSV")

    # 股票列表模式
    parser.add_argument("--pages", type=int, default=1,
                        help="拉取几页股票列表 (默认1)")
    parser.add_argument("--check-new", action="store_true",
                        help="只拉最后一页，检查是否有新上市股票")
    parser.add_argument("--reset", action="store_true",
                        help="重置列表拉取进度")
    parser.add_argument("--clear", action="store_true",
                        help="清空整个数据库")

    # K 线历史模式
    parser.add_argument("--history", action="store_true",
                        help="拉取 K 线历史 (而非股票列表)")
    parser.add_argument("--batch", type=int, default=5,
                        help="每次拉几只股票的 K 线 (默认5)")
    parser.add_argument("--days", type=int, default=365,
                        help="拉取多少天的历史 (默认365)")
    parser.add_argument("--reset-history", action="store_true",
                        help="重置 K 线拉取进度")

    # 实时分析模式
    parser.add_argument("--analyze", type=str, metavar="CODE",
                        help="分析单只股票 (本地历史+线上实时)")
    parser.add_argument("--analyze-days", type=int, default=120,
                        help="分析用多少天历史 (默认120)")

    args = parser.parse_args()

    if args.status:
        _print_status()
        return

    if args.clear:
        clear_all()
        return

    if args.reset:
        reset_progress()
        return

    if args.reset_history:
        reset_history_progress()
        return

    if args.export:
        export_to_csv(args.export)
        return

    if args.analyze:
        print_analysis(args.analyze, days=args.analyze_days)
        return

    if args.check_new:
        # 只拉最后一页，检查新上市股票
        conn = _get_db()
        try:
            total_expected = int(_meta_get(conn, "total_expected", "3500") or "3500")
            last_page = (total_expected + PAGE_SIZE - 1) // PAGE_SIZE
            before = conn.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]
            if before >= total_expected:
                log.info("股票列表已满 (%d 只)，如有新股上市会自动超出此数，当前无需检查。", before)
                conn.close()
                return
            log.info("检查新股: 拉取第 %d 页 (每页 %d 条)...", last_page, PAGE_SIZE)
            data = _fetch_page(last_page)
            api_data = data.get("data") or {}
            diff = api_data.get("diff")
            if diff:
                batch_id = _meta_get(conn, "batch_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
                _parse_and_store(conn, diff, batch_id)
            after = conn.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]
            new_count = after - before
            if new_count > 0:
                log.info("发现 %d 只新股票！总计: %d 只", new_count, after)
            else:
                log.info("没有新股票，总计: %d 只", after)
        finally:
            conn.close()
        _print_status()
        return

    # ══════════════════════════════════════════
    # K 线历史拉取模式
    # ══════════════════════════════════════════
    if args.history:
        conn = _get_db()
        try:
            stocks_count = conn.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]
            if stocks_count == 0:
                print("错误: stocks 表为空，请先拉取股票列表 (不带 --history)")
                return

            if args.auto:
                print(f"K 线自动模式: 每次 {args.batch} 只，间隔 {args.interval}s...")
                rounds = 0
                while True:
                    result = fetch_history_batch(
                        batch_size=args.batch,
                        days=args.days,
                        interval=args.interval,
                        conn=conn,
                    )
                    rounds += 1
                    if result["done"]:
                        print(f"\n全部 K 线拉取完成! (共 {rounds} 轮)")
                        break
                    log.info("剩余 %d 只，等待 %ds 后继续...",
                            result["remaining"], args.interval)
                    time.sleep(args.interval)
            else:
                result = fetch_history_batch(
                    batch_size=args.batch,
                    days=args.days,
                    interval=args.interval,
                    conn=conn,
                )
                if result["done"]:
                    print("\n全部 K 线拉取完成!")
                else:
                    print(f"\n本次拉了 {result['fetched']} 只，"
                          f"剩余 {result['remaining']} 只待拉")
                if result["errors"]:
                    print(f"失败: {result['errors']}")
        finally:
            conn.close()

        _print_status()
        return

    # ══════════════════════════════════════════
    # 股票列表拉取模式 (默认)
    # ══════════════════════════════════════════
    conn = _get_db()
    try:
        # 如果股票列表已满，不再继续刷新，直接提示
        stocks_in_db = conn.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]
        total_expected = int(_meta_get(conn, "total_expected", "0") or "0")
        if total_expected > 0 and stocks_in_db >= total_expected and not args.auto:
            log.info("股票列表已完整 (%d 只)，无需重新拉取。", stocks_in_db)
            log.info("如需检查新上市股票，运行: python slow_fetcher.py --check-new")
            log.info("如需更新 K 线数据，运行:  python slow_fetcher.py --history --auto --batch 100 --interval 10")
            conn.close()
            _print_status()
            return

        if args.auto:
            print("列表自动模式: 每隔 %ds 拉一页..." % args.interval)
            while True:
                result = fetch_next_page(conn)
                if result["error"]:
                    log.error("出错: %s，%ds 后重试...",
                             result["error"], args.interval)
                    time.sleep(args.interval)
                    continue
                if result["done"]:
                    print("\n股票列表拉取完成!")
                    break
                log.info("等待 %d 秒...", args.interval)
                time.sleep(args.interval)
        else:
            for i in range(args.pages):
                result = fetch_next_page(conn)
                if result["error"]:
                    log.error("出错: %s", result["error"])
                    break
                if result["done"]:
                    print("\n股票列表拉取完成!")
                    break
                if i < args.pages - 1:
                    log.info("等待 %d 秒...", args.interval)
                    time.sleep(args.interval)
    finally:
        conn.close()

    _print_status()


if __name__ == "__main__":
    main()
