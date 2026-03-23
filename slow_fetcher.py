"""
慢速增量拉取器 — 绕过东财 API 频率限制
==========================================
核心思路：不对抗限制，在限制内慢慢积累数据。

两种数据，全部来自东方财富：

  1) 股票列表元信息 (push2.eastmoney.com clist API)
     只维护 code / name / suspended 这类股票宇宙信息。

  2) 个股历史 K 线 (push2his.eastmoney.com kline API)
     每次拉 N 只股票各 1 年日 K，增量存入同一个 SQLite。
     日度行情、涨跌幅、换手率等实际分析数据全部以 stock_history 为准。

数据库:  data/stocks.db
  - stocks        表: 股票总览元信息 (code / name / suspended)
  - stock_history 表: 个股日 K 线历史 (OHLCV/涨跌幅/换手率...)
  - stock_fundamentals 表: 每日估值快照 (PE/PB/总市值/流通市值)
  - meta          表: 拉取进度追踪

用法:
  # ── 股票列表导入 ──
  python import_json.py                           # 从本地 JSON 导入 stocks 名单

  # ── K 线历史拉取 ──
  python slow_fetcher.py --history                    # 拉 5 只股票的 1 年日 K
  python slow_fetcher.py --history --batch 10         # 每次拉 10 只
  python slow_fetcher.py --history --batch 10 --interval 5  # 每只间隔 5 秒
  python slow_fetcher.py --history --auto --interval 3  # 自动模式，拉完为止
  python slow_fetcher.py --history --days 365         # 拉 N 天历史
  python slow_fetcher.py --intraday-update
      # 盘中刷新股票宇宙 + fundamentals 快照（不动正式日K）
  python slow_fetcher.py --daily-close-update --batch 100 --interval 3
      # 收盘后统一更新 fundamentals + 当日/缺失 history

  # ── 通用 ──
  python slow_fetcher.py --status                 # 查看所有进度
  python slow_fetcher.py --reset                  # 重置列表拉取进度
  python slow_fetcher.py --reset-history          # 重置 K 线拉取进度
  python slow_fetcher.py --export out.csv         # 导出股票列表到 CSV

数据来源: 东方财富 (push2 + push2his API)

Cron 示例:
  # 每 10 分钟拉 3 只股票的 K 线历史
  */10 * * * * cd /path/to/stock && .venv/bin/python slow_fetcher.py --history --batch 3
"""

import argparse
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

def _connect_db() -> sqlite3.Connection:
    """Open a SQLite connection with common pragmas."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _init_schema(conn: sqlite3.Connection):
    """Create base tables/indexes if they do not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stocks (
            code          TEXT PRIMARY KEY,
            name          TEXT,
            suspended     INTEGER DEFAULT 0
        )
    """)
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
        CREATE TABLE IF NOT EXISTS stock_fundamentals (
            code          TEXT NOT NULL,
            trade_date    TEXT NOT NULL,
            pe_ttm        REAL,
            pb            REAL,
            total_mv      REAL,
            float_mv      REAL,
            updated_at    TEXT,
            source        TEXT,
            batch_id      TEXT,
            PRIMARY KEY (code, trade_date)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_fundamentals_code_date
        ON stock_fundamentals(code, trade_date)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()


def _get_db() -> sqlite3.Connection:
    """Open a ready-to-use database connection without running migrations."""
    conn = _connect_db()
    _init_schema(conn)
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


def _upsert_stock_meta(conn: sqlite3.Connection, code: str, name: str,
                       suspended: int = 0):
    """Keep stocks as a lightweight universe table."""
    conn.execute(
        """
        INSERT INTO stocks (code, name, suspended)
        VALUES (?, ?, ?)
        ON CONFLICT(code) DO UPDATE SET
            name = excluded.name,
            suspended = excluded.suspended
        """,
        (code, name or "", int(bool(suspended))),
    )


def _upsert_fundamentals_snapshot(conn: sqlite3.Connection, code: str,
                                  trade_date: str, snapshot: dict,
                                  source: str = "clist",
                                  batch_id: str = ""):
    """Store daily valuation/market-cap snapshot."""
    conn.execute(
        """
        INSERT OR REPLACE INTO stock_fundamentals
        (code, trade_date, pe_ttm, pb, total_mv, float_mv,
         updated_at, source, batch_id)
        SELECT ?,?,?,?,?,?,?,?,?
        WHERE EXISTS (SELECT 1 FROM stocks WHERE code = ?)
        """,
        (
            code,
            trade_date,
            snapshot.get("pe_ttm"),
            snapshot.get("pb"),
            snapshot.get("total_mv"),
            snapshot.get("float_mv"),
            snapshot.get("updated_at"),
            source,
            batch_id,
            code,
        ),
    )



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


def _parse_and_store(conn: sqlite3.Connection, records) -> int:
    """解析东财返回的 diff 数据，存入数据库。返回入库条数。"""
    if isinstance(records, dict):
        records = list(records.values())
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

        _upsert_stock_meta(
            conn,
            code=code,
            name=r.get("f14", ""),
            suspended=0,
        )
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

        stored = _parse_and_store(conn, diff)

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


def refresh_fundamentals_snapshot(interval: float = 10.0,
                                  batch_size: int = 200,
                                  target_trade_date: str = None,
                                  conn: sqlite3.Connection = None) -> dict:
    """
    Full refresh of the market overview snapshot from Tencent.

    Purpose:
      - keep stocks universe up to date
      - write one daily snapshot per stock into stock_fundamentals
      - resume from the last unfinished batch within the same trade date

    Returns:
        {"batches": int, "stored": int, "total_expected": int, "done": bool, "errors": list}
    """
    own_conn = conn is None
    if own_conn:
        conn = _get_db()

    try:
        target_trade_date = target_trade_date or datetime.now().strftime("%Y-%m-%d")
        rows = conn.execute(
            "SELECT code, name FROM stocks WHERE suspended = 0 ORDER BY code"
        ).fetchall()
        total_expected = len(rows)
        stored_total = 0
        errors = []
        batches = 0
        resumed = False
        already_complete = False
        resume_offset = 0

        if not rows:
            return {
                "batches": 0,
                "stored": 0,
                "total_expected": 0,
                "done": False,
                "resumed": False,
                "already_complete": False,
                "resume_offset": 0,
                "next_offset": 0,
                "errors": ["stocks 表为空，无法刷新 fundamentals"],
            }

        progress_trade_date = _meta_get(conn, "fund_trade_date", "")
        progress_done = _meta_get(conn, "fund_done", "0") == "1"
        progress_offset = int(_meta_get(conn, "fund_next_offset", "0") or "0")

        if progress_trade_date != target_trade_date:
            _meta_set(conn, "fund_trade_date", target_trade_date)
            _meta_set(conn, "fund_next_offset", "0")
            _meta_set(conn, "fund_done", "0")
            _meta_set(conn, "fund_done_at", "")
            progress_offset = 0
            progress_done = False

        resume_offset = max(0, min(progress_offset, total_expected))
        resumed = resume_offset > 0 and not progress_done

        if progress_done and progress_trade_date == target_trade_date:
            already_complete = True
            return {
                "batches": 0,
                "stored": 0,
                "total_expected": total_expected,
                "done": True,
                "resumed": False,
                "already_complete": True,
                "resume_offset": total_expected,
                "next_offset": total_expected,
                "errors": [],
            }

        for start in range(resume_offset, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            symbols = [_to_tencent_symbol(code) for code, _ in batch]
            try:
                text = _fetch_tencent_batch(symbols)
                line_count = 0
                for line in text.splitlines():
                    item = _parse_tencent_quote_line(line)
                    if not item:
                        continue
                    _upsert_stock_meta(
                        conn,
                        code=item["code"],
                        name=item["name"],
                        suspended=0,
                    )
                    _upsert_fundamentals_snapshot(
                        conn,
                        code=item["code"],
                        trade_date=target_trade_date,
                        snapshot=item,
                        source=item.get("source", "tencent"),
                        batch_id=f"tencent_{target_trade_date}",
                    )
                    stored_total += 1
                    line_count += 1
                batches += 1
                next_offset = min(start + batch_size, total_expected)
                _meta_set(conn, "fund_trade_date", target_trade_date)
                _meta_set(conn, "fund_next_offset", str(next_offset))
                _meta_set(conn, "fund_last_fetch_at", datetime.now().isoformat())
                log.info(
                    "Fundamentals(Tencent): 批次 %d 写入 %d 条 (累计 %d/%d, 进度 %d/%d)",
                    (start // batch_size) + 1, line_count, stored_total, total_expected,
                    next_offset, total_expected,
                )
            except Exception as e:
                batch_no = (start // batch_size) + 1
                errors.append(f"batch {batch_no}: {e}")
                log.error("刷新 fundamentals 失败: batch %d: %s", batch_no, e)
                break
            if start + batch_size < len(rows) and interval > 0:
                time.sleep(interval)

        done = not errors and (resume_offset + batches * batch_size) >= total_expected
        if done:
            _meta_set(conn, "fund_done", "1")
            _meta_set(conn, "fund_done_at", datetime.now().isoformat())
            _meta_set(conn, "fund_next_offset", str(total_expected))
        else:
            _meta_set(conn, "fund_done", "0")

        return {
            "batches": batches,
            "stored": stored_total,
            "total_expected": total_expected,
            "done": done,
            "resumed": resumed,
            "already_complete": already_complete,
            "resume_offset": resume_offset,
            "next_offset": int(_meta_get(conn, "fund_next_offset", "0") or "0"),
            "errors": errors,
        }
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
                     "done", "done_at", "last_fetch_at",
                     "fund_trade_date", "fund_next_offset", "fund_done",
                     "fund_done_at", "fund_last_fetch_at"]:
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

def _stock_overview_query(updated_at_expr: str, legacy_select: str = "",
                          where_clause: str = "") -> str:
    """Build the shared stock overview SQL."""
    return f"""
        WITH last_trade AS (
            SELECT code, MAX(date) AS last_date
            FROM stock_history
            WHERE close IS NOT NULL
            GROUP BY code
        ),
        last_fund AS (
            SELECT code, MAX(trade_date) AS trade_date
            FROM stock_fundamentals
            GROUP BY code
        ),
        fundamentals AS (
            SELECT f.code, f.trade_date, f.pe_ttm, f.pb, f.total_mv,
                   f.float_mv, f.updated_at
            FROM stock_fundamentals f
            JOIN last_fund lf
              ON f.code = lf.code AND f.trade_date = lf.trade_date
        ),
        latest AS (
            SELECT h.code,
                   h.date AS updated_at_history,
                   h.close AS current,
                   h.pct_chg,
                   h.change,
                   h.volume,
                   h.amount,
                   h.amplitude,
                   h.high,
                   h.low,
                   h.open,
                   (
                     SELECT h2.close
                     FROM stock_history h2
                     WHERE h2.code = h.code
                       AND h2.date < h.date
                       AND h2.close IS NOT NULL
                     ORDER BY h2.date DESC
                     LIMIT 1
                   ) AS prev_close,
                   h.turnover AS turnover_rate
            FROM stock_history h
            JOIN last_trade lt
              ON h.code = lt.code AND h.date = lt.last_date
        ),
        hist AS (
            SELECT code,
                   COUNT(*) AS history_days,
                   AVG(CASE WHEN rn <= 5 THEN volume END) AS vol_5,
                   AVG(CASE WHEN rn <= 20 THEN volume END) AS vol_20,
                   MAX(CASE WHEN rn = 1 THEN close END) AS last_close,
                   MAX(CASE WHEN rn = 61 THEN close END) AS close_60d_ago,
                   MAX(CASE WHEN rn = 252 THEN close END) AS close_ytd_anchor
            FROM (
                SELECT code, date, close, volume,
                       ROW_NUMBER() OVER (PARTITION BY code ORDER BY date DESC) AS rn
                FROM stock_history
                WHERE close IS NOT NULL
            ) t
            GROUP BY code
        )
        SELECT s.code,
               s.name,
               s.suspended,
               l.current,
               l.pct_chg,
               l.change,
               l.volume,
               l.amount,
               l.amplitude,
               l.high,
               l.low,
               l.open,
               l.prev_close,
               CASE
                 WHEN hist.vol_20 IS NOT NULL AND hist.vol_20 > 0
                 THEN hist.vol_5 / hist.vol_20
               END AS volume_ratio,
               l.turnover_rate,
               CASE
                 WHEN hist.close_60d_ago IS NOT NULL AND hist.close_60d_ago > 0
                 THEN (hist.last_close / hist.close_60d_ago - 1.0) * 100
               END AS chg_60d,
               CASE
                 WHEN hist.close_ytd_anchor IS NOT NULL AND hist.close_ytd_anchor > 0
                 THEN (hist.last_close / hist.close_ytd_anchor - 1.0) * 100
               END AS chg_ytd,
               f.pe_ttm,
               f.pb,
               f.total_mv,
               f.float_mv,
               hist.history_days AS snapshot_history_days,
               COALESCE(f.updated_at, l.updated_at_history, {updated_at_expr}) AS updated_at
               {legacy_select}
        FROM stocks s
        LEFT JOIN latest l ON s.code = l.code
        LEFT JOIN fundamentals f ON s.code = f.code
        LEFT JOIN hist ON s.code = hist.code
        {where_clause}
        ORDER BY s.code
    """

def load_stocks_from_db() -> pd.DataFrame:
    """
    从本地数据库读取股票总览。

    新范式下:
      - stocks 仅保留 code / name / suspended 等元信息
      - 日度行情全部以 stock_history 为准

    因此这里会基于 stock_history 生成“最近一日市场快照”，并与 stocks
    元信息表合并，返回与 stock_screener.fetch_all_stocks() 兼容的 DataFrame。
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"数据库不存在: {DB_PATH}，请先运行 slow_fetcher.py")

    conn = _get_db()
    try:
        query = _stock_overview_query("NULL", "")
        df = pd.read_sql_query(query, conn)
        numeric_cols = [
            "current", "pct_chg", "change", "volume", "amount", "amplitude",
            "high", "low", "open", "prev_close", "volume_ratio",
            "turnover_rate", "chg_60d", "chg_ytd", "snapshot_history_days",
            "pe_ttm", "pb", "total_mv", "float_mv",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.attrs["source"] = "local_db"
        df.attrs["db_path"] = str(DB_PATH)
        if "current" in df.columns:
            df = df[df["current"].notna() & (df["current"] > 0)].copy()
        return df
    finally:
        conn.close()


def export_to_csv(path: str):
    """导出数据库到 CSV。"""
    df = load_stocks_from_db()
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log.info("导出 %d 条 → %s", len(df), path)


def load_stock_overview(code: str) -> dict:
    """Load a single-stock overview directly from SQL."""
    if not DB_PATH.exists():
        return {}
    conn = _get_db()
    try:
        query = _stock_overview_query(
            "NULL",
            "",
            "WHERE s.code = ?",
        )
        df = pd.read_sql_query(query, conn, params=(str(code).strip(),))
        if df.empty:
            return {}
        numeric_cols = [
            "current", "pct_chg", "change", "volume", "amount", "amplitude",
            "high", "low", "open", "prev_close", "volume_ratio",
            "turnover_rate", "chg_60d", "chg_ytd", "snapshot_history_days",
            "pe_ttm", "pb", "total_mv", "float_mv",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.iloc[0].to_dict()
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════
# K 线历史拉取 (东财 push2his API)
# ══════════════════════════════════════════════════════════════

KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
TENCENT_QUOTE_URL = "https://qt.gtimg.cn/q="


def _get_secid(code: str) -> str:
    code = str(code).strip()
    if code.startswith(("60", "68", "51", "11")):
        return f"1.{code}"
    return f"0.{code}"


def _to_tencent_symbol(code: str) -> str:
    code = str(code).strip()
    if code.startswith(("60", "68", "51", "11")):
        return f"sh{code}"
    return f"sz{code}"


def _fetch_tencent_batch(symbols: list[str]) -> str:
    """Fetch batched real-time quotes from Tencent."""
    if not symbols:
        return ""
    try:
        from curl_cffi import requests
        use_cffi = True
    except ImportError:
        import requests
        use_cffi = False

    url = TENCENT_QUOTE_URL + ",".join(symbols)
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Referer": "https://gu.qq.com/",
        "Accept": "*/*",
    }
    NO_PROXY = {"http": None, "https": None}

    if use_cffi:
        resp = requests.get(url, headers=headers, timeout=20, impersonate="chrome")
        content = resp.content
    else:
        resp = requests.get(url, headers=headers, timeout=20, proxies=NO_PROXY)
        content = resp.content

    resp.raise_for_status()
    return content.decode("gbk", errors="ignore")


def _parse_tencent_quote_line(line: str) -> dict:
    """Parse one qt.gtimg.cn line into a fundamentals snapshot."""
    line = (line or "").strip()
    if not line or '="' not in line:
        return {}
    left, right = line.split('="', 1)
    payload = right.rstrip('";')
    fields = payload.split("~")
    if len(fields) < 47:
        return {}

    code = fields[2].strip()
    if not code:
        return {}

    timestamp = fields[30].strip()
    trade_date = None
    updated_at = None
    if len(timestamp) >= 8 and timestamp.isdigit():
        trade_date = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
        if len(timestamp) >= 14:
            updated_at = (
                f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} "
                f"{timestamp[8:10]}:{timestamp[10:12]}:{timestamp[12:14]}"
            )

    return {
        "code": code,
        "name": fields[1].strip(),
        "trade_date": trade_date,
        "pe_ttm": _safe_float(fields[39]),
        "pb": _safe_float(fields[46]),
        "total_mv": _safe_float(fields[45]),
        "float_mv": _safe_float(fields[44]),
        "updated_at": updated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "tencent",
    }


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
    target_trade_date: str = None,
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

        # 默认用库中已有的最新交易日作为"已是最新"的判断标准；
        # 日更模式下可显式传入目标交易日（如当天收盘日期）。
        latest_trading_day = target_trade_date or conn.execute(
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


def daily_close_update(batch_size: int = 50,
                       days: int = 365,
                       interval: float = 1.0,
                       snapshot_interval: float = 10.0,
                       target_trade_date: str = None,
                       max_rounds: int = 500) -> dict:
    """
    收盘后统一更新入口:
      1) 刷新股票宇宙和当日 stock_fundamentals 快照
      2) 增量追平 stock_history 到目标交易日

    target_trade_date:
      - 默认使用今天日期
      - 可手动指定 YYYY-MM-DD，便于补历史或重跑
    """
    target_trade_date = target_trade_date or datetime.now().strftime("%Y-%m-%d")
    conn = _get_db()
    try:
        snapshot_result = refresh_fundamentals_snapshot(
            interval=snapshot_interval,
            batch_size=batch_size,
            target_trade_date=target_trade_date,
            conn=conn,
        )

        history_rounds = 0
        history_fetched = 0
        history_errors = []
        no_progress_rounds = 0
        while True:
            result = fetch_history_batch(
                batch_size=batch_size,
                days=days,
                interval=interval,
                target_trade_date=target_trade_date,
                conn=conn,
            )
            history_rounds += 1
            fetched_now = result.get("fetched", 0)
            history_fetched += fetched_now
            history_errors.extend(result.get("errors", []))
            if fetched_now <= 0:
                no_progress_rounds += 1
            else:
                no_progress_rounds = 0
            if result.get("done"):
                history_remaining = 0
                break
            history_remaining = result.get("remaining", 0)
            if history_remaining <= 0:
                break
            if no_progress_rounds >= 3:
                history_errors.append(
                    "连续 3 轮未取得任何增量进展，已提前停止，避免死循环。"
                )
                break
            if history_rounds >= max_rounds:
                history_errors.append(
                    f"达到最大轮次 {max_rounds}，已提前停止。"
                )
                break
            log.info("收盘更新继续: 剩余 %d 只，等待 %.1fs", history_remaining, interval)
            if interval > 0:
                time.sleep(interval)

        _meta_set(conn, "daily_close_last_run", datetime.now().isoformat())
        _meta_set(conn, "daily_close_target_date", target_trade_date)

        return {
            "target_trade_date": target_trade_date,
            "snapshot": snapshot_result,
            "history": {
                "rounds": history_rounds,
                "fetched": history_fetched,
                "remaining": history_remaining,
                "errors": history_errors,
                "done": history_remaining == 0 and not history_errors,
            },
            "done": snapshot_result.get("done", False) and history_remaining == 0 and not history_errors,
        }
    finally:
        conn.close()


def intraday_update(snapshot_interval: float = 10.0,
                    batch_size: int = 200,
                    target_trade_date: str = None) -> dict:
    """
    盘中更新入口:
      - 刷新股票宇宙
      - 刷新 stock_fundamentals 当日盘中快照

    注意:
      - 不更新 stock_history
      - 适合盘中看估值/快照，不适合作为正式收盘K线入库
    """
    conn = _get_db()
    try:
        snapshot_result = refresh_fundamentals_snapshot(
            interval=snapshot_interval,
            batch_size=batch_size,
            target_trade_date=target_trade_date,
            conn=conn,
        )
        _meta_set(conn, "intraday_last_run", datetime.now().isoformat())
        return {
            "snapshot": snapshot_result,
            "done": snapshot_result.get("done", False),
        }
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

def _fmt_pct(v) -> str:
    return f"{v:+.1f}%" if v is not None else "N/A"


def _fmt_price(v) -> str:
    return f"{v:.2f}" if v is not None else "N/A"


def assess_market_regime() -> dict:
    """
    基于本地 stock_history 最近日快照评估当前市场环境。
    优先离线可用，不依赖外部接口。
    """
    try:
        conn = sqlite3.connect(str(DB_PATH))
        row = conn.execute(
            """
            WITH last_trade AS (
                SELECT code, MAX(date) AS last_date
                FROM stock_history
                WHERE close IS NOT NULL
                GROUP BY code
            ),
            latest AS (
                SELECT h.code, h.close, h.pct_chg, h.turnover
                FROM stock_history h
                JOIN last_trade lt
                  ON h.code = lt.code AND h.date = lt.last_date
            ),
            hist AS (
                SELECT code,
                       MAX(CASE WHEN rn = 1 THEN close END) AS close_now,
                       MAX(CASE WHEN rn = 61 THEN close END) AS close_60d_ago
                FROM (
                    SELECT code, date, close,
                           ROW_NUMBER() OVER (PARTITION BY code ORDER BY date DESC) AS rn
                    FROM stock_history
                    WHERE close IS NOT NULL
                ) t
                GROUP BY code
            )
            SELECT
                COUNT(*) AS total,
                AVG(CASE WHEN l.pct_chg > 0 THEN 1.0 ELSE 0.0 END) AS up_ratio,
                AVG(CASE WHEN l.pct_chg < 0 THEN 1.0 ELSE 0.0 END) AS down_ratio,
                AVG(CASE WHEN l.pct_chg >= 2 THEN 1.0 ELSE 0.0 END) AS strong_up_ratio,
                AVG(CASE WHEN l.pct_chg <= -2 THEN 1.0 ELSE 0.0 END) AS strong_down_ratio,
                AVG(l.pct_chg) AS avg_pct_chg,
                AVG(
                    CASE
                      WHEN hist.close_60d_ago IS NOT NULL AND hist.close_60d_ago > 0
                           AND hist.close_now > hist.close_60d_ago
                      THEN 1.0 ELSE 0.0
                    END
                ) AS trend_60d_ratio,
                AVG(l.turnover) AS avg_turnover
            FROM latest l
            LEFT JOIN hist ON l.code = hist.code
            """
        ).fetchone()
        conn.close()
    except Exception:
        return {
            "score": 0.0,
            "label": "neutral",
            "action_bias": "neutral",
            "summary": "市场环境数据不可用，按中性环境处理。",
        }

    if not row or not row[0]:
        return {
            "score": 0.0,
            "label": "neutral",
            "action_bias": "neutral",
            "summary": "本地市场快照为空，按中性环境处理。",
        }

    total, up_ratio, down_ratio, strong_up_ratio, strong_down_ratio, avg_pct_chg, trend_60d_ratio, avg_turnover = row
    up_ratio = up_ratio or 0.0
    down_ratio = down_ratio or 0.0
    strong_up_ratio = strong_up_ratio or 0.0
    strong_down_ratio = strong_down_ratio or 0.0
    avg_pct_chg = avg_pct_chg or 0.0
    trend_60d_ratio = trend_60d_ratio or 0.0
    avg_turnover = avg_turnover or 0.0

    breadth = up_ratio - down_ratio
    score = (
        breadth * 45
        + avg_pct_chg * 10
        + (trend_60d_ratio - 0.5) * 30
        + (strong_up_ratio - strong_down_ratio) * 35
    )
    score = max(-100.0, min(100.0, score))

    if score >= 25:
        label = "bullish"
        action_bias = "risk_on"
        summary = "市场偏强，允许顺势型买入信号正常触发。"
    elif score >= 5:
        label = "slightly_bullish"
        action_bias = "cautious_long"
        summary = "市场略偏强，可以做精选个股，但不宜激进追高。"
    elif score <= -25:
        label = "bearish"
        action_bias = "risk_off"
        summary = "市场明显偏弱，应优先防守，买入门槛需要显著提高。"
    elif score <= -5:
        label = "slightly_bearish"
        action_bias = "defensive"
        summary = "市场略偏弱，更适合等待而不是主动进攻。"
    else:
        label = "neutral"
        action_bias = "neutral"
        summary = "市场中性，个股信号需要靠自身质量取胜。"

    return {
        "total_stocks": int(total),
        "up_ratio": round(up_ratio * 100, 1),
        "down_ratio": round(down_ratio * 100, 1),
        "strong_up_ratio": round(strong_up_ratio * 100, 1),
        "strong_down_ratio": round(strong_down_ratio * 100, 1),
        "avg_pct_chg": round(avg_pct_chg, 2),
        "trend_60d_ratio": round(trend_60d_ratio * 100, 1),
        "avg_turnover": round(avg_turnover, 2),
        "score": round(score, 1),
        "label": label,
        "action_bias": action_bias,
        "summary": summary,
    }


def assess_event_risk(code: str) -> dict:
    """
    评估公告/财报事件对当前交易决策的影响。
    网络不可用时自动降级，不阻断主流程。
    """
    result = {
        "available": False,
        "score": 0.0,
        "label": "unknown",
        "summary": "公告/财报数据不可用，未纳入事件约束。",
        "recent_earnings": False,
        "high_uncertainty": False,
        "positive_flags": [],
        "negative_flags": [],
    }

    try:
        from stock_tool import get_announcements, get_financial_history
    except Exception:
        return result

    today = datetime.today().date()
    anns = []
    try:
        anns = get_announcements(code, page_size=20)
        result["available"] = True
    except Exception:
        return result

    recent_30 = []
    for ann in anns:
        if ann.get("date"):
            try:
                ann_dt = datetime.strptime(ann["date"], "%Y-%m-%d").date()
                if (today - ann_dt).days <= 30:
                    recent_30.append(ann)
            except Exception:
                continue

    earnings_recent = [a for a in recent_30 if a.get("is_earnings")]
    result["recent_earnings"] = bool(earnings_recent)

    pos_keywords = ("回购", "增持", "中标", "分红", "预增", "扭亏", "业绩快报")
    neg_keywords = ("减持", "问询", "立案", "处罚", "亏损", "预减", "下修", "风险提示", "质押", "诉讼")

    score = 0.0
    positive_flags = []
    negative_flags = []

    for ann in recent_30[:10]:
        title = ann.get("title", "")
        if any(k in title for k in pos_keywords):
            score += 8
            positive_flags.append(title[:36])
        if any(k in title for k in neg_keywords):
            score -= 12
            negative_flags.append(title[:36])

    if earnings_recent:
        score -= 4
        result["high_uncertainty"] = True
        latest_title = earnings_recent[0].get("title", "")
        positive_flags.append(f"近期存在财报/业绩公告: {latest_title[:30]}")

        try:
            history = get_financial_history(code, periods=2)
            if history:
                latest = history[0]
                rev_yoy = latest.get("revenue_yoy")
                profit_yoy = latest.get("profit_yoy")
                roe = latest.get("roe")
                if profit_yoy is not None:
                    if profit_yoy >= 20:
                        score += 12
                        positive_flags.append(f"最新净利润同比 {profit_yoy:+.1f}%")
                        result["high_uncertainty"] = False
                    elif profit_yoy > 0:
                        score += 5
                        positive_flags.append(f"最新净利润同比 {profit_yoy:+.1f}%")
                    elif profit_yoy <= -20:
                        score -= 15
                        negative_flags.append(f"最新净利润同比 {profit_yoy:+.1f}%")
                    else:
                        score -= 6
                        negative_flags.append(f"最新净利润同比 {profit_yoy:+.1f}%")
                if rev_yoy is not None and rev_yoy < -10:
                    score -= 6
                    negative_flags.append(f"最新营收同比 {rev_yoy:+.1f}%")
                if roe is not None and roe >= 10:
                    score += 4
                    positive_flags.append(f"最新 ROE {roe:.1f}%")
        except Exception:
            pass

    score = max(-40.0, min(40.0, score))
    if score >= 10:
        label = "positive"
        summary = "近期公告/财报偏正面，对买入结论有一定支持。"
    elif score <= -10:
        label = "negative"
        summary = "近期公告/财报偏负面，应提高买入门槛或优先回避。"
    elif result["high_uncertainty"]:
        label = "uncertain"
        summary = "近期处于财报/事件窗口，信息仍在消化，适合更保守。"
    else:
        label = "neutral"
        summary = "近期公告/财报影响中性，未形成明显事件驱动。"

    result.update({
        "score": round(score, 1),
        "label": label,
        "summary": summary,
        "positive_flags": positive_flags[:5],
        "negative_flags": negative_flags[:5],
    })
    return result


def _build_trade_decision(
    history: pd.DataFrame,
    diag: dict,
    current: float,
    market_regime: dict | None = None,
    event_risk: dict | None = None,
    config: dict | None = None,
) -> dict:
    """
    将量化打分转换为更接近实盘的交易决策。
    只有条件足够充分时，才给出明确买入/卖出建议。
    """
    import risk_manager as rm

    close = history["close"].astype(float)
    ma20 = close.tail(min(20, len(close))).mean() if len(close) >= 20 else None
    ma60 = close.tail(min(60, len(close))).mean() if len(close) >= 60 else None
    ma120 = close.tail(min(120, len(close))).mean() if len(close) >= 120 else None

    trend = diag.get("trend_detail", {}) or {}
    momentum = diag.get("momentum_detail", {}) or {}
    volatility = diag.get("volatility_detail", {}) or {}
    mean_reversion = diag.get("mean_reversion_detail", {}) or {}
    volume = diag.get("volume_detail", {}) or {}
    support_resistance = diag.get("support_resistance_detail", {}) or {}
    statistics = diag.get("statistics_detail", {}) or {}
    mc = diag.get("monte_carlo") or {}
    market_regime = market_regime or {
        "score": 0.0,
        "label": "neutral",
        "action_bias": "neutral",
        "summary": "市场环境未评估，按中性处理。",
    }
    event_risk = event_risk or {
        "available": False,
        "score": 0.0,
        "label": "unknown",
        "summary": "公告/财报数据不可用，未纳入事件约束。",
        "recent_earnings": False,
        "high_uncertainty": False,
        "positive_flags": [],
        "negative_flags": [],
    }
    regime_score = market_regime.get("score", 0.0)
    regime_bias = market_regime.get("action_bias", "neutral")
    event_score = event_risk.get("score", 0.0)
    config = config or {}
    buy_score_min = config.get("buy_score_min", 18)
    sell_score_max = config.get("sell_score_max", -18)
    trend_buy_min = config.get("trend_buy_min", 15)
    trend_sell_max = config.get("trend_sell_max", -15)
    momentum_buy_min = config.get("momentum_buy_min", 5)
    momentum_sell_max = config.get("momentum_sell_max", -8)
    buy_prob_min = config.get("buy_prob_min", 55)
    sell_prob_min = config.get("sell_prob_min", 55)
    min_risk_reward = config.get("min_risk_reward", 1.8)
    expected_return_min = config.get("expected_return_min", 0.0)
    regime_buy_floor = config.get("regime_buy_floor", -5)
    regime_sell_cap = config.get("regime_sell_cap", -20)
    event_buy_floor = config.get("event_buy_floor", -5)
    event_sell_cap = config.get("event_sell_cap", -10)
    min_resistance_room = config.get("min_resistance_room", 4)
    min_support_gap_sell = config.get("min_support_gap_sell", 6)
    min_buy_checks = config.get("min_buy_checks", 7)
    min_sell_checks = config.get("min_sell_checks", 4)
    require_probabilistic_edge = config.get("require_probabilistic_edge", True)

    trend_score = trend.get("score", 0.0)
    momentum_score = momentum.get("score", 0.0)
    volume_score = volume.get("score", 0.0)
    total_score = diag.get("total_score", 0.0)
    prob_up = mc.get("prob_up")
    prob_down = mc.get("prob_down")
    expected_return = mc.get("expected_return")
    return_5th = mc.get("return_5th")
    risk_level = volatility.get("risk_level", "unknown")
    sr_details = support_resistance.get("details", {}) or {}
    dist_to_support = sr_details.get("dist_to_support_pct")
    dist_to_resistance = sr_details.get("dist_to_resistance_pct")
    nearest_support = sr_details.get("nearest_support")
    nearest_resistance = sr_details.get("nearest_resistance")

    var_95 = rm.calc_var(history, confidence=0.95, position_value=100000)
    stops = rm.calc_stop_levels(history, entry_price=current)
    kelly = rm.kelly_from_history(history)

    kelly_pct = None if "error" in kelly else kelly.get("kelly_adjusted_pct")
    stop_loss = (
        stops.get("support_stop_loss")
        or stops.get("atr_stop_loss")
        or stops.get("pct_5_stop_loss")
    )
    take_profit = (
        stops.get("support_take_profit")
        or stops.get("atr_take_profit")
        or stops.get("pct_5_take_profit")
    )

    stop_loss_pct = None
    take_profit_pct = None
    risk_reward = None
    if stop_loss is not None and current > stop_loss:
        stop_loss_pct = (current - stop_loss) / current * 100
    if take_profit is not None and take_profit > current:
        take_profit_pct = (take_profit - current) / current * 100
    if stop_loss is not None and take_profit is not None and current > stop_loss:
        risk_reward = (take_profit - current) / (current - stop_loss)

    if nearest_support is not None:
        entry_low = nearest_support
        entry_high = min(current, nearest_support * 1.03)
    elif ma20 is not None:
        entry_low = min(current, ma20 * 0.995)
        entry_high = max(current, ma20 * 1.015)
    else:
        entry_low = current * 0.99
        entry_high = current * 1.01

    buy_checks = []
    sell_checks = []

    def add_buy(ok: bool, ok_text: str, bad_text: str = None):
        buy_checks.append({"ok": ok, "text": ok_text if ok else (bad_text or ok_text)})

    def add_sell(ok: bool, ok_text: str, bad_text: str = None):
        sell_checks.append({"ok": ok, "text": ok_text if ok else (bad_text or ok_text)})

    add_buy(total_score >= buy_score_min,
            f"综合量化总分 {total_score:+.1f}，达到进攻阈值",
            f"综合量化总分仅 {total_score:+.1f}，进攻阈值不足")
    add_buy(trend_score >= trend_buy_min,
            f"趋势分 {trend_score:+.1f}，主趋势偏多",
            f"趋势分 {trend_score:+.1f}，主趋势仍不够强")
    add_buy(momentum_score >= momentum_buy_min,
            f"动量分 {momentum_score:+.1f}，上行动能未坏",
            f"动量分 {momentum_score:+.1f}，上行动能仍不足")
    add_buy(volume_score >= 0,
            f"量价分 {volume_score:+.1f}，没有明显派发",
            f"量价分 {volume_score:+.1f}，量价配合一般")
    add_buy(
        (not require_probabilistic_edge) or (prob_up is not None and prob_up >= buy_prob_min),
        f"20日上涨概率 {prob_up:.1f}% 达到偏多标准" if prob_up is not None else "已跳过上涨概率约束",
        (
            f"20日上涨概率仅 {prob_up:.1f}%"
            if prob_up is not None else
            ("校准模式下未使用上涨概率约束" if not require_probabilistic_edge else "缺少上涨概率数据")
        ),
    )
    add_buy(
        (not require_probabilistic_edge) or (expected_return is not None and expected_return > expected_return_min),
        f"20日期望收益 {expected_return:+.1f}% 为正" if expected_return is not None else "已跳过期望收益约束",
        (
            f"20日期望收益 {expected_return:+.1f}% 偏弱"
            if expected_return is not None else
            ("校准模式下未使用期望收益约束" if not require_probabilistic_edge else "缺少期望收益数据")
        ),
    )
    add_buy(risk_level not in ("extreme", "high"),
            f"波动风险为 {risk_level}，不是高波动博弈",
            f"波动风险为 {risk_level}，需要更保守")
    add_buy(risk_reward is not None and risk_reward >= min_risk_reward,
            f"预估盈亏比 1:{risk_reward:.2f}",
            f"预估盈亏比仅 1:{risk_reward:.2f}" if risk_reward is not None else "暂时无法计算盈亏比")
    add_buy(regime_score >= regime_buy_floor,
            f"市场环境分 {regime_score:+.1f}，系统性风险可控",
            f"市场环境分 {regime_score:+.1f}，系统性环境偏弱")
    add_buy(event_score >= event_buy_floor and not event_risk.get("high_uncertainty"),
            f"事件因子 {event_score:+.1f}，公告/财报没有明显拖累",
            f"事件因子 {event_score:+.1f}，近期公告/财报仍需谨慎")
    add_buy(
        dist_to_resistance is None or dist_to_resistance >= min_resistance_room,
        f"上方最近压力仍有 {dist_to_resistance:.1f}% 空间" if dist_to_resistance is not None
        else "上方压力位不密集",
        f"上方最近压力仅剩 {dist_to_resistance:.1f}% 空间" if dist_to_resistance is not None
        else "上方压力位未知",
    )

    add_sell(total_score <= sell_score_max and trend_score <= trend_sell_max,
             f"综合量化总分 {total_score:+.1f} 且趋势分 {trend_score:+.1f}，趋势破坏明显",
             f"总分 {total_score:+.1f} / 趋势分 {trend_score:+.1f}，尚未形成明确卖出破位")
    add_sell(momentum_score <= momentum_sell_max,
             f"动量分 {momentum_score:+.1f}，反弹动能不足",
             f"动量分 {momentum_score:+.1f}，尚未弱到必须卖")
    add_sell(volume_score <= -10,
             f"量价分 {volume_score:+.1f}，存在派发风险",
             f"量价分 {volume_score:+.1f}，未见强烈派发")
    add_sell(prob_down is not None and prob_down >= sell_prob_min,
             f"20日下跌概率 {prob_down:.1f}% 偏高" if prob_down is not None else "已触发下跌概率约束",
             f"20日下跌概率仅 {prob_down:.1f}%" if prob_down is not None else "缺少下跌概率数据")
    add_sell(expected_return is not None and expected_return < 0,
             f"20日期望收益 {expected_return:+.1f}% 为负" if expected_return is not None else "已触发收益为负约束",
             f"20日期望收益 {expected_return:+.1f}% 尚未转负" if expected_return is not None else "缺少期望收益数据")
    add_sell(return_5th is not None and return_5th <= -8,
             f"20日极端情景 5%分位收益 {return_5th:+.1f}%" if return_5th is not None else "已触发极端情景约束",
             f"20日极端情景 5%分位收益 {return_5th:+.1f}%" if return_5th is not None else "缺少极端情景数据")
    add_sell(risk_level == "extreme",
             f"波动风险为 {risk_level}，不适合继续硬扛",
             f"波动风险为 {risk_level}，尚未到强制卖出级别")
    add_sell(regime_score <= regime_sell_cap,
             f"市场环境分 {regime_score:+.1f}，弱市需要先保守",
             f"市场环境分 {regime_score:+.1f}，市场弱势尚未到极端防守")
    add_sell(event_score <= event_sell_cap or event_risk.get("high_uncertainty"),
             f"事件因子 {event_score:+.1f}，公告/财报风险偏高",
             f"事件因子 {event_score:+.1f}，事件风险尚未构成卖出")
    add_sell(
        dist_to_support is not None and dist_to_support >= min_support_gap_sell,
        f"距离最近支撑 {dist_to_support:.1f}%，回撤缓冲不足",
        f"距离最近支撑仅 {dist_to_support:.1f}%，暂未明显失守" if dist_to_support is not None else "支撑距离未知",
    )

    buy_passed = sum(1 for item in buy_checks if item["ok"])
    sell_passed = sum(1 for item in sell_checks if item["ok"])

    buy_core_ok = all([
        total_score >= buy_score_min,
        trend_score >= trend_buy_min,
        ((not require_probabilistic_edge) or (prob_up is not None and prob_up >= buy_prob_min)),
        ((not require_probabilistic_edge) or (expected_return is not None and expected_return > expected_return_min)),
        risk_level not in ("extreme", "high"),
        regime_score >= regime_buy_floor,
        event_score >= event_buy_floor,
        not event_risk.get("high_uncertainty"),
    ])
    sell_core_ok = (
        total_score <= sell_score_max and trend_score <= trend_sell_max and
        (((not require_probabilistic_edge) or (prob_down is not None and prob_down >= sell_prob_min))
         or (expected_return is not None and expected_return < 0)
         or momentum_score <= momentum_sell_max)
    )

    if buy_core_ok and buy_passed >= min_buy_checks:
        action = "buy"
        strength = "strong" if buy_passed >= 8 and (kelly_pct or 0) >= 15 else "medium"
        conclusion = "建议买入"
        reasons_for = [item["text"] for item in buy_checks if item["ok"]][:6]
        reasons_against = [item["text"] for item in buy_checks if not item["ok"]][:6]
        invalidation = (
            f"跌破止损位 {_fmt_price(stop_loss)}"
            if stop_loss is not None else "跌破近端支撑或动量转负"
        )
        why_not_opposite = "现在不是卖出，因为趋势、动量、盈亏比和上涨概率仍站在多头一侧。"
    elif sell_core_ok and sell_passed >= min_sell_checks:
        action = "sell"
        strength = "strong" if sell_passed >= 5 else "medium"
        conclusion = "建议卖出"
        reasons_for = [item["text"] for item in sell_checks if item["ok"]][:6]
        reasons_against = [item["text"] for item in sell_checks if not item["ok"]][:6]
        invalidation = (
            f"重新站回 {_fmt_price(ma20)} 上方并伴随量能改善"
            if ma20 is not None else "重新站回关键均线并放量"
        )
        why_not_opposite = "现在不是买入，因为趋势已偏弱，且收益期望与下行风险不匹配。"
    else:
        action = "hold"
        strength = "medium" if abs(total_score) >= 10 else "weak"
        conclusion = "建议观望"
        reasons_for = [item["text"] for item in buy_checks if item["ok"]][:6]
        reasons_against = [item["text"] for item in buy_checks if not item["ok"]][:6]
        invalidation = "等待趋势、动量、盈亏比三项至少同时转强后再考虑买入。"
        why_not_opposite = (
            "现在不是买入，因为关键条件还没完全凑齐；"
            "也不是卖出，因为暂未出现全面破位和高概率下跌共振。"
        )

    if action == "buy":
        if regime_score >= 20 and (kelly_pct or 0) >= 18 and risk_level == "low":
            position_advice = "中等偏重仓"
        elif (kelly_pct or 0) >= 10 and risk_level in ("low", "moderate"):
            position_advice = "半仓"
        else:
            position_advice = "轻仓试错"
    elif action == "sell":
        position_advice = "减仓或清仓"
    else:
        position_advice = "观察仓或空仓等待"

    if action == "buy":
        if trend_score >= 25 and momentum_score >= 10:
            holding_period = "波段偏中线（2-8周）"
        elif trend_score >= 15:
            holding_period = "波段（1-4周）"
        else:
            holding_period = "短线观察（3-10个交易日）"
    elif action == "sell":
        holding_period = "尽快执行，避免被动承受回撤"
    else:
        if trend_score > 10 and expected_return and expected_return > 0:
            holding_period = "短线跟踪，等待更好入场点"
        else:
            holding_period = "继续观察，暂不设持有计划"

    if action == "buy":
        if nearest_support is not None and current <= (nearest_support * 1.02):
            entry_style = "回踩支撑低吸"
        elif dist_to_resistance is not None and dist_to_resistance >= 6 and momentum_score > 10:
            entry_style = "顺势突破跟进"
        else:
            entry_style = "分批靠近入场区间布局"
    elif action == "sell":
        entry_style = "反弹减仓 / 破位卖出"
    else:
        if trend_score > 10:
            entry_style = "等回踩确认后再评估"
        else:
            entry_style = "不追价，先等待趋势明朗"

    return {
        "action": action,
        "strength": strength,
        "conclusion": conclusion,
        "confidence": diag.get("confidence", "moderate"),
        "position_advice": position_advice,
        "holding_period": holding_period,
        "entry_style": entry_style,
        "buy_checks_passed": buy_passed,
        "buy_checks_total": len(buy_checks),
        "sell_checks_passed": sell_passed,
        "sell_checks_total": len(sell_checks),
        "reasons_for": reasons_for,
        "reasons_against": reasons_against,
        "entry_low": round(entry_low, 2),
        "entry_high": round(entry_high, 2),
        "stop_loss": round(stop_loss, 2) if stop_loss is not None else None,
        "stop_loss_pct": round(stop_loss_pct, 2) if stop_loss_pct is not None else None,
        "take_profit": round(take_profit, 2) if take_profit is not None else None,
        "take_profit_pct": round(take_profit_pct, 2) if take_profit_pct is not None else None,
        "risk_reward": round(risk_reward, 2) if risk_reward is not None else None,
        "kelly_pct": kelly_pct,
        "var_95_pct": None if "error" in var_95 else var_95.get("var_pct"),
        "ma20": round(ma20, 2) if ma20 is not None else None,
        "ma60": round(ma60, 2) if ma60 is not None else None,
        "ma120": round(ma120, 2) if ma120 is not None else None,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "dist_to_support_pct": dist_to_support,
        "dist_to_resistance_pct": dist_to_resistance,
        "rsi_14": momentum.get("details", {}).get("rsi_14"),
        "zscore_20d": mean_reversion.get("details", {}).get("z_score_20d"),
        "position_in_range": sr_details.get("position_in_range"),
        "sharpe_ratio": statistics.get("details", {}).get("sharpe_ratio"),
        "risk_level": risk_level,
        "market_regime_score": regime_score,
        "market_regime_label": market_regime.get("label"),
        "market_action_bias": regime_bias,
        "event_score": event_score,
        "event_label": event_risk.get("label"),
        "invalidation": invalidation,
        "why_not_opposite": why_not_opposite,
    }


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

    # ── 3) 本地总览快照（由 stock_history 聚合） ──
    fundamentals = {}
    try:
        fundamentals = load_stock_overview(code)
    except Exception:
        pass

    name = realtime_info.get("name") or fundamentals.get("name") or code
    current = realtime_info.get("current") or float(history["close"].iloc[-1])
    pct_chg = realtime_info.get("pct_chg", 0)
    market_regime = assess_market_regime()
    event_risk = assess_event_risk(code)

    # ── 4) 量化诊断 ──
    diag = qe.comprehensive_diagnosis(history, run_monte_carlo=True, mc_days=20)

    mc = diag.get("monte_carlo") or {}
    td = diag.get("trend_detail", {})
    vd = diag.get("volatility_detail", {})
    md = diag.get("momentum_detail", {})
    mrd = diag.get("mean_reversion_detail", {})
    decision = _build_trade_decision(history, diag, current, market_regime, event_risk)

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
        "momentum": md.get("details", {}).get("rsi_14"),
        "mean_reversion": mrd.get("details", {}).get("z_score_20d"),
        "risk_level": vd.get("risk_level"),
        # 蒙特卡洛
        "prob_up": mc.get("prob_up"),
        "prob_down": mc.get("prob_down"),
        "expected_return": mc.get("expected_return"),
        # 交易决策
        "decision": decision,
        "market_regime": market_regime,
        "event_risk": event_risk,
        # 元信息
        "kline_source": kline_source,
        "history_bars": len(history),
        "risk_warnings": diag.get("risk_warnings", []),
        "summary": diag.get("summary", ""),
    }


def print_analysis(code: str, days: int = 120):
    """打印单只股票的完整分析报告。"""
    r = analyze_stock(code, days=days)
    d = r["decision"]
    mr = r["market_regime"]
    er = r["event_risk"]

    signal_colors = {
        "strong_buy": "🟢🟢", "buy": "🟢", "hold": "🟡",
        "sell": "🔴", "strong_sell": "🔴🔴",
    }
    sig = signal_colors.get(r["signal"], "⚪")
    action_icons = {"buy": "🟢", "hold": "🟡", "sell": "🔴"}
    action_sig = action_icons.get(d["action"], "⚪")

    print(f"""
{'='*60}
  {r['name']} ({r['code']})   {action_sig} {d['conclusion']} / {sig} {r['signal'].upper()}
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

  ── 市场环境 ──
  环境分:   {mr['score']:+.1f}  ({mr['label']}, {mr['action_bias']})
  广度:     上涨 {mr.get('up_ratio', 'N/A')}% / 下跌 {mr.get('down_ratio', 'N/A')}%
  均涨跌:   {mr.get('avg_pct_chg', 'N/A')}%
  说明:     {mr['summary']}

  ── 事件因子 ──
  事件分:   {er['score']:+.1f}  ({er['label']})
  说明:     {er['summary']}

  ── 交易决策 ──
  结论:     {d['conclusion']}  (强度: {d['strength']}, 置信度: {d['confidence']})
  买入条件: {d['buy_checks_passed']}/{d['buy_checks_total']} 满足
  仓位建议: {d['position_advice']}
  持有周期: {d['holding_period']}
  入场方式: {d['entry_style']}
  入场区间: {_fmt_price(d['entry_low'])} ~ {_fmt_price(d['entry_high'])}
  止损位:   {_fmt_price(d['stop_loss'])}  ({_fmt_pct(-d['stop_loss_pct']) if d['stop_loss_pct'] is not None else 'N/A'})
  目标位:   {_fmt_price(d['take_profit'])}  ({_fmt_pct(d['take_profit_pct']) if d['take_profit_pct'] is not None else 'N/A'})
  盈亏比:   {f"1:{d['risk_reward']:.2f}" if d['risk_reward'] is not None else 'N/A'}
  Kelly仓位:{f"{d['kelly_pct']:.1f}%" if d['kelly_pct'] is not None else 'N/A'}
  VaR(95):  {f"{d['var_95_pct']:.2f}%" if d['var_95_pct'] is not None else 'N/A'}
""")

    if r.get("prob_up") is not None:
        print(f"""  ── 蒙特卡洛模拟 (20日) ──
  上涨概率: {r['prob_up']:.0f}%
  下跌概率: {r['prob_down']:.0f}%
  预期收益: {r['expected_return']:+.1f}%
""")

    if d.get("reasons_for"):
        print("  ── 支持当前结论的理由 ──")
        for item in d["reasons_for"]:
            print(f"  + {item}")
        print()

    if d.get("reasons_against"):
        print("  ── 反对/待确认的因素 ──")
        for item in d["reasons_against"]:
            print(f"  - {item}")
        print()

    if er.get("positive_flags") or er.get("negative_flags"):
        print("  ── 公告/财报信号 ──")
        for item in er.get("positive_flags", [])[:3]:
            print(f"  + {item}")
        for item in er.get("negative_flags", [])[:3]:
            print(f"  - {item}")
        print()

    print("  ── 执行纪律 ──")
    print(f"  失效条件: {d['invalidation']}")
    print(f"  为什么不是相反结论: {d['why_not_opposite']}")
    print()

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
    parser.add_argument("--intraday-update", action="store_true",
                        help="盘中刷新股票宇宙和 fundamentals 快照")
    parser.add_argument("--daily-close-update", action="store_true",
                        help="收盘后统一更新: fundamentals + history")
    parser.add_argument("--trade-date", type=str,
                        help="指定目标交易日 YYYY-MM-DD (默认今天)")

    # 股票列表模式
    parser.add_argument("--pages", type=int, default=1,
                        help="已弃用: 在线拉取股票列表")
    parser.add_argument("--check-new", action="store_true",
                        help="已弃用: 在线检查新股")
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

    if args.intraday_update:
        result = intraday_update(
            snapshot_interval=max(float(args.interval), 10.0),
            batch_size=args.batch,
            target_trade_date=args.trade_date,
        )
        snap = result["snapshot"]
        progress_note = ""
        if snap.get("already_complete"):
            progress_note = "  Fundamentals: 当天快照已完整，无需重复刷新\n"
        elif snap.get("resumed"):
            progress_note = (
                f"  Fundamentals: 从 offset {snap.get('resume_offset', 0)} 继续，"
                f"本次写入 {snap['stored']} 条 / {snap['batches']} 批"
                f"{' (有错误)' if snap.get('errors') else ''}\n"
            )
        else:
            progress_note = (
                f"  Fundamentals: {snap['stored']} 条 / {snap['batches']} 批"
                f"{' (有错误)' if snap.get('errors') else ''}\n"
            )
        print(
            "\n盘中更新完成\n"
            f"{progress_note}"
        )
        if snap.get("errors"):
            print(f"  Fundamentals errors: {snap['errors'][:3]}")
        _print_status()
        return

    if args.daily_close_update:
        result = daily_close_update(
            batch_size=args.batch,
            days=args.days,
            interval=args.interval,
            snapshot_interval=max(float(args.interval), 10.0),
            target_trade_date=args.trade_date,
        )
        snap = result["snapshot"]
        hist = result["history"]
        if snap.get("already_complete"):
            fund_line = "  Fundamentals: 当天快照已完整，无需重复刷新"
        elif snap.get("resumed"):
            fund_line = (
                f"  Fundamentals: 从 offset {snap.get('resume_offset', 0)} 继续，"
                f"本次写入 {snap['stored']} 条 / {snap['batches']} 批"
                f"{' (有错误)' if snap.get('errors') else ''}"
            )
        else:
            fund_line = (
                f"  Fundamentals: {snap['stored']} 条 / {snap['batches']} 批"
                f"{' (有错误)' if snap.get('errors') else ''}"
            )
        print(
            "\n收盘更新完成\n"
            f"  目标交易日: {result['target_trade_date']}\n"
            f"{fund_line}\n"
            f"  History: {hist['fetched']} 只，轮次 {hist['rounds']}，剩余 {hist['remaining']}"
            f"{' (有错误)' if hist.get('errors') else ''}\n"
        )
        if snap.get("errors"):
            print(f"  Fundamentals errors: {snap['errors'][:3]}")
        if hist.get("errors"):
            print(f"  History errors: {hist['errors'][:5]}")
        _print_status()
        return

    if args.check_new:
        print("已停用在线 check-new。请手动保存东财 JSON 后运行 import_json.py 更新 stocks。")
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

    print("已停用在线 clist 拉取。请使用 import_json.py 导入 stocks 名单。")
    print("后续更新建议：")
    print("  1. python import_json.py")
    print("  2. python slow_fetcher.py --intraday-update")
    print("  3. python slow_fetcher.py --daily-close-update --batch 100 --interval 10")


if __name__ == "__main__":
    main()
