"""
Centralized PostgreSQL database connection management for A-share quantitative system.

Reads DATABASE_URL from .env file or environment variable.
Format: postgresql://user:pass@host:port/dbname?sslmode=require

Usage:
    from db import get_conn, execute, fetchone, fetchall, read_sql, upsert_sql

    # Basic usage
    conn = get_conn()
    cur = execute(conn, "SELECT * FROM stocks WHERE code = ?", ("600519",))
    row = cur.fetchone()
    conn.close()

    # Read into pandas
    df = read_sql("SELECT * FROM stock_history WHERE code = ?", conn, ("600519",))

    # Upsert data
    sql = upsert_sql("stocks", ["code", "name", "suspended"], ["code"])
    execute(conn, sql, ("600519", "贵州茅台", 0))
    conn.commit()
"""

import os
import sys
import psycopg2
import psycopg2.extras
import psycopg2.extensions
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)

DATABASE_URL = os.environ.get("DATABASE_URL", "")

# ── 全局类型适配器: PostgreSQL NUMERIC/DOUBLE → Python float ────
# psycopg2 默认把 DOUBLE PRECISION 返回为 float，但 NUMERIC 返回为 Decimal。
# 注册适配器后，所有数值列统一返回 float，避免 Decimal + float 运算报错。

def _cast_numeric_to_float(value, cur):
    if value is None:
        return None
    return float(value)

_NUMERIC = psycopg2.extensions.new_type((1700,), "NUMERIC_AS_FLOAT", _cast_numeric_to_float)
psycopg2.extensions.register_type(_NUMERIC)

# Connection pool for reuse
_shared_conn = None


def get_conn():
    """
    Get a new PostgreSQL connection. Caller is responsible for closing it.

    Returns:
        psycopg2 connection object with autocommit=False

    Raises:
        RuntimeError: if DATABASE_URL is not set
    """
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL not set. Create a .env file with "
            "DATABASE_URL=postgresql://user:pass@host:port/dbname?sslmode=require"
        )
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
    conn.autocommit = False
    # 单条查询最长 120 秒，防止大查询卡死 MCP server
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '120s'")
    cur.close()
    return conn


def get_shared_conn():
    """
    Get a shared connection (reusable, auto-reconnect on failure).

    Caller should NOT close this connection. It's managed globally.
    Use get_conn() if you need a connection you will close.

    Returns:
        psycopg2 connection object with autocommit=True
    """
    global _shared_conn
    if _shared_conn is None or _shared_conn.closed:
        _shared_conn = get_conn()
        _shared_conn.autocommit = True
    return _shared_conn


def execute(conn, sql, params=None):
    """
    Execute SQL with auto-conversion of SQLite ? placeholders to PostgreSQL %s.

    Args:
        conn: psycopg2 connection object
        sql (str): SQL statement with ? placeholders (SQLite style)
        params (tuple/list, optional): Query parameters

    Returns:
        psycopg2 cursor object with results available via fetchone/fetchall
    """
    sql = sql.replace("?", "%s")
    cur = conn.cursor()
    cur.execute(sql, params)
    return cur


def fetchone(conn, sql, params=None):
    """
    Execute SQL and fetch one row.

    Args:
        conn: psycopg2 connection object
        sql (str): SQL statement with ? placeholders
        params (tuple/list, optional): Query parameters

    Returns:
        tuple: One row or None if no results
    """
    cur = execute(conn, sql, params)
    return cur.fetchone()


def fetchall(conn, sql, params=None):
    """
    Execute SQL and fetch all rows.

    Args:
        conn: psycopg2 connection object
        sql (str): SQL statement with ? placeholders
        params (tuple/list, optional): Query parameters

    Returns:
        list: List of tuples, one per row
    """
    cur = execute(conn, sql, params)
    return cur.fetchall()


def read_sql(sql, conn, params=None):
    """
    Drop-in replacement for pd.read_sql_query with ? placeholder conversion.

    Automatically converts SQLite ? placeholders to PostgreSQL %s.

    Args:
        sql (str): SQL statement with ? placeholders
        conn: psycopg2 connection object
        params (tuple/list, optional): Query parameters

    Returns:
        pd.DataFrame: Query results as DataFrame
    """
    sql = sql.replace("?", "%s")
    df = pd.read_sql_query(sql, conn, params=params)
    # 强制把所有 Decimal/object 数值列转为 float，避免下游运算报错
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def upsert_sql(table: str, columns: list, conflict_keys: list) -> str:
    """
    Generate PostgreSQL UPSERT SQL from table, columns, and conflict keys.

    Converts SQLite INSERT OR REPLACE semantics to PostgreSQL INSERT ... ON CONFLICT.
    Non-conflict columns are updated on conflict; conflict key columns are left unchanged.

    Args:
        table (str): Table name
        columns (list): All column names in order (matches VALUES placeholders)
        conflict_keys (list): Columns that form the primary/unique key

    Returns:
        str: PostgreSQL UPSERT SQL with %s placeholders

    Example:
        >>> sql = upsert_sql("stocks", ["code", "name", "suspended"], ["code"])
        >>> print(sql)
        INSERT INTO stocks (code, name, suspended) VALUES (%s, %s, %s)
        ON CONFLICT (code) DO UPDATE SET name=EXCLUDED.name, suspended=EXCLUDED.suspended
    """
    placeholders = ", ".join(["%s"] * len(columns))
    cols = ", ".join(columns)
    update_cols = [c for c in columns if c not in conflict_keys]
    conflict = ", ".join(conflict_keys)

    sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"

    if update_cols:
        update_set = ", ".join(f"{c}=EXCLUDED.{c}" for c in update_cols)
        sql += f" ON CONFLICT ({conflict}) DO UPDATE SET {update_set}"
    else:
        sql += f" ON CONFLICT ({conflict}) DO NOTHING"

    return sql


def init_schema(conn=None):
    """
    Create all tables if they don't exist.

    Creates 5 tables:
    - stocks: Master stock list
    - stock_history: OHLCV data
    - stock_fundamentals: PE, PB, market cap
    - stock_moneyflow: Large order flows
    - meta: Configuration key-value store

    Args:
        conn (psycopg2 connection, optional): If None, creates a new connection
    """
    should_close = False
    if conn is None:
        conn = get_conn()
        should_close = True

    try:
        cur = conn.cursor()

        # stocks table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stocks (
                code TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                suspended INTEGER DEFAULT 0
            )
        """)

        # stock_history table - OHLCV data
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_history (
                code TEXT NOT NULL,
                date TEXT NOT NULL,
                "open" DOUBLE PRECISION,
                "close" DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                volume DOUBLE PRECISION,
                amount DOUBLE PRECISION,
                amplitude DOUBLE PRECISION,
                pct_chg DOUBLE PRECISION,
                "change" DOUBLE PRECISION,
                turnover DOUBLE PRECISION,
                PRIMARY KEY (code, date)
            )
        """)

        # Index for faster lookups by code and date range
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_stock_history_code
            ON stock_history (code)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_stock_history_date
            ON stock_history (date)
        """)

        # stock_fundamentals table - PE, PB, market cap
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_fundamentals (
                code TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                pe_ttm DOUBLE PRECISION,
                pb DOUBLE PRECISION,
                total_mv DOUBLE PRECISION,
                float_mv DOUBLE PRECISION,
                updated_at TEXT,
                source TEXT,
                batch_id TEXT,
                PRIMARY KEY (code, trade_date)
            )
        """)

        # Index for faster lookups by code
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_stock_fundamentals_code
            ON stock_fundamentals (code)
        """)

        # stock_moneyflow table - Large order flows
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_moneyflow (
                code TEXT NOT NULL,
                date TEXT NOT NULL,
                main_net DOUBLE PRECISION,
                main_net_pct DOUBLE PRECISION,
                super_net DOUBLE PRECISION,
                big_net DOUBLE PRECISION,
                mid_net DOUBLE PRECISION,
                small_net DOUBLE PRECISION,
                "close" DOUBLE PRECISION,
                pct_chg DOUBLE PRECISION,
                PRIMARY KEY (code, date)
            )
        """)

        # Index for faster lookups by code
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_stock_moneyflow_code
            ON stock_moneyflow (code)
        """)

        # fund_holdings table - 基金持仓（哪些基金持有某只股票）
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fund_holdings (
                code TEXT NOT NULL,
                report_date TEXT NOT NULL,
                fund_code TEXT NOT NULL,
                fund_name TEXT,
                hold_shares DOUBLE PRECISION,
                hold_mv DOUBLE PRECISION,
                nav_pct DOUBLE PRECISION,
                float_pct DOUBLE PRECISION,
                updated_at TEXT,
                PRIMARY KEY (code, report_date, fund_code)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_fund_holdings_code
            ON fund_holdings (code)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_fund_holdings_report_date
            ON fund_holdings (report_date)
        """)

        # stock_top_holders table - 十大流通股东
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_top_holders (
                code TEXT NOT NULL,
                report_date TEXT NOT NULL,
                rank INTEGER NOT NULL,
                holder_name TEXT,
                holder_type TEXT,
                hold_shares DOUBLE PRECISION,
                hold_pct DOUBLE PRECISION,
                change_shares DOUBLE PRECISION,
                change_type TEXT,
                updated_at TEXT,
                PRIMARY KEY (code, report_date, rank)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_stock_top_holders_code
            ON stock_top_holders (code)
        """)

        # stock_lhb_stat table - 龙虎榜近期统计（近实时机构动向）
        # 数据源: ak.stock_lhb_stock_statistic_em(symbol="近一月"/"近三月"/...)
        # 更新频率: 每交易日收盘后拉一次, 一次覆盖全市场
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_lhb_stat (
                code TEXT NOT NULL,
                period TEXT NOT NULL,
                last_lhb_date TEXT,
                lhb_count INTEGER,
                lhb_net_amt DOUBLE PRECISION,
                inst_buy_count INTEGER,
                inst_sell_count INTEGER,
                inst_net_amt DOUBLE PRECISION,
                inst_buy_total DOUBLE PRECISION,
                inst_sell_total DOUBLE PRECISION,
                updated_at TEXT,
                PRIMARY KEY (code, period)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_stock_lhb_stat_code
            ON stock_lhb_stat (code)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_stock_lhb_stat_updated_at
            ON stock_lhb_stat (updated_at)
        """)

        # stock_dzjy_stat table - 大宗交易近期统计（近实时机构动向第三个维度）
        # 数据源: ak.stock_dzjy_hygtj(symbol="近一月"/"近三月"/...)
        # 溢价交易 = 机构愿意更高价接盘 (看多信号)
        # 折价交易 = 持有人愿意打折出货 (中性偏弱)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_dzjy_stat (
                code TEXT NOT NULL,
                period TEXT NOT NULL,
                last_dzjy_date TEXT,
                total_count INTEGER,
                premium_count INTEGER,
                discount_count INTEGER,
                total_amt DOUBLE PRECISION,
                avg_premium_rate DOUBLE PRECISION,
                amt_to_float DOUBLE PRECISION,
                updated_at TEXT,
                PRIMARY KEY (code, period)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_stock_dzjy_stat_code
            ON stock_dzjy_stat (code)
        """)

        # meta table - Configuration key-value store
        cur.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error initializing schema: {e}")
        raise
    finally:
        if should_close:
            conn.close()


def migrate_from_sqlite(sqlite_path: str = None):
    """
    One-time migration: copy all data from local SQLite to PostgreSQL.

    Reads from data/stocks.db (or specified path) and inserts everything
    into PostgreSQL tables using batch inserts for performance.
    Prints progress for each table.

    Args:
        sqlite_path (str, optional): Path to SQLite database file.
                                     Defaults to data/stocks.db relative to this file.

    Raises:
        FileNotFoundError: If SQLite file doesn't exist
        RuntimeError: If DATABASE_URL not set
    """
    import sqlite3

    sqlite_path = sqlite_path or str(Path(__file__).parent / "data" / "stocks.db")

    if not Path(sqlite_path).exists():
        print(f"SQLite file not found: {sqlite_path}")
        return

    print(f"Starting migration from SQLite: {sqlite_path}")

    src = sqlite3.connect(sqlite_path)
    dst = get_conn()

    try:
        # Initialize PostgreSQL schema
        init_schema(dst)

        # Get list of tables from SQLite
        tables_to_migrate = ["stocks", "meta", "stock_history", "stock_fundamentals", "stock_moneyflow"]

        for table in tables_to_migrate:
            # Check if table exists in SQLite
            try:
                rows = src.execute(f"SELECT * FROM {table}").fetchall()
            except sqlite3.OperationalError:
                print(f"  {table}: table not found in SQLite, skip")
                continue

            if not rows:
                print(f"  {table}: 0 rows, skip")
                continue

            # Get column names from SQLite
            cols = [desc[0] for desc in src.execute(f"SELECT * FROM {table} LIMIT 1").description]

            # Determine conflict keys based on table
            if table == "stocks":
                conflict_keys = ["code"]
            elif table == "meta":
                conflict_keys = ["key"]
            elif table == "stock_history":
                conflict_keys = ["code", "date"]
            elif table == "stock_fundamentals":
                conflict_keys = ["code", "trade_date"]
            elif table == "stock_moneyflow":
                conflict_keys = ["code", "date"]
            else:
                conflict_keys = [cols[0]]  # Fallback to first column

            sql = upsert_sql(table, cols, conflict_keys)

            # Batch insert for performance
            batch_size = 1000
            total_inserted = 0

            for i in range(0, len(rows), batch_size):
                batch = rows[i:i+batch_size]
                cur = dst.cursor()
                try:
                    cur.executemany(sql, batch)
                    dst.commit()
                    total_inserted += len(batch)
                except Exception as e:
                    dst.rollback()
                    print(f"    Error inserting batch {i//batch_size}: {e}")
                    raise

            print(f"  {table}: {total_inserted} rows migrated")

        print("Migration complete!")
    except Exception as e:
        dst.rollback()
        print(f"Migration failed: {e}")
        raise
    finally:
        src.close()
        dst.close()


def health_check():
    """
    Check if PostgreSQL connection is working.

    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT version()")
        version = cur.fetchone()[0]
        conn.close()
        print(f"PostgreSQL connection OK: {version}")
        return True
    except Exception as e:
        print(f"PostgreSQL connection failed: {e}")
        return False


# CLI entry point
if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "migrate":
            # Optional: specify custom SQLite path
            sqlite_path = sys.argv[2] if len(sys.argv) > 2 else None
            migrate_from_sqlite(sqlite_path)
        elif command == "init":
            conn = get_conn()
            init_schema(conn)
            conn.close()
            print("Schema initialized")
        elif command == "health":
            health_check()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python db.py [migrate|init|health] [args]")
            sys.exit(1)
    else:
        # Test connection
        if health_check():
            sys.exit(0)
        else:
            sys.exit(1)
