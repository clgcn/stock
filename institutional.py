"""
机构持仓模块 (institutional) — 基金重仓 + 十大流通股东
=======================================================
提供:
  get_fund_holdings(code)        → 某只股票被哪些基金持有
  get_top_holders(code)          → 某只股票的十大流通股东
  fetch_fund_holdings_batch()    → 批量拉取全市场基金重仓数据
  fetch_top_holders_batch()      → 批量拉取全市场十大股东
  institutional_score(code)      → 机构共识度评分 (用于选股加分)
  format_institutional_report()  → 格式化报告

数据源: AKShare (新浪财经 + 东方财富)
  - 基金持仓: ak.stock_fund_stock_holder (新浪财经)
  - 十大流通股东: ak.stock_gdfx_free_top_10_em (东财个股页) → ak.stock_circulate_stock_holder (新浪) 回退
更新频率: 季度 (跟随基金季报/年报披露周期)
"""

import time
import logging
import db
from datetime import timedelta, date
from _http_utils import cn_now

_log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# AKShare 辅助
# ══════════════════════════════════════════════════════════════

def _ak_symbol(code: str) -> str:
    """转换为 AKShare stock_gdfx_free_top_10_em 所需的 'sh600519' / 'sz000001' 格式。"""
    code = str(code).strip().upper().replace("SH", "").replace("SZ", "")
    if code.startswith(("60", "68", "51", "11")):
        return f"sh{code}"
    return f"sz{code}"


def _quarter_date_str(report_date: str = None) -> str:
    """返回 AKShare 所需的季度日期字符串, 如 '20241231'。
    如果 report_date 为 None, 使用 _current_report_quarter()。"""
    if report_date is None:
        report_date = _current_report_quarter()  # e.g. "2024-12-31"
    return report_date.replace("-", "")


# ══════════════════════════════════════════════════════════════
# 1. 基金持仓 — 某只股票被哪些基金持有
# ══════════════════════════════════════════════════════════════

def get_fund_holdings(code: str, report_date: str = None, **_kwargs) -> dict:
    """
    通过 AKShare (新浪财经) 获取某只股票的基金持仓。

    数据源: ak.stock_fund_stock_holder → 新浪财经-股本股东-基金持股
    列: 基金名称, 基金代码, 持仓数量, 占流通股比例, 持股市值, 占净值比例, 截止日期

    Args:
        code: 股票代码 (如 "600519")
        report_date: 指定报告期 "YYYY-MM-DD"; None 时返回数据源最新一期

    Returns:
        {
            "report_date": "2025-12-31",
            "total_count": 245,
            "items": [
                {
                    "fund_code": "005827",
                    "fund_name": "易方达蓝筹精选混合",
                    "hold_shares": 12345678,
                    "hold_mv": 987654321.0,
                    "nav_pct": 5.23,
                    "float_pct": 0.12,
                }
            ]
        }
    """
    import akshare as ak

    code = str(code).strip()
    empty = {"report_date": "", "total_count": 0, "items": []}

    # akshare 对部分股票会返回空 DataFrame, 内部 rename 触发
    # "Length mismatch" / "None of [Index([...])] are in the [columns]"
    # 这类错误必须吞掉, 否则 MCP 链路会直接断
    try:
        df = ak.stock_fund_stock_holder(symbol=code)
    except Exception as e:
        _log.debug("ak.stock_fund_stock_holder(%s) 失败: %s", code, e)
        return empty

    if df is None or getattr(df, "empty", True):
        return empty

    if "截止日期" not in df.columns:
        _log.debug("基金持仓数据格式异常 (%s): 缺少 '截止日期' 列, 实际列=%s",
                   code, list(df.columns))
        return empty

    # 取最新一期 (截止日期最大), 或调用方指定的 report_date
    df["截止日期"] = df["截止日期"].astype(str)
    if report_date:
        target = report_date[:10]
        subset = df[df["截止日期"].str[:10] == target]
        if subset.empty:
            # 该股票在数据源里没有该季度数据, 返回空而不是回落到 max
            return empty
        latest = subset.copy()
        latest_date = target
    else:
        latest_date = df["截止日期"].max()
        latest = df[df["截止日期"] == latest_date].copy()

    # 转换日期格式: "2024-12-31" (可能是 datetime.date 或 str)
    report_date = str(latest_date)[:10]

    def _f(v):
        if isinstance(v, (dict, list, set)):
            return None
        try:
            return round(float(v), 4)
        except (TypeError, ValueError):
            return None

    def _s(v):
        if isinstance(v, (dict, list, set)):
            return str(v)
        return str(v) if v is not None else ""

    items = []
    for _, row in latest.iterrows():
        items.append({
            "fund_code": _s(row.get("基金代码")),
            "fund_name": _s(row.get("基金名称")),
            "hold_shares": _f(row.get("持仓数量")),
            "hold_mv": _f(row.get("持股市值")),
            "nav_pct": _f(row.get("占净值比例")),
            "float_pct": _f(row.get("占流通股比例")),
        })

    return {
        "report_date": report_date,
        "total_count": len(items),
        "items": items,
    }


# ══════════════════════════════════════════════════════════════
# 2. 十大流通股东
# ══════════════════════════════════════════════════════════════

_HOLDER_TYPE_KEYWORDS = {
    "fund": ["基金", "资产管理", "资管"],
    "social_security": ["社保", "社会保障"],
    "qfii": ["QFII", "合格境外"],
    "insurance": ["保险", "人寿", "太平", "泰康", "国寿"],
    "broker": ["证券金融", "证券", "期货"],
    "private": ["私募"],
    "connect": ["港股通", "陆股通", "HKSCC", "香港中央结算"],
}


def _classify_holder_type(name: str) -> str:
    """根据股东名称推断类型。"""
    if not name:
        return "unknown"
    name_upper = name.upper()
    for htype, keywords in _HOLDER_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw.upper() in name_upper:
                return htype
    return "individual"


def get_top_holders(code: str, report_date: str = None) -> dict:
    """
    通过 AKShare 获取某只股票的十大流通股东。

    主数据源: ak.stock_gdfx_free_top_10_em (东方财富个股页)
      列: 名次, 股东名称, 股东性质, 股份类型, 持股数, 占总流通股本持股比例, 增减, 变动比率
    回退: ak.stock_circulate_stock_holder (新浪财经)
      列: 截止日期, 公告日期, 编号, 股东名称, 持股数量(股), 占流通股比例(%), 股本性质

    Returns:
        {
            "report_date": "2025-12-31",
            "items": [
                {
                    "rank": 1,
                    "holder_name": "香港中央结算有限公司",
                    "holder_type": "connect",
                    "hold_shares": 123456789,
                    "hold_pct": 8.52,
                    "change_shares": 1234567,
                    "change_type": "增持",
                }
            ]
        }
    """
    import akshare as ak

    code = str(code).strip()

    def _f(v):
        try:
            return round(float(v), 4)
        except (TypeError, ValueError):
            return None

    empty_result = {"report_date": "", "items": []}

    # ── 主数据源: 东财个股页 ──
    try:
        sym = _ak_symbol(code)
        date_str = _quarter_date_str(report_date)
        try:
            df = ak.stock_gdfx_free_top_10_em(symbol=sym, date=date_str)
        except Exception as fetch_err:
            _log.debug("ak.stock_gdfx_free_top_10_em(%s, %s) 失败: %s",
                       sym, date_str, fetch_err)
            df = None

        if df is not None and not getattr(df, "empty", True):
            # 报告期就是请求的季度
            report_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

            items = []
            for _, row in df.iterrows():
                name = str(row.get("股东名称") or "")
                change_str = str(row.get("增减") or "")
                hold_shares = _f(row.get("持股数"))

                # 增减列: "新进" / "增持" / "减持" / "不变" 或具体数字
                if "新进" in change_str:
                    change_type = "新进"
                    change_shares = hold_shares  # 新进, 变动 = 全部
                elif "不变" in change_str:
                    change_type = "不变"
                    change_shares = 0
                else:
                    change_shares = _f(change_str)
                    if change_shares is not None:
                        if change_shares > 0:
                            change_type = "增持"
                        elif change_shares < 0:
                            change_type = "减持"
                        else:
                            change_type = "不变"
                    elif "增" in change_str:
                        change_type = "增持"
                        change_shares = None
                    elif "减" in change_str:
                        change_type = "减持"
                        change_shares = None
                    else:
                        change_type = "不变"
                        change_shares = 0

                items.append({
                    "rank": int(row.get("名次") or (len(items) + 1)),
                    "holder_name": name,
                    "holder_type": _classify_holder_type(name),
                    "hold_shares": hold_shares,
                    "hold_pct": _f(row.get("占总流通股本持股比例")),
                    "change_shares": change_shares,
                    "change_type": change_type,
                })

            if items:
                return {"report_date": report_date, "items": items}

    except Exception as e:
        _log.debug("东财十大流通股东失败 (%s): %s, 尝试新浪回退", code, e)

    # ── 回退: 新浪财经 ──
    try:
        df = ak.stock_circulate_stock_holder(symbol=code)
    except Exception as e:
        _log.debug("ak.stock_circulate_stock_holder(%s) 失败: %s", code, e)
        return empty_result

    if df is None or getattr(df, "empty", True):
        return empty_result

    if "截止日期" not in df.columns:
        _log.debug("十大流通股东数据格式异常 (%s): 缺少 '截止日期' 列, 实际列=%s",
                   code, list(df.columns))
        return empty_result

    # 取最新一期 (或调用方指定的 report_date)
    df["截止日期"] = df["截止日期"].astype(str)
    if report_date:
        target = report_date[:10]
        subset = df[df["截止日期"].str[:10] == target]
        if subset.empty:
            return empty_result
        latest = subset.copy()
        latest_date = target
    else:
        latest_date = df["截止日期"].max()
        latest = df[df["截止日期"] == latest_date].copy()
    report_date = str(latest_date)[:10]

    items = []
    for _, row in latest.iterrows():
        name = str(row.get("股东名称") or "")
        rank_val = row.get("编号")
        try:
            rank = int(rank_val)
        except (TypeError, ValueError):
            rank = len(items) + 1

        items.append({
            "rank": rank,
            "holder_name": name,
            "holder_type": _classify_holder_type(name),
            "hold_shares": _f(row.get("持股数量")),
            "hold_pct": _f(row.get("占流通股比例")),
            "change_shares": None,  # 新浪不提供变动数据
            "change_type": "不变",
        })

    return {
        "report_date": report_date,
        "items": items[:10],
    }


# ══════════════════════════════════════════════════════════════
# 3. 数据入库
# ══════════════════════════════════════════════════════════════

def _db_safe(v):
    """确保值可以被 psycopg2 适配。dict/list/set 等非标量转为字符串。"""
    if v is None:
        return None
    if isinstance(v, (dict, list, set, tuple)):
        import json
        return json.dumps(v, ensure_ascii=False)
    return v


def _get_stock_name(conn, code: str) -> str:
    """从 stocks 表查股票名, 查不到时 fallback 到 code 本身."""
    r = db.fetchone(conn, "SELECT name FROM stocks WHERE code=?", (code,))
    return r[0] if r else code


_QUARTER_ENDS = {"03-31", "06-30", "09-30", "12-31"}


def _is_quarter_end(report_date: str) -> bool:
    """是否合法季末日期 YYYY-(03-31|06-30|09-30|12-31)。"""
    if not report_date or not isinstance(report_date, str):
        return False
    rd = report_date.strip()[:10]
    if len(rd) != 10 or rd[4] != "-" or rd[7] != "-":
        return False
    return rd[5:] in _QUARTER_ENDS


def _snap_to_quarter_end(report_date: str) -> str:
    """把任意日期 snap 到"所属季度的末日"。
    A 股机构持仓按法定只在季末披露, 非季末日期的数据源通常是非标滚动披露,
    入库时统一归到所属季末, 防止脏日期混入。

    例:
      2026-03-23 → 2026-03-31  (Q1)
      2026-04-05 → 2026-06-30  (Q2)
      2026-07-15 → 2026-09-30  (Q3)
    """
    try:
        rd = report_date.strip()[:10]
        y = int(rd[:4])
        m = int(rd[5:7])
        if m <= 3:  return f"{y}-03-31"
        if m <= 6:  return f"{y}-06-30"
        if m <= 9:  return f"{y}-09-30"
        return f"{y}-12-31"
    except Exception:
        return ""


def _validate_report_date(report_date: str, table: str, code: str) -> str:
    """规范化 report_date:
       - 合法季末 → 原样返回
       - 非季末但可解析 → snap 到所属季末, 记 warning
       - 无法解析 → 返回 "", 调用方应跳过入库
    """
    if _is_quarter_end(report_date):
        return report_date.strip()[:10]
    snapped = _snap_to_quarter_end(report_date)
    if snapped and _is_quarter_end(snapped):
        _log.warning(
            "[%s] %s: report_date=%r 不是季末, snap 到 %s",
            table, code, report_date, snapped,
        )
        return snapped
    _log.warning(
        "[%s] %s: 无法解析 report_date=%r, 拒绝入库", table, code, report_date,
    )
    return ""


def store_fund_holdings(conn, code: str, holdings: dict) -> int:
    """将基金持仓数据写入 fund_holdings 表。返回入库条数。"""
    report_date = holdings.get("report_date", "")
    if not report_date:
        return 0
    # ── Guard: 非季末拒绝 / snap 到季末 ──
    report_date = _validate_report_date(report_date, "fund_holdings", code)
    if not report_date:
        return 0

    now_str = cn_now().strftime("%Y-%m-%d %H:%M:%S")
    sql = db.upsert_sql(
        "fund_holdings",
        ["code", "report_date", "fund_code", "fund_name",
         "hold_shares", "hold_mv", "nav_pct", "float_pct", "updated_at"],
        ["code", "report_date", "fund_code"],
    )
    count = 0
    for it in holdings.get("items", []):
        fund_code = it.get("fund_code", "")
        if not fund_code:
            continue
        db.execute(conn, sql, (
            code, report_date, _db_safe(fund_code),
            _db_safe(it.get("fund_name")),
            _db_safe(it.get("hold_shares")),
            _db_safe(it.get("hold_mv")),
            _db_safe(it.get("nav_pct")),
            _db_safe(it.get("float_pct")),
            now_str,
        ))
        count += 1
    conn.commit()
    return count


def store_top_holders(conn, code: str, holders: dict) -> int:
    """将十大流通股东数据写入 stock_top_holders 表。返回入库条数。"""
    report_date = holders.get("report_date", "")
    if not report_date:
        return 0
    # ── Guard: 新浪回退路径会返回非季末 (如 2026-03-23),snap 或拒绝 ──
    report_date = _validate_report_date(report_date, "stock_top_holders", code)
    if not report_date:
        return 0

    now_str = cn_now().strftime("%Y-%m-%d %H:%M:%S")
    sql = db.upsert_sql(
        "stock_top_holders",
        ["code", "report_date", "rank", "holder_name", "holder_type",
         "hold_shares", "hold_pct", "change_shares", "change_type", "updated_at"],
        ["code", "report_date", "rank"],
    )
    count = 0
    for it in holders.get("items", []):
        db.execute(conn, sql, (
            code, report_date, _db_safe(it.get("rank", 0)),
            _db_safe(it.get("holder_name")),
            _db_safe(it.get("holder_type")),
            _db_safe(it.get("hold_shares")),
            _db_safe(it.get("hold_pct")),
            _db_safe(it.get("change_shares")),
            _db_safe(it.get("change_type")),
            now_str,
        ))
        count += 1
    conn.commit()
    return count


# ══════════════════════════════════════════════════════════════
# 4. 批量拉取 (供 slow_fetcher 调用)
# ══════════════════════════════════════════════════════════════

def _refresh_interval_days() -> int:
    """季报披露高峰期 (1/4/7/10月) 每天刷新, 其余月份 7 天一次。"""
    month = cn_now().month
    if month in (1, 4, 7, 10):
        return 1
    return 7


def fetch_institutional_batch(
    batch_size: int = 20,
    interval: float = 2.0,
    conn=None,
) -> dict:
    """
    批量拉取机构持仓数据（基金重仓 + 十大股东），支持断点续传 + 定期刷新。

    更新策略:
      - 首轮: 拉取所有尚无当季数据的股票
      - 刷新: 已有数据但 updated_at > 7 天前的, 重新拉取 (捕捉增量披露)
      - 7 天内刚更新过的跳过

    Args:
        batch_size: 每批拉取股票数
        interval: 每只股票间隔秒数
        conn: 数据库连接

    Returns:
        {"fetched": int, "remaining": int, "done": bool, "errors": list}
    """
    own_conn = conn is None
    if own_conn:
        conn = db.get_conn()
        db.init_schema(conn)

    try:
        target_quarter = _current_report_quarter()
        refresh_days = _refresh_interval_days()
        stale_cutoff = (cn_now() - timedelta(days=refresh_days)).strftime("%Y-%m-%d %H:%M:%S")

        # 所有活跃非ST股票
        all_codes = [r[0] for r in db.fetchall(conn,
            "SELECT code FROM stocks "
            "WHERE suspended = 0 AND name NOT LIKE '%ST%' "
            "AND name NOT LIKE '%退%' "
            "ORDER BY code")]

        if not all_codes:
            return {"fetched": 0, "remaining": 0, "done": False,
                    "errors": ["stocks 表为空"]}

        # ── 判断每只股票的状态 ──
        # fund_holdings: 有当季数据 且 updated_at 在 7 天内 → 跳过
        fresh_fund = set(r[0] for r in db.fetchall(conn,
            "SELECT DISTINCT code FROM fund_holdings "
            "WHERE report_date >= ? AND updated_at >= ?",
            (target_quarter, stale_cutoff)))

        # stock_top_holders: 同理
        fresh_holders = set(r[0] for r in db.fetchall(conn,
            "SELECT DISTINCT code FROM stock_top_holders "
            "WHERE report_date >= ? AND updated_at >= ?",
            (target_quarter, stale_cutoff)))

        # 两个都新鲜的才跳过
        fresh_done = fresh_fund & fresh_holders
        remaining_codes = [c for c in all_codes if c not in fresh_done]

        if not remaining_codes:
            return {"fetched": 0, "remaining": 0, "done": True, "errors": []}

        batch = remaining_codes[:batch_size]
        fetched = 0
        errors = []

        for i, code in enumerate(batch):
            name = _get_stock_name(conn, code)

            fund_fresh = code in fresh_fund
            holders_fresh = code in fresh_holders

            # ── 基金持仓 ──
            if not fund_fresh:
                try:
                    holdings = get_fund_holdings(code)
                    stored = store_fund_holdings(conn, code, holdings)
                    _log.info("  [%d/%d] %s %s  基金持仓 +%d 条 (报告期 %s)",
                              i + 1, len(batch), code, name,
                              stored, holdings.get("report_date"))
                except Exception as e:
                    _log.warning("  [%d/%d] %s %s  基金持仓失败: %s",
                                 i + 1, len(batch), code, name, e)
                    errors.append(f"{code}: fund_holdings: {e}")

            # ── 十大股东 ──
            if not holders_fresh:
                try:
                    holders = get_top_holders(code)
                    stored = store_top_holders(conn, code, holders)
                    _log.info("  [%d/%d] %s %s  十大股东 +%d 条 (报告期 %s)",
                              i + 1, len(batch), code, name,
                              stored, holders.get("report_date"))
                except Exception as e:
                    _log.warning("  [%d/%d] %s %s  十大股东失败: %s",
                                 i + 1, len(batch), code, name, e)
                    errors.append(f"{code}: top_holders: {e}")

            fetched += 1
            if i < len(batch) - 1:
                time.sleep(interval)

        new_remaining = len(remaining_codes) - len(batch)
        done = new_remaining <= 0

        return {
            "fetched": fetched,
            "remaining": max(new_remaining, 0),
            "done": done,
            "errors": errors,
        }
    finally:
        if own_conn:
            conn.close()


def fetch_full_quarter(
    target_quarter: str,
    *,
    skip_if_exists: bool = True,
    batch_size: int = 100,
    interval: float = 1.0,
    do_funds: bool = True,
    do_top_holders: bool = True,
    conn=None,
    start_offset: int = 0,
    limit: int = 0,
) -> dict:
    """针对某一个"季末日期", 对全市场 3000 只股票做一次完整机构持仓回填。

    与 fetch_institutional_batch 的区别:
      - fetch_institutional_batch 是"滚动刷新"逻辑 (7 天过期重抓), 不保证一季度完整
      - fetch_full_quarter 是"一次性回填某季度", 遍历全部活跃股票, 每只抓一次,
        确保 stock_top_holders / fund_holdings 在该 quarter 的覆盖率接近 100%

    适用场景: 初始化 DB / 季报披露期结束后补齐 / 修复覆盖率低的历史数据

    Args:
        target_quarter:  目标季末 YYYY-MM-DD, 必须是 03-31/06-30/09-30/12-31
        skip_if_exists:  True=该 code 在该 quarter 已有数据则跳过 (断点续传),
                         False=强制重抓
        batch_size:      每批报 log 的数量
        interval:        每只股票之间 sleep 多少秒 (防封)
        do_funds/do_top_holders: 两种数据可单独开关
        start_offset:    从第几只开始 (CLI 中分片续跑)
        limit:           最多处理多少只 (0=不限)

    Returns: dict(total, processed, skipped, funds_stored, top_stored, errors)
    """
    if not _is_quarter_end(target_quarter):
        raise ValueError(f"target_quarter 必须是季末日期, 收到: {target_quarter!r}")

    own_conn = conn is None
    if own_conn:
        conn = db.get_conn()
        db.init_schema(conn)

    stats = {
        "target_quarter": target_quarter,
        "total": 0, "processed": 0, "skipped": 0,
        "funds_stored": 0, "top_stored": 0,
        "errors": [],
    }
    try:
        all_codes = [r[0] for r in db.fetchall(conn,
            "SELECT code FROM stocks "
            "WHERE suspended = 0 AND name NOT LIKE '%ST%' "
            "AND name NOT LIKE '%退%' "
            "ORDER BY code")]
        if start_offset > 0:
            all_codes = all_codes[start_offset:]
        if limit > 0:
            all_codes = all_codes[:limit]
        stats["total"] = len(all_codes)

        # 断点续传: 已有 target_quarter 数据的 code
        existing_funds = set()
        existing_top = set()
        if skip_if_exists:
            if do_funds:
                existing_funds = set(r[0] for r in db.fetchall(conn,
                    "SELECT DISTINCT code FROM fund_holdings WHERE report_date = ?",
                    (target_quarter,)))
            if do_top_holders:
                existing_top = set(r[0] for r in db.fetchall(conn,
                    "SELECT DISTINCT code FROM stock_top_holders WHERE report_date = ?",
                    (target_quarter,)))

        # 按目标季度逐只抓取; target_quarter 通过参数传入 get_*, 不再覆盖全局
        for i, code in enumerate(all_codes, 1):
            need_funds = do_funds and code not in existing_funds
            need_top = do_top_holders and code not in existing_top

            if not need_funds and not need_top:
                stats["skipped"] += 1
                continue

            try:
                if need_funds:
                    h = get_fund_holdings(code, report_date=target_quarter)
                    n = store_fund_holdings(conn, code, h)
                    stats["funds_stored"] += n
                if need_top:
                    h = get_top_holders(code, report_date=target_quarter)
                    n = store_top_holders(conn, code, h)
                    stats["top_stored"] += n
            except KeyboardInterrupt:
                raise  # 让上层处理 Ctrl-C
            except Exception as e:
                # 错误列表只保留前 200 条, 避免全市场全崩时爆内存
                if len(stats["errors"]) < 200:
                    stats["errors"].append(f"{code}: {type(e).__name__}: {e}")
                _log.warning("[%s] %s 抓取失败: %s", target_quarter, code, e)

            stats["processed"] += 1

            if stats["processed"] % batch_size == 0:
                _log.info(
                    "[%s] 进度 %d/%d processed=%d skipped=%d funds=+%d top=+%d errors=%d",
                    target_quarter, i, len(all_codes),
                    stats["processed"], stats["skipped"],
                    stats["funds_stored"], stats["top_stored"],
                    len(stats["errors"]),
                )

            if interval > 0 and i < len(all_codes):
                time.sleep(interval)

        return stats
    finally:
        if own_conn:
            conn.close()


def _current_report_quarter() -> str:
    """返回当前最近一次季报报告期 (YYYY-MM-DD)。

    基金季报披露规律:
      Q1 (03-31) → 4月下旬披露
      Q2 (06-30) → 7月下旬
      Q3 (09-30) → 10月下旬
      Q4 (12-31) → 次年1月下旬

    返回的是「已经能拿到数据的最新报告期」。
    """
    now = cn_now()
    year = now.year
    month = now.month

    if month >= 11:
        return f"{year}-09-30"     # Q3 已披露
    elif month >= 8:
        return f"{year}-06-30"     # Q2 已披露
    elif month >= 5:
        return f"{year}-03-31"     # Q1 已披露
    elif month >= 2:
        return f"{year - 1}-12-31"  # Q4 已披露
    else:
        return f"{year - 1}-09-30"  # Q3 已披露


# ══════════════════════════════════════════════════════════════
# 5. 机构共识度评分 (供 stock_screener 使用)
# ══════════════════════════════════════════════════════════════

def institutional_score(code: str, conn=None) -> dict:
    """
    计算某只股票的机构共识度评分。

    评分维度:
      - fund_count:     持仓基金数量 (0-40 分)
      - smart_money:    社保/QFII/险资/港股通 (0-30 分)
      - holder_change:  股东增减持动向 (0-30 分)

    Returns:
        {
            "score": float,         # 综合评分 0-100
            "fund_count": int,      # 持仓基金数
            "fund_score": float,    # 基金数量得分
            "smart_money_types": list,  # 包含的聪明资金类型
            "smart_money_score": float,
            "holder_change_score": float,
            "report_date": str,
            "detail": str,          # 一句话描述
        }
    """
    own_conn = conn is None
    if own_conn:
        conn = db.get_conn()

    try:
        target_q = _current_report_quarter()

        # ── 基金持仓数量 ──
        fund_row = db.fetchone(conn,
            "SELECT COUNT(*), MAX(report_date) FROM fund_holdings "
            "WHERE code = ? AND report_date >= ?",
            (code, target_q))
        fund_count = fund_row[0] if fund_row else 0
        report_date = fund_row[1] if fund_row else ""

        # 基金数量评分: ≥50只 满分40; 20只 20分; 5只以下 5分
        if fund_count >= 50:
            fund_score = 40.0
        elif fund_count >= 20:
            fund_score = 20.0 + (fund_count - 20) / 30 * 20
        elif fund_count >= 5:
            fund_score = 5.0 + (fund_count - 5) / 15 * 15
        elif fund_count >= 1:
            fund_score = fund_count * 1.0
        else:
            fund_score = 0.0

        # ── 聪明资金 (社保/QFII/险资/港股通) ──
        smart_types_found = []
        holders_row = db.fetchall(conn,
            "SELECT holder_name, holder_type, change_type FROM stock_top_holders "
            "WHERE code = ? AND report_date >= ? ORDER BY rank",
            (code, target_q))

        smart_money_types = {"social_security", "qfii", "insurance", "connect"}
        for _, htype, _ in holders_row:
            if htype in smart_money_types and htype not in smart_types_found:
                smart_types_found.append(htype)

        # 每种聪明资金 +7.5 分, 最多 30
        smart_money_score = min(len(smart_types_found) * 7.5, 30.0)

        # ── 增减持动向 ──
        # 增持/新进 每个 +5, 减持 每个 -3, 最终映射到 0-30
        change_raw = 0
        for _, _, change_type in holders_row:
            if change_type in ("新进", "增持"):
                change_raw += 5
            elif change_type == "减持":
                change_raw -= 3

        # 映射: raw ∈ [-30, +50] → score ∈ [0, 30]
        holder_change_score = max(0, min(30, 15 + change_raw))

        total_raw = round(fund_score + smart_money_score + holder_change_score, 1)

        # 时效衰减：机构持仓按季度披露，最长90天滞后，评分随时效衰减
        import datetime as _dt_inst
        lag_days = 0
        if report_date:
            try:
                rd = _dt_inst.datetime.strptime(str(report_date)[:10], "%Y-%m-%d")
                lag_days = max(0, (_dt_inst.datetime.now() - rd).days)
            except Exception:
                lag_days = 0
        time_decay = max(0.3, 1.0 - lag_days / 90.0)
        total = round(total_raw * time_decay, 1)

        # 一句话描述
        parts = []
        if fund_count > 0:
            parts.append(f"{fund_count}只基金持仓")
        if smart_types_found:
            type_names = {
                "social_security": "社保", "qfii": "QFII",
                "insurance": "险资", "connect": "港股通",
            }
            parts.append("+".join(type_names.get(t, t) for t in smart_types_found))
        increase_count = sum(1 for _, _, c in holders_row if c in ("新进", "增持"))
        decrease_count = sum(1 for _, _, c in holders_row if c == "减持")
        if increase_count or decrease_count:
            parts.append(f"增{increase_count}减{decrease_count}")

        detail = ", ".join(parts) if parts else "无机构数据"

        return {
            "score": total,
            "fund_count": fund_count,
            "fund_score": round(fund_score, 1),
            "smart_money_types": smart_types_found,
            "smart_money_score": round(smart_money_score, 1),
            "holder_change_score": round(holder_change_score, 1),
            "report_date": report_date or "",
            "detail": detail,
        }

    finally:
        if own_conn:
            conn.close()


# ══════════════════════════════════════════════════════════════
# 6. 格式化报告
# ══════════════════════════════════════════════════════════════

def format_institutional_report(code: str, conn=None) -> str:
    """生成某只股票的完整机构持仓报告。"""
    own_conn = conn is None
    if own_conn:
        conn = db.get_conn()

    try:
        name = _get_stock_name(conn, code)

        score_data = institutional_score(code, conn)

        # ── 新鲜度元数据 ──
        # A 股机构持仓法定只在季末披露, 且 fund_holdings (T+15) 与
        # stock_top_holders (年报/半年报 T+60/90) 披露节奏不同, 所以
        # 两张表的"最新报告期"可能错开一个季度. 这里分别给出.
        _ALLOWED_TABLES = ("stock_top_holders", "fund_holdings")

        def _max_rd(table: str) -> tuple:
            """返回 (max_report_date, lag_days) for a code in given table.

            注意: table 必须是白名单内的表名, f-string 拼接只为兼容 SQLite
            (不接受 ? 占位符绑定表名). 白名单防御 SQL 注入.
            """
            if table not in _ALLOWED_TABLES:
                raise ValueError(f"table must be one of {_ALLOWED_TABLES}")
            r = db.fetchone(conn,
                f"SELECT MAX(report_date) FROM {table} WHERE code = ?", (code,))
            rd = r[0] if r and r[0] else None
            if not rd:
                return (None, None)
            rd_str = rd if isinstance(rd, str) else str(rd)
            try:
                lag = (cn_now().date() - date.fromisoformat(rd_str[:10])).days
            except Exception:
                lag = None
            return (rd_str[:10], lag)

        top_rd, top_lag = _max_rd("stock_top_holders")
        fund_rd, fund_lag = _max_rd("fund_holdings")

        # 全市场同期覆盖度 (用各表自己的 max 日期)
        def _full_cov(table: str, rd: str) -> int:
            if table not in _ALLOWED_TABLES:
                raise ValueError(f"table must be one of {_ALLOWED_TABLES}")
            if not rd:
                return 0
            r = db.fetchone(conn,
                f"SELECT COUNT(DISTINCT code) FROM {table} WHERE report_date = ?",
                (rd,))
            return r[0] if r else 0

        cov_top = _full_cov("stock_top_holders", top_rd)
        cov_fund = _full_cov("fund_holdings", fund_rd)

        # "数据多老了" 用两者更大的 lag (即更老的那份数据) 做保守判断,
        # 这样只要任一表 > 90 天, 就提示调仓风险.
        max_lag_days = max(
            (l for l in (top_lag, fund_lag) if l is not None),
            default=None,
        )

        def _fmt_one(label: str, rd: str, lag: int, cov: int) -> str:
            if not rd:
                return f"{label}: (无数据)"
            s = f"{label}: {rd}"
            if lag is not None:
                s += f" (距今 {lag}d)"
            if cov:
                s += f" | 全市场覆盖 {cov}只"
            return s

        lines = [
            f"机构持仓报告: {code} {name}",
            _fmt_one("十大股东", top_rd, top_lag, cov_top),
            _fmt_one("基金持仓", fund_rd, fund_lag, cov_fund),
            "=" * 55,
            "",
            f"机构共识度评分: {score_data['score']}/100",
            f"  基金持仓 ({score_data['fund_count']}只): {score_data['fund_score']}/40",
            f"  聪明资金: {score_data['smart_money_score']}/30",
            f"  增减持动向: {score_data['holder_change_score']}/30",
            "",
        ]

        # ── 新鲜度提示 ──
        # 1) >90 天: 强警告 (期间可能大幅调仓)
        # 2) top_holders 覆盖 <1000: 提示跑 bulk-sync
        if max_lag_days is not None and max_lag_days > 90:
            lines.append(
                f"⚠️  数据距今 {max_lag_days} 天. A 股机构持仓按季度披露, "
                f"期间机构可能已大幅调仓. 要看当前动向请用 "
                f"institutional_realtime / northbound_flow / moneyflow."
            )
            lines.append("")
        elif top_rd and cov_top < 1000:
            lines.append(
                f"ℹ️  数据库中 {top_rd} 的十大股东覆盖仅 {cov_top}只 / ~3000. "
                f"如数据缺失, 运行: python sync_institutional_full.py --quarter {top_rd}"
            )
            lines.append("")

        # ── 十大流通股东 ──
        # 用 top_rd (该只股票该表的实际最新 report_date) 精确匹配,
        # 避免 target_q 是"预期季度"但该股票数据还在上一季度时查不到.
        holders = db.fetchall(conn,
            "SELECT rank, holder_name, holder_type, hold_pct, "
            "change_shares, change_type "
            "FROM stock_top_holders WHERE code = ? AND report_date = ? "
            "ORDER BY rank",
            (code, top_rd)) if top_rd else []

        if holders:
            lines.append("十大流通股东:")
            lines.append(f"  {'排名':<4} {'股东名称':<24} {'类型':<8} "
                         f"{'占比':>6} {'变动':>8}")
            lines.append("-" * 55)
            type_names = {
                "fund": "基金", "social_security": "社保", "qfii": "QFII",
                "insurance": "险资", "broker": "券商", "private": "私募",
                "connect": "港股通", "individual": "个人", "unknown": "-",
            }
            for rank, hname, htype, pct, chg_shares, chg_type in holders:
                pct_str = f"{pct:.2f}%" if pct else "-"
                type_str = type_names.get(htype, htype or "-")
                chg_str = chg_type or "-"
                lines.append(
                    f"  {rank:<4} {hname[:22]:<24} {type_str:<8} "
                    f"{pct_str:>6} {chg_str:>8}"
                )
            lines.append("")

        # ── 前10大持仓基金 ──
        funds = db.fetchall(conn,
            "SELECT fund_name, nav_pct, hold_mv "
            "FROM fund_holdings WHERE code = ? AND report_date = ? "
            "ORDER BY hold_mv DESC NULLS LAST LIMIT 10",
            (code, fund_rd)) if fund_rd else []

        if funds:
            lines.append(f"前10大持仓基金 (共{score_data['fund_count']}只):")
            lines.append(f"  {'基金名称':<28} {'占净值':>6} {'持仓市值':>12}")
            lines.append("-" * 55)
            for fname, nav_pct, hold_mv in funds:
                nav_str = f"{nav_pct:.2f}%" if nav_pct else "-"
                mv_str = f"{hold_mv / 1e8:.2f}亿" if hold_mv else "-"
                lines.append(f"  {fname[:26]:<28} {nav_str:>6} {mv_str:>12}")
            lines.append("")

        if not holders and not funds:
            lines.append("暂无当季机构持仓数据, 请先运行: python institutional.py --update")

        return "\n".join(lines)

    finally:
        if own_conn:
            conn.close()


# ══════════════════════════════════════════════════════════════
# 7. 近实时机构动向 (龙虎榜 + 主力资金)
#     — 不依赖季报披露, 日频滚动更新
# ══════════════════════════════════════════════════════════════

_LHB_PERIOD_DEFAULT = "近一月"
_LHB_PERIODS = ("近一月", "近三月", "近六月", "近一年")

_DZJY_PERIOD_DEFAULT = "近一月"
_DZJY_PERIODS = ("近一月", "近三月", "近六月", "近一年")


def fetch_lhb_snapshot(period: str = _LHB_PERIOD_DEFAULT) -> list:
    """
    一次性拉取全市场龙虎榜近期统计快照。

    数据源: ak.stock_lhb_stock_statistic_em
    列: 代码 / 名称 / 最近上榜日 / 上榜次数 / 龙虎榜净买额 /
        买方机构次数 / 卖方机构次数 / 机构买入净额 /
        机构买入总额 / 机构卖出总额 / ...

    Args:
        period: "近一月" / "近三月" / "近六月" / "近一年"

    Returns:
        list of dict, 每条对应一只上榜股票
    """
    import akshare as ak

    if period not in _LHB_PERIODS:
        period = _LHB_PERIOD_DEFAULT

    df = ak.stock_lhb_stock_statistic_em(symbol=period)
    if df is None or df.empty:
        return []

    def _f(v):
        try:
            return round(float(v), 2)
        except (TypeError, ValueError):
            return None

    def _i(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    rows = []
    for _, r in df.iterrows():
        code = str(r.get("代码") or "").strip().zfill(6)
        if not code or not code.isdigit():
            continue
        last_date = str(r.get("最近上榜日") or "")[:10]
        rows.append({
            "code": code,
            "period": period,
            "last_lhb_date": last_date,
            "lhb_count": _i(r.get("上榜次数")),
            "lhb_net_amt": _f(r.get("龙虎榜净买额")),
            "inst_buy_count": _i(r.get("买方机构次数")),
            "inst_sell_count": _i(r.get("卖方机构次数")),
            "inst_net_amt": _f(r.get("机构买入净额")),
            "inst_buy_total": _f(r.get("机构买入总额")),
            "inst_sell_total": _f(r.get("机构卖出总额")),
        })
    return rows


def store_lhb_snapshot(conn, rows: list) -> int:
    """把 fetch_lhb_snapshot 的结果写进 stock_lhb_stat 表。"""
    if not rows:
        return 0
    now_str = cn_now().strftime("%Y-%m-%d %H:%M:%S")
    sql = db.upsert_sql(
        "stock_lhb_stat",
        ["code", "period", "last_lhb_date", "lhb_count", "lhb_net_amt",
         "inst_buy_count", "inst_sell_count", "inst_net_amt",
         "inst_buy_total", "inst_sell_total", "updated_at"],
        ["code", "period"],
    )
    count = 0
    for r in rows:
        db.execute(conn, sql, (
            r["code"], r["period"], r.get("last_lhb_date"),
            _db_safe(r.get("lhb_count")),
            _db_safe(r.get("lhb_net_amt")),
            _db_safe(r.get("inst_buy_count")),
            _db_safe(r.get("inst_sell_count")),
            _db_safe(r.get("inst_net_amt")),
            _db_safe(r.get("inst_buy_total")),
            _db_safe(r.get("inst_sell_total")),
            now_str,
        ))
        count += 1
    conn.commit()
    return count


def refresh_lhb_stat(conn=None, period: str = _LHB_PERIOD_DEFAULT) -> dict:
    """拉一次全市场龙虎榜 + 落库。每日跑一次即可。"""
    own_conn = conn is None
    if own_conn:
        conn = db.get_conn()
        db.init_schema(conn)
    try:
        rows = fetch_lhb_snapshot(period)
        stored = store_lhb_snapshot(conn, rows)
        _log.info("龙虎榜快照 (%s): 抓取 %d 只 / 入库 %d 条",
                  period, len(rows), stored)
        return {"period": period, "fetched": len(rows), "stored": stored}
    finally:
        if own_conn:
            conn.close()


def _moneyflow_sub_score(conn, code: str, days: int = 10) -> tuple:
    """
    基于 stock_moneyflow 表(已入库, 日频)算主力资金动向分 (0-50)。

    评分:
      - 净流入分数 (0-25): 累计主力净流入, 映射 [-5亿, +5亿] → [0, 25]
      - 连续强度分 (0-25): 流入天数占比 * 25
    """
    rows = db.fetchall(
        conn,
        "SELECT date, main_net, main_net_pct FROM stock_moneyflow "
        "WHERE code = ? ORDER BY date DESC LIMIT ?",
        (code, days),
    )
    if not rows:
        return 0.0, {"available": False, "days_available": 0}

    nets = [float(r[1] or 0) for r in rows]
    net_sum = sum(nets)  # 单位: 亿元
    up_days = sum(1 for n in nets if n > 0)

    # 净流入分: 累计 +5 亿 → 25 分; -5 亿 → 0 分
    net_score = max(0.0, min(25.0, 12.5 + net_sum * 2.5))
    # 连续分: 流入天数占比
    up_score = (up_days / len(nets)) * 25.0

    return round(net_score + up_score, 1), {
        "available": True,
        "days_available": len(nets),
        "net_sum_yi": round(net_sum, 2),
        "up_days": up_days,
        "net_score": round(net_score, 1),
        "up_score": round(up_score, 1),
        "latest_date": str(rows[0][0]) if rows else "",
    }


def _lhb_sub_score(conn, code: str, period: str = _LHB_PERIOD_DEFAULT) -> tuple:
    """
    基于 stock_lhb_stat 表算龙虎榜机构席位动向分 (0-50)。

    评分:
      - 机构净买分 (0-30): inst_net_amt 映射 [-1亿, +1亿] → [0, 30]
      - 机构参与分 (0-20): 买方机构次数 × 2, 上限 20
    """
    row = db.fetchone(
        conn,
        "SELECT last_lhb_date, lhb_count, inst_buy_count, inst_sell_count, "
        "inst_net_amt, lhb_net_amt FROM stock_lhb_stat "
        "WHERE code = ? AND period = ?",
        (code, period),
    )
    if not row:
        return 0.0, {"available": False, "lhb_count": 0}

    last_date, lhb_count, inst_buy, inst_sell, inst_net, lhb_net = row
    inst_net_yi = float(inst_net or 0) / 1e8  # 元 → 亿
    lhb_net_yi = float(lhb_net or 0) / 1e8

    # 机构净买分: ±1 亿 → ±15 分
    net_score = max(0.0, min(30.0, 15.0 + inst_net_yi * 15.0))
    # 机构参与分: 买方机构次数 * 2, 最多 20
    participation = (inst_buy or 0) * 2.0
    participation_score = max(0.0, min(20.0, participation))

    return round(net_score + participation_score, 1), {
        "available": True,
        "period": period,
        "last_lhb_date": last_date or "",
        "lhb_count": int(lhb_count or 0),
        "inst_buy_count": int(inst_buy or 0),
        "inst_sell_count": int(inst_sell or 0),
        "inst_net_yi": round(inst_net_yi, 2),
        "lhb_net_yi": round(lhb_net_yi, 2),
        "net_score": round(net_score, 1),
        "participation_score": round(participation_score, 1),
    }


def fetch_dzjy_snapshot(period: str = _DZJY_PERIOD_DEFAULT) -> list:
    """
    一次性拉取全市场大宗交易近期统计快照。

    数据源: ak.stock_dzjy_hygtj
    列: 证券代码 / 证券简称 / 最近上榜日 /
        上榜次数-总计 / 上榜次数-溢价 / 上榜次数-折价 /
        总成交额(万) / 折溢率 / 成交总额/流通市值 / ...

    溢价 (premium) = 成交价高于当日收盘价 → 机构愿意溢价接盘 (看多)
    折价 (discount) = 成交价低于当日收盘价 → 持有人打折出货 (中性偏弱)
    """
    import akshare as ak

    if period not in _DZJY_PERIODS:
        period = _DZJY_PERIOD_DEFAULT

    df = ak.stock_dzjy_hygtj(symbol=period)
    if df is None or df.empty:
        return []

    def _f(v):
        try:
            return round(float(v), 4)
        except (TypeError, ValueError):
            return None

    def _i(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    rows = []
    for _, r in df.iterrows():
        code = str(r.get("证券代码") or "").strip().zfill(6)
        if not code or not code.isdigit():
            continue
        rows.append({
            "code": code,
            "period": period,
            "last_dzjy_date": str(r.get("最近上榜日") or "")[:10],
            "total_count": _i(r.get("上榜次数-总计")),
            "premium_count": _i(r.get("上榜次数-溢价")),
            "discount_count": _i(r.get("上榜次数-折价")),
            "total_amt": _f(r.get("总成交额")),       # 万元
            "avg_premium_rate": _f(r.get("折溢率")),   # 百分比
            "amt_to_float": _f(r.get("成交总额/流通市值")),
        })
    return rows


def store_dzjy_snapshot(conn, rows: list) -> int:
    """大宗交易快照入库 stock_dzjy_stat。"""
    if not rows:
        return 0
    now_str = cn_now().strftime("%Y-%m-%d %H:%M:%S")
    sql = db.upsert_sql(
        "stock_dzjy_stat",
        ["code", "period", "last_dzjy_date", "total_count",
         "premium_count", "discount_count", "total_amt",
         "avg_premium_rate", "amt_to_float", "updated_at"],
        ["code", "period"],
    )
    count = 0
    for r in rows:
        db.execute(conn, sql, (
            r["code"], r["period"], r.get("last_dzjy_date"),
            _db_safe(r.get("total_count")),
            _db_safe(r.get("premium_count")),
            _db_safe(r.get("discount_count")),
            _db_safe(r.get("total_amt")),
            _db_safe(r.get("avg_premium_rate")),
            _db_safe(r.get("amt_to_float")),
            now_str,
        ))
        count += 1
    conn.commit()
    return count


def refresh_dzjy_stat(conn=None, period: str = _DZJY_PERIOD_DEFAULT) -> dict:
    """拉一次全市场大宗交易 + 落库。"""
    own_conn = conn is None
    if own_conn:
        conn = db.get_conn()
        db.init_schema(conn)
    try:
        rows = fetch_dzjy_snapshot(period)
        stored = store_dzjy_snapshot(conn, rows)
        _log.info("大宗交易快照 (%s): 抓取 %d 只 / 入库 %d 条",
                  period, len(rows), stored)
        return {"period": period, "fetched": len(rows), "stored": stored}
    finally:
        if own_conn:
            conn.close()


def _dzjy_sub_score(conn, code: str, period: str = _DZJY_PERIOD_DEFAULT) -> tuple:
    """
    基于 stock_dzjy_stat 表算大宗交易动向分 (0-20)。

    评分:
      - 溢价分 (0-12): premium_count * 3, 上限 12  (溢价 4 次拉满)
      - 折价惩罚 (0-8): 从 8 开始, 每折价一次扣 1.5, 最低 0
    """
    row = db.fetchone(
        conn,
        "SELECT last_dzjy_date, total_count, premium_count, discount_count, "
        "total_amt, avg_premium_rate, amt_to_float FROM stock_dzjy_stat "
        "WHERE code = ? AND period = ?",
        (code, period),
    )
    if not row:
        return 0.0, {"available": False, "total_count": 0}

    last_date, total, premium, discount, amt, rate, amt2float = row
    total = int(total or 0)
    premium = int(premium or 0)
    discount = int(discount or 0)
    amt_yi = float(amt or 0) / 1e4  # 万 → 亿

    premium_score = min(premium * 3.0, 12.0)
    discount_penalty_score = max(0.0, 8.0 - discount * 1.5)

    return round(premium_score + discount_penalty_score, 1), {
        "available": True,
        "period": period,
        "last_dzjy_date": last_date or "",
        "total_count": total,
        "premium_count": premium,
        "discount_count": discount,
        "total_amt_yi": round(amt_yi, 2),
        "avg_premium_rate": float(rate or 0),
        "amt_to_float": float(amt2float or 0),
        "premium_score": round(premium_score, 1),
        "discount_penalty_score": round(discount_penalty_score, 1),
    }


def realtime_institutional_score(code: str, conn=None,
                                 moneyflow_days: int = 10,
                                 lhb_period: str = _LHB_PERIOD_DEFAULT,
                                 dzjy_period: str = _DZJY_PERIOD_DEFAULT) -> dict:
    """
    近实时机构动向评分 (0-100), 不依赖季报披露。

    三条腿:
      - 主力资金 (stock_moneyflow, 日频, 0-40)
      - 龙虎榜机构席位 (stock_lhb_stat, 日频快照, 0-40)
      - 大宗交易 (stock_dzjy_stat, 日频快照, 0-20)

    Args:
        code: 股票代码
        conn: DB 连接
        moneyflow_days: 主力资金统计天数 (默认 10)
        lhb_period: 龙虎榜统计周期 ("近一月" 默认)
        dzjy_period: 大宗交易统计周期 ("近一月" 默认)

    Returns:
        {
            "score": float,              # 综合 0-100
            "main_money_score": float,   # 0-40
            "lhb_score": float,          # 0-40
            "dzjy_score": float,         # 0-20
            "main_money": {...},
            "lhb": {...},
            "dzjy": {...},
            "detail": str,
        }
    """
    own_conn = conn is None
    if own_conn:
        conn = db.get_conn()
    try:
        # 子分函数原本返回 50 分制 / 50 分制 / 20 分制, 这里按新权重缩放
        mm_raw, mm_data = _moneyflow_sub_score(conn, code, moneyflow_days)
        lhb_raw, lhb_data = _lhb_sub_score(conn, code, lhb_period)
        dzjy_score, dzjy_data = _dzjy_sub_score(conn, code, dzjy_period)

        # 主力资金 50 → 40, 龙虎榜 50 → 40
        mm_score = round(mm_raw * 0.8, 1)
        lhb_score = round(lhb_raw * 0.8, 1)
        total = round(mm_score + lhb_score + dzjy_score, 1)

        parts = []
        if mm_data.get("available"):
            sign = "+" if mm_data["net_sum_yi"] >= 0 else ""
            parts.append(
                f"{mm_data['days_available']}日主力净流入"
                f"{sign}{mm_data['net_sum_yi']:.2f}亿"
                f"({mm_data['up_days']}/{mm_data['days_available']}日流入)"
            )
        if lhb_data.get("available"):
            sign = "+" if lhb_data["inst_net_yi"] >= 0 else ""
            parts.append(
                f"{lhb_period}上榜{lhb_data['lhb_count']}次, "
                f"机构席位净买{sign}{lhb_data['inst_net_yi']:.2f}亿"
            )
        if dzjy_data.get("available"):
            parts.append(
                f"{dzjy_period}大宗交易{dzjy_data['total_count']}次"
                f"(溢价{dzjy_data['premium_count']}/折价{dzjy_data['discount_count']}), "
                f"成交{dzjy_data['total_amt_yi']:.2f}亿"
            )

        detail = "; ".join(parts) if parts else \
            "近期无机构异动数据(主力资金/龙虎榜/大宗交易均无记录)"

        return {
            "score": total,
            "main_money_score": mm_score,
            "lhb_score": lhb_score,
            "dzjy_score": dzjy_score,
            "main_money": mm_data,
            "lhb": lhb_data,
            "dzjy": dzjy_data,
            "detail": detail,
        }
    finally:
        if own_conn:
            conn.close()


def format_realtime_report(code: str, conn=None) -> str:
    """
    生成某只股票的「近实时机构动向」报告。

    与 format_institutional_report (季度快照) 并列:
      - format_institutional_report → 基金持仓 + 十大股东 (滞后 15-90 天)
      - format_realtime_report       → 主力资金 + 龙虎榜 (滞后 T+1)
    """
    own_conn = conn is None
    if own_conn:
        conn = db.get_conn()
    try:
        name = _get_stock_name(conn, code)

        s = realtime_institutional_score(code, conn)
        mm = s["main_money"]
        lhb = s["lhb"]
        dzjy = s["dzjy"]

        lines = [
            f"近实时机构动向: {code} {name}",
            "=" * 55,
            f"综合评分: {s['score']}/100  "
            f"(主力 {s['main_money_score']}/40 + "
            f"龙虎榜 {s['lhb_score']}/40 + "
            f"大宗 {s['dzjy_score']}/20)",
            "",
        ]

        # 主力资金段
        if mm.get("available"):
            sign = "+" if mm["net_sum_yi"] >= 0 else ""
            lines.extend([
                f"▌ 主力资金 (近{mm['days_available']}日)",
                f"  累计净流入:       {sign}{mm['net_sum_yi']:.2f} 亿元",
                f"  净流入天数:       {mm['up_days']} / {mm['days_available']}",
                f"  最新数据日期:     {mm['latest_date']}",
                "",
            ])
        else:
            lines.extend(["▌ 主力资金: 暂无数据 (请先跑 slow_fetcher --moneyflow)", ""])

        # 龙虎榜段
        if lhb.get("available"):
            lines.extend([
                f"▌ 龙虎榜机构席位 ({lhb['period']})",
                f"  上榜次数:         {lhb['lhb_count']}",
                f"  买方机构次数:     {lhb['inst_buy_count']}",
                f"  卖方机构次数:     {lhb['inst_sell_count']}",
                f"  机构净买入:       {lhb['inst_net_yi']:+.2f} 亿元",
                f"  龙虎榜净买:       {lhb['lhb_net_yi']:+.2f} 亿元",
                f"  最近上榜日:       {lhb['last_lhb_date'] or '-'}",
                "",
            ])
        else:
            lines.extend([
                "▌ 龙虎榜机构席位: 该股近期未入选龙虎榜",
                "  (或 stock_lhb_stat 表为空, 请先跑 slow_fetcher --lhb-snapshot)",
                "",
            ])

        # 大宗交易段
        if dzjy.get("available"):
            lines.extend([
                f"▌ 大宗交易 ({dzjy['period']})",
                f"  总上榜次数:       {dzjy['total_count']}  "
                f"(溢价 {dzjy['premium_count']} / 折价 {dzjy['discount_count']})",
                f"  总成交额:         {dzjy['total_amt_yi']:.2f} 亿元",
                f"  平均折溢率:       {dzjy['avg_premium_rate']:+.2f}%",
                f"  成交额/流通市值:  {dzjy['amt_to_float']:.2f}%",
                f"  最近上榜日:       {dzjy['last_dzjy_date'] or '-'}",
                "",
            ])
        else:
            lines.extend([
                "▌ 大宗交易: 该股近期无大宗交易记录",
                "  (或 stock_dzjy_stat 表为空, 请先跑 slow_fetcher --dzjy-snapshot)",
                "",
            ])

        lines.append(f"一句话: {s['detail']}")
        return "\n".join(lines)
    finally:
        if own_conn:
            conn.close()


# ══════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════

def main():
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="机构持仓数据拉取")
    parser.add_argument("--update", action="store_true",
                        help="批量更新全市场机构持仓")
    parser.add_argument("--query", type=str, metavar="CODE",
                        help="查询某只股票的机构持仓")
    parser.add_argument("--score", type=str, metavar="CODE",
                        help="查看某只股票的机构共识度评分")
    parser.add_argument("--batch", type=int, default=20,
                        help="每批拉取股票数 (默认 20)")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="每只股票间隔秒数 (默认 2)")
    parser.add_argument("--max-minutes", type=float, default=0,
                        help="最长运行分钟数 (0=无限制)")
    parser.add_argument("--realtime", type=str, metavar="CODE",
                        help="查看某只股票的近实时机构动向 (主力资金+龙虎榜)")
    parser.add_argument("--lhb-snapshot", action="store_true",
                        help="抓取一次全市场龙虎榜快照 (写入 stock_lhb_stat)")
    parser.add_argument("--lhb-period", type=str, default=_LHB_PERIOD_DEFAULT,
                        choices=list(_LHB_PERIODS),
                        help="龙虎榜统计周期 (默认 近一月)")
    parser.add_argument("--dzjy-snapshot", action="store_true",
                        help="抓取一次全市场大宗交易快照 (写入 stock_dzjy_stat)")
    parser.add_argument("--dzjy-period", type=str, default=_DZJY_PERIOD_DEFAULT,
                        choices=list(_DZJY_PERIODS),
                        help="大宗交易统计周期 (默认 近一月)")

    args = parser.parse_args()

    if args.query:
        print(format_institutional_report(args.query))

    elif args.realtime:
        print(format_realtime_report(args.realtime))

    elif args.lhb_snapshot:
        r = refresh_lhb_stat(period=args.lhb_period)
        _log.info("✅ 龙虎榜快照完成: %s", r)

    elif args.dzjy_snapshot:
        r = refresh_dzjy_stat(period=args.dzjy_period)
        _log.info("✅ 大宗交易快照完成: %s", r)

    elif args.score:
        import json
        result = institutional_score(args.score)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.update:
        import time as _time
        _start = _time.monotonic()
        _deadline = _start + args.max_minutes * 60 if args.max_minutes > 0 else 0
        total_fetched = 0
        rounds = 0

        while True:
            if _deadline and _time.monotonic() >= _deadline:
                _log.warning("⏱️  达到时间限制，优雅退出")
                break

            rounds += 1
            result = fetch_institutional_batch(
                batch_size=args.batch,
                interval=args.interval,
            )
            total_fetched += result.get("fetched", 0)
            remaining = result.get("remaining", 0)
            errors = result.get("errors", [])

            _log.info("第 %d 轮: 本轮 %d 只, 累计 %d 只, 剩余 %d 只, 错误 %d",
                      rounds, result.get("fetched", 0), total_fetched,
                      remaining, len(errors))

            if result.get("done"):
                _log.info("✅ 全市场机构持仓更新完成!")
                break

            if result.get("fetched", 0) == 0:
                break

        elapsed = (_time.monotonic() - _start) / 60
        _log.info("总计: %d 只, %.1f 分钟", total_fetched, elapsed)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
