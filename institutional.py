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
from datetime import datetime, timedelta
from _http_utils import cn_now, cn_str

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

def get_fund_holdings(code: str, **_kwargs) -> dict:
    """
    通过 AKShare (新浪财经) 获取某只股票的基金持仓。

    数据源: ak.stock_fund_stock_holder → 新浪财经-股本股东-基金持股
    列: 基金名称, 基金代码, 持仓数量, 占流通股比例, 持股市值, 占净值比例, 截止日期

    Args:
        code: 股票代码 (如 "600519")

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
    df = ak.stock_fund_stock_holder(symbol=code)

    if df is None or df.empty:
        return {"report_date": "", "total_count": 0, "items": []}

    # 取最新一期 (截止日期最大)
    df["截止日期"] = df["截止日期"].astype(str)
    latest_date = df["截止日期"].max()
    latest = df[df["截止日期"] == latest_date].copy()

    # 转换日期格式: "2024-12-31" (可能是 datetime.date 或 str)
    report_date = str(latest_date)[:10]

    def _f(v):
        try:
            return round(float(v), 4)
        except (TypeError, ValueError):
            return None

    items = []
    for _, row in latest.iterrows():
        items.append({
            "fund_code": str(row.get("基金代码") or ""),
            "fund_name": str(row.get("基金名称") or ""),
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


def get_top_holders(code: str) -> dict:
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

    # ── 主数据源: 东财个股页 ──
    try:
        sym = _ak_symbol(code)
        date_str = _quarter_date_str()
        df = ak.stock_gdfx_free_top_10_em(symbol=sym, date=date_str)

        if df is not None and not df.empty:
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
    df = ak.stock_circulate_stock_holder(symbol=code)

    if df is None or df.empty:
        return {"report_date": "", "items": []}

    # 取最新一期
    df["截止日期"] = df["截止日期"].astype(str)
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

def store_fund_holdings(conn, code: str, holdings: dict) -> int:
    """将基金持仓数据写入 fund_holdings 表。返回入库条数。"""
    report_date = holdings.get("report_date", "")
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
            code, report_date, fund_code,
            it.get("fund_name"),
            it.get("hold_shares"),
            it.get("hold_mv"),
            it.get("nav_pct"),
            it.get("float_pct"),
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
            code, report_date, it.get("rank", 0),
            it.get("holder_name"),
            it.get("holder_type"),
            it.get("hold_shares"),
            it.get("hold_pct"),
            it.get("change_shares"),
            it.get("change_type"),
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
            name = db.fetchone(conn,
                "SELECT name FROM stocks WHERE code=?", (code,))
            name = name[0] if name else code

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

        total = round(fund_score + smart_money_score + holder_change_score, 1)

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
        name_row = db.fetchone(conn,
            "SELECT name FROM stocks WHERE code=?", (code,))
        name = name_row[0] if name_row else code

        score_data = institutional_score(code, conn)
        target_q = _current_report_quarter()

        lines = [
            f"机构持仓报告: {code} {name}",
            f"报告期: {score_data['report_date'] or target_q}",
            "=" * 55,
            "",
            f"机构共识度评分: {score_data['score']}/100",
            f"  基金持仓 ({score_data['fund_count']}只): {score_data['fund_score']}/40",
            f"  聪明资金: {score_data['smart_money_score']}/30",
            f"  增减持动向: {score_data['holder_change_score']}/30",
            "",
        ]

        # ── 十大流通股东 ──
        holders = db.fetchall(conn,
            "SELECT rank, holder_name, holder_type, hold_pct, "
            "change_shares, change_type "
            "FROM stock_top_holders WHERE code = ? AND report_date >= ? "
            "ORDER BY rank",
            (code, target_q))

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
            "FROM fund_holdings WHERE code = ? AND report_date >= ? "
            "ORDER BY hold_mv DESC NULLS LAST LIMIT 10",
            (code, target_q))

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

    args = parser.parse_args()

    if args.query:
        print(format_institutional_report(args.query))

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
