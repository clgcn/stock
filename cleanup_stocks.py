"""
清理数据库: 只保留沪深主板 (非ST) 股票。
删除创业板(300/301)、科创板(688)、北交所(8/4)、ST/*ST 及其关联的历史数据。

用法: python cleanup_stocks.py          # 预览 (不删除)
      python cleanup_stocks.py --apply  # 执行删除
"""

import db
import argparse


# 保留条件: 沪深主板 + 非ST
KEEP_CONDITION = """
    (code LIKE '600%%' OR code LIKE '601%%' OR code LIKE '603%%'
     OR code LIKE '000%%' OR code LIKE '001%%' OR code LIKE '002%%' OR code LIKE '003%%')
    AND name NOT LIKE '%%ST%%'
"""

# 要删除的板块 (用于统计)
BOARDS = {
    "创业板 (300/301)": "code LIKE '300%%' OR code LIKE '301%%'",
    "科创板 (688)":     "code LIKE '688%%'",
    "北交所 (8/4)":     "code LIKE '8%%' OR code LIKE '4%%'",
    "ST/*ST (主板)":    f"name LIKE '%%ST%%' AND ({KEEP_CONDITION.replace('AND name NOT LIKE', 'AND name LIKE')})",
}

# 关联表
RELATED_TABLES = ["stock_history", "stock_fundamentals", "stock_moneyflow"]


def preview(conn):
    total = db.fetchone(conn, "SELECT COUNT(*) FROM stocks")[0]
    keep = db.fetchone(conn, f"SELECT COUNT(*) FROM stocks WHERE {KEEP_CONDITION}")[0]
    delete = total - keep

    print(f"═══ 数据库清理预览 ═══\n")
    print(f"  当前总数: {total} 只\n")

    for label, cond in BOARDS.items():
        count = db.fetchone(conn, f"SELECT COUNT(*) FROM stocks WHERE {cond}")[0]
        if count:
            print(f"  删除 {label}: {count} 只")

    print(f"\n  保留: {keep} 只 (沪深主板, 非ST)")
    print(f"  删除: {delete} 只")
    print(f"  节省: {delete/total*100:.0f}%\n")

    print(f"  更新时间预估 (10秒/只):")
    print(f"    当前: {total * 10 / 3600:.1f} 小时")
    print(f"    清理后: {keep * 10 / 3600:.1f} 小时\n")

    # 关联数据量
    for table in RELATED_TABLES:
        total_rows = db.fetchone(conn, f"SELECT COUNT(*) FROM {table}")[0]
        keep_rows = db.fetchone(conn,
            f"SELECT COUNT(*) FROM {table} WHERE code IN (SELECT code FROM stocks WHERE {KEEP_CONDITION})")[0]
        delete_rows = total_rows - keep_rows
        print(f"  {table}: 删除 {delete_rows:,} 行 / 保留 {keep_rows:,} 行")

    print(f"\n  ⚠️  这是预览，未执行任何删除。加 --apply 参数执行。")
    return delete


def apply_cleanup(conn):
    # 先获取要删除的 code 列表
    codes_to_delete = db.fetchall(conn,
        f"SELECT code FROM stocks WHERE NOT ({KEEP_CONDITION})")
    delete_codes = [r[0] for r in codes_to_delete]

    if not delete_codes:
        print("没有需要删除的股票。")
        return

    print(f"开始清理 {len(delete_codes)} 只股票...\n")

    # 删除关联表数据
    for table in RELATED_TABLES:
        before = db.fetchone(conn, f"SELECT COUNT(*) FROM {table}")[0]
        db.execute(conn,
            f"DELETE FROM {table} WHERE code NOT IN (SELECT code FROM stocks WHERE {KEEP_CONDITION})")
        after = db.fetchone(conn, f"SELECT COUNT(*) FROM {table}")[0]
        deleted = before - after
        print(f"  {table}: 删除 {deleted:,} 行")

    # 删除 stocks 主表
    before = db.fetchone(conn, "SELECT COUNT(*) FROM stocks")[0]
    db.execute(conn,
        f"DELETE FROM stocks WHERE NOT ({KEEP_CONDITION})")
    after = db.fetchone(conn, "SELECT COUNT(*) FROM stocks")[0]
    deleted = before - after
    print(f"  stocks: 删除 {deleted} 只")

    conn.commit()

    print(f"\n✅ 清理完成! 剩余 {after} 只股票 (沪深主板, 非ST)")
    print(f"   预估更新时间: {after * 10 / 3600:.1f} 小时")


def main():
    parser = argparse.ArgumentParser(description="清理数据库: 只保留沪深主板非ST股票")
    parser.add_argument("--apply", action="store_true",
                        help="执行删除 (默认只预览)")
    args = parser.parse_args()

    conn = db.get_conn()
    try:
        if args.apply:
            preview(conn)
            print("\n" + "=" * 50)
            apply_cleanup(conn)
        else:
            preview(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
