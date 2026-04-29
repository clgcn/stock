"""
从 data/ 目录下的 JSON 文件（4.json, 5.json, ...）批量导入股票数据到 PostgreSQL
用法: python import_json.py
"""

import json
import db
import glob
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def ensure_tables(conn):
    db.execute(conn, """
        CREATE TABLE IF NOT EXISTS stocks (
            code          TEXT PRIMARY KEY,
            name          TEXT,
            suspended     INTEGER DEFAULT 0
        )
    """)
    conn.commit()

def import_file(conn, path: str) -> tuple[int, int]:
    """导入单个 JSON 文件，返回 (新增, 跳过) 数量。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    diff = data.get("data", {}).get("diff", [])
    if not diff:
        print(f"  ⚠️  {os.path.basename(path)}: diff 字段为空，跳过")
        return 0, 0

    rows = []
    skipped = 0
    for r in diff:
        code = r.get("f12", "")
        if not code:
            skipped += 1
            continue
        rows.append((code, r.get("f14", "")))

    if rows:
        sql = """
            INSERT INTO stocks (code, name, suspended)
            VALUES (%s, %s, 0)
            ON CONFLICT(code) DO UPDATE SET
                name = excluded.name
            """
        cur = conn.cursor()
        try:
            cur.executemany(sql, rows)
            conn.commit()
        finally:
            cur.close()

    return len(rows), skipped


def main():
    # 找所有数字命名的 json 文件，按页码排序
    pattern = os.path.join(DATA_DIR, "*.json")
    files = sorted(
        [f for f in glob.glob(pattern) if os.path.basename(f).replace(".json", "").isdigit()],
        key=lambda f: int(os.path.basename(f).replace(".json", ""))
    )

    if not files:
        print("data/ 目录下没有找到数字命名的 JSON 文件（如 4.json, 5.json）")
        return

    conn = db.get_conn()
    try:
        ensure_tables(conn)

        before = db.execute(conn, "SELECT COUNT(*) FROM stocks", ()).fetchone()[0]
        print(f"导入前数据库已有: {before} 条\n")

        total_inserted = 0
        total_skipped  = 0

        for path in files:
            inserted, skipped = import_file(conn, path)
            total_inserted += inserted
            total_skipped  += skipped
            print(f"  ✅ {os.path.basename(path):10s}  新增/覆盖: {inserted:3d} 条  跳过: {skipped} 条")

        after = db.execute(conn, "SELECT COUNT(*) FROM stocks", ()).fetchone()[0]
        print(f"\n导入完成！")
        print(f"  处理文件: {len(files)} 个")
        print(f"  本次写入: {total_inserted} 条（含覆盖更新）")
        print(f"  数据库总计: {before} → {after} 条（净增 {after - before} 条）")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
