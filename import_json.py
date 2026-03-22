"""
从 data/ 目录下的 JSON 文件（4.json, 5.json, ...）批量导入股票数据到 stocks.db
用法: python import_json.py
"""

import json
import sqlite3
import glob
import os
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH  = os.path.join(DATA_DIR, "stocks.db")


def _safe_float(v):
    if v is None or v == "-" or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def import_file(conn: sqlite3.Connection, path: str) -> tuple[int, int]:
    """导入单个 JSON 文件，返回 (新增, 跳过) 数量。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    diff = data.get("data", {}).get("diff", [])
    if not diff:
        print(f"  ⚠️  {os.path.basename(path)}: diff 字段为空，跳过")
        return 0, 0

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    batch_id = f"manual_{os.path.basename(path)}"

    inserted = 0
    skipped  = 0

    for r in diff:
        code = r.get("f12", "")
        if not code:
            skipped += 1
            continue

        current  = _safe_float(r.get("f2"))
        amount   = _safe_float(r.get("f6"))
        total_mv = _safe_float(r.get("f20"))
        float_mv = _safe_float(r.get("f21"))

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
        inserted += 1

    conn.commit()
    return inserted, skipped


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

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")

    # 导入前总数
    before = conn.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]
    print(f"导入前数据库已有: {before} 条\n")

    total_inserted = 0
    total_skipped  = 0

    for path in files:
        inserted, skipped = import_file(conn, path)
        total_inserted += inserted
        total_skipped  += skipped
        print(f"  ✅ {os.path.basename(path):10s}  新增/覆盖: {inserted:3d} 条  跳过: {skipped} 条")

    after = conn.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]
    conn.close()

    print(f"\n导入完成！")
    print(f"  处理文件: {len(files)} 个")
    print(f"  本次写入: {total_inserted} 条（含覆盖更新）")
    print(f"  数据库总计: {before} → {after} 条（净增 {after - before} 条）")


if __name__ == "__main__":
    main()
