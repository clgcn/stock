#!/usr/bin/env python3
"""
全量机构持仓同步脚本 — 按季末回填 stock_top_holders / fund_holdings
===================================================================

用法:
  # 同步单个季度 (全市场 ~3000 只)
  python sync_institutional_full.py --quarter 2025-12-31

  # 强制重抓 (不跳过已有)
  python sync_institutional_full.py --quarter 2025-12-31 --no-skip

  # 只抓十大股东, 不抓基金持仓
  python sync_institutional_full.py --quarter 2025-12-31 --only top

  # 分片 (便于并行 / 断点)
  python sync_institutional_full.py --quarter 2025-12-31 --offset 0 --limit 1000

  # 多季度一起
  python sync_institutional_full.py --quarter 2025-09-30 2025-12-31 2026-03-31

背景:
  A 股机构持仓按法定只在季末披露 (03-31/06-30/09-30/12-31).
  本脚本遍历全市场活跃股票, 对指定季度逐只 fetch + upsert, 确保覆盖率
  接近 100%. 脚本幂等: 已有数据默认跳过, 可随时中断后重跑.

  估时: 每只约 interval + 网络延迟, 默认 1 秒 interval → ~3000 秒 (约 50 分钟)
        可 --interval 0.3 加速, 但要注意被数据源限流.

推荐用法: 用 nohup 后台跑并写日志
  nohup python sync_institutional_full.py --quarter 2025-12-31 \\
    > logs/sync_2025q4.log 2>&1 &
"""
import argparse
import logging
import os
import sys
import time

# 加载 .env
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import institutional as inst


def _format_eta(remaining: int, interval: float) -> str:
    secs = int(remaining * (interval + 1.5))  # 保守:interval + 1.5s 网络
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def run_one_quarter(quarter: str, args) -> dict:
    do_funds = args.only in ("both", "funds")
    do_top = args.only in ("both", "top")

    print(f"\n{'='*72}")
    print(f"▶ 开始同步季度: {quarter}")
    print(f"  do_funds={do_funds}  do_top_holders={do_top}")
    print(f"  skip_if_exists={not args.no_skip}  interval={args.interval}s")
    print(f"  offset={args.offset}  limit={args.limit or 'ALL'}")
    print(f"{'='*72}")

    t0 = time.time()
    interrupted = False
    try:
        stats = inst.fetch_full_quarter(
            target_quarter=quarter,
            skip_if_exists=not args.no_skip,
            batch_size=args.batch_size,
            interval=args.interval,
            do_funds=do_funds,
            do_top_holders=do_top,
            start_offset=args.offset,
            limit=args.limit,
        )
    except KeyboardInterrupt:
        interrupted = True
        # 让调用方知道中断, fetch_full_quarter 保留了内部 stats 累积
        # 但目前没有暴露出来, 所以这里只能报最小骨架.
        stats = {
            "target_quarter": quarter, "total": 0, "processed": 0,
            "skipped": 0, "funds_stored": 0, "top_stored": 0,
            "errors": ["KeyboardInterrupt 用户中断 (Ctrl-C)"],
        }
        print(f"\n⚠️  收到中断信号, 当前季度 {quarter} 提前退出")
    stats["elapsed_sec"] = round(time.time() - t0, 1)
    stats["interrupted"] = interrupted

    print(f"\n─── {quarter} 完成 ───")
    print(f"  total:        {stats['total']}")
    print(f"  processed:    {stats['processed']}")
    print(f"  skipped:      {stats['skipped']}")
    print(f"  funds_stored: {stats['funds_stored']}")
    print(f"  top_stored:   {stats['top_stored']}")
    print(f"  errors:       {len(stats['errors'])}")
    print(f"  elapsed:      {stats['elapsed_sec']}s")
    if stats["errors"]:
        print("  前 5 个错误:")
        for e in stats["errors"][:5]:
            print(f"    - {e}")
    return stats


def main():
    ap = argparse.ArgumentParser(
        description="全量按季度回填 fund_holdings / stock_top_holders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--quarter", "-q", nargs="+", required=True,
                    help="一个或多个季末日期, 如 2025-12-31")
    ap.add_argument("--only", choices=["both", "funds", "top"], default="both",
                    help="默认 both; funds=只抓基金持仓; top=只抓十大股东")
    ap.add_argument("--no-skip", action="store_true",
                    help="强制重抓, 不跳过已有")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="每只股票之间 sleep 秒数 (防封), 默认 1.0")
    ap.add_argument("--batch-size", type=int, default=100,
                    help="多少只打一条进度 log, 默认 100")
    ap.add_argument("--offset", type=int, default=0,
                    help="从第几只股票开始, 默认 0")
    ap.add_argument("--limit", type=int, default=0,
                    help="最多处理多少只, 默认 0=不限")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # 校验季末
    for q in args.quarter:
        if not inst._is_quarter_end(q):
            ap.error(f"--quarter {q!r} 不是合法季末日期 (应为 YYYY-03-31 / 06-30 / 09-30 / 12-31)")

    all_stats = []
    for q in args.quarter:
        s = run_one_quarter(q, args)
        all_stats.append(s)
        # 中断后不再继续后面的季度, 避免用户二次 Ctrl-C
        if s.get("interrupted"):
            print(f"\n▎ 已中断, 跳过后续 {len(args.quarter) - len(all_stats)} 个季度")
            break

    # 汇总
    print(f"\n{'='*72}")
    print("▎ 总汇总")
    print(f"{'='*72}")
    tot_proc = sum(s["processed"] for s in all_stats)
    tot_fund = sum(s["funds_stored"] for s in all_stats)
    tot_top = sum(s["top_stored"] for s in all_stats)
    tot_err = sum(len(s["errors"]) for s in all_stats)
    tot_t = sum(s["elapsed_sec"] for s in all_stats)
    for s in all_stats:
        print(f"  {s['target_quarter']:12s}  processed={s['processed']:4d}  "
              f"funds=+{s['funds_stored']:5d}  top=+{s['top_stored']:5d}  "
              f"errors={len(s['errors']):3d}  elapsed={s['elapsed_sec']:>7.1f}s")
    print("─" * 72)
    print(f"  合计: processed={tot_proc}  funds=+{tot_fund}  top=+{tot_top}  "
          f"errors={tot_err}  elapsed={tot_t:.1f}s")

    return 0 if tot_err < tot_proc * 0.1 else 1  # 错误率 >10% 视为失败


if __name__ == "__main__":
    sys.exit(main())
