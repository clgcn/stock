"""Comparison script: v1 vs v2 pipeline for value strategy on live DB."""
import os
import sys
from dotenv import load_dotenv

# Load with explicit path to avoid frame-walking issue
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stock_screener as sc


def _fmt_pick(p, idx=None):
    prefix = f"{idx:>2}. " if idx is not None else "    "
    return (
        f"{prefix}{p.get('code'):<8}{p.get('name','')[:6]:<8}"
        f"EQS={p.get('entry_quality_score', float('nan')):>6.2f}  "
        f"prob_up={p.get('prob_up', float('nan')):>5.1f}%  "
        f"close={p.get('close', float('nan')):>7.2f}  "
        f"pct={p.get('pct_chg', float('nan')):>5.2f}%  "
        f"action={p.get('action_suggestion','')[:24]}"
    )


def run_compare(strategy="value", top_n=15):
    print(f"\n{'='*100}")
    print(f"COMPARISON: {strategy.upper()} strategy, top_n={top_n}")
    print(f"{'='*100}")

    # ---- v1 ----
    print("\n--- V1 (legacy, absolute thresholds, magic weights) ---")
    try:
        v1_ret = sc.screen_stocks(
            strategy=strategy,
            top_n=top_n,
            stage1_limit=60,
            quality_threshold=5.0,
            max_candidates=25,
            analysis_days=120,
        )
    except Exception as e:
        print(f"V1 ERROR: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return
    v1 = v1_ret.get("results", []) if isinstance(v1_ret, dict) else []
    dh = v1_ret.get("data_health") if isinstance(v1_ret, dict) else None
    if dh:
        print(f"data_health: coverage={dh.get('pct_chg_coverage')}  max_upd={dh.get('max_updated_at')}  stale_days={dh.get('staleness_days')}  is_stale={dh.get('is_stale')}")
        for w in dh.get("warnings", []):
            print(f"  warn: {w}")
    print(f"v1 stage1_count={v1_ret.get('stage1_count')}  stage2_count={v1_ret.get('stage2_count')}  qualified={v1_ret.get('qualified_count')}")
    for i, p in enumerate(v1[:top_n], 1):
        print(_fmt_pick(p, i))

    # ---- v2 ----
    print("\n--- V2 (z-score composite, quantile gate, relative refinement) ---")
    try:
        v2_ret = sc.screen_stocks_v2(
            strategy=strategy,
            top_n=top_n,
            stage1_limit=60,
            quality_quantile=0.20,
            min_keep=3,
            max_candidates=25,
            analysis_days=120,
        )
    except Exception as e:
        print(f"V2 ERROR: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return
    v2 = v2_ret.get("results", []) if isinstance(v2_ret, dict) else []
    print(f"v2 stage1_count={v2_ret.get('stage1_count')}  stage2_count={v2_ret.get('stage2_count')}  qualified={v2_ret.get('qualified_count')}")
    for i, p in enumerate(v2[:top_n], 1):
        print(_fmt_pick(p, i))

    # ---- Overlap analysis ----
    v1_codes = [p.get("code") for p in v1]
    v2_codes = [p.get("code") for p in v2]
    v1_set = set(v1_codes)
    v2_set = set(v2_codes)
    common = v1_set & v2_set
    only_v1 = v1_set - v2_set
    only_v2 = v2_set - v1_set

    print(f"\n--- Overlap analysis ---")
    print(f"v1 picks:         {len(v1)}")
    print(f"v2 picks:         {len(v2)}")
    print(f"common:           {len(common)}  ({sorted(common)})")
    print(f"only in v1:       {len(only_v1)}  ({sorted(only_v1)})")
    print(f"only in v2 (new): {len(only_v2)}  ({sorted(only_v2)})")
    if v1 and v2:
        pct_overlap = 100.0 * len(common) / max(len(v1_set), len(v2_set))
        print(f"overlap pct:      {pct_overlap:.1f}%")

    # rank shift for common
    if common:
        print("\n--- Rank shifts (common picks) ---")
        for code in sorted(common):
            r1 = v1_codes.index(code) + 1
            r2 = v2_codes.index(code) + 1
            delta = r1 - r2
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
            print(f"  {code}  v1 rank #{r1:>2}  ->  v2 rank #{r2:>2}   {arrow}{abs(delta)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="value")
    ap.add_argument("--top-n", type=int, default=15)
    args = ap.parse_args()
    run_compare(strategy=args.strategy, top_n=args.top_n)
