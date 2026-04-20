"""Probe v1 with small stage1_limit to see what happens now that NaN pct_chg is allowed."""
import os, sys, traceback, faulthandler
faulthandler.enable()

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stock_screener as sc
print("[probe] imported, calling v1 value (stage1_limit=10, max_candidates=5)", flush=True)
try:
    r = sc.screen_stocks(
        strategy="value",
        top_n=5,
        stage1_limit=10,
        quality_threshold=5.0,
        max_candidates=5,
        analysis_days=60,
        deep_scan_enabled=True,
    )
except BaseException as e:
    print(f"[probe] exception: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    raise

print(f"[probe] done. keys={list(r.keys())}", flush=True)
print(f"[probe] stage1={r.get('stage1_count')} stage2={r.get('stage2_count')} qualified={r.get('qualified_count')}", flush=True)
dh = r.get("data_health", {})
print(f"[probe] data_health: coverage={dh.get('pct_chg_coverage')} stale_days={dh.get('staleness_days')}", flush=True)
for p in r.get("results", []):
    print(f"  - {p.get('code')} {p.get('name')}  EQS={p.get('entry_quality_score')}", flush=True)
