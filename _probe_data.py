"""Probe: what's in fetch_all_stocks and which filters kill candidates?"""
import os, sys
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stock_screener as sc
import pandas as pd
import numpy as np

df = sc.fetch_all_stocks()
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print()

def coverage(col):
    if col not in df.columns:
        return "N/A"
    s = df[col]
    return f"non-null={s.notna().sum()}/{len(s)} ({100*s.notna().sum()/len(s):.1f}%)"

for c in ["pe_ttm", "pb", "total_mv", "pct_chg", "volume", "current", "turnover_rate", "volume_ratio", "name"]:
    print(f"  {c:<16}: {coverage(c)}")

print()
print("--- numeric quantiles ---")
for c in ["pe_ttm", "pb", "total_mv", "pct_chg", "turnover_rate"]:
    if c not in df.columns: continue
    s = df[c].dropna()
    if s.empty:
        print(f"  {c}: all NaN")
        continue
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
    print(f"  {c}: p05={q[0.05]:.2f} p25={q[0.25]:.2f} p50={q[0.5]:.2f} p75={q[0.75]:.2f} p95={q[0.95]:.2f}")

print()
print("--- applying filter_value mask step by step ---")
base = (
    ~df["name"].str.match(r"^\*?ST\b", case=False, na=False)
    & df["pct_chg"].notna() & (df["pct_chg"] > -9.5) & (df["pct_chg"] < 9.5)
    & df["volume"].notna() & (df["volume"] > 0)
    & df["current"].notna() & (df["current"] > 0)
)
print(f"  after base: {base.sum()}")
m1 = base & df["pe_ttm"].notna() & (df["pe_ttm"] > 0) & (df["pe_ttm"] <= 25)
print(f"  + pe_ttm (0,25]: {m1.sum()}")
m2 = m1 & df["pb"].notna() & (df["pb"] > 0) & (df["pb"] <= 3)
print(f"  + pb (0,3]: {m2.sum()}")
m3 = m2 & df["total_mv"].notna() & (df["total_mv"] >= 50)
print(f"  + total_mv >= 50: {m3.sum()}")
m4 = m3 & df["turnover_rate"].notna() & (df["turnover_rate"] >= 0.5)
print(f"  + turnover_rate >= 0.5: {m4.sum()}")

# check volume distribution
print()
print(f"volume>0 count: {(df['volume'] > 0).sum()}")
print(f"volume.isna count: {df['volume'].isna().sum()}")
print(f"volume==0 count: {(df['volume'] == 0).sum()}")
print(f"pct_chg.isna count: {df['pct_chg'].isna().sum()}")
print(f"pct_chg==0 count: {(df['pct_chg'] == 0).sum()}")

print()
print("--- sample rows ---")
print(df.head(5)[["code","name","current","pct_chg","volume","pe_ttm","pb","total_mv","turnover_rate"]].to_string())
