#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAIR_CSV = ROOT / "outputs/tables_killer/pair_samples.csv"  # you listed this

def main():
    if not PAIR_CSV.exists():
        raise FileNotFoundError(f"Missing {PAIR_CSV}. (Check outputs/tables_killer/)")
    df = pd.read_csv(PAIR_CSV)

    # Be flexible about column names
    cols = {c.lower(): c for c in df.columns}
    base_col = cols.get("d_base") or cols.get("d_base_hops") or cols.get("base_dist") or cols.get("d_baseline")
    sl_col   = cols.get("d_sl")   or cols.get("d_sl_hops")   or cols.get("sl_dist")   or cols.get("d_superloop")

    if base_col is None or sl_col is None:
        raise ValueError(f"Can't find distance columns. Found columns: {list(df.columns)}")

    dbase = df[base_col]
    dsl   = df[sl_col]
    delta = dbase - dsl

    n = int(delta.notna().sum())
    improved = int((delta > 0).sum())
    frac_improved = improved / n if n else 0.0

    print(f"OD pairs used (non-null): {n:,}")
    print(f"Share improved (Δd>0): {frac_improved*100:.2f}%")
    print(f"Mean Δd: {delta.mean():.3f}")
    print(f"Median Δd: {delta.median():.3f}")

if __name__ == "__main__":
    main()
