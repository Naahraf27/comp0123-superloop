#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CAND_BASE = ["d_base", "dist_base", "baseline_dist", "baseline_distance"]
CAND_SL   = ["d_sl", "dist_sl", "sl_dist", "superloop_dist", "post_dist"]

def pick_col(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find a {label} distance column. Columns are: {list(df.columns)}")

def ecdf(x):
    x = np.sort(np.asarray(x))
    y = np.arange(1, len(x)+1) / len(x)
    return x, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", default="outputs/tables_killer/pair_samples.csv")
    ap.add_argument("--S", type=int, default=49800)
    ap.add_argument("--out_dir", default="outputs/figures_plus")
    args = ap.parse_args()

    pairs_csv = Path(args.pairs_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pairs_csv)
    base_col = pick_col(df, CAND_BASE, "baseline")
    sl_col   = pick_col(df, CAND_SL,   "post(SL)")

    # Use exactly S rows (shuffle first so it's not "first chunk" biased)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df = df.head(args.S).copy()

    d_base = df[base_col].astype(float).to_numpy()
    d_sl   = df[sl_col].astype(float).to_numpy()
    delta  = d_base - d_sl

    # --- CDF plot
    xb, yb = ecdf(d_base)
    xs, ys = ecdf(d_sl)

    plt.figure()
    plt.plot(xb, yb, label="Baseline")
    plt.plot(xs, ys, label="With Superloop")
    plt.xlabel("Shortest-path hops")
    plt.ylabel("Empirical CDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pathlength_cdf.png", dpi=250)
    plt.close()

    # --- Histogram of improvements
    plt.figure()
    # bins: integer hops, clamp at 0..max
    max_d = int(np.nanmax(delta))
    bins = np.arange(-0.5, max_d + 1.5, 1.0)
    plt.hist(delta, bins=bins)
    plt.xlabel(r"Hop-savings $\Delta d = d_{\mathrm{base}} - d_{\mathrm{SL}}$")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "path_improvement_hist.png", dpi=250)
    plt.close()

    print(f"Wrote: {out_dir/'pathlength_cdf.png'}")
    print(f"Wrote: {out_dir/'path_improvement_hist.png'}")
    print(f"Used S={len(df):,} rows from {pairs_csv}")

if __name__ == "__main__":
    main()
