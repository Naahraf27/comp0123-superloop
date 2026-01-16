#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def load_simple_lcc(gexf_path: str) -> nx.Graph:
    G = nx.read_gexf(gexf_path)

    # Ensure simple undirected graph
    if G.is_directed():
        G = G.to_undirected()
    G = nx.Graph(G)  # drop multiedges if any
    G.remove_edges_from(nx.selfloop_edges(G))

    # Largest connected component
    if G.number_of_nodes() == 0:
        raise ValueError(f"{gexf_path} has 0 nodes after simplification.")
    lcc_nodes = max(nx.connected_components(G), key=len)
    G = G.subgraph(lcc_nodes).copy()
    return G


def richclub_phi(G: nx.Graph, k_values: np.ndarray):
    deg = dict(G.degree())
    # Precompute node list per k quickly via degrees array
    nodes = np.array(list(G.nodes()))
    degrees = np.array([deg[n] for n in nodes])

    phi = np.full(len(k_values), np.nan, dtype=float)
    Nk = np.zeros(len(k_values), dtype=int)

    for idx, k in enumerate(k_values):
        mask = degrees > k
        club = nodes[mask]
        n = int(club.size)
        Nk[idx] = n
        if n < 2:
            phi[idx] = np.nan
            continue
        sub = G.subgraph(club)
        m = sub.number_of_edges()
        phi[idx] = (2.0 * m) / (n * (n - 1))
    return phi, Nk


def degree_preserving_null(G: nx.Graph, nswap: int, seed: int) -> nx.Graph:
    H = G.copy()
    # double_edge_swap preserves degrees and stays simple
    # max_tries high enough to avoid failures on sparse graphs
    max_tries = max(1000, nswap * 20)
    rng = np.random.default_rng(seed)

    # NetworkX wants an int seed (not RNG), so we generate one
    nx_seed = int(rng.integers(0, 2**31 - 1))
    try:
        nx.double_edge_swap(H, nswap=nswap, max_tries=max_tries, seed=nx_seed)
    except Exception:
        # If swaps fail, we still return what we got (better than crashing).
        pass
    return H


def nanmean_safe(A: np.ndarray, axis=0):
    # Avoid RuntimeWarnings by explicitly handling all-NaN columns
    out = np.full(A.shape[1], np.nan, dtype=float)
    for j in range(A.shape[1]):
        col = A[:, j]
        if np.all(np.isnan(col)):
            out[j] = np.nan
        else:
            out[j] = np.nanmean(col)
    return out


def ratio_matrix(num_vec: np.ndarray, den_mat: np.ndarray) -> np.ndarray:
    """
    num_vec shape: (K,)
    den_mat shape: (R, K)
    returns shape: (R, K) with NaNs where den<=0 or nan
    """
    num = num_vec[None, :]  # (1,K)
    den = den_mat           # (R,K)
    out = np.full_like(den, np.nan, dtype=float)

    valid = np.isfinite(den) & (den > 0) & np.isfinite(num)
    np.divide(num, den, out=out, where=valid)  # broadcasts num across R
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/graphs/outer_stops_baseline_no_sl_exclusive.gexf")
    ap.add_argument("--sl", default="outputs/graphs/outer_stops_with_superloop.gexf")
    ap.add_argument("--R", type=int, default=50)
    ap.add_argument("--swap-factor", type=float, default=10.0, help="swaps per edge (nswap = swap_factor * M)")
    ap.add_argument("--min-rich", type=int, default=50, help="min N(>k) to consider stable")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-fig", default="figures/richclub_rho.png")
    ap.add_argument("--out-pdf", default="figures/richclub_rho.pdf")
    ap.add_argument("--out-csv", default="outputs/tables/richclub_rho.csv")
    args = ap.parse_args()

    out_fig = Path(args.out_fig)
    out_pdf = Path(args.out_pdf)
    out_csv = Path(args.out_csv)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    G_base = load_simple_lcc(args.base)
    G_sl = load_simple_lcc(args.sl)

    print(f"Base: {args.base} | N={G_base.number_of_nodes()} M={G_base.number_of_edges()}")
    print(f"SL  : {args.sl} | N={G_sl.number_of_nodes()} M={G_sl.number_of_edges()}")

    max_k = int(max(max(dict(G_base.degree()).values()), max(dict(G_sl.degree()).values())))
    k_values = np.arange(0, max_k + 1)

    base_phi, base_Nk = richclub_phi(G_base, k_values)
    sl_phi, sl_Nk = richclub_phi(G_sl, k_values)

    # Stable region based on BOTH graphs having enough rich nodes
    stable_mask = (np.minimum(base_Nk, sl_Nk) >= args.min_rich)
    stable_k_max = int(k_values[stable_mask].max()) if np.any(stable_mask) else -1

    nswap_base = int(args.swap_factor * G_base.number_of_edges())
    nswap_sl = int(args.swap_factor * G_sl.number_of_edges())
    print(f"Null ensemble: R={args.R}, swap_factor={args.swap_factor} swaps/edge")

    base_null = np.full((args.R, len(k_values)), np.nan, dtype=float)
    sl_null = np.full((args.R, len(k_values)), np.nan, dtype=float)

    for r in range(args.R):
        if (r + 1) % 5 == 0:
            print(f"  ... {r+1}/{args.R}")

        Hb = degree_preserving_null(G_base, nswap=nswap_base, seed=args.seed + 1000 + r)
        Hs = degree_preserving_null(G_sl, nswap=nswap_sl, seed=args.seed + 2000 + r)

        base_null[r, :], _ = richclub_phi(Hb, k_values)
        sl_null[r, :], _ = richclub_phi(Hs, k_values)

    # Normalised rich-club ρ(k) = φ_emp / <φ_null>
    base_rhos = ratio_matrix(base_phi, base_null)
    sl_rhos = ratio_matrix(sl_phi, sl_null)

    base_rho_mean = nanmean_safe(base_rhos, axis=0)
    sl_rho_mean = nanmean_safe(sl_rhos, axis=0)

    def pct(A, q):
        out = np.full(A.shape[1], np.nan, dtype=float)
        for j in range(A.shape[1]):
            col = A[:, j]
            col = col[np.isfinite(col)]
            if col.size == 0:
                out[j] = np.nan
            else:
                out[j] = np.percentile(col, q)
        return out

    base_rho_p05 = pct(base_rhos, 5)
    base_rho_p95 = pct(base_rhos, 95)
    sl_rho_p05 = pct(sl_rhos, 5)
    sl_rho_p95 = pct(sl_rhos, 95)

    # --- Plot (top: ONLY stable region to avoid insane tail; bottom: N(>k) full)
    fig = plt.figure(figsize=(11, 7.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1], hspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)

    title = f"Rich-club ordering (normalised by degree-preserving null, R={args.R})"
    ax.set_title(title)

    # Stable indices
    if stable_k_max >= 0:
        kk = k_values[k_values <= stable_k_max]
    else:
        kk = k_values[:1]

    idx = kk.astype(int)
    ax.plot(kk, base_rho_mean[idx], marker="o", label="Baseline")
    ax.plot(kk, sl_rho_mean[idx], marker="o", label="With Superloop")
    ax.fill_between(kk, base_rho_p05[idx], base_rho_p95[idx], alpha=0.15)
    ax.fill_between(kk, sl_rho_p05[idx], sl_rho_p95[idx], alpha=0.15)
    ax.axhline(1.0, linewidth=1.0)

    ax.set_ylabel("Normalised rich-club ρ(k)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")

    # Visual cue for "we're only plotting stable part"
    if stable_k_max >= 0:
        ax.text(
            0.99, 0.02,
            f"Plotted only where min(N(>k)) ≥ {args.min_rich}  (stable up to k={stable_k_max})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10
        )

    # Bottom panel: N(>k) on log scale (full)
    ax2.plot(k_values, base_Nk, marker="o", label="Baseline N(>k)")
    ax2.plot(k_values, sl_Nk, marker="o", label="SL N(>k)")
    ax2.set_yscale("log")
    ax2.set_ylabel("N(>k)")
    ax2.set_xlabel("k threshold (degree > k)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right")

    fig.savefig(out_fig, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Wrote {out_fig}")
    print(f"Wrote {out_pdf}")

    # CSV for reproducibility
    out = pd.DataFrame({
        "k": k_values,
        "base_phi": base_phi,
        "sl_phi": sl_phi,
        "base_rho_mean": base_rho_mean,
        "base_rho_p05": base_rho_p05,
        "base_rho_p95": base_rho_p95,
        "sl_rho_mean": sl_rho_mean,
        "sl_rho_p05": sl_rho_p05,
        "sl_rho_p95": sl_rho_p95,
        "base_Nk_gt": base_Nk,
        "sl_Nk_gt": sl_Nk,
        "stable": stable_mask,
    })
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    if stable_k_max >= 0:
        print(f"Stable up to k={stable_k_max} (min(N(>k)) >= {args.min_rich})")
    else:
        print("No stable k found (try lowering --min-rich).")


if __name__ == "__main__":
    main()
