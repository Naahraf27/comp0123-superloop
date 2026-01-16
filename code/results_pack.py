#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def haversine_km(lon1, lat1, lon2, lat2) -> float:
    # returns km
    if None in (lon1, lat1, lon2, lat2):
        return float("nan")
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_cached_sequences(cache_dir: Path) -> dict[str, dict]:
    seqs: dict[str, dict] = {}
    files = sorted(cache_dir.glob("route_sequence_*.json"))
    if not files:
        raise FileNotFoundError(f"No cached route_sequence_*.json files found in {cache_dir}")
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        line_id = obj.get("lineId") or fp.stem.replace("route_sequence_", "")
        seqs[str(line_id)] = obj
    return seqs


def extract_stop_lists(route_sequence: dict) -> list[list[dict]]:
    # TfL route sequence: stopPointSequences -> list of sequences, each has "stopPoint"
    out = []
    for sps in route_sequence.get("stopPointSequences", []) or []:
        stops = sps.get("stopPoint", []) or []
        if len(stops) >= 2:
            out.append(stops)
    return out


def is_superloop_route(route_sequence: dict) -> bool:
    name = str(route_sequence.get("lineName", "")).upper()
    return name.startswith("SL")


def build_graph_from_sequences(seqs: dict[str, dict]) -> tuple[nx.Graph, set[str], set[str]]:
    """
    Returns:
      G: stop-to-stop L-space graph (adjacent stops on any route)
      sl_lines: set of Superloop lineIds
      sl_nodes: set of stop ids served by any Superloop line
    """
    G = nx.Graph()
    sl_lines = {lid for lid, rs in seqs.items() if is_superloop_route(rs)}
    sl_nodes: set[str] = set()

    for line_id, rs in seqs.items():
        rs_is_sl = line_id in sl_lines
        for stops in extract_stop_lists(rs):
            for sp in stops:
                sid = str(sp.get("id"))
                if rs_is_sl and sid:
                    sl_nodes.add(sid)

            for a, b in zip(stops, stops[1:]):
                u = str(a.get("id"))
                v = str(b.get("id"))
                if not u or not v or u == v:
                    continue

                for sp in (a, b):
                    sid = str(sp.get("id"))
                    if sid not in G:
                        G.add_node(
                            sid,
                            name=sp.get("name"),
                            lat=sp.get("lat"),
                            lon=sp.get("lon"),
                        )

                lon1, lat1 = a.get("lon"), a.get("lat")
                lon2, lat2 = b.get("lon"), b.get("lat")
                w = haversine_km(lon1, lat1, lon2, lat2)

                if G.has_edge(u, v):
                    G[u][v]["routes"].add(line_id)
                    G[u][v]["is_superloop"] = G[u][v]["is_superloop"] or rs_is_sl
                    if not math.isnan(w) and not math.isnan(G[u][v].get("w_km", float("nan"))):
                        # keep the smaller if multiple (should be same)
                        G[u][v]["w_km"] = min(G[u][v]["w_km"], w)
                    elif not math.isnan(w):
                        G[u][v]["w_km"] = w
                else:
                    G.add_edge(u, v, routes=set([line_id]), is_superloop=rs_is_sl, w_km=w)

    return G, sl_lines, sl_nodes


def attach_boroughs(G: nx.Graph, boroughs_path: Path, borough_name_col: str) -> dict[str, str]:
    boroughs = gpd.read_file(boroughs_path)
    if boroughs.crs is None:
        raise ValueError("Borough file has no CRS. (Your GPKG should, so this is unexpected.)")

    boroughs_wgs = boroughs.to_crs("EPSG:4326").copy()
    # build inner/outer lookup from ons_inner if available
    inner_lookup = {}
    if "ons_inner" in boroughs_wgs.columns:
        for _, row in boroughs_wgs.iterrows():
            inner_lookup[str(row[borough_name_col])] = str(row["ons_inner"])

    rows = []
    for n, d in G.nodes(data=True):
        lon, lat = d.get("lon"), d.get("lat")
        if lon is None or lat is None:
            continue
        rows.append({"node": n, "geometry": Point(float(lon), float(lat))})
    pts = gpd.GeoDataFrame(rows, crs="EPSG:4326")

    joined = gpd.sjoin(pts, boroughs_wgs[[borough_name_col, "geometry"]], how="left", predicate="within")
    node_to_borough = dict(zip(joined["node"], joined[borough_name_col].astype("string")))

    # set attrs
    nx.set_node_attributes(G, node_to_borough, "borough")
    if inner_lookup:
        nx.set_node_attributes(
            G,
            {n: inner_lookup.get(b, None) for n, b in node_to_borough.items()},
            "ons_inner",
        )
    return node_to_borough


def outer_subgraph(G: nx.Graph) -> nx.Graph:
    # Prefer ons_inner flag from the dataset: 'F' means outer for your file
    outer_nodes = [n for n, d in G.nodes(data=True) if d.get("ons_inner") == "F"]
    H = G.subgraph(outer_nodes).copy()
    # keep largest component (optional but makes path plots nicer)
    if H.number_of_nodes() == 0:
        raise ValueError("Outer subgraph is empty. Borough join probably failed.")
    lcc = max(nx.connected_components(H), key=len)
    return H.subgraph(lcc).copy()


def build_variants(G_outer: nx.Graph, sl_lines: set[str]) -> tuple[nx.Graph, nx.Graph, nx.Graph, list[tuple[str, str]]]:
    # edges exclusively superloop
    excl = []
    any_sl = []
    for u, v, d in G_outer.edges(data=True):
        routes = d.get("routes", set())
        if routes & sl_lines:
            any_sl.append((u, v))
        if routes and routes.issubset(sl_lines):
            excl.append((u, v))

    G_sl = G_outer
    G_base_excl = G_outer.copy()
    G_base_excl.remove_edges_from(excl)

    G_base_all = G_outer.copy()
    G_base_all.remove_edges_from(any_sl)

    return G_sl, G_base_excl, G_base_all, excl


def approx_L_and_efficiency(G: nx.Graph, n_sources: int = 300, seed: int = 7) -> tuple[float, float]:
    rng = random.Random(seed)
    nodes = list(G.nodes())
    if len(nodes) < 10:
        return float("nan"), float("nan")
    sources = rng.sample(nodes, min(n_sources, len(nodes)))

    d_sum = 0
    d_cnt = 0
    inv_sum = 0.0
    inv_cnt = 0

    for s in sources:
        dist = nx.single_source_shortest_path_length(G, s)
        # drop self
        for t, d in dist.items():
            if t == s:
                continue
            d_sum += d
            d_cnt += 1
            inv_sum += 1.0 / d
            inv_cnt += 1

    L_hat = d_sum / d_cnt
    E_hat = inv_sum / inv_cnt
    return L_hat, E_hat


def ensure_dirs(outdir: Path):
    (outdir / "figures_plus").mkdir(parents=True, exist_ok=True)
    (outdir / "tables_plus").mkdir(parents=True, exist_ok=True)


def plot_map(G: nx.Graph, excl_edges: list[tuple[str, str]], out: Path):
    # Simple geographic plot: grey edges, red Superloop-exclusive edges
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # draw all edges (thin)
    for u, v in G.edges():
        du, dv = G.nodes[u], G.nodes[v]
        if du.get("lon") is None or dv.get("lon") is None:
            continue
        ax.plot([du["lon"], dv["lon"]], [du["lat"], dv["lat"]], linewidth=0.2, alpha=0.15)

    # draw superloop-exclusive edges (thicker)
    for u, v in excl_edges:
        du, dv = G.nodes[u], G.nodes[v]
        if du.get("lon") is None or dv.get("lon") is None:
            continue
        ax.plot([du["lon"], dv["lon"]], [du["lat"], dv["lat"]], linewidth=1.0, alpha=0.9)

    ax.set_title("Outer London bus stop graph (L-space): Superloop-exclusive edges overlaid")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_distance_cdfs(G_base: nx.Graph, G_sl: nx.Graph, out1: Path, out2: Path, n_sources: int = 200, seed: int = 7):
    rng = random.Random(seed)
    nodes = list(G_base.nodes())
    sources = rng.sample(nodes, min(n_sources, len(nodes)))

    d_base_all = []
    d_sl_all = []
    delta_all = []

    for s in sources:
        db = nx.single_source_shortest_path_length(G_base, s)
        ds = nx.single_source_shortest_path_length(G_sl, s)
        # align on targets reachable in both
        common = set(db.keys()) & set(ds.keys())
        common.discard(s)
        for t in common:
            d_base_all.append(db[t])
            d_sl_all.append(ds[t])
            delta_all.append(db[t] - ds[t])

    # CDF plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for arr, label in [(d_base_all, "Baseline (Superloop removed)"),
                       (d_sl_all, "With Superloop")]:
        x = np.sort(np.array(arr))
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, label=label)
    ax.set_xlabel("Shortest-path distance (hops)")
    ax.set_ylabel("CDF")
    ax.set_title("Distribution of shortest-path hop distances (sampled pairs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out1, dpi=220)
    plt.close(fig)

    # Delta distribution (improvement)
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.array(delta_all)
    ax.hist(x, bins=60)
    ax.set_xlabel(r"Improvement $\Delta d = d_{base} - d_{SL}$ (hops)")
    ax.set_ylabel("Count")
    ax.set_title("How much Superloop shortens paths (sampled pairs)")
    fig.tight_layout()
    fig.savefig(out2, dpi=220)
    plt.close(fig)


def random_shortcut_benchmark(G_base: nx.Graph, added_edges: list[tuple[str, str]], out: Path, reps: int = 30, seed: int = 7):
    rng = random.Random(seed)
    nodes = list(G_base.nodes())
    target_m = len(added_edges)

    # target distance distribution (km) from the actual added edges
    target_lengths = []
    for u, v in added_edges:
        du, dv = G_base.nodes.get(u, {}), G_base.nodes.get(v, {})
        if du.get("lon") is None or dv.get("lon") is None:
            continue
        target_lengths.append(haversine_km(du["lon"], du["lat"], dv["lon"], dv["lat"]))
    target_lengths = [x for x in target_lengths if not math.isnan(x) and x > 0]
    if not target_lengths:
        target_lengths = [2.0] * target_m  # fallback

    # real improvement
    L0, E0 = approx_L_and_efficiency(G_base, n_sources=250, seed=seed)
    G_real = G_base.copy()
    G_real.add_edges_from(added_edges)
    Lr, Er = approx_L_and_efficiency(G_real, n_sources=250, seed=seed)
    real_deltaE = Er - E0

    deltas = []
    for r in range(reps):
        H = G_base.copy()
        added = 0
        tries = 0
        # distance-matched-ish sampling with tolerance
        while added < target_m and tries < target_m * 400:
            tries += 1
            a, b = rng.sample(nodes, 2)
            if H.has_edge(a, b) or a == b:
                continue
            da, db = H.nodes[a], H.nodes[b]
            if da.get("lon") is None or db.get("lon") is None:
                continue
            d = haversine_km(da["lon"], da["lat"], db["lon"], db["lat"])
            Ltarget = rng.choice(target_lengths)
            if Ltarget <= 0:
                continue
            if abs(d - Ltarget) / Ltarget <= 0.25:  # 25% tolerance
                H.add_edge(a, b)
                added += 1

        # if we failed to match enough, just fill randomly
        while added < target_m:
            a, b = rng.sample(nodes, 2)
            if H.has_edge(a, b) or a == b:
                continue
            H.add_edge(a, b)
            added += 1

        _, E = approx_L_and_efficiency(H, n_sources=250, seed=seed + r + 1)
        deltas.append(E - E0)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(deltas, bins=25)
    ax.axvline(real_deltaE, linewidth=2.0)
    ax.set_xlabel(r"Efficiency gain $\Delta E$ from adding edges")
    ax.set_ylabel("Count")
    ax.set_title("Random shortcut benchmark (same number of edges as Superloop)")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def betweenness_plots(G: nx.Graph, sl_nodes: set[str], out: Path, seed: int = 7):
    # Approx betweenness (sampling) to keep it fast
    k = min(2000, G.number_of_nodes())
    bc = nx.betweenness_centrality(G, k=k, seed=seed, normalized=True)

    sl = np.array([bc[n] for n in G.nodes() if n in sl_nodes])
    non = np.array([bc[n] for n in G.nodes() if n not in sl_nodes])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([non, sl], labels=["Non-SL nodes", "SL-served nodes"], showfliers=False)
    ax.set_yscale("log")
    ax.set_ylabel("Betweenness centrality (log scale)")
    ax.set_title("Superloop nodes are structural bridges (approx betweenness)")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)

    # Save top nodes table
    top = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:30]
    rows = []
    for n, val in top:
        d = G.nodes[n]
        rows.append({
            "stop_id": n,
            "name": d.get("name"),
            "borough": d.get("borough"),
            "betweenness": val,
            "deg": G.degree(n),
            "is_superloop_node": n in sl_nodes,
        })
    pd.DataFrame(rows).to_csv(out.parent.parent / "tables_plus" / "top_betweenness_nodes.csv", index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--boroughs-path", required=True)
    p.add_argument("--borough-name-col", default="name")
    p.add_argument("--cache-dir", default="data/cache_tfl")
    p.add_argument("--outdir", default="outputs")
    args = p.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    ensure_dirs(outdir)

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    print(f"Loading cached route sequences from: {cache_dir}")
    seqs = load_cached_sequences(cache_dir)

    print("Building stop graph from cached sequences...")
    G_all, sl_lines, sl_nodes = build_graph_from_sequences(seqs)

    print("Attaching boroughs and extracting Outer London (ons_inner == 'F')...")
    attach_boroughs(G_all, Path(args.boroughs_path).expanduser().resolve(), args.borough_name_col)
    G_outer = outer_subgraph(G_all)

    print(f"Outer graph (LCC): N={G_outer.number_of_nodes()}, M={G_outer.number_of_edges()}")

    print("Building network variants (with SL / baseline removals)...")
    G_sl, G_base_excl, G_base_all, excl_edges = build_variants(G_outer, sl_lines)
    added_edges = excl_edges[:]  # edges that appear due to Superloop only

    # Quick headline metrics
    L0, E0 = approx_L_and_efficiency(G_base_excl, n_sources=300, seed=7)
    L1, E1 = approx_L_and_efficiency(G_sl, n_sources=300, seed=7)
    summary = pd.DataFrame([
        {"graph": "baseline_exclusive_removed", "N": G_base_excl.number_of_nodes(), "M": G_base_excl.number_of_edges(), "L_hat": L0, "E_hat": E0},
        {"graph": "with_superloop",            "N": G_sl.number_of_nodes(),       "M": G_sl.number_of_edges(),       "L_hat": L1, "E_hat": E1},
        {"graph": "delta", "N": 0, "M": len(added_edges), "L_hat": L1 - L0, "E_hat": E1 - E0},
    ])
    summary.to_csv(outdir / "tables_plus" / "headline_metrics.csv", index=False)

    # Figures
    figdir = outdir / "figures_plus"
    print("Plotting map...")
    plot_map(G_sl, excl_edges, figdir / "outer_map_superloop_overlay.png")

    print("Plotting distance CDFs and delta distribution...")
    plot_distance_cdfs(G_base_excl, G_sl,
                       figdir / "pathlength_cdf.png",
                       figdir / "path_improvement_hist.png",
                       n_sources=200)

    print("Running random-shortcut benchmark (same edge count as Superloop)...")
    random_shortcut_benchmark(G_base_excl, added_edges, figdir / "random_shortcut_benchmark.png", reps=30)

    print("Betweenness + top nodes table...")
    betweenness_plots(G_sl, sl_nodes, figdir / "betweenness_boxplot.png")

    print("Done. Wrote outputs to:", outdir)


if __name__ == "__main__":
    main()
