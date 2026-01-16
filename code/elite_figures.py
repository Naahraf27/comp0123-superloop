#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

try:
    import contextily as cx
    HAS_CX = True
except Exception:
    HAS_CX = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False


def haversine_km(lon1, lat1, lon2, lat2) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_cached_sequences(cache_dir: Path) -> dict[str, dict]:
    seqs = {}
    files = sorted(cache_dir.glob("route_sequence_*.json"))
    if not files:
        raise FileNotFoundError(f"No route_sequence_*.json in {cache_dir}")
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        line_id = str(obj.get("lineId") or fp.stem.replace("route_sequence_", ""))
        seqs[line_id] = obj
    return seqs


def extract_stop_lists(route_sequence: dict) -> list[list[dict]]:
    out = []
    for sps in route_sequence.get("stopPointSequences", []) or []:
        stops = sps.get("stopPoint", []) or []
        if len(stops) >= 2:
            out.append(stops)
    return out


def is_superloop(route_sequence: dict) -> bool:
    return str(route_sequence.get("lineName", "")).upper().startswith("SL")


def build_graph(seqs: dict[str, dict]) -> tuple[nx.Graph, set[str], set[str]]:
    G = nx.Graph()
    sl_lines = {lid for lid, rs in seqs.items() if is_superloop(rs)}
    sl_nodes = set()

    for line_id, rs in seqs.items():
        rs_is_sl = line_id in sl_lines

        for stops in extract_stop_lists(rs):
            for sp in stops:
                sid = str(sp.get("id"))
                if rs_is_sl and sid:
                    sl_nodes.add(sid)

            for a, b in zip(stops, stops[1:]):
                u = str(a.get("id")); v = str(b.get("id"))
                if not u or not v or u == v:
                    continue

                for sp in (a, b):
                    sid = str(sp.get("id"))
                    if sid not in G:
                        G.add_node(
                            sid,
                            name=sp.get("name"),
                            lat=float(sp.get("lat")),
                            lon=float(sp.get("lon")),
                        )

                w = haversine_km(a["lon"], a["lat"], b["lon"], b["lat"])
                if G.has_edge(u, v):
                    G[u][v]["routes"].add(line_id)
                    G[u][v]["is_superloop"] = G[u][v]["is_superloop"] or rs_is_sl
                    G[u][v]["w_km"] = min(G[u][v]["w_km"], w)
                else:
                    G.add_edge(u, v, routes=set([line_id]), is_superloop=rs_is_sl, w_km=w)

    return G, sl_lines, sl_nodes


def attach_boroughs(G: nx.Graph, boroughs_path: Path, borough_name_col: str) -> gpd.GeoDataFrame:
    boroughs = gpd.read_file(boroughs_path)
    boroughs = boroughs.to_crs("EPSG:4326")

    pts = []
    for n, d in G.nodes(data=True):
        pts.append({"node": n, "geometry": Point(d["lon"], d["lat"])})
    pts = gpd.GeoDataFrame(pts, crs="EPSG:4326")

    joined = gpd.sjoin(pts, boroughs[[borough_name_col, "ons_inner", "geometry"]], how="left", predicate="within")
    node_to_b = dict(zip(joined["node"], joined[borough_name_col].astype("string")))
    node_to_inner = dict(zip(joined["node"], joined["ons_inner"].astype("string")))

    nx.set_node_attributes(G, node_to_b, "borough")
    nx.set_node_attributes(G, node_to_inner, "ons_inner")

    return boroughs


def outer_lcc(G: nx.Graph) -> nx.Graph:
    outer_nodes = [n for n, d in G.nodes(data=True) if str(d.get("ons_inner")) == "F"]
    H = G.subgraph(outer_nodes).copy()
    lcc = max(nx.connected_components(H), key=len)
    return H.subgraph(lcc).copy()


def build_variants(G_outer: nx.Graph, sl_lines: set[str]) -> tuple[nx.Graph, nx.Graph, list[tuple[str, str]]]:
    excl_edges = []
    for u, v, d in G_outer.edges(data=True):
        routes = d.get("routes", set())
        if routes and routes.issubset(sl_lines):
            excl_edges.append((u, v))

    G_sl = G_outer
    G_base = G_outer.copy()
    G_base.remove_edges_from(excl_edges)
    return G_sl, G_base, excl_edges


def ensure_dirs(outdir: Path):
    (outdir / "figures_elite").mkdir(parents=True, exist_ok=True)
    (outdir / "tables_elite").mkdir(parents=True, exist_ok=True)


def plot_map_pretty(G: nx.Graph, boroughs: gpd.GeoDataFrame, excl_edges: list[tuple[str, str]], out: Path):
    fig, ax = plt.subplots(figsize=(9, 9))

    # borough outline
    boroughs.boundary.plot(ax=ax, linewidth=0.6, alpha=0.5)

    # baseline edges (light grey)
    for u, v in G.edges():
        du, dv = G.nodes[u], G.nodes[v]
        ax.plot([du["lon"], dv["lon"]], [du["lat"], dv["lat"]], linewidth=0.15, alpha=0.10)

    # superloop-exclusive edges (red, thick)
    for u, v in excl_edges:
        du, dv = G.nodes[u], G.nodes[v]
        ax.plot([du["lon"], dv["lon"]], [du["lat"], dv["lat"]], linewidth=1.5, alpha=0.9)

    ax.set_title("Outer London bus stop graph: Superloop-exclusive edges (red)")
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")

    if HAS_CX:
        try:
            cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.CartoDB.PositronNoLabels, alpha=0.45)
        except Exception:
            pass

    fig.tight_layout()
    fig.savefig(out, dpi=260)
    plt.close(fig)


def sampled_pair_deltas(G_base: nx.Graph, G_sl: nx.Graph, n_sources: int = 400, seed: int = 7):
    rng = random.Random(seed)
    nodes = list(G_base.nodes())
    sources = rng.sample(nodes, min(n_sources, len(nodes)))

    iterator = tqdm(sources, desc="BFS sources") if HAS_TQDM else sources

    d_base = defaultdict(int)
    d_sl = defaultdict(int)
    cnt = defaultdict(int)

    for s in iterator:
        db = nx.single_source_shortest_path_length(G_base, s)
        ds = nx.single_source_shortest_path_length(G_sl, s)
        common = set(db.keys()) & set(ds.keys())
        common.discard(s)
        for t in common:
            d_base[t] += db[t]
            d_sl[t] += ds[t]
            cnt[t] += 1

    rows = []
    for t in cnt:
        rows.append({
            "node": t,
            "mean_d_base": d_base[t] / cnt[t],
            "mean_d_sl": d_sl[t] / cnt[t],
            "delta_mean_d": (d_base[t] - d_sl[t]) / cnt[t],
            "borough": G_sl.nodes[t].get("borough"),
            "lon": G_sl.nodes[t]["lon"],
            "lat": G_sl.nodes[t]["lat"],
        })

    return pd.DataFrame(rows)


def borough_uplift_plots(df: pd.DataFrame, boroughs: gpd.GeoDataFrame, borough_name_col: str, out_map: Path, out_bar: Path):
    # borough mean uplift
    b = (df.dropna(subset=["borough"])
           .groupby("borough")["delta_mean_d"]
           .mean()
           .sort_values(ascending=False)
           .reset_index())

    # bar
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(b["borough"], b["delta_mean_d"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean hop-distance reduction (sampled sources)")
    ax.set_title("Borough-level uplift from Superloop (higher = more benefit)")
    fig.tight_layout()
    fig.savefig(out_bar, dpi=260)
    plt.close(fig)

    # map (choropleth)
    boroughs2 = boroughs.copy()
    boroughs2["borough"] = boroughs2[borough_name_col].astype("string")
    boroughs2 = boroughs2.merge(b, on="borough", how="left")

    fig, ax = plt.subplots(figsize=(9, 9))
    boroughs2.plot(column="delta_mean_d", ax=ax, legend=True, alpha=0.85, linewidth=0.3, edgecolor="white")
    ax.set_axis_off()
    ax.set_title("Mean hop-distance reduction by borough (Outer London)")
    ax.set_aspect("equal", adjustable="box")

    if HAS_CX:
        try:
            boroughs2 = boroughs2.to_crs("EPSG:3857")
            ax.clear()
            boroughs2.plot(column="delta_mean_d", ax=ax, legend=True, alpha=0.85, linewidth=0.3, edgecolor="white")
            cx.add_basemap(ax, source=cx.providers.CartoDB.PositronNoLabels, alpha=0.45)
            ax.set_axis_off()
            ax.set_title("Mean hop-distance reduction by borough (Outer London)")
        except Exception:
            pass

    fig.tight_layout()
    fig.savefig(out_map, dpi=260)
    plt.close(fig)


def edge_length_vs_edge_betweenness(G: nx.Graph, excl_edges: list[tuple[str, str]], out: Path, seed: int = 7):
    # approximate edge betweenness for speed
    k = min(1200, G.number_of_nodes())
    eb = nx.edge_betweenness_centrality(G, k=k, seed=seed, normalized=True)

    xs = []
    ys = []
    is_sl = []

    excl_set = set(tuple(sorted(e)) for e in excl_edges)

    for (u, v), val in eb.items():
        d = G[u][v]
        w = d.get("w_km", np.nan)
        if not np.isfinite(w):
            continue
        xs.append(w)
        ys.append(val)
        is_sl.append(tuple(sorted((u, v))) in excl_set)

    xs = np.array(xs); ys = np.array(ys); is_sl = np.array(is_sl)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(xs[~is_sl], ys[~is_sl], s=6, alpha=0.15, label="Non-SL edges")
    ax.scatter(xs[is_sl], ys[is_sl], s=18, alpha=0.9, label="Superloop-exclusive edges")
    ax.set_yscale("log")
    ax.set_xlabel("Edge length (km)")
    ax.set_ylabel("Edge betweenness (log scale, approx)")
    ax.set_title("Superloop edges: long-range and structurally central")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=260)
    plt.close(fig)


def robustness_curves(G_base: nx.Graph, G_sl: nx.Graph, out: Path, seed: int = 7):
    # Compare LCC size under targeted removal by degree and by betweenness (approx)
    rng = random.Random(seed)

    def lcc_frac(H):
        if H.number_of_nodes() == 0:
            return 0.0
        return len(max(nx.connected_components(H), key=len)) / H.number_of_nodes()

    # metrics
    deg_base = dict(G_base.degree())
    deg_sl = dict(G_sl.degree())

    k = min(2000, G_sl.number_of_nodes())
    bc_base = nx.betweenness_centrality(G_base, k=k, seed=seed, normalized=True)
    bc_sl = nx.betweenness_centrality(G_sl, k=k, seed=seed, normalized=True)

    fracs = np.linspace(0, 0.2, 21)  # remove up to 20%
    curves = []

    for label, metric_base, metric_sl in [
        ("degree", deg_base, deg_sl),
        ("betweenness", bc_base, bc_sl),
    ]:
        order_base = [n for n, _ in sorted(metric_base.items(), key=lambda x: x[1], reverse=True)]
        order_sl = [n for n, _ in sorted(metric_sl.items(), key=lambda x: x[1], reverse=True)]

        yb = []
        ys = []
        for f in fracs:
            rm_b = set(order_base[: int(f * len(order_base))])
            rm_s = set(order_sl[: int(f * len(order_sl))])

            Hb = G_base.copy()
            Hs = G_sl.copy()
            Hb.remove_nodes_from(rm_b)
            Hs.remove_nodes_from(rm_s)

            yb.append(lcc_frac(Hb))
            ys.append(lcc_frac(Hs))

        curves.append((label, yb, ys))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for label, yb, ys in curves:
        ax.plot(fracs, yb, label=f"Baseline, remove by {label}")
        ax.plot(fracs, ys, label=f"With SL, remove by {label}")
    ax.set_xlabel("Fraction of nodes removed (targeted)")
    ax.set_ylabel("Largest component fraction")
    ax.set_title("Robustness under targeted removals (Outer London L-space)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=260)
    plt.close(fig)


def community_stitching(G: nx.Graph, excl_edges: list[tuple[str, str]], out: Path, seed: int = 7):
    comms = nx.community.louvain_communities(G, seed=seed)
    cid = {}
    for i, c in enumerate(comms):
        for n in c:
            cid[n] = i
    modularity = nx.community.modularity(G, comms)

    inter = 0
    for u, v in excl_edges:
        if cid.get(u) != cid.get(v):
            inter += 1

    frac_inter = inter / max(1, len(excl_edges))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(["Modularity Q", "Inter-community SL edge share"], [modularity, frac_inter])
    ax.set_ylim(0, 1)
    ax.set_title("Community structure and Superloop stitching")
    fig.tight_layout()
    fig.savefig(out, dpi=260)
    plt.close(fig)


def route_route_graph(seqs: dict[str, dict], outer_nodes: set[str], out: Path):
    # Build stop -> lines index for OUTER nodes only
    stop_to_lines = defaultdict(set)
    line_is_sl = {}
    for line_id, rs in seqs.items():
        line_is_sl[line_id] = is_superloop(rs)
        stops_set = set()
        for stops in extract_stop_lists(rs):
            for sp in stops:
                sid = str(sp.get("id"))
                if sid in outer_nodes:
                    stops_set.add(sid)
        for sid in stops_set:
            stop_to_lines[sid].add(line_id)

    # Count shared stops between line pairs (sparse)
    pair_w = defaultdict(int)
    for _, lines in stop_to_lines.items():
        lines = sorted(lines)
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                pair_w[(lines[i], lines[j])] += 1

    R = nx.Graph()
    for (a, b), w in pair_w.items():
        if w >= 10:  # threshold to keep it readable
            R.add_edge(a, b, weight=w)

    # centrality
    bc = nx.betweenness_centrality(R, weight="weight", normalized=True)
    for n in R.nodes():
        R.nodes[n]["bc"] = bc.get(n, 0.0)
        R.nodes[n]["is_sl"] = line_is_sl.get(n, False)

    pos = nx.spring_layout(R, seed=7, k=0.6 / math.sqrt(max(1, R.number_of_nodes())))

    fig, ax = plt.subplots(figsize=(9, 7))
    # edges
    for u, v, d in R.edges(data=True):
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], alpha=0.15, linewidth=0.5)

    # nodes
    xs = []; ys = []; sizes = []; sl_mask = []
    for n in R.nodes():
        xs.append(pos[n][0]); ys.append(pos[n][1])
        sizes.append(80 + 4000 * R.nodes[n]["bc"])
        sl_mask.append(R.nodes[n]["is_sl"])
    xs = np.array(xs); ys = np.array(ys); sizes = np.array(sizes); sl_mask = np.array(sl_mask)

    ax.scatter(xs[~sl_mask], ys[~sl_mask], s=sizes[~sl_mask], alpha=0.35)
    ax.scatter(xs[sl_mask], ys[sl_mask], s=sizes[sl_mask], alpha=0.95)
    ax.set_axis_off()
    ax.set_title("Route-to-route network (Outer London): Superloop routes highlighted")
    fig.tight_layout()
    fig.savefig(out, dpi=260)
    plt.close(fig)


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
    seqs = load_cached_sequences(cache_dir)

    G_all, sl_lines, sl_nodes = build_graph(seqs)
    boroughs = attach_boroughs(G_all, Path(args.boroughs_path).expanduser().resolve(), args.borough_name_col)

    G_outer = outer_lcc(G_all)
    G_sl, G_base, excl_edges = build_variants(G_outer, sl_lines)

    figdir = outdir / "figures_elite"
    tabdir = outdir / "tables_elite"

    # 1) Pretty map
    plot_map_pretty(G_sl, boroughs[boroughs["ons_inner"].astype("string").fillna("") == "F"].copy(), excl_edges, figdir / "map_superloop_pretty.png")

    # 2) Borough/node uplift
    df = sampled_pair_deltas(G_base, G_sl, n_sources=400)
    df.to_csv(tabdir / "node_uplift.csv", index=False)
    borough_uplift_plots(df, boroughs[boroughs["ons_inner"].astype("string").fillna("") == "F"].copy(), args.borough_name_col,
                         figdir / "borough_uplift_map.png",
                         figdir / "borough_uplift_bar.png")

    # 3) Edge length vs edge betweenness
    edge_length_vs_edge_betweenness(G_sl, excl_edges, figdir / "edge_length_vs_edge_betweenness.png")

    # 4) Robustness
    robustness_curves(G_base, G_sl, figdir / "robustness_targeted.png")

    # 5) Community stitching
    community_stitching(G_sl, excl_edges, figdir / "community_stitching.png")

    # 6) Route-to-route network
    route_route_graph(seqs, set(G_outer.nodes()), figdir / "route_route_network.png")

    print("Done. Wrote elite figures to:", figdir)
    print("Wrote tables to:", tabdir)


if __name__ == "__main__":
    main()
