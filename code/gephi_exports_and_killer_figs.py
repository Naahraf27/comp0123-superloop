#!/usr/bin/env python3
"""
Build Outer London stop graph from cached TfL Route Sequence JSONs,
export Gephi-ready graphs (.gexf), and generate higher-value figures:

- Borough→Borough uplift OD heatmap
- Uplift vs angular separation ("closing the loop" mechanism)
- Uplift vs baseline path length (2D density/hex)
- Rank-shift plot (approx betweenness: baseline vs with Superloop)

Assumes your cache lives at: data/cache_tfl (created by earlier pipeline runs).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def haversine_km(lon1, lat1, lon2, lat2) -> float:
    # Earth radius (km)
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def angle_from_centre(lon, lat, centre_lon, centre_lat) -> float:
    # Simple bearing-like angle in radians using lon/lat deltas (fine for London-scale)
    dx = (lon - centre_lon) * math.cos(math.radians(centre_lat))
    dy = (lat - centre_lat)
    return math.atan2(dy, dx)


def safe_bool_equal(val, target: str) -> bool:
    # Handles pd.NA / None safely
    if val is None:
        return False
    try:
        if pd.isna(val):
            return False
    except Exception:
        pass
    return str(val) == target


# -----------------------------
# Cache parsing
# -----------------------------
def iter_route_sequence_json(cache_dir: Path) -> Iterable[Path]:
    # Accept anything starting route_sequence_ and ending .json
    for p in sorted(cache_dir.glob("route_sequence_*.json")):
        yield p


def extract_stop_sequences(payload: dict) -> List[List[dict]]:
    """
    TfL Route Sequence response usually contains:
    payload["stopPointSequences"] -> list of { "stopPoint": [ ... ] }

    This returns list-of-stop-lists.
    """
    sps = payload.get("stopPointSequences", None)
    if not isinstance(sps, list):
        return []

    out: List[List[dict]] = []
    for seq in sps:
        if not isinstance(seq, dict):
            continue
        stops = seq.get("stopPoint", None)
        if isinstance(stops, list) and len(stops) >= 2:
            out.append([s for s in stops if isinstance(s, dict)])
    return out


def stop_id_name_latlon(stop: dict) -> Optional[Tuple[str, str, float, float]]:
    sid = stop.get("id")
    name = stop.get("commonName") or stop.get("name") or ""
    lat = stop.get("lat")
    lon = stop.get("lon")
    if not sid or lat is None or lon is None:
        return None
    try:
        return str(sid), str(name), float(lat), float(lon)
    except Exception:
        return None


def build_graph_from_cache(cache_dir: Path) -> Tuple[nx.Graph, Dict[Tuple[str, str], Set[str]], Dict[str, Set[str]], Set[str]]:
    """
    Returns:
    - G_all: nodes=stops, edges=adjacent stops along any route
    - edge_routes: (u,v)-> set(route_ids)
    - node_routes: node-> set(route_ids)
    - sl_stop_ids: stops that appear on ANY Superloop route (route id startswith 'sl')
    """
    G = nx.Graph()
    edge_routes: Dict[Tuple[str, str], Set[str]] = {}
    node_routes: Dict[str, Set[str]] = {}
    sl_stop_ids: Set[str] = set()

    files = list(iter_route_sequence_json(cache_dir))
    if not files:
        raise FileNotFoundError(f"No route_sequence_*.json found in: {cache_dir}")

    print(f"[build] Reading {len(files)} cached Route Sequence files from {cache_dir}")

    for i, fp in enumerate(files, 1):
        try:
            payload = json.loads(fp.read_text())
        except Exception:
            continue

        route_id = str(payload.get("lineId") or payload.get("lineName") or fp.stem.replace("route_sequence_", "")).lower()
        is_sl_route = route_id.startswith("sl")

        sequences = extract_stop_sequences(payload)
        if not sequences:
            continue

        for stops in sequences:
            parsed: List[Tuple[str, str, float, float]] = []
            for s in stops:
                t = stop_id_name_latlon(s)
                if t is not None:
                    parsed.append(t)

            # Add nodes
            for sid, name, lat, lon in parsed:
                if sid not in G:
                    G.add_node(sid, name=name, lat=lat, lon=lon)
                else:
                    # keep first name/coords, but don't blow up if slightly different
                    G.nodes[sid].setdefault("name", name)
                    G.nodes[sid].setdefault("lat", lat)
                    G.nodes[sid].setdefault("lon", lon)

                node_routes.setdefault(sid, set()).add(route_id)
                if is_sl_route:
                    sl_stop_ids.add(sid)

            # Add edges between consecutive stops
            for (u, _, _, _), (v, _, _, _) in zip(parsed[:-1], parsed[1:]):
                if u == v:
                    continue
                a, b = (u, v) if u < v else (v, u)
                edge_routes.setdefault((a, b), set()).add(route_id)
                if not G.has_edge(a, b):
                    # length in km based on node coords
                    lon1, lat1 = G.nodes[a]["lon"], G.nodes[a]["lat"]
                    lon2, lat2 = G.nodes[b]["lon"], G.nodes[b]["lat"]
                    L = haversine_km(lon1, lat1, lon2, lat2)
                    G.add_edge(a, b, length_km=float(L))

        if i % 50 == 0:
            print(f"[build] ... {i}/{len(files)}")

    # Attach route summary attrs to edges
    for (u, v), routes in edge_routes.items():
        if G.has_edge(u, v):
            rlist = sorted(routes)
            G.edges[u, v]["routes_n"] = int(len(rlist))
            G.edges[u, v]["routes"] = ";".join(rlist)
            # edge is "superloop edge" if any SL route uses it
            G.edges[u, v]["is_sl_edge"] = any(r.startswith("sl") for r in rlist)
            # "SL-exclusive" if ONLY SL routes use it
            G.edges[u, v]["is_sl_exclusive"] = all(r.startswith("sl") for r in rlist)

    # Node attrs: is_sl_served + route count
    for n in G.nodes:
        G.nodes[n]["is_sl_served"] = bool(n in sl_stop_ids)
        G.nodes[n]["routes_n"] = int(len(node_routes.get(n, set())))

    print(f"[build] Done. Nodes={G.number_of_nodes():,} Edges={G.number_of_edges():,}")
    return G, edge_routes, node_routes, sl_stop_ids


# -----------------------------
# Outer London filtering (borough polygons)
# -----------------------------
def attach_boroughs(G: nx.Graph, boroughs_path: Path, borough_name_col: str) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
    # Build node dataframe
    rows = []
    for n, d in G.nodes(data=True):
        rows.append(
            {
                "node": n,
                "stop_name": d.get("name", ""),
                "lat": d.get("lat", np.nan),
                "lon": d.get("lon", np.nan),
                "is_sl_served": bool(d.get("is_sl_served", False)),
                "routes_n": int(d.get("routes_n", 0)),
            }
        )
    df = pd.DataFrame(rows).dropna(subset=["lat", "lon"])

    # Borough polygons
    boroughs = gpd.read_file(boroughs_path)
    if borough_name_col not in boroughs.columns:
        raise ValueError(f"borough_name_col='{borough_name_col}' not in columns: {list(boroughs.columns)}")

    # Ensure we have ons_inner if present; if not, we still can proceed, but "outer" filter will be unavailable
    has_ons_inner = "ons_inner" in boroughs.columns

    # Create points
    gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df["lon"], df["lat"])], crs="EPSG:4326")
    # Boroughs in EPSG:27700; project points
    gdf_27700 = gdf.to_crs(boroughs.crs)

    joined = gpd.sjoin(gdf_27700, boroughs[[borough_name_col] + (["ons_inner"] if has_ons_inner else []) + ["geometry"]], how="left", predicate="within")
    
    # GeoPandas may suffix columns if both sides share the same name.
    candidates = [
        borough_name_col,
        f"{borough_name_col}_right",
        f"{borough_name_col}_bor",
    ]

    boro_src = None
    for c in candidates:
        if c in joined.columns:
            boro_src = c
            break

    if boro_src is None:
        joined["borough"] = np.nan
    else:
        joined = joined.rename(columns={boro_src: "borough"})


    if has_ons_inner:
        joined["is_outer"] = joined["ons_inner"].apply(lambda x: safe_bool_equal(x, "F"))
    else:
        joined["ons_inner"] = None
        joined["is_outer"] = True  # fallback: keep all if no ons_inner field

    # push back to node df in WGS84
    joined_wgs = joined.to_crs("EPSG:4326").drop(columns=["index_right"], errors="ignore")
    return pd.DataFrame(joined_wgs.drop(columns="geometry")), boroughs


def induced_subgraph_with_node_attrs(G: nx.Graph, keep_nodes: Set[str], node_attr_df: pd.DataFrame) -> nx.Graph:
    H = G.subgraph(keep_nodes).copy()
    # attach borough attrs
    m = node_attr_df.set_index("node").to_dict(orient="index")
    for n in H.nodes:
        dd = m.get(n, {})
        # basic Gephi-friendly attrs
        H.nodes[n]["borough"] = str(dd.get("borough", "")) if dd.get("borough") is not None else ""
        H.nodes[n]["ons_inner"] = str(dd.get("ons_inner", "")) if dd.get("ons_inner") is not None else ""
        H.nodes[n]["is_outer"] = bool(dd.get("is_outer", False))
    return H


def make_outer_graphs(G_all: nx.Graph, node_attr_df: pd.DataFrame) -> Tuple[nx.Graph, nx.Graph]:
    # Outer only
    outer_nodes = set(node_attr_df.loc[node_attr_df["is_outer"] == True, "node"].astype(str))
    G_outer_full = induced_subgraph_with_node_attrs(G_all, outer_nodes, node_attr_df)

    # Baseline = remove SL-exclusive edges
    G_outer_base = G_outer_full.copy()
    remove_edges = [(u, v) for u, v, d in G_outer_base.edges(data=True) if bool(d.get("is_sl_exclusive", False))]
    G_outer_base.remove_edges_from(remove_edges)

    print(f"[outer] Outer nodes={G_outer_full.number_of_nodes():,} edges(full)={G_outer_full.number_of_edges():,} edges(base)={G_outer_base.number_of_edges():,}")
    return G_outer_full, G_outer_base


# -----------------------------
# Gephi export
# -----------------------------
def write_gexf(G: nx.Graph, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # Ensure GEXF-compatible primitive types
    for n, d in G.nodes(data=True):
        for k, v in list(d.items()):
            if isinstance(v, (np.integer,)):
                d[k] = int(v)
            elif isinstance(v, (np.floating,)):
                d[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                d[k] = bool(v)
            elif v is None:
                d[k] = ""

    for u, v, d in G.edges(data=True):
        for k, val in list(d.items()):
            if isinstance(val, (np.integer,)):
                d[k] = int(val)
            elif isinstance(val, (np.floating,)):
                d[k] = float(val)
            elif isinstance(val, (np.bool_,)):
                d[k] = bool(val)
            elif val is None:
                d[k] = ""

    nx.write_gexf(G, outpath)
    print(f"[gephi] Wrote {outpath}")


def build_route_route_graph(node_routes: Dict[str, Set[str]], keep_nodes: Set[str], sl_routes: Set[str], min_shared: int = 10) -> nx.Graph:
    # route -> set(nodes)
    route_nodes: Dict[str, Set[str]] = {}
    for n, rs in node_routes.items():
        if n not in keep_nodes:
            continue
        for r in rs:
            route_nodes.setdefault(r, set()).add(n)

    routes = sorted(route_nodes.keys())
    H = nx.Graph()
    for r in routes:
        H.add_node(r, is_sl_route=bool(r in sl_routes), stops_n=int(len(route_nodes[r])))

    # naive O(R^2) is fine for your scale; filter by min_shared
    for i in range(len(routes)):
        a = routes[i]
        A = route_nodes[a]
        for j in range(i + 1, len(routes)):
            b = routes[j]
            B = route_nodes[b]
            shared = len(A & B)
            if shared >= min_shared:
                H.add_edge(a, b, shared_stops=int(shared))
    return H


# -----------------------------
# Sampling + “killer” figures
# -----------------------------
def sample_pairs(G_base: nx.Graph, G_sl: nx.Graph, node_df: pd.DataFrame, n_sources: int, targets_per_source: int, seed: int,
                 centre_lon: float, centre_lat: float) -> pd.DataFrame:
    rng = random.Random(seed)

    # Work on the LCC to avoid infinities dominating everything
    lcc_nodes = max(nx.connected_components(G_sl), key=len)
    nodes = list(lcc_nodes)

    # Map attrs
    info = node_df.set_index("node")[["borough", "lat", "lon", "is_sl_served"]].to_dict(orient="index")

    sources = rng.sample(nodes, k=min(n_sources, len(nodes)))
    records = []

    print(f"[sample] BFS from {len(sources)} sources; {targets_per_source} targets/source")

    for si, s in enumerate(sources, 1):
        dist0 = nx.single_source_shortest_path_length(G_base, s)
        dist1 = nx.single_source_shortest_path_length(G_sl, s)

        # valid targets exist in both
        valid = [t for t in nodes if t != s and t in dist0 and t in dist1]
        if not valid:
            continue

        chosen = rng.sample(valid, k=min(targets_per_source, len(valid)))
        s_inf = info.get(s, {})
        s_lon, s_lat = float(s_inf.get("lon", np.nan)), float(s_inf.get("lat", np.nan))
        s_ang = angle_from_centre(s_lon, s_lat, centre_lon, centre_lat)
        s_b = str(s_inf.get("borough", ""))

        for t in chosen:
            t_inf = info.get(t, {})
            t_lon, t_lat = float(t_inf.get("lon", np.nan)), float(t_inf.get("lat", np.nan))
            t_ang = angle_from_centre(t_lon, t_lat, centre_lon, centre_lat)
            angdiff = abs((t_ang - s_ang + math.pi) % (2 * math.pi) - math.pi)  # [0,pi]

            d0 = int(dist0[t])
            d1 = int(dist1[t])
            dd = d0 - d1

            records.append(
                {
                    "src": s,
                    "dst": t,
                    "src_borough": s_b,
                    "dst_borough": str(t_inf.get("borough", "")),
                    "d_base": d0,
                    "d_sl": d1,
                    "delta_d": dd,
                    "angdiff_rad": angdiff,
                    "angdiff_deg": angdiff * 180 / math.pi,
                    "src_is_sl_served": bool(s_inf.get("is_sl_served", False)),
                    "dst_is_sl_served": bool(t_inf.get("is_sl_served", False)),
                }
            )

        if si % 25 == 0:
            print(f"[sample] ... {si}/{len(sources)}")

    df = pd.DataFrame(records)
    print(f"[sample] Done. Pair samples: {len(df):,}")
    return df


def plot_od_heatmap(pairs: pd.DataFrame, outpath: Path, top_n: int = 25) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # pick most frequent boroughs to keep readable
    counts = pd.concat([pairs["src_borough"], pairs["dst_borough"]]).value_counts()
    keep = list(counts.head(top_n).index)

    sub = pairs[pairs["src_borough"].isin(keep) & pairs["dst_borough"].isin(keep)].copy()
    mat = sub.pivot_table(index="src_borough", columns="dst_borough", values="delta_d", aggfunc="mean")

    # order by overall mean uplift
    order = sub.groupby("src_borough")["delta_d"].mean().sort_values(ascending=False).index.tolist()
    mat = mat.reindex(index=order, columns=order)

    plt.figure(figsize=(11, 9), dpi=200)
    plt.imshow(mat.to_numpy(), aspect="auto")
    plt.colorbar(label="Mean uplift Δd (hops)")
    plt.xticks(range(len(mat.columns)), mat.columns, rotation=90)
    plt.yticks(range(len(mat.index)), mat.index)
    plt.title("Borough → Borough mean uplift (sampled OD pairs)")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"[fig] Wrote {outpath}")


def plot_angular_profile(pairs: pd.DataFrame, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    bins = np.linspace(0, 180, 19)  # 10-degree bins
    pairs = pairs.copy()
    pairs["bin"] = pd.cut(pairs["angdiff_deg"], bins=bins, include_lowest=True)

    agg = pairs.groupby("bin")["delta_d"].agg(["mean", "median", "count"]).reset_index()
    mids = [b.mid for b in agg["bin"]]

    plt.figure(figsize=(9, 5), dpi=200)
    plt.plot(mids, agg["mean"], linewidth=2)
    plt.scatter(mids, agg["mean"], s=15)
    plt.xlabel("Angular separation around centre (degrees)")
    plt.ylabel("Mean uplift Δd (hops)")
    plt.title("Mechanism: Superloop helps more when trips are 'around' London (high angular separation)")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"[fig] Wrote {outpath}")


def plot_uplift_vs_baseline(pairs: pd.DataFrame, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    x = pairs["d_base"].to_numpy()
    y = pairs["delta_d"].to_numpy()

    plt.figure(figsize=(8, 6), dpi=200)
    plt.hexbin(x, y, gridsize=45, bins="log", mincnt=1)
    plt.colorbar(label="log10(count)")
    plt.xlabel("Baseline shortest-path distance d_base (hops)")
    plt.ylabel("Uplift Δd = d_base − d_SL (hops)")
    plt.title("Who gains most? Uplift grows with baseline trip complexity")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"[fig] Wrote {outpath}")


def approx_betweenness(G: nx.Graph, k: int, seed: int) -> Dict[str, float]:
    k = min(k, G.number_of_nodes())
    return nx.betweenness_centrality(G, k=k, seed=seed, normalized=True)


def plot_rank_shift(G_base: nx.Graph, G_sl: nx.Graph, outpath: Path, k: int, seed: int) -> pd.DataFrame:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # Use LCC of SL graph
    lcc = set(max(nx.connected_components(G_sl), key=len))
    Gb = G_base.subgraph(lcc).copy()
    Gs = G_sl.subgraph(lcc).copy()

    print(f"[rank] Approximating betweenness with k={min(k, len(lcc))} samples on LCC size={len(lcc):,}")

    b0 = approx_betweenness(Gb, k=k, seed=seed)
    b1 = approx_betweenness(Gs, k=k, seed=seed)

    df = pd.DataFrame(
        {
            "node": list(lcc),
            "b_base": [b0.get(n, 0.0) for n in lcc],
            "b_sl": [b1.get(n, 0.0) for n in lcc],
            "is_sl_served": [bool(Gs.nodes[n].get("is_sl_served", False)) for n in lcc],
            "borough": [str(Gs.nodes[n].get("borough", "")) for n in lcc],
        }
    )
    df["rank_base"] = df["b_base"].rank(ascending=False, method="average")
    df["rank_sl"] = df["b_sl"].rank(ascending=False, method="average")
    df["rank_gain"] = df["rank_base"] - df["rank_sl"]  # positive = moved up (more central)

    # Plot in log space (add tiny eps)
    eps = 1e-12
    x = (df["b_base"] + eps).to_numpy()
    y = (df["b_sl"] + eps).to_numpy()

    plt.figure(figsize=(7, 7), dpi=200)
    plt.scatter(x, y, s=6, alpha=0.25)
    # diagonal
    mn, mx = min(x.min(), y.min()), max(x.max(), y.max())
    plt.plot([mn, mx], [mn, mx], linewidth=1)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Betweenness (baseline, approx)")
    plt.ylabel("Betweenness (with Superloop, approx)")
    plt.title("Centrality re-ordering: which stops become more 'between' others?")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"[fig] Wrote {outpath}")

    return df.sort_values("rank_gain", ascending=False)


def add_python_communities(G: nx.Graph, seed: int) -> None:
    # Louvain built into NetworkX (3.x). Adds community id for Gephi colouring.
    try:
        from networkx.algorithms.community import louvain_communities
        comms = louvain_communities(G, seed=seed)
        for i, c in enumerate(comms):
            for n in c:
                G.nodes[n]["community"] = int(i)
    except Exception:
        # fallback: no community attribute
        pass

def make_full_baseline_graph(G_all, edge_routes, sl_prefix="sl"):
    """
    Return baseline graph where SL-exclusive edges are removed.
    SL-exclusive = edge served by >=1 Superloop route and by no non-Superloop route.
    Route IDs in this project are typically lowercase (e.g. 'sl1'), so matching is case-insensitive.
    """
    sl_prefix = sl_prefix.lower()

    def is_sl(r):
        return str(r).lower().startswith(sl_prefix)

    G_base = G_all.copy()
    remove = []

    for u, v in G_all.edges():
        routes = edge_routes.get((u, v)) or edge_routes.get((v, u)) or set()
        if not routes:
            continue  # safety

        has_sl = any(is_sl(r) for r in routes)
        only_sl = all(is_sl(r) for r in routes)

        sl_exclusive = has_sl and only_sl
        # Optional: keep attribute for debugging
        G_base.edges[u, v]["is_sl_exclusive"] = sl_exclusive

        if sl_exclusive:
            remove.append((u, v))

    G_base.remove_edges_from(remove)
    print(f"[full-baseline] Removed {len(remove)} SL-exclusive edges from full London graph")
    return G_base




# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data/cache_tfl", help="Directory containing route_sequence_*.json cache files")
    ap.add_argument("--boroughs-path", required=True, help="Path to London_Boroughs.gpkg")
    ap.add_argument("--borough-name-col", default="name", help="Borough name column in the GPKG")
    ap.add_argument("--outdir", required=True, help="Your outputs directory (e.g., outputs)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-sources", type=int, default=250)
    ap.add_argument("--targets-per-source", type=int, default=200)
    ap.add_argument("--btw-k", type=int, default=900)
    # Charing Cross-ish (good enough for this)
    ap.add_argument("--centre-lon", type=float, default=-0.1278)
    ap.add_argument("--centre-lat", type=float, default=51.5074)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    outdir = Path(args.outdir)
    figdir = outdir / "figures_killer"
    graphdir = outdir / "graphs"
    tabdir = outdir / "tables_killer"
    figdir.mkdir(parents=True, exist_ok=True)
    graphdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)

    G_all, edge_routes, node_routes, sl_stop_ids = build_graph_from_cache(cache_dir)
    G_base_all = make_full_baseline_graph(G_all, edge_routes)


    node_df, boroughs = attach_boroughs(G_all, Path(args.boroughs_path), args.borough_name_col)
    # Keep only nodes that successfully joined a borough (avoid blank boroughs)
    node_df = node_df.dropna(subset=["borough"])
    G_outer_full, G_outer_base = make_outer_graphs(G_all, node_df)

    # Add communities for Gephi colouring
    add_python_communities(G_outer_full, seed=args.seed)
    add_python_communities(G_outer_base, seed=args.seed)

    # # Export stop graphs for Gephi
    # write_gexf(G_outer_full, graphdir / "outer_stops_with_superloop.gexf")
    # write_gexf(G_outer_base, graphdir / "outer_stops_baseline_no_sl_exclusive.gexf")

    # --- Full London exports (inner + outer)
# --- Full London exports (inner + outer)
        # --- Export stop graphs for Gephi (full + outer)
    write_gexf(G_all,      graphdir / "all_stops_with_superloop.gexf")
    write_gexf(G_base_all, graphdir / "all_stops_baseline_no_sl_exclusive.gexf")

    write_gexf(G_outer_full, graphdir / "outer_stops_with_superloop.gexf")
    write_gexf(G_outer_base, graphdir / "outer_stops_baseline_no_sl_exclusive.gexf")



    # Export route-route graph for Gephi (often looks amazing in ForceAtlas2)
    keep_nodes = set(G_outer_full.nodes())
    sl_routes = {r for rs in node_routes.values() for r in rs if r.startswith("sl")}
    R = build_route_route_graph(node_routes, keep_nodes=keep_nodes, sl_routes=sl_routes, min_shared=10)
    write_gexf(R, graphdir / "outer_route_route_shared_stops.gexf")

    # Pair sampling + tables
    pairs = sample_pairs(
        G_base=G_outer_base,
        G_sl=G_outer_full,
        node_df=node_df,
        n_sources=args.n_sources,
        targets_per_source=args.targets_per_source,
        seed=args.seed,
        centre_lon=args.centre_lon,
        centre_lat=args.centre_lat,
    )
    pairs.to_csv(tabdir / "pair_samples.csv", index=False)
    print(f"[table] Wrote {tabdir / 'pair_samples.csv'}")

    # Killer figures
    plot_od_heatmap(pairs, figdir / "od_uplift_heatmap.png", top_n=25)
    plot_angular_profile(pairs, figdir / "uplift_vs_angular_separation.png")
    plot_uplift_vs_baseline(pairs, figdir / "uplift_vs_baseline_distance_hex.png")

    movers = plot_rank_shift(G_outer_base, G_outer_full, figdir / "betweenness_rank_shift.png", k=args.btw_k, seed=args.seed)
    movers.head(200).to_csv(tabdir / "top_rank_gainers.csv", index=False)
    print(f"[table] Wrote {tabdir / 'top_rank_gainers.csv'}")

    # quick headline table for your write-up
    headline = pd.DataFrame(
        [
            {
                "graph": "baseline_no_sl_exclusive",
                "nodes": G_outer_base.number_of_nodes(),
                "edges": G_outer_base.number_of_edges(),
                "components": nx.number_connected_components(G_outer_base),
            },
            {
                "graph": "with_superloop",
                "nodes": G_outer_full.number_of_nodes(),
                "edges": G_outer_full.number_of_edges(),
                "components": nx.number_connected_components(G_outer_full),
            },
        ]
    )
    headline.to_csv(tabdir / "graph_sizes.csv", index=False)
    print(f"[table] Wrote {tabdir / 'graph_sizes.csv'}")

    print("\nDone.")
    print(f"Gephi graphs: {graphdir}")
    print(f"Figures: {figdir}")
    print(f"Tables: {tabdir}")


if __name__ == "__main__":
    main()
