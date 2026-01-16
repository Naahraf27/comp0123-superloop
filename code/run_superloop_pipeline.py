# run_superloop_pipeline.py
# Python 3.11+ recommended

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import networkx as nx
import matplotlib.pyplot as plt

import geopandas as gpd
from shapely.geometry import Point



OUTER_BOROUGHS = {
    "Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley", "Croydon",
    "Ealing", "Enfield", "Harrow", "Havering", "Hillingdon", "Hounslow",
    "Kingston upon Thames", "Merton", "Redbridge", "Richmond upon Thames",
    "Sutton", "Waltham Forest"
}

TFL_BASE = "https://api.tfl.gov.uk"

def haversine_km(lat1, lon1, lat2, lon2):
    # WGS84 haversine
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))



def tfl_get(session, url, params, retries=5, backoff=1.5):
    for attempt in range(retries):
        r = session.get(url, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        # backoff on rate-limit-ish or transient
        sleep_s = (backoff ** attempt) + random.random()
        time.sleep(sleep_s)
    raise RuntimeError(f"TfL request failed after retries: {url} -> {r.status_code} {r.text[:200]}")

def fetch_lines(session, app_id, app_key):
    url = f"{TFL_BASE}/Line/Mode/bus"
    params = {"app_id": app_id, "app_key": app_key}
    lines = tfl_get(session, url, params=params)
    # Keep only regular “bus” lines (filter out weird empties defensively)
    lines = [ln for ln in lines if ln.get("id") and ln.get("name")]
    return lines

def fetch_route_sequence(session, line_id, app_id, app_key):
    url = f"{TFL_BASE}/Line/{line_id}/Route/Sequence/all"
    params = {"app_id": app_id, "app_key": app_key, "serviceTypes": "Regular"}
    return tfl_get(session, url, params=params)

def load_or_fetch_all_sequences(lines, cache_dir, app_id, app_key):
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "superloop-network-study/1.0"})

    sequences = {}
    for ln in lines:
        lid = ln["id"]
        cpath = cache_dir / f"route_sequence_{lid}.json"
        if cpath.exists():
            sequences[lid] = json.loads(cpath.read_text(encoding="utf-8"))
            continue
        data = fetch_route_sequence(session, lid, app_id, app_key)
        cpath.write_text(json.dumps(data), encoding="utf-8")
        sequences[lid] = data
        # tiny polite pause
        time.sleep(0.1)
    return sequences


# Parsing RouteSequence payload
# (TfL payloads can be branchy; we just need ordered stop lists)


def extract_stop_lists(route_seq_json):
    """
    Returns list of ordered stop lists.
    Each stop is a dict with at least: id, name, lat, lon (when available).
    """
    out = []
    sps = route_seq_json.get("stopPointSequences", []) or []

    for seq in sps:
        # Typically: seq["stopPoint"] is a list of matched stops
        stops = None
        for key in ("stopPoint", "stopPoints", "stops", "matchedStops"):
            if key in seq and isinstance(seq[key], list):
                stops = seq[key]
                break
        if not stops:
            continue

        cleaned = []
        for st in stops:
            sid = st.get("id") or st.get("naptanId") or st.get("stationId")
            name = st.get("name")
            lat = st.get("lat") if "lat" in st else st.get("latitude")
            lon = st.get("lon") if "lon" in st else st.get("longitude")
            # Some variants nest coords, so be defensive
            if lat is None and isinstance(st.get("coord"), dict):
                lat = st["coord"].get("lat")
                lon = st["coord"].get("lon")
            if sid is None:
                continue
            cleaned.append({"id": sid, "name": name, "lat": lat, "lon": lon})
        if len(cleaned) >= 2:
            out.append(cleaned)
    return out


# Build graphs


def build_graph(lines, sequences_by_line):
    """
    Builds a simple undirected graph from consecutive stop pairs on each line.
    Stores:
      - node attrs: name, lat, lon
      - edge attrs: routes(set), is_superloop_edge(bool), w_km(float)
    """
    # Identify Superloop lines by name (safe: auto-detect, no hardcoded list)
    superloop_line_ids = set()
    line_name = {}
    for ln in lines:
        lid = ln["id"]
        nm = str(ln.get("name", ""))
        line_name[lid] = nm
        if nm.upper().startswith("SL"):
            superloop_line_ids.add(lid)

    G = nx.Graph()
    # Node registry to fill coords if missing in some sequences
    node_info = {}

    for lid, rs in sequences_by_line.items():
        stop_lists = extract_stop_lists(rs)
        for stops in stop_lists:
            # Add nodes
            for st in stops:
                sid = st["id"]
                if sid not in node_info:
                    node_info[sid] = st
                else:
                    # fill missing coords
                    if node_info[sid].get("lat") is None and st.get("lat") is not None:
                        node_info[sid]["lat"] = st["lat"]
                        node_info[sid]["lon"] = st["lon"]

            # Add edges from consecutive pairs
            for a, b in zip(stops[:-1], stops[1:]):
                u, v = a["id"], b["id"]
                if u == v:
                    continue
                if not G.has_edge(u, v):
                    G.add_edge(u, v, routes=set())
                G[u][v]["routes"].add(lid)

    # Finalise node attrs
    for sid, info in node_info.items():
        G.add_node(
            sid,
            name=info.get("name"),
            lat=info.get("lat"),
            lon=info.get("lon"),
        )

    # Edge attrs: superloop + distance weight
    for u, v, data in G.edges(data=True):
        routes = data["routes"]
        data["is_superloop_edge"] = any(r in superloop_line_ids for r in routes)
        lu, lo = G.nodes[u].get("lat"), G.nodes[u].get("lon")
        lv, lvlo = G.nodes[v].get("lat"), G.nodes[v].get("lon")
        if lu is not None and lv is not None:
            data["w_km"] = haversine_km(lu, lo, lv, lvlo)
        else:
            data["w_km"] = None

    return G, superloop_line_ids, line_name

# -----------------------------
# Outer London filtering
# -----------------------------

def attach_boroughs(G, boroughs_path, borough_name_col):
    """
    Spatially joins nodes to borough polygons.
    Expects boroughs_path to be a GeoPackage/GeoJSON/Shapefile readable by geopandas.
    """
    boroughs = gpd.read_file(boroughs_path)
    if borough_name_col not in boroughs.columns:
        raise ValueError(f"borough_name_col '{borough_name_col}' not in columns: {list(boroughs.columns)}")

    # Build points gdf (drop nodes without coords)
    rows = []
    for n, d in G.nodes(data=True):
        if d.get("lat") is None or d.get("lon") is None:
            continue
        rows.append({"id": n, "geometry": Point(d["lon"], d["lat"])})
    pts = gpd.GeoDataFrame(rows, crs="EPSG:4326")

    boroughs = boroughs.to_crs("EPSG:4326")
    joined = gpd.sjoin(pts, boroughs[[borough_name_col, "geometry"]], how="left", predicate="within")
    bmap = dict(zip(joined["id"], joined[borough_name_col]))

    for n in G.nodes():
        b = bmap.get(n)
        G.nodes[n]["borough"] = b
        G.nodes[n]["outer"] = (b in OUTER_BOROUGHS)

def induced_outer_graph(G):
    outer_nodes = [n for n, d in G.nodes(data=True) if d.get("outer")]
    H = G.subgraph(outer_nodes).copy()
    # Drop edges with missing weight if you want weighted metrics
    return H

# Metrics (fast approximations)

def approx_char_path_len(G, n_sources=300, seed=42):
    if G.number_of_nodes() < 2:
        return np.nan
    # use LCC
    comps = list(nx.connected_components(G))
    lcc = max(comps, key=len)
    H = G.subgraph(lcc)
    nodes = list(H.nodes())
    rng = random.Random(seed)
    sources = rng.sample(nodes, min(n_sources, len(nodes)))

    total = 0
    count = 0
    for s in sources:
        dist = nx.single_source_shortest_path_length(H, s)
        # exclude self distance 0
        total += sum(dist.values())
        count += (len(dist) - 1)
    return total / count if count else np.nan

def approx_efficiency(G, n_sources=300, seed=42, weight=None):
    if G.number_of_nodes() < 2:
        return np.nan
    nodes = list(G.nodes())
    rng = random.Random(seed)
    sources = rng.sample(nodes, min(n_sources, len(nodes)))

    inv_sum = 0.0
    denom = (len(nodes) - 1) * len(sources)

    for s in sources:
        if weight:
            dist = nx.single_source_dijkstra_path_length(G, s, weight=weight)
        else:
            dist = nx.single_source_shortest_path_length(G, s)
        inv_sum += sum((1.0 / d) for t, d in dist.items() if t != s and d > 0)

    return inv_sum / denom if denom else np.nan

def basic_metrics(G):
    N = G.number_of_nodes()
    M = G.number_of_edges()
    degs = [d for _, d in G.degree()]
    return {
        "N": N,
        "M": M,
        "avg_degree": float(np.mean(degs)) if degs else np.nan,
        "density": nx.density(G) if N > 1 else np.nan,
        "clustering": nx.average_clustering(G) if N > 1 else np.nan,
        "assortativity_r": nx.degree_assortativity_coefficient(G) if M > 0 else np.nan,
        "lcc_frac": (len(max(nx.connected_components(G), key=len)) / N) if N else np.nan,
    }

def double_edge_swap_null(G, seed=42, nswap_mult=10):
    H = G.copy()
    m = H.number_of_edges()
    # nswap ~ 10M is usually plenty to mix
    nx.double_edge_swap(H, nswap=nswap_mult * m, max_tries=nswap_mult * m * 20, seed=seed)
    return H

def rich_club_phi(G, k):
    nodes = [n for n, d in G.degree() if d > k]
    n = len(nodes)
    if n < 2:
        return np.nan
    sub = G.subgraph(nodes)
    m = sub.number_of_edges()
    return (2 * m) / (n * (n - 1))

def permutation_test_mean(a, b, iters=5000, seed=42):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    obs = a.mean() - b.mean()
    combined = np.concatenate([a, b])
    na = len(a)
    hits = 0
    for _ in range(iters):
        rng.shuffle(combined)
        diff = combined[:na].mean() - combined[na:].mean()
        if abs(diff) >= abs(obs):
            hits += 1
    p = (hits + 1) / (iters + 1)
    return obs, p

# Plot helpers

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def plot_degree_ccdf(G, outpath):
    degs = np.array([d for _, d in G.degree()])
    degs = degs[degs > 0]
    xs = np.sort(np.unique(degs))
    ccdf = [(degs >= k).mean() for k in xs]
    plt.figure()
    plt.plot(xs, ccdf)
    plt.yscale("log")
    plt.xlabel("Degree k")
    plt.ylabel("CCDF P(K ≥ k)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=250)
    plt.close()

def plot_richclub(rho_df, outpath):
    plt.figure()
    plt.plot(rho_df["k"], rho_df["rho"])
    plt.axhline(1.0)
    plt.xlabel("k threshold")
    plt.ylabel("Normalised rich-club ρ(k)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=250)
    plt.close()

# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-id", default="", help="TfL app_id")
    ap.add_argument("--app-key", default="", help="TfL app_key")
    ap.add_argument("--boroughs-path", required=True, help="Path to borough boundaries (gpkg/geojson/shp)")
    ap.add_argument("--borough-name-col", default="NAME", help="Column containing borough name")
    ap.add_argument("--cache-dir", default="data/cache_tfl", help="Cache directory for API JSON")
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    figdir = outdir / "figures"
    tabdir = outdir / "tables"
    ensure_dir(figdir)
    ensure_dir(tabdir)

    # Fetch lines + route sequences
    session = requests.Session()
    session.headers.update({"User-Agent": "superloop-network-study/1.0"})
    lines = fetch_lines(session, args.app_id, args.app_key)

    sequences = load_or_fetch_all_sequences(
        lines, Path(args.cache_dir), args.app_id, args.app_key
    )

    # Build full bus graph (stop-level)
    G_all, superloop_line_ids, line_name = build_graph(lines, sequences)

    # Attach borough and filter to outer
    attach_boroughs(G_all, args.boroughs_path, args.borough_name_col)
    G_outer = induced_outer_graph(G_all)

    # Identify exclusive SL edges
    def is_exclusive_sl_edge(u, v, data):
        routes = data["routes"]
        return len(routes) > 0 and all(r in superloop_line_ids for r in routes)

    exclusive_edges = [(u, v) for u, v, d in G_outer.edges(data=True) if is_exclusive_sl_edge(u, v, d)]
    any_sl_edges = [(u, v) for u, v, d in G_outer.edges(data=True) if d.get("is_superloop_edge")]

    # Graphs: base (exclusive removed) and base (all SL removed), and full
    G_sl = G_outer.copy()

    G_base_excl = G_outer.copy()
    G_base_excl.remove_edges_from(exclusive_edges)

    G_base_all = G_outer.copy()
    G_base_all.remove_edges_from(any_sl_edges)

    # Core metrics
    def compute_suite(G, seed=42):
        m = basic_metrics(G)
        m["L_hat"] = approx_char_path_len(G, seed=seed)
        m["E_hat_unw"] = approx_efficiency(G, seed=seed, weight=None)
        # weighted efficiency (skip if weights missing)
        if all(G[u][v].get("w_km") is not None for u, v in list(G.edges())[:500]):  # quick check
            m["E_hat_w"] = approx_efficiency(G, seed=seed, weight="w_km")
        else:
            m["E_hat_w"] = np.nan
        return m

    res = []
    for name, G in [("G_base_exclusive_removed", G_base_excl),
                    ("G_base_allSL_removed", G_base_all),
                    ("G_SL_full", G_sl)]:
        row = compute_suite(G)
        row["graph"] = name
        res.append(row)

    metrics_df = pd.DataFrame(res).set_index("graph")
    metrics_df.to_csv(tabdir / "metrics_summary.csv")

    # ER small-world baseline (use G_base_excl for the headline comparison, plus repeat for G_SL if you want)
    def er_baseline(G, reps=25, seed=42):
        N, M = G.number_of_nodes(), G.number_of_edges()
        Cs, Ls = [], []
        for i in range(reps):
            R = nx.gnm_random_graph(N, M, seed=seed + i)
            Cs.append(nx.average_clustering(R))
            Ls.append(approx_char_path_len(R, seed=seed + i))
        return float(np.mean(Cs)), float(np.mean(Ls))

    C_er_base, L_er_base = er_baseline(G_base_excl)
    C_er_sl, L_er_sl = er_baseline(G_sl)

    # small-world sigma
    C_base = metrics_df.loc["G_base_exclusive_removed", "clustering"]
    L_base = metrics_df.loc["G_base_exclusive_removed", "L_hat"]
    C_sl_ = metrics_df.loc["G_SL_full", "clustering"]
    L_sl_ = metrics_df.loc["G_SL_full", "L_hat"]

    sigma_base = (C_base / C_er_base) / (L_base / L_er_base)
    sigma_sl = (C_sl_ / C_er_sl) / (L_sl_ / L_er_sl)

    sw_df = pd.DataFrame([
        {"graph": "G_base_exclusive_removed", "C_er": C_er_base, "L_er": L_er_base, "sigma": sigma_base},
        {"graph": "G_SL_full", "C_er": C_er_sl, "L_er": L_er_sl, "sigma": sigma_sl},
    ])
    sw_df.to_csv(tabdir / "small_world.csv", index=False)

    # Rich club (normalised via degree-preserving swaps)
    degs = np.array([d for _, d in G_sl.degree()])
    k_grid = np.unique(np.quantile(degs, np.linspace(0.50, 0.98, 25)).astype(int))

    # One null graph (you can average over many for smoother curves)
    G_null = double_edge_swap_null(G_sl, seed=123)

    rows = []
    for k in k_grid:
        phi_real = rich_club_phi(G_sl, int(k))
        phi_null = rich_club_phi(G_null, int(k))
        rho = (phi_real / phi_null) if (phi_real and phi_null and phi_null > 0) else np.nan
        rows.append({"k": int(k), "phi_real": phi_real, "phi_null": phi_null, "rho": rho})

    rho_df = pd.DataFrame(rows)
    rho_df.to_csv(tabdir / "rich_club.csv", index=False)
    plot_richclub(rho_df.dropna(), figdir / "richclub_rho.png")

    # Degree CCDF
    plot_degree_ccdf(G_sl, figdir / "degree_ccdf.png")

    # Centrality comparison: do Superloop-served nodes have higher betweenness?
    # Define served-by-SL nodes as endpoints of any SL edge
    sl_nodes = set()
    for u, v, d in G_sl.edges(data=True):
        if d.get("is_superloop_edge"):
            sl_nodes.add(u)
            sl_nodes.add(v)

    # Approx betweenness (sampling)
    btw = nx.betweenness_centrality(G_sl, k=min(2000, G_sl.number_of_nodes()), seed=42, normalized=True)
    sl_btw = [btw[n] for n in sl_nodes if n in btw]
    non_btw = [btw[n] for n in G_sl.nodes() if n not in sl_nodes]

    obs, p = permutation_test_mean(sl_btw, non_btw, iters=3000)
    pd.DataFrame([{"mean_diff_sl_minus_non": obs, "p_perm": p,
                   "n_sl": len(sl_btw), "n_non": len(non_btw)}]).to_csv(tabdir / "betweenness_permtest.csv", index=False)

    print("Done.")
    print(f"Wrote: {tabdir}/metrics_summary.csv, small_world.csv, rich_club.csv, betweenness_permtest.csv")
    print(f"Wrote figures: {figdir}/degree_ccdf.png, richclub_rho.png")

if __name__ == "__main__":
    main()
