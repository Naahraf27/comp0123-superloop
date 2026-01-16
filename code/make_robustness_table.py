#!/usr/bin/env python3
import argparse
import math
import random
from statistics import mean, median

import networkx as nx


def largest_cc_nodes(G: nx.Graph):
    return max(nx.connected_components(G), key=len)


def add_haversine_weights_km(G: nx.Graph, lat_key="lat", lon_key="lon", out_key="w_km"):
    # Pure python haversine to avoid extra deps.
    # Assumes node attributes contain lat/lon in decimal degrees.
    import math

    def hav_km(lat1, lon1, lat2, lon2):
        R = 6371.0088
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
        return 2 * R * math.asin(math.sqrt(a))

    missing = 0
    for u, v, data in G.edges(data=True):
        try:
            lat1 = float(G.nodes[u][lat_key])
            lon1 = float(G.nodes[u][lon_key])
            lat2 = float(G.nodes[v][lat_key])
            lon2 = float(G.nodes[v][lon_key])
            data[out_key] = hav_km(lat1, lon1, lat2, lon2)
        except Exception:
            missing += 1
            data[out_key] = 1.0  # fallback to avoid crashes
    if missing:
        print(f"[warn] {missing} edges missing lat/lon; set {out_key}=1.0 fallback.")


def sample_pairs(od_nodes, S, seed=0, origins_cap=2000):
    rng = random.Random(seed)
    od_nodes = list(od_nodes)
    n = len(od_nodes)

    # Pick a manageable number of origins to limit BFS/Dijkstra runs.
    num_origins = min(origins_cap, n)
    origins = rng.sample(od_nodes, k=num_origins)

    per_origin = int(math.ceil(S / num_origins))
    pairs_by_origin = {}
    for o in origins:
        ds = []
        for _ in range(per_origin):
            d = od_nodes[rng.randrange(n)]
            while d == o:
                d = od_nodes[rng.randrange(n)]
            ds.append(d)
        pairs_by_origin[o] = ds
    return pairs_by_origin


def compute_metrics(H_base, H_sl, od_nodes, S, seed, weighted=False, w_key="w_km"):
    pairs_by_origin = sample_pairs(od_nodes, S=S, seed=seed, origins_cap=2000)

    d_base_all = []
    d_sl_all = []
    delta_all = []

    improved = 0
    total = 0

    for o, dests in pairs_by_origin.items():
        if weighted:
            dist_base = nx.single_source_dijkstra_path_length(H_base, o, weight=w_key)
            dist_sl = nx.single_source_dijkstra_path_length(H_sl, o, weight=w_key)
        else:
            dist_base = nx.single_source_shortest_path_length(H_base, o)
            dist_sl = nx.single_source_shortest_path_length(H_sl, o)

        for t in dests:
            if t not in dist_base or t not in dist_sl:
                # Should be rare if OD nodes come from intersection of LCC node sets,
                # but keep safe.
                continue
            db = dist_base[t]
            ds = dist_sl[t]
            if db == 0 or ds == 0:
                continue

            d_base_all.append(db)
            d_sl_all.append(ds)

            delta = db - ds
            delta_all.append(delta)

            total += 1
            if delta > 0:
                improved += 1

    Lb = mean(d_base_all)
    Ls = mean(d_sl_all)
    Eb = mean([1.0 / x for x in d_base_all])
    Es = mean([1.0 / x for x in d_sl_all])

    out = {
        "S_effective": total,
        "L_base": Lb,
        "L_sl": Ls,
        "L_change_pct": 100.0 * (Ls - Lb) / Lb,
        "E_base": Eb,
        "E_sl": Es,
        "E_change_pct": 100.0 * (Es - Eb) / Eb,
        "share_improved": improved / total if total else float("nan"),
        "delta_mean": mean(delta_all) if delta_all else float("nan"),
        "delta_median": median(delta_all) if delta_all else float("nan"),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_outer", required=True)
    ap.add_argument("--sl_outer", required=True)
    ap.add_argument("--base_full", required=True)
    ap.add_argument("--sl_full", required=True)
    ap.add_argument("--S", type=int, default=49800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", default="outputs/tables/robustness_variants.csv")
    args = ap.parse_args()

    # Load graphs
    Gbo = nx.read_gexf(args.base_outer)
    Gso = nx.read_gexf(args.sl_outer)
    Gbf = nx.read_gexf(args.base_full)
    Gsf = nx.read_gexf(args.sl_full)

    # Outer OD set from outer graphs (no borough attrs needed)
    outer_nodes = set(Gso.nodes()) & set(Gbo.nodes())

    # Build LCC subgraphs for each
    Hbo = Gbo.subgraph(largest_cc_nodes(Gbo)).copy()
    Hso = Gso.subgraph(largest_cc_nodes(Gso)).copy()
    Hbf = Gbf.subgraph(largest_cc_nodes(Gbf)).copy()
    Hsf = Gsf.subgraph(largest_cc_nodes(Gsf)).copy()

    # OD nodes must exist in BOTH graphs for the variant
    od_outer = outer_nodes & set(Hbo.nodes()) & set(Hso.nodes())
    od_full = outer_nodes & set(Hbf.nodes()) & set(Hsf.nodes())

    # Add weights for weighted variants
    add_haversine_weights_km(Hbo)
    add_haversine_weights_km(Hso)
    add_haversine_weights_km(Hbf)
    add_haversine_weights_km(Hsf)

    rows = []

    def add_row(name, H_base, H_sl, od_nodes, cost):
        m = compute_metrics(
            H_base, H_sl, od_nodes,
            S=args.S, seed=args.seed,
            weighted=(cost == "km"),
            w_key="w_km"
        )
        rows.append({
            "variant": name,
            "paths_on": "outer_induced" if "Outer" in name else "full_london",
            "od_set": "outer_stops",
            "edge_cost": cost,
            **m
        })

    add_row("Main (Outer induced, hops)", Hbo, Hso, od_outer, "hops")
    add_row("Robustness 1 (Full London, hops)", Hbf, Hsf, od_full, "hops")
    add_row("Robustness 2 (Outer induced, km)", Hbo, Hso, od_outer, "km")
    add_row("Robustness 3 (Full London, km)", Hbf, Hsf, od_full, "km")

    # Write CSV
    import csv
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {args.out_csv}")
    for r in rows:
        print(r["variant"], "L:", round(r["L_base"], 3), "->", round(r["L_sl"], 3),
              "E:", round(r["E_base"], 6), "->", round(r["E_sl"], 6))

if __name__ == "__main__":
    main()
