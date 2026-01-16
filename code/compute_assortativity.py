#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BASE = ROOT / "outputs/graphs/outer_stops_baseline_no_sl_exclusive.gexf"
DEFAULT_SL   = ROOT / "outputs/graphs/outer_stops_with_superloop.gexf"

def load_graph(path: Path) -> nx.Graph:
    G = nx.read_gexf(path)
    # ensure simple undirected graph for assortativity
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        H = nx.Graph()
        H.add_nodes_from(G.nodes(data=True))
        for u, v in G.edges():
            if u != v:
                H.add_edge(u, v)
        G = H
    if G.is_directed():
        G = G.to_undirected()
    return G

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--sl", type=Path, default=DEFAULT_SL)
    args = ap.parse_args()

    if not args.base.exists():
        raise FileNotFoundError(f"Missing {args.base}")
    if not args.sl.exists():
        raise FileNotFoundError(f"Missing {args.sl}")

    Gb = load_graph(args.base)
    Gs = load_graph(args.sl)

    rb = nx.degree_assortativity_coefficient(Gb)
    rs = nx.degree_assortativity_coefficient(Gs)

    print(f"Base graph: {args.base} | N={Gb.number_of_nodes():,} M={Gb.number_of_edges():,}")
    print(f"SL   graph: {args.sl}   | N={Gs.number_of_nodes():,} M={Gs.number_of_edges():,}")
    print(f"Assortativity r (base): {rb:.4f}")
    print(f"Assortativity r (SL)  : {rs:.4f}")

if __name__ == "__main__":
    main()
