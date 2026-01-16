import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

BASE = "outputs/graphs/outer_stops_baseline_no_sl_exclusive.gexf"
SL   = "outputs/graphs/outer_stops_with_superloop.gexf"
OUT  = "figures/degree_ccdf_knn.png"

def lcc(G):
    if nx.is_connected(G):
        return G
    cc = max(nx.connected_components(G), key=len)
    return G.subgraph(cc).copy()

def degree_ccdf(G):
    degs = [d for _, d in G.degree()]
    c = Counter(degs)
    ks = np.array(sorted(c.keys()))
    counts = np.array([c[k] for k in ks], dtype=float)
    pmf = counts / counts.sum()
    ccdf = 1.0 - np.cumsum(pmf) + pmf  # P(K>=k)
    return ks, ccdf

def knn_by_k(G):
    # average neighbor degree per node, then average by degree k
    annd = nx.average_neighbor_degree(G)
    deg = dict(G.degree())
    buckets = {}
    for node, k in deg.items():
        buckets.setdefault(k, []).append(annd[node])
    ks = np.array(sorted(buckets.keys()))
    knn = np.array([np.mean(buckets[k]) for k in ks])
    return ks, knn

def main():
    G0 = lcc(nx.read_gexf(BASE))
    G1 = lcc(nx.read_gexf(SL))

    ks0, ccdf0 = degree_ccdf(G0)
    ks1, ccdf1 = degree_ccdf(G1)
    k0, knn0 = knn_by_k(G0)
    k1, knn1 = knn_by_k(G1)

    fig = plt.figure(figsize=(10,4))

    ax1 = plt.subplot(1,2,1)
    ax1.loglog(ks0, ccdf0, marker='o', linestyle='none', markersize=3, label="Baseline")
    ax1.loglog(ks1, ccdf1, marker='o', linestyle='none', markersize=3, label="With Superloop")
    ax1.set_xlabel("Degree k")
    ax1.set_ylabel("CCDF P(K ≥ k)")
    ax1.legend(frameon=False)

    ax2 = plt.subplot(1,2,2)
    ax2.loglog(k0, knn0, marker='o', linestyle='none', markersize=3, label="Baseline")
    ax2.loglog(k1, knn1, marker='o', linestyle='none', markersize=3, label="With Superloop")
    ax2.set_xlabel("Degree k")
    ax2.set_ylabel(r"$k_{nn}(k)$")
    ax2.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(OUT, dpi=300)
    print("Wrote", OUT)

if __name__ == "__main__":
    main()
