#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import networkx as nx

DEFAULT_L_BASE = 106.65
DEFAULT_L_SL   = 48.21

def norm_cols(df):
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def find_graphs(graph_dir="outputs/graphs"):
    files = []
    for fn in os.listdir(graph_dir):
        if fn.lower().endswith((".gexf", ".graphml", ".gpickle")):
            files.append(os.path.join(graph_dir, fn))
    if not files:
        raise FileNotFoundError(f"No graph files found in {graph_dir}")

    def is_base(p):
        s = os.path.basename(p).lower()
        return ("baseline" in s) or ("no_sl" in s) or ("no-superloop" in s) or ("no_superloop" in s)

    def is_sl(p):
        s = os.path.basename(p).lower()
        # must look like SL graph but not baseline/no_sl
        return (("sl" in s) or ("superloop" in s) or ("with" in s)) and (not is_base(p))

    base_candidates = [p for p in files if is_base(p)]
    sl_candidates   = [p for p in files if is_sl(p)]

    base = base_candidates[0] if base_candidates else None
    sl   = sl_candidates[0] if sl_candidates else None

    return base, sl, files

def load_graph(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".gexf":
        G = nx.read_gexf(path)
    elif ext == ".graphml":
        G = nx.read_graphml(path)
    elif ext == ".gpickle":
        import pickle
        with open(path, "rb") as f:
            G = pickle.load(f)
    else:
        raise ValueError(f"Unsupported graph format: {path}")

    # simplify to undirected simple graph
    if G.is_directed():
        G = G.to_undirected(as_view=False)
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        G = nx.Graph(G)

    # keep LCC
    if not nx.is_connected(G):
        lcc = max(nx.connected_components(G), key=len)
        G = G.subgraph(lcc).copy()

    return G

def try_get_L_from_headline(headline_path):
    if not headline_path or not os.path.exists(headline_path):
        return None

    df = pd.read_csv(headline_path)
    df = norm_cols(df)

    # Try wide format: columns include graph + L
    if "graph" in df.columns:
        # accept l, l_hat, path_length
        for lcol in ["l", "l_hat", "path_length", "path_length_hat", "l_est"]:
            if lcol in df.columns:
                out = {}
                for _, r in df.iterrows():
                    out[str(r["graph"]).lower()] = float(r[lcol])
                return out

    # Try long format: columns metric, base, sl
    if "metric" in df.columns:
        metric = df["metric"].astype(str).str.lower()
        # find a row that looks like characteristic path length
        mask = metric.str.contains("path") | metric.str.contains("l_hat") | metric.str.fullmatch("l")
        cand = df[mask]
        if len(cand) >= 1:
            row = cand.iloc[0]
            out = {}
            if "base" in df.columns: out["base"] = float(row["base"])
            if "sl"   in df.columns: out["sl"]   = float(row["sl"])
            if "g_base" in df.columns: out["base"] = float(row["g_base"])
            if "g_sl" in df.columns: out["sl"] = float(row["g_sl"])
            return out if out else None

    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small_world", default="outputs/tables/small_world.csv")
    ap.add_argument("--headline", default="outputs/tables/headline_metrics.csv")
    ap.add_argument("--out_tex", default="outputs/tables/nulls_table.tex")
    ap.add_argument("--graph_dir", default="outputs/graphs")
    ap.add_argument("--R", type=int, default=50, help="Number of null-model realisations (put what you actually used)")
    args = ap.parse_args()

    sw = pd.read_csv(args.small_world)
    sw = norm_cols(sw)

    required = {"graph", "c_er", "l_er", "sigma"}
    missing = required - set(sw.columns)
    if missing:
        raise ValueError(f"{args.small_world} missing columns {sorted(missing)}. Found {list(sw.columns)}")

    # Load graphs and compute empirical clustering
    base_path, sl_path, all_files = find_graphs(args.graph_dir)
    if not base_path or not sl_path or base_path == sl_path:
        raise ValueError(
            "Could not detect TWO distinct graph files for base and SL.\n"
            f"Detected base={base_path}\nDetected sl={sl_path}\n"
            f"All graph files: {all_files}\n"
            "Rename your SL graph file to include 'with_sl' or 'superloop' (and not 'no_sl'), "
            "or pass/adjust detection."
        )

    G_base = load_graph(base_path)
    G_sl   = load_graph(sl_path)

    C_base = nx.average_clustering(G_base)
    C_sl   = nx.average_clustering(G_sl)

    # empirical L: try headline_metrics.csv else fallback to your reported numbers
    L_map = try_get_L_from_headline(args.headline) or {}
    L_base = float(L_map.get("base", DEFAULT_L_BASE))
    L_sl   = float(L_map.get("sl", DEFAULT_L_SL))

    # Map small_world graph labels -> base/sl rows
    def row_for(name_options):
        for opt in name_options:
            m = sw["graph"].astype(str).str.lower() == opt
            if m.any():
                return sw[m].iloc[0]
        return None

    # --- Robust row matching: use edge counts (M) to map rows to base/SL.
    # We expect ER baselines were built with matched (N, M), so rows correspond to a specific M.
    M_base = G_base.number_of_edges()
    M_sl   = G_sl.number_of_edges()

    def parse_m_from_label(label: str):
        # Try to extract an integer from a label like "N12693_M15026" or "M=15026" etc.
        import re
        s = str(label)
        m = re.search(r"(?:^|[^0-9])m(?:=|_)?\s*([0-9]{3,})", s, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        # fallback: any long-ish integer token
        nums = re.findall(r"[0-9]{4,}", s)
        if nums:
            # take the last big number (often M)
            return int(nums[-1])
        return None

    # Try mapping by parsing M from the graph label first (common if you encoded M in the name)
    sw["_m_guess"] = sw["graph"].apply(parse_m_from_label)

    row_base = None
    row_sl = None

    if sw["_m_guess"].notna().any():
        # choose rows whose guessed M matches
        cand_base = sw[sw["_m_guess"] == M_base]
        cand_sl   = sw[sw["_m_guess"] == M_sl]
        if len(cand_base) >= 1:
            row_base = cand_base.iloc[0]
        if len(cand_sl) >= 1:
            row_sl = cand_sl.iloc[0]

    # If labels don't contain M, fall back to a simple heuristic:
    # baseline has smaller M than SL, so take the two rows and assign by sigma row order
    # (only works if small_world.csv contains exactly two rows for your two graphs)
    if row_base is None or row_sl is None:
        if len(sw) == 2:
            # assume first row corresponds to baseline (smaller M) IF your generator wrote baseline first
            # safer: assume baseline has larger L_er typically? no. Just use file order fallback.
            row_base = sw.iloc[0]
            row_sl   = sw.iloc[1]
        else:
            raise ValueError(
                "Couldn't match rows in small_world.csv to base/SL.\n"
                "Fix options:\n"
                "  (1) Edit outputs/tables/small_world.csv so graph labels are 'base' and 'sl' (recommended), OR\n"
                "  (2) Include M in the graph label (e.g., 'base_M15026', 'sl_M15201'), OR\n"
                f"  (3) Ensure small_world.csv contains exactly 2 rows (yours has {len(sw)} rows).\n"
                f"Detected M_base={M_base}, M_sl={M_sl}."
            )

    # clean helper column
    if "_m_guess" in sw.columns:
        sw.drop(columns=["_m_guess"], inplace=True)


    tab = pd.DataFrame([
        {"Graph": r"$G_{\mathrm{base}}$", "C": C_base, "L": L_base,
         "C_ER": float(row_base["c_er"]), "L_ER": float(row_base["l_er"]), "sigma": float(row_base["sigma"])},
        {"Graph": r"$G_{\mathrm{SL}}$",   "C": C_sl,   "L": L_sl,
         "C_ER": float(row_sl["c_er"]),   "L_ER": float(row_sl["l_er"]),   "sigma": float(row_sl["sigma"])},
    ])

    # write LaTeX tabular
    os.makedirs(os.path.dirname(args.out_tex), exist_ok=True)
    with open(args.out_tex, "w") as f:
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("Graph & $C$ & $L$ & $C_{\\mathrm{ER}}$ & $L_{\\mathrm{ER}}$ & $\\sigma$ \\\\\n")
        f.write("\\midrule\n")
        for _, r in tab.iterrows():
            f.write(f"{r['Graph']} & {r['C']:.4f} & {r['L']:.2f} & {r['C_ER']:.4f} & {r['L_ER']:.2f} & {r['sigma']:.2f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    print(f"Wrote {args.out_tex}")
    print(f"NOTE: set R={args.R} in your LaTeX caption/methods (this script doesn't infer it).")
    print("Graphs used:")
    print("  base:", base_path)
    print("  sl  :", sl_path)

if __name__ == "__main__":
    main()
