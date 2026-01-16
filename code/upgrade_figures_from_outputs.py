#!/usr/bin/env python3
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import geopandas as gpd
except Exception:
    gpd = None

# Optional basemap
try:
    import contextily as ctx
except Exception:
    ctx = None


CENTRE_LATLON = (51.5074, -0.1278)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def gini(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def cliffs_delta(a, b):
    """Effect size: P(a>b)-P(a<b). Cheap approximation using ranks if huge."""
    a = np.asarray(a); b = np.asarray(b)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    # If massive, subsample for speed
    rng = np.random.default_rng(42)
    if len(a) * len(b) > 3_000_000:
        a = rng.choice(a, size=min(2000, len(a)), replace=False)
        b = rng.choice(b, size=min(2000, len(b)), replace=False)
    gt = 0
    lt = 0
    for x in a:
        gt += np.sum(x > b)
        lt += np.sum(x < b)
    return (gt - lt) / (len(a) * len(b))


def load_outputs(outdir: Path):
    tables = outdir / "tables"
    need = {
        "node_uplift": tables / "node_uplift.csv",
        "headline": tables / "headline_metrics.csv",
    }
    missing = [k for k, p in need.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing expected files in outputs/tables: "
            + ", ".join([str(need[k]) for k in missing])
        )

    node_uplift = pd.read_csv(need["node_uplift"])
    headline = pd.read_csv(need["headline"])
    return node_uplift, headline


def save_lorenz(delta, outpath: Path):
    x = np.asarray(delta, dtype=float)
    x = x[np.isfinite(x)]
    x = np.clip(x, 0, None)
    x = np.sort(x)
    if x.sum() == 0:
        return
    cum = np.cumsum(x) / x.sum()
    p = np.arange(1, len(x) + 1) / len(x)
    G = gini(x)

    plt.figure(figsize=(6.2, 6.2))
    plt.plot(np.r_[0, p], np.r_[0, cum], linewidth=2)
    plt.plot([0, 1], [0, 1], linewidth=1)
    plt.title(f"Lorenz curve of node uplift (Gini = {G:.3f})")
    plt.xlabel("Share of nodes")
    plt.ylabel("Share of total uplift")
    plt.tight_layout()
    plt.savefig(outpath, dpi=250)
    plt.close()


def violin_sl_vs_non(df, outpath: Path):
    sl = df.loc[df["is_sl_served"] == True, "delta_mean_d"].to_numpy()
    non = df.loc[df["is_sl_served"] == False, "delta_mean_d"].to_numpy()
    cd = cliffs_delta(sl, non)

    plt.figure(figsize=(7.5, 4.5))
    parts = plt.violinplot([non[np.isfinite(non)], sl[np.isfinite(sl)]],
                           showmeans=False, showmedians=True, showextrema=False)
    plt.xticks([1, 2], ["Non-SL nodes", "SL-served nodes"])
    plt.ylabel("Mean hop-distance reduction Δd (baseline − with SL)")
    plt.title(f"Who benefits? Uplift distribution (Cliff’s δ ≈ {cd:.3f})")
    plt.tight_layout()
    plt.savefig(outpath, dpi=250)
    plt.close()


def radial_uplift(df, outpath: Path):
    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    delta = df["delta_mean_d"].to_numpy()
    r = np.array([haversine_km(lat[i], lon[i], CENTRE_LATLON[0], CENTRE_LATLON[1]) for i in range(len(df))])

    ok = np.isfinite(r) & np.isfinite(delta)
    r = r[ok]; delta = delta[ok]

    # Bin medians for a clean trend line
    bins = np.linspace(r.min(), r.max(), 25)
    b = np.digitize(r, bins)
    med = []
    mid = []
    for k in range(1, len(bins)):
        vals = delta[b == k]
        if len(vals) < 30:
            continue
        med.append(np.median(vals))
        mid.append(0.5 * (bins[k - 1] + bins[k]))

    plt.figure(figsize=(7.5, 4.8))
    plt.scatter(r, delta, s=6, alpha=0.08)
    if len(mid) > 0:
        plt.plot(mid, med, linewidth=3)
    plt.xlabel("Distance from centre (km)")
    plt.ylabel("Mean hop-distance reduction Δd")
    plt.title("Radial profile: where the Superloop helps most")
    plt.tight_layout()
    plt.savefig(outpath, dpi=250)
    plt.close()


def hexbin_map(df, boroughs_gdf, outpath: Path):
    # boroughs_gdf can be None; we still plot the hexbin
    lon = df["lon"].to_numpy()
    lat = df["lat"].to_numpy()
    delta = df["delta_mean_d"].to_numpy()

    ok = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(delta)
    lon, lat, delta = lon[ok], lat[ok], delta[ok]

    plt.figure(figsize=(8.5, 8.5))
    hb = plt.hexbin(lon, lat, C=delta, gridsize=55, reduce_C_function=np.mean, mincnt=10)
    plt.colorbar(hb, label="Mean Δd per hex (hops)")
    if boroughs_gdf is not None:
        # Plot outlines in lon/lat
        try:
            boroughs_gdf.to_crs(4326).boundary.plot(ax=plt.gca(), linewidth=0.6, alpha=0.9)
        except Exception:
            pass
    plt.title("Spatial heatmap of Superloop uplift (Outer London nodes)")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(outpath, dpi=250)
    plt.close()


def borough_choropleth(df, boroughs_path, borough_name_col, outpath: Path, outpath_bar: Path):
    if gpd is None:
        return None

    bsum = (
        df.groupby("borough", dropna=False)
          .agg(
              mean_uplift=("delta_mean_d", "mean"),
              median_uplift=("delta_mean_d", "median"),
              p90=("delta_mean_d", lambda x: np.nanpercentile(x, 90)),
              n=("delta_mean_d", "size"),
              sl_share=("is_sl_served", "mean"),
          )
          .reset_index()
    )

    gdf = gpd.read_file(boroughs_path)
    gdf[borough_name_col] = gdf[borough_name_col].astype(str)
    bsum["borough"] = bsum["borough"].astype(str)

    # Only Outer London if ons_inner exists
    if "ons_inner" in gdf.columns:
        gdf = gdf[gdf["ons_inner"].astype(str) == "F"].copy()

    gdf = gdf.merge(bsum, left_on=borough_name_col, right_on="borough", how="left")

    # Map
    # Use web mercator if basemap available, else keep native CRS
    ax = None
    if ctx is not None:
        gdf3857 = gdf.to_crs(3857)
        ax = gdf3857.plot(column="mean_uplift", legend=True, figsize=(9, 9), linewidth=0.6, edgecolor="white")
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, attribution_size=6)
        except Exception:
            pass
        ax.set_axis_off()
        ax.set_title("Mean hop-distance reduction by borough (Outer London)")
        plt.tight_layout()
        plt.savefig(outpath, dpi=250, bbox_inches="tight")
        plt.close()
    else:
        ax = gdf.to_crs(4326).plot(column="mean_uplift", legend=True, figsize=(9, 9), linewidth=0.6, edgecolor="white")
        ax.set_axis_off()
        ax.set_title("Mean hop-distance reduction by borough (Outer London)")
        plt.tight_layout()
        plt.savefig(outpath, dpi=250, bbox_inches="tight")
        plt.close()

    # Ranked bar (top 20 by mean uplift)
    top = bsum.sort_values("mean_uplift", ascending=False).head(20).iloc[::-1]
    plt.figure(figsize=(9.5, 6.8))
    plt.barh(top["borough"], top["mean_uplift"])
    plt.xlabel("Mean Δd (hops)")
    plt.title("Top boroughs by average uplift (Outer London)")
    plt.tight_layout()
    plt.savefig(outpath_bar, dpi=250)
    plt.close()

    return bsum


def coverage_vs_uplift(bsum, outpath: Path):
    if bsum is None or bsum.empty:
        return
    plt.figure(figsize=(7.2, 5.2))
    x = bsum["sl_share"].to_numpy()
    y = bsum["mean_uplift"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    plt.scatter(x[ok], y[ok], s=40, alpha=0.9)
    # simple linear fit for a readable trend
    if ok.sum() >= 3:
        m, c = np.polyfit(x[ok], y[ok], 1)
        xx = np.linspace(x[ok].min(), x[ok].max(), 50)
        plt.plot(xx, m * xx + c, linewidth=2)
    plt.xlabel("Share of nodes SL-served in borough")
    plt.ylabel("Mean uplift Δd (hops)")
    plt.title("Does more Superloop coverage predict more benefit?")
    plt.tight_layout()
    plt.savefig(outpath, dpi=250)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Path to outputs directory (contains tables/ and figures/).")
    ap.add_argument("--boroughs-path", default="", help="Path to London_Boroughs.gpkg (optional but recommended).")
    ap.add_argument("--borough-name-col", default="name", help="Borough name column in the GPKG.")
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    figdir = outdir / "figures_upgrade"
    tabdir = outdir / "tables_upgrade"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)

    node_uplift, headline = load_outputs(outdir)

    # Basic sanity
    if "delta_mean_d" not in node_uplift.columns:
        raise ValueError("Expected column delta_mean_d in node_uplift.csv")

    # Lorenz + Gini
    save_lorenz(node_uplift["delta_mean_d"], figdir / "uplift_lorenz_gini.png")

    # SL vs non-SL distributions
    violin_sl_vs_non(node_uplift, figdir / "uplift_violin_sl_vs_non.png")

    # Radial profile
    radial_uplift(node_uplift, figdir / "uplift_vs_radius.png")

    # Borough boundaries (optional)
    boroughs_gdf = None
    if args.boroughs_path and gpd is not None:
        try:
            boroughs_gdf = gpd.read_file(args.boroughs_path)
            if "ons_inner" in boroughs_gdf.columns:
                boroughs_gdf = boroughs_gdf[boroughs_gdf["ons_inner"].astype(str) == "F"].copy()
        except Exception:
            boroughs_gdf = None

    # Spatial hexbin heatmap
    hexbin_map(node_uplift, boroughs_gdf, figdir / "uplift_hexbin_map.png")

    # Borough choropleth + bar
    bsum = None
    if args.boroughs_path:
        bsum = borough_choropleth(
            node_uplift,
            args.boroughs_path,
            args.borough_name_col,
            figdir / "borough_uplift_map_clean.png",
            figdir / "borough_uplift_bar_top20.png",
        )

    # Coverage vs uplift
    coverage_vs_uplift(bsum, figdir / "coverage_vs_uplift.png")

    # Save summary tables you can cite in the report
    overall = {
        "n_nodes": int(node_uplift.shape[0]),
        "mean_uplift": float(np.nanmean(node_uplift["delta_mean_d"])),
        "median_uplift": float(np.nanmedian(node_uplift["delta_mean_d"])),
        "p90_uplift": float(np.nanpercentile(node_uplift["delta_mean_d"], 90)),
        "gini_uplift": float(gini(node_uplift["delta_mean_d"])),
        "sl_share_nodes": float(node_uplift["is_sl_served"].mean()),
    }
    pd.DataFrame([overall]).to_csv(tabdir / "uplift_overall_summary.csv", index=False)
    headline.to_csv(tabdir / "headline_metrics_copy.csv", index=False)
    if bsum is not None:
        bsum.to_csv(tabdir / "uplift_by_borough_summary.csv", index=False)

    print("Done.")
    print("Wrote figures to:", figdir)
    print("Wrote tables to:", tabdir)


if __name__ == "__main__":
    main()
