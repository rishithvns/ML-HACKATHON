"""Dataset exploration (train + student_test partitions only).

Outputs -> reports/eda/
  inventory.csv            one row per trip: files, sizes, counts, gaps, extent, network connectivity
  partition_summary.csv    aggregates per partition
  missing_values.csv       NaN / invalid counts per file type
  data_quality.md          automatically flagged anomalies
  *.png                    plots
  map_<partition>_<trip>.html  Folium map: GPS, nearby road network, reference route

Run:  python -m scripts.eda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

from pathsnap.config import load_config, resolve  # noqa: E402
from pathsnap.dataset import attach_labels, build_partition, process_trip  # noqa: E402
from pathsnap.geo import LocalProjection, haversine  # noqa: E402
from pathsnap.io import list_trips, load_route  # noqa: E402
from pathsnap.metrics import route_continuity  # noqa: E402
from pathsnap.viz import trip_map  # noqa: E402

PARTS = ["train", "student_test"]


def raw_nan_counts(path: Path) -> tuple[int, int]:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    return int(df.isna().sum().sum()), int(df.shape[1])


def inventory(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, miss = [], []
    for part in PARTS:
        pdir = resolve(cfg, part)
        for tid in list_trips(pdir):
            d = pdir / tid
            r = {"partition": part, "trip": tid}
            for ext in ("track", "nodes", "arcs", "route"):
                f = d / f"{tid}.{ext}"
                r[f"has_{ext}"] = f.exists()
                r[f"{ext}_kb"] = round(f.stat().st_size / 1024, 1) if f.exists() else None
                if f.exists():
                    nan, ncol = raw_nan_counts(f)
                    miss.append({"partition": part, "trip": tid, "file": ext, "nan_values": nan, "columns": ncol})
            tr = pd.read_csv(d / f"{tid}.track", sep=r"\s+", header=None).to_numpy()
            nodes = pd.read_csv(d / f"{tid}.nodes", sep=r"\s+", header=None).to_numpy()
            arcs = pd.read_csv(d / f"{tid}.arcs", sep=r"\s+", header=None).to_numpy(dtype=np.int64)
            route = load_route(d)
            dt = np.diff(tr[:, 2])
            step = haversine(tr[:-1, 0], tr[:-1, 1], tr[1:, 0], tr[1:, 1])
            proj = LocalProjection.around(tr[:, 0], tr[:, 1])
            x, y = proj.forward(tr[:, 0], tr[:, 1])
            n = len(nodes)
            g = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n, n))
            n_weak, lab_w = connected_components(g, directed=True, connection="weak")
            n_strong, lab_s = connected_components(g, directed=True, connection="strong")
            deg = np.bincount(arcs[:, 0], minlength=n) + np.bincount(arcs[:, 1], minlength=n)
            r.update({
                "n_points": len(tr), "n_track_cols": tr.shape[1],
                "n_nodes": n, "n_arcs": len(arcs), "route_len": len(route),
                "route_unique_arcs": len(np.unique(route)),
                "duration_min": (tr[-1, 2] - tr[0, 2]) / 60,
                "timestamps_sorted": bool((dt > 0).all()), "duplicate_timestamps": int((dt == 0).sum()),
                "dt_median": float(np.median(dt)), "frac_dt_1s": float(np.mean(dt == 1)),
                "dt_max_s": float(dt.max()), "gaps_gt_5s": int((dt > 5).sum()), "gaps_gt_60s": int((dt > 60).sum()),
                "gps_path_km": float(step.sum() / 1000), "max_step_m": float(step.max()),
                "max_speed_kmh": float(np.max(step / np.maximum(dt, 1e-9)) * 3.6),
                "extent_x_km": float(np.ptp(x) / 1000), "extent_y_km": float(np.ptp(y) / 1000),
                "lon_min": tr[:, 0].min(), "lon_max": tr[:, 0].max(),
                "lat_min": tr[:, 1].min(), "lat_max": tr[:, 1].max(),
                "isolated_nodes": int((deg == 0).sum()),
                "weak_components": int(n_weak),
                "largest_weak_frac": float(np.bincount(lab_w).max() / n),
                "strong_components": int(n_strong),
                "largest_strong_frac": float(np.bincount(lab_s).max() / n),
                "route_arcs_valid": bool(((route >= 0) & (route < len(arcs))).all()),
            })
            rows.append(r)
            print(f"{part}/{tid}: {len(tr)} pts", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(miss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-trips", nargs="*", default=["train/00000003", "student_test/00000052"])
    args = ap.parse_args()
    cfg = load_config()
    out = resolve(cfg, "reports") / "eda"
    out.mkdir(parents=True, exist_ok=True)

    inv, miss = inventory(cfg)
    # label-alignment diagnostics from the processed tables
    pts = pd.concat([build_partition(p, cfg)[1].assign(partition=p) for p in PARTS], ignore_index=True)
    lab = pts.groupby(["partition", "trip"]).agg(
        gps_route_dist_median=("label_dist", "median"),
        gps_route_dist_p95=("label_dist", lambda s: s.quantile(0.95)),
        label_valid_frac=("label_valid", "mean"),
        no_candidate_frac=("n_cand", lambda s: (s == 0).mean()),
        candidate_hit_frac=("label_in_cands", "mean"),
        median_speed_ms=("speed", "median"),
    ).reset_index()
    inv = inv.merge(lab, on=["partition", "trip"], how="left")
    inv.to_csv(out / "inventory.csv", index=False)
    miss.groupby(["partition", "file"])[["nan_values"]].sum().reset_index().to_csv(out / "missing_values.csv", index=False)

    summ = inv.groupby("partition").agg(
        trips=("trip", "count"), points_total=("n_points", "sum"), points_mean=("n_points", "mean"),
        points_min=("n_points", "min"), points_max=("n_points", "max"),
        duration_min_mean=("duration_min", "mean"), path_km_total=("gps_path_km", "sum"),
        frac_dt_1s_mean=("frac_dt_1s", "mean"), trips_with_gap_gt_60s=("gaps_gt_60s", lambda s: (s > 0).sum()),
        nodes_mean=("n_nodes", "mean"), arcs_mean=("n_arcs", "mean"), route_len_mean=("route_len", "mean"),
        gps_route_dist_median=("gps_route_dist_median", "median"), label_valid_frac=("label_valid_frac", "mean"),
    ).reset_index()
    summ.to_csv(out / "partition_summary.csv", index=False)
    print(summ.T)

    # ---- data-quality notes
    notes = ["# Data-quality findings (auto-generated by scripts/eda.py)\n",
             f"- Missing values across all raw files: {int(miss['nan_values'].sum())}",
             f"- Trips with unsorted timestamps: {int((~inv['timestamps_sorted']).sum())}",
             f"- Trips containing a GPS gap > 60 s: {int((inv['gaps_gt_60s'] > 0).sum())} "
             f"(largest gap {inv['dt_max_s'].max():.0f} s in trip {inv.loc[inv['dt_max_s'].idxmax(), 'trip']})",
             f"- Network: isolated nodes total {int(inv['isolated_nodes'].sum())}; median weak components per trip "
             f"{inv['weak_components'].median():.0f}; median largest-SCC share {inv['largest_strong_frac'].median():.3f}",
             "", "## Trips with poor GPS-to-reference agreement (label_valid < 0.95)", ""]
    bad = inv[inv["label_valid_frac"] < 0.95].sort_values("label_valid_frac")
    for _, r in bad.iterrows():
        notes.append(f"- {r['partition']}/{r['trip']}: median GPS-route distance {r['gps_route_dist_median']:.1f} m, "
                     f"p95 {r['gps_route_dist_p95']:.1f} m, label-valid {r['label_valid_frac']:.2f}")
    notes += ["", "## Assumptions", "",
              "- Observations > 50 m from their aligned reference arc are treated as unreliable labels "
              "(excluded from training and from point-level scoring).",
              "- Coordinates are projected with a per-trip local equirectangular projection (metres).",
              "- Duplicate timestamps would be dropped (keep first); none observed if count is 0 above."]

    # ---- reference-route continuity (are consecutive reference arcs actually connected?)
    cont = []
    for part in PARTS:
        pdir = resolve(cfg, part)
        for tid in list_trips(pdir):
            ctx = process_trip(pdir / tid, cfg)
            cont.append({"partition": part, "trip": tid,
                         "reference_route_continuity": route_continuity(load_route(pdir / tid), ctx.net)})
    cont = pd.DataFrame(cont)
    cont.to_csv(out / "reference_route_continuity.csv", index=False)
    broken = cont[cont["reference_route_continuity"] < 1]
    notes += ["", "## Reference routes with disconnected consecutive arcs", ""]
    notes += [f"- {r.partition}/{r.trip}: continuity {r.reference_route_continuity:.4f}" for r in broken.itertuples()]
    (out / "data_quality.md").write_text("\n".join(notes), encoding="utf-8")

    # ---- plots
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for part, g in inv.groupby("partition"):
        ax[0].hist(g["n_points"], bins=30, alpha=0.6, label=part)
        ax[1].hist(g["duration_min"], bins=30, alpha=0.6, label=part)
        ax[2].scatter(g["n_nodes"], g["n_arcs"], s=12, label=part)
    ax[0].set(title="GPS observations per trip", xlabel="points", ylabel="trips")
    ax[1].set(title="Trip duration", xlabel="minutes")
    ax[2].set(title="Road-network size per trip", xlabel="nodes", ylabel="arcs", xscale="log", yscale="log")
    for a in ax:
        a.legend()
    fig.tight_layout(); fig.savefig(out / "trip_sizes.png", dpi=130); plt.close(fig)

    dts, speeds = [], []
    for part in PARTS:
        pdir = resolve(cfg, part)
        for tid in list_trips(pdir):
            tr = pd.read_csv(pdir / tid / f"{tid}.track", sep=r"\s+", header=None).to_numpy()
            dts.append(np.diff(tr[:, 2]))
    dts = np.concatenate(dts)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    vals, cnts = np.unique(np.minimum(dts, 61), return_counts=True)
    ax[0].bar(vals, cnts); ax[0].set(yscale="log", title="Sampling interval (all trips, >60 s pooled at 61)",
                                    xlabel="seconds", ylabel="count")
    ax[1].hist(np.clip(pts["label_dist"], 0, 100), bins=100)
    ax[1].axvline(cfg["labels"]["max_label_dist_m"], color="r", ls="--", label="label-valid threshold")
    ax[1].set(yscale="log", title="GPS distance to aligned reference arc", xlabel="metres (clipped at 100)")
    ax[1].legend()
    ax[2].hist(np.clip(pts["speed"] * 3.6, 0, 150), bins=75)
    ax[2].set(title="Smoothed GPS speed", xlabel="km/h")
    fig.tight_layout(); fig.savefig(out / "sampling_distance_speed.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for part, mk in (("train", "o"), ("student_test", "^")):
        g = inv[inv.partition == part]
        ax[0].scatter((g.lon_min + g.lon_max) / 2, (g.lat_min + g.lat_max) / 2, marker=mk, label=part, s=20)
    ax[0].set(title="Trip centres (lon/lat)", xlabel="longitude", ylabel="latitude"); ax[0].legend()
    ax[1].bar(range(len(inv)), inv["label_valid_frac"], color=np.where(inv.partition == "train", "C0", "C1"))
    ax[1].set(title="Share of label-valid observations per trip", xlabel="trip index", ylim=(0, 1.02))
    fig.tight_layout(); fig.savefig(out / "coordinates_label_quality.png", dpi=130); plt.close(fig)

    sel = inv[inv.partition == "train"].sort_values("n_points").iloc[::10]["trip"].tolist()[:8]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for a, tid in zip(axes.ravel(), sel):
        tr = pd.read_csv(resolve(cfg, "train") / tid / f"{tid}.track", sep=r"\s+", header=None).to_numpy()
        sc = a.scatter(tr[:, 0], tr[:, 1], c=tr[:, 2] - tr[0, 2], s=2, cmap="viridis")
        a.set(title=f"train/{tid} ({len(tr)} pts)", xticks=[], yticks=[]); a.set_aspect(1 / np.cos(np.radians(tr[:, 1].mean())))
    fig.colorbar(sc, ax=axes, label="seconds since start", shrink=0.6)
    fig.savefig(out / "trajectories.png", dpi=130); plt.close(fig)

    # ---- folium maps
    for spec in args.map_trips:
        part, tid = spec.split("/")
        d = resolve(cfg, part) / tid
        ctx = process_trip(d, cfg)
        attach_labels(ctx, load_route(d), cfg)
        m = trip_map(ctx.trip.lon, ctx.trip.lat, ctx.net, network_arcs=ctx.cands["arc"].to_numpy(),
                     routes={"Reference route": (load_route(d), "reference", False)},
                     title=f"{part}/{tid}: raw GPS vs reference route")
        m.save(str(out / f"map_{part}_{tid}.html"))
    print("EDA outputs written to", out)


if __name__ == "__main__":
    main()
