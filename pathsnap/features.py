"""Candidate generation and feature engineering.

Each GPS observation gets up to `max_k` candidate arcs within `radius_m`.
Every (observation, candidate arc) pair becomes one row; the learning task is
binary: "is this candidate the arc the vehicle was on?". Scores of a point's
candidates are later compared (top-1) and fed to the HMM as emissions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import angle_diff, bearing, point_segment
from .network import RoadNetwork

FEATURE_COLUMNS = [
    "dist", "rank", "n_cand", "dist_minus_min", "dist_ratio",
    "t_frac", "at_endpoint", "log_arc_len",
    "head_diff", "cos_head", "head_valid", "along_speed",
    "speed", "dt_prev", "turn",
    "win_mean_dist", "win_max_dist", "win_minus_min", "win_rank",
    "out_deg", "in_deg", "is_oneway",
]


def kinematics(x, y, t, window: int, min_disp: float) -> dict:
    n = len(x)
    i = np.arange(n)
    i0, i1 = np.clip(i - window, 0, n - 1), np.clip(i + window, 0, n - 1)
    dx, dy = x[i1] - x[i0], y[i1] - y[i0]
    disp = np.hypot(dx, dy)
    dtw = np.maximum(t[i1] - t[i0], 1e-6)
    heading = bearing(dx, dy)
    head_valid = disp >= min_disp
    hin = bearing(x - x[i0], y - y[i0])
    hout = bearing(x[i1] - x, y[i1] - y)
    moving = (np.hypot(x - x[i0], y - y[i0]) > 1) & (np.hypot(x[i1] - x, y[i1] - y) > 1)
    turn = np.where(moving, angle_diff(hin, hout), 0.0)
    dt_prev = np.r_[0.0, np.diff(t)]
    return {"heading": heading, "head_valid": head_valid, "speed": disp / dtw,
            "ux": np.where(disp > 0, dx / np.maximum(disp, 1e-9), 0.0),
            "uy": np.where(disp > 0, dy / np.maximum(disp, 1e-9), 0.0),
            "disp_speed": disp / dtw, "turn": turn, "dt_prev": dt_prev}


def build_candidates(px, py, t, net: RoadNetwork, cand_cfg: dict, feat_cfg: dict):
    """Returns (point DataFrame, candidate DataFrame with FEATURE_COLUMNS)."""
    kin = kinematics(px, py, t, feat_cfg["heading_window"], feat_cfg["min_heading_disp_m"])
    net.build_index(px, py, cand_cfg["radius_m"], cand_cfg["densify_step_m"])
    c = net.candidates(px, py, cand_cfg["radius_m"], cand_cfg["max_k"])
    pt, arc, dist, tf, rank = c["pt"], c["arc"], c["dist"], c["t"], c["rank"]
    n = len(px)

    n_cand = np.bincount(pt, minlength=n)
    min_dist = np.full(n, np.nan)
    first = rank == 0  # candidates are sorted by distance, rank 0 is the nearest
    min_dist[pt[first]] = dist[first]

    hv = kin["head_valid"][pt]
    hd = np.where(hv, angle_diff(kin["heading"][pt], net.bearing[arc]), 90.0)
    L = net.length[arc]
    arc_ux = np.where(L > 0, (net.bx[arc] - net.ax[arc]) / np.maximum(L, 1e-9), 0.0)
    arc_uy = np.where(L > 0, (net.by[arc] - net.ay[arc]) / np.maximum(L, 1e-9), 0.0)
    along = kin["disp_speed"][pt] * (kin["ux"][pt] * arc_ux + kin["uy"][pt] * arc_uy)

    W = feat_cfg["neighbour_window"]
    wd = []
    for o in range(-W, W + 1):
        j = np.clip(pt + o, 0, n - 1)
        d, _ = point_segment(px[j], py[j], net.ax[arc], net.ay[arc], net.bx[arc], net.by[arc])
        wd.append(d)
    wd = np.vstack(wd)
    win_mean, win_max = wd.mean(0), wd.max(0)

    df = pd.DataFrame({
        "obs": pt, "arc": arc, "dist": dist, "rank": rank, "n_cand": n_cand[pt],
        "t_frac": tf, "at_endpoint": ((tf <= 0) | (tf >= 1)).astype(np.int8),
        "log_arc_len": np.log1p(L), "head_diff": hd, "cos_head": np.cos(np.radians(hd)) * hv,
        "head_valid": hv.astype(np.int8), "along_speed": along,
        "speed": kin["speed"][pt], "dt_prev": kin["dt_prev"][pt], "turn": kin["turn"][pt],
        "win_mean_dist": win_mean, "win_max_dist": win_max,
        "out_deg": net.out_degree[net.dst[arc]], "in_deg": net.in_degree[net.src[arc]],
        "is_oneway": (net.twin[arc] < 0).astype(np.int8),
    })
    df["dist_minus_min"] = df["dist"] - min_dist[pt]
    df["dist_ratio"] = (df["dist"] + 1.0) / (min_dist[pt] + 1.0)
    g = df.groupby("obs", sort=False)["win_mean_dist"]
    df["win_minus_min"] = df["win_mean_dist"] - g.transform("min")
    df["win_rank"] = g.rank(method="first").astype(np.int16) - 1

    points = pd.DataFrame({"x": px, "y": py, "t": t, "n_cand": n_cand,
                           "speed": kin["speed"], "heading": kin["heading"],
                           "head_valid": kin["head_valid"], "min_dist": min_dist})
    return points, df
