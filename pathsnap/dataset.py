"""Turns raw trips into candidate tables (with labels for train / student-test)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve
from .features import build_candidates
from .geo import LocalProjection
from .io import Trip, list_trips, load_route, load_trip
from .labels import align_to_route
from .network import RoadNetwork


@dataclass
class TripContext:
    trip: Trip
    proj: LocalProjection
    net: RoadNetwork
    points: pd.DataFrame
    cands: pd.DataFrame


def process_trip(trip_dir: str | Path, cfg: dict, trip: Trip | None = None) -> TripContext:
    """Prediction-side processing. Never reads the reference route."""
    trip = trip or load_trip(trip_dir)
    proj = LocalProjection.around(trip.lon, trip.lat)
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    px, py = proj.forward(trip.lon, trip.lat)
    points, cands = build_candidates(px, py, trip.t, net, cfg["candidates"], cfg["features"])
    points["lon"], points["lat"], points["obs_id"] = trip.lon, trip.lat, trip.obs_id
    cands.insert(0, "trip", trip.trip_id)
    points.insert(0, "obs", np.arange(len(points)))
    points.insert(0, "trip", trip.trip_id)
    return TripContext(trip, proj, net, points, cands)


def attach_labels(ctx: TripContext, route: np.ndarray, cfg: dict) -> TripContext:
    """Training/evaluation side: add per-observation reference arcs and candidate targets."""
    p = ctx.points
    lab = align_to_route(p["x"].to_numpy(), p["y"].to_numpy(), p["heading"].to_numpy(),
                         p["head_valid"].to_numpy(), ctx.net, route,
                         cfg["labels"]["heading_penalty_m"])
    for k, v in lab.items():
        p[k] = v
    p["label_valid"] = p["label_dist"] <= cfg["labels"]["max_label_dist_m"]
    c = ctx.cands
    c["y"] = (c["arc"].to_numpy() == p["label_arc"].to_numpy()[c["obs"].to_numpy()]).astype(np.int8)
    c["label_valid"] = p["label_valid"].to_numpy()[c["obs"].to_numpy()]
    hit = np.zeros(len(p), dtype=bool)
    hit[c.loc[c["y"] == 1, "obs"].to_numpy()] = True
    p["label_in_cands"] = hit
    return ctx


def build_partition(part: str, cfg: dict, force: bool = False, verbose: bool = True):
    """Candidate + point tables for every trip in a partition, cached as pickles."""
    out_dir = resolve(cfg, "processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    cpath, ppath = out_dir / f"{part}_cands.pkl", out_dir / f"{part}_points.pkl"
    if cpath.exists() and ppath.exists() and not force:
        return pd.read_pickle(cpath), pd.read_pickle(ppath)
    part_dir = resolve(cfg, part)
    all_c, all_p = [], []
    for tid in list_trips(part_dir):
        t0 = time.perf_counter()
        ctx = process_trip(part_dir / tid, cfg)
        attach_labels(ctx, load_route(part_dir / tid), cfg)
        all_c.append(ctx.cands)
        all_p.append(ctx.points)
        if verbose:
            p = ctx.points
            print(f"[{part}] {tid}: {len(p)} pts, {len(ctx.cands)} cand rows, "
                  f"label_valid={p.label_valid.mean():.3f}, cand_hit={p.label_in_cands.mean():.3f} "
                  f"({time.perf_counter() - t0:.1f}s)", flush=True)
    cands, points = pd.concat(all_c, ignore_index=True), pd.concat(all_p, ignore_index=True)
    cands.to_pickle(cpath)
    points.to_pickle(ppath)
    return cands, points
