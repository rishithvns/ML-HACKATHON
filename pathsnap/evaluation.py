"""Server-side scoring of a prediction against the reference route.

Only this module (and training-time labelling) reads .route files. Results are
aggregate metrics; reference arcs are never returned to the UI.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import TripContext, attach_labels
from .io import load_route
from .metrics import route_continuity, sequence_metrics


def score_prediction(trip_dir: str | Path, result: dict, ctx: TripContext, cfg: dict) -> dict:
    route = load_route(trip_dir)
    attach_labels(ctx, route, cfg)
    p = ctx.points
    valid = p["label_valid"].to_numpy()
    label = p["label_arc"].to_numpy()
    out = {
        "n_scored_points": int(valid.sum()),
        "candidate_miss_rate": float(1 - p.loc[valid, "label_in_cands"].mean()) if valid.any() else None,
        "reference_route_continuity": route_continuity(route, ctx.net),
    }
    for name in ("classifier", "hmm"):
        arcs = result[f"{name}_arc"]
        out[f"{name}_top1_accuracy"] = float((arcs[valid] == label[valid]).mean()) if valid.any() else None
        seq = sequence_metrics(result[f"{name}_route"], route, ctx.net)
        out.update({f"{name}_{k}": v for k, v in seq.items()})
        out[f"{name}_route_continuity"] = result[f"{name}_route_continuity"]
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}
