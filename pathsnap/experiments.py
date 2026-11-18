"""Helpers shared by the experiment scripts (route-level evaluation of a model)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve
from .dataset import TripContext, process_trip
from .evaluation import score_prediction
from .features import FEATURE_COLUMNS
from .io import list_trips
from .pipeline import decode_context


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def load_contexts(part: str, cfg: dict, trips: list[str] | None = None) -> list[tuple[Path, TripContext]]:
    part_dir = resolve(cfg, part)
    out = []
    for tid in trips or list_trips(part_dir):
        out.append((part_dir / tid, process_trip(part_dir / tid, cfg)))
    return out


def route_eval(contexts, cfg: dict, hmm_params: dict, model=None, probs: pd.DataFrame | None = None,
               ) -> pd.DataFrame:
    """Decode every trip (classifier-only and HMM) and score it against its reference route.
    Probabilities come either from `model` or from a (trip, obs, arc, prob) frame."""
    rows = []
    for trip_dir, ctx in contexts:
        c = ctx.cands
        if model is not None:
            c["prob"] = model.predict_proba(c[FEATURE_COLUMNS])[:, 1] if len(c) else np.empty(0)
        else:
            sub = probs[probs["trip"] == ctx.trip.trip_id][["obs", "arc", "prob"]]
            merged = c[["obs", "arc"]].merge(sub, on=["obs", "arc"], how="left")
            c["prob"] = merged["prob"].fillna(0.0).to_numpy()
        res = decode_context(ctx, hmm_params)
        s = score_prediction(trip_dir, res, ctx, cfg)
        s.update({"trip": ctx.trip.trip_id, "hmm_s": res["timings"]["hmm_s"],
                  "hmm_breaks": res["hmm_breaks"], "n_points": res["n_points"]})
        rows.append(s)
    return pd.DataFrame(rows)


def summarise_route_eval(df: pd.DataFrame) -> dict:
    """Point-weighted Top-1 and trip-averaged route metrics."""
    w = df["n_scored_points"]
    out = {}
    for name in ("classifier", "hmm"):
        out[f"{name}_top1_accuracy"] = float((df[f"{name}_top1_accuracy"] * w).sum() / w.sum())
        for k in ("arc_f1", "length_f1", "route_mismatch", "route_continuity"):
            out[f"{name}_{k}"] = float(df[f"{name}_{k}"].mean())
    out["hmm_breaks_total"] = int(df["hmm_breaks"].sum())
    out["hmm_seconds_total"] = float(df["hmm_s"].sum())
    return out


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
