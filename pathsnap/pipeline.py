"""End-to-end inference: raw trip folder -> candidates -> classifier -> HMM -> route.

This module never reads the reference route; scoring lives in evaluation.py.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np

from .dataset import TripContext, process_trip
from .features import FEATURE_COLUMNS
from .hmm import reconstruct_route, viterbi
from .metrics import dedupe, route_continuity, top1_arcs


class PathSnapPredictor:
    def __init__(self, model, cfg: dict, hmm_params: dict, meta: dict | None = None):
        self.model = model
        self.cfg = cfg
        self.hmm_params = hmm_params
        self.meta = meta or {}

    @classmethod
    def load(cls, models_dir: str | Path) -> "PathSnapPredictor":
        models_dir = Path(models_dir)
        with open(models_dir / "pipeline_config.json", encoding="utf-8") as f:
            pc = json.load(f)
        if pc["feature_columns"] != FEATURE_COLUMNS:
            raise ValueError("feature columns of the saved model do not match the code")
        model = joblib.load(models_dir / pc["model_file"])
        return cls(model, pc["config"], pc["hmm"], pc.get("meta"))

    def save(self, models_dir: str | Path, model_file: str = "final_model.joblib"):
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, models_dir / model_file, compress=3)
        pc = {"model_file": model_file, "feature_columns": FEATURE_COLUMNS,
              "config": {k: self.cfg[k] for k in ("candidates", "features", "labels")},
              "hmm": self.hmm_params, "meta": self.meta}
        with open(models_dir / "pipeline_config.json", "w", encoding="utf-8") as f:
            json.dump(pc, f, indent=2)

    def predict(self, trip_dir: str | Path) -> tuple[dict, TripContext]:
        timings = {}
        t0 = time.perf_counter()
        ctx = process_trip(trip_dir, self.cfg)
        timings["features_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        c = ctx.cands
        c["prob"] = self.model.predict_proba(c[FEATURE_COLUMNS])[:, 1] if len(c) else np.empty(0)
        timings["classifier_s"] = time.perf_counter() - t0

        result = decode_context(ctx, self.hmm_params)
        result["timings"] = {**timings, **result["timings"]}
        return result, ctx


def decode_context(ctx: TripContext, hmm_params: dict) -> dict:
    """Classifier-only and HMM predictions for a trip whose ctx.cands already has 'prob'."""
    c = ctx.cands
    n = len(ctx.points)
    clf_arc = np.full(n, -1, dtype=np.int64)
    if len(c):
        top = top1_arcs(c)
        clf_arc[top["obs"].to_numpy()] = top["arc"].to_numpy()

    t0 = time.perf_counter()
    dec = viterbi(c, ctx.points, ctx.net, hmm_params)
    rec = reconstruct_route(dec["obs_arc"], dec["segment"], ctx.net)
    hmm_s = time.perf_counter() - t0

    clf_route = dedupe(clf_arc)
    return {
        "trip_id": ctx.trip.trip_id,
        "n_points": n,
        "n_points_without_candidates": int((ctx.points["n_cand"] == 0).sum()),
        "classifier_arc": clf_arc,
        "hmm_arc": dec["obs_arc"],
        "hmm_segment": dec["segment"],
        "hmm_breaks": dec["n_breaks"],
        "classifier_route": clf_route,
        "hmm_route": rec["route"],
        "hmm_unfilled_gaps": rec["unfilled_gaps"],
        "classifier_route_continuity": route_continuity(clf_route, ctx.net),
        "hmm_route_continuity": route_continuity(rec["route"], ctx.net),
        "timings": {"hmm_s": hmm_s},
    }
