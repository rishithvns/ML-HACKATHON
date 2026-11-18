"""PathSnap inference API (runs on Laptop 1).

    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment variables
  PATHSNAP_MODELS_DIR  folder with final_model.joblib + pipeline_config.json (default: models)
  PATHSNAP_EVAL_DIR    folder with one sub-folder per evaluation trip (default: data/eval)
  PATHSNAP_CORS        comma-separated allowed UI origins (default: *)

The prediction pipeline only reads .track/.nodes/.arcs. The .route file of a
trip is read afterwards by pathsnap.evaluation to compute accuracy on the
server; reference arcs are never sent in a response.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pathsnap.evaluation import score_prediction  # noqa: E402
from pathsnap.io import list_trips  # noqa: E402
from pathsnap.pipeline import PathSnapPredictor  # noqa: E402


def _path(env: str, default: str) -> Path:
    p = Path(os.environ.get(env, default))
    return p if p.is_absolute() else ROOT / p


MODELS_DIR = _path("PATHSNAP_MODELS_DIR", "models")
EVAL_DIR = _path("PATHSNAP_EVAL_DIR", "data/eval")
LOG_DIR = ROOT / "reports" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("pathsnap.api")
log.setLevel(logging.INFO)
if not log.handlers:
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_DIR / "api.log", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)

predictor: PathSnapPredictor | None = None


@asynccontextmanager
async def lifespan(_app):
    global predictor
    predictor = PathSnapPredictor.load(MODELS_DIR)
    log.info("model loaded from %s: %s", MODELS_DIR, predictor.meta.get("model_name"))
    log.info("evaluation trips in %s: %s", EVAL_DIR, list_trips(EVAL_DIR))
    yield


app = FastAPI(title="PathSnap API", version="1.0", lifespan=lifespan,
              description="Map-matching of GPS trips: classifier + HMM/Viterbi route reconstruction")
origins = [o.strip() for o in os.environ.get("PATHSNAP_CORS", "*").split(",")]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def _access_log(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    client = request.client.host if request.client else "?"
    log.info("%s %s %s -> %d (%.0f ms)", client, request.method, request.url.path,
             response.status_code, 1000 * (time.perf_counter() - t0))
    return response


class PredictRequest(BaseModel):
    trip_id: str


def _round(a, nd=6):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None,
            "model": predictor.meta.get("model_name") if predictor else None,
            "eval_dir": str(EVAL_DIR), "n_trips": len(list_trips(EVAL_DIR))}


@app.get("/trips")
def trips():
    out = []
    for tid in list_trips(EVAL_DIR):
        with open(EVAL_DIR / tid / f"{tid}.track", "rb") as f:
            n = sum(1 for _ in f)
        out.append({"trip_id": tid, "n_points": n})
    return {"trips": out}


@app.post("/predict")
def predict(req: PredictRequest):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", req.trip_id) or req.trip_id not in list_trips(EVAL_DIR):
        raise HTTPException(status_code=404, detail=f"unknown trip_id {req.trip_id!r}")
    trip_dir = EVAL_DIR / req.trip_id
    log.info("predict request: trip %s", req.trip_id)
    t0 = time.perf_counter()
    result, ctx = predictor.predict(trip_dir)
    predict_s = time.perf_counter() - t0
    net = ctx.net
    gps = np.c_[ctx.trip.lat, ctx.trip.lon]
    nearby = np.unique(ctx.cands["arc"].to_numpy())

    # ---- server-side scoring (reads the reference route; nothing of it is returned)
    metrics = score_prediction(trip_dir, result, ctx, predictor.cfg) if (trip_dir / f"{req.trip_id}.route").exists() else None
    total_s = time.perf_counter() - t0
    if metrics:
        log.info("trip %s: %d pts | top-1 classifier %.4f -> HMM %.4f | arc F1 %.4f | continuity %.4f | %.2fs",
                 req.trip_id, result["n_points"], metrics["classifier_top1_accuracy"],
                 metrics["hmm_top1_accuracy"], metrics["hmm_arc_f1"], metrics["hmm_route_continuity"], total_s)

    matched = result["hmm_arc"]
    has = matched >= 0
    return {
        "trip_id": req.trip_id,
        "model": predictor.meta.get("model_name"),
        "n_points": result["n_points"],
        "n_points_without_candidates": result["n_points_without_candidates"],
        "gps": _round(gps),
        "predicted_route": [list(p) for p in net.route_lonlat(result["hmm_route"])],
        "classifier_route": [list(p) for p in net.route_lonlat(result["classifier_route"])],
        "predicted_arcs_per_point": matched.tolist(),
        "matched_fraction": float(has.mean()),
        "road_network": [[list(a), list(b)] for a, b in net.arc_lonlat(nearby)],
        "hmm_breaks": result["hmm_breaks"],
        "route_continuity": {"classifier": result["classifier_route_continuity"],
                             "hmm": result["hmm_route_continuity"]},
        "metrics": metrics,
        "timings_s": {**{k: round(v, 3) for k, v in result["timings"].items()},
                      "predict_total": round(predict_s, 3), "request_total": round(total_s, 3)},
    }
