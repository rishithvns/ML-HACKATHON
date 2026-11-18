"""Metrics: candidate-level classification, observation-level, and route-level."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)

from .network import RoadNetwork


def candidate_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    """Row-level (binary candidate) metrics."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(prob) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "log_loss": log_loss(y_true, np.clip(prob, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "confusion": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }


def top1_arcs(cands: pd.DataFrame, score_col: str = "prob") -> pd.DataFrame:
    """Highest-scoring candidate per (trip, obs)."""
    idx = cands.groupby(["trip", "obs"], sort=False)[score_col].idxmax()
    return cands.loc[idx, ["trip", "obs", "arc", score_col]]


def observation_metrics(cands: pd.DataFrame, points: pd.DataFrame,
                        pred: pd.DataFrame | None = None) -> dict:
    """Observation-level Top-1 accuracy and candidate miss rate over label-valid points.

    `pred` is a frame with columns trip, obs, arc (predicted arc per observation);
    observations without a prediction count as wrong."""
    pts = points[points["label_valid"]]
    if pred is None:
        pred = top1_arcs(cands)
    m = pts[["trip", "obs", "label_arc", "label_in_cands"]].merge(
        pred[["trip", "obs", "arc"]], on=["trip", "obs"], how="left")
    correct = (m["arc"].to_numpy() == m["label_arc"].to_numpy())
    hit = m["label_in_cands"].to_numpy()
    return {
        "top1_accuracy": float(correct.mean()),
        "top1_given_candidate": float(correct[hit].mean()) if hit.any() else float("nan"),
        "candidate_miss_rate": float(1 - hit.mean()),
        "n_points": int(len(m)),
    }


def dedupe(seq) -> list[int]:
    out: list[int] = []
    for a in seq:
        if a < 0:
            continue
        if not out or out[-1] != a:
            out.append(int(a))
    return out


def route_continuity(route_arcs, net: RoadNetwork) -> float:
    """Fraction of consecutive arcs in a route that are physically connected (head -> tail)."""
    r = list(route_arcs)
    if len(r) < 2:
        return 1.0
    ok = sum(int(net.dst[a]) == int(net.src[b]) for a, b in zip(r[:-1], r[1:]))
    return ok / (len(r) - 1)


def sequence_metrics(pred_route, ref_route, net: RoadNetwork) -> dict:
    """Route-level agreement between predicted and reference arc sequences.

    - arc precision / recall / F1 over the sets of traversed arcs
    - length-weighted versions (metres of road)
    - route mismatch fraction (Newson & Krumm 2009): (missing + extra length) / reference length
    """
    P, R = set(int(a) for a in pred_route), set(int(a) for a in ref_route)
    inter = P & R
    L = net.length
    lp, lr, li = L[list(P)].sum() if P else 0.0, L[list(R)].sum() if R else 0.0, L[list(inter)].sum() if inter else 0.0
    prec = len(inter) / len(P) if P else 0.0
    rec = len(inter) / len(R) if R else 0.0
    lprec = li / lp if lp else 0.0
    lrec = li / lr if lr else 0.0
    return {
        "arc_precision": prec, "arc_recall": rec,
        "arc_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "length_precision": lprec, "length_recall": lrec,
        "length_f1": 2 * lprec * lrec / (lprec + lrec) if lprec + lrec else 0.0,
        "route_mismatch": ((lr - li) + (lp - li)) / lr if lr else float("nan"),
    }
