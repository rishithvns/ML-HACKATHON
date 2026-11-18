"""Training utilities: row selection, group-aware CV, and evaluation of a fitted model."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .features import FEATURE_COLUMNS
from .metrics import candidate_metrics, observation_metrics


def training_rows(cands: pd.DataFrame, stride: int) -> pd.DataFrame:
    """Label-valid rows, keeping every `stride`-th observation (1 Hz fixes are highly
    redundant; striding cuts cost without dropping trips)."""
    m = cands["label_valid"].to_numpy() & (cands["obs"].to_numpy() % stride == 0)
    return cands[m]


def subsample_obs(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    keys = df["trip"].astype(str) + ":" + df["obs"].astype(str)
    uniq = keys.unique()
    rng = np.random.default_rng(seed)
    frac = max_rows / len(df)
    chosen = set(rng.choice(uniq, size=int(len(uniq) * frac), replace=False))
    return df[keys.isin(chosen).to_numpy()]


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X[FEATURE_COLUMNS])[:, 1]


def fit(model, df: pd.DataFrame):
    t0 = time.perf_counter()
    model.fit(df[FEATURE_COLUMNS], df["y"].to_numpy())
    return model, time.perf_counter() - t0


def evaluate(model, cands: pd.DataFrame, points: pd.DataFrame) -> dict:
    """Candidate-level metrics on label-valid rows and observation-level Top-1 on all
    label-valid points (points without candidates count as misses)."""
    t0 = time.perf_counter()
    prob = predict_proba(model, cands)
    infer = time.perf_counter() - t0
    c = cands.assign(prob=prob)
    valid = c["label_valid"].to_numpy()
    out = candidate_metrics(c.loc[valid, "y"], c.loc[valid, "prob"])
    out.update(observation_metrics(c, points))
    out["inference_s"] = infer
    out["inference_ms_per_1k_points"] = 1e6 * infer / max(1, len(points))
    return out, c


def group_cv(make_model, train_rows: pd.DataFrame, all_cands: pd.DataFrame, all_points: pd.DataFrame,
             folds: int, max_rows: int | None, seed: int) -> tuple[list[dict], pd.DataFrame]:
    """GroupKFold by trip. Fits on (strided, label-valid) training rows of the training
    folds and evaluates on *all* candidate rows of the held-out trips.
    Returns per-fold metrics and out-of-fold probabilities for every candidate row."""
    trips = np.array(sorted(all_cands["trip"].unique()))
    gkf = GroupKFold(n_splits=folds)
    fold_metrics, oof = [], []
    for f, (tr_idx, va_idx) in enumerate(gkf.split(trips, groups=trips)):
        tr_trips, va_trips = set(trips[tr_idx]), set(trips[va_idx])
        tr = subsample_obs(train_rows[train_rows["trip"].isin(tr_trips)], max_rows, seed + f)
        model, fit_s = fit(make_model(), tr)
        va_c = all_cands[all_cands["trip"].isin(va_trips)]
        va_p = all_points[all_points["trip"].isin(va_trips)]
        m, scored = evaluate(model, va_c, va_p)
        m.update({"fold": f, "fit_s": fit_s, "n_train_rows": len(tr)})
        fold_metrics.append(m)
        oof.append(scored[["trip", "obs", "arc", "prob"]])
    return fold_metrics, pd.concat(oof, ignore_index=True)
