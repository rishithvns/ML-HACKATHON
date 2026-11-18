"""Train and compare candidate-scoring models.

For every model in pathsnap.models.model_specs():
  1. 5-fold GroupKFold CV over the 80 training trips (groups = trips)
  2. refit on all training trips, evaluate on the 10 student-test trips
     (candidate metrics, observation Top-1, log loss, confusion matrix, timings)
  3. decode the student-test trips with the HMM using that model's probabilities
     (default decoder parameters from configs/default.json) to measure route
     continuity and route-level agreement
Models with iteration-wise training histories (LightGBM, XGBoost, MLP) also get
train/validation log-loss curves on CV fold 0.

Outputs -> reports/models/
Run:  python -m scripts.compare_models [--models lightgbm xgboost ...]
"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from pathsnap.config import load_config, resolve  # noqa: E402
from pathsnap.dataset import build_partition  # noqa: E402
from pathsnap.experiments import load_contexts, route_eval, save_json, set_seed, summarise_route_eval  # noqa: E402
from pathsnap.features import FEATURE_COLUMNS  # noqa: E402
from pathsnap.models import model_specs  # noqa: E402
from pathsnap.training import evaluate, fit, group_cv, subsample_obs, training_rows  # noqa: E402


def fold0_split(train_rows: pd.DataFrame, folds: int):
    trips = np.array(sorted(train_rows["trip"].unique()))
    tr_idx, va_idx = next(GroupKFold(n_splits=folds).split(trips, groups=trips))
    tr = train_rows[train_rows["trip"].isin(set(trips[tr_idx]))]
    va = train_rows[train_rows["trip"].isin(set(trips[va_idx]))]
    return tr, va


def history(name: str, spec: dict, train_rows: pd.DataFrame, folds: int, seed: int) -> dict | None:
    """Iteration-wise train/validation log loss on CV fold 0 (only for iterative learners)."""
    tr, va = fold0_split(train_rows, folds)
    tr = subsample_obs(tr, spec["max_train_rows"], seed)
    Xtr, ytr, Xva, yva = tr[FEATURE_COLUMNS], tr["y"].to_numpy(), va[FEATURE_COLUMNS], va["y"].to_numpy()
    if name == "lightgbm":
        m = spec["make"]()
        m.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], eval_names=["train", "valid"], eval_metric="binary_logloss")
        r = m.evals_result_
        return {"x_label": "boosting round", "train": r["train"]["binary_logloss"], "valid": r["valid"]["binary_logloss"]}
    if name == "xgboost":
        m = spec["make"]()
        m.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)
        r = m.evals_result()
        return {"x_label": "boosting round", "train": r["validation_0"]["logloss"], "valid": r["validation_1"]["logloss"]}
    if name == "mlp":
        sc = StandardScaler().fit(Xtr)
        A, B = sc.transform(Xtr), sc.transform(Xva)
        est = spec["make"]().steps[-1][1]
        m = MLPClassifier(hidden_layer_sizes=est.hidden_layer_sizes, alpha=est.alpha,
                          learning_rate_init=est.learning_rate_init, batch_size=est.batch_size, random_state=seed)
        rng = np.random.default_rng(seed)
        tl, vl = [], []
        for _ in range(est.max_iter):
            perm = rng.permutation(len(A))
            for s in range(0, len(A), est.batch_size):
                idx = perm[s:s + est.batch_size]
                m.partial_fit(A[idx], ytr[idx], classes=[0, 1])
            tl.append(log_loss(ytr, m.predict_proba(A)[:, 1], labels=[0, 1]))
            vl.append(log_loss(yva, m.predict_proba(B)[:, 1], labels=[0, 1]))
        return {"x_label": "epoch", "train": tl, "valid": vl}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    out = resolve(cfg, "reports") / "models"
    out.mkdir(parents=True, exist_ok=True)

    cands, points = build_partition("train", cfg)
    tcands, tpoints = build_partition("student_test", cfg)
    train_rows = training_rows(cands, cfg["training"]["train_point_stride"])
    contexts = load_contexts("student_test", cfg)
    folds = cfg["training"]["cv_folds"]
    specs = model_specs(seed)
    names = args.models or list(specs)
    print(f"training rows (label-valid, stride {cfg['training']['train_point_stride']}): {len(train_rows)}")

    for name in names:
        spec = specs[name]
        print(f"\n=== {name}", flush=True)
        t0 = time.perf_counter()
        fold_metrics, _ = group_cv(spec["make"], train_rows, cands, points, folds, spec["max_train_rows"], seed)
        cv_s = time.perf_counter() - t0
        fm = pd.DataFrame(fold_metrics).drop(columns=["confusion"])
        fm.insert(0, "model", name)
        fm.to_csv(out / f"cv_folds_{name}.csv", index=False)

        tr = subsample_obs(train_rows, spec["max_train_rows"], seed)
        model, fit_s = fit(spec["make"](), tr)
        test, _ = evaluate(model, tcands, tpoints)
        rt = route_eval(contexts, cfg, cfg["hmm"], model=model)
        rt.to_csv(out / f"student_test_trips_{name}.csv", index=False)
        rs = summarise_route_eval(rt)

        tr_points = points.merge(tr[["trip", "obs"]].drop_duplicates(), on=["trip", "obs"])
        train_eval, _ = evaluate(model, tr, tr_points)  # in-sample, for over/under-fitting checks
        res = {
            "model": name, "family": spec["family"], "config": spec["config"],
            "n_train_rows": len(tr), "seed": seed,
            "cv_macro_f1_mean": fm["macro_f1"].mean(), "cv_macro_f1_std": fm["macro_f1"].std(),
            "cv_top1_mean": fm["top1_accuracy"].mean(), "cv_top1_std": fm["top1_accuracy"].std(),
            "cv_log_loss_mean": fm["log_loss"].mean(), "cv_log_loss_std": fm["log_loss"].std(),
            "cv_seconds": cv_s,
            "train_macro_f1": train_eval["macro_f1"], "train_top1": train_eval["top1_accuracy"],
            "train_log_loss": train_eval["log_loss"],
            "test_accuracy": test["accuracy"], "test_precision": test["precision"], "test_recall": test["recall"],
            "test_f1": test["f1"], "test_macro_f1": test["macro_f1"], "test_log_loss": test["log_loss"],
            "test_top1": test["top1_accuracy"], "test_candidate_miss_rate": test["candidate_miss_rate"],
            "test_confusion": test["confusion"],
            "fit_seconds": fit_s, "inference_ms_per_1k_points": test["inference_ms_per_1k_points"],
            **{f"test_{k}": v for k, v in rs.items()},
        }
        hist = history(name, spec, train_rows, folds, seed)
        if hist:
            res["history"] = hist
        save_json(res, out / f"result_{name}.json")
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()
                          if k not in ("history", "test_confusion")}, indent=1), flush=True)

    collect_and_plot(out)


def collect_and_plot(out):
    results = []
    for f in sorted(out.glob("result_*.json")):
        with open(f, encoding="utf-8") as fh:
            results.append(json.load(fh))
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("history", "test_confusion")} for r in results])
    df = df.sort_values("cv_top1_mean", ascending=False)
    df.to_csv(out / "model_comparison_full.csv", index=False)
    table = pd.DataFrame({
        "Model": df["model"], "Algorithm family": df["family"], "Key configuration": df["config"],
        "CV macro F1": df["cv_macro_f1_mean"].round(4).astype(str) + " +/- " + df["cv_macro_f1_std"].round(4).astype(str),
        "CV Top-1": df["cv_top1_mean"].round(4),
        "Student-test Top-1 (classifier)": df["test_top1"].round(4),
        "Student-test Top-1 (+HMM)": df["test_hmm_top1_accuracy"].round(4),
        "Route continuity (classifier)": df["test_classifier_route_continuity"].round(4),
        "Route continuity (+HMM)": df["test_hmm_route_continuity"].round(4),
        "Student-test macro F1": df["test_macro_f1"].round(4),
        "Student-test log loss": df["test_log_loss"].round(4),
        "Fit time (s)": df["fit_seconds"].round(1),
        "Inference (ms / 1k points)": df["inference_ms_per_1k_points"].round(1),
    })
    table.to_csv(out / "model_comparison.csv", index=False)
    print(table.to_string(index=False))

    order = df["model"].tolist()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(order, df["cv_macro_f1_mean"], yerr=df["cv_macro_f1_std"], capsize=4, color="C0")
    ax.set(title="5-fold group CV macro F1 (training trips)", ylabel="macro F1",
           ylim=(max(0, df["cv_macro_f1_mean"].min() - 0.1), 1))
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout(); fig.savefig(out / "cv_macro_f1.png", dpi=130); plt.close(fig)

    x = np.arange(len(order)); w = 0.27
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - w, df["test_top1"], w, label="Top-1 (classifier only)")
    ax.bar(x, df["test_hmm_top1_accuracy"], w, label="Top-1 (classifier + HMM)")
    ax.bar(x + w, df["test_macro_f1"], w, label="Macro F1 (candidate level)")
    ax.set_xticks(x, order, rotation=35)
    ax.set(title="Student-test performance", ylim=(max(0, min(df["test_top1"].min(), df["test_macro_f1"].min()) - 0.1), 1))
    ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(out / "student_test_top1_f1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(order, df["test_log_loss"], color="C3"); ax[0].set(title="Student-test log loss (lower is better)")
    ax[1].bar(order, df["inference_ms_per_1k_points"], color="C2"); ax[1].set(title="Classifier inference ms / 1k points", yscale="log")
    for a in ax:
        a.tick_params(axis="x", rotation=35)
    fig.tight_layout(); fig.savefig(out / "log_loss_inference.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - w / 2, df["test_classifier_route_continuity"], w, label="classifier only")
    ax.bar(x + w / 2, df["test_hmm_route_continuity"], w, label="classifier + HMM")
    ax.set_xticks(x, order, rotation=35); ax.set(title="Route continuity on student-test", ylim=(0, 1.02))
    ax.legend(); fig.tight_layout(); fig.savefig(out / "route_continuity.png", dpi=130); plt.close(fig)

    n = len(results); cols = 5; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3 * rows))
    for a, r in zip(np.ravel(axes), sorted(results, key=lambda r: order.index(r["model"]))):
        cm = np.array(r["test_confusion"])
        a.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                a.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=9)
        a.set(title=r["model"], xticks=[0, 1], yticks=[0, 1], xlabel="predicted", ylabel="true")
    for a in np.ravel(axes)[n:]:
        a.axis("off")
    fig.suptitle("Student-test confusion matrices (candidate rows, threshold 0.5)")
    fig.tight_layout(); fig.savefig(out / "confusion_matrices.png", dpi=130); plt.close(fig)

    hist = [r for r in results if "history" in r]
    if hist:
        fig, axes = plt.subplots(1, len(hist), figsize=(5 * len(hist), 4))
        for a, r in zip(np.atleast_1d(axes), hist):
            h = r["history"]
            a.plot(h["train"], label="train (fold-0 training trips)")
            a.plot(h["valid"], label="validation (fold-0 held-out trips)")
            a.set(title=f"{r['model']} log loss", xlabel=h["x_label"], ylabel="log loss"); a.legend()
        fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
