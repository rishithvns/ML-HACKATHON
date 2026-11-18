"""Bayesian hyperparameter optimisation (Optuna TPE) of the selected model.

Search:   group-aware K-fold CV on the 80 training trips only (groups = trips).
          Objective = mean held-out observation-level Top-1 accuracy (the
          quantity the route decoder consumes), with median pruning on folds.
          Macro F1 and log loss of every trial are logged as well.
Then:     untuned vs tuned on identical protocol (5-fold CV + student-test,
          classifier-only and +HMM), out-of-fold probabilities of the tuned
          model are saved for decoder tuning (scripts/tune_hmm.py).

Outputs -> reports/tuning/, models/tuned_model.joblib, data/processed/oof_tuned.pkl
Run:  python -m scripts.tune_bayes --model lightgbm --trials 40
"""
from __future__ import annotations

import argparse
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from pathsnap.config import load_config, resolve  # noqa: E402
from pathsnap.dataset import build_partition  # noqa: E402
from pathsnap.experiments import load_contexts, route_eval, save_json, set_seed, summarise_route_eval  # noqa: E402
from pathsnap.models import model_specs  # noqa: E402
from pathsnap.training import evaluate, fit, group_cv, training_rows  # noqa: E402


def search_space(model: str, trial: optuna.Trial, seed: int):
    if model == "lightgbm":
        from lightgbm import LGBMClassifier
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
        return p, lambda: LGBMClassifier(**p, subsample_freq=1, random_state=seed, n_jobs=-1, verbose=-1)
    if model == "xgboost":
        from xgboost import XGBClassifier
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 50, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
        }
        return p, lambda: XGBClassifier(**p, tree_method="hist", eval_metric="logloss",
                                        random_state=seed, n_jobs=-1)
    raise ValueError(f"no search space defined for {model}")


def make_from_params(model: str, params: dict, seed: int):
    trial = optuna.trial.FixedTrial(params)
    return search_space(model, trial, seed)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lightgbm", choices=["lightgbm", "xgboost"])
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--search-folds", type=int, default=3)
    ap.add_argument("--search-stride", type=int, default=4,
                    help="observation stride for search rows (cheaper trials); final refit uses config stride")
    args = ap.parse_args()
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    out = resolve(cfg, "reports") / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    resolve(cfg, "models").mkdir(parents=True, exist_ok=True)

    cands, points = build_partition("train", cfg)
    tcands, tpoints = build_partition("student_test", cfg)
    search_rows = training_rows(cands, args.search_stride)
    trips = np.array(sorted(cands["trip"].unique()))
    splits = list(GroupKFold(n_splits=args.search_folds).split(trips, groups=trips))

    def objective(trial: optuna.Trial):
        params, make = search_space(args.model, trial, seed)
        tops, f1s, lls = [], [], []
        t0 = time.perf_counter()
        for f, (tr_idx, va_idx) in enumerate(splits):
            tr = search_rows[search_rows["trip"].isin(set(trips[tr_idx]))]
            va_set = set(trips[va_idx])
            model, _ = fit(make(), tr)
            m, _ = evaluate(model, cands[cands["trip"].isin(va_set)], points[points["trip"].isin(va_set)])
            tops.append(m["top1_accuracy"]); f1s.append(m["macro_f1"]); lls.append(m["log_loss"])
            trial.report(float(np.mean(tops)), f)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("cv_macro_f1", float(np.mean(f1s)))
        trial.set_user_attr("cv_log_loss", float(np.mean(lls)))
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        return float(np.mean(tops))

    study = optuna.create_study(direction="maximize", study_name=f"pathsnap_{args.model}",
                                sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=10),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    search_s = time.perf_counter() - t0
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state", "duration"))
    trials.to_csv(out / f"optuna_trials_{args.model}.csv", index=False)
    best = study.best_params
    print("best", study.best_value, best, flush=True)

    # ---------------- untuned vs tuned on the same protocol
    train_rows = training_rows(cands, cfg["training"]["train_point_stride"])
    contexts = load_contexts("student_test", cfg)
    untuned_make = model_specs(seed)[args.model]["make"]
    tuned_make = make_from_params(args.model, best, seed)
    rows, oof_tuned = [], None
    for label, make in (("untuned", untuned_make), ("tuned", tuned_make)):
        fm, oof = group_cv(make, train_rows, cands, points, cfg["training"]["cv_folds"], None, seed)
        fm = pd.DataFrame(fm)
        model, fit_s = fit(make(), train_rows)
        test, _ = evaluate(model, tcands, tpoints)
        rs = summarise_route_eval(route_eval(contexts, cfg, cfg["hmm"], model=model))
        rows.append({
            "variant": label, "cv_macro_f1": fm["macro_f1"].mean(), "cv_macro_f1_std": fm["macro_f1"].std(),
            "cv_top1": fm["top1_accuracy"].mean(), "cv_log_loss": fm["log_loss"].mean(),
            "test_top1": test["top1_accuracy"], "test_macro_f1": test["macro_f1"], "test_log_loss": test["log_loss"],
            "test_hmm_top1": rs["hmm_top1_accuracy"], "test_hmm_arc_f1": rs["hmm_arc_f1"],
            "fit_seconds": fit_s, "inference_ms_per_1k_points": test["inference_ms_per_1k_points"],
        })
        if label == "tuned":
            oof_tuned = oof
            joblib.dump(model, resolve(cfg, "models") / "tuned_model.joblib", compress=3)
    comp = pd.DataFrame(rows)
    comp.to_csv(out / f"tuned_vs_untuned_{args.model}.csv", index=False)
    print(comp.T, flush=True)
    oof_tuned.to_pickle(resolve(cfg, "processed") / "oof_tuned.pkl")

    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception as e:  # importance needs >1 completed trial with varying params
        importance = {"error": str(e)}
    save_json({
        "model": args.model, "objective": "mean group-CV observation Top-1 accuracy (maximise)",
        "search_folds": args.search_folds, "search_stride": args.search_stride,
        "n_trials": len(study.trials),
        "n_completed": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
        "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        "best_value": study.best_value, "best_params": best, "search_seconds": search_s,
        "param_importance": importance, "sampler": "TPESampler(seed, n_startup_trials=10)",
        "pruner": "MedianPruner(n_startup_trials=5)",
    }, out / f"best_params_{args.model}.json")

    done = trials[trials["state"] == "COMPLETE"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(done["number"], done["value"], "o", label="trial")
    ax[0].plot(done["number"], done["value"].cummax(), "-", label="best so far")
    ax[0].set(title="Optuna optimisation history", xlabel="trial", ylabel="CV Top-1"); ax[0].legend()
    if isinstance(importance, dict) and "error" not in importance:
        ax[1].barh(list(importance)[::-1], list(importance.values())[::-1])
        ax[1].set(title="Hyperparameter importance (fANOVA)")
    fig.tight_layout(); fig.savefig(out / f"optuna_history_{args.model}.png", dpi=130); plt.close(fig)

    pcols = [c for c in done.columns if c.startswith("params_")]
    fig, axes = plt.subplots(2, int(np.ceil(len(pcols) / 2)), figsize=(4 * int(np.ceil(len(pcols) / 2)), 7))
    for a, c in zip(np.ravel(axes), pcols):
        a.scatter(done[c], done["value"], s=14)
        a.set(xlabel=c.replace("params_", ""), ylabel="CV Top-1")
        if c.replace("params_", "") in ("learning_rate", "num_leaves", "min_child_samples", "reg_lambda",
                                        "reg_alpha", "min_child_weight", "gamma"):
            a.set_xscale("log")
    fig.tight_layout(); fig.savefig(out / f"optuna_params_{args.model}.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    metrics = ["cv_macro_f1", "cv_top1", "test_top1", "test_macro_f1", "test_hmm_top1"]
    x = np.arange(len(metrics))
    for i, r in comp.iterrows():
        ax.bar(x + (i - 0.5) * 0.35, [r[m] for m in metrics], 0.35, label=r["variant"])
    ax.set_xticks(x, metrics, rotation=20); ax.set(ylim=(0.8, 1), title=f"{args.model}: untuned vs tuned"); ax.legend()
    fig.tight_layout(); fig.savefig(out / f"tuned_vs_untuned_{args.model}.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
