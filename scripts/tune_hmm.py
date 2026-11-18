"""Tune the HMM decoder, compare classifier-only vs classifier+HMM, export final artifacts.

Decoder tuning uses *out-of-fold* probabilities of the tuned model on training
trips (so emissions are as (un)confident as they will be on unseen trips), on
training trips whose labels are reliable (label-valid share >= 0.95).
Objective: point-weighted HMM Top-1 accuracy. Student-test trips are then
used only to report the final comparison; the evaluator partition is never
touched.

Outputs -> reports/hmm/, models/final_model.joblib, models/pipeline_config.json
Run:  python -m scripts.tune_hmm [--dev-trips 30]
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pathsnap.config import load_config, resolve  # noqa: E402
from pathsnap.dataset import build_partition  # noqa: E402
from pathsnap.experiments import load_contexts, route_eval, save_json, set_seed, summarise_route_eval  # noqa: E402
from pathsnap.pipeline import PathSnapPredictor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-trips", type=int, default=30)
    ap.add_argument("--model-name", default="lightgbm")
    args = ap.parse_args()
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    out = resolve(cfg, "reports") / "hmm"
    out.mkdir(parents=True, exist_ok=True)

    _, points = build_partition("train", cfg)
    oof = pd.read_pickle(resolve(cfg, "processed") / "oof_tuned.pkl")
    model = joblib.load(resolve(cfg, "models") / "tuned_model.joblib")

    quality = points.groupby("trip")["label_valid"].mean()
    eligible = sorted(quality[quality >= 0.95].index)
    rng = np.random.default_rng(seed)
    dev = sorted(rng.choice(eligible, size=min(args.dev_trips, len(eligible)), replace=False).tolist())
    print(f"decoder-dev trips ({len(dev)}): {dev}", flush=True)
    dev_ctx = load_contexts("train", cfg, dev)

    base = cfg["hmm"]
    grid = []

    def run(params, tag):
        df = route_eval(dev_ctx, cfg, params, probs=oof)
        s = summarise_route_eval(df)
        row = {**{k: params[k] for k in ("beta", "emission_weight", "uturn_penalty")}, **s, "stage": tag}
        grid.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        return s

    # stage 1: beta x emission weight
    for beta, ew in itertools.product([1.0, 2.0, 5.0, 10.0, 20.0], [0.5, 1.0, 2.0]):
        run({**base, "beta": beta, "emission_weight": ew}, "beta_x_emission")
    g = pd.DataFrame(grid)
    b1 = g.sort_values(["hmm_top1_accuracy", "hmm_arc_f1"], ascending=False).iloc[0]
    # stage 2: U-turn penalty at the best (beta, emission weight)
    for up in [0.0, 1.0, 3.0, 10.0]:
        run({**base, "beta": b1["beta"], "emission_weight": b1["emission_weight"], "uturn_penalty": up}, "uturn")
    g = pd.DataFrame(grid)
    g.to_csv(out / "decoder_grid.csv", index=False)
    best_row = g.sort_values(["hmm_top1_accuracy", "hmm_arc_f1"], ascending=False).iloc[0]
    tuned = {**base, "beta": float(best_row["beta"]), "emission_weight": float(best_row["emission_weight"]),
             "uturn_penalty": float(best_row["uturn_penalty"])}
    print("tuned decoder:", tuned, flush=True)

    # ---------------- final comparison on student-test (reporting only)
    test_ctx = load_contexts("student_test", cfg)
    rows = {}
    for tag, params in (("default_decoder", base), ("tuned_decoder", tuned)):
        df = route_eval(test_ctx, cfg, params, model=model)
        df.to_csv(out / f"student_test_trips_{tag}.csv", index=False)
        rows[tag] = summarise_route_eval(df)
    summary = pd.DataFrame(rows).T
    summary.to_csv(out / "student_test_summary.csv")
    print(summary.T, flush=True)

    per = pd.read_csv(out / "student_test_trips_tuned_decoder.csv", dtype={"trip": str})
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    x = np.arange(len(per)); w = 0.38
    for a, (metric, title) in zip(ax, [("top1_accuracy", "Observation Top-1"), ("arc_f1", "Route arc F1"),
                                       ("route_mismatch", "Route mismatch (lower = better)")]):
        a.bar(x - w / 2, per[f"classifier_{metric}"], w, label="classifier only")
        a.bar(x + w / 2, per[f"hmm_{metric}"], w, label="classifier + HMM")
        a.set_xticks(x, per["trip"], rotation=60); a.set(title=title)
    ax[0].legend()
    fig.suptitle("Student-test trips: classifier-only vs classifier + HMM/Viterbi (tuned decoder)")
    fig.tight_layout(); fig.savefig(out / "classifier_vs_hmm_per_trip.png", dpi=130); plt.close(fig)

    s1 = g[g.stage == "beta_x_emission"].pivot(index="emission_weight", columns="beta", values="hmm_top1_accuracy")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(s1.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(s1.columns)), s1.columns); ax.set_yticks(range(len(s1.index)), s1.index)
    for i in range(s1.shape[0]):
        for j in range(s1.shape[1]):
            ax.text(j, i, f"{s1.values[i, j]:.4f}", ha="center", va="center", color="w", fontsize=8)
    ax.set(xlabel="beta (m)", ylabel="emission weight", title="Decoder grid: HMM Top-1 on dev training trips (OOF)")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(out / "decoder_grid.png", dpi=130); plt.close(fig)

    # ---------------- export final inference artifacts
    with open(resolve(cfg, "reports") / "tuning" / f"best_params_{args.model_name}.json", encoding="utf-8") as f:
        bp = json.load(f)
    meta = {
        "model_name": f"{args.model_name} (Optuna-tuned)", "model_params": bp["best_params"],
        "decoder_tuned_on": dev, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "student_test": rows["tuned_decoder"],
    }
    PathSnapPredictor(model, copy.deepcopy(cfg), tuned, meta).save(resolve(cfg, "models"))
    save_json({"tuned_hmm": tuned, "dev_trips": dev}, out / "tuned_decoder.json")
    print("exported final model to", resolve(cfg, "models"))


if __name__ == "__main__":
    main()
