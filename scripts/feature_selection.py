"""Feature analysis / selection on TRAINING trips only.

1. Correlation matrix of the 22 features (redundancy check)
2. LightGBM split-gain importance
3. Permutation importance (increase in log loss on held-out CV-fold-0 trips)
4. Ablation: drop the least important features and re-check fold-0 Top-1

The student-test and evaluator partitions are not used, so feature choices
do not leak test information.

Outputs -> reports/features/
Run:  python -m scripts.feature_selection
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from pathsnap.config import load_config, resolve  # noqa: E402
from pathsnap.dataset import build_partition  # noqa: E402
from pathsnap.features import FEATURE_COLUMNS  # noqa: E402
from pathsnap.metrics import observation_metrics  # noqa: E402
from pathsnap.models import make_lgbm  # noqa: E402
from pathsnap.training import subsample_obs, training_rows  # noqa: E402


def top1(model, cols, va_c, va_p):
    c = va_c.assign(prob=model.predict_proba(va_c[cols])[:, 1])
    return observation_metrics(c, va_p)["top1_accuracy"]


def main():
    cfg = load_config()
    seed = cfg["seed"]
    out = resolve(cfg, "reports") / "features"
    out.mkdir(parents=True, exist_ok=True)
    cands, points = build_partition("train", cfg)
    rows = training_rows(cands, cfg["training"]["train_point_stride"])
    trips = np.array(sorted(cands["trip"].unique()))
    tr_idx, va_idx = next(GroupKFold(n_splits=cfg["training"]["cv_folds"]).split(trips, groups=trips))
    tr_t, va_t = set(trips[tr_idx]), set(trips[va_idx])
    tr = rows[rows["trip"].isin(tr_t)]
    va_c, va_p = cands[cands["trip"].isin(va_t)], points[points["trip"].isin(va_t)]

    # 1. correlation
    corr = tr[FEATURE_COLUMNS].corr()
    corr.to_csv(out / "feature_correlation.csv")
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_COLUMNS)), FEATURE_COLUMNS, rotation=90)
    ax.set_yticks(range(len(FEATURE_COLUMNS)), FEATURE_COLUMNS)
    fig.colorbar(im); ax.set_title("Feature correlation (training trips)")
    fig.tight_layout(); fig.savefig(out / "feature_correlation.png", dpi=130); plt.close(fig)
    high = [(a, b, round(corr.loc[a, b], 3)) for i, a in enumerate(FEATURE_COLUMNS)
            for b in FEATURE_COLUMNS[i + 1:] if abs(corr.loc[a, b]) >= 0.9]

    # 2. gain importance
    model = make_lgbm(seed).fit(tr[FEATURE_COLUMNS], tr["y"])
    gain = pd.Series(model.booster_.feature_importance("gain"), index=FEATURE_COLUMNS)
    gain = (gain / gain.sum()).sort_values(ascending=False)

    # 3. permutation importance on held-out trips (log loss)
    vs = subsample_obs(va_c[va_c["label_valid"]], 120_000, seed)
    perm = permutation_importance(model, vs[FEATURE_COLUMNS], vs["y"], scoring="neg_log_loss",
                                  n_repeats=3, random_state=seed, n_jobs=1)
    perm_s = pd.Series(perm.importances_mean, index=FEATURE_COLUMNS)
    imp = pd.DataFrame({"gain_share": gain, "permutation_logloss_increase": perm_s,
                        "permutation_std": pd.Series(perm.importances_std, index=FEATURE_COLUMNS)})
    imp = imp.sort_values("permutation_logloss_increase", ascending=False)
    imp.to_csv(out / "feature_importance.csv")

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    g = imp["gain_share"].sort_values()
    ax[0].barh(g.index, g.values); ax[0].set(title="LightGBM split-gain share")
    p = imp["permutation_logloss_increase"].sort_values()
    ax[1].barh(p.index, p.values, xerr=imp.loc[p.index, "permutation_std"])
    ax[1].set(title="Permutation importance (log-loss increase, held-out trips)")
    fig.tight_layout(); fig.savefig(out / "feature_importance.png", dpi=130); plt.close(fig)

    # 4. ablation: drop the k least important (by permutation) features
    base = top1(model, FEATURE_COLUMNS, va_c, va_p)
    order = imp.index.tolist()
    abl = [{"n_features": len(FEATURE_COLUMNS), "dropped": "", "fold0_top1": base}]
    for k in (4, 8, 12):
        keep = order[:-k]
        m = make_lgbm(seed).fit(tr[keep], tr["y"])
        abl.append({"n_features": len(keep), "dropped": ", ".join(order[-k:]), "fold0_top1": top1(m, keep, va_c, va_p)})
    for grp, cols in {"no window features": ["win_mean_dist", "win_max_dist", "win_minus_min", "win_rank"],
                      "no heading features": ["head_diff", "cos_head", "head_valid", "along_speed"]}.items():
        keep = [c for c in FEATURE_COLUMNS if c not in cols]
        m = make_lgbm(seed).fit(tr[keep], tr["y"])
        abl.append({"n_features": len(keep), "dropped": grp, "fold0_top1": top1(m, keep, va_c, va_p)})
    abl = pd.DataFrame(abl)
    abl.to_csv(out / "feature_ablation.csv", index=False)

    with open(out / "SUMMARY.md", "w", encoding="utf-8") as f:
        f.write("# Feature selection summary (training trips, CV fold 0)\n\n")
        f.write("## Importance (sorted by permutation importance)\n\n" + "```\n" + imp.round(5).to_string() + "\n```\n\n")
        f.write("## Highly correlated pairs (|r| >= 0.9)\n\n")
        f.write("\n".join(f"- {a} / {b}: {r}" for a, b, r in high) or "- none")
        f.write("\n\n## Ablation (fold-0 held-out Top-1)\n\n" + "```\n" + abl.round(4).to_string(index=False) + "\n```\n")
    print(imp.round(4)); print(abl.round(4)); print("high corr:", high)


if __name__ == "__main__":
    main()
