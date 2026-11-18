"""Model zoo for candidate scoring. Every model exposes predict_proba.

`max_train_rows` caps the training rows for algorithms whose cost grows
super-linearly (KNN prediction, kernel/MLP training); rows are sampled by
whole observations so a point's candidate set stays together.
"""
from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def _lgbm(seed, **kw):
    from lightgbm import LGBMClassifier
    p = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
             random_state=seed, n_jobs=-1, verbose=-1)
    p.update(kw)
    return LGBMClassifier(**p)


def _xgb(seed, **kw):
    from xgboost import XGBClassifier
    p = dict(n_estimators=400, learning_rate=0.05, max_depth=7, subsample=0.8,
             colsample_bytree=0.8, tree_method="hist", eval_metric="logloss",
             random_state=seed, n_jobs=-1)
    p.update(kw)
    return XGBClassifier(**p)


def model_specs(seed: int = 42) -> dict:
    return {
        "logistic_regression": {
            "family": "Linear / probabilistic",
            "config": "StandardScaler + L2 LogisticRegression, C=1.0, lbfgs",
            "make": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
            "max_train_rows": None,
        },
        "gaussian_nb": {
            "family": "Linear / probabilistic",
            "config": "GaussianNB, var_smoothing=1e-9",
            "make": lambda: GaussianNB(),
            "max_train_rows": None,
        },
        "knn": {
            "family": "Instance-based",
            "config": "StandardScaler + KNN k=25, distance-weighted",
            "make": lambda: make_pipeline(StandardScaler(),
                                          KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)),
            "max_train_rows": 150_000,
        },
        "decision_tree": {
            "family": "Tree-based",
            "config": "DecisionTree max_depth=12, min_samples_leaf=50",
            "make": lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, random_state=seed),
            "max_train_rows": None,
        },
        "random_forest": {
            "family": "Ensemble (bagging)",
            "config": "RandomForest 200 trees, max_depth=20, min_samples_leaf=5",
            "make": lambda: RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=5,
                                                   n_jobs=-1, random_state=seed),
            "max_train_rows": 400_000,
        },
        "extra_trees": {
            "family": "Ensemble (bagging)",
            "config": "ExtraTrees 200 trees, max_depth=20, min_samples_leaf=5",
            "make": lambda: ExtraTreesClassifier(n_estimators=200, max_depth=20, min_samples_leaf=5,
                                                 n_jobs=-1, random_state=seed),
            "max_train_rows": 400_000,
        },
        "hist_gradient_boosting": {
            "family": "Boosting",
            "config": "HistGradientBoosting lr=0.1, max_iter=300, max_leaf_nodes=31",
            "make": lambda: HistGradientBoostingClassifier(learning_rate=0.1, max_iter=300, max_leaf_nodes=31,
                                                           early_stopping=False, random_state=seed),
            "max_train_rows": None,
        },
        "lightgbm": {
            "family": "Boosting",
            "config": "LightGBM 400 trees, lr=0.05, num_leaves=63, subsample=0.8",
            "make": lambda: _lgbm(seed),
            "max_train_rows": None,
        },
        "xgboost": {
            "family": "Boosting",
            "config": "XGBoost hist 400 trees, lr=0.05, max_depth=7, subsample=0.8",
            "make": lambda: _xgb(seed),
            "max_train_rows": None,
        },
        "mlp": {
            "family": "Neural network",
            "config": "StandardScaler + MLP (64, 32) ReLU, adam lr=1e-3, alpha=1e-4, 30 epochs",
            "make": lambda: make_pipeline(StandardScaler(), MLPClassifier(
                hidden_layer_sizes=(64, 32), alpha=1e-4, learning_rate_init=1e-3, batch_size=512,
                max_iter=30, early_stopping=True, n_iter_no_change=5, random_state=seed)),
            "max_train_rows": 400_000,
        },
    }


def make_lgbm(seed: int = 42, **params):
    return _lgbm(seed, **params)
