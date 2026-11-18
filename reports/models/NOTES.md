# Notes on these results

- The fit and inference times for random_forest, extra_trees, hist_gradient_boosting, lightgbm, xgboost and mlp were measured while `scripts/tune_bayes.py` (6 threads) ran at the same time on the same 16-thread laptop. Treat those timings as upper bounds. Rerun `python -m scripts.compare_models` on an idle machine for clean timings. Accuracy, F1 and log loss are not affected.
- Primary selection metric: observation-level Top-1 accuracy in 5-fold group CV over training trips, with log loss as a tiebreaker. The HMM consumes the probabilities, so their quality matters.
- Student-test Top-1 values are point-weighted over all label-valid points of the 10 student-test trips. Fixes with no candidate count as wrong.
