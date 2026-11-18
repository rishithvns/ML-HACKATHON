# PathSnap: results at a glance

Every number below comes from files in `reports/` and can be regenerated with the scripts listed. The partitions used are:
- **train**: 80 trips, for fitting, 5-fold group CV and tuning
- **student_test**: 10 trips, for development comparison

The evaluator trips were never used.

## 1. EDA (`python -m scripts.eda` → `reports/eda/`)

| | Train | Student-test |
|---|---|---|
| Trips | 80 | 10 |
| GPS points | 208,417 | 19,160 |
| Share of 1-second steps | 93.8 % | 98.2 % |
| Trips with a gap > 60 s | 22 | 1 |
| Median GPS-to-route distance (per-trip median) | 4.64 m | 4.75 m |

- **Data quality:** 0 missing values. Train trips 00000033 and 00000054 are offset ~120–140 m from their reference route. Points more than 50 m from the route get `label_valid = False`.
- **Files:** `data_quality.md`, `inventory.csv`, `partition_summary.csv`, plots `*.png`, and Folium maps `map_*.html`.

## 2. Feature engineering & selection (`pathsnap/features.py`, `python -m scripts.feature_selection` → `reports/features/`)

Each GPS point gets up to 8 candidate arcs within 50 m. The true arc is among them for 99.7 % of student-test points. Each (point, candidate) pair has 22 features, covering distance, heading, along-arc motion, a ±3-point window, and graph degree.

Selection was done on training trips only (held-out CV fold 0):

| Features used | Fold-0 Top-1 |
|---|---|
| all 22 | **0.9157** |
| drop the 4 least important | 0.9148 |
| drop 8 | 0.9146 |
| drop 12 | 0.9132 |
| without the window features | 0.9052 |
| without the heading features | 0.6335 |

- **Top features by permutation importance:** `along_speed`, `win_minus_min`, `at_endpoint`, `is_oneway`, `t_frac`.
- **Highly correlated pairs:** `head_diff`/`cos_head` (r = −0.995) and `dist`/`dist_minus_min` (r = 0.92).
- **Decision:** keep all 22. Dropping features never improves held-out Top-1.

## 3. Models and accuracies (`python -m scripts.compare_models` → `reports/models/`)

Ten models from 7 families were compared. CV is 5-fold `GroupKFold` by trip. "+HMM" means the same model's probabilities decoded with the HMM/Viterbi decoder.

| Model | Algorithm family | CV macro F1 | CV Top-1 | Student-test Top-1 (classifier) | Student-test Top-1 (+HMM) | Student-test macro F1 | Student-test log loss |
|---|---|---|---|---|---|---|---|
| random_forest | Ensemble (bagging) | 0.9345 ± 0.0255 | 0.8933 | 0.8936 | 0.9094 | 0.9344 | 0.0876 |
| **lightgbm** | Boosting | 0.9337 ± 0.0262 | 0.8932 | **0.8993** | 0.9091 | **0.9366** | **0.0855** |
| xgboost | Boosting | 0.9341 ± 0.026 | 0.8925 | 0.8974 | 0.9075 | 0.936 | 0.0857 |
| extra_trees | Ensemble (bagging) | 0.9328 ± 0.0248 | 0.8911 | 0.8896 | 0.9081 | 0.9309 | 0.0923 |
| mlp | Neural network | 0.9307 ± 0.0261 | 0.8896 | 0.8828 | 0.906 | 0.9259 | 0.0935 |
| decision_tree | Tree-based | 0.9296 ± 0.0275 | 0.888 | 0.89 | 0.9084 | 0.9284 | 0.1058 |
| logistic_regression | Linear / probabilistic | 0.9267 ± 0.0244 | 0.8858 | 0.8831 | 0.9062 | 0.9263 | 0.1086 |
| hist_gradient_boosting | Boosting | 0.9302 ± 0.0262 | 0.8849 | 0.8866 | 0.908 | 0.9303 | 0.1113 |
| knn | Instance-based | 0.926 ± 0.0239 | 0.8817 | 0.8827 | 0.9037 | 0.9253 | 0.1426 |
| gaussian_nb | Linear / probabilistic | 0.8851 ± 0.0215 | 0.8726 | 0.8732 | 0.905 | 0.8956 | 0.4778 |

**Selected model: LightGBM.**
- On CV Top-1 (the primary metric) it is tied with Random Forest, which leads by 0.0001.
- It has the best log loss, which matters because the HMM uses its probabilities.
- It trains about 4× faster than Random Forest.

Per-model details are in `reports/models/`:
- `result_*.json` and per-fold CSVs
- `confusion_matrices.png`
- `training_curves.png` (LightGBM, XGBoost, MLP)
- `cv_macro_f1.png`, `student_test_top1_f1.png`

## 4. Bayesian hyperparameter optimisation (`python -m scripts.tune_bayes` → `reports/tuning/`)

- **Method:** Optuna TPE sampler, 15 trials (13 completed, 2 pruned by the median pruner), 3-fold group CV on training trips.
- **Objective:** maximise mean CV observation Top-1.
- **Best parameters:** 300 trees, learning rate 0.048, 50 leaves, min_child_samples 26.

| | Untuned | Tuned |
|---|---|---|
| CV Top-1 | 0.8932 | 0.8925 |
| CV macro F1 | 0.9337 | 0.9339 |
| CV log loss | 0.0941 | 0.0931 |
| Student-test Top-1 | 0.8993 | 0.8966 |
| Fit time (s) | 12.3 | 6.9 |

Tuning gave no meaningful accuracy gain; the defaults were already near-optimal. It did give slightly better CV log loss and halved the training time.

## 5. HMM/Viterbi route reconstruction (`python -m scripts.tune_hmm` → `reports/hmm/`)

Student-test results for the tuned model with the tuned decoder (β = 10 m, emission weight 0.5, U-turn penalty 10):

| | Classifier only | Classifier + HMM |
|---|---|---|
| Observation Top-1 | 0.8966 | **0.9052** |
| Route continuity | 0.838 | **0.999** |
| Route arc F1 | 0.899 | **0.939** |
| Route mismatch (lower is better) | 0.114 | **0.078** |

The HMM did not help on trips 00000077 and 00000040 (see `reports/hmm/student_test_trips_tuned_decoder.csv`).
