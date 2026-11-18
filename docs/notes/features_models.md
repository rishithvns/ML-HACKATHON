# Features & model comparison

> Sources: `pathsnap/features.py`, `pathsnap/models.py`, `pathsnap/training.py`, `reports/models/model_comparison.csv`

## 1. What is a candidate, and why 50 m / 8 candidates?
A candidate is a road arc that a GPS point could belong to.
Why 50m? -> The radius comes from the data 
On normal trips, the typical (median per-trip) distance from the GPS points to the reference route is about 4.6 m, and the 95th percentile is about 15 m.
50 m leaves some margin for noise while keeping the candidate set small.  The candidate miss rate on student test is 0.28%.A larger radius adds more wrong candidates and costs more compute, with little gain in coverage


## 2. Five features and why each helps
| Feature | What it measures | Why it helps |
|---|---|---|
| `dist` | Distance in metres from the GPS point to the candidate arc |The nearest road is usually the right one, so it is the strongest basic signal.|
| `cos_head` | Cosine between the vehicle’s motion direction and the arc’s direction|It separates an arc from its reverse twin, which lies at the same distance.|
| `along_speed` |Speed (m/s) times cos_head, the part of the movement that goes along the arc | Fast movement along an arc supports it. Sideways or near-zero movement does not. |
| `win_mean_dist` |Mean distance of the +-3 neighbouring GPS points to the same arc | One noisy point is corrected by its neighbours. |
| `is_oneway` | 1 if the network has no arc running in the opposite direction, else 0 | A one-way arc has no reverse twin. The flag lets the model weigh cos_head differently on one-way and two-way roads.|

## 3. Why is cross-validation grouped by trip?
Points that come one after another on a trip are almost the same: same road, same speed, same neighbours. If we split the data randomly, a test point could have near-copies of itself in the training data. The model would then do well just by recognising trips it has already seen, and the score would look better than it should. GroupKFold keeps every point of a trip together, so each trip is either in training or in testing, never both. This matches the real situation, where the evaluator gives us whole trips the model has never seen. The downside is that the score jumps around more from fold to fold (a fold-to-fold standard deviation of about 0.043 in Top-1), because some trips are much harder than others.

## 4. Which model was best, and why? (use the numbers in model_comparison.csv)
LightGBM was best. It has the highest test top-1 (0.899), the highest test macro F1 (0.937) and the lowest test log loss (0.086), which XGBoost matches. It also fits in about 11 s, against about 42 s for Random Forest (idle-machine rerun).

The lead is small, though. LightGBM, Random Forest and XGBoost all have a CV top-1 of 0.893. After HMM decoding, every model lands between 0.904 and 0.909. So the model matters less than the decoder, and it would be wrong to claim LightGBM is decisively better than Random Forest or XGBoost. We picked it because it is at least as good on every metric and is cheaper to train.

Log loss matters here because the HMM uses the classifier’s probabilities. LightGBM’s 0.086 is much better than the weak models, which makes it a good fit for the decoder.

Boosting and bagging beat Logistic Regression (CV 0.886, log loss 0.109) because the features interact. Distance matters more when the heading also agrees.

## 5. One weakness you noticed (e.g. KNN inference time, Gaussian NB log loss)
KNN overfits. Its train top-1 is 0.992 but its CV top-1 is 0.882, a gap of about 0.11. It memorises the training points, with distance weighting letting each training point dominate its own neighbourhood. Its log loss (0.143) is also among the worst, so its probabilities are poorly calibrated.

Gaussian NB has the worst probabilities. Its test log loss is 0.478, against 0.086 for LightGBM, and it has the lowest top-1 (0.873). Naive Bayes assumes the features are independent, but several are strongly correlated (head_diff and cos_head r = -0.995, dist and dist_minus_min r = 0.92). That makes its probabilities overconfident, which would hurt a decoder that relies on them. Its train and CV scores are close (0.870 and 0.873), which points to underfitting, not overfitting.
