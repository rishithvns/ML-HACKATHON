# Bayesian tuning & HMM/Viterbi (M3: Rishj)



## 1. What are the hidden states?
In our project, the hidden states are the candidate road arcs generated for each GPS observation.
- For every GPS point, we generate up to 8 nearby candidate road arcs within a 50-metre search radius.
- Each candidate road arc represents a possible road on which the GPS point could have been recorded.
- We also calculate the projected position of the GPS point along each candidate arc using t_frac and the road arc's length.
The HMM uses these candidate road arcs as hidden states and selects the most likely sequence of road arcs corresponding to the GPS trajectory.

## 2. How is the emission probability calculated?
We use the LightGBM classifier to calculate the emission probability.
- The candidate road arcs are passed through the trained LightGBM model.
- The model uses 22 geometric and kinematic features to predict whether a candidate road arc is the correct match for a GPS point.
- The predicted probability is used as the emission probability.
In our implementation, the probability is converted into a log probability and multiplied by an emission_weight:
E = w*log(max(P,min_prob))
Here, P is the probability predicted by LightGBM, and \(w\) is the emission weight.
This allows the HMM to consider how confident the classifier is about each candidate road arc.

## 3. How is the transition probability calculated? (what does beta do?)
The transition probability checks whether moving from one candidate road arc to another is reasonable according to the road network.
We calculate two distances:
- GPS distance (d_gps): Straight-line distance between two consecutive GPS observations.
- Route distance (d_route): Distance travelled through the road network between the projected positions of the two candidate road arcs.
We calculate the transition score using:
  T = -|d_route - d_gps| / beta_eff
The transition score is higher when the road-network distance is close to the GPS distance.
Role of beta:
- Smaller beta: The model penalises differences between the two distances more heavily.
- Larger beta: The model allows larger differences between the distances.
Our implementation also adjusts beta according to the time difference between observations:
 beta_eff = beta*sqrt(max(1,delta_t))
 This makes the transition model more flexible when GPS observations are farther apart in time.
Additionally, our implementation applies a U-turn penalty (uturn_penalty) when transitioning onto the reverse or twin road arc.
  
## 4. What happens on a large GPS gap or when two points' roads are not connected?
Our implementation handles these situations as follows:
1. Large GPS time gap: If the time difference between consecutive observations exceeds max_gap_s, the HMM breaks the current sequence and starts a new segment.
2. Disconnected roads: If there is no valid path between two candidate road arcs in the road network, that transition is considered invalid and receives a negative infinity score.
3. No reachable candidate: If none of the current candidate road arcs can be reached from the previous states, the HMM starts a new segment.
4. No candidate road: If a GPS point has no candidate road within the search radius, it is skipped and receives no prediction (-1).
This prevents the HMM from forcing an unrealistic road connection.

## 5. Optuna: what was searched, objective, number of trials, best values
We used Optuna with a TPE sampler to tune eight LightGBM hyperparameters: number of estimators, learning rate, number of leaves, minimum child samples, subsampling ratio, feature sampling ratio, and L1 and L2 regularization. The objective was to maximize mean observation-level Top-1 accuracy using 3-fold GroupKFold cross-validation. We conducted 40 trials (the first 10 random, then guided by TPE), of which 33 were completed and 7 were pruned. The best configuration (trial 30) achieved 89.02% accuracy, with 150 estimators, a learning rate of 0.016, 149 leaves, 52 minimum child samples, a subsampling ratio of 0.92, a feature sampling ratio of 0.61, L1 regularization of 0.0055, and L2 regularization of 0.0125.

## 6. Did tuning help? (tuned_vs_untuned_lightgbm.csv)
Tuning did not significantly improve the classifier on its own. Cross-validation Top-1 stayed about the same (89.32% untuned vs 89.26% tuned), test Top-1 decreased from 89.93% to 89.41%, and log loss got worse (test 0.086 to 0.106), because the tuned model uses fewer trees with a low learning rate and gives softer probabilities. However, after HMM decoding the tuned model is slightly better (90.94% vs 91.14% with the default decoder), and it trains faster (6.5 s to 4.3 s). We deployed the tuned model.

## 7. Did the HMM help? (reports/hmm/student_test_summary.csv: top-1, route continuity, arc F1)
Yes, the HMM improved route prediction quality compared to using the classifier alone. Top-1 accuracy increased from 89.41% to 90.62%, Arc F1 increased from 89.50% to 94.14%, and route continuity improved significantly from 82.75% to 99.87%. Route mismatch also decreased from 11.54% to 7.92%. The HMM improved Top-1 on 9 of the 10 student-test trips; it did not help on trip 00000040 (94.6% to 93.1%). This indicates that the HMM helped produce more consistent and realistic routes by considering road-network connectivity and GPS movement consistency.
