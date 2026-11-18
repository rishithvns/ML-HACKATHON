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
  T = -((d_route - d_gps)/beta_eff)
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
We used Optuna with a TPE sampler to tune eight LightGBM hyperparameters: number of estimators, learning rate, number of leaves, minimum child samples, subsampling ratio, feature sampling ratio, and L1 and L2 regularization. The objective was to maximize mean observation-level Top-1 accuracy using 3-fold GroupKFold cross-validation. We conducted 15 trials, of which 13 were completed and 2 were pruned. The best configuration achieved 88.94% accuracy, with 300 estimators, a learning rate of 0.0482, 50 leaves, 26 minimum child samples, a subsampling ratio of 0.8059, a feature sampling ratio of 0.4837, L1 regularization of 0.0292, and L2 regularization of 0.0147.

## 6. Did tuning help? (tuned_vs_untuned_lightgbm.csv)
Tuning did not significantly improve the model's overall performance. The cross-validation Macro F1 score increased slightly from 93.37% to 93.39%, but test Top-1 accuracy decreased from 89.93% to 89.66%, and test Macro F1 decreased from 93.66% to 93.51%. Test log loss also increased slightly. Therefore, the tuned model showed no meaningful improvement over the original model on the test data.

## 7. Did the HMM help? (reports/hmm/student_test_summary.csv: top-1, route continuity, arc F1)
Yes, the HMM improved route prediction quality compared to using the classifier alone. Top-1 accuracy increased from 89.66% to 90.52%, Arc F1 increased from 89.86% to 93.88%, and route continuity improved significantly from 83.81% to 99.90%. Route mismatch also decreased from 11.43% to 7.81%. This indicates that the HMM helped produce more consistent and realistic routes by considering road-network connectivity and GPS movement consistency.
