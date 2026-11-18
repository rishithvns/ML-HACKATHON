# Feature selection summary (training trips, CV fold 0)

## Importance (sorted by permutation importance)

```
                gain_share  permutation_logloss_increase  permutation_std
along_speed        0.16336                       0.26529          0.00138
win_minus_min      0.12841                       0.09342          0.00087
at_endpoint        0.02781                       0.02849          0.00046
is_oneway          0.00769                       0.02222          0.00013
t_frac             0.01244                       0.01666          0.00066
head_diff          0.06194                       0.01581          0.00029
cos_head           0.11427                       0.01485          0.00033
dist_minus_min     0.34413                       0.01451          0.00040
out_deg            0.00632                       0.00564          0.00022
speed              0.01437                       0.00555          0.00025
n_cand             0.00675                       0.00346          0.00019
win_rank           0.02710                       0.00333          0.00014
win_max_dist       0.00756                       0.00280          0.00007
in_deg             0.00616                       0.00237          0.00008
win_mean_dist      0.01158                       0.00213          0.00011
dist_ratio         0.00447                       0.00096          0.00010
log_arc_len        0.04018                       0.00073          0.00031
dist               0.00508                       0.00057          0.00006
turn               0.00357                       0.00050          0.00007
rank               0.00618                       0.00028          0.00010
dt_prev            0.00063                       0.00018          0.00004
head_valid         0.00000                       0.00000          0.00000
```

## Highly correlated pairs (|r| >= 0.9)

- dist / dist_minus_min: 0.921
- head_diff / cos_head: -0.995

## Ablation (fold-0 held-out Top-1)

```
 n_features                                                                                                               dropped  fold0_top1
         22                                                                                                                            0.9157
         18                                                                                       turn, rank, dt_prev, head_valid      0.9148
         14                                         win_mean_dist, dist_ratio, log_arc_len, dist, turn, rank, dt_prev, head_valid      0.9146
         10 n_cand, win_rank, win_max_dist, in_deg, win_mean_dist, dist_ratio, log_arc_len, dist, turn, rank, dt_prev, head_valid      0.9132
         18                                                                                                    no window features      0.9052
         18                                                                                                   no heading features      0.6335
```
