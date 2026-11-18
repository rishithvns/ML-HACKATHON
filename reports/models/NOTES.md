# Notes on these results

- These results come from a full rerun on 18 Nov 2026 (15:13–15:45) on Rishit's laptop: Windows 11, 16 threads, 24 GB RAM, CPU only. Nothing else heavy was running, so the timings are clean. Only an idle API process was up.
- This rerun followed a fix in `pathsnap/hmm.py`. Before the fix, cached shortest-path searches could allow transitions beyond the step's distance cutoff, so HMM results depended on what had run earlier on the same trip. The classifier metrics (accuracy, F1, log loss) were identical before and after the fix; only the HMM columns moved, by at most 0.0005 in Top-1.
- Primary selection metric: observation-level Top-1 accuracy in 5-fold group CV over training trips, with log loss as a tiebreaker. The HMM consumes the probabilities, so their quality matters.
- Student-test Top-1 values are point-weighted over all label-valid points of the 10 student-test trips. Fixes with no candidate count as wrong.
