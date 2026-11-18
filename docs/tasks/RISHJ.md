# Rishj: M3, Bayesian tuning & HMM/Viterbi

## Before the 12:15 evaluation

1. Accept the GitHub invite to **rishithvns/ML-HACKATHON** (email or github.com/notifications).
2. Open https://github.com/rishithvns/ML-HACKATHON and read these files:
   - `pathsnap/hmm.py`: the docstring at the top explains the HMM
   - `scripts/tune_bayes.py`: the Optuna search
   - `reports/tuning/tuned_vs_untuned_lightgbm.csv`, `best_params_lightgbm.json`, `optuna_history_lightgbm.png`
   - `reports/hmm/student_test_summary.csv`, `decoder_grid.csv`
3. Click **Add file → Create new file** and name it `docs/notes/tuning_hmm.md`.
4. Write, in your own words:
   - **Hidden states:** the candidate road arcs of each GPS point
   - **Emission:** the classifier's probability
   - **Transition:** how well the road distance matches the GPS distance (beta)
   - **GPS gaps and disconnected roads:** the chain breaks and restarts
   - **Optuna:** what was searched, how many trials, and the honest result (tuned ≈ untuned, but faster training)
   - **Classifier vs HMM:** does the HMM help? Use `reports/hmm/student_test_summary.csv`
5. Click **Commit changes** and use the message "Add notes on Optuna and HMM decoding".
6. Open `api/main.py`, click a line number, then "…" → *Reference in new issue*. Ask Rhea a question and submit.

## After 12:15: rerun your slice on your own laptop

7. Set up (same as Rishib's steps 7–8): clone, create the venv, `pip install -r requirements.txt`, copy the data into `data\raw\`.
8. Run the tuning (about 50 min):
   ```
   git checkout -b m3-tuning-hmm
   .\.venv\Scripts\python -m scripts.tune_bayes --model lightgbm --trials 40
   .\.venv\Scripts\python -m scripts.tune_hmm
   ```
9. Commit and push:
   ```
   git add reports/tuning reports/hmm models/final_model.joblib models/pipeline_config.json
   git commit -m "Rerun Optuna and decoder tuning on my laptop"
   git push -u origin m3-tuning-hmm
   ```
10. Open a pull request. Rhea needs your model.
11. In the report, write §4 (Bayesian optimisation), §5 (HMM/Viterbi), and the §3.2 write-ups for LightGBM, XGBoost and HistGradientBoosting.
