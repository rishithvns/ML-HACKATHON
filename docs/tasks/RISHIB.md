# Rishib: M2, features & model comparison

## Before the 12:15 evaluation

1. Accept the GitHub invite to **rishithvns/ML-HACKATHON** (email or github.com/notifications).
2. Open https://github.com/rishithvns/ML-HACKATHON and read these files:
   - `pathsnap/features.py`: the 22 features
   - `pathsnap/models.py`: the 10 models
   - `pathsnap/training.py`: CV grouped by trip
   - `scripts/compare_models.py`
   - `reports/models/model_comparison.csv`
3. On the repo home click **Add file → Create new file** and name it `docs/notes/features_models.md`.
4. Write, in your own words:
   - **Candidates:** what counts as one (arcs within 50 m, at most 8) and why
   - **Features:** 5 of them (`dist`, `cos_head`, `along_speed`, `win_mean_dist`, `is_oneway`), with one line each on why it helps
   - **Group CV:** why the CV is grouped by trip
   - **Best model:** which one won in the comparison table and why
5. Click **Commit changes** and use the message "Add notes on features and model comparison".
6. Open `pathsnap/hmm.py`, click a line number, then "…" → *Reference in new issue*. Ask Rishj a question and submit.

## After 12:15: rerun your slice on your own laptop

7. Install Git and Python 3.11, then in PowerShell run:
   ```
   git clone https://github.com/rishithvns/ML-HACKATHON.git
   cd ML-HACKATHON
   python -m venv .venv
   .\.venv\Scripts\python -m pip install --upgrade pip
   .\.venv\Scripts\python -m pip install -r requirements.txt
   ```
8. Unzip `Student folder.zip` (Rishit will send it). Copy its `train` and `student_test` folders into `ML-HACKATHON\data\raw\`.
9. Run the comparison (about 30 min):
   ```
   git checkout -b m2-features-models
   .\.venv\Scripts\python -m scripts.compare_models
   ```
10. Commit and push:
   ```
   git add reports/models reports/logs/compare_models.log
   git commit -m "Rerun model comparison on my laptop"
   git push -u origin m2-features-models
   ```
11. On GitHub, click **Compare & pull request → Create pull request**.
12. In the report, write §3.1 (comparison table), §3.3 (plots), and the §3.2 write-ups for Decision Tree, Random Forest and Extra Trees.
