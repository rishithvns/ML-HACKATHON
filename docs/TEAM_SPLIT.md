# Team split: 4 members, equal ownership

Each member owns one vertical slice. For that slice they:

- commit the code from their own GitHub account,
- rerun its experiments on their own machine so the numbers in `reports/` come from them,
- write its report sections and answer viva questions on it.

Every member also writes the per-model subsections (report §3.2) for 2–3 of the compared models, so all four can talk about model results.

Repo: https://github.com/rishithvns/ML-HACKATHON

| | Member 1: Data & EDA | Member 2: Features & models | Member 3: Tuning & HMM | Member 4: Deployment & maps |
|---|---|---|---|---|
| **Code they own** | `pathsnap/io.py`, `pathsnap/geo.py`, `pathsnap/labels.py`, `pathsnap/config.py`, `configs/default.json`, `scripts/eda.py` | `pathsnap/network.py` (spatial index part), `pathsnap/features.py`, `pathsnap/dataset.py`, `pathsnap/models.py`, `pathsnap/training.py`, `scripts/compare_models.py` | `pathsnap/hmm.py`, `pathsnap/network.py` (Dijkstra part), `pathsnap/metrics.py`, `pathsnap/experiments.py`, `scripts/tune_bayes.py`, `scripts/tune_hmm.py` | `pathsnap/pipeline.py`, `pathsnap/evaluation.py`, `pathsnap/viz.py`, `api/`, `ui/`, `scripts/make_maps.py`, `README.md`, `requirements*.txt` |
| **Runs / produces** | `python -m scripts.eda` → `reports/eda/` | `python -m scripts.compare_models` → `reports/models/` | `python -m scripts.tune_bayes`, then `python -m scripts.tune_hmm` → `reports/tuning/`, `reports/hmm/`, `models/` | `python -m scripts.make_maps` → `reports/maps/`; the two-laptop demo; API log and UI screenshots |
| **Report sections** | §2 Dataset & EDA (preprocessing, patterns, missing values, trajectory plots, network plots, Folium map, label alignment and data-quality limits) | §3.1 comparison table, §3.3 comparison plots, feature-engineering rationale | §4 Bayesian optimisation, §5 HMM/Viterbi (states, emissions, transitions, gaps, disconnected segments, classifier vs HMM) | §6 Folium maps, §7 API/UI/deployment, final PDF assembly |
| **§3.2 models they write up** | Logistic Regression, Gaussian NB | Decision Tree, Random Forest, Extra Trees | LightGBM, XGBoost, HistGradientBoosting | KNN, MLP |
| **Viva topics** | file formats, projection choice, how labels come from `.route`, anomalous trips (33, 54), leakage rules | candidate radius/K, each feature and why, group CV, choosing the final model and the primary metric | search space, objective, trials and pruning, emission/transition formulas, the Viterbi recurrence, gap handling | how the API keeps labels server-side, request flow, UI, folium layers, running on two laptops |

## Right now (before the Deliverable-1 check): about 10 minutes each, no setup needed

1. **Repo owner:** Settings → Collaborators → invite the other three. They accept the invite from email or GitHub notifications.
2. **Every member**, from their own GitHub account in the browser, reads the files they own (table below). Then they create `docs/notes/<slice>.md` with *Add file → Create new file* and write, in their own words:
   - what their slice does and why it was designed that way,
   - the numbers it produced (from `reports/`),
   - one limitation they noticed.

   Commit it with a clear message. The slices are `data_eda`, `features_models`, `tuning_hmm` and `deployment`.
3. **Every member** opens one file of another member's slice on GitHub and leaves a review comment or question on a line. Use the line's "…" menu → *Reference in new issue*, or comment on a commit.

## Running your slice on your own laptop

**One-time setup** (Windows PowerShell, Python 3.11):

```powershell
git clone https://github.com/<your-username-or-owner>/ML-HACKATHON.git
cd ML-HACKATHON
python -m pip install --upgrade pip
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Unzip `Student folder.zip` and copy its `train` and `student_test` folders to `data\raw\train` and `data\raw\student_test`.

- If pip fails with `CERTIFICATE_VERIFY_FAILED`, upgrade pip first, as shown above. New pip uses the Windows certificate store.
- If `import pandas` is blocked by an Application Control policy, keep `pandas==3.0.3` as pinned.

**Then each member runs only their slice, on a branch:**

```powershell
git checkout -b m<N>-<slice>
```

| Member | Command(s) | Time on a 16-thread laptop | Commit |
|---|---|---|---|
| M1 | `.\.venv\Scripts\python -m scripts.eda` | ~1 min | `reports/eda/` |
| M2 | `.\.venv\Scripts\python -m scripts.compare_models` | ~25–35 min (KNN, RF and MLP are slowest) | `reports/models/`, `reports/logs/compare_models.log` |
| M3 | `.\.venv\Scripts\python -m scripts.tune_bayes --model lightgbm --trials 40`, then `.\.venv\Scripts\python -m scripts.tune_hmm` | ~30–45 min + ~8 min | `reports/tuning/`, `reports/hmm/`, `models/final_model.joblib`, `models/pipeline_config.json` |
| M4 | after M3's model is merged: `git pull`, then `.\.venv\Scripts\python -m scripts.make_maps`, then run the API + UI demo | ~2 min | `reports/maps/`, screenshots of the UI and the API console |

- The first script you run builds `data/processed/` from the raw data (~1 min). It is cached and not committed.
- The Laptop 1 used in the final evaluation needs `models/final_model.joblib` from M3's merged commit, or must rerun M3's steps.
- Push your branch and open a pull request. Another member reviews it and merges.
- If your numbers differ slightly from someone else's run, that's expected: thread scheduling and timing vary. Report the numbers from your own run.

## After Deliverable-1

Each member clones the repo, sets up the venv (see README), reruns their slice's scripts on their own laptop, and commits the regenerated `reports/` outputs from their own account. They also write their report sections. This is the main, genuine contribution. The notes above are only the starting point.

## Git workflow

The order matters because later slices import earlier ones.

1. **Member 1** creates `main` with `.gitignore`, `configs/`, `pathsnap/__init__.py` and their files, then pushes.
2. Each member works on a branch named after them, for example `m2-features-models`, and makes several small commits as they go, not one dump.
3. Open a pull request into `main`. Another member reviews it and leaves at least one comment before merging. Merge order: M1 → M2 → M3 → M4.
4. Commit generated outputs (`reports/**`, `models/final_model.joblib`, `models/pipeline_config.json`) from the machine that produced them. Never commit `data/raw/`, `data/eval/` or secrets. `.gitignore` already covers these.

Before any of this, the repo owner adds the other three members as collaborators under Settings → Collaborators, so everyone can push their own branch.

## Shared rules

- Never put evaluator trips anywhere except `data/eval/` on Laptop 1, and never use them before the final demo.
- If you change a number in `configs/default.json`, say so in your PR and rerun everything downstream of it.
- Disclose AI assistance in the report as the event's policy requires. This codebase was scaffolded with Claude Code. Each owner is responsible for understanding, verifying and being able to defend their slice. Write the report analysis in your own words from your own runs.
