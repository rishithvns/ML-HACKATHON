# PathSnap: map-matching and route reconstruction

PathSnap matches noisy GPS trajectories to a road network. The pipeline has five steps:

1. **Candidates**: up to 8 directed road arcs within 50 m of each GPS fix.
2. **Classifier**: scores each (fix, candidate arc) pair from 22 geometric and kinematic features.
3. **HMM/Viterbi**: picks one connected sequence of arcs using the classifier scores and road connectivity.
4. **Route filling**: inserts the shortest-path arcs between consecutive matched arcs so the route is continuous.
5. **Serving**: a FastAPI service (Laptop 1) runs the pipeline. A browser UI (Laptop 2) picks a trip and shows the result.

```
pathsnap/            library code
  io.py              raw file readers (load_trip never reads .route)
  geo.py             local projection, point-to-segment distance, bearings
  network.py         directed road graph, KD-tree candidate search, bounded Dijkstra
  labels.py          per-observation reference labels (monotone alignment to .route)
  features.py        candidate generation + feature engineering
  dataset.py         trip -> candidate/point tables, label attachment, caching
  models.py          the 10 compared models
  training.py        group-aware CV, fitting, evaluation helpers
  hmm.py             HMM/Viterbi decoder and route reconstruction
  metrics.py         candidate, observation and route-level metrics
  pipeline.py        inference: trip folder -> predicted route (no labels)
  evaluation.py      server-side scoring against .route
  viz.py             Folium map builder
scripts/             reproducible experiment entry points (run with python -m)
api/main.py          FastAPI backend (Laptop 1)
ui/                  static evaluator UI + tiny server (Laptop 2)
configs/default.json all tunable settings (radius, K, CV folds, decoder parameters, seed)
models/              final_model.joblib + pipeline_config.json (exported artifacts)
reports/             EDA, model comparison, Optuna, HMM results, Folium maps, logs
```

## Setup (Laptop 1: training and API)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Put the raw data under `data/raw/train/<trip>/...` and `data/raw/student_test/<trip>/...`.

## Reproduce the experiments

Run these in order from the project root. Each step caches what the next one needs.

```powershell
.\.venv\Scripts\python -m scripts.eda             # reports/eda: inventory, plots, data-quality notes, Folium map
.\.venv\Scripts\python -m scripts.compare_models  # reports/models: 10 models, 5-fold group CV + student-test + HMM
.\.venv\Scripts\python -m scripts.tune_bayes --model lightgbm --trials 40   # reports/tuning: Optuna TPE
.\.venv\Scripts\python -m scripts.tune_hmm        # reports/hmm + exports models/final_model.joblib
.\.venv\Scripts\python -m scripts.make_maps       # reports/maps: interactive Folium result maps
```

All randomness uses `seed` from `configs/default.json`.

### Data partitions

| Partition | Used for |
|---|---|
| `train` (80 trips) | model fitting, 5-fold `GroupKFold` CV (groups = trips), Optuna search, decoder tuning on out-of-fold probabilities |
| `student_test` (10 trips) | development-stage comparison of models, tuned vs untuned, classifier vs HMM |
| evaluator (10 trips) | never used in development; served only through the API from `data/eval/` |

### Labels

The `.route` file gives the arcs in travel order but does not say which GPS fix lies on which arc. `labels.py` assigns each fix to a route position using a monotone dynamic programme. Time order is preserved, and the cost is point-to-segment distance plus a penalty when the fix's heading opposes the arc. A fix more than 50 m from its aligned arc is marked `label_valid = False`. Such fixes are excluded from training and from point-level scoring. This mainly affects train trips 00000033 and 00000054, which show a constant ~130 m offset (see `reports/eda/data_quality.md`).

## Deployment on two laptops

**Laptop 1 (API)**

```powershell
.\.venv\Scripts\python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- Copy the 10 evaluator trips to `data/eval/<trip_id>/` before starting.
- Each request is logged to the console and to `reports/logs/api.log`.
- To rehearse with the student-test trips: `$env:PATHSNAP_EVAL_DIR="data/raw/student_test"` before starting.
- Windows Firewall must allow inbound TCP 8000.
- Find the laptop's IP with `ipconfig`.

**Laptop 2 (UI)** needs only Python and a browser.

```powershell
python ui/serve.py --port 8501
```

Open `http://localhost:8501/?api=http://<laptop-1-ip>:8000`, or type the API address into the page. The UI lists the trips, sends the chosen `trip_id` to `POST /predict`, and draws the GPS points, nearby roads, the classifier-only route and the HMM route, along with the server-computed metrics.

### API

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | model and evaluation-folder status |
| GET | `/trips` | evaluation trip ids and point counts |
| POST | `/predict` `{"trip_id": "..."}` | GPS points, predicted route geometry, per-point arcs, nearby road network, server-side metrics, timings |

The reference route is read only inside `pathsnap/evaluation.py`, after prediction. It is never returned in a response.

## Metrics

- **Candidate level** (binary: is this candidate the true arc?): accuracy, precision, recall, F1, macro F1, log loss, confusion matrix.
- **Observation Top-1**: fraction of label-valid fixes whose predicted arc equals the reference arc. Fixes without a prediction count as wrong.
- **Candidate miss rate**: share of label-valid fixes whose reference arc is not among their candidates.
- **Route continuity**: share of consecutive arcs in the reconstructed route that are physically connected (head of one arc = tail of the next).
- **Sequence level**: arc precision, recall and F1 between predicted and reference routes, length-weighted F1, and route mismatch fraction ((missing + extra length) / reference length, Newson & Krumm 2009).
