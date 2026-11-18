# Deployment slice (M4: Rhea)

Files I own: `pathsnap/pipeline.py`, `pathsnap/evaluation.py`, `pathsnap/viz.py`, `api/main.py`, `ui/`, `scripts/make_maps.py`, `README.md`, `requirements*.txt`.

## What the slice does

It turns the trained model into something an evaluator can use. Laptop 1 runs a FastAPI server that holds the model and the evaluation trips. Laptop 2 runs a static web page that only talks to that server. `scripts/make_maps.py` builds the offline Folium maps for the report.

## How a request flows from the UI to the API and back

1. **UI starts** (`ui/serve.py`, port 8501). It is a plain Python static file server, so Laptop 2 needs no packages. The API address comes from `?api=http://<laptop-1-ip>:8000`, the text box, or the last value saved in the browser.
2. **Connect**: `app.js` calls `GET /health` (is the model loaded, how many trips) and `GET /trips` (trip ids and point counts from `data/eval/`). It shows one button per trip.
3. **Pick a trip**: the UI sends `POST /predict` with only `{"trip_id": "..."}`. It never sends GPS data; the trip files are already on Laptop 1.
4. **API checks the id**: it must match `[A-Za-z0-9_-]+` and be one of the folders in the eval directory. Otherwise it returns 404. This stops path tricks like `../`.
5. **Prediction** (`PathSnapPredictor.predict` in `pipeline.py`):
   - `process_trip` → `load_trip` reads only `.track`, `.nodes`, `.arcs`, builds the road graph and up to 8 candidate arcs within 50 m per GPS fix, with 22 features each.
   - The classifier gives each (fix, arc) pair a probability.
   - `decode_context` builds two answers: classifier-only (top arc per fix) and HMM/Viterbi (connected arc sequence, gaps filled with shortest paths).
6. **Scoring** (`score_prediction` in `evaluation.py`): only now, after the prediction is finished, the server reads `.route`, aligns it to the fixes and computes Top-1 accuracy, arc precision/recall/F1, length-weighted F1, route mismatch and continuity for both answers.
7. **Response**: GPS points, the HMM route and classifier route as lon/lat lines, predicted arc per point, nearby road segments, HMM chain breaks, the metric numbers and timings. The request is also logged to the console and `reports/logs/api.log`.
8. **UI draws it** with Leaflet: grey road network, blue GPS dots, dashed orange classifier route, red HMM route, and a metrics table comparing classifier vs HMM.

## How the reference labels stay on the server

- Separate functions: `load_trip` (used for prediction) never opens `.route`. Only `load_route` does, and in the API it is called only from `evaluation.py`.
- Order: scoring runs after `predictor.predict` returns, so the prediction cannot depend on the labels.
- What leaves the server: only predicted routes and aggregate numbers (accuracy, F1, counts). No reference arc ids or reference geometry are ever in the JSON.
- The UI only sends a trip id, and the evaluator trips live only in `data/eval/` on Laptop 1, which `.gitignore` keeps out of the repo.
- If a trip has no `.route`, the API still predicts and returns `metrics: null`; the UI then shows the prediction only.

The offline report maps (`make_maps.py`) do draw the green reference route, but they run only on `train`/`student_test`, never on evaluator trips.

## Numbers

To fill in after I run `make_maps` and the two-laptop rehearsal with M3's final model: per-trip Top-1 (classifier → HMM), arc F1 and request time from `reports/logs/api.log`.

## One limitation

The API has no login and allows any origin (`PATHSNAP_CORS` defaults to `*`), so anyone on the same network can call it. Also, Laptop 2 needs internet: Leaflet is loaded from unpkg.com and the map tiles come from OpenStreetMap, so on a network without internet the map would be blank even though the API works.


