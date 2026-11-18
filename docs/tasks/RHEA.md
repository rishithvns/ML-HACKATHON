# Rhea: M4, API, UI, maps & demo

## Before the 12:15 evaluation

1. Accept the GitHub invite to **rishithvns/ML-HACKATHON** (email or github.com/notifications).
2. Open https://github.com/rishithvns/ML-HACKATHON and read these files:
   - `api/main.py`: the endpoints `/health`, `/trips` and `/predict`
   - `ui/app.js`: how the UI calls the API and draws the map
   - `pathsnap/pipeline.py`: the prediction, which never reads `.route`
   - `pathsnap/evaluation.py`: the scoring, done on the server only
   - the README section "Deployment on two laptops"
3. Click **Add file → Create new file** and name it `docs/notes/deployment.md`.
4. Write, in your own words:
   - **Request flow:** the UI calls `GET /trips`, the user picks a trip, `POST /predict` runs the pipeline, and the map and metrics are drawn
   - **Labels:** how they stay on the server and are never sent to the UI
   - **Two-laptop demo:**
     - Laptop 1: `python -m uvicorn api.main:app --host 0.0.0.0 --port 8000`; allow port 8000 in the firewall; get the IP from `ipconfig`
     - Laptop 2: `python ui/serve.py`, then open `http://localhost:8501/?api=http://<laptop1-ip>:8000`
5. Click **Commit changes** and use the message "Add notes on API, UI and deployment".
6. Open `pathsnap/features.py`, click a line number, then "…" → *Reference in new issue*. Ask Rishib a question and submit.

## After 12:15: rerun your slice on your own laptop

7. Set up (same as Rishib's steps 7–8): clone, create the venv, `pip install -r requirements.txt`, copy the data into `data\raw\`.
8. Generate the maps:
   ```
   git checkout -b m4-deployment
   .\.venv\Scripts\python -m scripts.make_maps
   ```
9. Rehearse the two-laptop demo. Take screenshots of the UI and of the API console showing the request logs, and save them in `reports/screenshots/`.
10. Commit and push:
   ```
   git add reports/maps reports/screenshots
   git commit -m "Result maps and demo screenshots"
   git push -u origin m4-deployment
   ```
   Then open a pull request.
11. In the report, write §6 (Folium maps) and §7 (API/UI/deployment), and the §3.2 write-ups for KNN and MLP. Then assemble the final PDF.
