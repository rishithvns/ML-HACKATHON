/* PathSnap evaluator UI: talks to the FastAPI backend on Laptop 1. */
const $ = (id) => document.getElementById(id);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* ignore */ } },
};

const params = new URLSearchParams(location.search);
const defaultApi = params.get("api") || store.get("pathsnap.api") || `http://${location.hostname || "localhost"}:8000`;
$("api-url").value = defaultApi;

const map = L.map("map", { preferCanvas: true }).setView([56.0, 54.0], 11);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);
const layers = {
  network: L.layerGroup(), gps: L.layerGroup(), classifier: L.layerGroup(), hmm: L.layerGroup(),
};
Object.values(layers).forEach((l) => l.addTo(map));
L.control.layers(null, {
  "Road network (near track)": layers.network,
  "GPS observations": layers.gps,
  "Classifier-only route": layers.classifier,
  "Predicted route (HMM/Viterbi)": layers.hmm,
}, { collapsed: false }).addTo(map);
const legend = L.control({ position: "bottomleft" });
legend.onAdd = () => {
  const d = L.DomUtil.create("div", "legend");
  d.innerHTML = '<div><i style="background:#1f77b4"></i>GPS observations</div>' +
    '<div><i style="background:#d62728"></i>Predicted route (HMM)</div>' +
    '<div><i style="background:#ff7f0e"></i>Classifier-only route</div>' +
    '<div><i style="background:#9e9e9e"></i>Road network</div>';
  return d;
};
legend.addTo(map);

function api() { return $("api-url").value.replace(/\/+$/, ""); }

function setStatus(text, cls) { const s = $("status"); s.textContent = text; s.className = `status ${cls}`; }

async function connect() {
  store.set("pathsnap.api", api());
  setStatus("connecting…", "idle");
  try {
    const h = await (await fetch(`${api()}/health`)).json();
    const t = await (await fetch(`${api()}/trips`)).json();
    setStatus(`connected · ${h.model || "model"}`, "ok");
    renderTrips(t.trips);
  } catch (e) {
    setStatus("cannot reach API", "err");
    $("trips").innerHTML = `<p class="muted">Could not reach ${api()}. Check the address and that the API is running.</p>`;
  }
}

function renderTrips(trips) {
  const box = $("trips");
  box.innerHTML = "";
  if (!trips.length) { box.innerHTML = '<p class="muted">The API has no evaluation trips.</p>'; return; }
  trips.forEach((t) => {
    const b = document.createElement("button");
    b.className = "trip";
    b.dataset.trip = t.trip_id;
    b.innerHTML = `${t.trip_id}<small>${t.n_points.toLocaleString()} GPS points</small>`;
    b.onclick = () => runTrip(t.trip_id);
    box.appendChild(b);
  });
}

const pct = (v) => (v === null || v === undefined ? "–" : `${(100 * v).toFixed(2)}%`);
const num = (v, d = 3) => (v === null || v === undefined ? "–" : Number(v).toFixed(d));

async function runTrip(tripId) {
  document.querySelectorAll(".trip").forEach((b) => b.classList.toggle("active", b.dataset.trip === tripId));
  $("loading").classList.remove("hidden");
  $("result-trip").textContent = `· trip ${tripId}`;
  try {
    const r = await fetch(`${api()}/predict`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ trip_id: tripId }),
    });
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    const res = await r.json();
    draw(res);
    renderMetrics(res);
  } catch (e) {
    $("metrics").innerHTML = `<p class="muted">Request failed: ${e.message}</p>`;
  } finally {
    $("loading").classList.add("hidden");
  }
}

function draw(res) {
  Object.values(layers).forEach((l) => l.clearLayers());
  res.road_network.forEach((seg) => L.polyline(seg, { color: "#9e9e9e", weight: 2, opacity: 0.6 }).addTo(layers.network));
  const step = Math.max(1, Math.ceil(res.gps.length / 5000));
  res.gps.forEach((p, i) => {
    if (i % step === 0) L.circleMarker(p, { radius: 2, color: "#1f77b4", weight: 0, fillOpacity: 0.8 }).addTo(layers.gps);
  });
  if (res.classifier_route.length) L.polyline(res.classifier_route, { color: "#ff7f0e", weight: 3, dashArray: "6 6", opacity: 0.9 }).addTo(layers.classifier);
  if (res.predicted_route.length) L.polyline(res.predicted_route, { color: "#d62728", weight: 5, opacity: 0.8 }).addTo(layers.hmm);
  map.fitBounds(L.latLngBounds(res.gps), { padding: [20, 20] });
}

function renderMetrics(res) {
  const m = res.metrics;
  const t = res.timings_s;
  let html = "";
  if (m) {
    html += `<div class="headline">
      <div class="kpi"><b>${pct(m.hmm_top1_accuracy)}</b><span>Top-1 accuracy (HMM)</span></div>
      <div class="kpi"><b>${pct(m.hmm_arc_f1)}</b><span>Route arc F1 (HMM)</span></div></div>
      <table><tr><th>Metric</th><th>Classifier</th><th>+ HMM</th></tr>
      <tr><td>Observation Top-1</td><td>${pct(m.classifier_top1_accuracy)}</td><td>${pct(m.hmm_top1_accuracy)}</td></tr>
      <tr><td>Route arc precision</td><td>${pct(m.classifier_arc_precision)}</td><td>${pct(m.hmm_arc_precision)}</td></tr>
      <tr><td>Route arc recall</td><td>${pct(m.classifier_arc_recall)}</td><td>${pct(m.hmm_arc_recall)}</td></tr>
      <tr><td>Route arc F1</td><td>${pct(m.classifier_arc_f1)}</td><td>${pct(m.hmm_arc_f1)}</td></tr>
      <tr><td>Length-weighted F1</td><td>${pct(m.classifier_length_f1)}</td><td>${pct(m.hmm_length_f1)}</td></tr>
      <tr><td>Route mismatch ↓</td><td>${num(m.classifier_route_mismatch)}</td><td>${num(m.hmm_route_mismatch)}</td></tr>
      <tr><td>Route continuity</td><td>${pct(m.classifier_route_continuity)}</td><td>${pct(m.hmm_route_continuity)}</td></tr>
      </table>
      <table style="margin-top:10px">
      <tr><td>Scored points</td><td>${m.n_scored_points.toLocaleString()} / ${res.n_points.toLocaleString()}</td></tr>
      <tr><td>Candidate miss rate</td><td>${pct(m.candidate_miss_rate)}</td></tr>`;
  } else {
    html += `<p class="muted">No reference route on the server for this trip; showing prediction only.</p><table>`;
  }
  html += `<tr><td>HMM chain breaks</td><td>${res.hmm_breaks}</td></tr>
    <tr><td>Points without candidates</td><td>${res.n_points_without_candidates}</td></tr>
    <tr><td>Features / classifier / HMM</td><td>${num(t.features_s, 2)} / ${num(t.classifier_s, 2)} / ${num(t.hmm_s, 2)} s</td></tr>
    <tr><td>Total request time</td><td>${num(t.request_total, 2)} s</td></tr>
    <tr><td>Model</td><td>${res.model || "–"}</td></tr></table>`;
  $("metrics").innerHTML = html;
}

$("connect").onclick = connect;
$("api-url").addEventListener("keydown", (e) => { if (e.key === "Enter") connect(); });
connect();
