"""Folium result maps: GPS, nearby road network, reference route, classifier-only route,
and HMM/Viterbi route, with layer control and legend.

Run:  python -m scripts.make_maps [--part student_test] [--trips 00000052 00000009]
Outputs -> reports/maps/<part>_<trip>.html
"""
from __future__ import annotations

import argparse

from pathsnap.config import load_config, resolve
from pathsnap.evaluation import score_prediction
from pathsnap.io import list_trips, load_route
from pathsnap.pipeline import PathSnapPredictor
from pathsnap.viz import trip_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="student_test", choices=["train", "student_test"])
    ap.add_argument("--trips", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config()
    out = resolve(cfg, "reports") / "maps"
    out.mkdir(parents=True, exist_ok=True)
    pred = PathSnapPredictor.load(resolve(cfg, "models"))
    pdir = resolve(cfg, args.part)
    for tid in args.trips or list_trips(pdir):
        res, ctx = pred.predict(pdir / tid)
        s = score_prediction(pdir / tid, res, ctx, pred.cfg)
        title = (f"{args.part}/{tid} &mdash; Top-1 classifier {s['classifier_top1_accuracy']:.3f} &rarr; "
                 f"HMM {s['hmm_top1_accuracy']:.3f}, route arc F1 {s['hmm_arc_f1']:.3f}")
        m = trip_map(ctx.trip.lon, ctx.trip.lat, ctx.net, network_arcs=ctx.cands["arc"].to_numpy(), title=title,
                     routes={"Reference route": (load_route(pdir / tid), "reference", False),
                             "Classifier-only route": (res["classifier_route"], "classifier", True),
                             "Predicted route (HMM/Viterbi)": (res["hmm_route"], "hmm", False)})
        m.save(str(out / f"{args.part}_{tid}.html"))
        print(f"{tid}: {title}")


if __name__ == "__main__":
    main()
