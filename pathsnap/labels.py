"""Per-observation reference labels derived from the reference route.

The .route file lists arcs in travel order but does not say which GPS point
lies on which arc. We align observations to route positions with a monotone
dynamic programme: observation k is assigned route position j_k with
j_0 <= j_1 <= ... (time order is preserved, arcs may be skipped when the
vehicle passes a short arc between two fixes). The cost of assigning k to j is
the point-to-segment distance, plus a penalty when the GPS heading opposes the
arc direction (this separates the two directions of a two-way street and
out-and-back routes).

Observations whose aligned arc is further than `max_label_dist_m` are flagged
label_valid = False: either the GPS or the reference route is wrong there
(e.g. trips 00000033 / 00000054 show a constant ~130 m offset), so they are
excluded from training and from point-level scoring.
"""
from __future__ import annotations

import numpy as np

from .geo import angle_diff, point_segment
from .network import RoadNetwork


def align_to_route(px, py, heading, head_valid, net: RoadNetwork, route: np.ndarray,
                   heading_penalty: float = 15.0, chunk: int = 2048):
    n, m = len(px), len(route)
    ax, ay, bx, by = net.ax[route], net.ay[route], net.bx[route], net.by[route]
    brg = net.bearing[route]
    cost = np.empty((n, m), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        d, _ = point_segment(px[s:e, None], py[s:e, None], ax, ay, bx, by)
        opp = (angle_diff(heading[s:e, None], brg) > 90) & head_valid[s:e, None]
        cost[s:e] = d + heading_penalty * opp
    back = np.empty((n, m), dtype=np.int32)
    idx = np.arange(m)
    acc = cost[0].astype(np.float64)
    for k in range(1, n):
        pm = np.minimum.accumulate(acc)
        arg = np.maximum.accumulate(np.where(acc == pm, idx, 0))
        back[k] = arg
        acc = cost[k] + pm
    j = np.empty(n, dtype=np.int64)
    j[-1] = int(np.argmin(acc))
    for k in range(n - 1, 0, -1):
        j[k - 1] = back[k, j[k]]
    arc = route[j]
    dist, frac = point_segment(px, py, net.ax[arc], net.ay[arc], net.bx[arc], net.by[arc])
    return {"label_arc": arc, "label_route_pos": j, "label_dist": dist, "label_t": frac}
