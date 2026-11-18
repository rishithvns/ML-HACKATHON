"""HMM + Viterbi route reconstruction.

Hidden state at observation k: one candidate arc of that observation, together
with the GPS point's projected position s along the arc (metres from its tail).

Emission:   log P(z_k | arc) = w * log(max(p_classifier, min_prob))
Transition: log P(arc_j | arc_i) = -|d_route - d_gps| / beta_eff
            (Newson & Krumm 2009): d_route is the network distance from the
            previous projected position to the current one, d_gps the straight
            line distance between the two fixes. On the same arc, d_route is the
            forward offset (backward jitter up to `back_tolerance_m` allowed).
            Otherwise d_route = rest of arc i + shortest path(head_i -> tail_j)
            + offset on arc j, with a bounded Dijkstra (cutoff grows with d_gps).
            Unreachable pairs are invalid (-inf). Turning onto the twin arc
            (U-turn) costs an extra `uturn_penalty`.
Gaps:       beta_eff = beta * sqrt(max(1, dt)); if dt > max_gap_s, or no
            candidate is reachable from any previous state, the chain is broken
            and a new segment starts (counted as a route break).
Missing candidates: observations with no arc within the search radius have
            no state; they are skipped and receive no prediction (-1).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .metrics import dedupe
from .network import RoadNetwork

NEG = -1e18


def _route_dist_matrix(net: RoadNetwork, arcs_i, s_i, arcs_j, s_j, cutoff, back_tol):
    R = np.full((len(arcs_i), len(arcs_j)), np.inf)
    for ia, (a, sa) in enumerate(zip(arcs_i, s_i)):
        rest = net.length[a] - sa
        dist = None
        for jb, (b, sb) in enumerate(zip(arcs_j, s_j)):
            if a == b:
                diff = sb - sa
                if diff >= -back_tol:
                    R[ia, jb] = max(diff, 0.0)
                continue
            if dist is None:
                dist, _ = net.shortest_from(int(net.dst[a]), cutoff)
            sp = dist.get(int(net.src[b]))
            if sp is not None:
                R[ia, jb] = rest + sp + sb
    return R


def viterbi(cands: pd.DataFrame, points: pd.DataFrame, net: RoadNetwork, params: dict,
            prob_col: str = "prob") -> dict:
    """Decode one trip.

    cands:  the trip's candidate rows (obs, arc, t_frac, prob_col)
    points: the trip's point table (x, y, t), indexed by obs position
    Returns per-observation arcs (-1 = no prediction), segment ids, break count.
    """
    c = cands.sort_values(["obs", prob_col], ascending=[True, False])
    obs_arr = c["obs"].to_numpy()
    n_points = len(points)
    if len(obs_arr) == 0:
        return {"obs_arc": np.full(n_points, -1), "segment": np.full(n_points, -1), "n_breaks": 0}
    starts = np.r_[0, np.flatnonzero(np.diff(obs_arr)) + 1]
    ends = np.r_[starts[1:], len(obs_arr)]
    arcs_all = c["arc"].to_numpy()
    s_all = c["t_frac"].to_numpy() * net.length[arcs_all]
    em_all = params["emission_weight"] * np.log(np.maximum(c[prob_col].to_numpy(), params["min_prob"]))
    x, y, t = points["x"].to_numpy(), points["y"].to_numpy(), points["t"].to_numpy()

    steps = []          # (obs, arcs, back, segment, delta)
    delta = None
    seg = -1
    for st, en in zip(starts, ends):
        k = int(obs_arr[st])
        arcs, s, em = arcs_all[st:en], s_all[st:en], em_all[st:en]
        back = None
        if steps:
            pk, parcs, ps = steps[-1][0], steps[-1][1], prev_s
            d = math.hypot(x[k] - x[pk], y[k] - y[pk])
            dt = t[k] - t[pk]
            if dt <= params["max_gap_s"]:
                cutoff = params["cutoff_base_m"] + params["cutoff_factor"] * d
                R = _route_dist_matrix(net, parcs, ps, arcs, s, cutoff, params["back_tolerance_m"])
                beta_eff = params["beta"] * math.sqrt(max(1.0, dt))
                logtr = np.where(np.isfinite(R), -np.abs(R - d) / beta_eff, NEG)
                logtr = logtr - params["uturn_penalty"] * (net.twin[parcs][:, None] == arcs[None, :])
                score = delta[:, None] + logtr
                best = score.argmax(0)
                bscore = score[best, np.arange(len(arcs))]
                if bscore.max() > NEG / 2:
                    delta = bscore + em
                    back = best
        if back is None:            # start of a new segment
            seg += 1
            delta = em.copy()
            back = np.full(len(arcs), -1)
        steps.append((k, arcs, back, seg, delta))
        prev_s = s

    obs_arc = np.full(n_points, -1, dtype=np.int64)
    segment = np.full(n_points, -1, dtype=np.int64)
    i = len(steps) - 1
    state = int(np.argmax(steps[i][4]))
    while i >= 0:
        k, arcs, back, sg, dl = steps[i]
        obs_arc[k] = arcs[state]
        segment[k] = sg
        if back[state] < 0:         # segment start: pick best final state of previous segment
            i -= 1
            if i >= 0:
                state = int(np.argmax(steps[i][4]))
        else:
            state = int(back[state])
            i -= 1
    return {"obs_arc": obs_arc, "segment": segment, "n_breaks": seg}


def reconstruct_route(obs_arc: np.ndarray, segment: np.ndarray, net: RoadNetwork,
                      fill_cutoff_m: float = 2000.0) -> dict:
    """Turn per-observation arcs into a continuous arc sequence by inserting the
    shortest connecting path between consecutive distinct arcs."""
    seq = dedupe(obs_arc)
    route: list[int] = []
    gaps = 0
    for a in seq:
        if not route:
            route.append(a)
            continue
        prev = route[-1]
        if int(net.dst[prev]) != int(net.src[a]):
            path = net.path_arcs(int(net.dst[prev]), int(net.src[a]), fill_cutoff_m)
            if path is None:
                gaps += 1
            else:
                route.extend(path)
        route.append(a)
    return {"route": route, "unfilled_gaps": gaps}
