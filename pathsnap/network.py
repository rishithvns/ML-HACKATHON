"""Directed road network: geometry, spatial index, and bounded shortest paths."""
from __future__ import annotations

import heapq

import numpy as np
from scipy.spatial import cKDTree

from .geo import LocalProjection, bearing, point_segment


class RoadNetwork:
    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray, proj: LocalProjection):
        self.proj = proj
        self.nodes_lonlat = nodes_lonlat
        self.nx, self.ny = proj.forward(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
        self.arcs = arcs
        self.src = arcs[:, 0]
        self.dst = arcs[:, 1]
        self.ax, self.ay = self.nx[self.src], self.ny[self.src]
        self.bx, self.by = self.nx[self.dst], self.ny[self.dst]
        self.length = np.hypot(self.bx - self.ax, self.by - self.ay)
        self.bearing = bearing(self.bx - self.ax, self.by - self.ay)
        n_nodes = len(nodes_lonlat)
        self.n_nodes, self.n_arcs = n_nodes, len(arcs)
        self.out_degree = np.bincount(self.src, minlength=n_nodes)
        self.in_degree = np.bincount(self.dst, minlength=n_nodes)
        order = np.argsort(self.src, kind="stable")
        self._out_arcs = order
        self._out_ptr = np.r_[0, np.cumsum(self.out_degree)]
        # twin = the arc running the opposite way between the same two nodes
        key = {(int(a), int(b)): i for i, (a, b) in enumerate(arcs)}
        self.twin = np.array([key.get((int(b), int(a)), -1) for a, b in arcs], dtype=np.int64)
        self._tree = None
        self._sp_cache: dict[int, tuple[float, dict, dict]] = {}

    # ------------------------------------------------------------------ spatial
    def build_index(self, px: np.ndarray, py: np.ndarray, radius: float, step: float):
        """Index only arcs whose bounding box comes within `radius` of the track's bbox,
        densified every `step` metres so a KD-tree ball query finds every arc within
        `radius` of a point (a sample lies within step/2 of the closest point)."""
        m = radius + step
        x0, x1, y0, y1 = px.min() - m, px.max() + m, py.min() - m, py.max() + m
        near = ((np.maximum(self.ax, self.bx) >= x0) & (np.minimum(self.ax, self.bx) <= x1)
                & (np.maximum(self.ay, self.by) >= y0) & (np.minimum(self.ay, self.by) <= y1))
        ids = np.flatnonzero(near)
        nseg = np.maximum(1, np.ceil(self.length[ids] / step)).astype(int)
        rep = np.repeat(ids, nseg + 1)
        within = np.arange(len(rep)) - np.repeat(np.r_[0, np.cumsum(nseg + 1)[:-1]], nseg + 1)
        frac = within / np.repeat(nseg, nseg + 1)
        sx = self.ax[rep] + frac * (self.bx[rep] - self.ax[rep])
        sy = self.ay[rep] + frac * (self.by[rep] - self.ay[rep])
        self._tree = cKDTree(np.c_[sx, sy])
        self._sample_arc = rep
        self._index_step = step
        self.indexed_arcs = ids

    def candidates(self, px: np.ndarray, py: np.ndarray, radius: float, k: int) -> dict:
        """Up to k nearest arcs within `radius` metres of each point.
        Returns flat arrays sorted by (point, distance)."""
        if self._tree is None:
            self.build_index(px, py, radius, 10.0)
        hits = self._tree.query_ball_point(np.c_[px, py], r=radius + self._index_step / 2 + 1.0,
                                           workers=-1)
        lens = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
        if lens.sum() == 0:
            e = np.empty(0, dtype=np.int64)
            return {"pt": e, "arc": e, "dist": e.astype(float), "t": e.astype(float), "rank": e}
        pt = np.repeat(np.arange(len(px)), lens)
        arc = self._sample_arc[np.concatenate([np.asarray(h, dtype=np.int64) for h in hits if len(h)])]
        key = np.unique(pt * self.n_arcs + arc)
        pt, arc = key // self.n_arcs, key % self.n_arcs
        dist, t = point_segment(px[pt], py[pt], self.ax[arc], self.ay[arc], self.bx[arc], self.by[arc])
        keep = dist <= radius
        pt, arc, dist, t = pt[keep], arc[keep], dist[keep], t[keep]
        order = np.lexsort((arc, dist, pt))
        pt, arc, dist, t = pt[order], arc[order], dist[order], t[order]
        starts = np.r_[0, np.flatnonzero(np.diff(pt)) + 1]
        rank = np.arange(len(pt)) - np.repeat(starts, np.diff(np.r_[starts, len(pt)]))
        keep = rank < k
        return {"pt": pt[keep], "arc": arc[keep], "dist": dist[keep], "t": t[keep], "rank": rank[keep]}

    # ----------------------------------------------------------- shortest paths
    def out_arcs(self, node: int) -> np.ndarray:
        return self._out_arcs[self._out_ptr[node]:self._out_ptr[node + 1]]

    def shortest_from(self, node: int, cutoff: float) -> tuple[dict, dict]:
        """Dijkstra from `node` over arc lengths, bounded by `cutoff` metres (cached)."""
        cached = self._sp_cache.get(node)
        if cached is not None and cached[0] >= cutoff:
            return cached[1], cached[2]
        dist = {node: 0.0}
        pred = {node: -1}
        heap = [(0.0, node)]
        length, dst, ptr, out = self.length, self.dst, self._out_ptr, self._out_arcs
        while heap:
            d, v = heapq.heappop(heap)
            if d > dist.get(v, np.inf):
                continue
            for a in out[ptr[v]:ptr[v + 1]]:
                w = int(dst[a])
                nd = d + length[a]
                if nd <= cutoff and nd < dist.get(w, np.inf):
                    dist[w] = nd
                    pred[w] = int(a)
                    heapq.heappush(heap, (nd, w))
        self._sp_cache[node] = (cutoff, dist, pred)
        return dist, pred

    def path_arcs(self, u: int, v: int, cutoff: float) -> list[int] | None:
        """Arc ids of the shortest path from node u to node v, or None if beyond cutoff."""
        if u == v:
            return []
        dist, pred = self.shortest_from(u, cutoff)
        if v not in dist:
            return None
        path = []
        while v != u:
            a = pred[v]
            path.append(a)
            v = int(self.src[a])
        return path[::-1]

    def connected(self, a: int, b: int) -> bool:
        return a == b or int(self.dst[a]) == int(self.src[b])

    def arc_lonlat(self, arc_ids) -> list[list[tuple[float, float]]]:
        arc_ids = np.asarray(arc_ids, dtype=np.int64)
        s, d = self.nodes_lonlat[self.src[arc_ids]], self.nodes_lonlat[self.dst[arc_ids]]
        return [[(float(a[1]), float(a[0])), (float(b[1]), float(b[0]))] for a, b in zip(s, d)]

    def route_lonlat(self, arc_ids) -> list[tuple[float, float]]:
        """Polyline (lat, lon) through a sequence of arcs (breaks are bridged straight)."""
        pts: list[tuple[float, float]] = []
        for a in arc_ids:
            s, d = self.nodes_lonlat[self.src[a]], self.nodes_lonlat[self.dst[a]]
            if not pts or pts[-1] != (s[1], s[0]):
                pts.append((float(s[1]), float(s[0])))
            pts.append((float(d[1]), float(d[0])))
        return pts
