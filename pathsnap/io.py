"""Raw-file readers.

File formats (headerless, whitespace separated):
  <id>.track  lon lat time_s       GPS observations (row = observation)
  <id>.nodes  lon lat              road nodes      (row index = node_id)
  <id>.arcs   from_node to_node    directed arcs   (row index = arc_id)
  <id>.route  arc_id               reference route (chronological arc ids)

`load_trip` deliberately never reads the .route file: the prediction pipeline
must not see reference labels. Only `load_route` (used by labelling and
server-side evaluation) touches it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Trip:
    trip_id: str
    lon: np.ndarray
    lat: np.ndarray
    t: np.ndarray
    obs_id: np.ndarray            # row index in the original .track file
    nodes: np.ndarray             # (n_nodes, 2) lon, lat
    arcs: np.ndarray              # (n_arcs, 2) from, to
    cleaning: dict = field(default_factory=dict)


def list_trips(part_dir: str | Path) -> list[str]:
    part_dir = Path(part_dir)
    if not part_dir.exists():
        return []
    return sorted(p.name for p in part_dir.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


def _read(path: Path, dtype=float) -> np.ndarray:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="c")
    return df.to_numpy(dtype=dtype)


def load_trip(trip_dir: str | Path) -> Trip:
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    raw = _read(trip_dir / f"{tid}.track")
    nodes = _read(trip_dir / f"{tid}.nodes")
    arcs = _read(trip_dir / f"{tid}.arcs", dtype=np.int64)

    n_raw = len(raw)
    obs_id = np.arange(n_raw)
    if raw.shape[1] < 3:  # no timestamps: assume 1 Hz
        raw = np.c_[raw[:, :2], np.arange(n_raw, dtype=float)]
    ok = np.isfinite(raw[:, :3]).all(1) & (np.abs(raw[:, 0]) <= 180) & (np.abs(raw[:, 1]) <= 90)
    raw, obs_id = raw[ok], obs_id[ok]
    order = np.argsort(raw[:, 2], kind="stable")
    raw, obs_id = raw[order], obs_id[order]
    keep = np.r_[True, np.diff(raw[:, 2]) > 0]     # drop duplicate timestamps
    raw, obs_id = raw[keep], obs_id[keep]

    valid_arc = (arcs >= 0).all(1) & (arcs < len(nodes)).all(1)
    cleaning = {
        "raw_rows": int(n_raw),
        "invalid_rows": int((~ok).sum()),
        "was_unsorted": bool((order != np.arange(len(order))).any()),
        "duplicate_timestamps": int((~keep).sum()),
        "kept_rows": int(len(raw)),
        "invalid_arcs": int((~valid_arc).sum()),
    }
    return Trip(tid, raw[:, 0].copy(), raw[:, 1].copy(), raw[:, 2].copy(), obs_id,
                nodes, arcs, cleaning)


def load_route(trip_dir: str | Path) -> np.ndarray:
    trip_dir = Path(trip_dir)
    return _read(trip_dir / f"{trip_dir.name}.route", dtype=np.int64).ravel()
