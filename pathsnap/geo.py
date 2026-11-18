"""Geometry helpers.

Coordinates are projected with a local equirectangular projection centred on
each trip. Trips span at most a few tens of km, where this projection's
distance error is far below GPS noise (<0.1 %), and it avoids a pyproj
dependency. Raw lon/lat degrees are never used as distances.
"""
from __future__ import annotations

import numpy as np

EARTH_R = 6_371_008.8  # mean Earth radius, metres


class LocalProjection:
    def __init__(self, lon0: float, lat0: float):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self.kx = np.radians(1.0) * EARTH_R * np.cos(np.radians(lat0))
        self.ky = np.radians(1.0) * EARTH_R

    @classmethod
    def around(cls, lon: np.ndarray, lat: np.ndarray) -> "LocalProjection":
        return cls(float(np.mean(lon)), float(np.mean(lat)))

    def forward(self, lon, lat):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky

    def inverse(self, x, y):
        return np.asarray(x) / self.kx + self.lon0, np.asarray(y) / self.ky + self.lat0


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def point_segment(px, py, ax, ay, bx, by):
    """Distance from points to segments (broadcasting) and projection fraction t in [0, 1]."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = np.where(l2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(l2 > 0, l2, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    qx, qy = ax + t * dx, ay + t * dy
    return np.hypot(px - qx, py - qy), t


def bearing(dx, dy):
    """Compass bearing in degrees (0 = north, 90 = east) of a displacement."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def angle_diff(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b)) % 360.0)
    return np.minimum(d, 360.0 - d)
