"""Folium map builder shared by EDA and result maps."""
from __future__ import annotations

import folium
import numpy as np

from .network import RoadNetwork

COLORS = {
    "gps": "#1f77b4",
    "network": "#9e9e9e",
    "reference": "#2ca02c",
    "classifier": "#ff7f0e",
    "hmm": "#d62728",
}


def _legend(items: list[tuple[str, str]]) -> folium.Element:
    rows = "".join(
        f'<div><span style="display:inline-block;width:14px;height:4px;background:{c};'
        f'margin-right:6px;vertical-align:middle"></span>{name}</div>' for name, c in items)
    return folium.Element(
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;'
        'padding:8px 10px;border:1px solid #bbb;border-radius:4px;font:12px sans-serif">'
        f'<b>Legend</b>{rows}</div>')


def trip_map(lon: np.ndarray, lat: np.ndarray, net: RoadNetwork, network_arcs=None,
             routes: dict | None = None, title: str = "", max_gps_points: int = 4000) -> folium.Map:
    """routes: {layer name: (arc id sequence, colour key or hex, dashed?)}"""
    m = folium.Map(location=[float(np.mean(lat)), float(np.mean(lon))], zoom_start=14,
                   tiles="OpenStreetMap", control_scale=True)
    legend = []
    if network_arcs is not None and len(network_arcs):
        fg = folium.FeatureGroup(name="Road network (arcs near track)", show=True)
        for seg in net.arc_lonlat(np.unique(network_arcs)):
            folium.PolyLine(seg, color=COLORS["network"], weight=2, opacity=0.6).add_to(fg)
        fg.add_to(m)
        legend.append(("Road network", COLORS["network"]))
    for name, (arcs, color, dashed) in (routes or {}).items():
        if arcs is None or len(arcs) == 0:
            continue
        col = COLORS.get(color, color)
        fg = folium.FeatureGroup(name=name, show=True)
        folium.PolyLine(net.route_lonlat(arcs), color=col, weight=5 if not dashed else 4,
                        opacity=0.85, dash_array="8 6" if dashed else None, tooltip=name).add_to(fg)
        fg.add_to(m)
        legend.append((name, col))
    step = max(1, int(np.ceil(len(lon) / max_gps_points)))
    fg = folium.FeatureGroup(name=f"GPS observations (every {step})" if step > 1 else "GPS observations")
    for x, y in zip(lon[::step], lat[::step]):
        folium.CircleMarker([float(y), float(x)], radius=2, color=COLORS["gps"], fill=True,
                            fill_opacity=0.8, weight=0).add_to(fg)
    fg.add_to(m)
    legend.append(("GPS observations", COLORS["gps"]))
    folium.Marker([float(lat[0]), float(lon[0])], tooltip="start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([float(lat[-1]), float(lon[-1])], tooltip="end",
                  icon=folium.Icon(color="black", icon="stop")).add_to(m)
    m.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]])
    folium.LayerControl(collapsed=False).add_to(m)
    if title:
        m.get_root().html.add_child(folium.Element(
            f'<div style="position:fixed;top:10px;left:60px;z-index:9999;background:white;padding:4px 8px;'
            f'border:1px solid #bbb;border-radius:4px;font:14px sans-serif"><b>{title}</b></div>'))
    m.get_root().html.add_child(_legend(legend))
    return m
