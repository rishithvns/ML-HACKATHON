"""Configuration loading. All paths are resolved relative to the project root."""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.json"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    """Absolute path for an entry of cfg['paths']."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p


def merged(cfg: dict, section: str, overrides: dict | None) -> dict:
    out = copy.deepcopy(cfg[section])
    out.update(overrides or {})
    return out
