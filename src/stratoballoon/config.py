"""Typed configuration loaded from YAML.

Each experiment is driven by one YAML file under configs/. Loading it into
dataclasses means a typo in a key fails loudly instead of silently falling back
to a default, and the resolved config can be written next to every result.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataConfig:
    """What to download and how to lay it out on disk."""
    name: str = "era5"
    dataset: str = "reanalysis-era5-pressure-levels"
    variables: list[str] = field(default_factory=lambda: [
        "u_component_of_wind", "v_component_of_wind", "temperature", "geopotential"])
    levels_hpa: list[int] = field(default_factory=lambda: [100, 70, 50, 30, 20])
    step_hours: int = 6
    start: str = "2015-01"          # first month, inclusive
    end: str = "2025-12"            # last month, inclusive
    area: list[float] = field(default_factory=lambda: [40, 40, 0, 130])  # N, W, S, E
    grid_deg: float = 1.0
    raw_dir: str = "data/era5"
    file_prefix: str = "era5"

    @property
    def raw_path(self) -> Path:
        return ROOT / self.raw_dir

    @property
    def hours(self) -> list[str]:
        return [f"{h:02d}:00" for h in range(0, 24, self.step_hours)]


def _build(cls, raw: dict[str, Any] | None):
    """Construct a dataclass from a dict, rejecting unknown keys."""
    raw = raw or {}
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - names
    if unknown:
        raise KeyError(f"{cls.__name__}: unknown config keys {sorted(unknown)}")
    return cls(**raw)


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def load_data_config(path: str | Path = "configs/data.yaml") -> DataConfig:
    return _build(DataConfig, load_yaml(path).get("data"))


def to_json(obj) -> str:
    """Serialise a config dataclass (or a dict of them) for run metadata."""
    def conv(o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))
    return json.dumps(obj, default=conv, indent=1)
