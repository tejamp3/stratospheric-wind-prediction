"""Prove the EDA notebook still runs.

A notebook rots silently: it is committed with stale outputs and nobody notices
the code stopped working until someone opens it. This executes every code cell
against the real downloaded data and fails if any of them raises.

Display calls are stripped rather than stubbed, because faking IPython well
enough for matplotlib's backend detection is more trouble than it is worth and
the thing under test is the analysis code, not the rendering.

Skipped when no ERA5 files are present, so a fresh clone can still run the suite.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "01_exploratory_analysis.ipynb"
sys.path.insert(0, str(ROOT / "src"))


def _has_data() -> bool:
    return any((ROOT / "data" / "raw").glob("era5_strat_*.nc"))


def test_notebook_is_valid_json():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert any(c["cell_type"] == "code" for c in nb["cells"])


@pytest.mark.skipif(not _has_data(),
                    reason="needs downloaded ERA5 data; run src/download_era5.py")
def test_notebook_cells_execute(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")

    # The notebook resolves "../src" and writes figures relative to the project,
    # so it has to run from its own directory.
    monkeypatch.chdir(NOTEBOOK.parent)

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    g = {"__name__": "__main__",
         "display": lambda *a, **k: None,
         "Image": lambda *a, **k: None}

    for i, cell in enumerate(cells, 1):
        lines = "".join(cell["source"]).splitlines()
        kept = [ln for ln in lines if not re.match(r"\s*(from|import) IPython", ln)]
        exec(compile("\n".join(kept), f"<notebook cell {i}>", "exec"), g)
