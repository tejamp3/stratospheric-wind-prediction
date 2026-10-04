"""Build and execute notebooks/balloon_autonomy_results.ipynb.

The notebook only presents results; every computation lives in the package and
the experiment scripts. Rebuilding it from this script keeps it in step with the
result files, and executing it stores the outputs so they render on GitHub.

Usage:  python experiments/build_results_notebook.py --results results
"""
from __future__ import annotations

import argparse

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from stratoballoon.config import ROOT

SECTIONS = [
    ("radiosondes", "## How good is the truth?\n\nERA5 against radiosondes at eight Indian "
                    "stations (**REAL DATA**).", "era5_vs_radiosondes.png", "summary.csv"),
    ("uncertainty", "## Are the forecast's uncertainty regions honest?\n\nConformal regions "
                    "calibrated on the previous year, checked on the test year "
                    "(**MODEL PREDICTION**).", "reliability.png", None),
    ("trajectory", "## Where will a drifting balloon be?\n\nPredicted against actual "
                   "positions (**MODEL PREDICTION**).", "position_error.png", "position_error.csv"),
    ("trajectory", "One launch, with the ensemble cone and the wind it flew through.",
     "example_cone.png", None),
    ("control", "## Which controller keeps the balloon on station?\n\nIdentical missions for "
                "every controller (**SIMULATION**).", "controller_comparison.png", "summary.csv"),
    ("control", "One mission in detail: track, altitude commands, battery.",
     "example_mission.png", None),
    ("feasibility", "## When is station keeping possible at all?\n\nWith a perfect forecast, "
                    "the upper bound (**SIMULATION**).", "feasibility.png", "twr50_by_month.csv"),
    ("robustness", "## What breaks it?\n\nInjected faults (**SIMULATION**).", "robustness.png",
     "summary.csv"),
    ("planner", "## Planning a mission from Vijayawada\n\n(**SIMULATION** over forecast "
                "scenarios).", "plan.png", None),
]


def build(results: str) -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells.append(new_markdown_cell(
        "# Balloon autonomy: results\n\n"
        "This notebook only shows results. Every number comes from the files the experiment "
        "scripts write under `" + results + "`; the code that produced them is in "
        "`src/stratoballoon/` and `experiments/`. Labels: **REAL DATA** (ERA5, radiosondes), "
        "**MODEL PREDICTION** (forecasts scored against ERA5), **SIMULATION** (a simplified "
        "research model of the balloon, not flight performance)."))
    nb.cells.append(new_code_cell(
        "from pathlib import Path\nimport pandas as pd\nfrom IPython.display import Image, display\n"
        f"R = Path('..') / '{results}'\npd.set_option('display.width', 160)\n"
        "def show(sub, png, csv=None):\n"
        "    p = R / sub / png\n"
        "    if p.exists():\n        display(Image(str(p)))\n"
        "    else:\n        print('not generated yet:', p)\n"
        "    if csv and (R / sub / csv).exists():\n"
        "        display(pd.read_csv(R / sub / csv).round(3))"))
    for sub, md, png, csv in SECTIONS:
        nb.cells.append(new_markdown_cell(md))
        nb.cells.append(new_code_cell(f"show('{sub}', '{png}'" + (f", '{csv}')" if csv else ")")))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    return nb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    nb = build(a.results)
    path = ROOT / "notebooks" / "balloon_autonomy_results.ipynb"
    NotebookClient(nb, timeout=300, kernel_name="python3",
                   resources={"metadata": {"path": str(path.parent)}}).execute()
    nbformat.write(nb, path)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
