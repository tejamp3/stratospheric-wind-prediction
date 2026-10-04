"""Run the whole pipeline in order, resumably.

Each step is skipped if its main output already exists (pass --force to redo),
so an interrupted run picks up where it stopped. Steps run as separate
processes, so one failure is reported and the rest still run unless a later
step depends on it. At the end the README, the project summary and the results
notebook are regenerated from the result files.

Usage:
    python experiments/run_all.py
    python experiments/run_all.py --only control feasibility
    python experiments/run_all.py --only control --force
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from stratoballoon.config import ROOT, load_yaml

FORECAST, EXPERIMENT, MISSION = ("configs/forecast.yaml", "configs/experiment.yaml",
                                 "configs/mission_example.yaml")


def steps():
    f_out = ROOT / load_yaml(FORECAST)["experiment"]["out_dir"]
    e_out = ROOT / load_yaml(EXPERIMENT)["experiment"]["out_dir"]
    py, fc, ex = "experiments/", FORECAST, EXPERIMENT
    return [
        ("ladder", [py + "forecast_ladder.py", "--config", fc], f_out / "skill_ci.csv", []),
        ("ablation", [py + "forecast_ablation.py", "--config", fc],
         f_out / "ablation" / "importance.csv", ["ladder"]),
        ("uncertainty", [py + "uncertainty_eval.py", "--config", ex],
         e_out / "uncertainty" / "coverage.csv", ["ladder"]),
        ("trajectory", [py + "trajectory_eval.py", "--config", ex],
         e_out / "trajectory" / "position_error.csv", ["ladder"]),
        ("rl", [py + "train_rl.py", "--config", ex, "--steps", "500000"],
         e_out / "rl" / "ppo_policy.zip", ["ladder"]),
        ("control", [py + "control_montecarlo.py", "--config", ex],
         e_out / "control" / "summary.csv", ["ladder", "rl"]),
        ("feasibility", [py + "feasibility_map.py", "--config", ex],
         e_out / "feasibility" / "twr50_by_month.csv", []),
        ("robustness", [py + "robustness.py", "--config", ex],
         e_out / "robustness" / "summary.csv", ["ladder"]),
        ("planner", [py + "plan_mission.py", "--config", ex, "--mission", MISSION],
         e_out / "planner" / "plan.json", ["ladder"]),
        ("radiosondes", [py + "era5_vs_radiosondes.py", "--config", ex],
         e_out / "radiosondes" / "summary.csv", ["ladder"]),
        ("vista", [py + "vista_case_study.py"], e_out / "vista" / "summary.json", []),
        ("anomaly", [py + "anomaly_eval.py"], e_out / "anomaly" / "summary.csv", []),
        ("edge", [py + "edge_profile.py", "--config", ex], e_out / "edge" / "edge_profile.csv",
         ["ladder"]),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    failed = set()
    for name, cmd, output, needs in steps():
        if a.only and name not in a.only:
            continue
        if failed & set(needs):
            print(f"[skip] {name}: depends on failed {sorted(failed & set(needs))}")
            failed.add(name)
            continue
        if output.exists() and not a.force:
            print(f"[done] {name}: {output.relative_to(ROOT)} exists")
            continue
        print(f"[run ] {name}: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, *cmd], cwd=ROOT)
        print(f"[{'ok' if r.returncode == 0 else 'FAIL'}] {name} in {(time.time() - t0) / 60:.1f} min",
              flush=True)
        if r.returncode:
            failed.add(name)
    for target in ("README.md", "docs/PROJECT_SUMMARY.md"):
        subprocess.run([sys.executable, "experiments/report.py", "--config", EXPERIMENT,
                        "--target", target], cwd=ROOT)
    subprocess.run([sys.executable, "experiments/build_results_notebook.py", "--results",
                    load_yaml(EXPERIMENT)["experiment"]["out_dir"]], cwd=ROOT)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
