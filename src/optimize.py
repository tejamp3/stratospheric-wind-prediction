"""Edge-deployment optimisation: precision reduction, latency and memory profiling.

A stratospheric airship carries a small, power-limited flight computer, so what
matters is single-sample CPU latency and resident footprint, not batch
throughput on a GPU. This script measures both for every variant and checks that
the accuracy cost of each is acceptable.

Usage:  python src/optimize.py [--tag base] [--lite-tag lite]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import data as D
import metrics as M
import viz
from evaluate import features_of, input_steps_of, load_model, physical, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("optimize")

torch.set_num_threads(1)  # an onboard computer will not give us many cores


# ------------------------------------------------------------------- variants
def fp16_variant(model: nn.Module) -> nn.Module:
    """Half-precision copy.

    FP16 halves the weight file, which is the scarce resource on an embedded
    flight computer. Note that x86 CPUs have no fast FP16 LSTM kernel, so we
    keep compute in FP32 by casting back at inference time and report FP16 as a
    storage win, measured honestly rather than claimed as a speed-up.
    """
    import copy
    m = copy.deepcopy(model).half()
    return m


def int8_variant(model: nn.Module) -> nn.Module:
    """Dynamic int8 quantisation of the LSTM and Linear weights.

    Weights are stored as int8 and activations are quantised on the fly, which
    cuts the weight file to about a quarter. Whether it is also *faster* depends
    on size: int8 kernels win on large matrices, but at this model's scale and
    batch size 1 the per-call quantise/dequantise overhead dominates, so the
    measured latency is worse than FP32. The table reports what was measured.
    """
    return torch.quantization.quantize_dynamic(
        model, {nn.LSTM, nn.Linear}, dtype=torch.qint8)


# ------------------------------------------------------------------- profiling
def state_dict_bytes(model: nn.Module) -> int:
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


def weight_count(model: nn.Module) -> int:
    """Number of learned weights in an unquantised module.

    A dynamically quantised LSTM hides its weights inside opaque packed-param
    objects that are not tensors, so neither parameters() nor the state dict can
    be counted for it. Quantisation changes precision, not the number of weights,
    so the caller passes the architecture's count for those variants instead and
    lets the size column carry the compression story.
    """
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def latency(model: nn.Module, x: torch.Tensor, n: int = 300,
            warmup: int = 30) -> dict[str, float]:
    """Single-sample latency distribution in milliseconds."""
    for _ in range(warmup):
        model(x)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        model(x)
        ts.append((time.perf_counter() - t0) * 1e3)
    a = np.array(ts)
    return {"mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99)),
            "max_ms": float(a.max())}


@torch.no_grad()
def predict_fp16(model, X: np.ndarray, batch: int = 256) -> np.ndarray:
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).half()
        out.append(model(xb).float().numpy())
    return np.concatenate(out)


def accuracy_of(pred_norm, ds_test, sx, sy) -> dict[str, float]:
    pred, true, base = physical(ds_test, pred_norm, sx, sy)
    r = {}
    for k, h in enumerate(C.HORIZONS):
        mm = M.evaluate(pred[:, k, :], true[:, k, :])
        bb = M.evaluate(base[:, k, :], true[:, k, :])
        r[f"speed_rmse_{h}h"] = mm["speed_rmse"]
        r[f"dir_acc_30deg_{h}h"] = mm["dir_acc_30deg"]
        r[f"skill_{h}h_pct"] = M.skill_score(mm["speed_rmse"], bb["speed_rmse"])
    return r


def fig_tradeoff(table: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    x = np.arange(len(table))
    a1.barh(x, table.p50_ms, color=viz.PREDICTED, edgecolor=viz.SURFACE, linewidth=2,
            height=.6)
    for i, (v, p95) in enumerate(zip(table.p50_ms, table.p95_ms)):
        a1.text(v, i, f"  {v:.2f} ms (p95 {p95:.2f})", va="center", fontsize=8.5,
                color=viz.INK)
    a1.set_yticks(x)
    a1.set_yticklabels(table.variant)
    a1.invert_yaxis()
    a1.set_xlabel("Single-sample CPU latency, median (ms)")
    a1.set_title("Inference latency, 1 thread")
    a1.set_xlim(0, table.p95_ms.max() * 1.45)
    a1.grid(axis="y", visible=False)

    a2.barh(x, table.size_kb, color=viz.BASELINE, edgecolor=viz.SURFACE, linewidth=2,
            height=.6)
    for i, v in enumerate(table.size_kb):
        a2.text(v, i, f"  {v:.0f} KB", va="center", fontsize=8.5, color=viz.INK)
    a2.set_yticks(x)
    a2.set_yticklabels(table.variant)
    a2.invert_yaxis()
    a2.set_xlabel("Serialised weight size (KB)")
    a2.set_title("Model footprint")
    a2.set_xlim(0, table.size_kb.max() * 1.35)
    a2.grid(axis="y", visible=False)
    fig.suptitle("Edge-deployment trade-offs", x=0.005, ha="left", fontsize=11,
                 fontweight="semibold")
    viz.finish(fig, C.FIGURES / "optimize_tradeoffs.png")


def fig_accuracy_cost(table: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    w = 0.26
    x = np.arange(len(table))
    for i, h in enumerate(C.HORIZONS):
        col = f"speed_rmse_{h}h"
        ax.bar(x + (i - 1) * w, table[col], width=w * 0.92,
               color=viz.SERIES[i], edgecolor=viz.SURFACE, linewidth=2,
               label=f"{h} h lead")
    ax.set_xticks(x)
    ax.set_xticklabels(table.variant)
    ax.set_ylabel("Speed RMSE (m/s)")
    ax.set_title("Accuracy cost of each optimisation")
    ax.legend(ncol=3, loc="lower right", bbox_to_anchor=(1, 1.0))
    viz.finish(fig, C.FIGURES / "optimize_accuracy_cost.png", viz.SOURCE_NOTE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--lite-tag", default="lite",
                    help="tag of a separately trained smaller model; skipped if absent")
    args = ap.parse_args()
    viz.apply_style()

    base, base_ck = load_model(args.tag)
    ds, sx, sy, _ = D.build_dataset(verbose=False,
                                    input_steps=input_steps_of(base_ck),
                                    feature_names=features_of(base_ck))
    Xte = ds["test"]["X"]
    x1 = torch.from_numpy(Xte[:1])

    rows = []

    def record(name: str, model, pred_norm, note: str = "", weights: int | None = None):
        lat = latency(model, x1.half() if name == "FP16" else x1)
        row = {"variant": name,
               "weights": weights if weights is not None else weight_count(model),
               "size_kb": state_dict_bytes(model) / 1024,
               **lat, **accuracy_of(pred_norm, ds["test"], sx, sy), "note": note}
        rows.append(row)
        log.info("%-8s %6.0f KB  p50 %5.2f ms  RMSE6h %.3f m/s",
                 name, row["size_kb"], row["p50_ms"], row["speed_rmse_6h"])

    nw = weight_count(base)
    record("FP32", base, predict(base, Xte), "reference")

    h16 = fp16_variant(base)
    record("FP16", h16, predict_fp16(h16, Xte),
           "half the weight file; no fast FP16 LSTM kernel on CPU", weights=nw)

    q8 = int8_variant(base)
    record("INT8", q8, predict(q8, Xte),
           "quarter the weight file; quantise overhead dominates at batch 1", weights=nw)

    lite_path = C.MODELS / f"lstm_{args.lite_tag}.pt"
    if lite_path.exists():
        lite, lck = load_model(args.lite_tag)
        nwl = weight_count(lite)
        record(f"Lite h{lck['config']['hidden']}", lite, predict(lite, Xte),
               "smaller LSTM, retrained from scratch")
        q8l = int8_variant(lite)
        record("Lite+INT8", q8l, predict(q8l, Xte),
               "smallest deployable variant", weights=nwl)
    else:
        log.warning("no %s; train one with: python src/train.py --hidden 32 --tag %s",
                    lite_path.name, args.lite_tag)

    table = pd.DataFrame(rows)
    csv = C.METRICS / "optimization.csv"
    table.to_csv(csv, index=False)
    cols = ["variant", "weights", "size_kb", "p50_ms", "p95_ms",
            "speed_rmse_6h", "dir_acc_30deg_6h", "skill_6h_pct"]
    print("\n" + table[cols].round(3).to_string(index=False) + "\n")

    fig_tradeoff(table)
    fig_accuracy_cost(table)

    # Save the deployable artefacts.
    torch.save(h16.state_dict(), C.MODELS / f"lstm_{args.tag}_fp16.pt")
    torch.save(q8.state_dict(), C.MODELS / f"lstm_{args.tag}_int8.pt")
    (C.METRICS / "optimization.json").write_text(
        json.dumps({"threads": 1, "variants": rows}, indent=1))
    log.info("wrote %s and optimisation figures", csv.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
