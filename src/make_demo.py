"""Build a standalone interactive demo page from the trained model's test forecasts.

Writes `results/demo.html`: one self-contained file with the data inlined, no
network access and no dependencies, so it opens straight from a clone of the
repo. Shows actual against forecast wind over the test period with a crosshair
readout, a lead-time selector, and a table view so every value is reachable
without hovering.

Usage:  python src/make_demo.py [--tag base]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import data as D
import metrics as M
import viz
from evaluate import features_of, fit_ridge, input_steps_of, load_model, physical, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("demo")

MAX_POINTS = 1200   # keeps the inlined payload small enough to open instantly


def build_payload(tag: str) -> dict:
    model, ck = load_model(tag)
    ds, sx, sy, df = D.build_dataset(verbose=False,
                                     input_steps=input_steps_of(ck),
                                     feature_names=features_of(ck))
    pred, true, base = physical(ds["test"], predict(model, ds["test"]["X"]), sx, sy)
    ridge = fit_ridge(ds, sy)
    times = pd.DatetimeIndex(ds["test"]["time"])

    step = max(1, len(times) // MAX_POINTS)
    sl = slice(None, None, step)

    series = {}
    for k, h in enumerate(C.HORIZONS):
        series[str(h)] = {
            "actual": np.round(M.speed(true[sl, k, 0], true[sl, k, 1]), 2).tolist(),
            "lstm": np.round(M.speed(pred[sl, k, 0], pred[sl, k, 1]), 2).tolist(),
            "persistence": np.round(M.speed(base[sl, k, 0], base[sl, k, 1]), 2).tolist(),
            "ridge": np.round(M.speed(ridge[sl, k, 0], ridge[sl, k, 1]), 2).tolist(),
            "actual_dir": np.round(M.direction(true[sl, k, 0], true[sl, k, 1]), 0).tolist(),
            "lstm_dir": np.round(M.direction(pred[sl, k, 0], pred[sl, k, 1]), 0).tolist(),
        }

    metrics_path = C.METRICS / f"test_metrics_{tag}.csv"
    table = pd.read_csv(metrics_path).to_dict("records") if metrics_path.exists() else []

    return {
        "times": [t.strftime("%Y-%m-%d %H:%M") for t in times[sl]],
        "series": series,
        "horizons": [str(h) for h in C.HORIZONS],
        "metrics": table,
        "meta": {
            "tag": tag,
            "station": f"{C.TARGET_LAT:.0f}N {C.TARGET_LON:.0f}E",
            "level": f"{C.PRIMARY_LEVEL} hPa",
            "period": f"{times[0]:%d %b %Y} to {times[-1]:%d %b %Y}",
            "step_hours": C.STEP_HOURS,
            "n_points": len(times[sl]),
            "source": viz.SOURCE_NOTE,
        },
    }


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stratospheric Wind Forecast</title>
<style>
  :root {
    color-scheme: light;
    --surface: #fcfcfb; --panel: #ffffff; --ink: #0b0b0b; --ink2: #52514e;
    --muted: #8a897f; --grid: #e6e5e0; --line: #dcdbd5;
    --actual: #52514e; --lstm: #2a78d6; --persist: #eb6834; --ridge: #1baf7a;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --surface: #1a1a19; --panel: #232322; --ink: #ffffff; --ink2: #c3c2b7;
      --muted: #8a897f; --grid: #33332f; --line: #3d3d38;
      --actual: #c3c2b7; --lstm: #3987e5; --persist: #d95926; --ridge: #199e70;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface: #1a1a19; --panel: #232322; --ink: #ffffff; --ink2: #c3c2b7;
    --muted: #8a897f; --grid: #33332f; --line: #3d3d38;
    --actual: #c3c2b7; --lstm: #3987e5; --persist: #d95926; --ridge: #199e70;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--surface); color: var(--ink);
    font: 15px/1.55 "Segoe UI", system-ui, -apple-system, sans-serif;
    padding: 28px 16px 56px;
  }
  .wrap { max-width: 1060px; margin: 0 auto; }
  h1 { font-size: 21px; margin: 0 0 4px; letter-spacing: -0.01em; }
  .sub { color: var(--ink2); font-size: 13.5px; margin: 0 0 22px; }
  .controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
              margin-bottom: 16px; }
  .controls span { font-size: 13px; color: var(--ink2); margin-right: 4px; }
  button {
    font: inherit; font-size: 13.5px; padding: 6px 13px; cursor: pointer;
    background: var(--panel); color: var(--ink);
    border: 1px solid var(--line); border-radius: 7px;
  }
  button[aria-pressed="true"] { background: var(--lstm); color: #fff;
                                border-color: var(--lstm); }
  button:focus-visible { outline: 2px solid var(--lstm); outline-offset: 2px; }
  .card { background: var(--panel); border: 1px solid var(--line);
          border-radius: 11px; padding: 16px 16px 8px; margin-bottom: 18px; }
  .legend { display: flex; flex-wrap: wrap; gap: 16px; margin: 2px 0 10px;
            font-size: 13px; color: var(--ink2); }
  .legend i { display: inline-block; width: 15px; height: 2.5px;
              border-radius: 2px; margin-right: 6px; vertical-align: 3px; }
  svg { display: block; width: 100%; height: auto; touch-action: none; }
  .tip {
    position: fixed; pointer-events: none; z-index: 20; opacity: 0;
    transition: opacity .1s; background: var(--panel); color: var(--ink);
    border: 1px solid var(--line); border-radius: 9px; padding: 9px 11px;
    font-size: 13px; box-shadow: 0 6px 22px rgba(0,0,0,.14); min-width: 178px;
  }
  .tip h4 { margin: 0 0 7px; font-size: 12px; font-weight: 600;
            color: var(--ink2); letter-spacing: .02em; }
  .tip .row { display: flex; align-items: center; gap: 8px;
              justify-content: space-between; margin-top: 3px; }
  .tip .nm { color: var(--ink2); font-size: 12.5px; }
  .tip .nm i { display: inline-block; width: 13px; height: 2.5px;
               border-radius: 2px; margin-right: 6px; vertical-align: 3px; }
  .tip .v { font-weight: 650; font-variant-numeric: tabular-nums; }
  table { border-collapse: collapse; width: 100%; font-size: 13px;
          font-variant-numeric: tabular-nums; }
  th, td { text-align: right; padding: 7px 9px;
           border-bottom: 1px solid var(--grid); }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--ink2); font-weight: 600; }
  details { margin-top: 6px; }
  summary { cursor: pointer; font-size: 13.5px; color: var(--ink2);
            padding: 6px 0; }
  .note { color: var(--muted); font-size: 11.5px; margin-top: 14px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Stratospheric wind forecast</h1>
  <p class="sub" id="sub"></p>

  <div class="controls" role="group" aria-label="Forecast lead time">
    <span>Lead time</span>
    <div id="leads"></div>
  </div>

  <div class="card">
    <div class="legend" id="legend"></div>
    <svg id="chart" viewBox="0 0 960 380" role="img"
         aria-label="Forecast and observed wind speed over the test period"></svg>
    <details>
      <summary>Show the numbers as a table</summary>
      <div id="tablewrap"></div>
    </details>
    <p class="note" id="note"></p>
  </div>
</div>
<div class="tip" id="tip" role="status" aria-live="polite"></div>

<script id="payload" type="application/json">__DATA__</script>
<script>
(function () {
  "use strict";
  var D = JSON.parse(document.getElementById("payload").textContent);
  var SERIES = [
    { key: "actual",      label: "Observed",    color: "var(--actual)",  w: 2.4 },
    { key: "lstm",        label: "LSTM",        color: "var(--lstm)",    w: 2 },
    { key: "persistence", label: "Persistence", color: "var(--persist)", w: 1.7 },
    { key: "ridge",       label: "Ridge",       color: "var(--ridge)",   w: 1.7 }
  ];
  var W = 960, H = 380, M = { t: 14, r: 14, b: 34, l: 48 };
  var IW = W - M.l - M.r, IH = H - M.t - M.b;
  var NS = "http://www.w3.org/2000/svg";
  var lead = D.horizons[0], hidden = {};

  var svg = document.getElementById("chart");
  var tip = document.getElementById("tip");

  function el(n, a) {
    var e = document.createElementNS(NS, n);
    for (var k in a) { if (a[k] !== null) e.setAttribute(k, a[k]); }
    return e;
  }
  function scaleX(i, n) { return M.l + (n < 2 ? IW / 2 : IW * i / (n - 1)); }

  document.getElementById("sub").textContent =
    D.meta.level + " wind above " + D.meta.station + ", " + D.meta.period +
    ", " + D.meta.step_hours + "-hourly. " + D.meta.n_points + " forecasts.";
  document.getElementById("note").textContent = D.meta.source;

  // ---- lead-time buttons
  var leads = document.getElementById("leads");
  D.horizons.forEach(function (h) {
    var b = document.createElement("button");
    b.type = "button";
    b.textContent = h + " h";
    b.setAttribute("aria-pressed", String(h === lead));
    b.addEventListener("click", function () {
      lead = h;
      [].forEach.call(leads.children, function (x) {
        x.setAttribute("aria-pressed", String(x.textContent === h + " h"));
      });
      draw();
    });
    leads.appendChild(b);
  });

  // ---- legend doubles as a series toggle
  var legend = document.getElementById("legend");
  SERIES.forEach(function (s) {
    var b = document.createElement("button");
    b.type = "button";
    b.style.border = "none";
    b.style.background = "none";
    b.style.padding = "0";
    b.style.color = "var(--ink2)";
    b.setAttribute("aria-pressed", "true");
    var i = document.createElement("i");
    i.style.background = s.color;
    b.appendChild(i);
    b.appendChild(document.createTextNode(s.label));
    b.addEventListener("click", function () {
      hidden[s.key] = !hidden[s.key];
      b.setAttribute("aria-pressed", String(!hidden[s.key]));
      b.style.opacity = hidden[s.key] ? ".42" : "1";
      draw();
    });
    legend.appendChild(b);
  });

  function visible() {
    return SERIES.filter(function (s) { return !hidden[s.key]; });
  }

  function draw() {
    while (svg.firstChild) { svg.removeChild(svg.firstChild); }
    var d = D.series[lead], n = D.times.length, vis = visible();

    var max = 0;
    vis.forEach(function (s) {
      d[s.key].forEach(function (v) { if (v > max) max = v; });
    });
    if (!max) max = 1;
    max = Math.ceil(max / 5) * 5;
    function y(v) { return M.t + IH - (v / max) * IH; }

    // grid + y axis, recessive
    for (var g = 0; g <= 4; g++) {
      var val = max * g / 4, yy = y(val);
      svg.appendChild(el("line", { x1: M.l, x2: W - M.r, y1: yy, y2: yy,
        stroke: "var(--grid)", "stroke-width": 1 }));
      var t = el("text", { x: M.l - 9, y: yy + 4, "text-anchor": "end",
        fill: "var(--muted)", "font-size": 11 });
      t.textContent = String(Math.round(val));
      svg.appendChild(t);
    }
    var ylab = el("text", { x: 12, y: M.t + IH / 2, fill: "var(--ink2)",
      "font-size": 11.5, transform: "rotate(-90 12 " + (M.t + IH / 2) + ")",
      "text-anchor": "middle" });
    ylab.textContent = "Wind speed (m/s)";
    svg.appendChild(ylab);

    // x ticks
    var ticks = Math.min(6, n);
    for (var k = 0; k < ticks; k++) {
      var idx = Math.round(k * (n - 1) / Math.max(ticks - 1, 1));
      var tx = el("text", { x: scaleX(idx, n), y: H - 10,
        "text-anchor": "middle", fill: "var(--muted)", "font-size": 11 });
      tx.textContent = D.times[idx].slice(0, 10);
      svg.appendChild(tx);
    }

    vis.forEach(function (s) {
      var pts = d[s.key].map(function (v, i) {
        return scaleX(i, n).toFixed(1) + "," + y(v).toFixed(1);
      }).join(" ");
      svg.appendChild(el("polyline", { points: pts, fill: "none",
        stroke: s.color, "stroke-width": s.w, "stroke-linejoin": "round",
        "stroke-linecap": "round", opacity: s.key === "actual" ? 1 : .92 }));
    });

    // crosshair: the reader aims at a date, never at a 2px line
    var hair = el("line", { y1: M.t, y2: M.t + IH, stroke: "var(--muted)",
      "stroke-width": 1, "stroke-dasharray": "3 3", opacity: 0 });
    svg.appendChild(hair);
    var dots = vis.map(function (s) {
      var c = el("circle", { r: 4, fill: s.color, stroke: "var(--panel)",
        "stroke-width": 2, opacity: 0 });
      svg.appendChild(c);
      return { s: s, c: c };
    });

    var hit = el("rect", { x: M.l, y: M.t, width: IW, height: IH,
      fill: "transparent" });
    svg.appendChild(hit);

    function at(clientX) {
      var r = svg.getBoundingClientRect();
      var px = (clientX - r.left) * (W / r.width);
      var i = Math.round((px - M.l) / IW * (n - 1));
      return Math.max(0, Math.min(n - 1, i));
    }

    function show(clientX, clientY) {
      var i = at(clientX), x = scaleX(i, n);
      hair.setAttribute("x1", x);
      hair.setAttribute("x2", x);
      hair.setAttribute("opacity", 1);
      while (tip.firstChild) { tip.removeChild(tip.firstChild); }
      var h4 = document.createElement("h4");
      h4.textContent = D.times[i] + "  ·  +" + lead + " h";
      tip.appendChild(h4);
      dots.forEach(function (o) {
        var v = D.series[lead][o.s.key][i];
        o.c.setAttribute("cx", x);
        o.c.setAttribute("cy", y(v));
        o.c.setAttribute("opacity", 1);
        var row = document.createElement("div");
        row.className = "row";
        var nm = document.createElement("span");
        nm.className = "nm";
        var ic = document.createElement("i");
        ic.style.background = o.s.color;
        nm.appendChild(ic);
        nm.appendChild(document.createTextNode(o.s.label));
        var val = document.createElement("span");
        val.className = "v";
        val.textContent = v.toFixed(1) + " m/s";
        row.appendChild(nm);
        row.appendChild(val);
        tip.appendChild(row);
      });
      var dirRow = document.createElement("div");
      dirRow.className = "row";
      var dn = document.createElement("span");
      dn.className = "nm";
      dn.textContent = "Direction, obs / LSTM";
      var dv = document.createElement("span");
      dv.className = "v";
      dv.textContent = D.series[lead].actual_dir[i] + "° / " +
                       D.series[lead].lstm_dir[i] + "°";
      dirRow.appendChild(dn);
      dirRow.appendChild(dv);
      tip.appendChild(dirRow);

      tip.style.opacity = 1;
      var tw = tip.offsetWidth, th = tip.offsetHeight;
      var lx = clientX + 16, ly = clientY - th / 2;
      if (lx + tw > window.innerWidth - 8) { lx = clientX - tw - 16; }
      tip.style.left = Math.max(8, lx) + "px";
      tip.style.top = Math.max(8, Math.min(window.innerHeight - th - 8, ly)) + "px";
    }

    function hide() {
      tip.style.opacity = 0;
      hair.setAttribute("opacity", 0);
      dots.forEach(function (o) { o.c.setAttribute("opacity", 0); });
    }

    hit.addEventListener("pointermove", function (e) { show(e.clientX, e.clientY); });
    hit.addEventListener("pointerleave", hide);
    svg.addEventListener("blur", hide);
  }

  // ---- table view, so no value is reachable only by hovering
  (function buildTable() {
    if (!D.metrics.length) { return; }
    var cols = [
      ["horizon_h", "Lead (h)"],
      ["model_speed_rmse", "LSTM RMSE"],
      ["persist_speed_rmse", "Persistence RMSE"],
      ["ridge_speed_rmse", "Ridge RMSE"],
      ["skill_speed_rmse_pct", "LSTM skill (%)"],
      ["model_dir_acc_30deg", "Within 30° (%)"]
    ].filter(function (c) { return c[0] in D.metrics[0]; });
    var t = document.createElement("table");
    var hr = document.createElement("tr");
    cols.forEach(function (c) {
      var th = document.createElement("th");
      th.textContent = c[1];
      hr.appendChild(th);
    });
    t.appendChild(hr);
    D.metrics.forEach(function (r) {
      var tr = document.createElement("tr");
      cols.forEach(function (c) {
        var td = document.createElement("td");
        var v = r[c[0]];
        td.textContent = typeof v === "number" ? v.toFixed(2) : String(v);
        tr.appendChild(td);
      });
      t.appendChild(tr);
    });
    document.getElementById("tablewrap").appendChild(t);
  })();

  draw();
  window.addEventListener("resize", draw);
})();
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--out", default=str(C.RESULTS / "demo.html"))
    args = ap.parse_args()

    payload = build_payload(args.tag)
    # json.dumps escapes nothing that can close a <script>, but be explicit.
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = HTML.replace("__DATA__", data)

    out = Path(args.out)
    out.write_text(html, encoding="utf-8")
    log.info("wrote %s (%.0f KB, %d forecast points)",
             out, out.stat().st_size / 1024, payload["meta"]["n_points"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
