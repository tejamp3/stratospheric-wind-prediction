"""Config-driven ERA5 downloader with a checksummed manifest.

Constraints of the CDS API, all found by measurement:

* request cost is 12 x days x times x levels x variables, capped at 60,000;
* that cost does not depend on area or grid, so time resolution is the only
  lever on queue time;
* CDS caps queued requests per user, so downloads are strictly sequential;
* a killed run leaves jobs queued that block the next one
  (`python -m stratoballoon.data.cds_queue --cancel` clears them).

Months are fetched in chronological order and existing files are skipped, so the
files on disk always form a contiguous prefix and the script is safe to re-run.
After every successful file the manifest is rewritten with the file's SHA-256,
so any result can name exactly which data it used.

Usage:  python -m stratoballoon.data.era5 [--config configs/data.yaml]
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import hashlib
import json
import logging
import os
import time
from pathlib import Path

from stratoballoon.config import DataConfig, load_data_config

log = logging.getLogger("era5")

COST_LIMIT = 60_000
THROTTLE_MARKERS = ("temporarily limited", "has been rejected", "too many", "rate limit")
# ERA5 is first published as preliminary "ERA5T" and finalised about 2-3 months
# later; values in that window can still be revised.
PRELIMINARY_MONTHS = 3


def months(cfg: DataConfig) -> list[tuple[int, int]]:
    y0, m0 = map(int, cfg.start.split("-"))
    y1, m1 = map(int, cfg.end.split("-"))
    out, y, m = [], y0, m0
    while (y, m) <= (y1, m1):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def chunks(cfg: DataConfig) -> list[tuple[int, int, list[int]]]:
    per_day = 12 * len(cfg.hours) * len(cfg.levels_hpa) * len(cfg.variables)
    max_days = max(1, COST_LIMIT // per_day)
    out = []
    for y, m in months(cfg):
        days = list(range(1, calendar.monthrange(y, m)[1] + 1))
        for i in range(0, len(days), max_days):
            out.append((y, m, days[i:i + max_days]))
    return out


def target(cfg: DataConfig, y: int, m: int, days: list[int]) -> Path:
    stem = f"{cfg.file_prefix}_{y}{m:02d}"
    if days[0] != 1 or days[-1] < calendar.monthrange(y, m)[1]:
        stem += f"_d{days[0]:02d}-{days[-1]:02d}"
    return cfg.raw_path / f"{stem}.nc"


def request(cfg: DataConfig, y: int, m: int, days: list[int]) -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": cfg.variables,
        "pressure_level": [str(p) for p in cfg.levels_hpa],
        "year": [str(y)], "month": [f"{m:02d}"],
        "day": [f"{d:02d}" for d in days],
        "time": cfg.hours,
        "area": cfg.area,
        "grid": [cfg.grid_deg, cfg.grid_deg],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(cfg: DataConfig) -> Path:
    """Record every file with its checksum, so results can cite their data."""
    path = cfg.raw_path / "manifest.json"
    old = json.loads(path.read_text()) if path.exists() else {"files": {}}
    today = dt.date.today()
    files = {}
    for p in sorted(cfg.raw_path.glob(f"{cfg.file_prefix}_*.nc")):
        prev = old["files"].get(p.name, {})
        size = p.stat().st_size
        digest = prev["sha256"] if prev.get("size") == size else sha256(p)
        ym = p.stem[len(cfg.file_prefix) + 1:][:6]
        y, m = int(ym[:4]), int(ym[4:])
        age = (today.year - y) * 12 + today.month - m
        files[p.name] = {"size": size, "sha256": digest,
                         "downloaded": prev.get("downloaded", today.isoformat()),
                         "preliminary_era5t": age <= PRELIMINARY_MONTHS}
    combined = hashlib.sha256("".join(f["sha256"] for f in files.values()).encode()).hexdigest()
    manifest = {"dataset": cfg.dataset, "config": cfg.__dict__, "n_files": len(files),
                "dataset_sha256": combined, "files": files}
    path.write_text(json.dumps(manifest, indent=1))
    return path


def fetch(cfg: DataConfig, y: int, m: int, days: list[int], retries: int = 12) -> str:
    import cdsapi
    out = target(cfg, y, m, days)
    if out.exists() and out.stat().st_size > 50_000:
        return "skip"
    tmp = out.with_suffix(".nc.part")
    for attempt in range(1, retries + 1):
        try:
            t0 = time.time()
            cdsapi.Client(quiet=True, progress=False).retrieve(
                cfg.dataset, request(cfg, y, m, days), str(tmp))
            os.replace(tmp, out)
            return f"ok {out.stat().st_size / 1e6:.1f} MB in {time.time() - t0:.0f}s"
        except Exception as exc:  # noqa: BLE001 - every CDS failure is retried
            tmp.unlink(missing_ok=True)
            if attempt == retries:
                return f"FAILED: {' '.join(str(exc).split())[:160]}"
            throttled = any(k in str(exc).lower() for k in THROTTLE_MARKERS)
            wait = 90 if throttled else min(30 * attempt, 300)
            log.warning("%d-%02d attempt %d failed (%s); retry in %ds", y, m, attempt,
                        " ".join(str(exc).split())[:100], wait)
            time.sleep(wait)
    return "FAILED"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--manifest-only", action="store_true")
    args = ap.parse_args()
    cfg = load_data_config(args.config)
    cfg.raw_path.mkdir(parents=True, exist_ok=True)
    if args.manifest_only:
        log.info("manifest -> %s", write_manifest(cfg))
        return 0

    todo = chunks(cfg)
    log.info("%d requests, %s to %s, %d levels x %d variables, %d-hourly -> %s",
             len(todo), cfg.start, cfg.end, len(cfg.levels_hpa), len(cfg.variables),
             cfg.step_hours, cfg.raw_path)
    failed = 0
    for i, (y, m, days) in enumerate(todo, 1):
        msg = fetch(cfg, y, m, days)
        (log.error if msg.startswith("FAILED") else log.info)(
            "[%3d/%d] %d-%02d %s", i, len(todo), y, m, msg)
        failed += msg.startswith("FAILED")
        if msg.startswith("ok"):
            write_manifest(cfg)
    write_manifest(cfg)
    log.info("done; %d failed", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
