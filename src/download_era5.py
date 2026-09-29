"""Download ERA5 stratospheric wind/temperature data in half-month chunks.

CDS enforces a per-request "size" cost limit of 60,000 for this dataset. At the
3-hourly sampling in config, one whole month costs ~26,800, so a month is one
request with room to spare. Months are requested in chronological order, so at
any moment the files on disk form a usable contiguous prefix of the record.
Existing files are skipped, so the script is safe to re-run.

Usage:  python src/download_era5.py [--workers 3]
"""
from __future__ import annotations

import argparse
import calendar
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cdsapi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("era5")


# CDS "size" cost here is 12 x days x times x levels x variables; limit 60,000.
COST_LIMIT = 60_000


def chunks() -> list[tuple[int, int, list[int]]]:
    """One entry per month, chronological, each split further only if over cost."""
    out = []
    per_day = 12 * len(C.HOURS) * len(C.PRESSURE_LEVELS) * len(C.VARIABLES)
    max_days = max(1, COST_LIMIT // per_day)
    for year in C.YEARS:
        for month in range(1, 13):
            days = list(range(1, calendar.monthrange(year, month)[1] + 1))
            for i in range(0, len(days), max_days):
                out.append((year, month, days[i:i + max_days]))
    return out


def chunk_target(year: int, month: int, days: list[int]) -> Path:
    stem = f"era5_strat_{year}{month:02d}"
    if days[0] != 1 or days[-1] < 28:
        stem += f"_d{days[0]:02d}-{days[-1]:02d}"
    return C.DATA_RAW / f"{stem}.nc"


# CDS rejects a submission outright when the user already has too many queued
# requests for this dataset. That is a capacity signal, not a bad request, so it
# is retried patiently rather than counted as a hard failure.
THROTTLE_MARKERS = ("temporarily limited", "has been rejected",
                    "too many", "rate limit")


def _is_throttle(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in THROTTLE_MARKERS)


def fetch_chunk(year: int, month: int, days: list[int],
                retries: int = 12) -> tuple[str, bool, str]:
    """Download one chunk. Returns (tag, downloaded_now, message)."""
    tag = f"{year}-{month:02d}" + (
        f" d{days[0]:02d}-{days[-1]:02d}" if (days[0] != 1 or days[-1] < 28) else "")
    out = chunk_target(year, month, days)
    if out.exists() and out.stat().st_size > 50_000:
        return tag, False, f"skip (exists, {out.stat().st_size/1e6:.1f} MB)"

    request = {
        "product_type": ["reanalysis"],
        "variable": C.VARIABLES,
        "pressure_level": C.PRESSURE_LEVELS,
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in days],
        "time": C.HOURS,
        "area": C.AREA,
        "grid": C.GRID,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    tmp = out.with_suffix(".nc.part")
    for attempt in range(1, retries + 1):
        try:
            client = cdsapi.Client(quiet=True, progress=False)
            t0 = time.time()
            client.retrieve(C.DATASET, request, str(tmp))
            os.replace(tmp, out)
            return tag, True, f"ok {out.stat().st_size/1e6:.1f} MB in {time.time()-t0:.0f}s"
        except Exception as exc:  # noqa: BLE001 - retry on anything transient
            tmp.unlink(missing_ok=True)
            if attempt == retries:
                return tag, False, f"FAILED after {retries} tries: {exc}"
            throttled = _is_throttle(exc)
            back = 90 if throttled else min(30 * attempt, 300)
            log.warning("%s attempt %d/%d %s (%s); retry in %ds", tag, attempt,
                        retries, "throttled" if throttled else "failed",
                        " ".join(str(exc).split())[:110], back)
            time.sleep(back)
    return tag, False, "unreachable"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel CDS requests. CDS caps the number of QUEUED "
                         "requests per dataset per user and rejects the excess, "
                         "so 1 (strictly sequential) is the reliable setting.")
    args = ap.parse_args()

    todo = chunks()
    log.info("Target: %d chunks (%d months) -> %s",
             len(todo), len(C.YEARS) * 12, C.DATA_RAW)

    done = failed = skipped = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(fetch_chunk, y, m, d) for y, m, d in todo]
        for i, fut in enumerate(futs, 1):   # chronological order
            tag, got, msg = fut.result()
            level = log.error if "FAILED" in msg else log.info
            level("[%2d/%d] %s  %s", i, len(todo), tag, msg)
            if "FAILED" in msg:
                failed += 1
            elif got:
                done += 1
            else:
                skipped += 1

    log.info("Downloaded %d, skipped %d, failed %d", done, skipped, failed)
    present = sorted(C.DATA_RAW.glob("era5_strat_*.nc"))
    total = sum(p.stat().st_size for p in present) / 1e6
    log.info("On disk: %d files, %.0f MB", len(present), total)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
