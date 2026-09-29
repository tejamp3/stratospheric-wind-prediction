"""Inspect or clear this account's CDS request queue.

CDS caps how many requests one user may have queued for a dataset and rejects
the excess outright. A crashed or killed download leaves its jobs queued, which
then blocks the next run, so being able to see and clear the queue matters.

Usage:
    python src/cds_queue.py            # show the queue
    python src/cds_queue.py --cancel   # cancel everything queued or running
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

CONFIG_CANDIDATES = [Path.home() / ".cdsapirc",
                     Path(__file__).resolve().parent.parent / ".cdsapirc"]
PENDING = ("accepted", "running")


def client():
    from ecmwf.datastores import Client
    for p in CONFIG_CANDIDATES:
        if p.exists():
            cfg = dict(re.findall(r"(\w+):\s*(\S+)", p.read_text()))
            return Client(url=cfg["url"], key=cfg["key"])
    raise FileNotFoundError(f"No .cdsapirc found in {CONFIG_CANDIDATES}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancel", action="store_true",
                    help="cancel every accepted/running job on this account")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    c = client()
    jobs = c.get_jobs(limit=args.limit).json["jobs"]
    print("queue:", dict(Counter(j["status"] for j in jobs)))

    pending = [j for j in jobs if j["status"] in PENDING]
    for j in pending:
        print(f"  {j['status']:10s} {j['jobID']}  created {j.get('created')}")

    if args.cancel and pending:
        ok = 0
        for j in pending:
            try:
                c.get_remote(j["jobID"]).delete()
                ok += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  could not cancel {j['jobID']}: {str(exc)[:80]}")
        print(f"cancelled {ok} of {len(pending)}")
    elif args.cancel:
        print("nothing to cancel")
    return 0


if __name__ == "__main__":
    sys.exit(main())
