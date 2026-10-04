"""Run metadata: what produced a result, so it can be reproduced or questioned.

Every experiment writes `run.json` next to its outputs with the resolved config,
the git commit (and whether the tree was dirty), the dataset checksum from the
manifest, seeds, package versions and wall time.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

from stratoballoon.config import ROOT


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - metadata must never break a run
        return "unknown"


def data_fingerprint(paths: list[Path]) -> dict:
    """Checksum from a manifest if present, else from file names and sizes."""
    paths = sorted(p for p in paths if Path(p).exists())
    if not paths:
        return {"n_files": 0}
    manifest = paths[0].parent / "manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        names = {p.name for p in paths}
        digests = [m["files"][n]["sha256"] for n in sorted(names) if n in m["files"]]
        if len(digests) == len(names):
            return {"n_files": len(paths),
                    "sha256": hashlib.sha256("".join(digests).encode()).hexdigest(),
                    "source": "manifest"}
    h = hashlib.sha256("".join(f"{p.name}:{p.stat().st_size}" for p in paths).encode())
    return {"n_files": len(paths), "sha256": h.hexdigest(), "source": "names+sizes"}


class Run:
    def __init__(self, out_dir: Path, config: dict, data_paths: list[Path]):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.meta = {
            "started": dt.datetime.now().isoformat(timespec="seconds"),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "python": platform.python_version(),
            "config": config,
            "data": data_fingerprint(data_paths),
        }

    def finish(self, **extra) -> Path:
        import numpy
        import sklearn
        self.meta.update(extra)
        self.meta["minutes"] = round((time.time() - self.t0) / 60, 2)
        self.meta["versions"] = {"numpy": numpy.__version__, "sklearn": sklearn.__version__}
        p = self.out / "run.json"
        p.write_text(json.dumps(self.meta, indent=1, default=str))
        return p
