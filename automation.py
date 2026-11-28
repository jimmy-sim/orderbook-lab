"""Run the daily research pipeline and record its outcome for launchd."""

from __future__ import annotations

import fcntl
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
STATUS = LOGS / "automation-status.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(**fields: object) -> None:
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(fields, indent=2) + "\n")
    tmp.replace(STATUS)


def main() -> int:
    LOGS.mkdir(exist_ok=True)
    with (LOGS / "automation.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"{now()} daily analysis is already running", flush=True)
            return 0

        started = now()
        started_epoch = time.time()
        write_status(state="running", started_at=started)
        try:
            for script in ("run_all.py", "plots.py"):
                print(f"{now()} starting {script}", flush=True)
                subprocess.run([sys.executable, "-u", script], cwd=ROOT, check=True)
                print(f"{now()} finished {script}", flush=True)
            outputs = [ROOT / "results" / name for name in (
                "data_summary.csv", "model_metrics.csv", "cost_fee_grid.csv",
                "predictions_BTC-USD.parquet", "predictions_ETH-USD.parquet",
            )]
            outputs += [ROOT / "docs" / f"{stem}.{suffix}"
                        for stem in ("signal", "decay", "edge_vs_fees")
                        for suffix in ("svg", "png")]
            stale = [str(path) for path in outputs
                     if not path.is_file() or path.stat().st_mtime < started_epoch - 1]
            if stale:
                raise RuntimeError(f"expected fresh outputs missing: {', '.join(stale)}")
            free_gb = round(shutil.disk_usage(ROOT).free / 1024**3, 1)
            if free_gb < 10:
                print(f"{now()} warning: only {free_gb} GiB disk space free", flush=True)
            write_status(state="ok", started_at=started, finished_at=now(), free_disk_gb=free_gb)
            return 0
        except Exception as exc:
            write_status(state="failed", started_at=started, finished_at=now(), error=str(exc))
            print(f"{now()} daily analysis failed: {exc}", file=sys.stderr, flush=True)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
