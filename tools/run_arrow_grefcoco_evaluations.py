#!/usr/bin/env python3
"""Durable three-seed launcher for preregistered gRefCOCO evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.arrow_grefcoco_common import RUN_SCHEMA, SEEDS, file_record, load_json, write_json_atomic

DEFAULT_PREREG = REPO_ROOT / "outputs/arrow_grefcoco_20260820/preregistration.json"


def _complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        row = load_json(path)
        records = file_record(Path(row["records"]["path"]))
        return row.get("schema") == RUN_SCHEMA and row.get("status") == "complete" and row.get("rows") == 29589 and all(records[key] == row["records"][key] for key in ("sha256", "size_bytes"))
    except Exception:
        return False


def status(preregistration: Path) -> dict:
    prereg = load_json(preregistration)
    root = Path(prereg["execution"]["results_root"])
    return {"schema": "arrow.grefcoco.runner_status/v1", "runs": {str(seed): {"complete": _complete(root / f"seed{seed}/run_receipt.json")} for seed in SEEDS}}


def run_all(preregistration: Path, device: str) -> dict:
    prereg = load_json(preregistration)
    root = Path(prereg["execution"]["results_root"])
    launches = []
    for seed in SEEDS:
        output = root / f"seed{seed}"
        receipt = output / "run_receipt.json"
        if _complete(receipt):
            launches.append({"seed": seed, "status": "reused_complete"})
            continue
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise ValueError(f"incomplete non-empty run directory requires audit: {output}")
        command = [sys.executable, str(REPO_ROOT / "tools/eval_arrow_grefcoco.py"), "--preregistration", str(preregistration), "--seed", str(seed), "--device", device]
        write_json_atomic(output / "launch.json", {"schema": "arrow.grefcoco.launch/v1", "seed": seed, "command": command, "preregistration": file_record(preregistration)})
        with (output / "run.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
        if completed.returncode != 0 or not _complete(receipt):
            raise RuntimeError(f"gRefCOCO seed{seed} failed; see {output / 'run.log'}")
        launches.append({"seed": seed, "status": "complete"})
    return {"schema": "arrow.grefcoco.runner_result/v1", "launches": launches}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "run"))
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    path = args.preregistration.resolve(strict=True)
    print(json.dumps(status(path) if args.command == "status" else run_all(path, args.device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
