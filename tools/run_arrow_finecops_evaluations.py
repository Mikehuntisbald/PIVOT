#!/usr/bin/env python3
"""Durable sequential launcher for the nine preregistered FineCops runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.arrow_finecops_common import file_record, load_json, write_json_atomic


DEFAULT_PREREG = REPO_ROOT / "outputs/arrow_finecops_20260819/preregistration.json"


def _receipt_ok(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = load_json(path)
        records = payload["records"]
        observed = file_record(Path(records["path"]))
        return (
            payload.get("schema") == "arrow.finecops.run_receipt/v1"
            and payload.get("status") == "complete"
            and int(payload.get("rows", -1)) == 27926
            and observed["sha256"] == records["sha256"]
            and observed["size_bytes"] == records["size_bytes"]
        )
    except Exception:
        return False


def status(preregistration: Path) -> dict:
    prereg = load_json(preregistration)
    root = Path(str(prereg["execution"]["results_root"])).resolve()
    rows = {}
    for route in ("A", "B", "C"):
        rows[route] = {}
        for seed in (17, 42, 73):
            receipt = root / route / f"seed{seed}" / "run_receipt.json"
            rows[route][str(seed)] = {
                "complete": _receipt_ok(receipt),
                "receipt": str(receipt),
            }
    return {"schema": "arrow.finecops.runner_status/v1", "runs": rows}


def run_all(preregistration: Path, device: str) -> dict:
    prereg = load_json(preregistration)
    root = Path(str(prereg["execution"]["results_root"])).resolve()
    launches = []
    for route in ("A", "B", "C"):
        for seed in (17, 42, 73):
            receipt = root / route / f"seed{seed}" / "run_receipt.json"
            if _receipt_ok(receipt):
                launches.append({"route": route, "seed": seed, "status": "reused_complete"})
                continue
            output_dir = receipt.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            if any(output_dir.iterdir()):
                raise ValueError(f"incomplete non-empty run directory requires audit: {output_dir}")
            command = [
                sys.executable,
                str(REPO_ROOT / "tools/eval_arrow_finecops.py"),
                "--preregistration",
                str(preregistration),
                "--route",
                route,
                "--seed",
                str(seed),
                "--device",
                device,
            ]
            launch = {
                "schema": "arrow.finecops.launch/v1",
                "route": route,
                "seed": seed,
                "command": command,
                "preregistration": file_record(preregistration),
            }
            write_json_atomic(output_dir / "launch.json", launch)
            with (output_dir / "run.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if completed.returncode != 0 or not _receipt_ok(receipt):
                raise RuntimeError(f"FineCops {route}/seed{seed} failed; see {output_dir / 'run.log'}")
            launches.append({"route": route, "seed": seed, "status": "complete"})
    return {"schema": "arrow.finecops.runner_result/v1", "launches": launches}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "run"))
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    prereg = args.preregistration.resolve(strict=True)
    payload = status(prereg) if args.command == "status" else run_all(prereg, args.device)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
