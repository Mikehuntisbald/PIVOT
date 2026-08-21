#!/usr/bin/env python3
"""Run or inspect the nine preregistered MM-GDINO e5 owner trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mmgdino_e5_ownership import OWNERSHIP_MODES
from tools.responsibility_isolation_cache import file_sha256
from tools.train_mmgdino_e5_ownership import (
    FORMAL_SEEDS,
    FormalConfig,
    run_formal_training,
    load_schedule,
)


PREREG = ROOT / "paper/data/mmgdino_e5_ownership_preregistration.json"
RUNTIME_AMENDMENT = ROOT / "paper/data/mmgdino_e5_ownership_runtime_amendment.json"
OUTPUT_ROOT = ROOT / "outputs/mmgdino_e5_ownership_transfer_20260821"


class MatrixRunError(RuntimeError):
    pass


def _paths(ownership: str, seed: int) -> dict[str, Path]:
    return {
        "cache": OUTPUT_ROOT / f"caches/seed{seed}.pt",
        "cache_receipt": OUTPUT_ROOT / f"caches/seed{seed}_receipt.json",
        "schedule": OUTPUT_ROOT / f"schedules/schedule_seed{seed}.json",
        "output": OUTPUT_ROOT / f"formal/{ownership}/seed{seed}",
    }


def _validate_seed_cache(seed: int) -> dict:
    paths = _paths(OWNERSHIP_MODES[0], seed)
    if not paths["cache"].is_file() or not paths["cache_receipt"].is_file():
        raise MatrixRunError(f"seed{seed} frozen cache is incomplete")
    receipt = json.loads(paths["cache_receipt"].read_text(encoding="utf-8"))
    schedule, schedule_sha = load_schedule(paths["schedule"])
    if receipt.get("status") != "complete":
        raise MatrixRunError(f"seed{seed} cache receipt is not complete")
    if receipt.get("output", {}).get("file_sha256") != file_sha256(paths["cache"]):
        raise MatrixRunError(f"seed{seed} cache bytes differ from receipt")
    if receipt.get("inputs", {}).get("rank_jsonl", {}).get("sha256") != schedule["source"]["rank_jsonl_sha256"]:
        raise MatrixRunError(f"seed{seed} rank source differs from schedule")
    if receipt.get("inputs", {}).get("d3_jsonl", {}).get("sha256") != schedule["source"]["d3_jsonl_sha256"]:
        raise MatrixRunError(f"seed{seed} D3 source differs from schedule")
    if receipt.get("source", {}).get("checkpoint_sha256") != "2ec6fbc01ee70e8c18f96e22614053c95f54932fee7fa14b488c404191c05d7b":
        raise MatrixRunError(f"seed{seed} cache checkpoint drifted")
    return {
        "cache_sha256": file_sha256(paths["cache"]),
        "cache_receipt_sha256": file_sha256(paths["cache_receipt"]),
        "schedule_sha256": schedule_sha,
    }


def _preflight() -> dict:
    value = json.loads(PREREG.read_text(encoding="utf-8"))
    if value.get("schema") != "arrow.mmgdino_e5_ownership.preregistration/v1":
        raise MatrixRunError("ownership preregistration schema drifted")
    if value.get("status") != "locked_before_candidate_extraction_and_owner_training":
        raise MatrixRunError("ownership preregistration status drifted")
    for ownership in OWNERSHIP_MODES:
        if ownership not in value["matrix"]:
            raise MatrixRunError(f"preregistration lost arm {ownership}")
    amendment = json.loads(RUNTIME_AMENDMENT.read_text(encoding="utf-8"))
    if amendment.get("status") != "locked_before_first_optimizer_update":
        raise MatrixRunError("runtime amendment status drifted")
    trainer = ROOT / "tools/train_mmgdino_e5_ownership.py"
    if amendment.get("change", {}).get("new_trainer_sha256") != file_sha256(trainer):
        raise MatrixRunError("runtime-amended trainer SHA drifted")
    return value


def status() -> dict:
    prereg = _preflight()
    rows = []
    seed_readiness = {}
    for seed in FORMAL_SEEDS:
        try:
            seed_readiness[seed] = _validate_seed_cache(seed)
        except MatrixRunError:
            seed_readiness[seed] = None
    for ownership in OWNERSHIP_MODES:
        for seed in FORMAL_SEEDS:
            paths = _paths(ownership, seed)
            receipt = paths["output"] / "training_receipt.json"
            checkpoint = paths["output"] / "checkpoint_u150.pt"
            rows.append(
                {
                    "ownership": ownership,
                    "seed": seed,
                    "cache_ready": seed_readiness[seed] is not None,
                    "schedule_ready": paths["schedule"].is_file(),
                    "output_exists": paths["output"].exists(),
                    "complete": receipt.is_file() and checkpoint.is_file(),
                    "receipt_sha256": file_sha256(receipt) if receipt.is_file() else None,
                    "checkpoint_sha256": (
                        file_sha256(checkpoint) if checkpoint.is_file() else None
                    ),
                }
            )
    return {
        "schema": "arrow.mmgdino_e5_ownership.matrix_status/v1",
        "preregistration": {
            "path": str(PREREG),
            "sha256": file_sha256(PREREG),
            "status": prereg["status"],
        },
        "trajectories": rows,
        "complete": sum(row["complete"] for row in rows),
        "total": len(rows),
    }


def run(*, ownership_values: Sequence[str], seeds: Sequence[int], device: str) -> dict:
    _preflight()
    completed = []
    validated_seeds = {seed: _validate_seed_cache(seed) for seed in seeds}
    for ownership in ownership_values:
        for seed in seeds:
            paths = _paths(ownership, seed)
            if seed not in validated_seeds:
                raise MatrixRunError(f"seed{seed} cache was not preflighted")
            if paths["output"].exists():
                raise MatrixRunError(
                    f"trajectory output already exists: {paths['output']}"
                )
            receipt = run_formal_training(
                cache_path=paths["cache"],
                schedule_path=paths["schedule"],
                output_dir=paths["output"],
                config=FormalConfig(
                    ownership=ownership, seed=seed, device=device
                ),
            )
            completed.append(
                {
                    "ownership": ownership,
                    "seed": seed,
                    "receipt": receipt["checkpoint"],
                }
            )
    return {"completed": completed, "status": status()}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "dry-run", "run", "status", "reconcile"))
    parser.add_argument("--ownership", nargs="+", choices=OWNERSHIP_MODES)
    parser.add_argument("--seeds", nargs="+", type=int, choices=FORMAL_SEEDS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    ownership = tuple(args.ownership or OWNERSHIP_MODES)
    seeds = tuple(args.seeds or FORMAL_SEEDS)
    if args.action in ("list", "status", "reconcile"):
        result = status()
    elif args.action == "dry-run":
        current = status()
        result = {
            "would_run": [
                {"ownership": arm, "seed": seed, **{key: str(value) for key, value in _paths(arm, seed).items()}}
                for arm in ownership
                for seed in seeds
            ],
            "status": current,
        }
    else:
        result = run(ownership_values=ownership, seeds=seeds, device=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
