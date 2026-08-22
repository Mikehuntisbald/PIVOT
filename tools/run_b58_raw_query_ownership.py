#!/usr/bin/env python3
"""Run the preregistered 100k Shared-Wide/Isolated replay on B58."""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

import tools.run_mmgdino_e6_ownership_2x2 as mature
from tools.b58_raw_query_ownership import (
    CHECKPOINT_SHA256,
    EVAL_CONFIG,
    EVAL_CONFIG_SHA256,
    EXPERIMENT_ROOT,
    FORMAL_SEEDS,
    IMAGE_ROOT,
    OWNERS,
    PARENT_AGGREGATE,
    PARENT_RESULT,
    PREREGISTRATION,
    REF_INPUTS,
    ROOT,
    SCHEDULE_RECEIPT,
    TRUNK_SPECS,
    eval_cache_path,
    eval_cache_receipt_path,
    evaluation_output_dir,
    owner_checkpoint_path,
    owner_output_dir,
    schedule_path,
    training_cache_path,
    training_cache_receipt_path,
)
from tools.responsibility_isolation_cache import file_sha256


PREREG_SCHEMA = "arrow.b58_raw_query_ownership.preregistration/v1"
STATUS_SCHEMA = "arrow.b58_raw_query_ownership.status/v1"


class RunnerError(RuntimeError):
    pass


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RunnerError(f"JSON object required: {path}")
    return value


def _preflight() -> dict[str, Any]:
    prereg = _json(PREREGISTRATION)
    if prereg.get("schema") != PREREG_SCHEMA:
        raise RunnerError("preregistration schema drifted")
    if prereg.get("status") != "locked_before_any_b58_raw_query_gpu_forward":
        raise RunnerError("preregistration status drifted")
    for name, record in prereg.get("code", {}).items():
        path = Path(record["path"])
        if file_sha256(path) != record["sha256"]:
            raise RunnerError(f"preregistered code drifted: {name}: {path}")
    spec = next(iter(TRUNK_SPECS.values()))
    if file_sha256(spec.checkpoint) != CHECKPOINT_SHA256:
        raise RunnerError("B58 checkpoint drifted")
    if file_sha256(EVAL_CONFIG) != EVAL_CONFIG_SHA256:
        raise RunnerError("evaluation config drifted")
    if file_sha256(SCHEDULE_RECEIPT) != prereg["schedule_contract"]["receipt"]["sha256"]:
        raise RunnerError("schedule receipt drifted")
    if file_sha256(PARENT_RESULT) != prereg["same_head_axis"]["parent_result"]["sha256"]:
        raise RunnerError("sealed parent result drifted")
    if file_sha256(PARENT_AGGREGATE) != prereg["same_head_axis"]["parent_aggregate"]["sha256"]:
        raise RunnerError("sealed parent aggregate drifted")
    for record in prereg["same_head_axis"]["parent_records"]:
        if file_sha256(Path(record["path"])) != record["sha256"]:
            raise RunnerError("sealed parent evaluation record drifted")
    return prereg


def _schedule_sources(seed: int) -> dict[str, Mapping[str, Any]]:
    try:
        return _json(SCHEDULE_RECEIPT)["outputs"][str(seed)]
    except KeyError as exc:
        raise RunnerError(f"schedule receipt lost seed{seed}") from exc


def _training_extract_command(
    trunk_id: str,
    seed: int,
    *,
    output: Path,
    receipt: Path,
    rank_limit: int = 0,
    pair_limit: int = 0,
) -> list[str]:
    spec = TRUNK_SPECS[trunk_id]
    source = _schedule_sources(seed)
    command = [
        sys.executable,
        str(ROOT / "tools/extract_b58_raw_query_ownership_cache.py"),
        "--mode", "training",
        "--config", str(EVAL_CONFIG),
        "--checkpoint", str(spec.checkpoint),
        "--checkpoint-sha256", spec.checkpoint_sha256,
        "--model-id", spec.model_id,
        "--image-root", str(IMAGE_ROOT),
        "--rank-jsonl", source["rank"]["path"],
        "--rank-jsonl-sha256", source["rank"]["sha256"],
        "--d3-jsonl", source["d3"]["path"],
        "--d3-jsonl-sha256", source["d3"]["sha256"],
        "--output", str(output),
        "--receipt", str(receipt),
        "--shard-id", f"{trunk_id}-owner-seed{seed}",
        "--device", "cuda:0",
    ]
    if rank_limit:
        command.extend(("--rank-limit", str(rank_limit)))
    if pair_limit:
        command.extend(("--pair-limit", str(pair_limit)))
    return command


def _eval_extract_command(trunk_id: str, surface: str) -> list[str]:
    spec = TRUNK_SPECS[trunk_id]
    source = REF_INPUTS[surface]
    return [
        sys.executable,
        str(ROOT / "tools/extract_b58_raw_query_ownership_cache.py"),
        "--mode", source["mode"],
        "--config", str(EVAL_CONFIG),
        "--checkpoint", str(spec.checkpoint),
        "--checkpoint-sha256", spec.checkpoint_sha256,
        "--model-id", spec.model_id,
        "--image-root", str(IMAGE_ROOT),
        "--input-jsonl", str(source["path"]),
        "--input-sha256", source["sha256"],
        "--surface", surface,
        "--output", str(eval_cache_path(trunk_id, surface)),
        "--receipt", str(eval_cache_receipt_path(trunk_id, surface)),
        "--device", "cuda:0",
    ]


def _replacements() -> dict[str, Any]:
    return {
        "EXPERIMENT_ROOT": EXPERIMENT_ROOT,
        "FORMAL_SEEDS": FORMAL_SEEDS,
        "IMAGE_ROOT": IMAGE_ROOT,
        "OWNERS": OWNERS,
        "PREREGISTRATION": PREREGISTRATION,
        "REF_INPUTS": REF_INPUTS,
        "ROOT": ROOT,
        "SCHEDULE_RECEIPT": SCHEDULE_RECEIPT,
        "TRUNK_SPECS": TRUNK_SPECS,
        "eval_cache_path": eval_cache_path,
        "eval_cache_receipt_path": eval_cache_receipt_path,
        "evaluation_output_dir": evaluation_output_dir,
        "owner_checkpoint_path": owner_checkpoint_path,
        "owner_output_dir": owner_output_dir,
        "schedule_path": schedule_path,
        "training_cache_path": training_cache_path,
        "training_cache_receipt_path": training_cache_receipt_path,
        "STATUS_SCHEMA": STATUS_SCHEMA,
        "_preflight": _preflight,
        "_training_extract_command": _training_extract_command,
        "_eval_extract_command": _eval_extract_command,
    }


@contextlib.contextmanager
def _mature_context():
    replacements = _replacements()
    previous = {name: getattr(mature, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(mature, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(mature, name, value)


def status() -> dict[str, Any]:
    with _mature_context():
        return mature.status()


def _choices(values: Sequence[str] | None, allowed: Sequence[str]) -> tuple[str, ...]:
    return mature._choices(values, allowed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "list", "status", "dry-run", "extract-smoke", "extract-training",
            "train", "extract-eval", "evaluate", "aggregate",
        ),
    )
    parser.add_argument("--trunks", nargs="+", choices=tuple(TRUNK_SPECS))
    parser.add_argument("--owners", nargs="+", choices=OWNERS)
    parser.add_argument("--seeds", nargs="+", type=int, choices=FORMAL_SEEDS)
    parser.add_argument("--surfaces", nargs="+", choices=tuple(REF_INPUTS))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    trunks = _choices(args.trunks, tuple(TRUNK_SPECS))
    owners = _choices(args.owners, OWNERS)
    seeds = tuple(FORMAL_SEEDS if args.seeds is None else args.seeds)
    surfaces = _choices(args.surfaces, tuple(REF_INPUTS))
    with _mature_context():
        if args.action in ("list", "status"):
            result = mature.status()
        elif args.action == "dry-run":
            _preflight()
            result = {
                "trunks": trunks,
                "owners": owners,
                "seeds": seeds,
                "surfaces": surfaces,
                "output_root": str(EXPERIMENT_ROOT),
            }
        elif args.action == "extract-smoke":
            result = mature.extract_smoke(trunks)
        elif args.action == "extract-training":
            result = mature.extract_training(trunks, seeds)
        elif args.action == "train":
            result = mature.train(trunks, owners, seeds)
        elif args.action == "extract-eval":
            result = mature.extract_eval(trunks, surfaces)
        elif args.action == "evaluate":
            result = mature.evaluate(trunks, surfaces)
        else:
            from tools.aggregate_b58_raw_query_ownership import aggregate

            result = aggregate(
                evaluation_root=EXPERIMENT_ROOT / "evaluation",
                formal_root=EXPERIMENT_ROOT / "formal",
                output=EXPERIMENT_ROOT / "aggregate.json",
            )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (
        OSError, ValueError, KeyError, RunnerError, mature.RunnerError,
        subprocess.CalledProcessError,
    ) as error:
        raise SystemExit(f"error: {error}") from error


__all__ = [
    "PREREG_SCHEMA", "RunnerError", "STATUS_SCHEMA", "_eval_extract_command",
    "_training_extract_command", "main", "status",
]
