#!/usr/bin/env python3
"""Run the frozen MM-GDINO-T pretrained Shared-Wide/Isolated replay."""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tools.run_mmgdino_e6_ownership_2x2 as mature
from tools.mmgdino_pretrain_ownership import (
    EVAL_CONFIG,
    EVAL_CONFIG_SHA256,
    EXPERIMENT_ROOT,
    FORMAL_SEEDS,
    IMAGE_ROOT,
    MMDET_COMMIT,
    MMDET_ROOT,
    OWNERS,
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


PREREG_SCHEMA = "arrow.mmgdino_pretrain_ownership.preregistration/v1"
STATUS_SCHEMA = "arrow.mmgdino_pretrain_ownership.status/v1"
RUNTIME_AMENDMENT = (
    ROOT / "paper/data/mmgdino_pretrain_ownership_runtime_amendment.json"
)


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
    if prereg.get("status") != "locked_before_any_pretrained_trunk_gpu_forward":
        raise RunnerError("preregistration status drifted")
    amendment = _json(RUNTIME_AMENDMENT) if RUNTIME_AMENDMENT.is_file() else None
    for name, record in prereg.get("code", {}).items():
        path = Path(record["path"])
        actual = file_sha256(path)
        if actual == record["sha256"]:
            continue
        amended = bool(
            name == "runner"
            and amendment is not None
            and amendment.get("schema")
            == "arrow.mmgdino_pretrain_ownership.runtime_amendment/v1"
            and amendment.get("status")
            == "locked_after_pre_forward_import_failure_before_retry"
            and amendment.get("parent_preregistration_sha256")
            == file_sha256(PREREGISTRATION)
            and amendment.get("change", {}).get("new_runner_sha256") == actual
        )
        if not amended:
            raise RunnerError(f"preregistered code drifted: {path}")
    for spec in TRUNK_SPECS.values():
        if file_sha256(spec.checkpoint) != spec.checkpoint_sha256:
            raise RunnerError("pretrained checkpoint drifted")
    if file_sha256(EVAL_CONFIG) != EVAL_CONFIG_SHA256:
        raise RunnerError("evaluation config drifted")
    if file_sha256(SCHEDULE_RECEIPT) != prereg["schedule_contract"]["receipt"]["sha256"]:
        raise RunnerError("schedule receipt drifted")
    commit = subprocess.run(
        ("git", "-C", str(MMDET_ROOT), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != MMDET_COMMIT:
        raise RunnerError("MMDetection checkout drifted")
    return prereg


def _replacements() -> dict[str, Any]:
    return {
        "EVAL_CONFIG": EVAL_CONFIG,
        "EVAL_CONFIG_SHA256": EVAL_CONFIG_SHA256,
        "EXPERIMENT_ROOT": EXPERIMENT_ROOT,
        "FORMAL_SEEDS": FORMAL_SEEDS,
        "IMAGE_ROOT": IMAGE_ROOT,
        "MMDET_COMMIT": MMDET_COMMIT,
        "MMDET_ROOT": MMDET_ROOT,
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


def _training_extract_command(*args, **kwargs):
    with _mature_context():
        return mature._training_extract_command(*args, **kwargs)


def _eval_extract_command(*args, **kwargs):
    with _mature_context():
        return mature._eval_extract_command(*args, **kwargs)


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
            from tools.aggregate_mmgdino_pretrain_ownership import aggregate

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
        OSError,
        ValueError,
        KeyError,
        RunnerError,
        mature.RunnerError,
        subprocess.CalledProcessError,
    ) as error:
        raise SystemExit(f"error: {error}") from error


__all__ = ["PREREG_SCHEMA", "RunnerError", "STATUS_SCHEMA", "main", "status"]
