#!/usr/bin/env python3
"""Run the preregistered MM-GDINO e6 Shared-Wide/Isolated 2x2."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.eval_mmgdino_e5_ownership_cache import evaluate_cache
from tools.mmgdino_e6_ownership_2x2 import (
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
from tools.train_mmgdino_e5_ownership import FormalConfig, load_schedule, run_formal_training


PREREG_SCHEMA = "arrow.mmgdino_e6_ownership_2x2.preregistration/v1"
STATUS_SCHEMA = "arrow.mmgdino_e6_ownership_2x2.status/v1"


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
    if prereg.get("status") != "locked_before_any_e6_owner_gpu_forward":
        raise RunnerError("preregistration status drifted")
    for record in prereg.get("code", {}).values():
        path = Path(record["path"])
        if file_sha256(path) != record["sha256"]:
            raise RunnerError(f"preregistered code drifted: {path}")
    for trunk_id, spec in TRUNK_SPECS.items():
        if file_sha256(spec.checkpoint) != spec.checkpoint_sha256:
            raise RunnerError(f"{trunk_id} checkpoint drifted")
    if file_sha256(EVAL_CONFIG) != EVAL_CONFIG_SHA256:
        raise RunnerError("evaluation config drifted")
    if file_sha256(SCHEDULE_RECEIPT) != prereg["schedule_contract"]["receipt"]["sha256"]:
        raise RunnerError("schedule receipt drifted")
    return prereg


def _schedule_sources(seed: int) -> dict[str, Mapping[str, Any]]:
    receipt = _json(SCHEDULE_RECEIPT)
    try:
        return receipt["outputs"][str(seed)]
    except KeyError as exc:
        raise RunnerError(f"schedule receipt lost seed{seed}") from exc


def _run_logged(command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment.setdefault("TRANSFORMERS_OFFLINE", "1")
    environment.setdefault("HF_HUB_OFFLINE", "1")
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("command=" + json.dumps(list(command)) + "\n")
        handle.flush()
        subprocess.run(
            list(command), cwd=ROOT, env=environment,
            stdout=handle, stderr=subprocess.STDOUT, check=True,
        )


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
        str(ROOT / "tools/extract_mmgdino_responsibility_cache.py"),
        "--mmdet-root", str(MMDET_ROOT),
        "--mmdet-commit", MMDET_COMMIT,
        "--config", str(EVAL_CONFIG),
        "--config-sha256", EVAL_CONFIG_SHA256,
        "--checkpoint", str(spec.checkpoint),
        "--checkpoint-sha256", spec.checkpoint_sha256,
        "--model-id", spec.model_id,
        "--rank-jsonl", source["rank"]["path"],
        "--rank-jsonl-sha256", source["rank"]["sha256"],
        "--rank-image-root", str(IMAGE_ROOT),
        "--d3-jsonl", source["d3"]["path"],
        "--d3-jsonl-sha256", source["d3"]["sha256"],
        "--d3-image-root", str(IMAGE_ROOT),
        "--output", str(output),
        "--receipt", str(receipt),
        "--shard-id", f"{trunk_id}-owner-seed{seed}",
        "--feature-dtype", "float32",
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
        str(ROOT / "tools/extract_mmgdino_e5_eval_cache.py"),
        "--mode", source["mode"],
        "--input-jsonl", str(source["path"]),
        "--input-sha256", source["sha256"],
        "--image-root", str(IMAGE_ROOT),
        "--surface", surface,
        "--mmdet-root", str(MMDET_ROOT),
        "--mmdet-commit", MMDET_COMMIT,
        "--config", str(EVAL_CONFIG),
        "--config-sha256", EVAL_CONFIG_SHA256,
        "--checkpoint", str(spec.checkpoint),
        "--checkpoint-sha256", spec.checkpoint_sha256,
        "--model-id", spec.model_id,
        "--output", str(eval_cache_path(trunk_id, surface)),
        "--receipt", str(eval_cache_receipt_path(trunk_id, surface)),
        "--device", "cuda:0",
    ]


def _validate_training_cache(trunk_id: str, seed: int) -> dict[str, Any]:
    cache = training_cache_path(trunk_id, seed)
    receipt_path = training_cache_receipt_path(trunk_id, seed)
    if not cache.is_file() or not receipt_path.is_file():
        raise RunnerError(f"{trunk_id}/seed{seed} training cache is incomplete")
    receipt = _json(receipt_path)
    spec = TRUNK_SPECS[trunk_id]
    if receipt.get("status") != "complete":
        raise RunnerError("training cache receipt is incomplete")
    if receipt.get("source", {}).get("checkpoint_sha256") != spec.checkpoint_sha256:
        raise RunnerError("training cache is bound to the wrong trunk")
    if receipt.get("output", {}).get("file_sha256") != file_sha256(cache):
        raise RunnerError("training cache SHA differs from receipt")
    schedule, schedule_sha = load_schedule(schedule_path(seed))
    source = _schedule_sources(seed)
    if schedule_sha != source["schedule"]["sha256"]:
        raise RunnerError("schedule bytes drifted")
    if receipt["inputs"]["rank_jsonl"]["sha256"] != schedule["source"]["rank_jsonl_sha256"]:
        raise RunnerError("rank identities differ from schedule")
    if receipt["inputs"]["d3_jsonl"]["sha256"] != schedule["source"]["d3_jsonl_sha256"]:
        raise RunnerError("D3 identities differ from schedule")
    return receipt


def extract_smoke(trunks: Sequence[str]) -> dict[str, Any]:
    _preflight()
    rows = []
    for trunk_id in trunks:
        output = EXPERIMENT_ROOT / f"smoke/{trunk_id}/cache.pt"
        receipt = EXPERIMENT_ROOT / f"smoke/{trunk_id}/cache_receipt.json"
        if output.exists() or receipt.exists():
            raise RunnerError(f"smoke output already exists for {trunk_id}")
        command = _training_extract_command(
            trunk_id, 17, output=output, receipt=receipt,
            rank_limit=2, pair_limit=2,
        )
        _run_logged(command, EXPERIMENT_ROOT / f"logs/smoke_{trunk_id}.log")
        value = _json(receipt)
        rows.append({
            "trunk": trunk_id,
            "cache_sha256": file_sha256(output),
            "checkpoint_sha256": value["source"]["checkpoint_sha256"],
            "rows": value["output"]["row_count"],
        })
    return {"status": "complete", "smoke": rows}


def extract_training(trunks: Sequence[str], seeds: Sequence[int]) -> dict[str, Any]:
    _preflight()
    completed = []
    for trunk_id in trunks:
        for seed in seeds:
            output = training_cache_path(trunk_id, seed)
            receipt = training_cache_receipt_path(trunk_id, seed)
            if output.exists() or receipt.exists():
                raise RunnerError(f"training cache output already exists: {output}")
            command = _training_extract_command(
                trunk_id, seed, output=output, receipt=receipt
            )
            _run_logged(
                command,
                EXPERIMENT_ROOT / f"logs/extract_training_{trunk_id}_seed{seed}.log",
            )
            value = _validate_training_cache(trunk_id, seed)
            completed.append({
                "trunk": trunk_id, "seed": seed,
                "cache_sha256": value["output"]["file_sha256"],
            })
    return {"status": "complete", "training_caches": completed}


def train(
    trunks: Sequence[str], owners: Sequence[str], seeds: Sequence[int]
) -> dict[str, Any]:
    _preflight()
    completed = []
    for trunk_id in trunks:
        for owner in owners:
            for seed in seeds:
                _validate_training_cache(trunk_id, seed)
                output = owner_output_dir(trunk_id, owner, seed)
                if output.exists():
                    raise RunnerError(f"formal output already exists: {output}")
                receipt = run_formal_training(
                    cache_path=training_cache_path(trunk_id, seed),
                    schedule_path=schedule_path(seed),
                    output_dir=output,
                    config=FormalConfig(ownership=owner, seed=seed, device="cuda"),
                )
                completed.append({
                    "trunk": trunk_id, "owner": owner, "seed": seed,
                    "checkpoint_sha256": receipt["checkpoint"]["sha256"],
                    "gradient_probe_u150": receipt["gradient_probes"]["150"],
                })
    return {"status": "complete", "formal": completed}


def extract_eval(trunks: Sequence[str], surfaces: Sequence[str]) -> dict[str, Any]:
    _preflight()
    completed = []
    for trunk_id in trunks:
        for surface in surfaces:
            output = eval_cache_path(trunk_id, surface)
            receipt = eval_cache_receipt_path(trunk_id, surface)
            if output.exists() or receipt.exists():
                raise RunnerError(f"evaluation cache output already exists: {output}")
            _run_logged(
                _eval_extract_command(trunk_id, surface),
                EXPERIMENT_ROOT / f"logs/extract_eval_{trunk_id}_{surface}.log",
            )
            value = _json(receipt)
            if value["assets"]["checkpoint_sha256"] != TRUNK_SPECS[trunk_id].checkpoint_sha256:
                raise RunnerError("evaluation cache is bound to wrong trunk")
            if value["output"]["file_sha256"] != file_sha256(output):
                raise RunnerError("evaluation cache SHA differs from receipt")
            completed.append({
                "trunk": trunk_id, "surface": surface,
                "cache_sha256": value["output"]["file_sha256"],
                "rows": value["output"]["row_count"],
            })
    return {"status": "complete", "evaluation_caches": completed}


def evaluate(trunks: Sequence[str], surfaces: Sequence[str]) -> dict[str, Any]:
    _preflight()
    completed = []
    for trunk_id in trunks:
        for surface in surfaces:
            cache = eval_cache_path(trunk_id, surface)
            if not cache.is_file():
                raise RunnerError(f"evaluation cache is absent: {cache}")
            for route in ("native", *OWNERS):
                seeds: Sequence[int | None] = (None,) if route == "native" else FORMAL_SEEDS
                for seed in seeds:
                    output = evaluation_output_dir(trunk_id, surface, route, seed)
                    if output.exists():
                        raise RunnerError(f"evaluation output already exists: {output}")
                    checkpoint = (
                        None if route == "native"
                        else owner_checkpoint_path(trunk_id, route, int(seed))
                    )
                    summary = evaluate_cache(
                        cache_path=cache,
                        route=route,
                        surface=surface,
                        output_dir=output,
                        checkpoint_path=checkpoint,
                        device="cuda",
                        batch_size=32,
                    )
                    completed.append({
                        "trunk": trunk_id, "surface": surface,
                        "route": route, "seed": seed,
                        "metrics": summary["metrics"],
                    })
    return {"status": "complete", "evaluations": completed}


def status() -> dict[str, Any]:
    prereg = _preflight()
    cache_rows = []
    formal_rows = []
    evaluation_cache_rows = []
    evaluation_rows = []
    for trunk_id in TRUNK_SPECS:
        for seed in FORMAL_SEEDS:
            cache_rows.append({
                "trunk": trunk_id, "seed": seed,
                "complete": training_cache_path(trunk_id, seed).is_file()
                and training_cache_receipt_path(trunk_id, seed).is_file(),
            })
        for owner in OWNERS:
            for seed in FORMAL_SEEDS:
                formal_rows.append({
                    "trunk": trunk_id, "owner": owner, "seed": seed,
                    "complete": owner_checkpoint_path(trunk_id, owner, seed).is_file()
                    and (owner_output_dir(trunk_id, owner, seed) / "training_receipt.json").is_file(),
                })
        for surface in REF_INPUTS:
            evaluation_cache_rows.append({
                "trunk": trunk_id, "surface": surface,
                "complete": eval_cache_path(trunk_id, surface).is_file()
                and eval_cache_receipt_path(trunk_id, surface).is_file(),
            })
            for route in ("native", *OWNERS):
                seeds: Sequence[int | None] = (None,) if route == "native" else FORMAL_SEEDS
                for seed in seeds:
                    output = evaluation_output_dir(trunk_id, surface, route, seed)
                    evaluation_rows.append({
                        "trunk": trunk_id, "surface": surface,
                        "route": route, "seed": seed,
                        "complete": (output / "summary.json").is_file()
                        and (output / "records.jsonl").is_file(),
                    })
    return {
        "schema": STATUS_SCHEMA,
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": file_sha256(PREREGISTRATION),
            "git_commit": prereg["git"]["commit"],
        },
        "training_caches": cache_rows,
        "formal": formal_rows,
        "evaluation_caches": evaluation_cache_rows,
        "evaluations": evaluation_rows,
        "counts": {
            "training_caches": sum(row["complete"] for row in cache_rows),
            "formal": sum(row["complete"] for row in formal_rows),
            "evaluation_caches": sum(row["complete"] for row in evaluation_cache_rows),
            "evaluations": sum(row["complete"] for row in evaluation_rows),
        },
    }


def _choices(values: Sequence[str] | None, allowed: Sequence[str]) -> tuple[str, ...]:
    result = tuple(allowed if values is None else values)
    if not result or any(value not in allowed for value in result):
        raise RunnerError(f"values must be a nonempty subset of {allowed}")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("list", "status", "dry-run", "extract-smoke", "extract-training", "train", "extract-eval", "evaluate", "aggregate"),
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
    if args.action in ("list", "status"):
        result = status()
    elif args.action == "dry-run":
        _preflight()
        result = {
            "trunks": trunks, "owners": owners, "seeds": seeds,
            "surfaces": surfaces, "output_root": str(EXPERIMENT_ROOT),
        }
    elif args.action == "extract-smoke":
        result = extract_smoke(trunks)
    elif args.action == "extract-training":
        result = extract_training(trunks, seeds)
    elif args.action == "train":
        result = train(trunks, owners, seeds)
    elif args.action == "extract-eval":
        result = extract_eval(trunks, surfaces)
    elif args.action == "evaluate":
        result = evaluate(trunks, surfaces)
    else:
        from tools.aggregate_mmgdino_e6_ownership_2x2 import aggregate

        result = aggregate(
            evaluation_root=EXPERIMENT_ROOT / "evaluation",
            formal_root=EXPERIMENT_ROOT / "formal",
            output=EXPERIMENT_ROOT / "aggregate.json",
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RunnerError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"error: {error}") from error
