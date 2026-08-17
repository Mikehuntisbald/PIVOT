#!/usr/bin/env python3
"""Launch and audit the leakage-clean U2-v5 CVPR ablation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.stageb_u2v5_ablation_registry import (
    FORMAL_ROWS,
    ROOT,
    SEEDS,
    Row,
    parse_run_id,
    validate_registry,
)


DEFAULT_PYTHON = Path("/home/haoyi/miniconda/envs/cvpr/bin/python")
DEFAULT_OUTPUT = ROOT / "outputs/u2v5_cvpr_ablation_20260817/training"
DEFAULT_ORCHESTRATION = ROOT / "outputs/u2v5_cvpr_ablation_20260817/orchestration"
SCHEMA = "pivot.stageb.u2v5_ablation_launch/v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _git() -> dict[str, str]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    return {"commit": commit, "status": "clean" if not status.strip() else "dirty"}


def _python() -> Path:
    path = Path(os.environ.get("PIVOT_PYTHON", str(DEFAULT_PYTHON))).resolve(
        strict=True
    )
    if not os.access(path, os.X_OK):
        raise PermissionError(f"python is not executable: {path}")
    return path


def _output_root() -> Path:
    return Path(os.environ.get("PIVOT_U2V5_ABLATION_OUTPUT", str(DEFAULT_OUTPUT))).resolve()


def _run_root(row: Row, seed: int) -> Path:
    return _output_root() / row.row_id / f"seed{seed}"


def _resolve_bound_path(raw: str) -> Path:
    value = str(raw).replace(
        "${DATA_ROOT}", os.environ.get("DATA_ROOT", "/media/haoyi/T9/data")
    )
    if value.startswith("/home/user/PIVOT/"):
        value = str(ROOT / value.removeprefix("/home/user/PIVOT/"))
    return Path(value).resolve(strict=True)


def _dataset_source_records(row: Row) -> dict[str, Any]:
    configs = [(ROOT / str(row.dataset)).resolve(strict=True)]
    if row.phase == "ownership":
        pointer = json.loads(configs[0].read_text(encoding="utf-8"))
        configs = [
            (ROOT / pointer["admission"]).resolve(strict=True),
            (ROOT / pointer["confidence"]).resolve(strict=True),
        ]
    result: dict[str, Any] = {}
    for config_index, config in enumerate(configs):
        payload = json.loads(config.read_text(encoding="utf-8"))
        result[f"dataset_config_{config_index}"] = _record(config)
        for entry_index, entry in enumerate(payload.get("train", [])):
            for field in ("anno", "paper_contract_audit"):
                raw = entry.get(field)
                if isinstance(raw, str) and raw != "/":
                    result[f"dataset_{config_index}_{entry_index}_{field}"] = _record(
                        _resolve_bound_path(raw)
                    )
    return result


def _parent(row: Row, seed: int) -> Path:
    if row.phase in {"admission", "ownership"}:
        return (
            ROOT
            / "outputs/u2v5_leakage_clean_anchor_20260817/initializer/"
            "checkpoint_clean_init.pth"
        ).resolve(strict=True)
    if row.phase == "confidence":
        return (
            ROOT
            / f"outputs/u2v5_leakage_clean_anchor_20260817/formal/"
            f"admission_seed{seed}_u100/checkpoint_iter.pth"
        ).resolve(strict=True)
    raise ValueError(f"row {row.row_id} has unsupported phase {row.phase}")


def _command(row: Row, seed: int) -> list[str]:
    python = _python()
    output = _run_root(row, seed)
    if row.phase == "ownership":
        return [
            str(python),
            "tools/train_stageb_u2v5_ownership.py",
            "--row-id", row.row_id,
            "--seed", str(seed),
            "--output-dir", str(output),
            "--initializer", str(_parent(row, seed)),
        ]
    return [
        str(python),
        "main.py",
        "--config_file", str(row.config),
        "--datasets", str(row.dataset),
        "--output_dir", str(output),
        "--pretrain_model_path", str(_parent(row, seed)),
        "--options", f"batch_size={row.batch_size}", "epochs=1",
        "--max_train_iters", str(row.updates),
        "--iter_checkpoint_interval", str(row.updates),
        "--num_workers", os.environ.get("PIVOT_NUM_WORKERS", "8"),
        "--prefetch_factor", os.environ.get("PIVOT_PREFETCH_FACTOR", "1"),
        "--mp_sharing_strategy", "file_system",
        "--min_nofile", "65536",
        "--amp", "--save_log", "--seed", str(seed),
    ]


def _plan(row: Row, seed: int) -> dict[str, Any]:
    config = (ROOT / str(row.config)).resolve(strict=True)
    dataset = (ROOT / str(row.dataset)).resolve(strict=True)
    inputs = {
        "config": _record(config),
        "dataset": _record(dataset),
        "parent": _record(_parent(row, seed)),
        **_dataset_source_records(row),
    }
    for source in (
        ROOT / "main.py",
        ROOT / "engine.py",
        ROOT / "models/GroundingDINO/groundingdino.py",
        ROOT / "models/GroundingDINO/stage_b_gdino_score_adapter.py",
        ROOT / "models/GroundingDINO/stage_b_u0_patch_rank.py",
        ROOT / "tools/stageb_u2v5_ablation_registry.py",
        ROOT / "tools/stageb_u2v5_ablation_contract.py",
        ROOT / "tools/run_stageb_u2v5_ablation_matrix.py",
    ):
        inputs[f"code_{source.name}_{len(inputs)}"] = _record(source)
    if row.phase == "ownership":
        inputs["code_ownership_trainer"] = _record(
            ROOT / "tools/train_stageb_u2v5_ownership.py"
        )
    return {
        "schema": SCHEMA,
        "created_at": _now(),
        "row": row.payload(),
        "seed": seed,
        "git": _git(),
        "inputs": inputs,
        "runtime": {
            "python": _record(_python()),
            "data_root": os.environ.get("DATA_ROOT", "/media/haoyi/T9/data"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        },
        "output_dir": str(_run_root(row, seed)),
        "command": _command(row, seed),
    }


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _postflight(row: Row, seed: int, plan: Mapping[str, Any]) -> dict[str, Any]:
    root = _run_root(row, seed)
    checkpoint = root / "checkpoint_iter.pth"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("optimizer_updates", -1)) != row.updates:
        raise RuntimeError(f"row {row.row_id}:{seed} update count drifted")
    args = payload.get("args")
    if not isinstance(args, Mapping):
        raise RuntimeError("ablation checkpoint lacks saved args")
    if args.get("stage_b_u2v2_c100_checkpoint") is not None or args.get(
        "stage_b_u2v2_c100_sha256"
    ) is not None:
        raise RuntimeError("ablation checkpoint serialized forbidden C100 provenance")
    if row.phase == "admission":
        contract = payload.get("u2v5_ablation")
        runtime = args.get("stage_b_u2v5_ablation_runtime_audit")
    elif row.phase == "confidence":
        contract = payload.get("u2v5_clean_confidence")
        runtime = contract.get("runtime_audit") if isinstance(contract, Mapping) else None
    else:
        contract = payload.get("u2v5_ownership")
        runtime = contract.get("runtime_audit") if isinstance(contract, Mapping) else None
    if not isinstance(contract, Mapping) or not isinstance(runtime, Mapping):
        raise RuntimeError("ablation checkpoint lacks ownership/runtime contract")
    if int(runtime.get("successful_optimizer_steps", -1)) != row.updates:
        raise RuntimeError("ablation runtime successful-step count drifted")
    if int(runtime.get("amp_skipped_optimizer_steps", -1)) != 0:
        raise RuntimeError("ablation runtime contains AMP-skipped steps")
    if int(runtime.get("nonfinite_gradient_boundaries", -1)) != 0:
        raise RuntimeError("ablation runtime contains nonfinite gradients")
    current = _git()
    if current != plan["git"]:
        raise RuntimeError("git state changed during ablation training")
    for name, bound in plan["inputs"].items():
        if _record(Path(bound["path"])) != bound:
            raise RuntimeError(f"ablation input changed during run: {name}")
    return {
        "schema": "pivot.stageb.u2v5_ablation_postflight/v1",
        "completed_at": _now(),
        "row_id": row.row_id,
        "seed": seed,
        "checkpoint": _record(checkpoint),
        "optimizer_updates": row.updates,
        "runtime_audit": dict(runtime),
        "inputs_rehashed": True,
        "status": "passed",
    }


def _execute(row: Row, seed: int) -> None:
    validate_registry()
    root = _run_root(row, seed)
    if root.exists():
        raise FileExistsError(f"formal output must be fresh: {root}")
    plan = _plan(row, seed)
    if plan["git"]["status"] != "clean":
        raise RuntimeError("formal ablation training requires a clean worktree")
    launch_path = root.parent / f"{root.name}.launch.json"
    _write(launch_path, plan)
    env = dict(os.environ)
    env.setdefault("DATA_ROOT", "/media/haoyi/T9/data")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        subprocess.run(plan["command"], cwd=ROOT, env=env, check=True)
    except subprocess.CalledProcessError as error:
        failure = {
            "schema": "pivot.stageb.u2v5_ablation_failure/v1",
            "failed_at": _now(),
            "row_id": row.row_id,
            "seed": seed,
            "returncode": int(error.returncode),
            "git": _git(),
            "launch_manifest": _record(launch_path),
            "status": "failed",
        }
        _write(root / "failed_postflight.json", failure)
        raise
    receipt = _postflight(row, seed, plan)
    _write(root / "postflight.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def _detach(row: Row, seed: int) -> None:
    orchestration = Path(
        os.environ.get("PIVOT_U2V5_ORCHESTRATION", str(DEFAULT_ORCHESTRATION))
    ).resolve()
    job = orchestration / f"{row.row_id}_seed{seed}"
    if job.exists():
        raise FileExistsError(f"orchestration job already exists: {job}")
    job.mkdir(parents=True)
    log = (job / "orchestrator.log").open("xb")
    command = [str(_python()), str(Path(__file__).resolve()), "run", "--run-id", f"{row.row_id}:{seed}"]
    process = subprocess.Popen(
        command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
    )
    launch = {"schema": SCHEMA, "pid": process.pid, "command": command, "created_at": _now()}
    _write(job / "launch.json", launch)
    print(json.dumps({"job": str(job), **launch}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    for name in ("dry-run", "run", "detach"):
        child = sub.add_parser(name)
        child.add_argument("--run-id", required=True)
    status = sub.add_parser("status")
    status.add_argument("job_dir")
    reconcile = sub.add_parser("reconcile")
    reconcile.add_argument("job_dir")
    args = parser.parse_args()
    validate_registry()
    if args.command == "list":
        print(json.dumps({"schema": SCHEMA, "seeds": SEEDS, "rows": [row.payload() for row in FORMAL_ROWS]}, indent=2))
        return
    if args.command in {"status", "reconcile"}:
        job = Path(args.job_dir).resolve(strict=True)
        launch = json.loads((job / "launch.json").read_text(encoding="utf-8"))
        pid = int(launch["pid"])
        alive = Path(f"/proc/{pid}").exists()
        result = {"job": str(job), "pid": pid, "alive": alive}
        if args.command == "reconcile":
            row, seed = parse_run_id(Path(job).name.replace("_seed", ":"))
            postflight = _run_root(row, seed) / "postflight.json"
            result["passed"] = postflight.is_file()
            _write(job / "status.json", result)
        print(json.dumps(result, indent=2))
        return
    row, seed = parse_run_id(args.run_id)
    if args.command == "dry-run":
        print(json.dumps(_plan(row, seed), indent=2, sort_keys=True))
    elif args.command == "run":
        _execute(row, seed)
    else:
        _detach(row, seed)


if __name__ == "__main__":
    main()
