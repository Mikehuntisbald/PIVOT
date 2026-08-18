#!/usr/bin/env python3
"""Launch and audit the six new ARROW Admission-input trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stageb_arrow_admission_contract import (  # noqa: E402
    RUNTIME_SCHEMA,
    SOURCES,
    validate_checkpoint_payload,
)


SCHEMA = "arrow.stageb.admission_input_launch/v1"
POSTFLIGHT_SCHEMA = "arrow.stageb.admission_input_postflight/v1"
SEEDS = (17, 42, 73)
ROWS = {
    "AR_B_TEXT": "config/ablations/cfg_arrow_admission_b_text_u100.py",
    "AR_C_NULL": "config/ablations/cfg_arrow_admission_c_null_u100.py",
}
DATASET = "config/datasets_stageb_u2_category_complete_three_ref.json"
INITIALIZER = (
    "outputs/u2v5_leakage_clean_anchor_20260817/initializer/"
    "checkpoint_clean_init.pth"
)
DEFAULT_OUTPUT = ROOT / "outputs/arrow_admission_input_20260818/training"
DEFAULT_PYTHON = Path("/home/haoyi/miniconda/envs/cvpr/bin/python")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha(path)}


def _env(primary: str, legacy: str, default: str) -> str:
    current, old = os.environ.get(primary), os.environ.get(legacy)
    if current and old and current != old:
        raise RuntimeError(f"{primary} conflicts with legacy {legacy}")
    return current or old or default


def _python() -> Path:
    path = Path(_env("ARROW_PYTHON", "PIVOT_PYTHON", str(DEFAULT_PYTHON))).resolve(strict=True)
    if not os.access(path, os.X_OK):
        raise PermissionError(f"python is not executable: {path}")
    return path


def _output() -> Path:
    return Path(_env("ARROW_ADMISSION_OUTPUT", "PIVOT_ARROW_ADMISSION_OUTPUT", str(DEFAULT_OUTPUT))).resolve()


def _parse(run_id: str) -> tuple[str, int]:
    row, separator, seed_text = str(run_id).partition(":")
    if separator != ":" or row not in ROWS:
        raise ValueError(f"invalid ARROW run id {run_id!r}")
    seed = int(seed_text)
    if seed not in SEEDS:
        raise ValueError(f"invalid ARROW seed {seed}")
    return row, seed


def _run_root(row: str, seed: int) -> Path:
    return _output() / row / f"seed{seed}"


def _resolve_data_path(value: str) -> Path:
    if value.startswith("/home/user/PIVOT/"):
        value = str(ROOT / value.removeprefix("/home/user/PIVOT/"))
    return Path(value.replace("${DATA_ROOT}", os.environ.get("DATA_ROOT", "/media/haoyi/T9/data"))).resolve(strict=True)


def _inputs(row: str) -> dict[str, Any]:
    config = (ROOT / ROWS[row]).resolve(strict=True)
    dataset = (ROOT / DATASET).resolve(strict=True)
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    result = {
        "config": _record(config), "dataset_config": _record(dataset),
        "initializer": _record((ROOT / INITIALIZER).resolve(strict=True)),
    }
    for index, entry in enumerate(payload["train"]):
        result[f"dataset_{index}"] = _record(_resolve_data_path(entry["anno"]))
    for relative in (
        "main.py", "engine.py", "models/GroundingDINO/groundingdino.py",
        "models/GroundingDINO/stage_b_u0_patch_rank.py",
        "tools/stageb_arrow_admission_contract.py",
        "tools/run_arrow_admission_matrix.py",
    ):
        result[f"code_{relative}"] = _record(ROOT / relative)
    return result


def _git() -> dict[str, str]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    return {"commit": commit, "status": "clean" if not status.strip() else "dirty"}


def _command(row: str, seed: int) -> list[str]:
    return [
        str(_python()), "main.py", "--config_file", ROWS[row],
        "--datasets", DATASET, "--output_dir", str(_run_root(row, seed)),
        "--pretrain_model_path", str((ROOT / INITIALIZER).resolve(strict=True)),
        "--options", "batch_size=56", "epochs=1", "--max_train_iters", "100",
        "--iter_checkpoint_interval", "100", "--num_workers",
        _env("ARROW_NUM_WORKERS", "PIVOT_NUM_WORKERS", "8"),
        "--prefetch_factor", _env("ARROW_PREFETCH_FACTOR", "PIVOT_PREFETCH_FACTOR", "1"),
        "--mp_sharing_strategy", "file_system", "--min_nofile", "65536",
        "--amp", "--save_log", "--seed", str(seed),
    ]


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _plan(row: str, seed: int) -> dict[str, Any]:
    return {
        "schema": SCHEMA, "created_at": _now(), "row_id": row,
        "source": SOURCES[row], "seed": seed, "updates": 100,
        "physical_batch": 56, "git": _git(), "inputs": _inputs(row),
        "command": _command(row, seed), "output_dir": str(_run_root(row, seed)),
    }


def _postflight(row: str, seed: int, plan: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = _run_root(row, seed) / "checkpoint_iter.pth"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("optimizer_updates", -1)) != 100:
        raise RuntimeError("ARROW optimizer update count drifted")
    args = payload.get("args")
    if not isinstance(args, Mapping):
        raise RuntimeError("ARROW checkpoint lacks args")
    runtime = args.get("stage_b_arrow_admission_runtime_audit")
    if not isinstance(runtime, Mapping) or runtime.get("schema") != RUNTIME_SCHEMA:
        raise RuntimeError("ARROW checkpoint lacks runtime audit")
    for key, expected in (
        ("successful_optimizer_steps", 100), ("amp_skipped_optimizer_steps", 0),
        ("nonfinite_gradient_boundaries", 0),
    ):
        if int(runtime.get(key, -1)) != expected:
            raise RuntimeError(f"ARROW runtime {key} drifted")
    # Load into the serialized model-shaped state validator without changing bytes.
    class StateOnly(torch.nn.Module):
        def __init__(self, state):
            super().__init__()
            self._state = state
        def state_dict(self, *args, **kwargs):
            return self._state
    validate_checkpoint_payload(
        StateOnly(payload["model"]), payload, row_id=row, source=SOURCES[row]
    )
    if _git() != plan["git"]:
        raise RuntimeError("git state changed during ARROW training")
    for name, bound in plan["inputs"].items():
        if _record(Path(bound["path"])) != bound:
            raise RuntimeError(f"ARROW input changed during training: {name}")
    return {
        "schema": POSTFLIGHT_SCHEMA, "status": "passed", "completed_at": _now(),
        "row_id": row, "source": SOURCES[row], "seed": seed,
        "checkpoint": _record(checkpoint), "optimizer_updates": 100,
        "runtime_audit": dict(runtime), "inputs_rehashed": True,
    }


def _run(row: str, seed: int) -> None:
    root = _run_root(row, seed)
    if root.exists():
        raise FileExistsError(f"formal ARROW output must be fresh: {root}")
    plan = _plan(row, seed)
    if plan["git"]["status"] != "clean":
        raise RuntimeError("formal ARROW training requires a clean worktree")
    launch = root.parent / f"seed{seed}.launch.json"
    _write(launch, plan)
    env = dict(os.environ)
    env.setdefault("DATA_ROOT", "/media/haoyi/T9/data")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    subprocess.run(plan["command"], cwd=ROOT, env=env, check=True)
    _write(root / "postflight.json", _postflight(row, seed, plan))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "dry-run", "run", "status"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.action == "list":
        print(json.dumps({"schema": SCHEMA, "rows": ROWS, "seeds": SEEDS}, indent=2))
        return
    if args.action == "status":
        print(json.dumps({
            f"{row}:{seed}": (_run_root(row, seed) / "postflight.json").is_file()
            for row in ROWS for seed in SEEDS
        }, indent=2))
        return
    if not args.run_id:
        raise ValueError("--run-id is required")
    row, seed = _parse(args.run_id)
    if args.action == "dry-run":
        print(json.dumps(_plan(row, seed), indent=2, sort_keys=True))
    else:
        _run(row, seed)


if __name__ == "__main__":
    main()
