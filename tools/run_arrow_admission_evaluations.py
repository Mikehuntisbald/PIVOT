#!/usr/bin/env python3
"""Run locked ARROW Admission val3, fresh-panel, calibration, and Test5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/haoyi/miniconda/envs/cvpr/bin/python")
OUTPUT = ROOT / "outputs/arrow_admission_input_20260818"
EVALUATIONS = OUTPUT / "evaluations"
TEST5 = ("refcoco_testA", "refcoco_testB", "refcocop_testA", "refcocop_testB", "refcocog_test")
VAL3 = ("refcoco_val", "refcocop_val", "refcocog_val")
ROWS = {
    "AR_B_TEXT": "config/ablations/cfg_arrow_admission_b_text_eval_gap3.py",
    "AR_C_NULL": "config/ablations/cfg_arrow_admission_c_null_eval_gap3.py",
}


class ArrowEvaluationError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha(path)}


def _git() -> dict[str, str]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status.strip():
        raise ArrowEvaluationError("ARROW evaluation requires a clean worktree")
    return {"commit": commit, "status": "clean"}


def _checkpoints(lock: dict, row: str, *, merged: bool) -> list[str]:
    if merged:
        return [str(OUTPUT / f"eval_checkpoints/{row}/seed{seed}.pth") for seed in (17, 42, 73)]
    key = "sealed_D3_confidence" if row == "AR_A_PATCH" else "training_checkpoints"
    values = lock[key] if row == "AR_A_PATCH" else lock[key][row]
    return [str(values[str(seed)]["path"]) for seed in (17, 42, 73)]


def _merge_commands(lock: dict) -> list[list[str]]:
    commands = []
    for row in ROWS:
        for seed in (17, 42, 73):
            commands.append([
                str(PYTHON), "tools/merge_arrow_admission_confidence.py",
                "--admission-checkpoint", lock["training_checkpoints"][row][str(seed)]["path"],
                "--confidence-checkpoint", lock["sealed_D3_confidence"][str(seed)]["path"],
                "--output", str(OUTPUT / f"eval_checkpoints/{row}/seed{seed}.pth"),
            ])
    return commands


def _ref_command(lock: dict, row: str, profile: str, splits: tuple[str, ...]) -> list[str]:
    return [
        str(PYTHON), "tools/eval_text_groundingdino_refcoco_tn.py",
        "--config", ROWS[row], "--ckpts", *_checkpoints(lock, row, merged=True),
        "--output_dir", str(EVALUATIONS / row / profile),
        "--data_root", "/media/haoyi/T9/data", "--device", "cuda:0",
        "--batch_size", "16", "--num_workers", "4", "--seed", "42", "--amp",
        "--ref_splits", *splits, "--skip_tn", "--topk", "1",
        "--holdout_level", "none",
    ]


def _calibration_command(lock: dict, row: str) -> list[str]:
    return [
        str(PYTHON), "tools/eval_text_groundingdino_refcoco_tn.py",
        "--config", "config/ablations/cfg_stageb_u2v5_clean_confidence_d3_u100.py",
        "--ckpts", *_checkpoints(lock, row, merged=True),
        "--output_dir", str(EVALUATIONS / row / "d3_calibration"),
        "--data_root", "/media/haoyi/T9/data", "--device", "cuda:0",
        "--batch_size", "16", "--num_workers", "4", "--seed", "42", "--amp",
        "--tn_jsonl", "data/ablations/stageb_tn_table_b_equal_exposure_20260717/d3_proposal_covered_calibration.jsonl",
        "--screen_calibration_manifest", "--screen_calibration_audit",
        "data/ablations/stageb_tn_table_b_equal_exposure_20260717/audit.json",
        "--skip_ref", "--holdout_level", "none",
    ]


def _panel_command(lock: dict, row: str) -> list[str]:
    config = (
        "config/ablations/cfg_arrow_admission_a_patch_eval_gap3.py"
        if row == "AR_A_PATCH" else ROWS[row]
    )
    return [
        str(PYTHON), "tools/eval_arrow_admission_panel.py", "--row-id", row,
        "--config", config, "--checkpoints",
        *_checkpoints(lock, row, merged=(row != "AR_A_PATCH")),
        "--output-dir", str(EVALUATIONS / row / "fresh_panel"),
        "--device", "cuda:0", "--batch-size", "16", "--num-workers", "4",
    ]


def commands(lock: dict) -> list[list[str]]:
    result = _merge_commands(lock)
    for row in ROWS:
        result.extend([
            _ref_command(lock, row, "val3", VAL3),
            _calibration_command(lock, row),
        ])
    result.append(_panel_command(lock, "AR_A_PATCH"))
    result.extend(_panel_command(lock, row) for row in ROWS)
    for row in ROWS:
        result.append(_ref_command(lock, row, "test5", TEST5))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("dry-run", "run"))
    parser.add_argument("--checkpoint-lock", required=True)
    args = parser.parse_args()
    lock_path = Path(args.checkpoint_lock).resolve(strict=True)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "arrow.stageb.admission_input_checkpoint_lock/v1" or lock.get("status") != "locked_before_model_evaluation":
        raise ArrowEvaluationError("invalid ARROW checkpoint lock")
    if _git() != lock.get("git"):
        raise ArrowEvaluationError("evaluation code differs from checkpoint lock")
    planned = commands(lock)
    if args.action == "dry-run":
        print(json.dumps({"commands": planned}, indent=2))
        return
    if EVALUATIONS.exists():
        raise ArrowEvaluationError(f"evaluation root must be fresh: {EVALUATIONS}")
    env = dict(os.environ)
    env.setdefault("DATA_ROOT", "/media/haoyi/T9/data")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    outputs = []
    for command in planned:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        if "--output_dir" in command:
            path = Path(command[command.index("--output_dir") + 1]) / "summary.json"
        elif "--output-dir" in command:
            path = Path(command[command.index("--output-dir") + 1]) / "summary.json"
        elif "--output" in command:
            path = Path(command[command.index("--output") + 1])
        else:
            continue
        if not path.is_file():
            raise ArrowEvaluationError(f"command did not produce {path}")
        outputs.append(_record(path))
    manifest = {
        "schema": "arrow.stageb.admission_input_evaluations/v1",
        "checkpoint_lock": _record(lock_path), "outputs": outputs,
        "strict_forwarded": False, "status": "complete",
    }
    EVALUATIONS.mkdir(parents=True, exist_ok=True)
    (EVALUATIONS / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, ArrowEvaluationError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
