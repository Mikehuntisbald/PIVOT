#!/usr/bin/env python3
"""Lock six ARROW checkpoints before val, fresh-panel, or Test5 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "arrow.stageb.admission_input_checkpoint_lock/v1"
TRAINING = ROOT / "outputs/arrow_admission_input_20260818/training"
EVALUATIONS = ROOT / "outputs/arrow_admission_input_20260818/evaluations"


class ArrowLockError(RuntimeError):
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
        raise ArrowLockError("checkpoint lock requires a clean worktree")
    return {"commit": commit, "status": "clean"}


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        raise ArrowLockError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--supersedes")
    args = parser.parse_args()
    prereg_path = Path(args.preregistration).resolve(strict=True)
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if prereg.get("schema") != "arrow.stageb.admission_input_preregistration/v1":
        raise ArrowLockError("invalid ARROW design preregistration")
    current_git = _git()
    previous_lock_record = None
    if current_git != prereg.get("git"):
        if not args.supersedes:
            raise ArrowLockError("code commit differs from design preregistration")
        previous_path = Path(args.supersedes).resolve(strict=True)
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if previous.get("schema") != SCHEMA:
            raise ArrowLockError("superseded checkpoint lock schema drifted")
        previous_lock_record = _record(previous_path)
    if EVALUATIONS.exists():
        raise ArrowLockError("evaluation root already exists before checkpoint lock")
    checkpoints, postflights = {}, {}
    for row in ("AR_B_TEXT", "AR_C_NULL"):
        checkpoints[row], postflights[row] = {}, {}
        for seed in (17, 42, 73):
            root = TRAINING / row / f"seed{seed}"
            checkpoint, postflight = root / "checkpoint_iter.pth", root / "postflight.json"
            receipt = json.loads(postflight.read_text(encoding="utf-8"))
            if receipt.get("status") != "passed" or int(receipt.get("optimizer_updates", -1)) != 100:
                raise ArrowLockError(f"training postflight failed for {row}:{seed}")
            checkpoints[row][str(seed)] = _record(checkpoint)
            postflights[row][str(seed)] = _record(postflight)
    sealed_a = {}
    clean_confidence = {}
    for seed in (17, 42, 73):
        sealed_a[str(seed)] = _record(
            ROOT / f"outputs/u2v5_leakage_clean_anchor_20260817/formal/admission_seed{seed}_u100/checkpoint_iter.pth"
        )
        clean_confidence[str(seed)] = _record(
            ROOT / f"outputs/u2v5_leakage_clean_anchor_20260817/formal/confidence_seed{seed}_u50/checkpoint_iter.pth"
        )
    payload = {
        "schema": SCHEMA, "status": "locked_before_model_evaluation",
        "git": current_git, "design_git": prereg.get("git"),
        "preregistration": _record(prereg_path),
        "training_checkpoints": checkpoints, "postflights": postflights,
        "sealed_A_admission": sealed_a, "sealed_D3_confidence": clean_confidence,
        "new_training_trajectory_count": 6,
        "strict_forward_forbidden": True,
    }
    if previous_lock_record is not None:
        payload["supersedes"] = previous_lock_record
        payload["evaluation_amendment"] = {
            "reason": "confidence-only forward bypasses Admission and merged checkpoint preserves clean-confidence provenance",
            "training_checkpoints_unchanged": True,
            "test5_or_fresh_panel_viewed": False,
        }
    _write(Path(args.output), payload)
    print(json.dumps({"status": "locked", "receipt": _record(Path(args.output))}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, ArrowLockError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
