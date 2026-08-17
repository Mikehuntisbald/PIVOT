#!/usr/bin/env python3
"""Seal all U2-v5 ablation rows before confirmatory Test5/strict evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.stageb_u2v5_ablation_registry import FORMAL_ROWS, ROOT, SEEDS, validate_registry


SCHEMA = "pivot.stageb.u2v5_ablation_preregistration/v1"
TRAINING = ROOT / "outputs/u2v5_cvpr_ablation_20260817/training"
EVALUATIONS = ROOT / "outputs/u2v5_cvpr_ablation_20260817/evaluations/mechanism"
FINAL_ROOT = ROOT / "outputs/u2v5_cvpr_ablation_20260817/evaluations/confirmatory"


class PreregError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _git() -> dict[str, str]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status.strip():
        raise PreregError("ablation preregistration requires a clean worktree")
    return {"commit": commit, "status": "clean"}


def _mechanism_summaries(row_id: str, block: str) -> list[Path]:
    root = EVALUATIONS / row_id
    if block == "A":
        paths = (
            [root / "deployment_parity.json"]
            if row_id == "A2" else [root / "val3/summary.json"]
        )
    elif block == "C":
        paths = [root / "d3_calibration/summary.json"]
    elif block == "D":
        paths = [root / "calibration/summary.json"]
    elif block == "O":
        paths = [root / "val3/summary.json", root / "d3_calibration/summary.json"]
    else:
        raise PreregError(f"unexpected formal block {block}")
    for path in paths:
        if not path.is_file():
            raise PreregError(f"row {row_id} lacks mechanism summary: {path}")
    return paths


def _checkpoint(row_id: str, seed: int) -> tuple[dict, dict]:
    root = TRAINING / row_id / f"seed{seed}"
    checkpoint = root / "checkpoint_iter.pth"
    postflight = root / "postflight.json"
    if not checkpoint.is_file() or not postflight.is_file():
        raise PreregError(f"row {row_id}:{seed} is incomplete")
    receipt = json.loads(postflight.read_text(encoding="utf-8"))
    if receipt.get("status") != "passed" or receipt.get("optimizer_updates") is None:
        raise PreregError(f"row {row_id}:{seed} postflight did not pass")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = payload.get("args")
    if not isinstance(args, dict) or args.get("stage_b_u2v2_c100_checkpoint") is not None or args.get("stage_b_u2v2_c100_sha256") is not None:
        raise PreregError(f"row {row_id}:{seed} has forbidden C100 provenance")
    return _record(checkpoint), _record(postflight)


def _strict_subset_contract() -> dict[str, Any]:
    root = ROOT / "data/eval_manifests/stageb_vlm_verified_strict_ann_umd_val_20260711"
    strict2031 = root / "eval_manifest.jsonl"
    strict1607 = root / "semantic_stageb_union_image_disjoint_manifest.jsonl"
    def ids(path):
        with path.open("r", encoding="utf-8") as handle:
            return {json.loads(line)["sample_id"] for line in handle if line.strip()}
    ids2031, ids1607 = ids(strict2031), ids(strict1607)
    if len(ids2031) != 2031 or len(ids1607) != 1607 or not ids1607 < ids2031:
        raise PreregError("strict1607 is not the exact registered strict2031 subset")
    return {
        "strict2031": _record(strict2031),
        "strict1607": _record(strict1607),
        "strict1607_is_subset": True,
        "additional_strict2031_rows": len(ids2031 - ids1607),
    }


def _write(path: Path, value: dict) -> None:
    path = path.resolve()
    if path.exists():
        raise PreregError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    validate_registry()
    if FINAL_ROOT.exists():
        raise PreregError(f"confirmatory output already exists: {FINAL_ROOT}")
    checkpoints, postflights, mechanism = {}, {}, {}
    for row in FORMAL_ROWS:
        checkpoints[row.row_id], postflights[row.row_id] = {}, {}
        for seed in SEEDS:
            checkpoint, postflight = _checkpoint(row.row_id, seed)
            checkpoints[row.row_id][str(seed)] = checkpoint
            postflights[row.row_id][str(seed)] = postflight
        mechanism[row.row_id] = [
            _record(path) for path in _mechanism_summaries(row.row_id, row.block)
        ]
    payload = {
        "schema": SCHEMA,
        "status": "locked_before_confirmatory_evaluation",
        "git": _git(),
        "formal_trajectory_count": 42,
        "checkpoints": checkpoints,
        "postflights": postflights,
        "mechanism_summaries": mechanism,
        "contrasts": {
            "admission": {"candidate": "A5", "reference": "A1", "surface": "test5"},
            "confidence": {"candidate": "C3", "reference": "C2", "surface": "strict2031"},
            "ownership_isolation": {"candidate": "O2", "reference": "O0", "surfaces": ["test5", "strict2031"]},
            "ownership_schedule": {"candidate": "O3", "reference": "O2", "surfaces": ["test5", "strict2031"], "test5_noninferiority_margin": 0.005},
            "matched_scope": {"candidate": "D3m", "reference": "D2m", "surface": "strict2031"},
        },
        "strict_contract": _strict_subset_contract(),
        "bootstrap": {"iterations": 5000, "generator": "PCG64", "seed": 20260719, "cluster": "image_id", "same_draw_all_seeds": True, "recompute_positive_q05": True},
        "final_output_root": str(FINAL_ROOT),
        "main_anchor_records_reused_without_reforward": True,
        "c100_formal_hypothesis_forbidden": True,
    }
    _write(Path(args.output), payload)
    print(json.dumps({"status": "locked", "receipt": _record(Path(args.output))}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, PreregError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
