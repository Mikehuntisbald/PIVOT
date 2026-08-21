#!/usr/bin/env python3
"""Seal the nine e5 owner checkpoints and held-out evaluation commands."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mmgdino_e5_ownership import OWNERSHIP_MODES
from tools.responsibility_isolation_cache import file_sha256
from tools.train_mmgdino_e5_ownership import FORMAL_SEEDS


SCHEMA = "arrow.mmgdino_e5_ownership.evaluation_preregistration/v1"
EXPERIMENT_ROOT = ROOT / "outputs/mmgdino_e5_ownership_transfer_20260821"


class EvaluationPreregError(RuntimeError):
    pass


def _record(path: Path, rows: int | None = None) -> dict[str, Any]:
    path = path.resolve(strict=True)
    value = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    if rows is not None:
        with path.open("r", encoding="utf-8") as handle:
            actual = sum(1 for _ in handle)
        if actual != rows:
            raise EvaluationPreregError(
                f"row count drift: {path}: expected {rows}, got {actual}"
            )
        value["rows"] = actual
    return value


def _atomic_json(value: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build(output: Path) -> dict[str, Any]:
    if output.exists():
        raise EvaluationPreregError("evaluation preregistration already exists")
    checkpoints = {}
    for ownership in OWNERSHIP_MODES:
        checkpoints[ownership] = {}
        for seed in FORMAL_SEEDS:
            directory = EXPERIMENT_ROOT / f"formal/{ownership}/seed{seed}"
            receipt_path = directory / "training_receipt.json"
            checkpoint_path = directory / "checkpoint_u150.pt"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("status") != "complete":
                raise EvaluationPreregError(
                    f"incomplete trajectory: {ownership}/seed{seed}"
                )
            if receipt["updates"] != {
                "total": 150,
                "rank": 100,
                "confidence": 50,
                "nonfinite": 0,
                "amp_skips": 0,
            }:
                raise EvaluationPreregError("formal update contract drifted")
            if receipt["optimizers"]["weight_decay"] != 0.0:
                raise EvaluationPreregError("formal weight decay drifted")
            if not receipt["optimizers"]["task_specific_states"]:
                raise EvaluationPreregError("task-specific optimizer states missing")
            if receipt["d3_queue"]["count"] != 400:
                raise EvaluationPreregError("D3 queue exposure drifted")
            if receipt["runtime"].get("cublas_workspace_config") not in (
                ":4096:8",
                ":16:8",
            ):
                raise EvaluationPreregError("deterministic cuBLAS contract missing")
            if not receipt["cache_unchanged"]:
                raise EvaluationPreregError("frozen candidate cache changed")
            checkpoints[ownership][str(seed)] = {
                "checkpoint": _record(checkpoint_path),
                "training_receipt": _record(receipt_path),
                "model_state_sha256": receipt["checkpoint"][
                    "model_state_sha256"
                ],
                "architecture": receipt["architecture"],
                "gradient_u150": receipt["gradient_probes"]["150"],
            }
    evaluation_root = EXPERIMENT_ROOT / "evaluation"
    cache_root = EXPERIMENT_ROOT / "evaluation_caches"
    if evaluation_root.exists() or cache_root.exists():
        raise EvaluationPreregError(
            "held-out evaluation or cache outputs already exist before lock"
        )
    ref_root = (
        ROOT
        / "outputs/paper_cvpr_v1/baseline_b58_ref8_seed42/refcoco_eval_inputs"
    )
    calibration = (
        ROOT
        / "outputs/u2v5_leakage_clean_anchor_20260817/formal/"
        "calibration_u25_u50_u100/tn_eval_inputs/tn_screen_calibration.jsonl"
    )
    strict = (
        ROOT
        / "outputs/u2v5_leakage_clean_anchor_20260817/final_once/"
        "strict2031_u50/tn_eval_inputs/tn_refcocop_val_refcocog_umd_val.jsonl"
    )
    payload = {
        "schema": SCHEMA,
        "status": "locked_after_u150_and_before_eval_cache_extraction",
        "locked_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        "parent_preregistration": _record(
            ROOT / "paper/data/mmgdino_e5_ownership_preregistration.json"
        ),
        "runtime_amendments": [
            _record(ROOT / "paper/data/mmgdino_e5_ownership_runtime_amendment.json"),
            _record(
                ROOT
                / "paper/data/mmgdino_e5_ownership_order_statistic_amendment.json"
            ),
        ],
        "checkpoints": checkpoints,
        "surfaces": {
            "refcoco_val": _record(ref_root / "refcoco_unc_val.jsonl", 10834),
            "refcoco_testA": _record(ref_root / "refcoco_unc_testA.jsonl", 5657),
            "refcoco_testB": _record(ref_root / "refcoco_unc_testB.jsonl", 5095),
            "d3_calibration": _record(calibration, 1570),
            "strict2031": _record(strict, 2031),
        },
        "routes": {
            "native": "frozen MM-GDINO max-token rank/sample-max score",
            "shared_128": "U150 learned rank and confidence",
            "shared_wide": "U150 learned rank and confidence",
            "isolated_128": "U150 learned rank and confidence",
        },
        "evaluation_contract": {
            "ref_primary": "RefCOCO TestA+TestB pooled micro P@1 IoU>=0.5",
            "tn_primary": "Strict2031 FPR95",
            "ref_val_mechanism_only": True,
            "fixed_threshold_source": "each seed D3 calibration positive q05",
            "no_checkpoint_or_milestone_selection": True,
            "testA_testB_and_strict_run_once_after_this_lock": True,
        },
        "statistics": {
            "bootstrap_replicates": 5000,
            "bootstrap_seed": 20260821,
            "cluster": "image_id; RefCOCO TestA/TestB stratified",
            "fpr_q05_recomputed_each_replicate": True,
            "noninferiority_margin": 0.005,
            "holm_family_size": 3,
        },
        "code": {
            "eval_cache_extractor": _record(
                ROOT / "tools/extract_mmgdino_e5_eval_cache.py"
            ),
            "cache_evaluator": _record(
                ROOT / "tools/eval_mmgdino_e5_ownership_cache.py"
            ),
            "bootstrap": _record(
                ROOT / "tools/aggregate_mmgdino_e5_ownership.py"
            ),
            "score_adapter": _record(
                ROOT / "models/GroundingDINO/stage_b_gdino_score_adapter.py"
            ),
        },
        "output_targets_absent": {
            "evaluation_caches": str(cache_root),
            "evaluation": str(evaluation_root),
        },
        "claim_policy": {
            "full_claim_requires_isolated_beats_native_shared128_and_sharedwide": True,
            "full_claim_requires_both_shared_controls_negative_in_all_seeds": True,
            "observed_u150_gradient_mechanism_before_heldout": {
                ownership: {
                    seed: checkpoints[ownership][seed]["gradient_u150"]
                    for seed in checkpoints[ownership]
                }
                for ownership in checkpoints
            },
            "heldout_results_cannot_change_model_or_protocol": True,
        },
    }
    _atomic_json(payload, output)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    payload = build(args.output.resolve())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
