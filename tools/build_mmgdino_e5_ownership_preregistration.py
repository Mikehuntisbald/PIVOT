#!/usr/bin/env python3
"""Seal the strong-e5 ownership transfer before cache/model-head execution."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mmgdino_e5_ownership import (
    MMGDinoE5ResponsibilityOwners,
    OWNERSHIP_ISOLATED_128,
    OWNERSHIP_SHARED_128,
    OWNERSHIP_SHARED_WIDE,
)
from tools.responsibility_isolation_cache import file_sha256


SCHEMA = "arrow.mmgdino_e5_ownership.preregistration/v1"
CHECKPOINT = ROOT / "weights/epoch_5.pth"
CHECKPOINT_SHA = "2ec6fbc01ee70e8c18f96e22614053c95f54932fee7fa14b488c404191c05d7b"
MMDET_ROOT = Path("/media/haoyi/T9/external/mmgdino_l_baseline/mmdetection")
MMDET_COMMIT = "cfd5d3a985b0249de009b67d04f37263e11cdf3d"
EVAL_CONFIG = Path(
    "/media/haoyi/T9/external/mmgdino_l_baseline/"
    "mmgdino_t_refcoco5e_formal_b4a8_seed20260819.py"
)
EVAL_CONFIG_SHA = "8f22719d7815563006de550e5f4f0173576bedfa8293922aa80c92e7524f2b82"
SCHEDULE_RECEIPT = (
    ROOT
    / "outputs/mmgdino_e5_ownership_transfer_20260821/"
    "schedules/schedule_receipt.json"
)
FORMAL_ROOT = ROOT / "outputs/mmgdino_e5_ownership_transfer_20260821/formal"


class PreregistrationError(RuntimeError):
    pass


def _record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    value = {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }
    if rows is not None:
        with resolved.open("r", encoding="utf-8") as handle:
            actual = sum(1 for _ in handle)
        if actual != rows:
            raise PreregistrationError(
                f"row count drift for {resolved}: expected {rows}, got {actual}"
            )
        value["rows"] = actual
    return value


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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
        raise PreregistrationError("preregistration output already exists")
    if file_sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise PreregistrationError("epoch-5 checkpoint SHA drifted")
    if _git_head(MMDET_ROOT) != MMDET_COMMIT:
        raise PreregistrationError("MMDetection commit drifted")
    if file_sha256(EVAL_CONFIG) != EVAL_CONFIG_SHA:
        raise PreregistrationError("MM-GDINO eval config drifted")
    try:
        checkpoint = torch.load(
            CHECKPOINT, map_location="cpu", mmap=True, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    meta = checkpoint.get("meta", {})
    if meta.get("epoch") != 5 or meta.get("iter") != 37695:
        raise PreregistrationError("provided checkpoint is not epoch5/iter37695")
    schedules = json.loads(SCHEDULE_RECEIPT.read_text(encoding="utf-8"))
    if schedules.get("status") != "complete_before_owner_training":
        raise PreregistrationError("schedule receipt status drifted")
    arms = (
        OWNERSHIP_SHARED_128,
        OWNERSHIP_SHARED_WIDE,
        OWNERSHIP_ISOLATED_128,
    )
    formal_targets = [
        FORMAL_ROOT / arm / f"seed{seed}" for arm in arms for seed in (17, 42, 73)
    ]
    existing = [str(path) for path in formal_targets if path.exists()]
    if existing:
        raise PreregistrationError(
            f"formal trajectory outputs already exist before lock: {existing}"
        )
    architecture = {
        arm: MMGDinoE5ResponsibilityOwners(
            ownership=arm
        ).architecture_report().as_dict()
        for arm in arms
    }
    calibration = (
        ROOT
        / "outputs/u2v5_leakage_clean_anchor_20260817/formal/"
        "calibration_u25_u50_u100/tn_eval_inputs/tn_screen_calibration.jsonl"
    )
    strict2031 = (
        ROOT
        / "outputs/u2v5_leakage_clean_anchor_20260817/final_once/"
        "strict2031_u50/tn_eval_inputs/tn_refcocop_val_refcocog_umd_val.jsonl"
    )
    annotations = Path(
        "/media/haoyi/T9/external/mmgdino_l_baseline/data/"
        "refcoco5e_coco/mdetr_annotations"
    )
    payload = {
        "schema": SCHEMA,
        "status": "locked_before_candidate_extraction_and_owner_training",
        "locked_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        "research_questions": [
            "Does shared ranking/rejection optimization still conflict when frozen query representations are strong?",
            "Does an isolated confidence owner improve TN rejection without degrading RefCOCO REC?",
        ],
        "frozen_candidate_generator": {
            "display_name": "MM-GDINO-T RefCOCO 5e",
            "checkpoint": _record(CHECKPOINT),
            "checkpoint_meta": {
                "epoch": meta.get("epoch"),
                "iter": meta.get("iter"),
                "seed": meta.get("seed"),
                "experiment_name": meta.get("experiment_name"),
            },
            "mmdetection_commit": MMDET_COMMIT,
            "eval_config": _record(EVAL_CONFIG),
            "frozen": True,
        },
        "evidence_status": {
            "trunk": "retrospective_provided_checkpoint_replay",
            "owner_transfer": "prospectively_frozen_after_this_receipt",
            "prior_test_exposure": "embedded trunk training evaluated RefCOCO val/testA/testB every epoch",
            "not_old_seed20260819_preregistration": True,
        },
        "matrix": {
            "native": {
                "trainable_parameters": 0,
                "macs_per_query_both_outputs": 0,
            },
            **architecture,
        },
        "training": {
            "arms": list(arms),
            "seeds": [17, 42, 73],
            "trajectories": 9,
            "updates": 150,
            "schedule": "rank,confidence,rank; R100+C50",
            "rank_batch_size": 32,
            "confidence_batch_size": 8,
            "rank_learning_rate": 3e-5,
            "confidence_learning_rate": 1e-4,
            "optimizer": "two task-specific AdamW optimizer states",
            "weight_decay": 0.0,
            "clip_norm": 0.1,
            "precision": "fp32 deterministic",
            "milestones_audit_only": [25, 50, 100, 150],
            "selected_update": 150,
        },
        "schedule_receipt": _record(SCHEDULE_RECEIPT),
        "evaluation_surfaces": {
            "refcoco_val": _record(annotations / "finetune_refcoco_val.json"),
            "refcoco_testA": _record(annotations / "finetune_refcoco_testA.json"),
            "refcoco_testB": _record(annotations / "finetune_refcoco_testB.json"),
            "d3_calibration": _record(calibration, rows=1570),
            "strict2031": _record(strict2031, rows=2031),
        },
        "statistics": {
            "bootstrap_replicates": 5000,
            "cluster": "image_id",
            "fpr95_recomputes_positive_q05_per_replicate": True,
            "refcoco_primary": "TestA+TestB pooled micro P@1",
            "strict_primary": "Strict2031 FPR95",
            "rec_noninferiority_margin": 0.005,
            "planned_contrasts": [
                "isolated_128-shared_128",
                "isolated_128-shared_wide",
                "isolated_128-native",
            ],
            "holm_family_size": 3,
        },
        "code": {
            name: _record(ROOT / path)
            for name, path in {
                "owners": "tools/mmgdino_e5_ownership.py",
                "trainer": "tools/train_mmgdino_e5_ownership.py",
                "extractor": "tools/extract_mmgdino_responsibility_cache.py",
                "schedule_builder": "tools/build_mmgdino_e5_ownership_schedules.py",
                "preregistration_builder": "tools/build_mmgdino_e5_ownership_preregistration.py",
            }.items()
        },
        "failed_attempt_ledger": [
            {
                "path": str(
                    ROOT
                    / "outputs/mmgdino_refcoco5e_strong_baseline_20260821/"
                    "eval_val/20260821_100225/20260821_100225.log"
                ),
                "status": "started_without_final_metric",
                "used_for_selection": False,
            }
        ],
        "formal_output_targets_absent": [str(path) for path in formal_targets],
        "prohibitions": [
            "do not change checkpoint, arm width, sample order, update count, loss, optimizer, or weight decay after an owner result",
            "do not select milestones per arm",
            "do not call the provided trunk or transfer a virgin held-out experiment",
            "do not add the e5 checkpoint to FineCops without a separate frozen evaluation",
        ],
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
