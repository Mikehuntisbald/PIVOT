#!/usr/bin/env python3
"""Shared contract for the MM-GDINO e6 ownership 2x2 experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.mmgdino_e5_ownership import (
    OWNERSHIP_ISOLATED_128,
    OWNERSHIP_SHARED_WIDE,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "outputs/mmgdino_e6_ownership_2x2_20260822"
PREREGISTRATION = ROOT / "paper/data/mmgdino_e6_ownership_2x2_preregistration.json"
E5_REFERENCE = ROOT / "paper/data/mmgdino_e5_ownership_results.json"
SCHEDULE_ROOT = ROOT / "outputs/mmgdino_e5_ownership_transfer_20260821/schedules"
SCHEDULE_RECEIPT = SCHEDULE_ROOT / "schedule_receipt.json"

MMDET_ROOT = Path("/media/haoyi/T9/external/mmgdino_l_baseline/mmdetection")
MMDET_COMMIT = "cfd5d3a985b0249de009b67d04f37263e11cdf3d"
EVAL_CONFIG = Path(
    "/media/haoyi/T9/external/mmgdino_l_baseline/"
    "mmgdino_t_refcoco5e_formal_b4a8_seed20260819.py"
)
EVAL_CONFIG_SHA256 = "8f22719d7815563006de550e5f4f0173576bedfa8293922aa80c92e7524f2b82"

FORMAL_SEEDS = (17, 42, 73)
OWNERS = (OWNERSHIP_SHARED_WIDE, OWNERSHIP_ISOLATED_128)
BOOTSTRAP_SEED = 20260822
BOOTSTRAP_REPLICATES = 5000
REC_NONINFERIORITY_MARGIN = 0.005


@dataclass(frozen=True)
class TrunkSpec:
    trunk_id: str
    display_name: str
    checkpoint: Path
    checkpoint_sha256: str
    expected_epoch: int
    expected_iter: int
    expected_experiment_name: str
    training_surface: str

    @property
    def model_id(self) -> str:
        return f"mmgroundingdino-t-{self.trunk_id}-{self.checkpoint_sha256[:8]}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "trunk_id": self.trunk_id,
            "display_name": self.display_name,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "expected_epoch": self.expected_epoch,
            "expected_iter": self.expected_iter,
            "expected_experiment_name": self.expected_experiment_name,
            "training_surface": self.training_surface,
            "model_id": self.model_id,
        }


TRUNK_SPECS = {
    "e6_posctrl": TrunkSpec(
        trunk_id="e6_posctrl",
        display_name="e5→e6 PosCtrl",
        checkpoint=ROOT / "weights/epoch_6_postctrl.pth",
        checkpoint_sha256="08177fac668d62de99100b292ee5ff157366c33c48eb56b742006263a42022c3",
        expected_epoch=6,
        expected_iter=45234,
        expected_experiment_name="grounding_dino_swin-t_finetune_2x16_e6_from_e5_refcoco_20260822_023402",
        training_surface="positive RefCOCO continuation control",
    ),
    "e6_tn10": TrunkSpec(
        trunk_id="e6_tn10",
        display_name="e5→e6 TN10",
        checkpoint=ROOT / "weights/epoch_6_tn10.pth",
        checkpoint_sha256="a7078f1139c847d99e85221c8228f7cfd00e5be5ca0b85820f5d4d6a02cfa66c",
        expected_epoch=6,
        expected_iter=45988,
        expected_experiment_name="grounding_dino_swin-t_finetune_2x16_e6_from_e5_refcoco_tn10_20260822_050029",
        training_surface="RefCOCO continuation with 10 percent TN exposure",
    ),
}


REF_INPUTS = {
    "refcoco_testA": {
        "mode": "ref",
        "path": ROOT / "outputs/paper_cvpr_v1/baseline_b58_ref8_seed42/refcoco_eval_inputs/refcoco_unc_testA.jsonl",
        "sha256": "47278ef1043382235a151cd90d1e6c18c79d30bb71cb4eb7df1932abc622946e",
        "rows": 5657,
    },
    "refcoco_testB": {
        "mode": "ref",
        "path": ROOT / "outputs/paper_cvpr_v1/baseline_b58_ref8_seed42/refcoco_eval_inputs/refcoco_unc_testB.jsonl",
        "sha256": "41687648194225a693da5c42c5448eb1a9f4d2f59ca4cd138d4063d818116c8f",
        "rows": 5095,
    },
    "strict2031": {
        "mode": "tn",
        "path": ROOT / "outputs/u2v5_leakage_clean_anchor_20260817/final_once/strict2031_u50/tn_eval_inputs/tn_refcocop_val_refcocog_umd_val.jsonl",
        "sha256": "999c6f77f6affcf0f24a9fb17132e59e824bd012d53adf73f9b287dae454702c",
        "rows": 2031,
    },
}

IMAGE_ROOT = Path("/media/haoyi/T9/data/COCO/coco2014/train2014")


def training_cache_path(trunk_id: str, seed: int) -> Path:
    return EXPERIMENT_ROOT / f"caches/{trunk_id}/seed{seed}.pt"


def training_cache_receipt_path(trunk_id: str, seed: int) -> Path:
    return EXPERIMENT_ROOT / f"caches/{trunk_id}/seed{seed}_receipt.json"


def schedule_path(seed: int) -> Path:
    return SCHEDULE_ROOT / f"schedule_seed{seed}.json"


def owner_output_dir(trunk_id: str, owner: str, seed: int) -> Path:
    return EXPERIMENT_ROOT / f"formal/{trunk_id}/{owner}/seed{seed}"


def owner_checkpoint_path(trunk_id: str, owner: str, seed: int) -> Path:
    return owner_output_dir(trunk_id, owner, seed) / "checkpoint_u150.pt"


def eval_cache_path(trunk_id: str, surface: str) -> Path:
    return EXPERIMENT_ROOT / f"evaluation_caches/{trunk_id}/{surface}.pt"


def eval_cache_receipt_path(trunk_id: str, surface: str) -> Path:
    return EXPERIMENT_ROOT / f"evaluation_caches/{trunk_id}/{surface}_receipt.json"


def evaluation_output_dir(
    trunk_id: str, surface: str, route: str, seed: int | None
) -> Path:
    suffix = "native" if route == "native" else f"{route}/seed{seed}"
    return EXPERIMENT_ROOT / f"evaluation/{trunk_id}/{surface}/{suffix}"


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "E5_REFERENCE",
    "EVAL_CONFIG",
    "EVAL_CONFIG_SHA256",
    "EXPERIMENT_ROOT",
    "FORMAL_SEEDS",
    "IMAGE_ROOT",
    "MMDET_COMMIT",
    "MMDET_ROOT",
    "OWNERS",
    "PREREGISTRATION",
    "REC_NONINFERIORITY_MARGIN",
    "REF_INPUTS",
    "SCHEDULE_RECEIPT",
    "SCHEDULE_ROOT",
    "TRUNK_SPECS",
    "TrunkSpec",
    "eval_cache_path",
    "eval_cache_receipt_path",
    "evaluation_output_dir",
    "owner_checkpoint_path",
    "owner_output_dir",
    "schedule_path",
    "training_cache_path",
    "training_cache_receipt_path",
]
