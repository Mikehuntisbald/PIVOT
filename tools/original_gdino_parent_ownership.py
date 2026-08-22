#!/usr/bin/env python3
"""Contract for the pure GroundingDINO-T pre-Stage-B ownership replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.mmgdino_e5_ownership import (
    OWNERSHIP_ISOLATED_128,
    OWNERSHIP_SHARED_WIDE,
)
from tools.mmgdino_pretrain_ownership import REF_INPUTS


ROOT = _REPO_ROOT
TRUNK_ID = "original_parent"
EXPERIMENT_ROOT = ROOT / "outputs/original_gdino_parent_ownership_20260822"
PREREGISTRATION = (
    ROOT / "paper/data/original_gdino_parent_ownership_preregistration.json"
)

RELEASE_OGC = ROOT / "weights/groundingdino_swint_ogc.pth"
RELEASE_OGC_SHA256 = (
    "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799"
)
CHECKPOINT = Path(
    "/media/haoyi/T9/gdino/outputs/ogc_original_finetune_stage_a/"
    "checkpoint0001.pth"
)
CHECKPOINT_SHA256 = (
    "2aa2b20bd777d0f3ef955f0a0a6ddb9f7ea2efc6886891e321bfa8f2ed8b45de"
)
B58_CHECKPOINT = Path(
    "/media/haoyi/T9/gdino/outputs/"
    "gdino_ft_stageb_from_gdino_ft_e1_with_tn_bs19_nopatchbranch/"
    "checkpoint0001.pth"
)
B58_CHECKPOINT_SHA256 = (
    "b58e5209dc07dbffb2e5ed3d792e0db70c3306ec2ed24389693b4aeeebab1157"
)
EVAL_CONFIG = ROOT / "config/ablations/cfg_original_gdino_parent_ownership_eval.py"
EVAL_CONFIG_SHA256 = (
    "b0c29f8043905f1fe5d1a2267315ac2610df042e76e4563aafb574baf5aee5e7"
)
PURE_TRUNK_SCHEMA_SHA256 = (
    "b93506604856b80042f8db06edb612f2ab57df617175e20e8dbd7f3838e013b6"
)
PURE_TRUNK_TENSORS = 938
PURE_TRUNK_NUMEL = 174_327_226
PARENT_UNUSED_PATCH_TENSORS = 200
PARENT_TO_B58_CHANGED_TENSORS = 727
PARENT_TO_B58_UNCHANGED_TENSORS = 211

SCHEDULE_ROOT = ROOT / "outputs/mmgdino_e5_ownership_transfer_20260821/schedules"
SCHEDULE_RECEIPT = SCHEDULE_ROOT / "schedule_receipt.json"
IMAGE_ROOT = Path("/media/haoyi/T9/data/COCO/coco2014/train2014")
FORMAL_SEEDS = (17, 42, 73)
OWNERS = (OWNERSHIP_SHARED_WIDE, OWNERSHIP_ISOLATED_128)
BOOTSTRAP_SEED = 20260824
BOOTSTRAP_REPLICATES = 5000
REC_NONINFERIORITY_MARGIN = 0.005
TEST5_SURFACES = tuple(
    name for name, spec in REF_INPUTS.items() if spec["test5"]
)
TESTAB_SURFACES = tuple(
    name for name, spec in REF_INPUTS.items() if spec["testab"]
)

MMGDINO_PRETRAIN_REFERENCE = (
    ROOT / "paper/data/mmgdino_pretrain_ownership_results.json"
)
E5_REFERENCE = ROOT / "paper/data/mmgdino_e5_ownership_results.json"
B58_REFERENCE = ROOT / "paper/data/b58_capacity_control_results.json"


@dataclass(frozen=True)
class TrunkSpec:
    trunk_id: str = TRUNK_ID
    display_name: str = "original GroundingDINO-T pre-Stage-B parent"
    checkpoint: Path = CHECKPOINT
    checkpoint_sha256: str = CHECKPOINT_SHA256

    @property
    def model_id(self) -> str:
        return f"groundingdino-t-parent-{self.checkpoint_sha256[:8]}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "trunk_id": self.trunk_id,
            "display_name": self.display_name,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_id": self.model_id,
            "native_score": "full-expression token-probability mean",
            "training_surface": (
                "same-data positive GroundingDINO fine-tune before the mixed "
                "Stage-B continuation that produced B58"
            ),
        }


TRUNK_SPECS = {TRUNK_ID: TrunkSpec()}


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
    "BOOTSTRAP_REPLICATES", "BOOTSTRAP_SEED", "B58_CHECKPOINT",
    "B58_CHECKPOINT_SHA256", "B58_REFERENCE", "CHECKPOINT",
    "CHECKPOINT_SHA256", "E5_REFERENCE", "EVAL_CONFIG", "EVAL_CONFIG_SHA256",
    "EXPERIMENT_ROOT", "FORMAL_SEEDS", "IMAGE_ROOT",
    "MMGDINO_PRETRAIN_REFERENCE", "OWNERS", "PARENT_TO_B58_CHANGED_TENSORS",
    "PARENT_TO_B58_UNCHANGED_TENSORS", "PARENT_UNUSED_PATCH_TENSORS",
    "PREREGISTRATION", "PURE_TRUNK_NUMEL", "PURE_TRUNK_SCHEMA_SHA256",
    "PURE_TRUNK_TENSORS", "REC_NONINFERIORITY_MARGIN", "REF_INPUTS",
    "RELEASE_OGC", "RELEASE_OGC_SHA256", "ROOT", "SCHEDULE_RECEIPT",
    "SCHEDULE_ROOT", "TEST5_SURFACES", "TESTAB_SURFACES", "TRUNK_ID",
    "TRUNK_SPECS", "eval_cache_path", "eval_cache_receipt_path",
    "evaluation_output_dir", "owner_checkpoint_path", "owner_output_dir",
    "schedule_path", "training_cache_path", "training_cache_receipt_path",
]
