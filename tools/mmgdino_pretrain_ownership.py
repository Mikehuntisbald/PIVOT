#!/usr/bin/env python3
"""Contract for the frozen MM-GDINO-T pretrained ownership replay."""

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


ROOT = _REPO_ROOT
TRUNK_ID = "pretrained"
EXPERIMENT_ROOT = ROOT / "outputs/mmgdino_pretrain_ownership_20260822"
PREREGISTRATION = (
    ROOT / "paper/data/mmgdino_pretrain_ownership_preregistration.json"
)
E5_REFERENCE = ROOT / "paper/data/mmgdino_e5_ownership_results.json"
B58_REFERENCE = ROOT / "paper/data/b58_capacity_control_results.json"
SCHEDULE_ROOT = ROOT / "outputs/mmgdino_e5_ownership_transfer_20260821/schedules"
SCHEDULE_RECEIPT = SCHEDULE_ROOT / "schedule_receipt.json"

MMDET_ROOT = Path("/media/haoyi/T9/external/mmgdino_l_baseline/mmdetection")
MMDET_COMMIT = "cfd5d3a985b0249de009b67d04f37263e11cdf3d"
EVAL_CONFIG = Path(
    "/media/haoyi/T9/external/mmgdino_l_baseline/"
    "mmgdino_t_refcoco5e_formal_b4a8_seed20260819.py"
)
EVAL_CONFIG_SHA256 = (
    "8f22719d7815563006de550e5f4f0173576bedfa8293922aa80c92e7524f2b82"
)
CHECKPOINT = Path(
    "/media/haoyi/T9/external/mmgdino_l_baseline/weights/"
    "grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_"
    "20231204_095047-b448804b.pth"
)
CHECKPOINT_SHA256 = (
    "b448804bb1af6fa688887f0f2454625edbeeae4e868bc95620e3e6413581051a"
)
CHECKPOINT_META = {
    "epoch": 30,
    "iter": 483060,
    "experiment_name": (
        "grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_"
        "20231204_095047"
    ),
}

FORMAL_SEEDS = (17, 42, 73)
OWNERS = (OWNERSHIP_SHARED_WIDE, OWNERSHIP_ISOLATED_128)
BOOTSTRAP_SEED = 20260823
BOOTSTRAP_REPLICATES = 5000
REC_NONINFERIORITY_MARGIN = 0.005


@dataclass(frozen=True)
class TrunkSpec:
    trunk_id: str = TRUNK_ID
    display_name: str = "MM-GDINO-T pretrained"
    checkpoint: Path = CHECKPOINT
    checkpoint_sha256: str = CHECKPOINT_SHA256

    @property
    def model_id(self) -> str:
        return f"mmgroundingdino-t-pretrained-{self.checkpoint_sha256[:8]}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "trunk_id": self.trunk_id,
            "display_name": self.display_name,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_id": self.model_id,
            "training_surface": (
                "Objects365 + GoldG + GRIT-9M + V3Det pretraining; "
                "no RefCOCO task-specific fine-tuning"
            ),
        }


TRUNK_SPECS = {TRUNK_ID: TrunkSpec()}

_REF_ROOT = (
    ROOT
    / "outputs/paper_cvpr_v1/baseline_b58_ref8_seed42/refcoco_eval_inputs"
)
REF_INPUTS = {
    "refcoco_testA": {
        "mode": "ref",
        "path": _REF_ROOT / "refcoco_unc_testA.jsonl",
        "sha256": "47278ef1043382235a151cd90d1e6c18c79d30bb71cb4eb7df1932abc622946e",
        "rows": 5657,
        "test5": True,
        "testab": True,
    },
    "refcoco_testB": {
        "mode": "ref",
        "path": _REF_ROOT / "refcoco_unc_testB.jsonl",
        "sha256": "41687648194225a693da5c42c5448eb1a9f4d2f59ca4cd138d4063d818116c8f",
        "rows": 5095,
        "test5": True,
        "testab": True,
    },
    "refcocop_testA": {
        "mode": "ref",
        "path": _REF_ROOT / "refcocoplus_unc_testA.jsonl",
        "sha256": "57a0fb2342f120d49a1174084a7748cb18ff75a7b789bf2ddaf6c8555dce1105",
        "rows": 5726,
        "test5": True,
        "testab": False,
    },
    "refcocop_testB": {
        "mode": "ref",
        "path": _REF_ROOT / "refcocoplus_unc_testB.jsonl",
        "sha256": "49fe753d28a45cfb47f3d33cf5fbe34a1fda0ae111c7dcd24063c68e2b411d36",
        "rows": 4889,
        "test5": True,
        "testab": False,
    },
    "refcocog_test": {
        "mode": "ref",
        "path": _REF_ROOT / "refcocog_umd_test.jsonl",
        "sha256": "6c1c9bf2006344167bdce1859578faf83ca594383cc1acac62792c3e6a0f0a1d",
        "rows": 9602,
        "test5": True,
        "testab": False,
    },
    "strict2031": {
        "mode": "tn",
        "path": ROOT / (
            "outputs/u2v5_leakage_clean_anchor_20260817/final_once/"
            "strict2031_u50/tn_eval_inputs/"
            "tn_refcocop_val_refcocog_umd_val.jsonl"
        ),
        "sha256": "999c6f77f6affcf0f24a9fb17132e59e824bd012d53adf73f9b287dae454702c",
        "rows": 2031,
        "test5": False,
        "testab": False,
    },
}
TEST5_SURFACES = tuple(
    name for name, spec in REF_INPUTS.items() if spec["test5"]
)
TESTAB_SURFACES = tuple(
    name for name, spec in REF_INPUTS.items() if spec["testab"]
)
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
    "BOOTSTRAP_REPLICATES", "BOOTSTRAP_SEED", "B58_REFERENCE",
    "CHECKPOINT", "CHECKPOINT_META", "CHECKPOINT_SHA256", "E5_REFERENCE",
    "EVAL_CONFIG", "EVAL_CONFIG_SHA256", "EXPERIMENT_ROOT", "FORMAL_SEEDS",
    "IMAGE_ROOT", "MMDET_COMMIT", "MMDET_ROOT", "OWNERS", "PREREGISTRATION",
    "REC_NONINFERIORITY_MARGIN", "REF_INPUTS", "ROOT", "SCHEDULE_RECEIPT",
    "SCHEDULE_ROOT", "TEST5_SURFACES", "TESTAB_SURFACES", "TRUNK_ID",
    "TRUNK_SPECS", "eval_cache_path", "eval_cache_receipt_path",
    "evaluation_output_dir", "owner_checkpoint_path", "owner_output_dir",
    "schedule_path", "training_cache_path", "training_cache_receipt_path",
]
