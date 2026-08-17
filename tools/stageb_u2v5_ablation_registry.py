#!/usr/bin/env python3
"""Canonical registry for the leakage-clean U2-v5 CVPR ablation block."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 42, 73)
SCHEMA = "pivot.stageb.u2v5_ablation_registry/v1"
CLEAN_INITIALIZER = (
    ROOT
    / "outputs/u2v5_leakage_clean_anchor_20260817/initializer/"
    "checkpoint_clean_init.pth"
)
ADMISSION_DATASET = ROOT / "config/datasets_stageb_u2_category_complete_three_ref.json"
D3_DATASET = ROOT / "config/datasets_stageb_u2v5_clean_confidence_d3.json"


@dataclass(frozen=True)
class Row:
    row_id: str
    block: str
    phase: str
    config: str | None
    dataset: str | None
    updates: int
    batch_size: int
    parent: str
    trainable_roles: tuple[str, ...]
    allowed_changes: tuple[str, ...]
    formal_training: bool = True
    confirmatory_surfaces: tuple[str, ...] = ()

    def payload(self) -> dict:
        value = asdict(self)
        value["schema"] = SCHEMA
        return value


ROWS = (
    Row("A0", "A", "eval", None, None, 0, 0, "clean_initializer", (), (), False),
    Row("A1", "A", "admission", "config/ablations/cfg_stageb_u2v5_ablation_a1_surface_only.py", "config/datasets_stageb_u2_category_complete_three_ref.json", 100, 56, "clean_initializer", ("surface8",), ("trainable_roles",), confirmatory_surfaces=("test5",)),
    Row("A2", "A", "admission", "config/ablations/cfg_stageb_u2v5_ablation_a2_auxiliary_only.py", "config/datasets_stageb_u2_category_complete_three_ref.json", 100, 56, "clean_initializer", ("auxiliary8",), ("trainable_roles",)),
    Row("A3", "A", "admission", "config/ablations/cfg_stageb_u2v5_ablation_a3_no_preserve.py", "config/datasets_stageb_u2_category_complete_three_ref.json", 100, 56, "clean_initializer", ("surface8", "auxiliary8"), ("stage_b_u2_target_preserve_weight",)),
    Row("A4", "A", "admission", "config/ablations/cfg_stageb_u2v5_ablation_a4_no_category_complete.py", "config/datasets_stageb_u2_category_complete_three_ref.json", 100, 56, "clean_initializer", ("surface8", "auxiliary8"), ("stage_b_u2_category_complete_supervision", "stage_b_u2_category_loss_weight")),
    Row("A5", "A", "sealed", "config/ablations/cfg_stageb_u2v5_clean_admission_u100.py", "config/datasets_stageb_u2_category_complete_three_ref.json", 100, 56, "clean_initializer", ("surface8", "auxiliary8"), (), False),
    Row("C0", "C", "eval", None, None, 0, 0, "A5", (), (), False),
    Row("C1", "C", "confidence", "config/ablations/cfg_stageb_u2v5_ablation_c1_current_batch.py", "config/datasets_stageb_u2v5_clean_confidence_d3.json", 50, 8, "A5", ("confidence12",), ("stage_b_gdino_confidence_objective", "stage_b_gdino_queue_size", "stage_b_gdino_queue_min_count", "stage_b_gdino_positive_trust_weight")),
    Row("C2", "C", "confidence", "config/ablations/cfg_stageb_u2v5_ablation_c2_no_positive_trust.py", "config/datasets_stageb_u2v5_clean_confidence_d3.json", 50, 8, "A5", ("confidence12",), ("stage_b_gdino_positive_trust_weight",), confirmatory_surfaces=("strict2031",)),
    Row("C3", "C", "sealed", "config/ablations/cfg_stageb_u2v5_clean_confidence_d3_u100.py", "config/datasets_stageb_u2v5_clean_confidence_d3.json", 50, 8, "A5", ("confidence12",), (), False),
    Row("C4", "C", "confidence", "config/ablations/cfg_stageb_u2v5_ablation_c4_paired_margin.py", "config/datasets_stageb_u2v5_clean_confidence_d3.json", 50, 8, "A5", ("confidence12",), ("stage_b_gdino_paired_margin_weight",)),
    Row("D0", "D", "eval", None, None, 0, 0, "A5", (), (), False),
    Row("D1", "D", "confidence", "config/ablations/cfg_stageb_u2v5_ablation_d1_unverified.py", "config/datasets_stageb_u2v5_confidence_d1_paired.json", 50, 8, "A5", ("confidence12",), ("tn_source", "tn_scope")),
    Row("D2", "D", "confidence", "config/ablations/cfg_stageb_u2v5_ablation_d2_traceable.py", "config/datasets_stageb_u2v5_confidence_d2_paired.json", 50, 8, "A5", ("confidence12",), ("tn_source", "tn_scope")),
    Row("D3", "D", "sealed", "config/ablations/cfg_stageb_u2v5_clean_confidence_d3_u100.py", "config/datasets_stageb_u2v5_clean_confidence_d3.json", 50, 8, "A5", ("confidence12",), (), False),
    Row("D2m", "D", "confidence", "config/ablations/cfg_stageb_u2v5_ablation_d2m_matched.py", "config/datasets_stageb_u2v5_confidence_d2m_paired.json", 50, 8, "A5", ("confidence12",), ("tn_source", "tn_scope", "matched_panel"), confirmatory_surfaces=("strict2031",)),
    Row("D3m", "D", "confidence", "config/ablations/cfg_stageb_u2v5_ablation_d3m_matched.py", "config/datasets_stageb_u2v5_confidence_d3m_paired.json", 50, 8, "A5", ("confidence12",), ("tn_source", "tn_scope", "matched_panel"), confirmatory_surfaces=("strict2031",)),
    Row("O0", "O", "ownership", "config/ablations/cfg_stageb_u2v5_ablation_o0_shared_score.py", "config/datasets_stageb_u2v5_ownership_interleaved.json", 150, 56, "clean_initializer", ("shared_rank8", "surface8", "auxiliary8"), ("ownership", "schedule"), confirmatory_surfaces=("test5", "strict2031")),
    Row("O1", "O", "ownership", "config/ablations/cfg_stageb_u2v5_ablation_o1_shared_trunk.py", "config/datasets_stageb_u2v5_ownership_interleaved.json", 150, 56, "clean_initializer", ("shared_rank8", "confidence_output6", "surface8", "auxiliary8"), ("ownership", "schedule")),
    Row("O2", "O", "ownership", "config/ablations/cfg_stageb_u2v5_ablation_o2_isolated_interleaved.py", "config/datasets_stageb_u2v5_ownership_interleaved.json", 150, 56, "clean_initializer", ("confidence12", "surface8", "auxiliary8"), ("schedule",), confirmatory_surfaces=("test5", "strict2031")),
    Row("O3", "O", "sealed", None, None, 150, 56, "clean_anchor", ("confidence12", "surface8", "auxiliary8"), (), False, confirmatory_surfaces=("test5", "strict2031")),
)

ROW_BY_ID = {row.row_id: row for row in ROWS}
FORMAL_ROWS = tuple(row for row in ROWS if row.formal_training)


def get_row(row_id: str) -> Row:
    key = str(row_id).strip()
    if key not in ROW_BY_ID:
        raise KeyError(f"unknown U2-v5 ablation row {key!r}")
    return ROW_BY_ID[key]


def parse_run_id(value: str) -> tuple[Row, int]:
    try:
        row_id, raw_seed = str(value).split(":", 1)
        seed = int(raw_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("run id must have form ROW:SEED") from exc
    row = get_row(row_id)
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}, got {seed}")
    if not row.formal_training:
        raise ValueError(f"row {row.row_id} is sealed/eval-only, not trainable")
    return row, seed


def validate_registry(rows: Iterable[Row] = ROWS) -> None:
    rows = tuple(rows)
    ids = [row.row_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate U2-v5 ablation row IDs")
    formal = [row for row in rows if row.formal_training]
    if len(formal) != 14 or len(formal) * len(SEEDS) != 42:
        raise ValueError("formal U2-v5 registry must define exactly 42 trajectories")
    for row in formal:
        if row.config is None or row.dataset is None:
            raise ValueError(f"formal row {row.row_id} lacks config/dataset")
        if row.updates <= 0 or row.batch_size <= 0 or not row.trainable_roles:
            raise ValueError(f"formal row {row.row_id} has invalid runtime contract")
        for path in (row.config, row.dataset):
            if not (ROOT / path).is_file():
                raise FileNotFoundError(f"row {row.row_id} input is missing: {path}")


__all__ = ["FORMAL_ROWS", "ROW_BY_ID", "ROWS", "ROOT", "SCHEMA", "SEEDS", "Row", "get_row", "parse_run_id", "validate_registry"]
