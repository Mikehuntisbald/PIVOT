#!/usr/bin/env python3
"""Contracts shared by the original GroundingDINO-T OGC replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from tools.arrow_finecops_common import file_record, load_json


PREREG_SCHEMA = "arrow.original_gdino_ogc.finecops_preregistration/v1"
RECORD_SCHEMA = "arrow.original_gdino_ogc.finecops_record/v1"
RESULTS_SCHEMA = "arrow.original_gdino_ogc.finecops_results/v1"
RUN_SCHEMA = "arrow.original_gdino_ogc.finecops_run_receipt/v1"

CHECKPOINT_SHA256 = (
    "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799"
)
CHECKPOINT_SIZE = 693_997_677
CHECKPOINT_TENSORS = 940
CHECKPOINT_NUMEL = 174_839_994
EXPECTED_UNUSED_CHECKPOINT_KEYS = (
    "bert.embeddings.position_ids",
    "label_enc.weight",
)

EXPECTED_FINECOPS_COUNTS = {
    "positive_total": 9605,
    "text_total": 9814,
    "image_total": 8507,
}

PRIMARY_SCORE = "expression_mean"
SENSITIVITY_SCORE = "expression_max"


def verify_file(expected: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(expected.get("path", ""))).expanduser().resolve(strict=True)
    observed = file_record(path)
    for key in ("sha256", "size_bytes"):
        if observed[key] != expected.get(key):
            raise ValueError(f"{label} {key} drifted")
    return path


def load_preregistration(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("schema") != PREREG_SCHEMA:
        raise ValueError("original OGC preregistration schema drifted")
    if payload.get("status") != "locked_before_original_ogc_forward":
        raise ValueError("original OGC preregistration is not locked")
    if payload.get("primary_score") != PRIMARY_SCORE:
        raise ValueError("original OGC primary score drifted")
    if payload.get("sensitivity_score") != SENSITIVITY_SCORE:
        raise ValueError("original OGC sensitivity score drifted")
    return payload
