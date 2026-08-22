#!/usr/bin/env python3
"""Seal the B58 100k raw-query ownership replay result receipt."""

from __future__ import annotations

import contextlib
import json
import subprocess
from pathlib import Path
from typing import Any

import tools.build_original_gdino_parent_ownership_results as engine


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs/b58_raw_query_ownership_20260822"
DESTINATION = ROOT / "paper/data/b58_raw_query_ownership_results.json"
PREREG = ROOT / "paper/data/b58_raw_query_ownership_preregistration.json"
PARENT_RESULT = ROOT / "paper/data/original_gdino_parent_ownership_results.json"
TRUNK = "b58_raw_query"
OWNERS = ("shared_wide", "isolated_128")
SEEDS = (17, 42, 73)
SURFACES = (
    "refcoco_testA", "refcoco_testB", "refcocop_testA",
    "refcocop_testB", "refcocog_test", "strict2031",
)


class ReceiptError(engine.ReceiptError):
    pass


@contextlib.contextmanager
def _context():
    replacements = {
        "OUTPUT_ROOT": OUTPUT_ROOT,
        "DESTINATION": DESTINATION,
        "PREREG": PREREG,
        "TRUNK": TRUNK,
        "OWNERS": OWNERS,
        "SEEDS": SEEDS,
        "SURFACES": SURFACES,
    }
    previous = {name: getattr(engine, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(engine, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(engine, name, value)


def _artifact_audit(
    training_caches: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    evaluation_caches: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(training_caches) != 3 or len(trajectories) != 6:
        raise ReceiptError("formal artifact count drifted")
    rank_input = sum(row["counters"]["rank_input"] for row in training_caches)
    rank_kept = sum(row["counters"]["rank_kept"] for row in training_caches)
    rank_no_positive = sum(
        row["counters"]["rank_no_positive"] for row in training_caches
    )
    confidence_rows = sum(
        row["counters"]["confidence_rows"] for row in training_caches
    )
    if (
        rank_input != 9600
        or rank_kept != 9599
        or rank_no_positive != 1
        or confidence_rows != 2400
    ):
        raise ReceiptError("B58 training candidate accounting drifted")
    if any(
        row["updates"] != {
            "amp_skips": 0, "confidence": 50, "nonfinite": 0,
            "rank": 100, "total": 150,
        }
        for row in trajectories
    ):
        raise ReceiptError("formal endpoint drifted")
    ref_caches = [row for row in evaluation_caches if row["surface_or_seed"] != "strict2031"]
    if any(
        row["oracle"]["rows_with_iou50_candidate"]
        != row["oracle"]["total_rows"]
        for row in ref_caches
    ):
        raise ReceiptError("B58 Test5 all-query oracle is incomplete")
    return {
        "training_rank_rows_input": rank_input,
        "training_rank_rows_kept": rank_kept,
        "training_rank_rows_without_positive": rank_no_positive,
        "training_confidence_rows": confidence_rows,
        "test5_all_query_oracle_complete_on_every_split": True,
        "formal_trajectories": 6,
        "evaluation_routes": 42,
    }


def main() -> None:
    if DESTINATION.exists():
        raise ReceiptError("result receipt already exists")
    aggregate_path = OUTPUT_ROOT / "aggregate.json"
    aggregate = engine.load(aggregate_path)
    if (
        aggregate.get("schema") != "arrow.b58_raw_query_ownership.aggregate/v1"
        or aggregate.get("status") != "complete"
    ):
        raise ReceiptError("aggregate is incomplete")
    with _context():
        training_caches = engine.caches("training")
        trajectories = engine.trajectories()
        evaluation_caches = engine.caches("evaluation")
        evaluations = engine.evaluations()
    artifact_audit = _artifact_audit(
        training_caches, trajectories, evaluation_caches
    )
    contrast = aggregate["isolated_minus_shared_wide"]
    axis = aggregate["same_head_parent_to_b58"]
    payload = {
        "schema": "arrow.b58_raw_query_ownership.results/v1",
        "status": "complete_same_head_conditional_result",
        "provenance": {
            "git_commit_before_packaging": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
            ).strip(),
            "preregistration": engine.binding(PREREG),
            "aggregate": engine.binding(aggregate_path),
            "sealed_parent_result": engine.binding(PARENT_RESULT),
        },
        "matrix": {
            "trunk": TRUNK,
            "owners": list(OWNERS),
            "seeds": list(SEEDS),
            "fixed_endpoint": "U150",
            "shared_128_ran": False,
            "strict_same_100k_head_contract_as_parent": True,
        },
        "artifacts": {
            "audit": artifact_audit,
            "training_caches": training_caches,
            "formal_trajectories": trajectories,
            "evaluation_caches": evaluation_caches,
            "evaluation_routes": evaluations,
        },
        "statistics": {
            "bootstrap": aggregate["bootstrap"],
            "point_metrics": aggregate["point_metrics"],
            "isolated_minus_shared_wide": contrast,
            "gradient_probes": aggregate["gradient_probes"],
            "claim_gate": aggregate["claim_gate"],
            "same_head_parent_to_b58": axis,
        },
        "interpretation": {
            "supported": [
                "B58 makes Isolated REC-noninferior to Shared-Wide under the matched 100k owner contract",
                "Stage-B adaptation significantly narrows the parent isolation REC penalty on Test5 and TestAB",
                "B58 Shared-Wide retains frequent negative gradient events while hard isolation has no cross-task path",
            ],
            "not_supported": [
                "Isolated is superior to Shared-Wide on B58 REC",
                "Isolated has statistically superior Strict2031 FPR95 on B58",
                "the parent-to-B58 change in FPR ownership effect excludes zero",
                "negative gradient cosine alone selects the deployed ownership topology",
            ],
            "paper_conclusion": (
                "Mixed Stage-B adaptation changes ownership sensitivity: the "
                "same hard-isolation topology moves from a large REC penalty on "
                "the direct parent to practical non-inferiority on B58, while "
                "its rejection advantage remains statistically unresolved."
            ),
        },
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    engine.atomic(payload, DESTINATION)
    print(json.dumps(engine.binding(DESTINATION), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
