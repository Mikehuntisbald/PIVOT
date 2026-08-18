from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.build_arrow_release_manifest import (
    ArrowReleaseError,
    METHOD_LONG_NAME,
    build_manifest,
    file_record,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path):
    prereg = tmp_path / "prereg.json"
    _write_json(prereg, {
        "schema": "pivot.stageb.u2v5_ablation_preregistration/v1",
        "formal_trajectory_count": 42,
    })
    final = tmp_path / "final.json"
    _write_json(final, {
        "schema": "pivot.stageb.u2v5_ablation_final_receipt/v1",
        "preregistration": file_record(prereg),
    })
    tables = tmp_path / "tables.json"
    _write_json(tables, {"schema": "pivot.stageb.u2v5_ablation_paper_tables/v1"})
    supplement = tmp_path / "supplement.json"
    _write_json(supplement, {"schema": "pivot.stageb.u2v5_zero_training_supplement/v1"})
    checkpoints = tmp_path / "checkpoints"
    for seed in (17, 42, 73):
        path = checkpoints / f"confidence_seed{seed}_u50/checkpoint_iter.pth"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"seed={seed}".encode())
    return prereg, final, tables, supplement, checkpoints


def test_arrow_release_wraps_legacy_evidence_without_schema_rewrite(tmp_path):
    prereg, final, tables, supplement, checkpoints = _fixture(tmp_path)
    payload = build_manifest(
        preregistration=prereg,
        final_receipt=final,
        paper_tables=tables,
        supplement=supplement,
        checkpoint_root=checkpoints,
    )
    assert payload["schema"] == "arrow.release_manifest/v1"
    assert payload["method"] == {"name": "ARROW", "long_name": METHOD_LONG_NAME}
    assert payload["implementation"]["legacy_schema_namespace"] == "pivot.stageb"
    assert payload["formal_trajectory_count"] == 42
    assert set(payload["legacy_evidence"]["main_checkpoints"]) == {"17", "42", "73"}


def test_arrow_release_rejects_legacy_schema_drift(tmp_path):
    prereg, final, tables, supplement, checkpoints = _fixture(tmp_path)
    _write_json(prereg, {"schema": "arrow.rewritten.invalid/v1", "formal_trajectory_count": 42})
    with pytest.raises(ArrowReleaseError, match="required schema"):
        build_manifest(
            preregistration=prereg,
            final_receipt=final,
            paper_tables=tables,
            supplement=supplement,
            checkpoint_root=checkpoints,
        )


def test_arrow_is_the_public_readme_name_and_history_is_bannnered():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# ARROW\n")
    assert METHOD_LONG_NAME in " ".join(readme.split())
    historical = (
        "docs/paper_cvpr_ablation_protocol.md",
        "docs/paper_cvpr_stage_b_development_20260802.md",
        "docs/stage_b_decoupled_scoring_handoff_20260716.md",
        "docs/stage_b_gdino_semantic_confidence_probe.md",
        "docs/stageb_paper_evaluation_runbook.md",
        "docs/stageb_serial_matrix_queue.md",
    )
    for relative in historical:
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert text.startswith("> **Historical pre-ARROW artifact.**")
