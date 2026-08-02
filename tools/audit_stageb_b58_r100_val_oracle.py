#!/usr/bin/env python3
"""Seal the canonical-val row oracle between the b58 and frozen R100 experts.

This is an upper-bound audit, not a selector.  It loads only the three official
Ref val record streams.  The R100 stream is taken from the all-query (gap=10)
row of the already sealed U2 category-gate sweep and is rejected unless every
query was eligible and the emitted winner is exactly the frozen teacher winner.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.select_stageb_u2_category_gate import (
    CANONICAL_SEEDS,
    EXACT_GAPS,
    SWEEP_CONTRACT,
    TEACHER_GAP,
    VAL_SPLITS,
    SelectionError,
    _artifact_unchanged,
    _canonical_sha256,
    _exact_gap,
    _exact_int,
    _load_formal,
    _read_summary,
    _read_sweep_extras,
    _validate_summary_row,
)
from tools.stageb_eval_records import RefRecords
from tools.stageb_ref_split_contract import REF_SPLIT_CONTRACT, REF_SPLITS


SCHEMA = "pivot.stageb.b58_r100_val_row_oracle/v1"


class OracleAuditError(ValueError):
    pass


def _raise_from_selection(error: Exception) -> OracleAuditError:
    return OracleAuditError(str(error))


def _index_baseline_rows(
    payload: Mapping[str, Any],
    *,
    expected_splits: Sequence[str],
    val_splits: Sequence[str],
    split_contract: Mapping[str, Mapping[str, Any]],
    canonical_seeds: Mapping[str, int],
) -> Dict[str, Mapping[str, Any]]:
    rows = payload.get("refcoco")
    if not isinstance(rows, list) or len(rows) != len(expected_splits):
        raise OracleAuditError("baseline summary is not the exact expected Ref split set")
    indexed: Dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise OracleAuditError(f"baseline row {index} is not an object")
        split = str(row.get("dataset", ""))
        if split not in expected_splits or split in indexed:
            raise OracleAuditError("baseline summary has duplicate or unexpected splits")
        indexed[split] = row
    if set(indexed) != set(expected_splits):
        raise OracleAuditError("baseline summary split coverage drifted")
    for split in val_splits:
        try:
            _validate_summary_row(
                indexed[split],
                split=split,
                seed=int(canonical_seeds[split]),
                split_contract=split_contract,
                label=f"baseline.{split}",
            )
        except SelectionError as error:
            raise _raise_from_selection(error) from error
    return indexed


def _index_sweep_rows(
    payload: Mapping[str, Any],
    *,
    val_splits: Sequence[str],
    gaps: Sequence[float],
    split_contract: Mapping[str, Mapping[str, Any]],
    canonical_seeds: Mapping[str, int],
) -> Dict[Tuple[str, float], Mapping[str, Any]]:
    rows = payload.get("refcoco")
    expected_keys = {(split, float(gap)) for split in val_splits for gap in gaps}
    if not isinstance(rows, list) or len(rows) != len(expected_keys):
        raise OracleAuditError(
            "sweep summary must contain exactly the canonical val split/gap grid"
        )
    indexed: Dict[Tuple[str, float], Mapping[str, Any]] = {}
    common_provenance: Optional[Tuple[Any, ...]] = None
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise OracleAuditError(f"sweep row {index} is not an object")
        split = str(row.get("dataset", ""))
        if split not in val_splits:
            raise OracleAuditError(
                f"sweep row {index}: test/non-val input is forbidden"
            )
        try:
            gap = _exact_gap(row.get("category_gate_max_gap"), label=f"sweep row {index}")
        except SelectionError as error:
            raise _raise_from_selection(error) from error
        key = (split, gap)
        if key in indexed:
            raise OracleAuditError(f"sweep row {index}: duplicate split/gap")
        try:
            _validate_summary_row(
                row,
                split=split,
                seed=int(canonical_seeds[split]),
                split_contract=split_contract,
                label=f"sweep.{split}.gap{gap:g}",
            )
            if row.get("category_gate_sweep_contract") != SWEEP_CONTRACT:
                raise OracleAuditError(f"sweep row {index}: sweep contract drifted")
            if _exact_int(
                row.get("category_gate_single_forward_gap_count"),
                label=f"sweep row {index}.gap_count",
            ) != len(gaps):
                raise OracleAuditError(f"sweep row {index}: gap count drifted")
        except SelectionError as error:
            raise _raise_from_selection(error) from error
        provenance = tuple(
            row.get(field)
            for field in (
                "checkpoint",
                "checkpoint_sha256",
                "config",
                "config_sha256",
                "amp",
                "device",
            )
        )
        if common_provenance is None:
            common_provenance = provenance
        elif provenance != common_provenance:
            raise OracleAuditError(
                "sweep checkpoint/config provenance drifted across rows"
            )
        indexed[key] = row
    if set(indexed) != expected_keys:
        raise OracleAuditError("sweep split/gap grid is incomplete")
    return indexed


def _oracle_counts(base: Sequence[bool], r100: Sequence[bool]) -> Dict[str, Any]:
    if len(base) != len(r100):
        raise OracleAuditError("paired expert correctness arrays differ in length")
    both_correct = base_only = r100_only = both_wrong = 0
    for base_correct, r100_correct in zip(base, r100):
        if base_correct and r100_correct:
            both_correct += 1
        elif base_correct:
            base_only += 1
        elif r100_correct:
            r100_only += 1
        else:
            both_wrong += 1
    n = len(base)
    b58_correct = both_correct + base_only
    r100_correct = both_correct + r100_only
    oracle_correct = n - both_wrong
    return {
        "n": n,
        "b58_correct": b58_correct,
        "r100_correct": r100_correct,
        "both_correct": both_correct,
        "base_only": base_only,
        "r100_only": r100_only,
        "both_wrong": both_wrong,
        "oracle_correct": oracle_correct,
        "b58_acc50": b58_correct / n,
        "r100_acc50": r100_correct / n,
        "oracle_acc50": oracle_correct / n,
        "r100_minus_b58": r100_correct - b58_correct,
        "oracle_gain_over_b58": oracle_correct - b58_correct,
        "oracle_gain_over_r100": oracle_correct - r100_correct,
    }


def build_oracle_receipt(
    *,
    baseline_summary: Path,
    sweep_summary: Path,
    split_contract: Mapping[str, Mapping[str, Any]] = REF_SPLIT_CONTRACT,
    baseline_splits: Sequence[str] = REF_SPLITS,
    val_splits: Sequence[str] = VAL_SPLITS,
    gaps: Sequence[float] = EXACT_GAPS,
    canonical_seeds: Mapping[str, int] = CANONICAL_SEEDS,
    expected_query_count: int = 900,
) -> Dict[str, Any]:
    if tuple(val_splits) != tuple(VAL_SPLITS):
        raise OracleAuditError(
            "oracle audit requires exactly the three canonical Ref val splits"
        )
    if float(TEACHER_GAP) not in {float(gap) for gap in gaps}:
        raise OracleAuditError("sweep gap grid does not contain the R100 gap10 row")
    baseline_summary = Path(baseline_summary).expanduser().resolve(strict=True)
    sweep_summary = Path(sweep_summary).expanduser().resolve(strict=True)
    try:
        baseline_payload, baseline_artifact = _read_summary(
            baseline_summary, label="baseline"
        )
        sweep_payload, sweep_artifact = _read_summary(sweep_summary, label="sweep")
    except SelectionError as error:
        raise _raise_from_selection(error) from error

    baseline_by_split = _index_baseline_rows(
        baseline_payload,
        expected_splits=baseline_splits,
        val_splits=val_splits,
        split_contract=split_contract,
        canonical_seeds=canonical_seeds,
    )
    sweep_by_key = _index_sweep_rows(
        sweep_payload,
        val_splits=val_splits,
        gaps=gaps,
        split_contract=split_contract,
        canonical_seeds=canonical_seeds,
    )

    baseline_records: Dict[str, RefRecords] = {}
    r100_records: Dict[str, RefRecords] = {}
    split_results: Dict[str, Dict[str, Any]] = {}
    all_base: list[bool] = []
    all_r100: list[bool] = []
    identity_hashes: Dict[str, str] = {}
    for split in val_splits:
        try:
            baseline = _load_formal(
                baseline_summary,
                baseline_by_split[split],
                split=split,
                split_contract=split_contract,
                label=f"baseline.{split}",
            )
            teacher = _load_formal(
                sweep_summary,
                sweep_by_key[(split, float(TEACHER_GAP))],
                split=split,
                split_contract=split_contract,
                label=f"sweep.{split}.gap{TEACHER_GAP:g}",
            )
            extras = _read_sweep_extras(
                teacher,
                gap=float(TEACHER_GAP),
                expected_query_count=expected_query_count,
                label=f"sweep.{split}.gap{TEACHER_GAP:g}",
            )
        except SelectionError as error:
            raise _raise_from_selection(error) from error
        if baseline.identities != teacher.identities:
            raise OracleAuditError(f"{split}: b58/R100 record identities drifted")
        if any(value != expected_query_count for value in extras["eligible"]):
            raise OracleAuditError(f"{split}: gap10 did not admit all queries")
        if extras["winner"] != extras["teacher_winner"]:
            raise OracleAuditError(f"{split}: gap10 is not the frozen R100 teacher")

        base_correct = baseline.correct50.tolist()
        r100_correct = teacher.correct50.tolist()
        baseline_records[split] = baseline
        r100_records[split] = teacher
        split_results[split] = _oracle_counts(base_correct, r100_correct)
        all_base.extend(base_correct)
        all_r100.extend(r100_correct)
        identity_hashes[split] = _canonical_sha256(list(baseline.identities))

    aggregate = _oracle_counts(all_base, all_r100)
    record_artifacts = {
        "b58": {
            split: dict(baseline_records[split].file_record) for split in val_splits
        },
        "r100_gap10": {
            split: dict(r100_records[split].file_record) for split in val_splits
        },
    }
    try:
        for expert, by_split in record_artifacts.items():
            for split, artifact in by_split.items():
                _artifact_unchanged(artifact, label=f"{expert}.{split}")
        _artifact_unchanged(baseline_artifact, label="baseline summary")
        _artifact_unchanged(sweep_artifact, label="sweep summary")
    except SelectionError as error:
        raise _raise_from_selection(error) from error

    receipt: Dict[str, Any] = {
        "schema": SCHEMA,
        "audit_kind": "paired_row_expert_oracle_upper_bound",
        "not_a_selector": True,
        "test_records_loaded": False,
        "contract": {
            "val_splits": list(val_splits),
            "canonical_seeds": {
                split: int(canonical_seeds[split]) for split in val_splits
            },
            "official_manifests": {
                split: dict(split_contract[split]) for split in val_splits
            },
            "r100_source_gap": float(TEACHER_GAP),
            "expected_query_count": int(expected_query_count),
            "sweep_contract": SWEEP_CONTRACT,
            "correctness": "top1_iou >= 0.5",
            "oracle_rule": "correct iff b58 correct or R100 correct on the same row",
        },
        "inputs": {
            "baseline_summary": baseline_artifact,
            "sweep_summary": sweep_artifact,
            "records": record_artifacts,
        },
        "paired_identity_sha256": identity_hashes,
        "split_results": split_results,
        "aggregate": aggregate,
    }
    receipt["payload_sha256"] = _canonical_sha256(receipt)
    return receipt


def write_receipt(path: Path, receipt: Mapping[str, Any]) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        dict(receipt), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise OracleAuditError(
                f"refusing to overwrite a different sealed receipt: {path}"
            )
        return path
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--sweep-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = build_oracle_receipt(
        baseline_summary=Path(args.baseline_summary),
        sweep_summary=Path(args.sweep_summary),
    )
    output = write_receipt(Path(args.output), receipt)
    print(
        json.dumps(
            {
                "output": str(output),
                "aggregate": receipt["aggregate"],
                "payload_sha256": receipt["payload_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
