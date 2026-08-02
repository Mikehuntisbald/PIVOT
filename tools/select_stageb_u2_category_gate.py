#!/usr/bin/env python3
"""Select and seal the U2 category-preserving gate from canonical Ref val."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.stageb_eval_records import (
    RefRecords,
    load_formal_ref_records,
    sha256_file,
)
from tools.stageb_ref_split_contract import REF_SPLIT_CONTRACT, REF_SPLITS

SCHEMA = "pivot.stageb.u2_category_gate_selection/v2"
D9_DATA_ONLY_SCHEMA = "pivot.stageb.data_only_category_gate_selection/v1"
SWEEP_CONTRACT = "stageb-u2-category-gate-sweep-lexicographic-v1"
VAL_SPLITS = ("refcoco_val", "refcocop_val", "refcocog_val")
EXACT_GAPS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0)
TEACHER_GAP = 10.0
FORMAL_EVAL_RUNTIME = {
    "batch_size": 16,
    "num_workers": 4,
    "max_batches": 0,
}
CANONICAL_SEEDS = {
    split: 42 + 100000 * index for index, split in enumerate(REF_SPLITS)
}
SELECTION_POLICY = {
    "schema": "pivot.stageb.u2_category_gate_selection_policy/v1",
    "feasibility": [
        "each_val_split_gate_correct_strictly_greater_than_b58",
        "aggregate_gate_correct_strictly_greater_than_gap10_teacher",
        "at_least_one_val_split_gate_correct_strictly_greater_than_gap10_teacher",
    ],
    "objective_order": [
        "minimize_total_teacher_correct_to_gate_wrong",
        "maximize_minimum_split_correct_gain_over_b58",
        "maximize_aggregate_correct_gain_over_teacher",
        "maximize_max_gap",
    ],
}
D9_DATA_ONLY_SELECTION_POLICY = {
    "schema": "pivot.stageb.data_only_category_gate_selection_policy/v1",
    "feasibility": [
        "each_val_split_gate_correct_strictly_greater_than_sealed_u2",
        "aggregate_gate_correct_strictly_greater_than_gap10_r100",
        "at_least_one_val_split_gate_correct_strictly_greater_than_gap10_r100",
    ],
    "objective_order": [
        "minimize_gap10_r100_correct_to_gate_wrong",
        "maximize_minimum_split_correct_gain_over_sealed_u2",
        "maximize_aggregate_correct_gain_over_gap10_r100",
        "maximize_max_gap",
    ],
}


class SelectionError(ValueError):
    pass


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_record(path: Path) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve(strict=True)
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _read_summary(path: Path, *, label: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    path = Path(path).expanduser().resolve(strict=True)
    artifact = _file_record(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SelectionError(f"{label}: cannot read JSON: {error}") from error
    if not isinstance(payload, dict) or set(payload) != {"refcoco", "tn"}:
        raise SelectionError(f"{label}: expected exact refcoco/tn summary keys")
    if payload["tn"] != [] or not isinstance(payload["refcoco"], list):
        raise SelectionError(f"{label}: TN must be empty and refcoco must be a list")
    if _file_record(path) != artifact:
        raise SelectionError(f"{label}: summary changed while being read")
    return payload, artifact


def _exact_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SelectionError(f"{label}: expected an exact integer")
    return int(value)


def _exact_gap(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectionError(f"{label}: expected a numeric max_gap")
    gap = float(value)
    if not math.isfinite(gap) or gap not in EXACT_GAPS:
        raise SelectionError(f"{label}: max_gap is outside the sealed grid")
    return gap


def _resolve_record_path(summary_path: Path, reported: Any, *, label: str) -> Path:
    if not isinstance(reported, str) or not reported.strip():
        raise SelectionError(f"{label}: records_jsonl is missing")
    raw = Path(reported).expanduser()
    candidates = [raw] if raw.is_absolute() else [
        REPO_ROOT / raw,
        summary_path.parent / raw,
    ]
    existing = []
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            if resolved not in existing:
                existing.append(resolved)
    if len(existing) != 1:
        raise SelectionError(
            f"{label}: records_jsonl must resolve to exactly one file, found {existing}"
        )
    return existing[0]


def _validate_summary_row(
    row: Mapping[str, Any],
    *,
    split: str,
    seed: int,
    split_contract: Mapping[str, Mapping[str, Any]],
    label: str,
    evaluation_runtime: Optional[Mapping[str, int]] = None,
) -> None:
    official = split_contract[split]
    if row.get("dataset") != split:
        raise SelectionError(f"{label}: dataset drifted")
    if _exact_int(row.get("seed"), label=f"{label}.seed") != int(seed):
        raise SelectionError(f"{label}: canonical seed drifted")
    if evaluation_runtime is not None:
        for field, expected in evaluation_runtime.items():
            if _exact_int(row.get(field), label=f"{label}.{field}") != int(expected):
                raise SelectionError(
                    f"{label}: fixed evaluation runtime {field} drifted "
                    f"(expected {expected})"
                )
    if _exact_int(row.get("manifest_n"), label=f"{label}.manifest_n") != int(
        official["rows"]
    ):
        raise SelectionError(f"{label}: manifest row count drifted")
    if str(row.get("manifest_sha256", "")) != str(official["sha256"]):
        raise SelectionError(f"{label}: manifest SHA-256 drifted")
    if _exact_int(
        row.get("num_expressions"), label=f"{label}.num_expressions"
    ) != int(official["rows"]):
        raise SelectionError(f"{label}: incomplete expressions")
    if _exact_int(row.get("invalid_records", 0), label=f"{label}.invalid_records") != 0:
        raise SelectionError(f"{label}: invalid records are forbidden")
    if not isinstance(row.get("run_id"), str) or not row["run_id"]:
        raise SelectionError(f"{label}: run_id is missing")
    if not isinstance(row.get("records_jsonl"), str):
        raise SelectionError(f"{label}: records_jsonl is missing")


def _load_formal(
    summary_path: Path,
    row: Mapping[str, Any],
    *,
    split: str,
    split_contract: Mapping[str, Mapping[str, Any]],
    label: str,
) -> RefRecords:
    record_path = _resolve_record_path(
        summary_path, row.get("records_jsonl"), label=label
    )
    try:
        return load_formal_ref_records(
            str(record_path),
            base_dir=REPO_ROOT,
            label=label,
            split=split,
            summary_row=row,
            summary_path=summary_path,
            split_contract=split_contract,
        )
    except Exception as error:
        raise SelectionError(f"{label}: formal record validation failed: {error}") from error


def _read_sweep_extras(
    records: RefRecords,
    *,
    gap: float,
    expected_query_count: int,
    label: str,
) -> Dict[str, Tuple[int, ...]]:
    rows = []
    with records.path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SelectionError(f"{label}:{line_number}: invalid JSON") from error
            rows.append(row)
    if len(rows) != records.manifest_n:
        raise SelectionError(f"{label}: record count changed during extra replay")
    if sha256_file(records.path) != records.file_record["sha256"]:
        raise SelectionError(f"{label}: records changed during extra replay")
    fields = {
        "eligible": [],
        "winner": [],
        "teacher_winner": [],
        "patch_winner": [],
    }
    source_fields = {
        "eligible": "category_gate_eligible_queries",
        "winner": "category_gate_winner_query",
        "teacher_winner": "category_gate_teacher_winner_query",
        "patch_winner": "category_gate_patch_winner_query",
    }
    for index, row in enumerate(rows):
        location = f"{label} record {index}"
        if row.get("category_gate_sweep_contract") != SWEEP_CONTRACT:
            raise SelectionError(f"{location}: sweep contract drifted")
        if _exact_gap(
            row.get("category_gate_max_gap"), label=f"{location}.max_gap"
        ) != gap:
            raise SelectionError(f"{location}: max_gap drifted")
        for output_name, source_name in source_fields.items():
            value = _exact_int(row.get(source_name), label=f"{location}.{source_name}")
            upper = expected_query_count if output_name == "eligible" else expected_query_count - 1
            lower = 1 if output_name == "eligible" else 0
            if not lower <= value <= upper:
                raise SelectionError(f"{location}: {source_name} is out of range")
            fields[output_name].append(value)
    return {key: tuple(values) for key, values in fields.items()}


def _paired_counts(reference: Sequence[bool], candidate: Sequence[bool]) -> Dict[str, int]:
    if len(reference) != len(candidate):
        raise SelectionError("paired correctness arrays have different lengths")
    both_correct = both_wrong = gain = regression = 0
    for left, right in zip(reference, candidate):
        if left and right:
            both_correct += 1
        elif not left and not right:
            both_wrong += 1
        elif not left and right:
            gain += 1
        else:
            regression += 1
    return {
        "reference_correct_gate_correct": both_correct,
        "reference_wrong_gate_wrong": both_wrong,
        "reference_wrong_to_gate_correct": gain,
        "reference_correct_to_gate_wrong": regression,
        "net_correct_gain": gain - regression,
    }


def _artifact_unchanged(record: Mapping[str, Any], *, label: str) -> None:
    path = Path(str(record["path"])).resolve(strict=True)
    if int(path.stat().st_size) != int(record["size_bytes"]) or sha256_file(path) != str(
        record["sha256"]
    ):
        raise SelectionError(f"{label}: input artifact changed before sealing")


def build_selection_receipt(
    *,
    sweep_summary: Path,
    baseline_summary: Path,
    split_contract: Mapping[str, Mapping[str, Any]] = REF_SPLIT_CONTRACT,
    baseline_splits: Sequence[str] = REF_SPLITS,
    canonical_seeds: Mapping[str, int] = CANONICAL_SEEDS,
    expected_query_count: int = 900,
    receipt_schema: str = SCHEMA,
    selection_policy: Mapping[str, Any] = SELECTION_POLICY,
) -> Dict[str, Any]:
    sweep_summary = Path(sweep_summary).expanduser().resolve(strict=True)
    baseline_summary = Path(baseline_summary).expanduser().resolve(strict=True)
    sweep_payload, sweep_artifact = _read_summary(sweep_summary, label="sweep")
    baseline_payload, baseline_artifact = _read_summary(
        baseline_summary, label="baseline"
    )

    baseline_rows = baseline_payload["refcoco"]
    if len(baseline_rows) != len(baseline_splits):
        raise SelectionError("baseline summary is not the exact expected Ref split set")
    baseline_by_split: Dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(baseline_rows):
        if not isinstance(row, dict):
            raise SelectionError(f"baseline row {index} is not an object")
        split = str(row.get("dataset", ""))
        if split not in baseline_splits or split in baseline_by_split:
            raise SelectionError("baseline summary has duplicate or unexpected split input")
        _validate_summary_row(
            row,
            split=split,
            seed=int(canonical_seeds[split]),
            split_contract=split_contract,
            label=f"baseline.{split}",
            evaluation_runtime=FORMAL_EVAL_RUNTIME,
        )
        baseline_by_split[split] = row
    if set(baseline_by_split) != set(baseline_splits):
        raise SelectionError("baseline summary split coverage drifted")

    sweep_rows = sweep_payload["refcoco"]
    if len(sweep_rows) != len(VAL_SPLITS) * len(EXACT_GAPS):
        raise SelectionError("sweep summary must contain exactly 11 gaps x 3 val splits")
    sweep_by_key: Dict[Tuple[str, float], Mapping[str, Any]] = {}
    common_provenance: Optional[Tuple[Any, ...]] = None
    for index, row in enumerate(sweep_rows):
        if not isinstance(row, dict):
            raise SelectionError(f"sweep row {index} is not an object")
        split = str(row.get("dataset", ""))
        if split not in VAL_SPLITS:
            raise SelectionError(f"sweep row {index}: test/non-val input is forbidden")
        gap = _exact_gap(row.get("category_gate_max_gap"), label=f"sweep row {index}")
        key = (split, gap)
        if key in sweep_by_key:
            raise SelectionError(f"sweep row {index}: duplicate split/gap")
        _validate_summary_row(
            row,
            split=split,
            seed=int(canonical_seeds[split]),
            split_contract=split_contract,
            label=f"sweep.{split}.gap{gap:g}",
            evaluation_runtime=FORMAL_EVAL_RUNTIME,
        )
        if row.get("category_gate_sweep_contract") != SWEEP_CONTRACT:
            raise SelectionError(f"sweep row {index}: sweep contract drifted")
        if _exact_int(
            row.get("category_gate_single_forward_gap_count"),
            label=f"sweep row {index}.gap_count",
        ) != len(EXACT_GAPS):
            raise SelectionError(f"sweep row {index}: gap count drifted")
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
            raise SelectionError("sweep checkpoint/config provenance drifted across rows")
        sweep_by_key[key] = row
    if set(sweep_by_key) != {
        (split, gap) for split in VAL_SPLITS for gap in EXACT_GAPS
    }:
        raise SelectionError("sweep split/gap grid is incomplete")

    baseline_records: Dict[str, RefRecords] = {}
    sweep_records: Dict[Tuple[str, float], RefRecords] = {}
    sweep_extras: Dict[Tuple[str, float], Dict[str, Tuple[int, ...]]] = {}
    for split in VAL_SPLITS:
        baseline_records[split] = _load_formal(
            baseline_summary,
            baseline_by_split[split],
            split=split,
            split_contract=split_contract,
            label=f"baseline.{split}",
        )
        for gap in EXACT_GAPS:
            record = _load_formal(
                sweep_summary,
                sweep_by_key[(split, gap)],
                split=split,
                split_contract=split_contract,
                label=f"sweep.{split}.gap{gap:g}",
            )
            if record.identities != baseline_records[split].identities:
                raise SelectionError(f"{split} gap {gap:g}: baseline identity drift")
            sweep_records[(split, gap)] = record
            sweep_extras[(split, gap)] = _read_sweep_extras(
                record,
                gap=gap,
                expected_query_count=expected_query_count,
                label=f"sweep.{split}.gap{gap:g}",
            )

        teacher = sweep_extras[(split, TEACHER_GAP)]
        if any(value != expected_query_count for value in teacher["eligible"]):
            raise SelectionError(f"{split}: gap10 did not admit all queries")
        if teacher["winner"] != teacher["teacher_winner"]:
            raise SelectionError(f"{split}: gap10 is not the frozen teacher")
        previous = None
        for gap in EXACT_GAPS:
            extras = sweep_extras[(split, gap)]
            if extras["teacher_winner"] != teacher["teacher_winner"] or extras[
                "patch_winner"
            ] != teacher["patch_winner"]:
                raise SelectionError(f"{split}: single-forward winner provenance drifted")
            if previous is not None and any(
                current < before
                for before, current in zip(previous, extras["eligible"])
            ):
                raise SelectionError(f"{split}: eligible counts are not monotonic in gap")
            previous = extras["eligible"]

    gap_results = []
    teacher_total = sum(
        int(sweep_records[(split, TEACHER_GAP)].correct50.sum())
        for split in VAL_SPLITS
    )
    for gap in EXACT_GAPS:
        split_counts: Dict[str, Any] = {}
        aggregate_gate = 0
        aggregate_baseline = 0
        teacher_regressions = 0
        min_baseline_gain: Optional[int] = None
        any_split_above_teacher = False
        each_split_above_baseline = True
        for split in VAL_SPLITS:
            baseline_correct = baseline_records[split].correct50.tolist()
            teacher_correct = sweep_records[(split, TEACHER_GAP)].correct50.tolist()
            gate_correct = sweep_records[(split, gap)].correct50.tolist()
            versus_baseline = _paired_counts(baseline_correct, gate_correct)
            versus_teacher = _paired_counts(teacher_correct, gate_correct)
            baseline_count = int(sum(baseline_correct))
            teacher_count = int(sum(teacher_correct))
            gate_count = int(sum(gate_correct))
            aggregate_gate += gate_count
            aggregate_baseline += baseline_count
            teacher_regressions += versus_teacher[
                "reference_correct_to_gate_wrong"
            ]
            min_baseline_gain = (
                versus_baseline["net_correct_gain"]
                if min_baseline_gain is None
                else min(min_baseline_gain, versus_baseline["net_correct_gain"])
            )
            each_split_above_baseline &= gate_count > baseline_count
            any_split_above_teacher |= gate_count > teacher_count
            split_counts[split] = {
                "n": len(gate_correct),
                "baseline_correct": baseline_count,
                "teacher_correct": teacher_count,
                "gate_correct": gate_count,
                "versus_baseline": versus_baseline,
                "versus_teacher": versus_teacher,
            }
        teacher_net = aggregate_gate - teacher_total
        feasible = bool(
            each_split_above_baseline
            and aggregate_gate > teacher_total
            and any_split_above_teacher
        )
        gap_results.append(
            {
                "max_gap": gap,
                "split_counts": split_counts,
                "aggregate": {
                    "n": sum(int(split_contract[split]["rows"]) for split in VAL_SPLITS),
                    "baseline_correct": aggregate_baseline,
                    "teacher_correct": teacher_total,
                    "gate_correct": aggregate_gate,
                    "gate_minus_baseline": aggregate_gate - aggregate_baseline,
                    "gate_minus_teacher": teacher_net,
                    "teacher_correct_to_gate_wrong": teacher_regressions,
                    "minimum_split_gain_over_baseline": int(min_baseline_gain or 0),
                },
                "feasible": feasible,
                "selection_key": [
                    teacher_regressions,
                    -int(min_baseline_gain or 0),
                    -teacher_net,
                    -gap,
                ],
            }
        )
    feasible = [row for row in gap_results if row["feasible"]]
    if not feasible:
        raise SelectionError("no max_gap satisfies the sealed val feasibility gates")
    selected = min(feasible, key=lambda row: tuple(row["selection_key"]))

    record_artifacts = {
        "baseline": {
            split: dict(baseline_records[split].file_record) for split in VAL_SPLITS
        },
        "sweep": {
            split: {
                format(gap, "g"): dict(sweep_records[(split, gap)].file_record)
                for gap in EXACT_GAPS
            }
            for split in VAL_SPLITS
        },
    }
    for group_name, group in record_artifacts.items():
        if group_name == "baseline":
            for split, artifact in group.items():
                _artifact_unchanged(artifact, label=f"baseline.{split}")
        else:
            for split, by_gap in group.items():
                for gap, artifact in by_gap.items():
                    _artifact_unchanged(artifact, label=f"sweep.{split}.gap{gap}")
    _artifact_unchanged(sweep_artifact, label="sweep summary")
    _artifact_unchanged(baseline_artifact, label="baseline summary")

    receipt: Dict[str, Any] = {
        "schema": str(receipt_schema),
        "selection_frozen": True,
        "selection_policy": dict(selection_policy),
        "selection_policy_sha256": _canonical_sha256(selection_policy),
        "contract": {
            "exact_gaps": list(EXACT_GAPS),
            "teacher_gap": TEACHER_GAP,
            "val_splits": list(VAL_SPLITS),
            "evaluation_runtime": dict(FORMAL_EVAL_RUNTIME),
            "canonical_seeds": {
                split: int(canonical_seeds[split]) for split in VAL_SPLITS
            },
            "official_manifests": {
                split: dict(split_contract[split]) for split in VAL_SPLITS
            },
            "expected_query_count": int(expected_query_count),
            "sweep_contract": SWEEP_CONTRACT,
        },
        "inputs": {
            "sweep_summary": sweep_artifact,
            "baseline_summary": baseline_artifact,
            "records": record_artifacts,
        },
        "gap_results": gap_results,
        "feasible_gaps": [row["max_gap"] for row in feasible],
        "selection": {
            "max_gap": selected["max_gap"],
            "selection_key": selected["selection_key"],
            "aggregate": selected["aggregate"],
            "split_counts": selected["split_counts"],
        },
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
            raise SelectionError(f"refusing to overwrite a different sealed receipt: {path}")
        return path
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-summary", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--profile",
        choices=("u2", "data_only_d9"),
        default="u2",
        help=(
            "Selection semantics. data_only_d9 treats the sealed U2 val summary "
            "as the baseline and keeps the same canonical single-forward grid."
        ),
    )
    args = parser.parse_args()
    is_data_only = args.profile == "data_only_d9"
    receipt = build_selection_receipt(
        sweep_summary=Path(args.sweep_summary),
        baseline_summary=Path(args.baseline_summary),
        baseline_splits=VAL_SPLITS if is_data_only else REF_SPLITS,
        receipt_schema=D9_DATA_ONLY_SCHEMA if is_data_only else SCHEMA,
        selection_policy=(
            D9_DATA_ONLY_SELECTION_POLICY if is_data_only else SELECTION_POLICY
        ),
    )
    output = write_receipt(Path(args.output), receipt)
    print(
        json.dumps(
            {
                "output": str(output),
                "selected_max_gap": receipt["selection"]["max_gap"],
                "payload_sha256": receipt["payload_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
