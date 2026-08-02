#!/usr/bin/env python3
"""Aggregate the supplemental, parent-matched Table-B D2m/D3m panel.

This report is intentionally separate from the paper's global-TN tables.  It
accepts only the proposal-covered calibration surface sealed by the v2 matched
panel audit and labels every resulting number as a diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_stageb_tn_matched_causal_panel import (  # noqa: E402
    MatchedPanelError,
    PAIR_SCHEMA,
    PRIMARY_CAUSAL_STRATUM,
    SCHEMA as AUDIT_SCHEMA,
    verify_panel,
)
from tools.compare_stageb_fpr95_records import (  # noqa: E402
    RecordComparisonError,
    exact_binary_auroc,
    exact_fpr95,
    load_manifest,
    load_tn_records,
)


REPORT_SCHEMA = "stage-b-table-b-matched-panel-report-v1"
FORMAL_REPORT_SCHEMA = "stage-b-table-b-matched-panel-formal-report-v1"
FORMAL_SEEDS = (17, 42, 73)
DECLARED_SURFACE = "proposal_covered_verified"
SPLIT = "calibration"
RELATIONS = ("identical", "different")
REPORT_STRATA = (*RELATIONS, PRIMARY_CAUSAL_STRATUM)
METRIC_LABEL = "proposal-covered matched-panel diagnostic"
PROVENANCE_SCHEMA = "stage-b-table-b-matched-eval-provenance-v1"
SCALAR_METRICS = (
    "positive_q05",
    "auroc",
    "fpr95_like_diagnostic",
    "positive_over_negative_win_rate",
    "positive_equals_negative_tie_rate",
    "positive_score_mean",
    "negative_score_mean",
    "score_gap_mean",
)


class MatchedPanelReportError(ValueError):
    """Raised when the matched-panel evidence cannot prove its contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    path = Path(path).expanduser().resolve(strict=True)
    result: dict[str, Any] = {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MatchedPanelReportError(f"{label}: invalid JSON: {error}") from error
    if not isinstance(value, Mapping):
        raise MatchedPanelReportError(f"{label}: expected a JSON object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise MatchedPanelReportError(f"{label}: invalid JSONL: {error}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise MatchedPanelReportError(
                f"{label}:{line_number}: blank rows are not permitted"
            )
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise MatchedPanelReportError(
                f"{label}:{line_number}: invalid JSON: {error}"
            ) from error
        if not isinstance(row, dict):
            raise MatchedPanelReportError(
                f"{label}:{line_number}: expected a JSON object"
            )
        rows.append(row)
    if not rows:
        raise MatchedPanelReportError(f"{label}: empty JSONL")
    return rows


def _exact_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise MatchedPanelReportError(
            f"{label}: expected an exact integer >= {minimum}"
        )
    return value


def _exact_bool(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise MatchedPanelReportError(f"{label}: expected an exact boolean")
    return value


def _hash_string(value: Any, *, label: str) -> str:
    rendered = str(value or "")
    if len(rendered) != 64 or any(ch not in "0123456789abcdef" for ch in rendered):
        raise MatchedPanelReportError(
            f"{label}: expected 64 lowercase hexadecimal characters"
        )
    return rendered


def _resolve_audit_path(value: Any, *, audit_path: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise MatchedPanelReportError(f"{label}: audit path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = audit_path.parent / path
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise MatchedPanelReportError(f"{label}: audited file is missing: {path}") from error


def _verify_audited_artifact(
    audit: Mapping[str, Any],
    *,
    audit_path: Path,
    key: str,
    supplied_path: Path,
) -> dict[str, Any]:
    outputs = audit.get("outputs")
    specification = outputs.get(key) if isinstance(outputs, Mapping) else None
    if not isinstance(specification, Mapping):
        raise MatchedPanelReportError(f"audit.outputs.{key}: missing artifact record")
    audited_path = _resolve_audit_path(
        specification.get("path"), audit_path=audit_path, label=f"audit.outputs.{key}"
    )
    supplied = Path(supplied_path).expanduser().resolve(strict=True)
    if supplied != audited_path:
        raise MatchedPanelReportError(
            f"{key}: supplied path is not the path sealed by the audit"
        )
    rows = _exact_int(specification.get("rows"), label=f"audit.outputs.{key}.rows", minimum=1)
    record = _file_record(supplied, rows=rows)
    expected_sha = _hash_string(
        specification.get("sha256"), label=f"audit.outputs.{key}.sha256"
    )
    if record["sha256"] != expected_sha:
        raise MatchedPanelReportError(f"{key}: SHA-256 drift from matched-panel audit")
    expected_size = _exact_int(
        specification.get("size_bytes"),
        label=f"audit.outputs.{key}.size_bytes",
    )
    if record["size_bytes"] != expected_size:
        raise MatchedPanelReportError(f"{key}: byte-size drift from matched-panel audit")
    return record


def _verify_named_audited_artifact(
    audit: Mapping[str, Any], *, audit_path: Path, key: str
) -> dict[str, Any]:
    outputs = audit.get("outputs")
    specification = outputs.get(key) if isinstance(outputs, Mapping) else None
    if not isinstance(specification, Mapping):
        raise MatchedPanelReportError(f"audit.outputs.{key}: missing artifact record")
    path = _resolve_audit_path(
        specification.get("path"), audit_path=audit_path, label=f"audit.outputs.{key}"
    )
    return _verify_audited_artifact(
        audit, audit_path=audit_path, key=key, supplied_path=path
    )


def _canonical_parent_hash(parent: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(parent), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _validate_audit(audit: Mapping[str, Any]) -> None:
    if audit.get("schema") != AUDIT_SCHEMA:
        raise MatchedPanelReportError(
            f"matched-panel audit schema must be exactly {AUDIT_SCHEMA!r}"
        )
    if audit.get("kind") != "completed_c2_parent_matched_tn_panel":
        raise MatchedPanelReportError("matched-panel audit kind is not sealed")
    runtime = audit.get("runtime_contract")
    if not isinstance(runtime, Mapping) or runtime.get(
        "D2m_D3m_supported_by_current_v24"
    ) is not True:
        raise MatchedPanelReportError("matched-panel runtime contract is incomplete")
    invariants = audit.get("invariants")
    split_invariants = (
        invariants.get(SPLIT) if isinstance(invariants, Mapping) else None
    )
    if not isinstance(split_invariants, Mapping):
        raise MatchedPanelReportError("audit lacks calibration invariants")
    for key in (
        "equal_rows",
        "aligned_pair_ids",
        "aligned_parent_key_sha256",
        "positive_phrase_normalized_match",
        "unique_pair_ids",
        "unique_parent_keys",
        "negative_text_relation_is_partition",
    ):
        if split_invariants.get(key) is not True:
            raise MatchedPanelReportError(f"audit calibration invariant {key} failed")
    if split_invariants.get("runtime_global_verified_true_rows") != 0:
        raise MatchedPanelReportError("audit upgrades matched rows to global TN")
    for key in (
        "strict_union_image_overlap",
        "train_calibration_image_overlap",
        "train_calibration_pair_id_overlap",
    ):
        if invariants.get(key) != 0:
            raise MatchedPanelReportError(f"audit invariant {key} must be zero")
    scopes = audit.get("scope_contract")
    expected = {
        "D2m": ("traceable_counterfactual_edit", False),
        "D3m": (DECLARED_SURFACE, False),
    }
    if not isinstance(scopes, Mapping):
        raise MatchedPanelReportError("audit scope contract is missing")
    for table_id, (scope, global_verified) in expected.items():
        value = scopes.get(table_id)
        if not isinstance(value, Mapping):
            raise MatchedPanelReportError(f"audit scope contract lacks {table_id}")
        if value.get("tn_scope") != scope or value.get("global_tn_verified") is not global_verified:
            raise MatchedPanelReportError(f"audit scope contract upgrades or relabels {table_id}")


def _population_contract(audit: Mapping[str, Any]) -> dict[str, Any]:
    expected_claim_scope = {
        "pairwise_effect_population": "matched_pairs_only",
        "primary_causal_stratum": PRIMARY_CAUSAL_STRATUM,
        "primary_causal_stratum_requires_exact_model_input": True,
        "canonical_class_id_equality_required": True,
        "unmatched_d3_parent_rows_are_out_of_scope": True,
        "generalization_to_unmatched_d3_parent_rows_supported": False,
    }
    if audit.get("claim_scope") != expected_claim_scope:
        raise MatchedPanelReportError("matched-panel claim_scope is incomplete")
    yields = audit.get("matching_yield")
    value = yields.get(SPLIT) if isinstance(yields, Mapping) else None
    if not isinstance(value, Mapping):
        raise MatchedPanelReportError("audit lacks calibration matching_yield")
    parent_n = _exact_int(
        value.get("d3_parent_rows"), label="matching_yield.calibration.d3_parent_rows", minimum=1
    )
    matched_n = _exact_int(
        value.get("matched_pairs"), label="matching_yield.calibration.matched_pairs", minimum=1
    )
    unmatched_n = _exact_int(
        value.get("unmatched_d3_parent_rows"),
        label="matching_yield.calibration.unmatched_d3_parent_rows",
    )
    primary_n = _exact_int(
        value.get("class_aligned_identical_claim_denominator"),
        label="matching_yield.calibration.class_aligned_identical_claim_denominator",
        minimum=1,
    )
    if parent_n != matched_n + unmatched_n:
        raise MatchedPanelReportError("calibration matched/unmatched population is not exact")
    return {
        "d3_parent_rows": parent_n,
        "matched_pairs": matched_n,
        "unmatched_d3_parent_rows": unmatched_n,
        "matched_fraction": float(value.get("matched_fraction")),
        "unmatched_fraction": float(value.get("unmatched_fraction")),
        "matched_pairwise_claim_denominator": _exact_int(
            value.get("matched_pairwise_claim_denominator"),
            label="matching_yield.calibration.matched_pairwise_claim_denominator",
            minimum=1,
        ),
        "class_aligned_identical_claim_denominator": primary_n,
        "pairwise_population_is_matched_only": True,
        "unmatched_rows_excluded_from_pairwise_metrics": True,
        "generalization_to_unmatched_rows_supported": False,
    }


def _shared_pair_value(row: Mapping[str, Any], field: str, *, label: str) -> Any:
    if field not in row:
        raise MatchedPanelReportError(f"{label}: missing {field}")
    return row[field]


def _validate_panel_rows(
    *,
    audit: Mapping[str, Any],
    ledger: Sequence[Mapping[str, Any]],
    d2_source: Sequence[Mapping[str, Any]],
    d3_source: Sequence[Mapping[str, Any]],
    surface: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sizes = {len(ledger), len(d2_source), len(d3_source), len(surface)}
    if len(sizes) != 1:
        raise MatchedPanelReportError(
            "ledger, D2m source, D3m source, and evaluation manifest row counts differ"
        )
    pair_ids: set[str] = set()
    parent_hashes: set[str] = set()
    d2_ids: set[str] = set()
    d3_ids: set[str] = set()
    metadata: list[dict[str, Any]] = []
    mismatch_directions: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    class_relation_counts: Counter[str] = Counter()
    causal_counts: Counter[str] = Counter()
    component_mismatches: Counter[str] = Counter()
    complete_by_relation: Counter[str] = Counter()
    mismatch_by_relation: Counter[str] = Counter()

    shared_fields = (
        "matched_pair_schema",
        "matched_pair_id",
        "matched_split",
        "matched_parent_key",
        "matched_parent_key_sha256",
        "matched_stratum",
        "positive_phrase_exact_match",
        "positive_phrase_normalized_match",
        "negative_text_exact_match",
        "negative_text_normalized_match",
        "d2_canonical_class_id",
        "d3_canonical_class_id",
        "canonical_class_id_match",
        "model_input_component_exact_matches",
        "base_parent_input_exact_match",
        "complete_model_input_exact_match",
        "class_aligned_identical_complete_input",
    )
    for index, (pair, d2, d3, manifest_row) in enumerate(
        zip(ledger, d2_source, d3_source, surface)
    ):
        location = f"matched row {index}"
        if pair.get("matched_pair_schema") != PAIR_SCHEMA:
            raise MatchedPanelReportError(f"{location}: wrong matched pair schema")
        if pair.get("matched_split") != SPLIT:
            raise MatchedPanelReportError(f"{location}: not a calibration pair")
        pair_id = str(pair.get("matched_pair_id") or "")
        if not pair_id or pair_id in pair_ids:
            raise MatchedPanelReportError(f"{location}: missing/duplicate matched_pair_id")
        pair_ids.add(pair_id)
        parent = pair.get("matched_parent_key")
        if not isinstance(parent, Mapping):
            raise MatchedPanelReportError(f"{location}: missing matched parent key")
        parent_hash = _hash_string(
            pair.get("matched_parent_key_sha256"),
            label=f"{location}.matched_parent_key_sha256",
        )
        if parent_hash != _canonical_parent_hash(parent):
            raise MatchedPanelReportError(f"{location}: parent-key hash mismatch")
        if parent_hash in parent_hashes:
            raise MatchedPanelReportError(f"{location}: duplicate parent-key hash")
        parent_hashes.add(parent_hash)
        stratum = pair.get("matched_stratum")
        if not isinstance(stratum, Mapping):
            raise MatchedPanelReportError(f"{location}: missing matched stratum")
        relation = str(stratum.get("negative_text_relation") or "")
        if relation not in RELATIONS:
            raise MatchedPanelReportError(
                f"{location}: negative-text relation does not partition the panel"
            )
        relation_counts[relation] += 1
        canonical_match = _exact_bool(
            pair.get("canonical_class_id_match"),
            label=f"{location}.canonical_class_id_match",
        )
        d2_class = _exact_int(
            pair.get("d2_canonical_class_id"), label=f"{location}.d2 class"
        )
        d3_class = _exact_int(
            pair.get("d3_canonical_class_id"), label=f"{location}.d3 class"
        )
        if canonical_match != (d2_class == d3_class):
            raise MatchedPanelReportError(f"{location}: canonical class-match contradiction")
        expected_class_relation = "aligned" if canonical_match else "mismatched"
        if stratum.get("canonical_class_relation") != expected_class_relation:
            raise MatchedPanelReportError(f"{location}: canonical class stratum mismatch")
        class_relation_counts[expected_class_relation] += 1
        if not canonical_match:
            mismatch_by_relation[relation] += 1
            mismatch_directions[f"{d2_class}->{d3_class}"] += 1
        causal_relation = str(stratum.get("causal_input_relation") or "")
        if not causal_relation:
            raise MatchedPanelReportError(f"{location}: missing causal input relation")
        causal_counts[causal_relation] += 1
        component_matches = pair.get("model_input_component_exact_matches")
        if not isinstance(component_matches, Mapping) or set(component_matches) != {
            "image_id",
            "file_name",
            "target_bbox_used",
            "positive_phrase",
            "negative_text",
            "canonical_class_id",
        }:
            raise MatchedPanelReportError(f"{location}: incomplete input-component audit")
        for component, value in component_matches.items():
            if type(value) is not bool:
                raise MatchedPanelReportError(
                    f"{location}: component match {component} is not boolean"
                )
            if not value:
                component_mismatches[str(component)] += 1
        expected_base_parent_match = all(
            component_matches[field]
            for field in (
                "image_id",
                "file_name",
                "target_bbox_used",
                "positive_phrase",
            )
        )
        base_parent_match = _exact_bool(
            pair.get("base_parent_input_exact_match"),
            label=f"{location}.base_parent_input_exact_match",
        )
        if base_parent_match != expected_base_parent_match:
            raise MatchedPanelReportError(f"{location}: base-parent flag contradiction")
        if pair.get("negative_text_exact_match") is not component_matches[
            "negative_text"
        ]:
            raise MatchedPanelReportError(f"{location}: exact negative-text flag contradiction")
        if canonical_match is not component_matches["canonical_class_id"]:
            raise MatchedPanelReportError(f"{location}: component class-match contradiction")
        if not base_parent_match:
            expected_causal_relation = "other_parent_input_mismatch"
        elif component_matches["negative_text"]:
            expected_causal_relation = (
                PRIMARY_CAUSAL_STRATUM
                if canonical_match
                else "identical_text_class_mismatch"
            )
        else:
            expected_causal_relation = (
                "different_text_class_aligned"
                if canonical_match
                else "different_text_class_mismatch"
            )
        if causal_relation != expected_causal_relation:
            raise MatchedPanelReportError(
                f"{location}: causal_input_relation is not implied by component facts"
            )
        complete_input = _exact_bool(
            pair.get("complete_model_input_exact_match"),
            label=f"{location}.complete_model_input_exact_match",
        )
        if complete_input != all(component_matches.values()):
            raise MatchedPanelReportError(f"{location}: complete-input flag contradiction")
        if complete_input:
            complete_by_relation[relation] += 1
        primary = _exact_bool(
            pair.get("class_aligned_identical_complete_input"),
            label=f"{location}.class_aligned_identical_complete_input",
        )
        if primary != (causal_relation == PRIMARY_CAUSAL_STRATUM):
            raise MatchedPanelReportError(f"{location}: primary causal stratum contradiction")

        sides = pair.get("d2m"), pair.get("d3m")
        if not all(isinstance(value, Mapping) for value in sides):
            raise MatchedPanelReportError(f"{location}: missing ledger side identities")
        d2_side, d3_side = sides
        d2_sample = str(d2.get("sample_id") or "")
        d3_sample = str(d3.get("sample_id") or "")
        if (
            not d2_sample
            or d2_sample in d2_ids
            or d2_side.get("sample_id") != d2_sample
        ):
            raise MatchedPanelReportError(f"{location}: missing/duplicate D2m sample_id")
        if (
            not d3_sample
            or d3_sample in d3_ids
            or d3_side.get("sample_id") != d3_sample
        ):
            raise MatchedPanelReportError(f"{location}: missing/duplicate D3m sample_id")
        d2_ids.add(d2_sample)
        d3_ids.add(d3_sample)
        for source_label, source, table_id, scope, side in (
            ("D2m", d2, "D2m", "traceable_counterfactual_edit", d2_side),
            ("D3m", d3, "D3m", DECLARED_SURFACE, d3_side),
        ):
            if source.get("table_b_id") != table_id:
                raise MatchedPanelReportError(f"{location}: {source_label} table ID drift")
            if source.get("tn_scope") != scope or source.get("global_tn_verified") is not False:
                raise MatchedPanelReportError(f"{location}: {source_label} scope upgrade")
            if side.get("tn_scope") != scope:
                raise MatchedPanelReportError(f"{location}: {source_label} ledger scope drift")
            if _exact_int(source.get("class_id"), label=f"{location}.{source_label}.class_id") != _exact_int(
                side.get("class_id"), label=f"{location}.{source_label}.ledger class_id"
            ):
                raise MatchedPanelReportError(f"{location}: {source_label} class ID drift")
            for field in shared_fields:
                if _shared_pair_value(source, field, label=f"{location}.{source_label}") != _shared_pair_value(
                    pair, field, label=location
                ):
                    raise MatchedPanelReportError(
                        f"{location}: {source_label} {field} differs from ledger"
                    )
        if d2.get("class_id") != d2_class or d3.get("class_id") != d3_class:
            raise MatchedPanelReportError(f"{location}: source/ledger canonical class drift")
        observed_components = {
            "image_id": d2.get("image_id") == d3.get("image_id"),
            "file_name": d2.get("file_name") == d3.get("file_name"),
            "target_bbox_used": d2.get("target_bbox_used")
            == d3.get("target_bbox_used"),
            "positive_phrase": d2.get("sent") == d3.get("sent"),
            "negative_text": d2.get("try_tn") == d3.get("try_tn"),
            "canonical_class_id": d2.get("class_id") == d3.get("class_id"),
        }
        if dict(component_matches) != observed_components:
            raise MatchedPanelReportError(
                f"{location}: component facts do not replay from audited source rows"
            )
        observed_negative_relation = (
            "identical"
            if _normalized_text(d2.get("try_tn"))
            == _normalized_text(d3.get("try_tn"))
            else "different"
        )
        if relation != observed_negative_relation:
            raise MatchedPanelReportError(
                f"{location}: negative-text relation does not replay from source rows"
            )
        if d3.get("proposal_covered_verified") is not True:
            raise MatchedPanelReportError(f"{location}: D3m source is not proposal covered")

        if manifest_row.get("sample_id") != d3_sample:
            raise MatchedPanelReportError(f"{location}: evaluation manifest sample_id/order drift")
        for field in (
            "image_id",
            "matched_pair_id",
            "matched_parent_key_sha256",
            "matched_stratum",
            "canonical_class_id_match",
            "d2_canonical_class_id",
            "d3_canonical_class_id",
        ):
            if manifest_row.get(field) != d3.get(field):
                raise MatchedPanelReportError(
                    f"{location}: evaluation manifest {field}/order drift"
                )
        if manifest_row.get("tn_scope") != DECLARED_SURFACE or manifest_row.get(
            "global_tn_verified"
        ) is not False:
            raise MatchedPanelReportError(f"{location}: evaluation manifest scope upgrade")

        image_id = _exact_int(pair.get("image_id"), label=f"{location}.image_id")
        if image_id != _exact_int(parent.get("image_id"), label=f"{location}.parent image_id"):
            raise MatchedPanelReportError(f"{location}: parent image identity drift")
        metadata.append(
            {
                "manifest_index": index,
                "sample_id": d3_sample,
                "image_id": image_id,
                "matched_pair_id": pair_id,
                "matched_parent_key_sha256": parent_hash,
                "negative_text_relation": relation,
                "canonical_class_match": canonical_match,
                "canonical_class_id_match": canonical_match,
                "d2_canonical_class_id": d2_class,
                "d3_canonical_class_id": d3_class,
                "causal_input_relation": causal_relation,
                "complete_model_input_exact_match": complete_input,
            }
        )

    derived_counts = {
        "pairs": len(metadata),
        "unique_images": len({row["image_id"] for row in metadata}),
        "negative_text_relation_pairs": dict(sorted(relation_counts.items())),
        "canonical_class_relation_pairs": dict(sorted(class_relation_counts.items())),
        "canonical_class_id_mismatch_pairs": int(sum(mismatch_directions.values())),
        "identical_negative_text_class_id_mismatch_pairs": int(
            mismatch_by_relation.get("identical", 0)
        ),
        "canonical_class_id_mismatch_direction_pairs": dict(
            sorted(mismatch_directions.items())
        ),
        "causal_input_relation_pairs": dict(sorted(causal_counts.items())),
        "class_aligned_identical_complete_input_pairs": int(
            causal_counts.get(PRIMARY_CAUSAL_STRATUM, 0)
        ),
        "model_input_component_mismatch_pairs": dict(
            sorted(component_mismatches.items())
        ),
    }
    stats = audit.get("statistics")
    sealed = stats.get(SPLIT) if isinstance(stats, Mapping) else None
    if not isinstance(sealed, Mapping):
        raise MatchedPanelReportError("audit lacks calibration statistics")
    # These fields are mandatory in v2: their absence must never be interpreted
    # as zero canonical mismatches or as a complete-input panel.
    for field, observed in derived_counts.items():
        if field not in sealed:
            raise MatchedPanelReportError(
                f"audit calibration statistics omit mandatory {field}"
            )
        if sealed[field] != observed:
            raise MatchedPanelReportError(
                f"audit calibration statistic {field} differs from ledger replay"
            )
    if set(relation_counts) != set(RELATIONS) or any(
        relation_counts[relation] <= 0 for relation in RELATIONS
    ):
        raise MatchedPanelReportError(
            "identical and different negative-text strata must both be non-empty"
        )
    canonical_report = {
        "total_n": len(metadata),
        "aligned_n": int(class_relation_counts.get("aligned", 0)),
        "mismatch_n": int(class_relation_counts.get("mismatched", 0)),
        "mismatch_by_negative_text_relation": {
            relation: int(mismatch_by_relation.get(relation, 0))
            for relation in RELATIONS
        },
        "mismatch_direction_counts": dict(sorted(mismatch_directions.items())),
        "complete_model_input_exact_match_n": int(sum(complete_by_relation.values())),
        "complete_model_input_exact_match_by_negative_text_relation": {
            relation: int(complete_by_relation.get(relation, 0))
            for relation in RELATIONS
        },
        "class_aligned_identical_complete_input_n": int(
            causal_counts.get(PRIMARY_CAUSAL_STRATUM, 0)
        ),
    }
    return metadata, canonical_report


def _record_manifest_identity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    fields = (
        "manifest_sha256",
        "manifest_n",
        "manifest_binding_schema",
        "manifest_derivation_algorithm",
        "manifest_row_mapping_sha256",
        "source_manifest_sha256",
        "source_manifest_n",
    )
    result: dict[str, Any] = {}
    for field in fields:
        values = {json.dumps(row.get(field), sort_keys=True) for row in rows}
        if len(values) != 1:
            raise MatchedPanelReportError(f"records mix {field} identities")
        result[field] = first.get(field)
    return result


def _load_condition_records(
    *,
    condition: str,
    seed: int,
    path: Path,
    source_manifest_path: Path,
    surface_record: Mapping[str, Any],
    audit_record: Mapping[str, Any],
    training_source_record: Mapping[str, Any],
    evaluation_source_record: Mapping[str, Any],
    declared_surface: str,
) -> tuple[Any, dict[str, Any]]:
    source_manifest = load_manifest(source_manifest_path)
    try:
        loaded = load_tn_records(path, source_manifest, label=f"{condition} seed {seed}")
    except (OSError, RecordComparisonError, ValueError) as error:
        raise MatchedPanelReportError(
            f"{condition} seed {seed}: record/manifest binding failed: {error}"
        ) from error
    if len(loaded.run_ids) != 1 or loaded.run_ids[0] == "unknown":
        raise MatchedPanelReportError(
            f"{condition} seed {seed}: records require exactly one non-empty run_id"
        )
    if not bool(np.all(loaded.valid)):
        raise MatchedPanelReportError(
            f"{condition} seed {seed}: matched diagnostics require all rows valid"
        )
    identity = _record_manifest_identity(loaded.rows)
    if identity["manifest_sha256"] != surface_record["sha256"]:
        raise MatchedPanelReportError(
            f"{condition} seed {seed}: records do not name the declared evaluation manifest"
        )
    if identity["manifest_n"] != surface_record["rows"]:
        raise MatchedPanelReportError(
            f"{condition} seed {seed}: evaluation manifest row count mismatch"
        )
    expected_provenance = {
        "provenance_schema": PROVENANCE_SCHEMA,
        "table_b_id": condition,
        "train_seed": seed,
        "training_source_sha256": training_source_record["sha256"],
        "training_source_n": training_source_record["rows"],
        "matched_panel_audit_sha256": audit_record["sha256"],
        "evaluation_source_sha256": evaluation_source_record["sha256"],
        "evaluation_source_n": evaluation_source_record["rows"],
        "evaluation_manifest_sha256": surface_record["sha256"],
        "declared_evaluation_surface": declared_surface,
    }
    checkpoint_sha256: str | None = None
    for index, row in enumerate(loaded.rows):
        if row.get("eval_scope") != declared_surface:
            raise MatchedPanelReportError(
                f"{condition} seed {seed} record {index}: missing/different declared surface"
            )
        if row.get("global_tn_verified") is True or row.get(
            "formal_global_fpr_eligible"
        ) is True:
            raise MatchedPanelReportError(
                f"{condition} seed {seed} record {index}: scope upgrade"
            )
        for field, expected in expected_provenance.items():
            if row.get(field) != expected:
                raise MatchedPanelReportError(
                    f"{condition} seed {seed} record {index}: "
                    f"provenance {field} mismatch"
                )
        observed_checkpoint = _hash_string(
            row.get("checkpoint_sha256"),
            label=f"{condition} seed {seed} record {index}.checkpoint_sha256",
        )
        if checkpoint_sha256 is None:
            checkpoint_sha256 = observed_checkpoint
        elif checkpoint_sha256 != observed_checkpoint:
            raise MatchedPanelReportError(
                f"{condition} seed {seed}: records mix checkpoint identities"
            )
        support_sha = _hash_string(
            row.get("support_input_sha256"),
            label=f"{condition} seed {seed} record {index}.support_input_sha256",
        )
        support_kind = row.get("support_input_kind")
        support_classes = row.get("support_class_ids")
        expected_class = _exact_int(
            source_manifest.rows[index].get("class_id"),
            label=f"{condition} seed {seed} record {index}.manifest class_id",
        )
        if support_kind not in {"patch", "patches", "patch_global"} or support_classes != [
            expected_class
        ]:
            raise MatchedPanelReportError(
                f"{condition} seed {seed} record {index}: support identity mismatch"
            )
    if checkpoint_sha256 is None:
        raise MatchedPanelReportError(f"{condition} seed {seed}: no checkpoint identity")
    record = dict(loaded.file_record)
    record.update(
        {
            "rows": len(loaded.rows),
            "run_id": loaded.run_ids[0],
            "manifest_binding_mode": loaded.manifest_binding_mode,
            "manifest_identity": identity,
            "provenance": {
                **expected_provenance,
                "checkpoint_sha256": checkpoint_sha256,
            },
        }
    )
    return loaded, record


def _metrics(positive: np.ndarray, negative: np.ndarray) -> dict[str, Any]:
    if positive.size == 0 or positive.shape != negative.shape:
        raise MatchedPanelReportError("diagnostic metrics require non-empty paired scores")
    fpr = exact_fpr95(positive, negative)
    return {
        "metric_label": METRIC_LABEL,
        "formal_global_fpr_eligible": False,
        "n": int(positive.size),
        "positive_q05": float(fpr["threshold"]),
        "positive_q05_definition": "exact 95%-TPR order statistic",
        "auroc": float(exact_binary_auroc(positive, negative)),
        "fpr95_like_diagnostic": float(fpr["fpr"]),
        "actual_positive_tpr_at_q05": float(fpr["actual_tpr"]),
        "accepted_positive_n": int(fpr["accepted_positive_n"]),
        "threshold_tie_policy": str(fpr["tie_policy"]),
        "positive_over_negative_win_rate": float(np.mean(positive > negative)),
        "positive_equals_negative_tie_rate": float(
            np.mean(positive == negative)
        ),
        "positive_score_mean": float(np.mean(positive)),
        "negative_score_mean": float(np.mean(negative)),
        "score_gap_mean": float(np.mean(positive - negative)),
    }


def _paired_delta(
    d2m: Mapping[str, Any],
    d3m: Mapping[str, Any],
    *,
    d2_positive: np.ndarray,
    d2_negative: np.ndarray,
    d3_positive: np.ndarray,
    d3_negative: np.ndarray,
) -> dict[str, Any]:
    d2_gap = d2_positive - d2_negative
    d3_gap = d3_positive - d3_negative
    return {
        "metric_label": METRIC_LABEL,
        "formal_global_fpr_eligible": False,
        "direction": "D3m_minus_D2m",
        "paired_on_exact_manifest_rows": True,
        "n": int(d2_positive.size),
        **{field: float(d3m[field]) - float(d2m[field]) for field in SCALAR_METRICS},
        "rowwise_paired_delta_means": {
            "positive_score": float(np.mean(d3_positive - d2_positive)),
            "negative_score": float(np.mean(d3_negative - d2_negative)),
            "score_gap": float(np.mean(d3_gap - d2_gap)),
        },
        "paired_model_comparison_rates": {
            "d3m_higher_positive_score_rate": float(
                np.mean(d3_positive > d2_positive)
            ),
            "equal_positive_score_tie_rate": float(
                np.mean(d3_positive == d2_positive)
            ),
            "d3m_lower_negative_score_rate": float(
                np.mean(d3_negative < d2_negative)
            ),
            "equal_negative_score_tie_rate": float(
                np.mean(d3_negative == d2_negative)
            ),
            "d3m_larger_score_gap_rate": float(np.mean(d3_gap > d2_gap)),
            "equal_score_gap_tie_rate": float(np.mean(d3_gap == d2_gap)),
        },
    }


def _mean_std(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else None,
        "std_ddof": 1,
    }


def _seed_summary(per_seed: Mapping[str, Any], seeds: Sequence[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for relation in REPORT_STRATA:
        relation_summary: dict[str, Any] = {
            "metric_label": METRIC_LABEL,
            "formal_global_fpr_eligible": False,
            "seed_set": list(seeds),
        }
        for key in ("D2m", "D3m", "D3m_minus_D2m"):
            relation_summary[key] = {
                metric: _mean_std(
                    per_seed[str(seed)]["strata"][relation][key][metric]
                    for seed in seeds
                )
                for metric in SCALAR_METRICS
            }
        result[relation] = relation_summary
    return result


def _normalize_seeds(values: Sequence[Any], *, label: str) -> tuple[int, ...]:
    seeds = tuple(_exact_int(value, label=f"{label}[{index}]") for index, value in enumerate(values))
    if not seeds:
        raise MatchedPanelReportError("expected seed set must be non-empty")
    if len(set(seeds)) != len(seeds):
        raise MatchedPanelReportError("expected seed set contains duplicates")
    return tuple(sorted(seeds))


def _normalize_record_map(
    values: Mapping[Any, Path | str], *, label: str
) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for raw_seed, raw_path in values.items():
        seed = _exact_int(raw_seed, label=f"{label} seed")
        if seed in result:
            raise MatchedPanelReportError(f"{label}: duplicate seed {seed}")
        result[seed] = Path(raw_path).expanduser().resolve(strict=True)
    return result


def aggregate_matched_panel(
    *,
    audit_path: Path | str,
    pair_ledger_path: Path | str,
    d2m_source_path: Path | str,
    d3m_source_path: Path | str,
    evaluation_manifest_path: Path | str,
    d2m_records: Mapping[int, Path | str],
    d3m_records: Mapping[int, Path | str],
    expected_seeds: Sequence[int],
    declared_surface: str = DECLARED_SURFACE,
) -> dict[str, Any]:
    if declared_surface != DECLARED_SURFACE:
        raise MatchedPanelReportError(
            f"matched-panel reports require declared surface {DECLARED_SURFACE!r}"
        )
    audit_path = Path(audit_path).expanduser().resolve(strict=True)
    pair_ledger_path = Path(pair_ledger_path).expanduser().resolve(strict=True)
    d2m_source_path = Path(d2m_source_path).expanduser().resolve(strict=True)
    d3m_source_path = Path(d3m_source_path).expanduser().resolve(strict=True)
    evaluation_manifest_path = Path(evaluation_manifest_path).expanduser().resolve(
        strict=True
    )
    audit = _read_json(audit_path, label="matched-panel audit")
    try:
        verified_audit = verify_panel(audit_path)
    except (OSError, KeyError, TypeError, MatchedPanelError) as error:
        raise MatchedPanelReportError(
            f"full v2 matched-panel audit verification failed: {error}"
        ) from error
    if dict(verified_audit) != dict(audit):
        raise MatchedPanelReportError("matched-panel audit changed during verification")
    _validate_audit(audit)
    population = _population_contract(audit)
    audit_record = _file_record(audit_path)
    expected = _normalize_seeds(expected_seeds, label="expected_seeds")
    d2_paths = _normalize_record_map(d2m_records, label="D2m records")
    d3_paths = _normalize_record_map(d3m_records, label="D3m records")
    expected_set = set(expected)
    if set(d2_paths) != expected_set or set(d3_paths) != expected_set:
        raise MatchedPanelReportError(
            "D2m and D3m record seed sets must each equal the exact expected seed set"
        )
    for condition, paths in (("D2m", d2_paths), ("D3m", d3_paths)):
        resolved = list(paths.values())
        if len(set(resolved)) != len(resolved):
            raise MatchedPanelReportError(f"{condition}: one record file labels multiple seeds")
        hashes = [_sha256(path) for path in resolved]
        if len(set(hashes)) != len(hashes):
            raise MatchedPanelReportError(
                f"{condition}: byte-identical record files label multiple seeds"
            )
    for seed in expected:
        if d2_paths[seed] == d3_paths[seed]:
            raise MatchedPanelReportError(
                f"seed {seed}: D2m and D3m cannot be the same record artifact"
            )

    audited_records = {
        "pair_ledger": _verify_audited_artifact(
            audit,
            audit_path=audit_path,
            key="pairs_calibration",
            supplied_path=pair_ledger_path,
        ),
        "D2m_source": _verify_audited_artifact(
            audit,
            audit_path=audit_path,
            key="d2m_calibration",
            supplied_path=d2m_source_path,
        ),
        "D3m_source": _verify_audited_artifact(
            audit,
            audit_path=audit_path,
            key="d3m_calibration",
            supplied_path=d3m_source_path,
        ),
        "D2m_training_source": _verify_named_audited_artifact(
            audit, audit_path=audit_path, key="d2m_train"
        ),
        "D3m_training_source": _verify_named_audited_artifact(
            audit, audit_path=audit_path, key="d3m_train"
        ),
    }
    row_n = audited_records["pair_ledger"]["rows"]
    if any(
        audited_records[key]["rows"] != row_n
        for key in ("pair_ledger", "D2m_source", "D3m_source")
    ):
        raise MatchedPanelReportError("audited calibration artifacts have unequal rows")
    if population["matched_pairs"] != row_n:
        raise MatchedPanelReportError("matched population denominator differs from ledger")
    surface_rows = _read_jsonl(
        evaluation_manifest_path, label="declared evaluation manifest"
    )
    surface_record = _file_record(evaluation_manifest_path, rows=len(surface_rows))
    if len(surface_rows) != row_n:
        raise MatchedPanelReportError(
            "declared evaluation manifest is not the full audited matched panel"
        )
    metadata, canonical_report = _validate_panel_rows(
        audit=audit,
        ledger=_read_jsonl(pair_ledger_path, label="matched pair ledger"),
        d2_source=_read_jsonl(d2m_source_path, label="D2m audited source"),
        d3_source=_read_jsonl(d3m_source_path, label="D3m audited source"),
        surface=surface_rows,
    )
    if population["class_aligned_identical_claim_denominator"] != canonical_report[
        "class_aligned_identical_complete_input_n"
    ]:
        raise MatchedPanelReportError(
            "primary causal denominator differs from ledger component replay"
        )

    loaded: dict[str, dict[int, Any]] = {"D2m": {}, "D3m": {}}
    record_inputs: dict[str, dict[str, Any]] = {"D2m": {}, "D3m": {}}
    surface_identity: Mapping[str, Any] | None = None
    run_ids: dict[str, set[str]] = {"D2m": set(), "D3m": set()}
    checkpoint_hashes: set[str] = set()
    support_identity: tuple[tuple[Any, ...], ...] | None = None
    for condition, paths in (("D2m", d2_paths), ("D3m", d3_paths)):
        for seed in expected:
            condition_records, input_record = _load_condition_records(
                condition=condition,
                seed=seed,
                path=paths[seed],
                source_manifest_path=d3m_source_path,
                surface_record=surface_record,
                audit_record=audit_record,
                training_source_record=audited_records[
                    f"{condition}_training_source"
                ],
                evaluation_source_record=audited_records["D3m_source"],
                declared_surface=declared_surface,
            )
            identity = input_record["manifest_identity"]
            if surface_identity is None:
                surface_identity = dict(identity)
            elif dict(identity) != dict(surface_identity):
                raise MatchedPanelReportError(
                    "D2m/D3m records were not evaluated on the same manifest identity"
                )
            run_id = str(input_record["run_id"])
            if run_id in run_ids[condition]:
                raise MatchedPanelReportError(
                    f"{condition}: one run_id labels multiple training seeds"
                )
            run_ids[condition].add(run_id)
            checkpoint_sha = str(
                input_record["provenance"]["checkpoint_sha256"]
            )
            if checkpoint_sha in checkpoint_hashes:
                raise MatchedPanelReportError(
                    "one checkpoint identity labels multiple conditions or seeds"
                )
            checkpoint_hashes.add(checkpoint_sha)
            observed_support = tuple(
                (
                    row.get("support_input_kind"),
                    row.get("support_input_sha256"),
                    tuple(row.get("support_class_ids", [])),
                )
                for row in condition_records.rows
            )
            if support_identity is None:
                support_identity = observed_support
            elif observed_support != support_identity:
                raise MatchedPanelReportError(
                    "conditions/seeds did not consume identical support inputs"
                )
            loaded[condition][seed] = condition_records
            record_inputs[condition][str(seed)] = input_record

    per_seed: dict[str, Any] = {}
    for seed in expected:
        d2 = loaded["D2m"][seed]
        d3 = loaded["D3m"][seed]
        for index, (left, right, meta) in enumerate(zip(d2.rows, d3.rows, metadata)):
            identity = ("manifest_index", "sample_id", "image_id", "split")
            if any(left.get(field) != right.get(field) for field in identity):
                raise MatchedPanelReportError(
                    f"seed {seed} record {index}: D2m/D3m record identity/order mismatch"
                )
            if any(
                left.get(field) != right.get(field)
                for field in (
                    "support_input_kind",
                    "support_input_sha256",
                    "support_class_ids",
                )
            ):
                raise MatchedPanelReportError(
                    f"seed {seed} record {index}: D2m/D3m support input mismatch"
                )
            if left.get("manifest_index") != meta["manifest_index"] or left.get(
                "sample_id"
            ) != meta["sample_id"]:
                raise MatchedPanelReportError(
                    f"seed {seed} record {index}: records do not join to pair ledger"
                )
        seed_result: dict[str, Any] = {
            "metric_label": METRIC_LABEL,
            "formal_global_fpr_eligible": False,
            "D2m_run_id": d2.run_ids[0],
            "D3m_run_id": d3.run_ids[0],
            "strata": {},
        }
        for relation in REPORT_STRATA:
            is_primary = relation == PRIMARY_CAUSAL_STRATUM
            indices = np.asarray(
                [
                    index
                    for index, row in enumerate(metadata)
                    if (
                        row["causal_input_relation"] == PRIMARY_CAUSAL_STRATUM
                        if is_primary
                        else row["negative_text_relation"] == relation
                    )
                ],
                dtype=np.int64,
            )
            d2_positive = d2.positive[indices]
            d2_negative = d2.negative[indices]
            d3_positive = d3.positive[indices]
            d3_negative = d3.negative[indices]
            d2_metrics = _metrics(d2_positive, d2_negative)
            d3_metrics = _metrics(d3_positive, d3_negative)
            joined = []
            for local, index in enumerate(indices.tolist()):
                row = dict(metadata[index])
                row.update(
                    {
                        "D2m": {
                            "pos_score": float(d2_positive[local]),
                            "neg_score": float(d2_negative[local]),
                            "score_gap": float(d2_positive[local] - d2_negative[local]),
                        },
                        "D3m": {
                            "pos_score": float(d3_positive[local]),
                            "neg_score": float(d3_negative[local]),
                            "score_gap": float(d3_positive[local] - d3_negative[local]),
                        },
                        "D3m_minus_D2m": {
                            "pos_score": float(d3_positive[local] - d2_positive[local]),
                            "neg_score": float(d3_negative[local] - d2_negative[local]),
                            "score_gap": float(
                                (d3_positive[local] - d3_negative[local])
                                - (d2_positive[local] - d2_negative[local])
                            ),
                        },
                    }
                )
                joined.append(row)
            mismatch_n = sum(not row["canonical_class_match"] for row in joined)
            complete_n = sum(
                row["complete_model_input_exact_match"] for row in joined
            )
            seed_result["strata"][relation] = {
                "metric_label": METRIC_LABEL,
                "formal_global_fpr_eligible": False,
                "stratum_kind": (
                    "primary_clean_causal_input"
                    if is_primary
                    else "descriptive_negative_text_relation"
                ),
                "negative_text_relation": "identical" if is_primary else relation,
                "causal_input_relation": (
                    PRIMARY_CAUSAL_STRATUM if is_primary else None
                ),
                "n": int(indices.size),
                "unique_images": len({row["image_id"] for row in joined}),
                "canonical_class_aligned_n": int(indices.size - mismatch_n),
                "canonical_class_mismatch_n": int(mismatch_n),
                "complete_model_input_exact_match_n": int(complete_n),
                "D2m": d2_metrics,
                "D3m": d3_metrics,
                "D3m_minus_D2m": _paired_delta(
                    d2_metrics,
                    d3_metrics,
                    d2_positive=d2_positive,
                    d2_negative=d2_negative,
                    d3_positive=d3_positive,
                    d3_negative=d3_negative,
                ),
                "paired_records": joined,
            }
        per_seed[str(seed)] = seed_result

    return {
        "schema": REPORT_SCHEMA,
        "status": "validated_internal_records_diagnostic",
        "report_role": "supplemental Table-B matched panel",
        "metric_label": METRIC_LABEL,
        "formal_global_fpr_eligible": False,
        "claim_contract": {
            "evaluation_surface": declared_surface,
            "surface_scope": "proposal-covered, not image-global",
            "fpr95_field_semantics": (
                "FPR95-like diagnostic on matched proposal-covered pairs; not formal global FPR95"
            ),
            "identical_and_different_negative_text_reported_separately": True,
            "primary_clean_causal_stratum_reported_separately": True,
            "canonical_mismatches_explicitly_reported": True,
            "pairwise_population": "matched pairs only",
            "unmatched_parent_generalization_supported": False,
        },
        "validation": {
            "pass": True,
            "full_v2_builder_verify_panel_passed": True,
            "audit_inputs_outputs_and_dataset_configs_rehashed": True,
            "claim_scope_verified": True,
            "matching_yield_verified": True,
            "split": SPLIT,
            "exact_seed_set": list(expected),
            "D2m_seed_set": sorted(d2_paths),
            "D3m_seed_set": sorted(d3_paths),
            "same_declared_surface": True,
            "same_manifest_identity": True,
            "identical_support_inputs_across_conditions_and_seeds": True,
            "full_audited_pair_count": len(metadata),
            "ledger_source_manifest_record_order_match": True,
            "all_records_valid": True,
            "no_scope_upgrade": True,
            "canonical_class_accounting": canonical_report,
            "population_accounting": population,
        },
        "inputs": {
            "audit": audit_record,
            **audited_records,
            "evaluation_manifest": surface_record,
            "record_files": record_inputs,
            "record_manifest_identity": dict(surface_identity or {}),
        },
        "per_seed": per_seed,
        "across_seed_summary": _seed_summary(per_seed, expected),
    }


def _require_common_formal_protocol(
    reference: Mapping[str, Any] | None,
    observed: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    required = {
        "schema",
        "common_runtime",
        "phase_command_templates",
        "command_template_sha256",
        "evaluation_code_closure",
        "evaluation_code_closure_sha256",
    }
    if set(observed) != required:
        raise MatchedPanelReportError(
            f"seed {seed}: formal protocol field set drifted"
        )
    if reference is None:
        return json.loads(json.dumps(dict(observed), sort_keys=True))
    for field, label in (
        ("common_runtime", "runtime protocol"),
        ("phase_command_templates", "phase command template"),
        ("command_template_sha256", "phase command template hash"),
        ("evaluation_code_closure", "evaluation code closure"),
        ("evaluation_code_closure_sha256", "evaluation code closure hash"),
    ):
        if observed.get(field) != reference.get(field):
            raise MatchedPanelReportError(
                f"seed {seed}: formal {label} differs across seeds"
            )
    if observed.get("schema") != reference.get("schema"):
        raise MatchedPanelReportError(
            f"seed {seed}: formal protocol schema differs across seeds"
        )
    return dict(reference)


def aggregate_formal_matched_panel(
    *,
    audit_path: Path | str,
    pair_ledger_path: Path | str,
    d2m_source_path: Path | str,
    d3m_source_path: Path | str,
    evaluation_manifest_path: Path | str,
    evaluation_outputs: Mapping[int, Path | str],
    declared_surface: str = DECLARED_SURFACE,
    source_resolver: Any = None,
    require_training_queue: bool = True,
) -> dict[str, Any]:
    """Aggregate only records recovered from three replayed formal outputs."""

    from tools import run_stageb_table_b_matched_evaluations as evaluator

    outputs = _normalize_record_map(
        evaluation_outputs, label="matched evaluation outputs"
    )
    if tuple(sorted(outputs)) != FORMAL_SEEDS:
        raise MatchedPanelReportError(
            f"formal matched aggregation requires exact seeds {FORMAL_SEEDS}"
        )
    audit_path = Path(audit_path).expanduser().resolve(strict=True)
    try:
        audit = verify_panel(audit_path)
    except (OSError, KeyError, TypeError, MatchedPanelError) as error:
        raise MatchedPanelReportError(
            f"formal matched-panel audit verification failed: {error}"
        ) from error
    d2_records: dict[int, Path] = {}
    d3_records: dict[int, Path] = {}
    attestations: dict[str, Any] = {}
    surface_identity: dict[str, Any] | None = None
    checkpoint_hashes: set[str] = set()
    common_protocol: dict[str, Any] | None = None
    common_training_source_contract: str | None = None
    surface_fields = (
        "matched_eval_surface_source_sha256",
        "matched_eval_surface_source_n",
        "matched_eval_surface_audit_sha256",
        "matched_eval_surface_ledger_sha256",
        "matched_eval_surface_derived_sha256",
        "matched_eval_surface_row_mapping_sha256",
        "matched_eval_surface_image_files_n",
        "matched_eval_surface_image_files_sha256",
        "matched_eval_surface_support_pool_classes_n",
        "matched_eval_surface_support_pool_files_n",
        "matched_eval_surface_support_pool_mapping_sha256",
        "matched_eval_surface_support_pool_files_sha256",
        "matched_eval_surface_scope",
        "matched_eval_surface_split",
        "formal_global_fpr_eligible",
    )
    for seed in FORMAL_SEEDS:
        output_dir = outputs[seed]
        try:
            postflight = evaluator.verify_completed_output(output_dir)
        except (OSError, ValueError, evaluator.MatchedEvaluationError) as error:
            raise MatchedPanelReportError(
                f"seed {seed}: matched evaluation replay failed: {error}"
            ) from error
        launch_path = (output_dir / "launch.json").resolve(strict=True)
        postflight_path = (output_dir / "postflight.json").resolve(strict=True)
        launch = _read_json(launch_path, label=f"seed {seed} matched launch")
        contract = launch.get("contract")
        if not (
            isinstance(contract, Mapping)
            and contract.get("seed") == seed
            and contract.get("evaluation_seed") == evaluator.EVAL_SEED
            and contract.get("conditions") == list(evaluator.CONDITIONS)
            and contract.get("formal_global_fpr_eligible") is False
        ):
            raise MatchedPanelReportError(
                f"seed {seed}: matched evaluation contract identity drift"
            )
        training_source_contract = contract.get(
            "training_source_contract", evaluator.LEGACY_TRAINING_SOURCE_CONTRACT
        )
        if training_source_contract not in evaluator.TRAINING_SOURCE_CONTRACTS:
            raise MatchedPanelReportError(
                f"seed {seed}: training source contract is invalid"
            )
        if common_training_source_contract is None:
            common_training_source_contract = str(training_source_contract)
        elif training_source_contract != common_training_source_contract:
            raise MatchedPanelReportError(
                "formal seed outputs mix training source contracts"
            )
        try:
            observed_protocol = evaluator.formal_protocol_identity(launch)
        except (OSError, ValueError, evaluator.MatchedEvaluationError) as error:
            raise MatchedPanelReportError(
                f"seed {seed}: formal evaluation protocol replay failed: {error}"
            ) from error
        common_protocol = _require_common_formal_protocol(
            common_protocol, observed_protocol, seed=seed
        )
        queue_dirs = set()
        for condition in evaluator.CONDITIONS:
            training = contract["training"][condition]
            queue = training.get("training_queue")
            if require_training_queue and not isinstance(queue, Mapping):
                raise MatchedPanelReportError(
                    f"seed {seed} {condition}: formal output lacks training queue attestation"
                )
            if isinstance(queue, Mapping):
                queue_dirs.add(
                    Path(str(queue["manifest"]["path"])).resolve(strict=True).parent
                )
        if len(queue_dirs) > 1:
            raise MatchedPanelReportError(
                f"seed {seed}: D2m/D3m name different training queues"
            )
        queue_dir = next(iter(queue_dirs)) if queue_dirs else None
        try:
            sources, _evidence = evaluator._resolve_sources(
                d2m_root=Path(contract["training"]["D2m"]["training_run_root"]),
                d3m_root=Path(contract["training"]["D3m"]["training_run_root"]),
                seed=seed,
                audit=audit,
                audit_path=audit_path,
                training_queue_dir=queue_dir,
                resolver=source_resolver,
                formal_v2=(
                    training_source_contract
                    == evaluator.FORMAL_V2_TRAINING_SOURCE_CONTRACT
                ),
            )
            evaluator._validate_runtime_sources(launch, sources)
        except (OSError, ValueError, evaluator.MatchedEvaluationError) as error:
            raise MatchedPanelReportError(
                f"seed {seed}: training provenance re-resolution failed: {error}"
            ) from error
        surface = postflight.get("surface")
        if not isinstance(surface, Mapping):
            raise MatchedPanelReportError(f"seed {seed}: postflight surface is missing")
        observed_surface = {field: surface.get(field) for field in surface_fields}
        if surface_identity is None:
            surface_identity = observed_surface
        elif observed_surface != surface_identity:
            raise MatchedPanelReportError(
                "formal seed outputs do not share one canonical matched surface"
            )
        condition_attestations = {}
        for condition in evaluator.CONDITIONS:
            artifacts = postflight["conditions"][condition]
            record_path = Path(str(artifacts["final_records"]["path"])).resolve(
                strict=True
            )
            if condition == "D2m":
                d2_records[seed] = record_path
            else:
                d3_records[seed] = record_path
            checkpoint_sha = str(
                contract["training"][condition]["checkpoint"]["sha256"]
            )
            if checkpoint_sha in checkpoint_hashes:
                raise MatchedPanelReportError(
                    "formal outputs reuse one checkpoint across conditions/seeds"
                )
            checkpoint_hashes.add(checkpoint_sha)
            condition_attestations[condition] = {
                "training_run_id": contract["training"][condition][
                    "training_run_id"
                ],
                "training_run_root": contract["training"][condition][
                    "training_run_root"
                ],
                "checkpoint": dict(contract["training"][condition]["checkpoint"]),
                "training_source": dict(
                    contract["training"][condition]["training_source"]
                ),
                "training_queue": contract["training"][condition].get(
                    "training_queue"
                ),
                "final_records": dict(artifacts["final_records"]),
            }
        attestations[str(seed)] = {
            "output_dir": str(output_dir),
            "launch": _file_record(launch_path),
            "postflight": _file_record(postflight_path),
            "contract_sha256": launch["contract_sha256"],
            "conditions": condition_attestations,
        }
    report = aggregate_matched_panel(
        audit_path=audit_path,
        pair_ledger_path=pair_ledger_path,
        d2m_source_path=d2m_source_path,
        d3m_source_path=d3m_source_path,
        evaluation_manifest_path=evaluation_manifest_path,
        d2m_records=d2_records,
        d3m_records=d3_records,
        expected_seeds=FORMAL_SEEDS,
        declared_surface=declared_surface,
    )
    report["schema"] = FORMAL_REPORT_SCHEMA
    report["status"] = "validated_formal_supplemental_diagnostic"
    report["validation"]["formal_evaluation_outputs_replayed"] = True
    report["validation"]["training_provenance_reresolved"] = True
    report["validation"]["training_queue_attestations_replayed"] = bool(
        require_training_queue
    )
    if common_protocol is None:
        raise MatchedPanelReportError("formal matched aggregation found no protocol")
    report["validation"]["common_runtime_protocol_verified"] = True
    report["validation"]["common_command_template_verified"] = True
    report["validation"]["common_evaluation_code_closure_verified"] = True
    report["formal_evaluation_protocol"] = {
        "training_source_contract": common_training_source_contract,
        "common_runtime": common_protocol["common_runtime"],
        "command_template_sha256": common_protocol["command_template_sha256"],
        "phase_command_templates": common_protocol["phase_command_templates"],
        "evaluation_code_closure_sha256": common_protocol[
            "evaluation_code_closure_sha256"
        ],
        "evaluation_code_closure": common_protocol["evaluation_code_closure"],
    }
    report["inputs"]["formal_evaluation_outputs"] = attestations
    return report


def _parse_seed_record(values: Sequence[str], *, label: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator or not seed_text or not path_text:
            raise MatchedPanelReportError(
                f"{label}: expected repeated SEED=/path/to/records.jsonl values"
            )
        try:
            seed = int(seed_text)
        except ValueError as error:
            raise MatchedPanelReportError(f"{label}: invalid seed {seed_text!r}") from error
        if seed in result:
            raise MatchedPanelReportError(f"{label}: duplicate seed {seed}")
        result[seed] = Path(path_text)
    return result


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--pair-ledger", required=True, type=Path)
    parser.add_argument("--d2m-source", required=True, type=Path)
    parser.add_argument("--d3m-source", required=True, type=Path)
    parser.add_argument("--evaluation-manifest", required=True, type=Path)
    parser.add_argument(
        "--eval-output-dir",
        required=True,
        action="append",
        help="Repeated SEED=/canonical/matched/evaluation/output values.",
    )
    parser.add_argument("--declared-surface", default=DECLARED_SURFACE)
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = aggregate_formal_matched_panel(
            audit_path=args.audit,
            pair_ledger_path=args.pair_ledger,
            d2m_source_path=args.d2m_source,
            d3m_source_path=args.d3m_source,
            evaluation_manifest_path=args.evaluation_manifest,
            evaluation_outputs=_parse_seed_record(
                args.eval_output_dir, label="matched evaluation outputs"
            ),
            declared_surface=args.declared_surface,
        )
    except (MatchedPanelReportError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output_json is not None:
        _write_atomic(args.output_json, rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
