#!/usr/bin/env python3
"""Freeze and verify a validation-only canonical-caption routing policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


SCHEMA = "stageb-canonical-caption-route-selection-v1"
KIND = "formal_validation_frozen_canonical_caption_route"
IDENTITY_ALGORITHM = "sha256-canonical-json-excluding-artifact-identity-v1"
VAL_SPLITS = ("refcoco_val", "refcocop_val", "refcocog_val")
EXPECTED_MANIFEST_N = {
    "refcoco_val": 10834,
    "refcocop_val": 10758,
    "refcocog_val": 4896,
}
EXPECTED_SEEDS = {
    "refcoco_val": 42,
    "refcocop_val": 300042,
    "refcocog_val": 600042,
}
SEED_PROTOCOL = "canonical_ref8_split_index_stride100000_v1"
DEFAULT_DESCRIPTOR_ID = "external_base_direct_v1"
BASE_IDENTITY_FIELDS = {
    "diagnostic_descriptor_kind": "external_gdino_base_identity",
    "diagnostic_identity_kind": "external_gdino_base_direct",
    "diagnostic_score_key": "stage_b_gdino_base_score",
    "diagnostic_query_count": 900,
    "diagnostic_output_box_source": (
        "external_outputs.pred_boxes_at_direct_global_argmax"
    ),
    "diagnostic_standard_ref_beta": 0.0,
    "diagnostic_winner_rule": "first_argmax_over_full_external_query_axis",
    "diagnostic_query_domain": "all_900_external_gdino_queries",
    "uses_external_base_score": True,
    "uses_external_rank_score": False,
    "uses_external_box": True,
    "uses_adapter_rank_residual": False,
    "uses_patch_top50_admission": False,
    "uses_top_query_mapping": False,
    "uses_fusion_weights": False,
    "external_gdino_base_identity_contract_version": 1,
}
CAPTION_CONTRACT = {
    "source": "patch_outputs.stage_a_captions",
    "normalization": "_norm_text",
    "key": "exact_normalized_caption",
    "uses_target_category_or_box_for_grouping": False,
}
MANIFEST_CAPTION_AUDIT_CONTRACT = {
    "schema": "stageb-canonical-caption-manifest-audit-v1",
    "class_id_source": "instances[0].class_id",
    "canonical_name_source": "canonical_classes_with_aliases.json",
    "preferred_name_order": ["base_name", "raw_name", "norm_name", "synset"],
    "missing_class_id_policy": "object",
    "normalization": "_norm_text",
}

# This registry was fixed before any exact-caption test breakdown was generated.
DESCRIPTOR_REGISTRY: Dict[str, Dict[str, Any]] = {
    DEFAULT_DESCRIPTOR_ID: {
        "descriptor_kind": "external_gdino_base_direct",
        "score_source": "stage_b_gdino_base_score",
        "selection": "first_argmax_over_all_900_external_queries",
        "output_box_source": "external_query_box",
        "uses_rank_residual": False,
        "uses_confidence": False,
        "uses_fusion": False,
    },
    "max_patch_p025_w003125_v1": {
        "descriptor_kind": "external_rank_transfer",
        "transfer_mode": "max_score_iou_power",
        "iou_power": 0.25,
        "patch_weight": 0.03125,
        "text_weight": 1.0,
        "score_source": "stage_b_gdino_rank_score",
        "output_box_source": "fixed_patch_candidate",
    },
    "max_external_p05_w0046875_v1": {
        "descriptor_kind": "external_rank_transfer",
        "transfer_mode": "max_score_iou_power_external_box",
        "iou_power": 0.5,
        "patch_weight": 0.046875,
        "text_weight": 1.0,
        "score_source": "stage_b_gdino_rank_score",
        "output_box_source": "matched_external_query",
    },
    "top_query_patch_w0_v1": {
        "descriptor_kind": "external_rank_transfer",
        "transfer_mode": "top_query_nearest_candidate",
        "iou_power": None,
        "patch_weight": 0.0,
        "text_weight": 1.0,
        "score_source": "stage_b_gdino_rank_score",
        "output_box_source": "fixed_patch_candidate",
    },
    "top_query_external_w0_v1": {
        "descriptor_kind": "external_rank_transfer",
        "transfer_mode": "top_query_nearest_candidate_external_box",
        "iou_power": None,
        "patch_weight": 0.0,
        "text_weight": 1.0,
        "score_source": "stage_b_gdino_rank_score",
        "output_box_source": "assigned_external_query",
    },
}
CANDIDATE_DESCRIPTOR_IDS = tuple(
    key for key in DESCRIPTOR_REGISTRY if key != DEFAULT_DESCRIPTOR_ID
)
SELECTION_CONTRACT = {
    "schema": "stageb-canonical-caption-route-selection-contract-v1",
    "input_splits": list(VAL_SPLITS),
    "full_validation_only": True,
    "batch_size": 32,
    "num_workers": 4,
    "amp": "enabled_by_invocation; legacy diagnostic row lacks a machine field",
    "minimum_total_expressions": 300,
    "supported_split_minimum_expressions": 50,
    "minimum_supported_splits": 2,
    "all_observed_splits_must_be_non_degrading": True,
    "minimum_supported_split_gain": "max(2, ceil(0.01 * split_n))",
    "minimum_supported_splits_meeting_gain": 2,
    "minimum_pooled_gain": "max(10, ceil(0.02 * pooled_n))",
    "equivalence_band": "max(2, ceil(0.005 * pooled_n))",
    "equivalence_primary": "maximum pooled correct gain",
    "equivalence_secondary": "maximum worst supported split gain rate",
    "descriptor_tiebreak": [
        "external_box_before_patch_box",
        "top_query_before_max_score",
        "smaller_patch_weight",
        "canonical_descriptor_id",
    ],
    "routing_key": "exact _norm_text(patch_outputs.stage_a_captions[row])",
    "forbidden_routing_inputs": [
        "targets",
        "target.labels",
        "target.boxes",
        "manifest.category",
        "refcoco_category_id",
        "dataset",
        "split",
    ],
    "unknown_or_ineligible_caption": DEFAULT_DESCRIPTOR_ID,
}

_WS_RE = re.compile(r"\s+")


class CaptionRouteArtifactError(RuntimeError):
    pass


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _norm_text(value: Any) -> str:
    return _WS_RE.sub(
        " ", str(value or "").replace("_", " ").replace(".", " ").strip().lower()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise CaptionRouteArtifactError(f"evidence file is missing: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _require_file_unchanged(
    record: Mapping[str, Any], *, label: str
) -> None:
    observed = _file_record(Path(str(record.get("path", ""))))
    if observed != dict(record):
        raise CaptionRouteArtifactError(f"{label} changed during validation")


def _source_component_records(source_identity: Mapping[str, Any]) -> Dict[str, Any]:
    bindings = {
        "patch_config": ("patch_config", "patch_config_sha256"),
        "patch_checkpoint": ("checkpoint", "patch_checkpoint_sha256"),
        "external_config": (
            "external_gdino_config",
            "external_gdino_config_sha256",
        ),
        "external_checkpoint": (
            "external_gdino_checkpoint",
            "external_gdino_checkpoint_sha256",
        ),
    }
    records: Dict[str, Any] = {}
    for label, (path_key, sha_key) in bindings.items():
        path_value = source_identity.get(path_key)
        expected_sha = source_identity.get(sha_key)
        if not isinstance(path_value, str) or not path_value:
            raise CaptionRouteArtifactError(f"diagnostic source lacks {path_key}")
        if (
            not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha)
        ):
            raise CaptionRouteArtifactError(f"diagnostic source has invalid {sha_key}")
        record = _file_record(Path(path_value))
        if record["sha256"] != expected_sha:
            raise CaptionRouteArtifactError(f"diagnostic source {label} SHA-256 drifted")
        records[label] = record
    return records


def _load_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CaptionRouteArtifactError(f"could not read {label} {path}: {error}") from error


def _finite_unit(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CaptionRouteArtifactError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise CaptionRouteArtifactError(f"{label} must be finite in [0,1]")
    return result


def _exact_int(value: Any, *, label: str) -> int:
    if type(value) is not int:
        raise CaptionRouteArtifactError(f"{label} must be an exact integer")
    return int(value)


def _matches(value: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return isinstance(value, (int, float)) and math.isclose(
            float(value), expected, rel_tol=0.0, abs_tol=1e-12
        )
    return value == expected


def _correct_count(acc50: Any, n: int, *, label: str) -> int:
    acc = _finite_unit(acc50, label=label)
    raw = acc * n
    correct = int(round(raw))
    if n <= 0 or correct < 0 or correct > n:
        raise CaptionRouteArtifactError(f"{label} has an invalid denominator")
    if not math.isclose(acc, correct / n, rel_tol=0.0, abs_tol=1e-12):
        raise CaptionRouteArtifactError(
            f"{label} is not an exact integer-correct-count ratio"
        )
    return correct


def _canonical_id_to_caption(path: Path) -> Dict[int, str]:
    payload = _load_json(path, label="canonical classes")
    if not isinstance(payload, list) or not payload:
        raise CaptionRouteArtifactError("canonical classes must be a non-empty list")
    result: Dict[int, str] = {}
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict) or entry.get("id") is None:
            continue
        try:
            class_id = int(entry["id"])
        except (TypeError, ValueError) as error:
            raise CaptionRouteArtifactError(
                f"canonical class row {index} has an invalid id"
            ) from error
        if class_id in result:
            raise CaptionRouteArtifactError(f"duplicate canonical class id {class_id}")
        preferred = None
        for key in ("base_name", "raw_name", "norm_name", "synset"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                preferred = _norm_text(value)
                break
        if preferred:
            result[class_id] = preferred
    if not result:
        raise CaptionRouteArtifactError("canonical classes contain no usable names")
    return result


def _manifest_caption_counts(
    path: Path,
    *,
    expected_n: int,
    canonical_id_to_caption: Mapping[int, str],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    observed = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                observed += 1
                row = json.loads(line)
                instances = row.get("instances") if isinstance(row, dict) else None
                if (
                    not isinstance(instances, list)
                    or len(instances) != 1
                    or not isinstance(instances[0], dict)
                ):
                    raise CaptionRouteArtifactError(
                        f"manifest {path}:{line_number} must contain one instance"
                    )
                instance = instances[0]
                class_id = instance.get("class_id")
                if isinstance(class_id, bool) or not isinstance(class_id, int):
                    raise CaptionRouteArtifactError(
                        f"manifest {path}:{line_number} has no exact integer class_id"
                    )
                normalized = canonical_id_to_caption.get(class_id, "object")
                if not normalized:
                    raise CaptionRouteArtifactError(
                        f"manifest {path}:{line_number} has no canonical caption"
                    )
                counts[normalized] += 1
    except (OSError, json.JSONDecodeError) as error:
        raise CaptionRouteArtifactError(f"could not parse manifest {path}: {error}") from error
    if observed != expected_n:
        raise CaptionRouteArtifactError(
            f"manifest {path} has {observed} rows, expected {expected_n}"
        )
    return counts


def _row_matches_descriptor(row: Mapping[str, Any], descriptor_id: str) -> bool:
    if descriptor_id == DEFAULT_DESCRIPTOR_ID:
        return all(_matches(row.get(key), value) for key, value in BASE_IDENTITY_FIELDS.items())
    descriptor = DESCRIPTOR_REGISTRY[descriptor_id]
    expected = {
        "diagnostic_transfer_mode": descriptor["transfer_mode"],
        "diagnostic_iou_power": descriptor["iou_power"],
        "diagnostic_patch_weight": descriptor["patch_weight"],
        "diagnostic_text_weight": descriptor["text_weight"],
    }
    return all(_matches(row.get(key), value) for key, value in expected.items())


def _select_descriptor_rows(
    rows: Any, *, split: str, expected_n: int, manifest_counts: Counter[str]
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise CaptionRouteArtifactError(f"{split} diagnostic result must be a non-empty row list")
    selected: Dict[str, Dict[str, Any]] = {}
    for descriptor_id in DESCRIPTOR_REGISTRY:
        matches = [row for row in rows if _row_matches_descriptor(row, descriptor_id)]
        if len(matches) != 1:
            raise CaptionRouteArtifactError(
                f"{split} must contain exactly one row for {descriptor_id}, got {len(matches)}"
            )
        selected[descriptor_id] = dict(matches[0])

    identity_fields = (
        "checkpoint",
        "patch_checkpoint_sha256",
        "patch_config",
        "patch_config_sha256",
        "external_gdino_checkpoint",
        "external_gdino_checkpoint_sha256",
        "external_gdino_config",
        "external_gdino_config_sha256",
        "external_gdino_query_count",
        "batch_size",
        "num_workers",
        "seed_protocol",
        "seed",
    )
    base = selected[DEFAULT_DESCRIPTOR_ID]
    common = {field: base.get(field) for field in identity_fields}
    if common["external_gdino_query_count"] != 900:
        raise CaptionRouteArtifactError(f"{split} external query count must be 900")
    if common["batch_size"] != 32 or common["num_workers"] != 4:
        raise CaptionRouteArtifactError(
            f"{split} selection runtime must use batch_size=32 and num_workers=4"
        )
    if common["seed_protocol"] != SEED_PROTOCOL or common["seed"] != EXPECTED_SEEDS[split]:
        raise CaptionRouteArtifactError(f"{split} seed protocol drifted")

    for descriptor_id, row in selected.items():
        label = f"{split}/{descriptor_id}"
        if row.get("diagnostic_only") is not True or row.get("formal_gate_eligible") is not False:
            raise CaptionRouteArtifactError(f"{label} must be explicitly diagnostic-only")
        if row.get("dataset") != split:
            raise CaptionRouteArtifactError(f"{label} dataset drifted")
        if _exact_int(row.get("num_expressions"), label=f"{label}.num_expressions") != expected_n:
            raise CaptionRouteArtifactError(f"{label} is not a full validation result")
        if _exact_int(row.get("max_images"), label=f"{label}.max_images") != 0:
            raise CaptionRouteArtifactError(f"{label} max_images must be zero")
        if _exact_int(row.get("max_batches"), label=f"{label}.max_batches") != 0:
            raise CaptionRouteArtifactError(f"{label} max_batches must be zero")
        for field, expected in common.items():
            if row.get(field) != expected:
                raise CaptionRouteArtifactError(f"{label} source identity field {field} drifted")
        if row.get("canonical_stage_a_caption_contract") != CAPTION_CONTRACT:
            raise CaptionRouteArtifactError(f"{label} canonical-caption contract drifted")
        groups = row.get("by_canonical_stage_a_caption")
        if not isinstance(groups, dict) or not groups:
            raise CaptionRouteArtifactError(f"{label} lacks exact-caption groups")
        observed_counts: Counter[str] = Counter()
        total_correct = 0
        for caption, group in groups.items():
            if not isinstance(caption, str) or not caption or _norm_text(caption) != caption:
                raise CaptionRouteArtifactError(f"{label} has a non-canonical caption key")
            if not isinstance(group, dict):
                raise CaptionRouteArtifactError(f"{label}/{caption} group must be an object")
            n = _exact_int(group.get("num_expressions"), label=f"{label}/{caption}.n")
            observed_counts[caption] = n
            total_correct += _correct_count(
                group.get("acc50"), n, label=f"{label}/{caption}.acc50"
            )
        if observed_counts != manifest_counts:
            raise CaptionRouteArtifactError(
                f"{label} caption counts differ from the bound validation manifest"
            )
        if total_correct != _correct_count(row.get("acc50"), expected_n, label=f"{label}.acc50"):
            raise CaptionRouteArtifactError(f"{label} caption correct counts do not sum")
    return selected, common


def _caption_count(row: Mapping[str, Any], caption: str) -> tuple[int, int]:
    group = row["by_canonical_stage_a_caption"][caption]
    n = int(group["num_expressions"])
    return n, _correct_count(group["acc50"], n, label=f"{caption}.acc50")


def _descriptor_tiebreak(descriptor_id: str) -> tuple[Any, ...]:
    descriptor = DESCRIPTOR_REGISTRY[descriptor_id]
    external_box = descriptor["output_box_source"] in {
        "matched_external_query",
        "assigned_external_query",
    }
    top_query = str(descriptor["transfer_mode"]).startswith("top_query_")
    return (
        0 if external_box else 1,
        0 if top_query else 1,
        float(descriptor["patch_weight"]),
        descriptor_id,
    )


def _candidate_stats(
    *,
    caption: str,
    descriptor_id: str,
    rows_by_split: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Dict[str, Any]:
    per_split: Dict[str, Dict[str, Any]] = {}
    pooled_n = 0
    pooled_gain = 0
    supported_rates: list[float] = []
    supported_meeting_gain = 0
    observed_non_degrading = True
    supported_splits = 0
    for split in VAL_SPLITS:
        base_row = rows_by_split[split][DEFAULT_DESCRIPTOR_ID]
        if caption not in base_row["by_canonical_stage_a_caption"]:
            continue
        n, base_correct = _caption_count(base_row, caption)
        candidate_n, candidate_correct = _caption_count(
            rows_by_split[split][descriptor_id], caption
        )
        if candidate_n != n:
            raise CaptionRouteArtifactError(
                f"{split}/{caption}/{descriptor_id} denominator drifted"
            )
        gain = candidate_correct - base_correct
        supported = n >= int(SELECTION_CONTRACT["supported_split_minimum_expressions"])
        minimum_gain = max(2, int(math.ceil(0.01 * n)))
        meets_gain = supported and gain >= minimum_gain
        pooled_n += n
        pooled_gain += gain
        observed_non_degrading = observed_non_degrading and gain >= 0
        if supported:
            supported_splits += 1
            supported_rates.append(gain / n)
        if meets_gain:
            supported_meeting_gain += 1
        per_split[split] = {
            "n": n,
            "base_correct": base_correct,
            "candidate_correct": candidate_correct,
            "delta_correct": gain,
            "supported": supported,
            "minimum_gain_if_supported": minimum_gain,
            "meets_supported_gain": meets_gain,
        }
    minimum_pooled_gain = max(10, int(math.ceil(0.02 * pooled_n)))
    checks = {
        "minimum_total_expressions": pooled_n >= 300,
        "minimum_supported_splits": supported_splits >= 2,
        "all_observed_splits_non_degrading": observed_non_degrading,
        "minimum_supported_splits_meeting_gain": supported_meeting_gain >= 2,
        "minimum_pooled_gain": pooled_gain >= minimum_pooled_gain,
    }
    return {
        "descriptor_id": descriptor_id,
        "per_split": per_split,
        "pooled_n": pooled_n,
        "pooled_delta_correct": pooled_gain,
        "minimum_pooled_gain": minimum_pooled_gain,
        "supported_splits": supported_splits,
        "supported_splits_meeting_gain": supported_meeting_gain,
        "worst_supported_gain_rate": min(supported_rates) if supported_rates else None,
        "checks": checks,
        "eligible": all(checks.values()),
        "rejection_reasons": sorted(key for key, passed in checks.items() if not passed),
    }


def _choose_candidate(caption: str, candidates: Sequence[Mapping[str, Any]]) -> str:
    eligible = [row for row in candidates if row["eligible"] is True]
    if not eligible:
        return DEFAULT_DESCRIPTOR_ID
    maximum_gain = max(int(row["pooled_delta_correct"]) for row in eligible)
    pooled_n = int(eligible[0]["pooled_n"])
    band = max(2, int(math.ceil(0.005 * pooled_n)))
    equivalent = [
        row
        for row in eligible
        if maximum_gain - int(row["pooled_delta_correct"]) <= band
    ]
    equivalent.sort(
        key=lambda row: (
            -float(row["worst_supported_gain_rate"]),
            _descriptor_tiebreak(str(row["descriptor_id"])),
        )
    )
    if not equivalent:
        raise CaptionRouteArtifactError(f"{caption} candidate selection became empty")
    return str(equivalent[0]["descriptor_id"])


def _selection_payload(
    *,
    result_paths: Mapping[str, Path],
    manifest_paths: Mapping[str, Path],
    canonical_classes_path: Path,
) -> Dict[str, Any]:
    if set(result_paths) != set(VAL_SPLITS) or set(manifest_paths) != set(VAL_SPLITS):
        raise CaptionRouteArtifactError("selection requires the exact three validation splits")
    result_records = {split: _file_record(result_paths[split]) for split in VAL_SPLITS}
    manifest_records = {split: _file_record(manifest_paths[split]) for split in VAL_SPLITS}
    canonical_classes_record = _file_record(canonical_classes_path)
    canonical_id_to_caption = _canonical_id_to_caption(canonical_classes_path)
    rows_by_split: Dict[str, Dict[str, Dict[str, Any]]] = {}
    source_identity: Dict[str, Any] | None = None
    for split in VAL_SPLITS:
        expected_n = EXPECTED_MANIFEST_N[split]
        manifest_counts = _manifest_caption_counts(
            manifest_paths[split],
            expected_n=expected_n,
            canonical_id_to_caption=canonical_id_to_caption,
        )
        rows, common = _select_descriptor_rows(
            _load_json(result_paths[split], label=f"{split} diagnostic result"),
            split=split,
            expected_n=expected_n,
            manifest_counts=manifest_counts,
        )
        comparable = {
            key: value
            for key, value in common.items()
            if key not in {"seed", "seed_protocol"}
        }
        if source_identity is None:
            source_identity = comparable
        elif comparable != source_identity:
            raise CaptionRouteArtifactError("diagnostic source identities differ across splits")
        rows_by_split[split] = rows
        _require_file_unchanged(result_records[split], label=f"{split} result")
        _require_file_unchanged(manifest_records[split], label=f"{split} manifest")

    captions = sorted(
        {
            caption
            for split in VAL_SPLITS
            for caption in rows_by_split[split][DEFAULT_DESCRIPTOR_ID][
                "by_canonical_stage_a_caption"
            ]
        }
    )
    if source_identity is None:
        raise CaptionRouteArtifactError("diagnostic source identity is missing")
    source_component_files = _source_component_records(source_identity)
    caption_decisions: Dict[str, Dict[str, Any]] = {}
    overrides: Dict[str, str] = {}
    for caption in captions:
        candidates = [
            _candidate_stats(
                caption=caption,
                descriptor_id=descriptor_id,
                rows_by_split=rows_by_split,
            )
            for descriptor_id in CANDIDATE_DESCRIPTOR_IDS
        ]
        selected = _choose_candidate(caption, candidates)
        if selected != DEFAULT_DESCRIPTOR_ID:
            overrides[caption] = selected
        caption_decisions[caption] = {
            "selected_descriptor_id": selected,
            "candidates": {row["descriptor_id"]: row for row in candidates},
        }

    route_by_split: Dict[str, Dict[str, Any]] = {}
    strict_improvements = 0
    for split in VAL_SPLITS:
        base_row = rows_by_split[split][DEFAULT_DESCRIPTOR_ID]
        base_correct = 0
        route_correct = 0
        for caption in base_row["by_canonical_stage_a_caption"]:
            n, base_k = _caption_count(base_row, caption)
            descriptor_id = overrides.get(caption, DEFAULT_DESCRIPTOR_ID)
            selected_n, selected_k = _caption_count(
                rows_by_split[split][descriptor_id], caption
            )
            if selected_n != n:
                raise CaptionRouteArtifactError("routed caption denominator drifted")
            base_correct += base_k
            route_correct += selected_k
        n = EXPECTED_MANIFEST_N[split]
        delta = route_correct - base_correct
        if delta < 0:
            raise CaptionRouteArtifactError(f"frozen route regresses {split}")
        strict_improvements += int(delta > 0)
        route_by_split[split] = {
            "n": n,
            "base_correct": base_correct,
            "base_acc50": base_correct / n,
            "route_correct": route_correct,
            "route_acc50": route_correct / n,
            "delta_correct": delta,
            "delta_acc50": delta / n,
        }
    if strict_improvements < 2:
        raise CaptionRouteArtifactError(
            "frozen route must strictly improve at least two validation splits"
        )

    payload = {
        "schema": SCHEMA,
        "kind": KIND,
        "selection_inputs": {
            split: {
                "result": result_records[split],
                "manifest": manifest_records[split],
                "manifest_n": EXPECTED_MANIFEST_N[split],
                "seed": EXPECTED_SEEDS[split],
            }
            for split in VAL_SPLITS
        },
        "canonical_classes": canonical_classes_record,
        "source_identity": source_identity,
        "source_component_files": source_component_files,
        "caption_contract": dict(CAPTION_CONTRACT),
        "manifest_caption_audit_contract": dict(MANIFEST_CAPTION_AUDIT_CONTRACT),
        "descriptor_registry": dict(DESCRIPTOR_REGISTRY),
        "descriptor_registry_sha256": canonical_sha256(DESCRIPTOR_REGISTRY),
        "selection_contract": dict(SELECTION_CONTRACT),
        "selection_contract_sha256": canonical_sha256(SELECTION_CONTRACT),
        "route": {
            "default_descriptor_id": DEFAULT_DESCRIPTOR_ID,
            "overrides": dict(sorted(overrides.items())),
            "override_count": len(overrides),
            "caption_count": len(captions),
        },
        "caption_decisions": caption_decisions,
        "validation_route_summary": route_by_split,
    }
    payload["artifact_identity"] = {
        "algorithm": IDENTITY_ALGORITHM,
        "sha256": canonical_sha256(payload),
    }
    for split in VAL_SPLITS:
        _require_file_unchanged(result_records[split], label=f"{split} result")
        _require_file_unchanged(manifest_records[split], label=f"{split} manifest")
    _require_file_unchanged(canonical_classes_record, label="canonical classes")
    for label, record in source_component_files.items():
        _require_file_unchanged(record, label=f"source {label}")
    return payload


def build_caption_route_artifact(
    *,
    result_paths: Mapping[str, str | Path],
    manifest_paths: Mapping[str, str | Path],
    canonical_classes_path: str | Path,
) -> Dict[str, Any]:
    return _selection_payload(
        result_paths={key: Path(value).expanduser().resolve() for key, value in result_paths.items()},
        manifest_paths={key: Path(value).expanduser().resolve() for key, value in manifest_paths.items()},
        canonical_classes_path=Path(canonical_classes_path).expanduser().resolve(),
    )


def create_caption_route_artifact(
    output: str | Path,
    *,
    result_paths: Mapping[str, str | Path],
    manifest_paths: Mapping[str, str | Path],
    canonical_classes_path: str | Path,
) -> Dict[str, Any]:
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise CaptionRouteArtifactError(f"refusing to overwrite artifact: {output_path}")
    payload = build_caption_route_artifact(
        result_paths=result_paths,
        manifest_paths=manifest_paths,
        canonical_classes_path=canonical_classes_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_path)
        verified = load_and_verify_caption_route_artifact(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise
    if verified != payload:
        output_path.unlink(missing_ok=True)
        raise CaptionRouteArtifactError("serialized artifact failed exact verification")
    return payload


def load_and_verify_caption_route_artifact(path: str | Path) -> Dict[str, Any]:
    artifact_path = Path(path).expanduser().resolve()
    artifact_record = _file_record(artifact_path)
    payload = _load_json(artifact_path, label="caption route artifact")
    if not isinstance(payload, dict):
        raise CaptionRouteArtifactError("caption route artifact must be an object")
    if payload.get("schema") != SCHEMA or payload.get("kind") != KIND:
        raise CaptionRouteArtifactError("unsupported caption route artifact")
    identity = payload.get("artifact_identity")
    unsigned = {key: value for key, value in payload.items() if key != "artifact_identity"}
    if identity != {
        "algorithm": IDENTITY_ALGORITHM,
        "sha256": canonical_sha256(unsigned),
    }:
        raise CaptionRouteArtifactError("caption route artifact identity mismatch")
    inputs = payload.get("selection_inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(VAL_SPLITS):
        raise CaptionRouteArtifactError("caption route artifact inputs drifted")
    rebuilt = build_caption_route_artifact(
        result_paths={split: inputs[split]["result"]["path"] for split in VAL_SPLITS},
        manifest_paths={split: inputs[split]["manifest"]["path"] for split in VAL_SPLITS},
        canonical_classes_path=payload["canonical_classes"]["path"],
    )
    if rebuilt != payload:
        raise CaptionRouteArtifactError(
            "caption route artifact no longer matches its bound validation evidence"
        )
    _require_file_unchanged(artifact_record, label="caption route artifact")
    return payload


def _parse_split_paths(values: Iterable[str], *, label: str) -> Dict[str, Path]:
    parsed: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise CaptionRouteArtifactError(f"{label} must use SPLIT=PATH")
        split, raw_path = value.split("=", 1)
        if split not in VAL_SPLITS or split in parsed or not raw_path:
            raise CaptionRouteArtifactError(f"invalid or duplicate {label} split: {split!r}")
        parsed[split] = Path(raw_path).expanduser().resolve()
    if set(parsed) != set(VAL_SPLITS):
        raise CaptionRouteArtifactError(f"{label} requires the exact three validation splits")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze or verify a full-validation canonical-caption route"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--output", required=True)
    create.add_argument("--result", action="append", required=True, help="SPLIT=PATH")
    create.add_argument("--manifest", action="append", required=True, help="SPLIT=PATH")
    create.add_argument("--canonical-classes", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--artifact", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            payload = create_caption_route_artifact(
                args.output,
                result_paths=_parse_split_paths(args.result, label="result"),
                manifest_paths=_parse_split_paths(args.manifest, label="manifest"),
                canonical_classes_path=args.canonical_classes,
            )
        else:
            payload = load_and_verify_caption_route_artifact(args.artifact)
    except CaptionRouteArtifactError as error:
        print(f"[FAIL] {error}")
        return 2
    print(json.dumps(payload["artifact_identity"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
