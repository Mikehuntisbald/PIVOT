#!/usr/bin/env python3
"""Seal the leakage-free, equal-exposure Stage-B Table-B data matrix.

The builder makes D1-D3 use one explicit paired-row surface while retaining
the strongest claim each source actually supports.  In particular, it never
turns a weak or proposal-covered label into ``global_tn_verified=true``.

D1 and D2 are split by image after removing the union of the two strict
evaluation manifests.  D3 consumes the already sealed single-edit semantic
partition and verifies its upstream split contract.  Every training output is
then exactly ``target_rows`` rows; D1/D2 use deterministic proportional
stratification over their native dataset/edit taxonomy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
STRICT_DIR = (
    REPO_ROOT
    / "data/eval_manifests/stageb_vlm_verified_strict_ann_umd_val_20260711"
)
DEFAULT_D1 = (
    REPO_ROOT
    / "data/ablations/stageb_gdino_adapter_dataft_20260711"
    / "benchmark_dataft_alltn_pairs.jsonl"
)
DEFAULT_D2 = (
    REPO_ROOT
    / "data/ablations/stageb_refexp_three_train_20260711"
    / "refexp_tn_stageb_v1.jsonl"
)
DEFAULT_D3_DIR = (
    REPO_ROOT
    / "data/ablations/stageb_gdino_adapter_semantic_partition_20260717"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "data/ablations/stageb_tn_table_b_equal_exposure_20260717"
)
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_SEED = "20260717-table-b"
DEFAULT_TARGET_ROWS = 14_196
DEFAULT_CALIBRATION_RATIO = Fraction(1, 10)
SCHEMA = "stage-b-paper-table-b-equal-exposure-v1"
PAIR_SCHEMA = "stage-b-paper-table-b-scope-preserving-pair-v1"

OUTPUT_NAMES = {
    "d1_train": "d1_unverified_allneg_train.jsonl",
    "d1_calibration": "d1_unverified_allneg_calibration.jsonl",
    "d2_train": "d2_traceable_edit_train.jsonl",
    "d2_calibration": "d2_traceable_edit_calibration.jsonl",
    "d3_train": "d3_proposal_covered_train.jsonl",
    "d3_calibration": "d3_proposal_covered_calibration.jsonl",
}
CONFIG_NAMES = {
    "D0": "datasets_stageb_table_b_d0_no_tn.json",
    "D1": "datasets_stageb_table_b_d1_unverified_allneg.json",
    "D2": "datasets_stageb_table_b_d2_traceable_edits.json",
    "D3": "datasets_stageb_table_b_d3_proposal_covered.json",
}


class TableBDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceRow:
    line_number: int
    raw: bytes
    value: dict[str, Any]
    image_id: int
    row_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _clean_text(value: Any, *, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TableBDataError(f"{context}: {field} must be non-empty text")
    return " ".join(value.strip().split())


def _image_id(value: Mapping[str, Any], *, context: str) -> int:
    raw = value.get("image_id")
    if isinstance(raw, bool):
        raise TableBDataError(f"{context}: invalid image_id {raw!r}")
    try:
        parsed = int(raw)
    except (TypeError, ValueError) as error:
        raise TableBDataError(f"{context}: invalid image_id {raw!r}") from error
    if parsed < 0 or str(parsed) != str(raw).strip():
        raise TableBDataError(f"{context}: non-canonical image_id {raw!r}")
    return parsed


def load_jsonl(path: Path, *, label: str) -> list[SourceRow]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise TableBDataError(f"missing {label}: {path}")
    rows: list[SourceRow] = []
    for line_number, raw in enumerate(path.read_bytes().splitlines(keepends=True), 1):
        context = f"{path}:{line_number}"
        if not raw.strip():
            raise TableBDataError(f"{context}: blank JSONL row")
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TableBDataError(f"{context}: invalid JSON: {error}") from error
        if not isinstance(value, dict):
            raise TableBDataError(f"{context}: row must be an object")
        rows.append(
            SourceRow(
                line_number=line_number,
                raw=raw,
                value=value,
                image_id=_image_id(value, context=context),
                row_sha256=hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest(),
            )
        )
    if not rows:
        raise TableBDataError(f"empty {label}: {path}")
    return rows


def _file_record(path: Path, rows: Sequence[SourceRow] | None = None) -> dict[str, Any]:
    path = path.resolve()
    record = {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if rows is not None:
        record.update(
            rows=len(rows), unique_images=len({row.image_id for row in rows})
        )
    return record


def _manifest_images(rows: Sequence[SourceRow]) -> set[int]:
    return {row.image_id for row in rows}


def _hash_int(*parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest, "big")


def image_is_calibration(image_id: int, *, seed: str, ratio: Fraction) -> bool:
    return _hash_int(seed, str(image_id)) * ratio.denominator < (
        ratio.numerator * (1 << 256)
    )


def filter_and_split(
    rows: Sequence[SourceRow],
    *,
    strict_images: set[int],
    seed: str,
    calibration_ratio: Fraction,
) -> tuple[list[SourceRow], list[SourceRow], list[SourceRow]]:
    filtered = [row for row in rows if row.image_id in strict_images]
    eligible = [row for row in rows if row.image_id not in strict_images]
    train = [
        row
        for row in eligible
        if not image_is_calibration(
            row.image_id, seed=seed, ratio=calibration_ratio
        )
    ]
    calibration = [
        row
        for row in eligible
        if image_is_calibration(row.image_id, seed=seed, ratio=calibration_ratio)
    ]
    if {row.image_id for row in train} & {row.image_id for row in calibration}:
        raise AssertionError("image-hash split leaked an image across partitions")
    return filtered, train, calibration


def _one_instance(row: SourceRow, *, source: str) -> dict[str, Any]:
    instances = row.value.get("instances")
    if not (
        isinstance(instances, list)
        and len(instances) == 1
        and isinstance(instances[0], dict)
    ):
        raise TableBDataError(
            f"{source} line {row.line_number}: expected exactly one instance"
        )
    instance = instances[0]
    bbox = instance.get("bbox")
    if not (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in bbox)
        and float(bbox[2]) > 0
        and float(bbox[3]) > 0
    ):
        raise TableBDataError(
            f"{source} line {row.line_number}: invalid xywh instance bbox"
        )
    return instance


def validate_d1(row: SourceRow) -> None:
    value = row.value
    instance = _one_instance(row, source="D1")
    required = (
        value.get("benchmark_dataft_alltn") is True
        and value.get("global_tn_verified") is False
        and value.get("proposalset_proxy_verified") is False
        and value.get("tn_scope") == "benchmark_dataft_alltn"
        and instance.get("benchmark_dataft_alltn") is True
        and instance.get("global_tn_verified") is False
        and instance.get("proposalset_proxy_verified") is False
        and instance.get("tn_scope") == "benchmark_dataft_alltn"
    )
    if not required:
        raise TableBDataError(
            f"D1 line {row.line_number}: recovered all-negative scope flags drifted"
        )
    positive = instance.get("positive_phrase", instance.get("phrase"))
    negative = instance.get("negative_phrase", instance.get("try_tn"))
    _clean_text(positive, field="positive_phrase", context=f"D1:{row.line_number}")
    _clean_text(negative, field="negative_phrase", context=f"D1:{row.line_number}")
    if "class_id" not in instance:
        raise TableBDataError(f"D1 line {row.line_number}: missing class_id")


def validate_d2(row: SourceRow) -> None:
    instance = _one_instance(row, source="D2")
    if instance.get("text_is_negative") is not True:
        raise TableBDataError(
            f"D2 line {row.line_number}: source row must be the synthetic negative view"
        )
    if instance.get("try_tn_method") != "synthetic_rule":
        raise TableBDataError(
            f"D2 line {row.line_number}: missing synthetic_rule provenance"
        )
    for field in (
        "positive_phrase",
        "try_tn",
        "replace_from",
        "replace_to",
        "replace_category",
        "try_tn_rule",
    ):
        _clean_text(
            instance.get(field), field=field, context=f"D2:{row.line_number}"
        )
    if _clean_text(
        instance["positive_phrase"], field="positive_phrase", context="D2"
    ).lower() == _clean_text(instance["try_tn"], field="try_tn", context="D2").lower():
        raise TableBDataError(f"D2 line {row.line_number}: edit does not change text")
    if row.value.get("global_tn_verified") is True or instance.get(
        "global_tn_verified"
    ) is True:
        raise TableBDataError(
            f"D2 line {row.line_number}: traceable-only source claims global verification"
        )


def validate_d3(row: SourceRow) -> None:
    value = row.value
    for field in ("sample_id", "sent", "try_tn", "dataset"):
        _clean_text(value.get(field), field=field, context=f"D3:{row.line_number}")
    edits = value.get("tn_edits")
    if not (isinstance(edits, list) and len(edits) == 1 and isinstance(edits[0], dict)):
        raise TableBDataError(
            f"D3 line {row.line_number}: expected provenance-certified single edit"
        )
    edit = edits[0]
    expected = {
        "replace_category": [edit.get("category")],
        "replace_from": [edit.get("replace_from")],
        "replace_to": [edit.get("replace_to")],
        "replace_span": [edit.get("replace_span")],
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise TableBDataError(
            f"D3 line {row.line_number}: top-level edit provenance drifted"
        )
    if not (
        value.get("cached_proposal_coverage_only") is True
        and value.get("all_900_gdino_queries_verified") is False
        and value.get("global_max_label_is_semantic_extrapolation") is True
    ):
        raise TableBDataError(
            f"D3 line {row.line_number}: proposal-coverage limitations are missing"
        )


def _basename(row: Mapping[str, Any]) -> str:
    value = row.get("file_name", row.get("filename", ""))
    return Path(str(value)).name


def _identity_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in ("image_id", "ann_id", "ref_id", "sent_id", "split")
        if key in value
    }


def convert_d1(row: SourceRow) -> dict[str, Any]:
    validate_d1(row)
    value = row.value
    instance = _one_instance(row, source="D1")
    positive = _clean_text(
        instance.get("positive_phrase", instance.get("phrase")),
        field="positive_phrase",
        context=f"D1:{row.line_number}",
    )
    negative = _clean_text(
        instance.get("negative_phrase", instance.get("try_tn")),
        field="negative_phrase",
        context=f"D1:{row.line_number}",
    )
    return {
        "table_b_pair_schema": PAIR_SCHEMA,
        "table_b_id": "D1",
        "sample_id": value.get(
            "sample_id", f"table-b-d1:{row.image_id}:{row.line_number}"
        ),
        **_identity_fields(value),
        "file_name": _basename(value),
        "dataset": value.get("pair_source", value.get("source", "unknown")),
        "pair_source": value.get("pair_source"),
        "class_id": instance["class_id"],
        "category_name": instance.get(
            "canonical_name", instance.get("head", instance.get("head_phrase"))
        ),
        "target_bbox_used": [float(v) for v in instance["bbox"]],
        "sent": positive,
        "try_tn": negative,
        "try_tn_head": instance.get("try_tn_head", instance.get("head")),
        "try_tn_head_phrase": positive,
        "replace_category": instance.get("replace_category"),
        "replace_from": instance.get("replace_from"),
        "replace_to": instance.get("replace_to"),
        "tn_scope": "unverified_all_negative",
        "global_tn_verified": False,
        "proposal_covered_verified": False,
        "visual_verified_negative": False,
        "proposalset_proxy_verified": False,
        "benchmark_dataft_alltn": True,
        "traceable_counterfactual_edit": False,
        "source_provenance": {
            "path_role": "recovered_benchmark_dataft_alltn",
            "line_number": row.line_number,
            "row_sha256": row.row_sha256,
            "source": value.get("source"),
            "source_tn_scope": value.get("tn_scope"),
            "source_global_tn_verified": value.get("global_tn_verified"),
        },
    }


def convert_d2(row: SourceRow) -> dict[str, Any]:
    validate_d2(row)
    value = row.value
    instance = _one_instance(row, source="D2")
    positive = _clean_text(
        instance["positive_phrase"],
        field="positive_phrase",
        context=f"D2:{row.line_number}",
    )
    negative = _clean_text(
        instance["try_tn"], field="try_tn", context=f"D2:{row.line_number}"
    )
    sample_id = (
        f"table-b-d2:{row.image_id}:{value.get('ann_id', 'na')}:"
        f"{value.get('ref_id', 'na')}:{value.get('sent_id', 'na')}:"
        f"{row.line_number}"
    )
    return {
        "table_b_pair_schema": PAIR_SCHEMA,
        "table_b_id": "D2",
        "sample_id": sample_id,
        **_identity_fields(value),
        "file_name": _basename(value),
        "dataset": value.get("source", instance.get("pair_source", "unknown")),
        "pair_source": instance.get("pair_source", value.get("source")),
        "class_id": instance["class_id"],
        "category_name": instance.get(
            "category_name",
            instance.get("canonical_name", instance.get("head")),
        ),
        "target_bbox_used": [float(v) for v in instance["bbox"]],
        "sent": positive,
        "try_tn": negative,
        "try_tn_head": instance.get("try_tn_head", instance.get("head")),
        "try_tn_head_phrase": positive,
        "replace_category": instance["replace_category"],
        "replace_from": instance["replace_from"],
        "replace_to": instance["replace_to"],
        "replace_token": instance.get("replace_token"),
        "try_tn_method": instance["try_tn_method"],
        "try_tn_rule": instance["try_tn_rule"],
        "tn_scope": "traceable_counterfactual_edit",
        "global_tn_verified": False,
        "proposal_covered_verified": False,
        "visual_verified_negative": False,
        "proposalset_proxy_verified": False,
        "benchmark_dataft_alltn": False,
        "traceable_counterfactual_edit": True,
        "source_provenance": {
            "path_role": "synthetic_edit_before_visual_filtering",
            "line_number": row.line_number,
            "row_sha256": row.row_sha256,
            "source": value.get("source"),
            "try_tn_method": instance["try_tn_method"],
            "try_tn_rule": instance["try_tn_rule"],
        },
    }


def convert_d3(row: SourceRow) -> dict[str, Any]:
    validate_d3(row)
    value = row.value
    # The source's legacy `global_tn_verified=true` name is stronger than its
    # own audit permits.  Preserve that fact under source_provenance, while the
    # Table-B runtime-facing label uses the actual proposal-covered scope.
    output = {
        key: value[key]
        for key in (
            "sample_id",
            "image_id",
            "ann_id",
            "ref_id",
            "sent_id",
            "split",
            "dataset",
            "pair_source",
            "class_id",
            "category_name",
            "class_norm_name",
            "target_bbox_used",
            "sent",
            "try_tn",
            "try_tn_head",
            "try_tn_head_phrase",
            "replace_category",
            "replace_from",
            "replace_to",
            "replace_span",
            "tn_edits",
            "proposal_count",
            "coverage_policy",
            "verification_contract",
        )
        if key in value
    }
    output.update(
        {
            "table_b_pair_schema": PAIR_SCHEMA,
            "table_b_id": "D3",
            "file_name": _basename(value),
            "tn_scope": "proposal_covered_verified",
            "global_tn_verified": False,
            "proposal_covered_verified": True,
            "visual_verified_negative": True,
            "proposalset_proxy_verified": False,
            "benchmark_dataft_alltn": False,
            "traceable_counterfactual_edit": True,
            "cached_proposal_coverage_only": True,
            "all_900_gdino_queries_verified": False,
            "global_max_label_is_semantic_extrapolation": True,
            "source_provenance": {
                "path_role": "proposal_covered_semantic_partition",
                "line_number": row.line_number,
                "row_sha256": row.row_sha256,
                "source": value.get("source"),
                "source_tn_scope": value.get("tn_scope"),
                "source_global_tn_verified": value.get("global_tn_verified"),
                "source_row_sha256": value.get("source_row_sha256"),
            },
        }
    )
    return output


def _taxonomy_key(value: Mapping[str, Any]) -> str:
    category = value.get("replace_category", "__missing__")
    if isinstance(category, list):
        category = "|".join(str(item) for item in category)
    return " ".join(str(category).strip().lower().split()) or "__missing__"


def _dataset_key(value: Mapping[str, Any]) -> str:
    raw = value.get("dataset", value.get("pair_source", "__missing__"))
    return " ".join(str(raw).strip().lower().split()) or "__missing__"


def _row_priority(value: Mapping[str, Any], *, seed: str, source_id: str) -> int:
    identity = str(value.get("sample_id", ""))
    return _hash_int(seed, source_id, identity)


def proportional_stratified_sample(
    values: Sequence[dict[str, Any]],
    *,
    target_rows: int,
    seed: str,
    source_id: str,
) -> list[dict[str, Any]]:
    if target_rows <= 0:
        raise TableBDataError("target_rows must be positive")
    if len(values) < target_rows:
        raise TableBDataError(
            f"{source_id} has only {len(values)} training rows, below target {target_rows}"
        )
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for value in values:
        groups[(_dataset_key(value), _taxonomy_key(value))].append(value)
    total = len(values)
    quotas: dict[tuple[str, str], int] = {}
    remainders = []
    allocated = 0
    for key, group in sorted(groups.items()):
        exact = Fraction(target_rows * len(group), total)
        floor = exact.numerator // exact.denominator
        quotas[key] = floor
        allocated += floor
        remainders.append((exact - floor, key))
    for _remainder, key in sorted(remainders, key=lambda item: (-item[0], item[1]))[
        : target_rows - allocated
    ]:
        quotas[key] += 1

    chosen: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        ordered = sorted(
            group,
            key=lambda value: (
                _row_priority(value, seed=seed, source_id=source_id),
                str(value.get("sample_id", "")),
            ),
        )
        chosen.extend(ordered[: quotas[key]])
    if len(chosen) != target_rows:
        raise AssertionError("stratified allocation did not hit the requested size")
    return sorted(
        chosen,
        key=lambda value: (
            _row_priority(value, seed=seed, source_id=source_id + "-output"),
            str(value.get("sample_id", "")),
        ),
    )


def sample_to_taxonomy_targets(
    values: Sequence[dict[str, Any]],
    *,
    taxonomy_targets: Mapping[str, int],
    seed: str,
    source_id: str,
) -> list[dict[str, Any]]:
    """Match a declared taxonomy exactly, preserving native dataset mix within it."""
    by_taxonomy_dataset: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for value in values:
        by_taxonomy_dataset[_taxonomy_key(value)][_dataset_key(value)].append(value)

    chosen: list[dict[str, Any]] = []
    for taxonomy, target in sorted(taxonomy_targets.items()):
        dataset_groups = by_taxonomy_dataset.get(taxonomy, {})
        capacity = sum(len(group) for group in dataset_groups.values())
        if capacity < target:
            raise TableBDataError(
                f"{source_id} taxonomy {taxonomy!r} has capacity {capacity}, "
                f"below matched target {target}"
            )
        quotas: dict[str, int] = {}
        remainders = []
        allocated = 0
        for dataset, group in sorted(dataset_groups.items()):
            exact = Fraction(target * len(group), capacity)
            floor = exact.numerator // exact.denominator
            quotas[dataset] = floor
            allocated += floor
            remainders.append((exact - floor, dataset))
        for _remainder, dataset in sorted(
            remainders, key=lambda item: (-item[0], item[1])
        )[: target - allocated]:
            quotas[dataset] += 1
        for dataset, group in sorted(dataset_groups.items()):
            ordered = sorted(
                group,
                key=lambda value: (
                    _row_priority(
                        value,
                        seed=seed,
                        source_id=f"{source_id}:{taxonomy}:{dataset}",
                    ),
                    str(value.get("sample_id", "")),
                ),
            )
            chosen.extend(ordered[: quotas[dataset]])

    expected = sum(int(value) for value in taxonomy_targets.values())
    if len(chosen) != expected:
        raise AssertionError("matched-taxonomy selection did not hit requested size")
    return sorted(
        chosen,
        key=lambda value: (
            _row_priority(value, seed=seed, source_id=source_id + "-output"),
            str(value.get("sample_id", "")),
        ),
    )


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temp.open("wb") as handle:
        for value in values:
            handle.write(canonical_bytes(value))
    os.replace(temp, path)


def _distribution(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dataset = Counter(_dataset_key(value) for value in values)
    taxonomy = Counter(_taxonomy_key(value) for value in values)
    images = {int(value["image_id"]) for value in values}
    return {
        "rows": len(values),
        "unique_images": len(images),
        "dataset_rows": dict(sorted(dataset.items())),
        "taxonomy_rows": dict(sorted(taxonomy.items())),
    }


def _output_record(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path, label=path.name)
    return _file_record(path, rows)


def _positive_dataset(anno: str) -> dict[str, Any]:
    return {
        "dataset_mode": "patch_episode",
        "root": "/",
        "anno": anno,
        "box_format": "xywh",
        "canonical_classes_json": "${DATA_ROOT}/canonical_classes_with_aliases.json",
        "support_patch_tsv": "${DATA_ROOT}/patches_quality_emb/emb_index_from_quality.tsv",
        "support_patch_bucket": "clean",
        "support_patch_use_embedding": False,
        "support_patch_image_root": "${DATA_ROOT}/patches_quality",
        "support_patch_max_per_class": 200,
        "patch_emb_cache_size": 4096,
        "keep_only_support_gt": True,
        "support_min_count": 1,
        "support_patch_size": 224,
        "build_text_token_masks": True,
        "text_mask_warn_limit": 0,
        "text_mask_skip_invalid_canonical": False,
        "tn_balance_sampling": False,
        "mix_weight": 1.0,
    }


def _tn_dataset(anno: str, *, table_b_id: str, scope: str, audit: str) -> dict[str, Any]:
    return {
        "dataset_mode": "patch_episode",
        "source": "sam3_tn_pair",
        "root": "${DATA_ROOT}/COCO/coco2014/train2014",
        "sam3_tn_image_root": "${DATA_ROOT}/COCO/coco2014/train2014",
        "anno": anno,
        "box_format": "xywh",
        "sam3_tn_bbox_key": "target_bbox_used",
        "canonical_classes_json": "${DATA_ROOT}/canonical_classes_with_aliases.json",
        "support_patch_tsv": "${DATA_ROOT}/patches_quality_emb/emb_index_from_quality.tsv",
        "support_patch_bucket": "clean",
        "support_patch_use_embedding": False,
        "support_patch_image_root": "${DATA_ROOT}/patches_quality",
        "support_patch_max_per_class": 200,
        "patch_emb_cache_size": 4096,
        "keep_only_support_gt": True,
        "neg_episode_prob": 0.0,
        "support_min_count": 1,
        "support_patch_size": 224,
        "build_text_token_masks": True,
        "text_mask_warn_limit": 0,
        "text_mask_skip_invalid_canonical": False,
        "tn_balance_sampling": False,
        "require_global_tn_verified": False,
        "require_single_edit_token_provenance": False,
        "paper_table_b_id": table_b_id,
        "paper_tn_scope": scope,
        "paper_contract_audit": audit,
        "mix_weight": 3.0,
    }


def _repo_alias(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return str(path.resolve())
    return "/home/user/PIVOT/" + relative.as_posix()


def write_dataset_configs(
    *, output_dir: Path, config_dir: Path, audit_path: Path
) -> dict[str, dict[str, Any]]:
    positive_annos = [
        "/home/user/PIVOT/data/ablations/stageb_refexp_three_train_20260711/refcoco_stageb_phrase_v1.jsonl",
        "/home/user/PIVOT/data/ablations/stageb_refexp_three_train_20260711/refcocoplus_stageb_phrase_v1.jsonl",
        "/home/user/PIVOT/data/ablations/stageb_refexp_three_train_20260711/refcocog_stageb_phrase_v1.jsonl",
    ]
    positives = [_positive_dataset(path) for path in positive_annos]
    audit_alias = _repo_alias(audit_path)
    specs: dict[str, list[dict[str, Any]]] = {"D0": [dict(x) for x in positives]}
    for table_b_id, output_key, scope in (
        ("D1", "d1_train", "unverified_all_negative"),
        ("D2", "d2_train", "traceable_counterfactual_edit"),
        ("D3", "d3_train", "proposal_covered_verified"),
    ):
        specs[table_b_id] = [dict(x) for x in positives] + [
            _tn_dataset(
                _repo_alias(output_dir / OUTPUT_NAMES[output_key]),
                table_b_id=table_b_id,
                scope=scope,
                audit=audit_alias,
            )
        ]

    records: dict[str, dict[str, Any]] = {}
    config_dir.mkdir(parents=True, exist_ok=True)
    for table_b_id, train in specs.items():
        path = config_dir / CONFIG_NAMES[table_b_id]
        payload = {"train": train, "val": []}
        temp = path.with_name(path.name + f".tmp-{os.getpid()}")
        temp.write_bytes(
            (
                json.dumps(payload, ensure_ascii=True, sort_keys=False, indent=2)
                + "\n"
            ).encode("ascii")
        )
        os.replace(temp, path)
        records[table_b_id] = _file_record(path)
    return records


def _upstream_d3_contract(
    audit_path: Path, train_path: Path, calibration_path: Path
) -> dict[str, Any]:
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TableBDataError(f"invalid D3 partition audit {audit_path}: {error}") from error
    if audit.get("schema") != "stage-b-semantic-tn-leakage-isolated-partition-v1":
        raise TableBDataError("D3 partition audit has an unexpected schema")
    invariants = audit.get("invariants", {})
    required = (
        invariants.get("eligible_strict_union_image_overlap") == 0
        and invariants.get("train_calibration_image_overlap") == 0
        and invariants.get("single_edit_invalid_metadata_rows_excluded") is True
    )
    if not required:
        raise TableBDataError("D3 upstream partition invariants are incomplete")
    outputs = audit.get("outputs", {})
    for key, path in (
        ("single_edit_train", train_path),
        ("single_edit_calibration", calibration_path),
    ):
        record = outputs.get(key, {})
        if record.get("sha256") != sha256_file(path):
            raise TableBDataError(f"D3 upstream {key} hash drift")
    return {
        "path": str(audit_path.resolve()),
        "sha256": sha256_file(audit_path),
        "schema": audit["schema"],
        "partition_contract_sha256": audit.get("partition_contract_sha256"),
        "partition_seed": audit.get("policy", {}).get("seed"),
        "calibration_ratio": audit.get("policy", {}).get("calibration_ratio"),
    }


def build_matrix(
    *,
    d1_path: Path,
    d2_path: Path,
    d3_train_path: Path,
    d3_calibration_path: Path,
    d3_audit_path: Path,
    strict2031_path: Path,
    strict1607_path: Path,
    output_dir: Path,
    config_dir: Path,
    seed: str = DEFAULT_SEED,
    target_rows: int = DEFAULT_TARGET_ROWS,
    calibration_ratio: Fraction | str = DEFAULT_CALIBRATION_RATIO,
) -> dict[str, Any]:
    if isinstance(calibration_ratio, str):
        calibration_ratio = Fraction(calibration_ratio)
    if not 0 < calibration_ratio < 1:
        raise TableBDataError("calibration_ratio must be strictly between zero and one")

    strict2031 = load_jsonl(strict2031_path, label="strict2031")
    strict1607 = load_jsonl(strict1607_path, label="strict1607")
    strict_images = _manifest_images(strict2031) | _manifest_images(strict1607)
    d1_rows = load_jsonl(d1_path, label="D1 source")
    d2_rows = load_jsonl(d2_path, label="D2 source")
    d3_train_rows = load_jsonl(d3_train_path, label="D3 single-edit train")
    d3_cal_rows = load_jsonl(
        d3_calibration_path, label="D3 single-edit calibration"
    )
    d3_upstream = _upstream_d3_contract(
        d3_audit_path, d3_train_path, d3_calibration_path
    )

    for row in d1_rows:
        validate_d1(row)
    for row in d2_rows:
        validate_d2(row)
    for row in d3_train_rows + d3_cal_rows:
        validate_d3(row)

    d1_filtered, d1_train_pool, d1_cal_pool = filter_and_split(
        d1_rows,
        strict_images=strict_images,
        seed=seed,
        calibration_ratio=calibration_ratio,
    )
    d2_filtered, d2_train_pool, d2_cal_pool = filter_and_split(
        d2_rows,
        strict_images=strict_images,
        seed=seed,
        calibration_ratio=calibration_ratio,
    )

    d3_train_images = {row.image_id for row in d3_train_rows}
    d3_cal_images = {row.image_id for row in d3_cal_rows}
    if d3_train_images & strict_images or d3_cal_images & strict_images:
        raise TableBDataError("D3 upstream partition overlaps a strict manifest")
    if d3_train_images & d3_cal_images:
        raise TableBDataError("D3 upstream train/calibration images overlap")
    if len(d3_train_rows) != target_rows:
        raise TableBDataError(
            f"D3 single-edit train has {len(d3_train_rows)} rows, expected {target_rows}; "
            "changing the target requires a separately sealed D3 subset"
        )

    d1_train_pool_values = [convert_d1(row) for row in d1_train_pool]
    d2_train_pool_values = [convert_d2(row) for row in d2_train_pool]
    d1_train_values = proportional_stratified_sample(
        d1_train_pool_values,
        target_rows=target_rows,
        seed=seed,
        source_id="D1",
    )
    d1_taxonomy_targets = Counter(_taxonomy_key(value) for value in d1_train_values)
    d1_cal_values = [convert_d1(row) for row in d1_cal_pool]
    d2_train_values = sample_to_taxonomy_targets(
        d2_train_pool_values,
        taxonomy_targets=d1_taxonomy_targets,
        seed=seed,
        source_id="D2",
    )
    d2_cal_values = [convert_d2(row) for row in d2_cal_pool]
    d3_train_values = [convert_d3(row) for row in d3_train_rows]
    d3_cal_values = [convert_d3(row) for row in d3_cal_rows]

    outputs = {
        "d1_train": d1_train_values,
        "d1_calibration": d1_cal_values,
        "d2_train": d2_train_values,
        "d2_calibration": d2_cal_values,
        "d3_train": d3_train_values,
        "d3_calibration": d3_cal_values,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, values in outputs.items():
        _write_jsonl(output_dir / OUTPUT_NAMES[key], values)

    audit_path = output_dir / "audit.json"
    audit: dict[str, Any] = {
        "schema": SCHEMA,
        "kind": "completed_table_b_data_matrix",
        "seed": seed,
        "target_train_rows_per_tn_source": target_rows,
        "split_contract": {
            "group_key": "canonical integer image_id",
            "strict_filter": "remove strict2031 union strict1607 before split",
            "assignment": "sha256(seed || NUL || image_id) threshold",
            "calibration_ratio": {
                "numerator": calibration_ratio.numerator,
                "denominator": calibration_ratio.denominator,
            },
            "selection": (
                "D1 uses Hamilton proportional allocation over native "
                "(dataset, edit taxonomy); D2 exactly matches D1 edit-taxonomy "
                "counts while retaining its native dataset proportions inside each "
                "taxonomy; all selections use seeded row-hash ordering"
            ),
            "D3_exception": (
                "D3 reuses its separately sealed upstream image-level split; "
                "see sources.D3.upstream_partition_audit for its seed and ratio"
            ),
        },
        "strict_manifests": {
            "strict2031": _file_record(strict2031_path, strict2031),
            "strict1607": _file_record(strict1607_path, strict1607),
            "union_unique_images": len(strict_images),
        },
        "sources": {
            "D1": {
                "input": _file_record(d1_path, d1_rows),
                "verification_scope": "unverified all-negative",
                "allowed_claim": "recovered all-negative data-FT TN",
                "forbidden_claims": ["true negative", "image-global verified"],
                "filtered_strict_rows": len(d1_filtered),
                "train_pool": _distribution(d1_train_pool_values),
                "calibration_pool": _distribution(d1_cal_values),
                "selected_train": _distribution(d1_train_values),
            },
            "D2": {
                "input": _file_record(d2_path, d2_rows),
                "verification_scope": "traceable counterfactual edit only",
                "allowed_claim": "traceable synthetic edit before visual filtering",
                "forbidden_claims": ["true negative", "visually absent", "image-global verified"],
                "filtered_strict_rows": len(d2_filtered),
                "train_pool": _distribution(d2_train_pool_values),
                "calibration_pool": _distribution(d2_cal_values),
                "selected_train": _distribution(d2_train_values),
                "pair_conversion": (
                    "positive_phrase becomes slot 0; try_tn becomes slot 1; "
                    "synthetic rule/from/to/category and source row hash are retained"
                ),
            },
            "D3": {
                "input_train": _file_record(d3_train_path, d3_train_rows),
                "input_calibration": _file_record(d3_calibration_path, d3_cal_rows),
                "upstream_partition_audit": d3_upstream,
                "verification_scope": "target plus cached proposals (proposal-covered)",
                "allowed_claim": "proposal-covered verified TN",
                "forbidden_claims": [
                    "whole-image absence",
                    "all 900 queries verified",
                    "deployed Stage-A Top-50 exact",
                ],
                "selected_train": _distribution(d3_train_values),
                "calibration_pool": _distribution(d3_cal_values),
                "scope_correction": (
                    "legacy source global_tn_verified is retained only under source_provenance; "
                    "runtime-facing global_tn_verified remains false"
                ),
            },
        },
        "outputs": {
            key: _output_record(output_dir / OUTPUT_NAMES[key]) for key in outputs
        },
        "sampling_contract": {
            "positive_sources": [
                {"dataset": "refcoco", "mix_weight": 1.0},
                {"dataset": "refcocoplus", "mix_weight": 1.0},
                {"dataset": "refcocog", "mix_weight": 1.0},
            ],
            "tn_mix_weight": 3.0,
            "expected_tn_draw_fraction_D1_D3": 0.5,
            "tn_balance_sampling": False,
            "fairness_note": (
                "D1-D3 have identical TN row count and sampler mass. Native taxonomies "
                "differ: D1 and D2 are exactly matched on their common color/size/spatial "
                "taxonomy, while D3 retains its broader sealed semantic taxonomy. "
                "Unsupported D3 categories are not fabricated in D1/D2."
            ),
        },
        "v19_loader_audit": {
            "paired_dataset_surface": "compatible after D1/D2 conversion",
            "base_v19_decoupled_confidence_status": (
                "blocked for D1-D3 truthful scopes unless the separately "
                "implemented fail-closed Table-B switch is enabled"
            ),
            "exact_blocker": (
                "engine.py checks paired_tn & ~global_tn_verified and raises: "
                "'Stage B score-confidence training received paired TN rows without "
                "global_tn_verified=True'"
            ),
            "why_not_bypass": (
                "Setting global_tn_verified=true would silently upgrade unverified or "
                "proposal-covered labels and invalidate Table B."
            ),
            "proposed_fail_closed_switch": {
                "name": "stage_b_v19_allow_scope_labeled_tn_ablation",
                "default": False,
                "required_scope_allowlist": [
                    "unverified_all_negative",
                    "traceable_counterfactual_edit",
                    "proposal_covered_verified",
                ],
                "required_bindings": [
                    "paper Table-B experiment id D1/D2/D3",
                    "this audit SHA-256",
                    "row table_b_id and tn_scope match the selected dataset config",
                ],
                "runtime_semantics": (
                    "create a separate confidence_ablation_eligible mask; never mutate or "
                    "alias global_tn_verified"
                ),
                "fail_closed_on": [
                    "missing/unknown scope",
                    "audit hash drift",
                    "mixed Table-B IDs",
                    "use outside the declared ablation block",
                ],
            },
        },
        "invariants": {
            "equal_train_rows_D1_D3": len(d1_train_values)
            == len(d2_train_values)
            == len(d3_train_values)
            == target_rows,
            "strict_union_overlap_D1_D3": {
                key: len({int(value["image_id"]) for value in outputs[key]} & strict_images)
                for key in ("d1_train", "d1_calibration", "d2_train", "d2_calibration", "d3_train", "d3_calibration")
            },
            "train_calibration_image_overlap": {
                table_id: len(
                    {int(value["image_id"]) for value in outputs[f"{table_id}_train"]}
                    & {
                        int(value["image_id"])
                        for value in outputs[f"{table_id}_calibration"]
                    }
                )
                for table_id in ("d1", "d2", "d3")
            },
            "runtime_global_verified_true_rows": {
                key: sum(value.get("global_tn_verified") is True for value in values)
                for key, values in outputs.items()
            },
            "scope_preserved_without_global_upgrade": True,
            "D1_D2_taxonomy_counts_equal": (
                _distribution(d1_train_values)["taxonomy_rows"]
                == _distribution(d2_train_values)["taxonomy_rows"]
            ),
        },
    }
    if not audit["invariants"]["equal_train_rows_D1_D3"]:
        raise AssertionError("equal-size invariant failed")
    if any(audit["invariants"]["strict_union_overlap_D1_D3"].values()):
        raise AssertionError("strict leakage invariant failed")
    if any(audit["invariants"]["train_calibration_image_overlap"].values()):
        raise AssertionError("train/calibration leakage invariant failed")
    if any(audit["invariants"]["runtime_global_verified_true_rows"].values()):
        raise AssertionError("a Table-B row was upgraded to global verification")
    if not audit["invariants"]["D1_D2_taxonomy_counts_equal"]:
        raise AssertionError("D1/D2 common taxonomy matching failed")

    audit_path.write_bytes(
        (json.dumps(audit, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode(
            "ascii"
        )
    )
    config_records = write_dataset_configs(
        output_dir=output_dir, config_dir=config_dir, audit_path=audit_path
    )
    audit["dataset_configs"] = config_records
    audit_path.write_bytes(
        (json.dumps(audit, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode(
            "ascii"
        )
    )
    return audit


def verify_matrix(*, audit_path: Path) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema") != SCHEMA:
        raise TableBDataError("unexpected Table-B audit schema")
    for section in ("outputs", "dataset_configs"):
        for label, record in audit.get(section, {}).items():
            path = Path(record["path"])
            if sha256_file(path) != record["sha256"]:
                raise TableBDataError(f"{section}.{label} hash drift")
    if audit.get("invariants", {}).get("equal_train_rows_D1_D3") is not True:
        raise TableBDataError("equal-size invariant was not sealed")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d1", type=Path, default=DEFAULT_D1)
    parser.add_argument("--d2", type=Path, default=DEFAULT_D2)
    parser.add_argument(
        "--d3-train", type=Path, default=DEFAULT_D3_DIR / "single_edit_train.jsonl"
    )
    parser.add_argument(
        "--d3-calibration",
        type=Path,
        default=DEFAULT_D3_DIR / "single_edit_calibration.jsonl",
    )
    parser.add_argument(
        "--d3-audit", type=Path, default=DEFAULT_D3_DIR / "audit.json"
    )
    parser.add_argument(
        "--strict2031", type=Path, default=STRICT_DIR / "eval_manifest.jsonl"
    )
    parser.add_argument(
        "--strict1607",
        type=Path,
        default=STRICT_DIR / "semantic_stageb_union_image_disjoint_manifest.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--target-rows", type=int, default=DEFAULT_TARGET_ROWS)
    parser.add_argument("--calibration-ratio", default="1/10")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify:
        audit = verify_matrix(audit_path=args.output_dir / "audit.json")
    else:
        audit = build_matrix(
            d1_path=args.d1,
            d2_path=args.d2,
            d3_train_path=args.d3_train,
            d3_calibration_path=args.d3_calibration,
            d3_audit_path=args.d3_audit,
            strict2031_path=args.strict2031,
            strict1607_path=args.strict1607,
            output_dir=args.output_dir,
            config_dir=args.config_dir,
            seed=args.seed,
            target_rows=args.target_rows,
            calibration_ratio=args.calibration_ratio,
        )
    print(
        json.dumps(
            {
                "schema": audit["schema"],
                "audit": str((args.output_dir / "audit.json").resolve()),
                "equal_train_rows": audit["invariants"]["equal_train_rows_D1_D3"],
                "v19_status": audit["v19_loader_audit"].get(
                    "base_v19_decoupled_confidence_status",
                    audit["v19_loader_audit"].get(
                        "current_decoupled_confidence_status"
                    ),
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
