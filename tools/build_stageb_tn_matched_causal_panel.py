#!/usr/bin/env python3
"""Build the class-audited parent-matched D2m/D3m causal panel for C2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from tools.build_stageb_tn_data_ablation_matrix import (
    DEFAULT_D2,
    DEFAULT_D3_DIR,
    REPO_ROOT,
    SourceRow,
    _file_record,
    _positive_dataset,
    _repo_alias,
    _tn_dataset,
    canonical_bytes,
    convert_d2,
    convert_d3,
    load_jsonl,
    sha256_file,
    validate_d2,
    validate_d3,
)


LEGACY_SCHEMA = "stage-b-paper-c2-parent-matched-tn-v1"
SCHEMA = "stage-b-paper-c2-parent-matched-tn-v2"
PAIR_SCHEMA = "stage-b-paper-c2-parent-matched-pair-v2"
PRIMARY_CAUSAL_STRATUM = "class_aligned_identical_complete_input"
DEFAULT_SEED = "20260717-c2-parent-match"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "data/ablations/stageb_tn_c2_parent_matched_class_aligned_20260718_v2"
)
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"
STRICT_DIR = (
    REPO_ROOT
    / "data/eval_manifests/stageb_vlm_verified_strict_ann_umd_val_20260711"
)
OUTPUT_NAMES = {
    "d2m_train": "d2m_traceable_edit_train.jsonl",
    "d3m_train": "d3m_proposal_covered_train.jsonl",
    "pairs_train": "matched_pairs_train.jsonl",
    "d2m_calibration": "d2m_traceable_edit_calibration.jsonl",
    "d3m_calibration": "d3m_proposal_covered_calibration.jsonl",
    "pairs_calibration": "matched_pairs_calibration.jsonl",
}
CONFIG_NAMES = {
    "D2m": "datasets_stageb_table_b_d2m_matched_class_aligned_v2_traceable.json",
    "D3m": (
        "datasets_stageb_table_b_d3m_matched_class_aligned_v2_"
        "proposal_covered.json"
    ),
}
DATASET_ALIASES = {
    "refcoco+_unc_train": "refcocoplus",
    "refcocoplus": "refcocoplus",
    "refcocog_umd_train": "refcocog",
    "refcocog": "refcocog",
    "refcoco_unc_train": "refcoco",
    "refcoco": "refcoco",
}


class MatchedPanelError(RuntimeError):
    pass


def _exact_class_id(value: Any, *, context: str) -> int:
    if type(value) is not int or value < 0:
        raise MatchedPanelError(
            f"{context}: canonical class_id must be an exact non-negative integer"
        )
    return value


def _d2_class_id(row: SourceRow) -> int:
    validate_d2(row)
    return _exact_class_id(
        row.value["instances"][0].get("class_id"),
        context=f"D2:{row.line_number}",
    )


def _d3_class_id(row: SourceRow) -> int:
    validate_d3(row)
    return _exact_class_id(
        row.value.get("class_id"), context=f"D3:{row.line_number}"
    )


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _dataset(value: Any) -> str:
    normalized = _norm(value)
    return DATASET_ALIASES.get(normalized, normalized)


def _single_category(value: Any, *, context: str) -> str:
    if isinstance(value, list):
        if len(value) != 1:
            raise MatchedPanelError(f"{context}: expected exactly one edit category")
        value = value[0]
    category = _norm(value)
    if not category:
        raise MatchedPanelError(f"{context}: missing edit category")
    return category


def d2_parent_key(row: SourceRow) -> tuple[str, int, int, str, str]:
    validate_d2(row)
    instance = row.value["instances"][0]
    return (
        _dataset(row.value.get("source", instance.get("pair_source"))),
        row.image_id,
        int(row.value["sent_id"]),
        _single_category(
            instance.get("replace_category"), context=f"D2:{row.line_number}"
        ),
        _norm(instance.get("positive_phrase")),
    )


def d3_parent_key(row: SourceRow) -> tuple[str, int, int, str, str]:
    validate_d3(row)
    return (
        _dataset(row.value.get("dataset")),
        row.image_id,
        int(row.value["sent_id"]),
        _single_category(
            row.value.get("replace_category"), context=f"D3:{row.line_number}"
        ),
        _norm(row.value.get("sent")),
    )


def _key_record(key: tuple[str, int, int, str, str]) -> dict[str, Any]:
    return {
        "dataset": key[0],
        "image_id": key[1],
        "sent_id": key[2],
        "edit_category": key[3],
        "positive_phrase_normalized": key[4],
    }


def _key_sha256(key: tuple[str, int, int, str, str]) -> str:
    payload = json.dumps(
        _key_record(key), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _priority(row: SourceRow, *, seed: str, key_sha256: str) -> str:
    return hashlib.sha256(
        (seed + "\0" + key_sha256 + "\0" + row.row_sha256).encode("ascii")
    ).hexdigest()


def _iter_source_rows(path: Path) -> Iterable[SourceRow]:
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                raise MatchedPanelError(f"blank D2 source row at line {line_number}")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise MatchedPanelError(f"non-object D2 source row at line {line_number}")
            raw_image_id = value.get("image_id")
            if isinstance(raw_image_id, bool):
                raise MatchedPanelError(f"invalid D2 image_id at line {line_number}")
            try:
                image_id = int(raw_image_id)
            except (TypeError, ValueError) as error:
                raise MatchedPanelError(
                    f"invalid D2 image_id at line {line_number}"
                ) from error
            yield SourceRow(
                line_number=line_number,
                raw=raw,
                value=value,
                image_id=image_id,
                row_sha256=hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest(),
            )


def _unique_d3_index(
    rows: Sequence[SourceRow], *, split: str
) -> dict[tuple[str, int, int, str, str], SourceRow]:
    index: dict[tuple[str, int, int, str, str], SourceRow] = {}
    for row in rows:
        _d3_class_id(row)
        key = d3_parent_key(row)
        if key in index:
            raise MatchedPanelError(
                f"D3 {split} has a duplicate parent key at lines "
                f"{index[key].line_number} and {row.line_number}"
            )
        index[key] = row
    return index


def _scan_d2_candidates(
    path: Path,
    *,
    requested_keys: set[tuple[str, int, int, str, str]],
) -> tuple[
    dict[tuple[str, int, int, str, str], list[SourceRow]], dict[str, Any]
]:
    candidates: dict[tuple[str, int, int, str, str], list[SourceRow]] = defaultdict(list)
    requested_prefixes = {(key[0], key[1], key[2]) for key in requested_keys}
    row_count = 0
    image_ids: set[int] = set()
    for row in _iter_source_rows(path):
        row_count += 1
        image_ids.add(row.image_id)
        # The three identity fields cheaply reject most rows before full schema
        # validation and phrase normalization.
        try:
            dataset = _dataset(row.value.get("source"))
            image_id = row.image_id
            sent_id = int(row.value["sent_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if (dataset, image_id, sent_id) not in requested_prefixes:
            continue
        key = d2_parent_key(row)
        if key in requested_keys:
            _d2_class_id(row)
            candidates[key].append(row)
    return candidates, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "rows": row_count,
        "unique_images": len(image_ids),
    }


def _guard_output_namespace(output_dir: Path) -> None:
    """Prevent a v2 build from overwriting the sealed v1 namespace."""

    if not output_dir.exists():
        return
    audit_path = output_dir / "audit.json"
    entries = list(output_dir.iterdir())
    if not entries:
        return
    if not audit_path.is_file():
        raise MatchedPanelError(
            f"refusing to overwrite non-empty unsealed output directory {output_dir}"
        )
    try:
        schema = json.loads(audit_path.read_text(encoding="utf-8")).get("schema")
    except (OSError, json.JSONDecodeError, AttributeError) as error:
        raise MatchedPanelError(
            f"refusing to overwrite output namespace with invalid audit {audit_path}"
        ) from error
    if schema == LEGACY_SCHEMA:
        raise MatchedPanelError(
            "refusing to overwrite sealed v1 matched-panel artifacts; choose a fresh "
            "v2 output directory"
        )
    if schema != SCHEMA:
        raise MatchedPanelError(
            f"refusing to overwrite output namespace with schema {schema!r}"
        )


def _annotate_pair(
    *,
    split: str,
    d2_row: SourceRow,
    d3_row: SourceRow,
    key: tuple[str, int, int, str, str],
    candidate_count: int,
    seed: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    key_sha256 = _key_sha256(key)
    pair_id = f"c2-parent:{split}:{key_sha256[:24]}"
    d2_value = convert_d2(d2_row)
    d3_value = convert_d3(d3_row)
    d2_class_id = _exact_class_id(
        d2_value.get("class_id"), context=f"D2:{d2_row.line_number}"
    )
    d3_class_id = _exact_class_id(
        d3_value.get("class_id"), context=f"D3:{d3_row.line_number}"
    )
    d2_negative = _norm(d2_value["try_tn"])
    d3_negative = _norm(d3_value["try_tn"])
    negative_text_normalized_match = d2_negative == d3_negative
    input_component_exact_matches = {
        "image_id": int(d2_value["image_id"]) == int(d3_value["image_id"]),
        "file_name": d2_value["file_name"] == d3_value["file_name"],
        "target_bbox_used": (
            d2_value["target_bbox_used"] == d3_value["target_bbox_used"]
        ),
        "positive_phrase": d2_value["sent"] == d3_value["sent"],
        "negative_text": d2_value["try_tn"] == d3_value["try_tn"],
        "canonical_class_id": d2_class_id == d3_class_id,
    }
    base_parent_input_exact_match = all(
        input_component_exact_matches[field]
        for field in (
            "image_id",
            "file_name",
            "target_bbox_used",
            "positive_phrase",
        )
    )
    if not base_parent_input_exact_match:
        causal_input_relation = "other_parent_input_mismatch"
    elif input_component_exact_matches["negative_text"]:
        causal_input_relation = (
            PRIMARY_CAUSAL_STRATUM
            if input_component_exact_matches["canonical_class_id"]
            else "identical_text_class_mismatch"
        )
    else:
        causal_input_relation = (
            "different_text_class_aligned"
            if input_component_exact_matches["canonical_class_id"]
            else "different_text_class_mismatch"
        )
    complete_model_input_exact_match = all(input_component_exact_matches.values())
    stratum = {
        "dataset": key[0],
        "edit_category": key[3],
        "negative_text_relation": (
            "identical" if negative_text_normalized_match else "different"
        ),
        "canonical_class_relation": (
            "aligned"
            if input_component_exact_matches["canonical_class_id"]
            else "mismatched"
        ),
        "causal_input_relation": causal_input_relation,
    }
    selection = {
        "seed": seed,
        "method": "minimum_sha256(seed,NUL,parent_key_sha256,NUL,d2_row_sha256)",
        "candidate_count": candidate_count,
        "selected_candidate_rank": 0,
        "selected_d2_source_line": d2_row.line_number,
        "selected_d2_source_row_sha256": d2_row.row_sha256,
    }
    shared = {
        "matched_pair_schema": PAIR_SCHEMA,
        "matched_pair_id": pair_id,
        "matched_split": split,
        "matched_parent_key": _key_record(key),
        "matched_parent_key_sha256": key_sha256,
        "matched_stratum": stratum,
        "matched_selection": selection,
        "positive_phrase_exact_match": d2_value["sent"] == d3_value["sent"],
        "positive_phrase_normalized_match": (
            _norm(d2_value["sent"]) == _norm(d3_value["sent"])
        ),
        "negative_text_exact_match": input_component_exact_matches["negative_text"],
        "negative_text_normalized_match": negative_text_normalized_match,
        "d2_canonical_class_id": d2_class_id,
        "d3_canonical_class_id": d3_class_id,
        "canonical_class_id_match": input_component_exact_matches[
            "canonical_class_id"
        ],
        "model_input_component_exact_matches": input_component_exact_matches,
        "base_parent_input_exact_match": base_parent_input_exact_match,
        "complete_model_input_exact_match": complete_model_input_exact_match,
        "class_aligned_identical_complete_input": (
            causal_input_relation == PRIMARY_CAUSAL_STRATUM
        ),
    }
    d2_value.update(shared)
    d2_value["table_b_id"] = "D2m"
    d3_value.update(shared)
    d3_value["table_b_id"] = "D3m"
    pair_record = {
        **shared,
        "image_id": key[1],
        "dataset": key[0],
        "sent_id": key[2],
        "edit_category": key[3],
        "d2m": {
            "sample_id": d2_value["sample_id"],
            "class_id": d2_class_id,
            "file_name": d2_value["file_name"],
            "target_bbox_used": d2_value["target_bbox_used"],
            "sent": d2_value["sent"],
            "try_tn": d2_value["try_tn"],
            "tn_scope": d2_value["tn_scope"],
            "source_line": d2_row.line_number,
            "source_row_sha256": d2_row.row_sha256,
        },
        "d3m": {
            "sample_id": d3_value["sample_id"],
            "class_id": d3_class_id,
            "file_name": d3_value["file_name"],
            "target_bbox_used": d3_value["target_bbox_used"],
            "sent": d3_value["sent"],
            "try_tn": d3_value["try_tn"],
            "tn_scope": d3_value["tn_scope"],
            "source_line": d3_row.line_number,
            "source_row_sha256": d3_row.row_sha256,
        },
    }
    return d2_value, d3_value, pair_record


def _build_split(
    *,
    split: str,
    d3_index: Mapping[tuple[str, int, int, str, str], SourceRow],
    d2_candidates: Mapping[
        tuple[str, int, int, str, str], Sequence[SourceRow]
    ],
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    d2_values: list[dict[str, Any]] = []
    d3_values: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for key, d3_row in sorted(d3_index.items(), key=lambda item: item[1].line_number):
        candidates = list(d2_candidates.get(key, ()))
        if not candidates:
            continue
        key_sha256 = _key_sha256(key)
        candidates.sort(
            key=lambda row: (
                _priority(row, seed=seed, key_sha256=key_sha256),
                row.line_number,
            )
        )
        d2_value, d3_value, pair = _annotate_pair(
            split=split,
            d2_row=candidates[0],
            d3_row=d3_row,
            key=key,
            candidate_count=len(candidates),
            seed=seed,
        )
        d2_values.append(d2_value)
        d3_values.append(d3_value)
        pairs.append(pair)
    return d2_values, d3_values, pairs


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        for value in values:
            handle.write(canonical_bytes(value))
    os.replace(temporary, path)


def _record(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path, label=path.name)
    return _file_record(path, rows)


def _load_pair_records(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                raise MatchedPanelError(
                    f"blank matched-pair row at {path}:{line_number}"
                )
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise MatchedPanelError(
                    f"invalid matched-pair JSON at {path}:{line_number}"
                ) from error
            if not isinstance(value, dict):
                raise MatchedPanelError(
                    f"non-object matched-pair row at {path}:{line_number}"
                )
            values.append(value)
    return values


def _split_stats(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    datasets = Counter(pair["matched_stratum"]["dataset"] for pair in pairs)
    categories = Counter(
        pair["matched_stratum"]["edit_category"] for pair in pairs
    )
    negative_relation = Counter(
        pair["matched_stratum"]["negative_text_relation"] for pair in pairs
    )
    negative_exact_relation = Counter(
        "identical" if pair["negative_text_exact_match"] else "different"
        for pair in pairs
    )
    positive_exact_relation = Counter(
        "identical" if pair["positive_phrase_exact_match"] else "different"
        for pair in pairs
    )
    candidate_counts = Counter(
        int(pair["matched_selection"]["candidate_count"]) for pair in pairs
    )
    canonical_class_relation = Counter(
        pair["matched_stratum"]["canonical_class_relation"] for pair in pairs
    )
    causal_input_relation = Counter(
        pair["matched_stratum"]["causal_input_relation"] for pair in pairs
    )
    class_mismatch_directions = Counter(
        f"{int(pair['d2_canonical_class_id'])}->{int(pair['d3_canonical_class_id'])}"
        for pair in pairs
        if not pair["canonical_class_id_match"]
    )
    component_mismatches = Counter(
        component
        for pair in pairs
        for component, matches in pair[
            "model_input_component_exact_matches"
        ].items()
        if not matches
    )
    return {
        "pairs": len(pairs),
        "unique_images": len(
            {int(pair["matched_parent_key"]["image_id"]) for pair in pairs}
        ),
        "dataset_pairs": dict(sorted(datasets.items())),
        "edit_category_pairs": dict(sorted(categories.items())),
        "negative_text_relation_pairs": dict(sorted(negative_relation.items())),
        "negative_text_exact_relation_pairs": dict(
            sorted(negative_exact_relation.items())
        ),
        "positive_text_exact_relation_pairs": dict(
            sorted(positive_exact_relation.items())
        ),
        "canonical_class_relation_pairs": dict(
            sorted(canonical_class_relation.items())
        ),
        "causal_input_relation_pairs": dict(sorted(causal_input_relation.items())),
        "class_aligned_identical_complete_input_pairs": causal_input_relation.get(
            PRIMARY_CAUSAL_STRATUM, 0
        ),
        "canonical_class_id_mismatch_pairs": canonical_class_relation.get(
            "mismatched", 0
        ),
        "identical_negative_text_class_id_mismatch_pairs": sum(
            pair["negative_text_exact_match"]
            and not pair["canonical_class_id_match"]
            for pair in pairs
        ),
        "canonical_class_id_mismatch_direction_pairs": dict(
            sorted(class_mismatch_directions.items())
        ),
        "model_input_component_mismatch_pairs": dict(
            sorted(component_mismatches.items())
        ),
        "d2_candidate_count_histogram": {
            str(key): value for key, value in sorted(candidate_counts.items())
        },
    }


def _write_configs(
    *, output_dir: Path, config_dir: Path, audit_path: Path
) -> dict[str, dict[str, Any]]:
    positive_annos = [
        "/home/user/PIVOT/data/ablations/stageb_refexp_three_train_20260711/refcoco_stageb_phrase_v1.jsonl",
        "/home/user/PIVOT/data/ablations/stageb_refexp_three_train_20260711/refcocoplus_stageb_phrase_v1.jsonl",
        "/home/user/PIVOT/data/ablations/stageb_refexp_three_train_20260711/refcocog_stageb_phrase_v1.jsonl",
    ]
    positives = [_positive_dataset(path) for path in positive_annos]
    specs = {
        "D2m": (
            "d2m_train",
            "traceable_counterfactual_edit",
        ),
        "D3m": (
            "d3m_train",
            "proposal_covered_verified",
        ),
    }
    records: dict[str, dict[str, Any]] = {}
    config_dir.mkdir(parents=True, exist_ok=True)
    for table_b_id, (output_key, scope) in specs.items():
        tn = _tn_dataset(
            _repo_alias(output_dir / OUTPUT_NAMES[output_key]),
            table_b_id=table_b_id,
            scope=scope,
            audit=_repo_alias(audit_path),
        )
        tn["paper_matched_causal_panel"] = True
        tn["paper_runtime_supported"] = True
        tn["paper_runtime_contract"] = (
            "v24_parent_matched_class_aligned_v2_fail_closed"
        )
        payload = {"train": [dict(value) for value in positives] + [tn], "val": []}
        path = config_dir / CONFIG_NAMES[table_b_id]
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
        records[table_b_id] = _file_record(path)
    return records


def build_panel(
    *,
    d2_path: Path,
    d3_train_path: Path,
    d3_calibration_path: Path,
    d3_partition_audit_path: Path,
    strict2031_path: Path,
    strict1607_path: Path,
    output_dir: Path,
    config_dir: Path,
    seed: str = DEFAULT_SEED,
) -> dict[str, Any]:
    _guard_output_namespace(output_dir)
    try:
        d3_partition_audit = json.loads(
            d3_partition_audit_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise MatchedPanelError(
            f"invalid D3 partition audit {d3_partition_audit_path}: {error}"
        ) from error
    if d3_partition_audit.get("schema") != (
        "stage-b-semantic-tn-leakage-isolated-partition-v1"
    ):
        raise MatchedPanelError("D3 partition audit schema mismatch")
    d3_outputs = d3_partition_audit.get("outputs", {})
    expected_d3_hashes = {
        "single_edit_train": sha256_file(d3_train_path),
        "single_edit_calibration": sha256_file(d3_calibration_path),
    }
    for key, observed in expected_d3_hashes.items():
        if d3_outputs.get(key, {}).get("sha256") != observed:
            raise MatchedPanelError(f"D3 partition audit {key} hash drift")
    d3_invariants = d3_partition_audit.get("invariants", {})
    if not (
        d3_invariants.get("eligible_strict_union_image_overlap") == 0
        and d3_invariants.get("train_calibration_image_overlap") == 0
        and d3_invariants.get("single_edit_invalid_metadata_rows_excluded") is True
    ):
        raise MatchedPanelError("D3 partition audit invariants are incomplete")

    d3_train_rows = load_jsonl(d3_train_path, label="D3 single-edit train")
    d3_calibration_rows = load_jsonl(
        d3_calibration_path, label="D3 single-edit calibration"
    )
    train_index = _unique_d3_index(d3_train_rows, split="train")
    calibration_index = _unique_d3_index(
        d3_calibration_rows, split="calibration"
    )
    overlap_keys = set(train_index) & set(calibration_index)
    if overlap_keys:
        raise MatchedPanelError("D3 train/calibration parent keys overlap")

    requested_keys = set(train_index) | set(calibration_index)
    candidates, d2_record = _scan_d2_candidates(
        d2_path, requested_keys=requested_keys
    )
    train = _build_split(
        split="train",
        d3_index=train_index,
        d2_candidates=candidates,
        seed=seed,
    )
    calibration = _build_split(
        split="calibration",
        d3_index=calibration_index,
        d2_candidates=candidates,
        seed=seed,
    )
    outputs = {
        "d2m_train": train[0],
        "d3m_train": train[1],
        "pairs_train": train[2],
        "d2m_calibration": calibration[0],
        "d3m_calibration": calibration[1],
        "pairs_calibration": calibration[2],
    }
    for key, values in outputs.items():
        _write_jsonl(output_dir / OUTPUT_NAMES[key], values)

    strict2031 = load_jsonl(strict2031_path, label="strict2031")
    strict1607 = load_jsonl(strict1607_path, label="strict1607")
    strict_images = {row.image_id for row in strict2031 + strict1607}
    train_images = {
        int(value["image_id"]) for value in outputs["d3m_train"]
    }
    calibration_images = {
        int(value["image_id"]) for value in outputs["d3m_calibration"]
    }
    if train_images & calibration_images:
        raise MatchedPanelError("matched train/calibration images overlap")
    if (train_images | calibration_images) & strict_images:
        raise MatchedPanelError("matched panel overlaps strict evaluation images")

    def pair_invariants(split: str) -> dict[str, Any]:
        d2_values = outputs[f"d2m_{split}"]
        d3_values = outputs[f"d3m_{split}"]
        pairs = outputs[f"pairs_{split}"]
        stats = _split_stats(pairs)
        return {
            "equal_rows": len(d2_values) == len(d3_values) == len(pairs),
            "aligned_pair_ids": [value["matched_pair_id"] for value in d2_values]
            == [value["matched_pair_id"] for value in d3_values]
            == [value["matched_pair_id"] for value in pairs],
            "aligned_parent_key_sha256": [
                value["matched_parent_key_sha256"] for value in d2_values
            ]
            == [value["matched_parent_key_sha256"] for value in d3_values],
            "positive_phrase_normalized_match": all(
                value["positive_phrase_normalized_match"] for value in pairs
            ),
            "unique_pair_ids": len(pairs)
            == len({value["matched_pair_id"] for value in pairs}),
            "unique_parent_keys": len(pairs)
            == len({value["matched_parent_key_sha256"] for value in pairs}),
            "negative_text_relation_is_partition": sum(
                stats["negative_text_relation_pairs"].values()
            )
            == len(pairs),
            "canonical_class_relation_is_partition": sum(
                stats["canonical_class_relation_pairs"].values()
            )
            == len(pairs),
            "causal_input_relation_is_partition": sum(
                stats["causal_input_relation_pairs"].values()
            )
            == len(pairs),
            "class_mismatch_count_is_audited": stats[
                "canonical_class_id_mismatch_pairs"
            ]
            == sum(not value["canonical_class_id_match"] for value in pairs),
            "class_aligned_identical_stratum_is_exact": all(
                (
                    value["matched_stratum"]["causal_input_relation"]
                    == PRIMARY_CAUSAL_STRATUM
                )
                == (
                    value["complete_model_input_exact_match"]
                    and value["canonical_class_id_match"]
                    and value["negative_text_exact_match"]
                )
                for value in pairs
            ),
            "class_aligned_identical_boolean_is_exact": all(
                value["class_aligned_identical_complete_input"]
                == (
                    value["matched_stratum"]["causal_input_relation"]
                    == PRIMARY_CAUSAL_STRATUM
                )
                for value in pairs
            ),
            "runtime_global_verified_true_rows": sum(
                value.get("global_tn_verified") is True
                for value in d2_values + d3_values
            ),
        }

    split_stats = {
        "train": _split_stats(outputs["pairs_train"]),
        "calibration": _split_stats(outputs["pairs_calibration"]),
    }

    def matching_yield(
        *, split: str, d3_parent_rows: int
    ) -> dict[str, Any]:
        if d3_parent_rows <= 0:
            raise MatchedPanelError(f"{split} has no D3 parent rows")
        matched_pairs = len(outputs[f"pairs_{split}"])
        unmatched_rows = d3_parent_rows - matched_pairs
        if unmatched_rows < 0:
            raise MatchedPanelError(f"{split} matched more rows than D3 parents")
        return {
            "d3_parent_rows": d3_parent_rows,
            "matched_pairs": matched_pairs,
            "unmatched_d3_parent_rows": unmatched_rows,
            "matched_fraction": matched_pairs / d3_parent_rows,
            "matched_pairwise_claim_denominator": matched_pairs,
            "class_aligned_identical_claim_denominator": split_stats[split][
                "class_aligned_identical_complete_input_pairs"
            ],
            "unmatched_d3_parent_rows_excluded_from_pairwise_claims": (
                unmatched_rows
            ),
            "unmatched_fraction": unmatched_rows / d3_parent_rows,
            "parent_row_partition_is_exact": (
                d3_parent_rows == matched_pairs + unmatched_rows
            ),
        }

    matching_yields = {
        "train": matching_yield(split="train", d3_parent_rows=len(train_index)),
        "calibration": matching_yield(
            split="calibration", d3_parent_rows=len(calibration_index)
        ),
    }

    audit_path = output_dir / "audit.json"
    audit: dict[str, Any] = {
        "schema": SCHEMA,
        "kind": "completed_c2_parent_matched_tn_panel",
        "seed": seed,
        "matching_contract": {
            "parent_key_fields": [
                "normalized dataset",
                "integer image_id",
                "integer sent_id",
                "normalized single edit category",
                "normalized positive phrase",
            ],
            "dataset_aliases": DATASET_ALIASES,
            "candidate_selection": (
                "minimum sha256(seed,NUL,parent_key_sha256,NUL,d2_row_sha256), "
                "then source line"
            ),
            "one_d2_candidate_per_d3_parent": True,
            "negative_phrase_is_not_a_parent_key": True,
            "canonical_class_id_is_not_a_parent_key": True,
            "canonical_class_id_is_exact_integer_audited": True,
            "selection_and_exposure_policy": (
                "Retain every deterministically selected parent match; isolate exact "
                "canonical-class equality in the causal-input stratum instead of "
                "silently dropping mismatches."
            ),
        },
        "inputs": {
            "D2_raw": d2_record,
            "D3_single_edit_train": _file_record(d3_train_path, d3_train_rows),
            "D3_single_edit_calibration": _file_record(
                d3_calibration_path, d3_calibration_rows
            ),
            "D3_partition_audit": _file_record(d3_partition_audit_path),
            "strict2031": _file_record(strict2031_path, strict2031),
            "strict1607": _file_record(strict1607_path, strict1607),
        },
        "statistics": split_stats,
        "matching_yield": matching_yields,
        "claim_scope": {
            "pairwise_effect_population": "matched_pairs_only",
            "primary_causal_stratum": PRIMARY_CAUSAL_STRATUM,
            "primary_causal_stratum_requires_exact_model_input": True,
            "canonical_class_id_equality_required": True,
            "unmatched_d3_parent_rows_are_out_of_scope": True,
            "generalization_to_unmatched_d3_parent_rows_supported": False,
        },
        "outputs": {
            key: _record(output_dir / OUTPUT_NAMES[key]) for key in outputs
        },
        "sampling_contract": {
            "D2m_D3m_train_rows_equal": len(outputs["d2m_train"])
            == len(outputs["d3m_train"]),
            "positive_mix_weights": [1.0, 1.0, 1.0],
            "tn_mix_weight": 3.0,
            "expected_tn_draw_fraction": 0.5,
            "tn_balance_sampling": False,
        },
        "scope_contract": {
            "D2m": {
                "tn_scope": "traceable_counterfactual_edit",
                "global_tn_verified": False,
            },
            "D3m": {
                "tn_scope": "proposal_covered_verified",
                "global_tn_verified": False,
            },
        },
        "limitations": {
            "matched_categories": ["color", "size", "spatial"],
            "negative_phrase_not_forced_equal": True,
            "clean_visual_filtering_subset": (
                f"Only causal_input_relation={PRIMARY_CAUSAL_STRATUM} holds every "
                "model-consumed identity component fixed, including canonical "
                "class_id. Text-identical class-mismatch rows are audited separately "
                "and are not part of this subset."
            ),
            "interpretation": (
                "The full matched panel controls normalized parent expression and "
                "edit category. Attribute differences specifically to visual filtering "
                f"only inside {PRIMARY_CAUSAL_STRATUM}; all other strata are "
                "descriptive."
            ),
            "unmatched_parent_rows": (
                "Pairwise estimates exclude unmatched D3 parents. Their counts and "
                "the resulting matched-only denominators are sealed per split."
            ),
        },
        "invariants": {
            "train": pair_invariants("train"),
            "calibration": pair_invariants("calibration"),
            "strict_union_image_overlap": len(
                (train_images | calibration_images) & strict_images
            ),
            "train_calibration_image_overlap": len(
                train_images & calibration_images
            ),
            "unique_train_parent_keys": len(outputs["pairs_train"])
            == len(
                {
                    value["matched_parent_key_sha256"]
                    for value in outputs["pairs_train"]
                }
            ),
            "unique_calibration_parent_keys": len(outputs["pairs_calibration"])
            == len(
                {
                    value["matched_parent_key_sha256"]
                    for value in outputs["pairs_calibration"]
                }
            ),
            "train_calibration_pair_id_overlap": len(
                {value["matched_pair_id"] for value in outputs["pairs_train"]}
                & {
                    value["matched_pair_id"]
                    for value in outputs["pairs_calibration"]
                }
            ),
            "train_matching_yield_partition": matching_yields["train"][
                "parent_row_partition_is_exact"
            ],
            "calibration_matching_yield_partition": matching_yields[
                "calibration"
            ]["parent_row_partition_is_exact"],
        },
        "runtime_contract": {
            "current_v24_supported_table_ids": ["D2m", "D3m"],
            "D2m_D3m_supported_by_current_v24": True,
            "status": "sealed causal data panel; fail-closed runtime enabled",
            "separate_audit_boundary": (
                f"D2m/D3m bind this {len(outputs['pairs_train']):,}-row matched-only "
                f"audit; {len(train_index) - len(outputs['pairs_train']):,} unmatched "
                "D3 train parents are explicitly outside pairwise claims."
            ),
        },
    }
    for split in ("train", "calibration"):
        for key, value in audit["invariants"][split].items():
            if key == "runtime_global_verified_true_rows":
                if value != 0:
                    raise MatchedPanelError(f"{split} upgraded a row to global TN")
            elif value is not True:
                raise MatchedPanelError(f"{split} invariant {key} failed")
    if audit["invariants"]["strict_union_image_overlap"] != 0:
        raise MatchedPanelError("strict leakage invariant failed")
    if audit["invariants"]["train_calibration_image_overlap"] != 0:
        raise MatchedPanelError("train/calibration leakage invariant failed")
    if audit["invariants"]["train_calibration_pair_id_overlap"] != 0:
        raise MatchedPanelError("train/calibration pair-ID overlap invariant failed")
    for key in (
        "unique_train_parent_keys",
        "unique_calibration_parent_keys",
        "train_matching_yield_partition",
        "calibration_matching_yield_partition",
    ):
        if audit["invariants"][key] is not True:
            raise MatchedPanelError(f"audit invariant {key} failed")

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n")
    configs = _write_configs(
        output_dir=output_dir, config_dir=config_dir, audit_path=audit_path
    )
    audit["dataset_configs"] = configs
    audit_path.write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n")
    return audit


def verify_panel(audit_path: Path) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text())
    if audit.get("schema") == LEGACY_SCHEMA:
        raise MatchedPanelError(
            "legacy v1 audit lacks canonical-class alignment; it is preserved but "
            "cannot support complete-input causal claims"
        )
    if audit.get("schema") != SCHEMA:
        raise MatchedPanelError("matched-panel audit schema mismatch")
    for section in ("inputs", "outputs", "dataset_configs"):
        for key, record in audit[section].items():
            if sha256_file(Path(record["path"])) != record["sha256"]:
                raise MatchedPanelError(f"{section}.{key} hash drift")
    for split in ("train", "calibration"):
        stats = audit.get("statistics", {}).get(split, {})
        pair_count = stats.get("pairs")
        if not isinstance(pair_count, int) or pair_count <= 0:
            raise MatchedPanelError(f"{split} pair count is not sealed")
        pair_output_key = f"pairs_{split}"
        pair_output = audit.get("outputs", {}).get(pair_output_key, {})
        pair_path = Path(str(pair_output.get("path", "")))
        computed_stats = _split_stats(_load_pair_records(pair_path))
        if stats != computed_stats:
            raise MatchedPanelError(
                f"{split} statistics do not match sealed pair records"
            )
        for field in (
            "negative_text_relation_pairs",
            "canonical_class_relation_pairs",
            "causal_input_relation_pairs",
        ):
            relation_counts = stats.get(field, {})
            if not isinstance(relation_counts, dict) or any(
                type(value) is not int or value < 0
                for value in relation_counts.values()
            ):
                raise MatchedPanelError(f"{split} {field} is malformed")
            if sum(relation_counts.values()) != pair_count:
                raise MatchedPanelError(
                    f"{split} {field} does not partition matched pairs"
                )
        class_relations = stats["canonical_class_relation_pairs"]
        class_mismatches = class_relations.get("mismatched", 0)
        if stats.get("canonical_class_id_mismatch_pairs") != class_mismatches:
            raise MatchedPanelError(f"{split} canonical-class mismatch count drifted")
        causal_relations = stats["causal_input_relation_pairs"]
        clean_pairs = causal_relations.get(PRIMARY_CAUSAL_STRATUM, 0)
        if (
            stats.get("class_aligned_identical_complete_input_pairs")
            != clean_pairs
        ):
            raise MatchedPanelError(f"{split} primary causal-stratum count drifted")
        identical_class_mismatches = stats.get(
            "identical_negative_text_class_id_mismatch_pairs"
        )
        if (
            type(identical_class_mismatches) is not int
            or identical_class_mismatches < 0
            or identical_class_mismatches > class_mismatches
        ):
            raise MatchedPanelError(
                f"{split} identical-text class-mismatch count is invalid"
            )
        matching_yield = audit.get("matching_yield", {}).get(split, {})
        d3_parent_rows = matching_yield.get("d3_parent_rows")
        unmatched_rows = matching_yield.get("unmatched_d3_parent_rows")
        expected_parent_rows = audit.get("inputs", {}).get(
            f"D3_single_edit_{split}", {}
        ).get("rows")
        expected_matched_fraction = (
            pair_count / d3_parent_rows
            if type(d3_parent_rows) is int and d3_parent_rows > 0
            else None
        )
        expected_unmatched_fraction = (
            unmatched_rows / d3_parent_rows
            if (
                type(unmatched_rows) is int
                and type(d3_parent_rows) is int
                and d3_parent_rows > 0
            )
            else None
        )
        if (
            type(d3_parent_rows) is not int
            or type(unmatched_rows) is not int
            or unmatched_rows < 0
            or d3_parent_rows != expected_parent_rows
            or d3_parent_rows != pair_count + unmatched_rows
            or matching_yield.get("matched_pairs") != pair_count
            or matching_yield.get("matched_fraction") != expected_matched_fraction
            or matching_yield.get("unmatched_fraction")
            != expected_unmatched_fraction
            or matching_yield.get("matched_pairwise_claim_denominator") != pair_count
            or matching_yield.get("class_aligned_identical_claim_denominator")
            != clean_pairs
            or matching_yield.get(
                "unmatched_d3_parent_rows_excluded_from_pairwise_claims"
            )
            != unmatched_rows
            or matching_yield.get("parent_row_partition_is_exact") is not True
        ):
            raise MatchedPanelError(
                f"{split} matched/unmatched claim denominator drifted"
            )
        invariants = audit.get("invariants", {}).get(split, {})
        for key, value in invariants.items():
            if key == "runtime_global_verified_true_rows":
                if value != 0:
                    raise MatchedPanelError(f"{split} upgraded a row to global TN")
            elif value is not True:
                raise MatchedPanelError(f"{split} invariant {key} failed")
    for key in (
        "strict_union_image_overlap",
        "train_calibration_image_overlap",
        "train_calibration_pair_id_overlap",
    ):
        if audit.get("invariants", {}).get(key) != 0:
            raise MatchedPanelError(f"audit invariant {key} is nonzero")
    for key in (
        "unique_train_parent_keys",
        "unique_calibration_parent_keys",
        "train_matching_yield_partition",
        "calibration_matching_yield_partition",
    ):
        if audit.get("invariants", {}).get(key) is not True:
            raise MatchedPanelError(f"audit invariant {key} failed")
    expected_claim_scope = {
        "pairwise_effect_population": "matched_pairs_only",
        "primary_causal_stratum": PRIMARY_CAUSAL_STRATUM,
        "primary_causal_stratum_requires_exact_model_input": True,
        "canonical_class_id_equality_required": True,
        "unmatched_d3_parent_rows_are_out_of_scope": True,
        "generalization_to_unmatched_d3_parent_rows_supported": False,
    }
    if audit.get("claim_scope") != expected_claim_scope:
        raise MatchedPanelError("matched-panel claim scope drifted")
    if audit.get("runtime_contract", {}).get(
        "D2m_D3m_supported_by_current_v24"
    ) is not True:
        raise MatchedPanelError("matched panel fail-closed runtime is not enabled")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--d3-partition-audit", type=Path, default=DEFAULT_D3_DIR / "audit.json"
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
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify:
        audit = verify_panel(args.output_dir / "audit.json")
    else:
        audit = build_panel(
            d2_path=args.d2,
            d3_train_path=args.d3_train,
            d3_calibration_path=args.d3_calibration,
            d3_partition_audit_path=args.d3_partition_audit,
            strict2031_path=args.strict2031,
            strict1607_path=args.strict1607,
            output_dir=args.output_dir,
            config_dir=args.config_dir,
            seed=args.seed,
        )
    print(
        json.dumps(
            {
                "schema": audit["schema"],
                "train_pairs": audit["statistics"]["train"]["pairs"],
                "calibration_pairs": audit["statistics"]["calibration"]["pairs"],
                "audit": str((args.output_dir / "audit.json").resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
