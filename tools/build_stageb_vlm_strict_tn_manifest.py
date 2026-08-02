#!/usr/bin/env python3
"""Build and verify the frozen VLM-verified Stage-B TN evaluation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/home/user/datasets/pivot_data"))
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "config" / "datasets_stageb_v10_aliasfix_synthetic_local_pairs.json"
DEFAULT_CANDIDATE_TRAIN_CONFIG = (
    REPO_ROOT / "config" / "datasets_stageb_v15_global_verified_pairs.json"
)
DEFAULT_GDINO_STAGEB_TRAIN_CONFIG = (
    REPO_ROOT
    / "config"
    / "ablations"
    / "gdino_ft_stage_b_rebuild_20260711"
    / "datasets_gdino_ft_stageb_with_tn_local.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "data"
    / "eval_manifests"
    / "stageb_vlm_verified_strict_ann_umd_val_20260711"
)
DEFAULT_PLUS_ACCEPTED = (
    REPO_ROOT
    / "sam3_and_tn"
    / "refcocoplus_sam3_washed_try_tn_llm_head_candidates_vlm_filter"
    / "accepted.jsonl"
)
DEFAULT_G_ACCEPTED = (
    REPO_ROOT
    / "sam3_and_tn"
    / "refcocog_sam3_washed_try_tn_llm_head_candidates_vlm_filter"
    / "accepted.jsonl"
)

SCHEMA_VERSION = "stageb_vlm_verified_strict_tn_v2"
STRICT_MANIFEST_NAME = "strict_ann_manifest.jsonl"
EVAL_MANIFEST_NAME = "eval_manifest.jsonl"
CANDIDATE_IMAGE_DISJOINT_MANIFEST_NAME = "candidate_train_image_disjoint_manifest.jsonl"
SEMANTIC_STAGEB_UNION_IMAGE_DISJOINT_MANIFEST_NAME = (
    "semantic_stageb_union_image_disjoint_manifest.jsonl"
)
GLOBAL_TN_DIAGNOSTIC_MANIFEST_NAME = "global_tn_only_image_disjoint_diagnostic.jsonl"
BASELINE_UNION_DIAGNOSTIC_MANIFEST_NAME = (
    "candidate_and_gdino_stageb_image_disjoint_diagnostic.jsonl"
)
AUDIT_NAME = "audit.json"

EXPECTED_STRICT_COUNTS = {
    "refcocoplus_unc_val": 1_251,
    "refcocog_umd_val": 784,
}
EXPECTED_TARGET_PLUS_PROPOSAL_COUNTS = {
    "refcocoplus_unc_val": 1_249,
    "refcocog_umd_val": 782,
}
EXPECTED_GLOBAL_TN_DIAGNOSTIC_COUNTS = {
    "refcocoplus_unc_val": 1_205,
    "refcocog_umd_val": 737,
}
EXPECTED_CANDIDATE_IMAGE_DISJOINT_COUNTS = {
    "refcocoplus_unc_val": 965,
    "refcocog_umd_val": 642,
}
EXPECTED_SEMANTIC_STAGEB_UNION_IMAGE_DISJOINT_COUNTS = {
    "refcocoplus_unc_val": 965,
    "refcocog_umd_val": 642,
}

COVERAGE_NONE = "none"
COVERAGE_TARGET_PLUS_PROPOSAL = "target_plus_proposal"
COVERAGE_ALL_PROPOSALS_ALL_NO = "all_proposals_all_no"
COVERAGE_POLICIES = (
    COVERAGE_NONE,
    COVERAGE_TARGET_PLUS_PROPOSAL,
    COVERAGE_ALL_PROPOSALS_ALL_NO,
)

_WS_RE = re.compile(r"\s+")


def normalize_sentence(value: Any) -> str:
    """Normalize RefCOCOg text for Google-to-UMD sentence matching."""
    text = str(value or "").replace("_", " ").replace(".", " ").strip().lower()
    return _WS_RE.sub(" ", text)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL row at {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"Expected JSON object at {path}:{line_number}")
            yield line_number, row


def _int_field(row: Mapping[str, Any], field: str, context: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer field {field!r} at {context}") from exc


def _nonempty_text(row: Mapping[str, Any], field: str, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing non-empty field {field!r} at {context}")
    return value.strip()


def _valid_xywh(row: Mapping[str, Any], field: str, context: str) -> List[float]:
    value = row.get(field)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"Invalid {field} at {context}: {value!r}")
    try:
        box = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric {field} at {context}: {value!r}") from exc
    if not all(math.isfinite(item) for item in box) or box[2] <= 0.0 or box[3] <= 0.0:
        raise ValueError(f"Non-finite or non-positive {field} at {context}: {box!r}")
    return box


def proposal_coverage(row: Mapping[str, Any], *, context: str = "row") -> Dict[str, Any]:
    """Return fail-closed target/proposal VLM coverage facts for one accepted row.

    ``target_plus_proposal_covered`` reproduces the audited final-eval rule. The
    target-local VLM judgment can account for at most one proposal that is not
    repeated in ``visual_proposal_judgments``.

    ``all_proposals_all_no`` is the stricter reusable training-data rule: the
    proposal cache must be non-empty, every proposal must have a judgment, and
    every nested judgment answer must be ``no``.
    """
    proposals = row.get("proposal_cache")
    judgments = row.get("visual_proposal_judgments")
    local = row.get("visual_local_judgment")
    if not isinstance(proposals, list):
        raise ValueError(f"proposal_cache must be a list at {context}")
    if not isinstance(judgments, list):
        raise ValueError(f"visual_proposal_judgments must be a list at {context}")
    if not isinstance(local, dict):
        raise ValueError(f"visual_local_judgment must be an object at {context}")

    proposal_num = _int_field(row, "proposal_num", context)
    if proposal_num < 0 or proposal_num != len(proposals):
        raise ValueError(
            f"proposal_num/cache mismatch at {context}: {proposal_num} vs {len(proposals)}"
        )
    if len(judgments) > proposal_num:
        raise ValueError(
            f"More proposal judgments than proposals at {context}: "
            f"{len(judgments)} vs {proposal_num}"
        )

    proposal_ids = []
    for index, proposal in enumerate(proposals):
        if not isinstance(proposal, dict):
            raise ValueError(f"Invalid proposal_cache[{index}] at {context}")
        proposal_ids.append(_int_field(proposal, "proposal_id", f"{context}.proposal_cache[{index}]"))
    if len(set(proposal_ids)) != len(proposal_ids):
        raise ValueError(f"Duplicate proposal ids at {context}")

    judgment_ids: List[int] = []
    judgment_answers: List[str] = []
    for index, item in enumerate(judgments):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid visual_proposal_judgments[{index}] at {context}")
        judgment_ids.append(
            _int_field(item, "proposal_id", f"{context}.visual_proposal_judgments[{index}]")
        )
        nested = item.get("judgment")
        if not isinstance(nested, dict):
            raise ValueError(f"Missing nested proposal judgment at {context}[{index}]")
        answer = nested.get("answer")
        if answer not in {"no", "unknown"}:
            raise ValueError(f"Unexpected proposal judgment answer at {context}[{index}]: {answer!r}")
        judgment_answers.append(str(answer))
    if len(set(judgment_ids)) != len(judgment_ids):
        raise ValueError(f"Duplicate proposal judgment ids at {context}")
    unknown_ids = set(judgment_ids).difference(proposal_ids)
    if unknown_ids:
        raise ValueError(f"Judgments reference unknown proposal ids at {context}: {sorted(unknown_ids)}")

    local_answer = local.get("answer")
    if local_answer not in {"no", "unknown"}:
        raise ValueError(f"Unexpected target-local answer at {context}: {local_answer!r}")
    target_verified_no = local_answer == "no"
    proposal_judgments_all_no = all(answer == "no" for answer in judgment_answers)
    all_proposals_judged = len(judgments) == proposal_num
    unjudged_ids = sorted(set(proposal_ids).difference(judgment_ids))

    return {
        "proposal_count": proposal_num,
        "proposal_judgment_count": len(judgments),
        "unjudged_proposal_count": len(unjudged_ids),
        "unjudged_proposal_ids": unjudged_ids,
        "target_local_answer": local_answer,
        "target_verified_no": target_verified_no,
        "proposal_judgments_all_no": proposal_judgments_all_no,
        "all_proposals_judged": all_proposals_judged,
        "target_plus_proposal_covered": (
            target_verified_no and proposal_num <= len(judgments) + 1
        ),
        "all_proposals_all_no": (
            target_verified_no
            and proposal_num > 0
            and all_proposals_judged
            and proposal_judgments_all_no
        ),
    }


def passes_coverage_policy(coverage: Mapping[str, Any], policy: str) -> bool:
    if policy == COVERAGE_NONE:
        return True
    if policy == COVERAGE_TARGET_PLUS_PROPOSAL:
        return bool(coverage.get("target_plus_proposal_covered", False))
    if policy == COVERAGE_ALL_PROPOSALS_ALL_NO:
        return bool(coverage.get("all_proposals_all_no", False))
    raise ValueError(f"Unknown coverage policy: {policy!r}")


def _load_refs(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        rows = pickle.load(handle)
    if not isinstance(rows, list):
        raise TypeError(f"Expected list in {path}, got {type(rows).__name__}")
    return rows


def _insert_split(
    mapping: MutableMapping[Tuple[Any, ...], str],
    key: Tuple[Any, ...],
    split: str,
    *,
    source: Path,
) -> bool:
    previous = mapping.get(key)
    if previous is not None and previous != split:
        raise ValueError(
            f"Conflicting official splits for {key!r} in {source}: {previous!r} vs {split!r}"
        )
    mapping[key] = split
    return previous is not None


def _build_plus_map(path: Path) -> Tuple[Dict[Tuple[int, int, int], str], int]:
    mapping: Dict[Tuple[int, int, int], str] = {}
    duplicates = 0
    for ref in _load_refs(path):
        split = str(ref["split"])
        for sentence in ref.get("sentences", []) or []:
            key = (int(ref["ref_id"]), int(ref["ann_id"]), int(sentence["sent_id"]))
            duplicates += int(_insert_split(mapping, key, split, source=path))
    return mapping, duplicates


def _build_g_maps(
    umd_path: Path,
    google_path: Path,
) -> Tuple[
    Dict[Tuple[int, int, str], str],
    Dict[Tuple[int, int, str], List[Tuple[int, int, str]]],
    Dict[Tuple[int, int, int], str],
    int,
    int,
]:
    umd: Dict[Tuple[int, int, str], str] = {}
    umd_identities: Dict[Tuple[int, int, str], List[Tuple[int, int, str]]] = {}
    umd_duplicates = 0
    for ref in _load_refs(umd_path):
        split = str(ref["split"])
        for sentence in ref.get("sentences", []) or []:
            normalized = normalize_sentence(sentence.get("sent") or sentence.get("raw"))
            if not normalized:
                raise ValueError(f"Empty normalized UMD sentence in {umd_path}")
            key = (int(ref["image_id"]), int(ref["ann_id"]), normalized)
            umd_duplicates += int(_insert_split(umd, key, split, source=umd_path))
            identity = (int(ref["ref_id"]), int(sentence["sent_id"]), split)
            candidates = umd_identities.setdefault(key, [])
            if identity in candidates:
                raise ValueError(f"Duplicate RefCOCOg UMD sentence identity in {umd_path}: {identity}")
            candidates.append(identity)

    google: Dict[Tuple[int, int, int], str] = {}
    google_duplicates = 0
    for ref in _load_refs(google_path):
        key = (int(ref["ref_id"]), int(ref["ann_id"]), int(ref["image_id"]))
        google_duplicates += int(
            _insert_split(google, key, str(ref["split"]), source=google_path)
        )
    return umd, umd_identities, google, umd_duplicates, google_duplicates


def _resolve_g_umd_identity(
    candidates: Sequence[Tuple[int, int, str]],
    *,
    accepted_sent_id: int,
    context: str,
) -> Tuple[int, int, str, str]:
    """Resolve a Google-annotated sentence to one official UMD sentence identity."""
    if not candidates:
        raise KeyError(f"No RefCOCOg UMD identity candidates at {context}")
    exact = [candidate for candidate in candidates if candidate[1] == accepted_sent_id]
    if len(exact) == 1:
        ref_id, sent_id, split = exact[0]
        mode = "accepted_sent_id" if len(candidates) > 1 else "unique_normalized_sentence"
        return ref_id, sent_id, split, mode
    if len(candidates) == 1:
        ref_id, sent_id, split = candidates[0]
        return ref_id, sent_id, split, "unique_normalized_sentence"
    raise ValueError(
        "Ambiguous RefCOCOg Google-to-UMD sentence mapping at "
        f"{context}: accepted_sent_id={accepted_sent_id}, candidates={list(candidates)!r}"
    )


def _resolve_config_path(value: str, *, data_root: Path) -> Path:
    expanded = value.replace("${DATA_ROOT}", str(data_root))
    path = Path(os.path.expandvars(expanded)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _load_holdout(
    train_config: Path,
    data_root: Path,
) -> Tuple[set[Tuple[int, int]], set[int], List[Dict[str, Any]]]:
    try:
        config = json.loads(train_config.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid training dataset config {train_config}: {exc}") from exc
    train_specs = config.get("train") if isinstance(config, dict) else None
    if not isinstance(train_specs, list) or not train_specs:
        raise ValueError(f"Training config has no non-empty train list: {train_config}")

    ann_keys: set[Tuple[int, int]] = set()
    image_ids: set[int] = set()
    inputs: List[Dict[str, Any]] = []
    for index, spec in enumerate(train_specs):
        if not isinstance(spec, dict) or not isinstance(spec.get("anno"), str):
            raise ValueError(f"Missing train[{index}].anno in {train_config}")
        path = _resolve_config_path(spec["anno"], data_root=data_root)
        rows = 0
        for line_number, row in _iter_jsonl(path):
            context = f"{path}:{line_number}"
            image_id = _int_field(row, "image_id", context)
            ann_id = _int_field(row, "ann_id", context)
            ann_keys.add((image_id, ann_id))
            image_ids.add(image_id)
            rows += 1
        inputs.append(
            {
                "path": str(path),
                "rows": rows,
                "sha256": sha256_file(path),
            }
        )
    return ann_keys, image_ids, inputs


def _load_train_image_sources(
    train_config: Path,
    data_root: Path,
) -> Tuple[set[int], List[Dict[str, Any]], List[set[int]]]:
    """Load every train source image id, failing closed on malformed rows."""
    try:
        config = json.loads(train_config.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid training dataset config {train_config}: {exc}") from exc
    train_specs = config.get("train") if isinstance(config, dict) else None
    if not isinstance(train_specs, list) or not train_specs:
        raise ValueError(f"Training config has no non-empty train list: {train_config}")

    union: set[int] = set()
    inputs: List[Dict[str, Any]] = []
    source_image_sets: List[set[int]] = []
    for index, spec in enumerate(train_specs):
        if not isinstance(spec, dict) or not isinstance(spec.get("anno"), str):
            raise ValueError(f"Missing train[{index}].anno in {train_config}")
        path = _resolve_config_path(spec["anno"], data_root=data_root)
        image_ids: set[int] = set()
        rows = 0
        for line_number, row in _iter_jsonl(path):
            image_ids.add(_int_field(row, "image_id", f"{path}:{line_number}"))
            rows += 1
        union.update(image_ids)
        source_image_sets.append(image_ids)
        inputs.append(
            {
                "config_index": index,
                "dataset_mode": spec.get("dataset_mode"),
                "declared_source": spec.get("source"),
                "path": str(path),
                "require_global_tn_verified": spec.get("require_global_tn_verified") is True,
                "rows": rows,
                "sha256": sha256_file(path),
                "unique_image_ids": len(image_ids),
            }
        )
    return union, inputs, source_image_sets


def _source_overlap_audit(
    records: Sequence[Mapping[str, Any]],
    inputs: Sequence[Mapping[str, Any]],
    source_image_sets: Sequence[set[int]],
) -> List[Dict[str, Any]]:
    if len(inputs) != len(source_image_sets):
        raise ValueError("Train source metadata/image-set length mismatch")
    cumulative: set[int] = set()
    audited: List[Dict[str, Any]] = []
    for metadata, image_ids in zip(inputs, source_image_sets):
        standalone_images = {int(row["image_id"]) for row in records if int(row["image_id"]) in image_ids}
        newly_covered = image_ids.difference(cumulative)
        incremental_images = {
            int(row["image_id"]) for row in records if int(row["image_id"]) in newly_covered
        }
        item = dict(metadata)
        item.update(
            {
                "incremental_union_image_ids": len(newly_covered),
                "incremental_overlap_eval_images": len(incremental_images),
                "incremental_overlap_eval_rows": sum(
                    int(int(row["image_id"]) in newly_covered) for row in records
                ),
                "standalone_overlap_eval_images": len(standalone_images),
                "standalone_overlap_eval_rows": sum(
                    int(int(row["image_id"]) in image_ids) for row in records
                ),
            }
        )
        audited.append(item)
        cumulative.update(image_ids)
    return audited


def _is_generic_stage_a_detection_source(metadata: Mapping[str, Any]) -> bool:
    """Identify the two generic detection sources excluded from semantic isolation."""
    return (
        metadata.get("dataset_mode") == "odvg"
        and Path(str(metadata.get("path", ""))).name
        in {
            "stagea_odvg_train_0_lvis.jsonl",
            "stagea_odvg_train_1_coco.jsonl",
        }
    )


def _semantic_stageb_source_audit(
    records: Sequence[Mapping[str, Any]],
    *,
    candidate_inputs: Sequence[Mapping[str, Any]],
    candidate_source_image_sets: Sequence[set[int]],
    baseline_inputs: Sequence[Mapping[str, Any]],
    baseline_source_image_sets: Sequence[set[int]],
) -> Tuple[set[int], List[Dict[str, Any]]]:
    """Build the task-specific Ref positive/TN image exclusion union and audit."""
    sources = [
        ("candidate", metadata, image_ids)
        for metadata, image_ids in zip(candidate_inputs, candidate_source_image_sets)
    ] + [
        ("gdino_stageb_baseline", metadata, image_ids)
        for metadata, image_ids in zip(baseline_inputs, baseline_source_image_sets)
    ]
    if len(sources) != len(candidate_inputs) + len(baseline_inputs):
        raise ValueError("Semantic Stage-B source metadata/image-set length mismatch")

    exclusion_union: set[int] = set()
    decisions: List[Dict[str, Any]] = []
    for origin, metadata, image_ids in sources:
        generic_stage_a = _is_generic_stage_a_detection_source(metadata)
        include = not generic_stage_a
        if generic_stage_a:
            rationale = (
                "Ignore this generic Stage-A LVIS/COCO detection ODVG source. It is shared "
                "localization pretraining rather than task-specific Ref positive/TN "
                "supervision; treating it as semantic leakage would cover every COCO dev "
                "image and make the supplemental metric undefined."
            )
        elif origin == "candidate":
            rationale = (
                "Include this candidate task-specific Ref positive/TN source in the image "
                "exclusion union."
            )
        else:
            rationale = (
                "Include this GDINO Stage-B task-specific Ref positive/TN source in the "
                "image exclusion union."
            )

        newly_excluded = image_ids.difference(exclusion_union) if include else set()
        standalone_overlap_images = {
            int(row["image_id"]) for row in records if int(row["image_id"]) in image_ids
        }
        incremental_overlap_images = {
            int(row["image_id"])
            for row in records
            if int(row["image_id"]) in newly_excluded
        }
        decision = dict(metadata)
        decision.update(
            {
                "included_in_semantic_exclusion_union": include,
                "incremental_exclusion_image_ids": len(newly_excluded),
                "incremental_overlap_eval_images": len(incremental_overlap_images),
                "incremental_overlap_eval_rows": sum(
                    int(int(row["image_id"]) in newly_excluded) for row in records
                ),
                "origin_config": origin,
                "rationale": rationale,
                "semantic_action": (
                    "exclude_eval_images_seen_by_source"
                    if include
                    else "ignore_generic_stage_a_detection_images"
                ),
                "standalone_overlap_eval_images": len(standalone_overlap_images),
                "standalone_overlap_eval_rows": sum(
                    int(int(row["image_id"]) in image_ids) for row in records
                ),
            }
        )
        decisions.append(decision)
        if include:
            exclusion_union.update(image_ids)
    return exclusion_union, decisions


def _source_record_hash(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(row)).hexdigest()


def _manifest_sort_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        str(row["eval_split"]),
        int(row["image_id"]),
        int(row["ann_id"]),
        int(row["ref_id"]),
        int(row["sent_id"]),
    )


def _validate_selected_row(
    row: Mapping[str, Any],
    *,
    context: str,
    data_root: Path,
) -> Dict[str, Any]:
    if row.get("visual_filter_status") != "accept":
        raise ValueError(f"Non-accepted VLM row selected at {context}")
    if row.get("visual_filter_reason") != "verified_negative":
        raise ValueError(f"Non-verified-negative VLM row selected at {context}")

    image_id = _int_field(row, "image_id", context)
    _int_field(row, "ann_id", context)
    _int_field(row, "ref_id", context)
    _int_field(row, "sent_id", context)
    _int_field(row, "class_id", context)
    positive = _nonempty_text(row, "sent", context)
    negative = _nonempty_text(row, "try_tn", context)
    if normalize_sentence(positive) == normalize_sentence(negative):
        raise ValueError(f"Positive and negative phrases are identical at {context}")
    _valid_xywh(row, "target_bbox_used", context)

    image_relpath = (
        Path("COCO")
        / "coco2014"
        / "train2014"
        / f"COCO_train2014_{image_id:012d}.jpg"
    )
    image_path = data_root / image_relpath
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing evaluation image at {context}: {image_path}")
    return proposal_coverage(row, context=context)


def _manifest_record(
    row: Mapping[str, Any],
    *,
    data_root: Path,
    eval_split: str,
    dataset: str,
    splitby: str,
    evaluation_pair_source: str,
    original_pair_source: str,
    official_ref_id: int,
    official_sent_id: int,
    source_path: Path,
    source_line: int,
    coverage: Mapping[str, Any],
    coverage_policy: str,
    google_split: str | None = None,
    g_umd_mapping_mode: str | None = None,
) -> Dict[str, Any]:
    image_id = int(row["image_id"])
    ann_id = int(row["ann_id"])
    accepted_ref_id = int(row["ref_id"])
    accepted_sent_id = int(row["sent_id"])
    ref_id = int(official_ref_id)
    sent_id = int(official_sent_id)
    class_id = int(row["class_id"])
    image_relpath = str(
        Path("COCO")
        / "coco2014"
        / "train2014"
        / f"COCO_train2014_{image_id:012d}.jpg"
    )
    image_path = str(data_root / image_relpath)
    positive_phrase = str(row["sent"]).strip()
    negative_phrase = str(row["try_tn"]).strip()
    canonical_name = str(
        row.get("class_norm_name")
        or row.get("category_name")
        or row.get("class_raw_name")
        or row.get("try_tn_head")
        or "object"
    ).strip()
    head = str(row.get("try_tn_head") or canonical_name).strip()
    head_phrase = str(row.get("try_tn_head_phrase") or head).strip()
    sample_id = (
        f"stageb-vlm-tn-v2:{eval_split}:{image_id}:{ann_id}:{ref_id}:{sent_id}"
    )
    coverage_pass = passes_coverage_policy(coverage, coverage_policy)
    bbox = [float(value) for value in row["target_bbox_used"]]
    instance = {
        "bbox": bbox,
        "canonical": canonical_name,
        "canonical_class_id": class_id,
        "canonical_name": canonical_name,
        "category_name": canonical_name,
        "class_id": class_id,
        "class_norm_name": canonical_name,
        "head": head,
        "head_phrase": head_phrase,
        "pair_source": evaluation_pair_source,
        "phrase": negative_phrase,
        "positive_phrase": positive_phrase,
        "raw_phrase": negative_phrase,
        "replace_category": row.get("replace_category"),
        "replace_from": row.get("replace_from"),
        "replace_span": row.get("replace_span"),
        "replace_to": row.get("replace_to"),
        "text_is_negative": True,
        "try_tn": negative_phrase,
        "try_tn_head": row.get("try_tn_head"),
        "try_tn_head_phrase": row.get("try_tn_head_phrase"),
        "try_tn_method": row.get("try_tn_method"),
        "visual_filter_reason": "verified_negative",
        "visual_filter_status": "accept",
        "visual_verified_negative": True,
    }
    record: Dict[str, Any] = {
        "accepted_ref_id": accepted_ref_id,
        "accepted_sent_id": accepted_sent_id,
        "ann_id": ann_id,
        "bbox": bbox,
        "box": bbox,
        "canonical": canonical_name,
        "canonical_class_id": class_id,
        "canonical_name": canonical_name,
        "category_name": canonical_name,
        "class_id": class_id,
        "class_norm_name": canonical_name,
        "coverage_pass": coverage_pass,
        "coverage_policy": coverage_policy,
        "eval_split": eval_split,
        "file_name": image_path,
        "filename": image_path,
        "image_id": image_id,
        "image_path": image_path,
        "image_relpath": image_relpath,
        "instances": [instance],
        "manifest_schema": SCHEMA_VERSION,
        "negative_phrase": negative_phrase,
        "official_dataset": dataset,
        "official_split": "val",
        "official_splitby": splitby,
        "original_pair_source": original_pair_source,
        "pair_source": evaluation_pair_source,
        "positive_phrase": positive_phrase,
        "proposal_audit": dict(coverage),
        "ref_id": ref_id,
        "sample_id": sample_id,
        "sent": positive_phrase,
        "sent_id": sent_id,
        "source": evaluation_pair_source,
        "source_file": str(source_path.resolve()),
        "source_line": int(source_line),
        "source_record_sha256": _source_record_hash(row),
        "split": "val",
        "stageb_manifest_source": f"stageb_vlm_verified_tn_{eval_split}",
        "target_bbox_used": bbox,
        "tn_eval_pair_source": evaluation_pair_source,
        "tn_eval_source_split": "val",
        "tn_eval_split": eval_split,
        "try_tn": negative_phrase,
        "visual_filter_reason": "verified_negative",
        "visual_filter_status": "accept",
        "visual_verified_negative": True,
    }
    if google_split is not None:
        record["refcocog_accepted_annotation_splitby"] = "google"
        record["refcocog_google_ref_id"] = accepted_ref_id
        record["refcocog_google_sent_id"] = accepted_sent_id
        record["refcocog_google_split"] = google_split
        record["refcocog_umd_mapping_mode"] = g_umd_mapping_mode
    return record


def _validate_manifest_records(
    records: Iterable[Mapping[str, Any]],
    *,
    require_coverage_pass: bool,
) -> Dict[str, Any]:
    rows = list(records)
    sample_ids: set[str] = set()
    identity_keys: set[Tuple[Any, ...]] = set()
    previous_key: Tuple[Any, ...] | None = None
    duplicate_sample_ids = 0
    duplicate_identity_keys = 0
    for index, row in enumerate(rows):
        if row.get("manifest_schema") != SCHEMA_VERSION:
            raise ValueError(f"Wrong manifest schema at row {index + 1}")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Missing sample_id at manifest row {index + 1}")
        if sample_id in sample_ids:
            duplicate_sample_ids += 1
        sample_ids.add(sample_id)
        key = _manifest_sort_key(row)
        if previous_key is not None and key < previous_key:
            raise ValueError(f"Manifest is not stably sorted at row {index + 1}")
        previous_key = key
        identity = (
            str(row["eval_split"]),
            int(row["image_id"]),
            int(row["ann_id"]),
            int(row["ref_id"]),
            int(row["sent_id"]),
        )
        if identity in identity_keys:
            duplicate_identity_keys += 1
        identity_keys.add(identity)
        if require_coverage_pass and row.get("coverage_pass") is not True:
            raise ValueError(f"Non-covered row present in eval manifest: {sample_id}")
        instances = row.get("instances")
        if not isinstance(instances, list) or len(instances) != 1:
            raise ValueError(f"Manifest row must contain exactly one instance: {sample_id}")
        instance_bbox = _valid_xywh(instances[0], "bbox", sample_id)
        top_bbox = _valid_xywh(row, "box", sample_id)
        if instance_bbox != top_bbox:
            raise ValueError(f"Top-level/instance bbox mismatch: {sample_id}")
        filename = _nonempty_text(row, "filename", sample_id)
        if not Path(filename).is_absolute():
            raise ValueError(f"Manifest filename must be absolute: {sample_id}")
        pair_source = _nonempty_text(row, "pair_source", sample_id)
        if instances[0].get("pair_source") != pair_source:
            raise ValueError(f"Top-level/instance pair_source mismatch: {sample_id}")
        _int_field(row, "class_id", sample_id)
        _nonempty_text(row, "canonical", sample_id)
        _nonempty_text(row, "positive_phrase", sample_id)
        _nonempty_text(row, "negative_phrase", sample_id)
    if duplicate_sample_ids or duplicate_identity_keys:
        raise ValueError(
            "Manifest duplicates detected: "
            f"sample_ids={duplicate_sample_ids}, identity_keys={duplicate_identity_keys}"
        )
    return {
        "rows": len(rows),
        "duplicate_sample_ids": duplicate_sample_ids,
        "duplicate_identity_keys": duplicate_identity_keys,
        "invalid_rows": 0,
    }


def _manifest_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in records)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def validate_manifest_file(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_rows: int | None = None,
    require_coverage_pass: bool,
) -> Dict[str, Any]:
    records = [row for _, row in _iter_jsonl(path)]
    validation = _validate_manifest_records(
        records,
        require_coverage_pass=require_coverage_pass,
    )
    actual_sha256 = sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"Manifest hash mismatch for {path}: {actual_sha256} != {expected_sha256}"
        )
    if expected_rows is not None and len(records) != int(expected_rows):
        raise ValueError(
            f"Manifest row-count mismatch for {path}: {len(records)} != {expected_rows}"
        )
    validation["sha256"] = actual_sha256
    return validation


def verify_output_dir(output_dir: Path) -> Dict[str, Any]:
    audit_path = output_dir / AUDIT_NAME
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid audit JSON {audit_path}: {exc}") from exc
    if audit.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unexpected audit schema in {audit_path}")
    manifests = audit.get("manifests")
    if not isinstance(manifests, dict):
        raise ValueError(f"Missing manifests section in {audit_path}")
    strict_meta = manifests.get("strict_ann")
    eval_meta = manifests.get("eval")
    if not isinstance(strict_meta, dict) or not isinstance(eval_meta, dict):
        raise ValueError(f"Incomplete manifests section in {audit_path}")
    verified: Dict[str, Any] = {}
    for name, metadata in manifests.items():
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid manifest metadata {name!r} in {audit_path}")
        verified[name] = validate_manifest_file(
            output_dir / str(metadata["file"]),
            expected_sha256=str(metadata["sha256"]),
            expected_rows=int(metadata["rows"]),
            require_coverage_pass=name != "strict_ann",
        )
    return verified


def build_manifests(
    *,
    data_root: Path,
    train_config: Path,
    candidate_train_config: Path = DEFAULT_CANDIDATE_TRAIN_CONFIG,
    gdino_stageb_train_config: Path = DEFAULT_GDINO_STAGEB_TRAIN_CONFIG,
    plus_accepted: Path,
    g_accepted: Path,
    output_dir: Path,
    coverage_policy: str = COVERAGE_TARGET_PLUS_PROPOSAL,
    expected_strict_count: int | None = 2_035,
    expected_eval_count: int | None = 2_031,
) -> Dict[str, Any]:
    if coverage_policy not in COVERAGE_POLICIES:
        raise ValueError(f"Unknown coverage policy: {coverage_policy!r}")
    data_root = data_root.resolve()
    train_config = train_config.resolve()
    candidate_train_config = candidate_train_config.resolve()
    gdino_stageb_train_config = gdino_stageb_train_config.resolve()
    plus_accepted = plus_accepted.resolve()
    g_accepted = g_accepted.resolve()
    output_dir = output_dir.resolve()

    plus_refs = data_root / "COCO" / "refcoco+" / "refs(unc).p"
    g_umd_refs = data_root / "COCO" / "refcocog" / "refs(umd).p"
    g_google_refs = data_root / "COCO" / "refcocog" / "refs(google).p"
    plus_map, plus_map_duplicates = _build_plus_map(plus_refs)
    (
        g_umd_map,
        g_umd_identities,
        g_google_map,
        g_umd_map_duplicates,
        g_google_map_duplicates,
    ) = _build_g_maps(
        g_umd_refs,
        g_google_refs,
    )
    holdout_ann_keys, holdout_image_ids, holdout_inputs = _load_holdout(
        train_config,
        data_root,
    )

    records: List[Dict[str, Any]] = []
    source_counts: Dict[str, Counter[str]] = {
        "refcocoplus": Counter(),
        "refcocog": Counter(),
    }
    input_identity_keys: set[Tuple[str, int, int, int, int]] = set()

    for source_name, source_path in (
        ("refcocoplus", plus_accepted),
        ("refcocog", g_accepted),
    ):
        counts = source_counts[source_name]
        for line_number, row in _iter_jsonl(source_path):
            context = f"{source_path}:{line_number}"
            counts["input_rows"] += 1
            if row.get("visual_filter_status") != "accept":
                raise ValueError(f"Unexpected non-accepted row in accepted file at {context}")
            if row.get("visual_filter_reason") != "verified_negative":
                raise ValueError(f"Unexpected VLM reason in accepted file at {context}")
            image_id = _int_field(row, "image_id", context)
            ann_id = _int_field(row, "ann_id", context)
            ref_id = _int_field(row, "ref_id", context)
            sent_id = _int_field(row, "sent_id", context)
            input_identity = (source_name, image_id, ann_id, ref_id, sent_id)
            if input_identity in input_identity_keys:
                raise ValueError(f"Duplicate accepted-row identity at {context}: {input_identity}")
            input_identity_keys.add(input_identity)

            if source_name == "refcocoplus":
                split_key = (ref_id, ann_id, sent_id)
                if split_key not in plus_map:
                    raise KeyError(f"Missing RefCOCO+ UNC split mapping at {context}: {split_key}")
                eval_split_value = plus_map[split_key]
                counts[f"unc_split_{eval_split_value}_rows"] += 1
                if eval_split_value != "val":
                    continue
                eval_split = "refcocoplus_unc_val"
                dataset = "refcoco+"
                splitby = "unc"
                evaluation_pair_source = "refcoco+_unc"
                original_pair_source = str(row.get("pair_source") or "refcoco+_unc")
                official_ref_id = ref_id
                official_sent_id = sent_id
                google_split = None
                g_umd_mapping_mode = None
            else:
                normalized = normalize_sentence(row.get("sent"))
                umd_key = (image_id, ann_id, normalized)
                google_key = (ref_id, ann_id, image_id)
                if umd_key not in g_umd_map:
                    raise KeyError(f"Missing RefCOCOg UMD split mapping at {context}: {umd_key}")
                if google_key not in g_google_map:
                    raise KeyError(
                        f"Missing RefCOCOg Google split mapping at {context}: {google_key}"
                    )
                (
                    official_ref_id,
                    official_sent_id,
                    eval_split_value,
                    g_umd_mapping_mode,
                ) = _resolve_g_umd_identity(
                    g_umd_identities[umd_key],
                    accepted_sent_id=sent_id,
                    context=context,
                )
                if eval_split_value != g_umd_map[umd_key]:
                    raise ValueError(
                        f"RefCOCOg UMD identity/split mismatch at {context}: "
                        f"{eval_split_value!r} != {g_umd_map[umd_key]!r}"
                    )
                google_split = g_google_map[google_key]
                counts[f"umd_mapping_{g_umd_mapping_mode}_rows"] += 1
                counts[f"umd_split_{eval_split_value}_rows"] += 1
                counts[f"google_split_{google_split}_rows"] += 1
                counts[f"google_{google_split}_umd_{eval_split_value}_rows"] += 1
                if (image_id, ann_id) not in holdout_ann_keys and google_split == "val":
                    counts["google_val_strict_ann_rows"] += 1
                if eval_split_value != "val":
                    continue
                eval_split = "refcocog_umd_val"
                dataset = "refcocog"
                splitby = "umd"
                evaluation_pair_source = "refcocog_umd"
                original_pair_source = str(row.get("pair_source") or "refcocog_google")

            counts["official_val_rows"] += 1
            if (image_id, ann_id) in holdout_ann_keys:
                counts["excluded_train_ann_rows"] += 1
                continue
            counts["strict_ann_rows"] += 1
            coverage = _validate_selected_row(row, context=context, data_root=data_root)
            counts["target_plus_proposal_covered_rows"] += int(
                coverage["target_plus_proposal_covered"]
            )
            counts["all_proposals_all_no_rows"] += int(coverage["all_proposals_all_no"])
            record = _manifest_record(
                row,
                data_root=data_root,
                eval_split=eval_split,
                dataset=dataset,
                splitby=splitby,
                evaluation_pair_source=evaluation_pair_source,
                original_pair_source=original_pair_source,
                official_ref_id=official_ref_id,
                official_sent_id=official_sent_id,
                source_path=source_path,
                source_line=line_number,
                coverage=coverage,
                coverage_policy=coverage_policy,
                google_split=google_split,
                g_umd_mapping_mode=g_umd_mapping_mode,
            )
            records.append(record)

    records.sort(key=_manifest_sort_key)
    eval_records = [row for row in records if row["coverage_pass"]]
    strict_validation = _validate_manifest_records(records, require_coverage_pass=False)
    eval_validation = _validate_manifest_records(eval_records, require_coverage_pass=True)

    strict_counts = Counter(str(row["eval_split"]) for row in records)
    eval_counts = Counter(str(row["eval_split"]) for row in eval_records)
    if expected_strict_count is not None and len(records) != int(expected_strict_count):
        raise ValueError(
            f"Unexpected strict manifest count: {len(records)} != {expected_strict_count}"
        )
    if expected_eval_count is not None and len(eval_records) != int(expected_eval_count):
        raise ValueError(
            f"Unexpected eval manifest count: {len(eval_records)} != {expected_eval_count}"
        )
    if coverage_policy == COVERAGE_TARGET_PLUS_PROPOSAL:
        if dict(strict_counts) != EXPECTED_STRICT_COUNTS:
            raise ValueError(
                f"Unexpected strict per-split counts: {dict(strict_counts)} != "
                f"{EXPECTED_STRICT_COUNTS}"
            )
        if dict(eval_counts) != EXPECTED_TARGET_PLUS_PROPOSAL_COUNTS:
            raise ValueError(
                f"Unexpected eval per-split counts: {dict(eval_counts)} != "
                f"{EXPECTED_TARGET_PLUS_PROPOSAL_COUNTS}"
            )

    candidate_image_ids, candidate_inputs, candidate_source_image_sets = (
        _load_train_image_sources(candidate_train_config, data_root)
    )
    baseline_image_ids, baseline_inputs, baseline_source_image_sets = (
        _load_train_image_sources(gdino_stageb_train_config, data_root)
    )
    semantic_stageb_union_image_ids, semantic_stageb_source_decisions = (
        _semantic_stageb_source_audit(
            eval_records,
            candidate_inputs=candidate_inputs,
            candidate_source_image_sets=candidate_source_image_sets,
            baseline_inputs=baseline_inputs,
            baseline_source_image_sets=baseline_source_image_sets,
        )
    )
    global_tn_image_ids: set[int] = set()
    for metadata, image_ids in zip(candidate_inputs, candidate_source_image_sets):
        if metadata["require_global_tn_verified"]:
            global_tn_image_ids.update(image_ids)

    global_tn_diagnostic_records = [
        row for row in eval_records if int(row["image_id"]) not in global_tn_image_ids
    ]
    candidate_image_disjoint_records = [
        row for row in eval_records if int(row["image_id"]) not in candidate_image_ids
    ]
    semantic_stageb_union_image_disjoint_records = [
        row
        for row in eval_records
        if int(row["image_id"]) not in semantic_stageb_union_image_ids
    ]
    candidate_baseline_union_image_ids = candidate_image_ids.union(baseline_image_ids)
    baseline_union_diagnostic_records = [
        row
        for row in eval_records
        if int(row["image_id"]) not in candidate_baseline_union_image_ids
    ]
    global_tn_diagnostic_validation = _validate_manifest_records(
        global_tn_diagnostic_records,
        require_coverage_pass=True,
    )
    candidate_image_disjoint_validation = _validate_manifest_records(
        candidate_image_disjoint_records,
        require_coverage_pass=True,
    )
    semantic_stageb_union_image_disjoint_validation = _validate_manifest_records(
        semantic_stageb_union_image_disjoint_records,
        require_coverage_pass=True,
    )
    baseline_union_diagnostic_validation = _validate_manifest_records(
        baseline_union_diagnostic_records,
        require_coverage_pass=True,
    )
    global_tn_diagnostic_counts = Counter(
        str(row["eval_split"]) for row in global_tn_diagnostic_records
    )
    candidate_image_disjoint_counts = Counter(
        str(row["eval_split"]) for row in candidate_image_disjoint_records
    )
    semantic_stageb_union_image_disjoint_counts = Counter(
        str(row["eval_split"]) for row in semantic_stageb_union_image_disjoint_records
    )
    if coverage_policy == COVERAGE_TARGET_PLUS_PROPOSAL:
        if dict(global_tn_diagnostic_counts) != EXPECTED_GLOBAL_TN_DIAGNOSTIC_COUNTS:
            raise ValueError(
                "Unexpected global-TN-only image-disjoint diagnostic counts: "
                f"{dict(global_tn_diagnostic_counts)} != {EXPECTED_GLOBAL_TN_DIAGNOSTIC_COUNTS}"
            )
        if dict(candidate_image_disjoint_counts) != EXPECTED_CANDIDATE_IMAGE_DISJOINT_COUNTS:
            raise ValueError(
                "Unexpected candidate image-disjoint counts: "
                f"{dict(candidate_image_disjoint_counts)} != "
                f"{EXPECTED_CANDIDATE_IMAGE_DISJOINT_COUNTS}"
            )
        if (
            dict(semantic_stageb_union_image_disjoint_counts)
            != EXPECTED_SEMANTIC_STAGEB_UNION_IMAGE_DISJOINT_COUNTS
        ):
            raise ValueError(
                "Unexpected semantic Stage-B union image-disjoint counts: "
                f"{dict(semantic_stageb_union_image_disjoint_counts)} != "
                f"{EXPECTED_SEMANTIC_STAGEB_UNION_IMAGE_DISJOINT_COUNTS}"
            )
        if baseline_union_diagnostic_records:
            raise ValueError(
                "Expected zero image-disjoint rows against the candidate/GDINO Stage-B "
                "training-source union"
            )

    strict_payload = _manifest_bytes(records)
    eval_payload = _manifest_bytes(eval_records)
    global_tn_diagnostic_payload = _manifest_bytes(global_tn_diagnostic_records)
    candidate_image_disjoint_payload = _manifest_bytes(candidate_image_disjoint_records)
    semantic_stageb_union_image_disjoint_payload = _manifest_bytes(
        semantic_stageb_union_image_disjoint_records
    )
    baseline_union_diagnostic_payload = _manifest_bytes(baseline_union_diagnostic_records)
    strict_sha256 = hashlib.sha256(strict_payload).hexdigest()
    eval_sha256 = hashlib.sha256(eval_payload).hexdigest()
    global_tn_diagnostic_sha256 = hashlib.sha256(global_tn_diagnostic_payload).hexdigest()
    candidate_image_disjoint_sha256 = hashlib.sha256(candidate_image_disjoint_payload).hexdigest()
    semantic_stageb_union_image_disjoint_sha256 = hashlib.sha256(
        semantic_stageb_union_image_disjoint_payload
    ).hexdigest()
    baseline_union_diagnostic_sha256 = hashlib.sha256(
        baseline_union_diagnostic_payload
    ).hexdigest()
    exclusions = [
        {
            "eval_split": row["eval_split"],
            "proposal_audit": row["proposal_audit"],
            "reason": f"coverage_policy_failed:{coverage_policy}",
            "sample_id": row["sample_id"],
            "source_file": row["source_file"],
            "source_line": row["source_line"],
        }
        for row in records
        if not row["coverage_pass"]
    ]

    input_files = {
        "refcocoplus_accepted": plus_accepted,
        "refcocog_accepted_google_annotations": g_accepted,
        "refcocoplus_unc_refs": plus_refs,
        "refcocog_umd_refs": g_umd_refs,
        "refcocog_google_refs_compatibility_audit": g_google_refs,
        "train_dataset_config": train_config,
        "candidate_train_dataset_config": candidate_train_config,
        "gdino_stageb_train_dataset_config": gdino_stageb_train_config,
    }
    eval_image_ids = {int(row["image_id"]) for row in eval_records}
    global_tn_overlap_images = eval_image_ids.intersection(global_tn_image_ids)
    candidate_overlap_images = eval_image_ids.intersection(candidate_image_ids)
    semantic_stageb_overlap_images = eval_image_ids.intersection(
        semantic_stageb_union_image_ids
    )
    baseline_overlap_images = eval_image_ids.intersection(baseline_image_ids)
    union_overlap_images = eval_image_ids.intersection(candidate_baseline_union_image_ids)
    audit: Dict[str, Any] = {
        "coverage_policy": {
            "all_proposals_all_no": (
                "target-local answer is no; proposal cache is non-empty; every proposal "
                "has a nested judgment; all nested answers are no"
            ),
            "selected": coverage_policy,
            "target_plus_proposal": (
                "target-local answer is no and proposal_num <= "
                "len(visual_proposal_judgments) + 1"
            ),
        },
        "exclusions": exclusions,
        "holdout": {
            "level": "ann",
            "key": ["image_id", "ann_id"],
            "train_annotation_keys": len(holdout_ann_keys),
            "train_image_ids": len(holdout_image_ids),
            "train_jsonls": holdout_inputs,
        },
        "image_disjoint_audit": {
            "all_configured_sources_image_disjoint_feasible": bool(
                baseline_union_diagnostic_records
            ),
            "primary_protocols": {
                "same_annotation_holdout_compatibility": EVAL_MANIFEST_NAME,
                "strict_semantic_stageb_union_supplemental": (
                    SEMANTIC_STAGEB_UNION_IMAGE_DISJOINT_MANIFEST_NAME
                ),
            },
            "candidate_train_config": {
                "config": str(candidate_train_config),
                "config_sha256": sha256_file(candidate_train_config),
                "excluded_eval_images": len(candidate_overlap_images),
                "excluded_eval_rows": len(eval_records) - len(candidate_image_disjoint_records),
                "remaining_eval_images": len(
                    {int(row["image_id"]) for row in candidate_image_disjoint_records}
                ),
                "remaining_eval_rows": len(candidate_image_disjoint_records),
                "role": "candidate-model-only image-disjoint subset; not cross-model final",
                "source_contributions": _source_overlap_audit(
                    eval_records,
                    candidate_inputs,
                    candidate_source_image_sets,
                ),
                "train_union_image_ids": len(candidate_image_ids),
            },
            "gdino_stageb_retrain_config": {
                "config": str(gdino_stageb_train_config),
                "config_sha256": sha256_file(gdino_stageb_train_config),
                "excluded_eval_images": len(baseline_overlap_images),
                "excluded_eval_rows": sum(
                    int(int(row["image_id"]) in baseline_image_ids) for row in eval_records
                ),
                "remaining_eval_images": len(eval_image_ids.difference(baseline_image_ids)),
                "remaining_eval_rows": sum(
                    int(int(row["image_id"]) not in baseline_image_ids) for row in eval_records
                ),
                "source_contributions": _source_overlap_audit(
                    eval_records,
                    baseline_inputs,
                    baseline_source_image_sets,
                ),
                "train_union_image_ids": len(baseline_image_ids),
            },
            "global_tn_only_compatibility": {
                "excluded_eval_images": len(global_tn_overlap_images),
                "excluded_eval_rows": len(eval_records) - len(global_tn_diagnostic_records),
                "not_authoritative": True,
                "remaining_eval_images": len(
                    {int(row["image_id"]) for row in global_tn_diagnostic_records}
                ),
                "remaining_eval_rows": len(global_tn_diagnostic_records),
                "role": "diagnostic only: excludes only require_global_tn_verified sources",
                "train_union_image_ids": len(global_tn_image_ids),
            },
            "semantic_stageb_union": {
                "currently_byte_equivalent_to_candidate_only": (
                    semantic_stageb_union_image_disjoint_payload
                    == candidate_image_disjoint_payload
                ),
                "definition": (
                    "Exclude images seen by candidate or GDINO baseline task-specific Ref "
                    "positive/TN sources; ignore generic Stage-A LVIS/COCO detection ODVG."
                ),
                "excluded_eval_images": len(semantic_stageb_overlap_images),
                "excluded_eval_rows": (
                    len(eval_records) - len(semantic_stageb_union_image_disjoint_records)
                ),
                "included_source_count": sum(
                    int(item["included_in_semantic_exclusion_union"])
                    for item in semantic_stageb_source_decisions
                ),
                "ignored_generic_stage_a_source_count": sum(
                    int(not item["included_in_semantic_exclusion_union"])
                    for item in semantic_stageb_source_decisions
                ),
                "protocol_role": "primary strict supplemental image-disjoint set",
                "remaining_eval_images": len(
                    {
                        int(row["image_id"])
                        for row in semantic_stageb_union_image_disjoint_records
                    }
                ),
                "remaining_eval_rows": len(semantic_stageb_union_image_disjoint_records),
                "source_decisions": semantic_stageb_source_decisions,
                "task_specific_train_union_image_ids": len(
                    semantic_stageb_union_image_ids
                ),
            },
            "candidate_and_gdino_stageb_union": {
                "excluded_eval_images": len(union_overlap_images),
                "excluded_eval_rows": len(eval_records) - len(baseline_union_diagnostic_records),
                "feasible": bool(baseline_union_diagnostic_records),
                "reason_if_infeasible": (
                    "Every evaluator-compatible development row shares its COCO image_id "
                    "with at least one configured training source; the Stage-A COCO source "
                    "in the GDINO Stage-B retraining config alone covers the full set."
                    if not baseline_union_diagnostic_records
                    else None
                ),
                "remaining_eval_images": len(
                    {int(row["image_id"]) for row in baseline_union_diagnostic_records}
                ),
                "remaining_eval_rows": len(baseline_union_diagnostic_records),
                "role": "diagnostic only; never use this zero-row set as a final metric",
                "train_union_image_ids": len(candidate_baseline_union_image_ids),
            },
            "eval_image_ids": len(eval_image_ids),
            "eval_rows": len(eval_records),
        },
        "inputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for name, path in input_files.items()
        },
        "manifests": {
            "baseline_union_image_disjoint_diagnostic": {
                "duplicate_identity_keys": baseline_union_diagnostic_validation[
                    "duplicate_identity_keys"
                ],
                "duplicate_sample_ids": baseline_union_diagnostic_validation[
                    "duplicate_sample_ids"
                ],
                "file": BASELINE_UNION_DIAGNOSTIC_MANIFEST_NAME,
                "invalid_rows": baseline_union_diagnostic_validation["invalid_rows"],
                "not_final": True,
                "role": "diagnostic proof of cross-model image-disjoint infeasibility",
                "rows": len(baseline_union_diagnostic_records),
                "sha256": baseline_union_diagnostic_sha256,
                "split_counts": {},
            },
            "candidate_image_disjoint": {
                "duplicate_identity_keys": candidate_image_disjoint_validation[
                    "duplicate_identity_keys"
                ],
                "duplicate_sample_ids": candidate_image_disjoint_validation[
                    "duplicate_sample_ids"
                ],
                "file": CANDIDATE_IMAGE_DISJOINT_MANIFEST_NAME,
                "invalid_rows": candidate_image_disjoint_validation["invalid_rows"],
                "role": "candidate-model-only image-disjoint subset",
                "rows": len(candidate_image_disjoint_records),
                "sha256": candidate_image_disjoint_sha256,
                "split_counts": dict(sorted(candidate_image_disjoint_counts.items())),
            },
            "eval": {
                "duplicate_identity_keys": eval_validation["duplicate_identity_keys"],
                "duplicate_sample_ids": eval_validation["duplicate_sample_ids"],
                "file": EVAL_MANIFEST_NAME,
                "invalid_rows": eval_validation["invalid_rows"],
                "protocol_role": "primary annotation-holdout compatibility set",
                "rows": len(eval_records),
                "sha256": eval_sha256,
                "split_counts": dict(sorted(eval_counts.items())),
            },
            "global_tn_only_image_disjoint_diagnostic": {
                "duplicate_identity_keys": global_tn_diagnostic_validation[
                    "duplicate_identity_keys"
                ],
                "duplicate_sample_ids": global_tn_diagnostic_validation[
                    "duplicate_sample_ids"
                ],
                "file": GLOBAL_TN_DIAGNOSTIC_MANIFEST_NAME,
                "invalid_rows": global_tn_diagnostic_validation["invalid_rows"],
                "not_authoritative": True,
                "role": "compatibility diagnostic only",
                "rows": len(global_tn_diagnostic_records),
                "sha256": global_tn_diagnostic_sha256,
                "split_counts": dict(sorted(global_tn_diagnostic_counts.items())),
            },
            "semantic_stageb_union_image_disjoint": {
                "duplicate_identity_keys": semantic_stageb_union_image_disjoint_validation[
                    "duplicate_identity_keys"
                ],
                "duplicate_sample_ids": semantic_stageb_union_image_disjoint_validation[
                    "duplicate_sample_ids"
                ],
                "file": SEMANTIC_STAGEB_UNION_IMAGE_DISJOINT_MANIFEST_NAME,
                "invalid_rows": semantic_stageb_union_image_disjoint_validation[
                    "invalid_rows"
                ],
                "protocol_role": "primary strict supplemental image-disjoint set",
                "rows": len(semantic_stageb_union_image_disjoint_records),
                "sha256": semantic_stageb_union_image_disjoint_sha256,
                "split_counts": dict(
                    sorted(semantic_stageb_union_image_disjoint_counts.items())
                ),
            },
            "strict_ann": {
                "duplicate_identity_keys": strict_validation["duplicate_identity_keys"],
                "duplicate_sample_ids": strict_validation["duplicate_sample_ids"],
                "file": STRICT_MANIFEST_NAME,
                "invalid_rows": strict_validation["invalid_rows"],
                "rows": len(records),
                "sha256": strict_sha256,
                "split_counts": dict(sorted(strict_counts.items())),
            },
        },
        "schema_version": SCHEMA_VERSION,
        "source_counts": {
            name: dict(sorted(counts.items()))
            for name, counts in source_counts.items()
        },
        "split_protocol": {
            "refcocoplus_unc_val": {
                "accepted_annotations": "RefCOCO+ UNC",
                "evaluation_split": "val",
                "mapping_key": ["ref_id", "ann_id", "sent_id"],
                "pair_source": "refcoco+_unc",
            },
            "refcocog_google_val_compatibility_only": {
                "evaluation_selected": False,
                "mapping_key": ["ref_id", "ann_id", "image_id"],
                "note": "This is not the RefCOCOg UMD validation split.",
                "pair_source": "refcocog_google",
            },
            "refcocog_umd_val": {
                "accepted_annotations": "RefCOCOg Google",
                "evaluation_split": "val",
                "mapping_key": ["image_id", "ann_id", "normalized_sentence"],
                "normalization": "lowercase; replace '_' and '.' with spaces; collapse whitespace",
                "output_identity": "official RefCOCOg UMD ref_id and sent_id",
                "pair_source": "refcocog_umd",
                "tie_break": "accepted Google sent_id, otherwise fail closed unless unique",
                "splitby": "umd",
            },
        },
        "split_map_duplicates": {
            "refcocoplus_unc": plus_map_duplicates,
            "refcocog_google": g_google_map_duplicates,
            "refcocog_umd_normalized_sentence": g_umd_map_duplicates,
        },
        "validation": {
            "duplicate_identity_keys": 0,
            "duplicate_sample_ids": 0,
            "invalid_rows": 0,
            "passed": True,
            "stable_sort_key": [
                "eval_split",
                "image_id",
                "ann_id",
                "ref_id",
                "sent_id",
            ],
        },
    }

    _atomic_write(output_dir / STRICT_MANIFEST_NAME, strict_payload)
    _atomic_write(output_dir / EVAL_MANIFEST_NAME, eval_payload)
    _atomic_write(
        output_dir / CANDIDATE_IMAGE_DISJOINT_MANIFEST_NAME,
        candidate_image_disjoint_payload,
    )
    _atomic_write(
        output_dir / SEMANTIC_STAGEB_UNION_IMAGE_DISJOINT_MANIFEST_NAME,
        semantic_stageb_union_image_disjoint_payload,
    )
    _atomic_write(
        output_dir / GLOBAL_TN_DIAGNOSTIC_MANIFEST_NAME,
        global_tn_diagnostic_payload,
    )
    _atomic_write(
        output_dir / BASELINE_UNION_DIAGNOSTIC_MANIFEST_NAME,
        baseline_union_diagnostic_payload,
    )
    _atomic_write(
        output_dir / AUDIT_NAME,
        json.dumps(audit, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False).encode(
            "ascii"
        )
        + b"\n",
    )
    verify_output_dir(output_dir)
    return audit


def _optional_expected(value: int) -> int | None:
    return None if int(value) < 0 else int(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify the frozen VLM-verified strict-ann Stage-B TN manifest."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument(
        "--candidate-train-config",
        type=Path,
        default=DEFAULT_CANDIDATE_TRAIN_CONFIG,
    )
    parser.add_argument(
        "--gdino-stageb-train-config",
        type=Path,
        default=DEFAULT_GDINO_STAGEB_TRAIN_CONFIG,
    )
    parser.add_argument("--refcocoplus-accepted", type=Path, default=DEFAULT_PLUS_ACCEPTED)
    parser.add_argument("--refcocog-accepted", type=Path, default=DEFAULT_G_ACCEPTED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--coverage-policy", choices=COVERAGE_POLICIES, default=COVERAGE_TARGET_PLUS_PROPOSAL)
    parser.add_argument(
        "--expected-strict-count",
        type=int,
        default=2_035,
        help="Fail unless the strict-ann candidate count matches; negative disables the check.",
    )
    parser.add_argument(
        "--expected-eval-count",
        type=int,
        default=None,
        help=(
            "Fail unless the selected eval count matches. Defaults to 2031 for "
            "target_plus_proposal and disabled for other policies."
        ),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing manifest hashes, order, uniqueness, and coverage without rebuilding.",
    )
    args = parser.parse_args()

    if args.verify_only:
        result = verify_output_dir(args.output_dir.resolve())
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    expected_eval = args.expected_eval_count
    if expected_eval is None and args.coverage_policy == COVERAGE_TARGET_PLUS_PROPOSAL:
        expected_eval = 2_031
    audit = build_manifests(
        data_root=args.data_root,
        train_config=args.train_config,
        candidate_train_config=args.candidate_train_config,
        gdino_stageb_train_config=args.gdino_stageb_train_config,
        plus_accepted=args.refcocoplus_accepted,
        g_accepted=args.refcocog_accepted,
        output_dir=args.output_dir,
        coverage_policy=args.coverage_policy,
        expected_strict_count=_optional_expected(args.expected_strict_count),
        expected_eval_count=(None if expected_eval is None else _optional_expected(expected_eval)),
    )
    print(
        json.dumps(
            {
                "audit": str((args.output_dir / AUDIT_NAME).resolve()),
                "coverage_policy": args.coverage_policy,
                "candidate_image_disjoint": audit["manifests"]["candidate_image_disjoint"],
                "eval": audit["manifests"]["eval"],
                "semantic_stageb_union_image_disjoint": audit["manifests"][
                    "semantic_stageb_union_image_disjoint"
                ],
                "strict_ann": audit["manifests"]["strict_ann"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
