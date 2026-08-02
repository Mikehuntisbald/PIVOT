#!/usr/bin/env python3
"""Verify fixed-GDINO max-region judgments and build fail-closed TN pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.judge_stageb_fixed_gdino_top1_qwen import (
    ASSET_POLICY_SHA256,
    EXTRACTION_SCHEMA,
    GENERATION_CONFIG_SHA256,
    INHERIT_CONFIDENCE_THRESHOLD,
    INHERIT_IOU_THRESHOLD,
    INFERENCE_BATCH_SIZE,
    JUDGE_RUNTIME_POLICY,
    JUDGE_RUNTIME_POLICY_SHA256,
    JUDGMENT_SCHEMA,
    MODEL_ID,
    MODEL_REVISION,
    PROMPT_TEMPLATE_SHA256,
    QwenJudgeError,
    VISION_PROCESSOR_CONFIG_SHA256,
    canonical_sha256,
    file_record,
    iter_jsonl,
    judgment_cache_key,
    region_bbox_xyxy_original,
    render_prompt,
    sha256_file,
    validate_extraction_row,
    validate_judgment_record,
)


AUDIT_SCHEMA = "stage-b-fixed-gdino-top1-vlm-results-audit-v1"
AUDIT_KIND = "completed_fixed_gdino_top1_vlm_results_verification"
ACCEPTED_PAIR_SCHEMA = "stage-b-gdino-adapter-fixed-top1-verified-pair-v1"
DECISION_SCHEMA = "stage-b-fixed-gdino-top1-vlm-decision-v1"
FIXED_MAX_SCOPE = (
    "checkpoint_train_and_deploy_transform_specific_top1_union_verified"
)

EXPECTED_FORWARD_CONTRACT = {
    "primary": "confidence_train_paired_2b_pos_then_neg_local_b4",
    "shadow": "formal_eval_separate_negative_then_positive_local_b4",
    "deploy": "formal_eval_val_resize_separate_negative_then_positive_b16",
    "local_batch_size": 4,
    "paired_batch_size": 8,
    "deploy_batch_size": 16,
}

DEFAULT_STRICT2031 = (
    REPO_ROOT
    / "data/eval_manifests/stageb_vlm_verified_strict_ann_umd_val_20260711"
    / "eval_manifest.jsonl"
)
DEFAULT_STRICT1607 = (
    REPO_ROOT
    / "data/eval_manifests/stageb_vlm_verified_strict_ann_umd_val_20260711"
    / "semantic_stageb_union_image_disjoint_manifest.jsonl"
)
STRICT2031_SHA256 = "0e47763c01178d63ee22430a6c93d4fc6a210848d43f32aafbb2e6cd7243e918"
STRICT1607_SHA256 = "f2dc97d58884b8de3ae2c8b4cefd281432e15c8952b23b5e0252eb8e5be36d25"
QWEN_CONFIDENCE_THRESHOLD = 0.90
EXTRACTION_AUDIT_SCHEMA = "stage-b-fixed-gdino-top1-vlm-extraction-audit-v1"
EXTRACTION_AUDIT_KIND = "completed_fixed_gdino_top1_vlm_extraction"
EXPECTED_EXTRACTION_ROWS = 17_738
EXPECTED_TIE_EPSILON = 1.0e-3


class VerificationError(RuntimeError):
    pass


_FILE_SHA_CACHE: Dict[Path, tuple[int, int, str]] = {}


def _cached_file_sha256(path: Path) -> str:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise VerificationError(f"required hashed file is missing: {path}")
    before = path.stat()
    cache_key = (int(before.st_size), int(before.st_mtime_ns))
    cached = _FILE_SHA_CACHE.get(path)
    if cached is not None and cached[:2] == cache_key:
        return cached[2]
    digest = sha256_file(path)
    after = path.stat()
    if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        raise VerificationError(f"file changed while hashing: {path}")
    _FILE_SHA_CACHE[path] = (cache_key[0], cache_key[1], digest)
    return digest


def _canonical_row_sha256(value: Mapping[str, Any]) -> str:
    return canonical_sha256(value)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
    )


def _read_json(path: Path, *, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise VerificationError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"invalid {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise VerificationError(f"{label} is not a JSON object: {path}")
    return value


def _identity(row: Mapping[str, Any]) -> Dict[str, Any]:
    source = row.get("identity")
    source = source if isinstance(source, Mapping) else row
    result = {}
    sample_id = source.get("sample_id", row.get("sample_id"))
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise VerificationError("row has no sample_id")
    result["sample_id"] = sample_id.strip()
    for key in ("image_id", "ann_id", "ref_id", "sent_id"):
        try:
            result[key] = int(source.get(key, row.get(key)))
        except (TypeError, ValueError) as error:
            raise VerificationError(f"{sample_id} has invalid {key}") from error
    result["dataset"] = str(source.get("dataset", row.get("dataset", "")))
    result["split"] = str(source.get("split", row.get("split", "")))
    return result


def _float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise VerificationError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(result):
        raise VerificationError(f"{label} is not finite")
    return result


def _bbox_xywh(value: Any, *, label: str) -> list[float]:
    if not (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 4
    ):
        raise VerificationError(f"{label} is not xywh[4]")
    box = [_float(item, label=label) for item in value]
    if box[2] <= 0.0 or box[3] <= 0.0:
        raise VerificationError(f"{label} has non-positive size")
    return box


def _xywh_to_xyxy(value: Sequence[float]) -> list[float]:
    x, y, width, height = [float(item) for item in value]
    return [x, y, x + width, y + height]


def _iou_xyxy(left: Sequence[float], right: Sequence[float]) -> float:
    lx0, ly0, lx1, ly1 = [float(item) for item in left]
    rx0, ry0, rx1, ry1 = [float(item) for item in right]
    ix0, iy0 = max(lx0, rx0), max(ly0, ry0)
    ix1, iy1 = min(lx1, rx1), min(ly1, ry1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    left_area = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
    right_area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _judgment_answer_confidence(value: Any) -> tuple[str, float, Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        return "", -1.0, {}
    nested = value.get("judgment")
    judgment = nested if isinstance(nested, Mapping) else value
    answer = str(judgment.get("answer", "")).strip().upper()
    try:
        confidence = float(judgment.get("confidence", -1.0))
    except (TypeError, ValueError):
        confidence = -1.0
    return answer, confidence, judgment


def _source_verification(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("source_verification")
    if not isinstance(value, Mapping):
        raise VerificationError(
            f"{_identity(row)['sample_id']} has no embedded source_verification"
        )
    return value


def _source_regions(row: Mapping[str, Any]) -> list[Dict[str, Any]]:
    """Normalize embedded target/cache judgments without trusting overlap metadata."""
    source = _source_verification(row)
    result: list[Dict[str, Any]] = []
    target = source.get("target")
    if isinstance(target, Mapping):
        bbox = (
            target.get("bbox_xywh_original")
            or target.get("target_bbox_used")
            or target.get("bbox")
        )
        judgment = target.get("judgment", target.get("visual_local_judgment"))
    else:
        bbox = source.get("target_bbox_used")
        judgment = source.get("visual_local_judgment")
    if bbox is None:
        bbox = row.get("target_bbox_used")
    target_box = _bbox_xywh(bbox, label="source target bbox")
    answer, confidence, normalized = _judgment_answer_confidence(judgment)
    result.append(
        {
            "kind": "target",
            "proposal_id": None,
            "bbox_xywh_original": target_box,
            "bbox_xyxy_original": _xywh_to_xyxy(target_box),
            "answer": answer,
            "confidence": confidence,
            "judgment": dict(normalized),
            "judgment_sha256": canonical_sha256(normalized),
        }
    )

    proposals = source.get("proposals")
    if isinstance(proposals, list):
        for index, proposal in enumerate(proposals):
            if not isinstance(proposal, Mapping):
                raise VerificationError("source_verification contains non-object proposal")
            proposal_id = int(proposal.get("proposal_id", index))
            bbox = proposal.get("bbox_xywh_original", proposal.get("bbox"))
            proposal_box = _bbox_xywh(bbox, label=f"source proposal {proposal_id} bbox")
            answer, confidence, normalized = _judgment_answer_confidence(
                proposal.get("judgment")
            )
            result.append(
                {
                    "kind": "proposal",
                    "proposal_id": proposal_id,
                    "bbox_xywh_original": proposal_box,
                    "bbox_xyxy_original": _xywh_to_xyxy(proposal_box),
                    "answer": answer,
                    "confidence": confidence,
                    "judgment": dict(normalized),
                    "judgment_sha256": canonical_sha256(normalized),
                }
            )
        return result

    proposal_cache = source.get("proposal_cache")
    proposal_judgments = source.get("visual_proposal_judgments")
    if not isinstance(proposal_cache, list) or not isinstance(
        proposal_judgments, list
    ):
        raise VerificationError("source_verification lacks proposal cache/judgments")
    judgments_by_id: Dict[int, Mapping[str, Any]] = {}
    for judgment_row in proposal_judgments:
        if not isinstance(judgment_row, Mapping):
            raise VerificationError("source proposal judgment is not an object")
        proposal_id = int(judgment_row.get("proposal_id", -1))
        if proposal_id in judgments_by_id:
            raise VerificationError(f"duplicate source proposal judgment {proposal_id}")
        judgments_by_id[proposal_id] = judgment_row
    for index, proposal in enumerate(proposal_cache):
        if not isinstance(proposal, Mapping):
            raise VerificationError("source proposal is not an object")
        proposal_id = int(proposal.get("proposal_id", index))
        if proposal_id not in judgments_by_id:
            raise VerificationError(f"missing source proposal judgment {proposal_id}")
        proposal_box = _bbox_xywh(
            proposal.get("bbox"), label=f"source proposal {proposal_id} bbox"
        )
        answer, confidence, normalized = _judgment_answer_confidence(
            judgments_by_id[proposal_id]
        )
        result.append(
            {
                "kind": "proposal",
                "proposal_id": proposal_id,
                "bbox_xywh_original": proposal_box,
                "bbox_xyxy_original": _xywh_to_xyxy(proposal_box),
                "answer": answer,
                "confidence": confidence,
                "judgment": dict(normalized),
                "judgment_sha256": canonical_sha256(normalized),
            }
        )
    return result


def _best_source_overlap(
    row: Mapping[str, Any], region: Mapping[str, Any]
) -> Dict[str, Any]:
    region_box = region_bbox_xyxy_original(region)
    candidates = []
    for source in _source_regions(row):
        candidate = dict(source)
        candidate["iou"] = _iou_xyxy(region_box, source["bbox_xyxy_original"])
        candidates.append(candidate)
    if not candidates:
        raise VerificationError("source verification has no target/proposal regions")
    return max(
        candidates,
        key=lambda item: (
            item["iou"],
            item["kind"] == "target",
            -(item["proposal_id"] if item["proposal_id"] is not None else -1),
        ),
    )


def _source_inherit_decision(
    row: Mapping[str, Any], region: Mapping[str, Any]
) -> tuple[bool, Dict[str, Any]]:
    match = _best_source_overlap(row, region)
    eligible = (
        match["answer"] == "NO"
        and math.isfinite(match["confidence"])
        and match["confidence"] >= INHERIT_CONFIDENCE_THRESHOLD
        and match["iou"] >= INHERIT_IOU_THRESHOLD
    )
    extraction_overlap = region.get("max_overlap")
    if isinstance(extraction_overlap, Mapping):
        observed_iou = _float(
            extraction_overlap.get("iou", -1.0), label="extraction max_overlap.iou"
        )
        if not math.isclose(observed_iou, match["iou"], rel_tol=0.0, abs_tol=1e-6):
            raise VerificationError("extraction/source max-overlap IoU drifted")
        if str(extraction_overlap.get("kind", "")).lower() != match["kind"]:
            raise VerificationError("extraction/source max-overlap kind drifted")
        if match["kind"] == "proposal" and int(
            extraction_overlap.get("proposal_id", -1)
        ) != int(match["proposal_id"]):
            raise VerificationError("extraction/source max-overlap proposal_id drifted")
        if str(extraction_overlap.get("source_answer", "")).upper() != match[
            "answer"
        ]:
            raise VerificationError("extraction/source max-overlap answer drifted")
        observed_confidence = extraction_overlap.get("source_confidence")
        if observed_confidence is None or not math.isclose(
            _float(observed_confidence, label="extraction source confidence"),
            match["confidence"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise VerificationError("extraction/source max-overlap confidence drifted")
        if extraction_overlap.get("source_judgment_sha256") != match[
            "judgment_sha256"
        ]:
            raise VerificationError("extraction/source judgment hash drifted")
    return eligible, match


def _query_id(value: Any) -> int | None:
    if isinstance(value, Mapping):
        value = value.get("query_id")
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _validate_region_union(row: Mapping[str, Any]) -> None:
    identity = _identity(row)
    regions = row.get("regions")
    assert isinstance(regions, list)
    query_to_regions: Dict[int, set[str]] = {}
    origin_to_queries: Dict[str, set[int]] = {}
    origin_names = set()
    for region in regions:
        region_id = str(region["region_id"])
        origins = region.get("origins")
        if not isinstance(origins, list) or not origins:
            raise VerificationError(f"{identity['sample_id']} region lacks origins")
        origin_names.update(str(value).lower() for value in origins)
        query_ids = region.get("query_ids")
        if not isinstance(query_ids, list) or not query_ids:
            raise VerificationError(f"{identity['sample_id']} region lacks query_ids")
        for query_id in query_ids:
            parsed = int(query_id)
            if not 0 <= parsed < 900:
                raise VerificationError(f"invalid union query id {parsed}")
            query_to_regions.setdefault(parsed, set()).add(region_id)
            for origin in origins:
                origin_to_queries.setdefault(str(origin).lower(), set()).add(parsed)

    queries = row.get("queries")
    queries = queries if isinstance(queries, Mapping) else row
    observation_queries = {
        origin: _query_id(queries.get(origin))
        for origin in ("primary", "shadow", "deploy")
    }
    for origin, query_id in observation_queries.items():
        if query_id is None or query_id not in query_to_regions:
            raise VerificationError(
                f"{identity['sample_id']} {origin} query is absent from union"
            )
        if origin not in origin_names or query_id not in origin_to_queries.get(
            origin, set()
        ):
            raise VerificationError(
                f"{identity['sample_id']} {origin} query is not bound to its origin"
            )

    stability = row.get("stability")
    stability = stability if isinstance(stability, Mapping) else {}
    if not math.isclose(
        _float(stability.get("epsilon"), label="stability epsilon"),
        EXPECTED_TIE_EPSILON,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise VerificationError("stability epsilon drifted from 1e-3")
    primary = observation_queries["primary"]
    for origin, agreement_key in (
        ("shadow", "primary_shadow_agree"),
        ("deploy", "primary_deploy_agree"),
    ):
        agrees = stability.get(agreement_key)
        if not isinstance(agrees, bool):
            raise VerificationError(f"stability lacks boolean {agreement_key}")
        if agrees != (observation_queries[origin] == primary):
            raise VerificationError(f"stability {agreement_key} drifted")
    query_ids_by_origin = stability.get("query_ids_by_origin")
    if not isinstance(query_ids_by_origin, Mapping):
        raise VerificationError("stability lacks query_ids_by_origin")
    for origin, query_id in observation_queries.items():
        if _query_id(query_ids_by_origin.get(origin)) != query_id:
            raise VerificationError(f"stability {origin} query_id drifted")
    near_ties = stability.get("near_tie_query_ids", [])
    if near_ties is None:
        near_ties = []
    if not isinstance(near_ties, list):
        raise VerificationError("near_tie_query_ids must be a list")
    for query_id in near_ties:
        parsed = int(query_id)
        if parsed not in query_to_regions:
            raise VerificationError(
                f"{identity['sample_id']} near-tie query {parsed} is absent from union"
            )
        if parsed not in origin_to_queries.get("near_tie", set()):
            raise VerificationError(
                f"{identity['sample_id']} query {parsed} lacks near_tie origin"
            )


def _claims(row: Mapping[str, Any]) -> None:
    claims = row.get("claims")
    if not isinstance(claims, Mapping):
        raise VerificationError("extraction row lacks claims")
    required_false = (
        "all_900_gdino_queries_verified",
        "image_global_semantic_absence_proven",
        "portable_to_other_checkpoint_or_transform",
    )
    for key in required_false:
        if claims.get(key) is not False:
            raise VerificationError(f"extraction claim {key} must be exact false")
    if claims.get("frozen_gdino_global_max_regions_extracted") is not True:
        raise VerificationError("extraction lost its frozen-max extraction claim")
    if claims.get("train_path_and_deploy_transform_regions_extracted") is not True:
        raise VerificationError("extraction lost its train/deploy extraction claim")


def _check_hashed_path(record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or not record.get("path") or not record.get(
        "sha256"
    ):
        raise VerificationError(f"missing path/hash for {label}")
    path = Path(str(record["path"])).expanduser().resolve()
    observed = _cached_file_sha256(path)
    if observed != record["sha256"]:
        raise VerificationError(f"hash drift for {label}: {path}")
    return path


class _SourceLines:
    def __init__(self) -> None:
        self._lines: Dict[Path, list[Dict[str, Any]]] = {}
        self._hashes: Dict[Path, str] = {}

    def row(self, record: Mapping[str, Any], *, label: str) -> Dict[str, Any]:
        path = _check_hashed_path(record, label=label)
        if path not in self._lines:
            self._lines[path] = [row for _line, row in iter_jsonl(path)]
            self._hashes[path] = sha256_file(path)
        try:
            line = int(record.get("line", record.get("source_line")))
        except (TypeError, ValueError) as error:
            raise VerificationError(f"{label} has no source line") from error
        if line <= 0 or line > len(self._lines[path]):
            raise VerificationError(f"{label} line {line} is outside {path}")
        row = self._lines[path][line - 1]
        expected_row_hash = record.get("row_sha256")
        if expected_row_hash and _canonical_row_sha256(row) != expected_row_hash:
            raise VerificationError(f"{label} row hash drifted at {path}:{line}")
        return dict(row)


def _source_pair(
    extraction: Mapping[str, Any], source_lines: _SourceLines
) -> Dict[str, Any]:
    record = extraction.get("source_pair")
    if not isinstance(record, Mapping):
        raise VerificationError("extraction lacks source_pair provenance")
    embedded = record.get("row")
    if isinstance(embedded, Mapping):
        pair = dict(embedded)
        expected = record.get("row_sha256")
        if expected and _canonical_row_sha256(pair) != expected:
            raise VerificationError("embedded source pair hash drifted")
    else:
        pair = source_lines.row(record, label="source pair")
    identity = _identity(extraction)
    if pair.get("sample_id") != identity["sample_id"]:
        raise VerificationError("source pair sample_id drifted")
    for key in ("image_id", "ann_id", "ref_id", "sent_id"):
        if int(pair.get(key, -1)) != identity[key]:
            raise VerificationError(f"source pair {key} drifted")
    return pair


def _source_region_signature(regions: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    signature = [
        {
            "kind": str(region["kind"]),
            "proposal_id": region.get("proposal_id"),
            "bbox_xywh_original": [
                float(value) for value in region["bbox_xywh_original"]
            ],
            "answer": str(region["answer"]),
            "confidence": float(region["confidence"]),
            "judgment_sha256": str(region["judgment_sha256"]),
        }
        for region in regions
    ]
    return sorted(
        signature,
        key=lambda item: (
            item["kind"] != "target",
            int(item["proposal_id"] if item["proposal_id"] is not None else -1),
        ),
    )


def _validate_source_lineage(
    extraction: Mapping[str, Any],
    source_pair: Mapping[str, Any],
    source_lines: _SourceLines,
) -> str:
    record = extraction.get("source_verified_row")
    if not isinstance(record, Mapping):
        raise VerificationError("extraction lacks source_verified_row provenance")
    raw_source = source_lines.row(record, label="raw verified source")
    identity = _identity(extraction)
    for key in ("image_id", "ann_id", "ref_id", "sent_id"):
        if int(raw_source.get(key, -1)) != identity[key]:
            raise VerificationError(f"raw verified source {key} drifted")
    required = {
        "split": "train",
        "visual_filter_status": "accept",
        "visual_filter_reason": "verified_negative",
        "tn_scope": "image_global_proposal_verified",
        "global_tn_verified": True,
    }
    for key, expected in required.items():
        if raw_source.get(key) != expected:
            raise VerificationError(
                f"raw verified source {key}: expected {expected!r}, "
                f"got {raw_source.get(key)!r}"
            )
    raw_hash = _canonical_row_sha256(raw_source)
    if record.get("row_sha256") != raw_hash:
        raise VerificationError("raw verified source row hash drifted")
    if source_pair.get("source_row_sha256") != raw_hash:
        raise VerificationError("semantic pair lost its raw source row hash")
    raw_path = Path(str(record["path"])).expanduser().resolve()
    if Path(str(source_pair.get("source_file", ""))).expanduser().resolve() != raw_path:
        raise VerificationError("semantic pair raw source path drifted")
    if int(source_pair.get("source_line", -1)) != int(record.get("line", -2)):
        raise VerificationError("semantic pair raw source line drifted")
    if source_pair.get("tn_scope") != "image_global_topk_verified":
        raise VerificationError("semantic source pair has the wrong TN scope")
    if source_pair.get("global_tn_verified") is not True:
        raise VerificationError("semantic source pair lost global_tn_verified=true")
    if source_pair.get("proposalset_proxy_verified") is not False:
        raise VerificationError("semantic source pair is a proposal proxy")
    if source_pair.get("cached_proposal_coverage_only") is not True:
        raise VerificationError("semantic source pair lost cached-only disclosure")
    if source_pair.get("all_900_gdino_queries_verified") is not False:
        raise VerificationError("semantic source pair lost non-all-900 disclosure")
    if source_pair.get("global_max_label_is_semantic_extrapolation") is not True:
        raise VerificationError("semantic source pair lost its original extrapolation disclosure")

    raw_wrapper = {
        "identity": identity,
        "target_bbox_used": raw_source.get("target_bbox_used"),
        "source_verification": raw_source,
    }
    raw_regions = _source_regions(raw_wrapper)
    embedded_regions = _source_regions(extraction)
    if _source_region_signature(raw_regions) != _source_region_signature(
        embedded_regions
    ):
        raise VerificationError(
            "embedded source_verification drifted from its frozen raw source row"
        )
    if len(raw_regions) < 2:
        raise VerificationError("raw source has no non-empty cached proposal set")
    for region in raw_regions:
        if region["answer"] != "NO":
            raise VerificationError("raw source target/proposal is not verified NO")
    return raw_hash


def _validate_provenance(row: Mapping[str, Any]) -> Dict[str, str]:
    checkpoint = row.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        provenance = row.get("provenance")
        checkpoint = provenance.get("checkpoint") if isinstance(provenance, Mapping) else None
    checkpoint_path = _check_hashed_path(checkpoint, label="frozen checkpoint")
    for key in ("model_sha256", "base_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(checkpoint.get(key, ""))):
            raise VerificationError(f"checkpoint provenance lacks {key}")
    completion = checkpoint.get("protocol_train_complete")
    if not isinstance(completion, Mapping):
        raise VerificationError("checkpoint lacks baseline completion audit")
    _check_hashed_path(completion, label="baseline completion audit")
    config = row.get("config")
    for label, value in (
        ("model config", config),
        ("data config", row.get("data_config")),
    ):
        if not isinstance(value, Mapping):
            raise VerificationError(f"extraction lacks {label} provenance")
        _check_hashed_path(value, label=label)
        chain = value.get("import_chain")
        if not isinstance(chain, list) or not chain:
            raise VerificationError(f"{label} import chain is empty")
        for index, record in enumerate(chain):
            _check_hashed_path(record, label=f"{label} import {index}")
        if canonical_sha256(chain) != value.get("import_chain_sha256"):
            raise VerificationError(f"{label} import-chain hash drifted")
    if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("code_sha256", ""))):
        raise VerificationError("extraction code closure hash is malformed")
    image = row.get("image")
    image_path = _check_hashed_path(image, label="source image")
    transform_hashes: Dict[str, str] = {}
    transform_contract_hashes: Dict[str, str] = {}
    for name, transform in (
        ("train", row.get("transform")),
        ("deploy", row.get("deploy_transform")),
    ):
        if not isinstance(transform, Mapping) or not re.fullmatch(
            r"[0-9a-f]{64}", str(transform.get("sha256", ""))
        ):
            raise VerificationError(
                f"{name} transform provenance is missing its canonical hash"
            )
        transform_payload = dict(transform)
        observed_transform_sha = str(transform_payload.pop("sha256"))
        if canonical_sha256(transform_payload) != observed_transform_sha:
            raise VerificationError(f"{name} transform canonical hash drifted")
        transform_hashes[name] = observed_transform_sha
        static_contract_sha = str(transform.get("static_contract_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", static_contract_sha):
            raise VerificationError(f"{name} static transform contract hash is malformed")
        transform_contract_hashes[name] = static_contract_sha
    if row.get("forward_contract") != EXPECTED_FORWARD_CONTRACT:
        raise VerificationError("train/deploy forward contract drifted")
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": str(checkpoint["sha256"]),
        "image_path": str(image_path),
        "image_sha256": str(image["sha256"]),
        "train_transform_row_sha256": transform_hashes["train"],
        "deploy_transform_row_sha256": transform_hashes["deploy"],
        "train_transform_contract_sha256": transform_contract_hashes["train"],
        "deploy_transform_contract_sha256": transform_contract_hashes["deploy"],
    }


def _load_judgments(path: Path) -> Dict[tuple[str, str], Dict[str, Any]]:
    result: Dict[tuple[str, str], Dict[str, Any]] = {}
    for line_number, row in iter_jsonl(path):
        try:
            validate_judgment_record(row, require_assets=True)
        except QwenJudgeError as error:
            raise VerificationError(f"invalid judgment at {path}:{line_number}: {error}") from error
        key = (str(row.get("sample_id")), str(row.get("region_id")))
        if key in result:
            raise VerificationError(f"duplicate judgment at {path}:{line_number}")
        result[key] = row
    return result


def _validate_qwen_against_extraction(
    row: Mapping[str, Any],
    region: Mapping[str, Any],
    judgment: Mapping[str, Any],
) -> None:
    expected_key = judgment_cache_key(row, region)
    try:
        validate_judgment_record(
            judgment,
            expected_sample_id=_identity(row)["sample_id"],
            expected_region_id=str(region["region_id"]),
            expected_cache_key=expected_key,
            require_assets=True,
        )
    except QwenJudgeError as error:
        raise VerificationError(str(error)) from error
    if judgment.get("bbox_xyxy_original") != region_bbox_xyxy_original(region):
        raise VerificationError("Qwen judgment bbox drifted from extraction")
    negative = row.get("negative_expression", row.get("negative_caption_model"))
    if judgment.get("negative_expression") != " ".join(str(negative).split()):
        raise VerificationError("Qwen judgment negative expression drifted")
    prompt_hash = hashlib.sha256(
        render_prompt(" ".join(str(negative).split())).encode("utf-8")
    ).hexdigest()
    if judgment.get("prompt", {}).get("rendered_sha256") != prompt_hash:
        raise VerificationError("Qwen rendered prompt hash drifted")
    extraction_image = row.get("image")
    assets = judgment.get("assets")
    source_image = assets.get("source_image") if isinstance(assets, Mapping) else None
    if not isinstance(source_image, Mapping) or source_image.get(
        "sha256"
    ) != extraction_image.get("sha256"):
        raise VerificationError("Qwen asset source image drifted")
    extraction_assets = region.get("assets")
    if not isinstance(extraction_assets, Mapping):
        raise VerificationError("extraction region lacks locked assets")
    if extraction_assets.get("asset_policy_sha256") != ASSET_POLICY_SHA256:
        raise VerificationError("extraction asset policy drifted")
    for name in ("tight", "full_boxed", "context_2x_boxed"):
        expected = extraction_assets.get(name)
        observed = assets.get(name) if isinstance(assets, Mapping) else None
        if not isinstance(expected, Mapping) or not isinstance(observed, Mapping):
            raise VerificationError(f"Qwen/extraction {name} asset is missing")
        if expected.get("path") != observed.get("path") or expected.get(
            "sha256"
        ) != observed.get("sha256"):
            raise VerificationError(f"Qwen {name} asset drifted from extraction")


def _region_decision(
    row: Mapping[str, Any],
    region: Mapping[str, Any],
    judgment: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    eligible, source_match = _source_inherit_decision(row, region)
    result: Dict[str, Any] = {
        "region_id": str(region["region_id"]),
        "origins": list(region.get("origins", [])),
        "query_ids": [int(value) for value in region.get("query_ids", [])],
        "bbox_xyxy_original": region_bbox_xyxy_original(region),
        "source_match": {
            key: source_match[key]
            for key in (
                "kind",
                "proposal_id",
                "iou",
                "answer",
                "confidence",
                "judgment_sha256",
            )
        },
        "source_inherit_eligible": eligible,
    }
    if judgment is not None:
        _validate_qwen_against_extraction(row, region, judgment)
        result["qwen_judgment_sha256"] = _canonical_row_sha256(judgment)
        result["qwen_status"] = judgment.get("status")
        result["qwen_answer"] = judgment.get("answer")
        result["qwen_confidence"] = judgment.get("confidence")
        if eligible:
            if judgment.get("status") != "complete" or judgment.get(
                "answer"
            ) == "UNKNOWN":
                result["source_qwen_comparison"] = "inconclusive"
            elif judgment.get("answer") == "NO":
                if (
                    float(judgment.get("confidence", -1.0))
                    >= QWEN_CONFIDENCE_THRESHOLD
                ):
                    result["source_qwen_comparison"] = "agree_no"
                else:
                    result["source_qwen_comparison"] = "agree_no_low_confidence"
            elif judgment.get("answer") == "YES":
                result["source_qwen_comparison"] = "conflict_yes"
            else:
                result["source_qwen_comparison"] = "inconclusive"
        if judgment.get("status") == "complete" and judgment.get("answer") == "YES":
            result.update(decision="rejected", reason="qwen_yes", method="qwen")
            return result
        if judgment.get("status") != "complete":
            result.update(decision="quarantine", reason="qwen_error", method="qwen")
            return result
        if judgment.get("answer") == "UNKNOWN":
            result.update(decision="quarantine", reason="qwen_unknown", method="qwen")
            return result
        if judgment.get("answer") != "NO":
            result.update(decision="quarantine", reason="qwen_malformed", method="qwen")
            return result
        if float(judgment.get("confidence", -1.0)) < QWEN_CONFIDENCE_THRESHOLD:
            result.update(decision="quarantine", reason="qwen_low_confidence", method="qwen")
            return result
        result.update(decision="accepted", reason="qwen_no", method="qwen")
        return result
    if eligible:
        result.update(decision="accepted", reason="source_inherited_no", method="source")
        return result
    result.update(decision="quarantine", reason="missing_qwen_judgment", method="missing")
    return result


def _manifest_image_ids(
    path: Path, *, expected_sha256: str, label: str
) -> tuple[set[int], Dict[str, Any]]:
    observed = sha256_file(path)
    if expected_sha256 and observed != expected_sha256:
        raise VerificationError(
            f"{label} manifest hash drifted: {observed} != {expected_sha256}"
        )
    images = set()
    rows = 0
    for line_number, row in iter_jsonl(path):
        try:
            images.add(int(row["image_id"]))
        except (KeyError, TypeError, ValueError) as error:
            raise VerificationError(
                f"invalid image_id at {path}:{line_number}"
            ) from error
        rows += 1
    record = file_record(path)
    record.update(rows=rows, unique_images=len(images))
    return images, record


def _static_contract_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, Mapping):
        raise VerificationError(f"extraction audit lacks {label}")
    payload = dict(value)
    observed = str(payload.pop("sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", observed):
        raise VerificationError(f"{label} SHA-256 is malformed")
    if canonical_sha256(payload) != observed:
        raise VerificationError(f"{label} canonical hash drifted")
    return observed


def _validate_extraction_audit(
    path: Path,
    *,
    extraction_path: Path,
    extraction_record: Mapping[str, Any],
    strict2031_record: Mapping[str, Any],
    strict1607_record: Mapping[str, Any],
) -> Dict[str, Any]:
    audit = _read_json(path, label="extraction audit")
    if audit.get("schema") != EXTRACTION_AUDIT_SCHEMA or audit.get(
        "kind"
    ) != EXTRACTION_AUDIT_KIND:
        raise VerificationError("extraction audit schema/kind drifted")
    if int(audit.get("rows", -1)) != EXPECTED_EXTRACTION_ROWS:
        raise VerificationError(
            f"extraction audit rows must be {EXPECTED_EXTRACTION_ROWS}"
        )
    counts = audit.get("counts")
    if not isinstance(counts, Mapping) or int(counts.get("rows", -1)) != int(
        audit["rows"]
    ):
        raise VerificationError("extraction audit count total drifted")
    manifest = audit.get("manifest")
    if not isinstance(manifest, Mapping):
        raise VerificationError("extraction audit lacks manifest file record")
    expected_manifest = {
        "path": str(extraction_path),
        "sha256": extraction_record["sha256"],
        "size_bytes": extraction_record["size_bytes"],
        "rows": EXPECTED_EXTRACTION_ROWS,
    }
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise VerificationError("extraction audit manifest record drifted")
    legacy_output = audit.get("output")
    if not isinstance(legacy_output, Mapping) or any(
        legacy_output.get(key) != expected_manifest[key]
        for key in ("path", "sha256", "size_bytes")
    ):
        raise VerificationError("extraction audit legacy output record drifted")

    checkpoint = audit.get("checkpoint")
    checkpoint_path = _check_hashed_path(
        checkpoint, label="extraction-audit checkpoint"
    )
    completion = checkpoint.get("protocol_train_complete")
    if not isinstance(completion, Mapping):
        raise VerificationError("extraction audit checkpoint lacks completion audit")
    _check_hashed_path(completion, label="extraction-audit completion audit")

    holdout = audit.get("holdout")
    manifests = holdout.get("manifests") if isinstance(holdout, Mapping) else None
    if not isinstance(manifests, Mapping):
        raise VerificationError("extraction audit lacks strict holdout manifests")
    for name, observed, expected in (
        ("strict2031", manifests.get("strict2031"), strict2031_record),
        ("strict1607", manifests.get("strict1607"), strict1607_record),
    ):
        if not isinstance(observed, Mapping):
            raise VerificationError(f"extraction audit lacks {name} holdout record")
        for key in ("sha256", "rows"):
            if observed.get(key) != expected.get(key):
                raise VerificationError(
                    f"extraction audit {name} {key} drifted"
                )

    contracts = audit.get("transform_contracts")
    if not isinstance(contracts, Mapping):
        raise VerificationError("extraction audit lacks transform contracts")
    train_contract_sha = _static_contract_sha256(
        contracts.get("train"), label="train transform contract"
    )
    deploy_contract_sha = _static_contract_sha256(
        contracts.get("deploy"), label="deploy transform contract"
    )
    model_config = audit.get("model_config")
    data_config = audit.get("data_config")
    code = audit.get("code")
    for label, value in (
        ("model config", model_config),
        ("data config", data_config),
    ):
        _check_hashed_path(value, label=f"extraction-audit {label}")
    code_sha = code.get("code_sha256") if isinstance(code, Mapping) else None
    if not re.fullmatch(r"[0-9a-f]{64}", str(code_sha or "")):
        raise VerificationError("extraction audit code closure hash is malformed")
    claims = audit.get("claims")
    if not isinstance(claims, Mapping) or claims.get(
        "train_path_and_deploy_transform_regions_extracted"
    ) is not True:
        raise VerificationError("extraction audit train/deploy claim drifted")
    for key in (
        "all_900_gdino_queries_verified",
        "image_global_semantic_absence_proven",
        "portable_to_other_checkpoint_or_transform",
    ):
        if claims.get(key) is not False:
            raise VerificationError(f"extraction audit claim {key} drifted")
    runtime = audit.get("runtime")
    if not isinstance(runtime, Mapping) or not math.isclose(
        _float(runtime.get("tie_epsilon"), label="audit tie epsilon"),
        EXPECTED_TIE_EPSILON,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise VerificationError("extraction audit tie epsilon drifted")
    return {
        "file": file_record(path),
        "checkpoint": dict(checkpoint),
        "model_config": dict(model_config),
        "data_config": dict(data_config),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": str(checkpoint["sha256"]),
        "model_config_sha256": str(model_config["sha256"]),
        "data_config_sha256": str(data_config["sha256"]),
        "code_sha256": str(code_sha),
        "train_transform_contract_sha256": train_contract_sha,
        "deploy_transform_contract_sha256": deploy_contract_sha,
    }


def _accepted_pair(
    source_pair: Mapping[str, Any],
    extraction: Mapping[str, Any],
    region_decisions: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, str],
) -> Dict[str, Any]:
    output = dict(source_pair)
    output.update(
        {
            "adapter_pair_schema": ACCEPTED_PAIR_SCHEMA,
            "source": "stage_b_gdino_adapter_fixed_top1_verified",
            "fixed_gdino_max_scope": FIXED_MAX_SCOPE,
            "fixed_gdino_global_max_verified": True,
            "fixed_gdino_global_max_verification_contract": (
                "target_plus_cached_sam3_all_no_and_all_frozen_gdino_"
                "train_primary_shadow_deploy_neartie_union_regions_no"
            ),
            "all_900_gdino_queries_verified": False,
            "image_global_semantic_absence_proven": False,
            "global_max_label_is_semantic_extrapolation": False,
            "portable_to_other_checkpoint_or_transform": False,
            "frozen_gdino_checkpoint_sha256": provenance["checkpoint_sha256"],
            # Compatibility field: this is the shared train transform contract,
            # not the image-specific affine trace seal.
            "frozen_gdino_transform_sha256": provenance[
                "train_transform_contract_sha256"
            ],
            "frozen_gdino_train_transform_contract_sha256": provenance[
                "train_transform_contract_sha256"
            ],
            "frozen_gdino_deploy_transform_contract_sha256": provenance[
                "deploy_transform_contract_sha256"
            ],
            "frozen_gdino_train_transform_row_sha256": provenance[
                "train_transform_row_sha256"
            ],
            "frozen_gdino_deploy_transform_row_sha256": provenance[
                "deploy_transform_row_sha256"
            ],
            "fixed_gdino_region_verifications": [
                {
                    key: decision.get(key)
                    for key in (
                        "region_id",
                        "origins",
                        "query_ids",
                        "bbox_xyxy_original",
                        "method",
                        "reason",
                        "source_match",
                        "source_qwen_comparison",
                        "qwen_judgment_sha256",
                    )
                    if key in decision
                }
                for decision in region_decisions
            ],
            "fixed_gdino_extraction_row_sha256": _canonical_row_sha256(extraction),
        }
    )
    # Keep the existing loader-compatible scope while making its narrower,
    # checkpoint-specific evidence explicit in independent fields above.
    output["tn_scope"] = "image_global_topk_verified"
    output["global_tn_verified"] = True
    output["proposalset_proxy_verified"] = False
    return output


def verify(args: argparse.Namespace) -> Dict[str, Any]:
    extraction_path = Path(args.extractions).expanduser().resolve()
    extraction_audit_path = Path(args.extraction_audit).expanduser().resolve()
    judgment_path = Path(args.judgments).expanduser().resolve()
    accepted_path = Path(args.accepted_output).expanduser().resolve()
    rejected_path = Path(args.rejected_output).expanduser().resolve()
    quarantine_path = Path(args.quarantine_output).expanduser().resolve()
    audit_path = Path(args.audit).expanduser().resolve()
    strict2031_path = Path(args.strict2031).expanduser().resolve()
    strict1607_path = Path(args.strict1607).expanduser().resolve()

    strict2031_images, strict2031_record = _manifest_image_ids(
        strict2031_path,
        expected_sha256=str(args.expected_strict2031_sha256),
        label="strict2031",
    )
    strict1607_images, strict1607_record = _manifest_image_ids(
        strict1607_path,
        expected_sha256=str(args.expected_strict1607_sha256),
        label="strict1607",
    )
    if not strict1607_images.issubset(strict2031_images):
        raise VerificationError("strict1607 image IDs are no longer a strict2031 subset")

    extraction_file_record = file_record(extraction_path)
    extraction_audit = _validate_extraction_audit(
        extraction_audit_path,
        extraction_path=extraction_path,
        extraction_record=extraction_file_record,
        strict2031_record=strict2031_record,
        strict1607_record=strict1607_record,
    )
    extractions = [row for _line, row in iter_jsonl(extraction_path)]
    if len(extractions) != EXPECTED_EXTRACTION_ROWS:
        raise VerificationError(
            f"extraction manifest rows={len(extractions)}, "
            f"expected {EXPECTED_EXTRACTION_ROWS}"
        )
    seen_samples = set()
    overlap2031 = set()
    overlap1607 = set()
    for row in extractions:
        try:
            validate_extraction_row(row)
        except QwenJudgeError as error:
            raise VerificationError(str(error)) from error
        identity = _identity(row)
        if identity["sample_id"] in seen_samples:
            raise VerificationError(f"duplicate sample_id {identity['sample_id']}")
        seen_samples.add(identity["sample_id"])
        overlap2031.update({identity["image_id"]} & strict2031_images)
        overlap1607.update({identity["image_id"]} & strict1607_images)
        holdout = row.get("holdout")
        if (
            not isinstance(holdout, Mapping)
            or holdout.get("image_disjoint") is not True
            or holdout.get("strict2031_manifest_sha256")
            != strict2031_record["sha256"]
            or holdout.get("strict1607_manifest_sha256")
            != strict1607_record["sha256"]
        ):
            raise VerificationError("extraction holdout provenance drifted")
        _claims(row)
        _validate_region_union(row)
    if overlap2031 or overlap1607:
        raise VerificationError(
            "training extraction is not strict-image-disjoint: "
            f"strict2031={len(overlap2031)}, strict1607={len(overlap1607)}"
        )

    judgments = _load_judgments(judgment_path)
    source_lines = _SourceLines()
    accepted: list[Dict[str, Any]] = []
    rejected: list[Dict[str, Any]] = []
    quarantine: list[Dict[str, Any]] = []
    region_reason_counts: Counter[str] = Counter()
    region_method_counts: Counter[str] = Counter()
    source_qwen_comparison_counts: Counter[str] = Counter()
    row_reason_counts: Counter[str] = Counter()
    checkpoint_hashes = set()
    train_transform_contract_hashes = set()
    deploy_transform_contract_hashes = set()
    transform_row_records: list[list[str]] = []
    used_judgments = set()
    total_regions = 0

    for row in extractions:
        identity = _identity(row)
        provenance = _validate_provenance(row)
        checkpoint_hashes.add(provenance["checkpoint_sha256"])
        train_transform_contract_hashes.add(
            provenance["train_transform_contract_sha256"]
        )
        deploy_transform_contract_hashes.add(
            provenance["deploy_transform_contract_sha256"]
        )
        transform_row_records.append(
            [
                identity["sample_id"],
                provenance["train_transform_row_sha256"],
                provenance["deploy_transform_row_sha256"],
            ]
        )
        if row.get("checkpoint") != extraction_audit["checkpoint"]:
            raise VerificationError("row checkpoint drifted from extraction audit")
        if (
            provenance["train_transform_contract_sha256"]
            != extraction_audit["train_transform_contract_sha256"]
            or provenance["deploy_transform_contract_sha256"]
            != extraction_audit["deploy_transform_contract_sha256"]
        ):
            raise VerificationError("row transform contract drifted from extraction audit")
        if row.get("config") != extraction_audit["model_config"] or row.get(
            "data_config"
        ) != extraction_audit["data_config"]:
            raise VerificationError("row config drifted from extraction audit")
        if row.get("code_sha256") != extraction_audit["code_sha256"]:
            raise VerificationError("row code closure drifted from extraction audit")
        source_pair = _source_pair(row, source_lines)
        _validate_source_lineage(row, source_pair, source_lines)
        decisions = []
        for region in row["regions"]:
            total_regions += 1
            key = (identity["sample_id"], str(region["region_id"]))
            judgment = judgments.get(key)
            if judgment is not None:
                used_judgments.add(key)
            decision = _region_decision(row, region, judgment)
            decisions.append(decision)
            region_reason_counts[decision["reason"]] += 1
            region_method_counts[decision["method"]] += 1
            comparison = decision.get("source_qwen_comparison")
            if comparison is not None:
                source_qwen_comparison_counts[str(comparison)] += 1

        if any(item["decision"] == "rejected" for item in decisions):
            row_decision = "rejected"
            reason = next(
                item["reason"] for item in decisions if item["decision"] == "rejected"
            )
        elif any(item["decision"] == "quarantine" for item in decisions):
            row_decision = "quarantine"
            reason = next(
                item["reason"]
                for item in decisions
                if item["decision"] == "quarantine"
            )
        else:
            row_decision = "accepted"
            reason = "all_union_regions_verified_no"
        row_reason_counts[reason] += 1
        decision_record = {
            "schema": DECISION_SCHEMA,
            "identity": identity,
            "decision": row_decision,
            "reason": reason,
            "region_decisions": decisions,
            "extraction_row_sha256": _canonical_row_sha256(row),
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "train_transform_contract_sha256": provenance[
                "train_transform_contract_sha256"
            ],
            "deploy_transform_contract_sha256": provenance[
                "deploy_transform_contract_sha256"
            ],
            "train_transform_row_sha256": provenance[
                "train_transform_row_sha256"
            ],
            "deploy_transform_row_sha256": provenance[
                "deploy_transform_row_sha256"
            ],
        }
        if row_decision == "accepted":
            accepted.append(_accepted_pair(source_pair, row, decisions, provenance))
        elif row_decision == "rejected":
            rejected.append(decision_record)
        else:
            quarantine.append(decision_record)

    if (
        len(checkpoint_hashes) != 1
        or len(train_transform_contract_hashes) != 1
        or len(deploy_transform_contract_hashes) != 1
    ):
        raise VerificationError(
            "extraction mixes checkpoint/train/deploy transform provenance: "
            f"checkpoints={len(checkpoint_hashes)}, "
            f"train_contracts={len(train_transform_contract_hashes)}, "
            f"deploy_contracts={len(deploy_transform_contract_hashes)}"
        )
    unused_judgments = set(judgments) - used_judgments
    if unused_judgments:
        raise VerificationError(
            f"judgment file contains {len(unused_judgments)} rows outside extraction"
        )

    transform_row_records.sort(key=lambda item: tuple(item))
    accepted_transform_rows = sorted(
        [
            [
                str(row["sample_id"]),
                str(row["frozen_gdino_train_transform_row_sha256"]),
                str(row["frozen_gdino_deploy_transform_row_sha256"]),
            ]
            for row in accepted
        ],
        key=lambda item: tuple(item),
    )

    _write_jsonl(accepted_path, accepted)
    _write_jsonl(rejected_path, rejected)
    _write_jsonl(quarantine_path, quarantine)
    audit: Dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "kind": AUDIT_KIND,
        "inputs": {
            "extractions": {
                **file_record(extraction_path),
                "rows": len(extractions),
            },
            "judgments": {
                **file_record(judgment_path),
                "rows": len(judgments),
            },
            "extraction_audit": {
                **extraction_audit["file"],
                "schema": EXTRACTION_AUDIT_SCHEMA,
                "kind": EXTRACTION_AUDIT_KIND,
                "rows": EXPECTED_EXTRACTION_ROWS,
            },
            "strict2031": strict2031_record,
            "strict1607": strict1607_record,
        },
        "locked_contract": {
            "extraction_schema": EXTRACTION_SCHEMA,
            "judgment_schema": JUDGMENT_SCHEMA,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
            "asset_policy_sha256": ASSET_POLICY_SHA256,
            "generation_config_sha256": GENERATION_CONFIG_SHA256,
            "vision_processor_config_sha256": VISION_PROCESSOR_CONFIG_SHA256,
            "inference_batch_size": INFERENCE_BATCH_SIZE,
            "judge_runtime_policy": JUDGE_RUNTIME_POLICY,
            "judge_runtime_policy_sha256": JUDGE_RUNTIME_POLICY_SHA256,
            "source_inherit_iou_threshold": INHERIT_IOU_THRESHOLD,
            "source_inherit_confidence_threshold": INHERIT_CONFIDENCE_THRESHOLD,
            "qwen_no_confidence_threshold": QWEN_CONFIDENCE_THRESHOLD,
        },
        "rows": len(extractions),
        "regions": total_regions,
        "decisions": {
            "accepted": len(accepted),
            "rejected": len(rejected),
            "quarantine": len(quarantine),
        },
        "row_reason_counts": dict(sorted(row_reason_counts.items())),
        "region_reason_counts": dict(sorted(region_reason_counts.items())),
        "region_method_counts": dict(sorted(region_method_counts.items())),
        "source_qwen_comparison_counts": dict(
            sorted(source_qwen_comparison_counts.items())
        ),
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        # Compatibility alias: transform_sha256 is the shared train static
        # contract, not any image-specific affine trace.
        "transform_sha256": next(iter(train_transform_contract_hashes)),
        "train_transform_contract_sha256": next(
            iter(train_transform_contract_hashes)
        ),
        "deploy_transform_contract_sha256": next(
            iter(deploy_transform_contract_hashes)
        ),
        "transform_rows_sha256": canonical_sha256(accepted_transform_rows),
        "extraction_transform_rows_sha256": canonical_sha256(
            transform_row_records
        ),
        "transform_rows_hash_contract": {
            "schema": "stage-b-transform-row-hash-list-v1",
            "payload": "[[sample_id,train_row_sha256,deploy_row_sha256],...]",
            "ordering": "lexicographic_by_all_three_string_fields",
            "canonicalization": (
                "json.ensure_ascii=true,sort_keys=true,separators=(',',':'),"
                "allow_nan=false;sha256(utf8)"
            ),
            "transform_rows_scope": "accepted_output_rows",
            "extraction_transform_rows_scope": "all_extraction_rows",
        },
        "strict_image_overlap": {"strict2031": 0, "strict1607": 0},
        "outputs": {
            "accepted": {**file_record(accepted_path), "rows": len(accepted)},
            "rejected": {**file_record(rejected_path), "rows": len(rejected)},
            "quarantine": {**file_record(quarantine_path), "rows": len(quarantine)},
        },
        "scope": {
            "tn_scope_compatibility": "image_global_topk_verified",
            "fixed_gdino_max_scope": FIXED_MAX_SCOPE,
            "all_900_gdino_queries_verified": False,
            "image_global_semantic_absence_proven": False,
            "portable_to_other_checkpoint_or_transform": False,
            "global_max_label_is_semantic_extrapolation": False,
            "claim": (
                "Every extracted train primary/shadow and formal-deploy top-1/"
                "near-tie union member is NO for this exact checkpoint and the "
                "two locked transforms, by audited high-overlap source inheritance "
                "or pinned Qwen judgment."
            ),
        },
    }
    _write_json(audit_path, audit)
    return audit


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify pinned-Qwen fixed-GDINO max-region judgments."
    )
    parser.add_argument("--extractions", required=True)
    parser.add_argument("--extraction-audit", required=True)
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--accepted-output", required=True)
    parser.add_argument("--rejected-output", required=True)
    parser.add_argument("--quarantine-output", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--strict2031", default=str(DEFAULT_STRICT2031))
    parser.add_argument("--strict1607", default=str(DEFAULT_STRICT1607))
    parser.add_argument(
        "--expected-strict2031-sha256", default=STRICT2031_SHA256
    )
    parser.add_argument(
        "--expected-strict1607-sha256", default=STRICT1607_SHA256
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    try:
        audit = verify(args)
    except (VerificationError, QwenJudgeError) as error:
        raise SystemExit(f"[ERROR] {error}") from error
    print(json.dumps(audit, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
