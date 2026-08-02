#!/usr/bin/env python3
"""Extract frozen-GDINO negative top-query regions for explicit VLM review.

The primary score is produced with the confidence-training call shape: local
batch four, one concatenated ``[positive_B, negative_B]`` forward, and AMP.
The formal evaluator's two separate forwards are retained only as a shadow
stability observation.  No cached proposal is promoted to an all-query label.

This tool is deliberately fail-closed.  It accepts only the completed
authoritative fixed baseline, locked semantic inputs, and the locked
strict2031/strict1607 manifests.  Rows sharing an image with either strict
manifest are excluded before batching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw, __version__ as PIL_VERSION


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SCHEMA = "stage-b-fixed-gdino-top1-vlm-extraction-v1"
AUDIT_SCHEMA = "stage-b-fixed-gdino-top1-vlm-extraction-audit-v1"
SCORE_CONTRACT = "float32_mean_sigmoid_over_generated_full_expression_tokens"
PRIMARY_FORWARD_CONTRACT = "confidence_train_paired_2b_pos_then_neg_local_b4"
SHADOW_FORWARD_CONTRACT = "formal_eval_separate_negative_then_positive_local_b4"
DEPLOY_FORWARD_CONTRACT = "formal_eval_val_resize_separate_negative_then_positive_b16"
EXPECTED_QUERIES = 900
LOCAL_BATCH_SIZE = 4
PAIRED_BATCH_SIZE = 8
DEPLOY_BATCH_SIZE = 16
DEFAULT_SEED = 42
DEFAULT_TIE_EPSILON = 1.0e-3
DEFAULT_BOX_DEDUPE_DECIMALS = 5

BASELINE_CONFIG = REPO_ROOT / (
    "config/ablations/cfg_stageb_from_gdino_ft_with_tn_alltn_tau05605_w036.py"
)
BASELINE_CONFIG_SHA256 = "62fb4fa21d6db11827f8729291ebb6f9b856ff696f0d9ed2c11cec05f13f2659"
DATA_CONFIG = REPO_ROOT / "config/ablations/cfg_stageb_gdino_score_adapter_semantic_verified.py"
SEMANTIC_PAIRS = REPO_ROOT / (
    "data/ablations/stageb_gdino_adapter_semantic_verified_20260711/"
    "semantic_verified_pairs.jsonl"
)
SEMANTIC_PAIRS_SHA256 = "bea2aca85d207d883da85cb219420f748a65a840516218731811e8e46449b645"
SEMANTIC_AUDIT = SEMANTIC_PAIRS.parent / "audit.json"
SEMANTIC_AUDIT_SHA256 = "0b08dc80d724b688154d6105c8b292c133177e36eb8659080e06a7d38fb77338"
SEMANTIC_ROWS = 17_829
FILTERED_ROWS = 17_738
FILTERED_OUT_ROWS = 91
RAW_SOURCE_AUDIT_SHA256 = "468934d229eb9de9e3641d43ab9fde7e9661d39c373e2f5e29cb3d3645bc1f98"
RAW_SOURCE_SHA256 = {
    "refcocoplus": "8025bb8c80e02c4e1e4f3f0f7f9dc8343062914f2ab50c1140d3ef79af1d9afc",
    "refcocog": "5cc41ca49e716dd96cc71ad63aa8acfff13b0fd47ccb17d51539677385217b24",
}

STRICT_ROOT = REPO_ROOT / "data/eval_manifests/stageb_vlm_verified_strict_ann_umd_val_20260711"
STRICT_SPECS = {
    "strict2031": {
        "path": STRICT_ROOT / "eval_manifest.jsonl",
        "rows": 2031,
        "unique_images": 1045,
        "sha256": "0e47763c01178d63ee22430a6c93d4fc6a210848d43f32aafbb2e6cd7243e918",
    },
    "strict1607": {
        "path": STRICT_ROOT / "semantic_stageb_union_image_disjoint_manifest.jsonl",
        "rows": 1607,
        "unique_images": 795,
        "sha256": "f2dc97d58884b8de3ae2c8b4cefd281432e15c8952b23b5e0252eb8e5be36d25",
    },
}

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/stageb_fixed_gdino_top1_vlm_20260712"
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs/gdino_ft_stage_b_fixed_baseline_20260711/checkpoint0000.pth"

CODE_ENTRIES = (
    "tools/extract_stageb_fixed_gdino_top1_vlm_manifest.py",
    "engine.py",
    "tools/eval_text_groundingdino_refcoco_tn.py",
)
CODE_INCLUDE = (
    "datasets/__init__.py",
    "datasets/patch_episode.py",
    "datasets/transforms.py",
    "models/__init__.py",
    "models/GroundingDINO/__init__.py",
    "models/GroundingDINO/groundingdino.py",
    "models/GroundingDINO/transformer.py",
    "models/GroundingDINO/bertwarper.py",
    "models/GroundingDINO/ms_deform_attn.py",
    "tools/stageb_eval_records.py",
    "tools/stageb_fixed_protocol_audit.py",
    "tools/stageb_dependency_audit.py",
)
CODE_ORCHESTRATION = (
    "tools/run_stageb_fixed_protocol_eval.sh",
)
NATIVE_EXTENSION_MODULE = "MultiScaleDeformableAttention"

ASSET_POLICY = {
    "schema": "stage-b-fixed-gdino-vlm-assets-v1",
    "box_color_rgb": [255, 0, 0],
    "box_width_px": 4,
    "tight_encoding": {"format": "PNG", "compress_level": 9, "optimize": False},
    "context_scale": 2.0,
    "context_encoding": {"format": "PNG", "compress_level": 9, "optimize": False},
    "full_boxed_encoding": {
        "format": "JPEG",
        "quality": 95,
        "subsampling": 0,
        "optimize": False,
        "progressive": False,
    },
    "rasterization": "floor_xy_min_ceil_xy_max_clamp_min_one_pixel",
}

CLAIMS = {
    "frozen_gdino_global_max_regions_extracted": True,
    "train_path_and_deploy_transform_regions_extracted": True,
    "all_900_gdino_queries_verified": False,
    "image_global_semantic_absence_proven": False,
    "portable_to_other_checkpoint_or_transform": False,
}


class ExtractionError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ExtractionError(f"value is not canonical-JSON serializable: {error}") from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def _numeric_bucket(value: float) -> str:
    value = float(value)
    for threshold in (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1):
        if value < threshold:
            return f"lt_{threshold:.0e}"
    return "ge_1e-01"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ExtractionError(f"required file is missing: {path}")
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExtractionError(f"could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ExtractionError(f"expected a JSON object: {path}")
    return value


def iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    if not path.is_file():
        raise ExtractionError(f"missing JSONL: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ExtractionError(f"blank JSONL row at {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ExtractionError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ExtractionError(f"non-object JSONL row at {path}:{line_number}")
            yield line_number, row


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _identity(row: Mapping[str, Any], *, context: str) -> Tuple[str, int, int, int, int]:
    dataset = str(row.get("dataset", "")).strip()
    if dataset not in RAW_SOURCE_SHA256:
        raise ExtractionError(f"invalid dataset at {context}: {dataset!r}")
    values: List[int] = []
    for key in ("image_id", "ann_id", "ref_id", "sent_id"):
        try:
            values.append(int(row[key]))
        except (KeyError, TypeError, ValueError) as error:
            raise ExtractionError(f"invalid {key} at {context}") from error
    return (dataset, *values)


def _require_sha(path: Path, expected: str, *, label: str) -> Dict[str, Any]:
    record = file_record(path)
    if record["sha256"] != expected:
        raise ExtractionError(
            f"{label} SHA-256 mismatch: expected {expected}, got {record['sha256']}"
        )
    return record


def validate_strict_manifests(
    paths: Optional[Mapping[str, Path]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], set[int]]:
    records: Dict[str, Dict[str, Any]] = {}
    image_sets: Dict[str, set[int]] = {}
    sample_sets: Dict[str, set[str]] = {}
    selected = paths or {name: spec["path"] for name, spec in STRICT_SPECS.items()}
    if set(selected) != set(STRICT_SPECS):
        raise ExtractionError("strict manifest mapping must contain exactly strict2031 and strict1607")
    for name, spec in STRICT_SPECS.items():
        path = Path(selected[name]).resolve()
        record = _require_sha(path, str(spec["sha256"]), label=name)
        images: set[int] = set()
        samples: set[str] = set()
        rows = 0
        for line_number, row in iter_jsonl(path):
            if row.get("manifest_schema") != "stageb_vlm_verified_strict_tn_v2":
                raise ExtractionError(f"wrong strict schema at {path}:{line_number}")
            try:
                images.add(int(row["image_id"]))
            except (KeyError, TypeError, ValueError) as error:
                raise ExtractionError(f"invalid strict image_id at {path}:{line_number}") from error
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id or sample_id in samples:
                raise ExtractionError(f"missing or duplicate strict sample_id at {path}:{line_number}")
            samples.add(sample_id)
            rows += 1
        if rows != int(spec["rows"]) or len(images) != int(spec["unique_images"]):
            raise ExtractionError(
                f"{name} count drift: rows={rows}, unique_images={len(images)}"
            )
        record.update({"rows": rows, "unique_images": len(images)})
        records[name] = record
        image_sets[name] = images
        sample_sets[name] = samples
    if not sample_sets["strict1607"].issubset(sample_sets["strict2031"]):
        raise ExtractionError("strict1607 is no longer a sample-id subset of strict2031")
    image_union = set().union(*image_sets.values())
    if len(image_union) != int(STRICT_SPECS["strict2031"]["unique_images"]):
        raise ExtractionError("strict2031/strict1607 image union drifted")
    return records, image_union


def _load_raw_source_rows(audit: Mapping[str, Any]) -> Dict[Path, List[Dict[str, Any]]]:
    source_records = audit.get("sources")
    if not isinstance(source_records, Mapping) or set(source_records) != set(RAW_SOURCE_SHA256):
        raise ExtractionError("semantic audit must describe exactly both frozen raw sources")
    result: Dict[Path, List[Dict[str, Any]]] = {}
    for dataset, expected_sha in RAW_SOURCE_SHA256.items():
        source_record = source_records.get(dataset)
        if not isinstance(source_record, Mapping) or not source_record.get("path"):
            raise ExtractionError(f"semantic audit has no raw source for {dataset}")
        path = Path(str(source_record["path"])).resolve()
        _require_sha(path, expected_sha, label=f"raw semantic source {dataset}")
        if source_record.get("sha256") != expected_sha:
            raise ExtractionError(f"semantic audit raw-source hash drifted for {dataset}")
        rows = [row for _, row in iter_jsonl(path)]
        if len(rows) != int(source_record.get("rows", -1)):
            raise ExtractionError(f"raw source row count drifted for {dataset}")
        result[path] = rows
    return result


def validate_semantic_inputs(
    pair_path: Path = SEMANTIC_PAIRS,
    audit_path: Path = SEMANTIC_AUDIT,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pair_path = pair_path.resolve()
    audit_path = audit_path.resolve()
    pair_record = _require_sha(pair_path, SEMANTIC_PAIRS_SHA256, label="semantic pairs")
    audit_record = _require_sha(audit_path, SEMANTIC_AUDIT_SHA256, label="semantic audit")
    audit = read_json(audit_path)
    if (
        audit.get("schema") != "stage-b-gdino-adapter-semantic-verified-pairs-v1"
        or int(audit.get("rows", -1)) != SEMANTIC_ROWS
        or audit.get("output_sha256") != SEMANTIC_PAIRS_SHA256
        or audit.get("cached_proposal_coverage_only") is not True
        or audit.get("all_900_gdino_queries_verified") is not False
    ):
        raise ExtractionError("semantic pair audit contract drifted")
    source_audit = audit.get("source_audit")
    if not isinstance(source_audit, Mapping) or not source_audit.get("path"):
        raise ExtractionError("semantic audit has no raw-source audit record")
    source_audit_path = Path(str(source_audit["path"])).resolve()
    raw_audit_record = _require_sha(
        source_audit_path, RAW_SOURCE_AUDIT_SHA256, label="raw semantic source audit"
    )
    if source_audit.get("sha256") != RAW_SOURCE_AUDIT_SHA256:
        raise ExtractionError("semantic audit raw-source-audit hash drifted")

    # Reuse the source builder's full frozen contract before binding raw lines.
    from tools.build_stageb_gdino_adapter_semantic_verified_pairs import verify as verify_semantic

    verify_args = argparse.Namespace(output=pair_path, audit=audit_path, expected_rows=SEMANTIC_ROWS)
    try:
        verify_semantic(verify_args)
    except Exception as error:
        raise ExtractionError(f"semantic builder verification failed: {error}") from error

    raw_rows = _load_raw_source_rows(audit)
    bindings: List[Dict[str, Any]] = []
    identities: set[Tuple[str, int, int, int, int]] = set()
    for pair_line, pair in iter_jsonl(pair_path):
        context = f"{pair_path}:{pair_line}"
        identity = _identity(pair, context=context)
        if identity in identities:
            raise ExtractionError(f"duplicate semantic identity at {context}: {identity}")
        identities.add(identity)
        if (
            pair.get("adapter_pair_schema")
            != "stage-b-gdino-adapter-semantic-verified-pair-v1"
            or pair.get("global_tn_verified") is not True
            or pair.get("proposalset_proxy_verified") is not False
            or pair.get("cached_proposal_coverage_only") is not True
            or pair.get("all_900_gdino_queries_verified") is not False
        ):
            raise ExtractionError(f"invalid semantic pair contract at {context}")
        source_path = Path(str(pair.get("source_file", ""))).resolve()
        rows = raw_rows.get(source_path)
        try:
            source_line = int(pair["source_line"])
        except (KeyError, TypeError, ValueError) as error:
            raise ExtractionError(f"invalid raw source line at {context}") from error
        if rows is None or source_line <= 0 or source_line > len(rows):
            raise ExtractionError(f"semantic pair raw source binding is invalid at {context}")
        source_row = rows[source_line - 1]
        if canonical_sha256(source_row) != pair.get("source_row_sha256"):
            raise ExtractionError(f"raw semantic source row hash drifted at {context}")
        if _identity(source_row, context=f"{source_path}:{source_line}") != identity:
            raise ExtractionError(f"raw semantic source identity drifted at {context}")
        bindings.append(
            {
                "pair_path": pair_path,
                "pair_line": pair_line,
                "pair": pair,
                "pair_row_sha256": canonical_sha256(pair),
                "source_path": source_path,
                "source_file_sha256": RAW_SOURCE_SHA256[identity[0]],
                "source_line": source_line,
                "source_row": source_row,
            }
        )
    if len(bindings) != SEMANTIC_ROWS:
        raise ExtractionError(f"semantic pair row count drifted: {len(bindings)}")
    provenance = {
        "pairs": pair_record,
        "audit": audit_record,
        "raw_source_audit": raw_audit_record,
        "raw_sources": {
            dataset: dict(audit["sources"][dataset]) for dataset in sorted(RAW_SOURCE_SHA256)
        },
    }
    return bindings, provenance


def exclude_holdout_image_union(
    bindings: Sequence[Mapping[str, Any]],
    excluded_images: set[int],
    *,
    enforce_frozen_counts: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    removed: List[Mapping[str, Any]] = []
    for binding in bindings:
        pair = binding.get("pair")
        if not isinstance(pair, Mapping):
            raise ExtractionError("semantic binding has no pair row")
        try:
            image_id = int(pair["image_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise ExtractionError("semantic binding has an invalid image_id") from error
        if image_id in excluded_images:
            removed.append(binding)
        else:
            kept.append(dict(binding))
    if enforce_frozen_counts and (len(kept) != FILTERED_ROWS or len(removed) != FILTERED_OUT_ROWS):
        raise ExtractionError(
            f"strict image-union exclusion drifted: kept={len(kept)}, removed={len(removed)}"
        )
    stats = {
        "input_rows": len(bindings),
        "kept_rows": len(kept),
        "excluded_rows": len(removed),
        "excluded_unique_images": len(
            {int(binding["pair"]["image_id"]) for binding in removed}
        ),
        "excluded_by_dataset": dict(
            sorted(Counter(str(binding["pair"]["dataset"]) for binding in removed).items())
        ),
        "kept_by_dataset": dict(
            sorted(Counter(str(binding["pair"]["dataset"]) for binding in kept).items())
        ),
        "policy": "exclude_union_by_image_id_before_batching",
    }
    return kept, stats


def aggregate_full_expression_scores(
    token_logits: torch.Tensor,
    generated_phrase_mask: torch.Tensor,
    *,
    expected_queries: int = EXPECTED_QUERIES,
) -> torch.Tensor:
    if token_logits.dim() != 3 or not token_logits.is_floating_point():
        raise ExtractionError(
            f"token logits must be floating BxQxT, got {tuple(token_logits.shape)}"
        )
    if int(token_logits.shape[1]) != int(expected_queries):
        raise ExtractionError(
            f"expected exactly {expected_queries} GDINO queries, got {token_logits.shape[1]}"
        )
    mask = torch.as_tensor(generated_phrase_mask, device=token_logits.device)
    if mask.dim() == 3:
        if int(mask.shape[0]) != int(token_logits.shape[0]):
            raise ExtractionError("phrase-mask batch does not match logits")
        mask = mask.to(dtype=torch.bool).any(dim=1)
    elif mask.dim() == 2:
        mask = mask.to(dtype=torch.bool)
    else:
        raise ExtractionError(f"generated phrase mask must be BxKxT or BxT, got {tuple(mask.shape)}")
    if tuple(mask.shape) != (int(token_logits.shape[0]), int(token_logits.shape[2])):
        raise ExtractionError(
            f"full-expression mask shape {tuple(mask.shape)} does not match logits {tuple(token_logits.shape)}"
        )
    token_count = mask.sum(dim=-1)
    if bool((token_count <= 0).any().item()):
        raise ExtractionError("every expression must contain at least one generated scored token")
    if not bool(torch.isfinite(token_logits).all().item()):
        raise ExtractionError("GDINO token logits contain non-finite values")
    probability = token_logits.float().sigmoid()
    scores = (
        probability.masked_fill(~mask[:, None, :], 0.0).sum(dim=-1)
        / token_count[:, None].to(dtype=torch.float32)
    )
    if not bool(torch.isfinite(scores).all().item()):
        raise ExtractionError("aggregated GDINO scores contain non-finite values")
    return scores


def _xywh_to_xyxy(box: Sequence[float]) -> List[float]:
    if len(box) != 4:
        raise ExtractionError(f"xywh box must have four values: {box!r}")
    x, y, w, h = [float(value) for value in box]
    if not all(math.isfinite(value) for value in (x, y, w, h)) or w <= 0.0 or h <= 0.0:
        raise ExtractionError(f"invalid xywh box: {box!r}")
    return [x, y, x + w, y + h]


def _xyxy_to_xywh(box: Sequence[float]) -> List[float]:
    if len(box) != 4:
        raise ExtractionError(f"xyxy box must have four values: {box!r}")
    x0, y0, x1, y1 = [float(value) for value in box]
    return [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]


def _cxcywh_to_xyxy(box: Sequence[float]) -> List[float]:
    if len(box) != 4:
        raise ExtractionError(f"cxcywh box must have four values: {box!r}")
    cx, cy, width, height = [float(value) for value in box]
    return [cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0]


def box_iou_xyxy(first: Sequence[float], second: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(value) for value in first]
    bx0, by0, bx1, by1 = [float(value) for value in second]
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0.0, min(ay1, by1) - max(ay0, by0)
    )
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def inverse_transform_box(
    bbox_cxcywh_norm: Sequence[float],
    trace: Mapping[str, Any],
) -> Dict[str, List[float]]:
    try:
        output_h, output_w = [int(value) for value in trace["output_hw"]]
        original_h, original_w = [int(value) for value in trace["original_hw"]]
        scale_x, scale_y = [float(value) for value in trace["scale_xy"]]
        offset_x, offset_y = [float(value) for value in trace["offset_xy"]]
    except (KeyError, TypeError, ValueError) as error:
        raise ExtractionError("malformed transform trace") from error
    if min(output_h, output_w, original_h, original_w) <= 0 or scale_x <= 0.0 or scale_y <= 0.0:
        raise ExtractionError("transform trace has non-positive dimensions or scale")
    xyxy_norm = [min(1.0, max(0.0, value)) for value in _cxcywh_to_xyxy(bbox_cxcywh_norm)]
    transformed = [
        xyxy_norm[0] * output_w,
        xyxy_norm[1] * output_h,
        xyxy_norm[2] * output_w,
        xyxy_norm[3] * output_h,
    ]
    original = [
        (transformed[0] - offset_x) / scale_x,
        (transformed[1] - offset_y) / scale_y,
        (transformed[2] - offset_x) / scale_x,
        (transformed[3] - offset_y) / scale_y,
    ]
    original = [
        min(float(original_w), max(0.0, original[0])),
        min(float(original_h), max(0.0, original[1])),
        min(float(original_w), max(0.0, original[2])),
        min(float(original_h), max(0.0, original[3])),
    ]
    return {
        "bbox_xyxy_norm": xyxy_norm,
        "bbox_xyxy_transformed": transformed,
        "bbox_xyxy_original": original,
        "bbox_xywh_original": _xyxy_to_xywh(original),
    }


def make_query_observation(
    scores: torch.Tensor,
    boxes_cxcywh_norm: torch.Tensor,
    *,
    trace: Mapping[str, Any],
    target_bbox_xywh: Sequence[float],
    tie_epsilon: float = DEFAULT_TIE_EPSILON,
) -> Dict[str, Any]:
    scores = torch.as_tensor(scores).detach().float().cpu()
    boxes = torch.as_tensor(boxes_cxcywh_norm).detach().float().cpu()
    if scores.dim() != 1 or boxes.shape != (scores.numel(), 4):
        raise ExtractionError(
            f"query score/box shape mismatch: scores={tuple(scores.shape)}, boxes={tuple(boxes.shape)}"
        )
    if int(scores.numel()) != EXPECTED_QUERIES:
        raise ExtractionError(f"expected {EXPECTED_QUERIES} query scores, got {scores.numel()}")
    if not bool(torch.isfinite(scores).all().item()) or not bool(torch.isfinite(boxes).all().item()):
        raise ExtractionError("query observation contains non-finite scores or boxes")
    if tie_epsilon < 0.0 or not math.isfinite(float(tie_epsilon)):
        raise ExtractionError("tie epsilon must be finite and non-negative")
    query_id = int(torch.argmax(scores).item())
    best_score = float(scores[query_id].item())
    without_best = scores.clone()
    without_best[query_id] = -torch.inf
    second_score = float(without_best.max().item())
    margin = best_score - second_score
    near_ids = (
        (best_score - scores <= float(tie_epsilon)).nonzero(as_tuple=False).flatten().tolist()
    )
    extra_near_ids = [int(value) for value in near_ids if int(value) != query_id]
    ordered_ids = [query_id] + extra_near_ids
    target_xyxy = _xywh_to_xyxy(target_bbox_xywh)
    candidates: List[Dict[str, Any]] = []
    for candidate_id in ordered_ids:
        normalized = [float(value) for value in boxes[candidate_id].tolist()]
        geometry = inverse_transform_box(normalized, trace)
        candidates.append(
            {
                "query_id": candidate_id,
                "base_score": float(scores[candidate_id].item()),
                "bbox_cxcywh_norm": normalized,
                **geometry,
                "target_iou": box_iou_xyxy(geometry["bbox_xyxy_original"], target_xyxy),
            }
        )
    primary_candidate = candidates[0]
    summary = {
        "query_id": query_id,
        "base_score": best_score,
        "second_score": second_score,
        "margin": margin,
        "bbox_cxcywh_norm": primary_candidate["bbox_cxcywh_norm"],
        "bbox_xyxy_norm": primary_candidate["bbox_xyxy_norm"],
        "bbox_xyxy_original": primary_candidate["bbox_xyxy_original"],
        "bbox_xywh_original": primary_candidate["bbox_xywh_original"],
        "target_iou": primary_candidate["target_iou"],
    }
    return {
        "summary": summary,
        "near_tie_query_ids": sorted(extra_near_ids),
        "candidates": candidates,
    }


def summarize_stability(
    observations: Mapping[str, Mapping[str, Any]],
    *,
    epsilon: float = DEFAULT_TIE_EPSILON,
) -> Dict[str, Any]:
    if any(name not in observations for name in ("primary", "shadow", "deploy")):
        raise ExtractionError("stability requires primary, shadow, and deploy observations")
    primary = observations["primary"].get("summary")
    if not isinstance(primary, Mapping):
        raise ExtractionError("primary observation has no summary")
    primary_q = int(primary["query_id"])
    query_ids: Dict[str, int] = {}
    score_drift: Dict[str, float] = {}
    box_drift: Dict[str, Optional[float]] = {}
    near_ties: set[int] = set()
    for origin, observation in observations.items():
        summary = observation.get("summary")
        if not isinstance(summary, Mapping):
            raise ExtractionError(f"observation {origin} has no summary")
        query_id = int(summary["query_id"])
        query_ids[origin] = query_id
        score_drift[origin] = abs(float(summary["base_score"]) - float(primary["base_score"]))
        if query_id == primary_q:
            box_drift[origin] = max(
                abs(float(a) - float(b))
                for a, b in zip(summary["bbox_cxcywh_norm"], primary["bbox_cxcywh_norm"])
            )
        else:
            box_drift[origin] = None
        near_ties.update(int(value) for value in observation.get("near_tie_query_ids", []))
    max_score_drift = max(score_drift.values(), default=0.0)
    all_agree = all(query_id == primary_q for query_id in query_ids.values())
    margin_guard = float(primary["margin"]) > 2.0 * max_score_drift + float(epsilon)
    return {
        "policy": "query_union_on_disagreement_or_near_tie_fail_closed",
        "epsilon": float(epsilon),
        "primary_shadow_agree": query_ids.get("shadow") == primary_q,
        "primary_deploy_agree": query_ids.get("deploy") == primary_q,
        "all_observations_agree": all_agree,
        "query_ids_by_origin": query_ids,
        "score_drift_from_primary": score_drift,
        "bbox_cxcywh_norm_max_abs_drift_from_primary": box_drift,
        "max_abs_score_drift": max_score_drift,
        "margin_guard_pass": margin_guard,
        "near_tie_query_ids": sorted(near_ties),
        "near_tie_count": len(near_ties),
        "exact_top1_stable": bool(all_agree and margin_guard and not near_ties),
    }


def _judgment_answer(value: Any) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    if not isinstance(value, Mapping):
        return None, None, None
    judgment = value.get("judgment") if isinstance(value.get("judgment"), Mapping) else value
    answer = str(judgment.get("answer", "")).strip().lower() or None
    confidence_value = judgment.get("confidence")
    try:
        confidence = float(confidence_value) if confidence_value is not None else None
    except (TypeError, ValueError):
        confidence = None
    return answer, confidence, canonical_sha256(judgment)


def source_max_overlap(
    bbox_xywh: Sequence[float],
    source_row: Mapping[str, Any],
) -> Dict[str, Any]:
    candidate_xyxy = _xywh_to_xyxy(bbox_xywh)
    candidates: List[Dict[str, Any]] = []
    target = source_row.get("target_bbox_used")
    if isinstance(target, (list, tuple)) and len(target) == 4:
        answer, confidence, judgment_sha = _judgment_answer(source_row.get("visual_local_judgment"))
        candidates.append(
            {
                "kind": "target",
                "proposal_id": None,
                "iou": box_iou_xyxy(candidate_xyxy, _xywh_to_xyxy(target)),
                "source_answer": answer,
                "source_confidence": confidence,
                "source_judgment_sha256": judgment_sha,
            }
        )
    judgments: Dict[str, Mapping[str, Any]] = {}
    for judgment in source_row.get("visual_proposal_judgments", []) or []:
        if isinstance(judgment, Mapping) and judgment.get("proposal_id") is not None:
            judgments[str(judgment["proposal_id"])] = judgment
    for proposal in source_row.get("proposal_cache", []) or []:
        if not isinstance(proposal, Mapping) or proposal.get("proposal_id") is None:
            continue
        proposal_bbox = proposal.get("bbox")
        if not isinstance(proposal_bbox, (list, tuple)) or len(proposal_bbox) != 4:
            continue
        proposal_id = proposal["proposal_id"]
        answer, confidence, judgment_sha = _judgment_answer(judgments.get(str(proposal_id)))
        candidates.append(
            {
                "kind": "proposal",
                "proposal_id": proposal_id,
                "iou": box_iou_xyxy(candidate_xyxy, _xywh_to_xyxy(proposal_bbox)),
                "source_answer": answer,
                "source_confidence": confidence,
                "source_judgment_sha256": judgment_sha,
            }
        )
    if not candidates:
        return {
            "kind": "none",
            "proposal_id": None,
            "iou": 0.0,
            "source_answer": None,
            "source_confidence": None,
            "source_judgment_sha256": None,
        }
    return max(candidates, key=lambda item: (float(item["iou"]), item["kind"] == "target"))


def build_regions(
    sample_id: str,
    observations: Mapping[str, Mapping[str, Any]],
    source_row: Mapping[str, Any],
    *,
    dedupe_decimals: int = DEFAULT_BOX_DEDUPE_DECIMALS,
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[float, ...], Dict[str, Any]] = {}
    for origin, observation in observations.items():
        candidates = observation.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ExtractionError(f"observation {origin} has no query candidates")
        for candidate in candidates:
            bbox = [float(value) for value in candidate["bbox_xywh_original"]]
            key = tuple(round(value, int(dedupe_decimals)) for value in bbox)
            group = groups.setdefault(
                key,
                {
                    "origins": [],
                    "query_ids": [],
                    "base_scores": {},
                    "candidate": candidate,
                },
            )
            if origin not in group["origins"]:
                group["origins"].append(origin)
            query_id = int(candidate["query_id"])
            top_query_id = int(observation["summary"]["query_id"])
            if query_id != top_query_id and "near_tie" not in group["origins"]:
                group["origins"].append("near_tie")
            if query_id not in group["query_ids"]:
                group["query_ids"].append(query_id)
            group["base_scores"][f"{origin}:q{query_id}"] = float(candidate["base_score"])
    regions: List[Dict[str, Any]] = []
    for key, group in groups.items():
        representative = group["candidate"]
        origins = sorted(group["origins"])
        query_ids = sorted(group["query_ids"])
        bbox_xywh = [float(value) for value in representative["bbox_xywh_original"]]
        identity_payload = {
            "sample_id": sample_id,
            "bbox_xywh_original": list(key),
            "origins": origins,
            "query_ids": query_ids,
        }
        overlap = source_max_overlap(bbox_xywh, source_row)
        inherit_eligible = bool(
            overlap.get("source_answer") == "no"
            and overlap.get("source_confidence") is not None
            and float(overlap["source_confidence"]) >= 0.90
            and float(overlap.get("iou", 0.0)) >= 0.70
        )
        regions.append(
            {
                "region_id": canonical_sha256(identity_payload),
                "origins": origins,
                "query_ids": query_ids,
                "base_scores": dict(sorted(group["base_scores"].items())),
                "bbox_xyxy_norm": [float(value) for value in representative["bbox_xyxy_norm"]],
                "bbox_xyxy_original": [float(value) for value in representative["bbox_xyxy_original"]],
                "bbox_xywh_original": bbox_xywh,
                "bbox_sha256": canonical_sha256(bbox_xywh),
                "max_overlap": overlap,
                "inherit_eligible": inherit_eligible,
                "assets": {},
                "judgment": {"status": "pending", "cache_key": None},
            }
        )
    regions.sort(
        key=lambda region: (
            0 if "primary" in region["origins"] else 1,
            min(region["query_ids"]),
            region["region_id"],
        )
    )
    return regions


def _raster_box(box: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(value) for value in box]
    left = min(max(0, math.floor(x0)), max(0, width - 1))
    top = min(max(0, math.floor(y0)), max(0, height - 1))
    right = min(width, max(left + 1, math.ceil(x1)))
    bottom = min(height, max(top + 1, math.ceil(y1)))
    return int(left), int(top), int(right), int(bottom)


def _context_box(box: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    left, top, right, bottom = box
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    context_w = (right - left) * float(ASSET_POLICY["context_scale"])
    context_h = (bottom - top) * float(ASSET_POLICY["context_scale"])
    return _raster_box(
        [center_x - context_w / 2.0, center_y - context_h / 2.0,
         center_x + context_w / 2.0, center_y + context_h / 2.0],
        width,
        height,
    )


def _draw_box(image: Image.Image, box: Sequence[int]) -> None:
    left, top, right, bottom = [int(value) for value in box]
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [left, top, max(left, right - 1), max(top, bottom - 1)],
        outline=tuple(ASSET_POLICY["box_color_rgb"]),
        width=int(ASSET_POLICY["box_width_px"]),
    )


def generate_region_assets(
    image: Image.Image,
    region: Mapping[str, Any],
    *,
    temporary_root: Path,
    final_root: Path,
    sample_key: str,
) -> Dict[str, Any]:
    image = image.convert("RGB")
    width, height = image.size
    raster = _raster_box(region["bbox_xyxy_original"], width, height)
    context = _context_box(raster, width, height)
    relative = Path(sample_key) / str(region["region_id"])
    temporary_dir = temporary_root / relative
    temporary_dir.mkdir(parents=True, exist_ok=False)

    tight_path = temporary_dir / "tight.png"
    image.crop(raster).save(tight_path, **ASSET_POLICY["tight_encoding"])

    context_image = image.crop(context)
    context_local = (
        raster[0] - context[0], raster[1] - context[1],
        raster[2] - context[0], raster[3] - context[1],
    )
    _draw_box(context_image, context_local)
    context_path = temporary_dir / "context_2x_boxed.png"
    context_image.save(context_path, **ASSET_POLICY["context_encoding"])

    full_image = image.copy()
    _draw_box(full_image, raster)
    full_path = temporary_dir / "full_boxed.jpg"
    full_image.save(full_path, **ASSET_POLICY["full_boxed_encoding"])

    policy_sha = canonical_sha256(ASSET_POLICY)
    result: Dict[str, Any] = {
        "asset_policy_sha256": policy_sha,
        "raster_bbox_xyxy": list(raster),
        "context_bbox_xyxy": list(context),
        "pillow_version": PIL_VERSION,
    }
    for key, local_path in (
        ("tight", tight_path),
        ("context_2x_boxed", context_path),
        ("full_boxed", full_path),
    ):
        final_path = (final_root / relative / local_path.name).resolve()
        with Image.open(local_path) as asset:
            asset_width, asset_height = asset.size
        result[key] = {
            "path": str(final_path),
            "sha256": sha256_file(local_path),
            "width": int(asset_width),
            "height": int(asset_height),
        }
    return result


def transform_contract_from_cfg(cfg: Any) -> Dict[str, Any]:
    contract = {
        "schema": "stage-b-confidence-traceable-train-transform-v1",
        "image_set": "train",
        "fix_size": bool(getattr(cfg, "fix_size", False)),
        "strong_aug": bool(getattr(cfg, "strong_aug", False)),
        "hflip_prob": float(getattr(cfg, "data_aug_hflip_prob", 0.5)),
        "scales": list(getattr(cfg, "data_aug_scales", [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800])),
        "max_size": int(getattr(cfg, "data_aug_max_size", 1333)),
        "scales2_resize": list(getattr(cfg, "data_aug_scales2_resize", [400, 500, 600])),
        "scales2_crop": list(getattr(cfg, "data_aug_scales2_crop", [384, 600])),
        "scale_overlap": getattr(cfg, "data_aug_scale_overlap", None),
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    }
    if contract["fix_size"] is not True or contract["strong_aug"] is not False:
        raise ExtractionError("frozen confidence extraction requires fix_size=true and strong_aug=false")
    if not math.isclose(contract["hflip_prob"], 0.0, rel_tol=0.0, abs_tol=0.0):
        raise ExtractionError("frozen confidence extraction requires data_aug_hflip_prob=0.0")
    if contract["scale_overlap"] not in (None, 0, 0.0):
        raise ExtractionError("frozen confidence extraction does not allow scale-overlap drift")
    contract["canonical_json"] = canonical_json(contract)
    contract["sha256"] = canonical_sha256(contract)
    return contract


class TraceableConfidenceTrainTransform:
    """The current confidence train transform with an invertible geometry trace."""

    def __init__(self, contract: Mapping[str, Any]) -> None:
        self.contract = dict(contract)

    def __call__(self, image: Image.Image, target: MutableMapping[str, Any]):
        from torchvision.transforms import RandomCrop
        import datasets.transforms as transforms

        original_w, original_h = image.size
        scale_x = scale_y = 1.0
        offset_x = offset_y = 0.0
        operations: List[Dict[str, Any]] = []

        # Match RandomHorizontalFlip's RNG consumption even though p is locked to zero.
        flip_draw = random.random()
        if flip_draw < float(self.contract["hflip_prob"]):
            raise ExtractionError("horizontal flip is forbidden for referring-expression extraction")
        operations.append({"op": "hflip_draw", "draw": flip_draw, "applied": False})

        def apply_resize(size: Any, *, max_size: Optional[int]) -> None:
            nonlocal image, target, scale_x, scale_y, offset_x, offset_y
            before_w, before_h = image.size
            requested = list(size) if isinstance(size, (list, tuple)) else int(size)
            image, target = transforms.resize(image, target, size, max_size)
            after_w, after_h = image.size
            ratio_x = float(after_w) / float(before_w)
            ratio_y = float(after_h) / float(before_h)
            scale_x *= ratio_x
            scale_y *= ratio_y
            offset_x *= ratio_x
            offset_y *= ratio_y
            operations.append(
                {
                    "op": "resize",
                    "requested_size": requested,
                    "before_hw": [before_h, before_w],
                    "after_hw": [after_h, after_w],
                    "ratio_xy": [ratio_x, ratio_y],
                }
            )

        if bool(self.contract["fix_size"]):
            fixed_sizes = [
                (int(self.contract["max_size"]), max(int(value) for value in self.contract["scales"]))
            ]
            fixed_size = random.choice(fixed_sizes)
            apply_resize(fixed_size, max_size=None)
            operations.append({"op": "fixed_resize", "size_wh": list(fixed_size)})
        else:
            branch_draw = random.random()
            if branch_draw < 0.5:
                apply_resize(
                    int(random.choice(self.contract["scales"])),
                    max_size=int(self.contract["max_size"]),
                )
                branch = "direct_resize"
            else:
                apply_resize(
                    int(random.choice(self.contract["scales2_resize"])),
                    max_size=None,
                )
                crop_min, crop_max = [int(value) for value in self.contract["scales2_crop"]]
                crop_w = random.randint(crop_min, min(image.width, crop_max))
                crop_h = random.randint(crop_min, min(image.height, crop_max))
                top, left, height, width = RandomCrop.get_params(image, [crop_h, crop_w])
                image, target = transforms.crop(image, target, (top, left, height, width))
                offset_x -= float(left)
                offset_y -= float(top)
                operations.append(
                    {"op": "crop", "top": int(top), "left": int(left), "height": int(height), "width": int(width)}
                )
                apply_resize(
                    int(random.choice(self.contract["scales"])),
                    max_size=int(self.contract["max_size"]),
                )
                branch = "resize_crop_resize"
            operations.insert(1, {"op": "random_select", "draw": branch_draw, "branch": branch})

        image, target = transforms.ToTensor()(image, target)
        image, target = transforms.Normalize(
            self.contract["normalize_mean"], self.contract["normalize_std"]
        )(image, target)
        output_h, output_w = [int(value) for value in image.shape[-2:]]
        trace = {
            "schema": "stage-b-image-affine-trace-v1",
            "original_hw": [int(original_h), int(original_w)],
            "output_hw": [output_h, output_w],
            "scale_xy": [scale_x, scale_y],
            "offset_xy": [offset_x, offset_y],
            "operations": operations,
        }
        trace["canonical_json"] = canonical_json(trace)
        trace["sha256"] = canonical_sha256(trace)
        target["_stageb_extraction_transform_trace"] = trace
        return image, target


def deploy_transform_contract_from_cfg(cfg: Any) -> Dict[str, Any]:
    scales = list(
        getattr(
            cfg,
            "data_aug_scales",
            [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
        )
    )
    contract = {
        "schema": "stage-b-formal-deploy-val-transform-v1",
        "image_set": "val",
        "resize_short_side": int(max(scales)),
        "max_size": int(getattr(cfg, "data_aug_max_size", 1333)),
        "aspect_preserving": True,
        "hflip_prob": 0.0,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    }
    if contract["resize_short_side"] != 800 or contract["max_size"] != 1333:
        raise ExtractionError("formal deploy transform must remain min=800,max=1333")
    contract["canonical_json"] = canonical_json(contract)
    contract["sha256"] = canonical_sha256(contract)
    return contract


class TraceableDeployEvalTransform:
    """The formal evaluator's aspect-preserving val resize with inverse trace."""

    def __init__(self, contract: Mapping[str, Any]) -> None:
        self.contract = dict(contract)

    def __call__(self, image: Image.Image, target: MutableMapping[str, Any]):
        import datasets.transforms as transforms

        original_w, original_h = image.size
        requested = random.choice([int(self.contract["resize_short_side"])])
        image, target = transforms.resize(
            image,
            target,
            requested,
            int(self.contract["max_size"]),
        )
        resized_w, resized_h = image.size
        scale_x = float(resized_w) / float(original_w)
        scale_y = float(resized_h) / float(original_h)
        image, target = transforms.ToTensor()(image, target)
        image, target = transforms.Normalize(
            self.contract["normalize_mean"], self.contract["normalize_std"]
        )(image, target)
        trace = {
            "schema": "stage-b-image-affine-trace-v1",
            "original_hw": [int(original_h), int(original_w)],
            "output_hw": [int(resized_h), int(resized_w)],
            "scale_xy": [scale_x, scale_y],
            "offset_xy": [0.0, 0.0],
            "operations": [
                {
                    "op": "aspect_preserving_resize",
                    "requested_short_side": requested,
                    "max_size": int(self.contract["max_size"]),
                    "before_hw": [int(original_h), int(original_w)],
                    "after_hw": [int(resized_h), int(resized_w)],
                    "ratio_xy": [scale_x, scale_y],
                }
            ],
        }
        trace["canonical_json"] = canonical_json(trace)
        trace["sha256"] = canonical_sha256(trace)
        target["_stageb_extraction_transform_trace"] = trace
        return image, target


def _model_output_tensors(outputs: Mapping[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_logits = outputs.get("pred_logits_text")
    if not torch.is_tensor(token_logits):
        token_logits = outputs.get("pred_logits")
    boxes = outputs.get("pred_boxes")
    phrase_mask = outputs.get("phrase_to_token_mask")
    if not torch.is_tensor(token_logits) or not torch.is_tensor(boxes) or not torch.is_tensor(phrase_mask):
        raise ExtractionError("model output lacks token logits, boxes, or generated phrase mask")
    return token_logits, boxes, phrase_mask


def _slice_output(outputs: Mapping[str, Any], start: int, end: int) -> Dict[str, torch.Tensor]:
    token_logits, boxes, phrase_mask = _model_output_tensors(outputs)
    return {
        "pred_logits_text": token_logits[start:end],
        "pred_boxes": boxes[start:end],
        "phrase_to_token_mask": phrase_mask[start:end],
    }


def forward_paired_pos_neg(
    model: Any,
    samples: Any,
    positive_captions: Sequence[str],
    negative_captions: Sequence[str],
    *,
    amp: bool,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    batch_size = len(positive_captions)
    if batch_size != len(negative_captions) or int(samples.tensors.shape[0]) != batch_size:
        raise ExtractionError("paired forward fields do not align")
    paired_samples = samples.__class__(
        torch.cat((samples.tensors, samples.tensors), dim=0),
        torch.cat((samples.mask, samples.mask), dim=0) if samples.mask is not None else None,
    )
    with torch.cuda.amp.autocast(enabled=bool(amp) and samples.tensors.device.type == "cuda"):
        outputs = model(
            paired_samples,
            captions=list(positive_captions) + list(negative_captions),
        )
    if int(_model_output_tensors(outputs)[0].shape[0]) != 2 * batch_size:
        raise ExtractionError("paired model output does not have 2B rows")
    return _slice_output(outputs, 0, batch_size), _slice_output(outputs, batch_size, 2 * batch_size)


def forward_separate(
    model: Any,
    samples: Any,
    captions: Sequence[str],
    *,
    amp: bool,
) -> Dict[str, torch.Tensor]:
    if int(samples.tensors.shape[0]) != len(captions):
        raise ExtractionError("separate forward fields do not align")
    with torch.cuda.amp.autocast(enabled=bool(amp) and samples.tensors.device.type == "cuda"):
        outputs = model(samples, captions=list(captions))
    return _slice_output(outputs, 0, len(captions))


def score_model_output(outputs: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    token_logits, boxes, phrase_mask = _model_output_tensors(outputs)
    scores = aggregate_full_expression_scores(token_logits, phrase_mask)
    boxes = boxes.detach().float()
    if tuple(boxes.shape) != (int(scores.shape[0]), EXPECTED_QUERIES, 4):
        raise ExtractionError(f"unexpected GDINO box shape: {tuple(boxes.shape)}")
    if not bool(torch.isfinite(boxes).all().item()):
        raise ExtractionError("GDINO boxes contain non-finite values")
    return scores, boxes


def _pad_nested_batch(samples: Any, targets: Sequence[Mapping[str, Any]], size: int = LOCAL_BATCH_SIZE):
    real = len(targets)
    if real <= 0 or real > size or int(samples.tensors.shape[0]) != real:
        raise ExtractionError("invalid real batch for cyclic padding")
    indices = [index % real for index in range(size)]
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=samples.tensors.device)
    padded = samples.__class__(
        samples.tensors.index_select(0, index_tensor),
        samples.mask.index_select(0, index_tensor) if samples.mask is not None else None,
    )
    return padded, [targets[index] for index in indices], indices, real


def _target_int(target: Mapping[str, Any], key: str) -> int:
    value = target.get(key)
    if torch.is_tensor(value) and value.numel() == 1:
        return int(value.detach().cpu().reshape(-1)[0].item())
    return int(value)


def _assert_batch_alignment(
    targets: Sequence[Mapping[str, Any]], bindings: Sequence[Mapping[str, Any]]
) -> None:
    if len(targets) != len(bindings):
        raise ExtractionError("dataset targets and semantic bindings do not align")
    for index, (target, binding) in enumerate(zip(targets, bindings)):
        pair = binding["pair"]
        for key in ("image_id", "ann_id", "ref_id", "sent_id"):
            if _target_int(target, key) != int(pair[key]):
                raise ExtractionError(f"batch identity mismatch for {key} at local row {index}")


def _source_verification_payload(source_row: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "target_bbox_used",
        "target_tight_crop_path",
        "target_context_crop_path",
        "target_context_boxed_crop_path",
        "visual_local_judgment",
        "proposal_cache",
        "visual_proposal_judgments",
        "proposal_num",
        "candidate_cache_version",
        "visual_filter_status",
        "visual_filter_reason",
        "tn_scope",
        "global_tn_verified",
    )
    return {key: source_row.get(key) for key in keys}


def _checkpoint_provenance(path: Path) -> Dict[str, Any]:
    from tools.stageb_gdino_adapter_probe_audit import ProbeAuditError, _validate_fixed_baseline
    from tools.stageb_fixed_protocol_audit import (
        ProtocolError,
        _verify_train_completion,
    )

    path = path.resolve()
    if not path.is_file():
        raise ExtractionError(
            f"completed authoritative fixed baseline is missing: {path}"
        )
    try:
        _verify_train_completion(path.parent, checkpoint=path)
        record = _validate_fixed_baseline(path)
    except (ProtocolError, ProbeAuditError, OSError, RuntimeError, ValueError) as error:
        raise ExtractionError(f"fixed baseline validation failed: {error}") from error
    base_sha = record.get("base_model_sha256")
    if not isinstance(base_sha, str) or len(base_sha) != 64:
        raise ExtractionError("completed baseline has no base-model tensor hash")
    return {
        "path": str(path.resolve()),
        "sha256": record["sha256"],
        "model_sha256": base_sha,
        "base_sha256": base_sha,
        "rank_sha256": None,
        "confidence_sha256": None,
        "protocol_train_complete": record["protocol_train_complete"],
        "size_bytes": int(record["size_bytes"]),
    }


def _config_provenance(path: Path, *, require_baseline: bool) -> Dict[str, Any]:
    from tools.stageb_dependency_audit import config_import_chain

    record = file_record(path)
    if require_baseline and record["sha256"] != BASELINE_CONFIG_SHA256:
        raise ExtractionError("model config is not the locked fixed-baseline config")
    chain = [file_record(item) for item in config_import_chain(path.resolve(), root=REPO_ROOT)]
    return {
        **record,
        "import_chain": chain,
        "import_chain_sha256": canonical_sha256(chain),
    }


def _code_provenance() -> Dict[str, Any]:
    import importlib

    from tools.stageb_dependency_audit import (
        DependencyAuditError,
        local_python_dependency_paths,
    )

    try:
        python_paths = local_python_dependency_paths(
            CODE_ENTRIES,
            root=REPO_ROOT,
            include=CODE_INCLUDE,
        )
    except DependencyAuditError as error:
        raise ExtractionError(f"could not resolve extraction code closure: {error}") from error
    records = [
        {"kind": "python", **file_record(path)} for path in python_paths
    ]
    records.extend(
        {
            "kind": "orchestration",
            **file_record(REPO_ROOT / relative),
        }
        for relative in CODE_ORCHESTRATION
    )
    try:
        extension = importlib.import_module(NATIVE_EXTENSION_MODULE)
    except Exception as error:
        raise ExtractionError(
            f"could not import required native extension {NATIVE_EXTENSION_MODULE}: {error}"
        ) from error
    extension_path = getattr(extension, "__file__", None)
    if not isinstance(extension_path, str) or not extension_path.strip():
        raise ExtractionError(
            f"native extension {NATIVE_EXTENSION_MODULE} has no import path"
        )
    records.append(
        {
            "kind": "native_extension",
            "module": NATIVE_EXTENSION_MODULE,
            **file_record(Path(extension_path)),
        }
    )
    return {"files": records, "code_sha256": canonical_sha256(records)}


def _image_path(binding: Mapping[str, Any], data_root: Path) -> Path:
    source_row = binding["source_row"]
    for key in ("image_path", "filename"):
        value = source_row.get(key)
        if isinstance(value, str) and value.strip() and Path(value).is_file():
            return Path(value).resolve()
    pair = binding["pair"]
    return (
        data_root.resolve()
        / "COCO/coco2014/train2014"
        / str(pair["file_name"])
    ).resolve()


def _row_transform_payload(
    static_contract: Mapping[str, Any], trace: Mapping[str, Any], *, amp: bool
) -> Dict[str, Any]:
    payload = {
        "image_set": str(static_contract.get("image_set", "train")),
        "fix_size": bool(static_contract.get("fix_size", False)),
        "hflip_prob": float(static_contract.get("hflip_prob", 0.0)),
        "output_hw": list(trace["output_hw"]),
        "amp": bool(amp),
        "dtype": "float16" if amp else "float32",
        "static_contract_sha256": static_contract["sha256"],
        "affine_trace": dict(trace),
    }
    payload["canonical_json"] = canonical_json(payload)
    payload["sha256"] = canonical_sha256(payload)
    return payload


def _manifest_row(
    context: Mapping[str, Any],
    *,
    checkpoint: Mapping[str, Any],
    model_config: Mapping[str, Any],
    data_config: Mapping[str, Any],
    code: Mapping[str, Any],
    transform_contract: Mapping[str, Any],
    deploy_transform_contract: Mapping[str, Any],
    holdout: Mapping[str, Any],
    image_record: Mapping[str, Any],
    regions: Sequence[Mapping[str, Any]],
    stability: Mapping[str, Any],
    amp: bool,
) -> Dict[str, Any]:
    binding = context["binding"]
    pair = binding["pair"]
    source_row = binding["source_row"]
    observations = context["observations"]
    primary = observations["primary"]["summary"]
    shadow = observations["shadow"]["summary"]
    positive_primary = context["positive_primary"]["summary"]
    positive_shadow = context["positive_shadow"]["summary"]
    deploy = observations["deploy"]["summary"]
    positive_deploy = context["positive_deploy"]["summary"]
    source_path = Path(binding["source_path"])
    trace = context["trace"]
    proposal_cache = []
    for rank, region in enumerate(regions):
        proposal_cache.append(
            {
                "proposal_id": region["region_id"],
                "bbox": region["bbox_xywh_original"],
                "score": max(float(value) for value in region["base_scores"].values()),
                "source_prompt": str(pair["try_tn"]),
                "prompt_rank": int(rank),
                "query_ids": region["query_ids"],
                "origins": region["origins"],
                "tight_crop_path": region["assets"]["tight"]["path"],
                "context_boxed_crop_path": region["assets"]["context_2x_boxed"]["path"],
                "full_boxed_path": region["assets"]["full_boxed"]["path"],
            }
        )
    return {
        "schema": SCHEMA,
        "sample_id": str(pair["sample_id"]),
        "dataset": str(pair["dataset"]),
        "image_id": int(pair["image_id"]),
        "ann_id": int(pair["ann_id"]),
        "ref_id": int(pair["ref_id"]),
        "sent_id": int(pair["sent_id"]),
        "split": str(pair["split"]),
        "source_pair": {
            "path": str(Path(binding["pair_path"]).resolve()),
            "line": int(binding["pair_line"]),
            "sha256": SEMANTIC_PAIRS_SHA256,
            "row_sha256": binding["pair_row_sha256"],
            "sample_id": str(pair["sample_id"]),
        },
        "source_verified_row": {
            "path": str(source_path.resolve()),
            "sha256": str(binding["source_file_sha256"]),
            "line": int(binding["source_line"]),
            "row_sha256": str(pair["source_row_sha256"]),
        },
        "positive_expression": str(pair["sent"]),
        "negative_expression": str(pair["try_tn"]),
        "positive_caption_model": str(context["positive_caption"]),
        "negative_caption_model": str(context["negative_caption"]),
        "checkpoint": dict(checkpoint),
        "config": dict(model_config),
        "data_config": dict(data_config),
        "code_sha256": str(code["code_sha256"]),
        "transform": _row_transform_payload(transform_contract, trace, amp=amp),
        "deploy_transform": _row_transform_payload(
            deploy_transform_contract, context["deploy_trace"], amp=amp
        ),
        "image": dict(image_record),
        "target_bbox_used": [float(value) for value in pair["target_bbox_used"]],
        "score_contract": SCORE_CONTRACT,
        "forward_contract": {
            "primary": PRIMARY_FORWARD_CONTRACT,
            "shadow": SHADOW_FORWARD_CONTRACT,
            "deploy": DEPLOY_FORWARD_CONTRACT,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "paired_batch_size": PAIRED_BATCH_SIZE,
            "deploy_batch_size": DEPLOY_BATCH_SIZE,
        },
        "num_queries": EXPECTED_QUERIES,
        "valid_query_count": EXPECTED_QUERIES,
        "primary": dict(primary),
        "shadow": dict(shadow),
        "deploy": dict(deploy),
        "positive_primary": dict(positive_primary),
        "positive_shadow": dict(positive_shadow),
        "positive_deploy": dict(positive_deploy),
        "stability": dict(stability),
        "regions": [dict(region) for region in regions],
        "proposal_num": len(proposal_cache),
        "proposal_cache": proposal_cache,
        "visual_local_judgment": {"status": "pending", "cache_key": None},
        "visual_proposal_judgments": [],
        "visual_filter_status": "pending",
        "source_verification": _source_verification_payload(source_row),
        "holdout": dict(holdout),
        "claims": dict(CLAIMS),
    }


def validate_manifest_row(row: Mapping[str, Any], *, excluded_images: Optional[set[int]] = None) -> None:
    context = str(row.get("sample_id", "<missing>"))
    if row.get("schema") != SCHEMA or row.get("claims") != CLAIMS:
        raise ExtractionError(f"manifest schema/claims mismatch at {context}")
    if int(row.get("num_queries", -1)) != EXPECTED_QUERIES or int(
        row.get("valid_query_count", -1)
    ) != EXPECTED_QUERIES:
        raise ExtractionError(f"manifest query count mismatch at {context}")
    if excluded_images is not None and int(row["image_id"]) in excluded_images:
        raise ExtractionError(f"strict image leakage at {context}")
    if row.get("score_contract") != SCORE_CONTRACT:
        raise ExtractionError(f"score contract mismatch at {context}")
    expected_forward = {
        "primary": PRIMARY_FORWARD_CONTRACT,
        "shadow": SHADOW_FORWARD_CONTRACT,
        "deploy": DEPLOY_FORWARD_CONTRACT,
        "local_batch_size": LOCAL_BATCH_SIZE,
        "paired_batch_size": PAIRED_BATCH_SIZE,
        "deploy_batch_size": DEPLOY_BATCH_SIZE,
    }
    if row.get("forward_contract") != expected_forward:
        raise ExtractionError(f"forward contract mismatch at {context}")
    for transform_key in ("transform", "deploy_transform"):
        transform = row.get(transform_key)
        if not isinstance(transform, Mapping) or not isinstance(
            transform.get("sha256"), str
        ) or len(str(transform["sha256"])) != 64:
            raise ExtractionError(f"{transform_key} provenance is missing at {context}")
        payload = dict(transform)
        observed_sha = str(payload.pop("sha256"))
        if canonical_sha256(payload) != observed_sha:
            raise ExtractionError(f"{transform_key} canonical hash drifted at {context}")
    query_ids: set[int] = set()
    for query_key in (
        "primary",
        "shadow",
        "deploy",
        "positive_primary",
        "positive_shadow",
        "positive_deploy",
    ):
        query = row.get(query_key)
        if not isinstance(query, Mapping):
            raise ExtractionError(f"{query_key} observation is missing at {context}")
        try:
            query_id = int(query["query_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise ExtractionError(f"{query_key} query ID is invalid at {context}") from error
        if not 0 <= query_id < EXPECTED_QUERIES:
            raise ExtractionError(f"{query_key} query ID is out of range at {context}")
        if query_key in {"primary", "shadow", "deploy"}:
            query_ids.add(query_id)
    regions = row.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ExtractionError(f"manifest has no review regions at {context}")
    if int(row.get("proposal_num", -1)) != len(regions):
        raise ExtractionError(f"proposal/region count mismatch at {context}")
    region_ids = [str(region.get("region_id", "")) for region in regions]
    if any(not value for value in region_ids) or len(set(region_ids)) != len(region_ids):
        raise ExtractionError(f"missing or duplicate region ID at {context}")
    for region in regions:
        if region.get("judgment") != {"status": "pending", "cache_key": None}:
            raise ExtractionError(f"extraction region is not pending at {context}")
        assets = region.get("assets")
        if not isinstance(assets, Mapping):
            raise ExtractionError(f"region assets are missing at {context}")
        for key in ("tight", "context_2x_boxed", "full_boxed"):
            asset = assets.get(key)
            if not isinstance(asset, Mapping) or not asset.get("path") or not asset.get("sha256"):
                raise ExtractionError(f"region asset {key} is missing at {context}")
        region_queries = region.get("query_ids")
        if not isinstance(region_queries, list) or not region_queries:
            raise ExtractionError(f"region query IDs are missing at {context}")
        query_ids.difference_update(int(value) for value in region_queries)
    stability = row.get("stability")
    if (
        not isinstance(stability, Mapping)
        or "primary_shadow_agree" not in stability
        or "primary_deploy_agree" not in stability
    ):
        raise ExtractionError(f"stability record is missing at {context}")
    if not math.isclose(
        float(stability.get("epsilon", -1.0)),
        DEFAULT_TIE_EPSILON,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ExtractionError(f"near-tie epsilon drifted at {context}")
    query_ids.update(int(value) for value in stability.get("near_tie_query_ids", []))
    union_query_ids = {
        int(value)
        for region in regions
        for value in region.get("query_ids", [])
    }
    query_ids.update(
        int(value)
        for value in stability.get("query_ids_by_origin", {}).values()
    )
    missing_union = query_ids.difference(union_query_ids)
    if missing_union:
        raise ExtractionError(
            f"required query IDs are absent from region union at {context}: {sorted(missing_union)}"
        )
    holdout = row.get("holdout")
    if not isinstance(holdout, Mapping) or holdout.get("image_disjoint") is not True:
        raise ExtractionError(f"holdout provenance is missing at {context}")


def _make_dataset(
    data_cfg: Any,
    annotation: Path,
    data_root: Path,
    transform_contract: Mapping[str, Any],
    *,
    deploy: bool = False,
):
    from datasets import build_dataset

    datasetinfo = {
        "name": "fixed_gdino_semantic_top1_extraction",
        "dataset_mode": "patch_episode",
        "source": "sam3_tn_pair",
        "root": str(data_root / "COCO/coco2014/train2014"),
        "sam3_tn_image_root": str(data_root / "COCO/coco2014/train2014"),
        "anno": str(annotation),
        "box_format": "xywh",
        "sam3_tn_bbox_key": "target_bbox_used",
        "canonical_classes_json": str(data_root / "canonical_classes_with_aliases.json"),
        "keep_only_support_gt": True,
        "neg_episode_prob": 0.0,
        "support_min_count": 1,
        "support_patch_size": 224,
        "build_text_token_masks": True,
        "text_mask_warn_limit": 0,
        "text_mask_skip_invalid_canonical": False,
        "tn_balance_sampling": False,
        "require_global_tn_verified": True,
        "stage_b_gdino_adapter_no_support": True,
        "anno_cache": False,
        "anno_cache_write": False,
    }
    image_set = "val" if deploy else "train"
    dataset = build_dataset(image_set=image_set, args=data_cfg, datasetinfo=datasetinfo)
    dataset.transforms = (
        TraceableDeployEvalTransform(transform_contract)
        if deploy
        else TraceableConfidenceTrainTransform(transform_contract)
    )
    return dataset


def _collate_pilot(items: Sequence[Mapping[str, Any]], device: torch.device):
    from util.misc import nested_tensor_from_tensor_list

    samples = nested_tensor_from_tensor_list([item["tensor"] for item in items]).to(device)
    targets = [item["target"] for item in items]
    return samples, targets


def _attach_pilot_observation(
    contexts: Sequence[MutableMapping[str, Any]],
    pilot_items: Sequence[Mapping[str, Any]],
    model: Any,
    device: torch.device,
    *,
    amp: bool,
) -> None:
    # Pair-2 singleton isolates all companion/padding effects.
    for item in pilot_items:
        samples, _ = _collate_pilot([item], device)
        pos_out, neg_out = forward_paired_pos_neg(
            model, samples, [item["positive_caption"]], [item["negative_caption"]], amp=amp
        )
        neg_scores, neg_boxes = score_model_output(neg_out)
        context = contexts[int(item["context_index"])]
        context["observations"]["singleton"] = make_query_observation(
            neg_scores[0], neg_boxes[0], trace=context["trace"],
            target_bbox_xywh=context["binding"]["pair"]["target_bbox_used"],
            tie_epsilon=context["tie_epsilon"],
        )
        del pos_out, neg_out, neg_scores, neg_boxes, samples

    # A fixed even/odd permutation changes batch companions while remaining replayable.
    order = list(range(0, len(pilot_items), 2)) + list(range(1, len(pilot_items), 2))
    for start in range(0, len(order), LOCAL_BATCH_SIZE):
        real_indices = order[start : start + LOCAL_BATCH_SIZE]
        if not real_indices:
            continue
        padded_indices = [real_indices[index % len(real_indices)] for index in range(LOCAL_BATCH_SIZE)]
        group = [pilot_items[index] for index in padded_indices]
        samples, _ = _collate_pilot(group, device)
        pos_out, neg_out = forward_paired_pos_neg(
            model,
            samples,
            [item["positive_caption"] for item in group],
            [item["negative_caption"] for item in group],
            amp=amp,
        )
        neg_scores, neg_boxes = score_model_output(neg_out)
        for local_index, pilot_index in enumerate(real_indices):
            item = pilot_items[pilot_index]
            context = contexts[int(item["context_index"])]
            context["observations"]["rebatch"] = make_query_observation(
                neg_scores[local_index], neg_boxes[local_index], trace=context["trace"],
                target_bbox_xywh=context["binding"]["pair"]["target_bbox_used"],
                tie_epsilon=context["tie_epsilon"],
            )
        del pos_out, neg_out, neg_scores, neg_boxes, samples


def _write_filtered_annotation(path: Path, bindings: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for binding in bindings:
            handle.write(canonical_json(binding["pair"]) + "\n")


def _prepare_locked_inputs(args: argparse.Namespace) -> Dict[str, Any]:
    checkpoint = _checkpoint_provenance(Path(args.checkpoint))
    strict_records, excluded_images = validate_strict_manifests(
        {"strict2031": Path(args.strict2031), "strict1607": Path(args.strict1607)}
    )
    bindings, semantic = validate_semantic_inputs(Path(args.semantic_pairs), Path(args.semantic_audit))
    kept, exclusion = exclude_holdout_image_union(bindings, excluded_images)
    model_config = _config_provenance(Path(args.model_config), require_baseline=True)
    data_config = _config_provenance(Path(args.data_config), require_baseline=False)
    code = _code_provenance()
    return {
        "checkpoint": checkpoint,
        "strict": strict_records,
        "excluded_images": excluded_images,
        "semantic": semantic,
        "bindings": kept,
        "exclusion": exclusion,
        "model_config": model_config,
        "data_config": data_config,
        "code": code,
    }


def extract(args: argparse.Namespace) -> Dict[str, Any]:
    prepared = _prepare_locked_inputs(args)
    output = Path(args.output).resolve()
    audit_path = Path(args.audit).resolve()
    assets_root = Path(args.assets_dir).resolve()
    for path, label in ((output, "output"), (audit_path, "audit"), (assets_root, "assets")):
        if path.exists():
            raise ExtractionError(f"refusing to overwrite existing {label}: {path}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ExtractionError("CUDA extraction requested but CUDA is unavailable")
    if int(args.seed) != DEFAULT_SEED:
        raise ExtractionError(f"the frozen extraction seed must remain {DEFAULT_SEED}")
    if int(args.stability_pilot_rows) < 0:
        raise ExtractionError("stability pilot row count must be non-negative")
    if not math.isclose(
        float(args.tie_epsilon), DEFAULT_TIE_EPSILON, rel_tol=0.0, abs_tol=0.0
    ):
        raise ExtractionError(
            f"the frozen near-tie union epsilon must remain {DEFAULT_TIE_EPSILON:g}"
        )

    from torch.utils.data import DataLoader, SequentialSampler
    from tools.eval_refcoco_stageb import _load_model, _seed_worker
    from tools.stageb_eval_records import extract_adapter_tn_pair_captions
    from util import misc as utils
    from util.slconfig import SLConfig

    random.seed(DEFAULT_SEED)
    torch.manual_seed(DEFAULT_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(DEFAULT_SEED)
    data_cfg = SLConfig.fromfile(str(Path(args.data_config).resolve()))
    model_cfg = SLConfig.fromfile(str(Path(args.model_config).resolve()))
    transform_contract = transform_contract_from_cfg(data_cfg)
    deploy_transform_contract = deploy_transform_contract_from_cfg(model_cfg)
    amp = True
    contexts: List[MutableMapping[str, Any]] = []
    pilot_items: List[Dict[str, Any]] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    assets_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_assets = assets_root.with_name(f".{assets_root.name}.tmp-{os.getpid()}")
    temporary_output = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary_assets.exists() or temporary_output.exists():
        raise ExtractionError("stale extraction temporary path exists")

    with tempfile.TemporaryDirectory(prefix="stageb-top1-input-", dir=str(output.parent)) as temporary:
        annotation = Path(temporary) / "filtered_semantic_pairs.jsonl"
        _write_filtered_annotation(annotation, prepared["bindings"])
        dataset = _make_dataset(data_cfg, annotation, Path(args.data_root).resolve(), transform_contract)
        if len(dataset) != FILTERED_ROWS:
            raise ExtractionError(f"filtered dataset normalized to {len(dataset)} rows, expected {FILTERED_ROWS}")
        generator = torch.Generator()
        generator.manual_seed(DEFAULT_SEED)
        loader = DataLoader(
            dataset,
            batch_size=LOCAL_BATCH_SIZE,
            sampler=SequentialSampler(dataset),
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=0,
            pin_memory=device.type == "cuda",
            worker_init_fn=_seed_worker,
            generator=generator,
        )
        model = _load_model(model_cfg, str(Path(args.checkpoint).resolve()), device)
        offset = 0
        with torch.inference_mode():
            for batch_index, (samples, raw_targets) in enumerate(loader):
                raw_targets = list(raw_targets)
                real = len(raw_targets)
                batch_bindings = prepared["bindings"][offset : offset + real]
                _assert_batch_alignment(raw_targets, batch_bindings)
                samples, padded_targets, indices, real = _pad_nested_batch(samples, raw_targets)
                positive, negative, valid = extract_adapter_tn_pair_captions(padded_targets)
                if not bool(valid.all().item()):
                    raise ExtractionError("paired caption extraction produced an invalid row")
                samples = samples.to(device)
                pos_primary_out, neg_primary_out = forward_paired_pos_neg(
                    model, samples, positive, negative, amp=amp
                )
                # Match formal evaluator call order: negative first, positive second.
                neg_shadow_out = forward_separate(model, samples, negative, amp=amp)
                pos_shadow_out = forward_separate(model, samples, positive, amp=amp)
                pos_primary_scores, pos_primary_boxes = score_model_output(pos_primary_out)
                neg_primary_scores, neg_primary_boxes = score_model_output(neg_primary_out)
                neg_shadow_scores, neg_shadow_boxes = score_model_output(neg_shadow_out)
                pos_shadow_scores, pos_shadow_boxes = score_model_output(pos_shadow_out)
                for local_index in range(real):
                    binding = batch_bindings[local_index]
                    target = raw_targets[local_index]
                    trace = target.get("_stageb_extraction_transform_trace")
                    if not isinstance(trace, Mapping):
                        raise ExtractionError("dataset target has no invertible transform trace")
                    target_bbox = binding["pair"]["target_bbox_used"]
                    context: MutableMapping[str, Any] = {
                        "binding": binding,
                        "trace": dict(trace),
                        "positive_caption": positive[local_index],
                        "negative_caption": negative[local_index],
                        "tie_epsilon": float(args.tie_epsilon),
                        "observations": {
                            "primary": make_query_observation(
                                neg_primary_scores[local_index], neg_primary_boxes[local_index],
                                trace=trace, target_bbox_xywh=target_bbox,
                                tie_epsilon=float(args.tie_epsilon),
                            ),
                            "shadow": make_query_observation(
                                neg_shadow_scores[local_index], neg_shadow_boxes[local_index],
                                trace=trace, target_bbox_xywh=target_bbox,
                                tie_epsilon=float(args.tie_epsilon),
                            ),
                        },
                        "positive_primary": make_query_observation(
                            pos_primary_scores[local_index], pos_primary_boxes[local_index],
                            trace=trace, target_bbox_xywh=target_bbox,
                            tie_epsilon=float(args.tie_epsilon),
                        ),
                        "positive_shadow": make_query_observation(
                            pos_shadow_scores[local_index], pos_shadow_boxes[local_index],
                            trace=trace, target_bbox_xywh=target_bbox,
                            tie_epsilon=float(args.tie_epsilon),
                        ),
                    }
                    context_index = len(contexts)
                    contexts.append(context)
                    if context_index < int(args.stability_pilot_rows):
                        output_h, output_w = [int(value) for value in trace["output_hw"]]
                        pilot_items.append(
                            {
                                "context_index": context_index,
                                "tensor": samples.tensors[local_index, :, :output_h, :output_w].detach().cpu().clone(),
                                "target": target,
                                "positive_caption": positive[local_index],
                                "negative_caption": negative[local_index],
                            }
                        )
                offset += real
                del (
                    pos_primary_out, neg_primary_out, neg_shadow_out, pos_shadow_out,
                    pos_primary_scores, pos_primary_boxes, neg_primary_scores, neg_primary_boxes,
                    neg_shadow_scores, neg_shadow_boxes, pos_shadow_scores, pos_shadow_boxes, samples,
                )
                if device.type == "cuda" and batch_index % 50 == 0:
                    torch.cuda.empty_cache()
                if int(args.log_every) > 0 and (batch_index + 1) % int(args.log_every) == 0:
                    print(f"[extract] batches={batch_index + 1} rows={offset}", flush=True)
            if offset != FILTERED_ROWS or len(contexts) != FILTERED_ROWS:
                raise ExtractionError(f"extraction row count drifted: {offset}")
            if pilot_items:
                _attach_pilot_observation(contexts, pilot_items, model, device, amp=amp)

            # Full formal-deploy pass: val geometry, B16, negative then positive.
            random.seed(DEFAULT_SEED)
            torch.manual_seed(DEFAULT_SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(DEFAULT_SEED)
            deploy_dataset = _make_dataset(
                data_cfg,
                annotation,
                Path(args.data_root).resolve(),
                deploy_transform_contract,
                deploy=True,
            )
            if len(deploy_dataset) != FILTERED_ROWS:
                raise ExtractionError(
                    f"deploy dataset normalized to {len(deploy_dataset)} rows, expected {FILTERED_ROWS}"
                )
            deploy_generator = torch.Generator()
            deploy_generator.manual_seed(DEFAULT_SEED)
            deploy_loader = DataLoader(
                deploy_dataset,
                batch_size=DEPLOY_BATCH_SIZE,
                sampler=SequentialSampler(deploy_dataset),
                drop_last=False,
                collate_fn=utils.collate_fn,
                num_workers=0,
                pin_memory=device.type == "cuda",
                worker_init_fn=_seed_worker,
                generator=deploy_generator,
            )
            deploy_offset = 0
            for deploy_batch_index, (deploy_samples, deploy_targets) in enumerate(
                deploy_loader
            ):
                deploy_targets = list(deploy_targets)
                deploy_real = len(deploy_targets)
                deploy_bindings = prepared["bindings"][
                    deploy_offset : deploy_offset + deploy_real
                ]
                _assert_batch_alignment(deploy_targets, deploy_bindings)
                deploy_positive, deploy_negative, deploy_valid = (
                    extract_adapter_tn_pair_captions(deploy_targets)
                )
                if not bool(deploy_valid.all().item()):
                    raise ExtractionError("deploy caption extraction produced an invalid row")
                deploy_samples = deploy_samples.to(device)
                deploy_neg_out = forward_separate(
                    model, deploy_samples, deploy_negative, amp=amp
                )
                deploy_pos_out = forward_separate(
                    model, deploy_samples, deploy_positive, amp=amp
                )
                deploy_neg_scores, deploy_neg_boxes = score_model_output(deploy_neg_out)
                deploy_pos_scores, deploy_pos_boxes = score_model_output(deploy_pos_out)
                for local_index in range(deploy_real):
                    context = contexts[deploy_offset + local_index]
                    target = deploy_targets[local_index]
                    trace = target.get("_stageb_extraction_transform_trace")
                    if not isinstance(trace, Mapping):
                        raise ExtractionError("deploy target has no invertible transform trace")
                    target_bbox = context["binding"]["pair"]["target_bbox_used"]
                    context["deploy_trace"] = dict(trace)
                    context["observations"]["deploy"] = make_query_observation(
                        deploy_neg_scores[local_index],
                        deploy_neg_boxes[local_index],
                        trace=trace,
                        target_bbox_xywh=target_bbox,
                        tie_epsilon=float(args.tie_epsilon),
                    )
                    context["positive_deploy"] = make_query_observation(
                        deploy_pos_scores[local_index],
                        deploy_pos_boxes[local_index],
                        trace=trace,
                        target_bbox_xywh=target_bbox,
                        tie_epsilon=float(args.tie_epsilon),
                    )
                deploy_offset += deploy_real
                del (
                    deploy_neg_out,
                    deploy_pos_out,
                    deploy_neg_scores,
                    deploy_neg_boxes,
                    deploy_pos_scores,
                    deploy_pos_boxes,
                    deploy_samples,
                )
                if device.type == "cuda" and deploy_batch_index % 50 == 0:
                    torch.cuda.empty_cache()
                if (
                    int(args.log_every) > 0
                    and (deploy_batch_index + 1) % int(args.log_every) == 0
                ):
                    print(
                        f"[deploy] batches={deploy_batch_index + 1} rows={deploy_offset}",
                        flush=True,
                    )
            if deploy_offset != FILTERED_ROWS:
                raise ExtractionError(f"deploy extraction row count drifted: {deploy_offset}")
        del model, dataset, loader, deploy_dataset, deploy_loader

    temporary_assets.mkdir(parents=True, exist_ok=False)
    image_hash_cache: Dict[Path, str] = {}
    counts: Counter[str] = Counter()
    distributions: Dict[str, Counter[str]] = {
        "primary_margin": Counter(),
        "max_abs_score_drift": Counter(),
        "near_tie_count": Counter(),
        "union_query_count": Counter(),
        "region_count": Counter(),
    }
    holdout_row = {
        "strict2031_manifest_sha256": prepared["strict"]["strict2031"]["sha256"],
        "strict1607_manifest_sha256": prepared["strict"]["strict1607"]["sha256"],
        "image_disjoint": True,
        "exclusion_policy": "strict2031_strict1607_image_id_union",
    }
    try:
        with temporary_output.open("w", encoding="utf-8") as handle:
            for row_index, context in enumerate(contexts):
                stability = summarize_stability(context["observations"], epsilon=float(args.tie_epsilon))
                if stability["exact_top1_stable"]:
                    counts["exact_top1_stable"] += 1
                if not stability["primary_shadow_agree"]:
                    counts["primary_shadow_disagree"] += 1
                if not stability["primary_deploy_agree"]:
                    counts["primary_deploy_disagree"] += 1
                regions = build_regions(
                    str(context["binding"]["pair"]["sample_id"]),
                    context["observations"],
                    context["binding"]["source_row"],
                )
                distributions["primary_margin"][_numeric_bucket(context["observations"]["primary"]["summary"]["margin"])] += 1
                distributions["max_abs_score_drift"][_numeric_bucket(stability["max_abs_score_drift"])] += 1
                distributions["near_tie_count"][str(int(stability["near_tie_count"]))] += 1
                union_query_count = len(
                    {
                        int(query_id)
                        for region in regions
                        for query_id in region["query_ids"]
                    }
                )
                distributions["union_query_count"][str(union_query_count)] += 1
                distributions["region_count"][str(len(regions))] += 1
                image_path = _image_path(context["binding"], Path(args.data_root))
                if not image_path.is_file():
                    raise ExtractionError(f"source image is missing: {image_path}")
                image_sha = image_hash_cache.get(image_path)
                if image_sha is None:
                    image_sha = sha256_file(image_path)
                    image_hash_cache[image_path] = image_sha
                with Image.open(image_path) as source_image:
                    source_image = source_image.convert("RGB")
                    width, height = source_image.size
                    for trace_label, trace_value in (
                        ("train", context["trace"]),
                        ("deploy", context["deploy_trace"]),
                    ):
                        if list(trace_value.get("original_hw", [])) != [height, width]:
                            raise ExtractionError(
                                f"{trace_label} transform original size drifted for {image_path}"
                            )
                    sample_key = canonical_sha256(context["binding"]["pair"]["sample_id"])[:20]
                    for region in regions:
                        region["assets"] = generate_region_assets(
                            source_image,
                            region,
                            temporary_root=temporary_assets,
                            final_root=assets_root,
                            sample_key=sample_key,
                        )
                image_record = {
                    "path": str(image_path),
                    "width": int(width),
                    "height": int(height),
                    "sha256": image_sha,
                }
                row = _manifest_row(
                    context,
                    checkpoint=prepared["checkpoint"],
                    model_config=prepared["model_config"],
                    data_config=prepared["data_config"],
                    code=prepared["code"],
                    transform_contract=transform_contract,
                    deploy_transform_contract=deploy_transform_contract,
                    holdout=holdout_row,
                    image_record=image_record,
                    regions=regions,
                    stability=stability,
                    amp=amp,
                )
                validate_manifest_row(row, excluded_images=prepared["excluded_images"])
                handle.write(canonical_json(row) + "\n")
                counts["regions"] += len(regions)
                counts["rows"] += 1
                counts["inherit_eligible_regions"] += sum(
                    1 for region in regions if region["inherit_eligible"]
                )
                if int(args.log_every) > 0 and (row_index + 1) % (int(args.log_every) * LOCAL_BATCH_SIZE) == 0:
                    print(f"[assets] rows={row_index + 1} regions={counts['regions']}", flush=True)
        if counts["rows"] != FILTERED_ROWS:
            raise ExtractionError("serialized manifest row count drifted")
        output_sha = sha256_file(temporary_output)
        os.replace(temporary_assets, assets_root)
        os.replace(temporary_output, output)
    except Exception:
        if temporary_assets.exists():
            shutil.rmtree(temporary_assets)
        if temporary_output.exists():
            temporary_output.unlink()
        raise

    audit = {
        "schema": AUDIT_SCHEMA,
        "kind": "completed_fixed_gdino_top1_vlm_extraction",
        "rows": int(counts["rows"]),
        "regions": int(counts["regions"]),
        "counts": dict(sorted(counts.items())),
        "stability_distributions": {
            name: dict(sorted(counter.items()))
            for name, counter in sorted(distributions.items())
        },
        "output": {"path": str(output), "sha256": output_sha, "size_bytes": int(output.stat().st_size)},
        "manifest": {
            "path": str(output),
            "sha256": output_sha,
            "size_bytes": int(output.stat().st_size),
            "rows": int(counts["rows"]),
        },
        "assets": {
            "path": str(assets_root),
            "policy": ASSET_POLICY,
            "policy_sha256": canonical_sha256(ASSET_POLICY),
        },
        "checkpoint": prepared["checkpoint"],
        "model_config": prepared["model_config"],
        "data_config": prepared["data_config"],
        "code": prepared["code"],
        "transform_contracts": {
            "train": transform_contract,
            "deploy": deploy_transform_contract,
        },
        "semantic_source": prepared["semantic"],
        "holdout": {"manifests": prepared["strict"], "image_union_size": len(prepared["excluded_images"]), **prepared["exclusion"]},
        "runtime": {
            "device": str(device),
            "seed": DEFAULT_SEED,
            "amp": True,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "paired_batch_size": PAIRED_BATCH_SIZE,
            "deploy_batch_size": DEPLOY_BATCH_SIZE,
            "num_workers": 0,
            "stability_pilot_rows": len(pilot_items),
            "tie_epsilon": DEFAULT_TIE_EPSILON,
            "primary_forward_contract": PRIMARY_FORWARD_CONTRACT,
            "shadow_forward_contract": SHADOW_FORWARD_CONTRACT,
            "deploy_forward_contract": DEPLOY_FORWARD_CONTRACT,
            "score_contract": SCORE_CONTRACT,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "pillow_version": PIL_VERSION,
        },
        "claims": dict(CLAIMS),
    }
    _atomic_write_json(audit_path, audit)
    return audit


def verify_output(args: argparse.Namespace) -> Dict[str, Any]:
    audit_path = Path(args.audit).resolve()
    audit = read_json(audit_path)
    if audit.get("schema") != AUDIT_SCHEMA or audit.get("kind") != "completed_fixed_gdino_top1_vlm_extraction":
        raise ExtractionError("extraction audit schema/kind mismatch")
    if audit.get("claims") != CLAIMS:
        raise ExtractionError("extraction audit claims drifted")
    prepared = _prepare_locked_inputs(args)
    output = Path(args.output).resolve()
    output_record = audit.get("output")
    if not isinstance(output_record, Mapping) or output_record.get("path") != str(output):
        raise ExtractionError("audit output path mismatch")
    if sha256_file(output) != output_record.get("sha256"):
        raise ExtractionError("extraction manifest hash drifted")
    manifest_record = audit.get("manifest")
    expected_manifest_record = {
        "path": str(output),
        "sha256": output_record.get("sha256"),
        "size_bytes": int(output.stat().st_size),
        "rows": FILTERED_ROWS,
    }
    if manifest_record != expected_manifest_record or int(audit.get("rows", -1)) != FILTERED_ROWS:
        raise ExtractionError("extraction manifest completeness record drifted")
    if audit.get("checkpoint") != prepared["checkpoint"]:
        raise ExtractionError("extraction checkpoint provenance drifted")
    if audit.get("model_config") != prepared["model_config"] or audit.get("data_config") != prepared["data_config"]:
        raise ExtractionError("extraction config provenance drifted")
    if audit.get("code") != prepared["code"]:
        raise ExtractionError("extraction code provenance drifted")
    from util.slconfig import SLConfig

    current_train_transform = transform_contract_from_cfg(
        SLConfig.fromfile(str(Path(args.data_config).resolve()))
    )
    current_deploy_transform = deploy_transform_contract_from_cfg(
        SLConfig.fromfile(str(Path(args.model_config).resolve()))
    )
    if audit.get("transform_contracts") != {
        "train": current_train_transform,
        "deploy": current_deploy_transform,
    }:
        raise ExtractionError("extraction transform provenance drifted")
    identities: set[Tuple[str, int, int, int, int]] = set()
    binding_by_sample = {
        str(binding["pair"]["sample_id"]): binding for binding in prepared["bindings"]
    }
    rows = 0
    regions = 0
    for line_number, row in iter_jsonl(output):
        validate_manifest_row(row, excluded_images=prepared["excluded_images"])
        identity = _identity(row, context=f"{output}:{line_number}")
        if identity in identities:
            raise ExtractionError(f"duplicate extraction identity at {output}:{line_number}")
        identities.add(identity)
        binding = binding_by_sample.get(str(row["sample_id"]))
        if binding is None or _identity(binding["pair"], context="binding") != identity:
            raise ExtractionError(f"manifest source-pair binding drifted at {output}:{line_number}")
        if row.get("source_pair", {}).get("row_sha256") != binding["pair_row_sha256"]:
            raise ExtractionError(f"manifest source-pair hash drifted at {output}:{line_number}")
        image = row.get("image")
        if not isinstance(image, Mapping) or sha256_file(Path(str(image["path"]))) != image.get("sha256"):
            raise ExtractionError(f"manifest source image hash drifted at {output}:{line_number}")
        for region in row["regions"]:
            for asset_key in ("tight", "context_2x_boxed", "full_boxed"):
                asset = region["assets"][asset_key]
                if sha256_file(Path(str(asset["path"]))) != asset["sha256"]:
                    raise ExtractionError(f"asset hash drifted at {output}:{line_number}:{asset_key}")
        rows += 1
        regions += len(row["regions"])
    if rows != FILTERED_ROWS or rows != int(audit.get("rows", -1)):
        raise ExtractionError(f"verified manifest row count drifted: {rows}")
    if regions != int(audit.get("regions", -1)):
        raise ExtractionError(f"verified manifest region count drifted: {regions}")
    return {
        "schema": AUDIT_SCHEMA,
        "verified": True,
        "rows": rows,
        "regions": regions,
        "output_sha256": output_record["sha256"],
    }


def dry_run(args: argparse.Namespace) -> Dict[str, Any]:
    if not math.isclose(
        float(args.tie_epsilon), DEFAULT_TIE_EPSILON, rel_tol=0.0, abs_tol=0.0
    ):
        raise ExtractionError(
            f"the frozen near-tie union epsilon must remain {DEFAULT_TIE_EPSILON:g}"
        )
    prepared = _prepare_locked_inputs(args)
    from util.slconfig import SLConfig

    data_cfg = SLConfig.fromfile(str(Path(args.data_config).resolve()))
    model_cfg = SLConfig.fromfile(str(Path(args.model_config).resolve()))
    transform_contract = transform_contract_from_cfg(data_cfg)
    deploy_transform_contract = deploy_transform_contract_from_cfg(model_cfg)
    return {
        "schema": AUDIT_SCHEMA,
        "kind": "dry_run_no_model_or_gpu",
        "checkpoint": prepared["checkpoint"],
        "semantic_source": prepared["semantic"],
        "holdout": {
            "manifests": prepared["strict"],
            "image_union_size": len(prepared["excluded_images"]),
            **prepared["exclusion"],
        },
        "model_config": prepared["model_config"],
        "data_config": prepared["data_config"],
        "code": prepared["code"],
        "transform_contracts": {
            "train": transform_contract,
            "deploy": deploy_transform_contract,
        },
        "planned_runtime": {
            "primary": PRIMARY_FORWARD_CONTRACT,
            "shadow": SHADOW_FORWARD_CONTRACT,
            "deploy": DEPLOY_FORWARD_CONTRACT,
            "score": SCORE_CONTRACT,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "paired_batch_size": PAIRED_BATCH_SIZE,
            "deploy_batch_size": DEPLOY_BATCH_SIZE,
            "stability_pilot_rows": int(args.stability_pilot_rows),
            "tie_epsilon": DEFAULT_TIE_EPSILON,
        },
        "claims": dict(CLAIMS),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-config", type=Path, default=BASELINE_CONFIG)
    parser.add_argument("--data-config", type=Path, default=DATA_CONFIG)
    parser.add_argument("--semantic-pairs", type=Path, default=SEMANTIC_PAIRS)
    parser.add_argument("--semantic-audit", type=Path, default=SEMANTIC_AUDIT)
    parser.add_argument("--strict2031", type=Path, default=STRICT_SPECS["strict2031"]["path"])
    parser.add_argument("--strict1607", type=Path, default=STRICT_SPECS["strict1607"]["path"])
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "/home/user/datasets/pivot_data")),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "manifest.jsonl")
    parser.add_argument("--audit", type=Path, default=DEFAULT_OUTPUT_ROOT / "audit.json")
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "assets")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tie-epsilon", type=float, default=DEFAULT_TIE_EPSILON)
    parser.add_argument("--stability-pilot-rows", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    try:
        if args.verify_only:
            result = verify_output(args)
        elif args.dry_run:
            result = dry_run(args)
        else:
            result = extract(args)
    except ExtractionError as error:
        raise SystemExit(f"[ERROR] {error}") from error
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
