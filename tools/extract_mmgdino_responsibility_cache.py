#!/usr/bin/env python3
"""Extract frozen MM-Grounding-DINO queries for responsibility isolation.

The extractor is intentionally a thin, fail-closed adapter around an external
MMDetection checkout.  It never modifies vendored code and imports MMDetection
only when the real runtime is constructed, so its tensor and manifest contract
can be tested on CPU without that dependency.

Two existing input surfaces are supported:

* RefCOCO JSONL rows become ``rank`` cache rows.
* Leakage-clean D3 screen rows become closed positive/negative confidence
  pairs, using the positive expression and its traceable edited expression.

The deployed detector is always queried with the complete expression and
``tokens_positive=-1`` (the native REC route).  Forward hooks capture the last
of six 900x256 decoder layers and the matching last-layer token logits and
normalized cxcywh boxes.  No detector parameter is trainable or changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.responsibility_isolation_cache import (
    CACHE_BOX_FORMAT,
    CACHE_FEATURE_DIM,
    CACHE_ROW_SCHEMA,
    CACHE_SHARD_SCHEMA,
    CACHE_SOURCE_SCHEMA,
    CACHE_TASK_CONFIDENCE_PAIR,
    CACHE_TASK_RANK,
    CachedCandidateContractError,
    file_sha256,
    normalized_cxcywh_iou,
    save_cached_candidate_shard,
    validate_cached_candidate_shard,
)


PINNED_MMDET_COMMIT = "cfd5d3a985b0249de009b67d04f37263e11cdf3d"
PINNED_FORMAL_CONFIG_PATH = Path(
    "/media/haoyi/T9/external/mmgdino_l_baseline/"
    "mmgdino_t_refcoco5e_formal_b8a4_partial_fusion4_seed20260819.py"
)
PINNED_FORMAL_CONFIG_SHA256 = (
    "c36ef5e02842cc276e4017396fbaf5c1c37622b19a76cb0691b2dd80fef3fb3a"
)
PINNED_MODEL_ID = "mmgroundingdino-t-refcoco5e-seed20260819-epoch5"
QUERY_FEATURE_NAME = "decoder.hidden_states[-1]:900x256"

EXTRACTION_RECEIPT_SCHEMA = (
    "responsibility_isolation.mmgdino_cache_extraction_receipt/v1"
)
EXPECTED_DECODER_LAYERS = 6
EXPECTED_REFERENCE_LAYERS = 7
EXPECTED_QUERY_COUNT = 900
EXPECTED_TOKEN_COUNT = 256
FEATURE_DTYPES = {"float16": torch.float16, "float32": torch.float32}


class MMGroundingDinoExtractionError(RuntimeError):
    """Raised when provenance, input, hook, or output contracts drift."""


@dataclass(frozen=True)
class ExtractionRequest:
    sample_id: str
    image_id: str
    image_path: Path
    image_sha256: str
    caption: str
    task: str
    gt_boxes: Tensor
    pair_id: str | None = None
    pair_role: str | None = None


@dataclass(frozen=True)
class HookBatch:
    query_features: Tensor
    native_score: Tensor
    boxes: Tensor
    candidate_mask: Tensor


class FrozenRuntime(Protocol):
    def infer(self, image_path: Path, caption: str) -> HookBatch:
        """Run one frozen full-expression forward."""


def _require_identifier(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise MMGroundingDinoExtractionError(
            f"{name} must be a nonempty, whitespace-trimmed string"
        )
    return value


def _require_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MMGroundingDinoExtractionError(
            f"{name} must be a nonnegative integer"
        )
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MMGroundingDinoExtractionError(
            f"{name} must be a lowercase 64-character SHA-256"
        )
    return value


def _strict_json_loads(line: str, *, location: str) -> Mapping[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MMGroundingDinoExtractionError(
                    f"duplicate JSON key {key!r} at {location}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(line, object_pairs_hook=pairs_hook)
    except MMGroundingDinoExtractionError:
        raise
    except Exception as exc:
        raise MMGroundingDinoExtractionError(
            f"invalid JSON at {location}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise MMGroundingDinoExtractionError(
            f"JSON row at {location} must be an object"
        )
    return value


def _read_bound_jsonl(
    path: str | Path, *, expected_sha256: str, name: str
) -> list[Mapping[str, Any]]:
    source = Path(path)
    if not source.is_absolute() or not source.is_file():
        raise MMGroundingDinoExtractionError(
            f"{name} must be an existing absolute file path"
        )
    expected = _require_sha256(expected_sha256, name=f"{name}.sha256")
    actual = file_sha256(source)
    if actual != expected:
        raise MMGroundingDinoExtractionError(
            f"{name} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    rows: list[Mapping[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.endswith("\n"):
                raise MMGroundingDinoExtractionError(
                    f"{name} line {line_number} is not newline terminated"
                )
            if not raw.strip():
                raise MMGroundingDinoExtractionError(
                    f"{name} line {line_number} is empty"
                )
            rows.append(
                _strict_json_loads(
                    raw, location=f"{source}:{line_number}"
                )
            )
    if not rows:
        raise MMGroundingDinoExtractionError(f"{name} must not be empty")
    return rows


def _image_id_from_path(path: Path) -> int:
    prefix = "COCO_train2014_"
    if not path.name.startswith(prefix) or path.suffix.lower() != ".jpg":
        raise MMGroundingDinoExtractionError(
            f"image path is not a COCO train2014 filename: {path}"
        )
    value = path.stem[len(prefix) :]
    if len(value) != 12 or not value.isdigit():
        raise MMGroundingDinoExtractionError(
            f"image path has malformed COCO image id: {path}"
        )
    return int(value)


def _load_image_identity(
    path_value: Any,
    *,
    image_id: int,
    cache: dict[Path, tuple[int, int, str]],
) -> tuple[Path, int, int, str]:
    if not isinstance(path_value, str):
        raise MMGroundingDinoExtractionError("filename must be a string")
    path = Path(path_value)
    if not path.is_absolute() or not path.is_file():
        raise MMGroundingDinoExtractionError(
            f"filename must be an existing absolute file: {path}"
        )
    if _image_id_from_path(path) != image_id:
        raise MMGroundingDinoExtractionError(
            f"filename/image_id mismatch for {path}: expected {image_id}"
        )
    resolved = path.resolve(strict=True)
    cached = cache.get(resolved)
    if cached is None:
        try:
            from PIL import Image

            with Image.open(resolved) as image:
                image.verify()
            with Image.open(resolved) as image:
                width, height = image.size
        except Exception as exc:
            raise MMGroundingDinoExtractionError(
                f"could not verify image {resolved}: {exc}"
            ) from exc
        if width <= 0 or height <= 0:
            raise MMGroundingDinoExtractionError(
                f"image has invalid dimensions: {resolved}"
            )
        cached = (int(width), int(height), file_sha256(resolved))
        cache[resolved] = cached
    width, height, image_sha = cached
    return resolved, width, height, image_sha


def _bound_image_path(
    row: Mapping[str, Any],
    *,
    location: str,
    image_root: str | Path | None,
) -> Path:
    raw_filename = row.get("filename")
    file_name = row.get("file_name")
    if raw_filename is None and file_name is None:
        raise MMGroundingDinoExtractionError(
            f"{location} requires filename or file_name"
        )
    if raw_filename is not None and not isinstance(raw_filename, str):
        raise MMGroundingDinoExtractionError(
            f"{location}.filename must be a string"
        )
    if file_name is not None:
        file_name = _require_identifier(
            file_name, name=f"{location}.file_name"
        )
        if Path(file_name).name != file_name:
            raise MMGroundingDinoExtractionError(
                f"{location}.file_name must be a basename without traversal"
            )
    raw_basename = Path(raw_filename).name if raw_filename is not None else None
    if file_name is not None and raw_basename is not None and file_name != raw_basename:
        raise MMGroundingDinoExtractionError(
            f"{location}.filename and file_name basenames disagree"
        )
    basename = file_name if file_name is not None else raw_basename
    if image_root is None:
        if raw_filename is None:
            raise MMGroundingDinoExtractionError(
                f"{location} requires an explicit image_root for relative file_name"
            )
        return Path(raw_filename)
    root = Path(image_root)
    if not root.is_absolute() or not root.is_dir():
        raise MMGroundingDinoExtractionError(
            "image_root must be an existing absolute directory"
        )
    resolved_root = root.resolve(strict=True)
    candidate = resolved_root / str(basename)
    if not candidate.is_file():
        raise MMGroundingDinoExtractionError(
            f"{location} image does not exist under the bound image_root: {candidate}"
        )
    resolved = candidate.resolve(strict=True)
    if resolved.parent != resolved_root:
        raise MMGroundingDinoExtractionError(
            f"{location} image escaped the bound image_root"
        )
    return resolved


def _normalized_gt_box(
    value: Any, *, width: int, height: int, name: str
) -> Tensor:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
    ):
        raise MMGroundingDinoExtractionError(
            f"{name} must be an absolute xywh list of length four"
        )
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise MMGroundingDinoExtractionError(f"{name} must contain numbers")
    x, y, box_width, box_height = (float(item) for item in value)
    values = torch.tensor([x, y, box_width, box_height], dtype=torch.float64)
    if not bool(torch.isfinite(values).all().item()):
        raise MMGroundingDinoExtractionError(f"{name} must be finite")
    tolerance = 1e-4
    if (
        x < -tolerance
        or y < -tolerance
        or box_width <= 0.0
        or box_height <= 0.0
        or x + box_width > width + tolerance
        or y + box_height > height + tolerance
    ):
        raise MMGroundingDinoExtractionError(
            f"{name} lies outside the {width}x{height} image"
        )
    normalized = torch.tensor(
        [
            (x + 0.5 * box_width) / width,
            (y + 0.5 * box_height) / height,
            box_width / width,
            box_height / height,
        ],
        dtype=torch.float32,
    )
    return normalized.unsqueeze(0).contiguous()


def _single_instance(row: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    instances = row.get("instances")
    if (
        not isinstance(instances, Sequence)
        or isinstance(instances, (str, bytes))
        or len(instances) != 1
        or not isinstance(instances[0], Mapping)
    ):
        raise MMGroundingDinoExtractionError(
            f"{name}.instances must contain exactly one object"
        )
    return instances[0]


def parse_refcoco_rank_requests(
    rows: Sequence[Mapping[str, Any]],
    *,
    image_root: str | Path | None = None,
    image_cache: dict[Path, tuple[int, int, str]] | None = None,
) -> list[ExtractionRequest]:
    cache = {} if image_cache is None else image_cache
    requests: list[ExtractionRequest] = []
    for index, row in enumerate(rows):
        location = f"rank[{index}]"
        source = _require_identifier(row.get("source"), name=f"{location}.source")
        image_id = _require_int(row.get("image_id"), name=f"{location}.image_id")
        ann_id = _require_int(row.get("ann_id"), name=f"{location}.ann_id")
        ref_id = _require_int(row.get("ref_id"), name=f"{location}.ref_id")
        sent_id = _require_int(row.get("sent_id"), name=f"{location}.sent_id")
        image_path_value = _bound_image_path(
            row, location=location, image_root=image_root
        )
        image_path, width, height, image_sha = _load_image_identity(
            str(image_path_value), image_id=image_id, cache=cache
        )
        instance = _single_instance(row, name=location)
        if instance.get("text_is_negative") is not False:
            raise MMGroundingDinoExtractionError(
                f"{location} must be a positive RefCOCO row"
            )
        caption = _require_identifier(
            instance.get("positive_phrase"),
            name=f"{location}.instances[0].positive_phrase",
        )
        gt_boxes = _normalized_gt_box(
            instance.get("bbox"),
            width=width,
            height=height,
            name=f"{location}.instances[0].bbox",
        )
        sample_id = f"refcoco:{source}:{image_id}:{ann_id}:{ref_id}:{sent_id}"
        requests.append(
            ExtractionRequest(
                sample_id=sample_id,
                image_id=str(image_id),
                image_path=image_path,
                image_sha256=image_sha,
                caption=caption,
                task=CACHE_TASK_RANK,
                gt_boxes=gt_boxes,
            )
        )
    _validate_request_identities(requests)
    return requests


def parse_d3_pair_requests(
    rows: Sequence[Mapping[str, Any]],
    *,
    image_root: str | Path | None = None,
    image_cache: dict[Path, tuple[int, int, str]] | None = None,
) -> list[ExtractionRequest]:
    cache = {} if image_cache is None else image_cache
    requests: list[ExtractionRequest] = []
    for index, row in enumerate(rows):
        location = f"d3[{index}]"
        for field in (
            "proposal_covered_verified",
            "visual_verified_negative",
            "traceable_counterfactual_edit",
        ):
            if row.get(field) is not True:
                raise MMGroundingDinoExtractionError(
                    f"{location}.{field} must be true"
                )
        if row.get("table_b_id") != "D3":
            raise MMGroundingDinoExtractionError(
                f"{location}.table_b_id must be 'D3'"
            )
        if row.get("tn_scope") != "proposal_covered_verified":
            raise MMGroundingDinoExtractionError(
                f"{location}.tn_scope must be 'proposal_covered_verified'"
            )
        if row.get("verification_contract") != "target_plus_all_cached_proposals_no":
            raise MMGroundingDinoExtractionError(
                f"{location}.verification_contract drifted"
            )
        if row.get("cached_proposal_coverage_only") is not True:
            raise MMGroundingDinoExtractionError(
                f"{location}.cached_proposal_coverage_only must be true"
            )
        if row.get("split") != "train":
            raise MMGroundingDinoExtractionError(
                f"{location}.split must be 'train'"
            )
        tn_eval_split = row.get("tn_eval_split")
        if tn_eval_split is not None and tn_eval_split != "screen_calibration":
            raise MMGroundingDinoExtractionError(
                f"{location}.tn_eval_split is not a sealed calibration surface"
            )
        source_sample_id = _require_identifier(
            row.get("sample_id"), name=f"{location}.sample_id"
        )
        image_id = _require_int(row.get("image_id"), name=f"{location}.image_id")
        _require_int(row.get("ann_id"), name=f"{location}.ann_id")
        _require_int(row.get("ref_id"), name=f"{location}.ref_id")
        _require_int(row.get("sent_id"), name=f"{location}.sent_id")
        image_path_value = _bound_image_path(
            row, location=location, image_root=image_root
        )
        image_path, width, height, image_sha = _load_image_identity(
            str(image_path_value), image_id=image_id, cache=cache
        )
        positive_caption = _require_identifier(
            row.get("sent"), name=f"{location}.sent"
        )
        negative_caption = _require_identifier(
            row.get("try_tn"), name=f"{location}.try_tn"
        )
        gt_boxes = _normalized_gt_box(
            row.get("target_bbox_used"),
            width=width,
            height=height,
            name=f"{location}.target_bbox_used",
        )
        if "instances" in row:
            instance = _single_instance(row, name=location)
            if instance.get("positive_phrase") != positive_caption:
                raise MMGroundingDinoExtractionError(
                    f"{location} positive phrase disagrees with sent"
                )
            if instance.get("negative_phrase") != negative_caption:
                raise MMGroundingDinoExtractionError(
                    f"{location} negative phrase disagrees with try_tn"
                )
            if instance.get("text_is_negative") is not True:
                raise MMGroundingDinoExtractionError(
                    f"{location} must carry the verified negative edit"
                )
            instance_gt_boxes = _normalized_gt_box(
                instance.get("bbox"),
                width=width,
                height=height,
                name=f"{location}.instances[0].bbox",
            )
            if not torch.equal(gt_boxes, instance_gt_boxes):
                raise MMGroundingDinoExtractionError(
                    f"{location} target_bbox_used disagrees with instances[0].bbox"
                )
        pair_id = f"d3:{source_sample_id}"
        common = dict(
            image_id=str(image_id),
            image_path=image_path,
            image_sha256=image_sha,
            task=CACHE_TASK_CONFIDENCE_PAIR,
            pair_id=pair_id,
        )
        requests.extend(
            [
                ExtractionRequest(
                    sample_id=f"{pair_id}:positive",
                    caption=positive_caption,
                    gt_boxes=gt_boxes,
                    pair_role="positive",
                    **common,
                ),
                ExtractionRequest(
                    sample_id=f"{pair_id}:negative",
                    caption=negative_caption,
                    gt_boxes=torch.empty(0, 4, dtype=torch.float32).contiguous(),
                    pair_role="negative",
                    **common,
                ),
            ]
        )
    _validate_request_identities(requests)
    return requests


def _validate_request_identities(requests: Sequence[ExtractionRequest]) -> None:
    sample_ids = [request.sample_id for request in requests]
    if len(sample_ids) != len(set(sample_ids)):
        raise MMGroundingDinoExtractionError(
            "extraction request sample_id values must be unique"
        )
    pair_roles: dict[str, list[str]] = {}
    for request in requests:
        if not isinstance(request, ExtractionRequest):
            raise MMGroundingDinoExtractionError(
                "all extraction requests must be ExtractionRequest values"
            )
        _require_identifier(request.sample_id, name="request.sample_id")
        _require_identifier(request.image_id, name="request.image_id")
        _require_identifier(request.caption, name="request.caption")
        _require_sha256(request.image_sha256, name="request.image_sha256")
        if (
            not isinstance(request.image_path, Path)
            or not request.image_path.is_absolute()
            or not request.image_path.is_file()
        ):
            raise MMGroundingDinoExtractionError(
                "request.image_path must be an existing absolute file"
            )
        gt_boxes = request.gt_boxes
        if (
            not torch.is_tensor(gt_boxes)
            or gt_boxes.dtype != torch.float32
            or gt_boxes.device.type != "cpu"
            or gt_boxes.dim() != 2
            or int(gt_boxes.shape[1]) != 4
            or not gt_boxes.is_contiguous()
            or gt_boxes.requires_grad
        ):
            raise MMGroundingDinoExtractionError(
                "request.gt_boxes must be detached contiguous CPU float32 (N,4)"
            )
        if not bool(torch.isfinite(gt_boxes).all().item()):
            raise MMGroundingDinoExtractionError(
                "request.gt_boxes must be finite"
            )
        if gt_boxes.numel():
            if bool(((gt_boxes < 0.0) | (gt_boxes > 1.0)).any().item()):
                raise MMGroundingDinoExtractionError(
                    "request.gt_boxes must lie in [0,1]"
                )
            if bool((gt_boxes[:, 2:] <= 0.0).any().item()):
                raise MMGroundingDinoExtractionError(
                    "request.gt_boxes widths/heights must be positive"
                )
            low = gt_boxes[:, :2] - 0.5 * gt_boxes[:, 2:]
            high = gt_boxes[:, :2] + 0.5 * gt_boxes[:, 2:]
            if bool(((low < -1e-5) | (high > 1.0 + 1e-5)).any().item()):
                raise MMGroundingDinoExtractionError(
                    "request.gt_boxes must not extend outside the image"
                )
        if request.task == CACHE_TASK_CONFIDENCE_PAIR:
            if request.pair_id is None or request.pair_role is None:
                raise MMGroundingDinoExtractionError(
                    "confidence request requires pair_id and pair_role"
                )
            _require_identifier(request.pair_id, name="request.pair_id")
            if request.pair_role not in ("positive", "negative"):
                raise MMGroundingDinoExtractionError(
                    "confidence request pair_role must be positive or negative"
                )
            if request.pair_role == "positive" and int(gt_boxes.shape[0]) == 0:
                raise MMGroundingDinoExtractionError(
                    "positive confidence request must contain gt_boxes"
                )
            if request.pair_role == "negative" and int(gt_boxes.shape[0]) != 0:
                raise MMGroundingDinoExtractionError(
                    "negative confidence request must have empty gt_boxes"
                )
            pair_roles.setdefault(request.pair_id, []).append(request.pair_role)
        elif request.task != CACHE_TASK_RANK:
            raise MMGroundingDinoExtractionError(
                f"unsupported request task {request.task!r}"
            )
        else:
            if int(gt_boxes.shape[0]) == 0:
                raise MMGroundingDinoExtractionError(
                    "rank request must contain gt_boxes"
                )
            if request.pair_id is not None or request.pair_role is not None:
                raise MMGroundingDinoExtractionError(
                    "rank request must not carry confidence pair fields"
                )
    for pair_id, roles in pair_roles.items():
        if sorted(roles) != ["negative", "positive"]:
            raise MMGroundingDinoExtractionError(
                f"request pair {pair_id!r} is not closed"
            )


def _require_hook_tensor(
    value: Any, *, name: str, shape: tuple[int, ...]
) -> Tensor:
    if not torch.is_tensor(value) or tuple(value.shape) != shape:
        actual = tuple(value.shape) if torch.is_tensor(value) else type(value)
        raise MMGroundingDinoExtractionError(
            f"{name} must have shape {shape}, got {actual}"
        )
    if not value.is_floating_point():
        raise MMGroundingDinoExtractionError(f"{name} must be floating point")
    return value


def parse_mmgdino_hook_outputs(
    decoder_output: Any,
    head_output: Any,
    *,
    feature_dtype: torch.dtype = torch.float16,
) -> HookBatch:
    """Validate pinned hook outputs and materialize frozen CPU tensors."""
    if feature_dtype not in (torch.float16, torch.float32):
        raise MMGroundingDinoExtractionError(
            "feature_dtype must be torch.float16 or torch.float32"
        )
    if not isinstance(decoder_output, (tuple, list)) or len(decoder_output) != 2:
        raise MMGroundingDinoExtractionError(
            "decoder hook must return (hidden_states, references)"
        )
    if not isinstance(head_output, (tuple, list)) or len(head_output) != 2:
        raise MMGroundingDinoExtractionError(
            "bbox-head hook must return (token_logits, boxes)"
        )
    hidden = decoder_output[0]
    if not torch.is_tensor(hidden) or hidden.dim() != 4:
        raise MMGroundingDinoExtractionError(
            "decoder hidden_states must be a four-dimensional tensor"
        )
    batch_size = int(hidden.shape[1])
    expected_hidden = (
        EXPECTED_DECODER_LAYERS,
        batch_size,
        EXPECTED_QUERY_COUNT,
        CACHE_FEATURE_DIM,
    )
    hidden = _require_hook_tensor(
        hidden, name="decoder.hidden_states", shape=expected_hidden
    )
    references = _require_hook_tensor(
        decoder_output[1],
        name="decoder.references",
        shape=(
            EXPECTED_REFERENCE_LAYERS,
            batch_size,
            EXPECTED_QUERY_COUNT,
            4,
        ),
    )
    token_logits = _require_hook_tensor(
        head_output[0],
        name="bbox_head.token_logits",
        shape=(
            EXPECTED_DECODER_LAYERS,
            batch_size,
            EXPECTED_QUERY_COUNT,
            EXPECTED_TOKEN_COUNT,
        ),
    )
    boxes = _require_hook_tensor(
        head_output[1],
        name="bbox_head.boxes",
        shape=(EXPECTED_DECODER_LAYERS, batch_size, EXPECTED_QUERY_COUNT, 4),
    )
    if batch_size != 1:
        raise MMGroundingDinoExtractionError(
            f"extractor requires batch size one, got {batch_size}"
        )
    final_features = hidden[-1]
    final_logits = token_logits[-1]
    final_boxes = boxes[-1]
    if not bool(torch.isfinite(final_features).all().item()):
        raise MMGroundingDinoExtractionError("final decoder features are nonfinite")
    if not bool(torch.isfinite(references).all().item()):
        raise MMGroundingDinoExtractionError("decoder references are nonfinite")
    candidate_mask = torch.isfinite(final_logits).any(dim=-1)
    if not bool(candidate_mask.all().item()):
        raise MMGroundingDinoExtractionError(
            "native REC route must retain all 900 queries"
        )
    native_score = final_logits.sigmoid().amax(dim=-1)
    if not bool(torch.isfinite(native_score).all().item()):
        raise MMGroundingDinoExtractionError("native scores are nonfinite")
    if not bool(torch.isfinite(final_boxes).all().item()):
        raise MMGroundingDinoExtractionError("final boxes are nonfinite")
    if bool(((final_boxes < 0.0) | (final_boxes > 1.0)).any().item()):
        raise MMGroundingDinoExtractionError("final boxes must lie in [0,1]")
    if bool((final_boxes[..., 2:] <= 0.0).any().item()):
        raise MMGroundingDinoExtractionError(
            "final box widths/heights must be positive"
        )
    return HookBatch(
        query_features=final_features[0]
        .detach()
        .to(device="cpu", dtype=feature_dtype)
        .contiguous(),
        native_score=native_score[0]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous(),
        boxes=final_boxes[0]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous(),
        candidate_mask=candidate_mask[0]
        .detach()
        .to(device="cpu", dtype=torch.bool)
        .contiguous(),
    )


class MMGroundingDinoHookRecorder:
    """Record exactly one decoder and bbox-head invocation per forward."""

    def __init__(self, model: Any):
        if not hasattr(model, "decoder") or not hasattr(model, "bbox_head"):
            raise MMGroundingDinoExtractionError(
                "model must expose decoder and bbox_head modules"
            )
        self._decoder_outputs: list[Any] = []
        self._head_outputs: list[Any] = []
        self._handles = [
            model.decoder.register_forward_hook(self._capture_decoder),
            model.bbox_head.register_forward_hook(self._capture_head),
        ]

    def _capture_decoder(self, _module: Any, _inputs: Any, output: Any) -> None:
        self._decoder_outputs.append(output)

    def _capture_head(self, _module: Any, _inputs: Any, output: Any) -> None:
        self._head_outputs.append(output)

    def reset(self) -> None:
        self._decoder_outputs.clear()
        self._head_outputs.clear()

    def consume(self, *, feature_dtype: torch.dtype) -> HookBatch:
        if len(self._decoder_outputs) != 1 or len(self._head_outputs) != 1:
            raise MMGroundingDinoExtractionError(
                "one forward must trigger exactly one decoder and one bbox-head hook; "
                f"got decoder={len(self._decoder_outputs)}, "
                f"bbox_head={len(self._head_outputs)}"
            )
        result = parse_mmgdino_hook_outputs(
            self._decoder_outputs[0],
            self._head_outputs[0],
            feature_dtype=feature_dtype,
        )
        self.reset()
        return result

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "MMGroundingDinoHookRecorder":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def _cxcywh_to_absolute_xyxy(box: Tensor, *, width: int, height: int) -> Tensor:
    center, size = box[:2], box[2:]
    xyxy = torch.cat((center - 0.5 * size, center + 0.5 * size))
    factor = xyxy.new_tensor([width, height, width, height])
    xyxy = xyxy * factor
    xyxy[0::2].clamp_(0, width)
    xyxy[1::2].clamp_(0, height)
    return xyxy


def validate_native_prediction_parity(
    hook: HookBatch,
    *,
    predicted_scores: Tensor,
    predicted_boxes: Tensor,
    image_width: int,
    image_height: int,
) -> None:
    if predicted_scores.dim() != 1 or predicted_boxes.dim() != 2:
        raise MMGroundingDinoExtractionError(
            "native prediction tensors have invalid ranks"
        )
    if not len(predicted_scores) or tuple(predicted_boxes.shape[1:]) != (4,):
        raise MMGroundingDinoExtractionError("native prediction is empty")
    top_query = int(hook.native_score.argmax().item())
    expected_score = hook.native_score[top_query]
    expected_box = _cxcywh_to_absolute_xyxy(
        hook.boxes[top_query], width=image_width, height=image_height
    )
    if not torch.allclose(
        predicted_scores[0].detach().cpu().float(), expected_score, atol=1e-6, rtol=0
    ):
        raise MMGroundingDinoExtractionError(
            "native predict top score disagrees with raw hook"
        )
    if not torch.allclose(
        predicted_boxes[0].detach().cpu().float(), expected_box, atol=1e-3, rtol=0
    ):
        raise MMGroundingDinoExtractionError(
            "native predict top box disagrees with raw hook"
        )


class MMDetectionFrozenRuntime:
    """Real pinned MMDetection runtime; imported only for GPU extraction."""

    def __init__(
        self,
        *,
        config_path: Path,
        checkpoint_path: Path,
        device: str,
        feature_dtype: torch.dtype,
    ) -> None:
        try:
            from mmcv.transforms import Compose
            from mmengine.config import Config
            from mmdet.apis import init_detector
        except Exception as exc:
            raise MMGroundingDinoExtractionError(
                "MMDetection runtime imports failed; use the pinned external env"
            ) from exc
        config = Config.fromfile(str(config_path))
        self.model = init_detector(
            config,
            str(checkpoint_path),
            palette="random",
            device=device,
        )
        if self.model.training:
            raise MMGroundingDinoExtractionError("detector must be in eval mode")
        self.model.requires_grad_(False)
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise MMGroundingDinoExtractionError(
                "detector parameters must all be frozen"
            )
        if int(getattr(self.model, "num_queries", -1)) != EXPECTED_QUERY_COUNT:
            raise MMGroundingDinoExtractionError("detector num_queries drifted")
        self.pipeline = Compose(config.test_pipeline)
        self.feature_dtype = feature_dtype
        self.recorder = MMGroundingDinoHookRecorder(self.model)

    def infer(self, image_path: Path, caption: str) -> HookBatch:
        data = dict(
            img_path=str(image_path),
            img_id=0,
            text=caption,
            custom_entities=False,
            tokens_positive=-1,
        )
        packed = self.pipeline(data)
        packed["inputs"] = [packed["inputs"]]
        packed["data_samples"] = [packed["data_samples"]]
        self.recorder.reset()
        with torch.inference_mode():
            result = self.model.test_step(packed)[0]
        hook = self.recorder.consume(feature_dtype=self.feature_dtype)
        if not hasattr(result, "pred_instances"):
            raise MMGroundingDinoExtractionError(
                "native predict did not return pred_instances"
            )
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
        validate_native_prediction_parity(
            hook,
            predicted_scores=result.pred_instances.scores,
            predicted_boxes=result.pred_instances.bboxes,
            image_width=width,
            image_height=height,
        )
        return hook

    def close(self) -> None:
        self.recorder.close()


def _git_head(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception as exc:
        raise MMGroundingDinoExtractionError(
            f"could not read MMDetection git HEAD at {path}: {exc}"
        ) from exc


def validate_pinned_runtime_assets(
    *,
    mmdet_root: str | Path,
    config_path: str | Path,
    checkpoint_path: str | Path,
    expected_checkpoint_sha256: str,
    expected_mmdet_commit: str | None = None,
    expected_config_path: str | Path | None = None,
    expected_config_sha256: str | None = None,
) -> dict[str, str]:
    root = Path(mmdet_root)
    config = Path(config_path)
    checkpoint = Path(checkpoint_path)
    for value, name, directory in (
        (root, "mmdet_root", True),
        (config, "config_path", False),
        (checkpoint, "checkpoint_path", False),
    ):
        if not value.is_absolute() or not (value.is_dir() if directory else value.is_file()):
            kind = "directory" if directory else "file"
            raise MMGroundingDinoExtractionError(
                f"{name} must be an existing absolute {kind}"
            )
    commit = _git_head(root)
    expected_commit = str(
        PINNED_MMDET_COMMIT
        if expected_mmdet_commit is None
        else expected_mmdet_commit
    ).strip().lower()
    if (
        len(expected_commit) != 40
        or any(character not in "0123456789abcdef" for character in expected_commit)
    ):
        raise MMGroundingDinoExtractionError(
            "expected_mmdet_commit must be a lowercase 40-character git SHA"
        )
    if commit != expected_commit:
        raise MMGroundingDinoExtractionError(
            f"MMDetection commit drift: expected {expected_commit}, got {commit}"
        )
    resolved_config = config.resolve(strict=True)
    expected_config = Path(
        PINNED_FORMAL_CONFIG_PATH
        if expected_config_path is None
        else expected_config_path
    )
    if (
        not expected_config.is_absolute()
        or not expected_config.is_file()
        or resolved_config != expected_config.resolve(strict=True)
    ):
        raise MMGroundingDinoExtractionError(
            "formal MM-GDINO config path drift: "
            f"expected {expected_config}, got {resolved_config}"
        )
    config_sha = file_sha256(config)
    expected_config_sha = _require_sha256(
        PINNED_FORMAL_CONFIG_SHA256
        if expected_config_sha256 is None
        else expected_config_sha256,
        name="expected_config_sha256",
    )
    if config_sha != expected_config_sha:
        raise MMGroundingDinoExtractionError(
            "formal MM-GDINO config SHA-256 drift: "
            f"expected {expected_config_sha}, got {config_sha}"
        )
    expected_checkpoint = _require_sha256(
        expected_checkpoint_sha256, name="expected_checkpoint_sha256"
    )
    checkpoint_sha = file_sha256(checkpoint)
    if checkpoint_sha != expected_checkpoint:
        raise MMGroundingDinoExtractionError(
            "checkpoint SHA-256 mismatch: "
            f"expected {expected_checkpoint}, got {checkpoint_sha}"
        )
    return {
        "mmdetection_commit": commit,
        "config_sha256": config_sha,
        "checkpoint_sha256": checkpoint_sha,
    }


def extractor_code_sha256(path: str | Path | None = None) -> str:
    return file_sha256(Path(__file__) if path is None else path)


def _validate_hook_batch(hook: HookBatch) -> HookBatch:
    """Enforce the stronger MM-GDINO contract before cache serialization."""
    if not isinstance(hook, HookBatch):
        raise MMGroundingDinoExtractionError(
            "runtime.infer must return a HookBatch"
        )
    expected = {
        "query_features": (
            (EXPECTED_QUERY_COUNT, CACHE_FEATURE_DIM),
            (torch.float16, torch.float32),
        ),
        "native_score": ((EXPECTED_QUERY_COUNT,), (torch.float32,)),
        "boxes": ((EXPECTED_QUERY_COUNT, 4), (torch.float32,)),
        "candidate_mask": ((EXPECTED_QUERY_COUNT,), (torch.bool,)),
    }
    for name, (shape, dtypes) in expected.items():
        value = getattr(hook, name)
        if not torch.is_tensor(value):
            raise MMGroundingDinoExtractionError(
                f"runtime {name} must be a tensor"
            )
        if tuple(value.shape) != shape:
            raise MMGroundingDinoExtractionError(
                f"runtime {name} must have shape {shape}, got {tuple(value.shape)}"
            )
        if value.dtype not in dtypes:
            raise MMGroundingDinoExtractionError(
                f"runtime {name} dtype must be one of {dtypes}, got {value.dtype}"
            )
        if value.device.type != "cpu" or not value.is_contiguous():
            raise MMGroundingDinoExtractionError(
                f"runtime {name} must be contiguous on CPU"
            )
        if value.requires_grad:
            raise MMGroundingDinoExtractionError(
                f"runtime {name} must be detached"
            )
    if not bool(torch.isfinite(hook.query_features).all().item()):
        raise MMGroundingDinoExtractionError(
            "runtime query_features must be finite"
        )
    if not bool(torch.isfinite(hook.native_score).all().item()):
        raise MMGroundingDinoExtractionError("runtime native_score must be finite")
    if bool(((hook.native_score < 0.0) | (hook.native_score > 1.0)).any().item()):
        raise MMGroundingDinoExtractionError(
            "runtime native_score must lie in [0,1]"
        )
    if not bool(torch.isfinite(hook.boxes).all().item()):
        raise MMGroundingDinoExtractionError("runtime boxes must be finite")
    if bool(((hook.boxes < 0.0) | (hook.boxes > 1.0)).any().item()):
        raise MMGroundingDinoExtractionError("runtime boxes must lie in [0,1]")
    if bool((hook.boxes[:, 2:] <= 0.0).any().item()):
        raise MMGroundingDinoExtractionError(
            "runtime box widths/heights must be positive"
        )
    if not bool(hook.candidate_mask.all().item()):
        raise MMGroundingDinoExtractionError(
            "native full-expression candidate mask must retain all 900 queries"
        )
    return hook


def _cache_row(request: ExtractionRequest, hook: HookBatch) -> dict[str, Any]:
    hook = _validate_hook_batch(hook)
    row: dict[str, Any] = {
        "schema": CACHE_ROW_SCHEMA,
        "sample_id": request.sample_id,
        "image_id": request.image_id,
        "task": request.task,
        "query_features": hook.query_features,
        "native_score": hook.native_score,
        "boxes": hook.boxes,
        "candidate_mask": hook.candidate_mask,
        "gt_boxes": request.gt_boxes.detach().cpu().float().contiguous(),
    }
    if request.task == CACHE_TASK_CONFIDENCE_PAIR:
        row["pair_id"] = request.pair_id
        row["pair_role"] = request.pair_role
    return row


def extract_cached_candidate_shard(
    *,
    rank_requests: Sequence[ExtractionRequest],
    pair_requests: Sequence[ExtractionRequest],
    runtime: FrozenRuntime,
    shard_id: str,
    checkpoint_sha256: str,
    extractor_sha256: str,
    model_id: str = PINNED_MODEL_ID,
    config_sha256: str = PINNED_FORMAL_CONFIG_SHA256,
    allow_rank_rows_without_positive: bool = False,
) -> tuple[dict[str, Any], dict[str, int]]:
    _require_identifier(shard_id, name="shard_id")
    checkpoint_sha = _require_sha256(
        checkpoint_sha256, name="checkpoint_sha256"
    )
    extractor_sha = _require_sha256(
        extractor_sha256, name="extractor_code_sha256"
    )
    model_id = _require_identifier(model_id, name="model_id")
    config_sha = _require_sha256(config_sha256, name="config_sha256")
    _validate_request_identities(tuple(rank_requests) + tuple(pair_requests))
    if not rank_requests or not pair_requests:
        raise MMGroundingDinoExtractionError(
            "extraction requires rank requests and confidence pairs"
        )
    rows: list[dict[str, Any]] = []
    counters = {
        "rank_input": len(rank_requests),
        "rank_kept": 0,
        "rank_no_positive": 0,
        "confidence_rows": len(pair_requests),
    }
    if any(request.task != CACHE_TASK_RANK for request in rank_requests):
        raise MMGroundingDinoExtractionError(
            "rank_requests may contain only rank tasks"
        )
    if any(
        request.task != CACHE_TASK_CONFIDENCE_PAIR
        for request in pair_requests
    ):
        raise MMGroundingDinoExtractionError(
            "pair_requests may contain only confidence-pair tasks"
        )
    for request in tuple(rank_requests) + tuple(pair_requests):
        if file_sha256(request.image_path) != request.image_sha256:
            raise MMGroundingDinoExtractionError(
                f"image SHA-256 changed before forward: {request.image_path}"
            )
        hook = runtime.infer(request.image_path, request.caption)
        row = _cache_row(request, hook)
        if request.task == CACHE_TASK_RANK:
            iou = normalized_cxcywh_iou(row["boxes"], row["gt_boxes"]).amax(dim=1)
            eligible = row["candidate_mask"]
            if not bool((eligible & (iou >= 0.5)).any().item()):
                counters["rank_no_positive"] += 1
                if not allow_rank_rows_without_positive:
                    raise MMGroundingDinoExtractionError(
                        f"rank request {request.sample_id!r} has no IoU>=0.5 "
                        "query; extraction is fail-closed"
                    )
            if not bool((eligible & (iou < 0.5)).any().item()):
                raise MMGroundingDinoExtractionError(
                    f"rank request {request.sample_id!r} has no hard negative "
                    "query; extraction is fail-closed"
                )
            if bool((eligible & (iou >= 0.5)).any().item()):
                counters["rank_kept"] += 1
        rows.append(row)
    payload = {
        "schema": CACHE_SHARD_SCHEMA,
        "shard_id": shard_id,
        "source": {
            "schema": CACHE_SOURCE_SCHEMA,
            "model_id": model_id,
            "checkpoint_sha256": checkpoint_sha,
            "config_sha256": config_sha,
            "extractor_code_sha256": extractor_sha,
            "query_feature_name": QUERY_FEATURE_NAME,
        },
        "feature_dim": CACHE_FEATURE_DIM,
        "box_format": CACHE_BOX_FORMAT,
        "rows": tuple(rows),
    }
    try:
        validated = validate_cached_candidate_shard(
            payload,
            allow_rank_rows_without_positive=allow_rank_rows_without_positive,
        )
    except CachedCandidateContractError as exc:
        raise MMGroundingDinoExtractionError(
            f"extracted shard violates cache contract: {exc}"
        ) from exc
    return validated, counters


def _image_set_sha256(requests: Sequence[ExtractionRequest]) -> str:
    identities = sorted(
        {(str(request.image_path), request.image_sha256) for request in requests}
    )
    encoded = json.dumps(identities, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json_dump(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmdet-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--mmdet-commit", default=PINNED_MMDET_COMMIT)
    parser.add_argument("--config-sha256", default=PINNED_FORMAL_CONFIG_SHA256)
    parser.add_argument("--model-id", default=PINNED_MODEL_ID)
    parser.add_argument("--rank-jsonl", type=Path, required=True)
    parser.add_argument("--rank-jsonl-sha256", required=True)
    parser.add_argument("--rank-image-root", type=Path, required=True)
    parser.add_argument("--d3-jsonl", type=Path, required=True)
    parser.add_argument("--d3-jsonl-sha256", required=True)
    parser.add_argument("--d3-image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--rank-limit", type=int, default=0)
    parser.add_argument("--pair-limit", type=int, default=0)
    parser.add_argument("--feature-dtype", choices=tuple(FEATURE_DTYPES), default="float16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--allow-rank-rows-without-positive",
        action="store_true",
        help=(
            "Preserve scheduled rank rows without an eligible IoU>=0.5 query; "
            "their rank margin is masked by the formal trainer."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    for value, name in ((args.rank_limit, "rank_limit"), (args.pair_limit, "pair_limit")):
        if value < 0:
            raise MMGroundingDinoExtractionError(f"{name} must be nonnegative")
    if not args.output.is_absolute() or not args.receipt.is_absolute():
        raise MMGroundingDinoExtractionError("output and receipt paths must be absolute")
    if args.output.exists() or args.receipt.exists():
        raise MMGroundingDinoExtractionError(
            "output and receipt must not already exist"
        )
    asset_binding = validate_pinned_runtime_assets(
        mmdet_root=args.mmdet_root,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        expected_mmdet_commit=args.mmdet_commit,
        expected_config_path=args.config,
        expected_config_sha256=args.config_sha256,
    )
    rank_source = _read_bound_jsonl(
        args.rank_jsonl,
        expected_sha256=args.rank_jsonl_sha256,
        name="rank_jsonl",
    )
    d3_source = _read_bound_jsonl(
        args.d3_jsonl,
        expected_sha256=args.d3_jsonl_sha256,
        name="d3_jsonl",
    )
    if args.rank_limit:
        rank_source = rank_source[: args.rank_limit]
    if args.pair_limit:
        d3_source = d3_source[: args.pair_limit]
    image_cache: dict[Path, tuple[int, int, str]] = {}
    rank_requests = parse_refcoco_rank_requests(
        rank_source,
        image_root=args.rank_image_root,
        image_cache=image_cache,
    )
    pair_requests = parse_d3_pair_requests(
        d3_source,
        image_root=args.d3_image_root,
        image_cache=image_cache,
    )
    extractor_sha = extractor_code_sha256()
    runtime = MMDetectionFrozenRuntime(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        feature_dtype=FEATURE_DTYPES[args.feature_dtype],
    )
    try:
        shard, counters = extract_cached_candidate_shard(
            rank_requests=rank_requests,
            pair_requests=pair_requests,
            runtime=runtime,
            shard_id=args.shard_id,
        checkpoint_sha256=asset_binding["checkpoint_sha256"],
        extractor_sha256=extractor_sha,
        model_id=args.model_id,
        config_sha256=asset_binding["config_sha256"],
        allow_rank_rows_without_positive=args.allow_rank_rows_without_positive,
        )
    finally:
        runtime.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        hashes = save_cached_candidate_shard(
            shard,
            temporary_path,
            allow_rank_rows_without_positive=args.allow_rank_rows_without_positive,
        )
        os.replace(temporary_path, args.output)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    all_requests = tuple(rank_requests) + tuple(pair_requests)
    receipt = {
        "schema": EXTRACTION_RECEIPT_SCHEMA,
        "status": "complete",
        "shard_id": args.shard_id,
        "source": shard["source"],
        "assets": {
            **asset_binding,
            "mmdet_root": str(args.mmdet_root.resolve(strict=True)),
            "config_path": str(args.config.resolve(strict=True)),
            "checkpoint_path": str(args.checkpoint.resolve(strict=True)),
            "extractor_path": str(Path(__file__).resolve(strict=True)),
            "extractor_code_sha256": extractor_sha,
        },
        "inputs": {
            "rank_jsonl": {
                "path": str(args.rank_jsonl.resolve(strict=True)),
                "sha256": args.rank_jsonl_sha256,
                "selected_rows": len(rank_source),
                "image_root": str(args.rank_image_root.resolve(strict=True)),
            },
            "d3_jsonl": {
                "path": str(args.d3_jsonl.resolve(strict=True)),
                "sha256": args.d3_jsonl_sha256,
                "selected_pairs": len(d3_source),
                "image_root": str(args.d3_image_root.resolve(strict=True)),
            },
            "unique_image_count": len(image_cache),
            "image_identity_sha256": _image_set_sha256(all_requests),
        },
        "output": {
            "path": str(args.output),
            **hashes,
            "row_count": len(shard["rows"]),
        },
        "counters": counters,
        "runtime": {
            "device": args.device,
            "feature_dtype": args.feature_dtype,
            "batch_size": 1,
            "training": False,
            "tokens_positive": -1,
            "rank_rows_without_positive_policy": (
                "preserve_and_zero_margin"
                if args.allow_rank_rows_without_positive
                else "fail_closed"
            ),
        },
    }
    _atomic_json_dump(receipt, args.receipt)


if __name__ == "__main__":
    main()


__all__ = [
    "EXPECTED_DECODER_LAYERS",
    "EXPECTED_QUERY_COUNT",
    "EXPECTED_REFERENCE_LAYERS",
    "EXPECTED_TOKEN_COUNT",
    "EXTRACTION_RECEIPT_SCHEMA",
    "ExtractionRequest",
    "HookBatch",
    "MMDetectionFrozenRuntime",
    "MMGroundingDinoExtractionError",
    "MMGroundingDinoHookRecorder",
    "PINNED_FORMAL_CONFIG_SHA256",
    "PINNED_FORMAL_CONFIG_PATH",
    "PINNED_MMDET_COMMIT",
    "PINNED_MODEL_ID",
    "QUERY_FEATURE_NAME",
    "extract_cached_candidate_shard",
    "extractor_code_sha256",
    "parse_d3_pair_requests",
    "parse_mmgdino_hook_outputs",
    "parse_refcoco_rank_requests",
    "validate_native_prediction_parity",
    "validate_pinned_runtime_assets",
]
