"""Versioned cached-candidate contract for responsibility-isolation pilots.

This cache is deliberately independent of any detector framework.  An
extractor only has to materialize frozen query features, native scores and
boxes.  The pilot trainer never imports or executes the source detector.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


CACHE_ROW_SCHEMA = "responsibility_isolation.cached_candidate_row/v1"
CACHE_SHARD_SCHEMA = "responsibility_isolation.cached_candidate_shard/v1"
CACHE_SOURCE_SCHEMA = "responsibility_isolation.cached_candidate_source/v1"
CACHE_FEATURE_DIM = 256
CACHE_BOX_FORMAT = "normalized_cxcywh"
CACHE_TASK_RANK = "rank"
CACHE_TASK_CONFIDENCE_PAIR = "confidence_pair"
CACHE_TASKS = (CACHE_TASK_RANK, CACHE_TASK_CONFIDENCE_PAIR)
CACHE_PAIR_ROLES = ("positive", "negative")

_BASE_ROW_FIELDS = {
    "schema",
    "sample_id",
    "image_id",
    "task",
    "query_features",
    "native_score",
    "boxes",
    "candidate_mask",
    "gt_boxes",
}
_PAIR_ROW_FIELDS = _BASE_ROW_FIELDS | {"pair_id", "pair_role"}
_SHARD_FIELDS = {
    "schema",
    "shard_id",
    "source",
    "feature_dim",
    "box_format",
    "rows",
}
_SOURCE_FIELDS = {
    "schema",
    "model_id",
    "checkpoint_sha256",
    "config_sha256",
    "extractor_code_sha256",
    "query_feature_name",
}


class CachedCandidateContractError(ValueError):
    """Raised when cache bytes do not satisfy the frozen pilot contract."""


def _require_identifier(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise CachedCandidateContractError(
            f"{name} must be a nonempty, whitespace-trimmed string"
        )
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CachedCandidateContractError(
            f"{name} must be a lowercase 64-character SHA-256"
        )
    return value


def _validate_source(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _SOURCE_FIELDS:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise CachedCandidateContractError(
            "cache source fields differ from contract: "
            f"missing={sorted(_SOURCE_FIELDS - actual)}, "
            f"extra={sorted(actual - _SOURCE_FIELDS)}"
        )
    if value["schema"] != CACHE_SOURCE_SCHEMA:
        raise CachedCandidateContractError(
            f"cache source schema must be {CACHE_SOURCE_SCHEMA!r}"
        )
    _require_identifier(value["model_id"], name="source.model_id")
    _require_identifier(
        value["query_feature_name"], name="source.query_feature_name"
    )
    for field in (
        "checkpoint_sha256",
        "config_sha256",
        "extractor_code_sha256",
    ):
        _require_sha256(value[field], name=f"source.{field}")
    return dict(value)


def _require_tensor(
    value: Any,
    *,
    name: str,
    shape: tuple[int | None, ...],
    dtypes: tuple[torch.dtype, ...],
) -> Tensor:
    if not torch.is_tensor(value):
        raise CachedCandidateContractError(f"{name} must be a tensor")
    if value.device.type != "cpu":
        raise CachedCandidateContractError(f"{name} must be stored on CPU")
    if value.dtype not in dtypes:
        raise CachedCandidateContractError(
            f"{name} dtype must be one of {dtypes}, got {value.dtype}"
        )
    if value.dim() != len(shape) or any(
        expected is not None and int(actual) != expected
        for actual, expected in zip(value.shape, shape)
    ):
        raise CachedCandidateContractError(
            f"{name} must have shape {shape}, got {tuple(value.shape)}"
        )
    if not value.is_contiguous():
        raise CachedCandidateContractError(f"{name} must be contiguous")
    return value


def _validate_normalized_boxes(value: Tensor, *, name: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise CachedCandidateContractError(f"{name} must be finite")
    if value.numel() == 0:
        return
    if bool(((value < 0.0) | (value > 1.0)).any().item()):
        raise CachedCandidateContractError(f"{name} must lie in [0,1]")
    if bool((value[:, 2:] <= 0.0).any().item()):
        raise CachedCandidateContractError(f"{name} widths/heights must be positive")


def normalized_cxcywh_iou(boxes: Tensor, gt_boxes: Tensor) -> Tensor:
    """Return pairwise IoU for normalized cxcywh boxes without repo imports."""
    if boxes.dim() != 2 or boxes.shape[-1] != 4:
        raise ValueError("boxes must be (Q,4)")
    if gt_boxes.dim() != 2 or gt_boxes.shape[-1] != 4:
        raise ValueError("gt_boxes must be (N,4)")

    def xyxy(value: Tensor) -> Tensor:
        center, size = value[..., :2], value[..., 2:]
        return torch.cat((center - 0.5 * size, center + 0.5 * size), dim=-1)

    left = xyxy(boxes.float())
    right = xyxy(gt_boxes.float())
    if int(right.shape[0]) == 0:
        return left.new_zeros((int(left.shape[0]), 0))
    intersection_min = torch.maximum(left[:, None, :2], right[None, :, :2])
    intersection_max = torch.minimum(left[:, None, 2:], right[None, :, 2:])
    intersection = (intersection_max - intersection_min).clamp(min=0.0)
    intersection_area = intersection[..., 0] * intersection[..., 1]
    left_area = ((left[:, 2] - left[:, 0]).clamp(min=0.0)) * (
        (left[:, 3] - left[:, 1]).clamp(min=0.0)
    )
    right_area = ((right[:, 2] - right[:, 0]).clamp(min=0.0)) * (
        (right[:, 3] - right[:, 1]).clamp(min=0.0)
    )
    union = left_area[:, None] + right_area[None, :] - intersection_area
    return intersection_area / union.clamp(min=torch.finfo(union.dtype).eps)


def validate_cached_candidate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one row and return it unchanged as a plain dictionary."""
    if not isinstance(row, Mapping):
        raise CachedCandidateContractError("cache row must be a mapping")
    task = row.get("task")
    if task not in CACHE_TASKS:
        raise CachedCandidateContractError(
            f"cache row task must be one of {CACHE_TASKS}, got {task!r}"
        )
    expected_fields = (
        _BASE_ROW_FIELDS if task == CACHE_TASK_RANK else _PAIR_ROW_FIELDS
    )
    actual_fields = set(row)
    if actual_fields != expected_fields:
        raise CachedCandidateContractError(
            "cache row fields differ from contract: "
            f"missing={sorted(expected_fields - actual_fields)}, "
            f"extra={sorted(actual_fields - expected_fields)}"
        )
    if row["schema"] != CACHE_ROW_SCHEMA:
        raise CachedCandidateContractError(
            f"cache row schema must be {CACHE_ROW_SCHEMA!r}"
        )
    _require_identifier(row["sample_id"], name="sample_id")
    _require_identifier(row["image_id"], name="image_id")
    features = _require_tensor(
        row["query_features"],
        name="query_features",
        shape=(None, CACHE_FEATURE_DIM),
        dtypes=(torch.float16, torch.float32),
    )
    query_count = int(features.shape[0])
    if query_count < 2:
        raise CachedCandidateContractError(
            "query_features must contain at least two candidates"
        )
    native = _require_tensor(
        row["native_score"],
        name="native_score",
        shape=(query_count,),
        dtypes=(torch.float16, torch.float32),
    )
    boxes = _require_tensor(
        row["boxes"],
        name="boxes",
        shape=(query_count, 4),
        dtypes=(torch.float32,),
    )
    mask = _require_tensor(
        row["candidate_mask"],
        name="candidate_mask",
        shape=(query_count,),
        dtypes=(torch.bool,),
    )
    gt_boxes = _require_tensor(
        row["gt_boxes"],
        name="gt_boxes",
        shape=(None, 4),
        dtypes=(torch.float32,),
    )
    if not bool(torch.isfinite(features).all().item()):
        raise CachedCandidateContractError("query_features must be finite")
    if not bool(torch.isfinite(native).all().item()):
        raise CachedCandidateContractError("native_score must be finite")
    if not bool(mask.any().item()):
        raise CachedCandidateContractError(
            "candidate_mask must retain at least one candidate"
        )
    _validate_normalized_boxes(boxes, name="boxes")
    _validate_normalized_boxes(gt_boxes, name="gt_boxes")

    if task == CACHE_TASK_RANK:
        if int(gt_boxes.shape[0]) == 0:
            raise CachedCandidateContractError("rank row must contain gt_boxes")
        best_iou = normalized_cxcywh_iou(boxes, gt_boxes).amax(dim=1)
        positives = mask & (best_iou >= 0.5)
        negatives = mask & (~positives)
        if not bool(positives.any().item()) or not bool(negatives.any().item()):
            raise CachedCandidateContractError(
                "rank row must contain eligible IoU>=0.5 positives and hard negatives"
            )
    else:
        _require_identifier(row["pair_id"], name="pair_id")
        if row["pair_role"] not in CACHE_PAIR_ROLES:
            raise CachedCandidateContractError(
                f"pair_role must be one of {CACHE_PAIR_ROLES}"
            )
        if row["pair_role"] == "positive" and int(gt_boxes.shape[0]) == 0:
            raise CachedCandidateContractError(
                "positive confidence-pair row must contain gt_boxes"
            )
        if row["pair_role"] == "negative" and int(gt_boxes.shape[0]) != 0:
            raise CachedCandidateContractError(
                "negative confidence-pair row must have empty gt_boxes"
            )
    return dict(row)


def validate_cached_candidate_shard(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate row identities, pair closure and the immutable shard envelope."""
    if not isinstance(payload, Mapping):
        raise CachedCandidateContractError("cache shard must be a mapping")
    actual_fields = set(payload)
    if actual_fields != _SHARD_FIELDS:
        raise CachedCandidateContractError(
            "cache shard fields differ from contract: "
            f"missing={sorted(_SHARD_FIELDS - actual_fields)}, "
            f"extra={sorted(actual_fields - _SHARD_FIELDS)}"
        )
    if payload["schema"] != CACHE_SHARD_SCHEMA:
        raise CachedCandidateContractError(
            f"cache shard schema must be {CACHE_SHARD_SCHEMA!r}"
        )
    _require_identifier(payload["shard_id"], name="shard_id")
    if payload["feature_dim"] != CACHE_FEATURE_DIM:
        raise CachedCandidateContractError(
            f"feature_dim must equal {CACHE_FEATURE_DIM}"
        )
    if payload["box_format"] != CACHE_BOX_FORMAT:
        raise CachedCandidateContractError(
            f"box_format must equal {CACHE_BOX_FORMAT!r}"
        )
    source = _validate_source(payload["source"])
    if not isinstance(payload["rows"], Sequence) or isinstance(
        payload["rows"], (str, bytes)
    ):
        raise CachedCandidateContractError("cache shard rows must be a sequence")
    rows = tuple(validate_cached_candidate_row(row) for row in payload["rows"])
    if not rows:
        raise CachedCandidateContractError("cache shard must not be empty")
    sample_ids = [row["sample_id"] for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise CachedCandidateContractError("cache sample_id values must be unique")
    rank_rows = [row for row in rows if row["task"] == CACHE_TASK_RANK]
    if not rank_rows:
        raise CachedCandidateContractError("cache shard requires rank rows")
    pairs: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["task"] == CACHE_TASK_CONFIDENCE_PAIR:
            pairs.setdefault(row["pair_id"], []).append(row)
    if not pairs:
        raise CachedCandidateContractError("cache shard requires confidence pairs")
    for pair_id, pair_rows in pairs.items():
        roles = sorted(row["pair_role"] for row in pair_rows)
        if roles != ["negative", "positive"]:
            raise CachedCandidateContractError(
                f"confidence pair {pair_id!r} must contain one positive and one negative"
            )
    return {
        "schema": payload["schema"],
        "shard_id": payload["shard_id"],
        "source": source,
        "feature_dim": payload["feature_dim"],
        "box_format": payload["box_format"],
        "rows": rows,
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def cached_candidate_content_sha256(payload: Mapping[str, Any]) -> str:
    """Hash validated semantic content, including tensor dtype/shape/bytes."""
    shard = validate_cached_candidate_shard(payload)
    digest = hashlib.sha256()

    def text(name: str, value: Any) -> None:
        encoded = json.dumps(
            [name, value], ensure_ascii=True, separators=(",", ":")
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)

    def tensor(name: str, value: Tensor) -> None:
        detached = value.detach().cpu().contiguous()
        text(f"{name}.dtype", str(detached.dtype))
        text(f"{name}.shape", list(detached.shape))
        raw = detached.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)

    for field in ("schema", "shard_id", "feature_dim", "box_format"):
        text(field, shard[field])
    for field in sorted(_SOURCE_FIELDS):
        text(f"source.{field}", shard["source"][field])
    for index, row in enumerate(shard["rows"]):
        prefix = f"rows[{index}]"
        for field in ("schema", "sample_id", "image_id", "task"):
            text(f"{prefix}.{field}", row[field])
        if row["task"] == CACHE_TASK_CONFIDENCE_PAIR:
            text(f"{prefix}.pair_id", row["pair_id"])
            text(f"{prefix}.pair_role", row["pair_role"])
        for field in (
            "query_features",
            "native_score",
            "boxes",
            "candidate_mask",
            "gt_boxes",
        ):
            tensor(f"{prefix}.{field}", row[field])
    return digest.hexdigest()


def save_cached_candidate_shard(
    payload: Mapping[str, Any], path: str | Path
) -> dict[str, str]:
    shard = validate_cached_candidate_shard(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(shard, destination)
    return {
        "file_sha256": file_sha256(destination),
        "content_sha256": cached_candidate_content_sha256(shard),
    }


def load_cached_candidate_shard(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CachedCandidateContractError(
            f"could not load cached-candidate shard {source}: {exc}"
        ) from exc
    return validate_cached_candidate_shard(payload)


def build_synthetic_cached_candidate_shard(
    path: str | Path, *, seed: int = 20260821
) -> dict[str, str]:
    """Create a tiny deterministic rank/confidence fixture for U1/U5 tests."""
    generator = torch.Generator().manual_seed(int(seed))
    base_boxes = torch.tensor(
        [
            [0.25, 0.25, 0.20, 0.20],
            [0.72, 0.25, 0.18, 0.18],
            [0.25, 0.72, 0.16, 0.16],
            [0.72, 0.72, 0.14, 0.14],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([True, True, True, False], dtype=torch.bool)

    def features(index: int, dtype: torch.dtype) -> Tensor:
        value = torch.randn(
            4, CACHE_FEATURE_DIM, generator=generator, dtype=torch.float32
        )
        value[:, index % CACHE_FEATURE_DIM] += torch.tensor(
            [2.0, -2.0, 1.0, 0.0]
        )
        return value.to(dtype=dtype).contiguous()

    def rank_row(index: int, *, teacher_correct: bool) -> dict[str, Any]:
        gt = base_boxes[index : index + 1].clone()
        native = (
            torch.tensor([0.45, 0.12, 0.05, -0.2])
            if teacher_correct and index == 0
            else torch.tensor([0.12, 0.42, 0.08, -0.2])
        )
        if index == 1:
            native = torch.tensor([0.36, 0.18, 0.09, -0.2])
        return {
            "schema": CACHE_ROW_SCHEMA,
            "sample_id": f"rank-{index}",
            "image_id": f"rank-image-{index}",
            "task": CACHE_TASK_RANK,
            "query_features": features(
                index, torch.float16 if index % 2 == 0 else torch.float32
            ),
            "native_score": native.float().contiguous(),
            "boxes": base_boxes.clone().contiguous(),
            "candidate_mask": mask.clone().contiguous(),
            "gt_boxes": gt.contiguous(),
        }

    def pair_row(pair: int, role: str) -> dict[str, Any]:
        positive = role == "positive"
        offset = 10 + 2 * pair + int(not positive)
        native = torch.tensor(
            [0.35, 0.18, 0.06, -0.15]
            if positive
            else [0.30, 0.28, 0.12, -0.15],
            dtype=torch.float32,
        )
        return {
            "schema": CACHE_ROW_SCHEMA,
            "sample_id": f"confidence-{pair}-{role}",
            "image_id": f"confidence-image-{pair}-{role}",
            "task": CACHE_TASK_CONFIDENCE_PAIR,
            "pair_id": f"pair-{pair}",
            "pair_role": role,
            "query_features": features(offset, torch.float16).contiguous(),
            "native_score": native.contiguous(),
            "boxes": base_boxes.clone().contiguous(),
            "candidate_mask": mask.clone().contiguous(),
            "gt_boxes": (
                base_boxes[:1].clone().contiguous()
                if positive
                else torch.empty(0, 4, dtype=torch.float32).contiguous()
            ),
        }

    rows = [rank_row(0, teacher_correct=True), rank_row(1, teacher_correct=False)]
    for pair_index in range(2):
        rows.extend(
            [pair_row(pair_index, "positive"), pair_row(pair_index, "negative")]
        )
    payload = {
        "schema": CACHE_SHARD_SCHEMA,
        "shard_id": "synthetic-pilot-v1",
        "source": {
            "schema": CACHE_SOURCE_SCHEMA,
            "model_id": "synthetic-frozen-candidate-fixture",
            "checkpoint_sha256": hashlib.sha256(
                b"synthetic-checkpoint-v1"
            ).hexdigest(),
            "config_sha256": hashlib.sha256(b"synthetic-config-v1").hexdigest(),
            "extractor_code_sha256": hashlib.sha256(
                b"synthetic-extractor-v1"
            ).hexdigest(),
            "query_feature_name": "decoder_query_features",
        },
        "feature_dim": CACHE_FEATURE_DIM,
        "box_format": CACHE_BOX_FORMAT,
        "rows": tuple(rows),
    }
    return save_cached_candidate_shard(payload, path)


__all__ = [
    "CACHE_BOX_FORMAT",
    "CACHE_FEATURE_DIM",
    "CACHE_PAIR_ROLES",
    "CACHE_ROW_SCHEMA",
    "CACHE_SHARD_SCHEMA",
    "CACHE_SOURCE_SCHEMA",
    "CACHE_TASK_CONFIDENCE_PAIR",
    "CACHE_TASK_RANK",
    "CACHE_TASKS",
    "CachedCandidateContractError",
    "build_synthetic_cached_candidate_shard",
    "cached_candidate_content_sha256",
    "file_sha256",
    "load_cached_candidate_shard",
    "normalized_cxcywh_iou",
    "save_cached_candidate_shard",
    "validate_cached_candidate_row",
    "validate_cached_candidate_shard",
]
