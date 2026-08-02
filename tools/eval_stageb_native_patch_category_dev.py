#!/usr/bin/env python3
"""Evaluate native full-text rank with a data-only patch category gate.

The evaluator consumes only the D1 checkpoint, its b58-derived initializer, and
the sealed row-locked D1 data.  It never reads teacher, U2, R100, P50, Stage-A,
or cached model scores.  Every result is bound to immutable input and record
hashes, and ``--verify`` replays the metric and bootstrap calculations.

Record schema v2 additionally measures category-gate mechanics against every
same-category box in the category-complete row.  The original Ref correctness
fields remain primary-instance IoU metrics; the new category fields use each
query's maximum IoU across all category-complete GT boxes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, SequentialSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataset  # noqa: E402
from main import build_model_main  # noqa: E402
from models.GroundingDINO.stage_b_gdino_score_adapter import (  # noqa: E402
    aggregate_gdino_full_expression_score,
)
from models.GroundingDINO.stage_b_native_patch_category import (  # noqa: E402
    apply_native_patch_category_gate,
)
from tools.build_stageb_native_patch_category_initializer import (  # noqa: E402
    RANDOM_TRAINABLE_PATCH_KEYS,
    _safe_load_checkpoint,
    stable_file_record,
    validate_native_patch_category_initializer_payload,
)
from tools.stageb_dependency_audit import config_import_chain  # noqa: E402
from util import box_ops  # noqa: E402
from util import misc as utils  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


RESULT_SCHEMA = "pivot.stageb.native_patch_category_dev_eval/v2"
RECORD_SCHEMA = "pivot.stageb.native_patch_category_dev_record/v2"
RECEIPT_SCHEMA = "pivot.stageb.native_patch_category_d1_receipt/v1"
ROW_SCHEMA = "pivot.stageb.native_patch_category_d1_row/v1"
VALID_SPLITS = frozenset({"dev_screen", "dev_full"})
VALID_SOURCES = frozenset({"refcoco", "refcocoplus", "refcocog"})
EXPECTED_QUERIES = 900
DEFAULT_GATE_MAX_GAP = 3.0
DEFAULT_GATE_CLIP = 5.0
DEFAULT_BOOTSTRAP_ITERATIONS = 2000
DEFAULT_BOOTSTRAP_SEED = 20260724
CATEGORY_POSITIVE_IOU_THRESHOLD = 0.5
CATEGORY_NEGATIVE_IOU_THRESHOLD = 0.3
CATEGORY_STATES = frozenset({"positive", "negative", "neutral"})


class NativePatchCategoryDevEvalError(RuntimeError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite constant {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise NativePatchCategoryDevEvalError(
            f"could not parse {label}: {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise NativePatchCategoryDevEvalError(f"{label} must be a JSON object")
    return value


def _atomic_write_bytes(path: Path, payload: bytes, *, overwrite: bool) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise NativePatchCategoryDevEvalError(
            f"refusing to overwrite existing output: {path}"
        )
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(
    path: Path, payload: Mapping[str, Any], *, overwrite: bool
) -> None:
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("ascii") + b"\n"
    _atomic_write_bytes(path, serialized, overwrite=overwrite)


def _write_records(
    path: Path, records: Sequence[Mapping[str, Any]], *, overwrite: bool
) -> dict[str, Any]:
    payload = b"".join(_canonical_bytes(record) + b"\n" for record in records)
    _atomic_write_bytes(path, payload, overwrite=overwrite)
    return stable_file_record(path, label="D1 dev evaluation records")


def _read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="ascii") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    raise NativePatchCategoryDevEvalError(
                        f"records line {line_number} is empty"
                    )
                value = json.loads(
                    raw,
                    parse_constant=lambda token: (_ for _ in ()).throw(
                        ValueError(f"non-finite constant {token}")
                    ),
                )
                if not isinstance(value, dict):
                    raise NativePatchCategoryDevEvalError(
                        f"records line {line_number} is not an object"
                    )
                records.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise NativePatchCategoryDevEvalError(
            f"could not parse records {path}: {error}"
        ) from error
    validate_records(records)
    return records


def _require_exact_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or int(value) < int(minimum):
        raise NativePatchCategoryDevEvalError(
            f"{label} must be an exact integer >= {minimum}"
        )
    return int(value)


def _require_finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NativePatchCategoryDevEvalError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise NativePatchCategoryDevEvalError(f"{label} must be finite")
    return result


def validate_config(cfg: Any) -> dict[str, Any]:
    expected = {
        "stage_b_native_patch_category": True,
        "enable_patch_branch": True,
        "patch_gate_with_text": False,
        "stage_b_gdino_score_adapter": False,
        "stage_b_u0_patch_rank": False,
        "stage_b_data_driven_score": False,
        "patch_only": False,
        "stage_b": False,
        "fix_size": False,
        "data_aug_train_deterministic_aspect_resize": True,
        "strong_aug": False,
        "data_aug_hflip_prob": 0.0,
        "data_aug_scales": [800],
        "data_aug_max_size": 1333,
        "num_queries": EXPECTED_QUERIES,
    }
    drift = {
        key: {"observed": getattr(cfg, key, None), "expected": expected_value}
        for key, expected_value in expected.items()
        if getattr(cfg, key, None) != expected_value
    }
    if drift:
        raise NativePatchCategoryDevEvalError(
            f"D1 evaluation config drifted: {drift}"
        )
    return expected


def load_receipt_binding(
    receipt_path: Path, *, split: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if split not in VALID_SPLITS:
        raise NativePatchCategoryDevEvalError(f"invalid D1 dev split: {split!r}")
    receipt_path = receipt_path.expanduser().resolve(strict=True)
    receipt_record = stable_file_record(receipt_path, label="D1 data receipt")
    receipt = _strict_json_object(receipt_path, label="D1 data receipt")
    canonical = receipt.get("canonical_payload_sha256")
    replay = dict(receipt)
    replay.pop("canonical_payload_sha256", None)
    if canonical != _sha256_bytes(_canonical_bytes(replay)):
        raise NativePatchCategoryDevEvalError(
            "D1 data receipt canonical hash drifted"
        )
    invariants = receipt.get("invariants")
    split_record = receipt.get("splits", {}).get(split)
    output = split_record.get("output") if isinstance(split_record, Mapping) else None
    if not (
        receipt.get("schema") == RECEIPT_SCHEMA
        and isinstance(invariants, Mapping)
        and bool(invariants)
        and all(value is True for value in invariants.values())
        and isinstance(split_record, Mapping)
        and isinstance(output, Mapping)
    ):
        raise NativePatchCategoryDevEvalError("D1 data receipt contract drifted")
    manifest_path_value = output.get("path")
    if not isinstance(manifest_path_value, str) or not manifest_path_value:
        raise NativePatchCategoryDevEvalError("D1 receipt has no manifest path")
    manifest_path = Path(manifest_path_value).expanduser().resolve(strict=True)
    manifest_record = stable_file_record(manifest_path, label=f"D1 {split} manifest")
    if manifest_record != {
        "path": str(manifest_path),
        "size_bytes": output.get("size_bytes"),
        "sha256": output.get("sha256"),
    }:
        raise NativePatchCategoryDevEvalError(
            f"D1 {split} manifest file binding drifted"
        )
    rows = _require_exact_int(output.get("rows"), label=f"{split} rows", minimum=1)
    groups = _require_exact_int(
        split_record.get("groups"), label=f"{split} groups", minimum=1
    )
    unique_images = _require_exact_int(
        split_record.get("unique_images"),
        label=f"{split} unique images",
        minimum=1,
    )
    if rows != 3 * groups or split_record.get("rows") != rows:
        raise NativePatchCategoryDevEvalError(
            f"D1 {split} row/group geometry drifted"
        )
    normalized = {
        "split": split,
        "rows": rows,
        "groups": groups,
        "unique_images": unique_images,
        "canonical_payload_sha256": canonical,
    }
    return normalized, receipt_record, manifest_record


def load_dataset_entry(
    datasets_path: Path,
    *,
    split: str,
    receipt_path: Path,
    receipt_record: Mapping[str, Any],
    manifest_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    datasets_path = datasets_path.expanduser().resolve(strict=True)
    datasets_record = stable_file_record(datasets_path, label="D1 datasets config")
    payload = _strict_json_object(datasets_path, label="D1 datasets config")
    train = payload.get("train")
    if not isinstance(train, list) or len(train) != 1 or not isinstance(train[0], dict):
        raise NativePatchCategoryDevEvalError(
            "D1 datasets config must expose exactly one train template"
        )
    entry = copy.deepcopy(train[0])
    required = {
        "dataset_mode": "patch_episode",
        "root": "/",
        "native_patch_category_row_locked_support": True,
        "stage_b_native_patch_category_variant": "d1",
        "neg_episode_prob": 0.0,
        "support_num_patches_min": 1,
        "support_num_patches_max": 1,
        "support_patch_use_embedding": False,
        "build_text_token_masks": True,
        "strict_sample_identity": True,
        "anno_cache": False,
        "anno_cache_write": False,
    }
    drift = {
        key: {"observed": entry.get(key), "expected": expected}
        for key, expected in required.items()
        if type(entry.get(key)) is not type(expected) or entry.get(key) != expected
    }
    if drift or entry.get("support_patch_tsv") not in {None, ""}:
        raise NativePatchCategoryDevEvalError(
            f"D1 datasets template drifted: {drift}"
        )
    entry.update(
        {
            "anno": manifest_record["path"],
            "stage_b_native_patch_category_manifest_sha256": manifest_record[
                "sha256"
            ],
            "stage_b_native_patch_category_receipt": str(
                receipt_path.expanduser().resolve(strict=True)
            ),
            "stage_b_native_patch_category_receipt_sha256": receipt_record[
                "sha256"
            ],
            "stage_b_native_patch_category_split": split,
        }
    )
    return entry, datasets_record


def validate_metadata(
    rows: Sequence[Mapping[str, Any]], binding: Mapping[str, Any]
) -> list[dict[str, Any]]:
    expected_rows = _require_exact_int(binding.get("rows"), label="bound rows", minimum=1)
    if len(rows) != expected_rows:
        raise NativePatchCategoryDevEvalError(
            f"dataset exposes {len(rows)} rows, expected {expected_rows}"
        )
    normalized: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise NativePatchCategoryDevEvalError(f"metadata row {index} is invalid")
        group_id = row.get("native_patch_category_group_id")
        source = row.get("native_patch_category_source_dataset")
        variant = row.get("native_patch_category_variant_index")
        image_id = row.get("image_id")
        coco_split = row.get("category_complete_coco_split")
        class_id = row.get("category_complete_coco_category_id")
        instances = row.get("instances")
        declared_instance_count = row.get("category_complete_instance_count")
        instance_class_ids = (
            [instance.get("class_id") for instance in instances]
            if isinstance(instances, list)
            and all(isinstance(instance, Mapping) for instance in instances)
            else []
        )
        primary_instances = (
            [
                instance
                for instance in instances
                if instance.get("category_complete_primary") is True
            ]
            if isinstance(instances, list)
            and all(isinstance(instance, Mapping) for instance in instances)
            else []
        )
        if not (
            row.get("stage_b_native_patch_category_d1") is True
            and row.get("stage_b_native_patch_category_d1_schema") == ROW_SCHEMA
            and isinstance(group_id, str)
            and bool(group_id)
            and source in VALID_SOURCES
            and type(variant) is int
            and variant in {0, 1, 2}
            and type(image_id) is int
            and isinstance(coco_split, str)
            and bool(coco_split)
            and type(class_id) is int
            and isinstance(instances, list)
            and bool(instances)
            and type(declared_instance_count) is int
            and declared_instance_count == len(instances)
            and len(instance_class_ids) == len(instances)
            and all(type(value) is int for value in instance_class_ids)
            and len(set(instance_class_ids)) == 1
            and len(primary_instances) == 1
        ):
            raise NativePatchCategoryDevEvalError(
                f"metadata row {index} lost its D1 identity contract"
            )
        identity: dict[str, int] = {}
        for key in ("ann_id", "ref_id", "sent_id"):
            value = row.get(key)
            if type(value) is not int:
                raise NativePatchCategoryDevEvalError(
                    f"metadata row {index} has invalid {key}"
                )
            identity[key] = int(value)
        item = {
            "row_index": index,
            "group_id": group_id,
            "image_cluster": f"{coco_split}:{image_id}",
            "image_id": int(image_id),
            "class_id": int(class_id),
            "support_class_id": int(instance_class_ids[0]),
            "source_dataset": str(source),
            "variant_index": int(variant),
            **identity,
        }
        normalized.append(item)
        grouped[group_id].append(item)
    for group_id, members in grouped.items():
        if (
            len(members) != 3
            or [member["variant_index"] for member in members] != [0, 1, 2]
            or len({member["image_cluster"] for member in members}) != 1
            or len({member["class_id"] for member in members}) != 1
            or len({member["support_class_id"] for member in members}) != 1
        ):
            raise NativePatchCategoryDevEvalError(
                f"D1 group {group_id} is not one ordered three-row unit"
            )
    if len(grouped) != binding.get("groups"):
        raise NativePatchCategoryDevEvalError("D1 metadata group count drifted")
    if len({row["image_cluster"] for row in normalized}) != binding.get(
        "unique_images"
    ):
        raise NativePatchCategoryDevEvalError("D1 metadata image count drifted")
    return normalized


def _clean_model_state(
    payload: Mapping[str, Any], *, label: str
) -> dict[str, Tensor]:
    value = payload.get("model")
    if not isinstance(value, Mapping) or not value:
        raise NativePatchCategoryDevEvalError(f"{label} has no model state")
    state = dict(utils.clean_state_dict(value))
    invalid = [
        key
        for key, tensor in state.items()
        if not isinstance(key, str) or not torch.is_tensor(tensor)
    ]
    if invalid:
        raise NativePatchCategoryDevEvalError(
            f"{label} contains non-tensor state: {invalid[:8]}"
        )
    return state


def audit_checkpoint(
    model: torch.nn.Module,
    initializer_payload: Mapping[str, Any],
    checkpoint_payload: Mapping[str, Any],
    *,
    initializer_label: str,
    checkpoint_label: str,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    validate_native_patch_category_initializer_payload(
        model, initializer_payload, checkpoint_label=initializer_label
    )
    initializer_state = _clean_model_state(
        initializer_payload, label=initializer_label
    )
    checkpoint_state = _clean_model_state(checkpoint_payload, label=checkpoint_label)
    if set(checkpoint_state) != set(initializer_state):
        missing = sorted(set(initializer_state).difference(checkpoint_state))
        unexpected = sorted(set(checkpoint_state).difference(initializer_state))
        raise NativePatchCategoryDevEvalError(
            "D1 checkpoint model coverage drifted: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    changed: list[str] = []
    for key, expected in initializer_state.items():
        observed = checkpoint_state[key]
        if observed.dtype != expected.dtype or tuple(observed.shape) != tuple(
            expected.shape
        ):
            raise NativePatchCategoryDevEvalError(
                f"D1 checkpoint tensor shape/dtype drifted at {key}"
            )
        if not torch.equal(observed, expected):
            changed.append(key)
    illegal = sorted(set(changed).difference(RANDOM_TRAINABLE_PATCH_KEYS))
    if illegal:
        raise NativePatchCategoryDevEvalError(
            f"D1 checkpoint changed frozen tensors: {illegal[:8]}"
        )
    if any(
        key.startswith(
            (
                "stage_b_gdino_score_adapter.",
                "stage_b_u0_patch_rank_adapter.",
                "stage_b_data_driven_score_heads.",
            )
        )
        for key in checkpoint_state
    ):
        raise NativePatchCategoryDevEvalError(
            "D1 checkpoint unexpectedly contains an old score adapter"
        )
    audit = {
        "model_tensor_count": len(checkpoint_state),
        "allowed_trainable_tensor_count": len(RANDOM_TRAINABLE_PATCH_KEYS),
        "changed_tensor_count": len(changed),
        "changed_tensor_keys": sorted(changed),
        "all_frozen_tensors_bitwise_equal_initializer": True,
        "only_eight_declared_patch_projection_tensors_may_change": True,
        "no_old_score_adapter_tensors": True,
    }
    return checkpoint_state, audit


def _target_identity(target: Mapping[str, Any], key: str) -> int:
    value = target.get(key)
    if not torch.is_tensor(value) or value.numel() != 1:
        raise NativePatchCategoryDevEvalError(f"target has no scalar {key}")
    return int(value.detach().cpu().item())


def _full_expression_mask(
    outputs: Mapping[str, Any], expected_mask: Tensor, *, token_count: int
) -> Tensor:
    observed = outputs.get("phrase_to_token_mask")
    if not torch.is_tensor(observed):
        raise NativePatchCategoryDevEvalError(
            "model output has no phrase_to_token_mask"
        )
    observed = observed.to(device=expected_mask.device, dtype=torch.bool)
    expected_mask = expected_mask.to(dtype=torch.bool)
    if observed.dim() != 3 or tuple(observed.shape) != tuple(expected_mask.shape):
        raise NativePatchCategoryDevEvalError(
            "model phrase_to_token_mask differs from the row-locked target mask"
        )
    if not torch.equal(observed, expected_mask):
        raise NativePatchCategoryDevEvalError(
            "model phrase_to_token_mask changed the row-locked expression mask"
        )
    if int(observed.shape[-1]) != int(token_count):
        raise NativePatchCategoryDevEvalError(
            "expression mask does not align with pred_logits_text"
        )
    if int(observed.shape[1]) != 1 or bool((~observed.any(dim=-1)).any().item()):
        raise NativePatchCategoryDevEvalError(
            "D1 evaluation requires one non-empty complete expression per row"
        )
    return observed.any(dim=1)


def _category_state(max_iou: float) -> str:
    if float(max_iou) >= CATEGORY_POSITIVE_IOU_THRESHOLD:
        return "positive"
    if float(max_iou) < CATEGORY_NEGATIVE_IOU_THRESHOLD:
        return "negative"
    return "neutral"


def evaluate_batch(
    outputs: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    metadata: Sequence[Mapping[str, Any]],
    *,
    expected_phrase_mask: Tensor,
    gate_max_gap: float,
    gate_clip: float,
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    text_logits = outputs.get("pred_logits_text")
    patch_logits = outputs.get("pred_logits_patch")
    pred_boxes = outputs.get("pred_boxes")
    if not all(torch.is_tensor(value) for value in (text_logits, patch_logits, pred_boxes)):
        raise NativePatchCategoryDevEvalError(
            "D1 model output lacks text, patch, or box query tensors"
        )
    if (
        text_logits.dim() != 3
        or pred_boxes.dim() != 3
        or int(pred_boxes.shape[-1]) != 4
        or tuple(text_logits.shape[:2]) != tuple(pred_boxes.shape[:2])
        or int(text_logits.shape[1]) != EXPECTED_QUERIES
        or len(targets) != int(text_logits.shape[0])
        or len(metadata) != len(targets)
    ):
        raise NativePatchCategoryDevEvalError("D1 query/batch geometry drifted")
    expression_mask = _full_expression_mask(
        outputs, expected_phrase_mask, token_count=int(text_logits.shape[-1])
    )
    native_score = aggregate_gdino_full_expression_score(
        text_logits, expression_mask
    )
    candidate_mask = torch.ones_like(native_score, dtype=torch.bool)
    adapted_score, eligible, standardized_patch = apply_native_patch_category_gate(
        native_score,
        patch_logits,
        candidate_mask,
        max_gap=float(gate_max_gap),
        clip=float(gate_clip),
    )
    base_query = native_score.argmax(dim=1)
    adapted_query = adapted_score.argmax(dim=1)
    pred_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes.float())
    records: list[dict[str, Any]] = []
    for batch_index, (target, meta) in enumerate(zip(targets, metadata)):
        for key in ("image_id", "ann_id", "ref_id", "sent_id"):
            if _target_identity(target, key) != int(meta[key]):
                raise NativePatchCategoryDevEvalError(
                    f"target/metadata identity mismatch at row {meta['row_index']}: {key}"
                )
        boxes = target.get("boxes")
        labels = target.get("labels")
        support_class = target.get("support_class")
        primary = target.get("primary_instance_mask")
        if not (
            torch.is_tensor(boxes)
            and boxes.is_floating_point()
            and boxes.dim() == 2
            and int(boxes.shape[1]) == 4
            and int(boxes.shape[0]) >= 1
            and bool(torch.isfinite(boxes).all().item())
            and torch.is_tensor(labels)
            and labels.dtype == torch.int64
            and tuple(labels.shape) == (int(boxes.shape[0]),)
            and torch.is_tensor(support_class)
            and support_class.dtype == torch.int64
            and support_class.numel() == 1
            and bool((labels == support_class.reshape(-1)[0]).all().item())
            and int(support_class.reshape(-1)[0].item())
            == int(meta["support_class_id"])
            and torch.is_tensor(primary)
            and primary.dtype == torch.bool
            and tuple(primary.shape) == (int(boxes.shape[0]),)
            and int(primary.sum().item()) == 1
        ):
            raise NativePatchCategoryDevEvalError(
                f"row {meta['row_index']} has no exact same-category complete GT set"
            )
        gt_xyxy = box_ops.box_cxcywh_to_xyxy(boxes.float())
        all_category_ious = box_ops.box_iou(
            pred_xyxy[batch_index], gt_xyxy
        )[0]
        if tuple(all_category_ious.shape) != (
            EXPECTED_QUERIES,
            int(boxes.shape[0]),
        ) or not bool(torch.isfinite(all_category_ious).all().item()):
            raise NativePatchCategoryDevEvalError(
                f"row {meta['row_index']} category IoU geometry drifted"
            )
        category_max_iou = all_category_ious.amax(dim=1)
        primary_index = int(primary.nonzero(as_tuple=False)[0, 0].item())
        primary_ious = all_category_ious[:, primary_index]
        category_positive = category_max_iou >= CATEGORY_POSITIVE_IOU_THRESHOLD
        category_negative = category_max_iou < CATEGORY_NEGATIVE_IOU_THRESHOLD
        category_neutral = ~(category_positive | category_negative)
        if not bool(
            (
                category_positive.to(torch.int8)
                + category_negative.to(torch.int8)
                + category_neutral.to(torch.int8)
                == 1
            ).all().item()
        ):
            raise NativePatchCategoryDevEvalError(
                f"row {meta['row_index']} category query states do not partition"
            )
        base_index = int(base_query[batch_index].item())
        adapted_index = int(adapted_query[batch_index].item())
        base_iou = float(primary_ious[base_index].item())
        adapted_iou = float(primary_ious[adapted_index].item())
        base_correct = bool(base_iou >= float(iou_threshold))
        adapted_correct = bool(adapted_iou >= float(iou_threshold))
        row_standardized_patch = standardized_patch[batch_index]
        best_standardized_patch = row_standardized_patch.amax()
        patch_gap = best_standardized_patch - row_standardized_patch
        if not bool(torch.isfinite(patch_gap).all().item()):
            raise NativePatchCategoryDevEvalError(
                f"row {meta['row_index']} patch gaps are not finite"
            )
        exists_category_positive = bool(category_positive.any().item())
        native_best_positive_index: int | None = None
        native_best_positive_eligible: bool | None = None
        native_best_positive_patch_gap: float | None = None
        native_best_positive_max_iou: float | None = None
        if exists_category_positive:
            native_best_positive_index = int(
                native_score[batch_index]
                .masked_fill(~category_positive, -torch.inf)
                .argmax()
                .item()
            )
            native_best_positive_eligible = bool(
                eligible[batch_index, native_best_positive_index].item()
            )
            native_best_positive_patch_gap = float(
                patch_gap[native_best_positive_index].item()
            )
            native_best_positive_max_iou = float(
                category_max_iou[native_best_positive_index].item()
            )
        base_category_max_iou = float(category_max_iou[base_index].item())
        adapted_category_max_iou = float(category_max_iou[adapted_index].item())
        record = {
            "schema": RECORD_SCHEMA,
            "row_index": int(meta["row_index"]),
            "group_id": str(meta["group_id"]),
            "image_cluster": str(meta["image_cluster"]),
            "image_id": int(meta["image_id"]),
            "class_id": int(meta["class_id"]),
            "support_class_id": int(meta["support_class_id"]),
            "source_dataset": str(meta["source_dataset"]),
            "variant_index": int(meta["variant_index"]),
            "ann_id": int(meta["ann_id"]),
            "ref_id": int(meta["ref_id"]),
            "sent_id": int(meta["sent_id"]),
            "base_query": base_index,
            "adapted_query": adapted_index,
            "base_iou": base_iou,
            "adapted_iou": adapted_iou,
            "base_correct": base_correct,
            "adapted_correct": adapted_correct,
            "fixed": bool(not base_correct and adapted_correct),
            "regressed": bool(base_correct and not adapted_correct),
            "winner_changed": bool(base_index != adapted_index),
            "eligible_queries": int(eligible[batch_index].sum().item()),
            "base_query_eligible": bool(eligible[batch_index, base_index].item()),
            "adapted_native_score": float(native_score[batch_index, adapted_index].item()),
            "base_native_score": float(native_score[batch_index, base_index].item()),
            "adapted_standardized_patch_score": float(
                standardized_patch[batch_index, adapted_index].item()
            ),
            "category_gt_count": int(boxes.shape[0]),
            "category_positive_query_count": int(category_positive.sum().item()),
            "category_negative_query_count": int(category_negative.sum().item()),
            "category_neutral_query_count": int(category_neutral.sum().item()),
            "exists_category_positive": exists_category_positive,
            "native_winner_category_state": _category_state(
                base_category_max_iou
            ),
            "native_winner_category_max_iou": base_category_max_iou,
            "native_winner_category_eligible": bool(
                eligible[batch_index, base_index].item()
            ),
            "native_winner_patch_gap": float(patch_gap[base_index].item()),
            "adapted_winner_category_state": _category_state(
                adapted_category_max_iou
            ),
            "adapted_winner_category_max_iou": adapted_category_max_iou,
            "adapted_winner_category_eligible": bool(
                eligible[batch_index, adapted_index].item()
            ),
            "adapted_winner_patch_gap": float(
                patch_gap[adapted_index].item()
            ),
            "native_best_positive_query": native_best_positive_index,
            "native_best_positive_query_max_iou": native_best_positive_max_iou,
            "native_best_positive_query_eligible": (
                native_best_positive_eligible
            ),
            "native_best_positive_query_patch_gap": (
                native_best_positive_patch_gap
            ),
        }
        records.append(record)
    return records


def validate_records(records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise NativePatchCategoryDevEvalError("D1 evaluation records are empty")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for expected_index, record in enumerate(records):
        if not isinstance(record, Mapping) or record.get("schema") != RECORD_SCHEMA:
            raise NativePatchCategoryDevEvalError(
                f"invalid D1 evaluation record {expected_index}"
            )
        if record.get("row_index") != expected_index:
            raise NativePatchCategoryDevEvalError(
                "D1 evaluation record order is not sequential"
            )
        source = record.get("source_dataset")
        variant = record.get("variant_index")
        group_id = record.get("group_id")
        if not (
            source in VALID_SOURCES
            and type(variant) is int
            and variant in {0, 1, 2}
            and isinstance(group_id, str)
            and bool(group_id)
            and isinstance(record.get("image_cluster"), str)
            and bool(record["image_cluster"])
            and type(record.get("support_class_id")) is int
            and record["support_class_id"] >= 0
        ):
            raise NativePatchCategoryDevEvalError(
                f"D1 evaluation record {expected_index} identity drifted"
            )
        for key in ("base_correct", "adapted_correct", "fixed", "regressed"):
            if type(record.get(key)) is not bool:
                raise NativePatchCategoryDevEvalError(
                    f"D1 record {expected_index} has invalid {key}"
                )
        base = record["base_correct"]
        adapted = record["adapted_correct"]
        if record["fixed"] != (not base and adapted) or record["regressed"] != (
            base and not adapted
        ):
            raise NativePatchCategoryDevEvalError(
                f"D1 record {expected_index} transition flags drifted"
            )
        for key in ("base_query", "adapted_query"):
            value = record.get(key)
            if type(value) is not int or not 0 <= value < EXPECTED_QUERIES:
                raise NativePatchCategoryDevEvalError(
                    f"D1 record {expected_index} has invalid {key}"
                )
        if record.get("winner_changed") != (
            record["base_query"] != record["adapted_query"]
        ):
            raise NativePatchCategoryDevEvalError(
                f"D1 record {expected_index} winner-change flag drifted"
            )
        eligible_queries = record.get("eligible_queries")
        if (
            type(eligible_queries) is not int
            or not 1 <= eligible_queries <= EXPECTED_QUERIES
        ):
            raise NativePatchCategoryDevEvalError(
                f"D1 record {expected_index} has invalid eligible query count"
            )
        for key in (
            "base_query_eligible",
            "exists_category_positive",
            "native_winner_category_eligible",
            "adapted_winner_category_eligible",
        ):
            if type(record.get(key)) is not bool:
                raise NativePatchCategoryDevEvalError(
                    f"D1 record {expected_index} has invalid {key}"
                )
        if (
            record["native_winner_category_eligible"]
            != record["base_query_eligible"]
            or record["adapted_winner_category_eligible"] is not True
        ):
            raise NativePatchCategoryDevEvalError(
                f"D1 record {expected_index} winner eligibility drifted"
            )
        category_gt_count = record.get("category_gt_count")
        state_counts = {
            state: record.get(f"category_{state}_query_count")
            for state in CATEGORY_STATES
        }
        if (
            type(category_gt_count) is not int
            or category_gt_count < 1
            or any(
                type(value) is not int or value < 0
                for value in state_counts.values()
            )
            or sum(state_counts.values()) != EXPECTED_QUERIES
            or record["exists_category_positive"]
            != (state_counts["positive"] > 0)
        ):
            raise NativePatchCategoryDevEvalError(
                f"D1 record {expected_index} category state counts drifted"
            )
        for prefix in ("native_winner", "adapted_winner"):
            state = record.get(f"{prefix}_category_state")
            max_iou = record.get(f"{prefix}_category_max_iou")
            patch_gap = record.get(f"{prefix}_patch_gap")
            if (
                state not in CATEGORY_STATES
                or isinstance(max_iou, bool)
                or not isinstance(max_iou, (int, float))
                or not math.isfinite(float(max_iou))
                or not 0.0 <= float(max_iou) <= 1.0
                or state != _category_state(float(max_iou))
                or isinstance(patch_gap, bool)
                or not isinstance(patch_gap, (int, float))
                or not math.isfinite(float(patch_gap))
                or float(patch_gap) < 0.0
            ):
                raise NativePatchCategoryDevEvalError(
                    f"D1 record {expected_index} has invalid {prefix} mechanism state"
                )
            if state_counts[state] < 1:
                raise NativePatchCategoryDevEvalError(
                    f"D1 record {expected_index} {prefix} state is absent from counts"
                )
        if (
            record["base_correct"]
            and record["native_winner_category_state"] != "positive"
        ) or (
            record["adapted_correct"]
            and record["adapted_winner_category_state"] != "positive"
        ):
            raise NativePatchCategoryDevEvalError(
                f"D1 record {expected_index} primary/category correctness drifted"
            )
        best_positive_values = (
            record.get("native_best_positive_query"),
            record.get("native_best_positive_query_max_iou"),
            record.get("native_best_positive_query_eligible"),
            record.get("native_best_positive_query_patch_gap"),
        )
        if record["exists_category_positive"]:
            query, max_iou, query_eligible, patch_gap = best_positive_values
            if (
                type(query) is not int
                or not 0 <= query < EXPECTED_QUERIES
                or isinstance(max_iou, bool)
                or not isinstance(max_iou, (int, float))
                or not math.isfinite(float(max_iou))
                or not CATEGORY_POSITIVE_IOU_THRESHOLD <= float(max_iou) <= 1.0
                or type(query_eligible) is not bool
                or isinstance(patch_gap, bool)
                or not isinstance(patch_gap, (int, float))
                or not math.isfinite(float(patch_gap))
                or float(patch_gap) < 0.0
            ):
                raise NativePatchCategoryDevEvalError(
                    f"D1 record {expected_index} native-best-positive state drifted"
                )
            if record["native_winner_category_state"] == "positive" and (
                query != record["base_query"]
                or float(max_iou)
                != float(record["native_winner_category_max_iou"])
                or query_eligible
                != record["native_winner_category_eligible"]
                or float(patch_gap) != float(record["native_winner_patch_gap"])
            ):
                raise NativePatchCategoryDevEvalError(
                    f"D1 record {expected_index} native-positive winner is not "
                    "the native-best positive query"
                )
        elif any(value is not None for value in best_positive_values):
            raise NativePatchCategoryDevEvalError(
                f"D1 record {expected_index} has a positive query without category recall"
            )
        grouped[group_id].append(record)
    for group_id, members in grouped.items():
        if (
            len(members) != 3
            or [member["variant_index"] for member in members] != [0, 1, 2]
            or len({member["image_cluster"] for member in members}) != 1
            or len({member["class_id"] for member in members}) != 1
            or len({member["support_class_id"] for member in members}) != 1
        ):
            raise NativePatchCategoryDevEvalError(
                f"record group {group_id} is not an ordered D1 unit"
            )


def _binary_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = len(records)
    if rows == 0:
        raise NativePatchCategoryDevEvalError("cannot summarize zero records")
    base = sum(bool(record["base_correct"]) for record in records)
    adapted = sum(bool(record["adapted_correct"]) for record in records)
    fixed = sum(bool(record["fixed"]) for record in records)
    regressed = sum(bool(record["regressed"]) for record in records)
    return {
        "rows": rows,
        "base_correct": base,
        "adapted_correct": adapted,
        "base_accuracy": base / rows,
        "adapted_accuracy": adapted / rows,
        "delta": (adapted - base) / rows,
        "fixed": fixed,
        "regressed": regressed,
        "net_fixed": fixed - regressed,
        "unchanged": rows - fixed - regressed,
    }


def _unit_summary(
    records: Sequence[Mapping[str, Any]], *, key: str, label: str
) -> dict[str, Any]:
    units: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        units[str(record[key])].append(record)
    base_macro = 0.0
    adapted_macro = 0.0
    improved = 0
    regressed = 0
    tied = 0
    for members in units.values():
        base = sum(bool(row["base_correct"]) for row in members) / len(members)
        adapted = sum(bool(row["adapted_correct"]) for row in members) / len(members)
        base_macro += base
        adapted_macro += adapted
        if adapted > base:
            improved += 1
        elif adapted < base:
            regressed += 1
        else:
            tied += 1
    count = len(units)
    return {
        label: count,
        "base_macro_accuracy": base_macro / count,
        "adapted_macro_accuracy": adapted_macro / count,
        "delta": (adapted_macro - base_macro) / count,
        "improved": improved,
        "regressed": regressed,
        "tied": tied,
    }


def _rate_summary(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "rate": (float(numerator) / int(denominator)) if denominator else None,
    }


def _category_mechanism_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    native_states = {
        state: sum(
            row["native_winner_category_state"] == state for row in records
        )
        for state in sorted(CATEGORY_STATES)
    }
    adapted_states = {
        state: sum(
            row["adapted_winner_category_state"] == state for row in records
        )
        for state in sorted(CATEGORY_STATES)
    }
    native_positive = native_states["positive"]
    native_positive_eligible = sum(
        row["native_winner_category_state"] == "positive"
        and row["native_winner_category_eligible"]
        for row in records
    )
    native_negative = native_states["negative"]
    native_negative_rejected = sum(
        row["native_winner_category_state"] == "negative"
        and not row["native_winner_category_eligible"]
        for row in records
    )
    category_fixable = sum(row["exists_category_positive"] for row in records)
    category_fixable_adapted_positive = sum(
        row["exists_category_positive"]
        and row["adapted_winner_category_state"] == "positive"
        for row in records
    )
    best_positive_eligible = sum(
        row["exists_category_positive"]
        and row["native_best_positive_query_eligible"]
        for row in records
    )
    eligible_counts = sorted(int(row["eligible_queries"]) for row in records)
    best_positive_gaps = sorted(
        float(row["native_best_positive_query_patch_gap"])
        for row in records
        if row["exists_category_positive"]
    )
    return {
        "definition": {
            "per_query_category_iou": (
                "maximum IoU across every same-category category-complete GT box"
            ),
            "positive": "max_category_iou_gte_0.5",
            "negative": "max_category_iou_lt_0.3",
            "neutral": "0.3_lte_max_category_iou_lt_0.5",
            "category_fixable": "at_least_one_category_positive_query",
        },
        "native_winner_state_counts": native_states,
        "adapted_winner_state_counts": adapted_states,
        "neutral_winner_counts": {
            "native": native_states["neutral"],
            "adapted": adapted_states["neutral"],
        },
        "rows_without_category_positive": len(records) - category_fixable,
        "positive_winner_eligibility": _rate_summary(
            native_positive_eligible, native_positive
        ),
        "negative_winner_rejection": _rate_summary(
            native_negative_rejected, native_negative
        ),
        "category_fixable_adapted_positive_recall": _rate_summary(
            category_fixable_adapted_positive, category_fixable
        ),
        "native_best_positive_query_eligibility": _rate_summary(
            best_positive_eligible, category_fixable
        ),
        "native_best_positive_query_patch_gap": {
            "rows": len(best_positive_gaps),
            "mean": (
                sum(best_positive_gaps) / len(best_positive_gaps)
                if best_positive_gaps
                else None
            ),
            "q05": _quantile(best_positive_gaps, 0.05)
            if best_positive_gaps
            else None,
            "q50": _quantile(best_positive_gaps, 0.50)
            if best_positive_gaps
            else None,
            "q95": _quantile(best_positive_gaps, 0.95)
            if best_positive_gaps
            else None,
        },
        "eligible_queries": {
            "rows": len(eligible_counts),
            "min": eligible_counts[0],
            "max": eligible_counts[-1],
            "mean": sum(eligible_counts) / len(eligible_counts),
            "q01": _quantile(eligible_counts, 0.01),
            "q05": _quantile(eligible_counts, 0.05),
            "q50": _quantile(eligible_counts, 0.50),
            "q95": _quantile(eligible_counts, 0.95),
            "q99": _quantile(eligible_counts, 0.99),
        },
    }


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    validate_records(records)
    by_source: dict[str, Any] = {}
    for source in sorted(VALID_SOURCES):
        source_rows = [row for row in records if row["source_dataset"] == source]
        if source_rows:
            by_source[source] = _binary_summary(source_rows)
    return {
        "rows": _binary_summary(records),
        "groups": _unit_summary(records, key="group_id", label="groups"),
        "images": _unit_summary(records, key="image_cluster", label="images"),
        "sources": by_source,
        "category_mechanism": _category_mechanism_summary(records),
    }


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise NativePatchCategoryDevEvalError("cannot quantile an empty sequence")
    position = (len(sorted_values) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower]
        + fraction * (sorted_values[upper] - sorted_values[lower])
    )


def deterministic_bootstrap(
    records: Sequence[Mapping[str, Any]],
    *,
    unit_key: str,
    unit_name: str,
    iterations: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    validate_records(records)
    iterations = _require_exact_int(
        iterations, label="bootstrap iterations", minimum=1
    )
    if type(seed) is not int:
        raise NativePatchCategoryDevEvalError(
            "bootstrap seed must be an exact integer"
        )
    if not 0.0 < float(confidence) < 1.0:
        raise NativePatchCategoryDevEvalError("bootstrap confidence is invalid")
    units: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        units[str(record[unit_key])].append(record)
    unit_ids = sorted(units)
    values = [
        sum(
            int(row["adapted_correct"]) - int(row["base_correct"])
            for row in units[unit_id]
        )
        / len(units[unit_id])
        for unit_id in unit_ids
    ]
    derived_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{unit_name}".encode("ascii")).digest()[:8],
        byteorder="big",
        signed=False,
    )
    rng = random.Random(derived_seed)
    count = len(values)
    replicates = [
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(iterations)
    ]
    replicates.sort()
    tail = (1.0 - float(confidence)) / 2.0
    return {
        "unit": unit_name,
        "unit_count": count,
        "iterations": iterations,
        "seed": int(seed),
        "derived_seed": derived_seed,
        "confidence": float(confidence),
        "point_estimate": sum(values) / count,
        "ci_lower": _quantile(replicates, tail),
        "ci_upper": _quantile(replicates, 1.0 - tail),
        "method": "deterministic_percentile_macro_cluster_bootstrap_v1",
    }


def bootstrap_records(
    records: Sequence[Mapping[str, Any]], *, iterations: int, seed: int
) -> dict[str, Any]:
    return {
        "group": deterministic_bootstrap(
            records,
            unit_key="group_id",
            unit_name="group",
            iterations=iterations,
            seed=seed,
        ),
        "image": deterministic_bootstrap(
            records,
            unit_key="image_cluster",
            unit_name="image",
            iterations=iterations,
            seed=seed,
        ),
    }


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _stack_phrase_masks(
    targets: Sequence[Mapping[str, Any]], device: torch.device
) -> Tensor:
    masks = [target.get("phrase_to_token_mask") for target in targets]
    if not masks or any(
        not torch.is_tensor(mask)
        or mask.dtype != torch.bool
        or mask.dim() != 2
        or int(mask.shape[0]) != 1
        for mask in masks
    ):
        raise NativePatchCategoryDevEvalError(
            "every D1 row requires one boolean phrase_to_token_mask"
        )
    shapes = {tuple(mask.shape) for mask in masks}
    if len(shapes) != 1:
        raise NativePatchCategoryDevEvalError(
            "D1 phrase masks must share one token geometry"
        )
    return torch.stack(masks, dim=0).to(device=device, non_blocking=True)


def _input_records(
    *,
    config_path: Path,
    datasets_path: Path,
    receipt_path: Path,
    manifest_path: Path,
    initializer_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    chain = config_import_chain(config_path, root=REPO_ROOT)
    return {
        "config": stable_file_record(config_path, label="D1 model config"),
        "config_import_chain": [
            stable_file_record(path, label=f"D1 config dependency {path.name}")
            for path in chain
        ],
        "datasets": stable_file_record(datasets_path, label="D1 datasets config"),
        "data_receipt": stable_file_record(receipt_path, label="D1 data receipt"),
        "manifest": stable_file_record(manifest_path, label="D1 dev manifest"),
        "initializer": stable_file_record(
            initializer_path, label="D1 initializer"
        ),
        "checkpoint": stable_file_record(checkpoint_path, label="D1 checkpoint"),
    }


def replay_checkpoint_audit(
    config_path: Path,
    initializer_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Rebuild the model and replay the tensor audit without inference."""
    cfg = SLConfig.fromfile(str(config_path))
    validate_config(cfg)
    cfg.device = "cpu"
    cfg.distributed = False
    model, _criterion, _postprocessors = build_model_main(cfg)
    initializer_payload = _safe_load_checkpoint(
        initializer_path, label="D1 initializer verification"
    )
    checkpoint_payload = _safe_load_checkpoint(
        checkpoint_path, label="D1 checkpoint verification"
    )
    _state, audit = audit_checkpoint(
        model,
        initializer_payload,
        checkpoint_payload,
        initializer_label="D1 initializer verification",
        checkpoint_label="D1 checkpoint verification",
    )
    return audit


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if "GFLOPS_DEBUG_SHILONG" in os.environ:
        raise NativePatchCategoryDevEvalError("GFLOPS_DEBUG_SHILONG is forbidden")
    split = str(args.split)
    config_path = args.config.expanduser().resolve(strict=True)
    datasets_path = args.datasets.expanduser().resolve(strict=True)
    receipt_path = args.receipt.expanduser().resolve(strict=True)
    initializer_path = args.initializer.expanduser().resolve(strict=True)
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    output_path = args.output_json.expanduser().resolve()
    records_path = (
        args.records_jsonl.expanduser().resolve()
        if args.records_jsonl is not None
        else output_path.with_suffix(".records.jsonl")
    )
    if output_path == records_path:
        raise NativePatchCategoryDevEvalError(
            "summary and records outputs must be different files"
        )
    batch_size = _require_exact_int(args.batch_size, label="batch size", minimum=1)
    num_workers = _require_exact_int(
        args.num_workers, label="num workers", minimum=0
    )
    gate_max_gap = _require_finite_float(
        args.gate_max_gap, label="category gate max gap"
    )
    gate_clip = _require_finite_float(args.gate_clip, label="category gate clip")
    if gate_max_gap < 0.0 or gate_clip <= 0.0:
        raise NativePatchCategoryDevEvalError("category gate settings are invalid")
    binding, receipt_record, manifest_record = load_receipt_binding(
        receipt_path, split=split
    )
    dataset_entry, datasets_record = load_dataset_entry(
        datasets_path,
        split=split,
        receipt_path=receipt_path,
        receipt_record=receipt_record,
        manifest_record=manifest_record,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise NativePatchCategoryDevEvalError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    cfg = SLConfig.fromfile(str(config_path))
    config_contract = validate_config(cfg)
    cfg.device = str(device)
    cfg.distributed = False
    seed = _require_exact_int(args.seed, label="evaluation seed", minimum=0)
    _seed_everything(seed)
    model, _criterion, _postprocessors = build_model_main(cfg)
    initializer_payload = _safe_load_checkpoint(
        initializer_path, label="D1 initializer"
    )
    checkpoint_payload = _safe_load_checkpoint(
        checkpoint_path, label="D1 checkpoint"
    )
    checkpoint_state, checkpoint_audit = audit_checkpoint(
        model,
        initializer_payload,
        checkpoint_payload,
        initializer_label=f"D1 initializer {initializer_path}",
        checkpoint_label=f"D1 checkpoint {checkpoint_path}",
    )
    model.load_state_dict(checkpoint_state, strict=True)

    # The dataset's fail-closed binding currently names this path "train";
    # deterministic aspect resize makes it the deployment 800/max1333 geometry.
    dataset = build_dataset(image_set="train", args=cfg, datasetinfo=dataset_entry)
    metas = getattr(dataset, "metas", None)
    if not isinstance(metas, list):
        raise NativePatchCategoryDevEvalError(
            "D1 patch-episode dataset does not expose ordered metadata"
        )
    metadata = validate_metadata(metas, binding)
    if len(dataset) != binding["rows"] or getattr(dataset, "sample_weights", None) is not None:
        raise NativePatchCategoryDevEvalError("D1 evaluation dataset geometry drifted")
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
        persistent_workers=bool(num_workers > 0),
    )

    model.to(device).eval()
    effective_amp = bool(not args.no_amp and device.type == "cuda")
    records: list[dict[str, Any]] = []
    cursor = 0
    with torch.inference_mode():
        for batch_index, (samples, raw_targets) in enumerate(loader):
            raw_targets = list(raw_targets)
            batch_metadata = metadata[cursor : cursor + len(raw_targets)]
            samples = samples.to(device)
            captions: list[str] = []
            for local_index, target in enumerate(raw_targets):
                caption = target.get("caption")
                expressions = target.get("cap_list")
                if not (
                    isinstance(caption, str)
                    and bool(caption.strip())
                    and isinstance(expressions, (list, tuple))
                    and len(expressions) == 1
                    and isinstance(expressions[0], str)
                    and bool(expressions[0].strip())
                ):
                    raise NativePatchCategoryDevEvalError(
                        f"row {cursor + local_index} lost its full expression"
                    )
                captions.append(caption)
            if not all(torch.is_tensor(target.get("patch")) for target in raw_targets):
                raise NativePatchCategoryDevEvalError(
                    "D1 row-locked pixel support is missing"
                )
            patches = torch.stack(
                [target["patch"] for target in raw_targets], dim=0
            ).to(device=device, non_blocking=True)
            phrase_mask = _stack_phrase_masks(raw_targets, device)
            targets = [
                {
                    key: value.to(device=device, non_blocking=True)
                    for key, value in target.items()
                    if torch.is_tensor(value) and key != "patch"
                }
                for target in raw_targets
            ]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=effective_amp,
            ):
                outputs = model(
                    samples,
                    captions=captions,
                    patches=patches,
                    phrase_to_token_mask=phrase_mask,
                )
            batch_records = evaluate_batch(
                outputs,
                targets,
                batch_metadata,
                expected_phrase_mask=phrase_mask,
                gate_max_gap=gate_max_gap,
                gate_clip=gate_clip,
            )
            records.extend(batch_records)
            cursor += len(batch_records)
            if int(args.log_every) > 0 and (
                (batch_index + 1) % int(args.log_every) == 0
                or cursor == binding["rows"]
            ):
                print(
                    f"D1 {split}: {cursor}/{binding['rows']} rows",
                    flush=True,
                )
    if cursor != binding["rows"]:
        raise NativePatchCategoryDevEvalError(
            f"D1 evaluation consumed {cursor} rows, expected {binding['rows']}"
        )
    validate_records(records)
    metrics = summarize_records(records)
    bootstrap = bootstrap_records(
        records,
        iterations=int(args.bootstrap_iterations),
        seed=int(args.bootstrap_seed),
    )
    records_record = _write_records(
        records_path, records, overwrite=bool(args.overwrite)
    )
    inputs = _input_records(
        config_path=config_path,
        datasets_path=datasets_path,
        receipt_path=receipt_path,
        manifest_path=Path(manifest_record["path"]),
        initializer_path=initializer_path,
        checkpoint_path=checkpoint_path,
    )
    if inputs["datasets"] != datasets_record or inputs["data_receipt"] != receipt_record:
        raise NativePatchCategoryDevEvalError(
            "D1 inputs changed during evaluation"
        )
    code = {
        "evaluator": stable_file_record(
            Path(__file__), label="D1 dev evaluator code"
        ),
        "native_aggregator": stable_file_record(
            REPO_ROOT
            / "models/GroundingDINO/stage_b_gdino_score_adapter.py",
            label="native score aggregation code",
        ),
        "patch_gate": stable_file_record(
            REPO_ROOT
            / "models/GroundingDINO/stage_b_native_patch_category.py",
            label="D1 patch gate code",
        ),
    }
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "evaluation": {
            "split": split,
            "native_score": (
                "mean(sigmoid(pred_logits_text)) over every token selected by "
                "phrase_to_token_mask.any(dim=1)"
            ),
            "adapted_score": "native rank with D1 patch-category eligibility gate",
            "candidate_mask": "all_900_queries_valid",
            "gate_max_gap": gate_max_gap,
            "gate_clip": gate_clip,
            "top1_correctness": "primary_gt_iou_gte_0.5",
        },
        "inputs": inputs,
        "code": code,
        "config_contract": config_contract,
        "data_binding": binding,
        "checkpoint_audit": checkpoint_audit,
        "runtime": {
            "device": str(device),
            "amp": effective_amp,
            "seed": seed,
            "sampler": "sequential",
            "dataset_image_set_route": "train",
            "dataset_transform": "deterministic_aspect_800_max1333",
            "batch_size": batch_size,
            "num_workers": num_workers,
            "batches": len(loader),
            "rows": cursor,
        },
        "metrics": metrics,
        "bootstrap": bootstrap,
        "outputs": {"records": records_record},
        "provenance": {
            "teacher_scores_or_logits_consumed": False,
            "u2_r100_p50_or_stagea_scores_consumed": False,
            "only_checkpoint_forward_scores_consumed": True,
            "native_baseline_and_adapted_share_one_forward": True,
        },
    }
    result["canonical_payload_sha256"] = _sha256_bytes(_canonical_bytes(result))
    _atomic_write_json(output_path, result, overwrite=bool(args.overwrite))
    return {
        "result": result,
        "result_file": stable_file_record(output_path, label="D1 dev result"),
        "records_file": records_record,
    }


def _verify_file_record(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise NativePatchCategoryDevEvalError(f"saved {label} record is malformed")
    path_value = value.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise NativePatchCategoryDevEvalError(f"saved {label} path is invalid")
    observed = stable_file_record(Path(path_value), label=label)
    if observed != dict(value):
        raise NativePatchCategoryDevEvalError(f"saved {label} hash binding drifted")
    return observed


def verify_result(args: argparse.Namespace) -> dict[str, Any]:
    output_path = args.output_json.expanduser().resolve(strict=True)
    result_record = stable_file_record(output_path, label="D1 dev result")
    result = _strict_json_object(output_path, label="D1 dev result")
    canonical = result.get("canonical_payload_sha256")
    replay = dict(result)
    replay.pop("canonical_payload_sha256", None)
    if canonical != _sha256_bytes(_canonical_bytes(replay)):
        raise NativePatchCategoryDevEvalError("D1 result canonical hash drifted")
    if result.get("schema") != RESULT_SCHEMA or result.get("status") != "complete":
        raise NativePatchCategoryDevEvalError("D1 result schema/status drifted")
    evaluation = result.get("evaluation")
    if not isinstance(evaluation, Mapping) or evaluation.get("split") != args.split:
        raise NativePatchCategoryDevEvalError("D1 result belongs to another split")
    if (
        evaluation.get("gate_max_gap") != float(args.gate_max_gap)
        or evaluation.get("gate_clip") != float(args.gate_clip)
    ):
        raise NativePatchCategoryDevEvalError(
            "D1 result category-gate settings differ from verification CLI"
        )
    inputs = result.get("inputs")
    if not isinstance(inputs, Mapping):
        raise NativePatchCategoryDevEvalError("D1 result inputs are missing")
    expected_paths = {
        "config": args.config.expanduser().resolve(strict=True),
        "datasets": args.datasets.expanduser().resolve(strict=True),
        "data_receipt": args.receipt.expanduser().resolve(strict=True),
        "initializer": args.initializer.expanduser().resolve(strict=True),
        "checkpoint": args.checkpoint.expanduser().resolve(strict=True),
    }
    verified_inputs: dict[str, Any] = {}
    for key, expected_path in expected_paths.items():
        observed = _verify_file_record(inputs.get(key), label=f"D1 input {key}")
        if Path(observed["path"]) != expected_path:
            raise NativePatchCategoryDevEvalError(
                f"D1 result belongs to another {key}"
            )
        verified_inputs[key] = observed
    chain = inputs.get("config_import_chain")
    if not isinstance(chain, list) or not chain:
        raise NativePatchCategoryDevEvalError(
            "D1 result config import-chain binding is missing"
        )
    expected_chain = config_import_chain(
        expected_paths["config"], root=REPO_ROOT
    )
    if len(chain) != len(expected_chain):
        raise NativePatchCategoryDevEvalError(
            "D1 result config import chain length drifted"
        )
    verified_chain = []
    for index, (record, expected_path) in enumerate(zip(chain, expected_chain)):
        observed = _verify_file_record(
            record, label=f"D1 config dependency {index}"
        )
        if Path(observed["path"]) != expected_path.resolve(strict=True):
            raise NativePatchCategoryDevEvalError(
                "D1 result config import chain path/order drifted"
            )
        verified_chain.append(observed)
    verified_inputs["config_import_chain"] = verified_chain
    binding, receipt_record, manifest_record = load_receipt_binding(
        expected_paths["data_receipt"], split=str(args.split)
    )
    if verified_inputs["data_receipt"] != receipt_record:
        raise NativePatchCategoryDevEvalError("D1 result receipt binding drifted")
    if _verify_file_record(inputs.get("manifest"), label="D1 dev manifest") != manifest_record:
        raise NativePatchCategoryDevEvalError("D1 result manifest binding drifted")
    if result.get("data_binding") != binding:
        raise NativePatchCategoryDevEvalError("D1 result data summary drifted")
    outputs = result.get("outputs")
    if not isinstance(outputs, Mapping):
        raise NativePatchCategoryDevEvalError("D1 result outputs are missing")
    records_record = _verify_file_record(
        outputs.get("records"), label="D1 dev records"
    )
    if args.records_jsonl is not None and Path(records_record["path"]) != args.records_jsonl.expanduser().resolve(strict=True):
        raise NativePatchCategoryDevEvalError(
            "D1 result belongs to another records output"
        )
    records = _read_records(Path(records_record["path"]))
    if len(records) != binding["rows"]:
        raise NativePatchCategoryDevEvalError("D1 records do not cover the split")
    metrics = summarize_records(records)
    runtime = result.get("runtime")
    bootstrap_saved = result.get("bootstrap")
    if not isinstance(runtime, Mapping) or not isinstance(bootstrap_saved, Mapping):
        raise NativePatchCategoryDevEvalError("D1 runtime/bootstrap is missing")
    group_bootstrap = bootstrap_saved.get("group")
    if not isinstance(group_bootstrap, Mapping):
        raise NativePatchCategoryDevEvalError("D1 group bootstrap is missing")
    if (
        group_bootstrap.get("iterations") != int(args.bootstrap_iterations)
        or group_bootstrap.get("seed") != int(args.bootstrap_seed)
    ):
        raise NativePatchCategoryDevEvalError(
            "D1 bootstrap settings differ from verification CLI"
        )
    replay_bootstrap = bootstrap_records(
        records,
        iterations=_require_exact_int(
            group_bootstrap.get("iterations"),
            label="saved bootstrap iterations",
            minimum=1,
        ),
        seed=_require_exact_int(
            group_bootstrap.get("seed"), label="saved bootstrap seed", minimum=0
        ),
    )
    if result.get("metrics") != metrics or bootstrap_saved != replay_bootstrap:
        raise NativePatchCategoryDevEvalError(
            "D1 result metrics/bootstrap do not replay from bound records"
        )
    code = result.get("code")
    if not isinstance(code, Mapping):
        raise NativePatchCategoryDevEvalError("D1 result code binding is missing")
    expected_code_paths = {
        "evaluator": Path(__file__).resolve(),
        "native_aggregator": (
            REPO_ROOT
            / "models/GroundingDINO/stage_b_gdino_score_adapter.py"
        ).resolve(strict=True),
        "patch_gate": (
            REPO_ROOT
            / "models/GroundingDINO/stage_b_native_patch_category.py"
        ).resolve(strict=True),
    }
    for key, expected_path in expected_code_paths.items():
        observed = _verify_file_record(code.get(key), label=f"D1 code {key}")
        if Path(observed["path"]) != expected_path:
            raise NativePatchCategoryDevEvalError(
                f"D1 code {key} path drifted"
            )
    replayed_checkpoint_audit = replay_checkpoint_audit(
        expected_paths["config"],
        expected_paths["initializer"],
        expected_paths["checkpoint"],
    )
    if result.get("checkpoint_audit") != replayed_checkpoint_audit:
        raise NativePatchCategoryDevEvalError(
            "D1 checkpoint tensor audit does not replay"
        )
    return {
        "schema": RESULT_SCHEMA,
        "verified": True,
        "result_file": result_record,
        "records_file": records_record,
        "input_hashes": verified_inputs,
        "canonical_payload_sha256": canonical,
        "split": args.split,
        "metrics": metrics,
        "bootstrap": replay_bootstrap,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--datasets", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--initializer", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=sorted(VALID_SPLITS))
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--records-jsonl", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate-max-gap", type=float, default=DEFAULT_GATE_MAX_GAP)
    parser.add_argument("--gate-clip", type=float, default=DEFAULT_GATE_CLIP)
    parser.add_argument(
        "--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify an existing result and replay metrics without model inference",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = verify_result(args) if args.verify else run_evaluation(args)
        print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    except (NativePatchCategoryDevEvalError, OSError, ValueError, KeyError) as error:
        raise SystemExit(f"[FAIL] {error}") from error


if __name__ == "__main__":
    main()
