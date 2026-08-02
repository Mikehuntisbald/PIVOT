#!/usr/bin/env python3
"""Audit the worst-case CUDA runtime needed by the fixed-GDINO extractor.

This is a post-training smoke test, not an extractor.  It replays the formal
extractor's first confidence-train batch and its fixed worst deploy batch, then
writes one atomic, self-contained audit JSON.  The completed fixed baseline and
all locked semantic/holdout inputs are revalidated before the model is loaded.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import extract_stageb_fixed_gdino_top1_vlm_manifest as extraction  # noqa: E402


SCHEMA = "stage-b-fixed-gdino-top1-runtime-smoke-v1"
SMOKE_OUTPUT_ROOT = (
    REPO_ROOT / "outputs/stageb_fixed_gdino_top1_vlm_smoke_20260712"
).resolve()
DEFAULT_OUTPUT = SMOKE_OUTPUT_ROOT / "runtime_smoke.json"
TRAIN_BATCH_INDEX = 0
TRAIN_ROW_INDICES = tuple(range(0, extraction.LOCAL_BATCH_SIZE))
TRAIN_TENSOR_SHAPE = (4, 3, 800, 1333)
DEPLOY_WORST_BATCH_INDEX = 48
DEPLOY_ROW_START = DEPLOY_WORST_BATCH_INDEX * extraction.DEPLOY_BATCH_SIZE
DEPLOY_ROW_INDICES = tuple(
    range(DEPLOY_ROW_START, DEPLOY_ROW_START + extraction.DEPLOY_BATCH_SIZE)
)
DEPLOY_TENSOR_SHAPE = (16, 3, 1333, 1333)
MIN_TOTAL_HEADROOM_BYTES = 1 << 30
AMP_DTYPES = {"torch.float16", "torch.bfloat16"}

RUNTIME_CONTRACT = {
    "schema": "stage-b-fixed-gdino-top1-runtime-contract-v1",
    "seed": extraction.DEFAULT_SEED,
    "queries": extraction.EXPECTED_QUERIES,
    "score": extraction.SCORE_CONTRACT,
    "train": {
        "batch_index_zero_based": TRAIN_BATCH_INDEX,
        "row_indices_zero_based": list(TRAIN_ROW_INDICES),
        "local_batch_size": extraction.LOCAL_BATCH_SIZE,
        "paired_batch_size": extraction.PAIRED_BATCH_SIZE,
        "tensor_shape": list(TRAIN_TENSOR_SHAPE),
        "forward_order": [
            "paired_positive_then_negative",
            "separate_negative",
            "separate_positive",
        ],
        "forward_contract": extraction.PRIMARY_FORWARD_CONTRACT,
        "shadow_forward_contract": extraction.SHADOW_FORWARD_CONTRACT,
    },
    "deploy": {
        "batch_index_zero_based": DEPLOY_WORST_BATCH_INDEX,
        "row_indices_zero_based": list(DEPLOY_ROW_INDICES),
        "batch_size": extraction.DEPLOY_BATCH_SIZE,
        "tensor_shape": list(DEPLOY_TENSOR_SHAPE),
        "replayed_batch_indices_zero_based": list(
            range(0, DEPLOY_WORST_BATCH_INDEX + 1)
        ),
        "warmup_batch_indices_zero_based": list(range(0, DEPLOY_WORST_BATCH_INDEX)),
        "empty_cache_after_batch_indices_zero_based": [0],
        "forward_order": ["separate_negative", "separate_positive"],
        "forward_contract": extraction.DEPLOY_FORWARD_CONTRACT,
    },
    "cuda": {
        "amp_required": True,
        "accepted_autocast_dtypes": sorted(AMP_DTYPES),
        "minimum_total_headroom_bytes": MIN_TOTAL_HEADROOM_BYTES,
        "minimum_observed_system_free_bytes": MIN_TOTAL_HEADROOM_BYTES,
    },
}


class RuntimeSmokeError(RuntimeError):
    pass


def _resolve(path: Path | str) -> Path:
    result = Path(path).expanduser()
    if not result.is_absolute():
        result = REPO_ROOT / result
    return result.resolve()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _autocast_enabled(device_type: str) -> bool:
    try:
        return bool(torch.is_autocast_enabled(device_type))
    except TypeError:  # pragma: no cover - compatibility with older torch
        return bool(torch.is_autocast_enabled()) if device_type == "cuda" else False


def _autocast_dtype(device_type: str) -> str | None:
    try:
        return str(torch.get_autocast_dtype(device_type))
    except (AttributeError, RuntimeError, TypeError):  # pragma: no cover
        if device_type == "cuda" and hasattr(torch, "get_autocast_gpu_dtype"):
            return str(torch.get_autocast_gpu_dtype())
        return None


def _caption_record(captions: Sequence[str]) -> Dict[str, Any]:
    values = [str(value) for value in captions]
    return {
        "count": len(values),
        "canonical_sha256": extraction.canonical_sha256(values),
    }


class _AmpObservedModel:
    """Transparent callable that records AMP at the actual model-call boundary."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.observations: list[Dict[str, Any]] = []

    def __call__(self, samples: Any, *, captions: Sequence[str]):
        device_type = str(samples.tensors.device.type)
        observation: Dict[str, Any] = {
            "device_type": device_type,
            "autocast_enabled_at_model_call": _autocast_enabled(device_type),
            "autocast_dtype": _autocast_dtype(device_type),
            "input_dtype": str(samples.tensors.dtype),
            "input_shape": list(samples.tensors.shape),
            "captions": _caption_record(captions),
        }
        outputs = self.model(samples, captions=list(captions))
        token_logits, boxes, phrase_mask = extraction._model_output_tensors(outputs)
        observation["outputs"] = {
            "token_logits": {
                "shape": list(token_logits.shape),
                "dtype": str(token_logits.dtype),
                "device_type": str(token_logits.device.type),
            },
            "boxes": {
                "shape": list(boxes.shape),
                "dtype": str(boxes.dtype),
                "device_type": str(boxes.device.type),
            },
            "phrase_mask": {
                "shape": list(phrase_mask.shape),
                "dtype": str(phrase_mask.dtype),
                "device_type": str(phrase_mask.device.type),
            },
        }
        self.observations.append(observation)
        return outputs


def _tensor_float32_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def _score_report(
    label: str,
    outputs: Mapping[str, torch.Tensor],
    *,
    expected_batch: int,
) -> Dict[str, Any]:
    scores, boxes = extraction.score_model_output(outputs)
    expected_scores = (int(expected_batch), extraction.EXPECTED_QUERIES)
    expected_boxes = (int(expected_batch), extraction.EXPECTED_QUERIES, 4)
    if tuple(scores.shape) != expected_scores:
        raise RuntimeSmokeError(
            f"{label} score shape drifted: expected {expected_scores}, got {tuple(scores.shape)}"
        )
    if tuple(boxes.shape) != expected_boxes:
        raise RuntimeSmokeError(
            f"{label} box shape drifted: expected {expected_boxes}, got {tuple(boxes.shape)}"
        )
    if scores.dtype != torch.float32 or boxes.dtype != torch.float32:
        raise RuntimeSmokeError(f"{label} scores and boxes must normalize to float32")
    if not bool(torch.isfinite(scores).all().item()):
        raise RuntimeSmokeError(f"{label} scores contain non-finite values")
    if not bool(torch.isfinite(boxes).all().item()):
        raise RuntimeSmokeError(f"{label} boxes contain non-finite values")
    return {
        "label": label,
        "batch_size": int(expected_batch),
        "queries": int(scores.shape[1]),
        "scores": {
            "shape": list(scores.shape),
            "dtype": str(scores.dtype),
            "finite": True,
            "minimum": float(scores.min().item()),
            "maximum": float(scores.max().item()),
            "mean": float(scores.mean().item()),
            "top_query_ids": [int(value) for value in scores.argmax(dim=1).cpu().tolist()],
            "float32_c_order_sha256": _tensor_float32_sha256(scores),
        },
        "boxes": {
            "shape": list(boxes.shape),
            "dtype": str(boxes.dtype),
            "finite": True,
            "minimum": float(boxes.min().item()),
            "maximum": float(boxes.max().item()),
            "float32_c_order_sha256": _tensor_float32_sha256(boxes),
        },
    }


def _validate_forward_observations(
    observations: Sequence[Mapping[str, Any]],
    expected: Sequence[tuple[str, Sequence[str]]],
    *,
    require_cuda_amp: bool,
) -> list[Dict[str, Any]]:
    if len(observations) != len(expected):
        raise RuntimeSmokeError(
            f"forward count drifted: expected {len(expected)}, got {len(observations)}"
        )
    result: list[Dict[str, Any]] = []
    for index, (observed, (label, captions)) in enumerate(zip(observations, expected)):
        row = dict(observed)
        row["call_index_within_stage"] = index
        row["label"] = label
        expected_captions = _caption_record(captions)
        if row.get("captions") != expected_captions:
            raise RuntimeSmokeError(f"{label} caption/order contract drifted")
        if require_cuda_amp:
            if row.get("device_type") != "cuda":
                raise RuntimeSmokeError(f"{label} did not execute on CUDA")
            if row.get("autocast_enabled_at_model_call") is not True:
                raise RuntimeSmokeError(f"{label} did not execute inside CUDA autocast")
            if row.get("autocast_dtype") not in AMP_DTYPES:
                raise RuntimeSmokeError(
                    f"{label} used unexpected autocast dtype {row.get('autocast_dtype')!r}"
                )
        output_shapes = {
            key: value.get("shape")
            for key, value in row.get("outputs", {}).items()
            if isinstance(value, Mapping)
        }
        expected_batch = len(captions)
        token_shape = output_shapes.get("token_logits")
        box_shape = output_shapes.get("boxes")
        phrase_shape = output_shapes.get("phrase_mask")
        if (
            not isinstance(token_shape, list)
            or len(token_shape) != 3
            or token_shape[0] != expected_batch
            or token_shape[1] != extraction.EXPECTED_QUERIES
        ):
            raise RuntimeSmokeError(f"{label} raw token-logit shape drifted: {token_shape}")
        if box_shape != [expected_batch, extraction.EXPECTED_QUERIES, 4]:
            raise RuntimeSmokeError(f"{label} raw box shape drifted: {box_shape}")
        if (
            not isinstance(phrase_shape, list)
            or len(phrase_shape) not in (2, 3)
            or phrase_shape[0] != expected_batch
            or phrase_shape[-1] != token_shape[-1]
        ):
            raise RuntimeSmokeError(f"{label} raw phrase-mask shape drifted: {phrase_shape}")
        if require_cuda_amp:
            output_devices = {
                value.get("device_type")
                for value in row.get("outputs", {}).values()
                if isinstance(value, Mapping)
            }
            if output_devices != {"cuda"}:
                raise RuntimeSmokeError(f"{label} produced non-CUDA core outputs")
        result.append(row)
    return result


def _run_train_forwards(
    observed_model: _AmpObservedModel,
    samples: Any,
    positive: Sequence[str],
    negative: Sequence[str],
    *,
    require_cuda_amp: bool,
) -> Dict[str, Any]:
    start = len(observed_model.observations)
    positive_paired, negative_paired = extraction.forward_paired_pos_neg(
        observed_model, samples, positive, negative, amp=True
    )
    negative_separate = extraction.forward_separate(
        observed_model, samples, negative, amp=True
    )
    positive_separate = extraction.forward_separate(
        observed_model, samples, positive, amp=True
    )
    batch_size = len(positive)
    outputs = {
        "paired_positive": _score_report(
            "train_paired_positive", positive_paired, expected_batch=batch_size
        ),
        "paired_negative": _score_report(
            "train_paired_negative", negative_paired, expected_batch=batch_size
        ),
        "separate_negative": _score_report(
            "train_separate_negative", negative_separate, expected_batch=batch_size
        ),
        "separate_positive": _score_report(
            "train_separate_positive", positive_separate, expected_batch=batch_size
        ),
    }
    expected_calls = [
        ("paired_positive_then_negative", list(positive) + list(negative)),
        ("separate_negative", negative),
        ("separate_positive", positive),
    ]
    calls = _validate_forward_observations(
        observed_model.observations[start:],
        expected_calls,
        require_cuda_amp=require_cuda_amp,
    )
    return {"calls": calls, "outputs": outputs}


def _run_deploy_forwards(
    observed_model: _AmpObservedModel,
    samples: Any,
    positive: Sequence[str],
    negative: Sequence[str],
    *,
    require_cuda_amp: bool,
) -> Dict[str, Any]:
    start = len(observed_model.observations)
    negative_separate = extraction.forward_separate(
        observed_model, samples, negative, amp=True
    )
    positive_separate = extraction.forward_separate(
        observed_model, samples, positive, amp=True
    )
    batch_size = len(positive)
    outputs = {
        "separate_negative": _score_report(
            "deploy_separate_negative", negative_separate, expected_batch=batch_size
        ),
        "separate_positive": _score_report(
            "deploy_separate_positive", positive_separate, expected_batch=batch_size
        ),
    }
    expected_calls = [
        ("separate_negative", negative),
        ("separate_positive", positive),
    ]
    calls = _validate_forward_observations(
        observed_model.observations[start:],
        expected_calls,
        require_cuda_amp=require_cuda_amp,
    )
    return {"calls": calls, "outputs": outputs}


def _validate_batch_tensor(
    samples: Any,
    *,
    label: str,
    expected_shape: Sequence[int],
) -> Dict[str, Any]:
    shape = tuple(int(value) for value in samples.tensors.shape)
    expected = tuple(int(value) for value in expected_shape)
    if shape != expected:
        raise RuntimeSmokeError(f"{label} tensor shape drifted: expected {expected}, got {shape}")
    expected_mask = (expected[0], expected[2], expected[3])
    if samples.mask is None or tuple(samples.mask.shape) != expected_mask:
        observed_mask = None if samples.mask is None else tuple(samples.mask.shape)
        raise RuntimeSmokeError(
            f"{label} mask shape drifted: expected {expected_mask}, got {observed_mask}"
        )
    if samples.tensors.dtype != torch.float32 or samples.mask.dtype != torch.bool:
        raise RuntimeSmokeError(f"{label} input dtype contract drifted")
    if not bool(torch.isfinite(samples.tensors).all().item()):
        raise RuntimeSmokeError(f"{label} input contains non-finite values")
    return {
        "tensor_shape": list(shape),
        "tensor_dtype": str(samples.tensors.dtype),
        "mask_shape": list(samples.mask.shape),
        "mask_dtype": str(samples.mask.dtype),
        "finite": True,
    }


def _extract_captions(targets: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    from tools.stageb_eval_records import extract_adapter_tn_pair_captions

    try:
        positive, negative, valid = extract_adapter_tn_pair_captions(list(targets))
    except (TypeError, ValueError) as error:
        raise RuntimeSmokeError(f"paired caption extraction failed: {error}") from error
    if not bool(valid.all().item()) or len(positive) != len(targets) or len(negative) != len(targets):
        raise RuntimeSmokeError("paired caption extraction returned an invalid batch")
    return positive, negative


def _row_records(
    bindings: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    row_indices: Sequence[int],
    positive: Sequence[str],
    negative: Sequence[str],
    *,
    data_root: Path,
) -> list[Dict[str, Any]]:
    if not (
        len(bindings)
        == len(targets)
        == len(row_indices)
        == len(positive)
        == len(negative)
    ):
        raise RuntimeSmokeError("row-audit fields do not align")
    image_cache: Dict[Path, Dict[str, Any]] = {}
    records: list[Dict[str, Any]] = []
    for binding, target, row_index, positive_caption, negative_caption in zip(
        bindings, targets, row_indices, positive, negative
    ):
        pair = binding.get("pair")
        if not isinstance(pair, Mapping):
            raise RuntimeSmokeError(f"binding at row {row_index} has no pair")
        trace = target.get("_stageb_extraction_transform_trace")
        if not isinstance(trace, Mapping):
            raise RuntimeSmokeError(f"target at row {row_index} has no transform trace")
        image_path = extraction._image_path(binding, data_root)
        if image_path not in image_cache:
            image_cache[image_path] = extraction.file_record(image_path)
        identity: Dict[str, Any] = {
            "dataset": str(pair.get("dataset", "")),
            "sample_id": str(pair.get("sample_id", "")),
        }
        for key in ("image_id", "ann_id", "ref_id", "sent_id"):
            try:
                identity[key] = int(pair[key])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeSmokeError(f"invalid {key} at selected row {row_index}") from error
        records.append(
            {
                "row_index_zero_based": int(row_index),
                "identity": identity,
                "pair_row_sha256": str(binding.get("pair_row_sha256", "")),
                "source_file_sha256": str(binding.get("source_file_sha256", "")),
                "source_line": int(binding.get("source_line", 0)),
                "image": dict(image_cache[image_path]),
                "transform_trace": dict(trace),
                "transform_trace_sha256": extraction.canonical_sha256(trace),
                "positive_caption_sha256": extraction.canonical_sha256(positive_caption),
                "negative_caption_sha256": extraction.canonical_sha256(negative_caption),
            }
        )
    return records


def _memory_snapshot(label: str, device: torch.device) -> Dict[str, Any]:
    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "label": label,
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "system_free_bytes": int(free_bytes),
        "system_total_bytes": int(total_bytes),
    }


def _validate_memory_capacity(
    *,
    total_bytes: int,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
    minimum_system_free_bytes: int,
    minimum_headroom_bytes: int = MIN_TOTAL_HEADROOM_BYTES,
) -> Dict[str, Any]:
    values = (
        int(total_bytes),
        int(peak_allocated_bytes),
        int(peak_reserved_bytes),
        int(minimum_system_free_bytes),
        int(minimum_headroom_bytes),
    )
    if any(value < 0 for value in values):
        raise RuntimeSmokeError("memory counters must be non-negative")
    if peak_allocated_bytes > peak_reserved_bytes or peak_reserved_bytes > total_bytes:
        raise RuntimeSmokeError("CUDA peak memory counters are inconsistent")
    total_headroom = int(total_bytes) - int(peak_reserved_bytes)
    total_pass = total_headroom >= int(minimum_headroom_bytes)
    system_pass = int(minimum_system_free_bytes) >= int(minimum_headroom_bytes)
    report = {
        "total_bytes": int(total_bytes),
        "peak_allocated_bytes": int(peak_allocated_bytes),
        "peak_reserved_bytes": int(peak_reserved_bytes),
        "total_headroom_bytes": total_headroom,
        "minimum_observed_system_free_bytes": int(minimum_system_free_bytes),
        "required_headroom_bytes": int(minimum_headroom_bytes),
        "total_headroom_pass": total_pass,
        "system_free_pass": system_pass,
        "pass": total_pass and system_pass,
    }
    if not report["pass"]:
        raise RuntimeSmokeError(
            "CUDA runtime leaves less than the required 1 GiB process/system headroom: "
            f"total_headroom={total_headroom}, system_free={minimum_system_free_bytes}"
        )
    return report


def _single_loader_batch(
    dataset: Any,
    *,
    row_indices: Sequence[int],
    batch_size: int,
    seed: int,
    pin_memory: bool,
):
    from torch.utils.data import DataLoader, Subset
    from tools.eval_refcoco_stageb import _seed_worker
    from util import misc as utils

    if len(row_indices) != int(batch_size):
        raise RuntimeSmokeError("exact smoke batch must contain batch_size rows")
    subset = Subset(dataset, [int(value) for value in row_indices])
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=0,
        pin_memory=bool(pin_memory),
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    iterator = iter(loader)
    batch = next(iterator)
    try:
        next(iterator)
    except StopIteration:
        pass
    else:  # pragma: no cover - defensive assertion
        raise RuntimeSmokeError("exact smoke subset unexpectedly yielded multiple batches")
    return batch


def _prepare_batch(
    dataset: Any,
    bindings: Sequence[Mapping[str, Any]],
    *,
    row_indices: Sequence[int],
    expected_shape: Sequence[int],
    seed: int,
    data_root: Path,
    label: str,
) -> tuple[Any, list[Mapping[str, Any]], list[str], list[str], Dict[str, Any]]:
    _set_seed(seed)
    samples, raw_targets = _single_loader_batch(
        dataset,
        row_indices=row_indices,
        batch_size=len(row_indices),
        seed=seed,
        pin_memory=True,
    )
    raw_targets = list(raw_targets)
    selected_bindings = [bindings[int(index)] for index in row_indices]
    extraction._assert_batch_alignment(raw_targets, selected_bindings)
    samples, padded_targets, cyclic_indices, real = extraction._pad_nested_batch(
        samples, raw_targets, size=len(row_indices)
    )
    if real != len(row_indices) or cyclic_indices != list(range(len(row_indices))):
        raise RuntimeSmokeError(f"{label} unexpectedly required cyclic padding")
    if any(padded is not raw for padded, raw in zip(padded_targets, raw_targets)):
        raise RuntimeSmokeError(f"{label} target order changed during exact padding")
    input_report = _validate_batch_tensor(samples, label=label, expected_shape=expected_shape)
    positive, negative = _extract_captions(raw_targets)
    input_report.update(
        {
            "row_indices_zero_based": [int(value) for value in row_indices],
            "positive_captions": _caption_record(positive),
            "negative_captions": _caption_record(negative),
            "rows": _row_records(
                selected_bindings,
                raw_targets,
                row_indices,
                positive,
                negative,
                data_root=data_root,
            ),
        }
    )
    return samples, raw_targets, positive, negative, input_report


def _prepare_loaded_deploy_batch(
    samples: Any,
    raw_targets: Sequence[Mapping[str, Any]],
    bindings: Sequence[Mapping[str, Any]],
    *,
    row_indices: Sequence[int],
    expected_shape: Optional[Sequence[int]],
    data_root: Path,
    label: str,
) -> tuple[Any, list[Mapping[str, Any]], list[str], list[str], Dict[str, Any]]:
    raw_targets = list(raw_targets)
    selected_bindings = [bindings[int(index)] for index in row_indices]
    if len(raw_targets) != len(row_indices):
        raise RuntimeSmokeError(f"{label} target count drifted")
    extraction._assert_batch_alignment(raw_targets, selected_bindings)
    shape = tuple(int(value) for value in samples.tensors.shape)
    if shape[:2] != (len(row_indices), 3):
        raise RuntimeSmokeError(
            f"{label} must contain {len(row_indices)} RGB tensors, got {shape}"
        )
    input_report = _validate_batch_tensor(
        samples,
        label=label,
        expected_shape=shape if expected_shape is None else expected_shape,
    )
    positive, negative = _extract_captions(raw_targets)
    input_report.update(
        {
            "row_indices_zero_based": [int(value) for value in row_indices],
            "positive_captions": _caption_record(positive),
            "negative_captions": _caption_record(negative),
            "rows": _row_records(
                selected_bindings,
                raw_targets,
                row_indices,
                positive,
                negative,
                data_root=data_root,
            ),
        }
    )
    return samples, raw_targets, positive, negative, input_report


def _runtime_code_provenance() -> Dict[str, Any]:
    records = [
        {"role": "runtime_smoke", **extraction.file_record(Path(__file__))},
        {
            "role": "locked_extractor_core",
            **extraction.file_record(Path(extraction.__file__)),
        },
    ]
    return {"files": records, "sha256": extraction.canonical_sha256(records)}


def _validate_output_path(output: Path, *, root: Path = SMOKE_OUTPUT_ROOT) -> Path:
    output = output.expanduser()
    if not output.is_absolute():
        output = REPO_ROOT / output
    root = root.expanduser()
    if not root.is_absolute():
        root = REPO_ROOT / root
    if root.is_symlink():
        raise RuntimeSmokeError(f"runtime smoke output root must not be a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    output = output.resolve()
    if output.parent != root:
        raise RuntimeSmokeError(
            f"runtime smoke output must be a direct child of the isolated root {root}: {output}"
        )
    return output


def _prepared_file_records(value: Any) -> list[Dict[str, Any]]:
    """Extract every concrete file record embedded in prepared inputs."""

    records: Dict[str, Dict[str, Any]] = {}

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            path_value = item.get("path")
            sha_value = item.get("sha256")
            if (
                isinstance(path_value, str)
                and isinstance(sha_value, str)
                and len(sha_value) == 64
            ):
                path = Path(path_value).expanduser().resolve()
                if not path.is_file():
                    raise RuntimeSmokeError(f"locked input file is missing: {path}")
                expected = {
                    "path": str(path),
                    "sha256": sha_value,
                    "size_bytes": int(item.get("size_bytes", path.stat().st_size)),
                }
                previous = records.get(str(path))
                if previous is not None and previous != expected:
                    raise RuntimeSmokeError(
                        f"conflicting locked records for input file {path}"
                    )
                records[str(path)] = expected
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return [records[path] for path in sorted(records)]


def _capture_prepared_file_seal(prepared: Mapping[str, Any]) -> Dict[str, Any]:
    expected_records = _prepared_file_records(prepared)
    if not expected_records:
        raise RuntimeSmokeError("prepared inputs expose no locked file records")
    observed: list[Dict[str, Any]] = []
    for expected in expected_records:
        current = extraction.file_record(Path(expected["path"]))
        normalized = {
            "path": current["path"],
            "sha256": current["sha256"],
            "size_bytes": int(current["size_bytes"]),
        }
        if normalized != expected:
            raise RuntimeSmokeError(f"locked input file drifted: {expected['path']}")
        observed.append(normalized)
    return {
        "files": observed,
        "count": len(observed),
        "sha256": extraction.canonical_sha256(observed),
    }


def _copy_verified_file(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> Dict[str, Any]:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle, length=16 << 20)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
    except FileExistsError as error:
        raise RuntimeSmokeError(f"snapshot destination already exists: {destination}") from error
    record = extraction.file_record(destination)
    if (
        record["sha256"] != expected_sha256
        or int(record["size_bytes"]) != int(expected_size_bytes)
    ):
        raise RuntimeSmokeError(f"locked input changed while snapshotting: {source}")
    return {
        "source_path": str(source),
        "sha256": expected_sha256,
        "size_bytes": int(expected_size_bytes),
    }


def _snapshot_model_inputs(
    prepared_args: argparse.Namespace,
    prepared: Mapping[str, Any],
    temporary_root: Path,
) -> tuple[argparse.Namespace, Dict[str, Any]]:
    """Create verified private bytes for every file parsed by the model loader."""

    copied: list[Dict[str, Any]] = []
    snapshot_configs: Dict[str, Path] = {}
    for role, argument_name, provenance_key in (
        ("model_config", "model_config", "model_config"),
        ("data_config", "data_config", "data_config"),
    ):
        snapshot_repo = temporary_root / "config_snapshot" / role / "repo"
        provenance = prepared[provenance_key]
        config_records: Dict[str, Mapping[str, Any]] = {}
        for record in [provenance, *list(provenance.get("import_chain", []))]:
            if isinstance(record, Mapping) and isinstance(record.get("path"), str):
                config_records[str(Path(record["path"]).resolve())] = record
        for source_text in sorted(config_records):
            source = Path(source_text)
            try:
                relative = source.relative_to(REPO_ROOT)
            except ValueError as error:
                raise RuntimeSmokeError(
                    f"locked config import lies outside the repository: {source}"
                ) from error
            record = config_records[source_text]
            destination = snapshot_repo / relative
            copied_record = _copy_verified_file(
                source,
                destination,
                expected_sha256=str(record["sha256"]),
                expected_size_bytes=int(record["size_bytes"]),
            )
            copied_record.update(
                {"role": f"{role}_closure", "logical_path": str(relative)}
            )
            copied.append(copied_record)
        argument_source = Path(getattr(prepared_args, argument_name)).resolve()
        snapshot_configs[argument_name] = (
            snapshot_repo / argument_source.relative_to(REPO_ROOT)
        )

    checkpoint_record = prepared["checkpoint"]
    checkpoint_snapshot = temporary_root / "checkpoint_snapshot" / "checkpoint.pth"
    copied_checkpoint = _copy_verified_file(
        Path(prepared_args.checkpoint),
        checkpoint_snapshot,
        expected_sha256=str(checkpoint_record["sha256"]),
        expected_size_bytes=int(checkpoint_record["size_bytes"]),
    )
    copied_checkpoint.update({"role": "checkpoint", "logical_path": "checkpoint.pth"})
    copied.append(copied_checkpoint)
    copied.sort(key=lambda row: (str(row["role"]), str(row["logical_path"])))
    snapshot_args = argparse.Namespace(
        checkpoint=checkpoint_snapshot,
        model_config=snapshot_configs["model_config"],
        data_config=snapshot_configs["data_config"],
    )
    return snapshot_args, {
        "files": copied,
        "count": len(copied),
        "sha256": extraction.canonical_sha256(copied),
    }


def _write_report(output: Path, report: Mapping[str, Any]) -> Dict[str, Any]:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise RuntimeSmokeError(
                f"refusing to overwrite runtime smoke output: {output}"
            ) from error
        directory_descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return extraction.file_record(output)


def run_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    started_at = _utc_now()
    started = time.monotonic()
    output = _validate_output_path(Path(args.output))
    if output.exists():
        raise RuntimeSmokeError(f"refusing to overwrite runtime smoke output: {output}")
    if "GFLOPS_DEBUG_SHILONG" in os.environ:
        raise RuntimeSmokeError("GFLOPS_DEBUG_SHILONG is forbidden for the runtime smoke")
    if int(args.seed) != extraction.DEFAULT_SEED:
        raise RuntimeSmokeError(
            f"runtime smoke seed must remain {extraction.DEFAULT_SEED}"
        )
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeSmokeError("production runtime smoke requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeSmokeError("CUDA runtime smoke requested but CUDA is unavailable")
    torch.cuda.set_device(device)
    device_index = int(torch.cuda.current_device() if device.index is None else device.index)
    device = torch.device("cuda", device_index)
    properties = torch.cuda.get_device_properties(device)

    prepared_args = argparse.Namespace(
        checkpoint=_resolve(args.checkpoint),
        model_config=_resolve(args.model_config),
        data_config=_resolve(args.data_config),
        semantic_pairs=_resolve(args.semantic_pairs),
        semantic_audit=_resolve(args.semantic_audit),
        strict2031=_resolve(args.strict2031),
        strict1607=_resolve(args.strict1607),
    )
    prepared = extraction._prepare_locked_inputs(prepared_args)
    prepared_file_seal_before = _capture_prepared_file_seal(prepared)
    runtime_code_before = _runtime_code_provenance()
    if len(prepared["bindings"]) != extraction.FILTERED_ROWS:
        raise RuntimeSmokeError("locked filtered binding count drifted")
    if DEPLOY_ROW_INDICES[-1] >= len(prepared["bindings"]):
        raise RuntimeSmokeError("fixed worst deploy batch lies outside the locked inputs")

    from torch.utils.data import DataLoader, SequentialSampler
    from tools.eval_refcoco_stageb import _load_model, _seed_worker
    from util import misc as utils
    from util.slconfig import SLConfig

    data_root = _resolve(args.data_root)
    image_root = (data_root / "COCO/coco2014/train2014").resolve()
    if not image_root.is_dir():
        raise RuntimeSmokeError(f"COCO train image root is missing: {image_root}")
    runtime_contract = dict(RUNTIME_CONTRACT)
    runtime_contract["canonical_json"] = extraction.canonical_json(runtime_contract)
    runtime_contract["sha256"] = extraction.canonical_sha256(runtime_contract)

    memory_snapshots: list[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="stageb-top1-smoke-", dir=str(output.parent)) as temporary:
        temporary_root = Path(temporary)
        snapshot_args, model_load_snapshot = _snapshot_model_inputs(
            prepared_args, prepared, temporary_root
        )
        data_cfg = SLConfig.fromfile(str(snapshot_args.data_config))
        model_cfg = SLConfig.fromfile(str(snapshot_args.model_config))
        train_transform_contract = extraction.transform_contract_from_cfg(data_cfg)
        deploy_transform_contract = extraction.deploy_transform_contract_from_cfg(
            model_cfg
        )

        annotation = temporary_root / "filtered_semantic_pairs.jsonl"
        extraction._write_filtered_annotation(annotation, prepared["bindings"])
        annotation_record = {
            "logical_name": "filtered_semantic_pairs.jsonl",
            "ephemeral": True,
            "rows": extraction.FILTERED_ROWS,
            "size_bytes": int(annotation.stat().st_size),
            "sha256": extraction.sha256_file(annotation),
        }
        train_dataset = extraction._make_dataset(
            data_cfg, annotation, data_root, train_transform_contract, deploy=False
        )
        deploy_dataset = extraction._make_dataset(
            data_cfg, annotation, data_root, deploy_transform_contract, deploy=True
        )
        if len(train_dataset) != extraction.FILTERED_ROWS:
            raise RuntimeSmokeError("train smoke dataset row count drifted")
        if len(deploy_dataset) != extraction.FILTERED_ROWS:
            raise RuntimeSmokeError("deploy smoke dataset row count drifted")

        model = _load_model(model_cfg, str(snapshot_args.checkpoint), device)
        observed_model = _AmpObservedModel(model)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        memory_snapshots.append(_memory_snapshot("model_loaded", device))

        train_samples, train_targets, train_positive, train_negative, train_input = (
            _prepare_batch(
                train_dataset,
                prepared["bindings"],
                row_indices=TRAIN_ROW_INDICES,
                expected_shape=TRAIN_TENSOR_SHAPE,
                seed=int(args.seed),
                data_root=data_root,
                label="train_first_b4",
            )
        )
        train_samples = train_samples.to(device)
        with torch.inference_mode():
            train_forwards = _run_train_forwards(
                observed_model,
                train_samples,
                train_positive,
                train_negative,
                require_cuda_amp=True,
            )
        memory_snapshots.append(_memory_snapshot("train_first_b4_complete", device))
        del train_samples, train_targets, train_positive, train_negative
        torch.cuda.empty_cache()

        _set_seed(int(args.seed))
        deploy_generator = torch.Generator()
        deploy_generator.manual_seed(int(args.seed))
        deploy_loader = DataLoader(
            deploy_dataset,
            batch_size=extraction.DEPLOY_BATCH_SIZE,
            sampler=SequentialSampler(deploy_dataset),
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=0,
            pin_memory=True,
            worker_init_fn=_seed_worker,
            generator=deploy_generator,
        )
        deploy_warmup: list[Dict[str, Any]] = []
        deploy_input: Optional[Dict[str, Any]] = None
        deploy_forwards: Optional[Dict[str, Any]] = None
        seen_deploy_batches: list[int] = []
        cumulative_peak_allocated = 0
        cumulative_peak_reserved = 0
        for deploy_batch_index, (deploy_samples, deploy_targets) in enumerate(
            deploy_loader
        ):
            if deploy_batch_index > DEPLOY_WORST_BATCH_INDEX:
                break
            deploy_targets = list(deploy_targets)
            real = len(deploy_targets)
            row_start = deploy_batch_index * extraction.DEPLOY_BATCH_SIZE
            row_indices = tuple(range(row_start, row_start + real))
            if real != extraction.DEPLOY_BATCH_SIZE:
                raise RuntimeSmokeError(
                    f"deploy batch {deploy_batch_index} is unexpectedly partial"
                )
            expected_shape = (
                DEPLOY_TENSOR_SHAPE
                if deploy_batch_index == DEPLOY_WORST_BATCH_INDEX
                else None
            )
            (
                deploy_samples,
                deploy_targets,
                deploy_positive,
                deploy_negative,
                current_input,
            ) = _prepare_loaded_deploy_batch(
                deploy_samples,
                deploy_targets,
                prepared["bindings"],
                row_indices=row_indices,
                expected_shape=expected_shape,
                data_root=data_root,
                label=f"deploy_batch_{deploy_batch_index}_b16",
            )
            if deploy_batch_index == DEPLOY_WORST_BATCH_INDEX:
                expected_warmup = list(range(DEPLOY_WORST_BATCH_INDEX))
                if seen_deploy_batches != expected_warmup:
                    raise RuntimeSmokeError(
                        "deploy warmup batches were skipped or reordered before batch 48"
                    )
                cumulative_peak_allocated = int(
                    torch.cuda.max_memory_allocated(device)
                )
                cumulative_peak_reserved = int(
                    torch.cuda.max_memory_reserved(device)
                )
                torch.cuda.reset_peak_memory_stats(device)
                memory_snapshots.append(
                    _memory_snapshot("deploy_batch_48_measure_start", device)
                )

            deploy_samples = deploy_samples.to(device)
            with torch.inference_mode():
                current_forwards = _run_deploy_forwards(
                    observed_model,
                    deploy_samples,
                    deploy_positive,
                    deploy_negative,
                    require_cuda_amp=True,
                )
            del deploy_samples, deploy_targets, deploy_positive, deploy_negative
            seen_deploy_batches.append(deploy_batch_index)

            if deploy_batch_index < DEPLOY_WORST_BATCH_INDEX:
                emptied_cache = deploy_batch_index == 0
                if emptied_cache:
                    torch.cuda.empty_cache()
                    memory_snapshots.append(
                        _memory_snapshot("deploy_warm_batch_0_empty_cache", device)
                    )
                if deploy_batch_index == DEPLOY_WORST_BATCH_INDEX - 1:
                    memory_snapshots.append(
                        _memory_snapshot("deploy_warm_batch_47_complete", device)
                    )
                deploy_warmup.append(
                    {
                        "batch_index_zero_based": deploy_batch_index,
                        "input": current_input,
                        "forwards": current_forwards,
                        "empty_cache_after_batch": emptied_cache,
                    }
                )
            else:
                deploy_input = current_input
                deploy_forwards = current_forwards
                memory_snapshots.append(
                    _memory_snapshot("deploy_worst_b16_complete", device)
                )

        expected_replay = list(range(DEPLOY_WORST_BATCH_INDEX + 1))
        if seen_deploy_batches != expected_replay:
            raise RuntimeSmokeError(
                f"deploy replay order drifted: expected {expected_replay}, "
                f"got {seen_deploy_batches}"
            )
        if deploy_input is None or deploy_forwards is None:
            raise RuntimeSmokeError("deploy batch 48 did not complete")

        measured_peak_allocated = int(torch.cuda.max_memory_allocated(device))
        measured_peak_reserved = int(torch.cuda.max_memory_reserved(device))
        peak_allocated = max(cumulative_peak_allocated, measured_peak_allocated)
        peak_reserved = max(cumulative_peak_reserved, measured_peak_reserved)
        minimum_system_free = min(
            int(snapshot["system_free_bytes"]) for snapshot in memory_snapshots
        )
        memory = _validate_memory_capacity(
            total_bytes=int(properties.total_memory),
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
            minimum_system_free_bytes=minimum_system_free,
        )
        memory["cumulative_pre_measure_peak_allocated_bytes"] = (
            cumulative_peak_allocated
        )
        memory["cumulative_pre_measure_peak_reserved_bytes"] = cumulative_peak_reserved
        memory["batch_48_peak_allocated_bytes"] = measured_peak_allocated
        memory["batch_48_peak_reserved_bytes"] = measured_peak_reserved
        memory["snapshots"] = memory_snapshots
        del observed_model, model, train_dataset, deploy_dataset, deploy_loader

    prepared_after = extraction._prepare_locked_inputs(prepared_args)
    if prepared_after != prepared:
        raise RuntimeSmokeError(
            "locked prepared inputs changed during the runtime smoke"
        )
    prepared_file_seal_after = _capture_prepared_file_seal(prepared_after)
    if prepared_file_seal_after != prepared_file_seal_before:
        raise RuntimeSmokeError("locked input file seal changed during the runtime smoke")
    runtime_code_after = _runtime_code_provenance()
    if runtime_code_after != runtime_code_before:
        raise RuntimeSmokeError("runtime smoke code changed during execution")

    gpu = {
        "logical_device": str(device),
        "logical_device_index": device_index,
        "visible_device_count": int(torch.cuda.device_count()),
        "name": str(properties.name),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "total_memory_bytes": int(properties.total_memory),
        "multi_processor_count": int(properties.multi_processor_count),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": str(torch.__version__),
        "torch_cuda_build": str(torch.version.cuda),
        "cudnn_version": int(torch.backends.cudnn.version() or 0),
    }
    report = {
        "schema": SCHEMA,
        "kind": "completed_post_baseline_cuda_runtime_smoke",
        "pass": True,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "duration_seconds": float(time.monotonic() - started),
        "output_path": str(output),
        "contract": runtime_contract,
        "inputs": {
            "checkpoint": prepared["checkpoint"],
            "model_config": prepared["model_config"],
            "data_config": prepared["data_config"],
            "semantic": prepared["semantic"],
            "strict_manifests": prepared["strict"],
            "holdout_exclusion": prepared["exclusion"],
            "filtered_annotation": annotation_record,
            "data_root": str(data_root),
            "image_root": str(image_root),
        },
        "input_integrity": {
            "prepared_exact_replay": True,
            "file_seal_before": prepared_file_seal_before,
            "file_seal_after": prepared_file_seal_after,
            "private_model_load_snapshot": model_load_snapshot,
        },
        "code": {
            "runtime": runtime_code_before,
            "runtime_after": runtime_code_after,
            "extractor_closure": prepared["code"],
        },
        "transforms": {
            "train": train_transform_contract,
            "deploy": deploy_transform_contract,
        },
        "gpu": gpu,
        "memory": memory,
        "batches": {
            "train_first_b4": {
                "batch_index_zero_based": TRAIN_BATCH_INDEX,
                "input": train_input,
                "forwards": train_forwards,
            },
            "deploy_warmup_batches_0_through_47": deploy_warmup,
            "deploy_worst_batch_48_b16": {
                "batch_index_zero_based": DEPLOY_WORST_BATCH_INDEX,
                "input": deploy_input,
                "forwards": deploy_forwards,
            },
        },
        "assertions": {
            "completed_authoritative_fixed_baseline_replayed": True,
            "locked_semantic_and_strict_inputs_replayed": True,
            "train_first_b4_exact_shape_and_finite": True,
            "train_paired_and_separate_cuda_amp": True,
            "deploy_batches_0_through_48_replayed_in_order": True,
            "deploy_batch_0_followed_by_empty_cache": True,
            "deploy_batch_48_rows_768_through_783_exact_shape_and_finite": True,
            "deploy_negative_then_positive_separate_cuda_amp": True,
            "all_outputs_have_exactly_900_queries": True,
            "peak_memory_leaves_at_least_1_gib_total_and_system_headroom": True,
            "locked_inputs_and_code_unchanged_pre_to_post": True,
            "checkpoint_and_config_loaded_from_verified_private_snapshots": True,
        },
    }
    output_record = _write_report(output, report)
    return {"report": report, "output": output_record}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=extraction.DEFAULT_CHECKPOINT)
    parser.add_argument("--model-config", type=Path, default=extraction.BASELINE_CONFIG)
    parser.add_argument("--data-config", type=Path, default=extraction.DATA_CONFIG)
    parser.add_argument("--semantic-pairs", type=Path, default=extraction.SEMANTIC_PAIRS)
    parser.add_argument("--semantic-audit", type=Path, default=extraction.SEMANTIC_AUDIT)
    parser.add_argument(
        "--strict2031", type=Path, default=extraction.STRICT_SPECS["strict2031"]["path"]
    )
    parser.add_argument(
        "--strict1607", type=Path, default=extraction.STRICT_SPECS["strict1607"]["path"]
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "/home/user/datasets/pivot_data")),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=extraction.DEFAULT_SEED)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    try:
        result = run_smoke(parse_args(argv))
    except (RuntimeSmokeError, extraction.ExtractionError) as error:
        raise SystemExit(f"[ERROR] {error}") from error
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
