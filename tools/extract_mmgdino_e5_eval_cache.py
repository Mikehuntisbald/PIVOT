#!/usr/bin/env python3
"""Extract frozen MM-GDINO e5 candidates for RefCOCO or paired TN evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.extract_mmgdino_responsibility_cache import (
    ExtractionRequest,
    MMDetectionFrozenRuntime,
    MMGroundingDinoExtractionError,
    QUERY_FEATURE_NAME,
    _bound_image_path,
    _cache_row,
    _image_set_sha256,
    _load_image_identity,
    _normalized_gt_box,
    _strict_json_loads,
    _validate_request_identities,
    extractor_code_sha256,
    parse_refcoco_rank_requests,
    validate_pinned_runtime_assets,
)
from tools.responsibility_isolation_cache import (
    CACHE_BOX_FORMAT,
    CACHE_FEATURE_DIM,
    CACHE_SOURCE_SCHEMA,
    CACHE_TASK_CONFIDENCE_PAIR,
    CACHE_TASK_RANK,
    file_sha256,
    normalized_cxcywh_iou,
    validate_cached_candidate_row,
)


RECEIPT_SCHEMA = "arrow.mmgdino_e5_ownership.eval_cache_receipt/v1"
EVAL_CACHE_SCHEMA = "arrow.mmgdino_e5_ownership.eval_cache/v1"
MODES = ("ref", "tn")


def _read_jsonl(path: Path, expected_sha256: str) -> list[Mapping[str, Any]]:
    if file_sha256(path) != expected_sha256:
        raise MMGroundingDinoExtractionError("evaluation JSONL SHA drifted")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.endswith("\n") or not raw.strip():
                raise MMGroundingDinoExtractionError(
                    f"evaluation JSONL line {line_number} is malformed"
                )
            rows.append(
                _strict_json_loads(raw, location=f"{path}:{line_number}")
            )
    if not rows:
        raise MMGroundingDinoExtractionError("evaluation JSONL is empty")
    return rows


def _identifier(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise MMGroundingDinoExtractionError(f"{name} must be trimmed string")
    return value


def parse_tn_eval_requests(
    rows: Sequence[Mapping[str, Any]],
    *,
    image_root: str | Path,
) -> list[ExtractionRequest]:
    cache: dict[Path, tuple[int, int, str]] = {}
    requests = []
    for index, row in enumerate(rows):
        location = f"tn_eval[{index}]"
        # Historical strict manifests store both ``filename`` and
        # ``file_name`` as the same absolute path, whereas calibration stores
        # ``file_name`` as a basename.  The shared path binder deliberately
        # rejects traversal in ``file_name``.  Canonicalize only after proving
        # that both fields name the same file; no fallback or search is used.
        bound_row = dict(row)
        file_name = bound_row.get("file_name")
        filename = bound_row.get("filename")
        if isinstance(file_name, str) and Path(file_name).name != file_name:
            if not isinstance(filename, str) or Path(filename).name != Path(file_name).name:
                raise MMGroundingDinoExtractionError(
                    f"{location}.filename and file_name basenames disagree"
                )
            bound_row["file_name"] = Path(file_name).name
        sample_id = _identifier(row.get("sample_id"), name=f"{location}.sample_id")
        image_id = row.get("image_id")
        if isinstance(image_id, bool) or not isinstance(image_id, int) or image_id < 0:
            raise MMGroundingDinoExtractionError(
                f"{location}.image_id must be nonnegative int"
            )
        positive = row.get("positive_phrase", row.get("sent"))
        negative = row.get("negative_phrase", row.get("try_tn"))
        positive = _identifier(positive, name=f"{location}.positive_phrase")
        negative = _identifier(negative, name=f"{location}.negative_phrase")
        image_path_value = _bound_image_path(
            bound_row, location=location, image_root=image_root
        )
        image_path, width, height, image_sha = _load_image_identity(
            str(image_path_value), image_id=image_id, cache=cache
        )
        bbox = row.get("target_bbox_used", row.get("bbox", row.get("box")))
        gt_boxes = _normalized_gt_box(
            bbox, width=width, height=height, name=f"{location}.bbox"
        )
        pair_id = f"eval:{sample_id}"
        common = {
            "image_id": str(image_id),
            "image_path": image_path,
            "image_sha256": image_sha,
            "task": CACHE_TASK_CONFIDENCE_PAIR,
            "pair_id": pair_id,
        }
        requests.extend(
            [
                ExtractionRequest(
                    sample_id=f"{pair_id}:positive",
                    caption=positive,
                    gt_boxes=gt_boxes,
                    pair_role="positive",
                    **common,
                ),
                ExtractionRequest(
                    sample_id=f"{pair_id}:negative",
                    caption=negative,
                    gt_boxes=torch.empty(0, 4, dtype=torch.float32).contiguous(),
                    pair_role="negative",
                    **common,
                ),
            ]
        )
    _validate_request_identities(requests)
    return requests


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
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


def _atomic_torch(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(dict(value), temporary_name)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def extract_eval_cache(
    *,
    mode: str,
    input_jsonl: Path,
    input_sha256: str,
    image_root: Path,
    surface: str,
    mmdet_root: Path,
    mmdet_commit: str,
    config_path: Path,
    config_sha256: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    model_id: str,
    output: Path,
    receipt: Path,
    device: str,
) -> dict[str, Any]:
    if mode not in MODES:
        raise MMGroundingDinoExtractionError(f"mode must be one of {MODES}")
    if output.exists() or receipt.exists():
        raise MMGroundingDinoExtractionError("evaluation cache output already exists")
    rows = _read_jsonl(input_jsonl, input_sha256)
    if mode == "ref":
        requests = parse_refcoco_rank_requests(rows, image_root=image_root)
    else:
        requests = parse_tn_eval_requests(rows, image_root=image_root)
    assets = validate_pinned_runtime_assets(
        mmdet_root=mmdet_root,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_mmdet_commit=mmdet_commit,
        expected_config_path=config_path,
        expected_config_sha256=config_sha256,
    )
    extractor_sha = extractor_code_sha256(Path(__file__))
    runtime = MMDetectionFrozenRuntime(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        feature_dtype=torch.float32,
    )
    cache_rows = []
    oracle_rows = 0
    try:
        for request in requests:
            if file_sha256(request.image_path) != request.image_sha256:
                raise MMGroundingDinoExtractionError(
                    f"image bytes changed: {request.image_path}"
                )
            cache_row = _cache_row(
                request, runtime.infer(request.image_path, request.caption)
            )
            if request.task == CACHE_TASK_RANK:
                iou = normalized_cxcywh_iou(
                    cache_row["boxes"], cache_row["gt_boxes"]
                ).amax(dim=1)
                oracle_rows += int(
                    bool((cache_row["candidate_mask"] & (iou >= 0.5)).any().item())
                )
            cache_rows.append(cache_row)
    finally:
        runtime.close()
    cache_rows = tuple(validate_cached_candidate_row(row) for row in cache_rows)
    expected_task = CACHE_TASK_RANK if mode == "ref" else CACHE_TASK_CONFIDENCE_PAIR
    if {row["task"] for row in cache_rows} != {expected_task}:
        raise MMGroundingDinoExtractionError("evaluation cache task drifted")
    payload = {
        "schema": EVAL_CACHE_SCHEMA,
        "surface": surface,
        "task": expected_task,
        "source": {
            "schema": CACHE_SOURCE_SCHEMA,
            "model_id": model_id,
            "checkpoint_sha256": assets["checkpoint_sha256"],
            "config_sha256": assets["config_sha256"],
            "extractor_code_sha256": extractor_sha,
            "query_feature_name": QUERY_FEATURE_NAME,
        },
        "feature_dim": CACHE_FEATURE_DIM,
        "box_format": CACHE_BOX_FORMAT,
        "rows": cache_rows,
    }
    _atomic_torch(payload, output)
    hashes = {
        "file_sha256": file_sha256(output),
        "size_bytes": output.stat().st_size,
    }
    result = {
        "schema": RECEIPT_SCHEMA,
        "status": "complete",
        "mode": mode,
        "surface": surface,
        "assets": {
            **assets,
            "model_id": model_id,
            "extractor_code_sha256": extractor_sha,
        },
        "input": {
            "path": str(input_jsonl.resolve(strict=True)),
            "sha256": input_sha256,
            "manifest_rows": len(rows),
            "request_rows": len(requests),
            "image_identity_sha256": _image_set_sha256(requests),
        },
        "output": {
            "path": str(output.resolve(strict=True)),
            **hashes,
            "row_count": len(cache_rows),
        },
        "oracle": {
            "rows_with_iou50_candidate": oracle_rows if mode == "ref" else None,
            "total_rows": len(requests) if mode == "ref" else None,
        },
        "runtime": {
            "device": device,
            "feature_dtype": "float32",
            "batch_size": 1,
            "training": False,
        },
    }
    _atomic_json(result, receipt)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--mmdet-root", type=Path, required=True)
    parser.add_argument("--mmdet-commit", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    result = extract_eval_cache(
        mode=args.mode,
        input_jsonl=args.input_jsonl.resolve(strict=True),
        input_sha256=args.input_sha256,
        image_root=args.image_root.resolve(strict=True),
        surface=args.surface,
        mmdet_root=args.mmdet_root.resolve(strict=True),
        mmdet_commit=args.mmdet_commit,
        config_path=args.config.resolve(strict=True),
        config_sha256=args.config_sha256,
        checkpoint_path=args.checkpoint.resolve(strict=True),
        checkpoint_sha256=args.checkpoint_sha256,
        model_id=args.model_id,
        output=args.output.resolve(),
        receipt=args.receipt.resolve(),
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "EVAL_CACHE_SCHEMA",
    "RECEIPT_SCHEMA",
    "extract_eval_cache",
    "parse_tn_eval_requests",
]
