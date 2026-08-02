#!/usr/bin/env python3
"""Judge frozen-GDINO top-score regions with a pinned local Qwen VLM.

This tool is intentionally independent from the fixed Stage-B baseline code
closure. Dry-run performs only CPU-side schema and hash validation. Formal runs
validate the pinned processor/runtime before reading caches; the 7B model
weights remain lazy until at least one uncached region needs a new judgment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, features as PIL_FEATURES, __version__ as PIL_VERSION


EXTRACTION_SCHEMA = "stage-b-fixed-gdino-top1-vlm-extraction-v1"
EXTRACTION_AUDIT_SCHEMA = "stage-b-fixed-gdino-top1-vlm-extraction-audit-v1"
EXTRACTION_AUDIT_KIND = "completed_fixed_gdino_top1_vlm_extraction"
EXPECTED_EXTRACTION_ROWS = 17_738
JUDGMENT_SCHEMA = "stage-b-fixed-gdino-top1-qwen-judgment-v1"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
INHERIT_IOU_THRESHOLD = 0.70
INHERIT_CONFIDENCE_THRESHOLD = 0.90

PROMPT_TEMPLATE = """You are verifying a candidate region for referring-expression grounding.

Two views of the same image are provided. The first is the full image with the
candidate outlined by a red rectangle. The second is a 2x context crop with the
same candidate outlined in red. Judge only the object or region identified by
the red rectangle, while using the full scene to evaluate relations, actions,
position, number, and attached descriptions.

Does the red-boxed candidate fully satisfy every part of this expression?
Expression: {expression_json}

Answer YES only when every important noun, modifier, attribute, relation,
action, number, and attached clause is supported. Answer NO when any important
part is contradicted. Answer UNKNOWN when the visual evidence is insufficient
or the red box does not identify one judgeable candidate.

Return exactly one JSON object and no markdown:
{{"answer":"YES|NO|UNKNOWN","confidence":0.0,"short_reason":"brief evidence"}}
"""

ASSET_POLICY: Dict[str, Any] = {
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

GENERATION_CONFIG: Dict[str, Any] = {
    "do_sample": False,
    "max_new_tokens": 160,
    "use_cache": True,
}
VISION_PROCESSOR_CONFIG: Dict[str, Any] = {
    "min_pixels": 256 * 28 * 28,
    "max_pixels": 1280 * 28 * 28,
}
INFERENCE_BATCH_SIZE = 1


class QwenJudgeError(RuntimeError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


PROMPT_TEMPLATE_SHA256 = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
ASSET_POLICY_SHA256 = canonical_sha256(ASSET_POLICY)
GENERATION_CONFIG_SHA256 = canonical_sha256(GENERATION_CONFIG)
VISION_PROCESSOR_CONFIG_SHA256 = canonical_sha256(VISION_PROCESSOR_CONFIG)

JUDGE_IMPLEMENTATION_REVISION = (
    "stage-b-fixed-gdino-top1-qwen-judge-v2-runtime-locked-20260712"
)
JUDGE_RUNTIME_POLICY: Dict[str, Any] = {
    "schema": "stage-b-fixed-gdino-top1-qwen-runtime-policy-v1",
    "implementation_revision": JUDGE_IMPLEMENTATION_REVISION,
    "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
    "inference": {
        "device": "cuda:0",
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "batch_size": INFERENCE_BATCH_SIZE,
        "local_files_only": True,
        "generation_config_sha256": GENERATION_CONFIG_SHA256,
        "vision_processor_config_sha256": VISION_PROCESSOR_CONFIG_SHA256,
    },
    "software": {
        "torch": "2.11.0+cu128",
        "transformers": "5.11.0",
        "pillow": "10.2.0",
        "libjpeg": "8.0",
        "libjpeg_turbo": "2.1.5",
    },
    "classes": {
        "model": (
            "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl."
            "Qwen2_5_VLForConditionalGeneration"
        ),
        "processor": (
            "transformers.models.qwen2_5_vl.processing_qwen2_5_vl."
            "Qwen2_5_VLProcessor"
        ),
        "image_processor": (
            "transformers.models.qwen2_vl.image_processing_qwen2_vl."
            "Qwen2VLImageProcessor"
        ),
        "tokenizer": (
            "transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer"
        ),
    },
    "cuda": {
        "torch_cuda_build": "12.8",
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "compute_capability": [8, 9],
        "driver_version": "595.71.05",
    },
    "local_model_files": {
        "chat_template.json": (
            "ad60d90252ed0b0705ba14e2d0ad0fec0beac1ea955642b54059b36052d8bc96"
        ),
        "preprocessor_config.json": (
            "f2058c716eef96ccaed1cc1e2d0c08306b62586d535b28d9d08e691b2fab7ca0"
        ),
        "tokenizer_config.json": (
            "4abd3520120e266da84c0864fee064d1fb10806f02225911a47253dd38dc5f56"
        ),
    },
}
JUDGE_RUNTIME_POLICY_SHA256 = canonical_sha256(JUDGE_RUNTIME_POLICY)


def _qualified_class_name(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _validate_fixed_cli_policy(args: argparse.Namespace) -> None:
    expected = JUDGE_RUNTIME_POLICY["inference"]
    observed = {
        "device": str(getattr(args, "device", "")),
        "dtype": str(getattr(args, "dtype", "")),
        "attn_implementation": str(getattr(args, "attn_implementation", "")),
        "batch_size": int(getattr(args, "batch_size", -1)),
        "local_files_only": not bool(getattr(args, "allow_download", False)),
    }
    for key in (
        "device",
        "dtype",
        "attn_implementation",
        "batch_size",
        "local_files_only",
    ):
        if observed[key] != expected[key]:
            raise QwenJudgeError(
                f"formal Qwen runtime {key} must remain exactly {expected[key]!r}; "
                f"got {observed[key]!r}"
            )


def _nvidia_driver_version() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise QwenJudgeError(f"failed to query NVIDIA driver version: {error}") from error
    versions = {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    if len(versions) != 1:
        raise QwenJudgeError(f"NVIDIA driver version is ambiguous: {sorted(versions)}")
    return next(iter(versions))


def _observe_runtime_policy(*, model_cache_dir: Path | None) -> tuple[Dict[str, Any], Path]:
    try:
        import torch
        import transformers
        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as error:
        raise QwenJudgeError(f"pinned Qwen runtime dependency is unavailable: {error}") from error

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise QwenJudgeError("formal Qwen runtime requires visible CUDA device cuda:0")
    device = torch.device("cuda:0")
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=MODEL_ID,
                revision=MODEL_REVISION,
                cache_dir=str(model_cache_dir) if model_cache_dir is not None else None,
                local_files_only=True,
            )
        ).resolve()
    except Exception as error:
        raise QwenJudgeError(
            f"pinned local Qwen snapshot is unavailable for {MODEL_REVISION}: {error}"
        ) from error

    observed_files: Dict[str, str] = {}
    for name in JUDGE_RUNTIME_POLICY["local_model_files"]:
        path = snapshot / name
        if not path.is_file():
            raise QwenJudgeError(f"pinned Qwen snapshot lacks {name}: {path}")
        observed_files[name] = sha256_file(path)

    try:
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            cache_dir=str(model_cache_dir) if model_cache_dir is not None else None,
            min_pixels=int(VISION_PROCESSOR_CONFIG["min_pixels"]),
            max_pixels=int(VISION_PROCESSOR_CONFIG["max_pixels"]),
        )
    except Exception as error:
        raise QwenJudgeError(f"failed to load pinned local Qwen processor: {error}") from error

    turbo_version = (
        PIL_FEATURES.version_feature("libjpeg_turbo")
        if PIL_FEATURES.check_feature("libjpeg_turbo")
        else None
    )
    observed: Dict[str, Any] = {
        "schema": JUDGE_RUNTIME_POLICY["schema"],
        "implementation_revision": JUDGE_IMPLEMENTATION_REVISION,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "inference": dict(JUDGE_RUNTIME_POLICY["inference"]),
        "software": {
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "pillow": str(PIL_VERSION),
            "libjpeg": str(PIL_FEATURES.version("jpg")),
            "libjpeg_turbo": str(turbo_version),
        },
        "classes": {
            "model": _qualified_class_name(Qwen2_5_VLForConditionalGeneration),
            "processor": _qualified_class_name(processor),
            "image_processor": _qualified_class_name(processor.image_processor),
            "tokenizer": _qualified_class_name(processor.tokenizer),
        },
        "cuda": {
            "torch_cuda_build": str(torch.version.cuda),
            "gpu_name": str(torch.cuda.get_device_name(device)),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "driver_version": _nvidia_driver_version(),
        },
        "local_model_files": observed_files,
    }
    return observed, snapshot


def validate_runtime_environment(*, model_cache_dir: Path | None) -> Dict[str, Any]:
    observed, snapshot = _observe_runtime_policy(model_cache_dir=model_cache_dir)
    if observed != JUDGE_RUNTIME_POLICY:
        mismatches = [
            key
            for key in JUDGE_RUNTIME_POLICY
            if observed.get(key) != JUDGE_RUNTIME_POLICY.get(key)
        ]
        raise QwenJudgeError(
            "formal Qwen runtime policy drifted in: " + ", ".join(mismatches)
        )
    return {
        "policy_sha256": JUDGE_RUNTIME_POLICY_SHA256,
        "model_snapshot_path": str(snapshot),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> Dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise QwenJudgeError(f"required file is missing: {path}")
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def validate_extraction_audit(
    audit_path: Path, *, extraction_path: Path, extraction_record: Mapping[str, Any]
) -> Dict[str, Any]:
    if not audit_path.is_file():
        raise QwenJudgeError(f"missing extraction audit: {audit_path}")
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QwenJudgeError(f"invalid extraction audit {audit_path}: {error}") from error
    if not isinstance(audit, dict) or audit.get(
        "schema"
    ) != EXTRACTION_AUDIT_SCHEMA or audit.get("kind") != EXTRACTION_AUDIT_KIND:
        raise QwenJudgeError("extraction audit schema/kind drifted")
    if int(audit.get("rows", -1)) != EXPECTED_EXTRACTION_ROWS:
        raise QwenJudgeError(
            f"extraction audit rows must be {EXPECTED_EXTRACTION_ROWS}"
        )
    manifest = audit.get("manifest")
    expected = {
        "path": str(extraction_path),
        "sha256": extraction_record["sha256"],
        "size_bytes": extraction_record["size_bytes"],
        "rows": EXPECTED_EXTRACTION_ROWS,
    }
    if not isinstance(manifest, Mapping) or any(
        manifest.get(key) != value for key, value in expected.items()
    ):
        raise QwenJudgeError("extraction audit manifest record drifted")
    return {**file_record(audit_path), "schema": audit["schema"], "kind": audit["kind"]}


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )
    _atomic_write_text(path, payload)


def iter_jsonl(path: Path) -> Iterable[tuple[int, Dict[str, Any]]]:
    if not path.is_file():
        raise QwenJudgeError(f"missing JSONL: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise QwenJudgeError(f"blank line at {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise QwenJudgeError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise QwenJudgeError(f"non-object row at {path}:{line_number}")
            yield line_number, row


def _identity(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("identity")
    return value if isinstance(value, Mapping) else row


def _sample_id(row: Mapping[str, Any]) -> str:
    identity = _identity(row)
    value = identity.get("sample_id", row.get("sample_id"))
    if not isinstance(value, str) or not value.strip():
        raise QwenJudgeError("extraction row has no non-empty sample_id")
    return value.strip()


def _negative_expression(row: Mapping[str, Any]) -> str:
    for key in ("negative_expression", "negative_caption_model", "try_tn"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    raise QwenJudgeError(f"{_sample_id(row)} has no negative expression")


def _finite_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise QwenJudgeError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(result):
        raise QwenJudgeError(f"{label} is not finite")
    return result


def region_bbox_xyxy_original(region: Mapping[str, Any]) -> list[float]:
    value = region.get("bbox_xyxy_original")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 4:
        box = [
            _finite_float(item, label="region bbox_xyxy_original") for item in value
        ]
    else:
        value = region.get("bbox_xywh_original")
        if not (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and len(value) == 4
        ):
            raise QwenJudgeError("region has no valid original-coordinate bbox")
        x, y, width, height = [
            _finite_float(item, label="region bbox_xywh_original") for item in value
        ]
        box = [x, y, x + width, y + height]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise QwenJudgeError(f"region has non-positive bbox: {box}")
    return box


def _image_record(row: Mapping[str, Any]) -> Dict[str, Any]:
    value = row.get("image")
    if not isinstance(value, Mapping):
        raise QwenJudgeError(f"{_sample_id(row)} has no image record")
    path_value = value.get("path")
    digest = value.get("sha256")
    if not isinstance(path_value, str) or not path_value.strip():
        raise QwenJudgeError(f"{_sample_id(row)} image path is missing")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise QwenJudgeError(f"{_sample_id(row)} image SHA-256 is malformed")
    width = int(value.get("width", 0) or 0)
    height = int(value.get("height", 0) or 0)
    if width <= 0 or height <= 0:
        raise QwenJudgeError(f"{_sample_id(row)} image dimensions are invalid")
    return {
        "path": str(Path(path_value).expanduser().resolve()),
        "sha256": digest,
        "width": width,
        "height": height,
    }


def _overlap_record(region: Mapping[str, Any]) -> Mapping[str, Any]:
    value = region.get("max_overlap")
    return value if isinstance(value, Mapping) else {}


def source_inheritance_eligible(region: Mapping[str, Any]) -> bool:
    overlap = _overlap_record(region)
    kind = str(overlap.get("kind", "none")).strip().lower()
    answer = str(overlap.get("source_answer", "")).strip().upper()
    try:
        iou = float(overlap.get("iou", -1.0))
        confidence = float(overlap.get("source_confidence", -1.0))
    except (TypeError, ValueError):
        return False
    return (
        kind in {"target", "proposal"}
        and answer == "NO"
        and math.isfinite(iou)
        and math.isfinite(confidence)
        and iou >= INHERIT_IOU_THRESHOLD
        and confidence >= INHERIT_CONFIDENCE_THRESHOLD
    )


def validate_extraction_row(row: Mapping[str, Any]) -> None:
    if row.get("schema") != EXTRACTION_SCHEMA:
        raise QwenJudgeError(
            f"unexpected extraction schema for {_sample_id(row)}: {row.get('schema')!r}"
        )
    sample_id = _sample_id(row)
    _negative_expression(row)
    _image_record(row)
    if int(row.get("num_queries", -1)) != 900 or int(
        row.get("valid_query_count", -1)
    ) != 900:
        raise QwenJudgeError(f"{sample_id} does not expose exactly 900 valid queries")
    claims = row.get("claims")
    if not isinstance(claims, Mapping):
        raise QwenJudgeError(f"{sample_id} has no claims record")
    if claims.get("all_900_gdino_queries_verified") is not False:
        raise QwenJudgeError(f"{sample_id} lost the non-all-900 scope claim")
    if claims.get("portable_to_other_checkpoint_or_transform") is not False:
        raise QwenJudgeError(f"{sample_id} lost the non-portable scope claim")
    if claims.get("train_path_and_deploy_transform_regions_extracted") is not True:
        raise QwenJudgeError(f"{sample_id} lacks train/deploy region extraction")
    regions = row.get("regions")
    if not isinstance(regions, list) or not regions:
        raise QwenJudgeError(f"{sample_id} has no extracted verification regions")
    seen = set()
    for region in regions:
        if not isinstance(region, Mapping):
            raise QwenJudgeError(f"{sample_id} contains a non-object region")
        region_id = region.get("region_id")
        if not isinstance(region_id, str) or not region_id.strip():
            raise QwenJudgeError(f"{sample_id} has a region without region_id")
        if region_id in seen:
            raise QwenJudgeError(f"{sample_id} duplicates region_id={region_id}")
        seen.add(region_id)
        region_bbox_xyxy_original(region)
        assets = region.get("assets")
        if isinstance(assets, Mapping) and assets:
            if assets.get("asset_policy_sha256") != ASSET_POLICY_SHA256:
                raise QwenJudgeError(f"{sample_id} region asset policy drifted")
            for name in ("tight", "context_2x_boxed", "full_boxed"):
                asset = assets.get(name)
                if not isinstance(asset, Mapping) or not asset.get("path") or not asset.get(
                    "sha256"
                ):
                    raise QwenJudgeError(f"{sample_id} region lacks {name} asset")


def render_prompt(expression: str) -> str:
    return PROMPT_TEMPLATE.format(
        expression_json=json.dumps(expression, ensure_ascii=False)
    )


def judgment_cache_key(row: Mapping[str, Any], region: Mapping[str, Any]) -> str:
    image = _image_record(row)
    region_assets = region.get("assets")
    asset_inputs = None
    if isinstance(region_assets, Mapping):
        asset_inputs = {
            "asset_policy_sha256": region_assets.get("asset_policy_sha256"),
            "full_boxed_sha256": (
                region_assets.get("full_boxed", {}).get("sha256")
                if isinstance(region_assets.get("full_boxed"), Mapping)
                else None
            ),
            "context_2x_boxed_sha256": (
                region_assets.get("context_2x_boxed", {}).get("sha256")
                if isinstance(region_assets.get("context_2x_boxed"), Mapping)
                else None
            ),
        }
    payload = {
        "schema": JUDGMENT_SCHEMA,
        "sample_id": _sample_id(row),
        "region_id": str(region["region_id"]),
        "image_sha256": image["sha256"],
        "bbox_xyxy_original": region_bbox_xyxy_original(region),
        "negative_expression": _negative_expression(row),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        "asset_policy_sha256": ASSET_POLICY_SHA256,
        "extraction_assets": asset_inputs,
        "generation_config_sha256": GENERATION_CONFIG_SHA256,
        "vision_processor_config_sha256": VISION_PROCESSOR_CONFIG_SHA256,
        "inference_batch_size": INFERENCE_BATCH_SIZE,
        "judge_runtime_policy_sha256": JUDGE_RUNTIME_POLICY_SHA256,
    }
    return canonical_sha256(payload)


def _clip_box(box: Sequence[float], width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = [float(value) for value in box]
    clipped = [
        min(max(x0, 0.0), float(width)),
        min(max(y0, 0.0), float(height)),
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise QwenJudgeError(f"bbox is empty after clipping: {list(box)}")
    return clipped


def _raster_box(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(value) for value in box]
    left = min(max(0, math.floor(x0)), max(0, width - 1))
    top = min(max(0, math.floor(y0)), max(0, height - 1))
    right = min(width, max(left + 1, math.ceil(x1)))
    bottom = min(height, max(top + 1, math.ceil(y1)))
    return int(left), int(top), int(right), int(bottom)


def _context_box(
    box: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    context_width = (right - left) * float(ASSET_POLICY["context_scale"])
    context_height = (bottom - top) * float(ASSET_POLICY["context_scale"])
    return _raster_box(
        [
            center_x - context_width / 2.0,
            center_y - context_height / 2.0,
            center_x + context_width / 2.0,
            center_y + context_height / 2.0,
        ],
        width,
        height,
    )


def _draw_red_box(image: Image.Image, box: Sequence[int]) -> None:
    left, top, right, bottom = [int(value) for value in box]
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [left, top, max(left, right - 1), max(top, bottom - 1)],
        outline=tuple(ASSET_POLICY["box_color_rgb"]),
        width=int(ASSET_POLICY["box_width_px"]),
    )


def _save_asset(
    path: Path, image: Image.Image, encoding: Mapping[str, Any]
) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        image.save(temporary, **dict(encoding))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    result = file_record(path)
    result.update({"width": image.width, "height": image.height})
    return result


def build_boxed_assets(
    row: Mapping[str, Any],
    region: Mapping[str, Any],
    *,
    asset_root: Path,
    cache_key: str,
) -> Dict[str, Any]:
    image_record = _image_record(row)
    image_path = Path(image_record["path"])
    observed_hash = sha256_file(image_path)
    if observed_hash != image_record["sha256"]:
        raise QwenJudgeError(
            f"image hash drift for {_sample_id(row)}: {observed_hash} "
            f"!= {image_record['sha256']}"
        )
    with Image.open(image_path) as opened:
        original = opened.convert("RGB")
    if original.size != (image_record["width"], image_record["height"]):
        raise QwenJudgeError(
            f"image dimensions drifted for {_sample_id(row)}: {original.size}"
        )
    box = _clip_box(
        region_bbox_xyxy_original(region), original.width, original.height
    )
    raster = _raster_box(box, original.width, original.height)
    context_box = _context_box(raster, original.width, original.height)
    tight = original.crop(raster)
    context = original.crop(context_box)
    context_local = (
        raster[0] - context_box[0],
        raster[1] - context_box[1],
        raster[2] - context_box[0],
        raster[3] - context_box[1],
    )
    _draw_red_box(context, context_local)
    full = original.copy()
    _draw_red_box(full, raster)

    destination = asset_root / cache_key[:2] / cache_key
    tight_record = _save_asset(
        destination / "tight.png", tight, ASSET_POLICY["tight_encoding"]
    )
    context_record = _save_asset(
        destination / "context_2x_boxed.png",
        context,
        ASSET_POLICY["context_encoding"],
    )
    full_record = _save_asset(
        destination / "full_boxed.jpg",
        full,
        ASSET_POLICY["full_boxed_encoding"],
    )
    return {
        "asset_policy_sha256": ASSET_POLICY_SHA256,
        "source_image": dict(image_record),
        "bbox_xyxy_original": box,
        "raster_bbox_xyxy": list(raster),
        "context_bbox_xyxy": list(context_box),
        "pillow_version": PIL_VERSION,
        "tight": tight_record,
        "full_boxed": full_record,
        "context_2x_boxed": context_record,
    }


def extraction_boxed_assets(
    row: Mapping[str, Any], region: Mapping[str, Any]
) -> Dict[str, Any]:
    """Validate and reuse the extractor's locked boxed assets."""
    assets = region.get("assets")
    if not isinstance(assets, Mapping):
        raise QwenJudgeError(f"{_sample_id(row)} region has no extraction assets")
    if assets.get("asset_policy_sha256") != ASSET_POLICY_SHA256:
        raise QwenJudgeError("extraction asset policy drifted from the judge contract")
    result = dict(assets)
    for name in ("tight", "full_boxed", "context_2x_boxed"):
        record = assets.get(name)
        if not isinstance(record, Mapping) or not record.get("path") or not record.get(
            "sha256"
        ):
            raise QwenJudgeError(f"extraction lacks {name} asset")
        path = Path(str(record["path"])).expanduser().resolve()
        if sha256_file(path) != record["sha256"]:
            raise QwenJudgeError(f"extraction {name} asset hash drifted")
        with Image.open(path) as image:
            dimensions = image.size
        if dimensions != (int(record.get("width", 0)), int(record.get("height", 0))):
            raise QwenJudgeError(f"extraction {name} asset dimensions drifted")
    image = _image_record(row)
    if sha256_file(Path(image["path"])) != image["sha256"]:
        raise QwenJudgeError("extraction source image hash drifted")
    result["source_image"] = image
    result["bbox_xyxy_original"] = region_bbox_xyxy_original(region)
    return result


def parse_structured_answer(raw_output: str) -> Dict[str, Any]:
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise QwenJudgeError("Qwen returned an empty response")
    text = raw_output.strip()
    decoder = json.JSONDecoder()
    parsed = None
    for match in re.finditer(r"\{", text):
        try:
            candidate, _end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            parsed = candidate
            break
    if parsed is None:
        raise QwenJudgeError("Qwen response contains no JSON object")
    answer = str(parsed.get("answer", "")).strip().upper()
    if answer not in {"YES", "NO", "UNKNOWN"}:
        raise QwenJudgeError(f"invalid Qwen answer: {answer!r}")
    confidence = _finite_float(parsed.get("confidence"), label="Qwen confidence")
    if not 0.0 <= confidence <= 1.0:
        raise QwenJudgeError("Qwen confidence must be in [0,1]")
    reason = parsed.get("short_reason")
    if not isinstance(reason, str) or not reason.strip():
        raise QwenJudgeError("Qwen response has no short_reason")
    return {
        "answer": answer,
        "confidence": confidence,
        "short_reason": " ".join(reason.split()),
    }


class LocalQwenRunner:
    """Lazy local runner for the pinned Qwen checkpoint."""

    def __init__(
        self,
        *,
        device: str,
        dtype: str,
        attn_implementation: str,
        local_files_only: bool,
        cache_dir: Path | None,
    ) -> None:
        self.requested_device = str(device)
        self.dtype_name = str(dtype)
        self.attn_implementation = str(attn_implementation)
        self.local_files_only = bool(local_files_only)
        self.cache_dir = cache_dir
        self._model = None
        self._processor = None
        self._runtime: Dict[str, Any] | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        if self.requested_device == "auto" and importlib.util.find_spec(
            "accelerate"
        ) is None:
            raise QwenJudgeError(
                "device=auto requires accelerate, which is not installed; "
                "select an explicit device such as cuda:0"
            )
        if self.requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise QwenJudgeError("CUDA was requested but torch.cuda.is_available() is false")
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if self.dtype_name not in dtype_map:
            raise QwenJudgeError(f"unsupported dtype: {self.dtype_name}")
        common: Dict[str, Any] = {
            "revision": MODEL_REVISION,
            "local_files_only": self.local_files_only,
        }
        if self.cache_dir is not None:
            common["cache_dir"] = str(self.cache_dir)
        model_kwargs = dict(common)
        model_kwargs.update(
            {
                "torch_dtype": dtype_map[self.dtype_name],
                "attn_implementation": self.attn_implementation,
            }
        )
        if self.requested_device == "auto":
            model_kwargs["device_map"] = "auto"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, **model_kwargs
        )
        if self.requested_device != "auto":
            model.to(torch.device(self.requested_device))
        model.eval()
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            min_pixels=int(VISION_PROCESSOR_CONFIG["min_pixels"]),
            max_pixels=int(VISION_PROCESSOR_CONFIG["max_pixels"]),
            **common,
        )
        commit_hash = getattr(model.config, "_commit_hash", None)
        if commit_hash is not None and str(commit_hash) != MODEL_REVISION:
            raise QwenJudgeError(
                f"loaded Qwen revision drifted: {commit_hash} != {MODEL_REVISION}"
            )
        model_device = str(next(model.parameters()).device)
        cuda_name = None
        if model_device.startswith("cuda"):
            cuda_name = torch.cuda.get_device_name(torch.device(model_device))
        self._runtime = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "pillow": getattr(sys.modules.get("PIL"), "__version__", None),
            "requested_device": self.requested_device,
            "model_device": model_device,
            "cuda_device_name": cuda_name,
            "dtype": self.dtype_name,
            "attn_implementation": self.attn_implementation,
            "local_files_only": self.local_files_only,
            "vision_processor_config": dict(VISION_PROCESSOR_CONFIG),
            "vision_processor_config_sha256": VISION_PROCESSOR_CONFIG_SHA256,
            "model_config_commit_hash": commit_hash,
            "judge_runtime_policy_sha256": JUDGE_RUNTIME_POLICY_SHA256,
        }
        self._model = model
        self._processor = processor

    def judge(
        self,
        *,
        full_image: Image.Image,
        context_image: Image.Image,
        prompt: str,
    ) -> tuple[str, Dict[str, Any]]:
        outputs, runtime = self.judge_batch(
            items=[(full_image, context_image, prompt)]
        )
        return outputs[0], runtime

    def judge_batch(
        self,
        *,
        items: Sequence[tuple[Image.Image, Image.Image, str]],
    ) -> tuple[list[str], Dict[str, Any]]:
        if not items:
            raise QwenJudgeError("Qwen batch must contain at least one item")
        self._load()
        assert self._model is not None and self._processor is not None
        import torch

        chats = []
        images = []
        for full_image, context_image, prompt in items:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            chats.append(
                self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
            # The processor consumes replacements in placeholder order across
            # the text batch, so each item's full/context pair stays adjacent.
            images.extend((full_image, context_image))
        inputs = self._processor(
            text=chats,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        model_device = next(self._model.parameters()).device
        inputs = inputs.to(model_device)
        with torch.inference_mode():
            generated = self._model.generate(**inputs, **GENERATION_CONFIG)
        input_ids = inputs["input_ids"]
        trimmed = [
            output_ids[len(source_ids) :]
            for source_ids, output_ids in zip(input_ids, generated)
        ]
        raw_outputs = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if len(raw_outputs) != len(items):
            raise QwenJudgeError(
                f"Qwen returned {len(raw_outputs)} outputs for {len(items)} inputs"
            )
        runtime = dict(self._runtime or {})
        runtime["generation_batch_size"] = len(items)
        return list(raw_outputs), runtime


def _cache_path(cache_root: Path, cache_key: str) -> Path:
    return cache_root / cache_key[:2] / f"{cache_key}.json"


def validate_judgment_record(
    record: Mapping[str, Any],
    *,
    expected_sample_id: str | None = None,
    expected_region_id: str | None = None,
    expected_cache_key: str | None = None,
    require_assets: bool = True,
) -> None:
    if record.get("schema") != JUDGMENT_SCHEMA:
        raise QwenJudgeError("cached judgment has the wrong schema")
    for key, expected in (
        ("sample_id", expected_sample_id),
        ("region_id", expected_region_id),
        ("cache_key", expected_cache_key),
    ):
        if expected is not None and record.get(key) != expected:
            raise QwenJudgeError(f"cached judgment {key} drifted")
    model = record.get("model")
    if not isinstance(model, Mapping) or model.get("id") != MODEL_ID or model.get(
        "revision"
    ) != MODEL_REVISION:
        raise QwenJudgeError("cached judgment model provenance drifted")
    prompt = record.get("prompt")
    if not isinstance(prompt, Mapping) or prompt.get(
        "template_sha256"
    ) != PROMPT_TEMPLATE_SHA256:
        raise QwenJudgeError("cached judgment prompt provenance drifted")
    if record.get("asset_policy_sha256") != ASSET_POLICY_SHA256:
        raise QwenJudgeError("cached judgment asset policy drifted")
    if record.get("generation_config_sha256") != GENERATION_CONFIG_SHA256:
        raise QwenJudgeError("cached judgment generation policy drifted")
    if (
        record.get("vision_processor_config_sha256")
        != VISION_PROCESSOR_CONFIG_SHA256
    ):
        raise QwenJudgeError("cached judgment vision processor policy drifted")
    if record.get("inference_batch_size") != INFERENCE_BATCH_SIZE:
        raise QwenJudgeError("cached judgment inference batch size drifted")
    if record.get("judge_runtime_policy_sha256") != JUDGE_RUNTIME_POLICY_SHA256:
        raise QwenJudgeError("cached judgment runtime policy hash drifted")
    if record.get("judge_runtime_policy") != JUDGE_RUNTIME_POLICY:
        raise QwenJudgeError("cached judgment runtime policy drifted")
    status = record.get("status")
    if status not in {"complete", "error"}:
        raise QwenJudgeError(f"cached judgment has invalid status: {status!r}")
    raw_output = record.get("raw_output")
    if not isinstance(raw_output, str):
        raise QwenJudgeError("cached judgment lacks raw_output")
    if hashlib.sha256(raw_output.encode("utf-8")).hexdigest() != record.get(
        "raw_output_sha256"
    ):
        raise QwenJudgeError("cached judgment raw output hash drifted")
    if status == "complete":
        parsed = parse_structured_answer(raw_output)
        if record.get("answer") != parsed["answer"] or not math.isclose(
            float(record.get("confidence", -1.0)),
            parsed["confidence"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise QwenJudgeError("cached judgment parsed fields drifted")
        if record.get("short_reason") != parsed["short_reason"]:
            raise QwenJudgeError("cached judgment parsed reason drifted")
    if require_assets:
        assets = record.get("assets")
        if not isinstance(assets, Mapping):
            raise QwenJudgeError("cached judgment lacks assets")
        for name in ("full_boxed", "context_2x_boxed"):
            asset = assets.get(name)
            if not isinstance(asset, Mapping) or not asset.get("path") or not asset.get(
                "sha256"
            ):
                raise QwenJudgeError(f"cached judgment lacks {name} asset")
            if sha256_file(Path(str(asset["path"]))) != asset["sha256"]:
                raise QwenJudgeError(f"cached judgment {name} asset hash drifted")


def _read_existing(path: Path) -> Dict[tuple[str, str], Dict[str, Any]]:
    result: Dict[tuple[str, str], Dict[str, Any]] = {}
    if not path.exists():
        return result
    for line_number, row in iter_jsonl(path):
        validate_judgment_record(row, require_assets=False)
        key = (str(row.get("sample_id")), str(row.get("region_id")))
        if key in result:
            raise QwenJudgeError(f"duplicate existing judgment at {path}:{line_number}")
        result[key] = row
    return result


def _load_cached(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QwenJudgeError(f"invalid content cache {path}: {error}") from error
    if not isinstance(value, dict):
        raise QwenJudgeError(f"content cache is not an object: {path}")
    return value


def _make_judgment_record(
    row: Mapping[str, Any],
    region: Mapping[str, Any],
    *,
    cache_key: str,
    assets: Mapping[str, Any],
    raw_output: str,
    runtime: Mapping[str, Any],
) -> Dict[str, Any]:
    expression = _negative_expression(row)
    prompt = render_prompt(expression)
    error_message = None
    try:
        parsed = parse_structured_answer(raw_output)
        status = "complete"
    except QwenJudgeError as error:  # Invalid semantic output is quarantined.
        status = "error"
        parsed = {"answer": "UNKNOWN", "confidence": 0.0, "short_reason": ""}
        error_message = f"{type(error).__name__}: {error}"
    record: Dict[str, Any] = {
        "schema": JUDGMENT_SCHEMA,
        "extraction_schema": EXTRACTION_SCHEMA,
        "sample_id": _sample_id(row),
        "identity": dict(_identity(row)),
        "region_id": str(region["region_id"]),
        "cache_key": cache_key,
        "status": status,
        "answer": parsed["answer"],
        "confidence": parsed["confidence"],
        "short_reason": parsed["short_reason"],
        "raw_output": raw_output,
        "raw_output_sha256": hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
        "error": error_message,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "prompt": {
            "template_sha256": PROMPT_TEMPLATE_SHA256,
            "rendered_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        },
        "asset_policy_sha256": ASSET_POLICY_SHA256,
        "generation_config": dict(GENERATION_CONFIG),
        "generation_config_sha256": GENERATION_CONFIG_SHA256,
        "vision_processor_config": dict(VISION_PROCESSOR_CONFIG),
        "vision_processor_config_sha256": VISION_PROCESSOR_CONFIG_SHA256,
        "inference_batch_size": INFERENCE_BATCH_SIZE,
        "judge_runtime_policy": JUDGE_RUNTIME_POLICY,
        "judge_runtime_policy_sha256": JUDGE_RUNTIME_POLICY_SHA256,
        "bbox_xyxy_original": region_bbox_xyxy_original(region),
        "negative_expression": expression,
        "assets": dict(assets),
        "runtime": dict(runtime),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return record


def _selection_sha256(
    row: Mapping[str, Any], region: Mapping[str, Any], *, seed: str, pool: str
) -> str:
    payload = {
        "schema": "stage-b-fixed-gdino-qwen-selection-v1",
        "seed": str(seed),
        "pool": str(pool),
        "sample_id": _sample_id(row),
        "region_id": str(region["region_id"]),
    }
    return canonical_sha256(payload)


def build_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int = 0,
    selection_seed: str = "20260712",
    audit_inherited: int = 0,
) -> tuple[list[tuple[Mapping[str, Any], Mapping[str, Any], str]], Dict[str, Any]]:
    if int(limit) < 0 or int(audit_inherited) < 0:
        raise QwenJudgeError("limit and audit_inherited must be non-negative")
    non_inherited = []
    inherited = []
    stats: Dict[str, Any] = {
        "rows": 0,
        "regions": 0,
        "source_inheritable_regions": 0,
        "regions_requiring_qwen": 0,
    }
    seen_samples = set()
    for row in rows:
        validate_extraction_row(row)
        sample_id = _sample_id(row)
        if sample_id in seen_samples:
            raise QwenJudgeError(f"duplicate extraction sample_id: {sample_id}")
        seen_samples.add(sample_id)
        stats["rows"] += 1
        for region in row["regions"]:
            stats["regions"] += 1
            if source_inheritance_eligible(region):
                stats["source_inheritable_regions"] += 1
                inherited.append(
                    (
                        _selection_sha256(
                            row, region, seed=selection_seed, pool="source_inherited_audit"
                        ),
                        row,
                        region,
                    )
                )
                continue
            non_inherited.append(
                (
                    _selection_sha256(
                        row, region, seed=selection_seed, pool="non_inherited"
                    ),
                    row,
                    region,
                )
            )
    non_inherited.sort(key=lambda item: item[0])
    inherited.sort(key=lambda item: item[0])
    stats["regions_requiring_qwen"] = len(non_inherited)
    if int(limit) > len(non_inherited):
        raise QwenJudgeError(
            f"requested non-inherited limit={int(limit)} exceeds available "
            f"pool={len(non_inherited)}"
        )
    if int(audit_inherited) > len(inherited):
        raise QwenJudgeError(
            f"requested inherited audit={int(audit_inherited)} exceeds available "
            f"pool={len(inherited)}"
        )
    selected_non_inherited = (
        non_inherited[: int(limit)] if int(limit) > 0 else non_inherited
    )
    selected_inherited = inherited[: int(audit_inherited)]
    selected = selected_non_inherited + selected_inherited
    selected.sort(key=lambda item: item[0])
    plan = [
        (row, region, judgment_cache_key(row, region))
        for _digest, row, region in selected
    ]
    stats["planned_qwen_regions"] = len(plan)
    stats["planned_non_inherited_regions"] = len(selected_non_inherited)
    stats["planned_inherited_audit_regions"] = len(selected_inherited)
    stats["selection_policy"] = {
        "schema": "stage-b-fixed-gdino-qwen-selection-v1",
        "method": "canonical-json-sha256-ascending",
        "seed": str(selection_seed),
        "identity_fields": ["sample_id", "region_id"],
        "separate_pools": ["non_inherited", "source_inherited_audit"],
        "non_inherited_limit": int(limit),
        "inherited_audit_limit": int(audit_inherited),
    }
    return plan, stats


def _validate_retry_error_coverage(
    existing: Mapping[tuple[str, str], Mapping[str, Any]],
    plan: Sequence[tuple[Mapping[str, Any], Mapping[str, Any], str]],
    *,
    retry_errors: bool,
) -> None:
    if not retry_errors:
        return
    planned_keys = {
        (_sample_id(row), str(region["region_id"])) for row, region, _cache_key in plan
    }
    uncovered = sorted(
        key
        for key, record in existing.items()
        if record.get("status") == "error" and key not in planned_keys
    )
    if uncovered:
        preview = ", ".join(f"{sample_id}/{region_id}" for sample_id, region_id in uncovered[:3])
        raise QwenJudgeError(
            f"retry-errors found {len(uncovered)} existing error judgment(s) outside "
            "the current plan; preserve the pilot --audit-inherited and "
            f"--selection-seed when resuming (examples: {preview})"
        )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    _validate_fixed_cli_policy(args)
    input_path = Path(args.input).expanduser().resolve()
    extraction_audit_path = Path(args.extraction_audit).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    cache_root = Path(args.cache_dir).expanduser().resolve()
    input_record = file_record(input_path)
    extraction_audit_record = validate_extraction_audit(
        extraction_audit_path,
        extraction_path=input_path,
        extraction_record=input_record,
    )
    rows = [row for _line, row in iter_jsonl(input_path)]
    if len(rows) != EXPECTED_EXTRACTION_ROWS:
        raise QwenJudgeError(
            f"extraction rows={len(rows)}, expected {EXPECTED_EXTRACTION_ROWS}"
        )
    batch_size = int(getattr(args, "batch_size", 1))
    if batch_size != INFERENCE_BATCH_SIZE:
        raise QwenJudgeError(
            f"batch_size must remain exactly {INFERENCE_BATCH_SIZE}; changing batch "
            "padding would invalidate deterministic judgment cache provenance"
        )
    plan, stats = build_plan(
        rows,
        limit=int(args.limit),
        selection_seed=str(getattr(args, "selection_seed", "20260712")),
        audit_inherited=int(getattr(args, "audit_inherited", 0)),
    )
    stats.update(
        {
            "schema": "stage-b-fixed-gdino-top1-qwen-plan-v1",
            "input": {**input_record, "rows": len(rows)},
            "extraction_audit": extraction_audit_record,
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
            "asset_policy_sha256": ASSET_POLICY_SHA256,
            "generation_config_sha256": GENERATION_CONFIG_SHA256,
            "vision_processor_config_sha256": VISION_PROCESSOR_CONFIG_SHA256,
            "judge_runtime_policy": JUDGE_RUNTIME_POLICY,
            "judge_runtime_policy_sha256": JUDGE_RUNTIME_POLICY_SHA256,
            "batch_size": batch_size,
            "dry_run": bool(args.dry_run),
        }
    )
    if args.dry_run:
        return stats

    model_cache_dir = (
        Path(args.model_cache_dir).expanduser().resolve()
        if args.model_cache_dir
        else None
    )
    stats["validated_runtime"] = validate_runtime_environment(
        model_cache_dir=model_cache_dir
    )

    if output_path.exists() and not args.resume:
        raise QwenJudgeError(
            f"output exists; pass --resume or choose a new path: {output_path}"
        )
    existing = _read_existing(output_path) if args.resume else {}
    extraction_regions = {
        (_sample_id(row), str(region["region_id"])): (row, region)
        for row in rows
        for region in row["regions"]
    }
    unknown_existing = set(existing) - set(extraction_regions)
    if unknown_existing:
        raise QwenJudgeError(
            f"existing output contains {len(unknown_existing)} regions outside extraction"
        )
    for result_key, previous in existing.items():
        row, region = extraction_regions[result_key]
        validate_judgment_record(
            previous,
            expected_sample_id=result_key[0],
            expected_region_id=result_key[1],
            expected_cache_key=judgment_cache_key(row, region),
            require_assets=True,
        )
    _validate_retry_error_coverage(
        existing,
        plan,
        retry_errors=bool(args.retry_errors),
    )
    runner = LocalQwenRunner(
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=not bool(args.allow_download),
        cache_dir=model_cache_dir,
    )
    results = dict(existing)
    cache_hits = 0
    output_hits = 0
    model_calls = 0
    pending = []
    for row, region, cache_key in plan:
        sample_id = _sample_id(row)
        region_id = str(region["region_id"])
        result_key = (sample_id, region_id)
        previous = results.get(result_key)
        if previous is not None:
            validate_judgment_record(
                previous,
                expected_sample_id=sample_id,
                expected_region_id=region_id,
                expected_cache_key=cache_key,
                require_assets=True,
            )
            if not (args.retry_errors and previous.get("status") == "error"):
                output_hits += 1
                continue
        cache_path = _cache_path(cache_root, cache_key)
        cached = _load_cached(cache_path)
        if cached is not None:
            validate_judgment_record(
                cached,
                expected_sample_id=sample_id,
                expected_region_id=region_id,
                expected_cache_key=cache_key,
                require_assets=True,
            )
            if not (args.retry_errors and cached.get("status") == "error"):
                results[result_key] = cached
                cache_hits += 1
                continue
        assets = extraction_boxed_assets(row, region)
        pending.append((row, region, cache_key, cache_path, result_key, assets))

    generation_batches = 0
    if pending:
        # Model/revision/CUDA/OOM failures are job-level failures. They must not
        # be cached as semantic UNKNOWN judgments.
        runner._load()
    for offset in range(0, len(pending), batch_size):
        chunk = pending[offset : offset + batch_size]
        model_items = []
        for row, _region, _cache_key, _cache_path_value, _result_key, assets in chunk:
            with Image.open(Path(str(assets["full_boxed"]["path"]))) as image:
                full = image.convert("RGB")
            with Image.open(Path(str(assets["context_2x_boxed"]["path"]))) as image:
                context = image.convert("RGB")
            model_items.append((full, context, render_prompt(_negative_expression(row))))
        raw_outputs, runtime = runner.judge_batch(items=model_items)
        if len(raw_outputs) != len(chunk):
            raise QwenJudgeError("Qwen batch output count drifted")
        generation_batches += 1
        for item, raw_output in zip(chunk, raw_outputs):
            row, region, cache_key, cache_path, result_key, assets = item
            record = _make_judgment_record(
                row,
                region,
                cache_key=cache_key,
                assets=assets,
                raw_output=raw_output,
                runtime=runtime,
            )
            _atomic_write_json(cache_path, record)
            results[result_key] = record
            model_calls += 1

    ordered = sorted(results.values(), key=lambda item: (item["sample_id"], item["region_id"]))
    _write_jsonl(output_path, ordered)
    stats.update(
        {
            "output": file_record(output_path),
            "existing_output_hits": output_hits,
            "content_cache_hits": cache_hits,
            "model_calls": model_calls,
            "generation_batches": generation_batches,
            "judgments_written": len(ordered),
            "complete": sum(row.get("status") == "complete" for row in ordered),
            "errors": sum(row.get("status") == "error" for row in ordered),
        }
    )
    return stats


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Judge frozen-GDINO max regions with pinned local Qwen2.5-VL."
    )
    parser.add_argument("--input", required=True, help="Extraction JSONL")
    parser.add_argument("--extraction-audit", required=True)
    parser.add_argument("--output", required=True, help="Judgment JSONL")
    parser.add_argument("--cache-dir", required=True, help="Content-addressed result cache")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Hash-selected non-inherited regions; 0 means all",
    )
    parser.add_argument(
        "--audit-inherited",
        type=int,
        default=0,
        help="Hash-selected source-inherited regions to force through Qwen",
    )
    parser.add_argument("--selection-seed", default="20260712")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Locked to 1 for deterministic cache provenance and 24GB safety",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa",), default="sdpa")
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Unsupported by the locked formal runtime; retained for explicit failure.",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    try:
        summary = run(args)
    except QwenJudgeError as error:
        raise SystemExit(f"[ERROR] {error}") from error
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
