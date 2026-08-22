#!/usr/bin/env python3
"""Extract frozen candidates from the pure GroundingDINO pre-Stage-B parent."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

import datasets.transforms as T
from tools.arrow_finecops_common import file_record
from tools.eval_text_groundingdino_refcoco_tn import (
    _load_model_with_checkpoint_contract,
)
from tools.extract_mmgdino_e5_eval_cache import (
    EVAL_CACHE_SCHEMA,
    _atomic_json,
    _atomic_torch,
    _read_jsonl,
    parse_tn_eval_requests,
)
from tools.extract_mmgdino_responsibility_cache import (
    EXTRACTION_RECEIPT_SCHEMA,
    FEATURE_DTYPES,
    ExtractionRequest,
    HookBatch,
    MMGroundingDinoExtractionError,
    _atomic_json_dump,
    _cache_row,
    _image_set_sha256,
    _read_bound_jsonl,
    extract_cached_candidate_shard,
    parse_d3_pair_requests,
    parse_refcoco_rank_requests,
)
from tools.original_gdino_parent_ownership import (
    CHECKPOINT_SHA256,
    EVAL_CONFIG_SHA256,
    PARENT_UNUSED_PATCH_TENSORS,
    PURE_TRUNK_NUMEL,
    PURE_TRUNK_TENSORS,
)
from tools.responsibility_isolation_cache import (
    CACHE_BOX_FORMAT,
    CACHE_FEATURE_DIM,
    CACHE_SOURCE_SCHEMA,
    CACHE_TASK_CONFIDENCE_PAIR,
    CACHE_TASK_RANK,
    file_sha256,
    normalized_cxcywh_iou,
    save_cached_candidate_shard,
    validate_cached_candidate_row,
    validate_cached_candidate_shard,
)
from util.slconfig import SLConfig
from util.utils import clean_state_dict


QUERY_FEATURE_NAME = "groundingdino.transformer.decoder.hs[-1]:900x256"
TRAINING_RECEIPT_SCHEMA = (
    "arrow.original_gdino_parent_ownership.cache_extraction_receipt/v1"
)
EVALUATION_RECEIPT_SCHEMA = (
    "arrow.original_gdino_parent_ownership.eval_cache_receipt/v1"
)
EXPECTED_QUERY_COUNT = 900


class OriginalGDINOExtractionError(MMGroundingDinoExtractionError):
    pass


def _unused_parent_tensor(name: str) -> bool:
    return (
        name == "patch_logit_scale"
        or name.startswith("patch_encoder.")
        or name.startswith("query_proj_for_patch.")
    )


def _audit_loaded_checkpoint(
    model: torch.nn.Module, checkpoint_path: Path
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise OriginalGDINOExtractionError("parent checkpoint lacks model state")
    provided = clean_state_dict(state)
    root = model.module if hasattr(model, "module") else model
    runtime = root.state_dict()
    extras = sorted(set(provided).difference(runtime))
    missing = sorted(set(runtime).difference(provided))
    mismatch = sorted(
        key for key in set(runtime).intersection(provided)
        if tuple(runtime[key].shape) != tuple(provided[key].shape)
    )
    if (
        len(runtime) != PURE_TRUNK_TENSORS
        or sum(int(value.numel()) for value in runtime.values()) != PURE_TRUNK_NUMEL
        or len(extras) != PARENT_UNUSED_PATCH_TENSORS
        or any(not _unused_parent_tensor(name) for name in extras)
        or missing
        or mismatch
    ):
        raise OriginalGDINOExtractionError(
            "parent/runtime ownership drifted: "
            f"runtime={len(runtime)}, extras={len(extras)}, "
            f"missing={missing[:4]}, mismatch={mismatch[:4]}"
        )
    unequal = [
        key for key in sorted(runtime)
        if not torch.equal(runtime[key].detach().cpu(), provided[key].detach().cpu())
    ]
    if unequal:
        raise OriginalGDINOExtractionError(
            f"loaded pure trunk is not bitwise equal: {unequal[:4]}"
        )
    if any(
        tensor.is_floating_point()
        and not bool(torch.isfinite(tensor).all().item())
        for tensor in provided.values()
    ):
        raise OriginalGDINOExtractionError("parent checkpoint is non-finite")
    return {
        "checkpoint_tensor_count": len(provided),
        "runtime_tensor_count": len(runtime),
        "runtime_numel": sum(int(value.numel()) for value in runtime.values()),
        "unused_patch_tensor_count": len(extras),
        "unused_patch_tensor_names_sha256": __import__("hashlib").sha256(
            json.dumps(extras, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "loaded_runtime_parity": "bitwise_equal",
    }


def original_expression_mean(
    logits: Tensor, phrase_to_token_mask: Tensor
) -> Tensor:
    if (
        not torch.is_tensor(logits)
        or logits.dim() != 3
        or not torch.is_tensor(phrase_to_token_mask)
        or phrase_to_token_mask.dtype != torch.bool
        or phrase_to_token_mask.dim() != 3
        or int(phrase_to_token_mask.shape[0]) != int(logits.shape[0])
        or int(phrase_to_token_mask.shape[2]) != int(logits.shape[2])
    ):
        raise OriginalGDINOExtractionError("expression score inputs are malformed")
    mask = phrase_to_token_mask.any(dim=1)
    count = mask.sum(dim=1)
    if bool((count == 0).any().item()):
        raise OriginalGDINOExtractionError("full expression token mask is empty")
    score = (
        logits.detach().float().sigmoid().masked_fill(~mask[:, None, :], 0.0)
        .sum(dim=2) / count[:, None].float()
    )
    if not bool(torch.isfinite(score).all().item()):
        raise OriginalGDINOExtractionError("native expression score is non-finite")
    return score


def preprocess_original_caption(caption: str) -> str:
    if not isinstance(caption, str) or not caption.strip():
        raise OriginalGDINOExtractionError("caption must be non-empty")
    value = caption.lower().strip()
    return value if value.endswith(".") else value + "."


class OriginalGDINOFrozenRuntime:
    def __init__(
        self,
        *,
        config_path: Path,
        checkpoint_path: Path,
        device: str,
        feature_dtype: torch.dtype,
        amp: bool = True,
    ) -> None:
        if file_sha256(config_path) != EVAL_CONFIG_SHA256:
            raise OriginalGDINOExtractionError("evaluation config drifted")
        if file_sha256(checkpoint_path) != CHECKPOINT_SHA256:
            raise OriginalGDINOExtractionError("parent checkpoint drifted")
        cfg = SLConfig.fromfile(str(config_path))
        if not bool(
            getattr(cfg, "stage_b_original_gdino_parent_ownership_eval", False)
        ):
            raise OriginalGDINOExtractionError("evaluation config marker is absent")
        forbidden = (
            "stage_b", "patch_only", "enable_patch_branch",
            "stage_b_gdino_score_adapter", "stage_b_u0_patch_rank",
            "stage_b_data_driven_score", "stage_b_native_patch_category",
        )
        if any(bool(getattr(cfg, name, False)) for name in forbidden):
            raise OriginalGDINOExtractionError("evaluation enabled a Stage-B branch")
        self.device = torch.device(device)
        cfg.device = str(self.device)
        self.model, self.checkpoint_summary = _load_model_with_checkpoint_contract(
            cfg, checkpoint_path, self.device
        )
        self.ownership = _audit_loaded_checkpoint(self.model, checkpoint_path)
        self.model.requires_grad_(False)
        self.model.eval()
        if self.model.training or any(
            parameter.requires_grad for parameter in self.model.parameters()
        ):
            raise OriginalGDINOExtractionError("frozen runtime left trainable state")
        self.feature_dtype = feature_dtype
        self.amp = bool(amp)
        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

    def infer(self, image_path: Path, caption: str) -> HookBatch:
        caption = preprocess_original_caption(caption)
        with Image.open(image_path) as source:
            image, _ = self.transform(source.convert("RGB"), None)
        image = image.to(self.device)
        with torch.inference_mode(), torch.cuda.amp.autocast(
            enabled=self.amp and self.device.type == "cuda"
        ):
            outputs = self.model(
                image[None],
                captions=[caption],
                return_stage_b_v7_features=True,
                stage_b_data_driven_return_main_phrase_mask=True,
            )
        features = outputs.get("hs")
        logits = outputs.get("pred_logits")
        boxes = outputs.get("pred_boxes")
        phrase = outputs.get("phrase_to_token_mask")
        if (
            not torch.is_tensor(features)
            or tuple(features.shape) != (1, EXPECTED_QUERY_COUNT, CACHE_FEATURE_DIM)
            or not torch.is_tensor(logits)
            or tuple(logits.shape[:2]) != (1, EXPECTED_QUERY_COUNT)
            or not torch.is_tensor(boxes)
            or tuple(boxes.shape) != (1, EXPECTED_QUERY_COUNT, 4)
        ):
            raise OriginalGDINOExtractionError("pure GDINO outputs are misaligned")
        score = original_expression_mean(logits, phrase)
        candidate_mask = torch.isfinite(logits).any(dim=-1)
        if not bool(candidate_mask.all().item()):
            raise OriginalGDINOExtractionError("pure GDINO lost a candidate query")
        if not bool(torch.isfinite(features).all().item()):
            raise OriginalGDINOExtractionError("decoder feature is non-finite")
        if not bool(torch.isfinite(boxes).all().item()):
            raise OriginalGDINOExtractionError("box is non-finite")
        return HookBatch(
            query_features=features[0].detach().to(
                device="cpu", dtype=self.feature_dtype
            ).contiguous(),
            native_score=score[0].detach().cpu().float().contiguous(),
            boxes=boxes[0].detach().cpu().float().contiguous(),
            candidate_mask=candidate_mask[0].detach().cpu().bool().contiguous(),
        )

    def close(self) -> None:
        del self.model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def _source(
    *, model_id: str, checkpoint_sha256: str, config_sha256: str
) -> dict[str, str]:
    return {
        "schema": CACHE_SOURCE_SCHEMA,
        "model_id": model_id,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "extractor_code_sha256": file_sha256(Path(__file__)),
        "query_feature_name": QUERY_FEATURE_NAME,
    }


def extract_training(
    *,
    rank_jsonl: Path,
    rank_jsonl_sha256: str,
    d3_jsonl: Path,
    d3_jsonl_sha256: str,
    image_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    model_id: str,
    shard_id: str,
    output: Path,
    receipt: Path,
    device: str,
    rank_limit: int = 0,
    pair_limit: int = 0,
) -> dict[str, Any]:
    if output.exists() or receipt.exists():
        raise OriginalGDINOExtractionError("training output already exists")
    rank_rows = _read_bound_jsonl(
        rank_jsonl, expected_sha256=rank_jsonl_sha256, name="rank_jsonl"
    )
    d3_rows = _read_bound_jsonl(
        d3_jsonl, expected_sha256=d3_jsonl_sha256, name="d3_jsonl"
    )
    if rank_limit:
        rank_rows = rank_rows[:rank_limit]
    if pair_limit:
        d3_rows = d3_rows[:pair_limit]
    image_cache: dict[Path, tuple[int, int, str]] = {}
    rank_requests = parse_refcoco_rank_requests(
        rank_rows, image_root=image_root, image_cache=image_cache
    )
    pair_requests = parse_d3_pair_requests(
        d3_rows, image_root=image_root, image_cache=image_cache
    )
    runtime = OriginalGDINOFrozenRuntime(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        feature_dtype=torch.float32,
    )
    try:
        shard, counters = extract_cached_candidate_shard(
            rank_requests=rank_requests,
            pair_requests=pair_requests,
            runtime=runtime,
            shard_id=shard_id,
            checkpoint_sha256=checkpoint_sha256,
            extractor_sha256=file_sha256(Path(__file__)),
            model_id=model_id,
            config_sha256=file_sha256(config_path),
            allow_rank_rows_without_positive=True,
        )
        shard = dict(shard)
        shard["source"] = _source(
            model_id=model_id,
            checkpoint_sha256=checkpoint_sha256,
            config_sha256=file_sha256(config_path),
        )
        shard = validate_cached_candidate_shard(
            shard, allow_rank_rows_without_positive=True
        )
        ownership = runtime.ownership
        checkpoint_summary = runtime.checkpoint_summary
    finally:
        runtime.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        hashes = save_cached_candidate_shard(
            shard, temporary, allow_rank_rows_without_positive=True
        )
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    all_requests = tuple(rank_requests) + tuple(pair_requests)
    result = {
        "schema": TRAINING_RECEIPT_SCHEMA,
        "status": "complete",
        "shard_id": shard_id,
        "source": shard["source"],
        "assets": {
            "checkpoint_path": str(checkpoint_path.resolve(strict=True)),
            "checkpoint_sha256": checkpoint_sha256,
            "config_path": str(config_path.resolve(strict=True)),
            "config_sha256": file_sha256(config_path),
            "extractor_path": str(Path(__file__).resolve(strict=True)),
            "extractor_code_sha256": file_sha256(Path(__file__)),
            "ownership": ownership,
            "checkpoint_summary": checkpoint_summary,
        },
        "inputs": {
            "rank_jsonl": {
                "path": str(rank_jsonl.resolve(strict=True)),
                "sha256": rank_jsonl_sha256,
                "selected_rows": len(rank_rows),
                "image_root": str(image_root.resolve(strict=True)),
            },
            "d3_jsonl": {
                "path": str(d3_jsonl.resolve(strict=True)),
                "sha256": d3_jsonl_sha256,
                "selected_pairs": len(d3_rows),
                "image_root": str(image_root.resolve(strict=True)),
            },
            "unique_image_count": len(image_cache),
            "image_identity_sha256": _image_set_sha256(all_requests),
        },
        "output": {
            "path": str(output.resolve(strict=True)),
            **hashes,
            "row_count": len(shard["rows"]),
        },
        "counters": counters,
        "runtime": {
            "device": device,
            "feature_dtype": "float32",
            "batch_size": 1,
            "training": False,
            "native_score": "full_expression_probability_mean",
            "rank_rows_without_positive_policy": "preserve_and_zero_margin",
        },
    }
    _atomic_json_dump(result, receipt)
    return result


def extract_evaluation(
    *,
    mode: str,
    input_jsonl: Path,
    input_sha256: str,
    image_root: Path,
    surface: str,
    config_path: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    model_id: str,
    output: Path,
    receipt: Path,
    device: str,
) -> dict[str, Any]:
    if mode not in ("ref", "tn"):
        raise OriginalGDINOExtractionError("evaluation mode must be ref or tn")
    if output.exists() or receipt.exists():
        raise OriginalGDINOExtractionError("evaluation output already exists")
    rows = _read_jsonl(input_jsonl, input_sha256)
    requests = (
        parse_refcoco_rank_requests(rows, image_root=image_root)
        if mode == "ref"
        else parse_tn_eval_requests(rows, image_root=image_root)
    )
    runtime = OriginalGDINOFrozenRuntime(
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
                raise OriginalGDINOExtractionError(
                    f"image bytes changed: {request.image_path}"
                )
            row = _cache_row(
                request, runtime.infer(request.image_path, request.caption)
            )
            if request.task == CACHE_TASK_RANK:
                iou = normalized_cxcywh_iou(row["boxes"], row["gt_boxes"]).amax(dim=1)
                oracle_rows += int(bool((row["candidate_mask"] & (iou >= 0.5)).any()))
            cache_rows.append(
                validate_cached_candidate_row(
                    row, require_trainable_rank_pair=(mode != "ref")
                )
            )
        ownership = runtime.ownership
        checkpoint_summary = runtime.checkpoint_summary
    finally:
        runtime.close()
    task = CACHE_TASK_RANK if mode == "ref" else CACHE_TASK_CONFIDENCE_PAIR
    payload = {
        "schema": EVAL_CACHE_SCHEMA,
        "surface": surface,
        "task": task,
        "source": _source(
            model_id=model_id,
            checkpoint_sha256=checkpoint_sha256,
            config_sha256=file_sha256(config_path),
        ),
        "feature_dim": CACHE_FEATURE_DIM,
        "box_format": CACHE_BOX_FORMAT,
        "rows": tuple(cache_rows),
    }
    _atomic_torch(payload, output)
    result = {
        "schema": EVALUATION_RECEIPT_SCHEMA,
        "status": "complete",
        "mode": mode,
        "surface": surface,
        "assets": {
            "checkpoint_sha256": checkpoint_sha256,
            "config_sha256": file_sha256(config_path),
            "model_id": model_id,
            "extractor_code_sha256": file_sha256(Path(__file__)),
            "ownership": ownership,
            "checkpoint_summary": checkpoint_summary,
        },
        "input": {
            "path": str(input_jsonl.resolve(strict=True)),
            "sha256": input_sha256,
            "manifest_rows": len(rows),
            "request_rows": len(requests),
            "image_identity_sha256": _image_set_sha256(requests),
        },
        "output": {
            **file_record(output, rows=len(cache_rows)),
            "file_sha256": file_sha256(output),
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
            "native_score": "full_expression_probability_mean",
        },
    }
    _atomic_json(result, receipt)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("training", "ref", "tn"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--input-sha256")
    parser.add_argument("--surface")
    parser.add_argument("--rank-jsonl", type=Path)
    parser.add_argument("--rank-jsonl-sha256")
    parser.add_argument("--d3-jsonl", type=Path)
    parser.add_argument("--d3-jsonl-sha256")
    parser.add_argument("--shard-id")
    parser.add_argument("--rank-limit", type=int, default=0)
    parser.add_argument("--pair-limit", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    common = {
        "config_path": args.config.resolve(strict=True),
        "checkpoint_path": args.checkpoint.resolve(strict=True),
        "checkpoint_sha256": args.checkpoint_sha256,
        "model_id": args.model_id,
        "output": args.output.resolve(),
        "receipt": args.receipt.resolve(),
        "device": args.device,
        "image_root": args.image_root.resolve(strict=True),
    }
    if args.checkpoint_sha256 != CHECKPOINT_SHA256:
        raise OriginalGDINOExtractionError("checkpoint identity is not the locked parent")
    if args.mode == "training":
        required = (
            args.rank_jsonl, args.rank_jsonl_sha256, args.d3_jsonl,
            args.d3_jsonl_sha256, args.shard_id,
        )
        if any(value is None for value in required):
            raise OriginalGDINOExtractionError("training inputs are incomplete")
        result = extract_training(
            rank_jsonl=args.rank_jsonl.resolve(strict=True),
            rank_jsonl_sha256=args.rank_jsonl_sha256,
            d3_jsonl=args.d3_jsonl.resolve(strict=True),
            d3_jsonl_sha256=args.d3_jsonl_sha256,
            shard_id=args.shard_id,
            rank_limit=args.rank_limit,
            pair_limit=args.pair_limit,
            **common,
        )
    else:
        if args.input_jsonl is None or args.input_sha256 is None or args.surface is None:
            raise OriginalGDINOExtractionError("evaluation inputs are incomplete")
        result = extract_evaluation(
            mode=args.mode,
            input_jsonl=args.input_jsonl.resolve(strict=True),
            input_sha256=args.input_sha256,
            surface=args.surface,
            **common,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "EVALUATION_RECEIPT_SCHEMA", "OriginalGDINOExtractionError",
    "OriginalGDINOFrozenRuntime", "QUERY_FEATURE_NAME",
    "TRAINING_RECEIPT_SCHEMA", "extract_evaluation", "extract_training",
    "original_expression_mean", "preprocess_original_caption",
]
