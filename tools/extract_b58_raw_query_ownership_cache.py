#!/usr/bin/env python3
"""Extract B58 candidates for the matched 100k raw-query ownership replay."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

import datasets.transforms as T
import tools.extract_original_gdino_ownership_cache as engine
from tools.b58_raw_query_ownership import (
    CHECKPOINT_SHA256,
    EVAL_CONFIG_SHA256,
    PURE_TRUNK_NUMEL,
    PURE_TRUNK_SCHEMA_SHA256,
    PURE_TRUNK_TENSORS,
)
from tools.eval_text_groundingdino_refcoco_tn import (
    _load_model_with_checkpoint_contract,
)
from tools.responsibility_isolation_cache import file_sha256
from util.slconfig import SLConfig
from util.utils import clean_state_dict


TRAINING_RECEIPT_SCHEMA = "arrow.b58_raw_query_ownership.cache_extraction_receipt/v1"
EVALUATION_RECEIPT_SCHEMA = "arrow.b58_raw_query_ownership.eval_cache_receipt/v1"


class B58RawQueryExtractionError(engine.OriginalGDINOExtractionError):
    pass


def _schema(state: Mapping[str, torch.Tensor]) -> str:
    rows = [
        [name, str(state[name].dtype), list(state[name].shape)]
        for name in sorted(state)
    ]
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _audit_loaded_checkpoint(
    model: torch.nn.Module, checkpoint_path: Path
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise B58RawQueryExtractionError("B58 checkpoint lacks model state")
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
        len(provided) != PURE_TRUNK_TENSORS
        or len(runtime) != PURE_TRUNK_TENSORS
        or sum(int(value.numel()) for value in provided.values())
        != PURE_TRUNK_NUMEL
        or _schema(provided) != PURE_TRUNK_SCHEMA_SHA256
        or extras
        or missing
        or mismatch
    ):
        raise B58RawQueryExtractionError(
            "B58/runtime ownership drifted: "
            f"provided={len(provided)}, runtime={len(runtime)}, "
            f"extras={extras[:4]}, missing={missing[:4]}, mismatch={mismatch[:4]}"
        )
    unequal = [
        key for key in sorted(runtime)
        if not torch.equal(runtime[key].detach().cpu(), provided[key].detach().cpu())
    ]
    if unequal:
        raise B58RawQueryExtractionError(
            f"loaded B58 trunk is not bitwise equal: {unequal[:4]}"
        )
    if any(
        value.is_floating_point() and not bool(torch.isfinite(value).all().item())
        for value in provided.values()
    ):
        raise B58RawQueryExtractionError("B58 checkpoint is non-finite")
    return {
        "checkpoint_tensor_count": len(provided),
        "runtime_tensor_count": len(runtime),
        "runtime_numel": sum(int(value.numel()) for value in runtime.values()),
        "unused_tensor_count": 0,
        "pure_trunk_schema_sha256": _schema(provided),
        "loaded_runtime_parity": "bitwise_equal",
    }


class B58FrozenRuntime(engine.OriginalGDINOFrozenRuntime):
    """The original extractor path with a B58-specific identity audit."""

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
            raise B58RawQueryExtractionError("B58 evaluation config drifted")
        if file_sha256(checkpoint_path) != CHECKPOINT_SHA256:
            raise B58RawQueryExtractionError("B58 checkpoint drifted")
        cfg = SLConfig.fromfile(str(config_path))
        if not bool(getattr(cfg, "stage_b_b58_raw_query_ownership_eval", False)):
            raise B58RawQueryExtractionError("B58 evaluation marker is absent")
        forbidden = (
            "stage_b", "patch_only", "enable_patch_branch",
            "stage_b_gdino_score_adapter", "stage_b_u0_patch_rank",
            "stage_b_data_driven_score", "stage_b_native_patch_category",
        )
        if any(bool(getattr(cfg, name, False)) for name in forbidden):
            raise B58RawQueryExtractionError("evaluation enabled a Stage-B branch")
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
            raise B58RawQueryExtractionError("B58 runtime left trainable state")
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


@contextlib.contextmanager
def _runtime_context():
    previous = engine.OriginalGDINOFrozenRuntime
    try:
        engine.OriginalGDINOFrozenRuntime = B58FrozenRuntime
        yield
    finally:
        engine.OriginalGDINOFrozenRuntime = previous


def _stamp_result(
    result: dict[str, Any], *, receipt: Path, schema: str, mode: str
) -> dict[str, Any]:
    result["schema"] = schema
    result["assets"]["runtime_adapter"] = {
        "path": str(Path(__file__).resolve(strict=True)),
        "sha256": file_sha256(Path(__file__)),
        "shared_extraction_engine_path": str(Path(engine.__file__).resolve()),
        "shared_extraction_engine_sha256": file_sha256(Path(engine.__file__)),
    }
    result["runtime"]["frozen_trunk"] = "B58"
    result["runtime"]["matched_owner_contract"] = "100k_raw_query/v1"
    if mode == "training":
        engine._atomic_json_dump(result, receipt)
    else:
        engine._atomic_json(result, receipt)
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
    if args.checkpoint_sha256 != CHECKPOINT_SHA256:
        raise B58RawQueryExtractionError("checkpoint identity is not locked B58")
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
    with _runtime_context():
        if args.mode == "training":
            required = (
                args.rank_jsonl, args.rank_jsonl_sha256, args.d3_jsonl,
                args.d3_jsonl_sha256, args.shard_id,
            )
            if any(value is None for value in required):
                raise B58RawQueryExtractionError("training inputs are incomplete")
            result = engine.extract_training(
                rank_jsonl=args.rank_jsonl.resolve(strict=True),
                rank_jsonl_sha256=args.rank_jsonl_sha256,
                d3_jsonl=args.d3_jsonl.resolve(strict=True),
                d3_jsonl_sha256=args.d3_jsonl_sha256,
                shard_id=args.shard_id,
                rank_limit=args.rank_limit,
                pair_limit=args.pair_limit,
                **common,
            )
            result = _stamp_result(
                result, receipt=args.receipt.resolve(),
                schema=TRAINING_RECEIPT_SCHEMA, mode="training",
            )
        else:
            if (
                args.input_jsonl is None
                or args.input_sha256 is None
                or args.surface is None
            ):
                raise B58RawQueryExtractionError("evaluation inputs are incomplete")
            result = engine.extract_evaluation(
                mode=args.mode,
                input_jsonl=args.input_jsonl.resolve(strict=True),
                input_sha256=args.input_sha256,
                surface=args.surface,
                **common,
            )
            result = _stamp_result(
                result, receipt=args.receipt.resolve(),
                schema=EVALUATION_RECEIPT_SCHEMA, mode=args.mode,
            )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "B58FrozenRuntime", "B58RawQueryExtractionError",
    "EVALUATION_RECEIPT_SCHEMA", "TRAINING_RECEIPT_SCHEMA", "main",
]
