#!/usr/bin/env python3
"""Build a v14 pretrain checkpoint with a decoder warm-start from v5.x.

The output is intentionally model-only and must be consumed with
``--pretrain_model_path``. The detector/localization state comes byte-for-byte
from the base checkpoint; only the fixed-text scorer's decoder layers,
reference-point head, and final norm are replaced.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SCORER_PREFIX = "stage_b_fixed_text_scorer."
TARGET_DECODER_PREFIX = f"{SCORER_PREFIX}decoder."
SOURCE_DECODER_PREFIX = "transformer.decoder."
VALIDITY_PREFIX = f"{SCORER_PREFIX}validity_head."
TARGET_LAYER_RE = re.compile(
    rf"^{re.escape(TARGET_DECODER_PREFIX)}layers\.(\d+)\.(.+)$"
)
SOURCE_LAYER_RE = re.compile(
    rf"^{re.escape(SOURCE_DECODER_PREFIX)}layers\.(\d+)\.(.+)$"
)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_model_state(checkpoint: Any, *, label: str) -> OrderedDict[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"{label} checkpoint must be a mapping")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, Mapping):
        raise TypeError(f"{label} model state must be a mapping")

    cleaned: OrderedDict[str, Any] = OrderedDict()
    for raw_key, value in state.items():
        key = str(raw_key)
        if key.startswith("module."):
            key = key[7:]
        if key in cleaned:
            raise ValueError(f"{label} contains a duplicate key after cleaning: {key}")
        cleaned[key] = value
    return cleaned


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    byte_view = tensor.view(torch.uint8)
    return hashlib.sha256(byte_view.numpy().tobytes()).hexdigest()


def _layer_indices(state: Mapping[str, Any], pattern: re.Pattern[str]) -> list[int]:
    return sorted(
        {
            int(match.group(1))
            for key in state
            if (match := pattern.match(str(key))) is not None
        }
    )


def _assert_tensor_compatible(
    *,
    target_key: str,
    target_value: Any,
    source_key: str,
    source_value: Any,
) -> None:
    if not torch.is_tensor(target_value) or not torch.is_tensor(source_value):
        raise TypeError(
            f"mapped values must be tensors: {target_key}={type(target_value).__name__}, "
            f"{source_key}={type(source_value).__name__}"
        )
    if tuple(target_value.shape) != tuple(source_value.shape):
        raise ValueError(
            f"shape mismatch: {target_key} {tuple(target_value.shape)} versus "
            f"{source_key} {tuple(source_value.shape)}"
        )
    if target_value.dtype != source_value.dtype:
        raise ValueError(
            f"dtype mismatch: {target_key} {target_value.dtype} versus "
            f"{source_key} {source_value.dtype}"
        )


def _build_mapping(
    base_state: Mapping[str, Any], source_state: Mapping[str, Any]
) -> tuple[OrderedDict[str, str], dict[str, Any]]:
    validity_keys = sorted(key for key in base_state if key.startswith(VALIDITY_PREFIX))
    if validity_keys:
        raise ValueError(
            "the base checkpoint already contains a validity head; use the v12 scorer "
            f"checkpoint so v14 keeps its constructor initialization: {validity_keys}"
        )

    target_layer_indices = _layer_indices(base_state, TARGET_LAYER_RE)
    source_layer_indices = _layer_indices(source_state, SOURCE_LAYER_RE)
    if not target_layer_indices:
        raise ValueError("base checkpoint has no fixed-text scorer decoder layers")
    if target_layer_indices != list(range(len(target_layer_indices))):
        raise ValueError(
            "target scorer layers must be contiguous from zero: "
            f"{target_layer_indices}"
        )
    if source_layer_indices != list(range(len(source_layer_indices))):
        raise ValueError(
            "source decoder layers must be contiguous from zero: "
            f"{source_layer_indices}"
        )
    if len(source_layer_indices) < len(target_layer_indices):
        raise ValueError(
            f"source has only {len(source_layer_indices)} decoder layers, but target needs "
            f"{len(target_layer_indices)}"
        )
    selected_source_layers = source_layer_indices[-len(target_layer_indices) :]

    mapping: OrderedDict[str, str] = OrderedDict()
    group_counts = {"layers": 0, "ref_point_head": 0, "norm": 0}
    for target_key, target_value in base_state.items():
        match = TARGET_LAYER_RE.match(target_key)
        if match is not None:
            target_layer = int(match.group(1))
            suffix = match.group(2)
            source_layer = selected_source_layers[target_layer]
            source_key = f"{SOURCE_DECODER_PREFIX}layers.{source_layer}.{suffix}"
            group = "layers"
        elif target_key.startswith(f"{TARGET_DECODER_PREFIX}ref_point_head."):
            suffix = target_key[len(f"{TARGET_DECODER_PREFIX}ref_point_head.") :]
            source_key = f"{SOURCE_DECODER_PREFIX}ref_point_head.{suffix}"
            group = "ref_point_head"
        elif target_key.startswith(f"{TARGET_DECODER_PREFIX}norm."):
            suffix = target_key[len(f"{TARGET_DECODER_PREFIX}norm.") :]
            source_key = f"{SOURCE_DECODER_PREFIX}norm.{suffix}"
            group = "norm"
        elif target_key.startswith(TARGET_DECODER_PREFIX):
            raise ValueError(f"unsupported scorer decoder state key: {target_key}")
        else:
            continue

        if source_key not in source_state:
            raise KeyError(f"source checkpoint is missing {source_key} for {target_key}")
        _assert_tensor_compatible(
            target_key=target_key,
            target_value=target_value,
            source_key=source_key,
            source_value=source_state[source_key],
        )
        if any(fragment in target_key for fragment in ("bbox_embed", "class_embed")):
            raise ValueError(f"refusing to map a box/class prediction head: {target_key}")
        mapping[target_key] = source_key
        group_counts[group] += 1

    target_decoder_keys = {
        key for key in base_state if key.startswith(TARGET_DECODER_PREFIX)
    }
    if set(mapping) != target_decoder_keys:
        missing = sorted(target_decoder_keys.difference(mapping))
        raise ValueError(f"not all scorer decoder keys were mapped: {missing}")
    empty_groups = sorted(group for group, count in group_counts.items() if count == 0)
    if empty_groups:
        raise ValueError(f"required scorer decoder groups are empty: {empty_groups}")

    return mapping, {
        "target_layer_indices": target_layer_indices,
        "source_layer_indices": source_layer_indices,
        "selected_source_layers": selected_source_layers,
        "group_counts": group_counts,
    }


def _validate_v14_load(
    *, config_path: Path, output_state: Mapping[str, Any]
) -> dict[str, Any]:
    from models.registry import MODULE_BUILD_FUNCS
    from util.slconfig import SLConfig

    cfg = SLConfig.fromfile(str(config_path))
    cfg.device = "cpu"
    build_func = MODULE_BUILD_FUNCS.get(cfg.modelname)
    if build_func is None:
        raise KeyError(f"unknown modelname={cfg.modelname}")
    model, _criterion, _postprocessors = build_func(cfg)

    scorer = getattr(model, "stage_b_fixed_text_scorer", None)
    if scorer is None or getattr(scorer, "validity_head", None) is None:
        raise ValueError(f"{config_path} did not build a v14 validity head")
    validity_before = {
        key: value.detach().clone()
        for key, value in scorer.validity_head.state_dict().items()
    }
    load_result = model.load_state_dict(output_state, strict=False)
    missing = sorted(load_result.missing_keys)
    unexpected = sorted(load_result.unexpected_keys)
    expected_missing = sorted(
        key for key in model.state_dict() if key.startswith(VALIDITY_PREFIX)
    )
    if missing != expected_missing or unexpected:
        raise ValueError(
            "v14 load compatibility failed: "
            f"missing={missing}, expected_missing={expected_missing}, unexpected={unexpected}"
        )

    validity_after = scorer.validity_head.state_dict()
    preserved = all(
        torch.equal(validity_before[key], validity_after[key])
        for key in validity_before
    )
    final_layer = scorer.validity_head[-1]
    final_layer_zero = bool(
        torch.count_nonzero(final_layer.weight).item() == 0
        and torch.count_nonzero(final_layer.bias).item() == 0
    )
    if not preserved or not final_layer_zero:
        raise ValueError(
            "loading the hybrid checkpoint modified the validity head or its residual "
            "output layer is not zero-initialized"
        )
    if scorer.decoder.bbox_embed is not None or scorer.decoder.class_embed is not None:
        raise ValueError("fixed-text scorer unexpectedly owns a box/class prediction head")

    return {
        "config": str(config_path.resolve()),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "only_missing_validity_head": True,
        "validity_state_preserved_across_load": preserved,
        "validity_output_layer_all_zero": final_layer_zero,
        "scorer_bbox_embed_is_none": scorer.decoder.bbox_embed is None,
        "scorer_class_embed_is_none": scorer.decoder.class_embed is None,
    }


def _validate_serialized_checkpoint(
    *, output_path: Path, expected_state: Mapping[str, Any]
) -> dict[str, Any]:
    checkpoint = _torch_load(output_path)
    if not isinstance(checkpoint, Mapping) or sorted(checkpoint) != [
        "hybrid_init",
        "model",
    ]:
        raise ValueError(
            f"serialized output has unexpected top-level keys: "
            f"{sorted(checkpoint) if isinstance(checkpoint, Mapping) else type(checkpoint)}"
        )
    serialized_state = _extract_model_state(checkpoint, label="serialized output")
    if list(serialized_state) != list(expected_state):
        raise ValueError("serialized output model keys/order differ from the built state")
    mismatches = [
        key
        for key in expected_state
        if not torch.equal(serialized_state[key], expected_state[key])
    ]
    if mismatches:
        raise ValueError(f"serialized output tensor mismatches: {mismatches[:20]}")
    return {
        "top_level_keys": sorted(checkpoint),
        "model_key_count": len(serialized_state),
        "all_model_tensors_exact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="v12 base checkpoint")
    parser.add_argument(
        "--decoder-source", type=Path, required=True, help="v5.x decoder checkpoint"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "config/ablations/cfg_stageb_v14_phrase_validity_cvar.py",
        help="v14 config used for model-load validation",
    )
    parser.add_argument("--skip-model-validation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base_path = args.base.resolve()
    source_path = args.decoder_source.resolve()
    output_path = args.output.resolve()
    audit_path = (
        args.audit_json.resolve()
        if args.audit_json is not None
        else output_path.with_suffix(output_path.suffix + ".audit.json")
    )
    for label, path in (("base", base_path), ("decoder source", source_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} checkpoint does not exist: {path}")
    if output_path == base_path or output_path == source_path:
        raise ValueError("output must not overwrite either input checkpoint")
    if audit_path in {base_path, source_path, output_path}:
        raise ValueError("audit JSON must be distinct from both inputs and the output")
    if not args.skip_model_validation and not args.config.is_file():
        raise FileNotFoundError(f"v14 validation config does not exist: {args.config}")
    if not args.overwrite:
        for label, path in (("output", output_path), ("audit", audit_path)):
            if path.exists():
                raise FileExistsError(f"{label} already exists: {path}; pass --overwrite")

    print("[1/5] hashing inputs", flush=True)
    base_file_sha256 = _sha256_file(base_path)
    source_file_sha256 = _sha256_file(source_path)
    print("[2/5] loading checkpoints", flush=True)
    base_checkpoint = _torch_load(base_path)
    source_checkpoint = _torch_load(source_path)
    base_state = _extract_model_state(base_checkpoint, label="base")
    source_state = _extract_model_state(source_checkpoint, label="decoder source")
    source_model_key_count = len(source_state)

    mapping, mapping_summary = _build_mapping(base_state, source_state)
    output_state: OrderedDict[str, Any] = OrderedDict(base_state)
    mapping_rows = []
    for target_key, source_key in mapping.items():
        base_value = base_state[target_key]
        source_value = source_state[source_key]
        output_state[target_key] = source_value.detach().clone()
        mapping_rows.append(
            {
                "target": target_key,
                "source": source_key,
                "shape": list(source_value.shape),
                "dtype": str(source_value.dtype),
                "base_sha256": _sha256_tensor(base_value),
                "source_sha256": _sha256_tensor(source_value),
                "changed_from_base": not torch.equal(base_value, source_value),
            }
        )

    changed_keys = sorted(
        key
        for key in mapping
        if not torch.equal(base_state[key], output_state[key])
    )
    non_target_keys = [key for key in output_state if key not in mapping]
    non_target_same_object = all(
        output_state[key] is base_state[key] for key in non_target_keys
    )
    if not non_target_same_object:
        raise RuntimeError("a non-target base value was replaced")
    if any(not key.startswith(TARGET_DECODER_PREFIX) for key in changed_keys):
        raise RuntimeError(f"a changed key escaped the scorer decoder: {changed_keys}")

    provenance = {
        "format_version": 1,
        "purpose": "stage_b_v14_hybrid_pretrain_initialization",
        "use_with": "--pretrain_model_path (never --resume)",
        "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": base_file_sha256,
        "decoder_source_checkpoint": str(source_path),
        "decoder_source_checkpoint_sha256": source_file_sha256,
        "mapped_tensor_count": len(mapping),
        "validity_head": (
            "intentionally absent; v14 constructor state is preserved and its final "
            "residual layer remains zero"
        ),
    }
    validation: dict[str, Any]
    if args.skip_model_validation:
        validation = {"skipped": True}
    else:
        print(f"[3/5] validating load against {args.config}", flush=True)
        validation = _validate_v14_load(
            config_path=args.config.resolve(), output_state=output_state
        )

    del source_state, source_checkpoint, base_checkpoint
    gc.collect()
    output_checkpoint = {"model": output_state, "hybrid_init": provenance}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    print(f"[4/5] saving model-only checkpoint to {output_path}", flush=True)
    try:
        torch.save(output_checkpoint, temporary_path)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    output_file_sha256 = _sha256_file(output_path)
    serialized_validation = _validate_serialized_checkpoint(
        output_path=output_path, expected_state=output_state
    )

    detector_keys = [key for key in output_state if not key.startswith(SCORER_PREFIX)]
    box_related_keys = [
        key
        for key in output_state
        if any(fragment in key for fragment in ("bbox_embed", "refpoint_embed"))
        and not key.startswith(SCORER_PREFIX)
    ]
    audit = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base": {
            "path": str(base_path),
            "file_sha256": base_file_sha256,
            "model_key_count": len(base_state),
        },
        "decoder_source": {
            "path": str(source_path),
            "file_sha256": source_file_sha256,
            "model_key_count": source_model_key_count,
        },
        "output": {
            "path": str(output_path),
            "file_sha256": output_file_sha256,
            "bytes": output_path.stat().st_size,
            "top_level_keys": sorted(output_checkpoint),
            "model_key_count": len(output_state),
        },
        "mapping": {
            **mapping_summary,
            "target_prefix": TARGET_DECODER_PREFIX,
            "tensor_count": len(mapping),
            "changed_tensor_count": len(changed_keys),
            "rows": mapping_rows,
        },
        "safety": {
            "base_validity_head_keys": [],
            "validity_head_written_by_tool": False,
            "non_target_key_count": len(non_target_keys),
            "non_target_values_preserved": non_target_same_object,
            "non_scorer_detector_key_count": len(detector_keys),
            "non_scorer_box_related_key_count": len(box_related_keys),
            "changed_keys_outside_fixed_text_scorer_decoder": [],
            "old_optimizer_or_scheduler_copied": False,
        },
        "model_load_validation": validation,
        "serialized_checkpoint_validation": serialized_validation,
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[5/5] wrote audit to {audit_path}", flush=True)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "output_sha256": output_file_sha256,
                "audit_json": str(audit_path),
                "mapped_tensor_count": len(mapping),
                "changed_tensor_count": len(changed_keys),
                "selected_source_layers": mapping_summary[
                    "selected_source_layers"
                ],
                "validation": validation,
                "serialized_validation": serialized_validation,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
