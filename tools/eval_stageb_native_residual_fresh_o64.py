#!/usr/bin/env python3
"""Audit and evaluate the fresh, aspect-preserving native-residual O64 probe."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, SequentialSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataset  # noqa: E402
from engine import _build_stage_b_gdino_adapter_rank_captions  # noqa: E402
from main import build_model_main  # noqa: E402
from models.GroundingDINO.stage_b_gdino_score_adapter import (  # noqa: E402
    validate_stage_b_gdino_score_adapter_checkpoint,
)
from tools.build_stageb_native_residual_fresh_o64 import (  # noqa: E402
    OUTPUT_MANIFEST,
    OUTPUT_RECEIPT_SCHEMA,
    OUTPUT_ROW_SCHEMA,
    verify as verify_fresh_o64_artifact,
)
from tools.build_stageb_native_residual_initializer import (  # noqa: E402
    EXPECTED_B58_SHA256,
    validate_initializer_payload,
    verify_external_bindings,
)
from tools.build_stageb_u0_training_receipt import (  # noqa: E402
    _safe_load_checkpoint,
    stable_file_record,
)
from tools.eval_stageb_gdino_adapter_o64_direct_rank import (  # noqa: E402
    EXPECTED_BASE_TENSORS,
    EXPECTED_CONFIDENCE_TENSORS,
    EXPECTED_QUERIES,
    EXPECTED_RANK_TENSORS,
    O64DirectRankAuditError,
    _atomic_write_json,
    _audit_optimizer,
    _checkpoint_args,
    _exact_int,
    _finite_scalar,
    _model_state,
    _saved_path,
    _seed_everything,
    aggregate_o64_records,
    audit_b58_lineage,
    audit_batch_outputs,
    audit_tensor_isolation,
)
from util import misc as utils  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


SCHEMA = "pivot.stageb.native_residual.fresh_o64_eval/v1"
EXPECTED_ROWS = 128
EXPECTED_PAIRS = 64
EXPECTED_UPDATES = 500
EXPECTED_BATCH_SIZE = 32
EXPECTED_GRADIENT_ACCUMULATION_STEPS = 2
EXPECTED_EFFECTIVE_BATCH_SIZE = (
    EXPECTED_BATCH_SIZE * EXPECTED_GRADIENT_ACCUMULATION_STEPS
)
EXPECTED_EPOCHS = 250
EXPECTED_RANK_LR = 3.0e-4
EXPECTED_CONTRACT_VERSION = 2

EXPECTED_FRESH_MANIFEST_SHA256 = (
    "17569fc88babdfdc8b23ffb38d33983b3e64c427badce50e1f3b47f8eb5b7433"
)
EXPECTED_FRESH_RECEIPT_SHA256 = (
    "7708f6e52a6542aa4a8a807b0b75853c74768b5c4edead1270c858fb254beeb8"
)

FreshO64AuditError = O64DirectRankAuditError


def _expected_config_contract() -> dict[str, Any]:
    return {
        "stage_b_native_residual_data_only": True,
        "stage_b_native_residual_contract_version": EXPECTED_CONTRACT_VERSION,
        "stage_b_gdino_score_adapter": True,
        "stage_b_gdino_adapter_train_mode": "rank_only",
        "stage_b_gdino_tn_scope": "",
        "stage_b_gdino_rank_weight": 1.0,
        "stage_b_gdino_confidence_weight": 0.0,
        "stage_b_gdino_paired_margin_weight": 0.0,
        "stage_b_gdino_queue_size": 0,
        "stage_b_gdino_queue_min_count": 0,
        "stage_b_gdino_rank_lr": EXPECTED_RANK_LR,
        "lr": EXPECTED_RANK_LR,
        "batch_size": EXPECTED_BATCH_SIZE,
        "epochs": EXPECTED_EPOCHS,
        "lr_drop": 1000,
        "fix_size": False,
        "data_aug_train_deterministic_aspect_resize": True,
        "strong_aug": False,
        "data_aug_hflip_prob": 0.0,
        "data_aug_scales": [800],
        "data_aug_max_size": 1333,
        "enable_patch_branch": False,
    }


def validate_config(cfg: Any) -> dict[str, Any]:
    expected = _expected_config_contract()
    drift = {
        key: {"observed": getattr(cfg, key, None), "expected": value}
        for key, value in expected.items()
        if getattr(cfg, key, None) != value
    }
    if drift:
        raise FreshO64AuditError(f"fresh O64 config drifted: {drift}")
    return dict(expected)


def validate_fresh_initializer(
    model: torch.nn.Module,
    payload: Mapping[str, Any],
    *,
    initializer_path: Path,
    config_path: Path,
    b58_path: Path,
) -> None:
    label = f"native-residual initializer {initializer_path}"
    validate_initializer_payload(model, payload, checkpoint_label=label)
    verify_external_bindings(
        payload,
        config=config_path,
        b58_path=b58_path,
        checkpoint_label=label,
    )


def _load_dataset_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FreshO64AuditError(
            f"could not read fresh O64 dataset config: {error}"
        ) from error
    if not isinstance(value, dict) or set(value) != {
        "artifact_binding",
        "train",
        "val",
    }:
        raise FreshO64AuditError(
            "fresh O64 dataset config must contain artifact_binding/train/val"
        )
    train = value.get("train")
    if not isinstance(train, list) or len(train) != 1 or value.get("val") != []:
        raise FreshO64AuditError(
            "fresh O64 dataset config requires one train entry and empty val"
        )
    entry = train[0]
    if (
        not isinstance(entry, dict)
        or entry.get("dataset_mode") != "odvg"
        or entry.get("root") != "/"
        or entry.get("mix_weight") != 1.0
        or entry.get("strong_aug") is not False
    ):
        raise FreshO64AuditError(
            "fresh O64 dataset entry must be ODVG/root=/, weight=1, strong_aug=false"
        )
    annotation_path = _saved_path(
        entry.get("anno"), label="fresh dataset annotation"
    )
    if annotation_path.name != OUTPUT_MANIFEST:
        raise FreshO64AuditError(
            "fresh O64 dataset does not point at the sealed manifest"
        )
    binding = value.get("artifact_binding")
    manifest_binding = binding.get("manifest") if isinstance(binding, Mapping) else None
    receipt_binding = binding.get("receipt") if isinstance(binding, Mapping) else None
    if (
        not isinstance(manifest_binding, Mapping)
        or set(manifest_binding) != {"path", "sha256", "rows"}
        or _saved_path(
            manifest_binding.get("path"), label="dataset-bound fresh manifest"
        )
        != annotation_path
        or manifest_binding.get("sha256") != EXPECTED_FRESH_MANIFEST_SHA256
        or manifest_binding.get("rows") != EXPECTED_ROWS
        or not isinstance(receipt_binding, Mapping)
        or set(receipt_binding) != {"path", "sha256", "schema"}
        or _saved_path(
            receipt_binding.get("path"), label="dataset-bound fresh receipt"
        )
        != annotation_path.parent / "receipt.json"
        or receipt_binding.get("sha256") != EXPECTED_FRESH_RECEIPT_SHA256
        or receipt_binding.get("schema") != OUTPUT_RECEIPT_SCHEMA
    ):
        raise FreshO64AuditError("fresh O64 dataset artifact binding drifted")
    return value


def validate_fresh_o64_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(rows) != EXPECTED_ROWS:
        raise FreshO64AuditError(
            f"fresh O64 manifest has {len(rows)} rows, expected {EXPECTED_ROWS}"
        )
    normalized: list[dict[str, Any]] = []
    pair_ids: dict[int, str] = {}
    pair_images: dict[int, int] = {}
    target_ids: set[int] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise FreshO64AuditError(f"fresh O64 row {index} is not an object")
        pair_index = row.get("pair_index")
        direction = row.get("direction")
        expected_pair = index // 2
        expected_direction = "anchor" if index % 2 == 0 else "partner"
        if (
            row.get("row_schema") != OUTPUT_ROW_SCHEMA
            or type(pair_index) is not int
            or pair_index != expected_pair
            or direction != expected_direction
        ):
            raise FreshO64AuditError(
                f"fresh O64 row {index} violates anchor/partner pair order"
            )

        pair_id = row.get("source_member_pair_id")
        image_id = row.get("image_id")
        target_id = row.get("target_coco_ann_id")
        manifest = row.get("source_assignment_manifest")
        line_number = row.get("source_assignment_line_number")
        filename = row.get("filename")
        source_row_sha = row.get("source_row_sha256")
        source_priority_sha = row.get("source_priority_sha256")
        if (
            not isinstance(pair_id, str)
            or len(pair_id) != 64
            or any(character not in "0123456789abcdef" for character in pair_id)
            or type(image_id) is not int
            or type(target_id) is not int
            or not isinstance(manifest, str)
            or not manifest
            or type(line_number) is not int
            or line_number <= 0
            or not isinstance(filename, str)
            or not filename
            or not isinstance(source_row_sha, str)
            or len(source_row_sha) != 64
            or not isinstance(source_priority_sha, str)
            or len(source_priority_sha) != 64
        ):
            raise FreshO64AuditError(f"fresh O64 row {index} identity is malformed")

        grounding = row.get("grounding")
        regions = grounding.get("regions") if isinstance(grounding, Mapping) else None
        region = regions[0] if isinstance(regions, list) and len(regions) == 1 else None
        expression = region.get("phrase") if isinstance(region, Mapping) else None
        bbox = region.get("bbox") if isinstance(region, Mapping) else None
        if (
            not isinstance(expression, str)
            or not expression.strip()
            or not isinstance(bbox, list)
            or len(bbox) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in bbox
            )
            or float(bbox[2]) <= float(bbox[0])
            or float(bbox[3]) <= float(bbox[1])
        ):
            raise FreshO64AuditError(
                f"fresh O64 row {index} must have one expression and xyxy box"
            )

        previous_pair = pair_ids.setdefault(pair_index, pair_id)
        previous_image = pair_images.setdefault(pair_index, image_id)
        if previous_pair != pair_id or previous_image != image_id:
            raise FreshO64AuditError(
                f"fresh O64 pair {pair_index} directions have different identities"
            )
        if target_id in target_ids:
            raise FreshO64AuditError(
                f"fresh O64 target annotation {target_id} is repeated"
            )
        target_ids.add(target_id)
        normalized.append(
            {
                "row_index": index,
                "pair_index": pair_index,
                "direction": direction,
                "image_id": image_id,
                "filename": filename,
                "expression": expression.strip(),
                "target_bbox_xyxy": [float(value) for value in bbox],
                "target_coco_ann_id": target_id,
                "source_assignment_manifest": manifest,
                "source_assignment_line_number": line_number,
                "source_member_pair_id": pair_id,
                "source_priority_sha256": source_priority_sha,
                "source_row_sha256": source_row_sha,
            }
        )
    if (
        set(pair_ids) != set(range(EXPECTED_PAIRS))
        or len(set(pair_images.values())) != EXPECTED_PAIRS
        or len(target_ids) != EXPECTED_ROWS
    ):
        raise FreshO64AuditError(
            "fresh O64 pair/image/target coverage is incomplete"
        )
    return normalized


def audit_training_checkpoint(
    payload: Mapping[str, Any],
    *,
    config_path: Path,
    dataset_path: Path,
    initializer_path: Path,
    checkpoint_path: Path,
    loader_batches: int,
) -> dict[str, Any]:
    if loader_batches <= 0:
        raise FreshO64AuditError("loader_batches must be positive")
    updates = _exact_int(payload.get("optimizer_updates"), label="optimizer_updates")
    if updates != EXPECTED_UPDATES or payload.get("checkpoint_reason") != "max_train_iters":
        raise FreshO64AuditError(
            "fresh O64 terminal checkpoint must record exactly 500 successful "
            "updates and reason=max_train_iters"
        )
    epoch = _exact_int(payload.get("epoch"), label="epoch")
    iteration = _exact_int(payload.get("iteration"), label="iteration")
    updates_per_epoch = math.ceil(
        loader_batches / EXPECTED_GRADIENT_ACCUMULATION_STEPS
    )
    expected_epoch = (EXPECTED_UPDATES - 1) // updates_per_epoch
    if epoch != expected_epoch or payload.get("epoch_finished") is not True:
        raise FreshO64AuditError(
            "fresh O64 terminal checkpoint does not match final epoch boundary"
        )
    allowed_iterations = {0, loader_batches - 1, loader_batches}
    if iteration not in allowed_iterations:
        raise FreshO64AuditError(
            "fresh O64 terminal iteration is inconsistent with loader length: "
            f"observed={iteration}, allowed={sorted(allowed_iterations)}"
        )
    for key in ("criterion", "optimizer", "lr_scheduler", "scaler", "rng_state"):
        if not isinstance(payload.get(key), Mapping):
            raise FreshO64AuditError(f"fresh O64 checkpoint is missing {key} state")

    args = _checkpoint_args(payload)
    if not args:
        raise FreshO64AuditError("fresh O64 checkpoint saved args are missing")
    expected_args = {
        **_expected_config_contract(),
        "max_train_iters": EXPECTED_UPDATES,
        "gradient_accumulation_steps": EXPECTED_GRADIENT_ACCUMULATION_STEPS,
        "amp": True,
    }
    drift = {
        key: {"observed": args.get(key), "expected": expected}
        for key, expected in expected_args.items()
        if args.get(key) != expected
    }
    if drift:
        raise FreshO64AuditError(f"fresh O64 saved args drifted: {drift}")
    forbidden_modes = (
        "stage_b_u0_patch_rank",
        "stage_b_data_driven_score",
        "stage_b_v7",
        "stage_b_v11_fixed_text",
        "stage_b_legacy_global_gate",
    )
    enabled_forbidden = [key for key in forbidden_modes if bool(args.get(key, False))]
    if enabled_forbidden:
        raise FreshO64AuditError(
            f"fresh O64 saved args enable forbidden paths: {enabled_forbidden}"
        )
    if _saved_path(args.get("config_file"), label="config_file") != config_path:
        raise FreshO64AuditError("fresh O64 saved config path drifted")
    if _saved_path(args.get("datasets"), label="datasets") != dataset_path:
        raise FreshO64AuditError("fresh O64 saved dataset path drifted")
    if (
        _saved_path(args.get("pretrain_model_path"), label="pretrain_model_path")
        != initializer_path
    ):
        raise FreshO64AuditError("fresh O64 saved initializer path drifted")
    if _saved_path(args.get("resume"), label="resume", allow_empty=True) is not None:
        raise FreshO64AuditError("fresh O64 must initialize with pretrain, not resume")
    if _saved_path(args.get("output_dir"), label="output_dir") != checkpoint_path.parent:
        raise FreshO64AuditError("fresh O64 saved output directory drifted")

    criterion = payload["criterion"]
    for key, expected in (
        ("criterion_train_mode_code", 1),
        ("criterion_scope_code", 0),
        ("criterion_queue_size", 0),
        ("criterion_queue_min_count", 0),
    ):
        if key not in criterion or _finite_scalar(
            criterion[key], label=f"criterion.{key}"
        ) != float(expected):
            raise FreshO64AuditError(
                f"fresh O64 criterion contract drifted at {key}"
            )
    optimizer = _audit_optimizer(payload)
    return {
        "optimizer_updates": updates,
        "checkpoint_reason": "max_train_iters",
        "epoch": epoch,
        "iteration": iteration,
        "epoch_finished": True,
        "loader_batches": loader_batches,
        "optimizer_updates_per_epoch": updates_per_epoch,
        "derived_terminal_epoch": expected_epoch,
        "train_micro_batch_size": EXPECTED_BATCH_SIZE,
        "train_gradient_accumulation_steps": (
            EXPECTED_GRADIENT_ACCUMULATION_STEPS
        ),
        "train_effective_batch_size": EXPECTED_EFFECTIVE_BATCH_SIZE,
        "saved_args_verified": sorted(expected_args),
        "lineage_uses_pretrain_not_resume": True,
        "optimizer": optimizer,
    }


def _validated_attribution_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(records) != EXPECTED_ROWS:
        raise FreshO64AuditError(
            f"fresh O64 result needs {EXPECTED_ROWS} attribution records"
        )
    normalized = [dict(record) for record in records]
    if any(record.get("row_index") != index for index, record in enumerate(normalized)):
        raise FreshO64AuditError("fresh O64 attribution record order drifted")
    return normalized


def _verify_published_artifact(annotation_path: Path) -> dict[str, Any]:
    manifest_record = stable_file_record(
        annotation_path, label="fresh O64 manifest"
    )
    receipt_path = annotation_path.parent / "receipt.json"
    receipt_record = stable_file_record(receipt_path, label="fresh O64 receipt")
    if manifest_record["sha256"] != EXPECTED_FRESH_MANIFEST_SHA256:
        raise FreshO64AuditError("fresh O64 manifest SHA-256 drifted")
    if receipt_record["sha256"] != EXPECTED_FRESH_RECEIPT_SHA256:
        raise FreshO64AuditError("fresh O64 receipt SHA-256 drifted")
    try:
        receipt = verify_fresh_o64_artifact(
            output_root=annotation_path.parent,
            output_manifest=annotation_path.name,
        )
    except Exception as error:
        raise FreshO64AuditError(
            f"fresh O64 artifact replay failed: {error}"
        ) from error
    if (
        receipt.get("schema") != OUTPUT_RECEIPT_SCHEMA
        or receipt.get("row_schema") != OUTPUT_ROW_SCHEMA
        or receipt.get("pairs") != EXPECTED_PAIRS
        or receipt.get("rows") != EXPECTED_ROWS
        or receipt.get("unique_images") != EXPECTED_PAIRS
        or receipt.get("unique_target_annotation_ids") != EXPECTED_ROWS
        or not isinstance(receipt.get("invariants"), Mapping)
        or any(value is not True for value in receipt["invariants"].values())
    ):
        raise FreshO64AuditError("fresh O64 artifact receipt contract drifted")
    output = receipt.get("output")
    if not isinstance(output, Mapping) or output.get("sha256") != manifest_record["sha256"]:
        raise FreshO64AuditError("fresh O64 receipt output hash drifted")
    return {
        "receipt": receipt,
        "manifest_record": manifest_record,
        "receipt_record": receipt_record,
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if "GFLOPS_DEBUG_SHILONG" in os.environ:
        raise FreshO64AuditError("GFLOPS_DEBUG_SHILONG is forbidden")
    config_path = args.config.expanduser().resolve(strict=True)
    dataset_path = args.datasets.expanduser().resolve(strict=True)
    b58_path = args.b58.expanduser().resolve(strict=True)
    initializer_path = args.initializer.expanduser().resolve(strict=True)
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    output_path = args.output_json.expanduser().resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise FreshO64AuditError("CUDA was requested but is unavailable")
    if stable_file_record(b58_path, label="b58 checkpoint")["sha256"] != EXPECTED_B58_SHA256:
        raise FreshO64AuditError("b58 checkpoint SHA-256 drifted")

    cfg = SLConfig.fromfile(str(config_path))
    config_contract = validate_config(cfg)
    dataset_config = _load_dataset_config(dataset_path)
    dataset_entry = dataset_config["train"][0]
    annotation_path = _saved_path(
        dataset_entry["anno"], label="fresh dataset annotation"
    )
    artifact = _verify_published_artifact(annotation_path)

    seed = int(getattr(cfg, "seed", 42))
    _seed_everything(seed)
    cfg.device = str(device)
    cfg.distributed = False
    model, _criterion, _postprocessors = build_model_main(cfg)
    initializer_payload = _safe_load_checkpoint(
        initializer_path, label="native-residual initializer"
    )
    validate_fresh_initializer(
        model,
        initializer_payload,
        initializer_path=initializer_path,
        config_path=config_path,
        b58_path=b58_path,
    )
    checkpoint_payload = _safe_load_checkpoint(
        checkpoint_path, label="fresh O64 checkpoint"
    )
    b58_payload = _safe_load_checkpoint(b58_path, label="b58 checkpoint")
    b58_lineage = audit_b58_lineage(b58_payload, initializer_payload)
    isolation = audit_tensor_isolation(
        initializer_payload, checkpoint_payload, identity=bool(args.identity)
    )
    checkpoint_state = _model_state(
        checkpoint_payload, label="fresh O64 checkpoint"
    )
    validate_stage_b_gdino_score_adapter_checkpoint(
        model,
        checkpoint_state,
        checkpoint_label=f"fresh O64 checkpoint {checkpoint_path}",
    )
    model.load_state_dict(checkpoint_state, strict=True)

    # Evaluation always uses the deployment-like validation transform. The
    # audited train config now has the same scalar 800/max1333 resize geometry.
    dataset = build_dataset(image_set="val", args=cfg, datasetinfo=dataset_entry)
    metas = getattr(dataset, "metas", None)
    if not isinstance(metas, list):
        raise FreshO64AuditError(
            "fresh O64 ODVG dataset does not expose ordered metadata"
        )
    metadata = validate_fresh_o64_rows(metas)
    if len(dataset) != EXPECTED_ROWS or getattr(dataset, "sample_weights", None) is not None:
        raise FreshO64AuditError("fresh O64 evaluation dataset geometry drifted")
    loader = DataLoader(
        dataset,
        batch_size=EXPECTED_BATCH_SIZE,
        sampler=SequentialSampler(dataset),
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=0,
        pin_memory=False,
    )
    expected_loader_batches = math.ceil(EXPECTED_ROWS / EXPECTED_BATCH_SIZE)
    if len(loader) != expected_loader_batches:
        raise FreshO64AuditError("fresh O64 sequential loader length drifted")
    training_audit = None
    if not args.identity:
        training_audit = audit_training_checkpoint(
            checkpoint_payload,
            config_path=config_path,
            dataset_path=dataset_path,
            initializer_path=initializer_path,
            checkpoint_path=checkpoint_path,
            loader_batches=len(loader),
        )

    model.to(device).eval()
    amp = not bool(args.no_amp)
    records: list[dict[str, Any]] = []
    identity_checks = {
        "rank_residual_exact_zero": True,
        "rank_score_bitwise_equals_base": True,
        "winner_query_equals_base": True,
    }
    cursor = 0
    with torch.inference_mode():
        for samples, raw_targets in loader:
            raw_targets = list(raw_targets)
            batch_metadata = metadata[cursor : cursor + len(raw_targets)]
            cursor += len(raw_targets)
            captions = _build_stage_b_gdino_adapter_rank_captions(raw_targets)
            samples = samples.to(device)
            targets = [
                {
                    key: value.to(device)
                    for key, value in target.items()
                    if torch.is_tensor(value)
                }
                for target in raw_targets
            ]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=bool(amp and device.type == "cuda"),
            ):
                outputs = model(samples, captions=captions)
            batch_records, batch_identity = audit_batch_outputs(
                outputs,
                targets,
                batch_metadata,
                identity=bool(args.identity),
            )
            records.extend(batch_records)
            identity_checks = {
                key: identity_checks[key] and batch_identity[key]
                for key in identity_checks
            }
    if cursor != EXPECTED_ROWS:
        raise FreshO64AuditError(
            "fresh O64 forward did not consume exactly 128 rows"
        )
    records = _validated_attribution_records(records)
    metrics = aggregate_o64_records(records)
    receipt = artifact["receipt"]
    result = {
        "schema": SCHEMA,
        "mode": "identity" if args.identity else "u500",
        "inputs": {
            "config": stable_file_record(config_path, label="fresh O64 config"),
            "datasets": stable_file_record(dataset_path, label="fresh O64 datasets"),
            "b58": stable_file_record(b58_path, label="b58 checkpoint"),
            "initializer": stable_file_record(
                initializer_path, label="native-residual initializer"
            ),
            "checkpoint": stable_file_record(
                checkpoint_path, label="fresh O64 checkpoint"
            ),
        },
        "runtime": {
            "device": str(device),
            "amp": bool(amp and device.type == "cuda"),
            "dataset_transform": "val_scalar_800_max1333_aspect_preserving",
            "train_transform_contract": {
                "fix_size": False,
                "deterministic_aspect_resize": True,
                "strong_aug": False,
                "hflip_prob": 0.0,
                "scales": [800],
                "max_size": 1333,
            },
            "sampler": "sequential",
            "eval_batch_size": EXPECTED_BATCH_SIZE,
            "batches": len(loader),
            "rows": EXPECTED_ROWS,
            "train_micro_batch_size": EXPECTED_BATCH_SIZE,
            "train_gradient_accumulation_steps": (
                EXPECTED_GRADIENT_ACCUMULATION_STEPS
            ),
            "train_effective_batch_size": EXPECTED_EFFECTIVE_BATCH_SIZE,
        },
        "config_contract": config_contract,
        "dataset_artifact": {
            "schema": receipt.get("schema"),
            "row_schema": receipt.get("row_schema"),
            "canonical_payload_sha256": receipt.get("canonical_payload_sha256"),
            "manifest": artifact["manifest_record"],
            "receipt": artifact["receipt_record"],
            "rows": receipt.get("rows"),
            "pairs": receipt.get("pairs"),
            "unique_images": receipt.get("unique_images"),
            "fresh_exclusion_invariants": {
                key: value
                for key, value in receipt.get("invariants", {}).items()
                if "disjoint" in key or "model" in key or "score" in key
            },
        },
        "b58_lineage": b58_lineage,
        "tensor_isolation": isolation,
        "training_audit": training_audit,
        "identity_checks": identity_checks if args.identity else None,
        "metrics": metrics,
        "records": records,
        "passed": True,
    }
    _atomic_write_json(output_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--datasets", required=True, type=Path)
    parser.add_argument("--b58", required=True, type=Path)
    parser.add_argument("--initializer", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--identity",
        action="store_true",
        help="require checkpoint tensors and every forward rank output to be b58 identity",
    )
    return parser


def main() -> None:
    result = run_evaluation(_parser().parse_args())
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
