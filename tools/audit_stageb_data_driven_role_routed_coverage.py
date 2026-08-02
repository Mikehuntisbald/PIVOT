#!/usr/bin/env python3
"""Audit real-query supervision coverage for the sealed clean role dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataset  # noqa: E402
from engine import (  # noqa: E402
    _build_stage_b_data_driven_assignment_captions,
    _set_stage_b_data_driven_training_mode,
)
from main import (  # noqa: E402
    _freeze_and_audit_stage_b_data_driven,
    _validate_stage_b_data_driven_role_routed_training_contract,
    build_model_main,
)
from models.GroundingDINO.stage_b_data_driven_score import (  # noqa: E402
    _candidate_official_assignment_role_iou,
    data_driven_category_gate_mask,
    role_routed_official_assignment_top1_loss,
    validate_data_driven_role_routed_initializer_payload,
)
from tools.audit_stageb_data_driven_role_routed_real_model import (  # noqa: E402
    CONFIG,
    DATASET_CONFIG,
    EXPECTED_A0_SHA256,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_DATASET_CONFIG_SHA256,
    EXPECTED_INITIALIZER_SHA256,
    EXPECTED_QUERY_COUNT,
    EXPECTED_SOURCE_SHA256,
    EXPECTED_SOURCE_UPDATES,
    INITIALIZER,
    RealModelAuditError,
    _load_initializer,
    _move_criterion_target,
    _require_file,
    _seed_everything,
    _sha256,
    _write_json_exclusive,
)
from util.misc import nested_tensor_from_tensor_list  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


MIN_DIRECTION_COVERAGE = 0.90
MIN_FULL_PAIR_COVERAGE = 0.80
MIN_PATCH_INSTANCE_COVERAGE = 0.90


def _as_count(value: torch.Tensor, *, label: str) -> int:
    observed = float(value.detach().float().item())
    rounded = int(round(observed))
    if observed != float(rounded):
        raise RealModelAuditError(f"{label} is not an exact integer count: {observed}")
    return rounded


def _add_count(accumulator: dict[str, int], losses: Mapping[str, Any], key: str) -> None:
    value = losses.get(key)
    if not torch.is_tensor(value) or value.numel() != 1:
        raise RealModelAuditError(f"criterion did not emit scalar {key}")
    accumulator[key] = accumulator.get(key, 0) + _as_count(value, label=key)


def _build_runtime(
    cfg: Any,
    device: torch.device,
    *,
    binding: Mapping[str, Any] | None = None,
):
    selected = binding or {
        "config": CONFIG,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "initializer": INITIALIZER,
        "initializer_sha256": EXPECTED_INITIALIZER_SHA256,
    }
    config_path = _require_file(
        Path(selected["config"]),
        str(selected["config_sha256"]),
        label="training config",
    )
    dataset_path = _require_file(
        DATASET_CONFIG,
        EXPECTED_DATASET_CONFIG_SHA256,
        label="dataset config",
    )
    initializer_path = _require_file(
        Path(selected["initializer"]),
        str(selected["initializer_sha256"]),
        label="model-only initializer",
    )
    cfg.device = "cpu"
    cfg.amp = True
    cfg.seed = 42
    cfg.num_workers = 8
    cfg.prefetch_factor = 1
    cfg.gradient_accumulation_steps = 1
    cfg.eval = False
    cfg.resume = ""
    cfg.datasets = str(dataset_path)
    cfg.pretrain_model_path = str(initializer_path)
    cfg.max_train_iters = 1000
    cfg.iter_checkpoint_interval = 1000
    _validate_stage_b_data_driven_role_routed_training_contract(
        cfg,
        base_path=initializer_path,
        dataset_path=dataset_path,
    )
    model, criterion, _postprocessors = build_model_main(cfg)
    initializer = _load_initializer(initializer_path)
    validate_data_driven_role_routed_initializer_payload(
        model,
        initializer,
        checkpoint_label=str(initializer_path),
        expected_source_checkpoint_sha256=EXPECTED_SOURCE_SHA256,
        expected_a0_initializer_sha256=EXPECTED_A0_SHA256,
        expected_source_optimizer_updates=EXPECTED_SOURCE_UPDATES,
    )
    model.load_state_dict(initializer["model"], strict=True)
    _freeze_and_audit_stage_b_data_driven(model, "rank_patch_only")
    model = model.to(device)
    criterion = criterion.to(device)
    model.train()
    criterion.train()
    _set_stage_b_data_driven_training_mode(model, "rank_patch_only")
    if criterion.weight_dict != {
        "loss_stage_b_data_driven_role_routed_rank": 1.0,
        "loss_stage_b_data_driven_patch": 1.0,
    }:
        raise RealModelAuditError("coverage audit criterion surface drifted")
    return model, criterion, config_path, dataset_path, initializer_path


def _audit_source(
    *,
    cfg: Any,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    source: Mapping[str, Any],
    source_index: int,
    samples_per_source: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    dataset = build_dataset("train", cfg, dict(source))
    if samples_per_source > len(dataset):
        raise RealModelAuditError(
            f"requested {samples_per_source} rows from source with {len(dataset)} rows"
        )
    source_seed = int(seed + 1009 * source_index)
    indices = random.Random(source_seed).sample(range(len(dataset)), samples_per_source)
    counts: dict[str, int] = {}
    first_runtime_index = None
    image_ids: list[int] = []
    failed_rows: list[dict[str, Any]] = []
    count_keys = (
        "stage_b_data_driven_assignment_data_rows",
        "stage_b_data_driven_assignment_runtime_rows",
        "stage_b_data_driven_assignment_runtime_directions",
        "stage_b_data_driven_assignment_unreachable_rows",
        "stage_b_data_driven_assignment_role0_queries",
        "stage_b_data_driven_assignment_role1_queries",
        "stage_b_data_driven_assignment_safe_sibling_queries",
        "stage_b_data_driven_patch_valid_instances",
        "stage_b_data_driven_patch_skipped_instances",
        "stage_b_data_driven_patch_category_positive_queries",
        "stage_b_data_driven_patch_category_negative_queries",
        "stage_b_data_driven_patch_category_neutral_queries",
        "stage_b_data_driven_patch_deployed_gate_queries",
        "audit_geometry_reachable_rows",
        "audit_gate_removed_geometry_rows",
    )

    _seed_everything(source_seed)
    for offset in range(0, samples_per_source, batch_size):
        batch_indices = indices[offset : offset + batch_size]
        loaded = [dataset[index] for index in batch_indices]
        images = [item[0] for item in loaded]
        raw_targets = [item[1] for item in loaded]
        image_ids.extend(
            int(target["image_id"].reshape(-1)[0].item()) for target in raw_targets
        )
        canonical, expressions = _build_stage_b_data_driven_assignment_captions(
            raw_targets
        )
        samples = nested_tensor_from_tensor_list(images).to(device)
        patches = torch.stack(
            [target["patch"] for target in raw_targets], dim=0
        ).to(device)
        targets = [_move_criterion_target(target, device) for target in raw_targets]

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=bool(cfg.amp)):
            outputs = model(
                samples,
                captions=canonical,
                patches=patches,
                stage_b_data_driven_expression_captions=expressions,
            )
            losses = criterion(outputs, targets)
            rank = outputs["stage_b_data_driven_text_rank_score"]
            candidate = outputs["stage_b_data_driven_candidate_mask"]
            if tuple(rank.shape) != (len(batch_indices), EXPECTED_QUERY_COUNT, 2):
                raise RealModelAuditError("coverage rank-score shape drifted")
            if tuple(candidate.shape) != tuple(rank.shape) or not bool(candidate.all()):
                raise RealModelAuditError("coverage candidate mask is not all 900 queries")
            assignment_iou, other_iou, pair_valid = (
                _candidate_official_assignment_role_iou(
                    outputs["pred_boxes"], targets
                )
            )
            role = role_routed_official_assignment_top1_loss(
                rank,
                assignment_iou,
                other_iou,
                pair_valid,
                candidate,
                outputs["pred_logits_patch"],
                positive_iou_threshold=cfg.stage_b_data_driven_positive_iou_threshold,
                negative_iou_threshold=(
                    cfg.stage_b_data_driven_rank_negative_iou_threshold
                ),
                category_gate_max_gap=cfg.stage_b_data_driven_category_gate_max_gap,
                patch_score_clip=cfg.stage_b_data_driven_patch_score_clip,
                margin=cfg.stage_b_data_driven_rank_margin,
                temperature=cfg.stage_b_data_driven_temperature,
            )
            patch_score = outputs["pred_logits_patch"]
            if patch_score.dim() == 3:
                patch_score = patch_score[..., 0]
            gate, _standardized_patch = data_driven_category_gate_mask(
                patch_score.detach(),
                candidate[..., 0],
                max_gap=cfg.stage_b_data_driven_category_gate_max_gap,
                clip=cfg.stage_b_data_driven_patch_score_clip,
            )
            separated_from_unassigned = (
                other_iou
                < cfg.stage_b_data_driven_rank_negative_iou_threshold
            )
            geometry_role0 = (
                candidate[..., 0]
                & (
                    assignment_iou[..., 0]
                    >= cfg.stage_b_data_driven_positive_iou_threshold
                )
                & (
                    assignment_iou[..., 1]
                    < cfg.stage_b_data_driven_rank_negative_iou_threshold
                )
                & separated_from_unassigned
            )
            geometry_role1 = (
                candidate[..., 0]
                & (
                    assignment_iou[..., 1]
                    >= cfg.stage_b_data_driven_positive_iou_threshold
                )
                & (
                    assignment_iou[..., 0]
                    < cfg.stage_b_data_driven_rank_negative_iou_threshold
                )
                & separated_from_unassigned
            )
            geometry_reachable = (
                pair_valid
                & geometry_role0.any(dim=1)
                & geometry_role1.any(dim=1)
            )
            gate_removed_geometry = (
                geometry_reachable & ~role["runtime_valid"]
            )

        direct_counts = {
            "stage_b_data_driven_assignment_data_rows": int(
                role["data_valid"].sum().item()
            ),
            "stage_b_data_driven_assignment_runtime_rows": int(
                role["runtime_valid"].sum().item()
            ),
            "stage_b_data_driven_assignment_runtime_directions": int(
                role["runtime_valid_direction"].sum().item()
            ),
            "stage_b_data_driven_assignment_unreachable_rows": int(
                (role["data_valid"] & ~role["runtime_valid"]).sum().item()
            ),
            "stage_b_data_driven_assignment_role0_queries": int(
                role["owned_mask"][..., 0].sum().item()
            ),
            "stage_b_data_driven_assignment_role1_queries": int(
                role["owned_mask"][..., 1].sum().item()
            ),
            "stage_b_data_driven_assignment_safe_sibling_queries": int(
                role["safe_sibling_mask"].sum().item()
            ),
        }
        for key, direct in direct_counts.items():
            observed = _as_count(losses[key], label=key)
            if observed != direct:
                raise RealModelAuditError(
                    f"criterion/role truth-table count mismatch for {key}: "
                    f"criterion={observed}, direct={direct}"
                )
        runtime_any = role["runtime_valid_direction"].any(dim=1)
        if first_runtime_index is None and bool(runtime_any.any().item()):
            local = int(runtime_any.to(torch.int64).argmax().item())
            first_runtime_index = int(batch_indices[local])
        for key in count_keys:
            if key == "audit_geometry_reachable_rows":
                counts[key] = counts.get(key, 0) + int(
                    geometry_reachable.sum().item()
                )
            elif key == "audit_gate_removed_geometry_rows":
                counts[key] = counts.get(key, 0) + int(
                    gate_removed_geometry.sum().item()
                )
            else:
                _add_count(counts, losses, key)
        unreachable = role["data_valid"] & ~role["runtime_valid"]
        for local in unreachable.nonzero(as_tuple=False).reshape(-1).tolist():
            roles = raw_targets[local][
                "stage_b_data_driven_assignment_role"
            ].reshape(-1)
            failed_rows.append(
                {
                    "dataset_index": int(batch_indices[local]),
                    "image_id": int(
                        raw_targets[local]["image_id"].reshape(-1)[0].item()
                    ),
                    "same_category_instances": int(roles.numel()),
                    "unassigned_same_category_instances": int(
                        (roles == -1).sum().item()
                    ),
                    "geometry_role0_queries": int(
                        geometry_role0[local].sum().item()
                    ),
                    "geometry_role1_queries": int(
                        geometry_role1[local].sum().item()
                    ),
                    "gated_role0_queries": int(
                        role["owned_mask"][local, :, 0].sum().item()
                    ),
                    "gated_role1_queries": int(
                        role["owned_mask"][local, :, 1].sum().item()
                    ),
                    "failure_stage": (
                        "gate_removed"
                        if bool(geometry_reachable[local].item())
                        else "geometry_unreachable"
                    ),
                }
            )

        del outputs, losses, role, samples, patches, targets

    data_rows = counts["stage_b_data_driven_assignment_data_rows"]
    runtime_rows = counts["stage_b_data_driven_assignment_runtime_rows"]
    runtime_directions = counts[
        "stage_b_data_driven_assignment_runtime_directions"
    ]
    patch_valid = counts["stage_b_data_driven_patch_valid_instances"]
    patch_skipped = counts["stage_b_data_driven_patch_skipped_instances"]
    if data_rows <= 0 or patch_valid + patch_skipped <= 0:
        raise RealModelAuditError("coverage sample contained no valid supervision")
    direction_coverage = runtime_directions / (2.0 * data_rows)
    full_pair_coverage = runtime_rows / data_rows
    patch_instance_coverage = patch_valid / (patch_valid + patch_skipped)
    geometry_reachable = counts["audit_geometry_reachable_rows"]
    return {
        "source_index": source_index,
        "manifest": str(source["anno"]),
        "dataset_rows": len(dataset),
        "sample_seed": source_seed,
        "sample_count": samples_per_source,
        "sample_indices_sha256": hashlib.sha256(
            json.dumps(indices, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "sample_image_ids_sha256": hashlib.sha256(
            json.dumps(image_ids, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "first_runtime_direction_index": first_runtime_index,
        "failed_runtime_rows": failed_rows,
        "counts": counts,
        "rates": {
            "data_valid_row_fraction": data_rows / samples_per_source,
            "runtime_direction_coverage_of_valid": direction_coverage,
            "runtime_full_pair_coverage_of_valid": full_pair_coverage,
            "geometry_full_pair_coverage_of_valid": (
                geometry_reachable / data_rows
            ),
            "gate_retention_of_geometry_reachable": (
                runtime_rows / geometry_reachable
                if geometry_reachable > 0
                else 0.0
            ),
            "patch_instance_coverage": patch_instance_coverage,
        },
    }


def run_audit(
    *,
    device_name: str,
    samples_per_source: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    if samples_per_source <= 0 or batch_size <= 0:
        raise RealModelAuditError("coverage sample and batch sizes must be positive")
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RealModelAuditError("coverage audit requires an available CUDA device")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    _seed_everything(seed)
    cfg = SLConfig.fromfile(str(CONFIG))
    model, criterion, config_path, dataset_path, initializer_path = _build_runtime(
        cfg, device
    )
    dataset_payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    sources = dataset_payload["train"]
    if len(sources) != 3:
        raise RealModelAuditError("coverage audit requires all three sealed sources")
    source_results = [
        _audit_source(
            cfg=cfg,
            model=model,
            criterion=criterion,
            source=source,
            source_index=source_index,
            samples_per_source=samples_per_source,
            batch_size=batch_size,
            seed=seed,
            device=device,
        )
        for source_index, source in enumerate(sources)
    ]
    failed = []
    for source in source_results:
        rates = source["rates"]
        if rates["runtime_direction_coverage_of_valid"] < MIN_DIRECTION_COVERAGE:
            failed.append(
                {
                    "source_index": source["source_index"],
                    "metric": "runtime_direction_coverage_of_valid",
                    "observed": rates[
                        "runtime_direction_coverage_of_valid"
                    ],
                    "required_minimum": MIN_DIRECTION_COVERAGE,
                }
            )
        if rates["runtime_full_pair_coverage_of_valid"] < MIN_FULL_PAIR_COVERAGE:
            failed.append(
                {
                    "source_index": source["source_index"],
                    "metric": "runtime_full_pair_coverage_of_valid",
                    "observed": rates[
                        "runtime_full_pair_coverage_of_valid"
                    ],
                    "required_minimum": MIN_FULL_PAIR_COVERAGE,
                }
            )
        if rates["patch_instance_coverage"] < MIN_PATCH_INSTANCE_COVERAGE:
            failed.append(
                {
                    "source_index": source["source_index"],
                    "metric": "patch_instance_coverage",
                    "observed": rates["patch_instance_coverage"],
                    "required_minimum": MIN_PATCH_INSTANCE_COVERAGE,
                }
            )
        counts = source["counts"]
        for label in (
            "stage_b_data_driven_assignment_role0_queries",
            "stage_b_data_driven_assignment_role1_queries",
            "stage_b_data_driven_patch_category_positive_queries",
            "stage_b_data_driven_patch_category_negative_queries",
        ):
            if counts[label] <= 0:
                failed.append(
                    {
                        "source_index": source["source_index"],
                        "metric": label,
                        "observed": counts[label],
                        "required_minimum": 1,
                    }
                )
    torch.cuda.synchronize(device)
    return {
        "schema": "pivot.stageb.data_driven.role_routed_coverage_audit/v1",
        "status": "failed" if failed else "passed",
        "threshold_failures": failed,
        "device": str(device),
        "amp_matches_training": bool(cfg.amp),
        "thresholds": {
            "minimum_runtime_direction_coverage_of_valid": MIN_DIRECTION_COVERAGE,
            "minimum_runtime_full_pair_coverage_of_valid": MIN_FULL_PAIR_COVERAGE,
            "minimum_patch_instance_coverage": MIN_PATCH_INSTANCE_COVERAGE,
        },
        "sampling": {
            "seed": seed,
            "samples_per_source": samples_per_source,
            "batch_size": batch_size,
            "total_samples": samples_per_source * len(source_results),
            "without_replacement_within_each_source": True,
        },
        "bindings": {
            "audit_script_sha256": _sha256(Path(__file__).resolve()),
            "real_model_audit_script_sha256": _sha256(
                REPO_ROOT / "tools/audit_stageb_data_driven_role_routed_real_model.py"
            ),
            "score_implementation_sha256": _sha256(
                REPO_ROOT / "models/GroundingDINO/stage_b_data_driven_score.py"
            ),
            "config_path": str(config_path),
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "dataset_config_path": str(dataset_path),
            "dataset_config_sha256": EXPECTED_DATASET_CONFIG_SHA256,
            "initializer_path": str(initializer_path),
            "initializer_sha256": EXPECTED_INITIALIZER_SHA256,
        },
        "sources": source_results,
        "invariants": {
            "criterion_counts_match_direct_role_truth_table": True,
            "all_900_queries_candidate_for_both_expression_slots": True,
            "frozen_feature_generators_use_engine_eval_mode": True,
            "no_optimizer_constructed": True,
            "no_parameter_update": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples-per-source", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(
        device_name=args.device,
        samples_per_source=args.samples_per_source,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    _write_json_exclusive(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
