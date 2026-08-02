#!/usr/bin/env python3
"""Test exact Gate3 feasibility with GT-selected free per-query patch scores."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from engine import _build_stage_b_data_driven_assignment_captions  # noqa: E402
from models.GroundingDINO.stage_b_data_driven_score import (  # noqa: E402
    DATA_DRIVEN_PATCH_DROP_ANCHOR_INSTANCE_BALANCED,
    deployment_gate_category_patch_loss,
)
from tools.audit_stageb_data_driven_role_routed_coverage import (  # noqa: E402
    _build_runtime,
)
from tools.audit_stageb_data_driven_role_routed_real_model import (  # noqa: E402
    AUDIT_VARIANTS,
    EXPECTED_DATASET_CONFIG_SHA256,
    EXPECTED_QUERY_COUNT,
    RealModelAuditError,
    _move_criterion_target,
    _seed_everything,
    _sha256,
    _write_json_exclusive,
)
from tools.run_stageb_data_driven_role_routed_overfit64 import (  # noqa: E402
    EXPECTED_ROLE_EXCLUSIVE_REACHABLE_DIRECTIONS,
    MAX_FINAL_GATED_CATEGORY_NEGATIVE_FRACTION,
    MAX_FINAL_PATCH_ALIGNED_AUXILIARY_RATIO,
    MAX_FINAL_PATCH_COMPONENT_RATIO,
    MAX_FINAL_PATCH_LOSS_RATIO,
    MIN_FINAL_PATCH_COVERAGE_FRACTION,
    _canonical_sha256,
    _select_rows,
    _tensor_sha256,
)
from util import box_ops  # noqa: E402
from util.misc import nested_tensor_from_tensor_list  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


RAW_AMPLITUDES = (
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
    64.0,
    128.0,
    256.0,
    512.0,
    1024.0,
    2048.0,
)


def _scalar(value: torch.Tensor) -> float:
    if not torch.is_tensor(value) or value.numel() != 1:
        raise RealModelAuditError("patch oracle expected a scalar metric")
    return float(value.detach().float().item())


def _patch_contract(
    score: torch.Tensor,
    boxes: torch.Tensor,
    targets: Sequence[Mapping[str, Any]],
    candidate: torch.Tensor,
    cfg: Any,
) -> dict[str, torch.Tensor]:
    return deployment_gate_category_patch_loss(
        score,
        boxes,
        targets,
        candidate,
        positive_iou_threshold=cfg.stage_b_data_driven_positive_iou_threshold,
        negative_iou_threshold=cfg.stage_b_data_driven_patch_negative_iou_threshold,
        category_gate_max_gap=cfg.stage_b_data_driven_category_gate_max_gap,
        patch_score_clip=cfg.stage_b_data_driven_patch_score_clip,
        boundary_margin=cfg.stage_b_data_driven_category_gate_boundary_margin,
        temperature=cfg.stage_b_data_driven_temperature,
        active_unsafe_auxiliary_weight=(
            cfg.stage_b_data_driven_patch_active_unsafe_auxiliary_weight
        ),
        drop_dense_tail_weight=float(
            getattr(
                cfg,
                "stage_b_data_driven_patch_drop_dense_tail_weight",
                0.0,
            )
        ),
        dense_category_focal_weight=(
            cfg.stage_b_data_driven_patch_dense_category_focal_weight
        ),
        dense_category_focal_alpha=(
            cfg.stage_b_data_driven_patch_dense_category_focal_alpha
        ),
        dense_category_focal_gamma=(
            cfg.stage_b_data_driven_patch_dense_category_focal_gamma
        ),
        dense_category_focal_negative_weight=(
            cfg.stage_b_data_driven_patch_dense_category_focal_negative_weight
        ),
        role_exclusive_keep=True,
        drop_positive_anchor_gradient_policy=(
            DATA_DRIVEN_PATCH_DROP_ANCHOR_INSTANCE_BALANCED
        ),
    )


def _metrics(contract: Mapping[str, torch.Tensor]) -> dict[str, float]:
    scalar_keys = (
        "loss",
        "keep_component",
        "keep_objective_component",
        "keep_mean_component",
        "role_exclusive_keep_component",
        "drop_component",
        "drop_objective_component",
        "drop_active_unsafe_component",
        "drop_dense_tail_component",
        "valid_instances",
        "keep_safe_instances",
        "keep_deployed_instances",
        "role_exclusive_reachable_instances",
        "role_exclusive_unreachable_instances",
        "role_exclusive_keep_safe_instances",
        "role_exclusive_keep_deployed_instances",
        "valid_drop_rows",
        "drop_safe_rows",
        "drop_deployed_rows",
    )
    result = {key: _scalar(contract[key]) for key in scalar_keys}
    negative = contract["category_negative_mask"]
    deployed = contract["deployed_gate"]
    result["category_negative_queries"] = float(negative.sum().item())
    result["deployed_category_negative_queries"] = float(
        (negative & deployed).sum().item()
    )
    result["deployed_gate_queries"] = float(deployed.sum().item())
    return result


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0 if numerator <= 0.0 else float("inf")
    return numerator / denominator


def _checks(
    metrics: Mapping[str, float], baseline: Mapping[str, float]
) -> tuple[dict[str, bool], dict[str, float]]:
    valid_instances = metrics["valid_instances"]
    valid_drop_rows = metrics["valid_drop_rows"]
    role_reachable = metrics["role_exclusive_reachable_instances"]
    derived = {
        "patch_loss_ratio": _ratio(metrics["loss"], baseline["loss"]),
        "patch_keep_component_ratio": _ratio(
            metrics["keep_component"], baseline["keep_component"]
        ),
        "patch_drop_component_ratio": _ratio(
            metrics["drop_component"], baseline["drop_component"]
        ),
        "patch_drop_active_unsafe_component_ratio": _ratio(
            metrics["drop_active_unsafe_component"],
            baseline["drop_active_unsafe_component"],
        ),
        "patch_role_exclusive_keep_component_ratio": _ratio(
            metrics["role_exclusive_keep_component"],
            baseline["role_exclusive_keep_component"],
        ),
        "patch_keep_safe_fraction": _ratio(
            metrics["keep_safe_instances"], valid_instances
        ),
        "patch_keep_deployed_fraction": _ratio(
            metrics["keep_deployed_instances"], valid_instances
        ),
        "patch_drop_safe_fraction": _ratio(
            metrics["drop_safe_rows"], valid_drop_rows
        ),
        "patch_drop_deployed_fraction": _ratio(
            metrics["drop_deployed_rows"], valid_drop_rows
        ),
        "patch_role_exclusive_safe_fraction": _ratio(
            metrics["role_exclusive_keep_safe_instances"], role_reachable
        ),
        "patch_role_exclusive_deployed_fraction": _ratio(
            metrics["role_exclusive_keep_deployed_instances"], role_reachable
        ),
        "patch_gated_category_negative_fraction": _ratio(
            metrics["deployed_category_negative_queries"],
            metrics["category_negative_queries"],
        ),
    }
    checks = {
        "patch_loss_materially_decreases": (
            derived["patch_loss_ratio"] <= MAX_FINAL_PATCH_LOSS_RATIO
        ),
        "patch_keep_component_materially_decreases": (
            derived["patch_keep_component_ratio"]
            <= MAX_FINAL_PATCH_COMPONENT_RATIO
        ),
        "patch_drop_component_materially_decreases": (
            derived["patch_drop_component_ratio"]
            <= MAX_FINAL_PATCH_COMPONENT_RATIO
        ),
        "patch_allnegative_active_severity_materially_decreases": (
            derived["patch_drop_active_unsafe_component_ratio"]
            <= MAX_FINAL_PATCH_ALIGNED_AUXILIARY_RATIO
        ),
        "patch_role_exclusive_component_materially_decreases": (
            derived["patch_role_exclusive_keep_component_ratio"]
            <= MAX_FINAL_PATCH_COMPONENT_RATIO
        ),
        "patch_keep_safe_fraction_reaches_threshold": (
            derived["patch_keep_safe_fraction"]
            >= MIN_FINAL_PATCH_COVERAGE_FRACTION
        ),
        "patch_keep_deployed_fraction_reaches_threshold": (
            derived["patch_keep_deployed_fraction"]
            >= MIN_FINAL_PATCH_COVERAGE_FRACTION
        ),
        "patch_drop_safe_fraction_reaches_threshold": (
            derived["patch_drop_safe_fraction"]
            >= MIN_FINAL_PATCH_COVERAGE_FRACTION
        ),
        "patch_drop_deployed_fraction_reaches_threshold": (
            derived["patch_drop_deployed_fraction"]
            >= MIN_FINAL_PATCH_COVERAGE_FRACTION
        ),
        "patch_role_exclusive_reachability_is_complete": (
            role_reachable
            == float(EXPECTED_ROLE_EXCLUSIVE_REACHABLE_DIRECTIONS)
            and metrics["role_exclusive_unreachable_instances"] == 0.0
        ),
        "patch_role_exclusive_safe_fraction_reaches_threshold": (
            derived["patch_role_exclusive_safe_fraction"]
            >= MIN_FINAL_PATCH_COVERAGE_FRACTION
        ),
        "patch_role_exclusive_deployed_fraction_reaches_threshold": (
            derived["patch_role_exclusive_deployed_fraction"]
            >= MIN_FINAL_PATCH_COVERAGE_FRACTION
        ),
        "patch_gated_category_negative_fraction_is_bounded": (
            derived["patch_gated_category_negative_fraction"]
            <= MAX_FINAL_GATED_CATEGORY_NEGATIVE_FRACTION
        ),
    }
    return checks, derived


def _strict_witness_checks(metrics: Mapping[str, float]) -> dict[str, bool]:
    """Require a GT witness to satisfy every reachable Gate3 constraint."""
    valid_instances = metrics["valid_instances"]
    valid_drop_rows = metrics["valid_drop_rows"]
    role_reachable = metrics["role_exclusive_reachable_instances"]
    return {
        "all_generic_instances_are_safe": (
            metrics["keep_safe_instances"] == valid_instances
        ),
        "all_generic_instances_are_deployed": (
            metrics["keep_deployed_instances"] == valid_instances
        ),
        "all_role_exclusive_instances_are_reachable": (
            role_reachable
            == float(EXPECTED_ROLE_EXCLUSIVE_REACHABLE_DIRECTIONS)
            and metrics["role_exclusive_unreachable_instances"] == 0.0
        ),
        "all_role_exclusive_instances_are_safe": (
            metrics["role_exclusive_keep_safe_instances"] == role_reachable
        ),
        "all_role_exclusive_instances_are_deployed": (
            metrics["role_exclusive_keep_deployed_instances"] == role_reachable
        ),
        "all_negative_rows_are_safe": (
            metrics["drop_safe_rows"] == valid_drop_rows
        ),
        "all_negative_rows_are_deployed": (
            metrics["drop_deployed_rows"] == valid_drop_rows
        ),
        "no_category_negative_query_is_deployed": (
            metrics["deployed_category_negative_queries"] == 0.0
        ),
    }


def _select_gt_queries(
    boxes: torch.Tensor,
    targets: Sequence[Mapping[str, Any]],
    candidate: torch.Tensor,
    cfg: Any,
) -> torch.Tensor:
    candidates = box_ops.box_cxcywh_to_xyxy(boxes.detach().float())
    selected = torch.zeros_like(candidate)
    for row_index, target in enumerate(targets):
        target_boxes = box_ops.box_cxcywh_to_xyxy(
            target["boxes"].detach().float()
        )
        iou, _ = box_ops.box_iou(candidates[row_index], target_boxes)
        positive = (
            iou >= cfg.stage_b_data_driven_positive_iou_threshold
        ) & candidate[row_index, :, None]
        for instance_index in range(int(target_boxes.shape[0])):
            mask = positive[:, instance_index]
            if not bool(mask.any().item()):
                continue
            query = iou[:, instance_index].masked_fill(~mask, -1.0).argmax()
            selected[row_index, query] = True

        roles = target["stage_b_data_driven_assignment_role"].reshape(-1)
        pair_valid = bool(
            target["stage_b_data_driven_assignment_valid"].reshape(-1)[0].item()
        )
        if pair_valid:
            for role in (0, 1):
                instance = int(
                    torch.nonzero(roles == role, as_tuple=False).reshape(-1)[0]
                )
                other = torch.ones(
                    (int(target_boxes.shape[0]),),
                    dtype=torch.bool,
                    device=iou.device,
                )
                other[instance] = False
                exclusive = positive[:, instance] & (
                    iou[:, other]
                    < cfg.stage_b_data_driven_patch_negative_iou_threshold
                ).all(dim=1)
                if not bool(exclusive.any().item()):
                    raise RealModelAuditError(
                        "sealed O64 role-exclusive oracle became unreachable"
                    )
                query = iou[:, instance].masked_fill(~exclusive, -1.0).argmax()
                selected[row_index, query] = True
    if bool((~selected.any(dim=1)).any().item()):
        raise RealModelAuditError("GT patch oracle produced an empty row")
    return selected


def run_audit(*, device_name: str, seed: int) -> dict[str, Any]:
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RealModelAuditError("patch oracle audit requires CUDA")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    _seed_everything(seed)
    binding = AUDIT_VARIANTS["raw_centered"]
    cfg = SLConfig.fromfile(str(binding["config"]))
    model, criterion, config_path, dataset_path, initializer_path = _build_runtime(
        cfg, device, binding=binding
    )
    del criterion
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    loaded, identities = _select_rows(cfg, payload["train"], seed=seed)
    raw_targets = [target for _image, target in loaded]
    canonical, expressions = _build_stage_b_data_driven_assignment_captions(
        raw_targets
    )
    samples = nested_tensor_from_tensor_list(
        [image for image, _target in loaded]
    ).to(device)
    patches = torch.stack(
        [target["patch"] for target in raw_targets], dim=0
    ).to(device)
    targets = [_move_criterion_target(target, device) for target in raw_targets]
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        outputs = model(
            samples,
            captions=canonical,
            patches=patches,
            stage_b_data_driven_expression_captions=expressions,
        )
    base = outputs["pred_logits_patch_base"].detach().float()
    adapted = outputs["pred_logits_patch"].detach().float()
    boxes = outputs["pred_boxes"].detach().float()
    paired_candidate = outputs["stage_b_data_driven_candidate_mask"].detach()
    if base.dim() == 3 and int(base.shape[-1]) == 1:
        base = base[..., 0]
        adapted = adapted[..., 0]
    if (
        tuple(base.shape) != (64, EXPECTED_QUERY_COUNT)
        or not torch.equal(base, adapted)
        or tuple(paired_candidate.shape) != (64, EXPECTED_QUERY_COUNT, 2)
        or not torch.equal(paired_candidate[..., 0], paired_candidate[..., 1])
        or not bool(paired_candidate.all().item())
    ):
        raise RealModelAuditError("patch oracle U0 model surface drifted")
    candidate = paired_candidate[..., 0]
    scale = float(
        model.patch_logit_scale.detach().float().exp().clamp(
            max=model.patch_logit_scale_max
        ).item()
    )
    del outputs, model, samples, patches
    gc.collect()
    torch.cuda.empty_cache()

    baseline_contract = _patch_contract(base, boxes, targets, candidate, cfg)
    baseline = _metrics(baseline_contract)
    selected = _select_gt_queries(boxes, targets, candidate, cfg)
    selected_counts = selected.sum(dim=1)

    direct_score = selected.to(dtype=base.dtype)
    direct_contract = _patch_contract(
        direct_score, boxes, targets, candidate, cfg
    )
    direct_metrics = _metrics(direct_contract)
    direct_checks, direct_derived = _checks(direct_metrics, baseline)
    direct_strict_checks = _strict_witness_checks(direct_metrics)
    direct = {
        "status": (
            "passed"
            if all(direct_checks.values()) and all(direct_strict_checks.values())
            else "failed"
        ),
        "checks": direct_checks,
        "strict_witness_checks": direct_strict_checks,
        "derived": direct_derived,
        "metrics": direct_metrics,
    }

    bounded_results = []
    residual_limit = float(cfg.stage_b_data_driven_patch_residual_limit)
    query_count = int(base.shape[1])
    for amplitude in RAW_AMPLITUDES:
        high_count = selected_counts[:, None].float()
        low_count = float(query_count) - high_count
        low_raw = -float(amplitude) * high_count / low_count
        raw = low_raw.expand_as(base).clone()
        raw[selected] = float(amplitude)
        centered_raw = raw - raw.mean(dim=1, keepdim=True)
        residual = residual_limit * torch.tanh(centered_raw / residual_limit)
        score = base + scale * residual
        contract = _patch_contract(score, boxes, targets, candidate, cfg)
        metrics = _metrics(contract)
        checks, derived = _checks(metrics, baseline)
        bounded_results.append(
            {
                "raw_amplitude": float(amplitude),
                "status": "passed" if all(checks.values()) else "failed",
                "checks": checks,
                "derived": derived,
                "metrics": metrics,
                "residual": {
                    "abs_max": float(residual.abs().max().item()),
                    "abs_mean": float(residual.abs().mean().item()),
                    "positive_min": float(residual[selected].min().item()),
                    "unselected_max": float(residual[~selected].max().item()),
                    "centered_raw_row_mean_abs_max": float(
                        centered_raw.mean(dim=1).abs().max().item()
                    ),
                },
            }
        )

    # This explicitly exercises the full negative side of the centered-tanh
    # domain. With sparse selected queries, one shared amplitude requires a very
    # large positive raw value before the many unselected queries approach -L.
    low_centered_raw = -residual_limit * math.atanh(0.99)
    selected_count = selected_counts[:, None].float()
    unselected_count = float(query_count) - selected_count
    high_centered_raw = -low_centered_raw * unselected_count / selected_count
    saturated_raw = torch.full_like(base, low_centered_raw)
    saturated_raw = torch.where(
        selected,
        high_centered_raw.expand_as(base),
        saturated_raw,
    )
    saturated_centered_raw = saturated_raw - saturated_raw.mean(
        dim=1, keepdim=True
    )
    saturated_residual = residual_limit * torch.tanh(
        saturated_centered_raw / residual_limit
    )
    saturated_score = base + scale * saturated_residual
    saturated_contract = _patch_contract(
        saturated_score, boxes, targets, candidate, cfg
    )
    saturated_metrics = _metrics(saturated_contract)
    saturated_checks, saturated_derived = _checks(saturated_metrics, baseline)
    saturated_strict_checks = _strict_witness_checks(saturated_metrics)
    saturated_bounded = {
        "construction": "unselected_negative_99pct_limit_zero_mean_raw_v1",
        "status": (
            "passed"
            if all(saturated_checks.values())
            and all(saturated_strict_checks.values())
            else "failed"
        ),
        "checks": saturated_checks,
        "strict_witness_checks": saturated_strict_checks,
        "derived": saturated_derived,
        "metrics": saturated_metrics,
        "residual": {
            "abs_max": float(saturated_residual.abs().max().item()),
            "abs_mean": float(saturated_residual.abs().mean().item()),
            "positive_min": float(saturated_residual[selected].min().item()),
            "unselected_max": float(saturated_residual[~selected].max().item()),
            "unselected_min": float(saturated_residual[~selected].min().item()),
            "centered_raw_row_mean_abs_max": float(
                saturated_centered_raw.mean(dim=1).abs().max().item()
            ),
        },
    }

    passing_bounded = [
        result for result in bounded_results if result["status"] == "passed"
    ]
    best_bounded = min(
        bounded_results,
        key=lambda result: (
            -sum(bool(value) for value in result["checks"].values()),
            result["metrics"]["loss"],
        ),
    )
    return {
        "schema": "pivot.stageb.data_driven.patch_oracle_overfit64/v2",
        "status": (
            "passed"
            if direct["status"] == "passed"
            and saturated_bounded["status"] == "passed"
            else "failed"
        ),
        "interpretation": {
            "direct_score_gate_loss_is_feasible": direct["status"] == "passed",
            "bounded_additive_residual_is_feasible": (
                saturated_bounded["status"] == "passed"
            ),
            "loss_gate_and_residual_bound_have_feasible_gt_witness": (
                direct["status"] == "passed"
                and saturated_bounded["status"] == "passed"
            ),
            "learned_shared_mapping_or_optimization_remains_unresolved": (
                direct["status"] == "passed"
                and saturated_bounded["status"] == "passed"
            ),
        },
        "bindings": {
            "audit_script_sha256": _sha256(Path(__file__).resolve()),
            "config_path": str(config_path),
            "config_sha256": binding["config_sha256"],
            "dataset_config_path": str(dataset_path),
            "dataset_config_sha256": EXPECTED_DATASET_CONFIG_SHA256,
            "initializer_path": str(initializer_path),
            "initializer_sha256": binding["initializer_sha256"],
            "selection_member_stream_sha256": _canonical_sha256(identities),
            "fixed_image_tensor_stream_sha256": _tensor_sha256(
                [image for image, _target in loaded]
            ),
            "fixed_patch_tensor_stream_sha256": _tensor_sha256(
                [target["patch"] for target in raw_targets]
            ),
        },
        "invariants": {
            "selection_uses_no_model_scores": True,
            "same_64_unique_images_as_strict_o64": True,
            "all_900_queries_are_candidates": True,
            "gt_selects_only_one_best_query_per_reachable_constraint": True,
            "no_trainable_network_or_teacher_score_is_used": True,
            "direct_and_bounded_oracles_use_exact_deployment_gate_loss": True,
        },
        "selection": {
            "seed": seed,
            "members": identities,
            "selected_queries_per_row": selected_counts.tolist(),
            "selected_queries_total": int(selected.sum().item()),
        },
        "base_patch_logit_scale": scale,
        "baseline": baseline,
        "direct_gt_score_oracle": direct,
        "bounded_raw_centered_residual_oracles": bounded_results,
        "bounded_raw_centered_two_sided_99pct_oracle": saturated_bounded,
        "best_bounded_oracle": best_bounded,
        "first_passing_bounded_amplitude": (
            passing_bounded[0]["raw_amplitude"] if passing_bounded else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(device_name=args.device, seed=args.seed)
    _write_json_exclusive(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
