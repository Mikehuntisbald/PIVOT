#!/usr/bin/env python3
"""Run a sealed, model-score-free clean Overfit64 role-routing probe."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataset  # noqa: E402
from engine import (  # noqa: E402
    _build_stage_b_data_driven_assignment_captions,
    _clip_stage_b_data_driven_optimizer_grad_norms,
    _set_stage_b_data_driven_training_mode,
)
from main import (  # noqa: E402
    _make_grad_scaler,
    _stage_b_data_driven_optimizer_groups,
)
from models.GroundingDINO.stage_b_data_driven_score import (  # noqa: E402
    _candidate_official_assignment_role_iou,
    data_driven_category_gate_mask,
)
from tools.audit_stageb_data_driven_role_routed_coverage import (  # noqa: E402
    _build_runtime,
)
from tools.audit_stageb_data_driven_role_routed_real_model import (  # noqa: E402
    AUDIT_VARIANTS,
    DATASET_CONFIG,
    EXPECTED_DATASET_CONFIG_SHA256,
    EXPECTED_QUERY_COUNT,
    RealModelAuditError,
    _move_criterion_target,
    _seed_everything,
    _sha256,
    _write_json_exclusive,
)
from util.misc import nested_tensor_from_tensor_list  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


QUOTAS = (22, 21, 21)
SELECTION_CANDIDATES_PER_SOURCE = 4096
MIN_FINAL_GATE_RETENTION = 0.95
MIN_FINAL_MARGIN_DIRECTION_FRACTION = 0.90
MIN_FINAL_DEPLOYMENT_OWNED_DIRECTION_FRACTION = 0.90
MAX_FINAL_LOSS_RATIO = 0.25
MAX_FINAL_PATCH_LOSS_RATIO = 0.25
MAX_FINAL_PATCH_COMPONENT_RATIO = 0.50
MAX_FINAL_PATCH_ALIGNED_AUXILIARY_RATIO = 0.25
MIN_FINAL_PATCH_COVERAGE_FRACTION = 0.95
MAX_FINAL_GATED_CATEGORY_NEGATIVE_FRACTION = 0.001
EXPECTED_ROLE_EXCLUSIVE_REACHABLE_DIRECTIONS = 128


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


def _tensor_sha256(values: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().cpu().contiguous()
        header = json.dumps(
            [str(tensor.dtype), list(tensor.shape)], separators=(",", ":")
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        if tensor.numel():
            digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def _select_rows(
    cfg: Any,
    sources: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> tuple[list[tuple[torch.Tensor, Mapping[str, Any]]], list[dict[str, Any]]]:
    if len(sources) != len(QUOTAS):
        raise RealModelAuditError("Overfit64 requires all three clean sources")
    selected: list[tuple[torch.Tensor, Mapping[str, Any]]] = []
    identities: list[dict[str, Any]] = []
    used_images: set[int] = set()
    for source_index, (source, quota) in enumerate(zip(sources, QUOTAS)):
        source_seed = int(seed + 1009 * source_index)
        _seed_everything(source_seed)
        dataset = build_dataset("train", cfg, dict(source))
        candidate_count = min(SELECTION_CANDIDATES_PER_SOURCE, len(dataset))
        candidate_indices = random.Random(source_seed).sample(
            range(len(dataset)), candidate_count
        )
        source_selected = 0
        for dataset_index in candidate_indices:
            image, target = dataset[dataset_index]
            pair_valid = target["stage_b_data_driven_assignment_valid"]
            if not bool(pair_valid.reshape(-1)[0].item()):
                continue
            image_id = int(target["image_id"].reshape(-1)[0].item())
            if image_id in used_images:
                continue
            used_images.add(image_id)
            selected.append((image, target))
            identities.append(
                {
                    "source_index": source_index,
                    "source_seed": source_seed,
                    "dataset_index": int(dataset_index),
                    "image_id": image_id,
                    "sample_id": str(target.get("sample_id", "")),
                    "anchor_expression": str(
                        target["stage_b_data_driven_assignment_expressions"][0]
                    ),
                    "partner_expression": str(
                        target["stage_b_data_driven_assignment_expressions"][1]
                    ),
                }
            )
            source_selected += 1
            if source_selected == quota:
                break
        if source_selected != quota:
            raise RealModelAuditError(
                f"could not select clean Overfit64 quota for source {source_index}: "
                f"selected={source_selected}, required={quota}"
            )
    if len(selected) != 64 or len(used_images) != 64:
        raise RealModelAuditError("clean Overfit64 selection is not 64 unique images")
    return selected, identities


def _criterion_metrics(
    outputs: Mapping[str, Any],
    losses: Mapping[str, torch.Tensor],
    targets: Sequence[Mapping[str, Any]],
    cfg: Any,
) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for key, value in losses.items():
        if torch.is_tensor(value) and value.numel() == 1:
            result[key] = float(value.detach().float().item())
    assignment_iou, other_iou, pair_valid = _candidate_official_assignment_role_iou(
        outputs["pred_boxes"], targets
    )
    candidate = outputs["stage_b_data_driven_candidate_mask"]
    if tuple(candidate.shape) != (len(targets), EXPECTED_QUERY_COUNT, 2):
        raise RealModelAuditError("Overfit64 candidate mask shape drifted")
    if not bool(candidate.all()):
        raise RealModelAuditError("Overfit64 did not retain all 900 canonical queries")
    patch_score = outputs["pred_logits_patch"]
    patch_base = outputs.get("pred_logits_patch_base")
    patch_residual = outputs.get("pred_logits_patch_residual")
    if not torch.is_tensor(patch_base) or not torch.is_tensor(patch_residual):
        raise RealModelAuditError("Overfit64 residual patch outputs are incomplete")
    residual_limit = float(cfg.stage_b_data_driven_patch_residual_limit)
    detached_residual = patch_residual.detach().float()
    if detached_residual.dim() == 3:
        if int(detached_residual.shape[-1]) != 1:
            raise RealModelAuditError(
                "Overfit64 residual diagnostics require one support patch"
            )
        diagnostic_residual = detached_residual[..., 0]
    else:
        diagnostic_residual = detached_residual
    if tuple(diagnostic_residual.shape) != tuple(candidate.shape[:2]):
        raise RealModelAuditError("Overfit64 residual diagnostics are misaligned")
    residual_row_mean = diagnostic_residual.mean(dim=1, keepdim=True)
    centered_residual = diagnostic_residual - residual_row_mean
    result["audit_patch_residual_abs_max"] = float(
        detached_residual.abs().max().item()
    )
    result["audit_patch_residual_abs_mean"] = float(
        detached_residual.abs().mean().item()
    )
    result["audit_patch_residual_nonzero_fraction"] = float(
        (detached_residual != 0).float().mean().item()
    )
    result["audit_patch_residual_saturation_fraction"] = float(
        (detached_residual.abs() >= 0.95 * residual_limit)
        .float()
        .mean()
        .item()
    )
    result["audit_patch_residual_row_signed_mean"] = float(
        residual_row_mean.mean().item()
    )
    result["audit_patch_residual_row_signed_mean_abs_mean"] = float(
        residual_row_mean.abs().mean().item()
    )
    result["audit_patch_residual_centered_rms"] = float(
        centered_residual.square().mean().sqrt().item()
    )
    result["audit_patch_residual_centered_abs_max"] = float(
        centered_residual.abs().max().item()
    )
    result["audit_patch_residual_positive_fraction"] = float(
        (diagnostic_residual > 0.0).float().mean().item()
    )
    for fraction, label in ((0.5, "50"), (0.7, "70"), (0.9, "90")):
        result[f"audit_patch_residual_abs_ge_{label}pct_limit_fraction"] = float(
            (
                diagnostic_residual.abs()
                >= float(fraction) * residual_limit
            )
            .float()
            .mean()
            .item()
        )
    category_max_iou = torch.maximum(
        assignment_iou.amax(dim=-1), other_iou
    )
    category_positive = (
        category_max_iou >= cfg.stage_b_data_driven_positive_iou_threshold
    ) & candidate[..., 0]
    category_negative = (
        category_max_iou < cfg.stage_b_data_driven_patch_negative_iou_threshold
    ) & candidate[..., 0]
    category_neutral = candidate[..., 0] & ~(
        category_positive | category_negative
    )
    for label, mask in (
        ("positive", category_positive),
        ("negative", category_negative),
        ("neutral", category_neutral),
    ):
        if not bool(mask.any().item()):
            raise RealModelAuditError(
                f"Overfit64 has no {label} residual diagnostic queries"
            )
        result[f"audit_patch_residual_{label}_query_mean"] = float(
            diagnostic_residual[mask].mean().item()
        )
    result["audit_patch_residual_positive_minus_negative_mean"] = float(
        result["audit_patch_residual_positive_query_mean"]
        - result["audit_patch_residual_negative_query_mean"]
    )
    result["audit_patch_matches_base_bitwise"] = int(
        torch.equal(patch_score.detach(), patch_base.detach())
    )
    if patch_score.dim() == 3:
        patch_score = patch_score[..., 0]
    gate, _standardized = data_driven_category_gate_mask(
        patch_score.detach(),
        candidate[..., 0],
        max_gap=cfg.stage_b_data_driven_category_gate_max_gap,
        clip=cfg.stage_b_data_driven_patch_score_clip,
    )
    separated = other_iou < cfg.stage_b_data_driven_rank_negative_iou_threshold
    geometry0 = (
        (assignment_iou[..., 0] >= cfg.stage_b_data_driven_positive_iou_threshold)
        & (
            assignment_iou[..., 1]
            < cfg.stage_b_data_driven_rank_negative_iou_threshold
        )
        & separated
    )
    geometry1 = (
        (assignment_iou[..., 1] >= cfg.stage_b_data_driven_positive_iou_threshold)
        & (
            assignment_iou[..., 0]
            < cfg.stage_b_data_driven_rank_negative_iou_threshold
        )
        & separated
    )
    geometry_rows = pair_valid & geometry0.any(dim=1) & geometry1.any(dim=1)
    gated_rows = (
        pair_valid
        & (geometry0 & gate).any(dim=1)
        & (geometry1 & gate).any(dim=1)
    )
    rank_score = outputs["stage_b_data_driven_text_rank_score"]
    if tuple(rank_score.shape) != tuple(candidate.shape):
        raise RealModelAuditError("Overfit64 paired rank score shape drifted")
    deployment_top = rank_score.masked_fill(~gate[:, :, None], -torch.inf).argmax(
        dim=1
    )
    deployment_own_iou = assignment_iou.gather(
        1, deployment_top[:, None, :]
    ).squeeze(1)
    pair_directions = pair_valid[:, None].expand(-1, 2)
    deployment_iou50 = (
        deployment_own_iou >= cfg.stage_b_data_driven_positive_iou_threshold
    ) & pair_directions
    result["audit_geometry_reachable_rows"] = int(geometry_rows.sum().item())
    result["audit_gated_geometry_rows"] = int(gated_rows.sum().item())
    result["audit_gate_retention"] = (
        float(gated_rows.sum().item()) / float(geometry_rows.sum().item())
        if bool(geometry_rows.any().item())
        else 0.0
    )
    result["audit_gate_queries"] = int(gate.sum().item())
    result["audit_deployment_iou50_directions"] = int(
        deployment_iou50.sum().item()
    )
    result["audit_deployment_valid_directions"] = int(
        pair_directions.sum().item()
    )
    result["audit_deployment_iou50_fraction"] = (
        float(deployment_iou50.sum().item())
        / float(pair_directions.sum().item())
        if bool(pair_directions.any().item())
        else 0.0
    )
    return result


def _forward_metrics(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    samples,
    patches: torch.Tensor,
    targets: Sequence[Mapping[str, Any]],
    canonical: Sequence[str],
    expressions: Sequence[Sequence[str]],
    cfg: Any,
) -> dict[str, float | int]:
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        outputs = model(
            samples,
            captions=canonical,
            patches=patches,
            stage_b_data_driven_expression_captions=expressions,
        )
        losses = criterion(outputs, targets)
    return _criterion_metrics(outputs, losses, targets, cfg)


def _weighted_loss(
    losses: Mapping[str, torch.Tensor], criterion: torch.nn.Module
) -> torch.Tensor:
    terms = [
        losses[key] * float(weight)
        for key, weight in criterion.weight_dict.items()
        if key in losses
    ]
    if len(terms) != len(criterion.weight_dict):
        raise RealModelAuditError("Overfit64 criterion lost a weighted loss")
    return torch.stack([term.float() for term in terms]).sum()


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0 if numerator <= 0.0 else float("inf")
    return numerator / denominator


def run_probe(
    *,
    device_name: str,
    steps: int,
    log_interval: int,
    seed: int,
    output_dir: Path,
    variant: str = "uncentered",
) -> dict[str, Any]:
    if steps <= 0 or log_interval <= 0:
        raise RealModelAuditError("Overfit64 steps/log interval must be positive")
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RealModelAuditError("Overfit64 requires an available CUDA device")
    output_dir = output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    _seed_everything(seed)
    binding = AUDIT_VARIANTS.get(variant)
    if binding is None:
        raise RealModelAuditError(f"unknown residual O64 variant: {variant!r}")
    cfg = SLConfig.fromfile(str(binding["config"]))
    model, criterion, config_path, dataset_path, initializer_path = _build_runtime(
        cfg, device, binding=binding
    )
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    loaded, selection = _select_rows(cfg, payload["train"], seed=seed)
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
    selection_binding = {
        "policy": "seeded_uniform_candidates_then_valid_pair_unique_image_quota_v1",
        "model_score_free": True,
        "seed": seed,
        "candidate_draws_per_source": SELECTION_CANDIDATES_PER_SOURCE,
        "source_quotas": list(QUOTAS),
        "members": selection,
        "member_stream_sha256": _canonical_sha256(selection),
        "fixed_image_tensor_stream_sha256": _tensor_sha256(
            [image for image, _target in loaded]
        ),
        "fixed_patch_tensor_stream_sha256": _tensor_sha256(
            [target["patch"] for target in raw_targets]
        ),
    }

    optimizer_groups = _stage_b_data_driven_optimizer_groups(
        model,
        train_mode="rank_patch_only",
        rank_lr=cfg.stage_b_data_driven_rank_lr,
        confidence_lr=cfg.stage_b_data_driven_confidence_lr,
        patch_lr=cfg.stage_b_data_driven_patch_lr,
    )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scaler = _make_grad_scaler(enabled=True, init_scale=cfg.amp_init_scale)
    residual_module = model.stage_b_data_driven_patch_residual
    if residual_module is None:
        raise RealModelAuditError("Overfit64 residual module is missing")
    base_patch_named = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith(
            (
                "patch_encoder.input_proj.",
                "patch_encoder.norm.",
                "query_proj_for_patch.",
            )
        )
    )
    if len(base_patch_named) != 8:
        raise RealModelAuditError("Overfit64 base patch tensor set drifted")
    base_patch_tensor_sha256 = _tensor_sha256(
        [parameter for _name, parameter in base_patch_named]
    )
    bootstrap_gradient_audit: dict[str, float | bool] = {}
    _set_stage_b_data_driven_training_mode(model, "rank_patch_only")
    criterion.train()
    baseline = _forward_metrics(
        model,
        criterion,
        samples,
        patches,
        targets,
        canonical,
        expressions,
        cfg,
    )
    history = [{"optimizer_updates": 0, **baseline}]
    print(json.dumps(history[-1], sort_keys=True), flush=True)
    progress_path = output_dir / "progress.jsonl"
    progress_handle = progress_path.open("x", encoding="utf-8", newline="\n")
    progress_handle.write(json.dumps(history[-1], sort_keys=True) + "\n")
    progress_handle.flush()
    os.fsync(progress_handle.fileno())
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    updates = 0
    attempts = 0
    amp_skips = 0
    last_grad_norms: dict[str, float] = {}
    while updates < steps:
        attempts += 1
        if attempts > steps + 20:
            raise RealModelAuditError("Overfit64 exceeded AMP retry budget")
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(
                samples,
                captions=canonical,
                patches=patches,
                stage_b_data_driven_expression_captions=expressions,
            )
            losses = criterion(outputs, targets)
            total_loss = _weighted_loss(losses, criterion)
        if not bool(torch.isfinite(total_loss).item()):
            raise RealModelAuditError("Overfit64 produced a non-finite total loss")
        scaler.scale(total_loss).backward()
        scale_before = float(scaler.get_scale())
        scaler.unscale_(optimizer)
        if updates == 0:
            first_output_grad = float(
                residual_module.output.weight.grad.detach().float().abs().sum().item()
            )
            first_input_grad = float(
                sum(
                    parameter.grad.detach().float().abs().sum()
                    for parameter in residual_module.input.parameters()
                ).item()
            )
            context_output = getattr(residual_module, "context_output", None)
            context_input = getattr(residual_module, "context_input", None)
            first_context_output_grad = (
                float(
                    context_output.weight.grad.detach().float().abs().sum().item()
                )
                if context_output is not None
                else 0.0
            )
            first_context_input_grad = (
                float(
                    sum(
                        parameter.grad.detach().float().abs().sum()
                        for parameter in context_input.parameters()
                    ).item()
                )
                if context_input is not None
                else 0.0
            )
            if (
                first_output_grad <= 0.0
                or first_input_grad != 0.0
                or (
                    context_output is not None
                    and first_context_output_grad <= 0.0
                )
                or first_context_input_grad != 0.0
            ):
                raise RealModelAuditError(
                    "zero-output residual output-layer bootstrap drifted"
                )
            bootstrap_gradient_audit.update(
                first_update_output_grad_abs_sum=(
                    first_output_grad + first_context_output_grad
                ),
                first_update_input_grad_abs_sum=(
                    first_input_grad + first_context_input_grad
                ),
                first_update_only_output_layers_have_gradient=True,
                first_update_context_output_grad_abs_sum=(
                    first_context_output_grad
                ),
            )
        elif updates == 1 and "second_update_trunk_has_gradient" not in bootstrap_gradient_audit:
            second_input_grad = float(
                sum(
                    parameter.grad.detach().float().abs().sum()
                    for parameter in residual_module.input.parameters()
                ).item()
            )
            context_input = getattr(residual_module, "context_input", None)
            second_context_input_grad = (
                float(
                    sum(
                        parameter.grad.detach().float().abs().sum()
                        for parameter in context_input.parameters()
                    ).item()
                )
                if context_input is not None
                else 0.0
            )
            if (
                second_input_grad <= 0.0
                or (
                    context_input is not None
                    and second_context_input_grad <= 0.0
                )
            ):
                raise RealModelAuditError(
                    "patch residual trunk did not receive gradient on update two"
                )
            bootstrap_gradient_audit.update(
                second_update_input_grad_abs_sum=(
                    second_input_grad + second_context_input_grad
                ),
                second_update_context_input_grad_abs_sum=(
                    second_context_input_grad
                ),
                second_update_trunk_has_gradient=True,
            )
        last_grad_norms = _clip_stage_b_data_driven_optimizer_grad_norms(
            optimizer,
            cfg.clip_max_norm,
            train_mode="rank_patch_only",
        )
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        del total_loss, losses, outputs
        if scale_after < scale_before:
            amp_skips += 1
            continue
        updates += 1
        if updates == 1 or updates % log_interval == 0 or updates == steps:
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            metrics = _forward_metrics(
                model,
                criterion,
                samples,
                patches,
                targets,
                canonical,
                expressions,
                cfg,
            )
            gc.collect()
            torch.cuda.empty_cache()
            record = {
                "optimizer_updates": updates,
                "amp_scale": float(scaler.get_scale()),
                "amp_skips": amp_skips,
                "rank_grad_norm_preclip": float(
                    last_grad_norms.get(
                        "grad_norm_data_driven_rank_preclip", 0.0
                    )
                ),
                "patch_grad_norm_preclip": float(
                    last_grad_norms.get(
                        "grad_norm_data_driven_patch_preclip", 0.0
                    )
                ),
                "cuda_memory_allocated_after_eval_cache": int(
                    torch.cuda.memory_allocated(device)
                ),
                "cuda_memory_reserved_after_eval_cache": int(
                    torch.cuda.memory_reserved(device)
                ),
                **metrics,
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            progress_handle.write(json.dumps(record, sort_keys=True) + "\n")
            progress_handle.flush()
            os.fsync(progress_handle.fileno())
    progress_handle.close()
    final = history[-1]
    runtime_directions = float(
        final["stage_b_data_driven_assignment_runtime_directions"]
    )
    margin_fraction = _ratio(
        float(final["stage_b_data_driven_assignment_margin_directions"]),
        runtime_directions,
    )
    deployment_fraction = _ratio(
        float(
            final[
                "stage_b_data_driven_assignment_deployment_correct_directions"
            ]
        ),
        runtime_directions,
    )
    rank_loss_ratio = _ratio(
        float(final["loss_stage_b_data_driven_role_routed_rank"]),
        float(baseline["loss_stage_b_data_driven_role_routed_rank"]),
    )
    patch_loss_ratio = _ratio(
        float(final["loss_stage_b_data_driven_patch"]),
        float(baseline["loss_stage_b_data_driven_patch"]),
    )
    patch_keep_component_ratio = _ratio(
        float(final["stage_b_data_driven_patch_keep_component"]),
        float(baseline["stage_b_data_driven_patch_keep_component"]),
    )
    patch_keep_objective_component_ratio = _ratio(
        float(
            final[
                "stage_b_data_driven_patch_keep_objective_component"
            ]
        ),
        float(
            baseline[
                "stage_b_data_driven_patch_keep_objective_component"
            ]
        ),
    )
    patch_keep_mean_component_ratio = _ratio(
        float(final["stage_b_data_driven_patch_keep_mean_component"]),
        float(baseline["stage_b_data_driven_patch_keep_mean_component"]),
    )
    patch_drop_component_ratio = _ratio(
        float(final["stage_b_data_driven_patch_drop_component"]),
        float(baseline["stage_b_data_driven_patch_drop_component"]),
    )
    patch_drop_objective_component_ratio = _ratio(
        float(
            final[
                "stage_b_data_driven_patch_drop_objective_component"
            ]
        ),
        float(
            baseline[
                "stage_b_data_driven_patch_drop_objective_component"
            ]
        ),
    )
    patch_drop_active_unsafe_component_ratio = _ratio(
        float(
            final[
                "stage_b_data_driven_patch_drop_active_unsafe_component"
            ]
        ),
        float(
            baseline[
                "stage_b_data_driven_patch_drop_active_unsafe_component"
            ]
        ),
    )
    patch_dense_category_focal_component_ratio = _ratio(
        float(
            final[
                "stage_b_data_driven_patch_dense_category_focal_component"
            ]
        ),
        float(
            baseline[
                "stage_b_data_driven_patch_dense_category_focal_component"
            ]
        ),
    )
    patch_dense_category_positive_focal_component_ratio = _ratio(
        float(
            final[
                "stage_b_data_driven_patch_dense_category_positive_focal_component"
            ]
        ),
        float(
            baseline[
                "stage_b_data_driven_patch_dense_category_positive_focal_component"
            ]
        ),
    )
    patch_dense_category_negative_focal_component_ratio = _ratio(
        float(
            final[
                "stage_b_data_driven_patch_dense_category_negative_focal_component"
            ]
        ),
        float(
            baseline[
                "stage_b_data_driven_patch_dense_category_negative_focal_component"
            ]
        ),
    )
    patch_role_exclusive_component_ratio = _ratio(
        float(
            final[
                "stage_b_data_driven_patch_role_exclusive_keep_component"
            ]
        ),
        float(
            baseline[
                "stage_b_data_driven_patch_role_exclusive_keep_component"
            ]
        ),
    )
    patch_keep_deployed_fraction = _ratio(
        float(final["stage_b_data_driven_patch_keep_deployed_instances"]),
        float(final["stage_b_data_driven_patch_valid_instances"]),
    )
    patch_keep_safe_fraction = _ratio(
        float(final["stage_b_data_driven_patch_keep_safe_instances"]),
        float(final["stage_b_data_driven_patch_valid_instances"]),
    )
    patch_drop_deployed_fraction = _ratio(
        float(final["stage_b_data_driven_patch_drop_deployed_rows"]),
        float(final["stage_b_data_driven_patch_valid_drop_rows"]),
    )
    patch_drop_safe_fraction = _ratio(
        float(final["stage_b_data_driven_patch_drop_safe_rows"]),
        float(final["stage_b_data_driven_patch_valid_drop_rows"]),
    )
    role_exclusive_reachable = float(
        final[
            "stage_b_data_driven_patch_role_exclusive_reachable_instances"
        ]
    )
    role_exclusive_unreachable = float(
        final[
            "stage_b_data_driven_patch_role_exclusive_unreachable_instances"
        ]
    )
    patch_role_exclusive_deployed_fraction = _ratio(
        float(
            final[
                "stage_b_data_driven_patch_role_exclusive_keep_deployed_instances"
            ]
        ),
        role_exclusive_reachable,
    )
    patch_role_exclusive_safe_fraction = _ratio(
        float(
            final[
                "stage_b_data_driven_patch_role_exclusive_keep_safe_instances"
            ]
        ),
        role_exclusive_reachable,
    )
    patch_gated_category_negative_fraction = _ratio(
        float(
            final[
                "stage_b_data_driven_patch_deployed_category_negative_queries"
            ]
        ),
        float(final["stage_b_data_driven_patch_category_negative_queries"]),
    )
    checks = {
        "amp_has_no_skips": amp_skips == 0,
        "gate_retention_reaches_threshold": (
            float(final["audit_gate_retention"]) >= MIN_FINAL_GATE_RETENTION
        ),
        "rank_margin_direction_fraction_reaches_threshold": (
            margin_fraction >= MIN_FINAL_MARGIN_DIRECTION_FRACTION
        ),
        "deployment_owned_direction_fraction_reaches_threshold": (
            deployment_fraction
            >= MIN_FINAL_DEPLOYMENT_OWNED_DIRECTION_FRACTION
        ),
        "rank_loss_ratio_reaches_threshold": (
            rank_loss_ratio <= MAX_FINAL_LOSS_RATIO
        ),
        "patch_loss_materially_decreases": (
            patch_loss_ratio <= MAX_FINAL_PATCH_LOSS_RATIO
        ),
        "patch_keep_component_materially_decreases": (
            patch_keep_component_ratio <= MAX_FINAL_PATCH_COMPONENT_RATIO
        ),
        "patch_drop_component_materially_decreases": (
            patch_drop_component_ratio <= MAX_FINAL_PATCH_COMPONENT_RATIO
        ),
        "patch_allnegative_active_severity_materially_decreases": (
            patch_drop_active_unsafe_component_ratio
            <= MAX_FINAL_PATCH_ALIGNED_AUXILIARY_RATIO
        ),
        "patch_role_exclusive_component_materially_decreases": (
            patch_role_exclusive_component_ratio
            <= MAX_FINAL_PATCH_COMPONENT_RATIO
        ),
        "patch_keep_deployed_fraction_reaches_threshold": (
            patch_keep_deployed_fraction
            >= MIN_FINAL_PATCH_COVERAGE_FRACTION
        ),
        "patch_keep_safe_fraction_reaches_threshold": (
            patch_keep_safe_fraction >= MIN_FINAL_PATCH_COVERAGE_FRACTION
        ),
        "patch_drop_deployed_fraction_reaches_threshold": (
            patch_drop_deployed_fraction
            >= MIN_FINAL_PATCH_COVERAGE_FRACTION
        ),
        "patch_drop_safe_fraction_reaches_threshold": (
            patch_drop_safe_fraction >= MIN_FINAL_PATCH_COVERAGE_FRACTION
        ),
        "patch_role_exclusive_reachability_is_complete": (
            role_exclusive_reachable
            == float(EXPECTED_ROLE_EXCLUSIVE_REACHABLE_DIRECTIONS)
            and role_exclusive_unreachable == 0.0
        ),
        "patch_role_exclusive_deployed_fraction_reaches_threshold": (
            patch_role_exclusive_deployed_fraction
            >= MIN_FINAL_PATCH_COVERAGE_FRACTION
        ),
        "patch_role_exclusive_safe_fraction_reaches_threshold": (
            patch_role_exclusive_safe_fraction
            >= MIN_FINAL_PATCH_COVERAGE_FRACTION
        ),
        "patch_gated_category_negative_fraction_is_bounded": (
            patch_gated_category_negative_fraction
            <= MAX_FINAL_GATED_CATEGORY_NEGATIVE_FRACTION
        ),
    }
    patch_check_names = {
        key
        for key in checks
        if key.startswith("patch_") or key == "gate_retention_reaches_threshold"
    }
    rank_check_names = {
        key
        for key in checks
        if key.startswith("rank_")
        or key.startswith("deployment_owned_")
    }
    active_state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    checkpoint_path = output_dir / "active_trainable_state.pth"
    final_base_patch_tensor_sha256 = _tensor_sha256(
        [parameter for _name, parameter in base_patch_named]
    )
    if final_base_patch_tensor_sha256 != base_patch_tensor_sha256:
        raise RealModelAuditError("frozen base patch tensors changed during Overfit64")
    with checkpoint_path.open("xb") as handle:
        torch.save(
            {
                "schema": "pivot.stageb.data_driven.role_routed_overfit64_active/v1",
                "optimizer_updates": updates,
                "selection_member_stream_sha256": selection_binding[
                    "member_stream_sha256"
                ],
                "model": active_state,
            },
            handle,
        )
        handle.flush()
        os.fsync(handle.fileno())
    result = {
        "schema": "pivot.stageb.data_driven.role_routed_overfit64_probe/v10",
        "status": "passed" if all(checks.values()) else "failed",
        "subsystem_status": {
            "patch": (
                "passed"
                if checks["amp_has_no_skips"]
                and all(checks[key] for key in patch_check_names)
                else "failed"
            ),
            "rank": (
                "passed"
                if checks["amp_has_no_skips"]
                and all(checks[key] for key in rank_check_names)
                else "failed"
            ),
        },
        "device": str(device),
        "optimizer_updates": updates,
        "attempts": attempts,
        "selection": selection_binding,
        "bindings": {
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "variant": variant,
            "config_path": str(config_path),
            "config_sha256": binding["config_sha256"],
            "dataset_config_path": str(dataset_path),
            "dataset_config_sha256": EXPECTED_DATASET_CONFIG_SHA256,
            "initializer_path": str(initializer_path),
            "initializer_sha256": binding["initializer_sha256"],
            "score_implementation_sha256": _sha256(
                REPO_ROOT / "models/GroundingDINO/stage_b_data_driven_score.py"
            ),
            "patch_residual_implementation_sha256": _sha256(
                REPO_ROOT
                / "models/GroundingDINO/stage_b_data_driven_patch_residual.py"
            ),
        },
        "optimizer": {
            "rank_lr": cfg.stage_b_data_driven_rank_lr,
            "patch_lr": cfg.stage_b_data_driven_patch_lr,
            "weight_decay": cfg.weight_decay,
            "per_branch_clip_max_norm": cfg.clip_max_norm,
            "amp_init_scale": cfg.amp_init_scale,
            "patch_training_surface": (
                cfg.stage_b_data_driven_patch_training_surface
            ),
            "patch_residual_architecture": residual_module.architecture(),
        },
        "thresholds": {
            "minimum_final_gate_retention": MIN_FINAL_GATE_RETENTION,
            "minimum_final_margin_direction_fraction": (
                MIN_FINAL_MARGIN_DIRECTION_FRACTION
            ),
            "minimum_final_deployment_owned_direction_fraction": (
                MIN_FINAL_DEPLOYMENT_OWNED_DIRECTION_FRACTION
            ),
            "maximum_final_rank_loss_ratio": MAX_FINAL_LOSS_RATIO,
            "maximum_final_patch_component_ratio": (
                MAX_FINAL_PATCH_COMPONENT_RATIO
            ),
            "maximum_final_patch_loss_ratio": MAX_FINAL_PATCH_LOSS_RATIO,
            "maximum_final_patch_aligned_auxiliary_ratio": (
                MAX_FINAL_PATCH_ALIGNED_AUXILIARY_RATIO
            ),
            "minimum_final_patch_coverage_fraction": (
                MIN_FINAL_PATCH_COVERAGE_FRACTION
            ),
            "maximum_final_gated_category_negative_fraction": (
                MAX_FINAL_GATED_CATEGORY_NEGATIVE_FRACTION
            ),
            "expected_role_exclusive_reachable_directions": (
                EXPECTED_ROLE_EXCLUSIVE_REACHABLE_DIRECTIONS
            ),
        },
        "checks": checks,
        "baseline": baseline,
        "final": final,
        "derived": {
            "rank_margin_direction_fraction": margin_fraction,
            "deployment_owned_direction_fraction": deployment_fraction,
            "rank_loss_ratio": rank_loss_ratio,
            "patch_loss_ratio": patch_loss_ratio,
            "patch_keep_component_ratio": patch_keep_component_ratio,
            "patch_keep_objective_component_ratio": (
                patch_keep_objective_component_ratio
            ),
            "patch_keep_mean_component_ratio": patch_keep_mean_component_ratio,
            "patch_drop_component_ratio": patch_drop_component_ratio,
            "patch_drop_objective_component_ratio": (
                patch_drop_objective_component_ratio
            ),
            "patch_drop_active_unsafe_component_ratio": (
                patch_drop_active_unsafe_component_ratio
            ),
            "patch_dense_category_focal_component_ratio": (
                patch_dense_category_focal_component_ratio
            ),
            "patch_dense_category_positive_focal_component_ratio": (
                patch_dense_category_positive_focal_component_ratio
            ),
            "patch_dense_category_negative_focal_component_ratio": (
                patch_dense_category_negative_focal_component_ratio
            ),
            "patch_role_exclusive_keep_component_ratio": (
                patch_role_exclusive_component_ratio
            ),
            "patch_keep_deployed_fraction": patch_keep_deployed_fraction,
            "patch_keep_safe_fraction": patch_keep_safe_fraction,
            "patch_drop_deployed_fraction": patch_drop_deployed_fraction,
            "patch_drop_safe_fraction": patch_drop_safe_fraction,
            "patch_role_exclusive_deployed_fraction": (
                patch_role_exclusive_deployed_fraction
            ),
            "patch_role_exclusive_safe_fraction": (
                patch_role_exclusive_safe_fraction
            ),
            "patch_gated_category_negative_fraction": (
                patch_gated_category_negative_fraction
            ),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "artifacts": {
            "progress_jsonl": {
                "path": str(progress_path.resolve()),
                "sha256": _sha256(progress_path),
            },
            "active_trainable_state": {
                "path": str(checkpoint_path.resolve()),
                "sha256": _sha256(checkpoint_path),
            },
        },
        "invariants": {
            "selection_uses_no_model_scores": True,
            "all_64_rows_are_clean_valid_pairs_from_unique_images": True,
            "fixed_images_and_support_patches_reused_each_update": True,
            "formal_amp_optimizer_groups_and_branch_clipping_used": True,
            "confidence_and_frozen_feature_generators_not_optimized": True,
            "drop_positive_gradient_reaches_every_reachable_instance": True,
            "zero_init_residual_bootstrap": bootstrap_gradient_audit,
            "base_patch_tensor_sha256_before": base_patch_tensor_sha256,
            "base_patch_tensor_sha256_after": final_base_patch_tensor_sha256,
            "base_patch_tensors_remain_bitwise_frozen": True,
            "residual_raw_centering_matches_config": (
                (
                    residual_module.architecture().get("query_centering")
                    == "raw_mean_before_tanh_v1"
                )
                == bool(
                    getattr(
                        cfg,
                        "stage_b_data_driven_patch_residual_center_raw",
                        False,
                    )
                )
            ),
            "residual_raw_centering_mode": residual_module.architecture().get(
                "query_centering", "none"
            ),
            "cuda_allocator_uses_expandable_segments": (
                os.environ.get("PYTORCH_ALLOC_CONF")
                == "expandable_segments:True"
                and os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
                == "expandable_segments:True"
            ),
        },
    }
    _write_json_exclusive(output_dir / "receipt.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variant",
        choices=sorted(AUDIT_VARIANTS),
        default="uncentered",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_probe(
        device_name=args.device,
        steps=args.steps,
        log_interval=args.log_interval,
        seed=args.seed,
        output_dir=args.output_dir,
        variant=args.variant,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
