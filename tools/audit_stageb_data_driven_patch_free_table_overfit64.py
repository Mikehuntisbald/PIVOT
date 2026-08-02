#!/usr/bin/env python3
"""Isolate Stage-B patch optimization with one free raw value per O64 query."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from engine import _build_stage_b_data_driven_assignment_captions  # noqa: E402
from models.GroundingDINO.stage_b_data_driven_score import (  # noqa: E402
    _candidate_official_assignment_role_iou,
    data_driven_category_gate_mask,
)
from tools.audit_stageb_data_driven_patch_oracle_overfit64 import (  # noqa: E402
    _checks,
    _metrics,
    _patch_contract,
    _strict_witness_checks,
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
    MIN_FINAL_GATE_RETENTION,
    _canonical_sha256,
    _select_rows,
    _tensor_sha256,
)
from util.misc import nested_tensor_from_tensor_list  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


REGIMES: dict[str, dict[str, Any]] = {
    "formal_adamw_lr3e4_clip01_u100": {
        "optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "clip_max_norm": 0.1,
        "steps": 100,
        "log_updates": (1, 10, 25, 50, 75, 100),
    },
    "formal_adamw_lr3e4_clip01_u1000": {
        "optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "clip_max_norm": 0.1,
        "steps": 1000,
        "log_updates": (1, 10, 50, 100, 200, 500, 750, 1000),
    },
    "diagnostic_adam_lr1e2_noclip_u500": {
        "optimizer": "adam",
        "lr": 1e-2,
        "weight_decay": 0.0,
        "clip_max_norm": None,
        "steps": 500,
        "log_updates": (1, 10, 25, 50, 100, 200, 300, 500),
    },
}


def _finite_gradient_norm(parameter: torch.Tensor) -> float:
    if parameter.grad is None:
        raise RealModelAuditError("free-table parameter lost its gradient")
    grad = parameter.grad.detach().float()
    if not bool(torch.isfinite(grad).all().item()):
        raise RealModelAuditError("free-table parameter produced a non-finite gradient")
    return float(torch.linalg.vector_norm(grad).item())


def _bounded_residual(raw: torch.Tensor, limit: float) -> tuple[torch.Tensor, torch.Tensor]:
    centered = raw - raw.mean(dim=1, keepdim=True)
    return centered, float(limit) * torch.tanh(centered / float(limit))


def _gate_retention(
    score: torch.Tensor,
    boxes: torch.Tensor,
    targets: Sequence[Mapping[str, Any]],
    candidate: torch.Tensor,
    cfg: Any,
) -> dict[str, float]:
    assignment_iou, other_iou, pair_valid = (
        _candidate_official_assignment_role_iou(boxes, targets)
    )
    gate, _standardized = data_driven_category_gate_mask(
        score.detach(),
        candidate,
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
    geometry_count = float(geometry_rows.sum().item())
    gated_count = float(gated_rows.sum().item())
    return {
        "geometry_reachable_rows": geometry_count,
        "gated_geometry_rows": gated_count,
        "gate_retention": gated_count / geometry_count if geometry_count else 0.0,
        "gate_queries": float(gate.sum().item()),
    }


def _complete_patch_checks(
    contract: Mapping[str, torch.Tensor],
    score: torch.Tensor,
    *,
    boxes: torch.Tensor,
    targets: Sequence[Mapping[str, Any]],
    candidate: torch.Tensor,
    cfg: Any,
    baseline: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, bool], dict[str, float]]:
    metrics = _metrics(contract)
    checks, derived = _checks(metrics, baseline)
    retention = _gate_retention(score, boxes, targets, candidate, cfg)
    metrics.update(retention)
    checks["gate_retention_reaches_threshold"] = (
        retention["gate_retention"] >= MIN_FINAL_GATE_RETENTION
    )
    return metrics, checks, derived


def _evaluate(
    raw: torch.Tensor,
    *,
    base: torch.Tensor,
    boxes: torch.Tensor,
    targets: Sequence[Mapping[str, Any]],
    candidate: torch.Tensor,
    cfg: Any,
    scale: float,
    baseline: Mapping[str, float],
) -> dict[str, Any]:
    with torch.no_grad():
        centered, residual = _bounded_residual(
            raw, float(cfg.stage_b_data_driven_patch_residual_limit)
        )
        score = base + float(scale) * residual
        contract = _patch_contract(
            score,
            boxes,
            targets,
            candidate,
            cfg,
        )
        metrics, checks, derived = _complete_patch_checks(
            contract,
            score,
            boxes=boxes,
            targets=targets,
            candidate=candidate,
            cfg=cfg,
            baseline=baseline,
        )
        strict_checks = _strict_witness_checks(metrics)
        positive = contract["category_positive_mask"]
        negative = contract["category_negative_mask"]
        neutral = contract["category_neutral_mask"]
        residual_metrics = {
            "abs_max": float(residual.abs().max().item()),
            "abs_mean": float(residual.abs().mean().item()),
            "centered_rms": float(centered.square().mean().sqrt().item()),
            "centered_abs_max": float(centered.abs().max().item()),
            "centered_row_mean_abs_max": float(
                centered.mean(dim=1).abs().max().item()
            ),
            "raw_abs_max": float(raw.abs().max().item()),
            "raw_row_mean_abs_mean": float(
                raw.mean(dim=1).abs().mean().item()
            ),
            "positive_mean": float(residual[positive].mean().item()),
            "negative_mean": float(residual[negative].mean().item()),
            "neutral_mean": float(residual[neutral].mean().item()),
            "positive_minus_negative_mean": float(
                residual[positive].mean().item()
                - residual[negative].mean().item()
            ),
            "saturation_fraction": float(
                (
                    residual.abs()
                    >= 0.95 * float(cfg.stage_b_data_driven_patch_residual_limit)
                )
                .float()
                .mean()
                .item()
            ),
        }
    return {
        "gate_status": "passed" if all(checks.values()) else "failed",
        "strict_witness_status": (
            "passed" if all(strict_checks.values()) else "failed"
        ),
        "checks": checks,
        "strict_witness_checks": strict_checks,
        "derived": derived,
        "metrics": metrics,
        "residual": residual_metrics,
    }


def _run_regime(
    name: str,
    spec: Mapping[str, Any],
    *,
    base: torch.Tensor,
    boxes: torch.Tensor,
    targets: Sequence[Mapping[str, Any]],
    candidate: torch.Tensor,
    cfg: Any,
    scale: float,
    baseline: Mapping[str, float],
    progress_handle,
    artifact_path: Path,
) -> dict[str, Any]:
    raw = torch.nn.Parameter(torch.zeros_like(base, dtype=torch.float32))
    optimizer_name = str(spec["optimizer"])
    optimizer_type = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }.get(optimizer_name)
    if optimizer_type is None:
        raise RealModelAuditError(f"unknown free-table optimizer: {optimizer_name!r}")
    optimizer = optimizer_type(
        [raw],
        lr=float(spec["lr"]),
        weight_decay=float(spec["weight_decay"]),
    )
    clip_max_norm = spec["clip_max_norm"]
    steps = int(spec["steps"])
    log_updates = {0, steps, *(int(value) for value in spec["log_updates"])}
    history: list[dict[str, Any]] = []
    gate_sweep: list[dict[str, Any]] = []
    passing_full_records: list[dict[str, Any]] = []
    best_passing_raw: torch.Tensor | None = None
    best_passing_key: tuple[float, float] | None = None
    last_preclip_norm = 0.0
    last_postclip_norm = 0.0

    def sweep_record(
        update: int,
        contract: Mapping[str, torch.Tensor],
        score: torch.Tensor,
    ) -> None:
        nonlocal best_passing_key, best_passing_raw
        metrics, checks, derived = _complete_patch_checks(
            contract,
            score,
            boxes=boxes,
            targets=targets,
            candidate=candidate,
            cfg=cfg,
            baseline=baseline,
        )
        compact = {
                "optimizer_updates": update,
                "gate_status": "passed" if all(checks.values()) else "failed",
                "passed_checks": sum(bool(value) for value in checks.values()),
                "total_checks": len(checks),
                "loss": metrics["loss"],
                "patch_loss_ratio": derived["patch_loss_ratio"],
                "keep_safe_instances": metrics["keep_safe_instances"],
                "drop_safe_rows": metrics["drop_safe_rows"],
                "drop_deployed_rows": metrics["drop_deployed_rows"],
                "deployed_category_negative_queries": metrics[
                    "deployed_category_negative_queries"
                ],
                "gated_category_negative_fraction": derived[
                    "patch_gated_category_negative_fraction"
                ],
                "gate_retention": metrics["gate_retention"],
            }
        gate_sweep.append(compact)
        if all(checks.values()):
            full = {
                "optimizer_updates": update,
                "checks": checks,
                "derived": derived,
                "metrics": metrics,
                "raw_tensor_sha256": _tensor_sha256([raw]),
            }
            passing_full_records.append(full)
            key = (
                metrics["deployed_category_negative_queries"],
                metrics["loss"],
            )
            if best_passing_key is None or key < best_passing_key:
                best_passing_key = key
                best_passing_raw = raw.detach().cpu().clone()

    def record(update: int) -> None:
        result = _evaluate(
            raw,
            base=base,
            boxes=boxes,
            targets=targets,
            candidate=candidate,
            cfg=cfg,
            scale=scale,
            baseline=baseline,
        )
        result.update(
            optimizer_updates=update,
            grad_norm_preclip=last_preclip_norm,
            grad_norm_postclip=last_postclip_norm,
        )
        history.append(result)
        summary = {
            "regime": name,
            "optimizer_updates": update,
            "loss": result["metrics"]["loss"],
            "keep": result["metrics"]["keep_component"],
            "drop": result["metrics"]["drop_component"],
            "keep_safe": result["metrics"]["keep_safe_instances"],
            "drop_safe": result["metrics"]["drop_safe_rows"],
            "gated_negative": result["metrics"][
                "deployed_category_negative_queries"
            ],
            "residual_abs_max": result["residual"]["abs_max"],
            "gate_status": result["gate_status"],
        }
        print(json.dumps(summary, sort_keys=True), flush=True)
        progress_handle.write(json.dumps(summary, sort_keys=True) + "\n")
        progress_handle.flush()

    record(0)
    for update in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        centered, residual = _bounded_residual(
            raw, float(cfg.stage_b_data_driven_patch_residual_limit)
        )
        score = base + float(scale) * residual
        contract = _patch_contract(
            score,
            boxes,
            targets,
            candidate,
            cfg,
        )
        if update > 1:
            sweep_record(update - 1, contract, score)
        loss = contract["loss"]
        if not bool(torch.isfinite(loss).item()):
            raise RealModelAuditError(
                f"free-table regime {name!r} produced a non-finite loss"
            )
        loss.backward()
        last_preclip_norm = _finite_gradient_norm(raw)
        if clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_([raw], float(clip_max_norm))
        last_postclip_norm = _finite_gradient_norm(raw)
        optimizer.step()
        del centered, residual, score, contract, loss
        if update in log_updates:
            record(update)

    final = history[-1]
    with torch.no_grad():
        final_score = base + float(scale) * _bounded_residual(
            raw, float(cfg.stage_b_data_driven_patch_residual_limit)
        )[1]
        final_contract = _patch_contract(
            final_score,
            boxes,
            targets,
            candidate,
            cfg,
        )
    sweep_record(steps, final_contract, final_score)
    passing_updates = [
        int(record["optimizer_updates"])
        for record in gate_sweep
        if record["gate_status"] == "passed"
    ]
    best_sweep = min(
        gate_sweep,
        key=lambda record: (
            -int(record["passed_checks"]),
            float(record["deployed_category_negative_queries"]),
            float(record["loss"]),
        ),
    )
    artifact = None
    if best_passing_raw is not None:
        best_full = min(
            passing_full_records,
            key=lambda record: (
                record["metrics"]["deployed_category_negative_queries"],
                record["metrics"]["loss"],
            ),
        )
        with artifact_path.open("xb") as handle:
            torch.save(
                {
                    "schema": (
                        "pivot.stageb.data_driven.patch_free_table_best_raw/v1"
                    ),
                    "regime": name,
                    "optimizer_updates": best_full["optimizer_updates"],
                    "raw": best_passing_raw,
                },
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
        artifact = {
            "path": str(artifact_path.resolve()),
            "sha256": _sha256(artifact_path),
            "optimizer_updates": best_full["optimizer_updates"],
            "raw_tensor_sha256": best_full["raw_tensor_sha256"],
        }
    return {
        "spec": dict(spec),
        "status": "passed" if passing_updates else "failed",
        "final_status": final["gate_status"],
        "strict_witness_status": final["strict_witness_status"],
        "first_passing_update": passing_updates[0] if passing_updates else None,
        "last_passing_update": passing_updates[-1] if passing_updates else None,
        "passing_updates": passing_updates,
        "best_gate_sweep_record": best_sweep,
        "passing_full_records": passing_full_records,
        "best_passing_raw_artifact": artifact,
        "gate_sweep": gate_sweep,
        "history": history,
        "final_raw_tensor_sha256": _tensor_sha256([raw]),
        "final": final,
    }


def run_audit(
    *,
    device_name: str,
    seed: int,
    regime_names: Sequence[str],
    progress_path: Path,
) -> dict[str, Any]:
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RealModelAuditError("free-table audit requires CUDA")
    unknown = sorted(set(regime_names) - set(REGIMES))
    if unknown or not regime_names:
        raise RealModelAuditError(f"unknown or empty free-table regimes: {unknown}")
    if len(set(regime_names)) != len(regime_names):
        raise RealModelAuditError("free-table regimes must be unique")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    _seed_everything(seed)
    binding = AUDIT_VARIANTS["raw_centered"]
    cfg = SLConfig.fromfile(str(binding["config"]))
    if not (
        math.isclose(float(cfg.stage_b_data_driven_patch_lr), 3e-4)
        and math.isclose(float(cfg.weight_decay), 1e-4)
        and math.isclose(float(cfg.clip_max_norm), 0.1)
        and math.isclose(
            float(cfg.stage_b_data_driven_patch_residual_limit), 0.25
        )
        and bool(cfg.stage_b_data_driven_patch_residual_center_raw)
    ):
        raise RealModelAuditError("sealed v19 optimizer or residual contract drifted")
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
        raise RealModelAuditError("free-table audit U0 model surface drifted")
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
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("x", encoding="utf-8", newline="\n") as handle:
        regimes = {
            name: _run_regime(
                name,
                REGIMES[name],
                base=base,
                boxes=boxes,
                targets=targets,
                candidate=candidate,
                cfg=cfg,
                scale=scale,
                baseline=baseline,
                progress_handle=handle,
                artifact_path=progress_path.with_name(
                    f"{progress_path.stem}.{name}.best_passing_raw.pth"
                ),
            )
            for name in regime_names
        }

    formal_100 = regimes.get("formal_adamw_lr3e4_clip01_u100")
    formal_1000 = regimes.get("formal_adamw_lr3e4_clip01_u1000")
    diagnostic = regimes.get("diagnostic_adam_lr1e2_noclip_u500")
    return {
        "schema": "pivot.stageb.data_driven.patch_free_table_overfit64/v2",
        "status": "completed",
        "interpretation": {
            "formal_hyperparameter_u100_free_table_meets_patch_gate": (
                None if formal_100 is None else formal_100["status"] == "passed"
            ),
            "formal_hyperparameter_u1000_free_table_meets_patch_gate": (
                None if formal_1000 is None else formal_1000["status"] == "passed"
            ),
            "diagnostic_free_table_meets_patch_gate": (
                None if diagnostic is None else diagnostic["status"] == "passed"
            ),
            "tested_schedules_found_free_table_patch_gate_witness": any(
                result["status"] == "passed" for result in regimes.values()
            ),
            "free_table_does_not_reproduce_v19_mlp_optimizer_dynamics": True,
            "shared_scorer_features_or_capacity_are_not_tested_here": True,
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
            "same_64_unique_images_as_strict_o64": True,
            "all_900_queries_are_candidates": True,
            "model_is_deleted_before_free_table_optimization": True,
            "only_one_zero_initialized_64_by_900_raw_table_is_trainable": True,
            "v19_post_raw_functional_form_is_used_in_fp32": True,
            "exact_deployment_gate_category_patch_loss_is_used": True,
            "formal_regimes_reuse_config_adamw_lr_weight_decay_and_branch_clip": True,
            "weight_decay_applies_to_free_logits_not_v19_mlp_weights": True,
            "no_teacher_or_winner_score_is_used": True,
        },
        "selection": {
            "seed": seed,
            "members": identities,
        },
        "base_patch_logit_scale": scale,
        "baseline": baseline,
        "regimes": regimes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=tuple(REGIMES),
        default=tuple(REGIMES),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    output_json = args.output_json.expanduser()
    if not output_json.is_absolute():
        output_json = Path.cwd() / output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_json.with_suffix(".progress.jsonl")
    result = run_audit(
        device_name=args.device,
        seed=args.seed,
        regime_names=args.regimes,
        progress_path=progress_path,
    )
    _write_json_exclusive(output_json, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "interpretation": result["interpretation"],
                "output_json": str(output_json),
                "progress_jsonl": str(progress_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
