# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import copy
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, List, Mapping, Optional, Tuple

from util.utils import to_device
import numpy as np
import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.cocogrounding_eval import CocoGroundingEvaluator

from datasets.panoptic_eval import PanopticEvaluator
from util.misc import NestedTensor
from util.stage_b_table_b_contract import build_confidence_ablation_eligible


class GracefulTrainingExit(Exception):
    """Raised after an interrupt checkpoint has been written."""


def _set_stage_b_v7_training_mode(model: torch.nn.Module) -> None:
    """Keep the frozen proposal tower deterministic while training the verifier."""
    root = model.module if hasattr(model, "module") else model
    verifier = getattr(root, "stage_b_verifier", None)
    if verifier is None:
        raise RuntimeError("stage_b_v7 training requires model.stage_b_verifier")

    # requires_grad=False does not disable Dropout/DropPath. Stage A is a frozen
    # proposal generator, so its feature and candidate distribution must match
    # evaluation exactly.
    root.eval()
    verifier.train(True)
    verifier_bert = getattr(verifier, "bert", None)
    if verifier_bert is not None:
        verifier_bert.eval()


def _set_stage_b_v11_training_mode(model: torch.nn.Module) -> None:
    """Train only the fixed-box scorer; keep every Stage-A module deterministic."""
    root = model.module if hasattr(model, "module") else model
    scorer = getattr(root, "stage_b_fixed_text_scorer", None)
    if scorer is None:
        raise RuntimeError(
            "stage_b_v11_fixed_text training requires model.stage_b_fixed_text_scorer"
        )
    root.eval()
    scorer.train(True)


def _set_stage_b_legacy_global_gate_training_mode(model: torch.nn.Module) -> None:
    """Train only the legacy confidence gate on a deterministic frozen model."""
    root = model.module if hasattr(model, "module") else model
    gate = getattr(root, "stage_b_legacy_global_gate", None)
    if gate is None:
        raise RuntimeError(
            "stage_b_legacy_global_gate training requires "
            "model.stage_b_legacy_global_gate"
        )
    root.eval()
    gate.train(True)


def _stage_b_gdino_adapter_train_mode(value) -> str:
    from models.GroundingDINO.stage_b_gdino_score_adapter import (
        stage_b_gdino_adapter_train_mode_code,
    )

    mode = str(value).strip()
    stage_b_gdino_adapter_train_mode_code(mode)
    return mode


def _set_stage_b_gdino_adapter_training_mode(
    model: torch.nn.Module,
    train_mode: str = "joint",
) -> None:
    """Keep pure GroundingDINO deterministic while training only its adapter."""
    root = model.module if hasattr(model, "module") else model
    adapter = getattr(root, "stage_b_gdino_score_adapter", None)
    if adapter is None:
        raise RuntimeError(
            "stage_b_gdino_score_adapter training requires the adapter module"
        )
    train_mode = _stage_b_gdino_adapter_train_mode(train_mode)
    root.eval()
    adapter.train(True)
    rank_modules = (adapter.rank_norm, adapter.rank_trunk, adapter.rank_output)
    confidence_modules = (
        adapter.confidence_norm,
        adapter.confidence_trunk,
        adapter.confidence_gate,
    )
    for module in rank_modules:
        module.train(train_mode in {"rank_only", "joint"})
    for module in confidence_modules:
        module.train(train_mode in {"confidence_only", "joint"})


def _set_stage_b_u0_patch_rank_training_mode(model: torch.nn.Module) -> None:
    """Train U0 patch-specific modules without changing the shared b58 backbone."""
    root = model.module if hasattr(model, "module") else model
    adapter = getattr(root, "stage_b_u0_patch_rank_adapter", None)
    patch_encoder = getattr(root, "patch_encoder", None)
    query_projection = getattr(root, "query_proj_for_patch", None)
    if adapter is None or patch_encoder is None or query_projection is None:
        raise RuntimeError("Stage-B U0 requires its residual and patch projection modules")
    if getattr(patch_encoder, "backbone", None) is not root.backbone:
        raise RuntimeError("Stage-B U0 patch encoder must share the frozen main backbone")
    root.eval()
    adapter.train(True)
    patch_encoder.input_proj.train(True)
    patch_encoder.norm.train(True)
    query_projection.train(True)


def _set_stage_b_u0_gate_aligned_d10_training_mode(
    model: torch.nn.Module,
) -> None:
    """Keep every selector frozen/eval while training D10 projections."""
    root = model.module if hasattr(model, "module") else model
    adapter = getattr(root, "stage_b_u0_patch_rank_adapter", None)
    score_adapter = getattr(root, "stage_b_gdino_score_adapter", None)
    patch_encoder = getattr(root, "patch_encoder", None)
    query_projection = getattr(root, "query_proj_for_patch", None)
    if any(
        module is None
        for module in (adapter, score_adapter, patch_encoder, query_projection)
    ):
        raise RuntimeError("D10 requires U0, R100/P50, and patch projections")
    if getattr(patch_encoder, "backbone", None) is not root.backbone:
        raise RuntimeError("D10 patch encoder must share the frozen backbone")
    root.eval()
    patch_encoder.input_proj.train(True)
    patch_encoder.norm.train(True)
    query_projection.train(True)
    if adapter.training or score_adapter.training or root.backbone.training:
        raise RuntimeError("a frozen D10 selector or backbone remains in train mode")


def _set_stage_b_u0_gate_aligned_d11_training_mode(
    model: torch.nn.Module,
) -> None:
    """Keep Gap2/P50/U0 deterministic while tuning only R100's output."""
    root = model.module if hasattr(model, "module") else model
    u0_adapter = getattr(root, "stage_b_u0_patch_rank_adapter", None)
    score_adapter = getattr(root, "stage_b_gdino_score_adapter", None)
    patch_encoder = getattr(root, "patch_encoder", None)
    if any(module is None for module in (u0_adapter, score_adapter, patch_encoder)):
        raise RuntimeError("D11 requires U0, R100/P50, and the D9 patch branch")
    root.eval()
    score_adapter.rank_output.train(True)
    frozen_modules = (
        root.backbone,
        patch_encoder,
        u0_adapter,
        score_adapter.rank_norm,
        score_adapter.rank_trunk,
        score_adapter.confidence_norm,
        score_adapter.confidence_trunk,
        score_adapter.confidence_gate,
    )
    if any(module.training for module in frozen_modules):
        raise RuntimeError("a frozen D11 module remains in train mode")


def _set_stage_b_u0_gate_aligned_d12_training_mode(
    model: torch.nn.Module,
) -> None:
    """Train only D12's residual; keep its R100 teacher and Gap2 gate fixed."""
    root = model.module if hasattr(model, "module") else model
    d12 = getattr(root, "stage_b_u0_gate_aligned_rank_residual", None)
    u0_adapter = getattr(root, "stage_b_u0_patch_rank_adapter", None)
    score_adapter = getattr(root, "stage_b_gdino_score_adapter", None)
    patch_encoder = getattr(root, "patch_encoder", None)
    if any(module is None for module in (d12, u0_adapter, score_adapter, patch_encoder)):
        raise RuntimeError("D12 training modules are incomplete")
    root.eval()
    d12.train(True)
    frozen_modules = (
        root.backbone,
        patch_encoder,
        u0_adapter,
        score_adapter,
    )
    if any(module.training for module in frozen_modules):
        raise RuntimeError("a frozen D12 teacher or gate remains in train mode")


def _set_stage_b_u0_gate_aligned_d13_training_mode(
    model: torch.nn.Module,
) -> None:
    """Train only D13 while keeping every patch/text teacher deterministic."""
    root = model.module if hasattr(model, "module") else model
    d13 = getattr(root, "stage_b_u0_gate_aligned_patch_residual", None)
    u0_adapter = getattr(root, "stage_b_u0_patch_rank_adapter", None)
    score_adapter = getattr(root, "stage_b_gdino_score_adapter", None)
    patch_encoder = getattr(root, "patch_encoder", None)
    query_projection = getattr(root, "query_proj_for_patch", None)
    if any(
        module is None
        for module in (d13, u0_adapter, score_adapter, patch_encoder, query_projection)
    ):
        raise RuntimeError("D13 training modules are incomplete")
    root.eval()
    d13.train(True)
    frozen_modules = (
        root.backbone,
        patch_encoder,
        query_projection,
        u0_adapter,
        score_adapter,
    )
    if any(module.training for module in frozen_modules):
        raise RuntimeError("a frozen D13 teacher or patch projection remains in train mode")


def _set_stage_b_native_patch_category_training_mode(
    model: torch.nn.Module,
) -> None:
    """Train only patch projections on deterministic native b58 features."""
    root = model.module if hasattr(model, "module") else model
    patch_encoder = getattr(root, "patch_encoder", None)
    query_projection = getattr(root, "query_proj_for_patch", None)
    if patch_encoder is None or query_projection is None:
        raise RuntimeError(
            "native patch-category training requires patch projection modules"
        )
    if getattr(patch_encoder, "backbone", None) is not root.backbone:
        raise RuntimeError(
            "native patch-category patch encoder must share the frozen b58 backbone"
        )
    root.eval()
    patch_encoder.input_proj.train(True)
    patch_encoder.norm.train(True)
    query_projection.train(True)
    frozen_modules = (root, root.backbone, root.bert, root.transformer)
    if any(module.training for module in frozen_modules):
        raise RuntimeError(
            "a frozen native patch-category feature generator remains in train mode"
        )
    if patch_encoder.backbone.training:
        raise RuntimeError(
            "the shared native patch-category backbone remains in train mode"
        )


def _set_stage_b_data_driven_training_mode(
    model: torch.nn.Module,
    train_mode: str,
) -> None:
    """Train only the declared score phase on deterministic b58 features."""
    from models.GroundingDINO.stage_b_data_driven_score import (
        normalize_data_driven_train_mode,
    )

    root = model.module if hasattr(model, "module") else model
    heads = getattr(root, "stage_b_data_driven_score_heads", None)
    patch_encoder = getattr(root, "patch_encoder", None)
    query_projection = getattr(root, "query_proj_for_patch", None)
    patch_residual = getattr(
        root, "stage_b_data_driven_patch_residual", None
    )
    if heads is None or patch_encoder is None or query_projection is None:
        raise RuntimeError(
            "data-driven training requires score heads and patch projection modules"
        )
    if getattr(patch_encoder, "backbone", None) is not root.backbone:
        raise RuntimeError(
            "data-driven patch encoder must share the frozen b58 backbone"
        )
    mode = normalize_data_driven_train_mode(train_mode)
    if bool(getattr(heads, "category_gate", False)):
        raise RuntimeError(
            "the data-driven category gate is inference-only and must be disabled "
            "in every training phase"
        )

    # requires_grad=False does not disable BERT dropout, fusion DropPath, or
    # Swin stochastic depth. Keep the frozen feature generators in exact eval
    # mode and re-enable only modules owned by the active phase.
    root.eval()
    if mode == "rank_patch_only":
        heads.rank_branch.train(True)
        heads.confidence_branch.eval()
        heads.confidence_gate.eval()
        if patch_residual is None:
            patch_encoder.input_proj.train(True)
            patch_encoder.norm.train(True)
            query_projection.train(True)
        else:
            patch_residual.train(True)
    else:
        heads.rank_branch.eval()
        heads.confidence_branch.train(True)
        heads.confidence_gate.train(True)
    frozen_modules = (root, root.backbone, root.bert, root.transformer)
    if any(module.training for module in frozen_modules):
        raise RuntimeError("a frozen data-driven feature generator remains in train mode")
    if patch_encoder.backbone.training:
        raise RuntimeError("the shared data-driven patch backbone remains in train mode")


def _grad_l2_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """Return the L2 norm of the gradients currently on ``parameters``."""
    grad_norms = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        if grad.is_sparse:
            grad = grad.coalesce().values()
        grad_norms.append(torch.linalg.vector_norm(grad.float(), ord=2))
    if not grad_norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(grad_norms), ord=2).item())


def _clip_stage_b_data_driven_optimizer_grad_norms(
    optimizer: torch.optim.Optimizer,
    max_norm: float,
    *,
    train_mode: str,
) -> dict:
    """Clip each declared data-driven optimizer branch independently."""
    from models.GroundingDINO.stage_b_data_driven_score import (
        normalize_data_driven_train_mode,
    )

    mode = normalize_data_driven_train_mode(train_mode)
    expected_labels = (
        ("rank", "patch") if mode == "rank_patch_only" else ("confidence",)
    )
    grouped_parameters = {}
    parameter_owners = {}
    for group_index, group in enumerate(optimizer.param_groups):
        label = group.get("stage_b_data_driven_branch")
        if label not in expected_labels:
            raise RuntimeError(
                "per_optimizer_branch_v1 found an unlabeled or inactive optimizer "
                f"group at index {group_index}: {label!r}"
            )
        if label in grouped_parameters:
            raise RuntimeError(
                f"per_optimizer_branch_v1 requires exactly one {label!r} group"
            )
        parameters = list(group.get("params", ()))
        if not parameters:
            raise RuntimeError(
                f"per_optimizer_branch_v1 found an empty {label!r} optimizer group"
            )
        for parameter in parameters:
            parameter_id = id(parameter)
            if parameter_id in parameter_owners:
                owner = parameter_owners[parameter_id]
                raise RuntimeError(
                    "per_optimizer_branch_v1 optimizer parameter is duplicated: "
                    f"{owner}/{label}"
                )
            parameter_owners[parameter_id] = label
        grouped_parameters[label] = parameters
    if tuple(grouped_parameters) != expected_labels:
        raise RuntimeError(
            "per_optimizer_branch_v1 optimizer labels drifted: "
            f"expected={expected_labels}, observed={tuple(grouped_parameters)}"
        )

    max_norm = float(max_norm)
    if not math.isfinite(max_norm) or max_norm <= 0.0:
        raise ValueError(
            "per_optimizer_branch_v1 requires a finite positive clip max norm"
        )
    stats = {}
    for label in expected_labels:
        parameters = [
            parameter
            for parameter in grouped_parameters[label]
            if parameter.requires_grad and parameter.grad is not None
        ]
        stats[f"grad_norm_data_driven_{label}_preclip"] = _grad_l2_norm(
            parameters
        )
        if parameters:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        stats[f"grad_norm_data_driven_{label}_postclip"] = _grad_l2_norm(
            parameters
        )
    return stats


def _clip_stage_b_v15_grad_norms(
    model: torch.nn.Module,
    max_norm: float,
) -> dict:
    """Clip the v15 rank and confidence branches independently."""
    root = model.module if hasattr(model, "module") else model
    rank_parameters = []
    confidence_parameters = []
    for name, parameter in root.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        if name.startswith("stage_b_fixed_text_scorer.validity_head."):
            confidence_parameters.append(parameter)
        else:
            rank_parameters.append(parameter)

    if not confidence_parameters:
        raise RuntimeError(
            "stage_b_v15_separate_grad_clip requires trainable validity_head "
            "parameters with gradients"
        )

    stats = {}
    for branch, parameters in (
        ("rank", rank_parameters),
        ("confidence", confidence_parameters),
    ):
        stats[f"grad_norm_{branch}_preclip"] = _grad_l2_norm(parameters)
        if parameters:
            torch.nn.utils.clip_grad_norm_(parameters, float(max_norm))
        stats[f"grad_norm_{branch}_postclip"] = _grad_l2_norm(parameters)
    return stats


def _clip_stage_b_dense_duty_grad_norms(
    model: torch.nn.Module,
    max_norm: float,
) -> dict:
    root = model.module if hasattr(model, "module") else model
    scorer = getattr(root, "stage_b_fixed_text_scorer", None)
    adapter = getattr(scorer, "confidence_adapter", None)
    head_contract = str(
        getattr(
            adapter,
            "head_gradient_contract",
            "shared_token_veto_global_absolute_v1",
        )
    ).strip()
    parameters = [
        parameter
        for parameter in root.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    preclip = _grad_l2_norm(parameters)
    stats = {
        "grad_norm_dense_duty_active_preclip": preclip,
        "grad_tensor_count_dense_duty_active": float(len(parameters)),
    }
    if head_contract == "shared_token_veto_global_absolute_v1":
        if parameters:
            torch.nn.utils.clip_grad_norm_(parameters, float(max_norm))
    elif head_contract in {
        "split_token_veto_global_absolute_v2",
        "split_token_veto_global_absolute_joint_clip_v3",
        "split_token_veto_global_trust_veto_v4",
        "split_token_veto_deployed_router_global_absolute_v5",
        "split_token_veto_candidate_absolute_sample_calibrator_v6",
        "split_token_veto_fulltext_global_absolute_v7",
        "split_token_veto_local_candidate_global_absolute_v8",
        "split_token_veto_deployment_owned_global_absolute_v9",
        "split_token_veto_deployment_owned_query_global_absolute_v10",
        "split_token_veto_deployment_owned_query_veto_global_absolute_v11",
    }:
        candidate_sample_contract = head_contract == (
            "split_token_veto_candidate_absolute_sample_calibrator_v6"
        )
        required_owner_apis = ["token_veto_parameters"]
        required_owner_apis.extend(
            ["candidate_absolute_parameters", "sample_calibrator_parameters"]
            if candidate_sample_contract
            else ["global_absolute_parameters"]
        )
        if (
            head_contract
            == "split_token_veto_deployed_router_global_absolute_v5"
        ):
            required_owner_apis.append("deployed_router_parameters")
        if scorer is None or not all(
            hasattr(scorer, name) for name in required_owner_apis
        ):
            raise RuntimeError(
                "split confidence heads require explicit parameter ownership APIs"
            )
        owned = {"token_veto": tuple(scorer.token_veto_parameters())}
        if (
            head_contract
            == "split_token_veto_deployed_router_global_absolute_v5"
        ):
            owned["deployed_router"] = tuple(
                scorer.deployed_router_parameters()
            )
        if candidate_sample_contract:
            owned["candidate_absolute"] = tuple(
                scorer.candidate_absolute_parameters()
            )
            owned["sample_calibrator"] = tuple(
                scorer.sample_calibrator_parameters()
            )
        else:
            owned["global_absolute"] = tuple(scorer.global_absolute_parameters())
        owner_ids = {
            label: {id(parameter) for parameter in group}
            for label, group in owned.items()
        }
        active_ids = {
            id(parameter)
            for parameter in root.parameters()
            if parameter.requires_grad
        }
        owner_labels = tuple(owner_ids)
        owners_overlap = any(
            bool(owner_ids[left] & owner_ids[right])
            for index, left in enumerate(owner_labels)
            for right in owner_labels[index + 1 :]
        )
        owned_union = set().union(*owner_ids.values())
        if not all(owner_ids.values()) or owners_overlap or owned_union != active_ids:
            raise RuntimeError(
                "split confidence-head clipping found empty, overlapping, or "
                "incomplete parameter ownership"
            )
        live_by_owner = {
            label: [
                parameter
                for parameter in group
                if parameter.requires_grad and parameter.grad is not None
            ]
            for label, group in owned.items()
        }
        live_by_global_subowner = {}
        if head_contract == "split_token_veto_global_trust_veto_v4":
            if not all(
                hasattr(scorer, name)
                for name in ("global_trust_parameters", "global_veto_parameters")
            ):
                raise RuntimeError(
                    "global trust/veto clipping requires explicit subowner APIs"
                )
            global_subowned = {
                "global_trust": tuple(scorer.global_trust_parameters()),
                "global_veto": tuple(scorer.global_veto_parameters()),
            }
            global_subowner_ids = {
                label: {id(parameter) for parameter in group}
                for label, group in global_subowned.items()
            }
            if (
                not all(global_subowner_ids.values())
                or global_subowner_ids["global_trust"]
                & global_subowner_ids["global_veto"]
                or global_subowner_ids["global_trust"]
                | global_subowner_ids["global_veto"]
                != owner_ids["global_absolute"]
            ):
                raise RuntimeError(
                    "global trust/veto clipping found empty, overlapping, or "
                    "incomplete parameter ownership"
                )
            live_by_global_subowner = {
                label: [
                    parameter
                    for parameter in group
                    if parameter.requires_grad and parameter.grad is not None
                ]
                for label, group in global_subowned.items()
            }
        for label, live in live_by_owner.items():
            stats[f"grad_norm_dense_duty_{label}_preclip"] = _grad_l2_norm(live)
            stats[f"grad_tensor_count_dense_duty_{label}"] = float(len(live))
        for label, live in live_by_global_subowner.items():
            stats[f"grad_norm_dense_duty_{label}_preclip"] = _grad_l2_norm(live)
            stats[f"grad_tensor_count_dense_duty_{label}"] = float(len(live))

        if head_contract in {
            "split_token_veto_global_absolute_v2",
            "split_token_veto_global_trust_veto_v4",
            "split_token_veto_deployed_router_global_absolute_v5",
            "split_token_veto_candidate_absolute_sample_calibrator_v6",
            "split_token_veto_fulltext_global_absolute_v7",
            "split_token_veto_local_candidate_global_absolute_v8",
            "split_token_veto_deployment_owned_global_absolute_v9",
            "split_token_veto_deployment_owned_query_global_absolute_v10",
            "split_token_veto_deployment_owned_query_veto_global_absolute_v11",
        }:
            # Every independently supervised owner receives the full max norm.
            for live in live_by_owner.values():
                if live:
                    torch.nn.utils.clip_grad_norm_(live, float(max_norm))
        else:
            joint_live = [
                parameter
                for live in live_by_owner.values()
                for parameter in live
            ]
            if joint_live:
                torch.nn.utils.clip_grad_norm_(joint_live, float(max_norm))

        for label, live in live_by_owner.items():
            stats[f"grad_norm_dense_duty_{label}_postclip"] = _grad_l2_norm(live)
        for label, live in live_by_global_subowner.items():
            stats[f"grad_norm_dense_duty_{label}_postclip"] = _grad_l2_norm(live)
    else:
        raise RuntimeError(
            f"unknown confidence head-gradient contract: {head_contract!r}"
        )
    active_postclip = _grad_l2_norm(parameters)
    stats["grad_norm_dense_duty_active_postclip"] = active_postclip
    clip_contract_owner_labels = None
    if head_contract == "split_token_veto_candidate_absolute_sample_calibrator_v6":
        clip_contract_owner_labels = (
            "token_veto",
            "candidate_absolute",
            "sample_calibrator",
        )
    elif head_contract in {
        "split_token_veto_fulltext_global_absolute_v7",
        "split_token_veto_local_candidate_global_absolute_v8",
        "split_token_veto_deployment_owned_global_absolute_v9",
        "split_token_veto_deployment_owned_query_global_absolute_v10",
        "split_token_veto_deployment_owned_query_veto_global_absolute_v11",
    }:
        clip_contract_owner_labels = ("token_veto", "global_absolute")
    if clip_contract_owner_labels is not None:
        owner_labels = clip_contract_owner_labels
        owner_preclip = [
            float(stats[f"grad_norm_dense_duty_{label}_preclip"])
            for label in owner_labels
        ]
        owner_postclip = [
            float(stats[f"grad_norm_dense_duty_{label}_postclip"])
            for label in owner_labels
        ]
        if not hasattr(
            scorer, "expected_live_confidence_parameter_tensor_counts"
        ):
            raise RuntimeError(
                "sealed split heads require an explicit live gradient-tensor contract"
            )
        expected_live_counts = dict(
            scorer.expected_live_confidence_parameter_tensor_counts()
        )
        if set(expected_live_counts) != set(owner_labels) or any(
            not isinstance(value, int) or value <= 0
            for value in expected_live_counts.values()
        ):
            raise RuntimeError(
                "split-head live gradient-tensor contract is malformed"
            )
        observed_live_counts = {
            label: len(live_by_owner[label]) for label in owner_labels
        }
        pre_decomposition = math.sqrt(sum(value * value for value in owner_preclip))
        post_decomposition = math.sqrt(sum(value * value for value in owner_postclip))
        pre_residual = abs(float(preclip) - pre_decomposition)
        post_residual = abs(float(active_postclip) - post_decomposition)
        decomposition_tolerance = max(
            1.0e-6,
            1.0e-5 * max(1.0, abs(float(preclip)), abs(float(active_postclip))),
        )
        clip_tolerance = max(1.0e-6, 1.0e-5 * float(max_norm))
        owner_clip_residuals = [
            max(0.0, post - min(pre, float(max_norm)))
            for pre, post in zip(owner_preclip, owner_postclip)
        ]
        active_monotonic_residual = max(
            0.0, float(active_postclip) - float(preclip)
        )
        stats.update(
            {
                "dense_duty_clip_contract_checked": 1,
                "dense_duty_clip_contract_owner_count": len(owner_labels),
                "dense_duty_clip_contract_max_norm": float(max_norm),
                "dense_duty_clip_contract_tolerance": decomposition_tolerance,
                "dense_duty_clip_contract_pre_decomposition_residual": pre_residual,
                "dense_duty_clip_contract_post_decomposition_residual": post_residual,
                "dense_duty_clip_contract_owner_clip_residual": max(
                    owner_clip_residuals
                ),
                "dense_duty_clip_contract_active_monotonic_residual": (
                    active_monotonic_residual
                ),
                "dense_duty_clip_contract_live_tensor_count_violation": int(
                    observed_live_counts != expected_live_counts
                    or len(parameters) != sum(expected_live_counts.values())
                ),
                "dense_duty_clip_contract_pre_decomposition_violation": int(
                    pre_residual > decomposition_tolerance
                ),
                "dense_duty_clip_contract_post_decomposition_violation": int(
                    post_residual > decomposition_tolerance
                ),
                "dense_duty_clip_contract_owner_clip_violation": int(
                    max(owner_clip_residuals) > clip_tolerance
                ),
                "dense_duty_clip_contract_active_monotonic_violation": int(
                    active_monotonic_residual > clip_tolerance
                ),
            }
        )
        for label in owner_labels:
            stats[f"dense_duty_clip_contract_expected_{label}_tensor_count"] = (
                expected_live_counts[label]
            )
            stats[f"dense_duty_clip_contract_observed_{label}_tensor_count"] = (
                observed_live_counts[label]
            )
    return stats


def _record_stage_b_dense_duty_runtime_audit(
    args,
    device: torch.device,
    *,
    optimizer_step_boundary: bool,
    optimizer_step_succeeded: bool,
    branch_grad_norms: Mapping[str, float],
    amp_scale: Optional[float] = None,
) -> None:
    if not bool(getattr(args, "stage_b_dense_duty", False)):
        return
    existing = getattr(args, "stage_b_dense_duty_runtime_audit", None)
    audit = dict(existing) if isinstance(existing, Mapping) else {}
    audit.setdefault("schema", "pivot.stageb.dense_duty_runtime_audit/v1")
    audit.setdefault("optimizer_step_boundaries", 0)
    audit.setdefault("successful_optimizer_steps", 0)
    audit.setdefault("amp_skipped_optimizer_steps", 0)
    audit.setdefault("nonfinite_gradient_boundaries", 0)
    audit.setdefault("zero_gradient_successful_steps", 0)
    audit.setdefault("max_active_grad_norm_preclip", 0.0)
    if optimizer_step_boundary:
        audit["optimizer_step_boundaries"] += 1
        grad_norm = float(
            branch_grad_norms.get("grad_norm_dense_duty_active_preclip", 0.0)
        )
        audit["last_active_grad_norm_preclip"] = grad_norm
        audit["max_active_grad_norm_preclip"] = max(
            float(audit["max_active_grad_norm_preclip"]), grad_norm
        )
        if not math.isfinite(grad_norm):
            audit["nonfinite_gradient_boundaries"] += 1
        if optimizer_step_succeeded:
            audit["successful_optimizer_steps"] += 1
            if grad_norm == 0.0:
                audit["zero_gradient_successful_steps"] += 1
        else:
            audit["amp_skipped_optimizer_steps"] += 1
        for head in (
            "token_veto",
            "deployed_router",
            "global_absolute",
            "candidate_absolute",
            "sample_calibrator",
            "global_trust",
            "global_veto",
        ):
            key = f"grad_norm_dense_duty_{head}_preclip"
            if key not in branch_grad_norms:
                continue
            head_grad_norm = float(branch_grad_norms[key])
            audit[f"last_{head}_grad_norm_preclip"] = head_grad_norm
            audit[f"max_{head}_grad_norm_preclip"] = max(
                float(audit.get(f"max_{head}_grad_norm_preclip", 0.0)),
                head_grad_norm,
            )
            if not math.isfinite(head_grad_norm):
                counter = f"nonfinite_{head}_gradient_boundaries"
                audit[counter] = int(audit.get(counter, 0)) + 1
            if optimizer_step_succeeded and head_grad_norm == 0.0:
                counter = f"zero_{head}_gradient_successful_steps"
                audit[counter] = int(audit.get(counter, 0)) + 1
        if int(branch_grad_norms.get("dense_duty_clip_contract_checked", 0)) == 1:
            clip_owner_count = int(
                branch_grad_norms.get("dense_duty_clip_contract_owner_count", 0)
            )
            if clip_owner_count == 3:
                clip_owners = (
                    "token_veto",
                    "candidate_absolute",
                    "sample_calibrator",
                )
                audit["clip_contract_schema"] = (
                    "pivot.stageb.dense_duty_three_owner_clip_contract/v1"
                )
            elif clip_owner_count == 2:
                clip_owners = ("token_veto", "global_absolute")
                audit["clip_contract_schema"] = (
                    "pivot.stageb.dense_duty_two_owner_clip_contract/v1"
                )
            else:
                raise RuntimeError("dense-duty clip audit has an invalid owner count")
            if optimizer_step_succeeded:
                audit["clip_contract_checked_steps"] = int(
                    audit.get("clip_contract_checked_steps", 0)
                ) + 1
                counter_fields = {
                    "owner_clip_violation_steps": (
                        "dense_duty_clip_contract_owner_clip_violation"
                    ),
                    "active_pre_decomposition_violation_steps": (
                        "dense_duty_clip_contract_pre_decomposition_violation"
                    ),
                    "active_post_decomposition_violation_steps": (
                        "dense_duty_clip_contract_post_decomposition_violation"
                    ),
                    "live_tensor_count_violation_steps": (
                        "dense_duty_clip_contract_live_tensor_count_violation"
                    ),
                    "active_monotonic_violation_steps": (
                        "dense_duty_clip_contract_active_monotonic_violation"
                    ),
                }
                for audit_key, stats_key in counter_fields.items():
                    audit[audit_key] = int(audit.get(audit_key, 0)) + int(
                        branch_grad_norms.get(stats_key, 0)
                    )
                residual_fields = {
                    "max_active_pre_decomposition_residual": (
                        "dense_duty_clip_contract_pre_decomposition_residual"
                    ),
                    "max_active_post_decomposition_residual": (
                        "dense_duty_clip_contract_post_decomposition_residual"
                    ),
                    "max_owner_clip_residual": (
                        "dense_duty_clip_contract_owner_clip_residual"
                    ),
                    "max_active_monotonic_residual": (
                        "dense_duty_clip_contract_active_monotonic_residual"
                    ),
                }
                for audit_key, stats_key in residual_fields.items():
                    value = float(branch_grad_norms.get(stats_key, math.inf))
                    audit[audit_key] = max(float(audit.get(audit_key, 0.0)), value)
                audit["clip_contract_tolerance"] = float(
                    branch_grad_norms[
                        "dense_duty_clip_contract_tolerance"
                    ]
                )
                audit["clip_contract_max_norm"] = float(
                    branch_grad_norms["dense_duty_clip_contract_max_norm"]
                )
                for owner in clip_owners:
                    expected_key = (
                        f"dense_duty_clip_contract_expected_{owner}_tensor_count"
                    )
                    observed_key = (
                        f"dense_duty_clip_contract_observed_{owner}_tensor_count"
                    )
                    audit[f"expected_{owner}_tensor_count"] = int(
                        branch_grad_norms[expected_key]
                    )
                    audit[f"last_observed_{owner}_tensor_count"] = int(
                        branch_grad_norms[observed_key]
                    )
    if amp_scale is not None and math.isfinite(float(amp_scale)):
        scale = float(amp_scale)
        audit["last_amp_scale"] = scale
        audit["min_amp_scale"] = min(
            float(audit.get("min_amp_scale", scale)), scale
        )
    if device.type == "cuda":
        current = {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        for key, value in current.items():
            audit[key] = max(int(audit.get(key, 0)), value)
    setattr(args, "stage_b_dense_duty_runtime_audit", audit)


def _clip_stage_b_gdino_adapter_grad_norms(
    model: torch.nn.Module,
    max_norm: float,
) -> dict:
    """Clip rank and confidence adapter parameters without cross-branch scaling."""
    root = model.module if hasattr(model, "module") else model
    adapter = getattr(root, "stage_b_gdino_score_adapter", None)
    if adapter is None:
        raise RuntimeError("missing stage_b_gdino_score_adapter for gradient clipping")
    stats = {}
    for branch, parameters in (
        ("gdino_rank", list(adapter.rank_parameters())),
        ("gdino_confidence", list(adapter.gate_parameters())),
    ):
        parameters = [
            parameter
            for parameter in parameters
            if parameter.requires_grad and parameter.grad is not None
        ]
        stats[f"grad_norm_{branch}_preclip"] = _grad_l2_norm(parameters)
        if parameters:
            torch.nn.utils.clip_grad_norm_(parameters, float(max_norm))
        stats[f"grad_norm_{branch}_postclip"] = _grad_l2_norm(parameters)
    return stats


def _clip_stage_b_u0_patch_rank_grad_norms(
    model: torch.nn.Module,
    max_norm: float,
) -> dict:
    root = model.module if hasattr(model, "module") else model
    adapter = getattr(root, "stage_b_u0_patch_rank_adapter", None)
    if adapter is None:
        raise RuntimeError("missing Stage-B U0 patch-rank adapter")
    direct_patch_gain = getattr(adapter, "direct_patch_gain", None)
    residual_ids = {
        id(parameter)
        for parameter in adapter.trainable_parameters()
        if parameter is not direct_patch_gain
    }
    residual = []
    gain = []
    patch_projection = []
    for parameter in root.parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        if parameter is direct_patch_gain:
            gain.append(parameter)
        elif id(parameter) in residual_ids:
            residual.append(parameter)
        else:
            patch_projection.append(parameter)
    stats = {}
    branches = [("u0_residual", residual)]
    if direct_patch_gain is not None:
        branches.append(("u1_direct_patch_gain", gain))
    branches.append(("u0_patch_projection", patch_projection))
    for branch, parameters in branches:
        stats[f"grad_norm_{branch}_preclip"] = _grad_l2_norm(parameters)
        if parameters:
            torch.nn.utils.clip_grad_norm_(parameters, float(max_norm))
        stats[f"grad_norm_{branch}_postclip"] = _grad_l2_norm(parameters)
    return stats


def _stage_b_v22_gradient_diagnostic(
    args,
    model: torch.nn.Module,
    loss_dict,
    weight_dict,
    *,
    step: int,
) -> dict:
    """Run the optional Table-D gradient probe without changing model grads."""
    interval = int(
        getattr(args, "stage_b_v22_gradient_diagnostic_interval", 0) or 0
    )
    if interval <= 0 or int(step) % interval != 0:
        return {}
    from models.GroundingDINO.stage_b_fixed_text_scorer import (
        normalize_stage_b_score_ownership,
    )
    from util.stage_b_task_gradients import (
        branch_isolation_report,
        gradient_conflict_report,
        weighted_stage_b_task_losses,
    )

    ownership = normalize_stage_b_score_ownership(
        getattr(args, "stage_b_v22_score_ownership", "")
    )
    if not ownership:
        raise RuntimeError(
            "stage_b_v22_gradient_diagnostic_interval requires an explicit "
            "stage_b_v22_score_ownership"
        )
    root = model.module if hasattr(model, "module") else model
    scorer = getattr(root, "stage_b_fixed_text_scorer", None)
    if scorer is None:
        raise RuntimeError("Stage-B v22 gradient diagnostic requires the fixed scorer")
    rank_loss, confidence_loss = weighted_stage_b_task_losses(
        loss_dict, weight_dict
    )
    named = tuple(
        (name, parameter)
        for name, parameter in scorer.named_parameters()
        if parameter.requires_grad
    )
    if ownership in {"shared_score", "shared_trunk_two_heads"}:
        report = gradient_conflict_report(rank_loss, confidence_loss, named)
        numeric = {
            "grad_cosine": report["cosine"],
            "grad_cosine_defined": float(report["cosine_defined"]),
            "grad_rank_norm": report["rank_norm"],
            "grad_confidence_norm": report["confidence_norm"],
            "grad_element_conflict_fraction": report[
                "element_conflict_fraction"
            ],
            "grad_tensor_conflict_fraction": report[
                "tensor_conflict_fraction"
            ],
            "grad_shared_parameter_count": report["shared_parameter_count"],
            "grad_shared_element_count": report["shared_element_count"],
        }
    else:
        rank_named = tuple(
            (name, parameter)
            for name, parameter in named
            if name.startswith("decoder.")
        )
        confidence_named = tuple(
            (name, parameter)
            for name, parameter in named
            if name.startswith("validity_head.")
        )
        report = branch_isolation_report(
            rank_loss,
            confidence_loss,
            rank_named,
            confidence_named,
        )
        numeric = {
            "branch_isolation_pass": float(report["passed"]),
            "branch_rank_parameter_count": report["rank_parameter_count"],
            "branch_confidence_parameter_count": report[
                "confidence_parameter_count"
            ],
        }
    reference = next(iter(loss_dict.values()))
    return {
        f"stage_b_v22_{key}": torch.as_tensor(
            float(value), device=reference.device, dtype=torch.float32
        )
        for key, value in numeric.items()
    }


def _make_grad_scaler(enabled: bool):
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "GradScaler"):
        try:
            return amp_mod.GradScaler("cuda", enabled=enabled)
        except TypeError:
            try:
                return amp_mod.GradScaler(device_type="cuda", enabled=enabled)
            except TypeError:
                pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _sum_weighted_training_losses(loss_dict, weight_dict) -> torch.Tensor:
    """Build the backward objective without traversing disabled loss graphs."""
    terms = [
        loss_dict[key] * float(weight_dict[key])
        for key in loss_dict
        if key in weight_dict and float(weight_dict[key]) != 0.0
    ]
    if not terms:
        raise RuntimeError("training objective has no non-zero weighted loss")
    return sum(terms)


class _IteratorWithLen:
    def __init__(self, iterator, length: int) -> None:
        self.iterator = iterator
        self.length = max(0, int(length))

    def __iter__(self):
        return self.iterator

    def __len__(self) -> int:
        return self.length


def _merge_stage_b_packed_batches(batches):
    if not batches:
        raise ValueError("packed Stage-B forward requires at least one batch")
    samples_list = [batch[0] for batch in batches]
    if any(not isinstance(samples, NestedTensor) for samples in samples_list):
        raise TypeError("packed Stage-B batches must contain NestedTensor samples")
    channels = int(samples_list[0].tensors.shape[1])
    dtype = samples_list[0].tensors.dtype
    device = samples_list[0].tensors.device
    if any(
        samples.tensors.dim() != 4
        or int(samples.tensors.shape[1]) != channels
        or samples.tensors.dtype != dtype
        or samples.tensors.device != device
        for samples in samples_list
    ):
        raise ValueError("packed Stage-B image batches have incompatible tensors")
    total = sum(int(samples.tensors.shape[0]) for samples in samples_list)
    max_height = max(int(samples.tensors.shape[-2]) for samples in samples_list)
    max_width = max(int(samples.tensors.shape[-1]) for samples in samples_list)
    tensors = torch.zeros(
        (total, channels, max_height, max_width), dtype=dtype, device=device
    )
    mask = torch.ones(
        (total, max_height, max_width), dtype=torch.bool, device=device
    )
    targets = []
    offset = 0
    for samples, batch_targets in batches:
        count, _, height, width = samples.tensors.shape
        end = offset + int(count)
        tensors[offset:end, :, :height, :width].copy_(samples.tensors)
        if samples.mask is None:
            mask[offset:end, :height, :width] = False
        else:
            if tuple(samples.mask.shape) != (int(count), int(height), int(width)):
                raise ValueError("packed Stage-B sample mask has an invalid shape")
            mask[offset:end, :height, :width].copy_(samples.mask)
        targets.extend(list(batch_targets))
        offset = end
    return NestedTensor(tensors, mask), targets


class _PackedStageBDataLoader:
    """Pack consecutive logical batches without changing sampler/data RNG order."""

    def __init__(self, data_loader, pack_factor: int) -> None:
        self.data_loader = data_loader
        self.pack_factor = int(pack_factor)
        if self.pack_factor <= 1:
            raise ValueError("packed Stage-B forward requires pack_factor > 1")
        self.logical_length = len(data_loader)
        self.length = int(math.ceil(self.logical_length / self.pack_factor))

    def __iter__(self):
        iterator = iter(self.data_loader)
        while True:
            batches = []
            for _ in range(self.pack_factor):
                try:
                    batches.append(next(iterator))
                except StopIteration:
                    break
            if not batches:
                return
            yield _merge_stage_b_packed_batches(batches)

    def __len__(self) -> int:
        return self.length


_V52_CANDIDATE_ABSOLUTE_SAMPLE_CALIBRATOR_CONTRACT = (
    "split_token_veto_candidate_absolute_sample_calibrator_v6"
)
_V53_FULLTEXT_GLOBAL_ABSOLUTE_CONTRACT = (
    "split_token_veto_fulltext_global_absolute_v7"
)
_V55_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE_CONTRACT = (
    "split_token_veto_local_candidate_global_absolute_v8"
)
_V56_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_CONTRACT = (
    "split_token_veto_deployment_owned_global_absolute_v9"
)
_V59_DEPLOYMENT_OWNED_QUERY_GLOBAL_ABSOLUTE_CONTRACT = (
    "split_token_veto_deployment_owned_query_global_absolute_v10"
)
_V60_DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE_CONTRACT = (
    "split_token_veto_deployment_owned_query_veto_global_absolute_v11"
)
_CANDIDATE_AND_SAMPLE_CONFIDENCE_CONTRACTS = {
    _V52_CANDIDATE_ABSOLUTE_SAMPLE_CALIBRATOR_CONTRACT,
    _V53_FULLTEXT_GLOBAL_ABSOLUTE_CONTRACT,
    _V55_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE_CONTRACT,
    _V56_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_CONTRACT,
    _V59_DEPLOYMENT_OWNED_QUERY_GLOBAL_ABSOLUTE_CONTRACT,
    _V60_DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE_CONTRACT,
}


def _select_dense_duty_confidence_loss_logits(
    *,
    outputs: Mapping[str, torch.Tensor],
    candidate_logits: torch.Tensor,
    confidence_logits: Optional[torch.Tensor],
    head_gradient_contract: str,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Bind positive and TN losses to their declared confidence routes."""
    if confidence_logits is not None and confidence_logits.shape != candidate_logits.shape:
        raise RuntimeError(
            "Stage B v15 confidence logits must align with phrase logits"
        )
    normalized_contract = str(head_gradient_contract).strip().lower()
    if normalized_contract in _CANDIDATE_AND_SAMPLE_CONFIDENCE_CONTRACTS:
        candidate_confidence = outputs.get(
            "stage_b_dense_duty_confidence_base_logits"
        )
        if candidate_confidence is None:
            raise RuntimeError(
                "candidate-absolute/sample-calibrator training requires live "
                "candidate confidence logits"
            )
        if (
            not torch.is_tensor(candidate_confidence)
            or not candidate_confidence.is_floating_point()
            or tuple(candidate_confidence.shape) != tuple(candidate_logits.shape)
            or candidate_confidence.device != candidate_logits.device
        ):
            raise RuntimeError(
                "candidate confidence logits must be a floating tensor aligned "
                "with candidate logits"
            )
        if not bool(torch.isfinite(candidate_confidence).all().item()):
            raise RuntimeError("candidate confidence logits must be finite")
        return candidate_confidence[..., 0], candidate_confidence[..., 1]

    positive = confidence_logits[..., 0] if confidence_logits is not None else None
    negative = confidence_logits[..., 1] if confidence_logits is not None else None
    if normalized_contract != "split_token_veto_global_trust_veto_v4":
        return positive, negative

    positive_route = outputs.get("stage_b_dense_duty_positive_confidence_logits")
    negative_route = outputs.get("stage_b_dense_duty_negative_confidence_logits")
    if positive_route is None or negative_route is None:
        raise RuntimeError(
            "global trust/veto training requires both routed confidence outputs"
        )
    if (
        tuple(positive_route.shape) != tuple(candidate_logits.shape)
        or tuple(negative_route.shape) != tuple(candidate_logits.shape)
        or confidence_logits is None
    ):
        raise RuntimeError(
            "global trust/veto confidence routes must align with the deployed "
            "confidence logits"
        )
    if confidence_logits.device.type != "cuda" and (
        not torch.equal(positive_route, confidence_logits)
        or not torch.equal(negative_route, confidence_logits)
    ):
        raise RuntimeError(
            "global trust/veto routes changed the deployed forward confidence value"
        )
    return positive_route[..., 0], negative_route[..., 1]


def _select_dense_duty_sample_confidence_logits(
    *,
    outputs: Mapping[str, torch.Tensor],
    candidate_logits: torch.Tensor,
    head_gradient_contract: str,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Select exact, unbroadcast logits for sample-global objectives."""
    if (
        str(head_gradient_contract).strip().lower()
        not in _CANDIDATE_AND_SAMPLE_CONFIDENCE_CONTRACTS
    ):
        return None, None
    sample_confidence = outputs.get(
        "stage_b_dense_duty_global_confidence_logits"
    )
    expected_shape = (int(candidate_logits.shape[0]), 2)
    if sample_confidence is None:
        raise RuntimeError(
            "candidate-absolute/sample-calibrator training requires exact "
            "sample-global confidence logits"
        )
    if (
        not torch.is_tensor(sample_confidence)
        or not sample_confidence.is_floating_point()
        or tuple(sample_confidence.shape) != expected_shape
        or sample_confidence.device != candidate_logits.device
    ):
        raise RuntimeError(
            "sample-global confidence logits must be a floating tensor with "
            f"shape {expected_shape}"
        )
    if not bool(torch.isfinite(sample_confidence).all().item()):
        raise RuntimeError("sample-global confidence logits must be finite")
    return sample_confidence[..., 0], sample_confidence[..., 1]


_RESIDUAL_POSITIVE_TRUST_CONTRACTS = {
    "net_total_confidence_delta_v1",
    "exact_frozen_rank_max_confidence_delta_v3",
}


def _select_dense_duty_positive_confidence_trust_logits(
    *,
    outputs: Mapping[str, torch.Tensor],
    candidate_logits: torch.Tensor,
    sample_positive_confidence_logits: Optional[torch.Tensor],
    decoupled_confidence: bool,
    positive_trust_contract: str,
    head_gradient_contract: str,
) -> Optional[torch.Tensor]:
    """Select the positive-tail gradient route without changing deployed scores."""
    if sample_positive_confidence_logits is None and not decoupled_confidence:
        return None

    trust_contract = str(positive_trust_contract).strip().lower()
    if trust_contract in _RESIDUAL_POSITIVE_TRUST_CONTRACTS:
        confidence_delta = outputs.get(
            "stage_b_dense_duty_confidence_delta_logits"
        )
        expected_shape = (int(candidate_logits.shape[0]), 2)
        if confidence_delta is None:
            raise RuntimeError(
                f"{trust_contract} positive trust requires the exact "
                "confidence-delta output"
            )
        if (
            not torch.is_tensor(confidence_delta)
            or not confidence_delta.is_floating_point()
            or tuple(confidence_delta.shape) != expected_shape
            or confidence_delta.device != candidate_logits.device
        ):
            raise RuntimeError(
                "positive trust confidence-delta logits must be a floating "
                f"tensor with shape {expected_shape} on the candidate-logit device"
            )
        if not bool(torch.isfinite(confidence_delta).all().item()):
            raise RuntimeError(
                "positive trust confidence-delta logits must be finite"
            )
        return confidence_delta[..., 0]

    if trust_contract == "absolute_global_confidence_logit_v2":
        if sample_positive_confidence_logits is not None:
            return sample_positive_confidence_logits
        trust_output_name = (
            "stage_b_dense_duty_positive_global_confidence_logits"
            if str(head_gradient_contract).strip().lower()
            == "split_token_veto_global_trust_veto_v4"
            else "stage_b_dense_duty_global_confidence_logits"
        )
        positive_global_confidence = outputs.get(trust_output_name)
        if positive_global_confidence is None:
            raise RuntimeError(
                "absolute confidence trust requires the independent "
                "global-confidence output"
            )
        return positive_global_confidence[..., 0]

    if trust_contract == "absolute_global_pool_logit_v4":
        pool_absolute = outputs.get(
            "stage_b_dense_duty_confidence_pool_absolute_logits"
        )
        expected_shape = (int(candidate_logits.shape[0]), 2)
        if pool_absolute is None:
            raise RuntimeError(
                "absolute global-pool trust requires the independent pool output"
            )
        if (
            not torch.is_tensor(pool_absolute)
            or not pool_absolute.is_floating_point()
            or tuple(pool_absolute.shape) != expected_shape
            or pool_absolute.device != candidate_logits.device
        ):
            raise RuntimeError(
                "absolute global-pool logits must be a floating tensor with "
                f"shape {expected_shape} on the candidate-logit device"
            )
        if not bool(torch.isfinite(pool_absolute).all().item()):
            raise RuntimeError("absolute global-pool logits must be finite")
        selected = pool_absolute[..., 0]
        deployment_owned_contract = (
            str(head_gradient_contract).strip().lower()
            == _V56_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_CONTRACT
        )
        if sample_positive_confidence_logits is not None and not torch.equal(
            selected.detach(), sample_positive_confidence_logits.detach()
        ):
            raise RuntimeError(
                "absolute global-pool trust changed the deployed confidence value"
            )
        if deployment_owned_contract:
            if sample_positive_confidence_logits is None:
                raise RuntimeError(
                    "V56 positive-q05 protection requires the true deployed "
                    "sample-global logit"
                )
            # Return the same tensor consumed by deployed inference/FPR95, not a
            # merely value-equal diagnostic alias of the pool output.
            return sample_positive_confidence_logits
        return selected

    if trust_contract == "pool_residual_v1":
        if sample_positive_confidence_logits is not None:
            return sample_positive_confidence_logits
        validity_gate = outputs.get("stage_b_v15_final_validity_gate_logits")
        if validity_gate is None:
            raise RuntimeError(
                "pool-residual confidence trust requires the validity-gate output"
            )
        return validity_gate[..., 0]

    raise RuntimeError(
        "unknown dense-duty positive trust contract: "
        f"{trust_contract!r}"
    )


def _call_fixed_text_criterion_in_logical_batches(
    criterion,
    *,
    logical_batch_size: int,
    **criterion_kwargs,
):
    candidate_logits = criterion_kwargs.get("candidate_logits")
    if not torch.is_tensor(candidate_logits) or candidate_logits.dim() < 1:
        raise ValueError("logical Stage-B criterion requires batched candidate_logits")
    batch_size = int(candidate_logits.shape[0])
    logical_batch_size = int(logical_batch_size)
    if logical_batch_size <= 0 or batch_size % logical_batch_size != 0:
        raise ValueError(
            "packed Stage-B forward batch must be divisible by logical_batch_size"
        )
    logical_batches = batch_size // logical_batch_size
    if logical_batches == 1:
        return criterion(**criterion_kwargs), 1

    defer_payload = getattr(criterion, "defer_tail_queue_payload", None)
    if not callable(defer_payload):
        raise RuntimeError(
            "packed Stage-B criterion requires deferred tail-queue commits"
        )
    loss_dicts = []
    for logical_index in range(logical_batches):
        start = logical_index * logical_batch_size
        end = start + logical_batch_size
        chunk = {}
        for name, value in criterion_kwargs.items():
            if value is None:
                chunk[name] = None
                continue
            if not torch.is_tensor(value) or value.dim() < 1:
                raise TypeError(
                    f"logical Stage-B criterion value {name!r} is not batched"
                )
            if int(value.shape[0]) != batch_size:
                raise ValueError(
                    f"logical Stage-B criterion value {name!r} has batch drift"
                )
            chunk[name] = value[start:end]
        loss_dicts.append(criterion(**chunk))
        defer_payload()

    keys = tuple(loss_dicts[0])
    if any(tuple(losses) != keys for losses in loss_dicts[1:]):
        raise RuntimeError("logical Stage-B criterion loss schema drifted")
    averaged = {
        name: sum(losses[name] for losses in loss_dicts) / float(logical_batches)
        for name in keys
    }
    return averaged, logical_batches


def _stage_b_token_edit_carrier_logits(outputs):
    base_logits = outputs.get("stage_b_dense_duty_confidence_base_logits")
    if base_logits is None:
        return None
    if (
        not torch.is_tensor(base_logits)
        or not base_logits.is_floating_point()
        or base_logits.dim() != 3
        or int(base_logits.shape[-1]) != 2
    ):
        raise RuntimeError(
            "Stage-B confidence base logits must have shape (B,N,2)"
        )
    return base_logits[..., 1].detach()


def _stage_b_token_role_carrier_logits(outputs):
    base_logits = outputs.get("stage_b_dense_duty_confidence_base_logits")
    if base_logits is None:
        return None
    if (
        not torch.is_tensor(base_logits)
        or not base_logits.is_floating_point()
        or base_logits.dim() != 3
        or int(base_logits.shape[-1]) != 2
    ):
        raise RuntimeError(
            "Stage-B confidence base logits must have shape (B,N,2)"
        )
    return base_logits.detach()


def _scale_loss_for_logical_accumulation(
    losses: torch.Tensor,
    *,
    logical_batches_in_forward: int,
    logical_batches_in_update: int,
) -> torch.Tensor:
    forward_count = int(logical_batches_in_forward)
    update_count = int(logical_batches_in_update)
    if forward_count <= 0 or update_count <= 0 or forward_count > update_count:
        raise ValueError("logical Stage-B accumulation counts are invalid")
    return losses * (float(forward_count) / float(update_count))


def _index_nested_tensor(samples: NestedTensor, indices: List[int]) -> NestedTensor:
    idx = torch.as_tensor(indices, dtype=torch.long, device=samples.tensors.device)
    tensors = samples.tensors.index_select(0, idx)
    mask = samples.mask.index_select(0, idx) if samples.mask is not None else None
    return NestedTensor(tensors, mask)


def _select_rank_patch_rows(value, indices: List[int], slots: List[int]):
    if value is None:
        return None
    batch_idx = torch.as_tensor(indices, dtype=torch.long, device=value.device)
    slot_idx = torch.as_tensor(slots, dtype=torch.long, device=value.device)
    if value.dim() == 2:
        return value.index_select(0, batch_idx)
    if value.dim() == 3:
        return value[batch_idx, slot_idx].unsqueeze(1)
    if value.dim() == 4:
        return value.index_select(0, batch_idx)
    if value.dim() == 5:
        return value[batch_idx, slot_idx].unsqueeze(1)
    raise ValueError(f"Unsupported rank patch tensor shape: {tuple(value.shape)}")


def _zero_from_stage_b_outputs(outputs) -> Optional[torch.Tensor]:
    zero = None
    for key in ("pred_logits", "pred_logits_text", "pred_logits_patch", "pred_boxes"):
        value = outputs.get(key, None) if isinstance(outputs, dict) else None
        if torch.is_tensor(value):
            term = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
            zero = term if zero is None else zero + term
    return zero


def _sync_bool_any(value: bool, device: torch.device) -> bool:
    if not utils.is_dist_avail_and_initialized():
        return bool(value)
    flag = torch.as_tensor([1 if value else 0], dtype=torch.int32, device=device)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    return bool(int(flag.item()) > 0)


def _sync_bool_all(value: bool, device: torch.device) -> bool:
    if not utils.is_dist_avail_and_initialized():
        return bool(value)
    flag = torch.as_tensor([1 if value else 0], dtype=torch.int32, device=device)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    return bool(int(flag.item()) > 0)


def _build_stage_b_dummy_rank_subbatch(samples, targets, patches, patch_global, patch_mask):
    if not targets:
        return None
    device = samples.tensors.device
    batch_idx = 0
    slot_idx = 0
    if torch.is_tensor(patch_mask) and patch_mask.dim() == 2 and patch_mask.shape[0] > 0:
        valid = torch.nonzero(patch_mask[0].to(torch.bool).view(-1), as_tuple=False).flatten()
        if int(valid.numel()) > 0:
            slot_idx = int(valid[0].item())

    target = {
        k: v.to(device, non_blocking=True)
        for k, v in targets[batch_idx].items()
        if torch.is_tensor(v)
    }
    for mask_key in ("phrase_to_token_mask", "canonical_to_token_mask"):
        mask = target.get(mask_key, None)
        if torch.is_tensor(mask) and mask.dim() == 2 and mask.shape[0] > 0:
            safe_slot = min(slot_idx, int(mask.shape[0]) - 1)
            target[mask_key] = mask[safe_slot : safe_slot + 1]
    if "support_classes" in target and torch.is_tensor(target["support_classes"]):
        sc = target["support_classes"].view(-1)
        if int(sc.numel()) > 0:
            safe_slot = min(slot_idx, int(sc.numel()) - 1)
            target["support_classes"] = sc[safe_slot : safe_slot + 1]
            target["support_class"] = sc[safe_slot : safe_slot + 1]
    target["rank_source_slot"] = torch.as_tensor([slot_idx], dtype=torch.long, device=device)
    dummy_patch_mask = None
    if torch.is_tensor(patch_mask):
        dummy_patch_mask = torch.ones((1, 1), dtype=torch.bool, device=patch_mask.device)
    return {
        "samples": _index_nested_tensor(samples, [batch_idx]),
        "captions": ["object ."],
        "targets": [target],
        "patches": _select_rank_patch_rows(patches, [batch_idx], [slot_idx]),
        "patch_global": _select_rank_patch_rows(patch_global, [batch_idx], [slot_idx]),
        "patch_mask": dummy_patch_mask,
    }


def _build_stage_b_rank_subbatch(args, samples, targets, captions, patches, patch_global, patch_mask):
    need_positive_prompt = (
        bool(getattr(args, "stage_b_enable_phrase_rank", False))
        or float(getattr(args, "stage_b_score_calib_loss_coef", 0.0)) > 0.0
    )
    if not need_positive_prompt:
        return None
    device = samples.tensors.device
    rank_indices: List[int] = []
    rank_slots: List[int] = []
    rank_captions: List[str] = []
    rank_targets: List[dict] = []
    rank_candidate_tn_count = 0
    rank_missing_positive_count = 0
    rank_invalid_positive_count = 0
    rank_truncated_positive_count = 0
    max_pairs = int(
        getattr(
            args,
            "stage_b_rank_pos_max_pairs_per_gpu",
            getattr(args, "stage_b_rank_pos_max_pairs", 0),
        )
        or 0
    )
    for batch_idx, target in enumerate(targets):
        is_tn = target.get("is_tn", None)
        if torch.is_tensor(is_tn):
            tn_slots = torch.nonzero(is_tn.to(torch.bool).view(-1), as_tuple=False).flatten()
        else:
            tn_slots = torch.zeros((0,), dtype=torch.long)
        if int(tn_slots.numel()) == 0:
            continue
        rank_candidate_tn_count += 1
        has_rank = target.get("has_rank_positive", None)
        rank_captions_i = target.get("rank_positive_captions", None)
        if (not torch.is_tensor(has_rank)) or rank_captions_i is None:
            rank_missing_positive_count += 1
            continue
        valid_slots = torch.nonzero(has_rank.to(torch.bool).view(-1) & is_tn.to(torch.bool).view(-1), as_tuple=False).flatten()
        if int(valid_slots.numel()) == 0:
            rank_missing_positive_count += 1
            continue
        phrase_mask = target.get("rank_positive_phrase_to_token_mask", None)
        canonical_mask = target.get("rank_positive_canonical_to_token_mask", None)
        if (not torch.is_tensor(phrase_mask)) or (not torch.is_tensor(canonical_mask)):
            rank_invalid_positive_count += int(valid_slots.numel())
            continue
        if phrase_mask.dim() != 2 or canonical_mask.dim() != 2:
            rank_invalid_positive_count += int(valid_slots.numel())
            continue

        for slot_idx_t in valid_slots:
            slot_idx = int(slot_idx_t.item())
            if not isinstance(rank_captions_i, list) or slot_idx >= len(rank_captions_i):
                rank_invalid_positive_count += 1
                continue
            caption = rank_captions_i[slot_idx]
            if not isinstance(caption, str) or not caption.strip():
                rank_missing_positive_count += 1
                continue
            if slot_idx >= int(phrase_mask.shape[0]) or slot_idx >= int(canonical_mask.shape[0]):
                rank_invalid_positive_count += 1
                continue
            if not bool(phrase_mask[slot_idx].any().item()) or not bool(canonical_mask[slot_idx].any().item()):
                rank_invalid_positive_count += 1
                continue
            if max_pairs > 0 and len(rank_indices) >= max_pairs:
                rank_truncated_positive_count += 1
                continue
            rank_target = {
                k: v.to(device, non_blocking=True)
                for k, v in target.items()
                if torch.is_tensor(v)
                and k
                not in {
                    "phrase_to_token_mask",
                    "canonical_to_token_mask",
                    "content_to_token_mask",
                    "attr_pos_to_token_mask",
                    "attr_neg_to_token_mask",
                    "phrase_semantic_token_mask",
                    "negative_to_token_mask",
                    "attr_neg_weight_mask",
                    "rank_positive_phrase_to_token_mask",
                    "rank_positive_canonical_to_token_mask",
                    "has_rank_positive",
                }
            }
            rank_target["phrase_to_token_mask"] = phrase_mask[slot_idx : slot_idx + 1].to(device, non_blocking=True)
            rank_target["canonical_to_token_mask"] = canonical_mask[slot_idx : slot_idx + 1].to(device, non_blocking=True)
            selected_class = None
            if "support_classes" in rank_target and torch.is_tensor(rank_target["support_classes"]):
                sc = rank_target["support_classes"].view(-1)
                if slot_idx < int(sc.numel()):
                    selected_class = int(sc[slot_idx].item())
                    rank_target["support_classes"] = sc[slot_idx : slot_idx + 1]
                    rank_target["support_class"] = sc[slot_idx : slot_idx + 1]
            elif "support_class" in rank_target and torch.is_tensor(rank_target["support_class"]):
                selected_class = int(rank_target["support_class"].view(-1)[0].item())
                rank_target["support_class"] = rank_target["support_class"].view(-1)[:1]
            if selected_class is not None and "labels" in rank_target and "boxes" in rank_target:
                label_mask = rank_target["labels"].to(torch.long) == int(selected_class)
                if not bool(label_mask.any().item()):
                    rank_invalid_positive_count += 1
                    continue
                rank_target["rank_target_ids"] = torch.nonzero(label_mask, as_tuple=False).flatten().to(device)
                rank_target["labels"] = rank_target["labels"][label_mask]
                rank_target["boxes"] = rank_target["boxes"][label_mask]
            rank_target["rank_source_slot"] = torch.as_tensor([slot_idx], dtype=torch.long, device=device)
            rank_indices.append(batch_idx)
            rank_slots.append(slot_idx)
            rank_captions.append(caption)
            rank_targets.append(rank_target)
    if rank_candidate_tn_count <= 0:
        return None
    if not rank_indices:
        return {
            "indices": [],
            "rank_candidate_tn_count": rank_candidate_tn_count,
            "rank_missing_positive_count": rank_missing_positive_count,
            "rank_invalid_positive_count": rank_invalid_positive_count,
            "rank_truncated_positive_count": rank_truncated_positive_count,
        }
    rank_patch_mask = None
    if patch_mask is not None:
        rank_patch_mask = torch.ones((len(rank_indices), 1), dtype=torch.bool, device=patch_mask.device)
    return {
        "indices": rank_indices,
        "rank_candidate_tn_count": rank_candidate_tn_count,
        "rank_missing_positive_count": rank_missing_positive_count,
        "rank_invalid_positive_count": rank_invalid_positive_count,
        "rank_truncated_positive_count": rank_truncated_positive_count,
        "samples": _index_nested_tensor(samples, rank_indices),
        "captions": rank_captions,
        "targets": rank_targets,
        "patches": _select_rank_patch_rows(patches, rank_indices, rank_slots),
        "patch_global": _select_rank_patch_rows(patch_global, rank_indices, rank_slots),
        "patch_mask": rank_patch_mask,
    }


def _stack_stage_b_v7_mask(targets, mask_key: str, device):
    if not all(mask_key in t for t in targets):
        return None
    values = [t[mask_key] for t in targets]
    if not all(torch.is_tensor(v) for v in values):
        return None
    if len({tuple(v.shape) for v in values}) == 1:
        return torch.stack(values, dim=0).to(device, non_blocking=True)
    if all(v.dim() == 2 for v in values):
        kmax = max(int(v.shape[0]) for v in values)
        tmax = max(int(v.shape[1]) for v in values)
        padded = values[0].new_zeros((len(values), kmax, tmax))
        for i, v in enumerate(values):
            padded[i, : int(v.shape[0]), : int(v.shape[1])] = v
        return padded.to(device, non_blocking=True)
    if all(v.dim() == 1 for v in values):
        kmax = max(int(v.shape[0]) for v in values)
        padded = values[0].new_zeros((len(values), kmax))
        for i, v in enumerate(values):
            padded[i, : int(v.shape[0])] = v
        return padded.to(device, non_blocking=True)
    return None


def _normalize_stage_b_v11_expression(value, *, sample_idx: int, slot_idx: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Stage B v11 sample {sample_idx} expression slot {slot_idx} must be non-empty text"
        )
    value = " ".join(value.strip().split())
    # GroundingDINO's special-token mask recognizes '.' and '?' but not '!'.
    # Append a period after '!' so the full expression still receives a mask.
    if value[-1] not in ".?":
        value = value + " ."
    return value


def _stage_b_v11_scalar_int(target, key: str, *, sample_idx: int) -> int:
    if key not in target:
        raise KeyError(f"Stage B v11 sample {sample_idx} is missing {key!r}")
    value = target[key]
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(
                f"Stage B v11 sample {sample_idx} {key!r} must be scalar, "
                f"got shape {tuple(value.shape)}"
            )
        return int(value.reshape(-1)[0].item())
    return int(value)


def _build_stage_b_v11_expression_slots(targets, device):
    """Build independent [positive, local-TN] expressions with a fixed DDP shape."""
    expression_captions = []
    expression_valid = []
    for sample_idx, target in enumerate(targets):
        patch_slots = _stage_b_v11_scalar_int(
            target, "verifier_num_patch_slots", sample_idx=sample_idx
        )
        if patch_slots != 1:
            raise ValueError(
                "Stage B v11 separates one localization patch from text scoring; "
                f"sample {sample_idx} declares {patch_slots} patch slots"
            )
        pair_stride = _stage_b_v11_scalar_int(
            target, "verifier_pair_stride", sample_idx=sample_idx
        )
        cap_list = target.get("cap_list", None)
        if not isinstance(cap_list, (list, tuple)):
            raise TypeError(
                f"Stage B v11 sample {sample_idx} cap_list must be list/tuple"
            )
        raw_is_tn = target.get("is_tn", None)
        if raw_is_tn is None:
            raise KeyError(f"Stage B v11 sample {sample_idx} is missing 'is_tn'")
        is_tn = torch.as_tensor(raw_is_tn, dtype=torch.bool).reshape(-1).tolist()

        if pair_stride == 2:
            if len(cap_list) != 2 or is_tn != [False, True]:
                raise ValueError(
                    "Stage B v11 paired samples must be exactly "
                    "[positive expression, local TN expression]; "
                    f"sample {sample_idx} has len(cap_list)={len(cap_list)}, is_tn={is_tn}"
                )
            expression_captions.append(
                [
                    _normalize_stage_b_v11_expression(
                        cap_list[0], sample_idx=sample_idx, slot_idx=0
                    ),
                    _normalize_stage_b_v11_expression(
                        cap_list[1], sample_idx=sample_idx, slot_idx=1
                    ),
                ]
            )
            expression_valid.append([True, True])
        elif pair_stride == 1:
            if len(cap_list) != 1 or is_tn != [False]:
                raise ValueError(
                    "Stage B v11 clean samples must contain exactly one positive expression; "
                    f"sample {sample_idx} has len(cap_list)={len(cap_list)}, is_tn={is_tn}"
                )
            expression_captions.append(
                [
                    _normalize_stage_b_v11_expression(
                        cap_list[0], sample_idx=sample_idx, slot_idx=0
                    ),
                    "object .",
                ]
            )
            expression_valid.append([True, False])
        else:
            raise ValueError(
                f"Stage B v11 sample {sample_idx} has unsupported pair stride {pair_stride}"
            )

    return expression_captions, torch.as_tensor(
        expression_valid, dtype=torch.bool, device=device
    )


def _preserve_stage_b_v21_trace_metadata(source, destination) -> None:
    trace = source.get("stage_b_data_driven_trace")
    if trace is None:
        return
    if not isinstance(trace, Mapping):
        raise TypeError(
            "stage_b_data_driven_trace must remain a mapping in patch-only training"
        )
    destination["stage_b_data_driven_trace"] = copy.deepcopy(dict(trace))


def _build_stage_b_v21_certified_edit_traces(targets):
    traces = []
    for sample_idx, target in enumerate(targets):
        provenance_valid = _strict_target_bool(
            target, "stage_b_v21_token_supervision_valid"
        )
        trace = target.get("stage_b_data_driven_trace")
        if provenance_valid and not isinstance(trace, Mapping):
            raise RuntimeError(
                "certified Stage B v21 sample lost its direct edit trace at "
                f"batch index {sample_idx}"
            )
        traces.append(
            copy.deepcopy(dict(trace))
            if provenance_valid and isinstance(trace, Mapping)
            else None
        )
    return traces


def _build_stage_b_v15_exact_candidate_replay(targets, *, topk: int, device):
    """Collect exact candidate bindings without imposing them on other rows."""
    from util.stageb_exact_topk_contract import EXACT_TOPK_TN_SCOPE

    exact_mask = []
    indices = []
    boxes = []
    atols = []
    any_exact = False
    for sample_idx, target in enumerate(targets):
        flag = target.get("fixed_stagea_topk_exact_verified", None)
        is_exact = (
            torch.is_tensor(flag)
            and flag.numel() == 1
            and flag.dtype == torch.bool
            and bool(flag.reshape(-1)[0].item())
        )
        exact_mask.append(is_exact)
        any_exact = any_exact or is_exact
        if not is_exact:
            indices.append(torch.zeros(topk, dtype=torch.int64, device=device))
            boxes.append(torch.zeros(topk, 4, dtype=torch.float32, device=device))
            atols.append(0.0)
            continue
        if target.get("tn_scope") != EXACT_TOPK_TN_SCOPE:
            raise RuntimeError(
                f"exact Stage-A candidate sample {sample_idx} has wrong TN scope"
            )
        global_flag = target.get("global_tn_verified", None)
        if not (
            torch.is_tensor(global_flag)
            and global_flag.numel() == 1
            and global_flag.dtype == torch.bool
            and bool(global_flag.reshape(-1)[0].item())
        ):
            raise RuntimeError(
                f"exact Stage-A candidate sample {sample_idx} lacks global verification"
            )
        candidate_indices = target.get("fixed_stagea_candidate_indices")
        candidate_boxes = target.get("fixed_stagea_candidate_boxes")
        candidate_atol = target.get("fixed_stagea_candidate_box_atol")
        if (
            not torch.is_tensor(candidate_indices)
            or candidate_indices.dtype != torch.int64
            or tuple(candidate_indices.shape) != (topk,)
            or int(torch.unique(candidate_indices).numel()) != topk
            or bool((candidate_indices < 0).any().item())
        ):
            raise RuntimeError(
                f"exact Stage-A candidate sample {sample_idx} has invalid ordered indices"
            )
        if (
            not torch.is_tensor(candidate_boxes)
            or tuple(candidate_boxes.shape) != (topk, 4)
            or not candidate_boxes.is_floating_point()
            or not bool(torch.isfinite(candidate_boxes).all().item())
        ):
            raise RuntimeError(
                f"exact Stage-A candidate sample {sample_idx} has invalid boxes"
            )
        if not torch.is_tensor(candidate_atol) or candidate_atol.numel() != 1:
            raise RuntimeError(
                f"exact Stage-A candidate sample {sample_idx} has invalid box tolerance"
            )
        atol = float(candidate_atol.reshape(-1)[0].item())
        if not 0.0 <= atol <= 1.0e-3:
            raise RuntimeError(
                f"exact Stage-A candidate sample {sample_idx} box tolerance is unsafe"
            )
        indices.append(candidate_indices.to(device=device))
        boxes.append(candidate_boxes.to(device=device, dtype=torch.float32))
        atols.append(atol)
    if not any_exact:
        return None
    exact_atols = {atols[index] for index, value in enumerate(exact_mask) if value}
    if len(exact_atols) != 1:
        raise RuntimeError("exact Stage-A candidate rows mix box tolerances")
    return {
        "mask": torch.as_tensor(exact_mask, dtype=torch.bool, device=device),
        "indices": torch.stack(indices, dim=0),
        "boxes": torch.stack(boxes, dim=0),
        "box_atol": exact_atols.pop(),
    }


def _build_stage_b_legacy_gate_pair_captions(targets):
    """Extract one positive and one proposal-set-proxy TN per sample."""
    positive_captions = []
    negative_captions = []
    for sample_idx, target in enumerate(targets):
        verified = target.get("proposalset_proxy_verified", None)
        verified = (
            bool(verified.detach().view(-1)[0].item() is True)
            if torch.is_tensor(verified)
            and verified.dtype == torch.bool
            and verified.numel() == 1
            else False
        )
        if not verified:
            raise RuntimeError(
                "Legacy global gate received a proposal-set TN row without "
                f"proposalset_proxy_verified=True at batch index {sample_idx}"
            )
        patch_slots = _stage_b_v11_scalar_int(
            target, "verifier_num_patch_slots", sample_idx=sample_idx
        )
        pair_stride = _stage_b_v11_scalar_int(
            target, "verifier_pair_stride", sample_idx=sample_idx
        )
        cap_list = target.get("cap_list", None)
        raw_is_tn = target.get("is_tn", None)
        if not isinstance(cap_list, (list, tuple)) or raw_is_tn is None:
            raise ValueError(
                f"Legacy global gate sample {sample_idx} is missing its paired expressions"
            )
        is_tn = torch.as_tensor(raw_is_tn, dtype=torch.bool).reshape(-1).tolist()
        if patch_slots != 1 or pair_stride != 2 or len(cap_list) != 2 or is_tn != [False, True]:
            raise ValueError(
                "Legacy global gate requires exactly one patch and expressions "
                "[positive, proposal-set TN proxy]; "
                f"sample {sample_idx} has patch_slots={patch_slots}, "
                f"pair_stride={pair_stride}, len(cap_list)={len(cap_list)}, is_tn={is_tn}"
            )
        positive_captions.append(
            _normalize_stage_b_v11_expression(
                cap_list[0], sample_idx=sample_idx, slot_idx=0
            )
        )
        negative_captions.append(
            _normalize_stage_b_v11_expression(
                cap_list[1], sample_idx=sample_idx, slot_idx=1
            )
        )
    return positive_captions, negative_captions


def _strict_target_bool(target, key: str) -> bool:
    value = target.get(key, None)
    if torch.is_tensor(value):
        return bool(
            value.dtype == torch.bool
            and value.numel() == 1
            and value.detach().reshape(-1)[0].item() is True
        )
    return value is True


def _strict_target_false(target, key: str) -> bool:
    value = target.get(key, None)
    if torch.is_tensor(value):
        return bool(
            value.dtype == torch.bool
            and value.numel() == 1
            and value.detach().reshape(-1)[0].item() is False
        )
    return value is False


def _strict_target_zero(target, key: str) -> bool:
    value = target.get(key, None)
    if torch.is_tensor(value):
        return bool(
            value.dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
            and value.numel() == 1
            and int(value.detach().reshape(-1)[0].item()) == 0
        )
    return isinstance(value, int) and not isinstance(value, bool) and value == 0


def _build_stage_b_gdino_adapter_rank_captions(
    targets,
    *,
    u0_patch_episode: bool = False,
):
    """Fail-closed audit for positive-only rank supervision."""
    captions = []
    for sample_idx, target in enumerate(targets):
        cap_list = target.get("cap_list", None)
        boxes = target.get("boxes", None)
        if not isinstance(cap_list, (list, tuple)) or len(cap_list) != 1:
            raise ValueError(
                "GDINO adapter rank_only requires exactly one positive expression; "
                f"sample {sample_idx} has cap_list={cap_list!r}"
            )
        if (
            not torch.is_tensor(boxes)
            or boxes.dim() != 2
            or boxes.numel() == 0
            or int(boxes.shape[-1]) != 4
        ):
            raise ValueError(
                f"GDINO adapter rank_only sample {sample_idx} requires non-empty boxes"
            )
        if u0_patch_episode:
            for key in (
                "is_negative_episode",
                "is_lvis_neg_category_episode",
            ):
                if not _strict_target_zero(target, key):
                    raise ValueError(
                        "Stage-B U0 rank_only requires exact integer "
                        f"{key}=0; sample {sample_idx} is malformed or negative"
                    )
        elif not _strict_target_zero(target, "is_negative"):
            raise ValueError(
                "GDINO adapter rank_only requires exact integer is_negative=0; "
                f"sample {sample_idx} is malformed or negative"
            )
        captions.append(
            _normalize_stage_b_v11_expression(
                cap_list[0], sample_idx=sample_idx, slot_idx=0
            )
        )
    return captions


def _build_stage_b_data_driven_positive_captions(targets):
    """Return canonical query prompts and independent full expressions."""
    canonical_captions = []
    expression_captions = []
    for sample_idx, target in enumerate(targets):
        cap_list = target.get("cap_list", None)
        canonical = target.get("stage_a_caption", None)
        boxes = target.get("boxes", None)
        if not isinstance(cap_list, (list, tuple)) or len(cap_list) != 1:
            raise ValueError(
                "data-driven rank/patch training requires exactly one positive "
                f"expression; sample {sample_idx} has cap_list={cap_list!r}"
            )
        if not isinstance(canonical, str) or not canonical.strip():
            raise ValueError(
                f"data-driven sample {sample_idx} has no canonical stage_a_caption"
            )
        if (
            not torch.is_tensor(boxes)
            or boxes.dim() != 2
            or boxes.numel() == 0
            or int(boxes.shape[-1]) != 4
        ):
            raise ValueError(
                f"data-driven sample {sample_idx} requires non-empty target boxes"
            )
        for key in ("is_negative_episode", "is_lvis_neg_category_episode"):
            if not _strict_target_zero(target, key):
                raise ValueError(
                    "data-driven rank/patch training requires exact integer "
                    f"{key}=0; sample {sample_idx} is malformed or negative"
                )
        canonical_captions.append(
            _normalize_stage_b_v11_expression(
                canonical, sample_idx=sample_idx, slot_idx=0
            )
        )
        expression_captions.append(
            _normalize_stage_b_v11_expression(
                cap_list[0], sample_idx=sample_idx, slot_idx=0
            )
        )
    return canonical_captions, expression_captions


def _build_stage_b_data_driven_assignment_captions(targets):
    """Return canonical prompts and two official referent expressions per image."""
    canonical_captions = []
    expression_captions = []
    expected_schema = "pivot.stageb.data_driven.official_assignment_pair/v1"
    for sample_idx, target in enumerate(targets):
        cap_list = target.get("cap_list", None)
        canonical = target.get("stage_a_caption", None)
        expressions = target.get(
            "stage_b_data_driven_assignment_expressions", None
        )
        schema = target.get("stage_b_data_driven_assignment_pair_schema", None)
        pair_valid = target.get("stage_b_data_driven_assignment_valid", None)
        roles = target.get("stage_b_data_driven_assignment_role", None)
        boxes = target.get("boxes", None)
        if schema != expected_schema:
            raise ValueError(
                f"official assignment sample {sample_idx} schema drifted"
            )
        if not isinstance(cap_list, (list, tuple)) or len(cap_list) != 1:
            raise ValueError(
                "official assignment keeps exactly one anchor Ref expression; "
                f"sample {sample_idx} has cap_list={cap_list!r}"
            )
        if not (
            isinstance(expressions, (list, tuple))
            and len(expressions) == 2
            and all(
                isinstance(expression, str) and bool(expression.strip())
                for expression in expressions
            )
        ):
            raise ValueError(
                f"official assignment sample {sample_idx} lost its expression pair"
            )
        if not isinstance(canonical, str) or not canonical.strip():
            raise ValueError(
                f"official assignment sample {sample_idx} has no canonical caption"
            )
        if (
            not torch.is_tensor(pair_valid)
            or pair_valid.dtype != torch.bool
            or pair_valid.numel() != 1
            or not torch.is_tensor(roles)
            or roles.dtype != torch.int64
            or not torch.is_tensor(boxes)
            or boxes.dim() != 2
            or tuple(roles.reshape(-1).shape) != (int(boxes.shape[0]),)
        ):
            raise ValueError(
                f"official assignment sample {sample_idx} has invalid role metadata"
            )
        if bool(pair_valid.reshape(-1)[0].item()) and any(
            int((roles.reshape(-1) == role).sum().item()) != 1
            for role in (0, 1)
        ):
            raise ValueError(
                f"official assignment sample {sample_idx} lacks two exact targets"
            )
        normalized_expressions = [
            _normalize_stage_b_v11_expression(
                expression, sample_idx=sample_idx, slot_idx=slot_idx
            )
            for slot_idx, expression in enumerate(expressions)
        ]
        normalized_anchor = _normalize_stage_b_v11_expression(
            cap_list[0], sample_idx=sample_idx, slot_idx=0
        )
        if normalized_expressions[0] != normalized_anchor:
            raise ValueError(
                f"official assignment sample {sample_idx} anchor expression drifted"
            )
        for key in ("is_negative_episode", "is_lvis_neg_category_episode"):
            if not _strict_target_zero(target, key):
                raise ValueError(
                    "official assignment requires positive image/support episodes; "
                    f"sample {sample_idx} has invalid {key}"
                )
        canonical_captions.append(
            _normalize_stage_b_v11_expression(
                canonical, sample_idx=sample_idx, slot_idx=0
            )
        )
        expression_captions.append(normalized_expressions)
    return canonical_captions, expression_captions


def _build_stage_b_data_driven_pair_captions(targets):
    """Extract one traceable [positive, image-global TN] pair per image."""
    canonical_captions = []
    expression_captions = []
    for sample_idx, target in enumerate(targets):
        cap_list = target.get("cap_list", None)
        canonical = target.get("stage_a_caption", None)
        trace = target.get("stage_b_data_driven_trace", None)
        if not isinstance(cap_list, (list, tuple)) or len(cap_list) != 2:
            raise ValueError(
                "data-driven confidence training requires exactly one aligned "
                f"[positive, TN] pair; sample {sample_idx} has cap_list={cap_list!r}"
            )
        if not isinstance(canonical, str) or not canonical.strip():
            raise ValueError(
                f"data-driven confidence sample {sample_idx} has no canonical caption"
            )
        if target.get("tn_scope", None) != "image_global_topk_verified":
            raise ValueError(
                "global-max data-driven confidence accepts only image-global "
                "verified TN rows; "
                f"sample {sample_idx} has scope={target.get('tn_scope', None)!r}"
            )
        if not _strict_target_bool(target, "global_tn_verified"):
            raise ValueError(
                "global-max data-driven confidence requires exact "
                f"global_tn_verified=true at sample {sample_idx}"
            )
        if not isinstance(trace, dict):
            raise ValueError(
                f"data-driven confidence sample {sample_idx} lost its edit trace"
            )
        pair_stride = _stage_b_v11_scalar_int(
            target, "verifier_pair_stride", sample_idx=sample_idx
        )
        if pair_stride != 2:
            raise ValueError(
                f"data-driven confidence sample {sample_idx} has pair stride {pair_stride}"
            )
        for key in ("is_negative_episode", "is_lvis_neg_category_episode"):
            if not _strict_target_zero(target, key):
                raise ValueError(
                    "data-driven confidence requires positive image/support "
                    f"episodes; sample {sample_idx} has invalid {key}"
                )
        canonical_captions.append(
            _normalize_stage_b_v11_expression(
                canonical, sample_idx=sample_idx, slot_idx=0
            )
        )
        expression_captions.append(
            [
                _normalize_stage_b_v11_expression(
                    caption, sample_idx=sample_idx, slot_idx=slot_idx
                )
                for slot_idx, caption in enumerate(cap_list)
            ]
        )
    return canonical_captions, expression_captions


def _build_stage_b_gdino_adapter_pair_captions(targets, expected_scope: str):
    """Extract one explicitly scoped [positive, TN] pair per sample."""
    from models.GroundingDINO.stage_b_gdino_score_adapter import (
        stage_b_gdino_tn_scope_code,
    )

    expected_scope = str(expected_scope).strip()
    expected_code = stage_b_gdino_tn_scope_code(expected_scope)
    positive_captions = []
    negative_captions = []
    for sample_idx, target in enumerate(targets):
        scope = target.get("tn_scope", None)
        if scope in {"proposal_set_verified", "proposalset_proxy"} or not _strict_target_false(
            target, "proposalset_proxy_verified"
        ):
            raise RuntimeError(
                "GDINO adapter requires an exact boolean false proposal-proxy "
                f"flag and rejects proposal-set proxy sample {sample_idx}"
            )
        if scope != expected_scope:
            raise RuntimeError(
                f"GDINO adapter sample {sample_idx} scope={scope!r} does not match "
                f"configured scope={expected_scope!r}"
            )
        verification_key = (
            "global_tn_verified"
            if expected_scope == "image_global_topk_verified"
            else "benchmark_dataft_alltn"
        )
        if not _strict_target_bool(target, verification_key):
            raise RuntimeError(
                f"GDINO adapter sample {sample_idx} requires exact boolean "
                f"{verification_key}=True"
            )
        patch_slots = _stage_b_v11_scalar_int(
            target, "verifier_num_patch_slots", sample_idx=sample_idx
        )
        pair_stride = _stage_b_v11_scalar_int(
            target, "verifier_pair_stride", sample_idx=sample_idx
        )
        cap_list = target.get("cap_list", None)
        raw_is_tn = target.get("is_tn", None)
        if not isinstance(cap_list, (list, tuple)) or raw_is_tn is None:
            raise ValueError(
                f"GDINO adapter sample {sample_idx} is missing paired expressions"
            )
        is_tn = torch.as_tensor(raw_is_tn, dtype=torch.bool).reshape(-1).tolist()
        if (
            patch_slots != 1
            or pair_stride != 2
            or len(cap_list) != 2
            or is_tn != [False, True]
        ):
            raise ValueError(
                "GDINO adapter requires one localization slot and exactly "
                "[positive expression, scoped TN expression]; "
                f"sample {sample_idx} has patch_slots={patch_slots}, "
                f"pair_stride={pair_stride}, len(cap_list)={len(cap_list)}, "
                f"is_tn={is_tn}"
            )
        positive_captions.append(
            _normalize_stage_b_v11_expression(
                cap_list[0], sample_idx=sample_idx, slot_idx=0
            )
        )
        negative_captions.append(
            _normalize_stage_b_v11_expression(
                cap_list[1], sample_idx=sample_idx, slot_idx=1
            )
        )
    scope_codes = torch.full(
        (len(targets),), expected_code, dtype=torch.int64
    )
    return positive_captions, negative_captions, scope_codes


def _split_stage_b_legacy_gate_batch(value, batch_size: int, take_negative: bool):
    """Split model outputs from the concatenated [positive, TN] batch."""
    if torch.is_tensor(value):
        if value.dim() > 0 and int(value.shape[0]) == 2 * int(batch_size):
            start = int(batch_size) if take_negative else 0
            return value.narrow(0, start, int(batch_size))
        return value
    if isinstance(value, dict):
        return {
            key: _split_stage_b_legacy_gate_batch(item, batch_size, take_negative)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _split_stage_b_legacy_gate_batch(item, batch_size, take_negative)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _split_stage_b_legacy_gate_batch(item, batch_size, take_negative)
            for item in value
        )
    return value


def _forward_stage_b_gdino_adapter_pair(
    model,
    samples,
    targets,
    positive_captions,
    negative_captions,
    scope_codes,
):
    """Run one concatenated DDP forward and split its positive/TN halves."""
    batch_size = len(targets)
    if (
        len(positive_captions) != batch_size
        or len(negative_captions) != batch_size
        or int(scope_codes.numel()) != batch_size
    ):
        raise ValueError("GDINO adapter paired batch fields do not align")
    paired_samples = NestedTensor(
        torch.cat((samples.tensors, samples.tensors), dim=0),
        (
            torch.cat((samples.mask, samples.mask), dim=0)
            if samples.mask is not None
            else None
        ),
    )
    paired_outputs = model(
        paired_samples,
        captions=list(positive_captions) + list(negative_captions),
    )
    outputs = _split_stage_b_legacy_gate_batch(
        paired_outputs, batch_size, take_negative=False
    )
    outputs["stage_b_gdino_tn_outputs"] = _split_stage_b_legacy_gate_batch(
        paired_outputs, batch_size, take_negative=True
    )
    outputs["stage_b_gdino_tn_scope_code"] = scope_codes.to(
        device=samples.tensors.device, non_blocking=True
    )
    outputs["stage_b_gdino_positive_captions"] = list(positive_captions)
    outputs["stage_b_gdino_tn_captions"] = list(negative_captions)
    return outputs


def _restore_rng_state(rng_state) -> None:
    if not rng_state:
        return
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
    if "torch" in rng_state:
        torch.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available() and rng_state.get("cuda", None) is not None:
        torch.cuda.set_rng_state_all(rng_state["cuda"])


def _unnormalize_img(img: torch.Tensor) -> torch.Tensor:
    """
    img: (3,H,W) normalized by ImageNet mean/std.
    Returns float tensor in [0,1].
    """
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=img.dtype, device=img.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=img.dtype, device=img.device)[:, None, None]
    x = img * std + mean
    return x.clamp(0, 1)


def _cxcywh_norm_to_xyxy_abs(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
    """
    boxes: (...,4) normalized cxcywh in [0,1]
    returns (...,4) absolute xyxy in pixels.
    """
    cx, cy, bw, bh = boxes.unbind(-1)
    x0 = (cx - 0.5 * bw) * w
    y0 = (cy - 0.5 * bh) * h
    x1 = (cx + 0.5 * bw) * w
    y1 = (cy + 0.5 * bh) * h
    out = torch.stack([x0, y0, x1, y1], dim=-1)
    out[..., 0::2] = out[..., 0::2].clamp(0, w - 1)
    out[..., 1::2] = out[..., 1::2].clamp(0, h - 1)
    return out


@torch.no_grad()
def _maybe_save_patch_sanity(
    *,
    args,
    samples,
    targets,
    outputs,
    criterion=None,
    epoch: int,
    step: int,
) -> None:
    if not bool(getattr(args, "log_patch_sanity", False)):
        return
    if not utils.is_main_process():
        return
    out_dir = getattr(args, "output_dir", None)
    if not out_dir:
        return

    interval = int(getattr(args, "patch_sanity_interval", 500))
    if interval <= 0 or (step % interval) != 0:
        return

    try:
        from PIL import Image, ImageDraw, ImageFont  # pylint: disable=import-error
    except Exception:
        return

    topk = int(getattr(args, "patch_sanity_topk", 20))
    max_images = int(getattr(args, "patch_sanity_max_images", 2))
    topk = max(1, topk)
    max_images = max(1, max_images)

    if "pred_logits_patch" not in outputs or outputs["pred_logits_patch"] is None:
        return
    if "pred_boxes" not in outputs:
        return

    img_t = samples.tensors.detach().float().cpu()  # (B,3,H,W)
    mask = getattr(samples, "mask", None)
    if mask is not None:
        mask = mask.detach().cpu()  # (B,H,W) True for padded

    pred_logits_raw = outputs["pred_logits_patch"].detach().float().cpu()  # (B,Q) or (B,Q,K)
    # Union score for ranking/quick view.
    pred_logits_union = pred_logits_raw
    if pred_logits_union.dim() == 3:
        pred_logits_union = pred_logits_union.max(dim=-1).values  # (B,Q)
    pred_boxes = outputs["pred_boxes"].detach().float().cpu()  # (B,Q,4) cxcywh norm
    patch_mask = outputs.get("patch_mask", None)
    if patch_mask is not None:
        patch_mask = patch_mask.detach().cpu().to(torch.bool)

    save_dir = Path(out_dir) / "patch_sanity"
    save_dir.mkdir(parents=True, exist_ok=True)

    dn_num = int(getattr(args, "patch_dn_num_queries", 0))
    B = int(img_t.shape[0])
    for b in range(min(B, max_images)):
        x = _unnormalize_img(img_t[b])
        if mask is not None:
            m = mask[b]
            x[:, m] = 0.0
        c, h, w = x.shape
        _ = c
        img_u8 = (x.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        pil = Image.fromarray(img_u8)
        draw = ImageDraw.Draw(pil)

        logits_union = pred_logits_union[b]  # (Q,)
        boxes_all = pred_boxes[b]
        Q = int(logits_union.numel())
        dn = max(0, min(int(dn_num), Q))
        is_neg = int(targets[b].get("is_negative_episode", torch.tensor([0])).item())
        is_lvis_neg = int(targets[b].get("is_lvis_neg_category_episode", torch.tensor([0])).item())

        def _load_font(size: int, bold: bool = False):
            try:
                candidates = [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                ]
                for p in candidates:
                    try:
                        return ImageFont.truetype(p, size=size)
                    except Exception:
                        continue
            except Exception:
                pass
            return ImageFont.load_default()

        font_small = _load_font(14, bold=False)
        font_big = _load_font(22, bold=True)

        def _draw_text_bold(xy, text: str, fill, stroke: int = 1, font=None):
            x0, y0 = xy
            for dx in range(-stroke, stroke + 1):
                for dy in range(-stroke, stroke + 1):
                    draw.text((x0 + dx, y0 + dy), text, fill=fill, font=font)

        # 1) Draw ALL GT boxes (green).
        gt_boxes = targets[b].get("boxes", None)
        gt_labels = targets[b].get("labels", None)
        has_gt = False
        if gt_boxes is not None and gt_labels is not None:
            gt_boxes = gt_boxes.detach().float().cpu()
            gt_labels = gt_labels.detach().long().cpu()
            has_gt = bool(gt_boxes.numel() > 0 and gt_labels.numel() > 0)
            gt_xyxy = _cxcywh_norm_to_xyxy_abs(gt_boxes, w=w, h=h).tolist()
            for (x0, y0, x1, y1) in gt_xyxy:
                draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=4)

        # 2) Draw top-k query boxes in red (union score).
        sc = logits_union.sigmoid()
        k = min(topk, int(sc.numel()))
        if k > 0:
            vals, idx = torch.topk(sc, k=k, largest=True)
            q_boxes = boxes_all[idx]
            q_xyxy = _cxcywh_norm_to_xyxy_abs(q_boxes, w=w, h=h).tolist()
            for (x0, y0, x1, y1), v in zip(q_xyxy, vals.tolist()):
                draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
                _draw_text_bold((x0 + 2, max(0, y0 - 14)), f"{v:.2f}", fill=(255, 0, 0), stroke=1, font=font_small)

        # 3) Draw Hungarian-matched query boxes in yellow, and annotate (canonical id, logit).
        if criterion is not None and hasattr(criterion, "matcher") and criterion.matcher is not None:
            # Prepare logits as (Q,K) for matching.
            if pred_logits_raw.dim() == 2:
                logits_qk = pred_logits_raw[b].unsqueeze(-1)  # (Q,1)
                support_classes = targets[b].get("support_class", None)
                if support_classes is not None:
                    support_classes = support_classes.detach().long().cpu().view(-1)
                else:
                    support_classes = torch.full((1,), -1, dtype=torch.long)
            else:
                logits_qk = pred_logits_raw[b]  # (Q,K)
                support_classes = targets[b].get("support_classes", None)
                if support_classes is not None:
                    support_classes = support_classes.detach().long().cpu().view(-1)
                else:
                    support_classes = torch.full((logits_qk.shape[1],), -1, dtype=torch.long)

            K = int(logits_qk.shape[1])
            if support_classes.numel() < K:
                pad = torch.full((K - int(support_classes.numel()),), -1, dtype=support_classes.dtype)
                support_classes = torch.cat([support_classes, pad], dim=0)
            support_classes = support_classes[:K]
            valid_k = support_classes >= 0
            if patch_mask is not None:
                valid_k = valid_k & patch_mask[b].view(-1)[:K]

            if has_gt and int(valid_k.sum().item()) > 0:
                keep = valid_k.nonzero(as_tuple=False).flatten()
                logits_b = logits_qk[:, keep]
                support_kept = support_classes[keep].to(torch.long)

                max_row = int(max(int(gt_labels.max().item()), int(support_kept.max().item()))) + 1
                label_map = torch.zeros((max_row, int(keep.numel())), dtype=torch.float32)
                for local_k, cid in enumerate(support_kept.tolist()):
                    if cid >= 0 and cid < max_row:
                        label_map[int(cid), int(local_k)] = 1.0

                try:
                    (src_idx, tgt_idx) = criterion.matcher(
                        {"pred_logits": logits_b.unsqueeze(0), "pred_boxes": boxes_all.unsqueeze(0)},
                        [{"labels": gt_labels, "boxes": gt_boxes}],
                        label_map,
                    )[0]
                except Exception:
                    src_idx = torch.zeros((0,), dtype=torch.int64)
                    tgt_idx = torch.zeros((0,), dtype=torch.int64)

                if src_idx.numel() > 0:
                    cid_to_local = {int(cid): int(i) for i, cid in enumerate(support_kept.tolist())}
                    pred_xyxy = _cxcywh_norm_to_xyxy_abs(boxes_all[src_idx], w=w, h=h).tolist()
                    for m_i, ((x0, y0, x1, y1), gt_i) in enumerate(zip(pred_xyxy, tgt_idx.tolist())):
                        cid = int(gt_labels[gt_i].item())
                        lk = cid_to_local.get(cid, None)
                        if lk is None:
                            continue
                        logit = float(logits_b[int(src_idx[m_i].item()), int(lk)].item())
                        score = 1.0 / (1.0 + torch.exp(torch.tensor(-logit))).item()
                        draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0), width=3)
                        _draw_text_bold((x0 + 2, max(0, y0 - 24)), f"{cid}", fill=(255, 255, 0), stroke=2, font=font_big)

        draw.text(
            (5, 5),
            f"epoch={epoch} step={step} neg={is_neg} lvis_neg={is_lvis_neg} dn={dn}",
            fill=(255, 255, 0),
        )
        pil.save(save_dir / f"e{epoch:03d}_s{step:06d}_b{b}.jpg", quality=90)


def _clone_target_value_for_drift(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    return copy.deepcopy(value)


def _move_target_value_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    return value


def _clone_stage_b_drift_batch(samples, targets, captions, patches, patch_global, patch_mask):
    return {
        "samples": utils.NestedTensor(
            samples.tensors.detach().cpu().clone(),
            samples.mask.detach().cpu().clone() if samples.mask is not None else None,
        ),
        "targets": [
            {k: _clone_target_value_for_drift(v) for k, v in t.items()}
            for t in targets
        ],
        "captions": list(captions),
        "patches": patches.detach().cpu().clone() if torch.is_tensor(patches) else None,
        "patch_global": patch_global.detach().cpu().clone() if torch.is_tensor(patch_global) else None,
        "patch_mask": patch_mask.detach().cpu().clone() if torch.is_tensor(patch_mask) else None,
    }


def _move_stage_b_drift_batch_to_device(batch, device):
    return {
        "samples": batch["samples"].to(device),
        "targets": [
            {k: _move_target_value_to_device(v, device) for k, v in t.items()}
            for t in batch["targets"]
        ],
        "captions": list(batch["captions"]),
        "patches": batch["patches"].to(device) if torch.is_tensor(batch["patches"]) else None,
        "patch_global": batch["patch_global"].to(device) if torch.is_tensor(batch["patch_global"]) else None,
        "patch_mask": batch["patch_mask"].to(device) if torch.is_tensor(batch["patch_mask"]) else None,
    }


@torch.no_grad()
def _eval_stage_b_drift_batch(args, model, device, batch):
    cached = _move_stage_b_drift_batch_to_device(batch, device)
    was_training = model.training
    model.eval()
    try:
        with torch.cuda.amp.autocast(enabled=getattr(args, "amp", False)):
            eval_outputs = model(
                cached["samples"],
                targets=cached["targets"],
                captions=cached["captions"],
                patches=cached["patches"],
                patch_global=cached["patch_global"],
                patch_mask=cached["patch_mask"],
                patch_only=True,
                patch_only_compute_text_logits=bool(getattr(args, "patch_only_compute_text_logits", False)),
            )
    finally:
        if was_training:
            model.train()
            if bool(getattr(args, "stage_b_legacy_global_gate", False)):
                _set_stage_b_legacy_global_gate_training_mode(model)
            elif bool(getattr(args, "stage_b_v11_fixed_text", False)):
                _set_stage_b_v11_training_mode(model)
            elif bool(getattr(args, "stage_b_v7", False)):
                _set_stage_b_v7_training_mode(model)
    return cached, eval_outputs


@torch.no_grad()
def _compute_stage_b_patch_metrics(outputs, targets, criterion, topk: int):
    pred_logits_patch = outputs.get("pred_logits_patch", None)
    if pred_logits_patch is None:
        return None
    pred_logits_patch = pred_logits_patch.detach().float()
    union_logits = pred_logits_patch if pred_logits_patch.dim() == 2 else pred_logits_patch.max(dim=-1).values

    metrics = {
        "patch_logit_mean": float(union_logits.mean().item()),
        "patch_logit_std": float(union_logits.std(unbiased=False).item()),
        "patch_match_topk_recall": 0.0,
    }
    if not hasattr(criterion, "compute_matching"):
        return metrics

    try:
        match_ctx = criterion.compute_matching(outputs, targets)
    except Exception:
        return metrics

    matched = 0.0
    recalled = 0.0
    k = max(1, int(topk))
    for b, (src_idx, _tgt_idx) in enumerate(match_ctx["all_indices"]):
        if src_idx.numel() == 0:
            continue
        topk_idx = torch.topk(union_logits[b], k=min(k, int(union_logits.shape[1])), largest=True).indices
        recalled += float(torch.isin(src_idx.detach().cpu(), topk_idx.detach().cpu()).float().sum().item())
        matched += float(src_idx.numel())
    if matched > 0:
        metrics["patch_match_topk_recall"] = recalled / matched
    return metrics


@torch.no_grad()
def _maybe_log_stage_b_patch_drift(
    *,
    args,
    model,
    criterion,
    device,
    drift_state,
    samples,
    targets,
    captions,
    patches,
    patch_global,
    patch_mask,
    outputs,
    step: int,
    logger=None,
):
    if not bool(getattr(args, "log_stage_b_patch_drift", False)):
        return drift_state
    if not utils.is_main_process():
        return drift_state

    topk = int(getattr(args, "stage_b_patch_drift_topk", 50))
    interval = int(getattr(args, "stage_b_patch_drift_interval", 100))

    if drift_state is None:
        drift_batch = _clone_stage_b_drift_batch(samples, targets, captions, patches, patch_global, patch_mask)
        cached, eval_outputs = _eval_stage_b_drift_batch(args, model, device, drift_batch)
        baseline_metrics = _compute_stage_b_patch_metrics(eval_outputs, cached["targets"], criterion, topk=topk)
        if baseline_metrics is None:
            return None
        drift_state = {
            "baseline_step": int(step),
            "baseline_metrics": baseline_metrics,
            "batch": drift_batch,
        }
        msg = f"Stage B patch drift baseline @step={step}: {baseline_metrics}"
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)
        return drift_state

    if interval <= 0 or step <= 0 or (step % interval) != 0:
        return drift_state

    cached, eval_outputs = _eval_stage_b_drift_batch(args, model, device, drift_state["batch"])
    cur_metrics = _compute_stage_b_patch_metrics(eval_outputs, cached["targets"], criterion, topk=topk)
    if cur_metrics is None:
        return drift_state
    baseline_metrics = drift_state["baseline_metrics"]
    delta = {k: float(cur_metrics[k] - baseline_metrics[k]) for k in baseline_metrics.keys()}
    msg = (
        f"Stage B patch drift @step={step}: current={cur_metrics} "
        f"baseline={baseline_metrics} delta={delta}"
    )
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)
    return drift_state


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None,
                    scaler: Optional[torch.cuda.amp.GradScaler] = None,
                    start_iter: int = 0,
                    start_optimizer_updates: int = 0,
                    epoch_rng_state=None,
                    runtime_rng_state=None,
                    iter_checkpoint_fn: Optional[Callable[..., None]] = None):
    if scaler is None:
        scaler = _make_grad_scaler(enabled=args.amp)


    model.train()
    if bool(getattr(args, "stage_b_native_patch_category", False)):
        _set_stage_b_native_patch_category_training_mode(model)
    elif bool(getattr(args, "stage_b_data_driven_score", False)):
        _set_stage_b_data_driven_training_mode(
            model,
            str(
                getattr(
                    args,
                    "stage_b_data_driven_train_mode",
                    "rank_patch_only",
                )
            ),
        )
    elif bool(getattr(args, "stage_b_u0_patch_rank", False)):
        if bool(getattr(args, "stage_b_u0_gate_aligned_d13", False)):
            _set_stage_b_u0_gate_aligned_d13_training_mode(model)
        elif bool(getattr(args, "stage_b_u0_gate_aligned_d12", False)):
            _set_stage_b_u0_gate_aligned_d12_training_mode(model)
        elif bool(getattr(args, "stage_b_u0_gate_aligned_d11", False)):
            _set_stage_b_u0_gate_aligned_d11_training_mode(model)
        elif bool(getattr(args, "stage_b_u0_gate_aligned_d10", False)):
            _set_stage_b_u0_gate_aligned_d10_training_mode(model)
        else:
            _set_stage_b_u0_patch_rank_training_mode(model)
    elif bool(getattr(args, "stage_b_gdino_score_adapter", False)):
        _set_stage_b_gdino_adapter_training_mode(
            model,
            str(getattr(args, "stage_b_gdino_adapter_train_mode", "joint")),
        )
    elif bool(getattr(args, "stage_b_legacy_global_gate", False)):
        _set_stage_b_legacy_global_gate_training_mode(model)
    elif bool(getattr(args, "stage_b_v11_fixed_text", False)):
        _set_stage_b_v11_training_mode(model)
    elif bool(getattr(args, "stage_b_v7", False)):
        _set_stage_b_v7_training_mode(model)
    criterion.train()
    data_driven_grad_clip_contract = str(
        getattr(args, "stage_b_data_driven_grad_clip_contract", "") or ""
    ).strip()
    if data_driven_grad_clip_contract not in {
        "",
        "per_optimizer_branch_v1",
    }:
        raise ValueError(
            "stage_b_data_driven_grad_clip_contract must be empty or "
            "'per_optimizer_branch_v1'"
        )
    separate_data_driven_grad_clip = bool(
        getattr(args, "stage_b_data_driven_score", False)
        and data_driven_grad_clip_contract == "per_optimizer_branch_v1"
    )
    separate_dense_duty_grad_clip = bool(
        getattr(args, "stage_b_dense_duty", False)
    )
    if data_driven_grad_clip_contract and not bool(
        getattr(args, "stage_b_data_driven_score", False)
    ):
        raise ValueError(
            "per_optimizer_branch_v1 requires stage_b_data_driven_score=True"
        )
    if separate_data_driven_grad_clip and (
        not math.isfinite(float(max_norm)) or float(max_norm) <= 0.0
    ):
        raise ValueError(
            "per_optimizer_branch_v1 requires clip_max_norm to be finite and positive"
        )
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if separate_dense_duty_grad_clip and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    forward_pack_factor = int(
        getattr(args, "stage_b_dense_duty_forward_pack_factor", 1) or 1
    )
    logical_loss_batch_size = int(
        getattr(args, "stage_b_dense_duty_logical_loss_batch_size", 0) or 0
    )
    if forward_pack_factor < 1:
        raise ValueError("stage_b_dense_duty_forward_pack_factor must be positive")
    if forward_pack_factor > 1:
        if not bool(getattr(args, "stage_b_dense_duty", False)):
            raise ValueError("packed forward is restricted to dense-duty Stage B")
        if logical_loss_batch_size != int(getattr(args, "batch_size", 0)):
            raise ValueError(
                "packed forward requires logical_loss_batch_size == batch_size"
            )
        data_loader = _PackedStageBDataLoader(data_loader, forward_pack_factor)

    _cnt = max(0, int(start_iter))
    optimizer_updates = max(0, int(start_optimizer_updates))
    gradient_accumulation_steps = int(
        getattr(args, "gradient_accumulation_steps", 1) or 1
    )
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    data_loader_len = len(data_loader) if hasattr(data_loader, "__len__") else None
    logical_batches_per_epoch = (
        int(data_loader.logical_length)
        if isinstance(data_loader, _PackedStageBDataLoader)
        else data_loader_len
    )
    if gradient_accumulation_steps > 1 and data_loader_len is None:
        raise ValueError(
            "gradient accumulation requires a DataLoader with a finite __len__ "
            "so the final partial update can be normalized correctly"
        )
    if (
        gradient_accumulation_steps > 1
        and _cnt > 0
        and data_loader_len is not None
        and _cnt < int(data_loader_len)
        and (_cnt % gradient_accumulation_steps) != 0
    ):
        raise ValueError(
            "gradient-accumulation resume must start at an optimizer-step boundary: "
            f"iteration={_cnt}, accumulation_steps={gradient_accumulation_steps}"
        )
    micro_batches_in_update = 0
    current_update_micro_batches = gradient_accumulation_steps
    current_update_logical_batches = (
        gradient_accumulation_steps * forward_pack_factor
    )
    logical_batches_consumed = (
        min(_cnt * forward_pack_factor, int(logical_batches_per_epoch))
        if logical_batches_per_epoch is not None
        else _cnt
    )
    optimizer.zero_grad()
    configured_update_limit = int(getattr(args, "max_train_iters", 0) or 0)
    if configured_update_limit > 0 and optimizer_updates >= configured_update_limit:
        raise GracefulTrainingExit(
            f"Checkpoint already reached max_train_iters={configured_update_limit} "
            "optimizer updates; no additional micro-batch was consumed."
        )
    drift_state = None
    consecutive_amp_skips = 0
    total_amp_skips = 0
    terminal_epoch_reason = None

    data_iterable = data_loader
    if _cnt > 0:
        if epoch_rng_state is not None:
            _restore_rng_state(epoch_rng_state)
        raw_iter = iter(data_loader)
        skipped = 0
        total_len = len(data_loader) if hasattr(data_loader, "__len__") else _cnt
        target_skip = min(_cnt, int(total_len))
        def log_skip(message: str) -> None:
            if logger is not None:
                logger.info(message)
            else:
                print(message, flush=True)

        log_every_batches = 100
        log_every_seconds = 30.0
        last_log_t = time.time()
        skip_start_t = last_log_t
        log_skip(
            f"Resuming epoch {epoch}: skipping {target_skip}/{total_len} already-finished batches. "
            "This is CPU/data-loader work, so GPU usage can stay low until the skip finishes."
        )
        for _ in range(target_skip):
            try:
                next(raw_iter)
                skipped += 1
                now_t = time.time()
                if (
                    skipped <= 10
                    or skipped % log_every_batches == 0
                    or skipped == target_skip
                    or (now_t - last_log_t) >= log_every_seconds
                ):
                    elapsed = max(1e-6, now_t - skip_start_t)
                    batches_per_sec = skipped / elapsed
                    remaining = max(0, target_skip - skipped)
                    eta = remaining / max(1e-6, batches_per_sec)
                    log_skip(
                        f"Resume skip progress: {skipped}/{target_skip} batches "
                        f"({batches_per_sec:.2f} batch/s, eta {int(eta)}s)."
                    )
                    last_log_t = now_t
            except StopIteration:
                break
        msg = f"Resuming epoch {epoch} from iteration {skipped}/{total_len}; skipped completed batches."
        log_skip(msg)
        if runtime_rng_state is not None:
            _restore_rng_state(runtime_rng_state)
        data_iterable = _IteratorWithLen(raw_iter, int(total_len) - skipped)
        _cnt = skipped

    for samples, targets in metric_logger.log_every(data_iterable, print_freq, header, logger=logger):

        samples = samples.to(device)
        logical_batches_in_forward = 1
        patch_only = bool(getattr(args, "patch_only", False))
        stage_b_gdino_adapter = bool(
            getattr(args, "stage_b_gdino_score_adapter", False)
        )
        stage_b_u0_patch_rank = bool(
            getattr(args, "stage_b_u0_patch_rank", False)
        )
        stage_b_data_driven_score = bool(
            getattr(args, "stage_b_data_driven_score", False)
        )
        stage_b_native_patch_category = bool(
            getattr(args, "stage_b_native_patch_category", False)
        )
        gdino_adapter_train_mode = (
            _stage_b_gdino_adapter_train_mode(
                getattr(args, "stage_b_gdino_adapter_train_mode", "joint")
            )
            if stage_b_gdino_adapter
            else None
        )
        raw_targets = list(targets)
        gdino_adapter_pair = None
        data_driven_expression_captions = None
        if patch_only:
            captions = [t.get("caption", "object .") for t in targets]
            patch_mask = None
            patch_global = None
            patches = None
            if all(("patch_global" in t) for t in targets):
                pg0 = targets[0]["patch_global"]
                if (not torch.is_tensor(pg0)) or pg0.dim() not in {1, 2}:
                    raise ValueError("targets[*]['patch_global'] must be (D,) or (K,D) in patch_only mode.")
                if pg0.dim() == 1:
                    patch_global = torch.stack([t["patch_global"] for t in targets], dim=0).to(device, non_blocking=True)
                else:
                    # Pad variable K across batch: (B,Kmax,D) + patch_mask.
                    Kmax = max(int(t["patch_global"].shape[0]) for t in targets)
                    D = int(pg0.shape[1])
                    patch_global = torch.zeros((len(targets), Kmax, D), dtype=pg0.dtype, device=device)
                    patch_mask = torch.zeros((len(targets), Kmax), dtype=torch.bool, device=device)
                    for i, t in enumerate(targets):
                        pg = t["patch_global"]
                        ki = int(pg.shape[0])
                        patch_global[i, :ki] = pg.to(device, non_blocking=True)
                        patch_mask[i, :ki] = True
            elif all(("patches" in t) for t in targets):
                p0 = targets[0]["patches"]
                if (not torch.is_tensor(p0)) or p0.dim() != 4:
                    raise ValueError("targets[*]['patches'] must be (K,3,H,W) in patch_only mode.")
                Kmax = max(int(t["patches"].shape[0]) for t in targets)
                C, H, W = map(int, p0.shape[1:])
                patches = torch.zeros((len(targets), Kmax, C, H, W), dtype=p0.dtype, device=device)
                patch_mask = torch.zeros((len(targets), Kmax), dtype=torch.bool, device=device)
                for i, t in enumerate(targets):
                    p = t["patches"]
                    ki = int(p.shape[0])
                    patches[i, :ki] = p.to(device, non_blocking=True)
                    patch_mask[i, :ki] = True
            else:
                patches = torch.stack([t["patch"] for t in targets], dim=0).to(device, non_blocking=True)
            filtered_targets = []
            Kmax = int(patch_mask.shape[1]) if (patch_mask is not None and torch.is_tensor(patch_mask)) else None
            for t in targets:
                t2 = {k: v.to(device) for k, v in t.items() if torch.is_tensor(v) and k not in {"patch", "patches", "patch_global"}}
                for text_key in (
                    "stage_a_caption",
                    "verifier_caption",
                    "caption",
                    "cap_list",
                ):
                    if text_key in t:
                        t2[text_key] = t[text_key]
                _preserve_stage_b_v21_trace_metadata(t, t2)
                if "table_b_id" in t:
                    for contract_key in (
                        "table_b_id",
                        "table_b_audit_sha256",
                        "tn_scope",
                    ):
                        t2[contract_key] = t[contract_key]
                if Kmax is not None and "support_classes" in t2 and torch.is_tensor(t2["support_classes"]):
                    sc = t2["support_classes"].view(-1)
                    if sc.numel() < Kmax:
                        pad = torch.full((Kmax - int(sc.numel()),), -1, dtype=sc.dtype, device=sc.device)
                        t2["support_classes"] = torch.cat([sc, pad], dim=0)
                filtered_targets.append(t2)
                if "rank_positive_captions" in t:
                    t2["rank_positive_captions"] = t["rank_positive_captions"]
            targets = filtered_targets
        else:
            captions = [t["caption"] for t in targets]
            cap_list = [t["cap_list"] for t in targets]
            if stage_b_native_patch_category:
                if data_driven_expression_captions is not None:
                    raise RuntimeError(
                        "native patch-category routing must not rewrite full-text captions"
                    )
                for row_index, (target, caption, expressions) in enumerate(
                    zip(raw_targets, captions, cap_list)
                ):
                    if caption != target.get("caption"):
                        raise RuntimeError(
                            "native patch-category caption routing changed row "
                            f"{row_index}"
                        )
                    if expressions != target.get("cap_list"):
                        raise RuntimeError(
                            "native patch-category cap_list routing changed row "
                            f"{row_index}"
                        )
            elif stage_b_data_driven_score:
                data_driven_mode = str(
                    getattr(
                        args,
                        "stage_b_data_driven_train_mode",
                        "rank_patch_only",
                    )
                ).strip()
                if data_driven_mode == "rank_patch_only":
                    rank_supervision = str(
                        getattr(
                            args,
                            "stage_b_data_driven_rank_supervision",
                            "all_nonpositive_negative_v1",
                        )
                    ).strip().lower()
                    if rank_supervision in {
                        "official_same_image_same_category_assignment_v1",
                        "role_routed_official_assignment_top1_v1",
                        "role_routed_official_assignment_all_exclusive_nonowned_v2",
                    }:
                        captions, data_driven_expression_captions = (
                            _build_stage_b_data_driven_assignment_captions(
                                raw_targets
                            )
                        )
                    else:
                        captions, data_driven_expression_captions = (
                            _build_stage_b_data_driven_positive_captions(
                                raw_targets
                            )
                        )
                elif data_driven_mode == "confidence_pair":
                    captions, data_driven_expression_captions = (
                        _build_stage_b_data_driven_pair_captions(raw_targets)
                    )
                else:
                    raise ValueError(
                        f"unknown data-driven train mode {data_driven_mode!r}"
                    )
            elif stage_b_gdino_adapter:
                if gdino_adapter_train_mode == "rank_only":
                    captions = _build_stage_b_gdino_adapter_rank_captions(
                        raw_targets,
                        u0_patch_episode=stage_b_u0_patch_rank,
                    )
                else:
                    gdino_adapter_pair = _build_stage_b_gdino_adapter_pair_captions(
                        raw_targets,
                        str(getattr(args, "stage_b_gdino_tn_scope", "")),
                    )
                    captions = gdino_adapter_pair[0]
            patches = None
            if (
                (
                    not stage_b_gdino_adapter
                    or stage_b_u0_patch_rank
                    or stage_b_data_driven_score
                )
                and all(("patch" in t) for t in targets)
            ):
                patches = torch.stack([t["patch"] for t in targets], dim=0).to(device, non_blocking=True)
            patch_global = None
            if (
                (
                    not stage_b_gdino_adapter
                    or stage_b_u0_patch_rank
                    or stage_b_data_driven_score
                )
                and all(("patch_global" in t) for t in targets)
            ):
                patch_global = torch.stack([t["patch_global"] for t in targets], dim=0).to(device, non_blocking=True)
            if (
                (
                    stage_b_u0_patch_rank
                    or stage_b_data_driven_score
                    or stage_b_native_patch_category
                )
                and patches is None
                and patch_global is None
            ):
                raise RuntimeError(
                    "Stage-B patch supervision requires one support patch per row"
                )
            targets = [
                {
                    k: v.to(device)
                    for k, v in t.items()
                    if torch.is_tensor(v) and k not in {"patch", "patch_global"}
                }
                for t in targets
            ]
        with torch.cuda.amp.autocast(enabled=args.amp):
            if patch_only:
                if bool(getattr(args, "stage_b_legacy_global_gate", False)):
                    positive_captions, negative_captions = (
                        _build_stage_b_legacy_gate_pair_captions(targets)
                    )
                    paired_samples = NestedTensor(
                        torch.cat((samples.tensors, samples.tensors), dim=0),
                        (
                            torch.cat((samples.mask, samples.mask), dim=0)
                            if samples.mask is not None
                            else None
                        ),
                    )

                    def repeat_pair_batch(value):
                        return (
                            torch.cat((value, value), dim=0)
                            if torch.is_tensor(value)
                            else None
                        )

                    paired_outputs = model(
                        paired_samples,
                        targets=targets + targets,
                        captions=positive_captions + negative_captions,
                        patches=repeat_pair_batch(patches),
                        patch_global=repeat_pair_batch(patch_global),
                        patch_mask=repeat_pair_batch(patch_mask),
                        patch_only=True,
                        disable_patch_dn=True,
                        patch_only_compute_text_logits=True,
                    )
                    pair_batch_size = len(targets)
                    outputs = _split_stage_b_legacy_gate_batch(
                        paired_outputs, pair_batch_size, take_negative=False
                    )
                    negative_outputs = _split_stage_b_legacy_gate_batch(
                        paired_outputs, pair_batch_size, take_negative=True
                    )
                    outputs["stage_b_legacy_global_tn_outputs"] = negative_outputs
                    outputs["stage_b_legacy_positive_captions"] = positive_captions
                    outputs["stage_b_legacy_tn_captions"] = negative_captions
                    loss_dict = criterion(outputs, targets)
                elif bool(getattr(args, "stage_b_v11_fixed_text", False)):
                    stage_a_captions = [
                        t.get("stage_a_caption", t.get("caption", "object ."))
                        for t in targets
                    ]
                    (
                        expression_captions,
                        expression_valid_mask,
                    ) = _build_stage_b_v11_expression_slots(targets, device)
                    candidate_topk = int(
                        getattr(args, "stage_b_v11_candidate_topk", 50)
                    )
                    exact_candidate_replay = (
                        _build_stage_b_v15_exact_candidate_replay(
                            targets, topk=candidate_topk, device=device
                        )
                    )
                    direct_trace_roles = bool(
                        getattr(args, "stage_b_dense_duty", False)
                    ) and str(
                        getattr(args, "stage_b_v21_token_objective", "off")
                    ).strip().lower() in {
                        "edit_bce",
                        "edit_focal",
                        "edit_bce_group_balanced",
                    }
                    edit_traces = None
                    if direct_trace_roles:
                        edit_traces = _build_stage_b_v21_certified_edit_traces(
                            targets
                        )
                    outputs = model(
                        samples,
                        targets=targets,
                        captions=stage_a_captions,
                        patches=patches,
                        patch_global=patch_global,
                        patch_mask=patch_mask,
                        patch_only=True,
                        disable_patch_dn=True,
                        patch_only_compute_text_logits=False,
                        stage_b_v11_expression_captions=expression_captions,
                        stage_b_v11_expression_valid_mask=expression_valid_mask,
                        stage_b_v21_edit_traces=edit_traces,
                        stage_b_v11_candidate_topk=candidate_topk,
                        stage_b_v15_exact_candidate_mask=(
                            exact_candidate_replay["mask"]
                            if exact_candidate_replay is not None
                            else None
                        ),
                        stage_b_v15_exact_candidate_indices=(
                            exact_candidate_replay["indices"]
                            if exact_candidate_replay is not None
                            else None
                        ),
                        stage_b_v15_exact_candidate_boxes=(
                            exact_candidate_replay["boxes"]
                            if exact_candidate_replay is not None
                            else None
                        ),
                        stage_b_v15_exact_candidate_box_atol=(
                            exact_candidate_replay["box_atol"]
                            if exact_candidate_replay is not None
                            else None
                        ),
                        stage_b_v11_expression_microbatch=int(
                            getattr(args, "stage_b_v11_expression_microbatch", 8)
                        ),
                        stage_b_v11_assert_fixed_candidates=bool(
                            getattr(args, "stage_b_v11_assert_fixed_candidates", False)
                        ),
                    )
                    from models.GroundingDINO.stage_b_fixed_text_criterion import (
                        candidate_max_iou,
                    )

                    decoupled_confidence = bool(
                        getattr(args, "stage_b_v15_decoupled_confidence", False)
                    )
                    from models.GroundingDINO.stage_b_fixed_text_scorer import (
                        normalize_stage_b_score_ownership,
                        select_stage_b_rank_confidence_logits,
                    )

                    score_ownership = normalize_stage_b_score_ownership(
                        getattr(args, "stage_b_v22_score_ownership", "")
                    )
                    candidate_logits, confidence_logits = (
                        select_stage_b_rank_confidence_logits(
                            outputs,
                            score_ownership=score_ownership,
                            legacy_decoupled_confidence=decoupled_confidence,
                            legacy_validity_head=bool(
                                getattr(args, "stage_b_v14_validity_head", False)
                            ),
                        )
                    )
                    predicate_logits = outputs[
                        "stage_b_v11_final_predicate_logits"
                    ]
                    if candidate_logits.shape[-1] != 2:
                        raise RuntimeError(
                            "Stage B v11 training requires exactly two expression slots"
                        )
                    global_trust_veto_contract = str(
                        getattr(
                            args,
                            "stage_b_dense_duty_confidence_head_gradient_contract",
                            "",
                        )
                    ).strip().lower()
                    (
                        positive_loss_confidence_logits,
                        negative_loss_confidence_logits,
                    ) = _select_dense_duty_confidence_loss_logits(
                        outputs=outputs,
                        candidate_logits=candidate_logits,
                        confidence_logits=confidence_logits,
                        head_gradient_contract=global_trust_veto_contract,
                    )
                    (
                        sample_positive_confidence_logits,
                        sample_tn_confidence_logits,
                    ) = _select_dense_duty_sample_confidence_logits(
                        outputs=outputs,
                        candidate_logits=candidate_logits,
                        head_gradient_contract=global_trust_veto_contract,
                    )
                    if predicate_logits.shape != candidate_logits.shape:
                        raise RuntimeError(
                            "Stage B v11 predicate logits must align with phrase logits"
                        )
                    candidate_ious = candidate_max_iou(
                        outputs["stage_b_v11_candidate_boxes"], targets
                    )
                    benchmark_global_tn = bool(
                        getattr(args, "stage_b_v14_global_tn_all_candidates", False)
                    )
                    global_tn_logits = (
                        candidate_logits[..., 1] if benchmark_global_tn else None
                    )
                    global_tn_confidence_logits = (
                        negative_loss_confidence_logits
                        if benchmark_global_tn
                        and sample_tn_confidence_logits is None
                        else None
                    )
                    token_edit_carrier_logits = (
                        _stage_b_token_edit_carrier_logits(outputs)
                    )
                    token_role_carrier_logits = (
                        _stage_b_token_role_carrier_logits(outputs)
                    )
                    global_tn_verified = None
                    confidence_ablation_eligible = None
                    confidence_tn_eligible = expression_valid_mask[:, 1]
                    if benchmark_global_tn:
                        paired_tn = expression_valid_mask[:, 1]
                        confidence_tn_scope = str(
                            getattr(
                                args,
                                "stage_b_dense_duty_confidence_tn_scope",
                                "all_verified_v1",
                            )
                        ).strip().lower()
                        if confidence_tn_scope == "direct_trace_valid_v1":
                            direct_trace_valid = outputs.get(
                                "stage_b_v21_direct_trace_valid"
                            )
                            if direct_trace_valid is None:
                                raise RuntimeError(
                                    "direct-trace confidence scope requires runtime "
                                    "trace validity"
                                )
                            direct_trace_valid = torch.as_tensor(
                                direct_trace_valid,
                                device=device,
                                dtype=torch.bool,
                            )
                            if tuple(direct_trace_valid.shape) != tuple(
                                paired_tn.shape
                            ):
                                raise RuntimeError(
                                    "direct-trace confidence validity must have shape (B,)"
                                )
                            confidence_tn_eligible = paired_tn & direct_trace_valid
                        elif confidence_tn_scope == "all_verified_v1":
                            confidence_tn_eligible = paired_tn
                        else:
                            raise RuntimeError(
                                "unknown dense-duty confidence TN scope: "
                                f"{confidence_tn_scope!r}"
                            )
                        if decoupled_confidence or bool(score_ownership):
                            verified_rows = torch.stack(
                                [
                                    t.get(
                                        "global_tn_verified",
                                        torch.zeros(1, dtype=torch.bool, device=device),
                                    ).reshape(-1)[0]
                                    for t in targets
                                ]
                            ).to(device=device, dtype=torch.bool)
                            confidence_ablation_eligible = (
                                build_confidence_ablation_eligible(
                                    args,
                                    targets,
                                    paired_tn,
                                    device=device,
                                )
                            )
                            confidence_eligible = verified_rows
                            if confidence_ablation_eligible is not None:
                                confidence_eligible = (
                                    confidence_eligible
                                    | confidence_ablation_eligible
                                )
                            unverified = paired_tn & ~confidence_eligible
                            if bool(unverified.any().item()):
                                bad = unverified.nonzero(as_tuple=False).flatten().tolist()
                                raise RuntimeError(
                                    "Stage B score-confidence training received paired TN "
                                    "rows without global_tn_verified=True at batch indices "
                                    f"{bad}"
                                )
                            global_tn_verified = paired_tn & verified_rows
                        else:
                            # Historical v14 benchmark compatibility mode.
                            global_tn_verified = paired_tn
                    token_supervision_valid = torch.as_tensor(
                        [
                            _strict_target_bool(
                                target,
                                "stage_b_v21_token_supervision_valid",
                            )
                            for target in targets
                        ],
                        dtype=torch.bool,
                        device=device,
                    )
                    compact_candidate_mask = outputs.get(
                        "stage_b_dense_duty_candidate_eligible_mask"
                    )
                    positive_candidate_mask = (
                        compact_candidate_mask[..., 0]
                        if compact_candidate_mask is not None
                        else None
                    )
                    local_tn_candidate_mask = (
                        compact_candidate_mask[..., 1]
                        if compact_candidate_mask is not None
                        else expression_valid_mask[:, 1:2]
                    )
                    positive_confidence_trust_logits = (
                        _select_dense_duty_positive_confidence_trust_logits(
                            outputs=outputs,
                            candidate_logits=candidate_logits,
                            sample_positive_confidence_logits=(
                                sample_positive_confidence_logits
                            ),
                            decoupled_confidence=decoupled_confidence,
                            positive_trust_contract=getattr(
                                args,
                                "stage_b_dense_duty_positive_trust_contract",
                                "pool_residual_v1",
                            ),
                            head_gradient_contract=global_trust_veto_contract,
                        )
                    )
                    loss_dict, logical_batches_in_forward = (
                        _call_fixed_text_criterion_in_logical_batches(
                        criterion,
                        logical_batch_size=(
                            logical_loss_batch_size
                            if logical_loss_batch_size > 0
                            else len(targets)
                        ),
                        candidate_logits=candidate_logits[..., 0],
                        candidate_ious=candidate_ious,
                        candidate_mask=positive_candidate_mask,
                        local_tn_logits=candidate_logits[..., 1],
                        confidence_logits=(
                            positive_loss_confidence_logits
                        ),
                        sample_positive_confidence_logits=(
                            sample_positive_confidence_logits
                        ),
                        sample_tn_confidence_logits=(
                            sample_tn_confidence_logits
                        ),
                        positive_confidence_gate_logits=(
                            positive_confidence_trust_logits
                        ),
                        local_tn_confidence_logits=(
                            negative_loss_confidence_logits
                        ),
                        local_tn_mask=local_tn_candidate_mask,
                        positive_predicate_logits=predicate_logits[..., 0],
                        local_tn_predicate_logits=predicate_logits[..., 1],
                        predicate_pair_valid=outputs[
                            "stage_b_v11_predicate_pair_valid"
                        ],
                        global_tn_logits=global_tn_logits,
                        global_tn_confidence_logits=global_tn_confidence_logits,
                        global_tn_verified=global_tn_verified,
                        global_tn_candidate_mask=(
                            local_tn_candidate_mask
                            if benchmark_global_tn
                            else None
                        ),
                        confidence_ablation_eligible=(
                            confidence_ablation_eligible
                        ),
                        confidence_tn_train_eligible=confidence_tn_eligible,
                        token_edit_carrier_logits=token_edit_carrier_logits,
                        token_role_carrier_logits=token_role_carrier_logits,
                        token_logits=outputs["stage_b_v11_final_token_logits"],
                        score_token_mask=outputs["stage_b_v15_score_token_mask"],
                        predicate_token_mask=outputs[
                            "stage_b_v11_predicate_token_mask"
                        ],
                        expression_valid_mask=outputs[
                            "stage_b_v11_expression_valid_mask"
                        ],
                        token_supervision_valid=token_supervision_valid,
                        token_positive_mask=outputs.get(
                            "stage_b_v21_positive_token_mask"
                        ),
                        token_shared_mask=outputs.get(
                            "stage_b_v21_shared_token_mask"
                        ),
                        token_changed_mask=outputs.get(
                            "stage_b_v21_changed_token_mask"
                        ),
                        token_direct_trace_valid=outputs.get(
                            "stage_b_v21_direct_trace_valid"
                        ),
                        token_residual_logits=outputs.get(
                            "stage_b_dense_duty_final_confidence_token_residual_logits"
                        ),
                        score_word_group_ids=outputs.get(
                            "stage_b_dense_duty_score_word_group_ids"
                        ),
                        positive_reference_base_logits=(
                            outputs.get(
                                "stage_b_dense_duty_reference_base_logits"
                            )[..., 0]
                            if outputs.get(
                                "stage_b_dense_duty_reference_base_logits"
                            ) is not None
                            else None
                        ),
                        confidence_veto_carrier_indices=(
                            outputs.get(
                                "stage_b_dense_duty_final_confidence_veto_carrier_index"
                            )
                        ),
                        confidence_mismatch_gate=outputs.get(
                            "stage_b_dense_duty_confidence_deployed_routing_gate",
                            outputs.get(
                                "stage_b_dense_duty_confidence_mismatch_gate"
                            ),
                        ),
                        confidence_veto_coverage=outputs.get(
                            "stage_b_dense_duty_confidence_veto_coverage"
                        ),
                        confidence_base_logits=outputs.get(
                            "stage_b_dense_duty_confidence_base_logits"
                        ),
                        )
                    )
                    confidence_delta = outputs.get(
                        "stage_b_dense_duty_confidence_delta_logits"
                    )
                    confidence_mismatch_gate = outputs.get(
                        "stage_b_dense_duty_confidence_deployed_routing_gate",
                        outputs.get(
                            "stage_b_dense_duty_confidence_mismatch_gate"
                        ),
                    )
                    if confidence_delta is not None:
                        valid_expression = outputs[
                            "stage_b_v11_expression_valid_mask"
                        ].to(dtype=torch.bool)
                        for slot, label in ((0, "positive"), (1, "tn")):
                            slot_valid = valid_expression[:, slot]
                            if slot == 1:
                                slot_valid = slot_valid & confidence_tn_eligible
                            if bool(slot_valid.any().item()):
                                loss_dict[
                                    f"stage_b_dense_confidence_{label}_delta_mean"
                                ] = confidence_delta[:, slot][slot_valid].detach().mean()
                    if (
                        confidence_mismatch_gate is not None
                        and compact_candidate_mask is not None
                    ):
                        for slot, label in ((0, "positive"), (1, "tn")):
                            gate_valid = (
                                compact_candidate_mask[..., slot]
                                if slot == 0
                                else (
                                    compact_candidate_mask[..., slot]
                                    & confidence_tn_eligible[:, None]
                                )
                            )
                            if bool(gate_valid.any().item()):
                                loss_dict[
                                    f"stage_b_dense_confidence_{label}_veto_gate_mean"
                                ] = confidence_mismatch_gate[..., slot][
                                    gate_valid
                                ].detach().mean()
                    for output_name, metric_name in (
                        (
                            "stage_b_dense_duty_confidence_veto_coverage",
                            "veto_coverage",
                        ),
                        (
                            "stage_b_dense_duty_confidence_veto_sample_gate",
                            "veto_sample_gate",
                        ),
                    ):
                        metric = outputs.get(output_name)
                        if metric is None:
                            continue
                        valid_expression = outputs[
                            "stage_b_v11_expression_valid_mask"
                        ].to(dtype=torch.bool)
                        for slot, label in ((0, "positive"), (1, "tn")):
                            metric_valid = valid_expression[:, slot]
                            if slot == 1:
                                metric_valid = metric_valid & confidence_tn_eligible
                            if bool(metric_valid.any().item()):
                                loss_dict[
                                    f"stage_b_dense_confidence_{label}_{metric_name}_mean"
                                ] = metric[:, slot][metric_valid].detach().mean()
                    veto_ceiling = outputs.get(
                        "stage_b_dense_duty_confidence_veto_absolute_ceiling"
                    )
                    if veto_ceiling is not None:
                        loss_dict["stage_b_dense_confidence_veto_ceiling"] = (
                            veto_ceiling.detach().mean()
                        )
                    outputs["stage_a_captions"] = stage_a_captions
                    outputs["stage_b_v11_expression_captions"] = expression_captions
                    outputs["stage_b_v11_candidate_ious"] = candidate_ious
                elif bool(getattr(args, "stage_b_v7", False)):
                    stage_a_captions = [
                        t.get("stage_a_caption", t.get("caption", "object ."))
                        for t in targets
                    ]
                    verifier_captions = [
                        t.get("verifier_caption", t.get("caption", "object ."))
                        for t in targets
                    ]
                    phrase_to_token_mask = _stack_stage_b_v7_mask(targets, "phrase_to_token_mask", device)
                    canonical_to_token_mask = _stack_stage_b_v7_mask(targets, "canonical_to_token_mask", device)
                    content_to_token_mask = _stack_stage_b_v7_mask(targets, "content_to_token_mask", device)
                    attr_pos_to_token_mask = _stack_stage_b_v7_mask(targets, "attr_pos_to_token_mask", device)
                    attr_neg_to_token_mask = _stack_stage_b_v7_mask(targets, "attr_neg_to_token_mask", device)
                    negative_to_token_mask = _stack_stage_b_v7_mask(targets, "negative_to_token_mask", device)
                    attr_neg_weight_mask = _stack_stage_b_v7_mask(targets, "attr_neg_weight_mask", device)
                    is_tn = _stack_stage_b_v7_mask(targets, "is_tn", device)
                    verifier_pair_stride = _stack_stage_b_v7_mask(targets, "verifier_pair_stride", device)
                    verifier_num_patch_slots = _stack_stage_b_v7_mask(targets, "verifier_num_patch_slots", device)
                    outputs = model(
                        samples,
                        targets=targets,
                        captions=stage_a_captions,
                        patches=patches,
                        patch_global=patch_global,
                        patch_mask=patch_mask,
                        patch_only=True,
                        disable_patch_dn=True,
                        patch_only_compute_text_logits=False,
                        return_stage_b_v7_features=True,
                        stage_b_v7_verifier_captions=verifier_captions,
                        phrase_to_token_mask=phrase_to_token_mask,
                        canonical_to_token_mask=canonical_to_token_mask,
                        content_to_token_mask=content_to_token_mask,
                        attr_pos_to_token_mask=attr_pos_to_token_mask,
                        attr_neg_to_token_mask=attr_neg_to_token_mask,
                        negative_to_token_mask=negative_to_token_mask,
                        attr_neg_weight_mask=attr_neg_weight_mask,
                        is_tn=is_tn,
                        verifier_pair_stride=verifier_pair_stride,
                        verifier_num_patch_slots=verifier_num_patch_slots,
                    )
                    outputs["stage_a_captions"] = stage_a_captions
                    outputs["verifier_captions"] = verifier_captions
                    outputs["phrase_to_token_mask"] = phrase_to_token_mask
                    outputs["canonical_to_token_mask"] = canonical_to_token_mask
                    outputs["content_to_token_mask"] = content_to_token_mask
                    outputs["attr_pos_to_token_mask"] = attr_pos_to_token_mask
                    outputs["attr_neg_to_token_mask"] = attr_neg_to_token_mask
                    outputs["negative_to_token_mask"] = negative_to_token_mask
                    outputs["attr_neg_weight_mask"] = attr_neg_weight_mask
                    outputs["is_tn"] = is_tn
                    outputs["verifier_pair_stride"] = verifier_pair_stride
                    outputs["verifier_num_patch_slots"] = verifier_num_patch_slots
                    loss_dict = criterion(outputs, targets)
                else:
                    # Pass `targets` so the model can optionally build GT-guided (DN) queries in patch-only mode.
                    stage_b_mask_kwargs = {}
                    for mask_key in (
                        "phrase_to_token_mask",
                        "canonical_to_token_mask",
                        "content_to_token_mask",
                        "attr_pos_to_token_mask",
                        "attr_neg_to_token_mask",
                        "phrase_semantic_token_mask",
                    ):
                        if all(mask_key in t for t in targets):
                            values = [t[mask_key] for t in targets]
                            if all(torch.is_tensor(v) for v in values):
                                if len({tuple(v.shape) for v in values}) == 1:
                                    stage_b_mask_kwargs[mask_key] = torch.stack(values, dim=0).to(
                                        device, non_blocking=True
                                    )
                                elif all(v.dim() == 2 for v in values):
                                    kmax = max(int(v.shape[0]) for v in values)
                                    tmax = max(int(v.shape[1]) for v in values)
                                    padded = values[0].new_zeros((len(values), kmax, tmax))
                                    for i, v in enumerate(values):
                                        padded[i, : int(v.shape[0]), : int(v.shape[1])] = v
                                    stage_b_mask_kwargs[mask_key] = padded.to(device, non_blocking=True)
                    rank_subbatch = _build_stage_b_rank_subbatch(
                        args,
                        samples,
                        targets,
                        captions,
                        patches,
                        patch_global,
                        patch_mask,
                    )
                    has_rank_pairs = bool(rank_subbatch is not None and rank_subbatch.get("indices"))
                    global_has_rank_pairs = _sync_bool_any(has_rank_pairs, device)
                    outputs = model(
                        samples,
                        captions=captions,
                        patches=patches,
                        patch_global=patch_global,
                        patch_mask=patch_mask,
                        patch_only=True,
                        disable_patch_dn=global_has_rank_pairs,
                        patch_only_compute_text_logits=bool(getattr(args, "patch_only_compute_text_logits", False)),
                        **stage_b_mask_kwargs,
                    )
                    if rank_subbatch is not None:
                        outputs["rank_candidate_tn_count"] = torch.as_tensor(
                            float(rank_subbatch.get("rank_candidate_tn_count", 0)), device=device
                        )
                        outputs["rank_missing_positive_count"] = torch.as_tensor(
                            float(rank_subbatch.get("rank_missing_positive_count", 0)), device=device
                        )
                        outputs["rank_invalid_positive_count"] = torch.as_tensor(
                            float(rank_subbatch.get("rank_invalid_positive_count", 0)), device=device
                        )
                        outputs["rank_truncated_positive_count"] = torch.as_tensor(
                            float(rank_subbatch.get("rank_truncated_positive_count", 0)), device=device
                        )
                    run_rank_pos_forward = bool(global_has_rank_pairs)
                    rank_forward_subbatch = rank_subbatch if has_rank_pairs else None
                    if run_rank_pos_forward and not has_rank_pairs:
                        rank_forward_subbatch = _build_stage_b_dummy_rank_subbatch(
                            samples,
                            targets,
                            patches,
                            patch_global,
                            patch_mask,
                        )
                    if run_rank_pos_forward and rank_forward_subbatch is not None:
                        rank_pos_outputs = model(
                            rank_forward_subbatch["samples"],
                            targets=rank_forward_subbatch["targets"],
                            captions=rank_forward_subbatch["captions"],
                            patches=rank_forward_subbatch["patches"],
                            patch_global=rank_forward_subbatch["patch_global"],
                            patch_mask=rank_forward_subbatch["patch_mask"],
                            patch_only=True,
                            disable_patch_dn=True,
                            patch_only_compute_text_logits=bool(getattr(args, "patch_only_compute_text_logits", False)),
                            phrase_to_token_mask=torch.stack(
                                [t["phrase_to_token_mask"] for t in rank_forward_subbatch["targets"]], dim=0
                            ),
                            canonical_to_token_mask=torch.stack(
                                [t["canonical_to_token_mask"] for t in rank_forward_subbatch["targets"]], dim=0
                            ),
                        )
                        if has_rank_pairs:
                            outputs["rank_pos_outputs"] = rank_pos_outputs
                            outputs["rank_pos_targets"] = rank_subbatch["targets"]
                            outputs["rank_pair_map"] = torch.as_tensor(
                                rank_subbatch["indices"], dtype=torch.long, device=device
                            )
                        else:
                            dummy_zero = _zero_from_stage_b_outputs(rank_pos_outputs)
                            if dummy_zero is not None:
                                outputs["rank_pos_dummy_loss"] = dummy_zero
                    loss_dict = criterion(outputs, targets)
            else:
                if stage_b_data_driven_score:
                    if data_driven_expression_captions is None:
                        raise RuntimeError(
                            "data-driven expression captions were not initialized"
                        )
                    outputs = model(
                        samples,
                        captions=captions,
                        patches=patches,
                        patch_global=patch_global,
                        stage_b_data_driven_expression_captions=(
                            data_driven_expression_captions
                        ),
                    )
                    criterion_targets = (
                        raw_targets
                        if str(
                            getattr(
                                args,
                                "stage_b_data_driven_train_mode",
                                "rank_patch_only",
                            )
                        ).strip()
                        == "confidence_pair"
                        else targets
                    )
                    if criterion_targets is raw_targets:
                        criterion_targets = [dict(target) for target in raw_targets]
                        for target, pair in zip(
                            criterion_targets, data_driven_expression_captions
                        ):
                            target[
                                "stage_b_data_driven_expression_captions"
                            ] = list(pair)
                    loss_dict = criterion(outputs, criterion_targets)
                elif stage_b_gdino_adapter:
                    if gdino_adapter_train_mode == "rank_only":
                        if gdino_adapter_pair is not None:
                            raise RuntimeError(
                                "rank_only must not initialize a paired TN batch"
                            )
                        outputs = model(
                            samples,
                            captions=captions,
                            patches=patches,
                            patch_global=patch_global,
                        )
                    else:
                        if gdino_adapter_pair is None:
                            raise RuntimeError(
                                "GDINO adapter pair extraction was not initialized"
                            )
                        positive_captions, negative_captions, scope_codes = (
                            gdino_adapter_pair
                        )
                        outputs = _forward_stage_b_gdino_adapter_pair(
                            model,
                            samples,
                            targets,
                            positive_captions,
                            negative_captions,
                            scope_codes,
                        )
                    loss_dict = criterion(outputs, targets)
                else:
                    gdino_mask_kwargs = {}
                    for mask_key in (
                        "phrase_to_token_mask",
                        "canonical_to_token_mask",
                        "content_to_token_mask",
                        "attr_pos_to_token_mask",
                        "attr_neg_to_token_mask",
                        "negative_to_token_mask",
                        "is_tn",
                    ):
                        if all(mask_key in t for t in targets):
                            values = [t[mask_key] for t in targets]
                            if all(torch.is_tensor(v) for v in values):
                                if len({tuple(v.shape) for v in values}) == 1:
                                    gdino_mask_kwargs[mask_key] = torch.stack(values, dim=0).to(
                                        device, non_blocking=True
                                    )
                                elif all(v.dim() == 2 for v in values):
                                    kmax = max(int(v.shape[0]) for v in values)
                                    tmax = max(int(v.shape[1]) for v in values)
                                    padded = values[0].new_zeros((len(values), kmax, tmax))
                                    for i, v in enumerate(values):
                                        padded[i, : int(v.shape[0]), : int(v.shape[1])] = v
                                    gdino_mask_kwargs[mask_key] = padded.to(device, non_blocking=True)
                                elif all(v.dim() == 1 for v in values):
                                    kmax = max(int(v.shape[0]) for v in values)
                                    padded = values[0].new_zeros((len(values), kmax))
                                    for i, v in enumerate(values):
                                        padded[i, : int(v.shape[0])] = v
                                    gdino_mask_kwargs[mask_key] = padded.to(device, non_blocking=True)
                    outputs = model(
                        samples,
                        captions=captions,
                        patches=patches,
                        patch_global=patch_global,
                        **gdino_mask_kwargs,
                    )
                    loss_dict = criterion(outputs, targets, cap_list, captions)

            weight_dict = criterion.weight_dict

            if bool(getattr(args, "stage_b_v11_fixed_text", False)):
                loss_dict.update(
                    _stage_b_v22_gradient_diagnostic(
                        args,
                        model,
                        loss_dict,
                        weight_dict,
                        step=_cnt,
                    )
                )

            losses = _sum_weighted_training_losses(loss_dict, weight_dict)
            if patch_only:
                _maybe_save_patch_sanity(
                    args=args, samples=samples, targets=targets, outputs=outputs, criterion=criterion, epoch=epoch, step=_cnt
                )
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        if micro_batches_in_update == 0:
            if data_loader_len is None:
                current_update_micro_batches = gradient_accumulation_steps
            else:
                remaining_micro_batches = max(1, int(data_loader_len) - _cnt)
                current_update_micro_batches = min(
                    gradient_accumulation_steps, remaining_micro_batches
                )
            if logical_batches_per_epoch is None:
                current_update_logical_batches = current_update_micro_batches
            else:
                remaining_logical_batches = max(
                    1,
                    int(logical_batches_per_epoch) - logical_batches_consumed,
                )
                current_update_logical_batches = min(
                    gradient_accumulation_steps * forward_pack_factor,
                    remaining_logical_batches,
                )
        micro_batches_in_update += 1
        logical_batches_consumed += int(logical_batches_in_forward)
        optimizer_step_boundary = (
            micro_batches_in_update >= current_update_micro_batches
        )
        backward_losses = _scale_loss_for_logical_accumulation(
            losses,
            logical_batches_in_forward=logical_batches_in_forward,
            logical_batches_in_update=current_update_logical_batches,
        )

        # Accumulate scaled gradients and update only at an accumulation boundary.
        amp_step_skipped = 0.0
        amp_scale = float(scaler.get_scale()) if args.amp else 1.0
        branch_grad_norms = {}
        separate_v15_grad_clip = bool(
            getattr(args, "stage_b_v15_separate_grad_clip", False)
        )
        separate_gdino_adapter_grad_clip = bool(
            getattr(args, "stage_b_gdino_score_adapter", False)
        )
        separate_u0_patch_rank_grad_clip = bool(
            getattr(args, "stage_b_u0_patch_rank", False)
        )
        if args.amp:
            scaler.scale(backward_losses).backward()
        else:
            backward_losses.backward()

        commit_tail_queue = getattr(criterion, "commit_tail_queue", None)
        if not optimizer_step_boundary and callable(commit_tail_queue):
            defer_tail_queue = getattr(criterion, "defer_tail_queue_payload", None)
            if not callable(defer_tail_queue):
                raise RuntimeError(
                    "criterion.commit_tail_queue requires defer_tail_queue_payload "
                    "when gradient_accumulation_steps > 1"
                )
            defer_tail_queue()

        optimizer_step_succeeded = False
        if optimizer_step_boundary:
            if args.amp:
                scale_before = float(scaler.get_scale())
                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    if separate_data_driven_grad_clip:
                        branch_grad_norms = (
                            _clip_stage_b_data_driven_optimizer_grad_norms(
                                optimizer,
                                max_norm,
                                train_mode=str(
                                    getattr(
                                        args,
                                        "stage_b_data_driven_train_mode",
                                        "rank_patch_only",
                                    )
                                ),
                            )
                        )
                    elif separate_u0_patch_rank_grad_clip:
                        branch_grad_norms = _clip_stage_b_u0_patch_rank_grad_norms(
                            model, max_norm
                        )
                    elif separate_gdino_adapter_grad_clip:
                        branch_grad_norms = _clip_stage_b_gdino_adapter_grad_norms(
                            model, max_norm
                        )
                    elif separate_dense_duty_grad_clip:
                        branch_grad_norms = _clip_stage_b_dense_duty_grad_norms(
                            model, max_norm
                        )
                    elif separate_v15_grad_clip:
                        branch_grad_norms = _clip_stage_b_v15_grad_norms(
                            model, max_norm
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
                amp_scale = float(scaler.get_scale())
                amp_step_skipped = float(amp_scale < scale_before)
                optimizer_step_succeeded = not bool(amp_step_skipped)
                if amp_step_skipped:
                    consecutive_amp_skips += 1
                    total_amp_skips += 1
                else:
                    consecutive_amp_skips = 0
                max_consecutive_skips = int(
                    getattr(args, "amp_max_consecutive_skips", 0) or 0
                )
                if (
                    max_consecutive_skips > 0
                    and consecutive_amp_skips >= max_consecutive_skips
                ):
                    raise FloatingPointError(
                        "AMP skipped "
                        f"{consecutive_amp_skips} consecutive optimizer steps "
                        f"(total={total_amp_skips}, scale={amp_scale:g}). "
                        "This usually means a finite forward loss has non-finite gradients."
                    )
            else:
                if max_norm > 0:
                    if separate_data_driven_grad_clip:
                        branch_grad_norms = (
                            _clip_stage_b_data_driven_optimizer_grad_norms(
                                optimizer,
                                max_norm,
                                train_mode=str(
                                    getattr(
                                        args,
                                        "stage_b_data_driven_train_mode",
                                        "rank_patch_only",
                                    )
                                ),
                            )
                        )
                    elif separate_u0_patch_rank_grad_clip:
                        branch_grad_norms = _clip_stage_b_u0_patch_rank_grad_norms(
                            model, max_norm
                        )
                    elif separate_gdino_adapter_grad_clip:
                        branch_grad_norms = _clip_stage_b_gdino_adapter_grad_norms(
                            model, max_norm
                        )
                    elif separate_dense_duty_grad_clip:
                        branch_grad_norms = _clip_stage_b_dense_duty_grad_norms(
                            model, max_norm
                        )
                    elif separate_v15_grad_clip:
                        branch_grad_norms = _clip_stage_b_v15_grad_norms(
                            model, max_norm
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                optimizer_step_succeeded = True

            any_step_succeeded = _sync_bool_any(
                optimizer_step_succeeded, device
            )
            all_steps_succeeded = _sync_bool_all(
                optimizer_step_succeeded, device
            )
            if any_step_succeeded != all_steps_succeeded:
                raise FloatingPointError(
                    "AMP optimizer-step success differed across distributed ranks; "
                    "aborting before counters or schedulers can diverge"
                )
            optimizer_step_succeeded = all_steps_succeeded
            if callable(commit_tail_queue):
                commit_tail_queue(optimizer_step_succeeded)

            _record_stage_b_dense_duty_runtime_audit(
                args,
                device,
                optimizer_step_boundary=optimizer_step_boundary,
                optimizer_step_succeeded=optimizer_step_succeeded,
                branch_grad_norms=branch_grad_norms,
                amp_scale=amp_scale,
            )

            optimizer.zero_grad()
            micro_batches_in_update = 0
            if optimizer_step_succeeded:
                optimizer_updates += 1
                if args.onecyclelr:
                    lr_scheduler.step()

        if patch_only and bool(getattr(args, "stage_b", False)):
            if optimizer_step_boundary and optimizer_step_succeeded:
                drift_state = _maybe_log_stage_b_patch_drift(
                    args=args,
                    model=model,
                    criterion=criterion,
                    device=device,
                    drift_state=drift_state,
                    samples=samples,
                    targets=targets,
                    captions=captions,
                    patches=patches,
                    patch_global=patch_global,
                    patch_mask=patch_mask,
                    outputs=outputs,
                    step=optimizer_updates,
                    logger=logger,
                )


        metric_logger.update(
            amp_step_skipped=amp_step_skipped,
            amp_scale=amp_scale,
            optimizer_step=float(optimizer_step_succeeded),
            **branch_grad_norms,
        )
        metric_weight = (
            int(logical_batches_in_forward) if forward_pack_factor > 1 else 1
        )
        weighted_metrics = {
            "loss": loss_value,
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        }
        for metric_name, metric_value in weighted_metrics.items():
            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.item()
            metric_logger.meters[metric_name].update(
                metric_value, n=metric_weight
            )
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        iter_interval = int(getattr(args, "iter_checkpoint_interval", 0) or 0)
        max_train_iters = int(getattr(args, "max_train_iters", 0) or 0)
        local_stop_requested = bool(getattr(args, "_stop_requested", False))
        stop_requested = (
            _sync_bool_any(local_stop_requested, device)
            if optimizer_step_boundary
            else local_stop_requested
        )
        if stop_requested:
            args._stop_requested = True
        epoch_finished_now = (
            data_loader_len is not None and _cnt >= int(data_loader_len)
        )
        stop_at_iter_limit = (
            optimizer_step_boundary
            and optimizer_step_succeeded
            and max_train_iters > 0
            and optimizer_updates >= max_train_iters
        )
        periodic_update_checkpoint = (
            optimizer_step_boundary
            and optimizer_step_succeeded
            and not epoch_finished_now
            and iter_interval > 0
            and (optimizer_updates % iter_interval == 0)
        )
        if epoch_finished_now and (stop_requested or stop_at_iter_limit):
            terminal_epoch_reason = (
                "signal" if stop_requested else "max_train_iters"
            )
            break
        should_save_iter = iter_checkpoint_fn is not None and (
            (stop_requested and optimizer_step_boundary)
            or stop_at_iter_limit
            or periodic_update_checkpoint
        )
        if should_save_iter:
            if stop_requested:
                reason = "signal"
            elif stop_at_iter_limit:
                reason = "max_train_iters"
            else:
                reason = "interval"
            iter_checkpoint_fn(
                epoch=epoch,
                iteration=_cnt,
                optimizer_updates=optimizer_updates,
                scaler=scaler,
                epoch_finished=False,
                reason=reason,
            )
        if stop_requested and optimizer_step_boundary:
            signum = getattr(args, "_stop_signal", None)
            raise GracefulTrainingExit(
                f"Stop requested by signal {signum}; saved optimizer-boundary checkpoint."
            )
        if stop_at_iter_limit:
            raise GracefulTrainingExit(
                f"Reached max_train_iters={max_train_iters} optimizer updates; "
                "saved iteration checkpoint."
            )
        if args.debug:
            if _cnt % 15 == 0 and optimizer_step_boundary:
                print("BREAK!"*5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    if terminal_epoch_reason is not None:
        if not args.onecyclelr:
            lr_scheduler.step()
        if iter_checkpoint_fn is not None:
            iter_checkpoint_fn(
                epoch=epoch,
                iteration=0,
                optimizer_updates=optimizer_updates,
                scaler=scaler,
                epoch_finished=True,
                reason=terminal_epoch_reason,
            )
        if terminal_epoch_reason == "signal":
            signum = getattr(args, "_stop_signal", None)
            raise GracefulTrainingExit(
                f"Stop requested by signal {signum}; saved epoch-boundary checkpoint."
            )
        raise GracefulTrainingExit(
            f"Reached max_train_iters={configured_update_limit} optimizer updates; "
            "saved epoch-boundary checkpoint after advancing the epoch scheduler."
        )


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    resstat["optimizer_updates"] = int(optimizer_updates)
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    
    coco_evaluator = CocoGroundingEvaluator(base_ds, iou_types, useCats=useCats)


    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only

    if args.use_coco_eval:
        from pycocotools.coco import COCO
        coco = COCO(args.coco_val_path)

        # 获取所有类别
        category_dict = coco.loadCats(coco.getCatIds())
        cat_list = [item['name'] for item in category_dict]
    else:
        cat_list=args.label_list
    caption = " . ".join(cat_list) + ' .'
    print("Input text prompt:", caption)

    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        bs = samples.tensors.shape[0]
        input_captions = [caption] * bs
        with torch.cuda.amp.autocast(enabled=args.amp):

            outputs = model(samples, captions=input_captions)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
        
        if args.save_results:



            for i, (tgt, res) in enumerate(zip(targets, results)):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)

                _res_bbox = res['boxes']
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
       

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]



    return stats, coco_evaluator
