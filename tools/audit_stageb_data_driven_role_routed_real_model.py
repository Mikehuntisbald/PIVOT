#!/usr/bin/env python3
"""Fail-closed CUDA audit for the clean role-routed Stage-B data path.

This is deliberately not a training smoke test. It builds the real model and
one real sealed dataset sample, then audits caption routing, branch isolation,
the v18 criterion, inference Gate3, and autograd ownership without constructing
an optimizer or taking a parameter update.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
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
    DATA_DRIVEN_PATCH_DROP_ANCHOR_INSTANCE_BALANCED,
    data_driven_category_gate_mask,
    validate_data_driven_role_routed_initializer_payload,
)
from util.misc import nested_tensor_from_tensor_list  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_data_driven_dd1_role_routed_patch_residual_clean_20260727.py"
)
DATASET_CONFIG = REPO_ROOT / "config/datasets_stageb_data_driven_role_routed_clean_train_20260727.json"
INITIALIZER = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/data_driven_role_routed_20260727/initializers/"
    "clean_dd1_u1000_lr3e4_patch_residual128_seed42_v2/"
    "checkpoint_model_only.pth"
)
EXPECTED_CONFIG_SHA256 = (
    "d545f4b12d9ac05950678f0442be4455b6767d35c831da4143040c38ea6e7d24"
)
EXPECTED_DATASET_CONFIG_SHA256 = (
    "909f2eb39934e5a263850c1f742d41bdf3f89f819992192696dbe99dd36ea245"
)
EXPECTED_INITIALIZER_SHA256 = (
    "c4275c575d8f7f3734806620b90572cf316adcd3fa8b42958ea6678d700c04c0"
)
RAW_CENTERED_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_data_driven_dd1_role_routed_patch_residual_"
    "raw_centered_clean_20260727.py"
)
RAW_CENTERED_INITIALIZER = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/data_driven_role_routed_20260727/initializers/"
    "clean_dd1_u1000_lr3e4_patch_residual128_raw_centered_seed42_v3/"
    "checkpoint_model_only.pth"
)
TOPK_SEMANTIC_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_data_driven_dd1_role_routed_patch_topk_semantic_"
    "clean_20260727.py"
)
TOPK_SEMANTIC_INITIALIZER = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/data_driven_role_routed_20260727/initializers/"
    "clean_dd1_u1000_lr3e4_patch_topksemantic128_ctx16_k10_seed42_v1/"
    "checkpoint_model_only.pth"
)
AUDIT_VARIANTS = {
    "uncentered": {
        "config": CONFIG,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "initializer": INITIALIZER,
        "initializer_sha256": EXPECTED_INITIALIZER_SHA256,
    },
    "raw_centered": {
        "config": RAW_CENTERED_CONFIG,
        "config_sha256": (
            "00bf18c02d7cdfa6c4d02eef4f7e89f933c0adabbe6ff4ea039f93a698d07fab"
        ),
        "initializer": RAW_CENTERED_INITIALIZER,
        "initializer_sha256": (
            "e65f62f342f3c36f6d6bd9322841eb028806f320b898eaf390a38ead36942799"
        ),
    },
    "topk_semantic": {
        "config": TOPK_SEMANTIC_CONFIG,
        "config_sha256": (
            "798634fd52e2484c6519b520914d5bb8782c28757710c7c840b22e1770c56c3d"
        ),
        "initializer": TOPK_SEMANTIC_INITIALIZER,
        "initializer_sha256": (
            "1c6472a8694eb606ceec8d74b7a5b2a7e8b3776790eb0f88edb4d5880a72ca0b"
        ),
    },
}
EXPECTED_SOURCE_SHA256 = (
    "dcfd1bf29668b7190f509587f1c9664345da168a9ee874bd97a1a032c01a1aa6"
)
EXPECTED_A0_SHA256 = (
    "c2c4ba71656054d3afc3d219ca2f6d56839396d6258bac0201878566b1937034"
)
EXPECTED_SOURCE_UPDATES = 1000
EXPECTED_QUERY_COUNT = 900

RANK_PREFIX = "stage_b_data_driven_score_heads.rank_branch."
CONFIDENCE_PREFIXES = (
    "stage_b_data_driven_score_heads.confidence_branch.",
    "stage_b_data_driven_score_heads.confidence_gate.",
)
PATCH_PREFIXES = (
    "stage_b_data_driven_patch_residual.",
)


class RealModelAuditError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, expected_sha256: str, *, label: str) -> Path:
    if path.is_symlink():
        raise RealModelAuditError(f"{label} must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise RealModelAuditError(f"{label} is not a regular file: {resolved}")
    observed = _sha256(resolved)
    if expected_sha256 and observed != expected_sha256:
        raise RealModelAuditError(
            f"{label} SHA256 drifted: expected={expected_sha256}, observed={observed}"
        )
    return resolved


def _load_initializer(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RealModelAuditError("role-routed initializer is not a mapping")
    return payload


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_real_sample(
    cfg: Any, dataset_index: int, *, dataset_path: Path = DATASET_CONFIG
):
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    rows = payload.get("train") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list) or len(rows) != 3 or payload.get("val") != []:
        raise RealModelAuditError("sealed clean dataset config coverage drifted")
    if dataset_index < 0:
        raise RealModelAuditError("dataset index must be nonnegative")
    dataset = build_dataset("train", cfg, dict(rows[0]))
    if dataset_index >= len(dataset):
        raise RealModelAuditError(
            f"dataset index {dataset_index} is outside {len(dataset)} sealed rows"
        )
    image, target = dataset[dataset_index]
    if not torch.is_tensor(image) or image.dim() != 3:
        raise RealModelAuditError("real sealed dataset sample did not return a CHW tensor")
    required = {
        "patch",
        "boxes",
        "stage_a_caption",
        "cap_list",
        "stage_b_data_driven_assignment_expressions",
        "stage_b_data_driven_assignment_pair_schema",
        "stage_b_data_driven_assignment_valid",
        "stage_b_data_driven_assignment_role",
    }
    missing = sorted(required - set(target))
    if missing:
        raise RealModelAuditError(f"real sealed target is incomplete: {missing}")
    if not bool(target["stage_b_data_driven_assignment_valid"].reshape(-1)[0].item()):
        raise RealModelAuditError(
            "selected sealed row has no valid assignment pair; choose another --dataset-index"
        )
    return image, target, len(dataset)


def _move_criterion_target(target: Mapping[str, Any], device: torch.device):
    return {
        key: value.to(device)
        for key, value in target.items()
        if torch.is_tensor(value) and key not in {"patch", "patch_global"}
    }


def _mutate_expression_pair(expressions: Sequence[Sequence[str]]) -> list[list[str]]:
    if len(expressions) != 1 or len(expressions[0]) != 2:
        raise RealModelAuditError("causal audit requires one Bx2 expression pair")
    changed = [list(expressions[0])]
    source = changed[0][0]
    replacements = (
        (" blue ", " red "),
        (" gray ", " green "),
        (" left ", " right "),
        (" right ", " left "),
        (" man ", " woman "),
        (" woman ", " man "),
    )
    padded = f" {source.strip()} "
    for old, new in replacements:
        if old in padded.lower():
            start = padded.lower().index(old)
            padded = padded[:start] + new + padded[start + len(old) :]
            changed[0][0] = padded.strip()
            break
    else:
        changed[0][0] = source.strip() + " altered"
    if changed[0][0] == source:
        raise RealModelAuditError("failed to create a full-expression intervention")
    return changed


def _expected_grounding_expression(value: str) -> str:
    normalized = " ".join(value.strip().split())
    if normalized[-1] not in ".?":
        normalized += " ."
    return normalized


def _assert_exact(left: torch.Tensor, right: torch.Tensor, *, label: str) -> None:
    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
        raise RealModelAuditError(f"{label} shape/dtype changed")
    if not torch.equal(left, right):
        delta = (
            (left.detach().float() - right.detach().float()).abs().max().item()
            if left.numel()
            else 0.0
        )
        raise RealModelAuditError(f"{label} changed; max_abs_delta={delta}")


def _nonzero_grad_names(model: torch.nn.Module) -> dict[str, float]:
    result = {}
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        magnitude = float(grad.detach().float().abs().sum().item())
        if not np.isfinite(magnitude):
            raise RealModelAuditError(f"non-finite gradient at {name}")
        if magnitude > 0.0:
            result[name] = magnitude
    return result


def _clear_grads(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def _audit_gradient_owner(
    model: torch.nn.Module,
    loss: torch.Tensor,
    *,
    expected_prefixes: Sequence[str],
    label: str,
    retain_graph: bool,
) -> dict[str, float]:
    _clear_grads(model)
    if not torch.is_tensor(loss) or loss.numel() != 1 or not torch.isfinite(loss):
        raise RealModelAuditError(f"{label} loss is not one finite scalar")
    loss.backward(retain_graph=retain_graph)
    nonzero = _nonzero_grad_names(model)
    if not nonzero:
        raise RealModelAuditError(f"{label} produced no nonzero parameter gradient")
    leaked = sorted(
        name
        for name in nonzero
        if not name.startswith(tuple(expected_prefixes))
    )
    if leaked:
        raise RealModelAuditError(f"{label} gradient leaked to {leaked}")
    return nonzero


def _audit_frozen_surfaces(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if name == "patch_logit_scale" or name.startswith(CONFIDENCE_PREFIXES):
            if parameter.requires_grad or parameter.grad is not None:
                raise RealModelAuditError(f"frozen score surface received grad: {name}")
        if name.startswith("backbone."):
            if parameter.requires_grad or parameter.grad is not None:
                raise RealModelAuditError(f"frozen backbone received grad: {name}")
        if name.startswith(
            (
                "patch_encoder.input_proj.",
                "patch_encoder.norm.",
                "query_proj_for_patch.",
            )
        ):
            if parameter.requires_grad or parameter.grad is not None:
                raise RealModelAuditError(
                    f"frozen base patch scorer received grad: {name}"
                )


def _audit_engine_training_mode(model: torch.nn.Module) -> None:
    root = model.module if hasattr(model, "module") else model
    heads = root.stage_b_data_driven_score_heads
    patch_encoder = root.patch_encoder
    expected_eval = {
        "root": root,
        "backbone": root.backbone,
        "bert": root.bert,
        "transformer": root.transformer,
        "shared_patch_backbone": patch_encoder.backbone,
        "confidence_branch": heads.confidence_branch,
        "confidence_gate": heads.confidence_gate,
        "patch_input_proj": patch_encoder.input_proj,
        "patch_norm": patch_encoder.norm,
        "query_proj_for_patch": root.query_proj_for_patch,
    }
    unexpected_train = sorted(
        label for label, module in expected_eval.items() if module.training
    )
    if unexpected_train:
        raise RealModelAuditError(
            f"frozen modules remain in train mode: {unexpected_train}"
        )
    expected_train = {
        "rank_branch": heads.rank_branch,
        "patch_residual": root.stage_b_data_driven_patch_residual,
    }
    unexpected_eval = sorted(
        label for label, module in expected_train.items() if not module.training
    )
    if unexpected_eval:
        raise RealModelAuditError(
            f"owned modules remain in eval mode: {unexpected_eval}"
        )


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    requested = path.expanduser()
    if not requested.is_absolute():
        requested = Path.cwd() / requested
    requested.parent.mkdir(parents=True, exist_ok=True)
    path = requested.parent.resolve(strict=True) / requested.name
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RealModelAuditError(
                "renameat2(RENAME_NOREPLACE) is unavailable; refusing unsafe receipt commit"
            )
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(temporary),
            -100,
            os.fsencode(path),
            1,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number == errno.EEXIST:
                raise RealModelAuditError(
                    f"refusing to overwrite audit receipt: {path}"
                )
            raise RealModelAuditError(
                "atomic no-replace receipt commit failed: "
                f"{temporary} -> {path}: {os.strerror(error_number)}"
            )
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def run_audit(
    *,
    device_name: str,
    dataset_index: int,
    seed: int,
    variant: str = "uncentered",
) -> dict[str, Any]:
    if not device_name.startswith("cuda"):
        raise RealModelAuditError("real-model audit requires an explicit CUDA device")
    if not torch.cuda.is_available():
        raise RealModelAuditError("torch.cuda.is_available() is false")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    _seed_everything(seed)

    binding = AUDIT_VARIANTS.get(variant)
    if binding is None:
        raise RealModelAuditError(f"unknown residual audit variant: {variant!r}")

    config_path = _require_file(
        binding["config"],
        binding["config_sha256"],
        label="training config",
    )
    dataset_path = _require_file(
        DATASET_CONFIG,
        EXPECTED_DATASET_CONFIG_SHA256,
        label="dataset config",
    )
    initializer_path = _require_file(
        binding["initializer"],
        binding["initializer_sha256"],
        label="model-only initializer",
    )

    cfg = SLConfig.fromfile(str(config_path))
    cfg.device = "cpu"
    cfg.amp = True
    cfg.seed = 42
    cfg.num_workers = int(cfg.stage_b_data_driven_role_expected_num_workers)
    cfg.pin_memory = bool(cfg.stage_b_data_driven_role_expected_pin_memory)
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
    trainable_numel = _freeze_and_audit_stage_b_data_driven(
        model, "rank_patch_only"
    )
    if int(criterion.criterion_contract_version.item()) != 18:
        raise RealModelAuditError("criterion contract is not v18")
    if int(criterion.rank_supervision_contract_id.item()) != 6:
        raise RealModelAuditError("criterion rank-supervision id is not 6")
    if criterion.patch_active_unsafe_auxiliary_weight != 1.0:
        raise RealModelAuditError(
            "criterion patch active-unsafe auxiliary weight is not 1.0"
        )
    if (
        criterion.patch_drop_positive_anchor_gradient_policy
        != DATA_DRIVEN_PATCH_DROP_ANCHOR_INSTANCE_BALANCED
    ):
        raise RealModelAuditError(
            "criterion does not use the instance-balanced drop anchor"
        )
    expected_dense_focal = (0.0, 0.25, 2.0, 1.0)
    observed_dense_focal = (
        criterion.patch_dense_category_focal_weight,
        criterion.patch_dense_category_focal_alpha,
        criterion.patch_dense_category_focal_gamma,
        criterion.patch_dense_category_focal_negative_weight,
    )
    if observed_dense_focal != expected_dense_focal:
        raise RealModelAuditError(
            "criterion dense-category focal contract drifted"
        )
    expected_weight_dict = {
        "loss_stage_b_data_driven_role_routed_rank": 1.0,
        "loss_stage_b_data_driven_patch": 1.0,
    }
    if criterion.weight_dict != expected_weight_dict:
        raise RealModelAuditError(
            "criterion weighted-loss surface drifted: "
            f"expected={expected_weight_dict}, observed={criterion.weight_dict}"
        )

    image, raw_target, dataset_rows = _build_real_sample(
        cfg, dataset_index, dataset_path=dataset_path
    )
    canonical, expressions = _build_stage_b_data_driven_assignment_captions(
        [raw_target]
    )
    expected_canonical = [
        _expected_grounding_expression(raw_target["stage_a_caption"])
    ]
    expected_expressions = [[
        _expected_grounding_expression(value)
        for value in raw_target["stage_b_data_driven_assignment_expressions"]
    ]]
    if canonical != expected_canonical:
        raise RealModelAuditError(
            "engine changed the canonical Stage-A caption beyond allowed syntax normalization"
        )
    if expressions != expected_expressions:
        raise RealModelAuditError(
            "engine changed a full expression beyond allowed syntax normalization"
        )

    samples = nested_tensor_from_tensor_list([image]).to(device)
    patches = raw_target["patch"].unsqueeze(0).to(device)
    target = _move_criterion_target(raw_target, device)
    model = model.to(device)
    criterion = criterion.to(device)

    captured_queries: list[torch.Tensor] = []

    def capture_query(_module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]):
            raise RealModelAuditError("score-head hook did not receive canonical queries")
        captured_queries.append(inputs[0].detach().clone())

    hook = model.stage_b_data_driven_score_heads.register_forward_pre_hook(
        capture_query
    )
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=bool(cfg.amp)):
        base_outputs = model(
            samples,
            captions=canonical,
            patches=patches,
            stage_b_data_driven_expression_captions=expressions,
        )
        intervened_expressions = _mutate_expression_pair(expressions)
        changed_outputs = model(
            samples,
            captions=canonical,
            patches=patches,
            stage_b_data_driven_expression_captions=intervened_expressions,
        )
    hook.remove()
    if len(captured_queries) != 2:
        raise RealModelAuditError("score-head hook call count drifted")
    _assert_exact(captured_queries[0], captured_queries[1], label="canonical query hs")
    _assert_exact(base_outputs["pred_boxes"], changed_outputs["pred_boxes"], label="boxes")
    _assert_exact(
        base_outputs["pred_logits_patch"],
        changed_outputs["pred_logits_patch"],
        label="patch scores",
    )
    _assert_exact(
        base_outputs["pred_logits_patch"],
        base_outputs["pred_logits_patch_base"],
        label="zero-init residual patch score",
    )
    residual_score = base_outputs["pred_logits_patch_residual"]
    if not torch.is_tensor(residual_score) or not bool(
        (residual_score == 0).all().item()
    ):
        raise RealModelAuditError("patch residual is not exactly zero at U0")
    _assert_exact(
        residual_score,
        changed_outputs["pred_logits_patch_residual"],
        label="expression-independent patch residual",
    )
    _assert_exact(
        base_outputs["stage_b_data_driven_candidate_mask"],
        changed_outputs["stage_b_data_driven_candidate_mask"],
        label="candidate mask",
    )
    candidate = base_outputs["stage_b_data_driven_candidate_mask"]
    expected_candidate_shape = (1, EXPECTED_QUERY_COUNT, 2)
    if tuple(candidate.shape) != expected_candidate_shape or not bool(candidate.all()):
        raise RealModelAuditError(
            f"candidate contract is not all {EXPECTED_QUERY_COUNT} queries in both slots: "
            f"shape={tuple(candidate.shape)}"
        )
    base_rank = base_outputs["stage_b_data_driven_text_rank_score"]
    changed_rank = changed_outputs["stage_b_data_driven_text_rank_score"]
    if tuple(base_rank.shape) != (1, EXPECTED_QUERY_COUNT, 2):
        raise RealModelAuditError(
            f"paired rank-score shape drifted: {tuple(base_rank.shape)}"
        )
    _assert_exact(
        base_rank[..., 1],
        changed_rank[..., 1],
        label="untouched expression-slot rank scores",
    )
    rank_delta = float(
        (
            base_rank[..., 0]
            - changed_rank[..., 0]
        )
        .abs()
        .max()
        .item()
    )
    if not rank_delta > 0.0:
        raise RealModelAuditError(
            "full-expression intervention did not change its own rank-score slot"
        )
    base_confidence = base_outputs["stage_b_data_driven_confidence_score"]
    changed_confidence = changed_outputs["stage_b_data_driven_confidence_score"]
    if tuple(base_confidence.shape) != (1, EXPECTED_QUERY_COUNT, 2):
        raise RealModelAuditError(
            f"paired confidence-score shape drifted: {tuple(base_confidence.shape)}"
        )
    _assert_exact(
        base_confidence[..., 1],
        changed_confidence[..., 1],
        label="untouched expression-slot confidence scores",
    )

    heads = model.stage_b_data_driven_score_heads
    if heads.category_gate:
        raise RealModelAuditError("training model unexpectedly enables inference Gate3")
    heads.category_gate = True
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=bool(cfg.amp)):
        gated_outputs = model(
            samples,
            captions=canonical,
            patches=patches,
            stage_b_data_driven_expression_captions=expressions,
        )
    heads.category_gate = False
    gated_candidate = gated_outputs["stage_b_data_driven_candidate_mask"]
    patch_score = gated_outputs["pred_logits_patch"]
    if patch_score.dim() == 3:
        if int(patch_score.shape[-1]) != 1:
            raise RealModelAuditError("Gate3 audit requires one support-patch slot")
        patch_score = patch_score[..., 0]
    slot_patch = (
        patch_score[:, :, None]
        .expand(-1, -1, 2)
        .permute(0, 2, 1)
        .reshape(2, EXPECTED_QUERY_COUNT)
    )
    slot_candidate = gated_candidate.permute(0, 2, 1).reshape(2, EXPECTED_QUERY_COUNT)
    hand_gate, _ = data_driven_category_gate_mask(
        slot_patch,
        slot_candidate,
        max_gap=3.0,
        clip=5.0,
    )
    model_gate = gated_outputs[
        "stage_b_data_driven_category_gate_eligible_mask"
    ].permute(0, 2, 1).reshape(2, EXPECTED_QUERY_COUNT)
    _assert_exact(hand_gate, model_gate, label="Gate3 eligibility")
    text_rank = gated_outputs["stage_b_data_driven_text_rank_score"]
    deployed_rank = gated_outputs["stage_b_data_driven_rank_score"]
    hand_top = text_rank.masked_fill(
        ~gated_outputs["stage_b_data_driven_category_gate_eligible_mask"],
        -torch.inf,
    ).argmax(dim=1)
    model_top = deployed_rank.argmax(dim=1)
    _assert_exact(hand_top, model_top, label="Gate3 top1")

    model.train()
    criterion.train()
    _set_stage_b_data_driven_training_mode(
        model, cfg.stage_b_data_driven_train_mode
    )
    _audit_engine_training_mode(model)
    with torch.cuda.amp.autocast(enabled=bool(cfg.amp)):
        train_outputs = model(
            samples,
            captions=canonical,
            patches=patches,
            stage_b_data_driven_expression_captions=expressions,
        )
        losses = criterion(train_outputs, [target])
    rank_loss = losses.get("loss_stage_b_data_driven_role_routed_rank")
    patch_loss = losses.get("loss_stage_b_data_driven_patch")
    if rank_loss is None or patch_loss is None:
        raise RealModelAuditError(
            "v18 criterion did not emit both role-rank and patch losses"
        )
    runtime_directions = int(
        losses["stage_b_data_driven_assignment_runtime_directions"].item()
    )
    paired_sibling_queries = int(
        losses[
            "stage_b_data_driven_assignment_paired_sibling_queries"
        ].item()
    )
    safe_nonowned_queries = int(
        losses[
            "stage_b_data_driven_assignment_safe_sibling_queries"
        ].item()
    )
    if safe_nonowned_queries < paired_sibling_queries:
        raise RealModelAuditError(
            "all-exclusive-nonowned supervision lost a paired sibling"
        )
    if runtime_directions <= 0:
        raise RealModelAuditError(
            "selected real row has no runtime-valid assignment direction; "
            "choose another --dataset-index"
        )
    rank_grads = _audit_gradient_owner(
        model,
        rank_loss,
        expected_prefixes=(RANK_PREFIX,),
        label="role-routed rank",
        retain_graph=True,
    )
    patch_grads = _audit_gradient_owner(
        model,
        patch_loss,
        expected_prefixes=PATCH_PREFIXES,
        label="category patch",
        retain_graph=False,
    )
    _audit_frozen_surfaces(model)
    _clear_grads(model)
    torch.cuda.synchronize(device)

    confidence_delta = float(
        (
            base_confidence[..., 0]
            - changed_confidence[..., 0]
        )
        .abs()
        .max()
        .item()
    )
    return {
        "status": "passed",
        "device": str(device),
        "dataset": {
            "manifest": str(
                json.loads(dataset_path.read_text())["train"][0]["anno"]
            ),
            "rows": dataset_rows,
            "index": dataset_index,
            "image_id": int(raw_target["image_id"].reshape(-1)[0].item()),
            "raw_canonical_caption": raw_target["stage_a_caption"],
            "routed_canonical_caption": canonical[0],
            "raw_expressions": raw_target[
                "stage_b_data_driven_assignment_expressions"
            ],
            "routed_expressions": expressions[0],
            "intervened_expressions": intervened_expressions[0],
        },
        "contracts": {
            "audit_script_sha256": _sha256(Path(__file__).resolve()),
            "variant": variant,
            "config_sha256": binding["config_sha256"],
            "dataset_config_sha256": EXPECTED_DATASET_CONFIG_SHA256,
            "initializer_sha256": binding["initializer_sha256"],
            "criterion_version": 18,
            "rank_supervision_id": 6,
            "rank_negative_contract": (
                "all_exclusive_category_complete_nonowned_v2"
            ),
            "patch_active_unsafe_auxiliary_weight": 1.0,
            "patch_dense_category_focal_contract": [0.0, 0.25, 2.0, 1.0],
            "patch_drop_gradient_contract": (
                "fixed_denominator_active_severity_instance_balanced_zero_sum_v3"
            ),
            "patch_residual_contract": (
                cfg.stage_b_data_driven_patch_residual_contract
            ),
            "patch_residual_u0_matches_base_bitwise": True,
            "patch_residual_is_expression_independent": True,
            "weighted_loss_keys": sorted(expected_weight_dict),
            "query_count": EXPECTED_QUERY_COUNT,
            "amp_matches_training": bool(cfg.amp),
            "candidate_slots_equal_and_all_true": True,
            "caption_normalization_only": True,
            "training_category_gate_disabled": True,
            "engine_frozen_feature_generators_in_eval": True,
            "engine_owned_rank_patch_modules_in_train": True,
        },
        "causal_intervention": {
            "canonical_query_hs_bitwise_equal": True,
            "boxes_bitwise_equal": True,
            "patch_scores_bitwise_equal": True,
            "candidate_masks_bitwise_equal": True,
            "mutated_slot_rank_score_max_abs_delta": rank_delta,
            "untouched_slot_rank_scores_bitwise_equal": True,
            "mutated_slot_confidence_score_max_abs_delta": confidence_delta,
            "untouched_slot_confidence_scores_bitwise_equal": True,
            "note": (
                "confidence is also a full-expression branch, so it may change; "
                "it is frozen and receives no rank/patch gradient"
            ),
        },
        "gate3": {
            "hand_mask_matches_model": True,
            "hand_argmax_matches_model": True,
            "eligible_queries_per_slot": hand_gate.sum(dim=1).tolist(),
            "top_query_per_slot": hand_top.reshape(-1).tolist(),
        },
        "autograd": {
            "trainable_numel": int(trainable_numel),
            "runtime_valid_directions": runtime_directions,
            "paired_sibling_queries": paired_sibling_queries,
            "safe_nonowned_queries": safe_nonowned_queries,
            "rank_loss": float(rank_loss.detach().item()),
            "patch_loss": float(patch_loss.detach().item()),
            "rank_nonzero_grad_parameters": sorted(rank_grads),
            "patch_nonzero_grad_parameters": sorted(patch_grads),
            "confidence_backbone_and_patch_scale_frozen": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variant",
        choices=sorted(AUDIT_VARIANTS),
        default="uncentered",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result = run_audit(
        device_name=args.device,
        dataset_index=args.dataset_index,
        seed=args.seed,
        variant=args.variant,
    )
    if args.output_json is not None:
        _write_json_exclusive(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
