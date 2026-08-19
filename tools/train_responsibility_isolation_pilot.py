#!/usr/bin/env python3
"""Train shared/isolated heads from a frozen cached-candidate shard.

The runner is a deliberately small pilot, not a detector extractor.  It binds
the cache bytes and semantic tensor content, uses a deterministic interleaved
rank/confidence exposure schedule, and emits a resumable checkpoint plus an
ownership/gradient/AMP receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.responsibility_isolation import (
    RESPONSIBILITY_OWNERSHIP_ISOLATED,
    RESPONSIBILITY_OWNERSHIP_MODES,
    FrozenCandidateResponsibilityHeads,
    responsibility_gradient_report,
    responsibility_ownership_report,
)
from tools.responsibility_isolation_cache import (
    CACHE_FEATURE_DIM,
    CACHE_SHARD_SCHEMA,
    CACHE_TASK_CONFIDENCE_PAIR,
    CACHE_TASK_RANK,
    cached_candidate_content_sha256,
    file_sha256,
    load_cached_candidate_shard,
    normalized_cxcywh_iou,
)


PILOT_CHECKPOINT_SCHEMA = "responsibility_isolation.pilot_checkpoint/v1"
PILOT_RECEIPT_SCHEMA = "responsibility_isolation.pilot_receipt/v1"
PILOT_SCHEDULE_SCHEMA = "responsibility_isolation.interleaved_schedule/v1"
PILOT_SEEDS = (17, 42, 73)
RANK_LOSS_CONTRACT = (
    "iou50_topk_hard_negative_margin_listwise_plus_teacher_preserve/v1"
)
CONFIDENCE_LOSS_CONTRACT = "paired_sample_max_balanced_bce/v1"


class PilotContractError(RuntimeError):
    """Raised when a pilot run cannot satisfy its preregistered contract."""


@dataclass(frozen=True)
class PilotConfig:
    ownership: str
    seed: int
    updates: int
    device: str = "cpu"
    amp: bool = False
    hidden_dim: int = 128
    rank_residual_limit: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    clip_norm: float = 0.1
    rank_temperature: float = 0.05
    rank_margin: float = 0.05
    rank_hard_negative_count: int = 32
    preserve_tolerance: float = 0.01
    preserve_floor: float = 0.005
    preserve_weight: float = 2.0
    amp_initial_scale: float = 8192.0

    def validate(self) -> "PilotConfig":
        if self.ownership not in RESPONSIBILITY_OWNERSHIP_MODES:
            raise PilotContractError(
                f"ownership must be one of {RESPONSIBILITY_OWNERSHIP_MODES}"
            )
        if self.seed not in PILOT_SEEDS:
            raise PilotContractError(f"seed must be one of {PILOT_SEEDS}")
        if (
            isinstance(self.updates, bool)
            or not isinstance(self.updates, int)
            or self.updates <= 0
        ):
            raise PilotContractError("updates must be a positive integer")
        for name, value in (
            ("hidden_dim", self.hidden_dim),
            ("rank_hard_negative_count", self.rank_hard_negative_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise PilotContractError(f"{name} must be a positive integer")
        if not isinstance(self.amp, bool):
            raise PilotContractError("amp must be boolean")
        if self.device not in ("cpu", "cuda"):
            raise PilotContractError("device must be 'cpu' or 'cuda'")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise PilotContractError("CUDA was requested but is unavailable")
        if self.amp and self.device != "cuda":
            raise PilotContractError("AMP pilot is supported only on CUDA")
        positive_values = {
            "rank_residual_limit": self.rank_residual_limit,
            "learning_rate": self.learning_rate,
            "clip_norm": self.clip_norm,
            "rank_temperature": self.rank_temperature,
            "rank_margin": self.rank_margin,
            "preserve_floor": self.preserve_floor,
            "preserve_weight": self.preserve_weight,
            "amp_initial_scale": self.amp_initial_scale,
        }
        for name, value in positive_values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise PilotContractError(f"{name} must be finite and positive")
        if (
            not math.isfinite(float(self.weight_decay))
            or float(self.weight_decay) < 0.0
            or not math.isfinite(float(self.preserve_tolerance))
            or float(self.preserve_tolerance) < 0.0
        ):
            raise PilotContractError(
                "weight_decay and preserve_tolerance must be finite and nonnegative"
            )
        return self

    def checkpoint_contract(self) -> dict[str, Any]:
        """Fields that must remain identical when extending a checkpoint."""
        return {
            "ownership": self.ownership,
            "seed": self.seed,
            "device": self.device,
            "amp": self.amp,
            "feature_dim": CACHE_FEATURE_DIM,
            "hidden_dim": self.hidden_dim,
            "rank_residual_limit": self.rank_residual_limit,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "clip_norm": self.clip_norm,
            "rank_temperature": self.rank_temperature,
            "rank_margin": self.rank_margin,
            "rank_hard_negative_count": self.rank_hard_negative_count,
            "preserve_tolerance": self.preserve_tolerance,
            "preserve_floor": self.preserve_floor,
            "preserve_weight": self.preserve_weight,
            "amp_initial_scale": self.amp_initial_scale,
            "rank_loss_contract": RANK_LOSS_CONTRACT,
            "confidence_loss_contract": CONFIDENCE_LOSS_CONTRACT,
            "schedule_schema": PILOT_SCHEDULE_SCHEMA,
        }


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_mapping_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        header = json.dumps(
            [name, str(value.dtype), list(value.shape)],
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = value.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _ordered_epoch_stream(
    identities: Iterable[str], *, seed: int, salt: str, count: int
) -> tuple[str, ...]:
    values = tuple(sorted(identities))
    if not values:
        raise PilotContractError(f"no identities available for {salt} schedule")
    result: list[str] = []
    epoch = 0
    while len(result) < count:
        ranked = sorted(
            values,
            key=lambda identity: hashlib.sha256(
                f"{PILOT_SCHEDULE_SCHEMA}:{seed}:{salt}:{epoch}:{identity}".encode()
            ).hexdigest(),
        )
        result.extend(ranked)
        epoch += 1
    return tuple(result[:count])


def _shard_indices(shard: Mapping[str, Any]) -> tuple[dict, dict]:
    rank = {
        row["sample_id"]: row
        for row in shard["rows"]
        if row["task"] == CACHE_TASK_RANK
    }
    pair_rows: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in shard["rows"]:
        if row["task"] != CACHE_TASK_CONFIDENCE_PAIR:
            continue
        pair_rows.setdefault(row["pair_id"], {})[row["pair_role"]] = row
    return rank, pair_rows


def build_interleaved_exposure_schedule(
    shard: Mapping[str, Any], *, seed: int, updates: int
) -> tuple[dict[str, Any], ...]:
    """Build an ownership-independent rank/confidence alternating schedule."""
    if seed not in PILOT_SEEDS:
        raise PilotContractError(f"seed must be one of {PILOT_SEEDS}")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
        raise PilotContractError("updates must be a positive integer")
    rank_rows, pairs = _shard_indices(shard)
    rank_count = (updates + 1) // 2
    confidence_count = updates // 2
    rank_stream = _ordered_epoch_stream(
        rank_rows, seed=seed, salt="rank", count=rank_count
    )
    confidence_stream = _ordered_epoch_stream(
        pairs, seed=seed, salt="confidence", count=confidence_count
    ) if confidence_count else ()
    result: list[dict[str, Any]] = []
    rank_cursor = 0
    confidence_cursor = 0
    for update_index in range(1, updates + 1):
        if update_index % 2 == 1:
            sample_id = rank_stream[rank_cursor]
            rank_cursor += 1
            result.append(
                {
                    "update": update_index,
                    "task": CACHE_TASK_RANK,
                    "sample_id": sample_id,
                    "image_id": rank_rows[sample_id]["image_id"],
                }
            )
        else:
            pair_id = confidence_stream[confidence_cursor]
            confidence_cursor += 1
            rows = pairs[pair_id]
            result.append(
                {
                    "update": update_index,
                    "task": CACHE_TASK_CONFIDENCE_PAIR,
                    "pair_id": pair_id,
                    "positive_sample_id": rows["positive"]["sample_id"],
                    "negative_sample_id": rows["negative"]["sample_id"],
                    "positive_image_id": rows["positive"]["image_id"],
                    "negative_image_id": rows["negative"]["image_id"],
                }
            )
    return tuple(result)


def _row_forward(
    module: FrozenCandidateResponsibilityHeads,
    row: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Tensor]:
    return module(
        row["query_features"].unsqueeze(0).to(device=device),
        row["native_score"].unsqueeze(0).to(device=device),
        row["candidate_mask"].unsqueeze(0).to(device=device),
    )


def rank_listwise_preserve_loss(
    output: Mapping[str, Tensor],
    row: Mapping[str, Any],
    *,
    temperature: float,
    margin: float,
    hard_negative_count: int,
    preserve_tolerance: float,
    preserve_floor: float,
    preserve_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    """IoU50 positive-mass listwise loss plus native-correct preservation."""
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise PilotContractError("rank temperature must be finite and positive")
    if not math.isfinite(float(margin)) or float(margin) < 0.0:
        raise PilotContractError("rank margin must be finite and nonnegative")
    if (
        isinstance(hard_negative_count, bool)
        or not isinstance(hard_negative_count, int)
        or hard_negative_count <= 0
    ):
        raise PilotContractError("hard_negative_count must be a positive integer")
    score = output["rank_score"][0]
    native = output["native_score"][0].detach()
    mask = output["candidate_mask"][0]
    boxes = row["boxes"].to(device=score.device)
    gt_boxes = row["gt_boxes"].to(device=score.device)
    best_iou = normalized_cxcywh_iou(boxes, gt_boxes).amax(dim=1)
    positive = mask & (best_iou >= 0.5)
    negative = mask & (~positive)
    if not bool(positive.any().item()) or not bool(negative.any().item()):
        raise PilotContractError(
            f"rank row {row['sample_id']!r} lost its positive/negative surface"
        )
    tau = float(temperature)
    negative_scores = score[negative]
    selected_hard_negative_count = min(
        int(hard_negative_count), int(negative_scores.numel())
    )
    hard_indices = torch.argsort(
        negative_scores.detach(), descending=True, stable=True
    )[:selected_hard_negative_count]
    hard_negative_scores = negative_scores[hard_indices]
    positive_partition = torch.logsumexp(score[positive] / tau, dim=0)
    negative_partition = torch.logsumexp(
        (hard_negative_scores + float(margin)) / tau, dim=0
    )
    listwise = tau * (
        torch.logaddexp(positive_partition, negative_partition)
        - positive_partition
    )
    learned_positive = score[positive].max()
    learned_hard_negative = score[negative].max()
    learned_gap = learned_positive - learned_hard_negative
    teacher_gap = native[positive].max() - native[negative].max()
    if bool((teacher_gap > 0).item()):
        preserve_target = torch.maximum(
            teacher_gap - float(preserve_tolerance),
            teacher_gap.new_tensor(float(preserve_floor)),
        )
        preserve = F.relu(preserve_target - learned_gap)
    else:
        preserve = learned_gap * 0.0
    loss = listwise + float(preserve_weight) * preserve
    if not bool(torch.isfinite(loss).item()):
        raise PilotContractError("rank loss is non-finite")
    return loss, {
        "loss": float(loss.detach()),
        "listwise": float(listwise.detach()),
        "preserve": float(preserve.detach()),
        "teacher_gap": float(teacher_gap.detach()),
        "learned_gap": float(learned_gap.detach()),
        "hard_negative_score": float(learned_hard_negative.detach()),
        "hard_negative_count": float(selected_hard_negative_count),
        "positive_count": float(positive.sum().item()),
        "negative_count": float(negative.sum().item()),
    }


def confidence_pair_sample_max_bce_loss(
    positive_output: Mapping[str, Tensor],
    negative_output: Mapping[str, Tensor],
) -> tuple[Tensor, dict[str, float]]:
    """Stable balanced BCE over the maximum valid absolute candidate logit."""
    positive_mask = positive_output["candidate_mask"][0]
    negative_mask = negative_output["candidate_mask"][0]
    positive_logit = positive_output["confidence_score"][0][positive_mask].max()
    negative_logit = negative_output["confidence_score"][0][negative_mask].max()
    positive_bce = F.softplus(-positive_logit)
    negative_bce = F.softplus(negative_logit)
    loss = 0.5 * (positive_bce + negative_bce)
    if not bool(torch.isfinite(loss).item()):
        raise PilotContractError("confidence pair loss is non-finite")
    return loss, {
        "loss": float(loss.detach()),
        "positive_bce": float(positive_bce.detach()),
        "negative_bce": float(negative_bce.detach()),
        "positive_sample_max_logit": float(positive_logit.detach()),
        "negative_sample_max_logit": float(negative_logit.detach()),
    }


def _task_loss(
    module: FrozenCandidateResponsibilityHeads,
    exposure: Mapping[str, Any],
    rank_rows: Mapping[str, Mapping[str, Any]],
    pairs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    config: PilotConfig,
    *,
    device: torch.device,
) -> tuple[Tensor, dict[str, float]]:
    if exposure["task"] == CACHE_TASK_RANK:
        row = rank_rows[exposure["sample_id"]]
        output = _row_forward(module, row, device=device)
        return rank_listwise_preserve_loss(
            output,
            row,
            temperature=config.rank_temperature,
            margin=config.rank_margin,
            hard_negative_count=config.rank_hard_negative_count,
            preserve_tolerance=config.preserve_tolerance,
            preserve_floor=config.preserve_floor,
            preserve_weight=config.preserve_weight,
        )
    pair = pairs[exposure["pair_id"]]
    positive = _row_forward(module, pair["positive"], device=device)
    negative = _row_forward(module, pair["negative"], device=device)
    return confidence_pair_sample_max_bce_loss(positive, negative)


def _gradient_norm_is_finite(module: torch.nn.Module) -> bool:
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return bool(gradients) and all(
        bool(torch.isfinite(gradient).all().item()) for gradient in gradients
    )


def _final_gradient_audit(
    module: FrozenCandidateResponsibilityHeads,
    rank_rows: Mapping[str, Mapping[str, Any]],
    pairs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    config: PilotConfig,
    *,
    device: torch.device,
) -> dict:
    rank_id = sorted(rank_rows)[0]
    pair_id = sorted(pairs)[0]
    rank_output = _row_forward(module, rank_rows[rank_id], device=device)
    rank_loss, _ = rank_listwise_preserve_loss(
        rank_output,
        rank_rows[rank_id],
        temperature=config.rank_temperature,
        margin=config.rank_margin,
        hard_negative_count=config.rank_hard_negative_count,
        preserve_tolerance=config.preserve_tolerance,
        preserve_floor=config.preserve_floor,
        preserve_weight=config.preserve_weight,
    )
    positive = _row_forward(module, pairs[pair_id]["positive"], device=device)
    negative = _row_forward(module, pairs[pair_id]["negative"], device=device)
    confidence_loss, _ = confidence_pair_sample_max_bce_loss(positive, negative)
    report = responsibility_gradient_report(module, rank_loss, confidence_loss)
    if not report["gradient_finite"]:
        raise PilotContractError("final ownership audit found non-finite gradients")
    if config.ownership == RESPONSIBILITY_OWNERSHIP_ISOLATED:
        if not report["structurally_isolated"]:
            raise PilotContractError("isolated pilot has cross-task autograd paths")
    elif not report["jointly_connected_parameter_names"]:
        raise PilotContractError("shared pilot has no shared task-gradient path")
    return report


def _checkpoint_load(
    path: Path,
    *,
    device: torch.device,
    config: PilotConfig,
    cache_file_sha256: str,
    cache_content_sha256: str,
    schedule: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        raise PilotContractError(f"could not load resume checkpoint: {exc}") from exc
    if not isinstance(checkpoint, Mapping):
        raise PilotContractError("resume checkpoint must be a mapping")
    if checkpoint.get("schema") != PILOT_CHECKPOINT_SCHEMA:
        raise PilotContractError("resume checkpoint schema mismatch")
    if checkpoint.get("config") != config.checkpoint_contract():
        raise PilotContractError("resume checkpoint training contract mismatch")
    if checkpoint.get("cache_file_sha256") != cache_file_sha256:
        raise PilotContractError("resume checkpoint cache file SHA mismatch")
    if checkpoint.get("cache_content_sha256") != cache_content_sha256:
        raise PilotContractError("resume checkpoint cache content SHA mismatch")
    completed = checkpoint.get("completed_updates")
    if not isinstance(completed, int) or completed < 0 or completed > config.updates:
        raise PilotContractError("resume checkpoint update cursor is invalid")
    expected_prefix = list(schedule[:completed])
    if checkpoint.get("exposure_history") != expected_prefix:
        raise PilotContractError("resume checkpoint exposure prefix mismatch")
    if len(checkpoint.get("loss_history", ())) != completed:
        raise PilotContractError("resume checkpoint loss history is incomplete")
    return dict(checkpoint)


@contextmanager
def _deterministic_algorithms():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def run_cached_feature_pilot(
    *,
    cache_path: str | Path,
    output_dir: str | Path,
    config: PilotConfig,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Run U{updates} and return receipt/checkpoint paths plus the receipt."""
    if config.device == "cuda":
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace is None:
            # This must happen before the first cuBLAS-backed head operation.
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        elif workspace not in (":4096:8", ":16:8"):
            raise PilotContractError(
                "deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
                "or :16:8"
            )
    config = config.validate()
    cache_path = Path(cache_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "run_receipt.json"
    checkpoint_path = output_dir / f"checkpoint_u{config.updates:06d}.pt"
    if receipt_path.exists() or checkpoint_path.exists():
        raise PilotContractError("pilot output artifacts already exist")

    cache_file_before = file_sha256(cache_path)
    shard = load_cached_candidate_shard(cache_path)
    if shard["schema"] != CACHE_SHARD_SCHEMA:
        raise PilotContractError("validated cache unexpectedly changed schema")
    cache_content = cached_candidate_content_sha256(shard)
    cache_requires_grad = sum(
        int(value.requires_grad)
        for row in shard["rows"]
        for value in row.values()
        if torch.is_tensor(value)
    )
    if cache_requires_grad:
        raise PilotContractError("cached tensors must not require gradients")
    schedule = build_interleaved_exposure_schedule(
        shard, seed=config.seed, updates=config.updates
    )
    schedule_sha = _canonical_json_sha256(schedule)
    rank_rows, pairs = _shard_indices(shard)
    device = torch.device(config.device)

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    module = FrozenCandidateResponsibilityHeads(
        feature_dim=CACHE_FEATURE_DIM,
        hidden_dim=config.hidden_dim,
        ownership=config.ownership,
        rank_residual_limit=config.rank_residual_limit,
    ).to(device=device)
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        foreach=False,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=config.amp_initial_scale,
        enabled=config.amp,
    )
    completed_updates = 0
    applied_updates = 0
    amp_skips = 0
    nonfinite_updates = 0
    exposure_history: list[dict[str, Any]] = []
    loss_history: list[dict[str, Any]] = []
    resume_sha: str | None = None

    if resume_from is not None:
        resume_path = Path(resume_from).resolve()
        resume_sha = file_sha256(resume_path)
        checkpoint = _checkpoint_load(
            resume_path,
            device=device,
            config=config,
            cache_file_sha256=cache_file_before,
            cache_content_sha256=cache_content,
            schedule=schedule,
        )
        module.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        completed_updates = int(checkpoint["completed_updates"])
        applied_updates = int(checkpoint["applied_updates"])
        amp_skips = int(checkpoint["amp_skips"])
        nonfinite_updates = int(checkpoint["nonfinite_updates"])
        exposure_history = list(checkpoint["exposure_history"])
        loss_history = list(checkpoint["loss_history"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
            )

    module.train()
    initial_scale = float(scaler.get_scale())
    with _deterministic_algorithms():
        for exposure in schedule[completed_updates:]:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=config.amp,
            ):
                loss, metrics = _task_loss(
                    module,
                    exposure,
                    rank_rows,
                    pairs,
                    config,
                    device=device,
                )
            if not bool(torch.isfinite(loss.detach()).item()):
                nonfinite_updates += 1
                raise PilotContractError(
                    f"non-finite {exposure['task']} loss at update {exposure['update']}"
                )
            scale_before = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if not _gradient_norm_is_finite(module):
                nonfinite_updates += 1
                raise PilotContractError(
                    f"non-finite gradient at update {exposure['update']}"
                )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                module.parameters(), config.clip_norm
            )
            if not bool(torch.isfinite(gradient_norm).item()):
                nonfinite_updates += 1
                raise PilotContractError(
                    f"non-finite clipped norm at update {exposure['update']}"
                )
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            skipped = config.amp and scale_after < scale_before
            amp_skips += int(skipped)
            applied_updates += int(not skipped)
            exposure_history.append(dict(exposure))
            loss_history.append(
                {
                    "update": exposure["update"],
                    "task": exposure["task"],
                    "gradient_norm_before_clip": float(gradient_norm.detach()),
                    "amp_scale_before": scale_before,
                    "amp_scale_after": scale_after,
                    "amp_skipped": bool(skipped),
                    **metrics,
                }
            )
        completed_updates = config.updates

    if exposure_history != list(schedule):
        raise PilotContractError("completed exposure history differs from schedule")
    ownership_report = responsibility_ownership_report(module)
    gradient_report = _final_gradient_audit(
        module, rank_rows, pairs, config, device=device
    )
    cache_file_after = file_sha256(cache_path)
    if cache_file_after != cache_file_before:
        raise PilotContractError("frozen cache bytes changed during training")
    cache_content_after = cached_candidate_content_sha256(
        load_cached_candidate_shard(cache_path)
    )
    if cache_content_after != cache_content:
        raise PilotContractError("frozen cache semantic content changed during training")

    checkpoint_payload = {
        "schema": PILOT_CHECKPOINT_SCHEMA,
        "config": config.checkpoint_contract(),
        "cache_file_sha256": cache_file_before,
        "cache_content_sha256": cache_content,
        "completed_updates": completed_updates,
        "applied_updates": applied_updates,
        "amp_skips": amp_skips,
        "nonfinite_updates": nonfinite_updates,
        "exposure_history": exposure_history,
        "loss_history": loss_history,
        "model_state_dict": module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if device.type == "cuda" else []
        ),
    }
    torch.save(checkpoint_payload, checkpoint_path)
    checkpoint_sha = file_sha256(checkpoint_path)
    model_state_sha = _tensor_mapping_sha256(module.state_dict())
    rank_updates = sum(
        int(item["task"] == CACHE_TASK_RANK) for item in exposure_history
    )
    confidence_updates = completed_updates - rank_updates
    receipt = {
        "schema": PILOT_RECEIPT_SCHEMA,
        "cache": {
            "path": str(cache_path),
            "file_sha256_before": cache_file_before,
            "file_sha256_after": cache_file_after,
            "content_sha256_before": cache_content,
            "content_sha256_after": cache_content_after,
            "unchanged": True,
            "requires_grad_tensor_count": cache_requires_grad,
            "shard_id": shard["shard_id"],
            "source": shard["source"],
            "row_count": len(shard["rows"]),
        },
        "config": {**config.checkpoint_contract(), "updates": config.updates},
        "schedule": {
            "schema": PILOT_SCHEDULE_SCHEMA,
            "sha256": schedule_sha,
            "exposures": exposure_history,
        },
        "updates": {
            "attempted": completed_updates,
            "applied": applied_updates,
            "rank": rank_updates,
            "confidence": confidence_updates,
            "nonfinite": nonfinite_updates,
        },
        "amp": {
            "enabled": config.amp,
            "initial_scale": initial_scale,
            "final_scale": float(scaler.get_scale()),
            "skip_count": amp_skips,
        },
        "ownership": ownership_report,
        "gradient_audit": gradient_report,
        "loss_history": loss_history,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "model_state_sha256": model_state_sha,
            "resume_from": (
                None
                if resume_from is None
                else {"path": str(Path(resume_from).resolve()), "sha256": resume_sha}
            ),
        },
        "runtime": {
            "torch_version": torch.__version__,
            "device": str(device),
            "deterministic_algorithms": True,
            "cublas_workspace_config": (
                os.environ.get("CUBLAS_WORKSPACE_CONFIG")
                if device.type == "cuda"
                else None
            ),
        },
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "checkpoint_path": checkpoint_path,
        "receipt_path": receipt_path,
        "receipt": receipt,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ownership", required=True, choices=RESPONSIBILITY_OWNERSHIP_MODES)
    parser.add_argument("--seed", required=True, type=int, choices=PILOT_SEEDS)
    parser.add_argument("--updates", required=True, type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume-from", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_cached_feature_pilot(
        cache_path=args.cache,
        output_dir=args.output_dir,
        config=PilotConfig(
            ownership=args.ownership,
            seed=args.seed,
            updates=args.updates,
            device=args.device,
            amp=args.amp,
        ),
        resume_from=args.resume_from,
    )
    print(json.dumps(result["receipt"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "CONFIDENCE_LOSS_CONTRACT",
    "PILOT_CHECKPOINT_SCHEMA",
    "PILOT_RECEIPT_SCHEMA",
    "PILOT_SCHEDULE_SCHEMA",
    "PILOT_SEEDS",
    "RANK_LOSS_CONTRACT",
    "PilotConfig",
    "PilotContractError",
    "build_interleaved_exposure_schedule",
    "confidence_pair_sample_max_bce_loss",
    "rank_listwise_preserve_loss",
    "run_cached_feature_pilot",
]
