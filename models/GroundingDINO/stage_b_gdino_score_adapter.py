"""Identity-initialized score adaptation for a frozen GroundingDINO model.

The adapter deliberately has no dependency on the patch Stage-B path.  Its two
outputs serve different decisions:

* ``rank_score`` may add a candidate-specific residual to the deployed base
  score and is trained only by localization/ranking objectives.
* ``confidence_score`` adds one image-expression offset to the frozen GDINO
  base score and is trained only by image-global confidence objectives.

Both input tensors are detached at the module boundary.  The confidence branch
owns its normalization/trunk and consumes only frozen query features plus the
detached base score.  The deployed confidence has no functional dependency on
the rank branch, so rank training cannot change FPR scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from groundingdino.util import box_ops


GDINO_TN_SCOPE_CODES = {
    "image_global_topk_verified": 1,
    "benchmark_dataft_alltn": 2,
    # D3 verifies the annotated target and every cached proposal, but not all
    # 900 runtime queries.  Keep this distinct from image-global/all-TN scope.
    "proposal_covered_verified": 3,
    # Clean Table-D rows keep their weaker source labels explicit.  They are
    # never aliases for benchmark/all-query or image-global verification.
    "unverified_all_negative": 4,
    "traceable_counterfactual_edit": 5,
}

GDINO_ADAPTER_TRAIN_MODE_CODES = {
    "rank_only": 1,
    "confidence_only": 2,
    "joint": 3,
}

GDINO_CONFIDENCE_OBJECTIVE_CODES = {
    "queue_q05_st": 1,
    "detached_recent_q05_trust": 2,
    # P3 protects the gate itself.  This variant protects the deployed
    # image-global score, which is the quantity used by the FPR evaluator.
    "detached_recent_q05_total_trust": 3,
    # Same deployed-score surrogate, with a deliberately weaker supervision
    # contract.  This name prevents a D3 run from claiming all-query labels.
    "detached_recent_q05_proposal_covered": 4,
    # Same deployed loss used by the clean D1/D2/D2m/D3m source ablation.
    # The configured TN scope, not this objective name, owns label strength.
    "detached_recent_q05_scope_labeled": 5,
}


def stage_b_gdino_tn_scope_code(scope: str) -> int:
    scope = str(scope).strip()
    if scope not in GDINO_TN_SCOPE_CODES:
        raise ValueError(
            "stage_b_gdino TN scope must be exactly one of "
            f"{sorted(GDINO_TN_SCOPE_CODES)}, got {scope!r}"
        )
    return GDINO_TN_SCOPE_CODES[scope]


def stage_b_gdino_adapter_train_mode_code(train_mode: str) -> int:
    train_mode = str(train_mode).strip()
    if train_mode not in GDINO_ADAPTER_TRAIN_MODE_CODES:
        raise ValueError(
            "stage_b_gdino_adapter_train_mode must be one of "
            f"{sorted(GDINO_ADAPTER_TRAIN_MODE_CODES)}, got {train_mode!r}"
        )
    return GDINO_ADAPTER_TRAIN_MODE_CODES[train_mode]


def stage_b_gdino_confidence_objective_code(objective: str) -> int:
    objective = str(objective).strip()
    if objective not in GDINO_CONFIDENCE_OBJECTIVE_CODES:
        raise ValueError(
            "stage_b_gdino confidence objective must be one of "
            f"{sorted(GDINO_CONFIDENCE_OBJECTIVE_CODES)}, got {objective!r}"
        )
    return GDINO_CONFIDENCE_OBJECTIVE_CODES[objective]


def validate_stage_b_gdino_score_adapter_checkpoint(
    model: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    checkpoint_label: str,
) -> None:
    """Require a complete, shape-compatible adapter for resume/evaluation."""

    root = model.module if hasattr(model, "module") else model
    adapter = getattr(root, "stage_b_gdino_score_adapter", None)
    if adapter is None:
        raise ValueError(
            f"{checkpoint_label}: model has no stage_b_gdino_score_adapter"
        )
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"{checkpoint_label}: checkpoint model state must be a mapping")
    prefix = "stage_b_gdino_score_adapter."
    expected = {prefix + key: value for key, value in adapter.state_dict().items()}
    provided = {
        str(key): value
        for key, value in state_dict.items()
        if str(key).startswith(prefix)
    }
    missing = sorted(set(expected).difference(provided))
    unexpected = sorted(set(provided).difference(expected))
    shape_mismatches = []
    for key in sorted(set(expected).intersection(provided)):
        value = provided[key]
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(expected[key].shape):
            shape_mismatches.append(
                (
                    key,
                    tuple(expected[key].shape),
                    tuple(value.shape) if torch.is_tensor(value) else type(value).__name__,
                )
            )
    if missing or unexpected or shape_mismatches:
        details = []
        if missing:
            details.append(f"missing={missing[:8]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:8]}")
        if shape_mismatches:
            details.append(f"shape_mismatches={shape_mismatches[:8]}")
        raise ValueError(
            f"{checkpoint_label}: incomplete stage_b_gdino_score_adapter "
            f"({'; '.join(details)}). Use --pretrain_model_path only for the "
            "frozen pure-GDINO base initialization; --resume/evaluation requires "
            "a complete adapter checkpoint."
        )


def aggregate_gdino_full_expression_score(
    token_logits: Tensor,
    expression_token_mask: Tensor,
) -> Tensor:
    """Match the pure-GDINO Ref/TN evaluator's full-expression aggregation.

    The evaluator averages sigmoid probabilities over all expression tokens for
    every query.  Requiring a non-empty mask prevents an invalid expression from
    silently becoming a zero-confidence training example.
    """

    if token_logits.dim() != 3:
        raise ValueError(
            "token_logits must have shape (B,Q,T), "
            f"got {tuple(token_logits.shape)}"
        )
    if not token_logits.is_floating_point():
        raise TypeError("token_logits must be floating point")
    mask = torch.as_tensor(expression_token_mask, device=token_logits.device)
    if mask.dim() == 3 and int(mask.shape[1]) == 1:
        mask = mask[:, 0]
    if mask.dim() != 2 or tuple(mask.shape) != (
        int(token_logits.shape[0]),
        int(token_logits.shape[2]),
    ):
        raise ValueError(
            "expression_token_mask must have shape (B,T) or (B,1,T), "
            f"got {tuple(mask.shape)}"
        )
    mask = mask.to(dtype=torch.bool)
    token_count = mask.sum(dim=-1)
    if bool((token_count == 0).any().item()):
        raise ValueError("every expression must contain at least one scored token")
    # The authoritative pure-GDINO evaluators cast logits to float32 before
    # sigmoid.  Keep this exact even when the forward runs under AMP.
    probability = token_logits.float().sigmoid()
    return (
        probability.masked_fill(~mask[:, None, :], 0.0).sum(dim=-1)
        / token_count[:, None].to(dtype=probability.dtype)
    )


def _validate_candidate_scores(
    scores: Tensor,
    candidate_mask: Optional[Tensor],
    *,
    name: str,
) -> Tensor:
    if scores.dim() != 2:
        raise ValueError(f"{name} must have shape (B,Q), got {tuple(scores.shape)}")
    if not scores.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if int(scores.shape[0]) == 0 or int(scores.shape[1]) == 0:
        raise ValueError(f"{name} must be non-empty")
    if candidate_mask is None:
        mask = torch.ones_like(scores, dtype=torch.bool)
    else:
        mask = torch.as_tensor(candidate_mask, device=scores.device, dtype=torch.bool)
        if tuple(mask.shape) != tuple(scores.shape):
            raise ValueError(
                f"{name} candidate_mask must have shape {tuple(scores.shape)}, "
                f"got {tuple(mask.shape)}"
            )
    if bool((~mask.any(dim=1)).any().item()):
        raise ValueError(f"every {name} row must contain a valid candidate")
    if not bool(torch.isfinite(scores[mask]).all().item()):
        raise ValueError(f"valid {name} entries must be finite")
    return mask


def _graph_zero(tensor: Tensor) -> Tensor:
    finite = torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))
    return (finite * 0.0).float().sum()


def b58_top1_anchored_rank_tail_score(
    base_score: Tensor,
    rank_score: Tensor,
    candidate_mask: Optional[Tensor] = None,
) -> Tensor:
    """Keep B58's top-1 query while allowing the rank tower to reorder its tail.

    The safety contract is structural rather than statistical: the query chosen
    by ``base_score`` is made strictly larger than every deployed rank-tail
    score.  This prevents a learned rank residual from regressing a B58-correct
    Ref example while retaining the trained residual ordering below top-1.
    """

    if tuple(rank_score.shape) != tuple(base_score.shape):
        raise ValueError("base_score and rank_score must share shape")
    mask = _validate_candidate_scores(
        base_score, candidate_mask, name="base_score"
    )
    _validate_candidate_scores(rank_score, mask, name="rank_score")
    masked_base = base_score.float().masked_fill(~mask, -torch.inf)
    masked_rank = rank_score.float().masked_fill(~mask, -torch.inf)
    base_top = masked_base.argmax(dim=1, keepdim=True)
    rank_max = masked_rank.max(dim=1, keepdim=True).values
    anchor = torch.nextafter(rank_max, torch.full_like(rank_max, torch.inf))
    if not bool(torch.isfinite(anchor).all().item()):
        raise ValueError("cannot construct a finite B58 top-1 rank anchor")
    guarded = rank_score.clone()
    guarded.scatter_(1, base_top, anchor.to(dtype=guarded.dtype))
    if not torch.equal(
        guarded.float().masked_fill(~mask, -torch.inf).argmax(dim=1),
        masked_base.argmax(dim=1),
    ):
        raise RuntimeError("B58 top-1 rank anchor failed")
    return guarded


def _distributed_masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Return a class-conditional mean with DDP-correct parameter gradients.

    Each rank owns only its local rows and DDP later averages parameter
    gradients.  Scaling the local sum by ``world_size / global_count`` makes
    that averaged gradient equal to a true mean over the selected global rows.
    Metric reduction across ranks likewise recovers the global forward mean.
    """

    if values.dim() != 1 or tuple(mask.shape) != tuple(values.shape):
        raise ValueError("masked mean values/mask must share one-dimensional shape")
    mask = mask.to(device=values.device, dtype=torch.bool)
    local_count = mask.sum().to(dtype=torch.float32)
    global_count = local_count.detach().clone()
    world_size = 1
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        world_size = int(dist.get_world_size())
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    if int(global_count.item()) == 0:
        return _graph_zero(values)
    local_sum = values.masked_fill(~mask, 0.0).sum()
    return local_sum * (float(world_size) / global_count)


def multi_positive_listwise_rank_loss(
    rank_score: Tensor,
    positive_mask: Tensor,
    *,
    eligible_mask: Optional[Tensor] = None,
    temperature: float = 0.2,
) -> Tensor:
    """Multi-positive listwise loss over eligible candidates in each image.

    ``eligible_mask`` should exclude ambiguous IoU candidates.  A row contributes
    only when it has at least one positive and at least one eligible negative.
    """

    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if rank_score.dim() != 2 or not rank_score.is_floating_point():
        raise ValueError("rank_score must be a floating-point tensor with shape (B,Q)")
    if int(rank_score.shape[0]) == 0 or int(rank_score.shape[1]) == 0:
        raise ValueError("rank_score must be non-empty")
    if eligible_mask is None:
        candidate_mask = torch.ones_like(rank_score, dtype=torch.bool)
    else:
        if not torch.is_tensor(eligible_mask) or eligible_mask.dtype != torch.bool:
            raise ValueError("eligible_mask must be an exact boolean tensor")
        candidate_mask = eligible_mask.to(device=rank_score.device)
        if tuple(candidate_mask.shape) != tuple(rank_score.shape):
            raise ValueError("eligible_mask must match rank_score")
    if not bool(torch.isfinite(rank_score[candidate_mask]).all().item()):
        raise ValueError("eligible rank_score entries must be finite")
    if not torch.is_tensor(positive_mask) or positive_mask.dtype != torch.bool:
        raise ValueError("positive_mask must be an exact boolean tensor")
    positive_mask = positive_mask.to(device=rank_score.device)
    if tuple(positive_mask.shape) != tuple(rank_score.shape):
        raise ValueError("positive_mask must match rank_score")
    if bool((positive_mask & ~candidate_mask).any().item()):
        raise ValueError("positive_mask must be a subset of eligible_mask")

    losses = []
    scaled_score = rank_score.float() / float(temperature)
    for batch_index in range(int(rank_score.shape[0])):
        row_eligible = candidate_mask[batch_index]
        row_positive = positive_mask[batch_index]
        row_negative = row_eligible & (~row_positive)
        if not bool(row_positive.any().item()) or not bool(row_negative.any().item()):
            continue
        positive_log_mass = torch.logsumexp(
            scaled_score[batch_index, row_positive], dim=0
        )
        eligible_log_mass = torch.logsumexp(
            scaled_score[batch_index, row_eligible], dim=0
        )
        losses.append(eligible_log_mass - positive_log_mass)
    if not losses:
        return _graph_zero(rank_score)
    return torch.stack(losses).mean()


@dataclass(frozen=True)
class BaselinePreservingRankOutput:
    loss: Tensor
    margin_loss: Tensor
    fix_loss: Tensor
    preserve_loss: Tensor
    residual_loss: Tensor
    valid_rows: Tensor
    fix_rows: Tensor
    preserve_rows: Tensor
    rows_no_positive: Tensor
    base_correct: Tensor
    adapted_correct: Tensor
    wrong_fixed: Tensor
    correct_regressed: Tensor


def baseline_preserving_top1_rank_loss(
    rank_score: Tensor,
    base_score: Tensor,
    rank_residual: Tensor,
    candidate_iou: Tensor,
    *,
    iou_threshold: float = 0.5,
    fix_margin: float = 0.05,
    preserve_margin: float = 0.02,
    temperature: float = 0.1,
    residual_weight: float = 1e-3,
) -> BaselinePreservingRankOutput:
    """Improve wrong top-1 rows while preserving already-correct GDINO rows.

    Every candidate below ``iou_threshold`` is a deployed top-1 error.  Rows
    whose frozen GDINO top-1 is wrong receive a smooth positive-vs-hardest-
    negative margin.  Correct rows receive only a one-sided constraint that
    permits their frozen positive-negative gap to shrink by at most
    ``preserve_margin``.  The two row classes are normalized separately so
    abundant already-correct rows cannot dilute the repair gradient.
    """

    if rank_score.dim() != 2 or not rank_score.is_floating_point():
        raise ValueError("rank_score must be floating point with shape (B,Q)")
    for name, value in (
        ("base_score", base_score),
        ("rank_residual", rank_residual),
        ("candidate_iou", candidate_iou),
    ):
        if tuple(value.shape) != tuple(rank_score.shape):
            raise ValueError(f"{name} must match rank_score")
    if not 0.0 < float(iou_threshold) <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    if float(fix_margin) < 0.0 or float(preserve_margin) < 0.0:
        raise ValueError("rank margins must be non-negative")
    if float(temperature) <= 0.0 or float(residual_weight) < 0.0:
        raise ValueError("rank temperature must be positive and residual_weight non-negative")
    if not bool(torch.isfinite(rank_score).all().item()):
        raise ValueError("rank_score must be finite")
    if not bool(torch.isfinite(base_score).all().item()):
        raise ValueError("base_score must be finite")

    positive = candidate_iou >= float(iou_threshold)
    negative = candidate_iou < float(iou_threshold)
    valid = positive.any(dim=1) & negative.any(dim=1)

    rank_positive = rank_score.masked_fill(~positive, -torch.inf).max(dim=1).values
    rank_negative = rank_score.masked_fill(~negative, -torch.inf).max(dim=1).values
    base_positive = base_score.masked_fill(~positive, -torch.inf).max(dim=1).values
    base_negative = base_score.masked_fill(~negative, -torch.inf).max(dim=1).values
    rank_gap = torch.where(valid, rank_positive - rank_negative, torch.zeros_like(rank_positive))
    base_gap = torch.where(valid, base_positive - base_negative, torch.zeros_like(base_positive))

    base_top = base_score.argmax(dim=1)
    adapted_top = rank_score.argmax(dim=1)
    base_correct = positive.gather(1, base_top[:, None]).squeeze(1) & positive.any(dim=1)
    adapted_correct = positive.gather(1, adapted_top[:, None]).squeeze(1) & positive.any(dim=1)

    tau = float(temperature)
    fix_row_mask = valid & (~base_correct)
    preserve_row_mask = valid & base_correct
    fix_row_loss = tau * F.softplus((float(fix_margin) - rank_gap) / tau)
    preserve_target = (base_gap.detach() - float(preserve_margin)).clamp_min(0.0)
    preserve_row_loss = F.relu(preserve_target - rank_gap)
    fix_loss = _distributed_masked_mean(fix_row_loss, fix_row_mask)
    preserve_loss = _distributed_masked_mean(
        preserve_row_loss, preserve_row_mask
    )
    margin_loss = fix_loss + preserve_loss
    residual_loss = rank_residual.float().square().mean()
    loss = margin_loss + float(residual_weight) * residual_loss
    return BaselinePreservingRankOutput(
        loss=loss,
        margin_loss=margin_loss,
        fix_loss=fix_loss,
        preserve_loss=preserve_loss,
        residual_loss=residual_loss,
        valid_rows=valid.float().sum().detach(),
        fix_rows=fix_row_mask.float().sum().detach(),
        preserve_rows=preserve_row_mask.float().sum().detach(),
        rows_no_positive=(~positive.any(dim=1)).float().sum().detach(),
        base_correct=base_correct.float().sum().detach(),
        adapted_correct=adapted_correct.float().sum().detach(),
        wrong_fixed=((~base_correct) & adapted_correct).float().sum().detach(),
        correct_regressed=(base_correct & (~adapted_correct)).float().sum().detach(),
    )


def image_expression_global_max(
    candidate_score: Tensor,
    candidate_mask: Optional[Tensor] = None,
    *,
    name: str = "candidate_score",
) -> Tensor:
    """Return the deployed image-expression score: max over valid queries."""

    mask = _validate_candidate_scores(candidate_score, candidate_mask, name=name)
    return candidate_score.float().masked_fill(~mask, -torch.inf).max(dim=1).values


def exact_tpr_operating_threshold(
    positive_global_score: Tensor,
    *,
    target_tpr: float = 0.95,
) -> Tensor:
    """Use the evaluator's exact order statistic for its ``score >= threshold`` rule."""

    score = positive_global_score.float().reshape(-1)
    if score.numel() == 0 or not bool(torch.isfinite(score).all().item()):
        raise ValueError("positive_global_score must be non-empty and finite")
    if not 0.0 < float(target_tpr) <= 1.0:
        raise ValueError("target_tpr must be in (0, 1]")
    accepted = max(1, int(math.ceil(float(target_tpr) * int(score.numel()))))
    # This exactly matches the final evaluator's
    # ascending_index = N - ceil(TPR*N) order statistic. ``torch.kthvalue``
    # has no deterministic CUDA implementation. Full sorting is deterministic
    # on the supported CUDA runtime and exactly value-equivalent, including
    # ties, so use it for formal training and CPU replay alike.
    ascending_index = int(score.numel()) - accepted
    return torch.sort(score, stable=True).values[ascending_index]


def distributed_gather_1d_with_local_grad(value: Tensor) -> tuple[Tensor, int]:
    """Gather a variable-length vector while retaining this rank's graph.

    Remote rank values are constants in the local autograd graph.  Every rank
    reconstructs the same rank-ordered global vector, while the local slice is
    replaced by the original tensor so an order statistic selected on this
    rank can propagate its gradient.  The caller must compensate for DDP's
    parameter-gradient averaging when differentiating a global mean.
    """

    if value.dim() != 1 or not value.is_floating_point():
        raise ValueError("distributed score gather requires a floating 1-D tensor")
    if value.numel() == 0 or not bool(torch.isfinite(value).all().item()):
        raise ValueError("distributed score gather requires non-empty finite values")
    if not (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        return value, 1

    world_size = int(dist.get_world_size())
    rank = int(dist.get_rank())
    local_count = torch.as_tensor(
        [int(value.numel())], dtype=torch.int64, device=value.device
    )
    gathered_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
    dist.all_gather(gathered_counts, local_count)
    counts = [int(item.item()) for item in gathered_counts]
    if any(count <= 0 for count in counts):
        raise RuntimeError(
            "every distributed rank must contribute at least one FPR score"
        )
    if counts[rank] != int(value.numel()):
        raise RuntimeError("distributed score count drifted on the local rank")

    max_count = max(counts)
    padded = value.detach().new_zeros((max_count,))
    padded[: int(value.numel())] = value.detach()
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    parts = []
    for source_rank, (remote, count) in enumerate(zip(gathered, counts)):
        parts.append(value if source_rank == rank else remote[:count].detach())
    return torch.cat(parts, dim=0), world_size


def _preserve_value_scale_local_gradient(value: Tensor, scale: int) -> Tensor:
    if int(scale) <= 0:
        raise ValueError("gradient scale must be positive")
    if int(scale) == 1:
        return value
    # All ranks evaluate the same global mean but only own one differentiable
    # slice.  DDP averages those parameter gradients, so compensate here while
    # preserving the exact forward value reported and checkpointed.
    detached = value.detach()
    return detached + float(scale) * (value - detached)


@dataclass(frozen=True)
class FPR95SurrogateOutput:
    loss: Tensor
    negative_loss: Tensor
    positive_threshold: Tensor
    current_positive_threshold: Tensor
    threshold_drift: Tensor
    surrogate_threshold: Tensor
    paired_separation_loss: Tensor
    positive_global_score: Tensor
    negative_global_score: Tensor
    local_positive_global_score: Tensor
    local_negative_global_score: Tensor
    exact_tpr: Tensor
    exact_fpr: Tensor
    current_exact_tpr: Tensor
    current_exact_fpr: Tensor


def fpr95_global_max_surrogate(
    positive_candidate_score: Tensor,
    negative_candidate_score: Tensor,
    *,
    positive_candidate_mask: Optional[Tensor] = None,
    negative_candidate_mask: Optional[Tensor] = None,
    positive_history: Optional[Tensor] = None,
    negative_history: Optional[Tensor] = None,
    temperature: float = 0.1,
    margin: float = 0.0,
    target_tpr: float = 0.95,
    paired_margin_weight: float = 0.0,
    paired_margin: float = 0.0,
) -> FPR95SurrogateOutput:
    """Smooth the actual FPR@95TPR decision over every negative global maximum.

    Positive history estimates the operating q05 and is detached.  Negative
    history is accepted only for queue-schema compatibility and never enters the
    loss mean: every *current* negative keeps weight ``1/current_batch_size``.
    Unlike a negative-q95 tail loss, every current negative example contributes.
    """

    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if float(paired_margin_weight) < 0.0:
        raise ValueError("paired_margin_weight must be non-negative")
    local_positive_global = image_expression_global_max(
        positive_candidate_score,
        positive_candidate_mask,
        name="positive_candidate_score",
    )
    local_negative_global = image_expression_global_max(
        negative_candidate_score,
        negative_candidate_mask,
        name="negative_candidate_score",
    )
    if local_negative_global.device != local_positive_global.device:
        raise ValueError("positive and negative candidate scores must share a device")
    positive_global, positive_world_size = distributed_gather_1d_with_local_grad(
        local_positive_global
    )
    negative_global, negative_world_size = distributed_gather_1d_with_local_grad(
        local_negative_global
    )
    if positive_world_size != negative_world_size:
        raise RuntimeError("positive and negative distributed world sizes differ")

    positive_bank = positive_global
    if positive_history is not None:
        history = torch.as_tensor(
            positive_history,
            device=positive_global.device,
            dtype=torch.float32,
        ).detach().reshape(-1)
        if history.numel() and not bool(torch.isfinite(history).all().item()):
            raise ValueError("positive_history must be finite")
        if history.numel():
            positive_bank = torch.cat((history, positive_global), dim=0)

    if negative_history is not None:
        history = torch.as_tensor(
            negative_history,
            device=negative_global.device,
            dtype=torch.float32,
        ).detach().reshape(-1)
        if history.numel() and not bool(torch.isfinite(history).all().item()):
            raise ValueError("negative_history must be finite")

    bank_threshold = exact_tpr_operating_threshold(
        positive_bank, target_tpr=float(target_tpr)
    )
    current_threshold = exact_tpr_operating_threshold(
        positive_global, target_tpr=float(target_tpr)
    )
    # Forward value follows the stable score bank, while the straight-through
    # term guarantees that current positives still receive the q05 gradient
    # after the selected bank order statistic comes from detached history.
    surrogate_threshold = (
        bank_threshold.detach()
        + current_threshold
        - current_threshold.detach()
    )
    tau = float(temperature)
    fpr_loss = tau * F.softplus(
        (negative_global - surrogate_threshold + float(margin)) / tau
    ).mean()
    paired_loss = _graph_zero(negative_global)
    if float(paired_margin_weight) > 0.0:
        if positive_global.shape != negative_global.shape:
            raise ValueError(
                "paired separation requires aligned positive and negative batches"
            )
        paired_loss = tau * F.softplus(
            (
                negative_global
                - positive_global
                + float(paired_margin)
            )
            / tau
        ).mean()
    unscaled_loss = fpr_loss + float(paired_margin_weight) * paired_loss
    loss = _preserve_value_scale_local_gradient(
        unscaled_loss, positive_world_size
    )
    with torch.no_grad():
        exact_tpr = (positive_bank >= bank_threshold).float().mean()
        exact_fpr = (negative_global >= bank_threshold).float().mean()
        current_exact_tpr = (
            positive_global >= current_threshold
        ).float().mean()
        current_exact_fpr = (
            negative_global >= current_threshold
        ).float().mean()
    return FPR95SurrogateOutput(
        loss=loss,
        negative_loss=fpr_loss,
        positive_threshold=bank_threshold,
        current_positive_threshold=current_threshold,
        threshold_drift=current_threshold.detach() - bank_threshold.detach(),
        surrogate_threshold=surrogate_threshold,
        paired_separation_loss=paired_loss,
        positive_global_score=positive_global,
        negative_global_score=negative_global,
        local_positive_global_score=local_positive_global,
        local_negative_global_score=local_negative_global,
        exact_tpr=exact_tpr,
        exact_fpr=exact_fpr,
        current_exact_tpr=current_exact_tpr,
        current_exact_fpr=current_exact_fpr,
    )


@dataclass(frozen=True)
class DetachedRecentQ05TrustOutput:
    loss: Tensor
    negative_loss: Tensor
    positive_trust_loss: Tensor
    positive_score_trust_loss: Tensor
    positive_threshold: Tensor
    current_positive_threshold: Tensor
    threshold_drift: Tensor
    paired_separation_loss: Tensor
    positive_global_score: Tensor
    negative_global_score: Tensor
    positive_global_gate: Tensor
    negative_global_gate: Tensor
    local_positive_global_score: Tensor
    local_negative_global_score: Tensor
    exact_tpr: Tensor
    exact_fpr: Tensor
    current_exact_tpr: Tensor
    current_exact_fpr: Tensor
    positive_trust_violation_rate: Tensor
    positive_score_trust_violation_rate: Tensor


def _validate_expression_gate(
    gate: Tensor,
    *,
    batch_size: int,
    device: torch.device,
    name: str,
) -> Tensor:
    if not torch.is_tensor(gate) or gate.dim() != 1:
        raise ValueError(f"{name} must be a floating tensor with shape (B,)")
    if not gate.is_floating_point() or int(gate.numel()) != int(batch_size):
        raise ValueError(f"{name} must be a floating tensor with shape (B,)")
    if gate.device != device:
        raise ValueError(f"{name} must share the confidence-score device")
    if not bool(torch.isfinite(gate).all().item()):
        raise ValueError(f"{name} must be finite")
    return gate


def detached_recent_q05_trust_surrogate(
    positive_candidate_score: Tensor,
    negative_candidate_score: Tensor,
    positive_gate: Tensor,
    negative_gate: Tensor,
    *,
    positive_candidate_mask: Optional[Tensor] = None,
    negative_candidate_mask: Optional[Tensor] = None,
    positive_history: Optional[Tensor] = None,
    temperature: float = 0.1,
    margin: float = 0.0,
    target_tpr: float = 0.95,
    positive_trust_margin: float = 0.02,
    positive_trust_weight: float = 1.0,
    paired_margin_weight: float = 0.0,
    paired_margin: float = 0.0,
    positive_score_trust: bool = False,
) -> DetachedRecentQ05TrustOutput:
    """Optimize current TNs against a detached recent q05 and protect positives.

    Once history is supplied, it alone defines the forward threshold.  Before
    queue warmup, the exact current-global q05 is detached.  No straight-through
    gradient is attached to a current-batch order statistic.  A zero-valued
    global-mean positive-gate proxy carries only the common-translation gradient,
    so lowering every positive and TN score cannot reduce this loss.  When
    ``positive_score_trust`` is true, the positive protection is applied to
    the final deployed image-global score instead of the auxiliary gate.  The
    latter is the objective used by the total-trust confidence branch.
    """

    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if float(positive_trust_margin) < 0.0:
        raise ValueError("positive_trust_margin must be non-negative")
    if float(positive_trust_weight) < 0.0:
        raise ValueError("positive_trust_weight must be non-negative")
    if float(paired_margin_weight) < 0.0:
        raise ValueError("paired_margin_weight must be non-negative")
    if not isinstance(positive_score_trust, bool):
        raise TypeError("positive_score_trust must be a bool")

    local_positive_global = image_expression_global_max(
        positive_candidate_score,
        positive_candidate_mask,
        name="positive_candidate_score",
    )
    local_negative_global = image_expression_global_max(
        negative_candidate_score,
        negative_candidate_mask,
        name="negative_candidate_score",
    )
    if local_negative_global.device != local_positive_global.device:
        raise ValueError("positive and negative candidate scores must share a device")
    positive_gate = _validate_expression_gate(
        positive_gate,
        batch_size=int(local_positive_global.numel()),
        device=local_positive_global.device,
        name="positive_gate",
    )
    negative_gate = _validate_expression_gate(
        negative_gate,
        batch_size=int(local_negative_global.numel()),
        device=local_negative_global.device,
        name="negative_gate",
    )

    positive_global, world_size = distributed_gather_1d_with_local_grad(
        local_positive_global
    )
    negative_global, negative_world_size = distributed_gather_1d_with_local_grad(
        local_negative_global
    )
    positive_gate_global, positive_gate_world_size = (
        distributed_gather_1d_with_local_grad(positive_gate)
    )
    negative_gate_global, negative_gate_world_size = (
        distributed_gather_1d_with_local_grad(negative_gate)
    )
    if len(
        {
            int(world_size),
            int(negative_world_size),
            int(positive_gate_world_size),
            int(negative_gate_world_size),
        }
    ) != 1:
        raise RuntimeError("confidence scores and gates have different world sizes")
    if positive_global.shape != negative_global.shape:
        raise ValueError("positive and negative global score batches must align")
    if positive_gate_global.shape != positive_global.shape or (
        negative_gate_global.shape != negative_global.shape
    ):
        raise ValueError("confidence gates must align with global score batches")

    current_threshold = exact_tpr_operating_threshold(
        positive_global, target_tpr=float(target_tpr)
    ).detach()
    threshold_bank = positive_global.detach()
    if positive_history is not None:
        history = torch.as_tensor(
            positive_history,
            device=positive_global.device,
            dtype=torch.float32,
        ).detach().reshape(-1)
        if history.numel() == 0 or not bool(torch.isfinite(history).all().item()):
            raise ValueError("positive_history must be non-empty and finite")
        # P3 intentionally excludes current scores once the recent queue is warm.
        threshold_bank = history
    bank_threshold = exact_tpr_operating_threshold(
        threshold_bank, target_tpr=float(target_tpr)
    ).detach()
    # The translation proxy must use the same score consumed by evaluation.
    # P3 historically used the gate; total-trust uses the deployed global max.
    positive_translation = (
        positive_global.mean()
        if positive_score_trust
        else positive_gate_global.mean()
    )
    surrogate_threshold = (
        bank_threshold + positive_translation - positive_translation.detach()
    )

    tau = float(temperature)
    negative_loss = tau * F.softplus(
        (negative_global - surrogate_threshold + float(margin)) / tau
    ).mean()
    if positive_score_trust:
        # Keep the final positive q05 above a detached floor.  The tolerance
        # mirrors P3's gate contract: a margin of 0.02 permits a small drop
        # below the historical q05 while still supplying a tail gradient.
        score_floor = bank_threshold - float(positive_trust_margin)
        trust_violation = F.relu(score_floor - positive_global)
        gate_violation_rate = positive_global.new_zeros(())
    else:
        trust_violation = F.relu(
            -float(positive_trust_margin) - positive_gate_global
        )
        gate_violation_rate = (
            positive_gate_global < -float(positive_trust_margin)
        ).float().mean()
    positive_trust_loss = trust_violation.mean()
    positive_score_trust_loss = (
        positive_trust_loss if positive_score_trust else _graph_zero(positive_global)
    )
    paired_loss = _graph_zero(negative_global)
    if float(paired_margin_weight) > 0.0:
        paired_loss = tau * F.softplus(
            (
                negative_global
                - positive_global
                + float(paired_margin)
            )
            / tau
        ).mean()
    unscaled_loss = (
        negative_loss
        + float(positive_trust_weight) * positive_trust_loss
        + float(paired_margin_weight) * paired_loss
    )
    loss = _preserve_value_scale_local_gradient(unscaled_loss, int(world_size))

    with torch.no_grad():
        exact_tpr = (threshold_bank >= bank_threshold).float().mean()
        exact_fpr = (negative_global >= bank_threshold).float().mean()
        current_exact_tpr = (
            positive_global >= current_threshold
        ).float().mean()
        current_exact_fpr = (
            negative_global >= current_threshold
        ).float().mean()
        violation_rate = gate_violation_rate
        score_violation_rate = (
            (
                positive_global
                < (bank_threshold - float(positive_trust_margin))
            ).float().mean()
            if positive_score_trust
            else positive_global.new_zeros(())
        )
    return DetachedRecentQ05TrustOutput(
        loss=loss,
        negative_loss=negative_loss,
        positive_trust_loss=positive_trust_loss,
        positive_score_trust_loss=positive_score_trust_loss,
        positive_threshold=bank_threshold,
        current_positive_threshold=current_threshold,
        threshold_drift=current_threshold - bank_threshold,
        paired_separation_loss=paired_loss,
        positive_global_score=positive_global,
        negative_global_score=negative_global,
        positive_global_gate=positive_gate_global,
        negative_global_gate=negative_gate_global,
        local_positive_global_score=local_positive_global,
        local_negative_global_score=local_negative_global,
        exact_tpr=exact_tpr,
        exact_fpr=exact_fpr,
        current_exact_tpr=current_exact_tpr,
        current_exact_fpr=current_exact_fpr,
        positive_trust_violation_rate=violation_rate,
        positive_score_trust_violation_rate=score_violation_rate,
    )


class StageBGDINOScoreAdapter(nn.Module):
    """Two isolated score branches on top of frozen final-query features."""

    score_feature_dim = 6

    def __init__(
        self,
        hidden_dim: int,
        *,
        adapter_dim: int = 128,
        gate_hidden_dim: int = 128,
        gate_pool_temperature: float = 0.1,
        gate_topk: int = 10,
        u2v5_score_ownership: str = "isolated_heads",
    ) -> None:
        super().__init__()
        if int(hidden_dim) <= 0 or int(adapter_dim) <= 0 or int(gate_hidden_dim) <= 0:
            raise ValueError("hidden dimensions must be positive")
        if float(gate_pool_temperature) <= 0.0:
            raise ValueError("gate_pool_temperature must be positive")
        if int(gate_topk) <= 0:
            raise ValueError("gate_topk must be positive")
        self.hidden_dim = int(hidden_dim)
        self.adapter_dim = int(adapter_dim)
        self.gate_pool_temperature = float(gate_pool_temperature)
        self.gate_topk = int(gate_topk)
        self.u2v5_score_ownership = str(u2v5_score_ownership).strip()
        if self.u2v5_score_ownership not in {
            "isolated_heads", "shared_score", "shared_trunk_two_heads",
        }:
            raise ValueError("invalid U2-v5 score ownership")

        self.rank_norm = nn.LayerNorm(self.hidden_dim)
        self.rank_trunk = nn.Sequential(
            nn.Linear(self.hidden_dim + 1, self.adapter_dim),
            nn.GELU(),
            nn.Linear(self.adapter_dim, self.adapter_dim),
            nn.GELU(),
        )
        self.rank_output = nn.Linear(self.adapter_dim, 1)

        self.confidence_norm = nn.LayerNorm(self.hidden_dim)
        self.confidence_trunk = nn.Sequential(
            nn.Linear(self.hidden_dim + 1, self.adapter_dim),
            nn.GELU(),
            nn.Linear(self.adapter_dim, self.adapter_dim),
            nn.GELU(),
        )
        self.confidence_gate = nn.Sequential(
            nn.Linear(
                self.adapter_dim + self.score_feature_dim,
                int(gate_hidden_dim),
            ),
            nn.GELU(),
            nn.Linear(int(gate_hidden_dim), int(gate_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(gate_hidden_dim), 1),
        )
        nn.init.zeros_(self.rank_output.weight)
        nn.init.zeros_(self.rank_output.bias)
        nn.init.zeros_(self.confidence_gate[-1].weight)
        nn.init.zeros_(self.confidence_gate[-1].bias)

    def rank_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.rank_norm.parameters()) + tuple(
            self.rank_trunk.parameters()
        ) + tuple(self.rank_output.parameters())

    def gate_parameters(self) -> tuple[nn.Parameter, ...]:
        return (
            tuple(self.confidence_norm.parameters())
            + tuple(self.confidence_trunk.parameters())
            + tuple(self.confidence_gate.parameters())
        )

    def _gate_inputs(
        self,
        confidence_feature: Tensor,
        base_score: Tensor,
        candidate_mask: Tensor,
    ) -> Tensor:
        # ``query_hs`` and ``base_score`` are detached once at the adapter
        # boundary.  Do not detach ``confidence_feature`` here: doing so would
        # silently make confidence_norm/confidence_trunk untrainable.
        feature = confidence_feature.float()
        score = base_score.float()
        valid_count = candidate_mask.sum(dim=1)
        masked_score = score.masked_fill(~candidate_mask, -torch.inf)
        weights = torch.softmax(
            masked_score / self.gate_pool_temperature, dim=1
        ).masked_fill(~candidate_mask, 0.0)
        pooled_feature = torch.einsum("bq,bqd->bd", weights, feature)

        score_max = masked_score.max(dim=1).values
        count_float = valid_count.float()
        score_mean = score.masked_fill(~candidate_mask, 0.0).sum(dim=1) / count_float
        centered = (score - score_mean[:, None]).masked_fill(~candidate_mask, 0.0)
        score_std = (
            centered.square().sum(dim=1) / count_float
        ).clamp_min(0.0).sqrt()

        top_count = min(self.gate_topk, int(score.shape[1]))
        top_values = torch.topk(
            masked_score, k=top_count, dim=1, largest=True, sorted=True
        ).values
        top_valid = torch.arange(top_count, device=score.device)[None, :] < torch.minimum(
            valid_count, valid_count.new_full(valid_count.shape, top_count)
        )[:, None]
        score_top_mean = top_values.masked_fill(~top_valid, 0.0).sum(dim=1) / top_valid.sum(
            dim=1
        ).clamp_min(1)
        if int(score.shape[1]) > 1:
            top_two = torch.topk(
                masked_score, k=2, dim=1, largest=True, sorted=True
            ).values
            score_margin = torch.where(
                valid_count > 1,
                top_two[:, 0] - top_two[:, 1],
                torch.zeros_like(score_max),
            )
        else:
            score_margin = torch.zeros_like(score_max)

        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=1)
        score_entropy = torch.where(
            valid_count > 1,
            entropy / count_float.log().clamp_min(1e-8),
            torch.zeros_like(entropy),
        )
        score_features = torch.stack(
            (
                score_max,
                score_top_mean,
                score_mean,
                score_std,
                score_margin,
                score_entropy,
            ),
            dim=-1,
        )
        return torch.cat((pooled_feature, score_features), dim=-1)

    def forward(
        self,
        query_hs: Tensor,
        base_score: Tensor,
        candidate_mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        if query_hs.dim() != 3:
            raise ValueError(
                f"query_hs must have shape (B,Q,D), got {tuple(query_hs.shape)}"
            )
        if not query_hs.is_floating_point():
            raise TypeError("query_hs must be floating point")
        if int(query_hs.shape[-1]) != self.hidden_dim:
            raise ValueError(
                f"query_hs hidden dimension must be {self.hidden_dim}, "
                f"got {query_hs.shape[-1]}"
            )
        if tuple(query_hs.shape[:2]) != tuple(base_score.shape):
            raise ValueError("query_hs and base_score must share (B,Q)")
        mask = _validate_candidate_scores(
            base_score, candidate_mask, name="base_score"
        ).to(device=query_hs.device)

        # The frozen Stage-B checkpoint remains outside both autograd graphs.
        hs = query_hs.detach()
        base = base_score.detach().to(device=hs.device)
        safe_base = base.masked_fill(~mask, 0.0).to(dtype=hs.dtype)
        rank_input = torch.cat(
            (self.rank_norm(hs), safe_base.unsqueeze(-1)), dim=-1
        )
        rank_feature = self.rank_trunk(rank_input)
        rank_residual = self.rank_output(rank_feature).squeeze(-1).to(dtype=base.dtype)
        rank_residual = rank_residual.masked_fill(~mask, 0.0)
        rank_score = base + rank_residual

        if self.u2v5_score_ownership == "shared_score":
            confidence_feature = rank_feature
            masked_residual = rank_residual.masked_fill(~mask, -torch.inf)
            gate = masked_residual.max(dim=1).values
            confidence_score = rank_score
        else:
            if self.u2v5_score_ownership == "shared_trunk_two_heads":
                confidence_feature = rank_feature
            else:
                confidence_input = torch.cat(
                    (self.confidence_norm(hs), safe_base.unsqueeze(-1)), dim=-1
                )
                confidence_feature = self.confidence_trunk(confidence_input)
            gate_input = self._gate_inputs(confidence_feature, base, mask)
            gate = self.confidence_gate(gate_input).squeeze(-1).to(dtype=base.dtype)
            confidence_score = base + gate[:, None]
        return {
            "base_score": base,
            "rank_feature": rank_feature,
            "rank_residual": rank_residual,
            "rank_score": rank_score,
            "confidence_feature": confidence_feature,
            "confidence_gate": gate,
            "confidence_score": confidence_score,
            "candidate_mask": mask,
        }


def _candidate_max_iou(
    candidate_boxes: Tensor,
    targets: Sequence[Mapping[str, Any]],
) -> Tensor:
    if candidate_boxes.dim() != 3 or int(candidate_boxes.shape[-1]) != 4:
        raise ValueError("candidate boxes must have shape (B,Q,4)")
    if len(targets) != int(candidate_boxes.shape[0]):
        raise ValueError("targets must align with candidate boxes")
    candidate_xyxy = box_ops.box_cxcywh_to_xyxy(candidate_boxes.detach().float())
    result = candidate_xyxy.new_zeros(candidate_xyxy.shape[:2])
    for batch_index, target in enumerate(targets):
        target_boxes = target.get("boxes", None)
        if not torch.is_tensor(target_boxes) or target_boxes.numel() == 0:
            continue
        target_xyxy = box_ops.box_cxcywh_to_xyxy(
            target_boxes.detach().to(device=result.device, dtype=torch.float32).reshape(-1, 4)
        )
        iou, _ = box_ops.box_iou(candidate_xyxy[batch_index], target_xyxy)
        if iou.numel():
            result[batch_index] = iou.max(dim=1).values
    return result


def _strict_scalar_bool(target: Mapping[str, Any], key: str) -> bool:
    value = target.get(key, None)
    if torch.is_tensor(value):
        return bool(
            value.dtype == torch.bool
            and value.numel() == 1
            and value.detach().reshape(-1)[0].item() is True
        )
    return value is True


def _strict_scalar_false(target: Mapping[str, Any], key: str) -> bool:
    value = target.get(key, None)
    if torch.is_tensor(value):
        return bool(
            value.dtype == torch.bool
            and value.numel() == 1
            and value.detach().reshape(-1)[0].item() is False
        )
    return value is False


class StageBGDINOScoreAdapterCriterion(nn.Module):
    """Phase-isolated rank and scope-checked FPR@95TPR objectives."""

    def __init__(
        self,
        *,
        tn_scope: str = "",
        train_mode: str = "joint",
        confidence_objective: str = "queue_q05_st",
        positive_iou_threshold: float = 0.5,
        negative_iou_threshold: float = 0.5,
        listwise_temperature: float = 0.2,
        rank_fix_margin: float = 0.05,
        rank_preserve_margin: float = 0.02,
        rank_residual_weight: float = 1e-3,
        rank_weight: float = 1.0,
        confidence_weight: float = 1.0,
        fpr_temperature: float = 0.1,
        fpr_margin: float = 0.0,
        paired_margin_weight: float = 0.25,
        paired_margin: float = 0.05,
        positive_trust_margin: float = 0.02,
        positive_trust_weight: float = 1.0,
        queue_size: int = 4096,
        queue_min_count: int = 256,
    ) -> None:
        super().__init__()
        if not 0.0 < float(positive_iou_threshold) <= 1.0:
            raise ValueError("positive_iou_threshold must be in (0, 1]")
        if not math.isclose(
            float(negative_iou_threshold),
            float(positive_iou_threshold),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "negative_iou_threshold must equal positive_iou_threshold so "
                "every IoU below the deployed acc50 boundary is a negative"
            )
        if float(listwise_temperature) <= 0.0 or float(fpr_temperature) <= 0.0:
            raise ValueError("loss temperatures must be positive")
        if (
            float(rank_fix_margin) < 0.0
            or float(rank_preserve_margin) < 0.0
            or float(rank_residual_weight) < 0.0
        ):
            raise ValueError("rank margins and residual weight must be non-negative")
        if float(paired_margin_weight) < 0.0:
            raise ValueError("paired_margin_weight must be non-negative")
        if int(queue_size) < 0 or int(queue_min_count) < 0:
            raise ValueError("queue sizes must be non-negative")
        if int(queue_min_count) > int(queue_size):
            raise ValueError("queue_min_count cannot exceed queue_size")
        self.train_mode = str(train_mode).strip()
        self.train_mode_code = stage_b_gdino_adapter_train_mode_code(self.train_mode)
        self.enable_rank = self.train_mode in {"rank_only", "joint"}
        self.enable_confidence = self.train_mode in {"confidence_only", "joint"}
        self.confidence_objective = (
            str(confidence_objective).strip() if self.enable_confidence else ""
        )
        self.confidence_objective_code = (
            stage_b_gdino_confidence_objective_code(self.confidence_objective)
            if self.enable_confidence
            else 0
        )
        self.tn_scope = str(tn_scope).strip() if self.enable_confidence else ""
        self.expected_scope_code = (
            stage_b_gdino_tn_scope_code(self.tn_scope)
            if self.enable_confidence
            else 0
        )
        self.positive_iou_threshold = float(positive_iou_threshold)
        self.negative_iou_threshold = float(negative_iou_threshold)
        self.listwise_temperature = float(listwise_temperature)
        self.rank_fix_margin = float(rank_fix_margin)
        self.rank_preserve_margin = float(rank_preserve_margin)
        self.rank_residual_weight = float(rank_residual_weight)
        self.fpr_temperature = float(fpr_temperature)
        self.fpr_margin = float(fpr_margin)
        self.paired_margin_weight = float(paired_margin_weight)
        self.paired_margin = float(paired_margin)
        trust_enabled = self.confidence_objective in {
            "detached_recent_q05_trust",
            "detached_recent_q05_total_trust",
            "detached_recent_q05_proposal_covered",
            "detached_recent_q05_scope_labeled",
        }
        if trust_enabled and float(positive_trust_margin) < 0.0:
            raise ValueError("positive_trust_margin must be non-negative")
        if trust_enabled and float(positive_trust_weight) < 0.0:
            raise ValueError("positive_trust_weight must be non-negative")
        self.positive_trust_margin = (
            float(positive_trust_margin) if trust_enabled else 0.0
        )
        self.positive_trust_weight = (
            float(positive_trust_weight) if trust_enabled else 0.0
        )
        self.queue_size = int(queue_size)
        self.queue_min_count = int(queue_min_count)
        if trust_enabled and (self.queue_size <= 0 or self.queue_min_count <= 0):
            raise ValueError(
                "detached q05 trust objectives require a positive queue size and warmup"
            )
        self.weight_dict = {}
        if self.enable_rank:
            self.weight_dict["loss_stage_b_gdino_rank"] = float(rank_weight)
        if self.enable_confidence:
            self.weight_dict["loss_stage_b_gdino_confidence"] = float(
                confidence_weight
            )
        self.register_buffer(
            "criterion_train_mode_code",
            torch.as_tensor(self.train_mode_code, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "criterion_scope_code",
            torch.as_tensor(self.expected_scope_code, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "criterion_confidence_objective_code",
            torch.as_tensor(self.confidence_objective_code, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "criterion_positive_trust_margin",
            torch.as_tensor(self.positive_trust_margin, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "criterion_positive_trust_weight",
            torch.as_tensor(self.positive_trust_weight, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "criterion_queue_size",
            torch.as_tensor(self.queue_size, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "criterion_queue_min_count",
            torch.as_tensor(self.queue_min_count, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "fpr_positive_queue",
            torch.zeros(self.queue_size, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "fpr_negative_queue",
            torch.zeros(self.queue_size, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "fpr_queue_count", torch.zeros((), dtype=torch.int64), persistent=True
        )
        self.register_buffer(
            "fpr_queue_ptr", torch.zeros((), dtype=torch.int64), persistent=True
        )
        self._pending_queue_payload: Optional[Tensor] = None
        self._deferred_queue_payloads: list[Tensor] = []

    def load_state_dict(self, state_dict, strict: bool = True):
        saved_mode = state_dict.get("criterion_train_mode_code", None)
        if not torch.is_tensor(saved_mode) or saved_mode.numel() != 1:
            raise RuntimeError(
                "Stage-B GDINO adapter criterion checkpoint is missing its train-mode code"
            )
        if int(saved_mode.reshape(-1)[0].item()) != self.train_mode_code:
            raise RuntimeError(
                "Stage-B GDINO adapter criterion train-mode mismatch: checkpoint="
                f"{int(saved_mode.reshape(-1)[0].item())}, configured="
                f"{self.train_mode_code} ({self.train_mode})"
            )
        saved_scope = state_dict.get("criterion_scope_code", None)
        if not torch.is_tensor(saved_scope) or saved_scope.numel() != 1:
            raise RuntimeError(
                "Stage-B GDINO adapter criterion checkpoint is missing its scope code"
            )
        if int(saved_scope.reshape(-1)[0].item()) != self.expected_scope_code:
            raise RuntimeError(
                "Stage-B GDINO adapter criterion scope mismatch: checkpoint="
                f"{int(saved_scope.reshape(-1)[0].item())}, configured="
                f"{self.expected_scope_code} ({self.tn_scope})"
            )
        saved_objective = state_dict.get(
            "criterion_confidence_objective_code", None
        )
        if not torch.is_tensor(saved_objective) or saved_objective.numel() != 1:
            raise RuntimeError(
                "Stage-B GDINO adapter criterion checkpoint is missing its "
                "confidence-objective code"
            )
        if int(saved_objective.reshape(-1)[0].item()) != self.confidence_objective_code:
            raise RuntimeError(
                "Stage-B GDINO adapter criterion confidence-objective mismatch: "
                f"checkpoint={int(saved_objective.reshape(-1)[0].item())}, "
                f"configured={self.confidence_objective_code} "
                f"({self.confidence_objective or 'disabled'})"
            )
        for key, expected, label in (
            (
                "criterion_positive_trust_margin",
                self.criterion_positive_trust_margin,
                "positive-trust margin",
            ),
            (
                "criterion_positive_trust_weight",
                self.criterion_positive_trust_weight,
                "positive-trust weight",
            ),
            ("criterion_queue_size", self.criterion_queue_size, "queue size"),
            (
                "criterion_queue_min_count",
                self.criterion_queue_min_count,
                "queue minimum count",
            ),
        ):
            saved = state_dict.get(key, None)
            if not torch.is_tensor(saved) or saved.numel() != 1:
                raise RuntimeError(
                    f"Stage-B GDINO adapter criterion checkpoint is missing its {label}"
                )
            saved_value = saved.detach().to(
                device=expected.device, dtype=expected.dtype
            ).reshape_as(expected)
            if not torch.equal(saved_value, expected):
                raise RuntimeError(
                    f"Stage-B GDINO adapter criterion {label} mismatch: "
                    f"checkpoint={saved.reshape(-1)[0].item()}, "
                    f"configured={expected.reshape(-1)[0].item()}"
                )
        return super().load_state_dict(state_dict, strict=strict)

    def _queue_values(self, queue: Tensor) -> Tensor:
        count = min(int(self.fpr_queue_count.item()), self.queue_size)
        if count <= 0:
            return queue[:0]
        if count < self.queue_size:
            return queue[:count]
        return queue

    @torch.no_grad()
    def _gather_pending(self, payload: Tensor) -> Tensor:
        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            return payload
        world_size = dist.get_world_size()
        local_count = torch.as_tensor(
            [int(payload.shape[0])], dtype=torch.int64, device=payload.device
        )
        counts = [torch.zeros_like(local_count) for _ in range(world_size)]
        dist.all_gather(counts, local_count)
        counts_int = [int(value.item()) for value in counts]
        max_count = max(counts_int, default=0)
        padded = payload.new_full((max_count, 2), torch.nan)
        if payload.numel():
            padded[: int(payload.shape[0])] = payload
        gathered = [torch.empty_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered, padded)
        return torch.cat(
            [value[:count] for value, count in zip(gathered, counts_int)], dim=0
        )

    @torch.no_grad()
    def defer_tail_queue_payload(self) -> None:
        """Hold one micro-batch payload until its accumulated optimizer step."""
        payload = self._pending_queue_payload
        self._pending_queue_payload = None
        if payload is not None:
            self._deferred_queue_payloads.append(payload)

    @torch.no_grad()
    def commit_tail_queue(self, step_succeeded: bool) -> None:
        payloads = list(self._deferred_queue_payloads)
        if self._pending_queue_payload is not None:
            payloads.append(self._pending_queue_payload)
        payload = (
            payloads[0]
            if len(payloads) == 1
            else (torch.cat(payloads, dim=0) if payloads else None)
        )
        self._deferred_queue_payloads.clear()
        self._pending_queue_payload = None
        if not bool(step_succeeded) or payload is None or self.queue_size <= 0:
            return
        payload = self._gather_pending(payload.detach().float())
        if payload.numel() == 0:
            return
        for row in payload:
            ptr = int(self.fpr_queue_ptr.item())
            self.fpr_positive_queue[ptr] = row[0]
            self.fpr_negative_queue[ptr] = row[1]
            self.fpr_queue_ptr.fill_((ptr + 1) % self.queue_size)
            self.fpr_queue_count.fill_(
                min(self.queue_size, int(self.fpr_queue_count.item()) + 1)
            )

    def _validate_scope(
        self,
        outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
        batch_size: int,
    ) -> None:
        scope_codes = outputs.get("stage_b_gdino_tn_scope_code", None)
        if not torch.is_tensor(scope_codes):
            raise RuntimeError("adapter training requires explicit TN scope codes")
        scope_codes = scope_codes.detach().reshape(-1)
        if scope_codes.numel() != batch_size or not bool(
            (scope_codes == self.expected_scope_code).all().item()
        ):
            raise RuntimeError(
                "adapter batch TN scope does not match configured scope "
                f"{self.tn_scope!r}"
            )
        for index, target in enumerate(targets):
            if not _strict_scalar_false(target, "proposalset_proxy_verified"):
                raise RuntimeError(
                    "proposal-set proxy/malformed proxy flag is forbidden for "
                    f"GDINO adapter sample {index}"
                )
            if self.tn_scope == "image_global_topk_verified":
                verified = _strict_scalar_bool(target, "global_tn_verified")
            elif self.tn_scope == "benchmark_dataft_alltn":
                verified = _strict_scalar_bool(target, "benchmark_dataft_alltn")
            elif self.tn_scope == "unverified_all_negative":
                # D1 is deliberately unverified.  The raw-target path already
                # binds table/audit/scope; never upgrade it by requiring or
                # claiming benchmark/all-TN verification here.
                verified = _strict_scalar_false(target, "global_tn_verified")
            else:
                verified = (
                    _strict_scalar_false(target, "global_tn_verified")
                    and _strict_scalar_false(target, "benchmark_dataft_alltn")
                )
            if not verified:
                raise RuntimeError(
                    f"sample {index} lacks exact boolean verification for "
                    f"{self.tn_scope}: tn_scope={target.get('tn_scope')!r}, "
                    f"table_b_id={target.get('table_b_id')!r}, "
                    "audit_sha_present="
                    f"{isinstance(target.get('table_b_audit_sha256'), str)}, "
                    f"global={target.get('global_tn_verified')!r}, "
                    f"benchmark={target.get('benchmark_dataft_alltn')!r}"
                )

    def forward(
        self,
        outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
    ) -> dict[str, Tensor]:
        result: dict[str, Tensor] = {
            "stage_b_gdino_train_mode_code": (
                self.criterion_train_mode_code.detach().float()
            )
        }
        batch_size: Optional[int] = None

        if self.enable_rank:
            rank_score = outputs.get("stage_b_gdino_rank_score", None)
            base_score = outputs.get("stage_b_gdino_base_score", None)
            rank_residual = outputs.get("stage_b_gdino_rank_residual", None)
            candidate_boxes = outputs.get("pred_boxes", None)
            if not all(
                torch.is_tensor(value)
                for value in (
                    rank_score,
                    base_score,
                    rank_residual,
                    candidate_boxes,
                )
            ):
                raise KeyError(
                    "rank phase requires rank/base/residual score and box outputs"
                )
            if rank_score.dim() != 2:
                raise ValueError("stage_b_gdino_rank_score must have shape (B,Q)")
            batch_size = int(rank_score.shape[0])
            if len(targets) != batch_size:
                raise ValueError("targets must align with adapter rank outputs")
            candidate_iou = _candidate_max_iou(candidate_boxes, targets)
            rank_output = baseline_preserving_top1_rank_loss(
                rank_score,
                base_score,
                rank_residual,
                candidate_iou,
                iou_threshold=self.positive_iou_threshold,
                fix_margin=self.rank_fix_margin,
                preserve_margin=self.rank_preserve_margin,
                temperature=self.listwise_temperature,
                residual_weight=self.rank_residual_weight,
            )
            result.update(
                {
                    "loss_stage_b_gdino_rank": rank_output.loss,
                    "stage_b_gdino_rank_margin_loss": rank_output.margin_loss.detach(),
                    "stage_b_gdino_rank_fix_loss": rank_output.fix_loss.detach(),
                    "stage_b_gdino_rank_preserve_loss": (
                        rank_output.preserve_loss.detach()
                    ),
                    "stage_b_gdino_rank_residual_l2": rank_output.residual_loss.detach(),
                    "stage_b_gdino_valid_rank_rows": rank_output.valid_rows,
                    "stage_b_gdino_rank_fix_rows": rank_output.fix_rows,
                    "stage_b_gdino_rank_preserve_rows": rank_output.preserve_rows,
                    "stage_b_gdino_rank_rows_no_positive": rank_output.rows_no_positive,
                    "stage_b_gdino_rank_base_correct": rank_output.base_correct,
                    "stage_b_gdino_rank_adapted_correct": rank_output.adapted_correct,
                    "stage_b_gdino_rank_wrong_fixed": rank_output.wrong_fixed,
                    "stage_b_gdino_rank_correct_regressed": (
                        rank_output.correct_regressed
                    ),
                }
            )

        if self.enable_confidence:
            negative_outputs = outputs.get("stage_b_gdino_tn_outputs", None)
            if not isinstance(negative_outputs, Mapping):
                raise KeyError(
                    "confidence phase requires stage_b_gdino_tn_outputs"
                )
            positive_confidence = outputs.get(
                "stage_b_gdino_confidence_score", None
            )
            negative_confidence = negative_outputs.get(
                "stage_b_gdino_confidence_score", None
            )
            if not torch.is_tensor(positive_confidence) or not torch.is_tensor(
                negative_confidence
            ):
                raise KeyError("confidence phase is missing positive/TN scores")
            if positive_confidence.dim() != 2 or (
                negative_confidence.shape != positive_confidence.shape
            ):
                raise ValueError(
                    "positive/TN confidence scores must share shape (B,Q)"
                )
            confidence_batch_size = int(positive_confidence.shape[0])
            if batch_size is not None and confidence_batch_size != batch_size:
                raise ValueError("rank and confidence batches do not align")
            batch_size = confidence_batch_size
            if len(targets) != batch_size:
                raise ValueError("targets must align with confidence outputs")
            self._validate_scope(outputs, targets, batch_size)

            positive_history = None
            if (
                self.queue_size > 0
                and int(self.fpr_queue_count.item()) >= self.queue_min_count
            ):
                positive_history = self._queue_values(self.fpr_positive_queue)
            positive_gate = None
            negative_gate = None
            if self.confidence_objective in {
                "detached_recent_q05_trust",
                "detached_recent_q05_total_trust",
                "detached_recent_q05_proposal_covered",
                "detached_recent_q05_scope_labeled",
            }:
                positive_gate = outputs.get(
                    "stage_b_gdino_confidence_gate", None
                )
                negative_gate = negative_outputs.get(
                    "stage_b_gdino_confidence_gate", None
                )
                if not torch.is_tensor(positive_gate) or not torch.is_tensor(
                    negative_gate
                ):
                    raise KeyError(
                        "detached_recent_q05_trust requires positive/TN "
                        "stage_b_gdino_confidence_gate outputs"
                    )
                fpr_output = detached_recent_q05_trust_surrogate(
                    positive_confidence,
                    negative_confidence,
                    positive_gate,
                    negative_gate,
                    positive_history=positive_history,
                    temperature=self.fpr_temperature,
                    margin=self.fpr_margin,
                    target_tpr=0.95,
                    positive_trust_margin=self.positive_trust_margin,
                    positive_trust_weight=self.positive_trust_weight,
                    paired_margin_weight=self.paired_margin_weight,
                    paired_margin=self.paired_margin,
                    positive_score_trust=(
                        self.confidence_objective in {
                            "detached_recent_q05_total_trust",
                            "detached_recent_q05_proposal_covered",
                            "detached_recent_q05_scope_labeled",
                        }
                    ),
                )
            else:
                fpr_output = fpr95_global_max_surrogate(
                    positive_confidence,
                    negative_confidence,
                    positive_history=positive_history,
                    temperature=self.fpr_temperature,
                    margin=self.fpr_margin,
                    target_tpr=0.95,
                    paired_margin_weight=self.paired_margin_weight,
                    paired_margin=self.paired_margin,
                )
            self._pending_queue_payload = torch.stack(
                (
                    fpr_output.local_positive_global_score.detach(),
                    fpr_output.local_negative_global_score.detach(),
                ),
                dim=1,
            )
            result.update(
                {
                    "loss_stage_b_gdino_confidence": fpr_output.loss,
                    "stage_b_gdino_confidence_negative_loss": (
                        fpr_output.negative_loss.detach()
                    ),
                    "stage_b_gdino_confidence_trust_loss": (
                        fpr_output.positive_trust_loss.detach()
                        if isinstance(fpr_output, DetachedRecentQ05TrustOutput)
                        else fpr_output.negative_loss.detach().new_zeros(())
                    ),
                    "stage_b_gdino_confidence_positive_score_trust_loss": (
                        fpr_output.positive_score_trust_loss.detach()
                        if isinstance(fpr_output, DetachedRecentQ05TrustOutput)
                        else fpr_output.negative_loss.detach().new_zeros(())
                    ),
                    "stage_b_gdino_positive_global_mean": (
                        fpr_output.positive_global_score.mean().detach()
                    ),
                    "stage_b_gdino_negative_global_mean": (
                        fpr_output.negative_global_score.mean().detach()
                    ),
                    "stage_b_gdino_positive_threshold": (
                        fpr_output.positive_threshold.detach()
                    ),
                    "stage_b_gdino_bank_q05": (
                        fpr_output.positive_threshold.detach()
                    ),
                    "stage_b_gdino_current_positive_threshold": (
                        fpr_output.current_positive_threshold.detach()
                    ),
                    "stage_b_gdino_current_q05": (
                        fpr_output.current_positive_threshold.detach()
                    ),
                    "stage_b_gdino_threshold_drift": (
                        fpr_output.threshold_drift.detach()
                    ),
                    "stage_b_gdino_q05_drift": (
                        fpr_output.threshold_drift.detach()
                    ),
                    "stage_b_gdino_exact_batch_tpr": fpr_output.exact_tpr.detach(),
                    "stage_b_gdino_exact_batch_fpr": fpr_output.exact_fpr.detach(),
                    "stage_b_gdino_current_q05_tpr": (
                        fpr_output.current_exact_tpr.detach()
                    ),
                    "stage_b_gdino_current_q05_fpr": (
                        fpr_output.current_exact_fpr.detach()
                    ),
                    "stage_b_gdino_paired_separation": (
                        fpr_output.paired_separation_loss.detach()
                    ),
                    "stage_b_gdino_queue_count": (
                        self.fpr_queue_count.detach().float()
                    ),
                    "stage_b_gdino_scope_code": (
                        self.criterion_scope_code.detach().float()
                    ),
                    "stage_b_gdino_confidence_objective_code": (
                        self.criterion_confidence_objective_code.detach().float()
                    ),
                }
            )
            if isinstance(fpr_output, DetachedRecentQ05TrustOutput):
                result.update(
                    {
                        "stage_b_gdino_positive_gate_mean": (
                            fpr_output.positive_global_gate.mean().detach()
                        ),
                        "stage_b_gdino_negative_gate_mean": (
                            fpr_output.negative_global_gate.mean().detach()
                        ),
                        "stage_b_gdino_positive_trust_violation_rate": (
                            fpr_output.positive_trust_violation_rate.detach()
                        ),
                        "stage_b_gdino_positive_score_trust_violation_rate": (
                            fpr_output.positive_score_trust_violation_rate.detach()
                        ),
                    }
                )
                quantiles = torch.as_tensor(
                    [0.01, 0.05, 0.5, 0.95, 0.99],
                    device=fpr_output.positive_global_gate.device,
                    dtype=torch.float32,
                )
                with torch.no_grad():
                    positive_gate_quantiles = torch.quantile(
                        fpr_output.positive_global_gate.detach().float(), quantiles
                    )
                    negative_gate_quantiles = torch.quantile(
                        fpr_output.negative_global_gate.detach().float(), quantiles
                    )
                for suffix, positive_value, negative_value in zip(
                    ("q01", "q05", "q50", "q95", "q99"),
                    positive_gate_quantiles,
                    negative_gate_quantiles,
                ):
                    result[f"stage_b_gdino_positive_gate_{suffix}"] = positive_value
                    result[f"stage_b_gdino_negative_gate_{suffix}"] = negative_value
        else:
            self._pending_queue_payload = None

        return result


__all__ = [
    "BaselinePreservingRankOutput",
    "GDINO_ADAPTER_TRAIN_MODE_CODES",
    "GDINO_CONFIDENCE_OBJECTIVE_CODES",
    "GDINO_TN_SCOPE_CODES",
    "DetachedRecentQ05TrustOutput",
    "FPR95SurrogateOutput",
    "StageBGDINOScoreAdapter",
    "StageBGDINOScoreAdapterCriterion",
    "aggregate_gdino_full_expression_score",
    "b58_top1_anchored_rank_tail_score",
    "baseline_preserving_top1_rank_loss",
    "detached_recent_q05_trust_surrogate",
    "distributed_gather_1d_with_local_grad",
    "exact_tpr_operating_threshold",
    "fpr95_global_max_surrogate",
    "image_expression_global_max",
    "multi_positive_listwise_rank_loss",
    "stage_b_gdino_adapter_train_mode_code",
    "stage_b_gdino_confidence_objective_code",
    "stage_b_gdino_tn_scope_code",
    "validate_stage_b_gdino_score_adapter_checkpoint",
]
