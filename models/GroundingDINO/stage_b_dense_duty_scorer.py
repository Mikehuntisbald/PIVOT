"""Dense, responsibility-separated Stage-B scoring over frozen candidates.

The patch branch owns candidate admission and category evidence.  One private
full-text ``rank_tower`` orders admitted candidates.  A lightweight confidence
adapter consumes stop-gradient rank logits/query/text features and learns a
zero-initialized token residual plus an absolute sample confidence.  Confidence
training therefore cannot change ranking or the frozen proposal path.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .stage_b_data_driven_score import data_driven_category_gate_mask
from .transformer import TransformerDecoder
from .utils import ContrastiveEmbed


RawContextProvider = Callable[[List[str], Tensor], Dict[str, Any]]

DENSE_DUTY_PHASES = ("rank", "confidence", "eval")
DENSE_DUTY_CONTRACT_VERSION = 3
CONFIDENCE_TOKEN_CONTRACT = "detached_rank_token_minus_zero_init_residual_v1"
CONFIDENCE_HEAD_GRADIENT_CONTRACT_SHARED = (
    "shared_token_veto_global_absolute_v1"
)
CONFIDENCE_HEAD_GRADIENT_CONTRACT_SPLIT = (
    "split_token_veto_global_absolute_v2"
)
CONFIDENCE_HEAD_GRADIENT_CONTRACT_SPLIT_JOINT_CLIP = (
    "split_token_veto_global_absolute_joint_clip_v3"
)
CONFIDENCE_HEAD_GRADIENT_CONTRACT_GLOBAL_TRUST_VETO = (
    "split_token_veto_global_trust_veto_v4"
)
CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYED_ROUTER = (
    "split_token_veto_deployed_router_global_absolute_v5"
)
CONFIDENCE_HEAD_GRADIENT_CONTRACT_CANDIDATE_SAMPLE = (
    "split_token_veto_candidate_absolute_sample_calibrator_v6"
)
CONFIDENCE_HEAD_GRADIENT_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE = (
    "split_token_veto_fulltext_global_absolute_v7"
)
CONFIDENCE_HEAD_GRADIENT_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE = (
    "split_token_veto_local_candidate_global_absolute_v8"
)
CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE = (
    "split_token_veto_deployment_owned_global_absolute_v9"
)
CONFIDENCE_HEAD_GRADIENT_CONTRACT_SPLIT_SEMANTICS = (
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_SPLIT,
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_SPLIT_JOINT_CLIP,
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_GLOBAL_TRUST_VETO,
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYED_ROUTER,
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_CANDIDATE_SAMPLE,
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE,
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE,
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE,
)
CONFIDENCE_HEAD_GRADIENT_CONTRACTS = (
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_SHARED,
    *CONFIDENCE_HEAD_GRADIENT_CONTRACT_SPLIT_SEMANTICS,
)
CONFIDENCE_POOL_FEATURE_CONTRACT = "patch_statistics_only_v1"
CONFIDENCE_POOL_FEATURE_CONTRACT_SIGNED_RANK_QUERY = (
    "detached_rank_query_plus_patch_statistics_signed_residual_v2"
)
CONFIDENCE_POOL_FEATURE_CONTRACT_TOKEN_CONDITIONED = (
    "detached_rank_query_token_context_plus_patch_statistics_monotone_v3"
)
CONFIDENCE_POOL_FEATURE_CONTRACT_CROSS_ATTENTION = (
    "detached_rank_query_modifier_cross_attention_plus_patch_statistics_absolute_v4"
)
CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ABSOLUTE = (
    "detached_query_modifier_cross_attention_candidate_absolute_logits_v5"
)
CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_CALIBRATED = (
    "detached_candidate_absolute_patch_invariant_monotone_veto_logits_v6"
)
CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_NORMALIZED = (
    "detached_candidate_absolute_normalized_patch_amplified_veto_logits_v7"
)
CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC = (
    "detached_candidate_absolute_raw_patch_asymmetric_veto_logits_v8"
)
CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_SET_ATTENTION = (
    "detached_candidate_set_attention_absolute_asymmetric_veto_logits_v9"
)
CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE = (
    "detached_rank_full_expression_candidate_residual_global_pool_v10"
)
CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_REFERENCE = (
    "detached_rank_full_expression_candidate_residual_global_pool_exact_rank_max_reference_v11"
)
CONFIDENCE_POOL_FEATURE_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE = (
    "detached_rank_full_expression_local_candidate_frozen_rank_global_pool_v12"
)
CONFIDENCE_POOL_FEATURE_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE = (
    "detached_rank_full_expression_deployment_owned_global_pool_v13"
)
CONFIDENCE_POOL_FEATURE_CONTRACTS = (
    CONFIDENCE_POOL_FEATURE_CONTRACT,
    CONFIDENCE_POOL_FEATURE_CONTRACT_SIGNED_RANK_QUERY,
    CONFIDENCE_POOL_FEATURE_CONTRACT_TOKEN_CONDITIONED,
    CONFIDENCE_POOL_FEATURE_CONTRACT_CROSS_ATTENTION,
    CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ABSOLUTE,
    CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_CALIBRATED,
    CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_NORMALIZED,
    CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC,
    CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_SET_ATTENTION,
    CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE,
    CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_REFERENCE,
    CONFIDENCE_POOL_FEATURE_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE,
    CONFIDENCE_POOL_FEATURE_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE,
)
CONFIDENCE_RANK_EVIDENCE_CONTRACT_OFF = "off_v1"
CONFIDENCE_RANK_EVIDENCE_CONTRACT_SCALE = "zero_init_rank_logit_scale_v1"
CONFIDENCE_RANK_EVIDENCE_CONTRACT_AFFINE = "zero_init_rank_logit_affine_v2"
CONFIDENCE_RANK_EVIDENCE_CONTRACT_GATE_MARGIN = (
    "zero_init_rank_logit_gate_margin_scale_v3"
)
CONFIDENCE_RANK_EVIDENCE_CONTRACT_CARRIER_TOKEN_SLOPE = (
    "zero_init_carrier_token_rank_slope_v4"
)
CONFIDENCE_RANK_EVIDENCE_CONTRACT_CARRIER_TOKEN_AFFINE = (
    "zero_init_carrier_token_rank_affine_v5"
)
CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH = (
    "zero_init_carrier_token_rank_affine_sparse_rank_channel_v6"
)
CONFIDENCE_GATE_GRADIENT_CONTRACT_HARD_DETACHED = "hard_detached_v1"
CONFIDENCE_GATE_GRADIENT_CONTRACT_HARD_FORWARD_SOFT_BACKWARD = (
    "hard_forward_soft_backward_v2"
)
CONFIDENCE_GATE_GRADIENT_CONTRACT_CONTINUOUS = "continuous_sigmoid_v3"
CONFIDENCE_GATE_GRADIENT_CONTRACT_CONTINUOUS_MONOTONE_DEPTH = (
    "continuous_sigmoid_monotone_depth_v4"
)
CONFIDENCE_GATE_GRADIENT_CONTRACT_COMPLEMENTARY_TRUST_VETO = (
    "continuous_sigmoid_complementary_trust_veto_v5"
)
CONFIDENCE_GATE_GRADIENT_CONTRACT_UNGATED_MONOTONE_DEPTH = (
    "token_conditioned_ungated_monotone_depth_v6"
)
CONFIDENCE_GATE_GRADIENT_CONTRACT_FLOOR_GATED_MONOTONE_DEPTH = (
    "token_conditioned_floor_gated_monotone_depth_v7"
)
CONFIDENCE_GATE_GRADIENT_CONTRACT_INDEPENDENT_ABSOLUTE_LOGIT = (
    "token_conditioned_independent_absolute_logit_v8"
)
CONFIDENCE_GATE_GRADIENT_CONTRACT_CROSS_ATTENTION_INDEPENDENT_ABSOLUTE_LOGIT = (
    "cross_attention_independent_absolute_logit_v9"
)
CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ABSOLUTE_LOGIT = (
    "candidate_cross_attention_independent_absolute_logit_v10"
)
CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_CALIBRATED_LOGIT = (
    "candidate_patch_invariant_monotone_veto_absolute_logit_v11"
)
CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_NORMALIZED_LOGIT = (
    "candidate_normalized_patch_amplified_monotone_veto_absolute_logit_v12"
)
CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT = (
    "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13"
)
CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_SET_ATTENTION_LOGIT = (
    "candidate_set_attention_asymmetric_monotone_veto_absolute_logit_v14"
)
CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_DEPLOYED_ROUTING_ST = (
    "candidate_raw_patch_asymmetric_deployed_routing_st_v15"
)
CONFIDENCE_MONOTONE_VETO_GATE_FLOOR = 0.25
CONFIDENCE_GATE_GRADIENT_CONTRACTS = (
    CONFIDENCE_GATE_GRADIENT_CONTRACT_HARD_DETACHED,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_HARD_FORWARD_SOFT_BACKWARD,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CONTINUOUS,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CONTINUOUS_MONOTONE_DEPTH,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_COMPLEMENTARY_TRUST_VETO,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_UNGATED_MONOTONE_DEPTH,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_FLOOR_GATED_MONOTONE_DEPTH,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_INDEPENDENT_ABSOLUTE_LOGIT,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CROSS_ATTENTION_INDEPENDENT_ABSOLUTE_LOGIT,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ABSOLUTE_LOGIT,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_CALIBRATED_LOGIT,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_NORMALIZED_LOGIT,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_SET_ATTENTION_LOGIT,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_DEPLOYED_ROUTING_ST,
)
CONFIDENCE_PHRASE_AGGREGATION_LEGACY = "legacy_prob_mean_add_v1"
CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO = (
    "trace_activated_word_veto_product_v1"
)
CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_PENALTY = (
    "trace_activated_word_veto_penalty_v2"
)
CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_ABSOLUTE_CAP = (
    "trace_activated_word_veto_absolute_cap_v4"
)
CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP = (
    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
)
CONFIDENCE_PHRASE_AGGREGATION_WORD_VETOES = (
    CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO,
    CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_PENALTY,
    CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_ABSOLUTE_CAP,
    CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
)
CONFIDENCE_PHRASE_AGGREGATIONS = (
    CONFIDENCE_PHRASE_AGGREGATION_LEGACY,
    *CONFIDENCE_PHRASE_AGGREGATION_WORD_VETOES,
)


class _ExactForwardSurrogateBackward(torch.autograd.Function):
    """Return the exact first input while differentiating through the second."""

    @staticmethod
    def forward(ctx, exact: Tensor, surrogate: Tensor) -> Tensor:
        del ctx, surrogate
        return exact

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[None, Tensor]:
        del ctx
        return None, grad_output


def normalize_dense_duty_phase(value: Any) -> str:
    phase = str(value or "eval").strip().lower()
    if phase not in DENSE_DUTY_PHASES:
        raise ValueError(
            f"dense-duty phase must be one of {DENSE_DUTY_PHASES}, got {phase!r}"
        )
    return phase


def _module_parameters(module: Any) -> tuple[nn.Parameter, ...]:
    if isinstance(module, nn.Module):
        return tuple(module.parameters())
    return ()


def _set_module_trainable(module: Any, enabled: bool) -> None:
    if not isinstance(module, nn.Module):
        return
    module.train(bool(enabled))
    for parameter in module.parameters():
        parameter.requires_grad_(bool(enabled))


def _masked_score_statistics(
    score: Tensor,
    candidate_mask: Tensor,
    *,
    topk: int,
) -> tuple[Tensor, Tensor]:
    """Return a differentiable pooled score and six stable row statistics."""
    if score.dim() != 2 or tuple(candidate_mask.shape) != tuple(score.shape):
        raise ValueError("score and candidate_mask must both have shape (B,N)")
    if candidate_mask.dtype != torch.bool:
        raise TypeError("candidate_mask must be boolean")
    valid_count = candidate_mask.sum(dim=1)
    if bool((valid_count <= 0).any().item()):
        raise ValueError("every confidence row requires an eligible candidate")

    score_float = score.float()
    masked_score = score_float.masked_fill(~candidate_mask, -torch.inf)
    weights = torch.softmax(masked_score, dim=1).masked_fill(~candidate_mask, 0.0)
    pooled_score = (weights * score_float.masked_fill(~candidate_mask, 0.0)).sum(
        dim=1
    )
    count = valid_count.float()
    score_max = masked_score.max(dim=1).values
    score_mean = score_float.masked_fill(~candidate_mask, 0.0).sum(dim=1) / count
    centered = (score_float - score_mean[:, None]).masked_fill(~candidate_mask, 0.0)
    # This is the population standard deviation, written as a vector norm so
    # singleton and tied-score rows have a finite zero gradient at variance 0.
    score_std = torch.linalg.vector_norm(centered, ord=2, dim=1) / count.sqrt()

    top_count = min(max(1, int(topk)), int(score.shape[1]))
    top_values = torch.topk(masked_score, k=top_count, dim=1).values
    top_valid = torch.arange(top_count, device=score.device)[None] < torch.minimum(
        valid_count, valid_count.new_full(valid_count.shape, top_count)
    )[:, None]
    score_top_mean = top_values.masked_fill(~top_valid, 0.0).sum(dim=1) / top_valid.sum(
        dim=1
    ).clamp_min(1)
    if int(score.shape[1]) > 1:
        top_two = torch.topk(masked_score, k=2, dim=1).values
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
        entropy / count.log().clamp_min(1e-8),
        torch.zeros_like(entropy),
    )
    statistics = torch.stack(
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
    return pooled_score, statistics


def _word_normalized_softmin_probability(
    token_logits: Tensor,
    token_residuals: Tensor,
    token_mask: Tensor,
    word_group_ids: Tensor,
    *,
    temperature: float,
    gate_scale: float,
    gate_offset: float = 0.0,
    gate_gradient_contract: str = (
        CONFIDENCE_GATE_GRADIENT_CONTRACT_HARD_DETACHED
    ),
) -> tuple[Tensor, Tensor]:
    """Return a WordPiece-invariant veto and an exact hard mismatch gate."""
    if token_logits.dim() != 4 or token_residuals.shape != token_logits.shape:
        raise ValueError("token logits/residuals must have shape (L,M,N,T)")
    if tuple(token_mask.shape) != tuple(token_logits.shape[1::2]):
        raise ValueError("token_mask must have shape (M,T)")
    if tuple(word_group_ids.shape) != tuple(token_mask.shape):
        raise ValueError("word_group_ids must align with token_mask")
    if word_group_ids.dtype != torch.long:
        raise TypeError("word_group_ids must be int64")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("word softmin temperature must be finite and positive")
    if not math.isfinite(float(gate_scale)) or float(gate_scale) <= 0.0:
        raise ValueError("word veto gate_scale must be finite and positive")
    if not math.isfinite(float(gate_offset)) or float(gate_offset) < 0.0:
        raise ValueError("word veto gate_offset must be finite and non-negative")
    gate_gradient_contract = str(gate_gradient_contract).strip().lower()
    if gate_gradient_contract not in CONFIDENCE_GATE_GRADIENT_CONTRACTS:
        raise ValueError("confidence gate-gradient contract is invalid")

    mask = token_mask.to(device=token_logits.device, dtype=torch.bool)
    groups = word_group_ids.to(device=token_logits.device, dtype=torch.long)
    if bool((mask & groups.lt(0)).any().item()):
        raise ValueError("every confidence score token requires a lexical word group")
    member_mask = mask & groups.ge(0)
    max_group = int(groups.masked_fill(~member_mask, 0).max().item())
    membership = F.one_hot(
        groups.clamp_min(0), num_classes=max(1, max_group + 1)
    ).to(dtype=token_logits.dtype)
    membership = membership * member_mask[..., None].to(dtype=membership.dtype)
    member_count = membership.sum(dim=1)
    group_valid = member_count.gt(0)
    denominator = member_count.clamp_min(1.0)[None, :, None, :]

    word_probability = torch.einsum(
        "lmnt,mtg->lmng", token_logits.float().sigmoid(), membership.float()
    ) / denominator.float()
    word_residual = torch.einsum(
        "lmnt,mtg->lmng", token_residuals.float(), membership.float()
    ) / denominator.float()
    expanded_valid = group_valid[None, :, None, :]
    log_sum = torch.logsumexp(
        (-word_probability / float(temperature)).masked_fill(
            ~expanded_valid, -torch.inf
        ),
        dim=-1,
    )
    valid_count = group_valid.sum(dim=-1).clamp_min(1).float()
    veto_probability = -float(temperature) * (
        log_sum - valid_count.log()[None, :, None]
    )
    has_word = group_valid.any(dim=-1)[None, :, None]
    veto_probability = torch.where(
        has_word, veto_probability, torch.ones_like(veto_probability)
    ).clamp(0.0, 1.0)

    max_word_residual = word_residual.masked_fill(
        ~expanded_valid, -torch.inf
    ).max(dim=-1).values
    max_word_residual = torch.where(
        has_word, max_word_residual, torch.zeros_like(max_word_residual)
    )
    # The legacy contracts keep the exact hard forward gate. The continuous
    # contract is used by the post-v20 conditional-residual readout: it avoids
    # both a dead closed gate and the absolute-cap score plateau at positive
    # q05 while remaining directly anchored by the raw edit-token objective.
    if gate_gradient_contract in {
        CONFIDENCE_GATE_GRADIENT_CONTRACT_CONTINUOUS,
        CONFIDENCE_GATE_GRADIENT_CONTRACT_CONTINUOUS_MONOTONE_DEPTH,
        CONFIDENCE_GATE_GRADIENT_CONTRACT_COMPLEMENTARY_TRUST_VETO,
        CONFIDENCE_GATE_GRADIENT_CONTRACT_UNGATED_MONOTONE_DEPTH,
        CONFIDENCE_GATE_GRADIENT_CONTRACT_FLOOR_GATED_MONOTONE_DEPTH,
        CONFIDENCE_GATE_GRADIENT_CONTRACT_INDEPENDENT_ABSOLUTE_LOGIT,
        CONFIDENCE_GATE_GRADIENT_CONTRACT_CROSS_ATTENTION_INDEPENDENT_ABSOLUTE_LOGIT,
    }:
        mismatch_gate = torch.sigmoid(
            (max_word_residual - float(gate_offset)) / float(gate_scale)
        )
        return veto_probability, mismatch_gate

    hard_mismatch_gate = (
        (max_word_residual.detach() - float(gate_offset)) / float(gate_scale)
    ).clamp(min=0.0, max=1.0)
    if (
        gate_gradient_contract
        == CONFIDENCE_GATE_GRADIENT_CONTRACT_HARD_FORWARD_SOFT_BACKWARD
    ):
        soft_mismatch_gate = torch.sigmoid(
            (max_word_residual - float(gate_offset)) / float(gate_scale)
        )
        mismatch_gate = _ExactForwardSurrogateBackward.apply(
            hard_mismatch_gate, soft_mismatch_gate
        )
    elif (
        gate_gradient_contract
        == CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_DEPLOYED_ROUTING_ST
    ):
        # Preserve v13's exact clamp at inference while exposing a smooth
        # backward path for the explicit deployed winner/coverage objective.
        soft_mismatch_gate = torch.sigmoid(
            (max_word_residual - float(gate_offset)) / float(gate_scale)
        )
        mismatch_gate = _ExactForwardSurrogateBackward.apply(
            hard_mismatch_gate, soft_mismatch_gate
        )
    else:
        mismatch_gate = hard_mismatch_gate
    return veto_probability, mismatch_gate


def _frozen_reference_carrier_gate(
    reference_logits: Tensor,
    candidate_mask: Tensor,
    mismatch_gate: Tensor,
) -> tuple[Tensor, Tensor]:
    """Gather the detached mismatch gate of the frozen reference carrier."""
    carrier_index = _frozen_reference_carrier_index(
        reference_logits, candidate_mask
    )
    carrier_gate = _gather_frozen_reference_carrier_gate(
        mismatch_gate, carrier_index
    )
    return carrier_gate, carrier_index


def _frozen_reference_carrier_index(
    reference_logits: Tensor,
    candidate_mask: Tensor,
) -> Tensor:
    """Select the eligible frozen-reference carrier for each expression row."""
    if reference_logits.dim() != 2:
        raise ValueError("reference_logits must have shape (B,N)")
    if tuple(candidate_mask.shape) != tuple(reference_logits.shape):
        raise ValueError("candidate_mask must align with reference_logits")
    if candidate_mask.dtype != torch.bool:
        raise TypeError("candidate_mask must be boolean")
    if bool((candidate_mask.sum(dim=1) <= 0).any().item()):
        raise ValueError("every confidence row requires an eligible carrier")

    frozen_reference = reference_logits.detach().float()
    return frozen_reference.masked_fill(
        ~candidate_mask, -torch.inf
    ).argmax(dim=1)


def _gather_frozen_reference_carrier_gate(
    mismatch_gate: Tensor,
    carrier_index: Tensor,
    *,
    detach_gate: bool = True,
) -> Tensor:
    """Gather a carrier gate using an already selected frozen carrier."""
    if mismatch_gate.dim() != 2:
        raise ValueError("mismatch_gate must have shape (B,N)")
    if carrier_index.dim() != 1 or int(carrier_index.shape[0]) != int(
        mismatch_gate.shape[0]
    ):
        raise ValueError("carrier_index must have shape (B,)")
    if carrier_index.dtype != torch.long:
        raise TypeError("carrier_index must be int64")
    if bool(
        (
            carrier_index.lt(0)
            | carrier_index.ge(int(mismatch_gate.shape[1]))
        ).any().item()
    ):
        raise ValueError("carrier_index is outside the candidate dimension")
    carrier_gate = mismatch_gate.float().gather(
        dim=1, index=carrier_index[:, None]
    ).squeeze(1)
    return carrier_gate.detach() if detach_gate else carrier_gate


class AbsoluteConfidencePool(nn.Module):
    """Pool private dense confidence features into one absolute sample logit."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        pool_hidden_dim: int = 256,
        score_topk: int = 10,
        pool_temperature: float = 0.2,
        set_attention: bool = False,
        set_seed_count: int = 4,
    ) -> None:
        super().__init__()
        if int(hidden_dim) <= 0 or int(pool_hidden_dim) <= 0 or int(score_topk) <= 0:
            raise ValueError("confidence pooling dimensions must be positive")
        if not math.isfinite(float(pool_temperature)) or float(pool_temperature) <= 0:
            raise ValueError("pool_temperature must be finite and positive")
        self.hidden_dim = int(hidden_dim)
        self.score_topk = int(score_topk)
        self.pool_temperature = float(pool_temperature)
        self.set_attention = bool(set_attention)
        self.set_seed_count = int(set_seed_count)
        if self.set_attention:
            if self.set_seed_count <= 0 or self.hidden_dim % 4 != 0:
                raise ValueError("set-attention pooling requires positive seeds and 4 heads")
            self.set_seed = nn.Parameter(
                torch.empty(self.set_seed_count, self.hidden_dim)
            )
            self.set_candidate_norm = nn.LayerNorm(self.hidden_dim)
            self.set_logit_projection = nn.Linear(1, self.hidden_dim)
            self.set_attention_pool = nn.MultiheadAttention(
                self.hidden_dim, 4, batch_first=True
            )
            self.set_ffn = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
                nn.GELU(),
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            )
            self.set_output_projection = nn.Sequential(
                nn.LayerNorm(self.set_seed_count * self.hidden_dim),
                nn.Linear(self.set_seed_count * self.hidden_dim, self.hidden_dim),
                nn.GELU(),
            )
            nn.init.normal_(self.set_seed, std=0.02)
        else:
            self.register_parameter("set_seed", None)
            self.set_candidate_norm = None
            self.set_logit_projection = None
            self.set_attention_pool = None
            self.set_ffn = None
            self.set_output_projection = None
        self.residual = nn.Sequential(
            nn.Linear(self.hidden_dim + 6, int(pool_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(pool_hidden_dim), int(pool_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(pool_hidden_dim), 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        dense_feature: Tensor,
        dense_logit: Tensor,
        candidate_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if dense_feature.dim() != 3 or int(dense_feature.shape[-1]) != self.hidden_dim:
            raise ValueError("dense_feature must have shape (B,N,D)")
        if tuple(dense_logit.shape) != tuple(dense_feature.shape[:2]):
            raise ValueError("dense_logit must align with dense_feature")
        if tuple(candidate_mask.shape) != tuple(dense_logit.shape):
            raise ValueError("candidate_mask must align with dense_logit")

        mask = candidate_mask.to(device=dense_logit.device, dtype=torch.bool)
        masked_dense = dense_logit.float().masked_fill(~mask, -torch.inf)
        weights = torch.softmax(
            masked_dense / self.pool_temperature, dim=1
        ).masked_fill(~mask, 0.0)
        pooled_feature = torch.einsum(
            "bn,bnd->bd", weights.to(dtype=dense_feature.dtype), dense_feature
        ).float()
        if self.set_attention:
            modules = (
                self.set_candidate_norm,
                self.set_logit_projection,
                self.set_attention_pool,
                self.set_ffn,
                self.set_output_projection,
            )
            if self.set_seed is None or any(module is None for module in modules):
                raise RuntimeError("set-attention confidence pool is incomplete")
            finite_logit = dense_logit.float().masked_fill(~mask, 0.0).clamp(
                min=-20.0, max=20.0
            ) / 20.0
            candidate = self.set_candidate_norm(dense_feature)
            candidate = candidate + self.set_logit_projection(
                finite_logit[..., None].to(dtype=dense_feature.dtype)
            )
            seed = self.set_seed[None].expand(int(candidate.shape[0]), -1, -1)
            attended, _ = self.set_attention_pool(
                seed.to(dtype=candidate.dtype),
                candidate,
                candidate,
                key_padding_mask=~mask,
                need_weights=False,
            )
            attended = attended + self.set_ffn(attended)
            set_feature = self.set_output_projection(
                attended.flatten(start_dim=1)
            ).float()
            pooled_feature = (pooled_feature + set_feature) / math.sqrt(2.0)
        dense_base, dense_stats = _masked_score_statistics(
            dense_logit, mask, topk=self.score_topk
        )
        residual = self.residual(torch.cat((pooled_feature, dense_stats), dim=-1)).squeeze(-1)
        return dense_base + residual, residual


class TokenAwareConfidenceAdapter(nn.Module):
    """Zero-initialized token residual and category confidence over detached rank state."""

    patch_stat_dim = 9
    deployed_router_feature_dim = 10

    def __init__(
        self,
        hidden_dim: int,
        *,
        adapter_dim: int = 64,
        max_text_len: int = 256,
        patch_hidden_dim: int = 64,
        score_topk: int = 10,
        patch_score_clip: float = 5.0,
        phrase_aggregation: str = CONFIDENCE_PHRASE_AGGREGATION_LEGACY,
        word_softmin_temperature: float = 0.1,
        veto_gate_scale: float = 1.0,
        veto_gate_offset: float = 0.0,
        veto_coverage_offset: float = 0.1,
        veto_coverage_ramp: float = 0.8,
        veto_cap_temperature: float = 0.1,
        veto_cap_initial_ceiling: float = -0.1,
        rank_evidence_contract: str = CONFIDENCE_RANK_EVIDENCE_CONTRACT_OFF,
        pool_feature_contract: str = CONFIDENCE_POOL_FEATURE_CONTRACT,
        residual_parameterization_gain: float = 1.0,
        gate_gradient_contract: str = (
            CONFIDENCE_GATE_GRADIENT_CONTRACT_HARD_DETACHED
        ),
        head_gradient_contract: str = CONFIDENCE_HEAD_GRADIENT_CONTRACT_SHARED,
    ) -> None:
        super().__init__()
        if min(
            int(hidden_dim),
            int(adapter_dim),
            int(max_text_len),
            int(patch_hidden_dim),
            int(score_topk),
        ) <= 0:
            raise ValueError("confidence adapter dimensions must be positive")
        if not math.isfinite(float(patch_score_clip)) or float(patch_score_clip) <= 0:
            raise ValueError("patch_score_clip must be finite and positive")
        phrase_aggregation = str(phrase_aggregation).strip().lower()
        if phrase_aggregation not in CONFIDENCE_PHRASE_AGGREGATIONS:
            raise ValueError(
                "confidence phrase_aggregation must be one of "
                f"{CONFIDENCE_PHRASE_AGGREGATIONS}, got {phrase_aggregation!r}"
            )
        gate_gradient_contract = str(gate_gradient_contract).strip().lower()
        if gate_gradient_contract not in CONFIDENCE_GATE_GRADIENT_CONTRACTS:
            raise ValueError("confidence gate-gradient contract is invalid")
        head_gradient_contract = str(head_gradient_contract).strip().lower()
        if head_gradient_contract not in CONFIDENCE_HEAD_GRADIENT_CONTRACTS:
            raise ValueError("confidence head-gradient contract is invalid")
        if (
            gate_gradient_contract
            == CONFIDENCE_GATE_GRADIENT_CONTRACT_HARD_FORWARD_SOFT_BACKWARD
            and phrase_aggregation
            != CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
        ):
            raise ValueError(
                "soft-backward gate gradients require gated-pool absolute-cap "
                "aggregation"
            )
        if (
            head_gradient_contract
            == CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYED_ROUTER
            and gate_gradient_contract
            != CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_DEPLOYED_ROUTING_ST
        ):
            raise ValueError(
                "the independent deployed router requires the V47 deployed-routing "
                "gate contract"
            )
        if (
            gate_gradient_contract
            in {
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CONTINUOUS,
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CONTINUOUS_MONOTONE_DEPTH,
                CONFIDENCE_GATE_GRADIENT_CONTRACT_COMPLEMENTARY_TRUST_VETO,
                CONFIDENCE_GATE_GRADIENT_CONTRACT_UNGATED_MONOTONE_DEPTH,
                CONFIDENCE_GATE_GRADIENT_CONTRACT_FLOOR_GATED_MONOTONE_DEPTH,
                CONFIDENCE_GATE_GRADIENT_CONTRACT_INDEPENDENT_ABSOLUTE_LOGIT,
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CROSS_ATTENTION_INDEPENDENT_ABSOLUTE_LOGIT,
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ABSOLUTE_LOGIT,
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_CALIBRATED_LOGIT,
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_NORMALIZED_LOGIT,
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT,
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_SET_ATTENTION_LOGIT,
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_DEPLOYED_ROUTING_ST,
            }
            and phrase_aggregation
            != CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
        ):
            raise ValueError(
                "continuous modifier gates require the gated-pool confidence "
                "aggregation"
            )
        if (
            not math.isfinite(float(word_softmin_temperature))
            or float(word_softmin_temperature) <= 0.0
        ):
            raise ValueError("word_softmin_temperature must be finite and positive")
        if not math.isfinite(float(veto_gate_scale)) or float(veto_gate_scale) <= 0.0:
            raise ValueError("veto_gate_scale must be finite and positive")
        if not math.isfinite(float(veto_gate_offset)) or float(veto_gate_offset) < 0.0:
            raise ValueError("veto_gate_offset must be finite and non-negative")
        if (
            not math.isfinite(float(veto_coverage_offset))
            or not 0.0 <= float(veto_coverage_offset) < 1.0
        ):
            raise ValueError("veto_coverage_offset must be finite and in [0, 1)")
        if (
            not math.isfinite(float(veto_coverage_ramp))
            or float(veto_coverage_ramp) <= 0.0
            or float(veto_coverage_offset) + float(veto_coverage_ramp) > 1.0
        ):
            raise ValueError(
                "veto_coverage_ramp must be positive and end within coverage 1"
            )
        if (
            not math.isfinite(float(veto_cap_temperature))
            or float(veto_cap_temperature) <= 0.0
        ):
            raise ValueError("veto_cap_temperature must be finite and positive")
        if (
            not math.isfinite(float(veto_cap_initial_ceiling))
            or float(veto_cap_initial_ceiling) >= 0.0
        ):
            raise ValueError("veto_cap_initial_ceiling must be finite and negative")
        rank_evidence_contract = str(rank_evidence_contract).strip().lower()
        if rank_evidence_contract not in {
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_OFF,
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_SCALE,
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_AFFINE,
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_GATE_MARGIN,
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_CARRIER_TOKEN_SLOPE,
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_CARRIER_TOKEN_AFFINE,
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH,
        }:
            raise ValueError("confidence rank-evidence contract is invalid")
        if (
            not math.isfinite(float(residual_parameterization_gain))
            or float(residual_parameterization_gain) <= 0.0
        ):
            raise ValueError(
                "confidence residual_parameterization_gain must be finite and positive"
            )
        pool_feature_contract = str(pool_feature_contract).strip().lower()
        if pool_feature_contract not in CONFIDENCE_POOL_FEATURE_CONTRACTS:
            raise ValueError("confidence pool-feature contract is invalid")
        if (
            gate_gradient_contract
            in {
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ABSOLUTE_LOGIT,
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_CALIBRATED_LOGIT,
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_NORMALIZED_LOGIT,
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT,
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_SET_ATTENTION_LOGIT,
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_DEPLOYED_ROUTING_ST,
            }
        ) != (
            pool_feature_contract
            in {
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ABSOLUTE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_CALIBRATED,
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_NORMALIZED,
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC,
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_SET_ATTENTION,
                CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_REFERENCE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE,
            }
        ):
            raise ValueError(
                "candidate-absolute gate and pool-feature contracts must be paired"
            )
        if (
            gate_gradient_contract
            == CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ABSOLUTE_LOGIT
        ) != (
            pool_feature_contract
            == CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ABSOLUTE
        ):
            raise ValueError("v10 candidate-absolute contracts must match exactly")
        if (
            gate_gradient_contract
            == CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_CALIBRATED_LOGIT
        ) != (
            pool_feature_contract
            == CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_CALIBRATED
        ):
            raise ValueError("v11 candidate-calibrated contracts must match exactly")
        if (
            gate_gradient_contract
            == CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_NORMALIZED_LOGIT
        ) != (
            pool_feature_contract
            == CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_NORMALIZED
        ):
            raise ValueError("v12 candidate-normalized contracts must match exactly")
        asymmetric_gate_contract = gate_gradient_contract in {
            CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT,
            CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_DEPLOYED_ROUTING_ST,
        }
        asymmetric_pool_contract = pool_feature_contract in {
            CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC,
            CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE,
            CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_REFERENCE,
            CONFIDENCE_POOL_FEATURE_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE,
            CONFIDENCE_POOL_FEATURE_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE,
        }
        if asymmetric_gate_contract != asymmetric_pool_contract:
            raise ValueError(
                "candidate-asymmetric gate and pool contracts must match exactly"
            )
        fulltext_global_contract = head_gradient_contract in {
            CONFIDENCE_HEAD_GRADIENT_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE,
            CONFIDENCE_HEAD_GRADIENT_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE,
            CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE,
        }
        fulltext_global_pool = (
            pool_feature_contract
            in {
                CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_REFERENCE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE,
            }
        )
        if fulltext_global_contract != fulltext_global_pool:
            raise ValueError("full-expression head and pool must match exactly")
        if (
            head_gradient_contract
            == CONFIDENCE_HEAD_GRADIENT_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE
        ) != (
            pool_feature_contract
            == CONFIDENCE_POOL_FEATURE_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE
        ):
            raise ValueError("V55 local-candidate/global-absolute contracts must match")
        if (
            head_gradient_contract
            == CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE
        ) != (
            pool_feature_contract
            == CONFIDENCE_POOL_FEATURE_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE
        ):
            raise ValueError("V56 deployment-owned global contracts must match")
        if fulltext_global_contract and (
            gate_gradient_contract
            != CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT
            or phrase_aggregation
            != CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
        ):
            raise ValueError(
                "full-expression global confidence requires the raw word-veto "
                "feature contract"
            )
        if (
            gate_gradient_contract
            == CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_SET_ATTENTION_LOGIT
        ) != (
            pool_feature_contract
            == CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_SET_ATTENTION
        ):
            raise ValueError("v14 candidate-set-attention contracts must match exactly")
        scaled_parameterization = rank_evidence_contract in {
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_GATE_MARGIN,
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_CARRIER_TOKEN_SLOPE,
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_CARRIER_TOKEN_AFFINE,
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH,
        }
        if scaled_parameterization != (
            not math.isclose(
                float(residual_parameterization_gain),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "scaled rank-evidence contracts require a non-unit residual gain, "
                "and all other rank-evidence contracts require gain=1"
            )
        self.hidden_dim = int(hidden_dim)
        self.adapter_dim = int(adapter_dim)
        self.max_text_len = int(max_text_len)
        self.score_topk = int(score_topk)
        self.patch_score_clip = float(patch_score_clip)
        self.phrase_aggregation = phrase_aggregation
        self.word_softmin_temperature = float(word_softmin_temperature)
        self.veto_gate_scale = float(veto_gate_scale)
        self.veto_gate_offset = float(veto_gate_offset)
        self.veto_coverage_offset = float(veto_coverage_offset)
        self.veto_coverage_ramp = float(veto_coverage_ramp)
        self.veto_cap_temperature = float(veto_cap_temperature)
        self.veto_cap_initial_ceiling = float(veto_cap_initial_ceiling)
        self.rank_evidence_contract = rank_evidence_contract
        self.pool_feature_contract = pool_feature_contract
        self.gate_gradient_contract = gate_gradient_contract
        self.head_gradient_contract = head_gradient_contract
        self.residual_parameterization_gain = float(
            residual_parameterization_gain
        )

        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.text_norm = nn.LayerNorm(self.hidden_dim)
        self.query_projection = nn.Linear(self.hidden_dim, self.adapter_dim)
        self.text_projection = nn.Linear(self.hidden_dim, self.adapter_dim)
        self.query_bias = nn.Linear(self.adapter_dim, 1)
        self.token_bias = nn.Linear(self.adapter_dim, 1)
        self.patch_residual: Optional[nn.Sequential] = None
        if head_gradient_contract not in {
            CONFIDENCE_HEAD_GRADIENT_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE,
            CONFIDENCE_HEAD_GRADIENT_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE,
            CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE,
        }:
            self.patch_residual = nn.Sequential(
                nn.Linear(self.patch_stat_dim, int(patch_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(patch_hidden_dim), 1),
            )
        self.patch_feature = nn.Sequential(
            nn.Linear(self.patch_stat_dim, int(patch_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(patch_hidden_dim), self.hidden_dim),
        )
        self.feature_norm = nn.LayerNorm(self.hidden_dim)
        self.register_parameter("veto_cap_raw_ceiling", None)
        self.register_parameter("rank_evidence_residual_scale", None)
        self.register_parameter("rank_evidence_residual_bias", None)
        self.carrier_rank_slope: Optional[nn.Linear] = None
        self.rank_channel_norm: Optional[nn.LayerNorm] = None
        self.rank_channel_projection: Optional[nn.Linear] = None
        self.rank_channel_logit_projection: Optional[nn.Linear] = None
        self.rank_channel_output: Optional[nn.Linear] = None
        self.deployed_router_norm: Optional[nn.LayerNorm] = None
        self.deployed_router_residual: Optional[nn.Sequential] = None
        self.global_query_norm: Optional[nn.LayerNorm] = None
        self.global_query_trunk: Optional[nn.Sequential] = None
        self.cross_query_norm: Optional[nn.LayerNorm] = None
        self.cross_text_norm: Optional[nn.LayerNorm] = None
        self.cross_query_projection: Optional[nn.Linear] = None
        self.cross_text_projection: Optional[nn.Linear] = None
        self.cross_evidence_projection: Optional[nn.Linear] = None
        self.cross_attention: Optional[nn.MultiheadAttention] = None
        self.cross_ffn: Optional[nn.Sequential] = None
        self.cross_output_projection: Optional[nn.Linear] = None
        self.candidate_absolute_head: Optional[nn.Sequential] = None
        self.register_parameter("candidate_patch_scale_raw", None)
        self.register_parameter("candidate_veto_depth_raw", None)
        self.register_parameter("candidate_coverage_depth_raw", None)
        if self.phrase_aggregation in {
            CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_ABSOLUTE_CAP,
            CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
        } and self.head_gradient_contract not in {
            CONFIDENCE_HEAD_GRADIENT_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE,
            CONFIDENCE_HEAD_GRADIENT_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE,
            CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE,
        }:
            positive_ceiling_magnitude = -float(veto_cap_initial_ceiling)
            raw_ceiling = positive_ceiling_magnitude + math.log(
                -math.expm1(-positive_ceiling_magnitude)
            )
            self.veto_cap_raw_ceiling = nn.Parameter(
                torch.tensor(raw_ceiling, dtype=torch.float32)
            )
        if self.rank_evidence_contract in {
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_SCALE,
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_AFFINE,
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_GATE_MARGIN,
        }:
            self.rank_evidence_residual_scale = nn.Parameter(
                torch.zeros((), dtype=torch.float32)
            )
        if (
            self.rank_evidence_contract
            == CONFIDENCE_RANK_EVIDENCE_CONTRACT_AFFINE
        ):
            self.rank_evidence_residual_bias = nn.Parameter(
                torch.zeros((), dtype=torch.float32)
            )
        if self.rank_evidence_contract in {
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_CARRIER_TOKEN_SLOPE,
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_CARRIER_TOKEN_AFFINE,
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH,
        }:
            # Do not advance the enclosing deterministic confidence RNG: every
            # pre-existing adapter/pool tensor must retain its prior U0 value.
            with torch.random.fork_rng(devices=[]):
                self.carrier_rank_slope = nn.Linear(
                    self.adapter_dim,
                    1,
                    bias=(
                        self.rank_evidence_contract
                        in {
                            CONFIDENCE_RANK_EVIDENCE_CONTRACT_CARRIER_TOKEN_AFFINE,
                            CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH,
                        }
                    ),
                )
            nn.init.zeros_(self.carrier_rank_slope.weight)
            if self.carrier_rank_slope.bias is not None:
                nn.init.zeros_(self.carrier_rank_slope.bias)
        if (
            self.rank_evidence_contract
            == CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
        ):
            # Keep every pre-existing v18 tensor and the following confidence
            # pool bitwise identical at U0. Only the new verifier receives a
            # deterministic random basis; its bias-free output is exactly zero.
            with torch.random.fork_rng(devices=[]):
                self.rank_channel_norm = nn.LayerNorm(self.hidden_dim)
                self.rank_channel_projection = nn.Linear(
                    self.hidden_dim, self.adapter_dim
                )
                self.rank_channel_logit_projection = nn.Linear(
                    1, self.adapter_dim
                )
                self.rank_channel_output = nn.Linear(
                    self.adapter_dim, 1, bias=False
                )
            nn.init.zeros_(self.rank_channel_output.weight)
        if (
            self.pool_feature_contract
            in {
                CONFIDENCE_POOL_FEATURE_CONTRACT_SIGNED_RANK_QUERY,
                CONFIDENCE_POOL_FEATURE_CONTRACT_TOKEN_CONDITIONED,
                CONFIDENCE_POOL_FEATURE_CONTRACT_CROSS_ATTENTION,
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ABSOLUTE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_CALIBRATED,
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_NORMALIZED,
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC,
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_SET_ATTENTION,
                CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_REFERENCE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE,
            }
        ):
            # Preserve every v19 tensor and the following confidence-pool RNG.
            # The pool's zero-initialized output keeps U0 exactly identical.
            with torch.random.fork_rng(devices=[]):
                if (
                    self.pool_feature_contract
                    not in {
                        CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE,
                        CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_REFERENCE,
                        CONFIDENCE_POOL_FEATURE_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE,
                        CONFIDENCE_POOL_FEATURE_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE,
                    }
                ):
                    self.global_query_norm = nn.LayerNorm(self.hidden_dim)
                self.global_query_trunk = nn.Sequential(
                    nn.Linear(self.hidden_dim + 1, self.adapter_dim),
                    nn.GELU(),
                    nn.Linear(self.adapter_dim, self.hidden_dim),
                    nn.GELU(),
                )
        if (
            self.pool_feature_contract
            in {
                CONFIDENCE_POOL_FEATURE_CONTRACT_CROSS_ATTENTION,
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ABSOLUTE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_CALIBRATED,
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_NORMALIZED,
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC,
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_SET_ATTENTION,
                CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_REFERENCE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE,
            }
        ):
            cross_dim = max(128, 2 * self.adapter_dim)
            if cross_dim % 4 != 0:
                raise ValueError("cross-attention width must be divisible by four")
            with torch.random.fork_rng(devices=[]):
                self.cross_query_norm = nn.LayerNorm(self.hidden_dim)
                self.cross_text_norm = nn.LayerNorm(self.hidden_dim)
                self.cross_query_projection = nn.Linear(
                    self.hidden_dim, cross_dim
                )
                self.cross_text_projection = nn.Linear(
                    self.hidden_dim, cross_dim
                )
                self.cross_evidence_projection = nn.Linear(3, cross_dim)
                self.cross_attention = nn.MultiheadAttention(
                    cross_dim, 4, batch_first=True
                )
                self.cross_ffn = nn.Sequential(
                    nn.LayerNorm(cross_dim),
                    nn.Linear(cross_dim, 2 * cross_dim),
                    nn.GELU(),
                    nn.Linear(2 * cross_dim, cross_dim),
                )
                self.cross_output_projection = nn.Linear(
                    cross_dim, self.hidden_dim
                )
                if (
                    self.pool_feature_contract
                    in {
                        CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ABSOLUTE,
                        CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_CALIBRATED,
                        CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_NORMALIZED,
                        CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC,
                        CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_SET_ATTENTION,
                        CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE,
                        CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_REFERENCE,
                        CONFIDENCE_POOL_FEATURE_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE,
                        CONFIDENCE_POOL_FEATURE_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE,
                    }
                ):
                    # The verifier emits one absolute logit per admitted
                    # candidate before any sample-level pooling. This is the
                    # surface supervised by positive/TN local-absolute losses;
                    # its final affine is zero initialized so U0 is neutral.
                    self.candidate_absolute_head = nn.Sequential(
                        nn.LayerNorm(self.hidden_dim),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        nn.GELU(),
                        nn.Linear(self.hidden_dim, 1),
                    )
                    nn.init.zeros_(self.candidate_absolute_head[-1].weight)
                    nn.init.zeros_(self.candidate_absolute_head[-1].bias)
                    if (
                        self.pool_feature_contract
                        in {
                            CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_CALIBRATED,
                            CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_NORMALIZED,
                            CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC,
                            CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_SET_ATTENTION,
                        }
                    ):
                        if (
                            self.pool_feature_contract
                            == CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_CALIBRATED
                        ):
                            self.candidate_patch_scale_raw = nn.Parameter(
                                torch.zeros((), dtype=torch.float32)
                            )
                        self.candidate_veto_depth_raw = nn.Parameter(
                            torch.zeros((), dtype=torch.float32)
                        )
                        self.candidate_coverage_depth_raw = nn.Parameter(
                            torch.zeros((), dtype=torch.float32)
                        )

        if (
            self.head_gradient_contract
            == CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYED_ROUTER
        ):
            # The router is the third confidence parameter owner. Forking the
            # RNG preserves every inherited V47 tensor and the following pool.
            with torch.random.fork_rng(devices=[]):
                self.deployed_router_norm = nn.LayerNorm(
                    self.deployed_router_feature_dim
                )
                self.deployed_router_residual = nn.Sequential(
                    nn.Linear(self.deployed_router_feature_dim, self.adapter_dim),
                    nn.GELU(),
                    nn.Linear(self.adapter_dim, 1),
                )
            nn.init.zeros_(self.deployed_router_residual[-1].weight)
            nn.init.zeros_(self.deployed_router_residual[-1].bias)

        # LoRA-style asymmetric initialization makes every token residual
        # exactly zero while preserving a random text basis. Token BCE can
        # update query_projection and token_bias on the first optimizer step.
        nn.init.zeros_(self.query_projection.weight)
        nn.init.zeros_(self.query_projection.bias)
        nn.init.zeros_(self.query_bias.weight)
        nn.init.zeros_(self.query_bias.bias)
        nn.init.zeros_(self.token_bias.weight)
        nn.init.zeros_(self.token_bias.bias)

        # Category-only rows initially inherit the patch logit exactly; the
        # adapter learns calibration without erasing the pretrained evidence.
        if self.patch_residual is not None:
            nn.init.zeros_(self.patch_residual[-1].weight)
            nn.init.zeros_(self.patch_residual[-1].bias)
        nn.init.zeros_(self.patch_feature[-1].weight)
        nn.init.zeros_(self.patch_feature[-1].bias)
        if (
            self.head_gradient_contract
            == CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE
        ):
            # V56 keeps the V55 checkpoint/state surface for exact migration and
            # diagnostics, but no non-deployed candidate parameter is active.
            # Freezing the complete head also prevents AdamW decay from moving a
            # nominally zero-loss branch.
            _set_module_trainable(self.candidate_absolute_head, False)
        self._assert_head_parameter_ownership()

    @staticmethod
    def _module_parameters(*modules: Optional[nn.Module]) -> tuple[nn.Parameter, ...]:
        parameters: list[nn.Parameter] = []
        seen: set[int] = set()
        for module in modules:
            if module is None:
                continue
            for parameter in module.parameters():
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    parameters.append(parameter)
        return tuple(parameters)

    @staticmethod
    def _merge_parameters(
        module_parameters: Sequence[nn.Parameter],
        optional_parameters: Sequence[Optional[nn.Parameter]],
    ) -> tuple[nn.Parameter, ...]:
        parameters = list(module_parameters)
        seen = {id(parameter) for parameter in parameters}
        for parameter in optional_parameters:
            if parameter is not None and id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
        return tuple(parameters)

    def token_veto_parameters(self) -> tuple[nn.Parameter, ...]:
        """Parameters owned by token logits and raw word-veto supervision."""
        modules = self._module_parameters(
            self.query_norm,
            self.text_norm,
            self.query_projection,
            self.text_projection,
            self.query_bias,
            self.token_bias,
            self.carrier_rank_slope,
            self.rank_channel_norm,
            self.rank_channel_projection,
            self.rank_channel_logit_projection,
            self.rank_channel_output,
        )
        return self._merge_parameters(
            modules,
            (
                self.rank_evidence_residual_scale,
                self.rank_evidence_residual_bias,
            ),
        )

    def deployed_router_parameters(self) -> tuple[nn.Parameter, ...]:
        """Parameters owned exclusively by deployed-routing supervision."""
        return self._module_parameters(
            self.deployed_router_norm,
            self.deployed_router_residual,
        )

    def global_absolute_parameters(self) -> tuple[nn.Parameter, ...]:
        """Parameters owned by the deployed sample-global confidence losses."""
        if (
            self.head_gradient_contract
            == CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE
        ):
            return self._merge_parameters(
                self.deployed_global_trunk_parameters(),
                self.sample_calibrator_parameters(),
            )
        return self._merge_parameters(
            self.candidate_absolute_parameters(),
            self.sample_calibrator_parameters(),
        )

    def deployed_global_trunk_parameters(self) -> tuple[nn.Parameter, ...]:
        """Trainable representation owned by the deployed global logit."""
        return self._merge_parameters(
            self._module_parameters(
                self.patch_residual,
                self.patch_feature,
                self.feature_norm,
                self.global_query_norm,
                self.global_query_trunk,
                self.cross_query_norm,
                self.cross_text_norm,
                self.cross_query_projection,
                self.cross_text_projection,
                self.cross_evidence_projection,
                self.cross_attention,
                self.cross_ffn,
                self.cross_output_projection,
            ),
            (
                self.veto_cap_raw_ceiling,
                self.candidate_patch_scale_raw,
                self.candidate_veto_depth_raw,
            ),
        )

    def candidate_diagnostic_parameters(self) -> tuple[nn.Parameter, ...]:
        """Serialized non-deployed candidate head, frozen by the V56 contract."""
        return self._module_parameters(self.candidate_absolute_head)

    def candidate_absolute_parameters(self) -> tuple[nn.Parameter, ...]:
        """Parameters owned by query-local candidate absolute supervision."""
        diagnostic = self.candidate_diagnostic_parameters()
        if (
            self.head_gradient_contract
            == CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE
        ):
            return diagnostic
        return self._merge_parameters(
            self.deployed_global_trunk_parameters(), diagnostic
        )

    def sample_calibrator_parameters(self) -> tuple[nn.Parameter, ...]:
        """Adapter parameters owned only by sample-global calibration losses."""
        return self._merge_parameters((), (self.candidate_coverage_depth_raw,))

    def _assert_head_parameter_ownership(self) -> None:
        token = {id(parameter) for parameter in self.token_veto_parameters()}
        router = {
            id(parameter) for parameter in self.deployed_router_parameters()
        }
        absolute = {id(parameter) for parameter in self.global_absolute_parameters()}
        candidate = {
            id(parameter) for parameter in self.candidate_absolute_parameters()
        }
        sample = {id(parameter) for parameter in self.sample_calibrator_parameters()}
        complete = {id(parameter) for parameter in self.parameters()}
        diagnostic = {
            id(parameter) for parameter in self.candidate_diagnostic_parameters()
        }
        deployment_owned = (
            self.head_gradient_contract
            == CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE
        )
        active_complete = complete - diagnostic if deployment_owned else complete
        deployed_router_enabled = (
            self.head_gradient_contract
            == CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYED_ROUTER
        )
        candidate_sample_enabled = (
            self.head_gradient_contract
            == CONFIDENCE_HEAD_GRADIENT_CONTRACT_CANDIDATE_SAMPLE
        )
        ownership_invalid = (
            not token
            or not absolute
            or bool(token & absolute)
            or bool(token & router)
            or bool(router & absolute)
            or token | router | absolute != active_complete
            or (deployed_router_enabled and not router)
            or (not deployed_router_enabled and bool(router))
            or (
                deployment_owned
                and (
                    not diagnostic
                    or bool(diagnostic & (token | router | absolute))
                    or any(parameter.requires_grad for parameter in self.candidate_diagnostic_parameters())
                )
            )
            or (
                candidate_sample_enabled
                and (
                    not candidate
                    or not sample
                    or bool(candidate & sample)
                    or candidate | sample != absolute
                )
            )
        )
        if ownership_invalid:
            raise RuntimeError(
                "confidence token-veto/deployed-router/global-absolute parameter "
                "ownership is empty, overlapping, or incomplete"
            )

    def _token_conditioned_global_feature(
        self,
        *,
        query: Tensor,
        text: Tensor,
        rank_token: Tensor,
        token_residual: Tensor,
        modifier_mask: Tensor,
    ) -> Tensor:
        """Inject full modifier-token identity into the lightweight global pool.

        The preceding signed pool saw only a detached query and one reference
        logit; text could affect its veto depth only through a scalar maximum
        residual gate. This parameter-free context keeps the identical
        zero-initialized pool surface while letting that pool distinguish which
        attribute/relation word is weak for each candidate.
        """
        if self.global_query_norm is None or self.global_query_trunk is None:
            raise RuntimeError("token-conditioned confidence pool is incomplete")
        if tuple(rank_token.shape) != tuple(token_residual.shape):
            raise ValueError("rank token and token residual tensors must align")
        if tuple(modifier_mask.shape) != tuple(text.shape[:2]):
            raise ValueError("modifier mask must align with text features")

        token_mask = modifier_mask[None, :, None, :]
        has_modifier = modifier_mask.any(dim=-1)[None, :, None, None]
        finite_rank = torch.where(
            torch.isfinite(rank_token), rank_token, torch.zeros_like(rank_token)
        ).clamp(min=-20.0, max=20.0)
        confidence_token = finite_rank - token_residual.float()
        # Learned edit residuals focus changed words, while weak inherited rank
        # evidence supplies a useful U0 prior. No edit label enters inference.
        priority = (
            token_residual.float()
            - 0.25 * torch.tanh(confidence_token / 5.0)
        ).masked_fill(~token_mask, -torch.inf)
        attention = torch.softmax(priority, dim=-1)
        attention = torch.where(
            has_modifier,
            attention,
            torch.zeros_like(attention),
        )
        normalized_text = F.layer_norm(
            text.float(), normalized_shape=(self.hidden_dim,)
        )
        token_context = torch.einsum(
            "lmnt,mtd->lmnd", attention, normalized_text
        )
        normalized_query = self.global_query_norm(query).float()
        interaction = normalized_query * token_context
        conditioned_query = (
            normalized_query + token_context + interaction
        ) / math.sqrt(3.0)
        return conditioned_query.to(dtype=query.dtype)

    def _cross_attention_global_feature(
        self,
        *,
        query: Tensor,
        text: Tensor,
        rank_token: Tensor,
        token_residual: Tensor,
        modifier_mask: Tensor,
        phrase_mask: Tensor,
    ) -> Tensor:
        """Verify every candidate against the available modifier-token set.

        Unlike the v27 weighted mean, this path preserves token identity until
        after candidate-specific attention.  Rank query/text/logit inputs are
        detached by ``forward``; the learned interaction therefore calibrates
        absolute confidence without changing the rank tower.
        """
        modules = (
            self.global_query_trunk,
            self.cross_query_norm,
            self.cross_text_norm,
            self.cross_query_projection,
            self.cross_text_projection,
            self.cross_evidence_projection,
            self.cross_attention,
            self.cross_ffn,
            self.cross_output_projection,
        )
        if any(module is None for module in modules):
            raise RuntimeError("cross-attention confidence pool is incomplete")
        if tuple(rank_token.shape) != tuple(token_residual.shape):
            raise ValueError("rank token and token residual tensors must align")
        if tuple(modifier_mask.shape) != tuple(text.shape[:2]):
            raise ValueError("modifier mask must align with text features")
        if tuple(phrase_mask.shape) != tuple(text.shape[:2]):
            raise ValueError("phrase mask must align with text features")

        # Modifiers carry attribute/relation/spatial evidence. Category-only
        # expressions fall back to their phrase tokens so the absolute head
        # remains defined, while patch statistics retain their dedicated path.
        has_modifier = modifier_mask.any(dim=-1)
        selected_mask = torch.where(
            has_modifier[:, None], modifier_mask, phrase_mask
        )
        has_selected = selected_mask.any(dim=-1)
        safe_mask = selected_mask.clone()
        safe_mask[~has_selected, 0] = True
        selected_count = safe_mask.sum(dim=-1)
        selected_width = int(selected_count.max().item())
        selected_index = torch.argsort(
            safe_mask.to(dtype=torch.int64), dim=-1, descending=True
        )[:, :selected_width]
        selected_valid = torch.gather(safe_mask, 1, selected_index)

        normalized_query = self.cross_query_norm(query)
        projected_query = self.cross_query_projection(normalized_query)
        normalized_text = self.cross_text_norm(text)
        projected_text = self.cross_text_projection(normalized_text)
        text_index = selected_index[..., None].expand(
            -1, -1, int(projected_text.shape[-1])
        )
        selected_text = torch.gather(projected_text, 1, text_index)

        layer_count, expression_count, candidate_count, _ = rank_token.shape
        token_index = selected_index[None, :, None, :].expand(
            layer_count, expression_count, candidate_count, selected_width
        )
        finite_rank = torch.where(
            torch.isfinite(rank_token), rank_token, torch.zeros_like(rank_token)
        ).clamp(min=-20.0, max=20.0)
        selected_rank = torch.gather(finite_rank, -1, token_index)
        selected_residual = torch.gather(token_residual.float(), -1, token_index)
        selected_confidence = selected_rank - selected_residual
        evidence = torch.stack(
            (
                torch.tanh(selected_rank / 5.0),
                torch.tanh(selected_residual / 5.0),
                torch.tanh(selected_confidence / 5.0),
            ),
            dim=-1,
        ).to(dtype=projected_query.dtype)
        key_value = (
            selected_text[None, :, None]
            + self.cross_evidence_projection(evidence)
        )

        flat_query = projected_query.reshape(-1, 1, projected_query.shape[-1])
        flat_key_value = key_value.reshape(
            -1, selected_width, key_value.shape[-1]
        )
        key_padding_mask = (
            ~selected_valid[None, :, None, :]
            .expand(layer_count, expression_count, candidate_count, selected_width)
            .reshape(-1, selected_width)
        )
        attended, _ = self.cross_attention(
            flat_query,
            flat_key_value,
            flat_key_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attended = attended + self.cross_ffn(attended)
        attended = self.cross_output_projection(attended[:, 0]).reshape_as(query)
        attended = attended.masked_fill(
            ~has_selected[None, :, None, None], 0.0
        )
        conditioned_query = (normalized_query + attended) / math.sqrt(2.0)
        return conditioned_query.to(dtype=query.dtype)

    def absolute_cap_ceiling(self) -> Tensor:
        if self.veto_cap_raw_ceiling is None:
            raise RuntimeError(
                "absolute cap ceiling requires an absolute-cap aggregation"
            )
        return -F.softplus(self.veto_cap_raw_ceiling.float())

    def candidate_calibration_depths(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return non-negative patch/gate/coverage calibration coefficients."""
        normalized_contract = (
            self.pool_feature_contract
            == CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_NORMALIZED
        )
        asymmetric_contract = (
            self.pool_feature_contract
            in {
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC,
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_SET_ATTENTION,
            }
        )
        required = (
            self.candidate_veto_depth_raw,
            self.candidate_coverage_depth_raw,
        )
        if any(parameter is None for parameter in required) or (
            not (normalized_contract or asymmetric_contract)
            and self.candidate_patch_scale_raw is None
        ):
            raise RuntimeError("candidate calibration parameters are unavailable")
        parameters = (
            self.candidate_patch_scale_raw,
            self.candidate_veto_depth_raw,
            self.candidate_coverage_depth_raw,
        )
        if normalized_contract:
            gains = (0.0, float(self.adapter_dim), float(self.adapter_dim))
        elif asymmetric_contract:
            # Raw patch logits retain absolute category evidence, but their
            # cross-sample carrier has units set by the configured clip.  The
            # fixed reciprocal therefore removes scale leakage without a
            # validation-tuned coefficient.  Token mismatch is a local veto
            # and uses the bottleneck's standard-deviation scale; aggregate
            # coverage is a sample-tail veto and uses its full width.
            gains = (
                0.0,
                math.sqrt(float(self.adapter_dim)),
                float(self.adapter_dim),
            )
        else:
            gains = (1.0, 1.0, 1.0)
        depths = []
        for index, (parameter, gain) in enumerate(zip(parameters, gains)):
            if parameter is None:
                value = self.candidate_veto_depth_raw.float().new_zeros(())
                if asymmetric_contract and index == 0:
                    value = value + (1.0 / float(self.patch_score_clip))
                depths.append(value)
                continue
            exact = F.relu(parameter.float())
            surrogate = F.softplus(parameter.float()) - math.log(2.0)
            depths.append(
                gain * _ExactForwardSurrogateBackward.apply(exact, surrogate)
            )
        return depths[0], depths[1], depths[2]

    def _patch_statistics(
        self,
        patch_logits: Tensor,
        patch_standardized: Tensor,
        candidate_mask: Tensor,
    ) -> Tensor:
        if patch_logits.dim() != 2:
            raise ValueError("patch logits must have shape (M,N)")
        if tuple(patch_standardized.shape) != tuple(patch_logits.shape):
            raise ValueError("standardized patch logits must align with patch logits")
        if tuple(candidate_mask.shape) != tuple(patch_logits.shape):
            raise ValueError("candidate mask must align with patch logits")
        mask = candidate_mask.to(device=patch_logits.device, dtype=torch.bool)
        standardized = patch_standardized.float()
        statistics_score = (
            standardized
            if self.pool_feature_contract
            == CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_NORMALIZED
            else patch_logits
        )
        _, row_statistics = _masked_score_statistics(
            statistics_score, mask, topk=self.score_topk
        )
        standardized_max = standardized.masked_fill(~mask, -torch.inf).max(
            dim=1
        ).values
        candidate_presence = (
            standardized.sigmoid()
            if self.pool_feature_contract
            == CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_NORMALIZED
            else patch_logits.float().sigmoid()
        )
        candidate_statistics = torch.stack(
            (
                candidate_presence,
                standardized,
                standardized_max[:, None] - standardized,
            ),
            dim=-1,
        )
        return torch.cat(
            (
                candidate_statistics,
                row_statistics[:, None].expand(-1, patch_logits.shape[1], -1),
            ),
            dim=-1,
        ).detach()

    def _sparse_rank_channel_residual(
        self,
        *,
        query: Tensor,
        text: Tensor,
        rank_token: Tensor,
        modifier_mask: Tensor,
    ) -> Tensor:
        modules = (
            self.rank_channel_norm,
            self.rank_channel_projection,
            self.rank_channel_logit_projection,
            self.rank_channel_output,
        )
        if any(module is None for module in modules):
            raise RuntimeError("sparse rank-channel verifier is incomplete")
        rank_channel_norm = self.rank_channel_norm
        rank_channel_projection = self.rank_channel_projection
        rank_channel_logit_projection = self.rank_channel_logit_projection
        rank_channel_output = self.rank_channel_output
        if not all(
            isinstance(module, nn.Module)
            for module in (
                rank_channel_norm,
                rank_channel_projection,
                rank_channel_logit_projection,
                rank_channel_output,
            )
        ):
            raise AssertionError("sparse rank-channel verifier was not materialized")

        expression_residuals = []
        for expression_index in range(int(text.shape[0])):
            token_indices = torch.nonzero(
                modifier_mask[expression_index], as_tuple=False
            ).flatten()
            if int(token_indices.numel()) == 0:
                expression_residuals.append(
                    rank_token[:, expression_index].new_zeros(
                        rank_token.shape[0],
                        rank_token.shape[2],
                        rank_token.shape[3],
                    )
                )
                continue

            expression_query = query[:, expression_index]
            expression_text = text[expression_index].index_select(
                0, token_indices
            )
            # Materialize only modifier-token pairs. Never construct the dense
            # (L,M,N,T,D) interaction tensor and never consume edit labels.
            rank_channel = (
                expression_query[:, :, None, :]
                * expression_text[None, None, :, :]
            )
            hidden = rank_channel_projection(rank_channel_norm(rank_channel))
            bounded_rank = torch.where(
                torch.isfinite(rank_token[:, expression_index]),
                rank_token[:, expression_index],
                torch.zeros_like(rank_token[:, expression_index]),
            ).clamp(min=-20.0, max=20.0)
            sparse_rank = bounded_rank.index_select(-1, token_indices) / 20.0
            hidden = hidden + rank_channel_logit_projection(
                sparse_rank[..., None].to(dtype=hidden.dtype)
            )
            sparse_residual = rank_channel_output(F.gelu(hidden)).squeeze(-1).float()
            expression_residuals.append(
                rank_token[:, expression_index]
                .new_zeros(
                    rank_token.shape[0],
                    rank_token.shape[2],
                    rank_token.shape[3],
                )
                .index_copy(-1, token_indices, sparse_residual)
            )
        return torch.stack(expression_residuals, dim=1)

    def _deployed_routing_gate(
        self,
        *,
        raw_gate: Tensor,
        veto_probability: Tensor,
        rank_token: Tensor,
        confidence_token: Tensor,
        token_residual: Tensor,
        modifier_mask: Tensor,
        patch: Tensor,
        standardized: Tensor,
        candidate_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply a bounded, independently trained residual to the raw veto gate."""
        if (
            self.head_gradient_contract
            != CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYED_ROUTER
        ):
            return raw_gate, torch.zeros_like(raw_gate)
        if self.deployed_router_norm is None or self.deployed_router_residual is None:
            raise RuntimeError("deployed-routing adapter is incomplete")

        raw = raw_gate.detach().float()
        rank = torch.where(
            torch.isfinite(rank_token.detach()),
            rank_token.detach().float(),
            torch.zeros_like(rank_token, dtype=torch.float32),
        ).clamp(min=-20.0, max=20.0)
        confidence = torch.where(
            torch.isfinite(confidence_token.detach()),
            confidence_token.detach().float(),
            torch.zeros_like(confidence_token, dtype=torch.float32),
        ).clamp(min=-20.0, max=20.0)
        residual = torch.where(
            torch.isfinite(token_residual.detach()),
            token_residual.detach().float(),
            torch.zeros_like(token_residual, dtype=torch.float32),
        ).clamp(min=-40.0, max=40.0)
        token_mask = modifier_mask.detach()[None, :, None, :]
        token_count = token_mask.sum(dim=-1).clamp_min(1).float()
        has_modifier = modifier_mask.detach().any(dim=-1)[None, :, None]

        def _masked_mean(value: Tensor) -> Tensor:
            return (value * token_mask.float()).sum(dim=-1) / token_count

        def _masked_min(value: Tensor) -> Tensor:
            minimum = value.masked_fill(~token_mask, torch.inf).min(dim=-1).values
            return torch.where(has_modifier, minimum, torch.zeros_like(minimum))

        residual_max = residual.masked_fill(~token_mask, -torch.inf).max(
            dim=-1
        ).values
        residual_max = torch.where(
            has_modifier, residual_max, torch.zeros_like(residual_max)
        )
        patch_feature = torch.tanh(
            patch.detach().float() / self.patch_score_clip
        )[None].expand_as(raw)
        standardized_feature = torch.tanh(
            standardized.detach().float() / self.patch_score_clip
        )[None].expand_as(raw)
        router_features = torch.stack(
            (
                raw,
                veto_probability.detach().float(),
                _masked_min(confidence).sigmoid(),
                _masked_mean(confidence).sigmoid(),
                _masked_min(rank).sigmoid(),
                _masked_mean(rank).sigmoid(),
                torch.tanh(residual_max / 5.0),
                torch.tanh(_masked_mean(residual) / 5.0),
                patch_feature,
                standardized_feature,
            ),
            dim=-1,
        ).detach()
        if int(router_features.shape[-1]) != self.deployed_router_feature_dim:
            raise AssertionError("deployed-routing feature width changed")

        router_hidden = self.deployed_router_norm(router_features)
        router_logit = self.deployed_router_residual(router_hidden).squeeze(-1)
        router_active = (
            has_modifier & candidate_mask.detach()[None].to(dtype=torch.bool)
        ).to(dtype=router_logit.dtype)
        signed_residual = torch.tanh(router_logit.float()) * router_active
        surrogate_gate = raw + signed_residual
        exact_gate = surrogate_gate.clamp(min=0.0, max=1.0)
        deployed_gate = _ExactForwardSurrogateBackward.apply(
            exact_gate, surrogate_gate
        )
        return deployed_gate, signed_residual

    def forward(
        self,
        *,
        rank_token_layers: Tensor,
        query_layers: Tensor,
        text_features: Tensor,
        phrase_token_mask: Tensor,
        score_token_mask: Tensor,
        patch_logits: Tensor,
        patch_standardized: Tensor,
        candidate_mask: Tensor,
        score_word_group_ids: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        if query_layers.dim() != 4 or int(query_layers.shape[-1]) != self.hidden_dim:
            raise ValueError("query_layers must have shape (L,M,N,D)")
        if rank_token_layers.dim() != 4 or tuple(rank_token_layers.shape[:3]) != tuple(
            query_layers.shape[:3]
        ):
            raise ValueError("rank_token_layers must have shape (L,M,N,T)")
        if text_features.dim() != 3 or int(text_features.shape[-1]) != self.hidden_dim:
            raise ValueError("text_features must have shape (M,T,D)")
        if int(text_features.shape[0]) != int(query_layers.shape[1]):
            raise ValueError("query and text expression batches must align")
        if int(text_features.shape[1]) != self.max_text_len:
            raise ValueError("text_features must be padded to max_text_len")
        if int(rank_token_layers.shape[-1]) != self.max_text_len:
            raise ValueError("rank token logits must be padded to max_text_len")
        expected_token_shape = tuple(text_features.shape[:2])
        if tuple(phrase_token_mask.shape) != expected_token_shape:
            raise ValueError("phrase token mask must align with text features")
        if tuple(score_token_mask.shape) != expected_token_shape:
            raise ValueError("score token mask must align with text features")
        if tuple(patch_logits.shape) != tuple(query_layers.shape[1:3]):
            raise ValueError("patch logits must align with query layers")

        # Detach again at the ownership boundary even when the caller already
        # evaluated the rank tower under no_grad.  This is the architectural
        # guarantee that absolute confidence cannot update relative ranking.
        # Fuse in FP32 even under AMP.  A newly learned small residual must not
        # disappear when it is subtracted from a large half-precision logit.
        rank_token = rank_token_layers.detach().float()
        query = query_layers.detach()
        text = text_features.detach()
        patch = patch_logits.detach().float()
        standardized = patch_standardized.detach().float()
        mask = candidate_mask.detach().to(dtype=torch.bool)
        phrase_mask = phrase_token_mask.detach().to(dtype=torch.bool)
        modifier_mask = score_token_mask.detach().to(dtype=torch.bool) & phrase_mask
        if self.phrase_aggregation in CONFIDENCE_PHRASE_AGGREGATION_WORD_VETOES:
            if score_word_group_ids is None:
                raise ValueError(
                    "word-veto confidence requires score_word_group_ids"
                )
            word_group_ids = score_word_group_ids.detach().to(
                device=rank_token.device, dtype=torch.long
            )
            if tuple(word_group_ids.shape) != tuple(modifier_mask.shape):
                raise ValueError("score_word_group_ids must align with score_token_mask")
        else:
            word_group_ids = None

        modifier_valid = modifier_mask.any(dim=-1)
        reference_modifier_layers = torch.stack(
            [
                DenseExpressionTower.aggregate_phrase_logits(
                    rank_token[layer], modifier_mask
                )
                for layer in range(int(rank_token.shape[0]))
            ],
            dim=0,
        )
        frozen_full_phrase_layers = torch.stack(
            [
                DenseExpressionTower.aggregate_phrase_logits(
                    rank_token[layer], phrase_mask
                )
                for layer in range(int(rank_token.shape[0]))
            ],
            dim=0,
        )
        reference_category_logit = patch.clamp(
            min=-self.patch_score_clip, max=self.patch_score_clip
        )
        reference_base_layers = reference_category_logit[None] + torch.where(
            modifier_valid[None, :, None],
            reference_modifier_layers,
            torch.zeros_like(reference_modifier_layers),
        )
        if self.head_gradient_contract in {
            CONFIDENCE_HEAD_GRADIENT_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE,
            CONFIDENCE_HEAD_GRADIENT_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE,
            CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE,
        }:
            # V53 inherits the complete frozen rank semantics at U0. Patch
            # evidence remains responsible for admission and is also available
            # to the learned residual through ``dense_feature`` below.
            reference_base_layers = frozen_full_phrase_layers
        reference_carrier_index_layers = torch.stack(
            [
                _frozen_reference_carrier_index(
                    reference_base_layers[layer], mask
                )
                for layer in range(int(reference_base_layers.shape[0]))
            ],
            dim=0,
        )

        query_latent = self.query_projection(self.query_norm(query))
        text_latent = self.text_projection(self.text_norm(text))
        token_residual_layers = torch.einsum(
            "lmnd,mtd->lmnt", query_latent, text_latent
        ) / math.sqrt(float(self.adapter_dim))
        token_residual_layers = (
            token_residual_layers
            + self.query_bias(query_latent)
            + self.token_bias(text_latent)[None, :, None, :, 0]
        )
        token_residual_layers = token_residual_layers.float()
        if self.rank_evidence_residual_scale is not None:
            rank_evidence = torch.where(
                torch.isfinite(rank_token), rank_token, torch.zeros_like(rank_token)
            ).clamp(min=-20.0, max=20.0)
            token_residual_layers = token_residual_layers + (
                self.rank_evidence_residual_scale.float() * rank_evidence
            )
            if self.rank_evidence_residual_bias is not None:
                token_residual_layers = token_residual_layers + (
                    self.rank_evidence_residual_bias.float()
                )
        if self.carrier_rank_slope is not None:
            carrier_token_slope = self.carrier_rank_slope(
                text_latent.detach()
            ).squeeze(-1).float()
            bounded_rank_evidence = torch.where(
                torch.isfinite(rank_token), rank_token, torch.zeros_like(rank_token)
            ).clamp(min=-20.0, max=20.0) / 20.0
            carrier_mask = F.one_hot(
                reference_carrier_index_layers,
                num_classes=int(rank_token.shape[2]),
            ).to(dtype=bounded_rank_evidence.dtype)
            carrier_token_residual = (
                carrier_mask[..., None]
                * modifier_mask[None, :, None, :].to(
                    dtype=bounded_rank_evidence.dtype
                )
                * carrier_token_slope[None, :, None, :]
                * bounded_rank_evidence
            )
            token_residual_layers = token_residual_layers + (
                self.residual_parameterization_gain * carrier_token_residual
            )
        elif (
            self.rank_evidence_contract
            == CONFIDENCE_RANK_EVIDENCE_CONTRACT_GATE_MARGIN
        ):
            token_residual_layers = (
                token_residual_layers * self.residual_parameterization_gain
            )
        if (
            self.rank_evidence_contract
            == CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
        ):
            token_residual_layers = token_residual_layers + (
                self.residual_parameterization_gain
                * self._sparse_rank_channel_residual(
                    query=query,
                    text=text,
                    rank_token=rank_token,
                    modifier_mask=modifier_mask,
                )
            )
        token_residual_layers = token_residual_layers.masked_fill(
            ~phrase_mask[None, :, None], 0.0
        )
        token_layers = rank_token - token_residual_layers
        global_token_residual_layers = token_residual_layers
        if (
            self.head_gradient_contract
            in CONFIDENCE_HEAD_GRADIENT_CONTRACT_SPLIT_SEMANTICS
        ):
            # Absolute calibration may consume the learned lexical evidence at
            # inference, but its local/global/tail losses cannot rewrite the
            # token-veto scorer that edit BCE and routing supervision own.
            global_token_residual_layers = token_residual_layers.detach()

        patch_statistics = self._patch_statistics(patch, standardized, mask)
        patch_input = patch_statistics.to(dtype=query.dtype)
        category_logit = patch.clamp(
            min=-self.patch_score_clip, max=self.patch_score_clip
        )
        if self.patch_residual is not None:
            category_logit = category_logit + self.patch_residual(
                patch_input
            ).squeeze(-1).float()
        modifier_layers = torch.stack(
            [
                DenseExpressionTower.aggregate_phrase_logits(
                    token_layers[layer], modifier_mask
                )
                for layer in range(int(token_layers.shape[0]))
            ],
            dim=0,
        )
        identity_base_layers = category_logit[None] + torch.where(
            modifier_valid[None, :, None],
            modifier_layers.detach(),
            torch.zeros_like(modifier_layers),
        )
        if self.phrase_aggregation in CONFIDENCE_PHRASE_AGGREGATION_WORD_VETOES:
            if word_group_ids is None:
                raise AssertionError("word group contract was not materialized")
            veto_probability, raw_mismatch_gate_layers = (
                _word_normalized_softmin_probability(
                    token_layers,
                    token_residual_layers,
                    modifier_mask,
                    word_group_ids,
                    temperature=self.word_softmin_temperature,
                    gate_scale=self.veto_gate_scale,
                    gate_offset=self.veto_gate_offset,
                    gate_gradient_contract=self.gate_gradient_contract,
                )
            )
            (
                deployed_routing_gate_layers,
                deployed_routing_residual_layers,
            ) = self._deployed_routing_gate(
                raw_gate=raw_mismatch_gate_layers,
                veto_probability=veto_probability,
                rank_token=rank_token,
                confidence_token=token_layers,
                token_residual=token_residual_layers,
                modifier_mask=modifier_mask,
                patch=patch,
                standardized=standardized,
                candidate_mask=mask,
            )
            mismatch_gate_layers = deployed_routing_gate_layers
            epsilon = max(float(torch.finfo(veto_probability.dtype).eps), 1e-6)
            if self.phrase_aggregation in {
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_ABSOLUTE_CAP,
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
            }:
                # The v4 veto acts only after absolute pooling. Keeping the
                # candidate base at identity prevents a small lexical penalty
                # from changing the scale that the post-pool cap calibrates.
                base_layers = identity_base_layers
            elif (
                self.phrase_aggregation
                == CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_PENALTY
            ):
                # Preserve inherited category and modifier evidence. The veto is
                # a one-sided log-likelihood penalty, so a mismatched word can
                # only lower confidence and positive gate leakage cannot replace
                # the complete inherited score with a differently scaled logit.
                veto_penalty = veto_probability.clamp_min(epsilon).log()
                base_layers = (
                    identity_base_layers
                    + mismatch_gate_layers * veto_penalty
                )
            else:
                joint_probability = (
                    category_logit[None].sigmoid() * veto_probability
                )
                joint_logit = torch.logit(
                    joint_probability.clamp(epsilon, 1.0 - epsilon)
                )
                base_layers = identity_base_layers + mismatch_gate_layers * (
                    joint_logit - identity_base_layers
                )
            base_layers = torch.where(
                modifier_valid[None, :, None],
                base_layers,
                category_logit[None].expand_as(base_layers),
            )
        else:
            raw_mismatch_gate_layers = torch.zeros_like(identity_base_layers)
            deployed_routing_gate_layers = raw_mismatch_gate_layers
            deployed_routing_residual_layers = torch.zeros_like(
                raw_mismatch_gate_layers
            )
            mismatch_gate_layers = deployed_routing_gate_layers
            base_layers = category_logit[None] + torch.where(
                modifier_valid[None, :, None],
                modifier_layers,
                torch.zeros_like(modifier_layers),
            )
        # Build the pool feature declared by the active contract. Full-text
        # contracts may read detached rank query/text state; V55 still keeps
        # its local candidate affine out of the sample-global score path.
        patch_dense_feature = self.feature_norm(self.patch_feature(patch_input))
        if (
            self.pool_feature_contract
            == CONFIDENCE_POOL_FEATURE_CONTRACT_SIGNED_RANK_QUERY
        ):
            if self.global_query_norm is None or self.global_query_trunk is None:
                raise RuntimeError("signed rank-query confidence pool is incomplete")
            bounded_reference = torch.where(
                torch.isfinite(reference_base_layers),
                reference_base_layers,
                torch.zeros_like(reference_base_layers),
            ).clamp(min=-20.0, max=20.0) / 20.0
            global_query_input = torch.cat(
                (
                    self.global_query_norm(query),
                    bounded_reference[..., None].to(dtype=query.dtype),
                ),
                dim=-1,
            )
            dense_feature = self.global_query_trunk(global_query_input)
            dense_feature = dense_feature + patch_dense_feature[None]
        elif (
            self.pool_feature_contract
            == CONFIDENCE_POOL_FEATURE_CONTRACT_TOKEN_CONDITIONED
        ):
            token_conditioned_query = self._token_conditioned_global_feature(
                query=query,
                text=text,
                rank_token=rank_token,
                token_residual=global_token_residual_layers,
                modifier_mask=modifier_mask,
            )
            bounded_reference = torch.where(
                torch.isfinite(reference_base_layers),
                reference_base_layers,
                torch.zeros_like(reference_base_layers),
            ).clamp(min=-20.0, max=20.0) / 20.0
            global_query_input = torch.cat(
                (
                    token_conditioned_query,
                    bounded_reference[..., None].to(dtype=query.dtype),
                ),
                dim=-1,
            )
            dense_feature = self.global_query_trunk(global_query_input)
            dense_feature = dense_feature + patch_dense_feature[None]
        elif (
            self.pool_feature_contract
            in {
                CONFIDENCE_POOL_FEATURE_CONTRACT_CROSS_ATTENTION,
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ABSOLUTE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_CALIBRATED,
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_NORMALIZED,
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC,
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_SET_ATTENTION,
                CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_REFERENCE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE,
            }
        ):
            cross_attention_query = self._cross_attention_global_feature(
                query=query,
                text=text,
                rank_token=rank_token,
                token_residual=global_token_residual_layers,
                modifier_mask=modifier_mask,
                phrase_mask=phrase_mask,
            )
            bounded_reference = torch.where(
                torch.isfinite(reference_base_layers),
                reference_base_layers,
                torch.zeros_like(reference_base_layers),
            ).clamp(min=-20.0, max=20.0) / 20.0
            global_query_input = torch.cat(
                (
                    cross_attention_query,
                    bounded_reference[..., None].to(dtype=query.dtype),
                ),
                dim=-1,
            )
            if self.global_query_trunk is None:
                raise RuntimeError("cross-attention confidence pool is incomplete")
            dense_feature = self.global_query_trunk(global_query_input)
            dense_feature = dense_feature + patch_dense_feature[None]
        else:
            dense_feature = patch_dense_feature[None].expand(
                int(query.shape[0]), -1, -1, -1
            )
        if (
            self.pool_feature_contract
            in {
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ABSOLUTE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_CALIBRATED,
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_NORMALIZED,
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC,
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_SET_ATTENTION,
                CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_REFERENCE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE,
            }
        ):
            if self.candidate_absolute_head is None:
                raise RuntimeError("candidate-absolute confidence head is incomplete")
            candidate_head_input = (
                dense_feature.detach()
                if self.head_gradient_contract
                == CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE
                else dense_feature
            )
            candidate_residual_layers = self.candidate_absolute_head(
                candidate_head_input
            ).squeeze(-1)
            if self.head_gradient_contract == (
                CONFIDENCE_HEAD_GRADIENT_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE
            ):
                # V53/V54 retain their historical frozen-rank carrier.
                base_layers = frozen_full_phrase_layers + candidate_residual_layers
            elif self.head_gradient_contract == (
                CONFIDENCE_HEAD_GRADIENT_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE
            ):
                # V55 local supervision owns only the independent candidate
                # confidence coordinate. Frozen rank remains an input feature
                # and query-ordering signal, never an additive confidence score.
                base_layers = candidate_residual_layers
            elif self.head_gradient_contract == (
                CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE
            ):
                # V56 retains the V55 local coordinate only as a frozen,
                # detached diagnostic. It cannot update or enter the deployed
                # representation and remains excluded from the global score.
                base_layers = candidate_residual_layers
            else:
                # Historical candidate heads learn an absolute score from zero.
                base_layers = candidate_residual_layers
            if (
                self.head_gradient_contract
                != CONFIDENCE_HEAD_GRADIENT_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE
                and self.pool_feature_contract
                in {
                    CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_CALIBRATED,
                    CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_NORMALIZED,
                    CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC,
                    CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_SET_ATTENTION,
                }
            ):
                patch_scale, veto_depth, _coverage_depth = (
                    self.candidate_calibration_depths()
                )
                patch_row_max = patch.float().masked_fill(~mask, -torch.inf).max(
                    dim=1
                ).values
                # Patch owns candidate admission and within-image category
                # evidence, not cross-sample confidence scale.  The first
                # monotone term removes learned patch-scale leakage; the
                # second prevents the absolute winner from bypassing its
                # trace-supervised changed-word mismatch.
                base_layers = (
                    base_layers
                    - patch_scale * patch_row_max[None, :, None]
                    - veto_depth * mismatch_gate_layers.detach()
                )
            base_layers = base_layers.float().masked_fill(
                ~mask[None], torch.finfo(torch.float32).min
            )
        return {
            "token_layers": token_layers,
            "token_residual_layers": token_residual_layers,
            "hidden_layers": dense_feature,
            "base_layers": base_layers,
            "reference_base_layers": reference_base_layers,
            "reference_carrier_index_layers": (
                reference_carrier_index_layers
            ),
            "raw_mismatch_gate_layers": raw_mismatch_gate_layers,
            "deployed_routing_gate_layers": deployed_routing_gate_layers,
            "deployed_routing_residual_layers": (
                deployed_routing_residual_layers
            ),
            "mismatch_gate_layers": mismatch_gate_layers,
            "modifier_valid": modifier_valid,
        }


class DenseExpressionTower(nn.Module):
    """Private full expression encoder and fixed-reference GDINO decoder."""

    def __init__(
        self,
        source_feat_map: nn.Module,
        source_encoder: nn.Module,
        source_decoder: TransformerDecoder,
        source_level_embed: Optional[Tensor],
        *,
        max_text_len: int,
    ) -> None:
        super().__init__()
        if int(max_text_len) <= 0:
            raise ValueError("max_text_len must be positive")
        self.max_text_len = int(max_text_len)
        self.hidden_dim = int(getattr(source_decoder, "d_model", 0))
        if self.hidden_dim <= 0:
            raise ValueError("source decoder must expose a positive d_model")

        self.feat_map = copy.deepcopy(source_feat_map)
        self.encoder = copy.deepcopy(source_encoder)
        self.decoder = copy.deepcopy(source_decoder)
        self.decoder.bbox_embed = None
        self.decoder.class_embed = None
        self.num_layers = int(getattr(self.decoder, "num_layers", 0))
        if self.num_layers <= 0 or len(getattr(self.decoder, "layers", ())) != self.num_layers:
            raise ValueError("dense-duty scorer requires every source decoder layer")
        if source_level_embed is None:
            self.register_parameter("level_embed", None)
        else:
            if not torch.is_tensor(source_level_embed) or source_level_embed.dim() != 2:
                raise ValueError("source_level_embed must be a (levels,D) tensor")
            if int(source_level_embed.shape[-1]) != self.hidden_dim:
                raise ValueError("source_level_embed hidden dimension is incompatible")
            self.level_embed = nn.Parameter(
                source_level_embed.detach().clone(), requires_grad=False
            )

        self.token_head = ContrastiveEmbed(max_text_len=self.max_text_len)
        self._owned_parameter_ids: set[int] = set()
        self._refresh_parameter_contract()
        self.set_active(False)

    def _refresh_parameter_contract(self) -> None:
        visual_layers = getattr(self.encoder, "layers", None)
        for parameter in _module_parameters(visual_layers):
            parameter.requires_grad_(False)
        if self.level_embed is not None:
            self.level_embed.requires_grad_(False)
        owned = []
        owned.extend(self.feat_map.parameters())
        owned.extend(_module_parameters(getattr(self.encoder, "fusion_layers", None)))
        owned.extend(_module_parameters(getattr(self.encoder, "text_layers", None)))
        owned.extend(self.decoder.parameters())
        self._owned_parameter_ids = {id(parameter) for parameter in owned}
        if not self._owned_parameter_ids:
            raise RuntimeError("dense expression tower has no owned trainable surface")

    def owned_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(
            parameter
            for parameter in self.parameters()
            if id(parameter) in self._owned_parameter_ids
        )

    def set_active(self, active: bool) -> None:
        active = bool(active)
        nn.Module.train(self, active)
        for parameter in self.parameters():
            parameter.requires_grad_(id(parameter) in self._owned_parameter_ids and active)
        self.feat_map.train(active)
        _set_module_trainable(getattr(self.encoder, "fusion_layers", None), active)
        _set_module_trainable(getattr(self.encoder, "text_layers", None), active)
        self.decoder.train(active)
        visual_layers = getattr(self.encoder, "layers", None)
        if isinstance(visual_layers, nn.Module):
            visual_layers.eval()
            for parameter in visual_layers.parameters():
                parameter.requires_grad_(False)
        if self.level_embed is not None:
            self.level_embed.requires_grad_(False)

    @torch.no_grad()
    def load_from_components(
        self,
        feat_map: nn.Module,
        encoder: nn.Module,
        decoder: TransformerDecoder,
        level_embed: Optional[Tensor],
    ) -> None:
        source_num_layers = int(getattr(decoder, "num_layers", 0))
        if source_num_layers != self.num_layers:
            raise ValueError(
                f"source decoder has {source_num_layers} layers, expected {self.num_layers}"
            )
        self.feat_map.load_state_dict(feat_map.state_dict(), strict=True)
        self.encoder.load_state_dict(encoder.state_dict(), strict=True)
        decoder_state = copy.deepcopy(decoder)
        decoder_state.bbox_embed = None
        decoder_state.class_embed = None
        self.decoder.load_state_dict(decoder_state.state_dict(), strict=True)
        if self.level_embed is None:
            if level_embed is not None:
                raise ValueError("source supplies level_embed but tower does not own one")
        else:
            if level_embed is None or tuple(level_embed.shape) != tuple(self.level_embed.shape):
                raise ValueError("source level_embed is missing or shape-incompatible")
            self.level_embed.copy_(level_embed.detach())
        self.decoder.bbox_embed = None
        self.decoder.class_embed = None
        self._refresh_parameter_contract()

    @staticmethod
    def _required_context(context: Mapping[str, Any]) -> None:
        required = {
            "bert_hidden",
            "text_token_mask",
            "position_ids",
            "text_self_attention_masks",
            "phrase_token_mask",
            "srcs",
            "masks",
            "poss",
        }
        missing = sorted(required.difference(context))
        if missing:
            raise KeyError(f"raw context provider omitted required keys: {missing}")

    def _build_context(
        self,
        context: Mapping[str, Any],
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[dict[str, Tensor], Tensor]:
        self._required_context(context)
        bert_hidden = context["bert_hidden"]
        text_token_mask = context["text_token_mask"]
        position_ids = context["position_ids"]
        text_self_attention_masks = context["text_self_attention_masks"]
        phrase_token_mask = context["phrase_token_mask"]
        srcs = context["srcs"]
        masks = context["masks"]
        poss = context["poss"]
        if not torch.is_tensor(bert_hidden) or bert_hidden.dim() != 3:
            raise ValueError("bert_hidden must have shape (M,T,Cbert)")
        if int(bert_hidden.shape[0]) != int(batch_size) or bert_hidden.device != device:
            raise ValueError("bert_hidden batch/device does not align with candidates")
        if not torch.is_tensor(text_token_mask) or tuple(text_token_mask.shape) != tuple(
            bert_hidden.shape[:2]
        ):
            raise ValueError("text_token_mask must align with bert_hidden")
        if not torch.is_tensor(position_ids) or tuple(position_ids.shape) != tuple(
            bert_hidden.shape[:2]
        ):
            raise ValueError("position_ids must align with bert_hidden")
        expected_self_mask = (
            int(batch_size),
            int(bert_hidden.shape[1]),
            int(bert_hidden.shape[1]),
        )
        if not torch.is_tensor(text_self_attention_masks) or tuple(
            text_self_attention_masks.shape
        ) != expected_self_mask:
            raise ValueError("text_self_attention_masks must have shape (M,T,T)")
        if not torch.is_tensor(phrase_token_mask) or tuple(phrase_token_mask.shape) != tuple(
            bert_hidden.shape[:2]
        ):
            raise ValueError("phrase_token_mask must align with bert_hidden")
        if len(srcs) == 0 or len(srcs) != len(masks) or len(srcs) != len(poss):
            raise ValueError("raw image srcs/masks/poss must be non-empty and aligned")
        if self.level_embed is not None and len(srcs) > int(self.level_embed.shape[0]):
            raise ValueError("raw context has more feature levels than level_embed")

        encoded_text = self.feat_map(bert_hidden.detach())
        src_flatten = []
        mask_flatten = []
        pos_flatten = []
        selected_masks = []
        spatial_shapes = []
        for level, (src, mask, pos) in enumerate(zip(srcs, masks, poss)):
            if not all(torch.is_tensor(value) for value in (src, mask, pos)):
                raise TypeError("srcs, masks, and poss must contain tensors")
            if src.dim() != 4 or int(src.shape[0]) != batch_size:
                raise ValueError("every src must have shape (M,D,H,W)")
            if int(src.shape[1]) != self.hidden_dim:
                raise ValueError("raw src hidden dimension is incompatible")
            if tuple(mask.shape) != (
                batch_size,
                int(src.shape[-2]),
                int(src.shape[-1]),
            ):
                raise ValueError("raw mask must align with src")
            if tuple(pos.shape) != tuple(src.shape):
                raise ValueError("raw positional embedding must align with src")
            src = src.detach()
            mask = mask.detach().to(dtype=torch.bool)
            pos = pos.detach()
            spatial_shapes.append((int(src.shape[-2]), int(src.shape[-1])))
            src_flatten.append(src.flatten(2).transpose(1, 2))
            mask_flatten.append(mask.flatten(1))
            flat_pos = pos.flatten(2).transpose(1, 2)
            if self.level_embed is not None:
                flat_pos = flat_pos + self.level_embed[level].view(1, 1, -1)
            pos_flatten.append(flat_pos)
            selected_masks.append(mask)

        src_flat = torch.cat(src_flatten, dim=1)
        mask_flat = torch.cat(mask_flatten, dim=1)
        pos_flat = torch.cat(pos_flatten, dim=1)
        spatial = torch.as_tensor(spatial_shapes, dtype=torch.long, device=device)
        level_start = torch.cat(
            (spatial.new_zeros((1,)), spatial.prod(1).cumsum(0)[:-1])
        )

        valid_ratios = context.get("valid_ratios")
        if valid_ratios is None:
            get_valid_ratio = getattr(context.get("transformer"), "get_valid_ratio", None)
            if not callable(get_valid_ratio):
                # GroundingDINO padding occupies the lower/right rectangle.
                ratios = []
                for mask in selected_masks:
                    height, width = int(mask.shape[1]), int(mask.shape[2])
                    valid_height = (~mask[:, :, 0]).sum(dim=1).float() / float(height)
                    valid_width = (~mask[:, 0, :]).sum(dim=1).float() / float(width)
                    ratios.append(torch.stack((valid_width, valid_height), dim=-1))
                valid_ratios = torch.stack(ratios, dim=1)
            else:
                valid_ratios = torch.stack(
                    [get_valid_ratio(mask) for mask in selected_masks], dim=1
                )
        if tuple(valid_ratios.shape) != (batch_size, len(srcs), 2):
            raise ValueError("valid_ratios must have shape (M,F,2)")

        memory, memory_text = self.encoder(
            src_flat,
            pos=pos_flat,
            level_start_index=level_start,
            spatial_shapes=spatial,
            valid_ratios=valid_ratios.detach(),
            key_padding_mask=mask_flat,
            memory_text=encoded_text,
            text_attention_mask=~text_token_mask.detach().to(dtype=torch.bool),
            position_ids=position_ids.detach(),
            text_self_attention_masks=text_self_attention_masks.detach(),
        )
        return {
            "memory": memory,
            "memory_key_padding_mask": mask_flat,
            "memory_pos": pos_flat,
            "level_start_index": level_start,
            "spatial_shapes": spatial,
            "valid_ratios": valid_ratios.detach(),
            "encoded_text": memory_text,
            "text_token_mask": text_token_mask.detach().to(dtype=torch.bool),
        }, phrase_token_mask.detach().to(device=device, dtype=torch.bool)

    def _pad_mask(self, mask: Tensor) -> Tensor:
        result = torch.zeros(
            (int(mask.shape[0]), self.max_text_len),
            dtype=torch.bool,
            device=mask.device,
        )
        width = min(int(mask.shape[1]), self.max_text_len)
        if width > 0:
            result[:, :width] = mask[:, :width]
        return result

    def _pad_text_features(self, features: Tensor) -> Tensor:
        if features.dim() != 3 or int(features.shape[-1]) != self.hidden_dim:
            raise ValueError("text features must have shape (M,T,D)")
        result = features.new_zeros(
            (int(features.shape[0]), self.max_text_len, self.hidden_dim)
        )
        width = min(int(features.shape[1]), self.max_text_len)
        if width > 0:
            result[:, :width] = features[:, :width]
        return result

    @staticmethod
    def aggregate_phrase_logits(token_logits: Tensor, phrase_mask: Tensor) -> Tensor:
        token_float = token_logits.float()
        finite = torch.where(
            torch.isfinite(token_float),
            token_float,
            torch.full_like(token_float, -20.0),
        )
        weight = phrase_mask[:, None].to(dtype=finite.dtype)
        denominator = weight.sum(dim=-1).clamp_min(1.0)
        probability = (finite.sigmoid() * weight).sum(dim=-1) / denominator
        epsilon = max(float(torch.finfo(finite.dtype).eps), 1e-6)
        result = torch.logit(probability.clamp(epsilon, 1.0 - epsilon))
        return result.masked_fill(
            ~phrase_mask.any(dim=-1)[:, None], torch.finfo(result.dtype).min
        )

    def forward(
        self,
        *,
        candidate_hs: Tensor,
        candidate_boxes: Tensor,
        captions: List[str],
        owner_indices: Tensor,
        raw_context_provider: RawContextProvider,
        score_token_mask: Optional[Tensor],
    ) -> dict[str, Tensor]:
        context_raw = raw_context_provider(captions, owner_indices)
        if not isinstance(context_raw, Mapping):
            raise TypeError("raw_context_provider must return a mapping")
        context, phrase_mask = self._build_context(
            context_raw,
            batch_size=len(captions),
            device=candidate_hs.device,
        )
        chunk_hs = candidate_hs.index_select(0, owner_indices).detach()
        chunk_boxes = candidate_boxes.index_select(0, owner_indices).detach()
        decoded_layers, references = self.decoder.forward_fixed_external(
            tgt=chunk_hs,
            reference_boxes=chunk_boxes,
            memory=context["memory"],
            memory_key_padding_mask=context["memory_key_padding_mask"],
            memory_pos=context["memory_pos"],
            level_start_index=context["level_start_index"],
            spatial_shapes=context["spatial_shapes"],
            valid_ratios=context["valid_ratios"],
            memory_text=context["encoded_text"],
            text_attention_mask=~context["text_token_mask"],
        )
        if len(decoded_layers) != self.num_layers or len(references) != 1:
            raise RuntimeError("dense-duty decoder changed its fixed-reference contract")
        text_dict = {
            "encoded_text": context["encoded_text"],
            "text_token_mask": context["text_token_mask"],
        }
        token_layers = torch.stack(
            [self.token_head(hidden, text_dict) for hidden in decoded_layers], dim=0
        )
        padded_phrase = self._pad_mask(phrase_mask)
        if score_token_mask is None:
            effective_score = padded_phrase
        else:
            if score_token_mask.dim() != 2 or int(score_token_mask.shape[0]) != len(
                captions
            ):
                raise ValueError("score_token_mask must have shape (M,Tmax)")
            effective_score = self._pad_mask(score_token_mask.to(dtype=torch.bool))
            effective_score &= padded_phrase
        phrase_layers = torch.stack(
            [
                self.aggregate_phrase_logits(token_layers[layer], effective_score)
                for layer in range(self.num_layers)
            ],
            dim=0,
        )
        full_phrase_layers = torch.stack(
            [
                self.aggregate_phrase_logits(token_layers[layer], padded_phrase)
                for layer in range(self.num_layers)
            ],
            dim=0,
        )
        return {
            "hidden_layers": torch.stack(list(decoded_layers), dim=0),
            "token_layers": token_layers,
            "phrase_layers": phrase_layers,
            "full_phrase_layers": full_phrase_layers,
            "text_features": self._pad_text_features(context["encoded_text"]),
            "score_token_mask": effective_score,
            "phrase_token_mask": padded_phrase,
            "modifier_valid": effective_score.any(dim=-1),
        }


class StageBDenseDutyScorer(nn.Module):
    """Patch-admitted rank tower plus stop-gradient confidence adapter."""

    is_dense_duty = True
    warmstart_components = (
        "feat_map",
        "transformer.encoder",
        "transformer.decoder",
        "transformer.level_embed",
    )

    def __init__(
        self,
        source_feat_map: nn.Module,
        source_encoder: nn.Module,
        source_decoder: TransformerDecoder,
        source_level_embed: Optional[Tensor],
        *,
        max_text_len: int = 256,
        candidate_topk: int = 50,
        category_gate_max_gap: float = 3.0,
        patch_score_clip: float = 5.0,
        confidence_adapter_dim: int = 64,
        confidence_init_seed: int = 42,
        confidence_hidden_dim: int = 256,
        confidence_pool_temperature: float = 0.2,
        confidence_pool_topk: int = 10,
        confidence_phrase_aggregation: str = CONFIDENCE_PHRASE_AGGREGATION_LEGACY,
        confidence_word_softmin_temperature: float = 0.1,
        confidence_veto_gate_scale: float = 1.0,
        confidence_veto_gate_offset: float = 0.0,
        confidence_veto_coverage_offset: float = 0.1,
        confidence_veto_coverage_ramp: float = 0.8,
        confidence_veto_cap_temperature: float = 0.1,
        confidence_veto_cap_initial_ceiling: float = -0.1,
        confidence_rank_evidence_contract: str = (
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_OFF
        ),
        confidence_pool_feature_contract: str = CONFIDENCE_POOL_FEATURE_CONTRACT,
        confidence_residual_parameterization_gain: float = 1.0,
        confidence_gate_gradient_contract: str = (
            CONFIDENCE_GATE_GRADIENT_CONTRACT_HARD_DETACHED
        ),
        confidence_head_gradient_contract: str = (
            CONFIDENCE_HEAD_GRADIENT_CONTRACT_SHARED
        ),
        expression_microbatch: int = 1,
        phase: str = "rank",
    ) -> None:
        super().__init__()
        if int(candidate_topk) <= 0:
            raise ValueError("candidate_topk must be positive")
        if not math.isfinite(float(category_gate_max_gap)) or float(
            category_gate_max_gap
        ) < 0:
            raise ValueError("category_gate_max_gap must be finite and non-negative")
        if not math.isfinite(float(patch_score_clip)) or float(patch_score_clip) <= 0:
            raise ValueError("patch_score_clip must be finite and positive")
        if int(expression_microbatch) <= 0:
            raise ValueError("expression_microbatch must be positive")
        if isinstance(confidence_init_seed, bool) or int(confidence_init_seed) < 0:
            raise ValueError("confidence_init_seed must be a non-negative integer")
        self.max_text_len = int(max_text_len)
        self.candidate_topk = int(candidate_topk)
        self.expression_microbatch = int(expression_microbatch)
        self.category_gate_max_gap = float(category_gate_max_gap)
        self.patch_score_clip = float(patch_score_clip)
        self.rank_tower = DenseExpressionTower(
            source_feat_map,
            source_encoder,
            source_decoder,
            source_level_embed,
            max_text_len=self.max_text_len,
        )
        self.confidence_init_seed = int(confidence_init_seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.confidence_init_seed)
            self.confidence_adapter = TokenAwareConfidenceAdapter(
                self.rank_tower.hidden_dim,
                adapter_dim=int(confidence_adapter_dim),
                max_text_len=self.max_text_len,
                patch_hidden_dim=int(confidence_adapter_dim),
                score_topk=int(confidence_pool_topk),
                patch_score_clip=self.patch_score_clip,
                phrase_aggregation=confidence_phrase_aggregation,
                word_softmin_temperature=confidence_word_softmin_temperature,
                veto_gate_scale=confidence_veto_gate_scale,
                veto_gate_offset=confidence_veto_gate_offset,
                veto_coverage_offset=confidence_veto_coverage_offset,
                veto_coverage_ramp=confidence_veto_coverage_ramp,
                veto_cap_temperature=confidence_veto_cap_temperature,
                veto_cap_initial_ceiling=confidence_veto_cap_initial_ceiling,
                rank_evidence_contract=confidence_rank_evidence_contract,
                pool_feature_contract=confidence_pool_feature_contract,
                residual_parameterization_gain=(
                    confidence_residual_parameterization_gain
                ),
                gate_gradient_contract=confidence_gate_gradient_contract,
                head_gradient_contract=confidence_head_gradient_contract,
            )
            self.confidence_pool = AbsoluteConfidencePool(
                self.rank_tower.hidden_dim,
                pool_hidden_dim=int(confidence_hidden_dim),
                score_topk=int(confidence_pool_topk),
                pool_temperature=float(confidence_pool_temperature),
                set_attention=(
                    confidence_pool_feature_contract
                    == CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_SET_ATTENTION
                ),
            )
            self.confidence_veto_pool = (
                AbsoluteConfidencePool(
                    self.rank_tower.hidden_dim,
                    pool_hidden_dim=int(confidence_hidden_dim),
                    score_topk=int(confidence_pool_topk),
                    pool_temperature=float(confidence_pool_temperature),
                    set_attention=False,
                )
                if confidence_head_gradient_contract
                == CONFIDENCE_HEAD_GRADIENT_CONTRACT_GLOBAL_TRUST_VETO
                else None
            )
        self.num_layers = self.rank_tower.num_layers
        self.register_buffer(
            "_dense_duty_contract_version",
            torch.tensor(DENSE_DUTY_CONTRACT_VERSION, dtype=torch.int64),
            persistent=True,
        )
        self.phase = normalize_dense_duty_phase(phase)
        self._assert_parameter_disjointness()
        self._apply_phase_contract()

    def _assert_parameter_disjointness(self) -> None:
        rank_ids = {id(parameter) for parameter in self.rank_parameters()}
        confidence_ids = {id(parameter) for parameter in self.confidence_parameters()}
        diagnostic_ids = {
            id(parameter) for parameter in self.candidate_diagnostic_parameters()
        }
        deployment_owned = (
            self.confidence_adapter.head_gradient_contract
            == CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE
        )
        overlap = rank_ids & confidence_ids
        if not rank_ids or not confidence_ids or overlap:
            raise RuntimeError(
                "dense-duty rank/confidence parameter ownership is empty or overlapping"
            )
        if deployment_owned and (
            not diagnostic_ids
            or bool(diagnostic_ids & (rank_ids | confidence_ids))
            or any(
                parameter.requires_grad
                for parameter in self.candidate_diagnostic_parameters()
            )
        ):
            raise RuntimeError(
                "deployment-owned confidence requires a complete frozen diagnostic "
                "candidate head outside every active parameter owner"
            )
        token_ids = {id(parameter) for parameter in self.token_veto_parameters()}
        router_ids = {
            id(parameter) for parameter in self.deployed_router_parameters()
        }
        absolute_ids = {
            id(parameter) for parameter in self.global_absolute_parameters()
        }
        deployed_router_enabled = (
            self.confidence_adapter.head_gradient_contract
            == CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYED_ROUTER
        )
        candidate_sample_enabled = (
            self.confidence_adapter.head_gradient_contract
            == CONFIDENCE_HEAD_GRADIENT_CONTRACT_CANDIDATE_SAMPLE
        )
        if (
            not token_ids
            or not absolute_ids
            or token_ids & absolute_ids
            or token_ids & router_ids
            or router_ids & absolute_ids
            or token_ids | router_ids | absolute_ids != confidence_ids
            or (deployed_router_enabled and not router_ids)
            or (not deployed_router_enabled and bool(router_ids))
        ):
            raise RuntimeError(
                "dense-duty token-veto/deployed-router/global-absolute ownership "
                "is empty, overlapping, or incomplete"
            )
        if candidate_sample_enabled:
            candidate_ids = {
                id(parameter) for parameter in self.candidate_absolute_parameters()
            }
            sample_ids = {
                id(parameter) for parameter in self.sample_calibrator_parameters()
            }
            if (
                not candidate_ids
                or not sample_ids
                or candidate_ids & sample_ids
                or candidate_ids | sample_ids != absolute_ids
            ):
                raise RuntimeError(
                    "dense-duty candidate-absolute/sample-calibrator ownership is "
                    "empty, overlapping, or incomplete"
                )
        if (
            self.confidence_adapter.head_gradient_contract
            == CONFIDENCE_HEAD_GRADIENT_CONTRACT_GLOBAL_TRUST_VETO
        ):
            trust_ids = {
                id(parameter) for parameter in self.global_trust_parameters()
            }
            veto_ids = {
                id(parameter) for parameter in self.global_veto_parameters()
            }
            if (
                not trust_ids
                or not veto_ids
                or trust_ids & veto_ids
                or trust_ids | veto_ids != absolute_ids
            ):
                raise RuntimeError(
                    "dense-duty global trust/veto ownership is empty, "
                    "overlapping, or incomplete"
                )

    def rank_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.rank_tower.owned_parameters()

    def confidence_parameters(self) -> tuple[nn.Parameter, ...]:
        if (
            self.confidence_adapter.head_gradient_contract
            == CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE
        ):
            return self.token_veto_parameters() + self.global_absolute_parameters()
        return (
            tuple(self.confidence_adapter.parameters())
            + tuple(self.confidence_pool.parameters())
            + (
                tuple(self.confidence_veto_pool.parameters())
                if self.confidence_veto_pool is not None
                else ()
            )
        )

    def token_veto_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.confidence_adapter.token_veto_parameters()

    def deployed_router_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.confidence_adapter.deployed_router_parameters()

    def candidate_absolute_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.confidence_adapter.candidate_absolute_parameters()

    def candidate_diagnostic_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.confidence_adapter.candidate_diagnostic_parameters()

    def deployed_global_trunk_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.confidence_adapter.deployed_global_trunk_parameters()

    def sample_calibrator_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.confidence_adapter.sample_calibrator_parameters() + tuple(
            self.confidence_pool.parameters()
        )

    def expected_live_confidence_parameter_tensor_counts(self) -> dict[str, int]:
        """Return exact per-step gradient-tensor counts for sealed split heads."""
        head_contract = self.confidence_adapter.head_gradient_contract
        if head_contract in {
            CONFIDENCE_HEAD_GRADIENT_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE,
            CONFIDENCE_HEAD_GRADIENT_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE,
            CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE,
        }:
            return {
                "token_veto": len(self.token_veto_parameters()),
                "global_absolute": len(self.global_absolute_parameters()),
            }
        if head_contract != CONFIDENCE_HEAD_GRADIENT_CONTRACT_CANDIDATE_SAMPLE:
            raise RuntimeError("live tensor counts are unavailable for this head")
        adapter = self.confidence_adapter
        dormant_candidate = {
            id(parameter)
            for parameter in adapter._module_parameters(
                adapter.patch_residual,
                adapter.global_query_norm,
            )
        }
        if adapter.veto_cap_raw_ceiling is not None:
            dormant_candidate.add(id(adapter.veto_cap_raw_ceiling))
        candidate = {
            id(parameter) for parameter in self.candidate_absolute_parameters()
        }
        if not dormant_candidate or not dormant_candidate.issubset(candidate):
            raise RuntimeError(
                "split V6 dormant-candidate tensor contract drifted"
            )
        return {
            "token_veto": len(self.token_veto_parameters()),
            "candidate_absolute": len(candidate - dormant_candidate),
            "sample_calibrator": len(self.sample_calibrator_parameters()),
        }

    def global_trust_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.confidence_adapter.global_absolute_parameters() + tuple(
            self.confidence_pool.parameters()
        )

    def global_veto_parameters(self) -> tuple[nn.Parameter, ...]:
        if self.confidence_veto_pool is None:
            return ()
        return tuple(self.confidence_veto_pool.parameters())

    def global_absolute_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.global_trust_parameters() + self.global_veto_parameters()

    def set_phase(self, phase: str) -> None:
        self.phase = normalize_dense_duty_phase(phase)
        self._apply_phase_contract()

    def _apply_phase_contract(self) -> None:
        rank_active = bool(self.training and self.phase == "rank")
        confidence_active = bool(self.training and self.phase == "confidence")
        self.rank_tower.set_active(rank_active)
        deployment_owned = (
            self.confidence_adapter.head_gradient_contract
            == CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE
        )
        if deployment_owned:
            _set_module_trainable(self.confidence_adapter, False)
            _set_module_trainable(self.confidence_pool, False)
            if self.confidence_veto_pool is not None:
                _set_module_trainable(self.confidence_veto_pool, False)
            if confidence_active:
                for parameter in self.confidence_parameters():
                    parameter.requires_grad_(True)
        else:
            _set_module_trainable(self.confidence_adapter, confidence_active)
            _set_module_trainable(self.confidence_pool, confidence_active)
            if self.confidence_veto_pool is not None:
                _set_module_trainable(self.confidence_veto_pool, confidence_active)

    def train(self, mode: bool = True):
        super().train(mode)
        self._apply_phase_contract()
        return self

    @torch.no_grad()
    def load_from_groundingdino(self, source_model: nn.Module) -> dict[str, Any]:
        transformer = getattr(source_model, "transformer", None)
        feat_map = getattr(source_model, "feat_map", None)
        if transformer is None or not isinstance(feat_map, nn.Module):
            raise ValueError("source model lacks GroundingDINO feat_map/transformer")
        encoder = getattr(transformer, "encoder", None)
        decoder = getattr(transformer, "decoder", None)
        if not isinstance(encoder, nn.Module) or not isinstance(decoder, nn.Module):
            raise ValueError("source GroundingDINO transformer is incomplete")
        level_embed = getattr(transformer, "level_embed", None)
        self.rank_tower.load_from_components(feat_map, encoder, decoder, level_embed)
        self._assert_parameter_disjointness()
        self._apply_phase_contract()
        return {
            "loaded_components": [
                "feat_map",
                "transformer.encoder",
                "transformer.decoder",
                "transformer.level_embed",
            ],
            "decoder_num_layers": self.rank_tower.num_layers,
            "confidence_adapter_initialized_independently": True,
            "confidence_token_contract": CONFIDENCE_TOKEN_CONTRACT,
            "confidence_pool_feature_contract": (
                self.confidence_adapter.pool_feature_contract
            ),
        }

    @staticmethod
    def _extract_component_state(
        source_state: Mapping[str, Any],
        *,
        prefix: str,
        expected: Mapping[str, Tensor],
        checkpoint_label: str,
        allowed_unexpected_prefixes: Sequence[str] = (),
    ) -> dict[str, Tensor]:
        root = prefix + "."
        provided = {
            str(key)[len(root) :]: value
            for key, value in source_state.items()
            if str(key).startswith(root)
        }
        missing = sorted(set(expected).difference(provided))
        unexpected = sorted(
            key
            for key in set(provided).difference(expected)
            if not any(
                key.startswith(prefix) for prefix in allowed_unexpected_prefixes
            )
        )
        mismatches = []
        mapped = {}
        for key in sorted(set(expected).intersection(provided)):
            value = provided[key]
            if not torch.is_tensor(value) or tuple(value.shape) != tuple(expected[key].shape):
                mismatches.append(
                    (
                        key,
                        tuple(expected[key].shape),
                        tuple(value.shape) if torch.is_tensor(value) else type(value).__name__,
                    )
                )
            else:
                mapped[key] = value
        if missing or unexpected or mismatches:
            raise ValueError(
                f"{checkpoint_label}: incompatible {prefix} "
                f"(missing={missing[:8]}, unexpected={unexpected[:8]}, "
                f"shape_mismatches={mismatches[:8]})"
            )
        return mapped

    @torch.no_grad()
    def load_from_full_text_checkpoint_state(
        self,
        source_state_dict: Mapping[str, Any],
        *,
        checkpoint_label: str,
        source_model_prefix: str = "",
    ) -> dict[str, Any]:
        if not isinstance(source_state_dict, Mapping):
            raise TypeError(f"{checkpoint_label}: source state must be a mapping")
        model_prefix = str(source_model_prefix).strip().rstrip(".")

        def key(name: str) -> str:
            return f"{model_prefix}.{name}" if model_prefix else name

        template = self.rank_tower
        feat_state = self._extract_component_state(
            source_state_dict,
            prefix=key("feat_map"),
            expected=template.feat_map.state_dict(),
            checkpoint_label=checkpoint_label,
        )
        encoder_state = self._extract_component_state(
            source_state_dict,
            prefix=key("transformer.encoder"),
            expected=template.encoder.state_dict(),
            checkpoint_label=checkpoint_label,
        )
        decoder_state = self._extract_component_state(
            source_state_dict,
            prefix=key("transformer.decoder"),
            expected=template.decoder.state_dict(),
            checkpoint_label=checkpoint_label,
            allowed_unexpected_prefixes=("bbox_embed.", "class_embed."),
        )
        level_key = key("transformer.level_embed")
        level_value = source_state_dict.get(level_key)
        if template.level_embed is None:
            if level_value is not None:
                raise ValueError(
                    f"{checkpoint_label}: unexpected {level_key} for a one-level tower"
                )
        elif not torch.is_tensor(level_value) or tuple(level_value.shape) != tuple(
            template.level_embed.shape
        ):
            raise ValueError(
                f"{checkpoint_label}: missing or shape-incompatible {level_key}"
            )

        self.rank_tower.feat_map.load_state_dict(feat_state, strict=True)
        self.rank_tower.encoder.load_state_dict(encoder_state, strict=True)
        self.rank_tower.decoder.load_state_dict(decoder_state, strict=True)
        if self.rank_tower.level_embed is not None:
            self.rank_tower.level_embed.copy_(level_value.detach())
        self.rank_tower.decoder.bbox_embed = None
        self.rank_tower.decoder.class_embed = None
        self.rank_tower._refresh_parameter_contract()
        self._assert_parameter_disjointness()
        self._apply_phase_contract()
        return {
            "source_decoder_num_layers": template.num_layers,
            "selected_source_layer_indices": list(range(template.num_layers)),
            "loaded_num_layers": template.num_layers,
            "loaded_components": list(self.warmstart_components),
            "loaded_tensor_count": (
                len(feat_state)
                + len(encoder_state)
                + len(decoder_state)
                + int(level_value is not None)
            ),
        }

    def _run_tower_microbatched(
        self,
        tower: DenseExpressionTower,
        *,
        candidate_hs: Tensor,
        candidate_boxes: Tensor,
        captions: List[str],
        owner_indices: Tensor,
        raw_context_provider: RawContextProvider,
        score_token_mask: Optional[Tensor],
        expression_microbatch: int,
    ) -> dict[str, Tensor]:
        chunk_size = int(expression_microbatch)
        if chunk_size <= 0:
            raise ValueError("expression_microbatch must be positive")
        chunks: List[dict[str, Tensor]] = []
        for start in range(0, len(captions), chunk_size):
            end = min(start + chunk_size, len(captions))
            chunks.append(
                tower(
                    candidate_hs=candidate_hs,
                    candidate_boxes=candidate_boxes,
                    captions=captions[start:end],
                    owner_indices=owner_indices[start:end],
                    raw_context_provider=raw_context_provider,
                    score_token_mask=(
                        score_token_mask[start:end]
                        if score_token_mask is not None
                        else None
                    ),
                )
            )
        if not chunks:
            raise RuntimeError("dense-duty scorer produced no expression chunks")
        layer_keys = (
            "hidden_layers",
            "token_layers",
            "phrase_layers",
            "full_phrase_layers",
        )
        row_keys = (
            "text_features",
            "score_token_mask",
            "phrase_token_mask",
            "modifier_valid",
        )
        return {
            **{
                name: torch.cat([chunk[name] for chunk in chunks], dim=1)
                for name in layer_keys
            },
            **{
                name: torch.cat([chunk[name] for chunk in chunks], dim=0)
                for name in row_keys
            },
        }

    @staticmethod
    def _normalize_captions(
        expression_captions: Sequence[Sequence[str]],
        *,
        batch_size: int,
    ) -> tuple[List[str], int]:
        if len(expression_captions) != batch_size or batch_size <= 0:
            raise ValueError("expression captions must align with the non-empty image batch")
        slot_count = len(expression_captions[0])
        if slot_count <= 0:
            raise ValueError("every image requires an expression slot")
        flattened = []
        for row in expression_captions:
            if len(row) != slot_count:
                raise ValueError("expression captions must be rectangular")
            for caption in row:
                if not isinstance(caption, str):
                    raise TypeError("every expression caption must be a string")
                flattened.append(caption if caption.strip() else "object .")
        return flattened, slot_count

    def _prepare_candidates(
        self,
        candidate_hs: Tensor,
        candidate_boxes: Tensor,
        candidate_indices: Tensor,
        candidate_patch_logits: Tensor,
    ) -> dict[str, Tensor]:
        if candidate_hs.dim() != 3:
            raise ValueError("candidate_hs must have shape (B,N,D)")
        if candidate_boxes.dim() != 3 or int(candidate_boxes.shape[-1]) != 4:
            raise ValueError("candidate_boxes must have shape (B,N,4)")
        if tuple(candidate_hs.shape[:2]) != tuple(candidate_boxes.shape[:2]):
            raise ValueError("candidate_hs and candidate_boxes must share (B,N)")
        if int(candidate_hs.shape[1]) <= 0 or int(candidate_hs.shape[1]) > self.candidate_topk:
            raise ValueError(
                "compact candidate count must be positive and no larger than candidate_topk"
            )
        indices = torch.as_tensor(candidate_indices, device=candidate_hs.device)
        if indices.dtype != torch.int64 or tuple(indices.shape) != tuple(
            candidate_hs.shape[:2]
        ):
            raise ValueError("candidate_indices must be int64 with shape (B,N)")
        if bool((indices < 0).any().item()):
            raise ValueError("candidate_indices must be non-negative")
        for row in indices.detach().cpu().tolist():
            if len(set(int(value) for value in row)) != len(row):
                raise ValueError("candidate_indices must be unique within each row")
        if candidate_patch_logits.dim() == 3 and int(candidate_patch_logits.shape[-1]) == 1:
            candidate_patch_logits = candidate_patch_logits[..., 0]
        if tuple(candidate_patch_logits.shape) != tuple(candidate_hs.shape[:2]):
            raise ValueError("candidate_patch_logits must align with candidate_hs")
        patch = candidate_patch_logits.detach().float()
        if not bool(torch.isfinite(patch).all().item()):
            raise ValueError("candidate_patch_logits must be finite")
        all_candidates = torch.ones_like(patch, dtype=torch.bool)
        gap_eligible, standardized = data_driven_category_gate_mask(
            patch,
            all_candidates,
            max_gap=self.category_gate_max_gap,
            clip=self.patch_score_clip,
        )
        return {
            "indices": indices.detach(),
            "hs": candidate_hs.detach(),
            "boxes": candidate_boxes.detach(),
            "patch_logits": patch,
            "patch_standardized": standardized.detach(),
            "eligible": gap_eligible.detach(),
        }

    @staticmethod
    def _reshape_tower_output(
        value: Tensor,
        *,
        batch_size: int,
        slot_count: int,
    ) -> Tensor:
        if value.dim() == 4:  # L, M, N, D/T
            return (
                value.view(value.shape[0], batch_size, slot_count, *value.shape[2:])
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )
        if value.dim() == 3:  # L, M, N
            return (
                value.view(value.shape[0], batch_size, slot_count, value.shape[2])
                .permute(0, 1, 3, 2)
                .contiguous()
            )
        raise ValueError(f"unsupported dense tower output shape {tuple(value.shape)}")

    def forward(
        self,
        *,
        candidate_hs: Tensor,
        candidate_boxes: Tensor,
        candidate_indices: Tensor,
        candidate_patch_logits: Tensor,
        expression_captions: Sequence[Sequence[str]],
        expression_valid_mask: Tensor,
        raw_context_provider: RawContextProvider,
        expression_score_token_mask: Optional[Tensor] = None,
        expression_score_word_group_ids: Optional[Tensor] = None,
        expression_predicate_token_mask: Optional[Tensor] = None,
        expression_microbatch: Optional[int] = None,
    ) -> dict[str, Tensor]:
        if not callable(raw_context_provider):
            raise TypeError("raw_context_provider must be callable")
        batch_size = int(candidate_hs.shape[0])
        flat_captions, slot_count = self._normalize_captions(
            expression_captions, batch_size=batch_size
        )
        valid = torch.as_tensor(
            expression_valid_mask, device=candidate_hs.device, dtype=torch.bool
        )
        if tuple(valid.shape) != (batch_size, slot_count):
            raise ValueError("expression_valid_mask must have shape (B,K)")
        flat_score_mask = None
        if expression_score_token_mask is None:
            raise ValueError(
                "dense-duty scoring requires a noncanonical score-token mask; "
                "an empty row delegates category-only ranking to patch scores"
            )
        score_mask = torch.as_tensor(
            expression_score_token_mask,
            device=candidate_hs.device,
            dtype=torch.bool,
        )
        if tuple(score_mask.shape[:2]) != (batch_size, slot_count) or score_mask.dim() != 3:
            raise ValueError("expression_score_token_mask must have shape (B,K,T)")
        flat_score_mask = score_mask.reshape(batch_size * slot_count, -1)
        flat_score_word_groups = None
        if expression_score_word_group_ids is not None:
            score_word_groups = torch.as_tensor(
                expression_score_word_group_ids,
                device=candidate_hs.device,
                dtype=torch.long,
            )
            if score_word_groups.dim() != 3 or tuple(
                score_word_groups.shape[:2]
            ) != (batch_size, slot_count):
                raise ValueError(
                    "expression_score_word_group_ids must have shape (B,K,T)"
                )
            if int(score_word_groups.shape[-1]) != int(score_mask.shape[-1]):
                raise ValueError(
                    "expression score word groups must align with score-token mask"
                )
            flat_score_word_groups = score_word_groups.reshape(
                batch_size * slot_count, -1
            )
        flat_predicate_mask = None
        if expression_predicate_token_mask is not None:
            predicate_mask = torch.as_tensor(
                expression_predicate_token_mask,
                device=candidate_hs.device,
                dtype=torch.bool,
            )
            if predicate_mask.dim() != 3 or tuple(predicate_mask.shape[:2]) != (
                batch_size,
                slot_count,
            ):
                raise ValueError(
                    "expression_predicate_token_mask must have shape (B,K,T)"
                )
            flat_predicate_mask = predicate_mask.reshape(
                batch_size * slot_count, -1
            )

        candidates = self._prepare_candidates(
            candidate_hs,
            candidate_boxes,
            candidate_indices,
            candidate_patch_logits,
        )
        candidate_count = int(candidates["indices"].shape[1])
        owner_indices = torch.arange(
            batch_size, device=candidate_hs.device, dtype=torch.long
        ).repeat_interleave(slot_count)
        flat_eligible = candidates["eligible"].index_select(0, owner_indices)
        flat_patch_logits = candidates["patch_logits"].index_select(0, owner_indices)
        flat_patch_standardized = candidates["patch_standardized"].index_select(
            0, owner_indices
        )
        flat_valid = valid.reshape(-1)

        output: dict[str, Tensor] = {
            "candidate_idx": candidates["indices"],
            "candidate_boxes": candidates["boxes"],
            "candidate_patch_logits": candidates["patch_logits"],
            "candidate_patch_standardized": candidates["patch_standardized"],
            "candidate_eligible_mask": (
                candidates["eligible"][:, :, None]
                & valid[:, None, :]
            ),
            "expression_valid_mask": valid,
        }

        microbatch = (
            self.expression_microbatch
            if expression_microbatch is None
            else int(expression_microbatch)
        )
        if microbatch <= 0:
            raise ValueError("expression_microbatch must be positive")

        # The full rank tower is the single source of multimodal features.  It
        # remains trainable only in rank phase and is evaluated once under
        # no_grad for confidence training and evaluation.
        run_confidence = not self.training or self.phase in {"confidence", "eval"}
        confidence: Optional[dict[str, Tensor]] = None
        rank_grad = bool(self.training and self.phase == "rank")
        with torch.set_grad_enabled(torch.is_grad_enabled() and rank_grad):
            rank = self._run_tower_microbatched(
                self.rank_tower,
                candidate_hs=candidates["hs"],
                candidate_boxes=candidates["boxes"],
                captions=flat_captions,
                owner_indices=owner_indices,
                raw_context_provider=raw_context_provider,
                score_token_mask=flat_score_mask,
                expression_microbatch=microbatch,
            )

        if run_confidence:
            confidence_grad = bool(self.training and self.phase == "confidence")
            with torch.set_grad_enabled(torch.is_grad_enabled() and confidence_grad):
                confidence = self.confidence_adapter(
                    rank_token_layers=rank["token_layers"],
                    query_layers=rank["hidden_layers"],
                    text_features=rank["text_features"],
                    phrase_token_mask=rank["phrase_token_mask"],
                    score_token_mask=rank["score_token_mask"],
                    patch_logits=flat_patch_logits,
                    patch_standardized=flat_patch_standardized,
                    candidate_mask=flat_eligible,
                    score_word_group_ids=flat_score_word_groups,
                )

        rank_placeholder = False
        confidence_placeholder = confidence is None

        rank_token_flat = rank["token_layers"]
        rank_phrase_flat = rank["phrase_layers"]
        rank_full_phrase_flat = rank["full_phrase_layers"].detach().float()
        rank_modifier_valid = rank["modifier_valid"]
        rank_score_token_mask = rank["score_token_mask"]
        rank_phrase_token_mask = rank["phrase_token_mask"]
        fallback = flat_patch_standardized[None].expand(
            int(rank_phrase_flat.shape[0]), -1, -1
        )
        layer_phrase_flat = torch.where(
            rank_modifier_valid[None, :, None], rank_phrase_flat, fallback
        )
        layer_phrase_flat = layer_phrase_flat.masked_fill(
            ~flat_eligible[None], torch.finfo(layer_phrase_flat.dtype).min
        )
        layer_phrase_flat = layer_phrase_flat.masked_fill(
            ~flat_valid[None, :, None], torch.finfo(layer_phrase_flat.dtype).min
        )
        layer_token = self._reshape_tower_output(
            rank_token_flat, batch_size=batch_size, slot_count=slot_count
        )
        layer_token = layer_token.masked_fill(
            ~valid[None, :, None, :, None], torch.finfo(layer_token.dtype).min
        )
        layer_phrase = self._reshape_tower_output(
            layer_phrase_flat, batch_size=batch_size, slot_count=slot_count
        )
        final_phrase = layer_phrase[-1]
        frozen_rank_full_global_flat = rank_full_phrase_flat.masked_fill(
            ~flat_eligible[None], -torch.inf
        ).max(dim=-1).values
        frozen_rank_full_global = frozen_rank_full_global_flat.view(
            int(rank_full_phrase_flat.shape[0]), batch_size, slot_count
        ).masked_fill(
            ~valid[None], torch.finfo(rank_full_phrase_flat.dtype).min
        )

        if confidence_placeholder:
            confidence_token_flat = rank["token_layers"].detach()
            confidence_token_residual_flat = torch.zeros_like(
                confidence_token_flat
            )
            confidence_base_flat = rank["full_phrase_layers"].detach()
            confidence_reference_base_flat = confidence_base_flat
            confidence_hidden_flat = rank["hidden_layers"].detach()
            confidence_mismatch_gate_flat = torch.zeros_like(confidence_base_flat)
            confidence_raw_mismatch_gate_flat = confidence_mismatch_gate_flat
            confidence_deployed_routing_gate_flat = confidence_mismatch_gate_flat
            confidence_deployed_routing_residual_flat = torch.zeros_like(
                confidence_mismatch_gate_flat
            )
            confidence_reference_carrier_index_flat = torch.stack(
                [
                    _frozen_reference_carrier_index(
                        confidence_reference_base_flat[layer], flat_eligible
                    )
                    for layer in range(
                        int(confidence_reference_base_flat.shape[0])
                    )
                ],
                dim=0,
            )
        else:
            if confidence is None:
                raise AssertionError("confidence adapter output is unavailable")
            confidence_token_flat = confidence["token_layers"]
            confidence_token_residual_flat = confidence["token_residual_layers"]
            confidence_base_flat = confidence["base_layers"]
            confidence_reference_base_flat = confidence["reference_base_layers"]
            confidence_hidden_flat = confidence["hidden_layers"]
            confidence_mismatch_gate_flat = confidence["mismatch_gate_layers"]
            confidence_raw_mismatch_gate_flat = confidence[
                "raw_mismatch_gate_layers"
            ]
            confidence_deployed_routing_gate_flat = confidence[
                "deployed_routing_gate_layers"
            ]
            confidence_deployed_routing_residual_flat = confidence[
                "deployed_routing_residual_layers"
            ]
            confidence_reference_carrier_index_flat = confidence[
                "reference_carrier_index_layers"
            ]
        gated_pool_absolute_cap_enabled = (
            self.confidence_adapter.phrase_aggregation
            == CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
        )
        fulltext_global_absolute_enabled = (
            gated_pool_absolute_cap_enabled
            and self.confidence_adapter.head_gradient_contract
            in {
                CONFIDENCE_HEAD_GRADIENT_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE,
                CONFIDENCE_HEAD_GRADIENT_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE,
                CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE,
            }
            and self.confidence_adapter.gate_gradient_contract
            == CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT
            and self.confidence_adapter.pool_feature_contract
            in {
                CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_REFERENCE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE,
            }
        )
        independent_global_absolute_enabled = (
            fulltext_global_absolute_enabled
            and (
                (
                    self.confidence_adapter.head_gradient_contract
                    == CONFIDENCE_HEAD_GRADIENT_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE
                    and self.confidence_adapter.pool_feature_contract
                    == CONFIDENCE_POOL_FEATURE_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE
                )
                or (
                    self.confidence_adapter.head_gradient_contract
                    == CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE
                    and self.confidence_adapter.pool_feature_contract
                    == CONFIDENCE_POOL_FEATURE_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE
                )
            )
        )
        exact_rank_max_reference_enabled = (
            fulltext_global_absolute_enabled
            and self.confidence_adapter.pool_feature_contract
            in {
                CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_REFERENCE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE,
                CONFIDENCE_POOL_FEATURE_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE,
            }
        )
        continuous_conditional_residual_enabled = (
            gated_pool_absolute_cap_enabled
            and self.confidence_adapter.gate_gradient_contract
            == CONFIDENCE_GATE_GRADIENT_CONTRACT_CONTINUOUS
        )
        continuous_monotone_depth_enabled = (
            gated_pool_absolute_cap_enabled
            and self.confidence_adapter.gate_gradient_contract
            == CONFIDENCE_GATE_GRADIENT_CONTRACT_CONTINUOUS_MONOTONE_DEPTH
        )
        complementary_trust_veto_enabled = (
            gated_pool_absolute_cap_enabled
            and self.confidence_adapter.gate_gradient_contract
            == CONFIDENCE_GATE_GRADIENT_CONTRACT_COMPLEMENTARY_TRUST_VETO
        )
        ungated_monotone_depth_enabled = (
            gated_pool_absolute_cap_enabled
            and self.confidence_adapter.gate_gradient_contract
            == CONFIDENCE_GATE_GRADIENT_CONTRACT_UNGATED_MONOTONE_DEPTH
        )
        floor_gated_monotone_depth_enabled = (
            gated_pool_absolute_cap_enabled
            and self.confidence_adapter.gate_gradient_contract
            == CONFIDENCE_GATE_GRADIENT_CONTRACT_FLOOR_GATED_MONOTONE_DEPTH
        )
        independent_absolute_logit_enabled = (
            gated_pool_absolute_cap_enabled
            and self.confidence_adapter.gate_gradient_contract
            in {
                CONFIDENCE_GATE_GRADIENT_CONTRACT_INDEPENDENT_ABSOLUTE_LOGIT,
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CROSS_ATTENTION_INDEPENDENT_ABSOLUTE_LOGIT,
            }
        )
        candidate_calibrated_logit_enabled = (
            gated_pool_absolute_cap_enabled
            and self.confidence_adapter.head_gradient_contract
            != CONFIDENCE_HEAD_GRADIENT_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE
            and (
                (
                    self.confidence_adapter.gate_gradient_contract
                    == CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_CALIBRATED_LOGIT
                    and self.confidence_adapter.pool_feature_contract
                    == CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_CALIBRATED
                )
                or (
                    self.confidence_adapter.gate_gradient_contract
                    == CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_NORMALIZED_LOGIT
                    and self.confidence_adapter.pool_feature_contract
                    == CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_NORMALIZED
                )
                or (
                    self.confidence_adapter.gate_gradient_contract
                    in {
                        CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT,
                        CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_DEPLOYED_ROUTING_ST,
                    }
                    and self.confidence_adapter.pool_feature_contract
                    == CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC
                )
                or (
                    self.confidence_adapter.gate_gradient_contract
                    == CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_SET_ATTENTION_LOGIT
                    and self.confidence_adapter.pool_feature_contract
                    == CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_SET_ATTENTION
                )
            )
        )
        candidate_absolute_logit_enabled = (
            candidate_calibrated_logit_enabled
            or fulltext_global_absolute_enabled
            or (
                gated_pool_absolute_cap_enabled
                and self.confidence_adapter.gate_gradient_contract
                == CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ABSOLUTE_LOGIT
                and self.confidence_adapter.pool_feature_contract
                == CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ABSOLUTE
            )
        )
        split_global_trust_veto_enabled = (
            self.confidence_adapter.head_gradient_contract
            == CONFIDENCE_HEAD_GRADIENT_CONTRACT_GLOBAL_TRUST_VETO
        )
        split_candidate_sample_enabled = (
            self.confidence_adapter.head_gradient_contract
            == CONFIDENCE_HEAD_GRADIENT_CONTRACT_CANDIDATE_SAMPLE
        )
        if split_candidate_sample_enabled and not candidate_absolute_logit_enabled:
            raise RuntimeError(
                "candidate/sample confidence splitting requires the candidate-absolute "
                "confidence contract"
            )
        if split_global_trust_veto_enabled and not candidate_absolute_logit_enabled:
            raise RuntimeError(
                "global trust/veto routing requires the candidate-absolute "
                "confidence contract"
            )
        signed_rank_query_pool_enabled = (
            self.confidence_adapter.pool_feature_contract
            == CONFIDENCE_POOL_FEATURE_CONTRACT_SIGNED_RANK_QUERY
        )
        absolute_cap_enabled = (
            self.confidence_adapter.phrase_aggregation
            in {
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_ABSOLUTE_CAP,
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
            }
            and not fulltext_global_absolute_enabled
        )
        absolute_cap_ceiling = (
            self.confidence_adapter.absolute_cap_ceiling()
            if absolute_cap_enabled
            else None
        )
        global_layers = []
        positive_global_layers = []
        negative_global_layers = []
        global_veto_raw_layers = []
        global_veto_depth_layers = []
        reference_global_layers = []
        residual_layers = []
        veto_coverage_layers = []
        veto_sample_gate_layers = []
        veto_carrier_index_layers = []
        for layer in range(int(confidence_base_flat.shape[0])):
            if exact_rank_max_reference_enabled:
                reference_global_logit = (
                    confidence_reference_base_flat[layer]
                    .detach()
                    .float()
                    .masked_fill(~flat_eligible, -torch.inf)
                    .max(dim=1)
                    .values
                )
            else:
                reference_global_logit, _reference_statistics = (
                    _masked_score_statistics(
                        confidence_reference_base_flat[layer],
                        flat_eligible,
                        topk=self.confidence_pool.score_topk,
                    )
                )
            if confidence_placeholder:
                global_logit, _statistics = _masked_score_statistics(
                    confidence_base_flat[layer],
                    flat_eligible,
                    topk=self.confidence_pool.score_topk,
                )
                residual = torch.zeros_like(global_logit)
            else:
                pool_hidden = confidence_hidden_flat[layer]
                pool_base = (
                    confidence_reference_base_flat[layer].detach()
                    if independent_global_absolute_enabled
                    else confidence_base_flat[layer]
                )
                if split_candidate_sample_enabled:
                    pool_hidden = pool_hidden.detach()
                    pool_base = pool_base.detach()
                global_logit, residual = self.confidence_pool(
                    pool_hidden,
                    pool_base,
                    flat_eligible,
                )
            if fulltext_global_absolute_enabled:
                if independent_global_absolute_enabled:
                    # The pool's zero-init affine is the complete deployed
                    # absolute-confidence logit. Rank and candidate-local
                    # confidence affect it only through detached input
                    # features, so neither can leak an additive score.
                    global_layers.append(residual)
                else:
                    candidate_absolute_max = confidence_base_flat[
                        layer
                    ].float().masked_fill(~flat_eligible, -torch.inf).max(dim=1).values
                    global_layers.append(candidate_absolute_max + residual)
                reference_global_layers.append(reference_global_logit)
                residual_layers.append(residual)
                continue
            if absolute_cap_enabled:
                if absolute_cap_ceiling is None:
                    raise AssertionError("absolute cap ceiling was not materialized")
                frozen_reference = confidence_reference_base_flat[layer].detach().float()
                reference_weights = torch.softmax(
                    frozen_reference.masked_fill(~flat_eligible, -torch.inf)
                    / self.confidence_pool.pool_temperature,
                    dim=1,
                ).masked_fill(~flat_eligible, 0.0)
                coverage_gate = confidence_mismatch_gate_flat[layer]
                if (
                    self.confidence_adapter.gate_gradient_contract
                    != CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_DEPLOYED_ROUTING_ST
                ):
                    coverage_gate = coverage_gate.detach()
                veto_coverage = (
                    reference_weights * coverage_gate.float()
                ).sum(dim=1)
                veto_coverage = veto_coverage.masked_fill(~flat_valid, 0.0)
                if gated_pool_absolute_cap_enabled:
                    # The criterion protects this same frozen-reference
                    # argmax on positives and supervises every admitted query
                    # on TNs. Coverage remains diagnostic only in v5.
                    carrier_index = (
                        confidence_reference_carrier_index_flat[layer]
                    )
                    veto_sample_gate = (
                        _gather_frozen_reference_carrier_gate(
                            confidence_mismatch_gate_flat[layer],
                            carrier_index,
                            detach_gate=(
                                self.confidence_adapter.gate_gradient_contract
                                == CONFIDENCE_GATE_GRADIENT_CONTRACT_HARD_DETACHED
                            ),
                        )
                    )
                    veto_sample_gate = veto_sample_gate.masked_fill(
                        ~flat_valid, 0.0
                    )
                    carrier_index = carrier_index.masked_fill(~flat_valid, -1)
                    veto_carrier_index_layers.append(carrier_index)
                else:
                    veto_sample_gate = (
                        (
                            veto_coverage
                            - self.confidence_adapter.veto_coverage_offset
                        )
                        / self.confidence_adapter.veto_coverage_ramp
                    ).clamp(min=0.0, max=1.0).detach()
                cap_temperature = self.confidence_adapter.veto_cap_temperature
                if gated_pool_absolute_cap_enabled:
                    if candidate_absolute_logit_enabled:
                        # FPR95 observes the highest-confidence admitted
                        # candidate. Use that exact candidate maximum as the
                        # independent sample carrier, while the pool residual
                        # supplies cross-candidate calibration. Local absolute
                        # losses train every positive/TN candidate directly.
                        candidate_absolute_max = confidence_base_flat[
                            layer
                        ].float().masked_fill(~flat_eligible, -torch.inf).max(
                            dim=1
                        ).values
                        if split_candidate_sample_enabled:
                            candidate_absolute_max = candidate_absolute_max.detach()
                        global_logit = candidate_absolute_max + residual
                        if candidate_calibrated_logit_enabled:
                            _patch_scale, _veto_depth, coverage_depth = (
                                self.confidence_adapter.candidate_calibration_depths()
                            )
                            global_logit = global_logit - (
                                coverage_depth * veto_coverage.detach()
                            )
                        if split_global_trust_veto_enabled:
                            if self.confidence_veto_pool is None:
                                raise RuntimeError(
                                    "global trust/veto confidence pool is missing"
                                )
                            if confidence_placeholder:
                                veto_raw = torch.zeros_like(global_logit)
                            else:
                                _, veto_raw = self.confidence_veto_pool(
                                    confidence_hidden_flat[layer].detach(),
                                    confidence_base_flat[layer].detach(),
                                    flat_eligible,
                                )
                            veto_depth = _ExactForwardSurrogateBackward.apply(
                                F.relu(veto_raw), veto_raw
                            )
                            trust_logit = global_logit
                            positive_global_logit = trust_logit - veto_depth
                            negative_global_logit = (
                                trust_logit.detach() - veto_depth
                            )
                            global_logit = positive_global_logit
                            positive_global_layers.append(positive_global_logit)
                            negative_global_layers.append(negative_global_logit)
                            global_veto_raw_layers.append(veto_raw)
                            global_veto_depth_layers.append(veto_depth)
                        veto_coverage_layers.append(veto_coverage)
                        veto_sample_gate_layers.append(veto_sample_gate)
                        global_layers.append(global_logit)
                        reference_global_layers.append(reference_global_logit)
                        residual_layers.append(residual)
                        continue
                    if independent_absolute_logit_enabled:
                        # The rank logit is a within-image ordering score, not
                        # a calibrated cross-sample confidence carrier. Emit
                        # the pool head itself as the absolute confidence logit.
                        # Its inputs retain detached query/text interaction,
                        # normalized candidate statistics, patch evidence and
                        # token-mismatch evidence, while no gradient or score
                        # path is shared with rank.
                        global_logit = residual
                        veto_coverage_layers.append(veto_coverage)
                        veto_sample_gate_layers.append(veto_sample_gate)
                        global_layers.append(global_logit)
                        reference_global_layers.append(reference_global_logit)
                        residual_layers.append(residual)
                        continue
                    if (
                        continuous_conditional_residual_enabled
                        or continuous_monotone_depth_enabled
                        or complementary_trust_veto_enabled
                        or ungated_monotone_depth_enabled
                        or floor_gated_monotone_depth_enabled
                    ):
                        # v21 keeps U0 exactly equal to the frozen reference,
                        # but replaces the absolute-cap smooth-min with a
                        # continuous modifier-conditioned signed residual.
                        # Positive raw-token supervision drives this gate near
                        # zero; changed TN tokens drive it near one. Therefore
                        # the global pool can lower hard TNs without collapsing
                        # positives and negatives onto one learned ceiling.
                        initial_gain = -float(
                            self.confidence_adapter.veto_cap_initial_ceiling
                        )
                        conditional_gain = (
                            -absolute_cap_ceiling / initial_gain
                        )
                        carrier_index = (
                            confidence_reference_carrier_index_flat[layer]
                        )
                        carrier_base_logit = reference_global_logit.detach()
                        if complementary_trust_veto_enabled:
                            # A scalar evidence head chooses between two
                            # one-sided depths. The learned mismatch gate may
                            # only lower confidence; its exact complement may
                            # only lift a trusted low-tail positive. This
                            # restores positive-q05 calibration without the
                            # unconstrained TN-raising path of v21.
                            exact_veto_depth = F.relu(residual)
                            surrogate_veto_depth = (
                                F.softplus(residual) - math.log(2.0)
                            )
                            veto_depth = _ExactForwardSurrogateBackward.apply(
                                exact_veto_depth, surrogate_veto_depth
                            )
                            exact_trust_depth = F.relu(-residual)
                            surrogate_trust_depth = (
                                F.softplus(-residual) - math.log(2.0)
                            )
                            trust_depth = _ExactForwardSurrogateBackward.apply(
                                exact_trust_depth, surrogate_trust_depth
                            )
                            trust_gate = 1.0 - veto_sample_gate
                            global_logit = carrier_base_logit + (
                                conditional_gain
                                * (
                                    trust_gate * trust_depth
                                    - veto_sample_gate * veto_depth
                                )
                            )
                        elif ungated_monotone_depth_enabled:
                            # The modifier mismatch gate remains supervised and
                            # enters the token-conditioned pool as evidence, but
                            # it no longer blocks the global readout. This lets
                            # the independent confidence pool suppress a hard
                            # high-score TN even when its carrier edit residual
                            # has not crossed the lexical gate. Exact inference
                            # remains one-sided and U0 remains rank-identical.
                            exact_depth = F.relu(residual)
                            surrogate_depth = (
                                F.softplus(residual) - math.log(2.0)
                            )
                            veto_depth = _ExactForwardSurrogateBackward.apply(
                                exact_depth, surrogate_depth
                            )
                            global_logit = carrier_base_logit - (
                                conditional_gain * veto_depth
                            )
                        elif floor_gated_monotone_depth_enabled:
                            # Keep lexical selectivity but guarantee a gradient
                            # and inference path for hard TNs whose carrier gate
                            # has not opened. The fixed floor is deliberately
                            # below the observed positive/TN gate separation;
                            # the positive-delta trust objective prevents this
                            # fallback path from eroding the positive low tail.
                            exact_depth = F.relu(residual)
                            surrogate_depth = (
                                F.softplus(residual) - math.log(2.0)
                            )
                            veto_depth = _ExactForwardSurrogateBackward.apply(
                                exact_depth, surrogate_depth
                            )
                            effective_gate = (
                                CONFIDENCE_MONOTONE_VETO_GATE_FLOOR
                                + (
                                    1.0
                                    - CONFIDENCE_MONOTONE_VETO_GATE_FLOOR
                                )
                                * veto_sample_gate
                            )
                            global_logit = carrier_base_logit - (
                                effective_gate
                                * conditional_gain
                                * veto_depth
                            )
                        elif continuous_monotone_depth_enabled:
                            # Exact inference depth is non-negative, so this
                            # confidence path can never raise a TN. The smooth
                            # centered softplus surrogate gives a nonzero
                            # derivative at U0 even though ReLU(0) is exactly
                            # zero, preserving bitwise rank inheritance.
                            exact_depth = F.relu(residual)
                            surrogate_depth = (
                                F.softplus(residual) - math.log(2.0)
                            )
                            veto_depth = _ExactForwardSurrogateBackward.apply(
                                exact_depth, surrogate_depth
                            )
                            global_logit = carrier_base_logit - (
                                veto_sample_gate
                                * conditional_gain
                                * veto_depth
                            )
                        else:
                            global_logit = carrier_base_logit + (
                                veto_sample_gate
                                * conditional_gain
                                * residual
                            )
                        veto_coverage_layers.append(veto_coverage)
                        veto_sample_gate_layers.append(veto_sample_gate)
                        global_layers.append(global_logit)
                        reference_global_layers.append(reference_global_logit)
                        residual_layers.append(residual)
                        continue
                    # Smooth-min the carrier reference with the absolute cap,
                    # then let the shared pool add only downward veto depth.
                    # Thus an open gate can never raise either a high or an
                    # already-low U6551 score.
                    frozen_reference_global = reference_global_logit.detach()
                    capped_reference_logit = frozen_reference_global - (
                        cap_temperature
                        * F.softplus(
                            (
                                frozen_reference_global
                                - absolute_cap_ceiling
                            )
                            / cap_temperature
                        )
                    )
                    veto_target_logit = capped_reference_logit - (
                        cap_temperature
                        * F.softplus(-residual / cap_temperature)
                    )
                    carrier_base_logit = frozen_reference_global
                    if signed_rank_query_pool_enabled:
                        # The signed pool is the lightweight P50-style global
                        # calibration path. It remains active when the exact
                        # word-veto gate is closed, so hard-tail samples cannot
                        # pass through unchanged solely because their gate is 0.
                        carrier_base_logit = carrier_base_logit + residual
                    blended_global_logit = carrier_base_logit + (
                        veto_sample_gate
                        * (veto_target_logit - carrier_base_logit)
                    )
                    exact_global_logit = torch.where(
                        veto_sample_gate.eq(0.0),
                        carrier_base_logit,
                        blended_global_logit,
                    )
                    if (
                        self.confidence_adapter.gate_gradient_contract
                        == CONFIDENCE_GATE_GRADIENT_CONTRACT_HARD_FORWARD_SOFT_BACKWARD
                    ):
                        global_logit = _ExactForwardSurrogateBackward.apply(
                            exact_global_logit, blended_global_logit
                        )
                    else:
                        global_logit = exact_global_logit
                else:
                    capped_global_logit = global_logit - cap_temperature * F.softplus(
                        (global_logit - absolute_cap_ceiling) / cap_temperature
                    )
                    blended_global_logit = global_logit + veto_sample_gate * (
                        capped_global_logit - global_logit
                    )
                    # Avoid even a rounding-only change when the veto is closed.
                    global_logit = torch.where(
                        veto_sample_gate.eq(0.0), global_logit, blended_global_logit
                    )
                veto_coverage_layers.append(veto_coverage)
                veto_sample_gate_layers.append(veto_sample_gate)
            global_layers.append(global_logit)
            reference_global_layers.append(reference_global_logit)
            residual_layers.append(residual)
        global_layers_tensor = torch.stack(global_layers, dim=0)
        if split_global_trust_veto_enabled:
            expected_layers = int(confidence_base_flat.shape[0])
            if not all(
                len(values) == expected_layers
                for values in (
                    positive_global_layers,
                    negative_global_layers,
                    global_veto_raw_layers,
                    global_veto_depth_layers,
                )
            ):
                raise RuntimeError(
                    "global trust/veto routing did not cover every confidence layer"
                )
            positive_global_layers_tensor = torch.stack(
                positive_global_layers, dim=0
            )
            negative_global_layers_tensor = torch.stack(
                negative_global_layers, dim=0
            )
            global_veto_raw_layers_tensor = torch.stack(
                global_veto_raw_layers, dim=0
            )
            global_veto_depth_layers_tensor = torch.stack(
                global_veto_depth_layers, dim=0
            )
            if (
                positive_global_layers_tensor.device.type != "cuda"
                and not torch.equal(
                    positive_global_layers_tensor, negative_global_layers_tensor
                )
            ):
                raise RuntimeError(
                    "positive and TN confidence routes changed the forward value"
                )
        else:
            positive_global_layers_tensor = None
            negative_global_layers_tensor = None
            global_veto_raw_layers_tensor = None
            global_veto_depth_layers_tensor = None
        reference_global_layers_tensor = torch.stack(
            reference_global_layers, dim=0
        )
        confidence_delta_layers_tensor = (
            global_layers_tensor - reference_global_layers_tensor.detach()
        )
        residual_layers_tensor = torch.stack(residual_layers, dim=0)
        veto_coverage_layers_tensor = (
            torch.stack(veto_coverage_layers, dim=0)
            if absolute_cap_enabled
            else None
        )
        veto_sample_gate_layers_tensor = (
            torch.stack(veto_sample_gate_layers, dim=0)
            if absolute_cap_enabled
            else None
        )
        veto_carrier_index_layers_tensor = (
            torch.stack(veto_carrier_index_layers, dim=0)
            if absolute_cap_enabled and gated_pool_absolute_cap_enabled
            else None
        )
        broadcast_validity = global_layers_tensor[:, :, None].expand(
            -1, -1, candidate_count
        )
        broadcast_validity = broadcast_validity.masked_fill(
            ~flat_eligible[None], torch.finfo(broadcast_validity.dtype).min
        )
        broadcast_validity = broadcast_validity.masked_fill(
            ~flat_valid[None, :, None], torch.finfo(broadcast_validity.dtype).min
        )
        layer_validity = self._reshape_tower_output(
            broadcast_validity, batch_size=batch_size, slot_count=slot_count
        )
        if split_global_trust_veto_enabled:
            if (
                positive_global_layers_tensor is None
                or negative_global_layers_tensor is None
            ):
                raise AssertionError("global trust/veto routes are unavailable")

            def _broadcast_global_route(route: Tensor) -> Tensor:
                broadcast = route[:, :, None].expand(-1, -1, candidate_count)
                broadcast = broadcast.masked_fill(
                    ~flat_eligible[None], torch.finfo(broadcast.dtype).min
                )
                broadcast = broadcast.masked_fill(
                    ~flat_valid[None, :, None], torch.finfo(broadcast.dtype).min
                )
                return self._reshape_tower_output(
                    broadcast, batch_size=batch_size, slot_count=slot_count
                )

            positive_layer_validity = _broadcast_global_route(
                positive_global_layers_tensor
            )
            negative_layer_validity = _broadcast_global_route(
                negative_global_layers_tensor
            )
        else:
            positive_layer_validity = None
            negative_layer_validity = None
        confidence_token = self._reshape_tower_output(
            confidence_token_flat, batch_size=batch_size, slot_count=slot_count
        )
        confidence_token = confidence_token.masked_fill(
            ~valid[None, :, None, :, None],
            torch.finfo(confidence_token.dtype).min,
        )
        confidence_token_residual = self._reshape_tower_output(
            confidence_token_residual_flat,
            batch_size=batch_size,
            slot_count=slot_count,
        )
        confidence_token_residual = confidence_token_residual.masked_fill(
            ~valid[None, :, None, :, None], 0.0
        )
        confidence_base = self._reshape_tower_output(
            confidence_base_flat, batch_size=batch_size, slot_count=slot_count
        )
        confidence_base = confidence_base.masked_fill(
            ~valid[None, :, None, :], torch.finfo(confidence_base.dtype).min
        )
        confidence_reference_base = self._reshape_tower_output(
            confidence_reference_base_flat,
            batch_size=batch_size,
            slot_count=slot_count,
        )
        confidence_reference_base = confidence_reference_base.masked_fill(
            ~valid[None, :, None, :],
            torch.finfo(confidence_reference_base.dtype).min,
        )
        confidence_layers = int(confidence_base_flat.shape[0])
        global_shaped = global_layers_tensor.view(
            confidence_layers, batch_size, slot_count
        ).masked_fill(~valid[None], torch.finfo(global_layers_tensor.dtype).min)
        if split_global_trust_veto_enabled:
            if any(
                value is None
                for value in (
                    positive_global_layers_tensor,
                    negative_global_layers_tensor,
                    global_veto_raw_layers_tensor,
                    global_veto_depth_layers_tensor,
                )
            ):
                raise AssertionError("global trust/veto diagnostics are unavailable")
            positive_global_shaped = positive_global_layers_tensor.view(
                confidence_layers, batch_size, slot_count
            ).masked_fill(~valid[None], torch.finfo(global_layers_tensor.dtype).min)
            negative_global_shaped = negative_global_layers_tensor.view(
                confidence_layers, batch_size, slot_count
            ).masked_fill(~valid[None], torch.finfo(global_layers_tensor.dtype).min)
            global_veto_raw_shaped = global_veto_raw_layers_tensor.view(
                confidence_layers, batch_size, slot_count
            ).masked_fill(~valid[None], 0.0)
            global_veto_depth_shaped = global_veto_depth_layers_tensor.view(
                confidence_layers, batch_size, slot_count
            ).masked_fill(~valid[None], 0.0)
        else:
            positive_global_shaped = None
            negative_global_shaped = None
            global_veto_raw_shaped = None
            global_veto_depth_shaped = None
        residual_shaped = residual_layers_tensor.view(
            confidence_layers, batch_size, slot_count
        ).masked_fill(~valid[None], 0.0)
        reference_global_shaped = reference_global_layers_tensor.view(
            confidence_layers, batch_size, slot_count
        ).masked_fill(~valid[None], torch.finfo(global_layers_tensor.dtype).min)
        confidence_delta_shaped = confidence_delta_layers_tensor.view(
            confidence_layers, batch_size, slot_count
        ).masked_fill(~valid[None], 0.0)
        confidence_mismatch_gate = self._reshape_tower_output(
            confidence_mismatch_gate_flat,
            batch_size=batch_size,
            slot_count=slot_count,
        ).masked_fill(~valid[None, :, None, :], 0.0)
        confidence_raw_mismatch_gate = self._reshape_tower_output(
            confidence_raw_mismatch_gate_flat,
            batch_size=batch_size,
            slot_count=slot_count,
        ).masked_fill(~valid[None, :, None, :], 0.0)
        confidence_deployed_routing_gate = self._reshape_tower_output(
            confidence_deployed_routing_gate_flat,
            batch_size=batch_size,
            slot_count=slot_count,
        ).masked_fill(~valid[None, :, None, :], 0.0)
        confidence_deployed_routing_residual = self._reshape_tower_output(
            confidence_deployed_routing_residual_flat,
            batch_size=batch_size,
            slot_count=slot_count,
        ).masked_fill(~valid[None, :, None, :], 0.0)
        if absolute_cap_enabled:
            if (
                veto_coverage_layers_tensor is None
                or veto_sample_gate_layers_tensor is None
                or absolute_cap_ceiling is None
            ):
                raise AssertionError("absolute-cap diagnostics are unavailable")
            confidence_veto_coverage = veto_coverage_layers_tensor.view(
                confidence_layers, batch_size, slot_count
            ).masked_fill(~valid[None], 0.0)
            confidence_veto_sample_gate = veto_sample_gate_layers_tensor.view(
                confidence_layers, batch_size, slot_count
            ).masked_fill(~valid[None], 0.0)
            confidence_veto_ceiling = absolute_cap_ceiling.expand_as(
                confidence_veto_coverage
            )
        else:
            confidence_veto_coverage = None
            confidence_veto_sample_gate = None
            confidence_veto_ceiling = None
        confidence_veto_carrier_index = (
            confidence_reference_carrier_index_flat.view(
                confidence_layers, batch_size, slot_count
            ).masked_fill(~valid[None], -1)
            if gated_pool_absolute_cap_enabled
            else None
        )
        final_validity = layer_validity[-1]
        active_layer_token = (
            confidence_token
            if not confidence_placeholder and self.training and self.phase == "confidence"
            else layer_token
        )

        if flat_predicate_mask is None:
            effective_predicate = torch.zeros_like(rank_phrase_token_mask)
        else:
            effective_predicate = self.rank_tower._pad_mask(flat_predicate_mask)
            effective_predicate &= rank_phrase_token_mask
        predicate_flat = torch.stack(
            [
                self.rank_tower.aggregate_phrase_logits(
                    rank_token_flat[layer], effective_predicate
                )
                for layer in range(int(rank_token_flat.shape[0]))
            ],
            dim=0,
        )
        predicate_flat = predicate_flat.masked_fill(
            ~flat_eligible[None], torch.finfo(predicate_flat.dtype).min
        )
        predicate_valid = valid & effective_predicate.view(
            batch_size, slot_count, self.max_text_len
        ).any(dim=-1)
        layer_predicate = self._reshape_tower_output(
            predicate_flat, batch_size=batch_size, slot_count=slot_count
        ).masked_fill(
            ~predicate_valid[None, :, None, :],
            torch.finfo(predicate_flat.dtype).min,
        )

        score_token_mask = rank_score_token_mask.view(
            batch_size, slot_count, self.max_text_len
        )
        phrase_token_mask = rank_phrase_token_mask.view(
            batch_size, slot_count, self.max_text_len
        )
        final_rank_score = final_phrase.sigmoid()
        final_score = final_validity.sigmoid()
        output.update(
            {
                "layer_token_logits": active_layer_token,
                "final_token_logits": active_layer_token[-1],
                "layer_phrase_logits": layer_phrase,
                "final_phrase_logits": final_phrase,
                "layer_validity_logits": layer_validity,
                "final_validity_logits": final_validity,
                "layer_validity_gate_logits": residual_shaped,
                "final_validity_gate_logits": residual_shaped[-1],
                "layer_confidence_pool_absolute_logits": residual_shaped,
                "final_confidence_pool_absolute_logits": residual_shaped[-1],
                "layer_confidence_base_logits": confidence_base,
                "final_confidence_base_logits": confidence_base[-1],
                "layer_reference_base_logits": confidence_reference_base,
                "final_reference_base_logits": confidence_reference_base[-1],
                "confidence_layer_token_logits": confidence_token,
                "confidence_token_logits": confidence_token[-1],
                "final_rank_token_logits": layer_token[-1],
                "layer_frozen_rank_full_expression_global_logits": (
                    frozen_rank_full_global
                ),
                "final_frozen_rank_full_expression_global_logits": (
                    frozen_rank_full_global[-1]
                ),
                "final_confidence_token_logits": confidence_token[-1],
                "layer_confidence_token_residual_logits": (
                    confidence_token_residual
                ),
                "final_confidence_token_residual_logits": (
                    confidence_token_residual[-1]
                ),
                "layer_confidence_global_logits": global_shaped,
                "final_confidence_global_logits": global_shaped[-1],
                "final_global_confidence_logits": global_shaped[-1],
                "layer_reference_global_confidence_logits": reference_global_shaped,
                "final_reference_global_confidence_logits": reference_global_shaped[-1],
                "layer_confidence_delta_logits": confidence_delta_shaped,
                "final_confidence_delta_logits": confidence_delta_shaped[-1],
                "layer_confidence_mismatch_gate": confidence_mismatch_gate,
                "final_confidence_mismatch_gate": confidence_mismatch_gate[-1],
                "layer_predicate_logits": layer_predicate,
                "final_predicate_logits": layer_predicate[-1],
                "predicate_token_mask": effective_predicate.view(
                    batch_size, slot_count, self.max_text_len
                ),
                "predicate_valid_mask": predicate_valid,
                "score_token_mask": score_token_mask,
                "expression_token_mask": phrase_token_mask,
                "modifier_valid_mask": rank_modifier_valid.view(
                    batch_size, slot_count
                ),
                "category_only_patch_fallback_mask": (
                    ~rank_modifier_valid.view(batch_size, slot_count)
                    & valid
                ),
                "final_rank_score": final_rank_score,
                "final_score": final_score,
                "rank_score": final_rank_score,
                "text_rank_score": final_rank_score,
                "confidence_score": final_score,
                "confidence_base_score": confidence_base[-1].sigmoid(),
                "candidate_mask": output["candidate_eligible_mask"],
                "rank_output_is_placeholder": torch.as_tensor(
                    rank_placeholder, device=candidate_hs.device, dtype=torch.bool
                ),
                "confidence_output_is_placeholder": torch.as_tensor(
                    confidence_placeholder,
                    device=candidate_hs.device,
                    dtype=torch.bool,
                ),
            }
        )
        if (
            self.confidence_adapter.head_gradient_contract
            == CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYED_ROUTER
        ):
            output.update(
                {
                    "layer_confidence_raw_mismatch_gate": (
                        confidence_raw_mismatch_gate
                    ),
                    "final_confidence_raw_mismatch_gate": (
                        confidence_raw_mismatch_gate[-1]
                    ),
                    "layer_confidence_deployed_routing_gate": (
                        confidence_deployed_routing_gate
                    ),
                    "final_confidence_deployed_routing_gate": (
                        confidence_deployed_routing_gate[-1]
                    ),
                    "layer_confidence_deployed_routing_residual": (
                        confidence_deployed_routing_residual
                    ),
                    "final_confidence_deployed_routing_residual": (
                        confidence_deployed_routing_residual[-1]
                    ),
                }
            )
        if split_global_trust_veto_enabled:
            if any(
                value is None
                for value in (
                    positive_layer_validity,
                    negative_layer_validity,
                    positive_global_shaped,
                    negative_global_shaped,
                    global_veto_raw_shaped,
                    global_veto_depth_shaped,
                )
            ):
                raise AssertionError("global trust/veto outputs are unavailable")
            output.update(
                {
                    "layer_positive_confidence_logits": positive_layer_validity,
                    "final_positive_confidence_logits": positive_layer_validity[-1],
                    "layer_negative_confidence_logits": negative_layer_validity,
                    "final_negative_confidence_logits": negative_layer_validity[-1],
                    "layer_positive_global_confidence_logits": (
                        positive_global_shaped
                    ),
                    "final_positive_global_confidence_logits": (
                        positive_global_shaped[-1]
                    ),
                    "layer_negative_global_confidence_logits": (
                        negative_global_shaped
                    ),
                    "final_negative_global_confidence_logits": (
                        negative_global_shaped[-1]
                    ),
                    "layer_global_veto_raw_logits": global_veto_raw_shaped,
                    "final_global_veto_raw_logits": global_veto_raw_shaped[-1],
                    "layer_global_veto_depth": global_veto_depth_shaped,
                    "final_global_veto_depth": global_veto_depth_shaped[-1],
                }
            )
        if absolute_cap_enabled:
            if (
                confidence_veto_coverage is None
                or confidence_veto_sample_gate is None
                or confidence_veto_ceiling is None
                or absolute_cap_ceiling is None
            ):
                raise AssertionError("absolute-cap outputs are unavailable")
            output.update(
                {
                    "layer_confidence_veto_coverage": confidence_veto_coverage,
                    "final_confidence_veto_coverage": confidence_veto_coverage[-1],
                    "layer_confidence_veto_sample_gate": (
                        confidence_veto_sample_gate
                    ),
                    "final_confidence_veto_sample_gate": (
                        confidence_veto_sample_gate[-1]
                    ),
                    "confidence_veto_absolute_ceiling": absolute_cap_ceiling,
                    "layer_confidence_veto_absolute_ceiling": (
                        confidence_veto_ceiling
                    ),
                    "final_confidence_veto_absolute_ceiling": (
                        confidence_veto_ceiling[-1]
                    ),
                }
            )
        if gated_pool_absolute_cap_enabled:
            if confidence_veto_carrier_index is None:
                raise AssertionError("carrier indices are unavailable")
            output.update(
                {
                    "layer_confidence_veto_carrier_index": (
                        confidence_veto_carrier_index
                    ),
                    "final_confidence_veto_carrier_index": (
                        confidence_veto_carrier_index[-1]
                    ),
                }
            )
        return output


__all__ = [
    "AbsoluteConfidencePool",
    "CONFIDENCE_HEAD_GRADIENT_CONTRACT_SHARED",
    "CONFIDENCE_HEAD_GRADIENT_CONTRACT_SPLIT",
    "CONFIDENCE_HEAD_GRADIENT_CONTRACT_SPLIT_JOINT_CLIP",
    "CONFIDENCE_HEAD_GRADIENT_CONTRACT_GLOBAL_TRUST_VETO",
    "CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYED_ROUTER",
    "CONFIDENCE_HEAD_GRADIENT_CONTRACT_CANDIDATE_SAMPLE",
    "CONFIDENCE_HEAD_GRADIENT_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE",
    "CONFIDENCE_HEAD_GRADIENT_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE",
    "CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE",
    "CONFIDENCE_HEAD_GRADIENT_CONTRACT_SPLIT_SEMANTICS",
    "CONFIDENCE_HEAD_GRADIENT_CONTRACTS",
    "CONFIDENCE_GATE_GRADIENT_CONTRACT_HARD_DETACHED",
    "CONFIDENCE_GATE_GRADIENT_CONTRACT_HARD_FORWARD_SOFT_BACKWARD",
    "CONFIDENCE_GATE_GRADIENT_CONTRACT_CONTINUOUS",
    "CONFIDENCE_GATE_GRADIENT_CONTRACT_CONTINUOUS_MONOTONE_DEPTH",
    "CONFIDENCE_GATE_GRADIENT_CONTRACT_COMPLEMENTARY_TRUST_VETO",
    "CONFIDENCE_GATE_GRADIENT_CONTRACT_UNGATED_MONOTONE_DEPTH",
    "CONFIDENCE_GATE_GRADIENT_CONTRACT_FLOOR_GATED_MONOTONE_DEPTH",
    "CONFIDENCE_GATE_GRADIENT_CONTRACT_INDEPENDENT_ABSOLUTE_LOGIT",
    "CONFIDENCE_GATE_GRADIENT_CONTRACT_CROSS_ATTENTION_INDEPENDENT_ABSOLUTE_LOGIT",
    "CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ABSOLUTE_LOGIT",
    "CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_CALIBRATED_LOGIT",
    "CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_NORMALIZED_LOGIT",
    "CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT",
    "CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_SET_ATTENTION_LOGIT",
    "CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_DEPLOYED_ROUTING_ST",
    "CONFIDENCE_MONOTONE_VETO_GATE_FLOOR",
    "CONFIDENCE_GATE_GRADIENT_CONTRACTS",
    "DENSE_DUTY_CONTRACT_VERSION",
    "CONFIDENCE_POOL_FEATURE_CONTRACT",
    "CONFIDENCE_POOL_FEATURE_CONTRACT_SIGNED_RANK_QUERY",
    "CONFIDENCE_POOL_FEATURE_CONTRACT_TOKEN_CONDITIONED",
    "CONFIDENCE_POOL_FEATURE_CONTRACT_CROSS_ATTENTION",
    "CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ABSOLUTE",
    "CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_CALIBRATED",
    "CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_NORMALIZED",
    "CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC",
    "CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_SET_ATTENTION",
    "CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE",
    "CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_REFERENCE",
    "CONFIDENCE_POOL_FEATURE_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE",
    "CONFIDENCE_POOL_FEATURE_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE",
    "CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH",
    "CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP",
    "CONFIDENCE_TOKEN_CONTRACT",
    "DENSE_DUTY_PHASES",
    "DenseExpressionTower",
    "StageBDenseDutyScorer",
    "TokenAwareConfidenceAdapter",
    "normalize_dense_duty_phase",
]
