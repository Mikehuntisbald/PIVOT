from __future__ import annotations

import torch

from models.GroundingDINO.stage_b_dense_duty_scorer import (
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT,
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE,
    CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
    CONFIDENCE_POOL_FEATURE_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE,
    CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH,
    StageBDenseDutyScorer,
)
from tests.test_stage_b_dense_duty_scorer import _FakeGroundingDINO
from tests.test_stage_b_v54_exact_reference_scorer import _forward, _valid_values


def _build_scorer() -> StageBDenseDutyScorer:
    torch.manual_seed(20260802)
    source = _FakeGroundingDINO(seed=131)
    scorer = StageBDenseDutyScorer(
        source.feat_map,
        source.transformer.encoder,
        source.transformer.decoder,
        source.transformer.level_embed,
        max_text_len=8,
        candidate_topk=3,
        category_gate_max_gap=100.0,
        patch_score_clip=5.0,
        confidence_adapter_dim=3,
        confidence_hidden_dim=7,
        confidence_pool_topk=2,
        confidence_phrase_aggregation=(
            CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
        ),
        confidence_veto_gate_offset=0.0,
        confidence_veto_gate_scale=0.03,
        confidence_rank_evidence_contract=(
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
        ),
        confidence_pool_feature_contract=(
            CONFIDENCE_POOL_FEATURE_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE
        ),
        confidence_residual_parameterization_gain=0.25 / 0.03,
        confidence_gate_gradient_contract=(
            CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT
        ),
        confidence_head_gradient_contract=(
            CONFIDENCE_HEAD_GRADIENT_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE
        ),
        phase="confidence",
    )
    scorer.train()
    return scorer


def test_v55_u0_candidate_and_deployed_global_are_independent_zero_logits():
    scorer = _build_scorer()
    output, _inputs, _provider = _forward(scorer)
    valid = output["expression_valid_mask"]
    eligible = output["candidate_eligible_mask"]
    local = output["final_confidence_base_logits"]
    deployed = _valid_values(output, "final_confidence_global_logits")
    pool = _valid_values(output, "final_confidence_pool_absolute_logits")
    reference = _valid_values(
        output, "final_reference_global_confidence_logits"
    )

    assert torch.equal(local[eligible], torch.zeros_like(local[eligible]))
    assert torch.equal(deployed, torch.zeros_like(deployed))
    assert torch.equal(deployed, pool)
    assert bool(reference.ne(0).any().item())
    assert torch.equal(
        output["final_confidence_delta_logits"][valid],
        -reference,
    )


def test_v55_candidate_and_global_final_affines_have_disjoint_forward_values():
    scorer = _build_scorer()
    with torch.no_grad():
        scorer.confidence_adapter.candidate_absolute_head[-1].bias.fill_(0.375)
        scorer.confidence_pool.residual[-1].bias.fill_(-0.125)

    output, _inputs, _provider = _forward(scorer)
    eligible = output["candidate_eligible_mask"]
    deployed = _valid_values(output, "final_confidence_global_logits")
    pool = _valid_values(output, "final_confidence_pool_absolute_logits")

    torch.testing.assert_close(
        output["final_confidence_base_logits"][eligible],
        torch.full_like(output["final_confidence_base_logits"][eligible], 0.375),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        deployed,
        torch.full_like(deployed, -0.125),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(deployed, pool)


def test_v55_global_loss_cannot_reach_candidate_final_affine_or_rank():
    scorer = _build_scorer()
    output, inputs, provider = _forward(scorer, requires_grad=True)
    valid = output["expression_valid_mask"]
    output["final_confidence_global_logits"][valid].sum().backward()

    assert scorer.confidence_adapter.candidate_absolute_head[-1].bias.grad is None
    pool_bias_grad = scorer.confidence_pool.residual[-1].bias.grad
    assert pool_bias_grad is not None
    assert bool(pool_bias_grad.ne(0).any().item())
    assert all(parameter.grad is None for parameter in scorer.token_veto_parameters())
    assert all(parameter.grad is None for parameter in scorer.rank_parameters())
    assert inputs["candidate_hs"].grad is None
    assert inputs["candidate_boxes"].grad is None
    assert inputs["candidate_patch_logits"].grad is None
    assert all(value.grad is None for value in provider.leaves)


def test_v55_local_loss_cannot_reach_pool_final_affine():
    scorer = _build_scorer()
    output, _inputs, _provider = _forward(scorer)
    eligible = output["candidate_eligible_mask"]
    output["final_confidence_base_logits"][eligible].sum().backward()

    candidate_bias_grad = (
        scorer.confidence_adapter.candidate_absolute_head[-1].bias.grad
    )
    assert candidate_bias_grad is not None
    assert bool(candidate_bias_grad.ne(0).any().item())
    assert scorer.confidence_pool.residual[-1].bias.grad is None
