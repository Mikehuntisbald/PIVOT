from __future__ import annotations

import torch

from models.GroundingDINO.stage_b_dense_duty_scorer import (
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT,
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE,
    CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
    CONFIDENCE_POOL_FEATURE_CONTRACT_DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE,
    CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH,
    StageBDenseDutyScorer,
)
from tests.test_stage_b_dense_duty_scorer import _FakeGroundingDINO
from tests.test_stage_b_v54_exact_reference_scorer import _forward, _valid_values


def _build() -> StageBDenseDutyScorer:
    torch.manual_seed(20260803)
    source = _FakeGroundingDINO(seed=137)
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
            CONFIDENCE_POOL_FEATURE_CONTRACT_DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE
        ),
        confidence_residual_parameterization_gain=0.25 / 0.03,
        confidence_gate_gradient_contract=(
            CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT
        ),
        confidence_head_gradient_contract=(
            CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE
        ),
        confidence_full_decoder_verifier=True,
        confidence_veto_only_patch_softmin=True,
        phase="confidence",
    )
    scorer.copy_confidence_verifier_from_rank()
    scorer.train()
    return scorer


def test_v62_removes_the_trainable_absolute_pool_from_active_ownership():
    scorer = _build()
    active = {id(parameter) for parameter in scorer.confidence_parameters()}
    pool = {id(parameter) for parameter in scorer.confidence_pool.parameters()}
    head = {
        id(parameter)
        for parameter in scorer.confidence_verifier_veto_head.parameters()
    }
    assert pool
    assert head <= active
    assert not pool & active
    assert all(not parameter.requires_grad for parameter in scorer.confidence_pool.parameters())
    assert scorer.expected_live_confidence_parameter_tensor_counts() == {
        "token_veto": len(scorer.token_veto_parameters()),
        "global_absolute": 6,
    }


def test_v62_u0_is_exact_zero_and_contains_no_free_absolute_score():
    scorer = _build()
    output, _inputs, _provider = _forward(scorer)
    valid = output["expression_valid_mask"]
    assert torch.equal(
        output["final_confidence_token_logits"],
        output["final_rank_token_logits"],
    )
    assert torch.count_nonzero(output["final_deployed_query_veto_depth"]) == 0
    assert torch.count_nonzero(output["final_confidence_pool_absolute_logits"]) == 0
    assert torch.count_nonzero(output["final_confidence_global_logits"][valid]) == 0


def test_v62_deployment_is_exactly_negative_patch_softmin_veto():
    scorer = _build()
    with torch.no_grad():
        scorer.confidence_verifier_veto_head[-1].bias.fill_(3.0)
    output, _inputs, _provider = _forward(scorer, requires_grad=True)
    global_logit = _valid_values(output, "final_confidence_global_logits")
    depth = _valid_values(output, "final_deployed_query_veto_depth")
    pool = _valid_values(output, "final_confidence_pool_absolute_logits")
    assert torch.all(depth >= 0.0)
    assert torch.all(global_logit <= 0.0)
    assert torch.equal(pool, torch.zeros_like(pool))
    assert torch.allclose(global_logit, -depth, atol=1e-6, rtol=0.0)

    global_logit.sum().backward()
    assert all(parameter.grad is None for parameter in scorer.rank_parameters())
    assert all(parameter.grad is None for parameter in scorer.confidence_pool.parameters())
    assert any(
        parameter.grad is not None
        for parameter in scorer.confidence_verifier_veto_head.parameters()
    )


def test_v62_deployed_score_does_not_change_with_rank_only_perturbation():
    scorer = _build()
    with torch.no_grad():
        scorer.confidence_verifier_veto_head[-1].bias.fill_(2.0)
    before, inputs, provider = _forward(scorer)
    before_global = before["final_confidence_global_logits"].detach().clone()
    with torch.no_grad():
        for parameter in scorer.rank_tower.token_head.parameters():
            parameter.add_(7.0)
    after = scorer(raw_context_provider=provider, **inputs)
    assert torch.equal(before_global, after["final_confidence_global_logits"])
