from __future__ import annotations

import torch

from models.GroundingDINO.stage_b_dense_duty_scorer import (
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT,
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE,
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_QUERY_GLOBAL_ABSOLUTE,
    CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
    CONFIDENCE_POOL_FEATURE_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE,
    CONFIDENCE_POOL_FEATURE_CONTRACT_DEPLOYMENT_OWNED_QUERY_GLOBAL_ABSOLUTE,
    CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH,
    StageBDenseDutyScorer,
)
from tests.test_stage_b_dense_duty_scorer import _FakeGroundingDINO
from tests.test_stage_b_v54_exact_reference_scorer import _forward, _valid_values


def _build_scorer(*, query_global: bool) -> StageBDenseDutyScorer:
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
            CONFIDENCE_POOL_FEATURE_CONTRACT_DEPLOYMENT_OWNED_QUERY_GLOBAL_ABSOLUTE
            if query_global
            else CONFIDENCE_POOL_FEATURE_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE
        ),
        confidence_residual_parameterization_gain=0.25 / 0.03,
        confidence_gate_gradient_contract=(
            CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT
        ),
        confidence_head_gradient_contract=(
            CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_QUERY_GLOBAL_ABSOLUTE
            if query_global
            else CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE
        ),
        phase="confidence",
    )
    scorer.train()
    return scorer


def test_v59_preserves_v56_state_and_exact_deployed_u0():
    v56 = _build_scorer(query_global=False)
    v59 = _build_scorer(query_global=True)

    v56_state = v56.state_dict()
    v59_state = v59.state_dict()
    assert tuple(v56_state) == tuple(v59_state)
    assert all(torch.equal(v56_state[name], v59_state[name]) for name in v56_state)

    v56_output, _inputs, _provider = _forward(v56)
    v59_output, _inputs, _provider = _forward(v59)
    assert torch.equal(
        v56_output["final_confidence_global_logits"],
        v59_output["final_confidence_global_logits"],
    )


def test_v59_query_head_belongs_only_to_deployed_global_owner():
    scorer = _build_scorer(query_global=True)
    query_head = tuple(scorer.candidate_diagnostic_parameters())
    token = tuple(scorer.token_veto_parameters())
    global_absolute = tuple(scorer.global_absolute_parameters())
    active = tuple(scorer.confidence_parameters())

    query_ids = {id(parameter) for parameter in query_head}
    token_ids = {id(parameter) for parameter in token}
    global_ids = {id(parameter) for parameter in global_absolute}
    active_ids = {id(parameter) for parameter in active}
    assert len(query_head) == 6
    assert all(parameter.requires_grad for parameter in query_head)
    assert query_ids <= global_ids
    assert not query_ids & token_ids
    assert not token_ids & global_ids
    assert token_ids | global_ids == active_ids
    assert len(active) == 65


def test_v59_normalized_query_aggregate_is_translation_equivariant():
    scorer = _build_scorer(query_global=True)
    baseline, _inputs, _provider = _forward(scorer)
    with torch.no_grad():
        scorer.confidence_adapter.candidate_absolute_head[-1].bias.fill_(0.75)
    shifted, _inputs, _provider = _forward(scorer)

    baseline_global = _valid_values(baseline, "final_confidence_global_logits")
    shifted_global = _valid_values(shifted, "final_confidence_global_logits")
    assert torch.allclose(
        shifted_global - baseline_global,
        torch.full_like(baseline_global, 0.75),
        atol=2e-6,
        rtol=0.0,
    )


def test_v59_deployed_global_loss_updates_query_head_and_trunk_only():
    scorer = _build_scorer(query_global=True)
    with torch.no_grad():
        scorer.confidence_adapter.candidate_absolute_head[-1].weight.fill_(0.125)
        scorer.confidence_pool.residual[-1].weight.fill_(0.125)
    output, inputs, provider = _forward(scorer, requires_grad=True)
    valid = output["expression_valid_mask"]
    output["final_confidence_global_logits"][valid].sum().backward()

    assert any(
        parameter.grad is not None and bool(parameter.grad.ne(0).any().item())
        for parameter in scorer.candidate_diagnostic_parameters()
    )
    assert any(
        parameter.grad is not None and bool(parameter.grad.ne(0).any().item())
        for parameter in scorer.deployed_global_trunk_parameters()
    )
    assert all(parameter.grad is None for parameter in scorer.token_veto_parameters())
    assert all(parameter.grad is None for parameter in scorer.rank_parameters())
    assert inputs["candidate_hs"].grad is None
    assert inputs["candidate_boxes"].grad is None
    assert inputs["candidate_patch_logits"].grad is None
    assert all(value.grad is None for value in provider.leaves)
