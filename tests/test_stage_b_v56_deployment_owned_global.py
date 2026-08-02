from __future__ import annotations

import torch

from models.GroundingDINO.stage_b_dense_duty_scorer import (
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT,
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE,
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE,
    CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
    CONFIDENCE_POOL_FEATURE_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE,
    CONFIDENCE_POOL_FEATURE_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE,
    CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH,
    StageBDenseDutyScorer,
)
from tests.test_stage_b_dense_duty_scorer import _FakeGroundingDINO
from tests.test_stage_b_v54_exact_reference_scorer import _forward, _valid_values


def _build_scorer(*, deployment_owned: bool) -> StageBDenseDutyScorer:
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
            CONFIDENCE_POOL_FEATURE_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE
            if deployment_owned
            else CONFIDENCE_POOL_FEATURE_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE
        ),
        confidence_residual_parameterization_gain=0.25 / 0.03,
        confidence_gate_gradient_contract=(
            CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT
        ),
        confidence_head_gradient_contract=(
            CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE
            if deployment_owned
            else CONFIDENCE_HEAD_GRADIENT_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE
        ),
        phase="confidence",
    )
    scorer.train()
    return scorer


def test_v56_preserves_v55_state_and_deployed_u0_forward():
    v55 = _build_scorer(deployment_owned=False)
    v56 = _build_scorer(deployment_owned=True)

    v55_state = v55.state_dict()
    v56_state = v56.state_dict()
    assert tuple(v55_state) == tuple(v56_state)
    assert all(torch.equal(v55_state[name], v56_state[name]) for name in v55_state)

    v55_output, _inputs, _provider = _forward(v55)
    v56_output, _inputs, _provider = _forward(v56)
    for name in (
        "final_confidence_base_logits",
        "final_confidence_global_logits",
        "final_confidence_pool_absolute_logits",
    ):
        assert torch.equal(v55_output[name], v56_output[name])


def test_v56_candidate_head_is_frozen_and_outside_active_owners():
    scorer = _build_scorer(deployment_owned=True)
    candidate = tuple(scorer.candidate_diagnostic_parameters())
    token = tuple(scorer.token_veto_parameters())
    global_absolute = tuple(scorer.global_absolute_parameters())
    active = tuple(scorer.confidence_parameters())

    candidate_ids = {id(parameter) for parameter in candidate}
    token_ids = {id(parameter) for parameter in token}
    global_ids = {id(parameter) for parameter in global_absolute}
    active_ids = {id(parameter) for parameter in active}
    assert len(candidate) == 6
    assert all(not parameter.requires_grad for parameter in candidate)
    assert not candidate_ids & active_ids
    assert not token_ids & global_ids
    assert token_ids | global_ids == active_ids
    assert len(active) == 59


def test_v56_candidate_coordinate_is_detached_from_global_representation():
    scorer = _build_scorer(deployment_owned=True)
    output, _inputs, _provider = _forward(scorer)
    eligible = output["candidate_eligible_mask"]
    local = output["final_confidence_base_logits"][eligible]

    assert not local.requires_grad
    assert all(
        parameter.grad is None
        for parameter in scorer.deployed_global_trunk_parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in scorer.candidate_diagnostic_parameters()
    )


def test_v56_deployed_global_loss_updates_trunk_but_not_candidate_or_token():
    scorer = _build_scorer(deployment_owned=True)
    with torch.no_grad():
        scorer.confidence_pool.residual[-1].weight.fill_(0.125)
    output, inputs, provider = _forward(scorer, requires_grad=True)
    valid = output["expression_valid_mask"]
    deployed = _valid_values(output, "final_confidence_global_logits")
    assert torch.equal(
        deployed,
        _valid_values(output, "final_confidence_pool_absolute_logits"),
    )
    output["final_confidence_global_logits"][valid].sum().backward()

    trunk_gradients = [
        parameter.grad for parameter in scorer.deployed_global_trunk_parameters()
    ]
    assert any(
        gradient is not None and bool(gradient.ne(0).any().item())
        for gradient in trunk_gradients
    )
    assert all(
        parameter.grad is None
        for parameter in scorer.candidate_diagnostic_parameters()
    )
    assert all(parameter.grad is None for parameter in scorer.token_veto_parameters())
    assert all(parameter.grad is None for parameter in scorer.rank_parameters())
    assert inputs["candidate_hs"].grad is None
    assert inputs["candidate_boxes"].grad is None
    assert inputs["candidate_patch_logits"].grad is None
    assert all(value.grad is None for value in provider.leaves)
