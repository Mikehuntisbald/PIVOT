from __future__ import annotations

import math

import pytest
import torch

from engine import _build_stage_b_candidate_complete_trace_mask
from models.GroundingDINO.stage_b_dense_duty_scorer import (
    CONFIDENCE_CANDIDATE_TRACE_CONTRACT_FREE_HEAD_COVERAGE,
    CONFIDENCE_CANDIDATE_TRACE_CONTRACT_MONOTONE_TOKEN_ENTAILMENT,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT,
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE,
    CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
    CONFIDENCE_POOL_FEATURE_CONTRACT_DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE,
    CONFIDENCE_QUERY_VETO_MAX_DEPTH,
    CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH,
    StageBDenseDutyScorer,
    _word_normalized_softmin_probability,
)
from tests.test_stage_b_dense_duty_scorer import _FakeGroundingDINO
from tests.test_stage_b_v21_token_supervision import _token_criterion, _token_inputs
from tests.test_stage_b_v54_exact_reference_scorer import _forward


def _build_scorer(contract: str) -> StageBDenseDutyScorer:
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
        confidence_candidate_trace_contract=contract,
        confidence_token_depth_base_scale=1.0,
        phase="confidence",
    )
    scorer.copy_confidence_verifier_from_rank()
    scorer.train()
    return scorer


def _candidate_complete_inputs(*, scope_code: int, edit_mask: torch.Tensor):
    inputs = _token_inputs()
    inputs.update(
        {
            "global_tn_verified": torch.tensor([True]),
            "confidence_tn_train_eligible": torch.tensor([True]),
            "global_tn_candidate_mask": torch.tensor([[True, True]]),
            "token_edit_candidate_mask": edit_mask,
            "token_trace_scope_codes": torch.tensor([scope_code]),
            "score_word_group_ids": torch.tensor(
                [[[0, 1, 1, -1], [0, 1, 1, -1]]], dtype=torch.long
            ),
        }
    )
    return inputs


def test_c1_exposes_exact_per_candidate_depth_without_changing_deployment():
    scorer = _build_scorer(
        CONFIDENCE_CANDIDATE_TRACE_CONTRACT_FREE_HEAD_COVERAGE
    )
    with torch.no_grad():
        scorer.confidence_verifier_veto_head[-1].bias.fill_(3.0)
    output, _inputs, _provider = _forward(scorer, requires_grad=True)
    depth = output["final_deployed_candidate_veto_depth"]
    eligible = output["candidate_eligible_mask"]
    assert depth.shape == eligible.shape
    assert torch.count_nonzero(depth[~eligible]) == 0
    assert torch.all(depth[eligible] > 0.0)
    assert torch.allclose(
        output["final_confidence_global_logits"][output["expression_valid_mask"]],
        -output["final_deployed_query_veto_depth"][
            output["expression_valid_mask"]
        ],
        atol=1e-6,
        rtol=0.0,
    )


def test_c2_depth_is_monotone_token_entailment_and_free_head_is_dormant():
    scorer = _build_scorer(
        CONFIDENCE_CANDIDATE_TRACE_CONTRACT_MONOTONE_TOKEN_ENTAILMENT
    )
    head_parameters = list(scorer.confidence_verifier_veto_head.parameters())
    active_ids = {id(parameter) for parameter in scorer.confidence_parameters()}
    assert all(id(parameter) not in active_ids for parameter in head_parameters)
    assert all(not parameter.requires_grad for parameter in head_parameters)
    assert scorer.expected_live_confidence_parameter_tensor_counts() == {
        "token_veto": len(scorer.token_veto_parameters())
    }

    before, inputs, provider = _forward(scorer, requires_grad=True)
    with torch.no_grad():
        scorer.confidence_verifier_veto_head[-1].weight.fill_(1000.0)
        scorer.confidence_verifier_veto_head[-1].bias.fill_(-1000.0)
    after = scorer(raw_context_provider=provider, **inputs)
    assert torch.equal(
        before["final_deployed_candidate_veto_depth"],
        after["final_deployed_candidate_veto_depth"],
    )
    assert torch.equal(
        before["final_confidence_global_logits"],
        after["final_confidence_global_logits"],
    )

    mismatch = after["final_confidence_mismatch_probability"]
    expected_depth = CONFIDENCE_QUERY_VETO_MAX_DEPTH * torch.tanh(
        mismatch / CONFIDENCE_QUERY_VETO_MAX_DEPTH
    )
    expected_depth = expected_depth.masked_fill(
        ~after["candidate_eligible_mask"], 0.0
    )
    assert torch.allclose(
        after["final_deployed_candidate_veto_depth"],
        expected_depth,
        atol=1e-6,
        rtol=0.0,
    )

    after["final_confidence_global_logits"][
        after["expression_valid_mask"]
    ].sum().backward()
    assert all(parameter.grad is None for parameter in head_parameters)
    assert any(
        parameter.grad is not None
        for parameter in scorer.confidence_verifier_tower.parameters()
    )


@pytest.mark.parametrize(
    "surface", ("entailment_probability_layers", "mismatch_probability_layers")
)
def test_c2_entailment_surface_is_fail_closed(monkeypatch, surface):
    scorer = _build_scorer(
        CONFIDENCE_CANDIDATE_TRACE_CONTRACT_MONOTONE_TOKEN_ENTAILMENT
    )
    build_output = scorer._build_full_decoder_verifier_output

    def without_required_surface(**kwargs):
        output = build_output(**kwargs)
        output.pop(surface)
        return output

    monkeypatch.setattr(
        scorer, "_build_full_decoder_verifier_output", without_required_surface
    )
    with pytest.raises(RuntimeError, match="requires an explicit finite"):
        _forward(scorer, requires_grad=True)


def test_c2_mismatch_stays_positive_and_differentiable_for_high_entailment_logits():
    logits = torch.full((1, 1, 3, 2), 20.0, requires_grad=True)
    residuals = torch.zeros_like(logits)
    _entailment, _gate, mismatch = _word_normalized_softmin_probability(
        logits,
        residuals,
        torch.tensor([[True, True]]),
        torch.tensor([[0, 1]], dtype=torch.long),
        temperature=0.1,
        gate_scale=1.0,
        return_stable_mismatch=True,
    )
    assert torch.all(mismatch > 0.0)
    mismatch.sum().backward()
    assert logits.grad is not None
    assert torch.all(logits.grad < 0.0)


def test_trace_scope_builder_is_fail_closed_and_uses_original_query_indices():
    candidates = torch.tensor([[5, 8, 13], [2, 3, 7]], dtype=torch.long)
    deployed = torch.tensor([[True, True, False], [True, True, True]])
    targets = [
        {
            "stage_b_candidate_trace_scope": "global_word_absent",
            "stage_b_changed_word_global_absent_verified": torch.tensor(True),
        },
        {
            "stage_b_candidate_trace_scope": "candidate_verified",
            "stage_b_changed_word_candidate_verified_indices": torch.tensor([3, 99]),
        },
    ]
    mask, scopes = _build_stage_b_candidate_complete_trace_mask(
        targets,
        candidate_indices=candidates,
        deployed_tn_mask=deployed,
        provenance_gate=torch.tensor([True, True]),
    )
    assert torch.equal(mask, torch.tensor([[True, True, False], [False, True, False]]))
    assert torch.equal(scopes, torch.tensor([1, 2]))

    mask, scopes = _build_stage_b_candidate_complete_trace_mask(
        targets,
        candidate_indices=candidates,
        deployed_tn_mask=deployed,
        provenance_gate=torch.tensor([False, False]),
    )
    assert torch.count_nonzero(mask) == 0
    assert torch.count_nonzero(scopes) == 0


def test_candidate_complete_changed_tokens_cover_only_provenance_approved_queries():
    criterion = _token_criterion(
        "edit_bce_group_balanced",
        token_edit_query_scope="candidate_complete_trace_v4",
    )
    inputs = _candidate_complete_inputs(
        scope_code=1, edit_mask=torch.tensor([[True, True]])
    )
    result = criterion(**inputs)
    result["loss_stage_b_fixed_text"].backward()
    assert float(result["fixed_text_token_edit_count"]) == 2.0
    assert float(result["fixed_text_token_trace_broadcast_candidate_count"]) == 2.0
    assert inputs["token_logits"].grad[0, :, 1, 0].gt(0.0).all()
    assert torch.count_nonzero(inputs["token_logits"].grad[0, 1, 1, 1:]) == 0

    bad = _candidate_complete_inputs(
        scope_code=1, edit_mask=torch.tensor([[True, False]])
    )
    with pytest.raises(ValueError, match="every deployed candidate"):
        criterion(**bad)

    bad = _candidate_complete_inputs(
        scope_code=0, edit_mask=torch.tensor([[False, True]])
    )
    with pytest.raises(ValueError, match="expression-only"):
        criterion(**bad)


def test_candidate_depth_losses_hit_all_tn_escapes_and_only_target_positive():
    criterion = _token_criterion(
        "edit_bce_group_balanced",
        token_edit_query_scope="candidate_complete_trace_v4",
        token_weight=0.0,
        candidate_depth_all_weight=1.0,
        candidate_depth_escape_weight=1.0,
        candidate_depth_positive_weight=1.0,
    )
    inputs = _candidate_complete_inputs(
        scope_code=0, edit_mask=torch.tensor([[False, False]])
    )
    depth = torch.tensor(
        [[[0.20, 0.00], [0.80, 1.00]]], requires_grad=True
    )
    inputs["candidate_veto_depth"] = depth
    result = criterion(**inputs)
    result["loss_stage_b_fixed_text"].backward()
    assert float(result["fixed_text_candidate_depth_tn_query_count"]) == 2.0
    assert float(result["fixed_text_candidate_depth_positive_query_count"]) == 1.0
    assert depth.grad[0, 0, 0] > 0.0
    assert depth.grad[0, 1, 0] == 0.0
    assert depth.grad[0, :, 1].lt(0.0).all()
    assert abs(float(result["fixed_text_candidate_depth_tn_min_mean"])) < 1e-8


def test_candidate_complete_token_loss_is_wordpiece_invariant():
    criterion = _token_criterion(
        "edit_bce_group_balanced",
        token_edit_query_scope="candidate_complete_trace_v4",
        token_shared_weight=0.0,
        token_edit_weight=0.0,
    )

    single = _candidate_complete_inputs(
        scope_code=0, edit_mask=torch.tensor([[False, False]])
    )
    duplicate = _candidate_complete_inputs(
        scope_code=0, edit_mask=torch.tensor([[False, False]])
    )
    for inputs in (single, duplicate):
        with torch.no_grad():
            inputs["token_logits"][0, 0, 0, 0:2] = 2.0
            inputs["token_logits"][0, 0, 0, 2] = -2.0
    single["token_positive_mask"][0, 0] = torch.tensor(
        [True, False, True, False]
    )
    single["score_word_group_ids"][0, 0] = torch.tensor([0, 2, 1, -1])
    duplicate["token_positive_mask"][0, 0] = torch.tensor(
        [True, True, True, False]
    )
    duplicate["score_word_group_ids"][0, 0] = torch.tensor([0, 0, 1, -1])

    single_loss = criterion(**single)["loss_fixed_text_token"]
    duplicate_loss = criterion(**duplicate)["loss_fixed_text_token"]
    assert math.isfinite(float(single_loss.detach()))
    assert torch.allclose(single_loss, duplicate_loss, atol=1e-7, rtol=0.0)
