import torch

from models.GroundingDINO.stage_b_dense_duty_scorer import (
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT,
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE,
    CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
    CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE,
    CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_REFERENCE,
    CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH,
    StageBDenseDutyScorer,
)
from tests.test_stage_b_dense_duty_scorer import (
    _ContextProvider,
    _FakeGroundingDINO,
)


def _build_scorer(pool_feature_contract):
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
        confidence_pool_feature_contract=pool_feature_contract,
        confidence_residual_parameterization_gain=0.25 / 0.03,
        confidence_gate_gradient_contract=(
            CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT
        ),
        confidence_head_gradient_contract=(
            CONFIDENCE_HEAD_GRADIENT_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE
        ),
        phase="confidence",
    )
    scorer.train()
    return scorer


def _forward(scorer, *, requires_grad=False):
    torch.manual_seed(7)
    batch_size, candidate_count, hidden_dim, slot_count = 2, 3, 4, 2
    candidate_boxes = torch.rand(batch_size, candidate_count, 4).clamp(
        0.05, 0.95
    )
    candidate_boxes.requires_grad_(requires_grad)
    score_mask = torch.zeros(batch_size, slot_count, 8, dtype=torch.bool)
    score_mask[:, :, 0] = True
    score_mask[1, 0].zero_()
    word_groups = torch.full(score_mask.shape, -1, dtype=torch.long)
    word_groups[score_mask] = 0
    inputs = {
        "candidate_hs": torch.randn(
            batch_size,
            candidate_count,
            hidden_dim,
            requires_grad=requires_grad,
        ),
        "candidate_boxes": candidate_boxes,
        "candidate_indices": torch.tensor(
            [[4, 0, 2], [5, 2, 1]], dtype=torch.long
        ),
        "candidate_patch_logits": torch.tensor(
            [[4.0, 3.0, 2.0], [2.0, 3.0, 1.0]],
            requires_grad=requires_grad,
        ),
        "expression_captions": [
            ["red person", "blue person"],
            ["left cup", "right cup"],
        ],
        "expression_valid_mask": torch.tensor(
            [[True, True], [True, False]]
        ),
        "expression_score_token_mask": score_mask,
        "expression_score_word_group_ids": word_groups,
    }
    provider = _ContextProvider()
    output = scorer(raw_context_provider=provider, **inputs)
    return output, inputs, provider


def _valid_values(output, name):
    return output[name][output["expression_valid_mask"]]


def test_v54_u0_reference_deployed_and_frozen_rank_max_are_exactly_equal():
    scorer = _build_scorer(
        CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_REFERENCE
    )
    output, _inputs, _provider = _forward(scorer)

    reference = _valid_values(
        output, "final_reference_global_confidence_logits"
    )
    deployed = _valid_values(output, "final_confidence_global_logits")
    frozen_rank_max = _valid_values(
        output, "final_frozen_rank_full_expression_global_logits"
    )
    delta = _valid_values(output, "final_confidence_delta_logits")

    assert torch.equal(reference, frozen_rank_max)
    assert torch.equal(deployed, reference)
    assert torch.equal(delta, torch.zeros_like(delta))


def test_v54_nonzero_candidate_and_pool_residuals_produce_exact_reference_delta():
    scorer = _build_scorer(
        CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_REFERENCE
    )
    with torch.no_grad():
        scorer.confidence_adapter.candidate_absolute_head[-1].bias.fill_(0.375)
        scorer.confidence_pool.residual[-1].bias.fill_(-0.125)

    output, _inputs, _provider = _forward(scorer)
    delta = _valid_values(output, "final_confidence_delta_logits")

    torch.testing.assert_close(
        delta,
        torch.full_like(delta, 0.25),
        rtol=0.0,
        atol=2e-7,
    )


def test_v54_changes_only_reference_contract_not_deployed_global_formula():
    v54 = _build_scorer(
        CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_REFERENCE
    )
    v53 = _build_scorer(
        CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE
    )
    assert {
        name: tuple(value.shape) for name, value in v54.state_dict().items()
    } == {
        name: tuple(value.shape) for name, value in v53.state_dict().items()
    }
    v53.load_state_dict(v54.state_dict(), strict=True)
    with torch.no_grad():
        torch.manual_seed(41)
        v54.confidence_adapter.candidate_absolute_head[-1].weight.normal_(
            mean=0.0, std=0.1
        )
        v54.confidence_adapter.candidate_absolute_head[-1].bias.fill_(0.2)
        v54.confidence_pool.residual[-1].weight.normal_(mean=0.0, std=0.1)
        v54.confidence_pool.residual[-1].bias.fill_(-0.05)
    v53.load_state_dict(v54.state_dict(), strict=True)

    v54_output, _inputs, _provider = _forward(v54)
    v53_output, _inputs, _provider = _forward(v53)
    eligible = v54_output["candidate_eligible_mask"]
    candidate_max = v54_output["final_confidence_base_logits"].masked_fill(
        ~eligible, -torch.inf
    ).max(dim=1).values
    expected = candidate_max + v54_output["final_validity_gate_logits"]
    valid = v54_output["expression_valid_mask"]

    torch.testing.assert_close(
        v54_output["final_confidence_global_logits"][valid],
        expected[valid],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        v54_output["final_confidence_global_logits"][valid],
        v53_output["final_confidence_global_logits"][valid],
        rtol=0.0,
        atol=0.0,
    )


def test_v54_global_loss_stays_in_global_owner_and_cannot_reach_rank():
    scorer = _build_scorer(
        CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_REFERENCE
    )
    output, inputs, provider = _forward(scorer, requires_grad=True)
    assert not output["final_reference_global_confidence_logits"].requires_grad

    valid = output["expression_valid_mask"]
    output["final_confidence_global_logits"][valid].sum().backward()

    candidate_bias_grad = (
        scorer.confidence_adapter.candidate_absolute_head[-1].bias.grad
    )
    pool_bias_grad = scorer.confidence_pool.residual[-1].bias.grad
    assert candidate_bias_grad is not None
    assert bool(candidate_bias_grad.ne(0).any().item())
    assert pool_bias_grad is not None
    assert bool(pool_bias_grad.ne(0).any().item())
    assert all(parameter.grad is None for parameter in scorer.token_veto_parameters())
    assert all(parameter.grad is None for parameter in scorer.rank_parameters())
    assert inputs["candidate_hs"].grad is None
    assert inputs["candidate_boxes"].grad is None
    assert inputs["candidate_patch_logits"].grad is None
    assert all(value.grad is None for value in provider.leaves)
