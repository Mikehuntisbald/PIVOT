from __future__ import annotations

import torch

from models.GroundingDINO.stage_b_dense_duty_scorer import (
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_DEPLOYED_ROUTING_ST,
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYED_ROUTER,
    CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
    CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC,
    StageBDenseDutyScorer,
)
from tests.test_stage_b_dense_duty_scorer import (
    _ContextProvider,
    _FakeGroundingDINO,
    StageBDenseDutyScorerTest,
)


def _build_scorer(last_n: int) -> StageBDenseDutyScorer:
    source = _FakeGroundingDINO(seed=20260813)
    scorer = StageBDenseDutyScorer(
        source.feat_map,
        source.transformer.encoder,
        source.transformer.decoder,
        source.transformer.level_embed,
        max_text_len=8,
        candidate_topk=3,
        category_gate_max_gap=100.0,
        confidence_adapter_dim=3,
        confidence_hidden_dim=7,
        confidence_pool_topk=2,
        confidence_phrase_aggregation=(
            CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
        ),
        confidence_gate_gradient_contract=(
            CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_DEPLOYED_ROUTING_ST
        ),
        confidence_pool_feature_contract=(
            CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC
        ),
        confidence_head_gradient_contract=(
            CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYED_ROUTER
        ),
        confidence_rank_decoder_unfreeze_last_n=last_n,
        phase="confidence",
    )
    scorer.train()
    return scorer


def test_v51_rank_decoder_adaptation_is_exact_tail_and_receives_gradient():
    scorer = _build_scorer(last_n=1)
    decoder_layers = list(scorer.rank_tower.decoder.layers)
    assert len(decoder_layers) >= 2
    assert not any(parameter.requires_grad for parameter in decoder_layers[-2].parameters())
    assert all(parameter.requires_grad for parameter in decoder_layers[-1].parameters())

    harness = StageBDenseDutyScorerTest()
    inputs = harness._inputs(requires_grad=True)
    score_mask = inputs["expression_score_token_mask"]
    word_groups = torch.full(score_mask.shape, -1, dtype=torch.long)
    word_groups[score_mask] = 0
    inputs["expression_score_word_group_ids"] = word_groups
    output = scorer(raw_context_provider=_ContextProvider(), **inputs)
    loss = output["final_confidence_global_logits"].float().sum()
    loss = loss + output["confidence_token_logits"].float().sum()
    loss.backward()

    adapted = tuple(scorer.confidence_rank_adaptation_parameters())
    assert adapted
    assert all(parameter.requires_grad for parameter in adapted)
    assert any(parameter.grad is not None for parameter in adapted)
    assert all(parameter.grad is None for parameter in decoder_layers[-2].parameters())
    assert inputs["candidate_hs"].grad is None


def test_v51_default_preserves_rank_stop_gradient_contract():
    scorer = _build_scorer(last_n=0)
    assert scorer.confidence_rank_adaptation_parameters() == ()
    assert not any(
        parameter.requires_grad for parameter in scorer.rank_tower.parameters()
    )
