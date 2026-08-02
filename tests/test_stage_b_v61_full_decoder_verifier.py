from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

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
from tools import eval_refcoco_stageb as ref_eval
from tools import eval_text_groundingdino_refcoco_tn as combined_eval
from tools import run_stageb_confidence_full_decoder_verifier_probe_u0400 as training
from util.slconfig import SLConfig
from util.stage_b_dense_duty_audit import fingerprint_state


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
        phase="confidence",
    )
    scorer.copy_confidence_verifier_from_rank()
    scorer.train()
    return scorer


def test_v61_clones_complete_rank_tower_but_keeps_parameter_ownership_independent():
    scorer = _build()
    rank_state = scorer.rank_tower.state_dict()
    verifier_state = scorer.confidence_verifier_tower.state_dict()
    assert tuple(rank_state) == tuple(verifier_state)
    assert all(
        torch.equal(rank_state[name], verifier_state[name]) for name in rank_state
    )
    rank_ids = {id(parameter) for parameter in scorer.rank_parameters()}
    confidence_ids = {id(parameter) for parameter in scorer.confidence_parameters()}
    assert rank_ids
    assert confidence_ids
    assert not rank_ids & confidence_ids
    assert all(not parameter.requires_grad for parameter in scorer.rank_parameters())
    assert all(parameter.requires_grad for parameter in scorer.confidence_parameters())
    root = nn.Module()
    root.add_module("stage_b_fixed_text_scorer", scorer)
    active_ids = {id(parameter) for parameter in scorer.confidence_parameters()}
    active_names = [
        name
        for name, parameter in root.named_parameters()
        if id(parameter) in active_ids
    ]
    fingerprint = fingerprint_state(
        root.state_dict(),
        active_parameter_names=active_names,
        phase="confidence",
    )
    assert fingerprint["active"]["tensor_count"] == len(active_names)


def test_v61_u0_inherits_token_entailment_and_emits_exactly_zero_veto():
    scorer = _build()
    output, _inputs, _provider = _forward(scorer)
    assert torch.equal(
        output["final_confidence_token_logits"],
        output["final_rank_token_logits"],
    )
    assert torch.count_nonzero(output["final_deployed_query_veto_depth"]) == 0
    assert torch.equal(
        _valid_values(output, "final_confidence_global_logits"),
        _valid_values(output, "final_confidence_pool_absolute_logits"),
    )
    assert torch.count_nonzero(
        scorer.confidence_verifier_veto_head[-1].weight
    ) == 0
    assert torch.count_nonzero(
        scorer.confidence_verifier_veto_head[-1].bias
    ) == 0


def test_v61_verifier_only_lowers_the_absolute_pool_and_never_updates_rank():
    scorer = _build()
    with torch.no_grad():
        scorer.confidence_verifier_veto_head[-1].bias.fill_(3.0)
    output, _inputs, _provider = _forward(scorer, requires_grad=True)
    global_logit = _valid_values(output, "final_confidence_global_logits")
    pool_logit = _valid_values(output, "final_confidence_pool_absolute_logits")
    depth = _valid_values(output, "final_deployed_query_veto_depth")
    assert torch.all(depth >= 0.0)
    assert torch.all(global_logit <= pool_logit)
    assert torch.allclose(global_logit, pool_logit - depth, atol=1e-6, rtol=0.0)

    output["final_confidence_global_logits"][
        output["expression_valid_mask"]
    ].sum().backward()
    assert all(parameter.grad is None for parameter in scorer.rank_parameters())
    assert any(
        parameter.grad is not None
        for parameter in scorer.confidence_verifier_veto_head.parameters()
    )


def _diagnostic_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        partial_dense_duty_rank_diagnostic=False,
        partial_dense_duty_confidence_diagnostic=True,
        config=str(training.CONFIG),
        output_dir=str(tmp_path / "v61-strict1607"),
        ckpts=["full-decoder-verifier-u400-checkpoint.pth"],
        tn_jsonl=str(
            combined_eval._PARTIAL_CONFIDENCE_TN_SPECS["strict1607"]["path"]
        ),
        tn_splits=["refcocop_val", "refcocog_umd_val"],
        skip_tn=False,
        skip_ref=True,
        device="cuda:0",
        batch_size=16,
        num_workers=4,
        seed=42,
        amp=True,
        topk=[1],
        threshold_tprs=[0.75, 0.9, 0.95],
        score_thresholds=[0.5],
        max_ref_batches=0,
        max_tn_batches=0,
        no_per_example_records=False,
        screen_calibration_manifest=False,
        direct_prebuilt_tn=False,
        category_gate_max_gaps=None,
        category_gate_include_base_expert=False,
        candidate_count_control=0,
        holdout_level="none",
        exclude_train_jsonl=[],
    )


def test_v61_is_distinct_from_v60_and_registered_by_combined_evaluator(tmp_path):
    cfg = SLConfig.fromfile(str(training.CONFIG))
    assert not ref_eval._validate_v60_deployment_owned_query_veto_config(cfg)
    assert ref_eval._validate_v61_full_decoder_verifier_config(cfg)
    combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
        _diagnostic_args(tmp_path), cfg
    )


def test_v61_runtime_requires_exact_full_verifier_owner_counts():
    runtime = {
        "clip_contract_schema": ref_eval._V56_TWO_OWNER_CLIP_CONTRACT_SCHEMA,
        "clip_contract_checked_steps": 400,
        "owner_clip_violation_steps": 0,
        "active_pre_decomposition_violation_steps": 0,
        "active_post_decomposition_violation_steps": 0,
        "live_tensor_count_violation_steps": 0,
        "active_monotonic_violation_steps": 0,
        "clip_contract_tolerance": 1e-6,
        "clip_contract_max_norm": 0.1,
        "max_active_pre_decomposition_residual": 0.0,
        "max_active_post_decomposition_residual": 0.0,
        "max_owner_clip_residual": 0.0,
        "max_active_monotonic_residual": 0.0,
    }
    for owner, count in (("token_veto", 356), ("global_absolute", 12)):
        runtime[f"last_{owner}_grad_norm_preclip"] = 1.0
        runtime[f"max_{owner}_grad_norm_preclip"] = 1.0
        runtime[f"expected_{owner}_tensor_count"] = count
        runtime[f"last_observed_{owner}_tensor_count"] = count
    ref_eval._validate_v61_two_owner_runtime_audit(
        runtime, optimizer_updates=400
    )
    runtime["last_observed_token_veto_tensor_count"] = 355
    with pytest.raises(RuntimeError, match="v61 confidence checkpoint"):
        ref_eval._validate_v61_two_owner_runtime_audit(
            runtime, optimizer_updates=400
        )
