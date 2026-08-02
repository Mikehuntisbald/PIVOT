from types import SimpleNamespace

import pytest

import main as training_main
from tools import run_stageb_confidence_adapter_veto_probe_evaluation as promotion
from tools import (
    run_stageb_confidence_adapter_veto_gate_probe_evaluation as gate_promotion,
)
from tools import (
    run_stageb_confidence_adapter_veto_gated_pool_rank_evidence_probe_evaluation as rank_evidence_promotion,
)


def _args(
    *,
    scope: str,
    contract: str,
    report: str,
    aggregation: str = "trace_activated_word_veto_product_v1",
    revision: str = "word_veto_net_trust_v1",
):
    return SimpleNamespace(
        stage_b_dense_duty=True,
        stage_b_v22_score_ownership=(
            "rank_tower_stopgrad_token_adapter_two_phase"
        ),
        stage_b_dense_duty_confidence_phrase_aggregation=aggregation,
        stage_b_dense_duty_confidence_revision=revision,
        stage_b_dense_duty_execution_scope=scope,
        stage_b_dense_duty_confidence_probe_admission_contract=contract,
        stage_b_dense_duty_confidence_probe_admission_report=report,
    )


def _v11_validation_args(**overrides):
    values = {
        "stage_b_dense_duty": True,
        "stage_b_dense_duty_phase": "confidence",
        "stage_b_dense_duty_no_stageb_teacher": True,
        "stage_b_v11_num_layers": 6,
        "stage_b_v11_candidate_topk": 50,
        "stage_b_v15_patch_rank_fusion": False,
        "stage_b_v15_exclude_canonical_from_score": True,
        "stage_b_v21_token_objective": "edit_bce",
        "stage_b_v21_allow_legacy_token_diff_fallback": False,
        "stage_b_dense_duty_allow_incidental_trace_edits": False,
        "stage_b_dense_duty_token_role_source": "exact_direct_trace_v1",
        "stage_b_v22_score_ownership": (
            "rank_tower_stopgrad_token_adapter_two_phase"
        ),
        "stage_b_dense_duty_confidence_adapter_dim": 64,
        "stage_b_dense_duty_confidence_init_seed": 42,
        "stage_b_dense_duty_confidence_token_contract": (
            "detached_rank_token_minus_zero_init_residual_v1"
        ),
        "stage_b_dense_duty_confidence_pool_feature_contract": (
            "patch_statistics_only_v1"
        ),
        "stage_b_dense_duty_confidence_revision": (
            "word_veto_gated_pool_rank_evidence_v11"
        ),
        "stage_b_dense_duty_confidence_rank_evidence_contract": (
            "zero_init_rank_logit_scale_v1"
        ),
        "stage_b_dense_duty_confidence_phrase_aggregation": (
            "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        ),
        "stage_b_dense_duty_positive_trust_contract": (
            "net_total_confidence_delta_v1"
        ),
        "stage_b_dense_duty_confidence_tn_scope": "direct_trace_valid_v1",
        "stage_b_dense_duty_confidence_word_softmin_temperature": 0.1,
        "stage_b_dense_duty_confidence_veto_gate_scale": 0.03,
        "stage_b_dense_duty_confidence_veto_gate_offset": 0.02,
        "stage_b_dense_duty_raw_veto_gate_weight": 1.0,
        "stage_b_dense_duty_raw_veto_positive_margin": 0.1,
        "stage_b_dense_duty_raw_veto_tn_margin": 0.15,
        "stage_b_dense_duty_raw_veto_query_scope": (
            "tn_all_admitted_carrier_balanced_positive_carrier_paired_v4"
        ),
        "stage_b_dense_duty_raw_veto_tn_carrier_balance": 0.25,
        "stage_b_dense_duty_confidence_carrier_selector_contract": (
            "final_layer_reference_argmax_exact_eligible_v1"
        ),
        "stage_b_dense_duty_raw_veto_carrier_pair_weight": 0.25,
        "stage_b_dense_duty_raw_veto_carrier_pair_margin": 0.25,
        "stage_b_dense_duty_raw_veto_positive_carrier_balance": 0.0,
        "stage_b_dense_duty_confidence_veto_coverage_offset": 0.1,
        "stage_b_dense_duty_confidence_veto_coverage_ramp": 0.8,
        "stage_b_dense_duty_confidence_veto_cap_temperature": 0.1,
        "stage_b_dense_duty_confidence_veto_cap_initial_ceiling": -0.1,
        "stage_b_v11_global_tn_negative_weight": 1.0,
        "stage_b_v11_global_tn_tail_weight": 0.0,
        "stage_b_v15_tail_queue_global_scores": True,
        "stage_b_v15_tail_queue_objective": "fpr95",
        "stage_b_dense_duty_execution_scope": "probe",
        "stage_b_dense_duty_confidence_probe_admission_contract": (
            "disabled_for_probe_v1"
        ),
        "stage_b_dense_duty_confidence_probe_admission_report": "",
        "stage_b_dense_duty_confidence_probe_admission_audit": None,
        "finetune_ignore": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_u300_probe_disables_the_circular_formal_admission_gate():
    args = _args(
        scope="probe",
        contract="disabled_for_probe_v1",
        report="",
    )
    assert training_main._bind_stage_b_confidence_probe_admission(args) is None
    assert not hasattr(
        args, "stage_b_dense_duty_confidence_probe_admission_audit"
    )


def test_formal_training_binds_verified_admission_into_saved_args(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    report = tmp_path / "admission.json"
    report.write_text("{}\n", encoding="ascii")
    audit = {
        "status": "verified",
        "decision": "admit_to_formal_training",
        "formal_training_admitted": True,
        "diagnostic_only": True,
    }
    monkeypatch.setattr(promotion, "REPORT", report)
    monkeypatch.setattr(
        promotion,
        "verify_admission_report",
        lambda path: audit if path == report.resolve() else None,
    )
    args = _args(
        scope="formal",
        contract="u300_word_veto_strict1607_v1",
        report=str(report),
    )

    assert training_main._bind_stage_b_confidence_probe_admission(args) == audit
    assert (
        args.stage_b_dense_duty_confidence_probe_admission_audit == audit
    )


def test_formal_training_rejects_noncanonical_admission_report(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    canonical = tmp_path / "canonical.json"
    other = tmp_path / "other.json"
    canonical.write_text("{}\n", encoding="ascii")
    other.write_text("{}\n", encoding="ascii")
    monkeypatch.setattr(promotion, "REPORT", canonical)
    args = _args(
        scope="formal",
        contract="u300_word_veto_strict1607_v1",
        report=str(other),
    )
    with pytest.raises(RuntimeError, match="noncanonical promotion report"):
        training_main._bind_stage_b_confidence_probe_admission(args)


def test_v3_formal_training_binds_its_own_verified_admission(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    report = tmp_path / "gate-admission.json"
    report.write_text("{}\n", encoding="ascii")
    audit = {
        "status": "verified",
        "decision": "admit_to_formal_training",
        "formal_training_admitted": True,
        "diagnostic_only": True,
    }
    monkeypatch.setattr(gate_promotion, "REPORT", report)
    monkeypatch.setattr(
        gate_promotion,
        "verify_admission_report",
        lambda path: audit if path == report.resolve() else None,
    )
    args = _args(
        scope="formal",
        contract="u300_word_veto_gate_strict1607_v3",
        report=str(report),
        aggregation="trace_activated_word_veto_penalty_v2",
        revision="word_veto_raw_gate_margin_v3",
    )

    assert training_main._bind_stage_b_confidence_probe_admission(args) == audit
    assert args.stage_b_dense_duty_confidence_probe_admission_audit == audit


def test_v3_formal_training_rejects_v1_admission_contract(tmp_path):
    report = tmp_path / "gate-admission.json"
    report.write_text("{}\n", encoding="ascii")
    args = _args(
        scope="formal",
        contract="u300_word_veto_strict1607_v1",
        report=str(report),
        aggregation="trace_activated_word_veto_penalty_v2",
        revision="word_veto_raw_gate_margin_v3",
    )
    with pytest.raises(RuntimeError, match="U300 strict1607 promotion contract"):
        training_main._bind_stage_b_confidence_probe_admission(args)


def test_v11_formal_training_binds_rank_evidence_promotion(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    report = tmp_path / "rank-evidence-admission.json"
    report.write_text("{}\n", encoding="ascii")
    audit = {
        "status": "verified",
        "decision": "admit_to_formal_training",
        "formal_training_admitted": True,
        "diagnostic_only": True,
    }
    monkeypatch.setattr(rank_evidence_promotion, "REPORT", report)
    monkeypatch.setattr(
        rank_evidence_promotion,
        "verify_admission_report",
        lambda path: audit if path == report.resolve() else None,
    )
    args = _args(
        scope="formal",
        contract="u300_word_veto_gated_pool_rank_evidence_strict1607_v11",
        report=str(report),
        aggregation="trace_activated_word_veto_gated_pool_absolute_cap_v5",
        revision="word_veto_gated_pool_rank_evidence_v11",
    )

    assert training_main._bind_stage_b_confidence_probe_admission(args) == audit
    assert args.stage_b_dense_duty_confidence_probe_admission_audit == audit


def test_v11_formal_training_rejects_v9_admission_contract(tmp_path):
    report = tmp_path / "rank-evidence-admission.json"
    report.write_text("{}\n", encoding="ascii")
    args = _args(
        scope="formal",
        contract="u300_word_veto_gated_pool_carrier_pair_strict1607_v9",
        report=str(report),
        aggregation="trace_activated_word_veto_gated_pool_absolute_cap_v5",
        revision="word_veto_gated_pool_rank_evidence_v11",
    )
    with pytest.raises(RuntimeError, match="U300 strict1607 promotion contract"):
        training_main._bind_stage_b_confidence_probe_admission(args)


def test_v11_main_validation_accepts_only_its_exact_parameter_group():
    with pytest.raises(RuntimeError, match="exact Stage-A base path/SHA"):
        training_main._validate_stage_b_dense_duty_args(
            _v11_validation_args()
        )

    drift_cases = (
        (
            "stage_b_dense_duty_confidence_rank_evidence_contract",
            "off_v1",
            "rank-evidence v11",
        ),
        (
            "stage_b_dense_duty_raw_veto_query_scope",
            "tn_all_admitted_carrier_balanced_positive_carrier_v3",
            "raw supervision scope",
        ),
        (
            "stage_b_dense_duty_raw_veto_tn_carrier_balance",
            0.5,
            "exact 0.25",
        ),
        (
            "stage_b_dense_duty_raw_veto_carrier_pair_weight",
            0.5,
            "weight=0.25",
        ),
        (
            "stage_b_dense_duty_raw_veto_carrier_pair_margin",
            0.5,
            "margin=0.25",
        ),
        (
            "stage_b_dense_duty_raw_veto_positive_carrier_balance",
            0.25,
            "forbid positive carrier",
        ),
    )
    for field, value, message in drift_cases:
        with pytest.raises(RuntimeError, match=message):
            training_main._validate_stage_b_dense_duty_args(
                _v11_validation_args(**{field: value})
            )


def test_pre_v11_revision_forbids_rank_evidence_contract():
    args = SimpleNamespace()
    assert (
        training_main._validate_stage_b_confidence_rank_evidence_contract(
            args,
            revision="word_veto_gated_pool_carrier_pair_v9",
        )
        == "off_v1"
    )
    args.stage_b_dense_duty_confidence_rank_evidence_contract = (
        "zero_init_rank_logit_scale_v1"
    )
    with pytest.raises(RuntimeError, match="pre-v11"):
        training_main._validate_stage_b_confidence_rank_evidence_contract(
            args,
            revision="word_veto_gated_pool_carrier_pair_v9",
        )


def test_v12_rank_affine_requires_its_exact_contract():
    args = SimpleNamespace(
        stage_b_dense_duty_confidence_rank_evidence_contract=(
            "zero_init_rank_logit_affine_v2"
        )
    )
    assert training_main._validate_stage_b_confidence_rank_evidence_contract(
        args,
        revision="word_veto_gated_pool_rank_affine_v12",
    ) == "zero_init_rank_logit_affine_v2"
    args.stage_b_dense_duty_confidence_rank_evidence_contract = (
        "zero_init_rank_logit_scale_v1"
    )
    with pytest.raises(RuntimeError, match="rank-affine v12"):
        training_main._validate_stage_b_confidence_rank_evidence_contract(
            args,
            revision="word_veto_gated_pool_rank_affine_v12",
        )


def test_v19_sparse_rank_channel_requires_its_exact_contract_and_gain():
    args = SimpleNamespace(
        stage_b_dense_duty_confidence_rank_evidence_contract=(
            "zero_init_carrier_token_rank_affine_sparse_rank_channel_v6"
        ),
        stage_b_dense_duty_confidence_residual_parameterization_gain=(0.25 / 0.03),
        stage_b_dense_duty_confidence_gate_gradient_contract="hard_detached_v1",
    )
    assert training_main._validate_stage_b_confidence_rank_evidence_contract(
        args,
        revision="word_veto_gated_pool_tail_paired_rank_channel_v19",
    ) == "zero_init_carrier_token_rank_affine_sparse_rank_channel_v6"

    args.stage_b_dense_duty_confidence_rank_evidence_contract = (
        "zero_init_carrier_token_rank_affine_v5"
    )
    with pytest.raises(RuntimeError, match="rank-channel v19"):
        training_main._validate_stage_b_confidence_rank_evidence_contract(
            args,
            revision="word_veto_gated_pool_tail_paired_rank_channel_v19",
        )


def test_v20_main_validation_requires_its_signed_rank_query_pool_contract():
    args = _v11_validation_args(
        stage_b_dense_duty_confidence_revision=(
            "word_veto_gated_pool_tail_paired_signed_rank_pool_v20"
        ),
        stage_b_dense_duty_confidence_pool_feature_contract=(
            "detached_rank_query_plus_patch_statistics_signed_residual_v2"
        ),
        stage_b_dense_duty_confidence_rank_evidence_contract="off_v1",
    )

    # The signed pool is accepted first, so validation advances to the
    # deliberately invalid rank-evidence contract.
    with pytest.raises(RuntimeError, match="rank-channel v19"):
        training_main._validate_stage_b_dense_duty_args(args)

    args.stage_b_dense_duty_confidence_pool_feature_contract = (
        "patch_statistics_only_v1"
    )
    with pytest.raises(RuntimeError, match="pool-feature contract"):
        training_main._validate_stage_b_dense_duty_args(args)


def test_v20_signed_rank_pool_requires_exact_sparse_rank_contract_and_gain():
    args = SimpleNamespace(
        stage_b_dense_duty_confidence_rank_evidence_contract=(
            "zero_init_carrier_token_rank_affine_sparse_rank_channel_v6"
        ),
        stage_b_dense_duty_confidence_residual_parameterization_gain=(0.25 / 0.03),
        stage_b_dense_duty_confidence_gate_gradient_contract="hard_detached_v1",
    )
    revision = "word_veto_gated_pool_tail_paired_signed_rank_pool_v20"

    assert training_main._validate_stage_b_confidence_rank_evidence_contract(
        args,
        revision=revision,
    ) == "zero_init_carrier_token_rank_affine_sparse_rank_channel_v6"

    args.stage_b_dense_duty_confidence_rank_evidence_contract = (
        "zero_init_carrier_token_rank_affine_v5"
    )
    with pytest.raises(RuntimeError, match="rank-channel v19"):
        training_main._validate_stage_b_confidence_rank_evidence_contract(
            args,
            revision=revision,
        )

    args.stage_b_dense_duty_confidence_rank_evidence_contract = (
        "zero_init_carrier_token_rank_affine_sparse_rank_channel_v6"
    )
    args.stage_b_dense_duty_confidence_residual_parameterization_gain = 1.0
    with pytest.raises(RuntimeError, match="residual gain=0.25/0.03"):
        training_main._validate_stage_b_confidence_rank_evidence_contract(
            args,
            revision=revision,
        )
