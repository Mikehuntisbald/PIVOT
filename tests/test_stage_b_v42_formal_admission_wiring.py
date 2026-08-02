from pathlib import Path
from types import SimpleNamespace

import pytest

import main as training_main
import tools as tools_package
from tools import (
    run_stageb_confidence_adapter_candidate_tn_only_carrier_pair_probe_u0400 as probe,
)
from util.stage_b_dense_duty_audit import build_training_contract


V39_CONTRACT = (
    "u400_word_veto_candidate_gate_zero_offset_confidence_strict1607_v39"
)
V42_CONTRACT = (
    "u400_word_veto_candidate_tn_only_carrier_pair_confidence_strict1607_v42"
)
V42_PROMOTION_MODULE = (
    "run_stageb_confidence_adapter_candidate_tn_only_carrier_pair_"
    "probe_evaluation"
)
V39_PROMOTION_MODULE = (
    "run_stageb_confidence_adapter_candidate_gate_zero_offset_probe_evaluation"
)


def _args(
    *,
    gradient_contract: str,
    admission_contract: str,
    report: Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        stage_b_dense_duty=True,
        stage_b_v22_score_ownership=(
            "rank_tower_stopgrad_token_adapter_two_phase"
        ),
        stage_b_dense_duty_confidence_phrase_aggregation=(
            "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        ),
        stage_b_dense_duty_confidence_revision=(
            "word_veto_candidate_asymmetric_confidence_v32"
        ),
        stage_b_v15_tail_queue_positive_gradient_contract=(
            "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
        ),
        stage_b_dense_duty_confidence_veto_gate_offset=0.0,
        stage_b_v21_token_edit_query_scope="target_iou_v1",
        stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract=(
            gradient_contract
        ),
        stage_b_dense_duty_execution_scope="formal",
        stage_b_dense_duty_confidence_probe_admission_contract=(
            admission_contract
        ),
        stage_b_dense_duty_confidence_probe_admission_report=str(report),
    )


def _promotion(report: Path, audit: dict) -> SimpleNamespace:
    return SimpleNamespace(
        REPORT=report,
        verify_admission_report=lambda path: (
            audit if path == report.resolve(strict=True) else None
        ),
    )


def test_target_iou_gradient_route_selects_v42_before_v39():
    report = Path("unused.json")
    assert training_main._stage_b_target_iou_carrier_pair_admission_contract(
        _args(
            gradient_contract="tn_only_positive_detached_v2",
            admission_contract=V42_CONTRACT,
            report=report,
        )
    ) == V42_CONTRACT
    assert training_main._stage_b_target_iou_carrier_pair_admission_contract(
        _args(
            gradient_contract="bidirectional_v1",
            admission_contract=V39_CONTRACT,
            report=report,
        )
    ) == V39_CONTRACT


@pytest.mark.parametrize(
    ("scope", "gradient_contract", "message"),
    (
        (
            "target_iou_union_detached_final_confidence_base_argmax_v2",
            "tn_only_positive_detached_v2",
            "requires target_iou_v1",
        ),
        ("target_iou_v1", "unbound_v3", "unknown gradient contract"),
    ),
)
def test_target_iou_gradient_route_fails_closed(
    scope: str, gradient_contract: str, message: str
):
    args = _args(
        gradient_contract=gradient_contract,
        admission_contract=V42_CONTRACT,
        report=Path("unused.json"),
    )
    args.stage_b_v21_token_edit_query_scope = scope

    with pytest.raises(RuntimeError, match=message):
        training_main._stage_b_target_iou_carrier_pair_admission_contract(args)


def test_v42_formal_binding_uses_its_dedicated_verified_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    report = tmp_path / "v42-admission.json"
    report.write_text("{}\n", encoding="ascii")
    audit = {
        "status": "verified",
        "decision": "admit_to_formal_training",
        "formal_training_admitted": True,
    }
    monkeypatch.setattr(
        tools_package,
        V42_PROMOTION_MODULE,
        _promotion(report, audit),
        raising=False,
    )
    args = _args(
        gradient_contract="tn_only_positive_detached_v2",
        admission_contract=V42_CONTRACT,
        report=report,
    )

    assert training_main._bind_stage_b_confidence_probe_admission(args) == audit
    assert args.stage_b_dense_duty_confidence_probe_admission_audit == audit


def test_v42_gradient_route_rejects_the_v39_admission_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    report = tmp_path / "admission.json"
    report.write_text("{}\n", encoding="ascii")
    monkeypatch.setattr(
        tools_package,
        V42_PROMOTION_MODULE,
        _promotion(report, {}),
        raising=False,
    )
    args = _args(
        gradient_contract="tn_only_positive_detached_v2",
        admission_contract=V39_CONTRACT,
        report=report,
    )

    with pytest.raises(RuntimeError, match="promotion contract"):
        training_main._bind_stage_b_confidence_probe_admission(args)


def test_v39_bidirectional_route_remains_bound_to_v39(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    report = tmp_path / "v39-admission.json"
    report.write_text("{}\n", encoding="ascii")
    audit = {
        "status": "verified",
        "decision": "admit_to_formal_training",
        "formal_training_admitted": True,
    }
    monkeypatch.setattr(
        tools_package,
        V39_PROMOTION_MODULE,
        _promotion(report, audit),
        raising=False,
    )
    args = _args(
        gradient_contract="bidirectional_v1",
        admission_contract=V39_CONTRACT,
        report=report,
    )

    assert training_main._bind_stage_b_confidence_probe_admission(args) == audit


def test_v42_admission_fields_are_bound_into_the_v24_training_contract():
    values = probe._BASE._formal_current_args()
    audit = {
        "status": "verified",
        "decision": "admit_to_formal_training",
        "formal_training_admitted": True,
    }
    values.update(
        stage_b_dense_duty_execution_scope="formal",
        stage_b_dense_duty_confidence_probe_admission_contract=V42_CONTRACT,
        stage_b_dense_duty_confidence_probe_admission_report=(
            "/tmp/v42-admission.json"
        ),
        stage_b_dense_duty_confidence_probe_admission_audit=audit,
    )

    contract = build_training_contract(values)

    assert contract["schema"] == "pivot.stageb.dense_duty_training_contract/v24"
    assert contract["values"][
        "stage_b_dense_duty_confidence_probe_admission_contract"
    ] == V42_CONTRACT
    assert contract["values"][
        "stage_b_dense_duty_confidence_probe_admission_report"
    ] == "/tmp/v42-admission.json"
    assert contract["values"][
        "stage_b_dense_duty_confidence_probe_admission_audit"
    ] == audit
