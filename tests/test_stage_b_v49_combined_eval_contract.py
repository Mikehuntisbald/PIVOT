from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

import main as training_main
from tools import eval_refcoco_stageb as ref_eval
from tools import eval_text_groundingdino_refcoco_tn as combined_eval
from util import stage_b_dense_duty_audit as dense_duty_audit
from util.slconfig import SLConfig


V47_CONFIG = combined_eval._CANDIDATE_SPLIT_BOUNDARY_ROUTING_CONFIDENCE_U0400_CONFIG
V49_CONFIG = (
    combined_eval._CANDIDATE_SPLIT_GLOBAL_TRUST_VETO_CONFIDENCE_U0400_CONFIG
)


def _v49_checkpoint_case(tmp_path, monkeypatch, *, updates, reason):
    cfg = SLConfig.fromfile(str(V49_CONFIG))
    cfg.stage_b_dense_duty_partial_rank_diagnostic = False
    cfg.stage_b_dense_duty_partial_confidence_diagnostic = True

    train_dir = tmp_path / f"train-u{updates:04d}"
    train_dir.mkdir()
    checkpoint = train_dir / "checkpoint_iter.pth"
    checkpoint.write_bytes(b"v49-checkpoint-contract")
    source_closure = {"sha256": "a" * 64}
    saved_args = cfg._cfg_dict.to_dict()
    saved_args.update(
        {
            "output_dir": str(train_dir),
            "max_train_iters": 400,
            "stage_b_dense_duty_execution_scope": "probe",
            dense_duty_audit.SOURCE_CLOSURE_ARG: source_closure,
            "stage_b_dense_duty_runtime_audit": {
                "successful_optimizer_steps": updates,
                "optimizer_step_boundaries": updates,
                "amp_skipped_optimizer_steps": 0,
                "nonfinite_gradient_boundaries": 0,
                "zero_gradient_successful_steps": 0,
                "max_active_grad_norm_preclip": 1.0,
                "peak_reserved_bytes": 1,
            },
        }
    )
    payload = {
        "args": saved_args,
        "optimizer_updates": updates,
        "checkpoint_reason": reason,
    }
    audit = mock.Mock(
        return_value={
            "status": "passed",
            "phase": "confidence",
            "optimizer_updates": updates,
            "lineage": {
                "no_stage_b_teacher": True,
                "execution_scope": "probe",
            },
        }
    )
    resume_result = {
        "status": "passed",
        "phase": "confidence",
        "optimizer_updates": updates,
        "checkpoint_reason": reason,
        "rank_handoff": {"status": "passed"},
    }
    strict_resume = mock.Mock(return_value=resume_result)
    terminal_evaluation = mock.Mock(return_value=resume_result)
    monkeypatch.setattr(dense_duty_audit, "audit_checkpoint_payload", audit)
    monkeypatch.setattr(
        dense_duty_audit,
        "validate_strict_resume_checkpoint_payload",
        strict_resume,
    )
    monkeypatch.setattr(
        dense_duty_audit,
        "validate_evaluation_checkpoint_payload",
        terminal_evaluation,
    )
    monkeypatch.setattr(
        dense_duty_audit,
        "build_source_closure",
        lambda _path: source_closure,
    )
    monkeypatch.setattr(
        dense_duty_audit,
        "validate_source_closure",
        lambda value: value,
    )
    return cfg, payload, checkpoint, strict_resume, terminal_evaluation


def _diagnostic_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        partial_dense_duty_rank_diagnostic=False,
        partial_dense_duty_confidence_diagnostic=True,
        config=str(V49_CONFIG),
        output_dir=str(tmp_path / "v49-strict1607"),
        ckpts=["candidate-u400-checkpoint.pth"],
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


def test_v49_is_exact_structural_delta_over_v47():
    v47 = SLConfig.fromfile(str(V47_CONFIG))._cfg_dict.to_dict()
    v49 = SLConfig.fromfile(str(V49_CONFIG))._cfg_dict.to_dict()
    changed = {key for key in set(v47) | set(v49) if v47.get(key) != v49.get(key)}
    assert changed == {
        "stage_b_dense_duty_confidence_revision",
        "stage_b_dense_duty_confidence_head_gradient_contract",
        "stage_b_v11_trainable_params_min",
        "stage_b_v11_trainable_params_max",
        "stage_b_v15_tail_queue_negative_reduction_contract",
        "stage_b_dense_duty_trace_audit_path",
        "stage_b_dense_duty_trace_audit_sha256",
    }
    assert v49.get(
        "stage_b_v15_tail_queue_negative_reduction_contract",
        "all_mean_v1",
    ) == "all_mean_v1"


def test_v49_probe_config_is_admitted_by_both_evaluators(tmp_path):
    cfg = SLConfig.fromfile(str(V49_CONFIG))
    assert ref_eval._validate_v49_split_global_trust_veto_config(cfg) is True
    combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
        _diagnostic_args(tmp_path), cfg
    )


def test_v49_terminal_checkpoint_is_admitted_through_real_model_loader(
    tmp_path, monkeypatch
):
    cfg, payload, checkpoint, strict_resume, terminal_evaluation = (
        _v49_checkpoint_case(
            tmp_path,
            monkeypatch,
            updates=400,
            reason="max_train_iters",
        )
    )
    model = torch.nn.Linear(1, 1)
    payload["model"] = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    monkeypatch.setattr(
        ref_eval.MODULE_BUILD_FUNCS,
        "get",
        lambda _name: lambda _cfg: (model, None, None),
    )
    monkeypatch.setattr(
        ref_eval,
        "_torch_load_compat",
        lambda _path, map_location: payload,
    )
    monkeypatch.setattr(
        ref_eval,
        "validate_stage_b_fixed_text_scorer_checkpoint",
        lambda *_args, **_kwargs: None,
    )

    loaded = ref_eval._load_model(cfg, str(checkpoint), torch.device("cpu"))

    assert loaded is model
    assert loaded.stage_b_dense_duty_checkpoint_audit["terminal_checkpoint"] is True
    assert cfg.stage_b_dense_duty_partial_confidence_diagnostic_optimizer_updates == 400
    assert cfg.stage_b_dense_duty_partial_confidence_diagnostic_checkpoint_reason == (
        "max_train_iters"
    )
    terminal_evaluation.assert_called_once()
    strict_resume.assert_not_called()


@pytest.mark.parametrize(
    ("field", "drifted_value"),
    (
        ("stage_b_dense_duty_raw_veto_tn_carrier_balance", 0.5),
        ("stage_b_dense_duty_raw_veto_carrier_pair_weight", 0.5),
        ("stage_b_dense_duty_confidence_rank_evidence_contract", "drifted_v1"),
        ("stage_b_dense_duty_confidence_gate_gradient_contract", "drifted_v1"),
        (
            "stage_b_dense_duty_confidence_head_gradient_contract",
            "split_token_veto_global_absolute_v2",
        ),
        (
            "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
            "balanced_mean_v1",
        ),
        (
            "stage_b_v15_tail_queue_positive_trust_reduction_contract",
            "mean_v1",
        ),
        (
            "stage_b_v15_tail_queue_negative_reduction_contract",
            "exact_fpr95_active_set_mean_v1",
        ),
    ),
)
def test_v49_partial_checkpoint_rejects_inherited_and_split_head_drift(
    tmp_path, monkeypatch, field, drifted_value
):
    cfg, payload, checkpoint, strict_resume, terminal_evaluation = (
        _v49_checkpoint_case(
            tmp_path,
            monkeypatch,
            updates=399,
            reason="signal",
        )
    )
    payload["args"][field] = drifted_value

    with pytest.raises(RuntimeError, match="configuration drifted") as error:
        ref_eval._validate_dense_duty_partial_confidence_diagnostic_checkpoint(
            payload,
            cfg,
            checkpoint_path=checkpoint,
        )

    assert field in str(error.value)
    strict_resume.assert_called_once()
    terminal_evaluation.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("stage_b_dense_duty_raw_veto_tn_carrier_balance", 0.5),
        ("stage_b_dense_duty_raw_veto_carrier_pair_weight", 0.5),
        ("stage_b_dense_duty_confidence_gate_gradient_contract", "drifted_v1"),
        (
            "stage_b_dense_duty_confidence_head_gradient_contract",
            "split_token_veto_global_absolute_v2",
        ),
        (
            "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
            "balanced_mean_v1",
        ),
        (
            "stage_b_v15_tail_queue_positive_trust_reduction_contract",
            "mean_v1",
        ),
        (
            "stage_b_v15_tail_queue_negative_reduction_contract",
            "exact_fpr95_active_set_mean_v1",
        ),
    ),
)
def test_v49_training_and_eval_validators_reject_drift(tmp_path, field, value):
    cfg = SLConfig.fromfile(str(V49_CONFIG))
    setattr(cfg, field, value)
    with pytest.raises(RuntimeError):
        training_main._validate_stage_b_dense_duty_args(cfg)
    with pytest.raises(ValueError, match="contract failed"):
        combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
            _diagnostic_args(tmp_path), cfg
        )
