from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from engine import (
    _clip_stage_b_dense_duty_grad_norms,
    _record_stage_b_dense_duty_runtime_audit,
)
from tools import (
    run_stageb_confidence_candidate_complete_trace_c2_probe_u0400 as c2_training,
)
import util.stage_b_confidence_adapter_migration as migration_contract
import util.stage_b_dense_duty_audit as dense_duty_audit
from util.stage_b_dense_duty_audit import (
    FINGERPRINT_ARG,
    TRAINING_CONTRACT_ARG,
    _validate_fulltext_global_absolute_runtime_audit,
    audit_checkpoint_payload,
    build_training_contract,
    fingerprint_state,
    validate_strict_resume_checkpoint_payload,
)


class _TokenOnlyConfidenceScorer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.confidence_adapter = SimpleNamespace(
            head_gradient_contract=(
                "split_token_veto_deployment_owned_query_veto_"
                "global_absolute_v11"
            )
        )
        self.confidence_candidate_trace_contract = (
            "candidate_complete_monotone_token_entailment_v2"
        )
        self.verifier = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.LayerNorm(4),
            torch.nn.Linear(4, 1),
        )

    def token_veto_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return tuple(self.verifier.parameters())

    def expected_live_confidence_parameter_tensor_counts(self) -> dict[str, int]:
        return {"token_veto": len(self.token_veto_parameters())}


class _TokenOnlyRoot(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage_b_fixed_text_scorer = _TokenOnlyConfidenceScorer()


def _one_owner_step():
    root = _TokenOnlyRoot()
    loss = sum(
        parameter.float().square().sum()
        for parameter in root.parameters()
        if parameter.requires_grad
    )
    loss.backward()
    stats = _clip_stage_b_dense_duty_grad_norms(root, 0.1)
    args = SimpleNamespace(
        stage_b_dense_duty=True,
        stage_b_dense_duty_runtime_audit=None,
    )
    _record_stage_b_dense_duty_runtime_audit(
        args,
        torch.device("cpu"),
        optimizer_step_boundary=True,
        optimizer_step_succeeded=True,
        branch_grad_norms=stats,
        amp_scale=256.0,
    )
    return root, stats, args.stage_b_dense_duty_runtime_audit


def test_c2_token_only_clip_has_one_exact_live_owner():
    root, stats, runtime = _one_owner_step()
    expected_count = len(
        root.stage_b_fixed_text_scorer.token_veto_parameters()
    )
    assert stats["dense_duty_clip_contract_owner_count"] == 1
    assert stats["dense_duty_clip_contract_expected_token_veto_tensor_count"] == (
        expected_count
    )
    assert stats["dense_duty_clip_contract_observed_token_veto_tensor_count"] == (
        expected_count
    )
    assert "grad_norm_dense_duty_global_absolute_preclip" not in stats
    assert stats["grad_norm_dense_duty_token_veto_postclip"] <= 0.100001
    assert runtime["clip_contract_schema"] == (
        "pivot.stageb.dense_duty_one_owner_clip_contract/v1"
    )
    assert "expected_global_absolute_tensor_count" not in runtime


def test_c2_one_owner_runtime_audit_is_checkpoint_strict():
    root, _stats, runtime = _one_owner_step()
    expected_count = len(
        root.stage_b_fixed_text_scorer.token_veto_parameters()
    )
    validated = _validate_fulltext_global_absolute_runtime_audit(
        runtime,
        expected_steps=1,
        expected_owner_tensor_counts={"token_veto": expected_count},
        contract_label="candidate-complete-monotone-token-entailment",
    )
    assert validated["clip_contract_checked_steps"] == 1

    bad = dict(runtime)
    bad["zero_token_veto_gradient_successful_steps"] = 1
    with pytest.raises(RuntimeError, match="one-owner runtime audit is invalid"):
        _validate_fulltext_global_absolute_runtime_audit(
            bad,
            expected_steps=1,
            expected_owner_tensor_counts={"token_veto": expected_count},
            contract_label="candidate-complete-monotone-token-entailment",
        )


def _healthy_c2_u100_runtime_audit() -> dict[str, object]:
    return {
        "schema": "pivot.stageb.dense_duty_runtime_audit/v1",
        "optimizer_step_boundaries": 100,
        "successful_optimizer_steps": 100,
        "amp_skipped_optimizer_steps": 0,
        "nonfinite_gradient_boundaries": 0,
        "zero_gradient_successful_steps": 0,
        "last_active_grad_norm_preclip": 1.0,
        "max_active_grad_norm_preclip": 1.0,
        "clip_contract_schema": (
            "pivot.stageb.dense_duty_one_owner_clip_contract/v1"
        ),
        "clip_contract_checked_steps": 100,
        "clip_contract_max_norm": 0.1,
        "clip_contract_tolerance": 1.0e-5,
        "owner_clip_violation_steps": 0,
        "active_pre_decomposition_violation_steps": 0,
        "active_post_decomposition_violation_steps": 0,
        "live_tensor_count_violation_steps": 0,
        "active_monotonic_violation_steps": 0,
        "max_active_pre_decomposition_residual": 0.0,
        "max_active_post_decomposition_residual": 0.0,
        "max_owner_clip_residual": 0.0,
        "max_active_monotonic_residual": 0.0,
        "last_token_veto_grad_norm_preclip": 1.0,
        "max_token_veto_grad_norm_preclip": 1.0,
        "nonfinite_token_veto_gradient_boundaries": 0,
        "zero_token_veto_gradient_successful_steps": 0,
        "expected_token_veto_tensor_count": 356,
        "last_observed_token_veto_tensor_count": 356,
    }


def _c2_u100_partial_checkpoint(tmp_path: Path):
    active_prefix = "stage_b_fixed_text_scorer.confidence_verifier_tower."
    active_names = [
        f"{active_prefix}parameter_{index:03d}" for index in range(356)
    ]
    large_parameter_elements = 25_464_320 - (len(active_names) - 1)
    model_state = {
        active_names[0]: torch.zeros(large_parameter_elements),
        **{name: torch.zeros(()) for name in active_names[1:]},
        "stage_b_fixed_text_scorer._dense_duty_contract_version": torch.tensor(
            2, dtype=torch.int64
        ),
        "stage_b_fixed_text_scorer.rank_tower.synthetic": torch.tensor(7.0),
        "stage_b_fixed_text_scorer.confidence_verifier_veto_head.synthetic": (
            torch.tensor(0.0)
        ),
        "stage_b_fixed_text_scorer.confidence_pool.synthetic": torch.tensor(0.0),
    }
    initial_fingerprint = fingerprint_state(
        model_state,
        active_parameter_names=active_names,
        phase="confidence",
    )
    model_state[active_names[0]][0] = 1.0

    checkpoint_path = tmp_path / "checkpoint_iter.pth"
    checkpoint_path.touch()
    current_args = c2_training._BASE._formal_current_args()
    current_args.update(
        {
            "output_dir": str(tmp_path),
            "resume": str(checkpoint_path),
            "pretrain_model_path": None,
        }
    )
    saved_args = copy.deepcopy(current_args)
    saved_args[FINGERPRINT_ARG] = initial_fingerprint
    saved_args["stage_b_dense_duty_runtime_audit"] = (
        _healthy_c2_u100_runtime_audit()
    )
    saved_args["stage_b_dense_duty_confidence_adapter_migration_audit"] = {
        "status": "synthetic-test"
    }
    saved_args["stage_b_dense_duty_rank_source_checkpoint_audit"] = {
        "status": "synthetic-test"
    }
    saved_args[TRAINING_CONTRACT_ARG] = build_training_contract(saved_args)

    rng_state = {
        "python": (3, (), None),
        "numpy": ("MT19937", (), 0, 0, 0.0),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    optimizer_parameter_ids = list(range(len(active_names)))
    payload = {
        "model": model_state,
        "criterion": {},
        "optimizer": {
            "state": {
                parameter_id: {"step": torch.tensor(100.0)}
                for parameter_id in optimizer_parameter_ids
            },
            "param_groups": [{"params": optimizer_parameter_ids}],
        },
        "lr_scheduler": {},
        "scaler": {},
        "epoch": 0,
        "iteration": 400,
        "optimizer_updates": 100,
        "epoch_finished": False,
        "rng_state": copy.deepcopy(rng_state),
        "epoch_rng_state": copy.deepcopy(rng_state),
        "args": saved_args,
        "checkpoint_reason": "interval",
    }
    return current_args, payload, checkpoint_path


def test_c2_u100_partial_checkpoint_uses_real_strict_audit_and_controller_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    current_args, payload, checkpoint_path = _c2_u100_partial_checkpoint(tmp_path)
    monkeypatch.setattr(
        migration_contract,
        "validate_confidence_adapter_migration_audit",
        lambda *_args, **_kwargs: {"status": "synthetic-test"},
    )
    monkeypatch.setattr(
        dense_duty_audit,
        "validate_confidence_adapter_rank_source_audit",
        lambda *_args, **_kwargs: {"status": "synthetic-test"},
    )

    strict_audit = validate_strict_resume_checkpoint_payload(
        payload,
        current_args,
        checkpoint_path=checkpoint_path,
    )
    assert strict_audit["optimizer_updates"] == 100
    assert strict_audit["checkpoint_reason"] == "interval"
    assert strict_audit["runtime_audit"]["clip_contract_schema"].endswith(
        "one_owner_clip_contract/v1"
    )

    checkpoint_audit = audit_checkpoint_payload(
        payload, checkpoint_path=checkpoint_path
    )
    assert checkpoint_audit["status"] == "passed"
    assert checkpoint_audit["ownership"]["active_parameter_count"] == 356
    assert checkpoint_audit["current"]["active"]["element_count"] == 25_464_320

    controller = c2_training._BASE
    monkeypatch.setattr(controller, "OUTPUT", tmp_path)
    monkeypatch.setattr(controller, "CHECKPOINT", checkpoint_path)
    monkeypatch.setattr(controller, "_load", lambda _path: payload)
    monkeypatch.setattr(controller, "_validate_migration", lambda _args: {})
    monkeypatch.setattr(
        controller, "_formal_current_args", lambda: copy.deepcopy(current_args)
    )
    monkeypatch.setattr(
        controller,
        "fingerprint_named_tensors",
        lambda *_args, **_kwargs: {"sha256": controller.RANK_SHA256},
    )
    assert c2_training.inspect() == {
        "status": "partial",
        "action": "resume",
        "updates": 100,
    }

    bad_runtime = dict(payload["args"]["stage_b_dense_duty_runtime_audit"])
    bad_runtime["clip_contract_schema"] = (
        "pivot.stageb.dense_duty_two_owner_clip_contract/v1"
    )
    bad_saved_args = dict(payload["args"])
    bad_saved_args["stage_b_dense_duty_runtime_audit"] = bad_runtime
    bad_payload = dict(payload)
    bad_payload["args"] = bad_saved_args

    with pytest.raises(RuntimeError, match="one-owner clip check"):
        validate_strict_resume_checkpoint_payload(
            bad_payload,
            current_args,
            checkpoint_path=checkpoint_path,
        )
    with pytest.raises(RuntimeError, match="one-owner clip check"):
        audit_checkpoint_payload(bad_payload, checkpoint_path=checkpoint_path)

    monkeypatch.setattr(controller, "_load", lambda _path: bad_payload)
    rejected = c2_training.inspect()
    assert rejected["status"] == "invalid"
    assert "one-owner clip check" in rejected["reason"]
