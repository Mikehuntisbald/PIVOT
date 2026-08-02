import copy
import math
from pathlib import Path

import pytest
import torch

from tools import (
    audit_stageb_confidence_adapter_fulltext_global_absolute_probe_health as health,
)
from tools import (
    run_stageb_confidence_adapter_fulltext_global_absolute_probe_evaluation
    as evaluation,
)
from tools import (
    run_stageb_confidence_adapter_fulltext_global_absolute_probe_u0400 as training,
)
from util.slconfig import SLConfig
from util.stage_b_confidence_adapter_migration import (
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_ADAPTER_TENSOR_COUNT,
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_ELEMENT_COUNT,
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_SHA256,
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_STORAGE_BYTES,
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_TENSOR_COUNT,
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_POOL_TENSOR_COUNT,
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_STRICT_TARGET_TENSOR_COUNT,
    FULLTEXT_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT,
    FULLTEXT_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA,
    SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT,
)


def test_v53_training_controller_uses_fresh_u400_probe_surface():
    assert training.UPDATES == 400
    assert training.CONFIG.name == (
        "cfg_stageb_dense_duty_confidence_adapter_fulltext_global_absolute_"
        "probe_u0400_20260802.py"
    )
    assert training.OUTPUT.name == "u000400_fresh"
    assert training.CHECKPOINT == training.OUTPUT / "checkpoint_iter.pth"
    assert "fulltext_global_absolute" in training.LOCK.name


def test_v53_probe_config_and_controller_constants_are_exact():
    cfg = SLConfig.fromfile(str(training.CONFIG))
    assert cfg.stage_b_dense_duty_confidence_revision == evaluation.EXPECTED_REVISION
    assert (
        cfg.stage_b_dense_duty_confidence_head_gradient_contract
        == evaluation.EXPECTED_HEAD_CONTRACT
    )
    assert (
        cfg.stage_b_dense_duty_confidence_pool_feature_contract
        == evaluation.EXPECTED_POOL_CONTRACT
    )
    assert (
        cfg.stage_b_dense_duty_confidence_gate_gradient_contract
        == evaluation.EXPECTED_GATE_CONTRACT
    )
    assert cfg.stage_b_dense_duty_deployed_veto_routing_weight == 0.0
    assert cfg.stage_b_dense_duty_confidence_expected_optimizer_updates == 400
    assert cfg.stage_b_v11_trainable_params_min == 534_725
    assert cfg.stage_b_v11_trainable_params_max == 534_725
    assert evaluation._CORE.FORMAL_PROMOTION_OVERRIDES[
        "stage_b_dense_duty_confidence_expected_optimizer_updates"
    ] == (400, 4412)


def test_v53_probe_controller_builds_fixed_tn_only_strict1607_command():
    command = evaluation.build_command()
    assert command[0] == str(evaluation.FIXED_PYTHON)
    assert command[1].endswith("tools/eval_text_groundingdino_refcoco_tn.py")
    assert command[command.index("--config") + 1] == str(training.CONFIG)
    assert command[command.index("--ckpts") + 1] == str(training.CHECKPOINT)
    assert command[command.index("--output_dir") + 1] == str(evaluation.OUTPUT)
    assert "--skip_ref" in command
    assert "--partial_dense_duty_confidence_diagnostic" in command
    assert command[command.index("--topk") + 1] == "1"
    split_index = command.index("--tn_splits")
    assert command[split_index + 1 : split_index + 3] == [
        "refcocop_val",
        "refcocog_umd_val",
    ]
    assert command[command.index("--max_tn_batches") + 1] == "0"
    assert evaluation._CORE._load_health_audit() is health.audit


def test_v53_postflight_replaces_every_stale_v52_owner_claim(monkeypatch):
    monkeypatch.setattr(
        evaluation,
        "_BASE_POSTFLIGHT",
        lambda *_args, **_kwargs: {
            "contracts": {
                "terminal_u400_diagnostic": True,
                "v52_candidate_sample_calibrator_split_v34": True,
                "split_token_veto_candidate_absolute_sample_calibrator_v6": True,
                "three_independent_confidence_owners": True,
                "candidate_local_and_sample_global_logits_are_distinct": True,
            }
        },
    )
    result = evaluation._v53_postflight({})
    contracts = result["contracts"]
    assert contracts["v53_rank_full_expression_global_absolute_v35"] is True
    assert contracts[evaluation.EXPECTED_HEAD_CONTRACT] is True
    assert contracts[evaluation.EXPECTED_POOL_CONTRACT] is True
    assert contracts[evaluation.EXPECTED_GATE_CONTRACT] is True
    assert contracts["two_independent_confidence_owners"] is True
    assert not any("v52" in key for key in contracts)
    assert "three_independent_confidence_owners" not in contracts
    assert "candidate_local_and_sample_global_logits_are_distinct" not in contracts


def _admission_source() -> str:
    return f'''\
def _bind_stage_b_confidence_probe_admission(args):
    if (
        str(getattr(args, "aggregation", "")) == {evaluation.EXPECTED_AGGREGATION!r}
        and str(getattr(args, "revision", "")) == {evaluation.EXPECTED_REVISION!r}
        and str(getattr(args, "head", "")) == {evaluation.EXPECTED_HEAD_CONTRACT!r}
        and str(getattr(args, "pool", "")) == {evaluation.EXPECTED_POOL_CONTRACT!r}
        and str(getattr(args, "gate", "")) == {evaluation.EXPECTED_GATE_CONTRACT!r}
        and str(
            getattr(args, "routing_reduction", "")
        ) == {evaluation.EXPECTED_ROUTING_REDUCTION!r}
        and str(
            getattr(args, "trust_reduction", "")
        ) == {evaluation.EXPECTED_TRUST_REDUCTION!r}
        and str(
            getattr(args, "negative_reduction", "")
        ) == {evaluation.EXPECTED_NEGATIVE_REDUCTION!r}
        and str(
            getattr(args, "token_scope", "")
        ) == {evaluation.EXPECTED_TOKEN_EDIT_QUERY_SCOPE!r}
        and str(
            getattr(args, "positive_gradient", "")
        ) == {evaluation.EXPECTED_POSITIVE_GRADIENT_CONTRACT!r}
        and str(getattr(args, "carrier_pair", "")) == "bidirectional_v1"
        and str(
            getattr(args, "positive_trust", "")
        ) == "absolute_global_confidence_logit_v2"
        and float(
            getattr(
                args,
                "stage_b_dense_duty_deployed_veto_routing_weight",
                -1.0,
            )
        ) == 0.0
        and float(
            getattr(args, "stage_b_dense_duty_deployed_veto_positive_max", -1.0)
        ) == 0.1
        and float(getattr(args, "stage_b_dense_duty_deployed_veto_tn_min", -1.0)) == 0.9
        and float(
            getattr(
                args,
                "stage_b_dense_duty_confidence_veto_gate_offset",
                -1.0,
            )
        ) == 0.0
    ):
        formal_contract = {evaluation.FORMAL_ADMISSION_CONTRACT!r}
        from tools import {evaluation.CONTROLLER_IMPORT} as promotion
        return formal_contract, promotion
'''


def test_v53_formal_admission_ast_requires_head_pool_raw_gate_and_routing(tmp_path):
    source = tmp_path / "main.py"
    source.write_text(_admission_source(), encoding="utf-8")
    assert evaluation._formal_main_admission_is_wired(source) is True

    source.write_text(
        _admission_source().replace(evaluation.EXPECTED_POOL_CONTRACT, "drifted_pool"),
        encoding="utf-8",
    )
    assert evaluation._formal_main_admission_is_wired(source) is False


def _migration_audit() -> dict:
    return {
        "schema": FULLTEXT_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA,
        "fresh_confidence_contract": (
            FULLTEXT_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT
        ),
        "rank_evidence_contract": SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT,
        "head_gradient_contract": health.EXPECTED_HEAD_CONTRACT,
        "token_logit_contract": "detached_rank_token_minus_zero_init_residual_v1",
        "pool_feature_contract": health.EXPECTED_POOL_CONTRACT,
        "source_checkpoint_sha256": "a" * 64,
        "source_optimizer_updates": 6551,
        "source_checkpoint_reason": "max_train_iters",
        "rank": {
            "sha256": health._CORE.EXPECTED_RANK_SHA256,
            "tensor_count": 453,
            "nonfinite_count": 0,
        },
        "transferred": {
            "sha256": "c" * 64,
            "tensor_count": 1588,
            "nonfinite_count": 0,
        },
        "fresh_confidence": {
            "sha256": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_SHA256,
            "tensor_count": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_TENSOR_COUNT,
            "element_count": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_ELEMENT_COUNT,
            "storage_bytes": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_STORAGE_BYTES,
            "nonfinite_count": 0,
        },
        "retired_confidence_tower_tensor_count": 453,
        "retired_confidence_pool_tensor_count": 6,
        "retired_confidence_loaded_tensor_count": 0,
        "adapter_tensor_count": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_ADAPTER_TENSOR_COUNT,
        "pool_tensor_count": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_POOL_TENSOR_COUNT,
        "confidence_parameter_tensor_count": 65,
        "confidence_parameter_element_count": 534_725,
        "strict_target_tensor_count": (
            EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_STRICT_TARGET_TENSOR_COUNT
        ),
    }


def _migration_args() -> dict:
    return {
        "stage_b_dense_duty_rank_source_checkpoint_sha256": "a" * 64,
        "stage_b_dense_duty_rank_source_optimizer_updates": 6551,
        "stage_b_dense_duty_rank_source_checkpoint_reason": "max_train_iters",
        "stage_b_dense_duty_rank_source_rank_sha256": (
            health._CORE.EXPECTED_RANK_SHA256
        ),
        "stage_b_dense_duty_rank_source_transferred_sha256": "c" * 64,
        "stage_b_dense_duty_confidence_adapter_migration_audit": _migration_audit(),
    }


def test_v53_health_accepts_only_exact_v20_fresh_migration():
    result = health._audit_v53_migration(_migration_args())
    assert result["schema"] == FULLTEXT_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA
    assert result["fresh_confidence"]["sha256"] == health.MIGRATION_FRESH_SHA256

    drift = _migration_args()
    drift["stage_b_dense_duty_confidence_adapter_migration_audit"] = copy.deepcopy(
        drift["stage_b_dense_duty_confidence_adapter_migration_audit"]
    )
    drift["stage_b_dense_duty_confidence_adapter_migration_audit"][
        "fresh_confidence"
    ]["sha256"] = "0" * 64
    with pytest.raises(health.ProbeHealthEvidenceError):
        health._audit_v53_migration(drift)


def _runtime_audit() -> dict:
    runtime = {
        "schema": health._CORE.RUNTIME_SCHEMA,
        "optimizer_step_boundaries": 400,
        "successful_optimizer_steps": 400,
        "amp_skipped_optimizer_steps": 0,
        "nonfinite_gradient_boundaries": 0,
        "zero_gradient_successful_steps": 0,
        "last_active_grad_norm_preclip": 1.0,
        "max_active_grad_norm_preclip": 2.0,
        "last_amp_scale": 65536.0,
        "min_amp_scale": 32768.0,
        "clip_contract_schema": health.TWO_OWNER_CLIP_CONTRACT_SCHEMA,
        "clip_contract_checked_steps": 400,
        "owner_clip_violation_steps": 0,
        "active_pre_decomposition_violation_steps": 0,
        "active_post_decomposition_violation_steps": 0,
        "live_tensor_count_violation_steps": 0,
        "active_monotonic_violation_steps": 0,
        "max_active_pre_decomposition_residual": 0.0,
        "max_active_post_decomposition_residual": 0.0,
        "max_owner_clip_residual": 0.0,
        "max_active_monotonic_residual": 0.0,
        "clip_contract_tolerance": 1e-6,
        "clip_contract_max_norm": 0.1,
    }
    for owner, count in (("token_veto", 21), ("global_absolute", 44)):
        runtime[f"last_{owner}_grad_norm_preclip"] = 0.5
        runtime[f"max_{owner}_grad_norm_preclip"] = 1.0
        runtime[f"nonfinite_{owner}_gradient_boundaries"] = 0
        runtime[f"zero_{owner}_gradient_successful_steps"] = 0
        runtime[f"expected_{owner}_tensor_count"] = count
        runtime[f"last_observed_{owner}_tensor_count"] = count
    return runtime


def test_v53_runtime_accepts_exact_two_owner_400_step_evidence():
    result = health._audit_runtime(_runtime_audit())
    assert result["clip_contract_schema"] == health.TWO_OWNER_CLIP_CONTRACT_SCHEMA
    assert result["expected_token_veto_tensor_count"] == 21
    assert result["expected_global_absolute_tensor_count"] == 44


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "clip_contract_schema",
            "pivot.stageb.dense_duty_three_owner_clip_contract/v1",
        ),
        ("clip_contract_checked_steps", 399),
        ("owner_clip_violation_steps", 1),
        ("expected_global_absolute_tensor_count", 43),
        ("max_token_veto_grad_norm_preclip", 0.0),
        ("zero_global_absolute_gradient_successful_steps", 1),
    ),
)
def test_v53_runtime_fails_closed_on_two_owner_drift(field, value):
    runtime = _runtime_audit()
    runtime[field] = value
    with pytest.raises(health.ProbeHealthEvidenceError):
        health._audit_runtime(runtime)


@pytest.mark.parametrize(
    "owner", ("candidate_absolute", "sample_calibrator", "deployed_router")
)
def test_v53_runtime_rejects_retired_owner_traces(owner):
    runtime = _runtime_audit()
    runtime[f"max_{owner}_grad_norm_preclip"] = 1.0
    with pytest.raises(health.ProbeHealthEvidenceError, match="only token-veto"):
        health._audit_runtime(runtime)


def _trajectory() -> dict:
    return {
        "train_grad_norm_dense_duty_active_preclip": 0.5,
        "train_grad_tensor_count_dense_duty_active": 65.0,
        "train_grad_norm_dense_duty_token_veto_preclip": 0.3,
        "train_grad_tensor_count_dense_duty_token_veto": 21.0,
        "train_grad_norm_dense_duty_global_absolute_preclip": 0.4,
        "train_grad_tensor_count_dense_duty_global_absolute": 44.0,
        "train_grad_norm_dense_duty_token_veto_postclip": 0.1,
        "train_grad_norm_dense_duty_global_absolute_postclip": 0.1,
        "train_grad_norm_dense_duty_active_postclip": math.sqrt(0.02),
        "train_amp_step_skipped": 0.0,
    }


def test_v53_health_checks_lock_65_21_44_and_independent_clips():
    checks = health._health_checks(
        health._audit_runtime(_runtime_audit()), _trajectory()
    )
    assert checks["u222_v53_owner_live_counts_exact"]["passed"] is True
    assert checks["u222_v53_two_independent_clips_exact"]["passed"] is True
    assert all(check["passed"] for check in checks.values())


def _sized_tensors(prefix: str, count: int, elements: int) -> tuple[list[str], dict]:
    names = [f"{prefix}synthetic_{index:02d}" for index in range(count)]
    sizes = [elements - count + 1, *([1] * (count - 1))]
    return names, {name: torch.zeros(size) for name, size in zip(names, sizes)}


def test_v53_ownership_requires_all_65_optimizer_states(monkeypatch):
    token_names, token = _sized_tensors(
        health._CORE._CONFIDENCE_ADAPTER_PREFIX + "query_norm.", 21, 51_267
    )
    global_adapter_names, global_adapter = _sized_tensors(
        health._CORE._CONFIDENCE_ADAPTER_PREFIX + "candidate_absolute_head.",
        38,
        350_081,
    )
    pool_names, pool = _sized_tensors(
        health._CORE._CONFIDENCE_POOL_PREFIX + "residual.", 6, 133_377
    )
    names = token_names + global_adapter_names + pool_names
    model = {**token, **global_adapter, **pool}
    rank_name = health._CORE._RANK_PREFIX + "synthetic.weight"
    model[rank_name] = torch.zeros(1)
    initial = {
        "active_parameter_names": names,
        "active": {
            "tensor_count": 65,
            "element_count": 534_725,
            "nonfinite_count": 0,
            "sha256": "initial",
        },
    }

    monkeypatch.setattr(
        health, "validate_initial_fingerprint", lambda *_args, **_kwargs: initial
    )

    def fingerprint(_model, selected):
        if list(selected) == [rank_name]:
            return {
                "sha256": health._CORE.EXPECTED_RANK_SHA256,
                "nonfinite_count": 0,
            }
        return {"sha256": "trained", "nonfinite_count": 0}

    monkeypatch.setattr(health, "fingerprint_named_tensors", fingerprint)
    monkeypatch.setattr(
        health,
        "_audit_v53_migration",
        lambda _args: {"schema": FULLTEXT_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA},
    )
    args = {
        "stage_b_dense_duty_initial_state_fingerprint": {},
        "stage_b_dense_duty_rank_source_rank_sha256": (
            health._CORE.EXPECTED_RANK_SHA256
        ),
    }
    payload = {
        "model": model,
        "optimizer": {
            "param_groups": [{"params": list(range(65))}],
            "state": {index: {} for index in range(65)},
        },
    }
    result = health._audit_split_ownership(payload, args)
    assert result["token_veto"]["tensor_count"] == 21
    assert result["global_absolute"]["tensor_count"] == 44
    assert result["token_veto"]["optimizer_state_tensor_count"] == 21
    assert result["global_absolute"]["optimizer_state_tensor_count"] == 44
    assert result["adapter_tensor_count"] == 59
    assert result["pool_tensor_count"] == 6

    payload["optimizer"]["state"].pop(64)
    with pytest.raises(health.ProbeHealthEvidenceError, match="all 65"):
        health._audit_split_ownership(payload, args)
