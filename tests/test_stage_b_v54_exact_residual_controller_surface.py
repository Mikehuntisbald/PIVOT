from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from tools import (
    audit_stageb_confidence_adapter_fulltext_global_absolute_exact_residual_probe_health
    as health,
)
from tools import (
    run_stageb_confidence_adapter_fulltext_global_absolute_exact_residual_highmem_formal
    as formal,
)
from tools import (
    run_stageb_confidence_adapter_fulltext_global_absolute_exact_residual_probe_evaluation
    as evaluation,
)
from tools import (
    run_stageb_confidence_adapter_fulltext_global_absolute_exact_residual_probe_u0400
    as training,
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
    FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_FRESH_CONFIDENCE_CONTRACT,
    FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_MIGRATION_SCHEMA,
    FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_POOL_FEATURE_CONTRACT,
    SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT,
)


def test_v54_training_controller_uses_fresh_u400_exact_residual_surface():
    assert training.UPDATES == 400
    assert training.CONFIG.name == (
        "cfg_stageb_dense_duty_confidence_adapter_fulltext_global_absolute_"
        "exact_residual_probe_u0400_20260802.py"
    )
    assert training.OUTPUT == Path(
        "outputs/paper_cvpr_v1/"
        "dense_duty_adapter_fulltext_global_absolute_exact_residual_highmem_"
        "20260802/probe/u000400_fresh"
    ).resolve()
    assert training.CHECKPOINT == training.OUTPUT / "checkpoint_iter.pth"
    command = training.command("start")
    assert "--resume" not in command
    assert command[command.index("--max_train_iters") + 1] == "400"


def test_v54_config_and_controller_constants_are_exact():
    cfg = SLConfig.fromfile(str(training.CONFIG))
    assert cfg.stage_b_dense_duty_confidence_revision == evaluation.EXPECTED_REVISION
    assert (
        cfg.stage_b_dense_duty_confidence_head_gradient_contract
        == evaluation.EXPECTED_HEAD_CONTRACT
    )
    assert (
        cfg.stage_b_dense_duty_confidence_pool_feature_contract
        == evaluation.EXPECTED_POOL_CONTRACT
        == FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_POOL_FEATURE_CONTRACT
    )
    assert (
        cfg.stage_b_dense_duty_positive_trust_contract
        == evaluation.EXPECTED_POSITIVE_TRUST_CONTRACT
        == "exact_frozen_rank_max_confidence_delta_v3"
    )
    assert health.TRAINING_CONTRACT_SCHEMA == (
        "pivot.stageb.dense_duty_training_contract/v36"
    )
    assert cfg.stage_b_dense_duty_confidence_expected_optimizer_updates == 400
    assert cfg.stage_b_v11_trainable_params_min == 534_725
    assert cfg.stage_b_v11_trainable_params_max == 534_725


def test_v54_strict1607_controller_keeps_exact_gate_and_tn_only_command():
    assert evaluation.BASELINE_FALSE_ACCEPTS == 801
    assert evaluation.MAX_ADMITTED_FALSE_ACCEPTS == 800
    assert evaluation._CORE.BASELINE_FALSE_ACCEPTS == 801
    assert evaluation._CORE.MAX_ADMITTED_FALSE_ACCEPTS == 800
    command = evaluation.build_command()
    assert command[0] == str(evaluation.FIXED_PYTHON)
    assert command[command.index("--config") + 1] == str(training.CONFIG)
    assert command[command.index("--ckpts") + 1] == str(training.CHECKPOINT)
    assert command[command.index("--output_dir") + 1] == str(evaluation.OUTPUT)
    assert "--skip_ref" in command
    assert "--partial_dense_duty_confidence_diagnostic" in command
    assert command[command.index("--topk") + 1] == "1"
    assert command[command.index("--max_tn_batches") + 1] == "0"
    split = command.index("--tn_splits")
    assert command[split + 1 : split + 3] == [
        "refcocop_val",
        "refcocog_umd_val",
    ]
    assert evaluation._CORE._load_health_audit() is health.audit


def test_v54_postflight_replaces_v52_claims_with_exact_residual_contract(
    monkeypatch,
):
    monkeypatch.setattr(
        evaluation,
        "_BASE_POSTFLIGHT",
        lambda *_args, **_kwargs: {
            "contracts": {
                "terminal_u400_diagnostic": True,
                "v53_rank_full_expression_global_absolute_v35": True,
                "two_independent_confidence_owners": True,
            }
        },
    )

    contracts = evaluation._v54_postflight({})["contracts"]

    assert contracts[
        "v54_rank_full_expression_global_absolute_exact_residual_v36"
    ] is True
    assert contracts[evaluation.EXPECTED_POOL_CONTRACT] is True
    assert contracts[evaluation.EXPECTED_POSITIVE_TRUST_CONTRACT] is True
    assert contracts[
        "positive_tail_trust_uses_exact_frozen_rank_max_residual"
    ] is True
    assert contracts["two_independent_confidence_owners"] is True
    assert "v53_rank_full_expression_global_absolute_v35" not in contracts


def _admission_source(*, trust_contract: str | None = None) -> str:
    trust = (
        evaluation.EXPECTED_POSITIVE_TRUST_CONTRACT
        if trust_contract is None
        else trust_contract
    )
    return f'''\
def _bind_stage_b_confidence_probe_admission(args):
    if (
        str(getattr(args, "aggregation", "")) == {evaluation.EXPECTED_AGGREGATION!r}
        and str(getattr(args, "revision", "")) == {evaluation.EXPECTED_REVISION!r}
        and str(getattr(args, "head", "")) == {evaluation.EXPECTED_HEAD_CONTRACT!r}
        and str(getattr(args, "pool", "")) == {evaluation.EXPECTED_POOL_CONTRACT!r}
        and str(getattr(args, "gate", "")) == {evaluation.EXPECTED_GATE_CONTRACT!r}
        and str(getattr(args, "routing_reduction", "")) == {evaluation.EXPECTED_ROUTING_REDUCTION!r}
        and str(getattr(args, "trust_reduction", "")) == {evaluation.EXPECTED_TRUST_REDUCTION!r}
        and str(getattr(args, "negative_reduction", "")) == {evaluation.EXPECTED_NEGATIVE_REDUCTION!r}
        and str(getattr(args, "token_scope", "")) == {evaluation.EXPECTED_TOKEN_EDIT_QUERY_SCOPE!r}
        and str(getattr(args, "positive_gradient", "")) == {evaluation.EXPECTED_POSITIVE_GRADIENT_CONTRACT!r}
        and str(getattr(args, "carrier_pair", "")) == "bidirectional_v1"
        and str(getattr(args, "positive_trust", "")) == {trust!r}
        and float(getattr(args, "stage_b_dense_duty_deployed_veto_routing_weight", -1.0)) == 0.0
        and float(getattr(args, "stage_b_dense_duty_deployed_veto_positive_max", -1.0)) == 0.1
        and float(getattr(args, "stage_b_dense_duty_deployed_veto_tn_min", -1.0)) == 0.9
        and float(getattr(args, "stage_b_dense_duty_confidence_veto_gate_offset", -1.0)) == 0.0
    ):
        formal_contract = {evaluation.FORMAL_ADMISSION_CONTRACT!r}
        from tools import {evaluation.CONTROLLER_IMPORT} as promotion
        return formal_contract, promotion
'''


def test_v54_main_admission_ast_requires_exact_residual_trust(tmp_path):
    source = tmp_path / "main.py"
    source.write_text(_admission_source(), encoding="utf-8")
    assert evaluation._formal_main_admission_is_wired(source) is True

    source.write_text(
        _admission_source(trust_contract="absolute_global_confidence_logit_v2"),
        encoding="utf-8",
    )
    assert evaluation._formal_main_admission_is_wired(source) is False


def _migration_audit() -> dict:
    return {
        "schema": FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_MIGRATION_SCHEMA,
        "fresh_confidence_contract": (
            FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_FRESH_CONFIDENCE_CONTRACT
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


def test_v54_health_accepts_only_exact_v21_migration():
    result = health._audit_v54_migration(_migration_args())
    assert result["schema"] == FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_MIGRATION_SCHEMA
    assert result["fresh_confidence_contract"] == (
        FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_FRESH_CONFIDENCE_CONTRACT
    )
    assert result["pool_feature_contract"] == (
        FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_POOL_FEATURE_CONTRACT
    )

    drift = copy.deepcopy(_migration_args())
    drift["stage_b_dense_duty_confidence_adapter_migration_audit"]["schema"] = (
        "pivot.stageb.rank_to_token_confidence_adapter_fulltext_global_absolute/v20"
    )
    with pytest.raises(health.ProbeHealthEvidenceError):
        health._audit_v54_migration(drift)


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


def test_v54_health_keeps_exact_65_21_44_two_owner_runtime():
    runtime = health._audit_runtime(_runtime_audit())
    checks = health._health_checks(runtime, _trajectory())
    assert checks["u222_v53_owner_live_counts_exact"]["passed"] is True
    assert checks["u222_v53_two_independent_clips_exact"]["passed"] is True
    assert all(check["passed"] for check in checks.values())


def test_v54_formal_controller_is_fresh_u4412_and_requires_admission(monkeypatch):
    assert formal.CONFIG.resolve() == evaluation.FORMAL_CONFIG.resolve()
    assert formal.UPDATES == 4412
    assert formal.CHECKPOINT == formal.OUTPUT / "checkpoint_iter.pth"
    assert formal.OUTPUT == Path(
        "outputs/paper_cvpr_v1/"
        "dense_duty_adapter_fulltext_global_absolute_exact_residual_highmem_"
        "20260802/formal/confidence"
    ).resolve()
    command = formal.command("start")
    assert "--resume" not in command
    assert command[command.index("--max_train_iters") + 1] == "4412"

    sentinel = {"formal_training_admitted": True}
    monkeypatch.setattr(evaluation, "verify_admission_report", lambda: sentinel)
    assert formal.verify_probe_admission() is sentinel
