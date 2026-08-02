import copy
import hashlib
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
    SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
V53_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_fulltext_global_absolute_"
    "20260802.py"
)
V54_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_fulltext_global_absolute_"
    "exact_residual_20260802.py"
)


def _config(path: Path) -> dict:
    return SLConfig.fromfile(str(path))._cfg_dict.to_dict()


def test_v54_changes_only_the_seven_declared_v53_contract_fields():
    v53 = _config(V53_CONFIG)
    v54 = _config(V54_CONFIG)
    assert set(v54) == set(v53)
    changed = {key for key in v53 if v53[key] != v54[key]}
    assert changed == {
        "stage_b_dense_duty_confidence_revision",
        "stage_b_dense_duty_confidence_pool_feature_contract",
        "stage_b_dense_duty_positive_trust_contract",
        "stage_b_dense_duty_trace_audit_path",
        "stage_b_dense_duty_trace_audit_sha256",
        "stage_b_dense_duty_confidence_probe_admission_contract",
        "stage_b_dense_duty_confidence_probe_admission_report",
    }
    trace_path = Path(v54["stage_b_dense_duty_trace_audit_path"])
    assert trace_path.is_file()
    assert v54["stage_b_dense_duty_trace_audit_sha256"] == hashlib.sha256(
        trace_path.read_bytes()
    ).hexdigest()
    assert v54["stage_b_v11_trainable_params_min"] == 534_725
    assert v54["stage_b_v11_trainable_params_max"] == 534_725


def test_v54_probe_controller_and_strict1607_gate_are_exact():
    cfg = _config(training.CONFIG)
    assert training.UPDATES == 400
    assert training.OUTPUT.name == "u000400_fresh"
    assert training.CHECKPOINT == training.OUTPUT / "checkpoint_iter.pth"
    assert cfg["stage_b_dense_duty_confidence_revision"] == (
        evaluation.EXPECTED_REVISION
    )
    assert cfg["stage_b_dense_duty_confidence_pool_feature_contract"] == (
        evaluation.EXPECTED_POOL_CONTRACT
    )
    assert cfg["stage_b_dense_duty_positive_trust_contract"] == (
        evaluation.EXPECTED_POSITIVE_TRUST_CONTRACT
    )
    assert cfg["stage_b_dense_duty_confidence_expected_optimizer_updates"] == 400
    assert evaluation.BASELINE_FALSE_ACCEPTS == 801
    assert evaluation.MAX_ADMITTED_FALSE_ACCEPTS == 800

    command = evaluation.build_command()
    assert command[command.index("--config") + 1] == str(training.CONFIG)
    assert command[command.index("--ckpts") + 1] == str(training.CHECKPOINT)
    assert command[command.index("--max_tn_batches") + 1] == "0"
    assert command[command.index("--topk") + 1] == "1"
    split_index = command.index("--tn_splits")
    assert command[split_index + 1 : split_index + 3] == [
        "refcocop_val",
        "refcocog_umd_val",
    ]
    assert "--skip_ref" in command


def test_v54_health_contract_uses_v36_v21_v19_and_exact_trust():
    assert health.TRAINING_CONTRACT_SCHEMA == (
        "pivot.stageb.dense_duty_training_contract/v36"
    )
    assert health.MIGRATION_SCHEMA == (
        FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_MIGRATION_SCHEMA
    )
    assert health.FRESH_CONFIDENCE_CONTRACT == (
        FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_FRESH_CONFIDENCE_CONTRACT
    )
    assert health.EXPECTED_ACTIVE_TENSORS == 65
    assert health.EXPECTED_ACTIVE_ELEMENTS == 534_725
    assert health.EXPECTED_TOKEN_TENSORS == 21
    assert health.EXPECTED_GLOBAL_TENSORS == 44
    assert health._CORE.EXPECTED_CONTRACT_VALUES[
        "stage_b_dense_duty_positive_trust_contract"
    ] == "exact_frozen_rank_max_confidence_delta_v3"


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
            "storage_bytes": (
                EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_STORAGE_BYTES
            ),
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
        "stage_b_dense_duty_confidence_adapter_migration_audit": (
            _migration_audit()
        ),
    }


def test_v54_health_accepts_only_exact_v21_migration_surface():
    result = health._audit_v54_migration(_migration_args())
    assert result["schema"] == FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_MIGRATION_SCHEMA
    assert result["fresh_confidence_contract"] == (
        FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_FRESH_CONFIDENCE_CONTRACT
    )

    drift = _migration_args()
    drift["stage_b_dense_duty_confidence_adapter_migration_audit"] = copy.deepcopy(
        drift["stage_b_dense_duty_confidence_adapter_migration_audit"]
    )
    drift["stage_b_dense_duty_confidence_adapter_migration_audit"][
        "pool_feature_contract"
    ] = "drifted_pool"
    with pytest.raises(health.ProbeHealthEvidenceError):
        health._audit_v54_migration(drift)


def test_v54_postflight_replaces_v53_with_exact_residual_claim(monkeypatch):
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
    assert contracts[evaluation.EXPECTED_POSITIVE_TRUST_CONTRACT] is True
    assert contracts["deployed_global_confidence_remains_absolute"] is True
    assert contracts[
        "positive_tail_trust_uses_exact_frozen_rank_max_residual"
    ] is True
    assert "v53_rank_full_expression_global_absolute_v35" not in contracts


def _admission_source() -> str:
    return f'''\
def _bind_stage_b_confidence_probe_admission(args):
    if (
        str(getattr(args, "aggregation", "")) == {evaluation.EXPECTED_AGGREGATION!r}
        and str(getattr(args, "revision", "")) == {evaluation.EXPECTED_REVISION!r}
        and str(getattr(args, "head", "")) == {evaluation.EXPECTED_HEAD_CONTRACT!r}
        and str(getattr(args, "pool", "")) == {evaluation.EXPECTED_POOL_CONTRACT!r}
        and str(getattr(args, "gate", "")) == {evaluation.EXPECTED_GATE_CONTRACT!r}
        and str(getattr(args, "routing", "")) == {evaluation.EXPECTED_ROUTING_REDUCTION!r}
        and str(getattr(args, "trust", "")) == {evaluation.EXPECTED_TRUST_REDUCTION!r}
        and str(getattr(args, "negative", "")) == {evaluation.EXPECTED_NEGATIVE_REDUCTION!r}
        and str(getattr(args, "scope", "")) == {evaluation.EXPECTED_TOKEN_EDIT_QUERY_SCOPE!r}
        and str(getattr(args, "gradient", "")) == {evaluation.EXPECTED_POSITIVE_GRADIENT_CONTRACT!r}
        and str(getattr(args, "positive_trust", "")) == {evaluation.EXPECTED_POSITIVE_TRUST_CONTRACT!r}
        and str(getattr(args, "pair", "")) == "bidirectional_v1"
        and float(getattr(args, "stage_b_dense_duty_deployed_veto_routing_weight", -1.0)) == 0.0
        and float(getattr(args, "stage_b_dense_duty_deployed_veto_positive_max", -1.0)) == 0.1
        and float(getattr(args, "stage_b_dense_duty_deployed_veto_tn_min", -1.0)) == 0.9
        and float(getattr(args, "stage_b_dense_duty_confidence_veto_gate_offset", -1.0)) == 0.0
    ):
        formal_contract = {evaluation.FORMAL_ADMISSION_CONTRACT!r}
        from tools import {evaluation.CONTROLLER_IMPORT} as promotion
        return formal_contract, promotion
'''


def test_v54_admission_ast_requires_exact_residual_trust(tmp_path):
    source = tmp_path / "main.py"
    source.write_text(_admission_source(), encoding="utf-8")
    assert evaluation._formal_main_admission_is_wired(source) is True

    source.write_text(
        _admission_source().replace(
            evaluation.EXPECTED_POSITIVE_TRUST_CONTRACT,
            "absolute_global_confidence_logit_v2",
        ),
        encoding="utf-8",
    )
    assert evaluation._formal_main_admission_is_wired(source) is False


def test_v54_formal_controller_is_bound_to_the_promoted_config(monkeypatch):
    assert formal.CONFIG.resolve() == evaluation.FORMAL_CONFIG.resolve()
    assert formal.UPDATES == 4412
    assert formal.CHECKPOINT == formal.OUTPUT / "checkpoint_iter.pth"
    sentinel = {"formal_training_admitted": True}
    monkeypatch.setattr(evaluation, "verify_admission_report", lambda: sentinel)
    assert formal.verify_probe_admission() is sentinel
