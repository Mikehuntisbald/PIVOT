from pathlib import Path

from tools import (
    audit_stageb_confidence_adapter_fulltext_global_independent_absolute_probe_health
    as health,
)
from tools import (
    run_stageb_confidence_adapter_fulltext_global_independent_absolute_highmem_formal
    as formal,
)
from tools import (
    run_stageb_confidence_adapter_fulltext_global_independent_absolute_probe_evaluation
    as evaluation,
)
from tools import (
    run_stageb_confidence_adapter_fulltext_global_independent_absolute_probe_u0400
    as training,
)
from util.slconfig import SLConfig
from util.stage_b_confidence_adapter_migration import (
    FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT,
    FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_MIGRATION_SCHEMA,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
V54_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_fulltext_global_absolute_"
    "exact_residual_20260802.py"
)


def _config(path: Path) -> dict:
    return SLConfig.fromfile(str(path))._cfg_dict.to_dict()


def test_v55_changes_only_its_eight_declared_v54_contract_fields():
    v54 = _config(V54_CONFIG)
    v55 = _config(evaluation.FORMAL_CONFIG)
    assert set(v55) == set(v54)
    assert {key for key in v54 if v54[key] != v55[key]} == {
        "stage_b_dense_duty_confidence_revision",
        "stage_b_dense_duty_confidence_head_gradient_contract",
        "stage_b_dense_duty_confidence_pool_feature_contract",
        "stage_b_dense_duty_positive_trust_contract",
        "stage_b_dense_duty_trace_audit_path",
        "stage_b_dense_duty_trace_audit_sha256",
        "stage_b_dense_duty_confidence_probe_admission_contract",
        "stage_b_dense_duty_confidence_probe_admission_report",
    }
    assert v55["stage_b_v11_trainable_params_min"] == 534_725
    assert v55["stage_b_v11_trainable_params_max"] == 534_725


def test_v55_training_controller_is_fresh_u400():
    assert training.UPDATES == 400
    assert training.OUTPUT.name == "u000400_fresh"
    assert training.CHECKPOINT == training.OUTPUT / "checkpoint_iter.pth"
    command = training.command("start")
    assert "--resume" not in command
    assert command[command.index("--max_train_iters") + 1] == "400"
    assert command[command.index("--pretrain_model_path") + 1] == str(
        training._BASE.RANK_SOURCE
    )


def test_v55_health_seals_v37_v22_v20_and_two_owner_surface():
    assert health.TRAINING_CONTRACT_SCHEMA == (
        "pivot.stageb.dense_duty_training_contract/v37"
    )
    assert health.MIGRATION_SCHEMA == (
        FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_MIGRATION_SCHEMA
    )
    assert health.FRESH_CONFIDENCE_CONTRACT == (
        FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT
    )
    assert health.EXPECTED_ACTIVE_TENSORS == 65
    assert health.EXPECTED_ACTIVE_ELEMENTS == 534_725
    assert health.EXPECTED_TOKEN_TENSORS == 21
    assert health.EXPECTED_GLOBAL_TENSORS == 44
    assert health._CORE.EXPECTED_CONTRACT_VALUES[
        "stage_b_dense_duty_positive_trust_contract"
    ] == "absolute_global_pool_logit_v4"


def test_v55_strict1607_command_and_gate_are_exact():
    assert evaluation.BASELINE_FALSE_ACCEPTS == 801
    assert evaluation.MAX_ADMITTED_FALSE_ACCEPTS == 800
    command = evaluation.build_command()
    assert command[command.index("--config") + 1] == str(training.CONFIG)
    assert command[command.index("--ckpts") + 1] == str(training.CHECKPOINT)
    assert command[command.index("--max_tn_batches") + 1] == "0"
    assert command[command.index("--topk") + 1] == "1"
    assert "--skip_ref" in command
    split = command.index("--tn_splits")
    assert command[split + 1 : split + 3] == [
        "refcocop_val",
        "refcocog_umd_val",
    ]


def test_v55_postflight_replaces_v54_carrier_claims(monkeypatch):
    monkeypatch.setattr(
        evaluation,
        "_BASE_POSTFLIGHT",
        lambda *_args, **_kwargs: {
            "contracts": {
                "terminal_u400_diagnostic": True,
                "v54_rank_full_expression_global_absolute_exact_residual_v36": True,
                "positive_tail_trust_uses_exact_frozen_rank_max_residual": True,
                "frozen_rank_full_expression_u0_carrier": True,
            }
        },
    )
    contracts = evaluation._v55_postflight({})["contracts"]
    assert contracts[
        "v55_rank_full_expression_global_independent_absolute_v37"
    ] is True
    assert contracts["deployed_global_confidence_is_pool_absolute_only"] is True
    assert contracts["local_candidate_logits_excluded_from_deployed_global"] is True
    assert contracts[
        "global_tn_pair_queue_and_inference_use_pool_absolute_only"
    ] is True
    assert "positive_tail_trust_uses_exact_frozen_rank_max_residual" not in contracts
    assert "frozen_rank_full_expression_u0_carrier" not in contracts


def test_v55_main_admission_is_wired_to_canonical_controller():
    assert evaluation._formal_main_admission_is_wired(REPO_ROOT / "main.py")
    assert evaluation.CONTROLLER_IMPORT == (
        "run_stageb_confidence_adapter_fulltext_global_independent_absolute_"
        "probe_evaluation"
    )


def test_v55_formal_controller_is_fresh_u4412_and_admission_bound(monkeypatch):
    assert formal.CONFIG.resolve() == evaluation.FORMAL_CONFIG.resolve()
    assert formal.UPDATES == 4412
    assert formal.CHECKPOINT == formal.OUTPUT / "checkpoint_iter.pth"
    command = formal.command("start")
    assert "--resume" not in command
    assert command[command.index("--max_train_iters") + 1] == "4412"
    sentinel = {"formal_training_admitted": True}
    monkeypatch.setattr(evaluation, "verify_admission_report", lambda: sentinel)
    assert formal.verify_probe_admission() is sentinel
