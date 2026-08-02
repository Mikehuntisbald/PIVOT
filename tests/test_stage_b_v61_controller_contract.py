from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import audit_stageb_confidence_full_decoder_verifier_probe_health as health
from tools import run_stageb_confidence_full_decoder_verifier_formal as formal
from tools import (
    run_stageb_confidence_full_decoder_verifier_probe_evaluation as evaluation,
)
from tools import run_stageb_confidence_full_decoder_verifier_probe_u0400 as training
from util.slconfig import SLConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_v61_health_seals_exact_capacity_and_owner_surface():
    assert health.TRAINING_CONTRACT_SCHEMA.endswith("/v42")
    assert health.MIGRATION_SCHEMA.endswith("confidence_verifier/v26")
    assert health.EXPECTED_ACTIVE_TENSORS == 368
    assert health.EXPECTED_ACTIVE_ELEMENTS == 25_664_258
    assert health.EXPECTED_TOKEN_TENSORS == 356
    assert health.EXPECTED_TOKEN_ELEMENTS == 25_464_320
    assert health.EXPECTED_GLOBAL_TENSORS == 12
    assert health.EXPECTED_GLOBAL_ELEMENTS == 199_938
    assert health.EXPECTED_ADAPTER_TENSORS == 362
    assert health.EXPECTED_POOL_TENSORS == 6


def test_v61_health_replays_the_real_u0_migration_receipt():
    path = training.OUTPUT / "config_args_all.json"
    args = json.loads(path.read_text(encoding="utf-8"))
    migration_path = training.OUTPUT / "stage_b_confidence_adapter_migration_audit.json"
    args["stage_b_dense_duty_confidence_adapter_migration_audit"] = json.loads(
        migration_path.read_text(encoding="utf-8")
    )
    audit = health._audit_v61_migration(args)
    assert audit["verifier_matches_rank"] is True
    assert audit["verifier_veto_output_nonzero_count"] == 0
    assert audit["pool_output_nonzero_count"] == 0


def test_v61_direct_training_contract_is_fail_closed(monkeypatch):
    sentinel = {"schema": health.TRAINING_CONTRACT_SCHEMA}
    monkeypatch.setattr(
        health,
        "_BASE_AUDIT_TRAINING_CONTRACT",
        lambda _args: sentinel,
    )
    args = {
        "stage_b_dense_duty_confidence_full_decoder_verifier": True,
        "stage_b_dense_duty_confidence_capacity_contract": (
            health.EXPECTED_CAPACITY_CONTRACT
        ),
        "stage_b_dense_duty_confidence_variant": health.EXPECTED_VARIANT,
        "stage_b_dense_duty_deployed_global_absolute_weight": 0.0,
    }
    assert health._audit_v61_training_contract(args) is sentinel
    args["stage_b_dense_duty_confidence_full_decoder_verifier"] = False
    with pytest.raises(health.ProbeHealthEvidenceError, match="V61 direct"):
        health._audit_v61_training_contract(args)


def test_v61_probe_evaluator_and_formal_controller_are_exact():
    command = evaluation.build_command()
    assert str(training.CONFIG) in command
    assert str(training.CHECKPOINT) in command
    assert evaluation.MAX_ADMITTED_FALSE_ACCEPTS == 800
    assert formal.UPDATES == 4412
    assert "--resume" not in formal.command("start")
    cfg = SLConfig.fromfile(str(formal.CONFIG))
    assert cfg.stage_b_dense_duty_execution_scope == "formal"
    assert cfg.stage_b_dense_duty_evaluation_scope == "formal"
    assert cfg.stage_b_dense_duty_confidence_expected_optimizer_updates == 4412
    assert cfg.stage_b_dense_duty_confidence_probe_admission_contract == (
        evaluation.FORMAL_ADMISSION_CONTRACT
    )
    assert cfg.stage_b_dense_duty_confidence_probe_admission_report == str(
        evaluation.REPORT
    )


def test_v61_postflight_relabels_inherited_veto_contract(monkeypatch):
    monkeypatch.setattr(
        evaluation._V60,
        "_v60_postflight",
        lambda *_args, **_kwargs: {
            "contracts": {
                "v60_deployment_owned_query_veto_representation_v42": True
            }
        },
    )
    contracts = evaluation._v61_postflight({})["contracts"]
    assert "v60_deployment_owned_query_veto_representation_v42" not in contracts
    assert contracts["v61_rank_cloned_full_decoder_verifier_v26"] is True
    assert contracts[health.EXPECTED_CAPACITY_CONTRACT] is True
    assert contracts["verifier_has_no_free_signed_absolute_score"] is True
