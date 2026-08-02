from tools import (
    run_stageb_confidence_adapter_veto_gated_pool_rank_affine_probe as probe,
)
from tools import (
    run_stageb_confidence_adapter_veto_gated_pool_rank_affine_probe_u0050 as probe_u0050,
)
from util.stage_b_dense_duty_audit import build_training_contract


def test_rank_affine_v12_probe_is_fresh_and_binds_u6551():
    values = probe_u0050._BASE._formal_current_args()
    assert values["stage_b_dense_duty_confidence_revision"] == (
        "word_veto_gated_pool_rank_affine_v12"
    )
    assert values["stage_b_dense_duty_confidence_rank_evidence_contract"] == (
        "zero_init_rank_logit_affine_v2"
    )
    assert values["stage_b_dense_duty_raw_veto_query_scope"] == (
        "tn_all_admitted_carrier_balanced_positive_carrier_paired_v4"
    )
    assert values["stage_b_dense_duty_raw_veto_tn_carrier_balance"] == 0.25
    assert values["stage_b_dense_duty_raw_veto_positive_carrier_balance"] == 0.0
    assert values["stage_b_dense_duty_raw_veto_carrier_pair_weight"] == 0.25
    assert values["stage_b_dense_duty_raw_veto_carrier_pair_margin"] == 0.25
    assert values["stage_b_v11_trainable_params_min"] == 185_927
    assert values["stage_b_v11_trainable_params_max"] == 185_927
    assert values["stage_b_dense_duty_rank_source_optimizer_updates"] == 6551

    contract = build_training_contract(values)
    assert contract["schema"] == (
        "pivot.stageb.dense_duty_training_contract/v14"
    )
    assert contract["values"][
        "stage_b_dense_duty_confidence_rank_evidence_contract"
    ] == "zero_init_rank_logit_affine_v2"

    command = probe_u0050.command("start")
    assert "--resume" not in command
    assert command[command.index("--max_train_iters") + 1] == "50"
    assert command[command.index("--pretrain_model_path") + 1] == str(
        probe_u0050._BASE.RANK_SOURCE
    )


def test_rank_affine_v12_u300_uses_independent_output():
    command = probe.command("start")
    assert "--resume" not in command
    assert command[command.index("--max_train_iters") + 1] == "300"
    assert "dense_duty_adapter_veto_gated_pool_rank_affine_highmem_20260731" in str(
        probe.OUTPUT
    )
