from tools import (
    run_stageb_confidence_adapter_veto_gated_pool_tail_paired_probe as probe,
)
from tools import (
    run_stageb_confidence_adapter_veto_gated_pool_tail_paired_probe_u0050 as probe_u0050,
)
from tools import (
    run_stageb_confidence_adapter_veto_gated_pool_tail_paired_probe_u0100 as probe_u0100,
)
from tools import (
    run_stageb_confidence_adapter_veto_gated_pool_tail_paired_rank_channel_probe_u0050 as rank_channel_u0050,
)
from tools import (
    run_stageb_confidence_adapter_veto_gated_pool_tail_paired_signed_rank_pool_probe_u0050 as signed_rank_pool_u0050,
)
from util.stage_b_dense_duty_audit import build_training_contract


def test_tail_paired_v18_probe_is_fresh_and_binds_u6551():
    values = probe._BASE._formal_current_args()
    assert values["stage_b_dense_duty_confidence_revision"] == (
        "word_veto_gated_pool_tail_paired_v18"
    )
    assert values["stage_b_dense_duty_raw_veto_query_scope"] == (
        "tn_all_admitted_tail_weighted_carrier_tail_paired_v7"
    )
    assert values["stage_b_dense_duty_confidence_gate_gradient_contract"] == (
        "hard_detached_v1"
    )
    assert values["stage_b_dense_duty_raw_veto_tail_quantile"] == 0.95
    assert values["stage_b_dense_duty_raw_veto_tail_temperature"] == 0.1
    assert values["stage_b_dense_duty_raw_veto_tail_min_count"] == 256
    assert values["stage_b_dense_duty_confidence_expected_optimizer_updates"] == 300
    assert values["stage_b_dense_duty_execution_scope"] == "probe"
    assert values["resume"] == str(probe.CHECKPOINT)
    assert values["stage_b_dense_duty_rank_source_checkpoint_sha256"] == (
        probe._BASE.SOURCE_SHA256
    )
    assert build_training_contract(values)["schema"] == (
        "pivot.stageb.dense_duty_training_contract/v20"
    )


def test_tail_paired_v18_u50_uses_independent_output():
    values = probe_u0050._BASE._formal_current_args()
    assert values["stage_b_dense_duty_confidence_expected_optimizer_updates"] == 50
    assert values["max_train_iters"] == 50
    assert probe_u0050.OUTPUT != probe.OUTPUT
    assert "tail_paired_highmem_20260731" in str(probe_u0050.OUTPUT)


def test_tail_paired_v18_u100_is_a_fresh_deterministic_screen():
    values = probe_u0100._BASE._formal_current_args()
    assert values["stage_b_dense_duty_confidence_revision"] == (
        "word_veto_gated_pool_tail_paired_v18"
    )
    assert values["stage_b_dense_duty_confidence_expected_optimizer_updates"] == 100
    assert values["max_train_iters"] == 100
    assert values["pretrain_model_path"] is None
    assert values["resume"] == str(probe_u0100.CHECKPOINT)
    assert probe_u0100.OUTPUT not in {probe_u0050.OUTPUT, probe.OUTPUT}
    assert probe_u0100.command("start")[-2:] == [
        "--pretrain_model_path",
        str(probe_u0100._BASE.RANK_SOURCE),
    ]


def test_rank_channel_v19_u50_is_fresh_and_preserves_v18_training_contract():
    values = rank_channel_u0050._BASE._formal_current_args()
    assert values["stage_b_dense_duty_confidence_revision"] == (
        "word_veto_gated_pool_tail_paired_rank_channel_v19"
    )
    assert values["stage_b_dense_duty_confidence_rank_evidence_contract"] == (
        "zero_init_carrier_token_rank_affine_sparse_rank_channel_v6"
    )
    assert values["stage_b_dense_duty_raw_veto_query_scope"] == (
        "tn_all_admitted_tail_weighted_carrier_tail_paired_v7"
    )
    assert values["stage_b_dense_duty_confidence_expected_optimizer_updates"] == 50
    assert values["max_train_iters"] == 50
    assert values["stage_b_v11_trainable_params_min"] == 203_142
    assert values["stage_b_v11_trainable_params_max"] == 203_142
    assert values["pretrain_model_path"] is None
    assert values["resume"] == str(rank_channel_u0050.CHECKPOINT)
    assert rank_channel_u0050.OUTPUT not in {
        probe_u0050.OUTPUT,
        probe_u0100.OUTPUT,
        probe.OUTPUT,
    }
    assert rank_channel_u0050.command("start")[-2:] == [
        "--pretrain_model_path",
        str(rank_channel_u0050._BASE.RANK_SOURCE),
    ]
    assert build_training_contract(values)["schema"] == (
        "pivot.stageb.dense_duty_training_contract/v20"
    )


def test_signed_rank_pool_v20_u50_has_an_independent_exact_launch_contract():
    values = signed_rank_pool_u0050._BASE._formal_current_args()
    assert values["stage_b_dense_duty_confidence_revision"] == (
        "word_veto_gated_pool_tail_paired_signed_rank_pool_v20"
    )
    assert values["stage_b_dense_duty_confidence_pool_feature_contract"] == (
        "detached_rank_query_plus_patch_statistics_signed_residual_v2"
    )
    assert values["stage_b_dense_duty_confidence_rank_evidence_contract"] == (
        "zero_init_carrier_token_rank_affine_sparse_rank_channel_v6"
    )
    assert values[
        "stage_b_dense_duty_confidence_residual_parameterization_gain"
    ] == (0.25 / 0.03)
    assert values["stage_b_v11_trainable_params_min"] == 236_806
    assert values["stage_b_v11_trainable_params_max"] == 236_806
    assert values["stage_b_dense_duty_confidence_expected_optimizer_updates"] == 50
    assert values["max_train_iters"] == 50
    assert values["pretrain_model_path"] is None
    assert values["resume"] == str(signed_rank_pool_u0050.CHECKPOINT)
    assert signed_rank_pool_u0050.CONFIG != rank_channel_u0050.CONFIG
    assert signed_rank_pool_u0050.OUTPUT not in {
        rank_channel_u0050.OUTPUT,
        probe_u0050.OUTPUT,
        probe_u0100.OUTPUT,
        probe.OUTPUT,
    }

    start_command = signed_rank_pool_u0050.command("start")
    assert start_command[start_command.index("--config_file") + 1] == str(
        signed_rank_pool_u0050.CONFIG
    )
    assert start_command[start_command.index("--output_dir") + 1] == str(
        signed_rank_pool_u0050.OUTPUT
    )
    assert start_command[start_command.index("--max_train_iters") + 1] == "50"
    assert start_command[-2:] == [
        "--pretrain_model_path",
        str(signed_rank_pool_u0050._BASE.RANK_SOURCE),
    ]
    assert "--resume" not in start_command
    assert build_training_contract(values)["schema"] == (
        "pivot.stageb.dense_duty_training_contract/v20"
    )
