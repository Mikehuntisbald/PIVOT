from tools import run_stageb_confidence_adapter_formal as original
from tools import run_stageb_confidence_adapter_highmem_formal as controller


def test_highmem_recipe_preserves_training_semantics():
    values = controller._formal_current_args()
    assert values["batch_size"] == 16
    assert values["gradient_accumulation_steps"] == 2
    assert values["stage_b_dense_duty_forward_pack_factor"] == 2
    assert values["stage_b_dense_duty_logical_loss_batch_size"] == 16
    assert values["stage_b_dense_duty_expected_forward_batch_size"] == 32
    assert values["stage_b_dense_duty_expected_logical_batches_per_epoch"] == 887
    assert values["stage_b_dense_duty_expected_physical_forwards_per_epoch"] == 444
    assert values["stage_b_v11_expression_microbatch"] == 64
    assert values["stage_b_dense_duty_expected_expression_microbatch"] == 64
    assert values["max_train_iters"] == 4412
    assert values["stage_b_v22_score_ownership"] == (
        "rank_tower_stopgrad_token_adapter_two_phase"
    )
    assert values[controller.SOURCE_CLOSURE_ARG]["config"]["entry"] == (
        "config/ablations/"
        "cfg_stageb_dense_duty_confidence_adapter_20260730.py"
    )


def test_highmem_controller_uses_an_isolated_output():
    assert controller.CONFIG == original.CONFIG
    assert controller.OUTPUT != original.OUTPUT
    assert controller.CHECKPOINT != original.CHECKPOINT
    assert original.CONFIG.name == (
        "cfg_stageb_dense_duty_confidence_adapter_20260730.py"
    )
    assert "dense_duty_adapter_20260730" in str(original.OUTPUT)
    assert "dense_duty_adapter_packed_highmem_20260730" in str(
        controller.OUTPUT
    )


def test_highmem_start_command_uses_fresh_rank_initializer():
    argv = controller.command("start")
    assert argv[argv.index("--config_file") + 1] == str(controller.CONFIG)
    assert argv[argv.index("--output_dir") + 1] == str(controller.OUTPUT)
    assert "--pretrain_model_path" in argv
    assert "--resume" not in argv
    assert argv[argv.index("--gradient_accumulation_steps") + 1] == "2"
