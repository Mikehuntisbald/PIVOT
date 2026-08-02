from pathlib import Path

import pytest

from tools import run_stageb_confidence_adapter_formal as controller


def test_formal_current_args_are_rebuilt_from_fixed_recipe():
    values = controller._formal_current_args()
    assert values["config_file"] == str(controller.CONFIG)
    assert values["datasets"] == str(controller.DATASET)
    assert values["output_dir"] == str(controller.OUTPUT)
    assert values["stage_b_v22_score_ownership"] == (
        "rank_tower_stopgrad_token_adapter_two_phase"
    )
    assert values["stage_b_dense_duty_confidence_token_contract"] == (
        "detached_rank_token_minus_zero_init_residual_v1"
    )
    assert values["stage_b_dense_duty_confidence_pool_feature_contract"] == (
        "patch_statistics_only_v1"
    )
    assert values["batch_size"] == 16
    assert values["gradient_accumulation_steps"] == 2
    assert values["max_train_iters"] == 4412
    assert values["amp"] is True
    assert values["distributed"] is False
    assert values[controller.SOURCE_CLOSURE_ARG]["config"]["entry"] == (
        "config/ablations/cfg_stageb_dense_duty_confidence_adapter_20260730.py"
    )


def test_partial_inspection_compares_checkpoint_to_current_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "confidence"
    output.mkdir()
    checkpoint = output / "checkpoint_iter.pth"
    checkpoint.touch()
    saved_args = {
        "stage_b_v22_score_ownership": (
            "rank_tower_stopgrad_token_adapter_two_phase"
        )
    }
    payload = {
        "args": saved_args,
        "optimizer_updates": 1,
        "model": {
            "stage_b_fixed_text_scorer.rank_tower.weight": object(),
        },
    }
    intended = {"sentinel": "current-formal-recipe"}
    observed = {}

    def validate_resume(value, current, *, checkpoint_path):
        observed["payload"] = value
        observed["current"] = current
        observed["checkpoint_path"] = checkpoint_path

    monkeypatch.setattr(controller, "OUTPUT", output)
    monkeypatch.setattr(controller, "CHECKPOINT", checkpoint)
    monkeypatch.setattr(controller, "_load", lambda _path: payload)
    monkeypatch.setattr(controller, "_validate_migration", lambda _args: {})
    monkeypatch.setattr(controller, "_formal_current_args", lambda: intended)
    monkeypatch.setattr(
        controller, "validate_strict_resume_checkpoint_payload", validate_resume
    )
    monkeypatch.setattr(
        controller, "audit_checkpoint_payload", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        controller,
        "fingerprint_named_tensors",
        lambda *_args, **_kwargs: {"sha256": controller.RANK_SHA256},
    )

    assert controller.inspect() == {
        "status": "partial",
        "action": "resume",
        "updates": 1,
    }
    assert observed["payload"] is payload
    assert observed["current"] is intended
    assert observed["current"] is not saved_args
    assert observed["checkpoint_path"] == checkpoint


def test_terminal_training_state_rejects_missing_tail_queue():
    payload = {
        key: {}
        for key in controller.STRICT_RESUME_REQUIRED_KEYS
        if key not in {
            "epoch",
            "iteration",
            "optimizer_updates",
            "epoch_finished",
            "checkpoint_reason",
        }
    }
    payload.update(
        {
            "epoch": 0,
            "iteration": 4,
            "optimizer_updates": controller.UPDATES,
            "epoch_finished": False,
            "checkpoint_reason": "max_train_iters",
        }
    )
    with pytest.raises(controller.ControllerError, match="tail-queue schema"):
        controller._validate_terminal_training_state(
            payload,
            {
                "stage_b_v14_tail_queue_size": 4096,
                "gradient_accumulation_steps": 4,
                "amp": True,
            },
        )
