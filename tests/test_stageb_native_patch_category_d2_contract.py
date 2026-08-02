from types import SimpleNamespace

import pytest
import torch

from tools.build_stageb_native_patch_category_initializer import (
    RANDOM_TRAINABLE_PATCH_KEYS,
)
from tools.stageb_native_patch_category_d2_contract import (
    NativePatchCategoryD2ContractError,
    audit_d2_source_transition,
)


def _payloads():
    frozen_key = "backbone.weight"
    initial_state = {
        frozen_key: torch.tensor([1.0]),
        **{
            key: torch.zeros(1, dtype=torch.float32)
            for key in RANDOM_TRAINABLE_PATCH_KEYS
        },
    }
    source_state = {key: value.clone() for key, value in initial_state.items()}
    for index, key in enumerate(sorted(RANDOM_TRAINABLE_PATCH_KEYS), start=1):
        source_state[key].fill_(float(index))
    initializer = {"model": initial_state}
    source = {
        "model": source_state,
        "optimizer_updates": 500,
        "checkpoint_reason": "max_train_iters",
        "iteration": 1000,
        "epoch": 0,
        "epoch_finished": False,
        "args": SimpleNamespace(
            max_train_iters=500,
            gradient_accumulation_steps=2,
            batch_size=36,
            seed=42,
            stage_b_native_patch_execution_scope=(
                "native_patch_category_d1_u500_v1"
            ),
        ),
    }
    expected = {key: value.clone() for key, value in initial_state.items()}
    return expected, initializer, source


def test_exact_d1_transition_is_accepted():
    expected, initializer, source = _payloads()
    audit = audit_d2_source_transition(expected, initializer, source)

    assert audit["changed_tensor_count"] == 8
    assert audit["changed_tensor_keys"] == sorted(RANDOM_TRAINABLE_PATCH_KEYS)
    assert audit["all_frozen_tensors_bitwise_equal_b58_initializer"] is True


def test_frozen_drift_is_rejected():
    expected, initializer, source = _payloads()
    source["model"]["backbone.weight"].add_(1.0)

    with pytest.raises(NativePatchCategoryD2ContractError, match="exactly"):
        audit_d2_source_transition(expected, initializer, source)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("optimizer_updates", 499, "update count"),
        ("iteration", 999, "iteration checkpoint"),
        ("checkpoint_reason", "epoch", "iteration checkpoint"),
    ],
)
def test_training_metadata_drift_is_rejected(field, value, message):
    expected, initializer, source = _payloads()
    source[field] = value

    with pytest.raises(NativePatchCategoryD2ContractError, match=message):
        audit_d2_source_transition(expected, initializer, source)


def test_nonfinite_trainable_tensor_is_rejected():
    expected, initializer, source = _payloads()
    source["model"][next(iter(RANDOM_TRAINABLE_PATCH_KEYS))].fill_(torch.nan)

    with pytest.raises(NativePatchCategoryD2ContractError, match="non-finite"):
        audit_d2_source_transition(expected, initializer, source)
