from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch.nn as nn

pytest.importorskip("mmengine")
pytest.importorskip("fairscale")

from fairscale.nn.checkpoint import checkpoint_wrapper

from tools.mmgdino_checkpoint_policy import (
    MMGDinoPartialEncoderCheckpointHook,
    is_fairscale_checkpointed,
)


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(6)])
        self.fusion_layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(6)])
        for index in range(4):
            checkpoint_wrapper(self.layers[index])
            checkpoint_wrapper(self.fusion_layers[index])


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()


class _Logger:
    def info(self, *_args, **_kwargs) -> None:
        return None


def test_partial_fusion_checkpoint_preserves_state_and_parameters() -> None:
    model = _Model()
    state_keys = tuple(model.state_dict())
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    runner = SimpleNamespace(model=model, logger=_Logger())
    hook = MMGDinoPartialEncoderCheckpointHook(components=("fusion",))

    hook.before_train(runner)

    assert not is_fairscale_checkpointed(model.encoder.layers[4])
    assert is_fairscale_checkpointed(model.encoder.fusion_layers[4])
    assert tuple(model.state_dict()) == state_keys
    assert tuple(id(parameter) for parameter in model.parameters()) == parameter_ids


def test_policy_fails_closed_on_wrong_prefix() -> None:
    model = _Model()
    model.encoder.layers[0] = nn.Linear(4, 4)
    runner = SimpleNamespace(model=model, logger=_Logger())
    hook = MMGDinoPartialEncoderCheckpointHook(components=("visual",))
    with pytest.raises(RuntimeError, match="not checkpointed"):
        hook.before_train(runner)


@pytest.mark.parametrize("components", [(), ("bad",), ("fusion", "fusion")])
def test_invalid_components_fail_closed(components) -> None:
    with pytest.raises(ValueError):
        MMGDinoPartialEncoderCheckpointHook(components=components)
