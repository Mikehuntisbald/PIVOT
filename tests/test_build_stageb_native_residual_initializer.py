from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from models.GroundingDINO.stage_b_gdino_score_adapter import (
    StageBGDINOScoreAdapter,
)
from tools.build_stageb_native_residual_initializer import (
    ADAPTER_PREFIX,
    NativeResidualInitializerError,
    compose_native_residual_state,
    functional_identity_check,
)


def _template_and_source():
    adapter = StageBGDINOScoreAdapter(
        hidden_dim=8,
        adapter_dim=4,
        gate_hidden_dim=4,
        gate_pool_temperature=0.1,
        gate_topk=2,
    )
    source = OrderedDict(
        {
            "backbone.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "decoder.bias": torch.arange(2, dtype=torch.float32),
        }
    )
    template = OrderedDict((key, value.clone()) for key, value in source.items())
    template.update(
        (ADAPTER_PREFIX + key, value.clone())
        for key, value in adapter.state_dict().items()
    )
    return adapter, template, source


def test_compose_native_residual_state_is_exhaustive_and_preserves_b58():
    _adapter, template, source = _template_and_source()
    state, roles = compose_native_residual_state(template, source)

    assert set(state) == set(template)
    assert set(roles) == {"b58_base", "random_identity_adapter"}
    assert set(roles["b58_base"]) == set(source)
    assert len(roles["random_identity_adapter"]) == 20
    for key, value in source.items():
        assert torch.equal(state[key], value)


def test_compose_rejects_unbound_or_polluted_source_keys():
    _adapter, template, source = _template_and_source()
    polluted = OrderedDict(source)
    polluted["foreign.weight"] = torch.ones(1)
    with pytest.raises(NativeResidualInitializerError, match="do not map exactly"):
        compose_native_residual_state(template, polluted)

    template["unexpected.weight"] = torch.ones(1)
    with pytest.raises(NativeResidualInitializerError, match="unbound tensor"):
        compose_native_residual_state(template, source)


def test_functional_identity_requires_zero_terminal_outputs():
    adapter, _template, _source = _template_and_source()
    checks = functional_identity_check(adapter)
    assert all(checks.values())

    with torch.no_grad():
        adapter.rank_output.bias.fill_(0.25)
    with pytest.raises(NativeResidualInitializerError, match="not exact zero"):
        functional_identity_check(adapter)
