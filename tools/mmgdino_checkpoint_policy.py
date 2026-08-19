"""Resource-only activation-checkpoint policies for MM-Grounding-DINO.

This module deliberately changes no parameter, buffer, loss, or optimizer
ownership.  It fills the coarse gap between ``encoder.num_cp=4`` and
``encoder.num_cp=5`` by checkpointing exactly one component of encoder layer
index 4.  The public experiment config must import this module explicitly.
"""

from __future__ import annotations

import functools
from typing import Iterable

from fairscale.nn.checkpoint import checkpoint_wrapper
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


def is_fairscale_checkpointed(module) -> bool:
    """Return whether ``checkpoint_wrapper`` patched ``module.forward``."""

    forward = module.__dict__.get("forward")
    return (
        isinstance(forward, functools.partial)
        and getattr(forward.func, "__name__", "") == "_checkpointed_forward"
    )


@HOOKS.register_module()
class MMGDinoPartialEncoderCheckpointHook(Hook):
    """Checkpoint selected submodules of one otherwise-unwrapped layer.

    Args:
        layer_index: Zero-based transformer encoder layer.
        components: Any subset of ``("visual", "fusion")``.  ``visual`` maps
            to ``encoder.layers`` and ``fusion`` to
            ``encoder.fusion_layers``.
        expected_layers: Fail-closed architecture check.
        expected_wrapped_prefix: Layers below this index must already have
            both components checkpointed by the detector config.
    """

    priority = "VERY_HIGH"

    def __init__(
        self,
        layer_index: int = 4,
        components: Iterable[str] = ("fusion",),
        expected_layers: int = 6,
        expected_wrapped_prefix: int = 4,
    ) -> None:
        self.layer_index = int(layer_index)
        self.components = tuple(components)
        self.expected_layers = int(expected_layers)
        self.expected_wrapped_prefix = int(expected_wrapped_prefix)
        allowed = {"visual", "fusion"}
        if not self.components or len(set(self.components)) != len(self.components):
            raise ValueError("components must be a non-empty unique sequence")
        if not set(self.components) <= allowed:
            raise ValueError(f"components must be a subset of {sorted(allowed)}")
        if not 0 <= self.layer_index < self.expected_layers:
            raise ValueError("layer_index is outside expected encoder layers")
        self._applied = False

    def before_train(self, runner) -> None:
        if self._applied:
            raise RuntimeError("partial checkpoint policy was applied twice")

        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        if not hasattr(model, "encoder"):
            raise RuntimeError("model has no MM-Grounding-DINO encoder")
        encoder = model.encoder
        if (
            len(encoder.layers) != self.expected_layers
            or len(encoder.fusion_layers) != self.expected_layers
        ):
            raise RuntimeError("unexpected MM-Grounding-DINO encoder depth")

        for index in range(self.expected_wrapped_prefix):
            if not is_fairscale_checkpointed(encoder.layers[index]):
                raise RuntimeError(f"encoder visual layer {index} is not checkpointed")
            if not is_fairscale_checkpointed(encoder.fusion_layers[index]):
                raise RuntimeError(f"encoder fusion layer {index} is not checkpointed")

        mapping = {
            "visual": encoder.layers,
            "fusion": encoder.fusion_layers,
        }
        state_keys_before = tuple(model.state_dict().keys())
        parameter_ids_before = tuple(id(parameter) for parameter in model.parameters())
        for component in self.components:
            target = mapping[component][self.layer_index]
            if is_fairscale_checkpointed(target):
                raise RuntimeError(
                    f"encoder {component} layer {self.layer_index} was already checkpointed"
                )
            checkpoint_wrapper(target)
            if not is_fairscale_checkpointed(target):
                raise RuntimeError("FairScale checkpoint wrapper did not take effect")

        if tuple(model.state_dict().keys()) != state_keys_before:
            raise RuntimeError("checkpoint policy changed model state-dict keys")
        if tuple(id(parameter) for parameter in model.parameters()) != parameter_ids_before:
            raise RuntimeError("checkpoint policy changed parameter ownership")

        self._applied = True
        runner.logger.info(
            "Applied resource-only partial encoder checkpoint policy: "
            "layer=%d components=%s",
            self.layer_index,
            ",".join(self.components),
        )
