#!/usr/bin/env python3
"""Run the sealed new-head probe evaluator with its CLI device in the config.

The preregistered evaluator computes the effective device contract from the
CLI but omitted the equivalent assignment required by the model builder. Keep
that evaluator byte-for-byte sealed during LR selection and apply only this
runtime plumbing fix, identically, to all three candidates.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import eval_stageb_data_driven_new_head_dev as evaluator


_load_model = evaluator._load_model
_forward = evaluator._forward


def _load_model_with_cli_device(cfg, checkpoint: str, device):
    configured = getattr(cfg, "device", None)
    if configured is not None and str(configured) != str(device):
        raise evaluator.NewHeadDevEvalError(
            f"config/CLI device drifted: config={configured!r}, cli={str(device)!r}"
        )
    cfg.device = str(device)
    return _load_model(cfg, checkpoint, device)


def _forward_with_source_identity(model, batch, device, *, amp: bool, cfg):
    raw_targets = list(batch[1])
    outputs, targets = _forward(model, batch, device, amp=amp, cfg=cfg)
    if len(targets) != len(raw_targets):
        raise evaluator.NewHeadDevEvalError(
            "forward target count drifted while restoring source identity"
        )
    for row, (raw_target, target) in enumerate(zip(raw_targets, targets)):
        source = raw_target.get("dataset_name")
        if not isinstance(source, str) or not source:
            raise evaluator.NewHeadDevEvalError(
                f"raw target {row} has no source identity"
            )
        observed = target.get("dataset_name")
        if observed is not None and observed != source:
            raise evaluator.NewHeadDevEvalError(
                f"forward target {row} source identity drifted"
            )
        target["dataset_name"] = source
    return outputs, targets


def main() -> None:
    evaluator._load_model = _load_model_with_cli_device
    evaluator._forward = _forward_with_source_identity
    evaluator.main()


if __name__ == "__main__":
    main()
