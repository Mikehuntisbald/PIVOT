#!/usr/bin/env python3
"""Fail-closed ownership/runtime audit for a U2-v2 training milestone."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import build_model_main  # noqa: E402
from tools.build_stageb_u2v2_initializer import (  # noqa: E402
    U2V2InitializerError,
    validate_runtime_payload,
)
from tools.stageb_gdino_adapter_probe_audit import file_record, load_checkpoint  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


class U2V2TrainingAuditError(RuntimeError):
    pass


def audit(checkpoint: Path, *, config: Path, expected_updates: int) -> dict[str, Any]:
    checkpoint = checkpoint.resolve(strict=True)
    payload = load_checkpoint(checkpoint)
    cfg = SLConfig.fromfile(str(config.resolve(strict=True)))
    cfg.device = "cpu"
    model, _criterion, _post = build_model_main(cfg)
    validate_runtime_payload(
        model, payload, checkpoint_label=f"U2-v2 milestone {checkpoint}"
    )
    state = payload.get("model")
    if int(payload.get("optimizer_updates", -1)) != int(expected_updates):
        raise U2V2TrainingAuditError("optimizer update count drifted")
    if len(state) != 1174:
        raise U2V2TrainingAuditError("trained U2-v2 must contain 1,174 tensors")
    residual_prefix = "stage_b_u2v2_rank_residual."
    residual_state = {k: v for k, v in state.items() if k.startswith(residual_prefix)}
    if len(residual_state) != 9:
        raise U2V2TrainingAuditError("residual state must be 7 params + 2 buffers")
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise U2V2TrainingAuditError("optimizer state is missing")
    groups = optimizer.get("param_groups")
    slots = optimizer.get("state")
    if not isinstance(groups, list) or len(groups) != 1:
        raise U2V2TrainingAuditError("U2-v2 requires one optimizer group")
    if groups[0].get("stage_b_u2v2_branch") != "post_gate_rank_residual":
        raise U2V2TrainingAuditError("optimizer branch ownership drifted")
    if len(groups[0].get("params", [])) != 7 or not isinstance(slots, Mapping) or len(slots) != 7:
        raise U2V2TrainingAuditError("optimizer must own exactly seven residual tensors")
    saved_args = payload.get("args")
    runtime = saved_args.get("stage_b_u2v2_runtime_audit") if isinstance(saved_args, Mapping) else None
    if not isinstance(runtime, Mapping) or runtime.get("schema") != "pivot.stageb.u2v2_runtime_audit/v1":
        raise U2V2TrainingAuditError("U2-v2 runtime audit is missing")
    if int(runtime.get("successful_optimizer_steps", -1)) != int(expected_updates):
        raise U2V2TrainingAuditError("successful-step audit drifted")
    if int(runtime.get("amp_skipped_optimizer_steps", -1)) != 0:
        raise U2V2TrainingAuditError("U2-v2 had an AMP skip")
    if int(runtime.get("nonfinite_gradient_boundaries", -1)) != 0:
        raise U2V2TrainingAuditError("U2-v2 had a nonfinite gradient")
    if int(runtime.get("zero_gradient_successful_steps", -1)) != 0:
        raise U2V2TrainingAuditError("U2-v2 had a zero-gradient optimizer step")
    if float(runtime.get("min_amp_scale", 0.0)) <= 0.0:
        raise U2V2TrainingAuditError("AMP scale audit is invalid")
    minimum_free = int(runtime.get("minimum_device_free_bytes", 0))
    if minimum_free < 1024 ** 3:
        raise U2V2TrainingAuditError("U2-v2 retained less than 1 GiB unreserved VRAM")
    args_contract = {
        "seed": int(saved_args.get("seed", -1)),
        "batch_size": int(saved_args.get("batch_size", -1)),
        "gradient_accumulation_steps": int(saved_args.get("gradient_accumulation_steps", -1)),
        "weight_decay": float(saved_args.get("weight_decay", -1)),
        "clip_max_norm": float(saved_args.get("clip_max_norm", -1)),
        "amp_init_scale": float(saved_args.get("amp_init_scale", -1)),
        "forward_microbatch": int(
            saved_args.get("stage_b_u2v2_forward_microbatch", -1)
        ),
    }
    expected = {
        "seed": 42, "batch_size": 38, "gradient_accumulation_steps": 2,
        "weight_decay": 1e-4, "clip_max_norm": 0.1, "amp_init_scale": 8192.0,
        "forward_microbatch": 19,
    }
    if args_contract != expected:
        raise U2V2TrainingAuditError(f"training runtime drifted: {args_contract}")
    return {
        "schema": "pivot.stageb.u2v2_training_checkpoint_audit/v1",
        "status": "verified",
        "checkpoint": file_record(checkpoint),
        "optimizer_updates": expected_updates,
        "state_tensors": len(state),
        "residual_state_tensors": len(residual_state),
        "optimizer_owned_tensors": len(slots),
        "runtime": dict(runtime),
        "training": args_contract,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-updates", type=int, required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = audit(
        Path(args.checkpoint), config=Path(args.config),
        expected_updates=args.expected_updates,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output).resolve()
        if output.exists():
            raise U2V2TrainingAuditError(f"refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, U2V2InitializerError, U2V2TrainingAuditError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
