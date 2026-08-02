#!/usr/bin/env python3
"""Audit that U0 training changes only its declared patch-rank surface."""

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

from models.GroundingDINO.stage_b_u0_patch_rank import (  # noqa: E402
    U0_PATCH_SOURCE_KEYS,
    stage_b_u0_tensor_state_sha256,
    validate_stage_b_u0_initializer_payload,
)
from tools.stageb_gdino_adapter_probe_audit import (  # noqa: E402
    file_record,
    load_checkpoint,
)


SCHEMA = "pivot.stageb.u0_transition_audit/v1"
U0_PREFIX = "stage_b_u0_patch_rank_adapter."
FROZEN_PATCH_KEYS = frozenset({"patch_logit_scale"})


class U0TransitionAuditError(RuntimeError):
    pass


def _model_state(payload: Mapping[str, Any], *, label: str) -> Mapping[str, torch.Tensor]:
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise U0TransitionAuditError(f"{label} has no model state")
    if any(not isinstance(key, str) or not torch.is_tensor(value) for key, value in state.items()):
        raise U0TransitionAuditError(f"{label} model state is not tensor-only")
    return state


def _tensor_changed(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    key: str,
) -> bool:
    return stage_b_u0_tensor_state_sha256(before, [key]) != (
        stage_b_u0_tensor_state_sha256(after, [key])
    )


def audit_u0_transition(
    initializer_payload: Mapping[str, Any],
    trained_payload: Mapping[str, Any],
) -> dict[str, Any]:
    initial = _model_state(initializer_payload, label="U0 initializer")
    validate_stage_b_u0_initializer_payload(
        initial,
        initializer_payload,
        checkpoint_label="U0 transition initializer",
    )
    trained = _model_state(trained_payload, label="U0 trained checkpoint")
    if set(trained) != set(initial):
        raise U0TransitionAuditError(
            "trained U0 full-model key coverage drifted "
            f"(missing={sorted(set(initial) - set(trained))[:8]}, "
            f"unexpected={sorted(set(trained) - set(initial))[:8]})"
        )
    changed = sorted(
        key for key in initial if _tensor_changed(initial, trained, key)
    )
    contract = initializer_payload["u0_initializer"]
    roles = contract["role_keys"]
    u0_trainable = {
        key
        for key in roles["u0_zero"]
        if not key.removeprefix(U0_PREFIX).startswith("_contract_")
    }
    patch_trainable = set(U0_PATCH_SOURCE_KEYS).difference(FROZEN_PATCH_KEYS)
    allowed = u0_trainable | patch_trainable
    forbidden = sorted(set(changed).difference(allowed))
    if forbidden:
        raise U0TransitionAuditError(
            f"U0 training changed frozen tensors: {forbidden[:20]}"
        )
    changed_output = sorted(
        key
        for key in changed
        if key in {
            U0_PREFIX + "output.weight",
            U0_PREFIX + "output.bias",
        }
    )
    if not changed_output:
        raise U0TransitionAuditError("U0 optimizer step did not change its output layer")
    merged_hash = stage_b_u0_tensor_state_sha256(trained, roles["merged"])
    if merged_hash != contract["merged_teacher_tensor_sha256"]:
        raise U0TransitionAuditError("sealed merged R100/P50 teacher changed")
    alias_hash = stage_b_u0_tensor_state_sha256(
        trained, roles["shared_backbone_alias"]
    )
    if alias_hash != contract["shared_backbone_alias_tensor_sha256"]:
        raise U0TransitionAuditError("shared b58 patch-backbone alias changed")
    for key in FROZEN_PATCH_KEYS:
        if _tensor_changed(initial, trained, key):
            raise U0TransitionAuditError(f"frozen normalized-scale tensor changed: {key}")
    for key, value in trained.items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise U0TransitionAuditError(f"trained U0 tensor is non-finite: {key}")
    optimizer_updates = trained_payload.get("optimizer_updates")
    if isinstance(optimizer_updates, bool) or not isinstance(optimizer_updates, int):
        raise U0TransitionAuditError("trained U0 checkpoint lacks optimizer_updates")
    if optimizer_updates <= 0:
        raise U0TransitionAuditError("trained U0 checkpoint has no successful update")
    return {
        "schema": SCHEMA,
        "status": "verified",
        "optimizer_updates": optimizer_updates,
        "iteration": trained_payload.get("iteration"),
        "checkpoint_reason": trained_payload.get("checkpoint_reason"),
        "state_keys": len(trained),
        "changed_key_count": len(changed),
        "changed_keys": changed,
        "changed_output_keys": changed_output,
        "frozen_key_count": len(trained) - len(changed),
        "merged_teacher_tensor_sha256": merged_hash,
        "shared_backbone_alias_tensor_sha256": alias_hash,
        "u0_trainable_tensor_sha256": stage_b_u0_tensor_state_sha256(
            trained, sorted(allowed)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initializer", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    initializer = Path(args.initializer).resolve(strict=True)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    result = audit_u0_transition(
        load_checkpoint(initializer),
        load_checkpoint(checkpoint),
    )
    result["initializer"] = file_record(initializer)
    result["checkpoint"] = file_record(checkpoint)
    encoded = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.output:
        output = Path(args.output).resolve()
        if output.exists():
            raise U0TransitionAuditError(f"refusing to overwrite audit: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(str(output) + ".tmp")
        if temporary.exists():
            raise U0TransitionAuditError(f"stale audit temporary exists: {temporary}")
        temporary.write_text(encoded, encoding="ascii")
        temporary.replace(output)
    print(encoded, end="")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, U0TransitionAuditError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
