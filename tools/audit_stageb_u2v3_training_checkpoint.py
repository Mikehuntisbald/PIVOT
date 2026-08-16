"""Audit U2-v3 category-admission checkpoints and emit a compact receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from groundingdino.util.slconfig import SLConfig
from models.GroundingDINO import build_groundingdino
from tools.stageb_u2v3_category_admission_contract import (
    TRAINABLE_KEYS,
    validate_runtime_payload,
)


def _file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def audit(
    *, checkpoint: Path, config: Path, initializer: Path,
    initializer_sha256: str,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve(strict=True)
    config = config.resolve(strict=True)
    initializer = initializer.resolve(strict=True)
    cfg = SLConfig.fromfile(str(config))
    cfg.device = "cpu"
    model, criterion, _postprocessors = build_groundingdino(cfg)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    contract = validate_runtime_payload(
        model,
        payload,
        checkpoint_label=str(checkpoint),
        initializer_path=initializer,
        initializer_sha256=initializer_sha256,
    )
    optimizer = payload.get("optimizer")
    groups = optimizer.get("param_groups") if isinstance(optimizer, Mapping) else None
    state = optimizer.get("state") if isinstance(optimizer, Mapping) else None
    if not isinstance(groups, list) or len(groups) != 1:
        raise RuntimeError("U2-v3 checkpoint requires one optimizer group")
    if groups[0].get("stage_b_u2v3_branch") != "category_admission_projection":
        raise RuntimeError("U2-v3 optimizer ownership tag drifted")
    if len(groups[0].get("params", [])) != len(TRAINABLE_KEYS):
        raise RuntimeError("U2-v3 optimizer must own exactly eight tensors")
    if not isinstance(state, Mapping) or len(state) != len(TRAINABLE_KEYS):
        raise RuntimeError("U2-v3 optimizer state must cover exactly eight tensors")
    criterion_state = payload.get("criterion")
    if not isinstance(criterion_state, Mapping):
        raise RuntimeError("U2-v3 checkpoint lacks criterion state")
    expected_criterion = criterion.state_dict()
    if set(criterion_state) != set(expected_criterion):
        raise RuntimeError("U2-v3 criterion-state keys drifted")
    saved_args = payload.get("args")
    runtime = saved_args.get("stage_b_u2v3_runtime_audit") if isinstance(saved_args, Mapping) else None
    if not isinstance(runtime, Mapping):
        raise RuntimeError("U2-v3 checkpoint lacks runtime audit")
    updates = int(payload.get("optimizer_updates", -1))
    if updates <= 0 or int(runtime.get("successful_optimizer_steps", -1)) != updates:
        raise RuntimeError("U2-v3 optimizer update count drifted")
    if int(runtime.get("amp_skipped_optimizer_steps", -1)) != 0:
        raise RuntimeError("U2-v3 checkpoint contains AMP skips")
    if int(runtime.get("zero_gradient_successful_steps", -1)) != 0:
        raise RuntimeError("U2-v3 checkpoint contains zero-gradient updates")
    return {
        "schema": "pivot.stageb.u2v3_category_admission_checkpoint_audit/v1",
        "status": "verified",
        "checkpoint": _file_record(checkpoint),
        "config": _file_record(config),
        "initializer": _file_record(initializer),
        "optimizer_updates": updates,
        "trainable_tensor_count": len(TRAINABLE_KEYS),
        "trainable_parameter_count": sum(
            int(payload["model"][key].numel()) for key in TRAINABLE_KEYS
        ),
        "runtime_audit": dict(runtime),
        "ownership_contract": contract,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--initializer-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = audit(
        checkpoint=args.checkpoint,
        config=args.config,
        initializer=args.initializer,
        initializer_sha256=args.initializer_sha256,
    )
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
