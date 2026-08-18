#!/usr/bin/env python3
"""Overlay sealed clean D3 confidence12 onto an ARROW Admission checkpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.GroundingDINO.stage_b_u0_patch_rank import stage_b_u0_tensor_state_sha256
from tools.stageb_arrow_admission_contract import SCHEMA as ADMISSION_SCHEMA


SCHEMA = "arrow.stageb.confidence_overlay/v1"


class ArrowMergeError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha(path)}


def merge(admission_path: Path, confidence_path: Path) -> dict[str, Any]:
    admission_record, confidence_record = _record(admission_path), _record(confidence_path)
    admission = torch.load(admission_record["path"], map_location="cpu", weights_only=False)
    confidence = torch.load(confidence_record["path"], map_location="cpu", weights_only=False)
    contract = admission.get("arrow_admission_input")
    source_contract = confidence.get("u2v5_clean_confidence")
    if not isinstance(contract, dict) or contract.get("schema") != ADMISSION_SCHEMA:
        raise ArrowMergeError("admission checkpoint lacks ARROW contract")
    if not isinstance(source_contract, dict) or source_contract.get("schema") != "pivot.stageb.u2v5_clean_confidence_handoff/v1":
        raise ArrowMergeError("confidence source is not sealed clean D3")
    if source_contract.get("c100_confidence_imported") is not False:
        raise ArrowMergeError("C100 confidence is forbidden")
    keys = list(source_contract.get("identity_confidence_keys", ()))
    if len(keys) != 12 or len(set(keys)) != 12:
        raise ArrowMergeError("clean confidence ownership is not confidence12")
    state, source_state = admission["model"], confidence["model"]
    if set(keys) - set(state) or set(keys) - set(source_state):
        raise ArrowMergeError("confidence12 keys are missing")
    result = copy.deepcopy(admission)
    before = {key: state[key].detach().clone() for key in state}
    for key in keys:
        result["model"][key] = source_state[key].detach().clone()
    changed = sorted(
        key for key in state
        if not torch.equal(before[key], result["model"][key])
    )
    if changed != sorted(keys):
        raise ArrowMergeError("confidence overlay changed tensors outside confidence12")
    arrow = dict(result["arrow_admission_input"])
    frozen = [key for key in arrow["frozen_keys"] if key not in set(keys)]
    arrow["frozen_keys"] = frozen
    arrow["frozen_tensor_sha256"] = stage_b_u0_tensor_state_sha256(
        result["model"], frozen
    )
    result["arrow_admission_input"] = arrow
    result["arrow_confidence_overlay"] = {
        "schema": SCHEMA,
        "admission_source": admission_record,
        "confidence_source": confidence_record,
        "confidence_keys": keys,
        "confidence_tensor_sha256": stage_b_u0_tensor_state_sha256(
            result["model"], keys
        ),
        "changed_keys": changed,
        "scope": source_contract.get("scope"),
        "table_b_id": source_contract.get("table_b_id"),
        "c100_confidence_imported": False,
        "eval_only": True,
    }
    result.pop("optimizer", None)
    result.pop("lr_scheduler", None)
    result.pop("scaler", None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-checkpoint", required=True)
    parser.add_argument("--confidence-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise ArrowMergeError(f"refusing to overwrite {output}")
    payload = merge(Path(args.admission_checkpoint), Path(args.confidence_checkpoint))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    receipt = output.with_suffix(".receipt.json")
    receipt.write_text(json.dumps({
        "schema": SCHEMA, "checkpoint": _record(output),
        "contract": payload["arrow_confidence_overlay"],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": _record(output), "receipt": _record(receipt)}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, ArrowMergeError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
