#!/usr/bin/env python3
"""Verify sealed A/B/C ownership and confidence-only runtime parity."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import util.misc as utils
from tools.arrow_grefcoco_common import SEEDS, file_record, load_json, write_json_atomic
from tools.eval_text_groundingdino_refcoco_tn import _load_model_with_checkpoint_contract
from util.slconfig import SLConfig

DEFAULT_FINECOPS_PREREG = REPO_ROOT / "outputs/arrow_finecops_20260819/preregistration.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/arrow_grefcoco_20260820/preflight.json"
CONFIGS = {
    "A": REPO_ROOT / "config/ablations/cfg_arrow_finecops_a.py",
    "B": REPO_ROOT / "config/ablations/cfg_arrow_finecops_b.py",
    "C": REPO_ROOT / "config/ablations/cfg_arrow_finecops_c.py",
}
TRUNK_PREFIXES = ("transformer.", "bert.", "backbone.", "bbox_embed.", "input_proj.", "feat_map.")
RANK_PREFIX = "stage_b_gdino_score_adapter.rank_"
CONFIDENCE_PREFIX = "stage_b_gdino_score_adapter.confidence_"


def _tensor_group_sha(state: Mapping[str, torch.Tensor], predicate) -> tuple[str, int]:
    digest = hashlib.sha256()
    keys = sorted(key for key in state if predicate(key))
    for key in keys:
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest(), len(keys)


def _state(path: Path) -> Mapping[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise ValueError(f"{path} has no model state")
    return state


def _output_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def run(finecops_prereg: Path, output: Path, device_name: str) -> dict[str, Any]:
    prior = load_json(finecops_prereg)
    checkpoints = prior["checkpoints"]
    ownership: dict[str, Any] = {}
    runtime: dict[str, Any] = {}
    device = torch.device(device_name)
    synthetic = utils.nested_tensor_from_tensor_list([torch.zeros((3, 256, 256), dtype=torch.float32)]).to(device)
    for seed in SEEDS:
        ownership[str(seed)] = {}
        runtime[str(seed)] = {}
        expected_groups: dict[str, tuple[str, int]] | None = None
        expected_outputs: dict[str, str] | None = None
        for route in ("A", "B", "C"):
            checkpoint = Path(checkpoints[route][str(seed)]["path"]).resolve(strict=True)
            observed = file_record(checkpoint)
            if any(observed[key] != checkpoints[route][str(seed)][key] for key in ("sha256", "size_bytes")):
                raise ValueError(f"{route}/seed{seed} checkpoint drifted")
            state = _state(checkpoint)
            groups = {
                "trunk938": _tensor_group_sha(state, lambda key: key.startswith(TRUNK_PREFIXES)),
                "rank8": _tensor_group_sha(state, lambda key: key.startswith(RANK_PREFIX)),
                "confidence12": _tensor_group_sha(state, lambda key: key.startswith(CONFIDENCE_PREFIX)),
            }
            if {key: count for key, (_, count) in groups.items()} != {"trunk938": 938, "rank8": 8, "confidence12": 12}:
                raise ValueError(f"{route}/seed{seed} ownership counts drifted: {groups}")
            if expected_groups is None:
                expected_groups = groups
            elif groups != expected_groups:
                raise ValueError(f"{route}/seed{seed} trunk/rank/confidence is not bitwise equal to A")
            ownership[str(seed)][route] = {key: {"sha256": sha, "tensors": count} for key, (sha, count) in groups.items()}
            del state
            cfg = SLConfig.fromfile(str(CONFIGS[route]))
            cfg.device = str(device)
            model, _ = _load_model_with_checkpoint_contract(cfg, checkpoint, device)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(synthetic, captions=["a synthetic object ."], stage_b_u2v2_confidence_only=True)
            if any(key.startswith("stage_b_u0_") or key.startswith("stage_b_arrow_admission_") for key in outputs):
                raise RuntimeError("confidence-only synthetic forward did not bypass admission")
            keys = ("pred_boxes", "stage_b_gdino_base_score", "stage_b_gdino_rank_score", "stage_b_gdino_confidence_score")
            hashes = {key: _output_hash(outputs[key]) for key in keys}
            if expected_outputs is None:
                expected_outputs = hashes
            elif hashes != expected_outputs:
                raise ValueError(f"{route}/seed{seed} synthetic confidence-only parity failed")
            runtime[str(seed)][route] = hashes
            del model, outputs
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    payload = {"schema": "arrow.grefcoco.preflight/v1", "status": "passed_before_grefcoco_forward", "finecops_preregistration": file_record(finecops_prereg), "ownership": ownership, "synthetic_runtime_parity": runtime, "synthetic_input": {"shape": [1, 3, 256, 256], "caption": "a synthetic object .", "contains_grefcoco_data": False}}
    write_json_atomic(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finecops-preregistration", type=Path, default=DEFAULT_FINECOPS_PREREG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(args.finecops_preregistration.resolve(strict=True), args.output.resolve(), args.device)
    print(json.dumps({"schema": result["schema"], "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
