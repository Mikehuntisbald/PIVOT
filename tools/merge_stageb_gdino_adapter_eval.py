#!/usr/bin/env python3
"""Merge isolated pure-GDINO rank/confidence adapters for evaluation only."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.GroundingDINO.stage_b_gdino_score_adapter import (  # noqa: E402
    StageBGDINOScoreAdapter,
    validate_stage_b_gdino_score_adapter_checkpoint,
)
from tools.make_stageb_gdino_adapter_p0 import (  # noqa: E402
    build_adapter_from_config,
)
from tools.stageb_dependency_audit import (  # noqa: E402
    DependencyAuditError,
    config_import_chain,
)
from tools.stageb_gdino_adapter_probe_audit import (  # noqa: E402
    ADAPTER_PREFIX,
    CONFIDENCE_PARTS,
    RANK_PARTS,
    ProbeAuditError,
    checkpoint_args,
    file_record,
    load_checkpoint,
    model_hash_record,
    resolve_path,
    tensor_state_sha256,
)
from util.slconfig import SLConfig  # noqa: E402


LINEAGE_SCHEMA = "stageb-gdino-adapter-merged-eval-lineage-v1"
CONTRACT_SCHEMA = "stageb-gdino-adapter-merged-eval-contract-v1"
DEFAULT_EVAL_CONFIG = (
    REPO_ROOT
    / "config/ablations/cfg_stageb_gdino_score_adapter_merged_eval.py"
)
HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

ARCHITECTURE_FIELDS = (
    "modelname",
    "hidden_dim",
    "num_queries",
    "stage_b_gdino_score_adapter",
    "stage_b_gdino_adapter_dim",
    "stage_b_gdino_gate_hidden_dim",
    "stage_b_gdino_gate_pool_temperature",
    "stage_b_gdino_gate_topk",
    "patch_only",
    "stage_b",
    "stage_b_v7",
    "stage_b_v11_fixed_text",
    "stage_b_legacy_global_gate",
    "enable_patch_branch",
)
EXPECTED_ARCHITECTURE = {
    "modelname": "groundingdino",
    "hidden_dim": 256,
    "num_queries": 900,
    "stage_b_gdino_score_adapter": True,
    "stage_b_gdino_adapter_dim": 128,
    "stage_b_gdino_gate_hidden_dim": 128,
    "stage_b_gdino_gate_pool_temperature": 0.01,
    "stage_b_gdino_gate_topk": 3,
    "patch_only": False,
    "stage_b": False,
    "stage_b_v7": False,
    "stage_b_v11_fixed_text": False,
    "stage_b_legacy_global_gate": False,
    "enable_patch_branch": False,
}
ROLE_CONFIG = {
    "rank": {
        "stage_b_gdino_adapter_train_mode": "rank_only",
        "stage_b_gdino_tn_scope": "",
        "stage_b_gdino_rank_weight": 1.0,
        "stage_b_gdino_confidence_weight": 0.0,
        "stage_b_gdino_queue_size": 0,
        "stage_b_gdino_queue_min_count": 0,
    },
    "confidence": {
        "stage_b_gdino_adapter_train_mode": "confidence_only",
        "stage_b_gdino_tn_scope": "image_global_topk_verified",
        "stage_b_gdino_rank_weight": 0.0,
        "stage_b_gdino_confidence_weight": 1.0,
        "stage_b_gdino_confidence_objective": "detached_recent_q05_trust",
        "stage_b_gdino_queue_size": 512,
        "stage_b_gdino_queue_min_count": 256,
    },
}
CRITERION_CODES = {
    "rank": {
        "criterion_train_mode_code": 1,
        "criterion_scope_code": 0,
        "criterion_confidence_objective_code": 0,
        "criterion_queue_size": 0,
        "criterion_queue_min_count": 0,
    },
    "confidence": {
        "criterion_train_mode_code": 2,
        "criterion_scope_code": 1,
        "criterion_confidence_objective_code": 2,
        "criterion_queue_size": 512,
        "criterion_queue_min_count": 256,
    },
}


class MergedEvalCheckpointError(RuntimeError):
    pass


def _matches(observed: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return isinstance(observed, (int, float)) and math.isclose(
            float(observed), expected, rel_tol=0.0, abs_tol=1e-12
        )
    return observed == expected


def _require_sha256(value: str, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if HEX_SHA256_RE.fullmatch(normalized) is None:
        raise MergedEvalCheckpointError(f"{label} must be a lowercase SHA256 digest")
    return normalized


def _bound_file(path: Path, expected_sha256: str, *, label: str) -> Dict[str, Any]:
    try:
        record = file_record(path)
    except ProbeAuditError as error:
        raise MergedEvalCheckpointError(str(error)) from error
    wanted = _require_sha256(expected_sha256, label=f"{label} expected hash")
    if record["sha256"] != wanted:
        raise MergedEvalCheckpointError(
            f"{label} hash mismatch: expected {wanted}, got {record['sha256']}"
        )
    return record


def _config_binding(path: Path) -> Dict[str, Any]:
    path = path.resolve()
    try:
        chain = config_import_chain(path, root=REPO_ROOT)
        records = [file_record(item) for item in chain]
    except (DependencyAuditError, ProbeAuditError) as error:
        raise MergedEvalCheckpointError(str(error)) from error
    if not chain or path not in chain:
        raise MergedEvalCheckpointError(
            f"config import chain does not contain its leaf: {path}"
        )
    leaf = next(record for record in records if record["path"] == str(path))
    return {"leaf": leaf, "import_chain": records}


def _verify_binding_unchanged(binding: Mapping[str, Any], *, label: str) -> None:
    rows = binding.get("import_chain")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise MergedEvalCheckpointError(f"{label} config binding is malformed")
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise MergedEvalCheckpointError(f"{label} config record is malformed")
        current = file_record(Path(row["path"]))
        if current != dict(row):
            raise MergedEvalCheckpointError(
                f"{label} config/import-chain file changed during merge: {row['path']}"
            )


def _load_config(path: Path, *, label: str):
    try:
        return SLConfig.fromfile(str(path))
    except Exception as error:
        raise MergedEvalCheckpointError(f"could not load {label} config {path}: {error}") from error


def _require_config_values(cfg: Any, expected: Mapping[str, Any], *, label: str) -> None:
    for key, wanted in expected.items():
        observed = getattr(cfg, key, None)
        if not _matches(observed, wanted):
            raise MergedEvalCheckpointError(
                f"{label} config {key}: expected {wanted!r}, got {observed!r}"
            )


def _architecture_record(cfg: Any, *, label: str) -> Dict[str, Any]:
    values = {key: getattr(cfg, key, None) for key in ARCHITECTURE_FIELDS}
    _require_config_values(cfg, EXPECTED_ARCHITECTURE, label=label)
    return values


def _resolve_recorded_config(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    try:
        return resolve_path(str(value))
    except Exception:
        return None


def _require_checkpoint_metadata(
    payload: Mapping[str, Any], checkpoint_record: Mapping[str, Any], *, label: str
) -> Dict[str, Any]:
    iteration = payload.get("iteration")
    epoch = payload.get("epoch")
    epoch_finished = payload.get("epoch_finished")
    reason = payload.get("checkpoint_reason")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration <= 0:
        raise MergedEvalCheckpointError(f"{label} iteration must be a positive integer")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise MergedEvalCheckpointError(f"{label} epoch must be a non-negative integer")
    if type(epoch_finished) is not bool:
        raise MergedEvalCheckpointError(f"{label} epoch_finished must be boolean")
    if not isinstance(reason, str) or not reason:
        raise MergedEvalCheckpointError(f"{label} checkpoint_reason must be non-empty")
    result = dict(checkpoint_record)
    result.update(
        {
            "iteration": iteration,
            "epoch": epoch,
            "epoch_finished": epoch_finished,
            "checkpoint_reason": reason,
        }
    )
    return result


def _scalar_int(state: Mapping[str, Any], key: str, *, label: str) -> int:
    value = state.get(key)
    if not torch.is_tensor(value) or value.numel() != 1:
        raise MergedEvalCheckpointError(f"{label} is missing scalar tensor {key}")
    return int(value.detach().cpu().reshape(-1)[0].item())


def _validate_criterion(payload: Mapping[str, Any], *, role: str) -> None:
    criterion = payload.get("criterion")
    if not isinstance(criterion, Mapping):
        raise MergedEvalCheckpointError(f"{role} source has no criterion state")
    for key, wanted in CRITERION_CODES[role].items():
        observed = _scalar_int(criterion, key, label=f"{role} criterion")
        if observed != wanted:
            raise MergedEvalCheckpointError(
                f"{role} criterion {key}: expected {wanted}, got {observed}"
            )
    queue_size = CRITERION_CODES[role]["criterion_queue_size"]
    for key in ("fpr_positive_queue", "fpr_negative_queue"):
        value = criterion.get(key)
        if (
            not torch.is_tensor(value)
            or value.dim() != 1
            or value.numel() != queue_size
            or not bool(torch.isfinite(value.detach()).all().item())
        ):
            raise MergedEvalCheckpointError(
                f"{role} criterion {key} must be finite length-{queue_size}"
            )
    count = _scalar_int(criterion, "fpr_queue_count", label=f"{role} criterion")
    pointer = _scalar_int(criterion, "fpr_queue_ptr", label=f"{role} criterion")
    minimum = CRITERION_CODES[role]["criterion_queue_min_count"]
    if not 0 <= count <= queue_size or count < minimum:
        raise MergedEvalCheckpointError(f"{role} criterion queue is invalid or cold")
    if queue_size == 0:
        if count != 0 or pointer != 0:
            raise MergedEvalCheckpointError("rank source must keep its queue disabled")
    elif not 0 <= pointer < queue_size:
        raise MergedEvalCheckpointError("confidence source queue pointer is invalid")


def _validate_optimizer(payload: Mapping[str, Any], *, role: str) -> None:
    optimizer = payload.get("optimizer")
    groups = optimizer.get("param_groups") if isinstance(optimizer, Mapping) else None
    if not isinstance(groups, list) or not groups:
        raise MergedEvalCheckpointError(f"{role} source has no optimizer param groups")
    branches = {group.get("stage_b_gdino_branch") for group in groups if isinstance(group, Mapping)}
    if branches != {role}:
        raise MergedEvalCheckpointError(
            f"{role} source optimizer branch mismatch: {sorted(map(str, branches))}"
        )


def _adapter_key_sets(adapter: StageBGDINOScoreAdapter) -> Dict[str, list[str]]:
    all_keys = sorted(ADAPTER_PREFIX + key for key in adapter.state_dict())
    rank = sorted(
        key
        for key in all_keys
        if key.removeprefix(ADAPTER_PREFIX).startswith(RANK_PARTS)
    )
    confidence = sorted(
        key
        for key in all_keys
        if key.removeprefix(ADAPTER_PREFIX).startswith(CONFIDENCE_PARTS)
    )
    if len(all_keys) != 20 or len(rank) != 8 or len(confidence) != 12:
        raise MergedEvalCheckpointError(
            "adapter whitelist drifted from the required 20=8 rank+12 confidence keys"
        )
    if set(all_keys) != set(rank).union(confidence):
        raise MergedEvalCheckpointError("adapter whitelist contains unknown branch keys")
    return {"all": all_keys, "rank": rank, "confidence": confidence}


def _require_model_state(payload: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise MergedEvalCheckpointError(f"{label} has no model state")
    invalid = [key for key, value in state.items() if not isinstance(key, str) or not torch.is_tensor(value)]
    if invalid:
        raise MergedEvalCheckpointError(f"{label} contains non-tensor model entries: {invalid[:8]}")
    return state


def _require_exact_tensor(
    observed: torch.Tensor, expected: torch.Tensor, *, key: str, label: str
) -> None:
    if tuple(observed.shape) != tuple(expected.shape):
        raise MergedEvalCheckpointError(
            f"{label} tensor {key} shape mismatch: {tuple(observed.shape)} != {tuple(expected.shape)}"
        )
    if observed.dtype != expected.dtype:
        raise MergedEvalCheckpointError(
            f"{label} tensor {key} dtype mismatch: {observed.dtype} != {expected.dtype}"
        )
    if not torch.equal(observed, expected):
        raise MergedEvalCheckpointError(f"{label} tensor {key} is not bitwise equal")


def _validate_source(
    *,
    role: str,
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    config: Path,
    adapter: StageBGDINOScoreAdapter,
    whitelist: Mapping[str, Sequence[str]],
) -> Dict[str, Any]:
    checkpoint_file = _bound_file(
        checkpoint, expected_checkpoint_sha256, label=f"{role} source checkpoint"
    )
    config_binding = _config_binding(config)
    cfg = _load_config(config, label=role)
    architecture = _architecture_record(cfg, label=role)
    _require_config_values(cfg, ROLE_CONFIG[role], label=role)
    payload = load_checkpoint(checkpoint)
    state = _require_model_state(payload, label=f"{role} source checkpoint")
    args = checkpoint_args(payload)
    if _resolve_recorded_config(args.get("config_file")) != config.resolve():
        raise MergedEvalCheckpointError(
            f"{role} source checkpoint config_file does not match {config}"
        )
    for key in tuple(ARCHITECTURE_FIELDS) + tuple(ROLE_CONFIG[role]):
        wanted = getattr(cfg, key, None)
        if not _matches(args.get(key), wanted):
            raise MergedEvalCheckpointError(
                f"{role} source arg {key}: expected {wanted!r}, got {args.get(key)!r}"
            )
    try:
        validate_stage_b_gdino_score_adapter_checkpoint(
            SimpleNamespace(stage_b_gdino_score_adapter=adapter),
            state,
            checkpoint_label=f"{role} source checkpoint",
        )
        hashes = model_hash_record(state)
    except (ValueError, TypeError, ProbeAuditError) as error:
        raise MergedEvalCheckpointError(str(error)) from error
    observed_adapter = sorted(key for key in state if key.startswith(ADAPTER_PREFIX))
    if observed_adapter != list(whitelist["all"]):
        raise MergedEvalCheckpointError(f"{role} source adapter key whitelist mismatch")
    expected_adapter = adapter.state_dict()
    for key in whitelist["all"]:
        suffix = key.removeprefix(ADAPTER_PREFIX)
        value = state[key]
        expected = expected_adapter[suffix]
        if tuple(value.shape) != tuple(expected.shape) or value.dtype != expected.dtype:
            raise MergedEvalCheckpointError(
                f"{role} source adapter tensor {key} shape/dtype mismatch"
            )
    if role == "rank":
        if hashes.get("rank_final_zero") is not False or hashes.get("confidence_final_zero") is not True:
            raise MergedEvalCheckpointError(
                "rank source must have trained rank output and untouched zero confidence output"
            )
    else:
        if hashes.get("confidence_final_zero") is not False or hashes.get("rank_final_zero") is not True:
            raise MergedEvalCheckpointError(
                "confidence source must have trained confidence output and untouched zero rank output"
            )
    _validate_criterion(payload, role=role)
    _validate_optimizer(payload, role=role)
    metadata = _require_checkpoint_metadata(
        payload, checkpoint_file, label=f"{role} source"
    )
    _verify_binding_unchanged(config_binding, label=role)
    if file_record(checkpoint) != checkpoint_file:
        raise MergedEvalCheckpointError(f"{role} source checkpoint changed during validation")
    return {
        "payload": payload,
        "state": state,
        "hashes": hashes,
        "architecture": architecture,
        "lineage": {
            "checkpoint": metadata,
            "config": config_binding,
            "model": {
                "base_tensor_sha256": hashes["base_model_sha256"],
                f"{role}_tensor_sha256": hashes[f"{role}_sha256"],
                "rank_final_zero": hashes["rank_final_zero"],
                "confidence_final_zero": hashes["confidence_final_zero"],
            },
        },
    }


def _validate_baseline(
    *, checkpoint: Path, expected_checkpoint_sha256: str
) -> Dict[str, Any]:
    checkpoint_file = _bound_file(
        checkpoint, expected_checkpoint_sha256, label="baseline checkpoint"
    )
    payload = load_checkpoint(checkpoint)
    state = _require_model_state(payload, label="baseline checkpoint")
    try:
        hashes = model_hash_record(state)
    except ProbeAuditError as error:
        raise MergedEvalCheckpointError(str(error)) from error
    if hashes.get("adapter_state_keys") != 0:
        raise MergedEvalCheckpointError("baseline checkpoint contains adapter tensors")
    forbidden_prefixes = (
        "stage_b_fixed_text_scorer.",
        "patch_encoder.",
        "query_proj_for_patch.",
    )
    forbidden = sorted(
        key
        for key in state
        if key.startswith(forbidden_prefixes) or key == "patch_logit_scale"
    )
    if forbidden:
        raise MergedEvalCheckpointError(
            f"baseline is not ordinary pure GDINO: {forbidden[:8]}"
        )
    if file_record(checkpoint) != checkpoint_file:
        raise MergedEvalCheckpointError("baseline checkpoint changed during validation")
    return {
        "payload": payload,
        "state": state,
        "hashes": hashes,
        "lineage": {
            "checkpoint": checkpoint_file,
            "base_tensor_sha256": hashes["base_model_sha256"],
        },
    }


def _adapter_from_state(
    config: Path, state: Mapping[str, Any]
) -> StageBGDINOScoreAdapter:
    adapter = build_adapter_from_config(config, seed=0)
    selected = {
        key.removeprefix(ADAPTER_PREFIX): value
        for key, value in state.items()
        if key.startswith(ADAPTER_PREFIX)
    }
    result = adapter.load_state_dict(selected, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise MergedEvalCheckpointError("adapter strict reload unexpectedly reported key drift")
    return adapter.eval()


def _functional_equivalence(
    *,
    merged_state: Mapping[str, Any],
    rank_state: Mapping[str, Any],
    confidence_state: Mapping[str, Any],
    eval_config: Path,
) -> Dict[str, bool]:
    rank_adapter = _adapter_from_state(eval_config, rank_state)
    confidence_adapter = _adapter_from_state(eval_config, confidence_state)
    merged_adapter = _adapter_from_state(eval_config, merged_state)
    generator = torch.Generator(device="cpu").manual_seed(20_260_716)
    query_hs = torch.randn(2, 900, 256, generator=generator, dtype=torch.float32)
    base_score = torch.rand(2, 900, generator=generator, dtype=torch.float32)
    candidate_mask = torch.ones_like(base_score, dtype=torch.bool)
    with torch.inference_mode():
        rank_output = rank_adapter(query_hs, base_score, candidate_mask)
        confidence_output = confidence_adapter(query_hs, base_score, candidate_mask)
        merged_output = merged_adapter(query_hs, base_score, candidate_mask)
    checks = {
        "rank_feature_equals_rank_source": torch.equal(
            merged_output["rank_feature"], rank_output["rank_feature"]
        ),
        "rank_residual_equals_rank_source": torch.equal(
            merged_output["rank_residual"], rank_output["rank_residual"]
        ),
        "rank_score_equals_rank_source": torch.equal(
            merged_output["rank_score"], rank_output["rank_score"]
        ),
        "confidence_feature_equals_confidence_source": torch.equal(
            merged_output["confidence_feature"], confidence_output["confidence_feature"]
        ),
        "confidence_gate_equals_confidence_source": torch.equal(
            merged_output["confidence_gate"], confidence_output["confidence_gate"]
        ),
        "confidence_score_equals_confidence_source": torch.equal(
            merged_output["confidence_score"], confidence_output["confidence_score"]
        ),
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise MergedEvalCheckpointError(
            f"merged adapter failed bitwise functional equivalence: {failed}"
        )
    return checks


def _prepare_merge(
    *,
    rank_checkpoint: Path,
    rank_checkpoint_sha256: str,
    rank_config: Path,
    confidence_checkpoint: Path,
    confidence_checkpoint_sha256: str,
    confidence_config: Path,
    baseline_checkpoint: Path,
    baseline_checkpoint_sha256: str,
    eval_config: Path,
) -> Dict[str, Any]:
    eval_config = eval_config.resolve()
    if eval_config != DEFAULT_EVAL_CONFIG.resolve():
        raise MergedEvalCheckpointError(
            f"merged checkpoint requires the dedicated eval config {DEFAULT_EVAL_CONFIG}"
        )
    eval_binding = _config_binding(eval_config)
    eval_cfg = _load_config(eval_config, label="merged eval")
    eval_architecture = _architecture_record(eval_cfg, label="merged eval")
    _require_config_values(
        eval_cfg,
        {
            "stage_b_gdino_adapter_merged_eval_only": True,
            "stage_b_gdino_adapter_merged_eval_contract_version": 1,
            "stage_b_gdino_rank_weight": 0.0,
            "stage_b_gdino_confidence_weight": 0.0,
        },
        label="merged eval",
    )
    adapter = build_adapter_from_config(eval_config, seed=0)
    whitelist = _adapter_key_sets(adapter)
    rank = _validate_source(
        role="rank",
        checkpoint=rank_checkpoint,
        expected_checkpoint_sha256=rank_checkpoint_sha256,
        config=rank_config,
        adapter=adapter,
        whitelist=whitelist,
    )
    confidence = _validate_source(
        role="confidence",
        checkpoint=confidence_checkpoint,
        expected_checkpoint_sha256=confidence_checkpoint_sha256,
        config=confidence_config,
        adapter=adapter,
        whitelist=whitelist,
    )
    baseline = _validate_baseline(
        checkpoint=baseline_checkpoint,
        expected_checkpoint_sha256=baseline_checkpoint_sha256,
    )
    if rank["architecture"] != confidence["architecture"] or rank["architecture"] != eval_architecture:
        raise MergedEvalCheckpointError("rank/confidence/eval architecture configs differ")
    rank_state = rank["state"]
    confidence_state = confidence["state"]
    baseline_state = baseline["state"]
    expected_model_keys = set(baseline_state).union(whitelist["all"])
    for label, state in (("rank", rank_state), ("confidence", confidence_state)):
        if set(state) != expected_model_keys:
            raise MergedEvalCheckpointError(
                f"{label} source model keys do not exactly equal baseline+adapter whitelist"
            )
        for key in baseline_state:
            _require_exact_tensor(
                state[key], baseline_state[key], key=key, label=f"{label} base"
            )
    if rank["hashes"]["base_model_sha256"] != baseline["hashes"]["base_model_sha256"]:
        raise MergedEvalCheckpointError("rank source base tensor hash differs from baseline")
    if confidence["hashes"]["base_model_sha256"] != baseline["hashes"]["base_model_sha256"]:
        raise MergedEvalCheckpointError("confidence source base tensor hash differs from baseline")

    merged_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in rank_state.items():
        merged_state[key] = confidence_state[key] if key in whitelist["confidence"] else value
    try:
        validate_stage_b_gdino_score_adapter_checkpoint(
            SimpleNamespace(stage_b_gdino_score_adapter=adapter),
            merged_state,
            checkpoint_label="merged eval checkpoint",
        )
        merged_hashes = model_hash_record(merged_state)
        full_model_hash = tensor_state_sha256(merged_state, merged_state.keys())
    except (ValueError, TypeError, ProbeAuditError) as error:
        raise MergedEvalCheckpointError(str(error)) from error
    if merged_hashes["rank_sha256"] != rank["hashes"]["rank_sha256"]:
        raise MergedEvalCheckpointError("merged rank tensors do not exactly match rank source")
    if merged_hashes["confidence_sha256"] != confidence["hashes"]["confidence_sha256"]:
        raise MergedEvalCheckpointError(
            "merged confidence tensors do not exactly match confidence source"
        )
    functional = _functional_equivalence(
        merged_state=merged_state,
        rank_state=rank_state,
        confidence_state=confidence_state,
        eval_config=eval_config,
    )
    _verify_binding_unchanged(eval_binding, label="merged eval")
    for source, label in ((rank, "rank"), (confidence, "confidence"), (baseline, "baseline")):
        if file_record(Path(source["lineage"]["checkpoint"]["path"]))["sha256"] != source["lineage"]["checkpoint"]["sha256"]:
            raise MergedEvalCheckpointError(f"{label} checkpoint changed during merge")

    lineage = {
        "schema": LINEAGE_SCHEMA,
        "rank_source": rank["lineage"],
        "confidence_source": confidence["lineage"],
        "baseline": baseline["lineage"],
        "eval_config": eval_binding,
    }
    contract = {
        "schema": CONTRACT_SCHEMA,
        "eval_only": True,
        "resumable": False,
        "architecture": eval_architecture,
        "adapter_key_whitelist": list(whitelist["all"]),
        "rank_key_whitelist": list(whitelist["rank"]),
        "confidence_key_whitelist": list(whitelist["confidence"]),
        "branch_selection": {
            "base": "declared_baseline_bitwise_equal_in_both_sources",
            "rank": "rank_source_only",
            "confidence": "confidence_source_only",
        },
        "model_state_keys": len(merged_state),
        "base_state_keys": merged_hashes["base_state_keys"],
        "adapter_state_keys": merged_hashes["adapter_state_keys"],
        "base_tensor_sha256": merged_hashes["base_model_sha256"],
        "rank_tensor_sha256": merged_hashes["rank_sha256"],
        "confidence_tensor_sha256": merged_hashes["confidence_sha256"],
        "adapter_tensor_sha256": merged_hashes["adapter_sha256"],
        "full_model_tensor_sha256": full_model_hash,
        "rank_final_zero": merged_hashes["rank_final_zero"],
        "confidence_final_zero": merged_hashes["confidence_final_zero"],
        "functional_bitwise": functional,
        "synthetic_input_contract": {
            "seed": 20_260_716,
            "shape": [2, 900, 256],
            "candidate_mask": "all_900_queries_valid",
            "device": "cpu",
            "dtype": "torch.float32",
        },
    }
    return {"model": merged_state, "lineage": lineage, "contract": contract}


def verify_merged_eval_checkpoint(checkpoint: Path) -> Dict[str, Any]:
    checkpoint = checkpoint.resolve()
    payload = load_checkpoint(checkpoint)
    if set(payload) != {"model", "lineage", "contract"}:
        raise MergedEvalCheckpointError(
            "merged eval checkpoint top-level keys must be exactly model/lineage/contract"
        )
    lineage = payload.get("lineage")
    contract = payload.get("contract")
    if not isinstance(lineage, Mapping) or lineage.get("schema") != LINEAGE_SCHEMA:
        raise MergedEvalCheckpointError("merged eval checkpoint lineage schema mismatch")
    if not isinstance(contract, Mapping) or contract.get("schema") != CONTRACT_SCHEMA:
        raise MergedEvalCheckpointError("merged eval checkpoint contract schema mismatch")
    if contract.get("eval_only") is not True or contract.get("resumable") is not False:
        raise MergedEvalCheckpointError("merged checkpoint lost its eval-only/non-resumable flags")

    def source_args(name: str) -> tuple[Path, str, Path]:
        source = lineage.get(name)
        if not isinstance(source, Mapping):
            raise MergedEvalCheckpointError(f"lineage is missing {name}")
        checkpoint_record = source.get("checkpoint")
        config_record = source.get("config")
        leaf = config_record.get("leaf") if isinstance(config_record, Mapping) else None
        if not isinstance(checkpoint_record, Mapping) or not isinstance(leaf, Mapping):
            raise MergedEvalCheckpointError(f"lineage {name} records are malformed")
        return (
            Path(str(checkpoint_record.get("path"))).resolve(),
            str(checkpoint_record.get("sha256")),
            Path(str(leaf.get("path"))).resolve(),
        )

    rank_checkpoint, rank_sha, rank_config = source_args("rank_source")
    confidence_checkpoint, confidence_sha, confidence_config = source_args(
        "confidence_source"
    )
    baseline = lineage.get("baseline")
    baseline_record = baseline.get("checkpoint") if isinstance(baseline, Mapping) else None
    eval_binding = lineage.get("eval_config")
    eval_leaf = eval_binding.get("leaf") if isinstance(eval_binding, Mapping) else None
    if not isinstance(baseline_record, Mapping) or not isinstance(eval_leaf, Mapping):
        raise MergedEvalCheckpointError("baseline/eval-config lineage is malformed")
    expected = _prepare_merge(
        rank_checkpoint=rank_checkpoint,
        rank_checkpoint_sha256=rank_sha,
        rank_config=rank_config,
        confidence_checkpoint=confidence_checkpoint,
        confidence_checkpoint_sha256=confidence_sha,
        confidence_config=confidence_config,
        baseline_checkpoint=Path(str(baseline_record.get("path"))).resolve(),
        baseline_checkpoint_sha256=str(baseline_record.get("sha256")),
        eval_config=Path(str(eval_leaf.get("path"))).resolve(),
    )
    if dict(lineage) != expected["lineage"]:
        raise MergedEvalCheckpointError("merged checkpoint lineage no longer matches its sources")
    if dict(contract) != expected["contract"]:
        raise MergedEvalCheckpointError("merged checkpoint contract no longer matches its sources")
    state = _require_model_state(payload, label="merged eval checkpoint")
    if list(state) != list(expected["model"]):
        raise MergedEvalCheckpointError("serialized merged model key order/set drifted")
    for key, value in expected["model"].items():
        _require_exact_tensor(state[key], value, key=key, label="serialized merged model")
    adapter = build_adapter_from_config(DEFAULT_EVAL_CONFIG, seed=0)
    try:
        validate_stage_b_gdino_score_adapter_checkpoint(
            SimpleNamespace(stage_b_gdino_score_adapter=adapter),
            state,
            checkpoint_label="serialized merged eval checkpoint",
        )
    except (ValueError, TypeError) as error:
        raise MergedEvalCheckpointError(str(error)) from error
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "verified",
        "checkpoint": file_record(checkpoint),
        "contract": dict(contract),
    }


def create_merged_eval_checkpoint(
    *,
    output: Path,
    rank_checkpoint: Path,
    rank_checkpoint_sha256: str,
    rank_config: Path,
    confidence_checkpoint: Path,
    confidence_checkpoint_sha256: str,
    confidence_config: Path,
    baseline_checkpoint: Path,
    baseline_checkpoint_sha256: str,
    eval_config: Path = DEFAULT_EVAL_CONFIG,
) -> Dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise MergedEvalCheckpointError(f"refusing to overwrite merged checkpoint: {output}")
    payload = _prepare_merge(
        rank_checkpoint=rank_checkpoint.resolve(),
        rank_checkpoint_sha256=rank_checkpoint_sha256,
        rank_config=rank_config.resolve(),
        confidence_checkpoint=confidence_checkpoint.resolve(),
        confidence_checkpoint_sha256=confidence_checkpoint_sha256,
        confidence_config=confidence_config.resolve(),
        baseline_checkpoint=baseline_checkpoint.resolve(),
        baseline_checkpoint_sha256=baseline_checkpoint_sha256,
        eval_config=eval_config.resolve(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + ".tmp")
    if temporary.exists():
        raise MergedEvalCheckpointError(f"refusing to replace stale temporary file: {temporary}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    del payload
    gc.collect()
    try:
        return verify_merged_eval_checkpoint(output)
    except Exception:
        output.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--output", required=True)
    create.add_argument("--rank-checkpoint", required=True)
    create.add_argument("--rank-checkpoint-sha256", required=True)
    create.add_argument("--rank-config", required=True)
    create.add_argument("--confidence-checkpoint", required=True)
    create.add_argument("--confidence-checkpoint-sha256", required=True)
    create.add_argument("--confidence-config", required=True)
    create.add_argument("--baseline-checkpoint", required=True)
    create.add_argument("--baseline-checkpoint-sha256", required=True)
    create.add_argument("--eval-config", default=str(DEFAULT_EVAL_CONFIG))
    verify = subparsers.add_parser("verify")
    verify.add_argument("--checkpoint", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "create":
            receipt = create_merged_eval_checkpoint(
                output=resolve_path(args.output),
                rank_checkpoint=resolve_path(args.rank_checkpoint),
                rank_checkpoint_sha256=args.rank_checkpoint_sha256,
                rank_config=resolve_path(args.rank_config),
                confidence_checkpoint=resolve_path(args.confidence_checkpoint),
                confidence_checkpoint_sha256=args.confidence_checkpoint_sha256,
                confidence_config=resolve_path(args.confidence_config),
                baseline_checkpoint=resolve_path(args.baseline_checkpoint),
                baseline_checkpoint_sha256=args.baseline_checkpoint_sha256,
                eval_config=resolve_path(args.eval_config),
            )
        else:
            receipt = verify_merged_eval_checkpoint(resolve_path(args.checkpoint))
        print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True))
    except (MergedEvalCheckpointError, ProbeAuditError) as error:
        raise SystemExit(f"[FAIL] {error}") from error


if __name__ == "__main__":
    main()
