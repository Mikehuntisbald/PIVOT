#!/usr/bin/env python3
"""Create and verify a fail-closed two-model Stage-B composite artifact.

The artifact is a deployment/evaluation handoff contract only.  It does not
load both models for inference and deliberately does not modify either Stage-B
evaluator.  Rank is supplied by a fixed-StageA patch/full-text checkpoint;
absolute confidence is supplied by an independently trained pure-GDINO score
adapter checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, MutableMapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.GroundingDINO.stage_b_gdino_score_adapter import (  # noqa: E402
    StageBGDINOScoreAdapterCriterion,
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
    ProbeAuditError,
    checkpoint_args,
    file_record,
    load_checkpoint,
    model_hash_record,
    read_json,
    tensor_state_sha256,
    write_json,
)
from util.path_compat import remap_legacy_path  # noqa: E402
from util.stageb_exact_topk_contract import canonical_sha256  # noqa: E402


SCHEMA = "stageb-two-model-composite-artifact-v1"
IDENTITY_ALGORITHM = "sha256-canonical-json-excluding-artifact-identity-v1"
RANK_SCORER_PREFIX = "stage_b_fixed_text_scorer."
CONFIDENCE_CONFIG_LEAF = "cfg_stageb_gdino_score_adapter_semantic_verified.py"
RANK_RECIPE_RE = re.compile(r"^cfg_stageb_v(15|16|17|18|19|20)(?:_|\.py)")

RANK_ROLE_CONTRACT_KEYS = {
    "role",
    "model_family",
    "recipe_version",
    "score_key",
    "candidate_topk",
    "patch_rank_fusion",
    "patch_rank_weight",
    "exclude_canonical_from_score",
    "confidence_output_mode_in_rank_checkpoint",
    "scorer_contract_version",
    "scorer_contract_output_mode_code",
    "rank_confidence_policy",
}
CONFIDENCE_ROLE_CONTRACT_KEYS = {
    "role",
    "model_family",
    "adapter_train_mode",
    "tn_scope",
    "score_key",
    "base_expression_reduction",
    "confidence_query_reduction",
    "query_count",
    "candidate_mask_contract",
    "gate_application",
    "gate_pool_temperature",
    "gate_score_topk",
    "confidence_objective",
    "queue_size",
    "queue_min_count",
    "rank_confidence_policy",
}


class CompositeArtifactError(RuntimeError):
    """Raised when any composite artifact invariant is not provable."""


def _resolve_path(value: str | Path, *, repo_root: Path) -> Path:
    path = remap_legacy_path(value, repo_root=repo_root)
    if not path.is_absolute():
        path = repo_root / path
    return path.expanduser().resolve()


def _stat_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _stable_file_record(path: Path, *, label: str) -> Dict[str, Any]:
    try:
        before = _stat_identity(path)
        record = file_record(path)
        after = _stat_identity(path)
    except OSError as error:
        raise CompositeArtifactError(f"could not bind {label} {path}: {error}") from error
    if before != after:
        raise CompositeArtifactError(f"{label} changed while it was being hashed: {path}")
    return record


def _load_bound_checkpoint(
    path: Path, *, label: str
) -> tuple[MutableMapping[str, Any], Dict[str, Any], tuple[int, int, int, int]]:
    try:
        before = _stat_identity(path)
        payload = load_checkpoint(path)
        after_load = _stat_identity(path)
    except (OSError, ProbeAuditError) as error:
        raise CompositeArtifactError(f"could not load {label} {path}: {error}") from error
    if before != after_load:
        raise CompositeArtifactError(f"{label} changed while it was being loaded: {path}")
    record = _stable_file_record(path, label=label)
    if before != _stat_identity(path):
        raise CompositeArtifactError(f"{label} changed while it was being bound: {path}")
    return payload, record, before


def _require_unchanged(
    path: Path, expected: tuple[int, int, int, int], *, label: str
) -> None:
    try:
        observed = _stat_identity(path)
    except OSError as error:
        raise CompositeArtifactError(f"could not recheck {label} {path}: {error}") from error
    if observed != expected:
        raise CompositeArtifactError(f"{label} changed during artifact construction: {path}")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompositeArtifactError(f"{label} must be a mapping")
    return value


def _require_model_state(
    payload: Mapping[str, Any], *, label: str
) -> Mapping[str, Any]:
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise CompositeArtifactError(f"{label} has no non-empty model state")
    non_tensors = [str(key) for key, value in state.items() if not torch.is_tensor(value)]
    if non_tensors:
        raise CompositeArtifactError(
            f"{label} model state contains non-tensors: {non_tensors[:8]}"
        )
    return state


def _checkpoint_metadata(
    path: Path,
    payload: Mapping[str, Any],
    file_identity: Mapping[str, Any],
    *,
    require_positive_iteration: bool,
) -> Dict[str, Any]:
    epoch = payload.get("epoch")
    iteration = payload.get("iteration")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise CompositeArtifactError(f"checkpoint epoch must be a non-negative integer: {path}")
    if (
        isinstance(iteration, bool)
        or not isinstance(iteration, int)
        or iteration < 0
        or (require_positive_iteration and iteration <= 0)
    ):
        qualifier = "positive" if require_positive_iteration else "non-negative"
        raise CompositeArtifactError(
            f"checkpoint iteration must be a {qualifier} integer: {path}"
        )
    epoch_finished = payload.get("epoch_finished")
    if type(epoch_finished) is not bool:
        raise CompositeArtifactError(f"checkpoint epoch_finished must be boolean: {path}")
    reason = payload.get("checkpoint_reason")
    if not isinstance(reason, str) or not reason:
        raise CompositeArtifactError(f"checkpoint_reason must be non-empty: {path}")
    result = dict(file_identity)
    result.update(
        {
            "epoch": epoch,
            "iteration": iteration,
            "epoch_finished": epoch_finished,
            "checkpoint_reason": reason,
        }
    )
    return result


def _load_cfg(path: Path):
    from util.slconfig import SLConfig

    try:
        return SLConfig.fromfile(str(path))
    except Exception as error:
        raise CompositeArtifactError(f"could not load structured config {path}: {error}") from error


def _config_record(path: Path, *, repo_root: Path) -> Dict[str, Any]:
    try:
        chain = config_import_chain(path, root=repo_root)
    except DependencyAuditError as error:
        raise CompositeArtifactError(str(error)) from error
    path = path.resolve()
    if not chain or path not in chain:
        raise CompositeArtifactError(
            f"configuration import chain does not contain its leaf: {path}"
        )
    records = [
        _stable_file_record(item, label="configuration import") for item in chain
    ]
    leaf = next((record for record in records if record["path"] == str(path)), None)
    if leaf is None:
        raise CompositeArtifactError(f"configuration leaf record is absent: {path}")
    return {"leaf": dict(leaf), "import_chain": records}


def _config_chain_paths(binding: Mapping[str, Any]) -> set[Path]:
    chain = binding.get("import_chain")
    if not isinstance(chain, list):
        raise CompositeArtifactError("configuration binding has no import chain")
    result = set()
    for row in chain:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise CompositeArtifactError("configuration import record is malformed")
        result.add(Path(row["path"]).resolve())
    return result


def _validate_rank_config_chain(
    binding: Mapping[str, Any], *, recipe_version: int, repo_root: Path
) -> None:
    paths = _config_chain_paths(binding)
    ablations = (repo_root / "config" / "ablations").resolve()
    fixed_base = (ablations / "cfg_stageb_v11_fixed_text_scorer.py").resolve()
    version_roots = {
        path
        for path in paths
        if path.parent == ablations
        and (match := RANK_RECIPE_RE.match(path.name)) is not None
        and int(match.group(1)) == recipe_version
    }
    if fixed_base not in paths or not version_roots:
        raise CompositeArtifactError(
            "rank config chain must include the repository fixed-text base and "
            f"a canonical v{recipe_version} ablation leaf"
        )


def _validate_confidence_config_chain(
    binding: Mapping[str, Any], *, repo_root: Path
) -> None:
    expected = (
        repo_root / "config" / "ablations" / CONFIDENCE_CONFIG_LEAF
    ).resolve()
    if expected not in _config_chain_paths(binding):
        raise CompositeArtifactError(
            "confidence config chain does not include the repository semantic recipe"
        )


def _matches(observed: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return isinstance(observed, (int, float)) and not isinstance(
            observed, bool
        ) and math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-12)
    return observed == expected


def _require_cfg_values(cfg: Any, expected: Mapping[str, Any], *, label: str) -> None:
    for key, wanted in expected.items():
        observed = getattr(cfg, key, None)
        if not _matches(observed, wanted):
            raise CompositeArtifactError(
                f"{label} config mismatch for {key}: expected {wanted!r}, "
                f"got {observed!r}"
            )


def _rank_recipe_version(config: Path) -> int:
    match = RANK_RECIPE_RE.match(config.name)
    if match is None:
        raise CompositeArtifactError(
            "rank config leaf must identify a fixed patch/full-text v15-v20 recipe, "
            f"got {config.name!r}"
        )
    return int(match.group(1))


def _expected_rank_scorer_contract(cfg: Any, *, recipe_version: int) -> Dict[str, Any]:
    output_mode = str(
        getattr(cfg, "stage_b_v16_confidence_output_mode", "base_plus_gate")
    ).strip().lower()
    explicit = bool(
        getattr(cfg, "stage_b_v19_explicit_confidence_output_contract", False)
    )
    expected_by_recipe = {
        15: ("base_plus_gate", False, 3, None),
        16: ("gate_only", False, 4, 1),
        17: ("gate_only", False, 4, 1),
        18: ("gate_only", False, 4, 1),
        19: ("base_plus_gate", True, 5, 0),
        20: ("gate_only", False, 4, 1),
    }
    expected_mode, expected_explicit, contract_version, output_code = (
        expected_by_recipe[recipe_version]
    )
    if output_mode != expected_mode or explicit is not expected_explicit:
        raise CompositeArtifactError(
            f"rank v{recipe_version} output contract mismatch: expected "
            f"mode={expected_mode!r}, explicit={expected_explicit}, got "
            f"mode={output_mode!r}, explicit={explicit}"
        )
    return {
        "version": contract_version,
        "decoupled_confidence": True,
        "validity_pool_temperature": float(
            getattr(cfg, "stage_b_v15_validity_pool_temperature", 0.2)
        ),
        "patch_rank_fusion": True,
        "patch_rank_weight": 1.0,
        "exclude_canonical": bool(
            getattr(cfg, "stage_b_v15_exclude_canonical_from_score", False)
        ),
        "candidate_topk": 50,
        "confidence_output_mode": output_mode,
        "confidence_output_mode_code": output_code,
    }


def _scalar_value(
    state: Mapping[str, Any], key: str, *, label: str
) -> int | float | bool:
    value = state.get(key)
    if not torch.is_tensor(value) or value.numel() != 1:
        raise CompositeArtifactError(f"{label} is missing scalar tensor {key}")
    return value.detach().cpu().reshape(-1)[0].item()


def _validate_rank_scorer_state(
    state: Mapping[str, Any], expected: Mapping[str, Any], *, num_layers: int
) -> Dict[str, Any]:
    scorer_keys = sorted(
        str(key) for key in state if str(key).startswith(RANK_SCORER_PREFIX)
    )
    if not scorer_keys:
        raise CompositeArtifactError("rank checkpoint has no fixed full-text scorer state")
    if any(str(key).startswith(ADAPTER_PREFIX) for key in state):
        raise CompositeArtifactError("rank checkpoint must not contain GDINO adapter state")

    contract_suffixes = {
        "_score_contract_version": expected["version"],
        "_score_contract_decoupled_confidence": expected[
            "decoupled_confidence"
        ],
        "_score_contract_validity_pool_temperature": expected[
            "validity_pool_temperature"
        ],
        "_score_contract_patch_rank_fusion": expected["patch_rank_fusion"],
        "_score_contract_patch_rank_weight": expected["patch_rank_weight"],
        "_score_contract_exclude_canonical": expected["exclude_canonical"],
        "_score_contract_candidate_topk": expected["candidate_topk"],
    }
    if expected["confidence_output_mode_code"] is not None:
        contract_suffixes["_score_contract_confidence_output_mode"] = expected[
            "confidence_output_mode_code"
        ]
    expected_contract_keys = {
        RANK_SCORER_PREFIX + suffix for suffix in contract_suffixes
    }
    observed_contract_keys = {
        key
        for key in scorer_keys
        if key.startswith(RANK_SCORER_PREFIX + "_score_contract_")
    }
    if observed_contract_keys != expected_contract_keys:
        raise CompositeArtifactError(
            "rank scorer contract buffers are incomplete or unexpected: "
            f"missing={sorted(expected_contract_keys - observed_contract_keys)}, "
            f"unexpected={sorted(observed_contract_keys - expected_contract_keys)}"
        )
    for suffix, wanted in contract_suffixes.items():
        key = RANK_SCORER_PREFIX + suffix
        observed = _scalar_value(state, key, label="rank scorer contract")
        if isinstance(wanted, float):
            matches = isinstance(observed, (int, float)) and math.isclose(
                float(observed), wanted, rel_tol=0.0, abs_tol=1e-6
            )
        else:
            matches = observed == wanted
        if not matches:
            raise CompositeArtifactError(
                f"rank scorer contract mismatch for {suffix}: "
                f"expected {wanted!r}, got {observed!r}"
            )

    decoder_prefix = RANK_SCORER_PREFIX + "decoder."
    confidence_prefix = RANK_SCORER_PREFIX + "confidence_decoder."
    decoder_keys = sorted(key for key in scorer_keys if key.startswith(decoder_prefix))
    confidence_keys = sorted(
        key for key in scorer_keys if key.startswith(confidence_prefix)
    )
    if not decoder_keys or not confidence_keys:
        raise CompositeArtifactError(
            "rank scorer requires both rank and frozen confidence decoder states"
        )
    decoder_suffixes = {key[len(decoder_prefix) :] for key in decoder_keys}
    confidence_suffixes = {key[len(confidence_prefix) :] for key in confidence_keys}
    if decoder_suffixes != confidence_suffixes:
        raise CompositeArtifactError(
            "rank/confidence decoder state key sets are not structurally identical"
        )
    for suffix in sorted(decoder_suffixes):
        rank_value = state[decoder_prefix + suffix]
        confidence_value = state[confidence_prefix + suffix]
        if tuple(rank_value.shape) != tuple(confidence_value.shape):
            raise CompositeArtifactError(
                f"rank/confidence decoder shape mismatch for {suffix}"
            )
    layer_indices = sorted(
        {
            int(match.group(1))
            for suffix in decoder_suffixes
            if (match := re.match(r"layers\.(\d+)\.", suffix)) is not None
        }
    )
    if layer_indices != list(range(num_layers)):
        raise CompositeArtifactError(
            f"rank scorer decoder layers must be exactly 0..{num_layers - 1}, "
            f"got {layer_indices}"
        )

    validity_keys = {
        RANK_SCORER_PREFIX + suffix
        for suffix in (
            "validity_head.0.weight",
            "validity_head.0.bias",
            "validity_head.2.weight",
            "validity_head.2.bias",
        )
    }
    observed_validity = {
        key
        for key in scorer_keys
        if key.startswith(RANK_SCORER_PREFIX + "validity_head.")
    }
    if observed_validity != validity_keys:
        raise CompositeArtifactError(
            "rank scorer validity head state is incomplete or unexpected"
        )
    accounted = (
        expected_contract_keys
        | set(decoder_keys)
        | set(confidence_keys)
        | validity_keys
    )
    unexpected = sorted(set(scorer_keys).difference(accounted))
    if unexpected:
        raise CompositeArtifactError(
            f"rank scorer contains unbound state keys: {unexpected[:8]}"
        )
    return {
        "state_keys": len(scorer_keys),
        "tensor_state_sha256": tensor_state_sha256(state, scorer_keys),
        "rank_decoder_state_keys": len(decoder_keys),
        "confidence_decoder_state_keys": len(confidence_keys),
        "validity_head_state_keys": len(validity_keys),
    }


def _validate_patch_state(state: Mapping[str, Any]) -> None:
    patch_encoder = [str(key) for key in state if str(key).startswith("patch_encoder.")]
    required = {
        "query_proj_for_patch.weight",
        "query_proj_for_patch.bias",
        "patch_logit_scale",
    }
    missing = sorted(required.difference(str(key) for key in state))
    if not patch_encoder or missing:
        raise CompositeArtifactError(
            "rank checkpoint has incomplete Stage-A patch scoring state: "
            f"patch_encoder_keys={len(patch_encoder)}, missing={missing}"
        )


def _validate_rank_stagea_binding(
    rank_state: Mapping[str, Any], stagea_state: Mapping[str, Any]
) -> Dict[str, Any]:
    if any(str(key).startswith(RANK_SCORER_PREFIX) for key in stagea_state):
        raise CompositeArtifactError("declared Stage-A source unexpectedly contains scorer state")
    if any(str(key).startswith(ADAPTER_PREFIX) for key in stagea_state):
        raise CompositeArtifactError("declared Stage-A source unexpectedly contains adapter state")
    rank_base_keys = sorted(
        str(key)
        for key in rank_state
        if not str(key).startswith(RANK_SCORER_PREFIX)
    )
    stagea_keys = sorted(str(key) for key in stagea_state)
    if rank_base_keys != stagea_keys:
        raise CompositeArtifactError(
            "rank checkpoint non-scorer state is not the exact declared Stage-A state: "
            f"rank_only={sorted(set(rank_base_keys) - set(stagea_keys))[:8]}, "
            f"stagea_only={sorted(set(stagea_keys) - set(rank_base_keys))[:8]}"
        )
    rank_hash = tensor_state_sha256(rank_state, rank_base_keys)
    stagea_hash = tensor_state_sha256(stagea_state, stagea_keys)
    if rank_hash != stagea_hash:
        raise CompositeArtifactError(
            "rank checkpoint changed frozen Stage-A tensors outside the full-text scorer"
        )
    return {"state_keys": len(stagea_keys), "tensor_state_sha256": stagea_hash}


def _validate_warmstart_audit(
    audit: Any, *, num_layers: int, repo_root: Path
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        from main import _validate_stage_b_v15_scorer_init_audit

        validated = _validate_stage_b_v15_scorer_init_audit(
            audit,
            scorer=SimpleNamespace(num_layers=num_layers),
            checkpoint_label="composite rank checkpoint",
        )
    except Exception as error:
        raise CompositeArtifactError(f"invalid rank scorer warm-start audit: {error}") from error
    source = _resolve_path(validated["resolved_source_path"], repo_root=repo_root)
    record = _stable_file_record(source, label="rank scorer warm-start source")
    if (
        record["sha256"] != validated["source_sha256"]
        or record["size_bytes"] != int(validated["source_size_bytes"])
    ):
        raise CompositeArtifactError(
            "rank scorer warm-start source file changed after initialization"
        )
    return dict(validated), record


def _build_rank_component(
    *, checkpoint: Path, config: Path, stagea_checkpoint: Path, repo_root: Path
) -> Dict[str, Any]:
    config_identity = _stat_identity(config)
    cfg = _load_cfg(config)
    recipe_version = _rank_recipe_version(config)
    _require_cfg_values(
        cfg,
        {
            "modelname": "groundingdino",
            "num_queries": 900,
            "patch_only": True,
            "stage_b": False,
            "stage_b_v11_fixed_text": True,
            "stage_b_v14_validity_head": True,
            "stage_b_v15_decoupled_confidence": True,
            "stage_b_v15_patch_rank_fusion": True,
            "stage_b_v15_patch_rank_weight": 1.0,
            "stage_b_v11_candidate_topk": 50,
        },
        label="rank",
    )
    if bool(getattr(cfg, "stage_b_gdino_score_adapter", False)):
        raise CompositeArtifactError("rank config must not enable the GDINO adapter")
    scorer_contract = _expected_rank_scorer_contract(
        cfg, recipe_version=recipe_version
    )
    num_layers = int(getattr(cfg, "stage_b_v11_num_layers", 3))
    if num_layers <= 0:
        raise CompositeArtifactError("rank scorer layer count must be positive")

    payload, checkpoint_file, checkpoint_identity = _load_bound_checkpoint(
        checkpoint, label="rank checkpoint"
    )
    state = _require_model_state(payload, label="rank checkpoint")
    metadata = _checkpoint_metadata(
        checkpoint,
        payload,
        checkpoint_file,
        require_positive_iteration=False,
    )
    args = checkpoint_args(payload)
    recorded_config = args.get("config_file")
    if not recorded_config or _resolve_path(recorded_config, repo_root=repo_root) != config:
        raise CompositeArtifactError("rank checkpoint config_file does not match rank config")
    _validate_patch_state(state)
    scorer_state = _validate_rank_scorer_state(
        state, scorer_contract, num_layers=num_layers
    )
    warm_audit, warm_source = _validate_warmstart_audit(
        args.get("stage_b_v15_scorer_init_audit"),
        num_layers=num_layers,
        repo_root=repo_root,
    )

    stagea_payload, stagea_record, stagea_identity = _load_bound_checkpoint(
        stagea_checkpoint, label="Stage-A source checkpoint"
    )
    stagea_state = _require_model_state(stagea_payload, label="Stage-A source checkpoint")
    stagea_tensor = _validate_rank_stagea_binding(state, stagea_state)
    stagea_record.update(stagea_tensor)
    role_contract = {
        "role": "rank",
        "model_family": "fixed_stagea_patch_fulltext_v15_v20",
        "recipe_version": recipe_version,
        "score_key": "stage_b_v11_rank_score",
        "candidate_topk": 50,
        "patch_rank_fusion": True,
        "patch_rank_weight": 1.0,
        "exclude_canonical_from_score": scorer_contract["exclude_canonical"],
        "confidence_output_mode_in_rank_checkpoint": scorer_contract[
            "confidence_output_mode"
        ],
        "scorer_contract_version": scorer_contract["version"],
        "scorer_contract_output_mode_code": scorer_contract[
            "confidence_output_mode_code"
        ],
        "rank_confidence_policy": "rank_component_confidence_is_never_consumed",
    }
    if set(role_contract) != RANK_ROLE_CONTRACT_KEYS:
        raise AssertionError("internal rank role contract key drift")
    metadata.update(scorer_state)
    config_binding = _config_record(config, repo_root=repo_root)
    _validate_rank_config_chain(
        config_binding, recipe_version=recipe_version, repo_root=repo_root
    )
    _require_unchanged(checkpoint, checkpoint_identity, label="rank checkpoint")
    _require_unchanged(
        stagea_checkpoint, stagea_identity, label="Stage-A source checkpoint"
    )
    _require_unchanged(config, config_identity, label="rank config")
    return {
        "role_contract": role_contract,
        "checkpoint": metadata,
        "config": config_binding,
        "stagea_source": stagea_record,
        "scorer_warmstart_audit": warm_audit,
        "scorer_warmstart_source": warm_source,
    }


def _confidence_expected_config() -> Dict[str, Any]:
    return {
        "modelname": "groundingdino",
        "num_queries": 900,
        "stage_b_gdino_score_adapter": True,
        "stage_b_gdino_adapter_train_mode": "confidence_only",
        "stage_b_gdino_tn_scope": "image_global_topk_verified",
        "patch_only": False,
        "stage_b": False,
        "stage_b_v7": False,
        "stage_b_v11_fixed_text": False,
        "stage_b_legacy_global_gate": False,
        "enable_patch_branch": False,
        "stage_b_gdino_rank_weight": 0.0,
        "stage_b_gdino_confidence_weight": 1.0,
        "stage_b_gdino_confidence_objective": "detached_recent_q05_trust",
        "stage_b_gdino_fpr_temperature": 0.1,
        "stage_b_gdino_fpr_margin": 0.0,
        "stage_b_gdino_paired_margin_weight": 0.25,
        "stage_b_gdino_paired_margin": 0.05,
        "stage_b_gdino_positive_trust_margin": 0.02,
        "stage_b_gdino_positive_trust_weight": 1.0,
        "stage_b_gdino_gate_pool_temperature": 0.01,
        "stage_b_gdino_gate_topk": 3,
        "stage_b_gdino_queue_size": 512,
        "stage_b_gdino_queue_min_count": 256,
        "stage_b_gdino_positive_iou_threshold": 0.5,
        "stage_b_gdino_negative_iou_threshold": 0.5,
        "data_aug_hflip_prob": 0.0,
    }


def _build_adapter_criterion(cfg: Any) -> StageBGDINOScoreAdapterCriterion:
    return StageBGDINOScoreAdapterCriterion(
        tn_scope=str(getattr(cfg, "stage_b_gdino_tn_scope")),
        train_mode=str(getattr(cfg, "stage_b_gdino_adapter_train_mode")),
        confidence_objective=str(
            getattr(cfg, "stage_b_gdino_confidence_objective")
        ),
        positive_iou_threshold=float(
            getattr(cfg, "stage_b_gdino_positive_iou_threshold", 0.5)
        ),
        negative_iou_threshold=float(
            getattr(cfg, "stage_b_gdino_negative_iou_threshold", 0.5)
        ),
        listwise_temperature=float(
            getattr(cfg, "stage_b_gdino_listwise_temperature", 0.2)
        ),
        rank_fix_margin=float(getattr(cfg, "stage_b_gdino_rank_fix_margin", 0.05)),
        rank_preserve_margin=float(
            getattr(cfg, "stage_b_gdino_rank_preserve_margin", 0.02)
        ),
        rank_residual_weight=float(
            getattr(cfg, "stage_b_gdino_rank_residual_weight", 1e-3)
        ),
        rank_weight=float(getattr(cfg, "stage_b_gdino_rank_weight", 1.0)),
        confidence_weight=float(
            getattr(cfg, "stage_b_gdino_confidence_weight", 1.0)
        ),
        fpr_temperature=float(getattr(cfg, "stage_b_gdino_fpr_temperature", 0.1)),
        fpr_margin=float(getattr(cfg, "stage_b_gdino_fpr_margin", 0.0)),
        paired_margin_weight=float(
            getattr(cfg, "stage_b_gdino_paired_margin_weight", 0.25)
        ),
        paired_margin=float(getattr(cfg, "stage_b_gdino_paired_margin", 0.05)),
        positive_trust_margin=float(
            getattr(cfg, "stage_b_gdino_positive_trust_margin", 0.02)
        ),
        positive_trust_weight=float(
            getattr(cfg, "stage_b_gdino_positive_trust_weight", 1.0)
        ),
        queue_size=int(getattr(cfg, "stage_b_gdino_queue_size", 4096)),
        queue_min_count=int(
            getattr(cfg, "stage_b_gdino_queue_min_count", 256)
        ),
    )


def _validate_confidence_criterion(
    criterion_state: Any, *, cfg: Any
) -> Dict[str, Any]:
    if not isinstance(criterion_state, Mapping) or not criterion_state:
        raise CompositeArtifactError(
            "confidence checkpoint is missing required adapter criterion state"
        )
    criterion = _build_adapter_criterion(cfg)
    try:
        criterion.load_state_dict(criterion_state, strict=True)
    except Exception as error:
        raise CompositeArtifactError(
            f"confidence criterion state is incomplete or incompatible: {error}"
        ) from error
    queue_size = int(getattr(cfg, "stage_b_gdino_queue_size"))
    queue_min_count = int(getattr(cfg, "stage_b_gdino_queue_min_count"))
    for key in ("fpr_positive_queue", "fpr_negative_queue"):
        value = criterion_state.get(key)
        if (
            not torch.is_tensor(value)
            or not value.is_floating_point()
            or tuple(value.shape) != (queue_size,)
            or not bool(torch.isfinite(value.detach()).all().item())
        ):
            raise CompositeArtifactError(
                f"confidence criterion {key} must be a finite float[{queue_size}] tensor"
            )
    count = int(
        _scalar_value(
            criterion_state, "fpr_queue_count", label="confidence criterion"
        )
    )
    pointer = int(
        _scalar_value(
            criterion_state, "fpr_queue_ptr", label="confidence criterion"
        )
    )
    if not queue_min_count <= count <= queue_size:
        raise CompositeArtifactError(
            "confidence criterion queue is not warm or has an invalid count: "
            f"{count}, required [{queue_min_count},{queue_size}]"
        )
    if not 0 <= pointer < queue_size:
        raise CompositeArtifactError(
            f"confidence criterion queue pointer is out of range: {pointer}"
        )
    if count < queue_size and pointer != count:
        raise CompositeArtifactError(
            f"confidence partial queue pointer/count mismatch: {pointer}/{count}"
        )
    criterion_keys = sorted(str(key) for key in criterion_state)
    return {
        "state_keys": len(criterion_keys),
        "tensor_state_sha256": tensor_state_sha256(
            criterion_state, criterion_keys
        ),
        "queue_count": count,
        "queue_pointer": pointer,
        "queue_size": queue_size,
        "queue_min_count": queue_min_count,
        "queue_warm": True,
    }


def _is_forbidden_pure_gdino_key(key: str) -> bool:
    return (
        key.startswith(RANK_SCORER_PREFIX)
        or key.startswith("patch_encoder.")
        or key.startswith("query_proj_for_patch.")
        or key == "patch_logit_scale"
    )


def _validate_confidence_baseline(
    confidence_state: Mapping[str, Any], baseline_state: Mapping[str, Any]
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    baseline_hashes = model_hash_record(baseline_state)
    if int(baseline_hashes.get("adapter_state_keys", 0)) != 0:
        raise CompositeArtifactError(
            "declared confidence baseline must not contain adapter parameters"
        )
    forbidden = sorted(
        str(key) for key in baseline_state if _is_forbidden_pure_gdino_key(str(key))
    )
    if forbidden:
        raise CompositeArtifactError(
            f"declared confidence baseline is not pure non-patch GDINO: {forbidden[:8]}"
        )
    confidence_hashes = model_hash_record(confidence_state)
    if int(confidence_hashes.get("adapter_state_keys", 0)) <= 0:
        raise CompositeArtifactError("confidence checkpoint has no adapter state")
    if confidence_hashes["base_model_sha256"] != baseline_hashes[
        "base_model_sha256"
    ]:
        raise CompositeArtifactError(
            "confidence checkpoint non-adapter tensors drifted from the declared baseline"
        )
    if confidence_hashes.get("confidence_final_zero") is not False:
        raise CompositeArtifactError(
            "confidence checkpoint still has an identity/zero final gate"
        )
    return confidence_hashes, baseline_hashes


def _require_checkpoint_args(
    args: Mapping[str, Any], cfg: Any, expected: Mapping[str, Any]
) -> None:
    for key in expected:
        wanted = getattr(cfg, key, None)
        observed = args.get(key)
        if not _matches(observed, wanted):
            raise CompositeArtifactError(
                f"confidence checkpoint arg {key}: expected {wanted!r}, got {observed!r}"
            )


def _build_confidence_component(
    *, checkpoint: Path, config: Path, baseline_checkpoint: Path, repo_root: Path
) -> Dict[str, Any]:
    if config.name != CONFIDENCE_CONFIG_LEAF:
        raise CompositeArtifactError(
            "confidence config leaf must be the semantic confidence-only recipe "
            f"{CONFIDENCE_CONFIG_LEAF!r}, got {config.name!r}"
        )
    config_identity = _stat_identity(config)
    cfg = _load_cfg(config)
    expected_cfg = _confidence_expected_config()
    _require_cfg_values(cfg, expected_cfg, label="confidence")

    payload, checkpoint_file, checkpoint_identity = _load_bound_checkpoint(
        checkpoint, label="confidence checkpoint"
    )
    state = _require_model_state(payload, label="confidence checkpoint")
    metadata = _checkpoint_metadata(
        checkpoint,
        payload,
        checkpoint_file,
        require_positive_iteration=True,
    )
    if any(_is_forbidden_pure_gdino_key(str(key)) for key in state):
        raise CompositeArtifactError(
            "confidence checkpoint contains patch/fixed-text state and is not pure GDINO"
        )
    args = checkpoint_args(payload)
    recorded_config = args.get("config_file")
    if not recorded_config or _resolve_path(recorded_config, repo_root=repo_root) != config:
        raise CompositeArtifactError(
            "confidence checkpoint config_file does not match confidence config"
        )
    _require_checkpoint_args(args, cfg, expected_cfg)

    adapter = build_adapter_from_config(config, seed=0)
    try:
        validate_stage_b_gdino_score_adapter_checkpoint(
            SimpleNamespace(stage_b_gdino_score_adapter=adapter),
            state,
            checkpoint_label="composite confidence checkpoint",
        )
    except Exception as error:
        raise CompositeArtifactError(
            f"confidence adapter state is incomplete or incompatible: {error}"
        ) from error

    baseline_payload, baseline_record, baseline_identity = _load_bound_checkpoint(
        baseline_checkpoint, label="confidence baseline checkpoint"
    )
    baseline_state = _require_model_state(
        baseline_payload, label="confidence baseline checkpoint"
    )
    confidence_hashes, baseline_hashes = _validate_confidence_baseline(
        state, baseline_state
    )
    criterion = _validate_confidence_criterion(payload.get("criterion"), cfg=cfg)
    baseline_record.update(
        {
            "model_state_keys": baseline_hashes["model_state_keys"],
            "tensor_state_sha256": baseline_hashes["base_model_sha256"],
        }
    )
    metadata.update(
        {
            "model_state_keys": confidence_hashes["model_state_keys"],
            "base_state_keys": confidence_hashes["base_state_keys"],
            "adapter_state_keys": confidence_hashes["adapter_state_keys"],
            "base_tensor_state_sha256": confidence_hashes["base_model_sha256"],
            "adapter_tensor_state_sha256": confidence_hashes["adapter_sha256"],
            "confidence_tensor_state_sha256": confidence_hashes[
                "confidence_sha256"
            ],
        }
    )
    role_contract = {
        "role": "confidence",
        "model_family": "pure_gdino_semantic_confidence_adapter",
        "adapter_train_mode": "confidence_only",
        "tn_scope": "image_global_topk_verified",
        "score_key": "stage_b_gdino_confidence_score",
        "base_expression_reduction": (
            "float32_mean_sigmoid_over_generated_full_expression_tokens"
        ),
        "confidence_query_reduction": "max_over_all_900_queries",
        "query_count": 900,
        "candidate_mask_contract": "all_900_queries_valid_no_topk_mask",
        "gate_application": "one_uniform_image_expression_scalar_added_to_every_query",
        "gate_pool_temperature": 0.01,
        "gate_score_topk": 3,
        "confidence_objective": "detached_recent_q05_trust",
        "queue_size": 512,
        "queue_min_count": 256,
        "rank_confidence_policy": "confidence_component_rank_is_never_consumed",
    }
    if set(role_contract) != CONFIDENCE_ROLE_CONTRACT_KEYS:
        raise AssertionError("internal confidence role contract key drift")
    config_binding = _config_record(config, repo_root=repo_root)
    _validate_confidence_config_chain(config_binding, repo_root=repo_root)
    _require_unchanged(
        checkpoint, checkpoint_identity, label="confidence checkpoint"
    )
    _require_unchanged(
        baseline_checkpoint,
        baseline_identity,
        label="confidence baseline checkpoint",
    )
    _require_unchanged(config, config_identity, label="confidence config")
    return {
        "role_contract": role_contract,
        "checkpoint": metadata,
        "config": config_binding,
        "declared_baseline": baseline_record,
        "criterion": criterion,
    }


def _identity_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        str(key): value
        for key, value in payload.items()
        if str(key) != "artifact_identity"
    }


def _attach_identity(payload: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(payload)
    result["artifact_identity"] = {
        "algorithm": IDENTITY_ALGORITHM,
        "sha256": canonical_sha256(_identity_payload(result)),
    }
    return result


def _validate_document_identity(payload: Mapping[str, Any]) -> None:
    if set(payload) != {"schema", "kind", "composition", "rank", "confidence", "artifact_identity"}:
        raise CompositeArtifactError("composite artifact has unexpected or missing top-level fields")
    if payload.get("schema") != SCHEMA or payload.get("kind") != "two_model_stageb_composite":
        raise CompositeArtifactError("unsupported composite artifact schema or kind")
    identity = _require_mapping(payload.get("artifact_identity"), "artifact_identity")
    if set(identity) != {"algorithm", "sha256"} or identity.get(
        "algorithm"
    ) != IDENTITY_ALGORITHM:
        raise CompositeArtifactError("composite artifact identity contract is invalid")
    expected = canonical_sha256(_identity_payload(payload))
    if identity.get("sha256") != expected:
        raise CompositeArtifactError("composite artifact canonical identity mismatch")


def build_composite_artifact(
    *,
    rank_checkpoint: str | Path,
    rank_config: str | Path,
    rank_stagea_checkpoint: str | Path,
    confidence_checkpoint: str | Path,
    confidence_config: str | Path,
    confidence_baseline_checkpoint: str | Path,
    repo_root: str | Path = REPO_ROOT,
) -> Dict[str, Any]:
    """Validate all inputs and return a deterministic composite artifact."""

    root = Path(repo_root).resolve()
    rank_checkpoint_path = _resolve_path(rank_checkpoint, repo_root=root)
    confidence_checkpoint_path = _resolve_path(confidence_checkpoint, repo_root=root)
    if rank_checkpoint_path == confidence_checkpoint_path:
        raise CompositeArtifactError(
            "rank and confidence roles cannot use the same checkpoint path"
        )
    if (
        rank_checkpoint_path.is_file()
        and confidence_checkpoint_path.is_file()
        and os.path.samefile(rank_checkpoint_path, confidence_checkpoint_path)
    ):
        raise CompositeArtifactError(
            "rank and confidence roles cannot use the same checkpoint inode"
        )
    try:
        rank = _build_rank_component(
            checkpoint=rank_checkpoint_path,
            config=_resolve_path(rank_config, repo_root=root),
            stagea_checkpoint=_resolve_path(rank_stagea_checkpoint, repo_root=root),
            repo_root=root,
        )
        confidence = _build_confidence_component(
            checkpoint=confidence_checkpoint_path,
            config=_resolve_path(confidence_config, repo_root=root),
            baseline_checkpoint=_resolve_path(
                confidence_baseline_checkpoint, repo_root=root
            ),
            repo_root=root,
        )
    except CompositeArtifactError:
        raise
    except (ProbeAuditError, OSError, ValueError, TypeError, RuntimeError) as error:
        raise CompositeArtifactError(str(error)) from error
    if rank["checkpoint"]["sha256"] == confidence["checkpoint"]["sha256"]:
        raise CompositeArtifactError(
            "rank and confidence roles cannot bind byte-identical checkpoints"
        )
    payload = {
        "schema": SCHEMA,
        "kind": "two_model_stageb_composite",
        "composition": {
            "rank_source": "rank.role_contract.score_key",
            "confidence_source": "confidence.role_contract.score_key",
            "shared_trainable_score": False,
            "cross_component_parameter_sharing": "none",
            "evaluator_integration_status": "not_implemented_by_this_artifact_utility",
        },
        "rank": rank,
        "confidence": confidence,
    }
    return _attach_identity(payload)


def create_composite_artifact(
    output: str | Path,
    *,
    rank_checkpoint: str | Path,
    rank_config: str | Path,
    rank_stagea_checkpoint: str | Path,
    confidence_checkpoint: str | Path,
    confidence_config: str | Path,
    confidence_baseline_checkpoint: str | Path,
    repo_root: str | Path = REPO_ROOT,
) -> Dict[str, Any]:
    """Build and atomically write a new artifact, refusing overwrite."""

    root = Path(repo_root).resolve()
    output_path = _resolve_path(output, repo_root=root)
    if output_path.exists():
        raise CompositeArtifactError(f"refusing to overwrite composite artifact: {output_path}")
    payload = build_composite_artifact(
        rank_checkpoint=rank_checkpoint,
        rank_config=rank_config,
        rank_stagea_checkpoint=rank_stagea_checkpoint,
        confidence_checkpoint=confidence_checkpoint,
        confidence_config=confidence_config,
        confidence_baseline_checkpoint=confidence_baseline_checkpoint,
        repo_root=root,
    )
    write_json(output_path, payload)
    recorded = read_json(output_path)
    if recorded != payload:
        raise CompositeArtifactError(
            f"written composite artifact did not round-trip exactly: {output_path}"
        )
    _validate_document_identity(recorded)
    return payload


def verify_composite_artifact(
    artifact: str | Path, *, repo_root: str | Path = REPO_ROOT
) -> Dict[str, Any]:
    """Replay every bound file/config/checkpoint invariant and return a receipt."""

    root = Path(repo_root).resolve()
    artifact_path = _resolve_path(artifact, repo_root=root)
    try:
        payload = read_json(artifact_path)
    except (ProbeAuditError, OSError, ValueError) as error:
        raise CompositeArtifactError(str(error)) from error
    _validate_document_identity(payload)
    rank = _require_mapping(payload.get("rank"), "rank component")
    confidence = _require_mapping(payload.get("confidence"), "confidence component")
    try:
        rebuilt = build_composite_artifact(
            rank_checkpoint=rank["checkpoint"]["path"],
            rank_config=rank["config"]["leaf"]["path"],
            rank_stagea_checkpoint=rank["stagea_source"]["path"],
            confidence_checkpoint=confidence["checkpoint"]["path"],
            confidence_config=confidence["config"]["leaf"]["path"],
            confidence_baseline_checkpoint=confidence["declared_baseline"]["path"],
            repo_root=root,
        )
    except (KeyError, TypeError) as error:
        raise CompositeArtifactError(
            f"composite artifact is missing a bound input path: {error}"
        ) from error
    if rebuilt != payload:
        raise CompositeArtifactError(
            "composite artifact no longer exactly matches replayed role contracts and inputs"
        )
    return {
        "schema": SCHEMA,
        "kind": "two_model_stageb_composite_verified",
        "artifact": file_record(artifact_path),
        "artifact_identity": dict(payload["artifact_identity"]),
        "rank_checkpoint": dict(rank["checkpoint"]),
        "confidence_checkpoint": dict(confidence["checkpoint"]),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--output", required=True)
    create.add_argument("--rank-checkpoint", required=True)
    create.add_argument("--rank-config", required=True)
    create.add_argument("--rank-stagea-checkpoint", required=True)
    create.add_argument("--confidence-checkpoint", required=True)
    create.add_argument("--confidence-config", required=True)
    create.add_argument("--confidence-baseline-checkpoint", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--artifact", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        if args.command == "create":
            payload = create_composite_artifact(
                args.output,
                rank_checkpoint=args.rank_checkpoint,
                rank_config=args.rank_config,
                rank_stagea_checkpoint=args.rank_stagea_checkpoint,
                confidence_checkpoint=args.confidence_checkpoint,
                confidence_config=args.confidence_config,
                confidence_baseline_checkpoint=args.confidence_baseline_checkpoint,
            )
            print(
                json.dumps(
                    {
                        "status": "created",
                        "artifact": str(Path(args.output).expanduser().resolve()),
                        "artifact_identity": payload["artifact_identity"],
                    },
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                )
            )
        else:
            receipt = verify_composite_artifact(args.artifact)
            print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True))
    except CompositeArtifactError as error:
        raise SystemExit(f"[FAIL] {error}") from error


if __name__ == "__main__":
    main()
