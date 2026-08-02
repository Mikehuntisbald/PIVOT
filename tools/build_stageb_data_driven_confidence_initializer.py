#!/usr/bin/env python3
"""Seal a trained DD1 model as the shared model-only DD2/DD3 initializer."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import (  # noqa: E402
    _stage_b_data_driven_code_paths,
    _stage_b_data_driven_dataset_asset_paths,
    _stage_b_data_driven_software_record,
    _stage_b_data_driven_support_pool_content_records,
    build_model_main,
)
from models.GroundingDINO.stage_b_data_driven_score import (  # noqa: E402
    DATA_DRIVEN_CONFIDENCE_INITIALIZER_SCHEMA,
    data_driven_tensor_state_sha256,
    validate_data_driven_confidence_initializer_payload,
    validate_data_driven_initializer_payload,
    validate_stage_b_data_driven_score_checkpoint,
)
from tools.build_stageb_u0_training_receipt import (  # noqa: E402
    _safe_load_checkpoint,
    stable_file_record,
)
from tools.stageb_dependency_audit import config_import_chain  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


DEFAULT_CONFIG = (
    REPO_ROOT
    / "config/ablations/cfg_stageb_data_driven_confidence_template.py"
)
FORMAL_DD1_CONFIG = (
    REPO_ROOT / "config/ablations/cfg_stageb_data_driven_dd1_category_complete.py"
)
FORMAL_DD1_DATASETS = (
    REPO_ROOT
    / "config/datasets_stageb_data_driven_dd1_category_complete_three_ref.json"
)
FORMAL_DD1_OPTIMIZER_UPDATES = 1000
FORMAL_DD1_BATCH_SIZE = 64
FORMAL_ALLOCATOR_ENV = "PYTORCH_CUDA_ALLOC_CONF"
FORMAL_ALLOCATOR_CONF = "expandable_segments:True"
DEFAULT_BASE_INITIALIZER = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/data_driven_initializers/seed42/checkpoint_dd_init.pth"
)
EXPECTED_BASE_INITIALIZER_SHA256 = (
    "99189fb802329765d13b7700b88b76c61a81d41222ad01736aaf98e337d65032"
)
RANK_PREFIX = "stage_b_data_driven_score_heads.rank_branch."
CONFIDENCE_PREFIXES = (
    "stage_b_data_driven_score_heads.confidence_branch.",
    "stage_b_data_driven_score_heads.confidence_gate.",
)
PATCH_PREFIXES = (
    "patch_encoder.input_proj.",
    "patch_encoder.norm.",
    "query_proj_for_patch.",
)
CONTRACT_KEY = "stage_b_data_driven_score_heads._contract_version"


class ConfidenceInitializerError(RuntimeError):
    pass


def _require_source_provenance(
    saved_args: Mapping[str, Any], *, source_datasets: Path
) -> dict[str, Any]:
    provenance = saved_args.get("stage_b_data_driven_training_provenance")
    if not isinstance(provenance, Mapping) or provenance.get("schema") != (
        "pivot.stageb.data_driven_training_provenance/v1"
    ):
        raise ConfidenceInitializerError(
            "formal DD1 source is missing training provenance"
        )
    required_allocator = {
        "environment_variable": FORMAL_ALLOCATOR_ENV,
        "value": FORMAL_ALLOCATOR_CONF,
    }
    if provenance.get("required_allocator") != required_allocator:
        raise ConfidenceInitializerError("formal DD1 allocator contract drifted")
    expected_allocator_environment = {
        "PYTORCH_ALLOC_CONF": None,
        "PYTORCH_CUDA_ALLOC_CONF": FORMAL_ALLOCATOR_CONF,
    }
    if provenance.get("allocator_environment") != expected_allocator_environment:
        raise ConfidenceInitializerError(
            "formal DD1 allocator environment was not sealed exactly"
        )

    expected_path_groups = {
        "code_files": _stage_b_data_driven_code_paths(),
        "dataset_asset_files": _stage_b_data_driven_dataset_asset_paths(
            source_datasets
        ),
    }
    for group, paths in expected_path_groups.items():
        observed = provenance.get(group)
        if not isinstance(observed, list) or any(
            not isinstance(record, Mapping) for record in observed
        ):
            raise ConfidenceInitializerError(
                f"formal DD1 provenance {group} is malformed"
            )
        expected = [
            stable_file_record(path, label=f"formal DD1 provenance {group}")
            for path in paths
        ]
        if observed != expected:
            raise ConfidenceInitializerError(
                f"formal DD1 provenance {group} drifted"
            )
    software = provenance.get("software")
    if software != _stage_b_data_driven_software_record():
        raise ConfidenceInitializerError(
            "formal DD1 software provenance drifted from the sealer environment"
        )
    expected_support_pool = _stage_b_data_driven_support_pool_content_records(
        expected_path_groups["dataset_asset_files"]
    )
    if provenance.get("support_patch_pool_content") != expected_support_pool:
        raise ConfidenceInitializerError(
            "formal DD1 support-patch image content drifted"
        )
    return dict(provenance)


def _build_template(config: Path):
    cfg = SLConfig.fromfile(str(config))
    cfg.device = "cpu"
    if not bool(getattr(cfg, "stage_b_data_driven_score", False)) or str(
        getattr(cfg, "stage_b_data_driven_train_mode", "")
    ) != "confidence_pair":
        raise ConfidenceInitializerError(
            "confidence initializer config must enable data-driven confidence_pair"
        )
    if not bool(getattr(cfg, "stage_b_data_driven_category_complete", False)):
        raise ConfidenceInitializerError("confidence phase must inherit DD1 category-complete")
    torch.manual_seed(int(getattr(cfg, "seed", 42)))
    model, _criterion, _postprocessors = build_model_main(cfg)
    return model.eval(), cfg


def _model_state(payload: Mapping[str, Any], *, label: str) -> Mapping[str, torch.Tensor]:
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state or any(
        not isinstance(key, str) or not torch.is_tensor(value)
        for key, value in state.items()
    ):
        raise ConfidenceInitializerError(f"{label} has no tensor-only model state")
    return state


def _role_keys(state: Mapping[str, torch.Tensor]) -> dict[str, list[str]]:
    roles = {
        "frozen_b58_base": [],
        "frozen_shared_backbone_alias": [],
        "dd1_trained_rank": [],
        "dd1_trained_patch": [],
        "untouched_random_confidence": [],
        "score_contract_buffer": [],
    }
    for key in state:
        if key == CONTRACT_KEY:
            role = "score_contract_buffer"
        elif key.startswith(RANK_PREFIX):
            role = "dd1_trained_rank"
        elif key.startswith(CONFIDENCE_PREFIXES):
            role = "untouched_random_confidence"
        elif key == "patch_logit_scale" or key.startswith(PATCH_PREFIXES):
            role = "dd1_trained_patch"
        elif key.startswith("patch_encoder.backbone."):
            role = "frozen_shared_backbone_alias"
        else:
            role = "frozen_b58_base"
        roles[role].append(key)
    if any(not keys for keys in roles.values()):
        raise ConfidenceInitializerError("confidence initializer has an empty role")
    return roles


def _config_binding(path: Path) -> dict[str, Any]:
    return {
        "leaf": stable_file_record(path, label="confidence initializer config"),
        "import_chain": [
            stable_file_record(item, label=f"config dependency {item.name}")
            for item in config_import_chain(path, root=REPO_ROOT)
        ],
    }


def _require_source_contract(
    payload: Mapping[str, Any], *, scope: str, minimum_updates: int
) -> tuple[dict[str, Any], int]:
    saved_args = payload.get("args")
    if not isinstance(saved_args, Mapping):
        raise ConfidenceInitializerError("DD1 source is missing saved args")
    expected = {
        "stage_b_data_driven_experiment_id": "DD1",
        "stage_b_data_driven_train_mode": "rank_patch_only",
        "stage_b_data_driven_category_complete": True,
        "seed": 42,
    }
    drift = {
        key: (saved_args.get(key), value)
        for key, value in expected.items()
        if saved_args.get(key) != value
    }
    if drift:
        raise ConfidenceInitializerError(f"DD1 source args drifted: {drift}")
    if scope == "formal" and saved_args.get(
        "stage_b_data_driven_confidence_trained"
    ) is not False:
        raise ConfidenceInitializerError(
            "formal DD1 source must seal confidence_trained=false"
        )
    if scope == "formal":
        source_config = Path(str(saved_args.get("config_file", "")))
        source_datasets = Path(str(saved_args.get("datasets", "")))
        if not source_config.is_absolute():
            source_config = REPO_ROOT / source_config
        if not source_datasets.is_absolute():
            source_datasets = REPO_ROOT / source_datasets
        if source_config.resolve(strict=True) != FORMAL_DD1_CONFIG.resolve(strict=True):
            raise ConfidenceInitializerError("formal DD1 source config path drifted")
        if source_datasets.resolve(strict=True) != FORMAL_DD1_DATASETS.resolve(strict=True):
            raise ConfidenceInitializerError("formal DD1 dataset config path drifted")
        expected_chain = [
            {
                "path": str(path.resolve()),
                "sha256": stable_file_record(
                    path, label=f"formal DD1 config dependency {path.name}"
                )["sha256"],
            }
            for path in config_import_chain(FORMAL_DD1_CONFIG, root=REPO_ROOT)
        ]
        expected_dataset = {
            "path": str(FORMAL_DD1_DATASETS.resolve()),
            "sha256": stable_file_record(
                FORMAL_DD1_DATASETS, label="formal DD1 dataset config"
            )["sha256"],
        }
        if saved_args.get("stage_b_data_driven_config_import_chain") != expected_chain:
            raise ConfidenceInitializerError(
                "formal DD1 config import-chain hashes drifted"
            )
        if saved_args.get("stage_b_data_driven_dataset_config") != expected_dataset:
            raise ConfidenceInitializerError("formal DD1 dataset config hash drifted")
        if saved_args.get("stage_b_data_driven_base_initializer_sha256") != (
            EXPECTED_BASE_INITIALIZER_SHA256
        ):
            raise ConfidenceInitializerError("formal DD1 base initializer binding drifted")
        source_base_record = saved_args.get("stage_b_data_driven_base_initializer")
        full_base_record = stable_file_record(
            DEFAULT_BASE_INITIALIZER, label="formal DD1 base initializer"
        )
        expected_base_record = {
            "path": full_base_record["path"],
            "sha256": full_base_record["sha256"],
        }
        if source_base_record != expected_base_record:
            raise ConfidenceInitializerError("formal DD1 pretrain lineage drifted")
        exact_runtime = {
            "batch_size": FORMAL_DD1_BATCH_SIZE,
            "amp": True,
            "amp_init_scale": 8192.0,
            "gradient_accumulation_steps": 1,
            "seed": 42,
            "world_size": 1,
            "distributed": False,
            "num_workers": 4,
            "prefetch_factor": 1,
            "pin_memory": None,
            "persistent_workers": None,
            "max_train_iters": FORMAL_DD1_OPTIMIZER_UPDATES,
            "stage_b_data_driven_rank_weight": 1.0,
            "stage_b_data_driven_patch_weight": 1.0,
            "stage_b_data_driven_positive_iou_threshold": 0.5,
            "stage_b_data_driven_patch_negative_iou_threshold": 0.3,
            "stage_b_data_driven_temperature": 0.1,
            "stage_b_data_driven_category_margin": 0.1,
            "stage_b_data_driven_rank_lr": 3e-5,
            "stage_b_data_driven_patch_lr": 3e-4,
            "weight_decay": 1e-4,
            "clip_max_norm": 0.1,
            "data_aug_hflip_prob": 0.0,
            "fix_size": True,
            "stage_b_data_driven_category_gate": False,
            "stage_b_data_driven_required_allocator_env": FORMAL_ALLOCATOR_ENV,
            "stage_b_data_driven_required_allocator_conf": FORMAL_ALLOCATOR_CONF,
        }
        runtime_drift = {
            key: (saved_args.get(key), value)
            for key, value in exact_runtime.items()
            if saved_args.get(key) != value
        }
        if runtime_drift:
            raise ConfidenceInitializerError(
                f"formal DD1 runtime contract drifted: {runtime_drift}"
            )
        if saved_args.get("options") != {"batch_size": 64, "epochs": 1}:
            raise ConfidenceInitializerError(
                "formal DD1 allows only batch_size=64, epochs=1 config options"
            )
        _require_source_provenance(saved_args, source_datasets=source_datasets)
    updates = payload.get("optimizer_updates")
    if (
        not isinstance(updates, int)
        or isinstance(updates, bool)
        or updates < int(minimum_updates)
    ):
        raise ConfidenceInitializerError(
            f"DD1 source optimizer_updates={updates!r} is below {minimum_updates}"
        )
    if scope == "formal" and (
        int(minimum_updates) != FORMAL_DD1_OPTIMIZER_UPDATES
        or int(updates) != FORMAL_DD1_OPTIMIZER_UPDATES
    ):
        raise ConfidenceInitializerError(
            "formal DD1 handoff requires exactly 1000 optimizer updates"
        )
    optimizer = payload.get("optimizer")
    groups = optimizer.get("param_groups") if isinstance(optimizer, Mapping) else None
    branches = (
        [group.get("stage_b_data_driven_branch") for group in groups]
        if isinstance(groups, list) and all(isinstance(group, Mapping) for group in groups)
        else None
    )
    if branches != ["rank", "patch"]:
        raise ConfidenceInitializerError(
            f"DD1 source optimizer branch contract drifted: {branches!r}"
        )
    selected = dict(expected)
    selected["stage_b_data_driven_confidence_trained"] = saved_args.get(
        "stage_b_data_driven_confidence_trained"
    )
    selected["config_file"] = saved_args.get("config_file")
    selected["datasets"] = saved_args.get("datasets")
    return selected, int(updates)


def build_payload(
    *,
    source_checkpoint: Path,
    source_sha256: str,
    base_initializer: Path,
    base_sha256: str,
    config: Path,
    scope: str,
    minimum_updates: int,
) -> tuple[dict[str, Any], torch.nn.Module]:
    if scope not in {"smoke", "formal"}:
        raise ConfidenceInitializerError("scope must be smoke or formal")
    source_record = stable_file_record(source_checkpoint, label="DD1 source checkpoint")
    if source_record["sha256"] != source_sha256.lower():
        raise ConfidenceInitializerError("DD1 source checkpoint SHA256 mismatch")
    base_record = stable_file_record(base_initializer, label="base data-driven initializer")
    if base_record["sha256"] != base_sha256.lower():
        raise ConfidenceInitializerError("base initializer SHA256 mismatch")
    if scope == "formal" and (
        base_initializer.resolve(strict=True) != DEFAULT_BASE_INITIALIZER.resolve(strict=True)
        or base_sha256.lower() != EXPECTED_BASE_INITIALIZER_SHA256
        or config.resolve(strict=True) != DEFAULT_CONFIG.resolve(strict=True)
    ):
        raise ConfidenceInitializerError(
            "formal confidence initializer forbids base/config overrides"
        )

    model, cfg = _build_template(config)
    source_payload = _safe_load_checkpoint(source_checkpoint, label="DD1 source checkpoint")
    base_payload = _safe_load_checkpoint(base_initializer, label="base initializer")
    source_state = _model_state(source_payload, label="DD1 source")
    base_state = _model_state(base_payload, label="base initializer")
    nonfinite = [
        key
        for key, value in source_state.items()
        if (value.is_floating_point() or value.is_complex())
        and not bool(torch.isfinite(value).all().item())
    ]
    if nonfinite:
        raise ConfidenceInitializerError(
            f"DD1 source contains non-finite tensors: {nonfinite[:8]}"
        )
    validate_stage_b_data_driven_score_checkpoint(
        model, source_state, checkpoint_label="DD1 source checkpoint"
    )
    validate_data_driven_initializer_payload(
        model, base_payload, checkpoint_label="base data-driven initializer"
    )
    source_args, updates = _require_source_contract(
        source_payload, scope=scope, minimum_updates=minimum_updates
    )
    source_training_provenance = source_payload.get("args", {}).get(
        "stage_b_data_driven_training_provenance"
    )
    roles = _role_keys(source_state)
    unchanged_roles = (
        "frozen_b58_base",
        "frozen_shared_backbone_alias",
        "untouched_random_confidence",
        "score_contract_buffer",
    )
    for role in unchanged_roles:
        drifted = [
            key for key in roles[role] if not torch.equal(source_state[key], base_state[key])
        ]
        if drifted:
            raise ConfidenceInitializerError(
                f"DD1 changed an immutable {role} tensor: {drifted[:8]}"
            )
    for role in ("dd1_trained_rank", "dd1_trained_patch"):
        if not any(
            not torch.equal(source_state[key], base_state[key]) for key in roles[role]
        ):
            raise ConfidenceInitializerError(f"DD1 did not train its {role} role")

    state = OrderedDict(
        (key, value.detach().cpu().clone()) for key, value in source_state.items()
    )
    model.load_state_dict(state, strict=True)
    contract = {
        "schema": DATA_DRIVEN_CONFIDENCE_INITIALIZER_SCHEMA,
        "scope": scope,
        "training_initializer": True,
        "resumable": False,
        "source_checkpoint": source_record,
        "source_optimizer_updates": updates,
        "minimum_source_optimizer_updates": int(minimum_updates),
        "source_training_position": {
            key: source_payload.get(key)
            for key in ("epoch", "iteration", "epoch_finished")
        },
        "source_args_contract": source_args,
        "source_training_provenance": source_training_provenance,
        "base_initializer": base_record,
        "config": _config_binding(config),
        "architecture": {
            "hidden_dim": int(cfg.hidden_dim),
            "num_queries": int(cfg.num_queries),
            "rank_dim": int(cfg.stage_b_data_driven_rank_dim),
            "confidence_dim": int(cfg.stage_b_data_driven_confidence_dim),
        },
        "role_keys": roles,
        "role_key_counts": {role: len(keys) for role, keys in roles.items()},
        "full_model_tensor_sha256": data_driven_tensor_state_sha256(
            state, sorted(state)
        ),
        "optimizer_state_carried": False,
        "criterion_state_carried": False,
        "scheduler_scaler_rng_carried": False,
        "invariants": {
            "dd1_model_preserved_bitwise": True,
            "rank_changed_from_base_initializer": True,
            "patch_changed_from_base_initializer": True,
            "confidence_unchanged_from_base_initializer": True,
            "frozen_b58_and_alias_unchanged_from_base_initializer": True,
            "new_confidence_optimizer_required": True,
            "dd2_dd3_share_this_initializer": True,
        },
    }
    for role, keys in roles.items():
        contract[f"{role}_tensor_sha256"] = data_driven_tensor_state_sha256(
            state, keys
        )
    payload = {
        "model": state,
        "data_driven_confidence_initializer": contract,
    }
    validate_data_driven_confidence_initializer_payload(
        model, payload, checkpoint_label="in-memory confidence initializer"
    )
    del source_payload, base_payload
    gc.collect()
    return payload, model


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"confidence initializer output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary initializer exists: {temporary}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--base-initializer", type=Path, default=DEFAULT_BASE_INITIALIZER)
    parser.add_argument(
        "--base-sha256", default=EXPECTED_BASE_INITIALIZER_SHA256
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scope", choices=("smoke", "formal"), required=True)
    parser.add_argument("--minimum-updates", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload, model = build_payload(
        source_checkpoint=args.source_checkpoint.resolve(strict=True),
        source_sha256=args.source_sha256,
        base_initializer=args.base_initializer.resolve(strict=True),
        base_sha256=args.base_sha256,
        config=args.config.resolve(strict=True),
        scope=args.scope,
        minimum_updates=args.minimum_updates,
    )
    _write(args.output, payload)
    reloaded = _safe_load_checkpoint(args.output, label="published confidence initializer")
    validate_data_driven_confidence_initializer_payload(
        model,
        reloaded,
        checkpoint_label=f"published confidence initializer {args.output}",
    )
    record = stable_file_record(args.output, label="published confidence initializer")
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
