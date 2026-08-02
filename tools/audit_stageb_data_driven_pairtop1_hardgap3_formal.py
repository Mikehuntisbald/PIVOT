#!/usr/bin/env python3
"""Audit the terminal DD1-PairTop1-HardGap3 formal U5020 checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.audit_stageb_data_driven_pairtop1_hardgap3_probe import (  # noqa: E402
    EXPECTED_ASSIGNMENT_RECEIPT_SHA256,
    EXPECTED_CHECKPOINT_KEYS,
    EXPECTED_CODE_FILE_COUNT,
    EXPECTED_CRITERION_VERSION,
    EXPECTED_DATASET_CONFIG_SHA256,
    EXPECTED_INITIALIZER,
    EXPECTED_INITIALIZER_PAIR_SHA256,
    EXPECTED_INITIALIZER_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_MODEL_TENSOR_COUNT,
    EXPECTED_PATCH_PARAMETER_COUNT,
    EXPECTED_RANK_SUPERVISION,
    EXPECTED_RANK_SUPERVISION_ID,
    EXPECTED_RANK_TRAINABLE_COUNT,
    EXPECTED_ROWS,
    EXPECTED_SAMPLING_STATE,
    EXPECTED_SCOPE,
    EXPECTED_VALID_ROWS,
    HARDGAP3_VARIANT,
    PATCH_PARAMETER_NAMES,
    PROVENANCE_SCHEMA,
    RANK_FIXED_BUFFER,
    RANK_PREFIX,
    PairTop1HardGap3ProbeAuditError,
    _atomic_publish_fresh_json,
    _audit_finite_tree,
    _deep_equal,
    _safe_load_checkpoint,
    _tensor_bitwise_equal,
    _validate_data_and_initializer_bindings,
    canonical_json_sha256,
    stable_file_record,
)
from tools.stageb_dependency_audit import config_import_chain  # noqa: E402


SCHEMA = "pivot.stageb.data_driven.pairtop1_hardgap3_formal_training_receipt/v1"
EXPECTED_EXECUTION_SCOPE = "formal_fresh_a1_u5020_v1"
EXPECTED_UPDATES = 5020
EXPECTED_CRITERION_TENSOR_COUNT = 5
EXPECTED_FORMAL_CONFIG_SHA256 = (
    "01e18b44c3b64cbaedb8a7e146a7f6ad6852f7fc156e01ab2ba195b8278d0abd"
)
EXPECTED_PROBE_RECEIPT_SHA256 = (
    "d530a2d97be56fe08fa0aa27468c7791965b43bb148395f1e42ddf110f66bf6f"
)
EXPECTED_GATE_SHA256 = (
    "067abfe62838adee0de343d6be5c396132322db21bfdfc9c195490b4282adea0"
)
EXPECTED_PREFLIGHT_SHA256 = (
    "8ffc0ac5b174949d8799e66b69cfa3c3474c63ea05be33a5a9891d5d93c7de38"
)
EXPECTED_OVERFIT_AUDIT_SHA256 = (
    "754f71f38e37c2aac04e2428eb38952ef4ecac955684d3db220e15683ebd71e3"
)
EXPECTED_PROBE_INTERNAL_SHA256 = (
    "e35bdbe20f28fef0f0eb0c09b9f635ec488b77486717f4826d466358e6210e3d"
)
EXPECTED_TRAINING_CODE_MANIFEST_SHA256 = (
    "e22b87cb6206175d6de19a1bb142d72ec49c34898128e805b74eed5b1910b958"
)
EXPECTED_WEIGHTS_SHA256 = (
    "257c1ab2f46a4328641ee49356e046f84db87c1deeaae15b8de10678080fed6a"
)
EXPECTED_LEDGER_SHA256 = (
    "6ac698c5d20bbfd4917f2b366f2088eb047a8905b4af2090d167c1c97a0a5ba9"
)

EXPECTED_FORMAL_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_data_driven_dd1_pairtop1_hardgap3_fair_v2_formal.py"
)
EXPECTED_BASE_CONFIG = (
    REPO_ROOT
    / "config/ablations/cfg_stageb_data_driven_dd1_pairtop1_hardgap3_fair_v2.py"
)
EXPECTED_DATASET_CONFIG = (
    REPO_ROOT / "config/datasets_stageb_data_driven_dd1_official_assignment_three_ref.json"
)
EXPECTED_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/"
    "data_driven_dd1_pairtop1_hardgap3_fair_v2_seed42_b64_u5020_v1"
)
EXPECTED_PROBE_RECEIPT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/"
    "data_driven_dd1_pairtop1_hardgap3_fair_v2_seed42_b64_fullprobe_u50_v2/"
    "probe_receipt.json"
)
EXPECTED_PREFLIGHT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/"
    "data_driven_dd1_pairtop1_hardgap3_fair_v2_preflight/"
    "metadata_causal_and_ledger_audit.json"
)
EXPECTED_GATE = EXPECTED_PREFLIGHT.parent / "formal_gate_contract.json"
EXPECTED_OVERFIT_AUDIT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/"
    "data_driven_dd1_pairtop1_hardgap3_overfit64_seed42_b64_u500_v3/"
    "pairtop1_hardgap3_eval_final_v4b.json"
)

EXPECTED_RELATIVE_CONFIG = str(EXPECTED_FORMAL_CONFIG.relative_to(REPO_ROOT))
EXPECTED_RELATIVE_DATASET = str(EXPECTED_DATASET_CONFIG.relative_to(REPO_ROOT))
EXPECTED_RELATIVE_OUTPUT = str(EXPECTED_OUTPUT_DIR.relative_to(REPO_ROOT))
EXPECTED_RELATIVE_INITIALIZER = str(EXPECTED_INITIALIZER.relative_to(REPO_ROOT))

CONFIDENCE_PREFIXES = (
    "stage_b_data_driven_score_heads.confidence_branch.",
    "stage_b_data_driven_score_heads.confidence_gate.",
)
EXPECTED_INITIALIZER_ROLE_COUNTS = {
    "b58_base": 938,
    "shared_backbone_alias": 187,
    "random_patch_projection": 9,
    "random_relational_rank": 40,
    "random_absolute_confidence": 15,
    "score_contract_buffer": 1,
}
EXPECTED_SCALER = {
    "scale": 32768.0,
    "growth_factor": 2.0,
    "backoff_factor": 0.5,
    "growth_interval": 2000,
    "_growth_tracker": 1020,
}
EXPECTED_SCHEDULER = {
    "step_size": 100,
    "gamma": 0.1,
    "base_lrs": [3e-5, 3e-4],
    "last_epoch": 1,
    "_step_count": 2,
    "_is_initial": False,
    "_get_lr_called_within_step": False,
    "_last_lr": [3e-5, 3e-4],
}


class PairTop1HardGap3FormalAuditError(RuntimeError):
    """The formal checkpoint does not satisfy the sealed U5020 contract."""


def _strict_equal(observed: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(observed, bool) and observed is expected
    if isinstance(expected, int):
        return type(observed) is int and observed == expected
    if isinstance(expected, float):
        return type(observed) is float and observed == expected
    return type(observed) is type(expected) and observed == expected


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PairTop1HardGap3FormalAuditError(f"{label} must be a mapping")
    return value


def _require_exact_fields(
    value: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    drift = {
        key: {"expected": wanted, "observed": value.get(key)}
        for key, wanted in expected.items()
        if not _strict_equal(value.get(key), wanted)
    }
    if drift:
        raise PairTop1HardGap3FormalAuditError(
            f"{label} exact contract drifted: {drift}"
        )


def _strict_json_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PairTop1HardGap3FormalAuditError(
                f"JSON evidence contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise PairTop1HardGap3FormalAuditError(
        f"JSON evidence contains non-finite constant {value!r}"
    )


def _load_strict_json(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    before = stable_file_record(path, label=label)
    try:
        payload = json.loads(
            Path(before["path"]).read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except PairTop1HardGap3FormalAuditError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PairTop1HardGap3FormalAuditError(
            f"could not parse {label} as strict JSON: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise PairTop1HardGap3FormalAuditError(f"{label} root must be an object")
    after = stable_file_record(Path(before["path"]), label=label)
    if before != after:
        raise PairTop1HardGap3FormalAuditError(f"{label} changed while it was read")
    return payload, before


def _require_input_record(
    path: Path,
    *,
    expected_path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        resolved = path.expanduser().resolve(strict=True)
        canonical_path = expected_path.resolve(strict=True)
    except OSError as error:
        raise PairTop1HardGap3FormalAuditError(
            f"could not resolve {label}: {error}"
        ) from error
    if resolved != canonical_path:
        raise PairTop1HardGap3FormalAuditError(
            f"{label} path drifted: expected {canonical_path}, got {resolved}"
        )
    payload, record = _load_strict_json(resolved, label=label)
    if record["sha256"] != expected_sha256:
        raise PairTop1HardGap3FormalAuditError(
            f"{label} SHA drifted: expected {expected_sha256}, "
            f"got {record['sha256']}"
        )
    return payload, record


def _validate_file_reference(
    value: Any,
    *,
    label: str,
    expected_path: Path | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    reference = _require_mapping(value, label=label)
    if set(reference) != {"path", "sha256"}:
        raise PairTop1HardGap3FormalAuditError(f"{label} schema drifted")
    raw_path = reference.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise PairTop1HardGap3FormalAuditError(f"{label}.path is invalid")
    path = Path(os.path.expandvars(os.path.expanduser(raw_path)))
    if not path.is_absolute():
        path = REPO_ROOT / path
    record = stable_file_record(path, label=label)
    if reference.get("sha256") != record["sha256"]:
        raise PairTop1HardGap3FormalAuditError(f"{label} saved SHA drifted")
    if expected_path is not None and Path(record["path"]) != expected_path.resolve(
        strict=True
    ):
        raise PairTop1HardGap3FormalAuditError(f"{label} path drifted")
    if expected_sha256 is not None and record["sha256"] != expected_sha256:
        raise PairTop1HardGap3FormalAuditError(f"{label} canonical SHA drifted")
    return record


def _audit_probe_receipt(payload: Mapping[str, Any]) -> None:
    _require_exact_fields(
        payload,
        {
            "schema": "pivot.stageb.data_driven.pairtop1_hardgap3_probe_receipt/v1",
            "status": "passed",
            "scope": "memory_and_protocol_probe_only_do_not_resume_into_formal",
            "receipt_sha256": EXPECTED_PROBE_INTERNAL_SHA256,
        },
        label="sealed U50 probe receipt",
    )
    unsigned = dict(payload)
    observed_internal = unsigned.pop("receipt_sha256", None)
    if canonical_json_sha256(unsigned) != observed_internal:
        raise PairTop1HardGap3FormalAuditError(
            "sealed U50 probe internal receipt digest drifted"
        )
    checkpoint = _require_mapping(payload.get("checkpoint"), label="U50 checkpoint")
    _require_exact_fields(
        checkpoint,
        {
            "optimizer_updates": 50,
            "iteration": 50,
            "checkpoint_reason": "max_train_iters",
            "epoch": 0,
            "epoch_finished": False,
        },
        label="U50 terminal checkpoint",
    )
    source = _require_mapping(payload.get("source"), label="U50 source")
    _require_exact_fields(
        source,
        {
            "training_code_file_count": EXPECTED_CODE_FILE_COUNT,
            "training_code_files_manifest_sha256": (
                EXPECTED_TRAINING_CODE_MANIFEST_SHA256
            ),
            "saved_training_provenance_bitwise_equal": True,
        },
        label="U50 source",
    )
    criterion = _require_mapping(payload.get("criterion"), label="U50 criterion")
    _require_exact_fields(
        criterion,
        {
            "criterion_contract_version": EXPECTED_CRITERION_VERSION,
            "rank_supervision_contract_id": EXPECTED_RANK_SUPERVISION_ID,
            "rank_supervision": EXPECTED_RANK_SUPERVISION,
            "assignment_weight": 1.0,
            "deployment_weight": 1.0,
        },
        label="U50 criterion",
    )
    invariants = _require_mapping(payload.get("invariants"), label="U50 invariants")
    required_invariants = {
        "fifty_of_fifty_optimizer_updates_succeeded",
        "amp_step_skips_zero",
        "all_model_and_optimizer_tensors_are_finite",
        "all_criterion_and_scaler_state_is_finite",
        "official_assignment_full_data_and_initializer_are_identical",
        "training_code_manifest_is_identical",
        "only_rank_trainable_model_and_optimizer_state_changed",
        "no_teacher_logits_weights_or_loss_targets_are_used",
        "probe_checkpoint_is_forbidden_as_formal_resume_source",
    }
    if set(invariants) != required_invariants or any(
        invariants[key] is not True for key in required_invariants
    ):
        raise PairTop1HardGap3FormalAuditError("U50 invariants drifted")


def _audit_gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_fields(
        payload,
        {
            "schema": "pivot.stageb.data_driven.pairtop1_hardgap3_formal_gate/v1",
            "status": "sealed_before_training",
        },
        label="formal gate",
    )
    training = _require_mapping(payload.get("training"), label="formal gate training")
    _require_exact_fields(
        training,
        {
            "variant_id": HARDGAP3_VARIANT,
            "config": str(EXPECTED_FORMAL_CONFIG),
            "output_dir": str(EXPECTED_OUTPUT_DIR),
            "fresh_start": True,
            "resume_forbidden": True,
            "optimizer_updates": EXPECTED_UPDATES,
            "epochs": 1,
            "seed": 42,
            "batch_size": 64,
            "num_workers": 4,
            "prefetch_factor": 1,
            "persistent_workers": False,
            "gradient_accumulation_steps": 1,
            "amp": True,
            "amp_init_scale": 8192.0,
            "iter_checkpoint_interval": 500,
            "world_size": 1,
            "distributed": False,
            "rank_supervision_contract_id": EXPECTED_RANK_SUPERVISION_ID,
            "assignment_weight": 1.0,
            "deployment_weight": 1.0,
            "category_gate_max_gap": 3.0,
            "patch_score_clip": 5.0,
        },
        label="formal gate training",
    )
    _validate_file_reference(
        training.get("base_intervention_config"),
        label="formal gate base intervention config",
        expected_path=EXPECTED_BASE_CONFIG,
    )
    _validate_file_reference(
        training.get("dataset_config"),
        label="formal gate dataset config",
        expected_path=EXPECTED_DATASET_CONFIG,
        expected_sha256=EXPECTED_DATASET_CONFIG_SHA256,
    )
    _validate_file_reference(
        training.get("initializer"),
        label="formal gate initializer",
        expected_path=EXPECTED_INITIALIZER,
        expected_sha256=EXPECTED_INITIALIZER_SHA256,
    )
    preconditions = _require_mapping(
        payload.get("preconditions"), label="formal gate preconditions"
    )
    _validate_file_reference(
        preconditions.get("metadata_causal_and_ledger_preflight"),
        label="formal gate preflight",
        expected_path=EXPECTED_PREFLIGHT,
        expected_sha256=EXPECTED_PREFLIGHT_SHA256,
    )
    _validate_file_reference(
        preconditions.get("sealed_u50_probe_receipt"),
        label="formal gate U50 receipt",
        expected_path=EXPECTED_PROBE_RECEIPT,
        expected_sha256=EXPECTED_PROBE_RECEIPT_SHA256,
    )
    _validate_file_reference(
        preconditions.get("overfit64_deployment_audit"),
        label="formal gate Overfit64 audit",
        expected_path=EXPECTED_OVERFIT_AUDIT,
        expected_sha256=EXPECTED_OVERFIT_AUDIT_SHA256,
    )
    if (
        preconditions.get("training_code_files_manifest_sha256")
        != EXPECTED_TRAINING_CODE_MANIFEST_SHA256
    ):
        raise PairTop1HardGap3FormalAuditError(
            "formal gate training-code manifest drifted"
        )
    headline = _require_mapping(
        payload.get("headline_evaluation"), label="formal headline evaluation"
    )
    _require_exact_fields(
        headline,
        {
            "checkpoint": str(EXPECTED_OUTPUT_DIR / "checkpoint_iter.pth"),
            "split": "refcocog_val",
            "batch_size": 16,
            "num_workers": 4,
            "base_seed": 42,
            "effective_split_seed": 600042,
            "amp": True,
            "max_batches": 0,
            "max_images": 0,
            "primary_metric": "acc50",
            "score_route": "patch_category_gate_then_full_text_rank",
            "category_gate_max_gap": 3.0,
            "patch_score_clip": 5.0,
            "required_records": 4896,
            "required_valid_records": 4896,
            "required_unique_manifest_indices": 4896,
        },
        label="formal headline evaluation",
    )
    _validate_file_reference(headline.get("config"), label="headline config")
    _validate_file_reference(headline.get("evaluator"), label="headline evaluator")
    headline_gate = _require_mapping(
        payload.get("headline_gate"), label="formal headline gate"
    )
    _require_exact_fields(
        headline_gate,
        {
            "metric": "acc50",
            "comparison": "strictly_greater_than_historical_gdino_stage_b_data_ft",
            "baseline_correct": 3942,
            "minimum_correct": 3943,
            "total": 4896,
            "baseline_accuracy": 3942 / 4896,
            "minimum_accuracy": 3943 / 4896,
        },
        label="formal headline gate",
    )
    _validate_file_reference(
        headline_gate.get("baseline_summary"), label="headline baseline summary"
    )
    _validate_file_reference(
        headline_gate.get("baseline_split_record"),
        label="headline baseline split record",
    )
    return dict(training)


def _audit_preflight(payload: Mapping[str, Any]) -> None:
    _require_exact_fields(
        payload,
        {
            "schema": "pivot.stageb.data_driven.pairtop1_hardgap3_preflight/v1",
            "status": "passed",
            "variant_id": HARDGAP3_VARIANT,
        },
        label="formal metadata preflight",
    )
    intervention = _require_mapping(
        payload.get("intervention"), label="preflight intervention"
    )
    _require_exact_fields(
        intervention,
        {
            "rank_supervision": EXPECTED_RANK_SUPERVISION,
            "assignment_weight": 1.0,
            "deployment_weight": 1.0,
            "rank_weight": 0.0,
            "category_gate_max_gap": 3.0,
            "patch_score_clip": 5.0,
            "new_model_parameters": 0,
        },
        label="preflight intervention",
    )
    invariants = _require_mapping(
        payload.get("invariants"), label="preflight invariants"
    )
    required = {
        "formal_training_starts_fresh_from_a1",
        "u50_probe_checkpoints_are_not_resumed",
        "full_official_assignment_data_is_used",
        "patch_scores_are_detached_before_building_the_hardgap3_mask",
        "hardgap3_rank_loss_has_no_patch_gradient_path",
        "no_r100_p50_u2_or_teacher_checkpoint_tensor_is_loaded",
        "no_teacher_logits_probabilities_or_loss_targets_are_used",
        "training_code_must_match_the_sealed_u50_manifest",
    }
    if set(invariants) != required or any(
        invariants[key] is not True for key in required
    ):
        raise PairTop1HardGap3FormalAuditError("preflight invariants drifted")
    initializer = _require_mapping(
        payload.get("initializer"), label="preflight initializer"
    )
    _require_exact_fields(
        initializer,
        {
            "path": str(EXPECTED_INITIALIZER),
            "sha256": EXPECTED_INITIALIZER_SHA256,
            "only_checkpoint_tensor_source": "b58",
            "rank_patch_and_confidence_heads": "random_independent",
        },
        label="preflight initializer",
    )
    data = _require_mapping(
        payload.get("data_and_sampling"), label="preflight data and sampling"
    )
    _require_exact_fields(
        data,
        {
            "rows": EXPECTED_ROWS,
            "valid_assignment_rows": EXPECTED_VALID_ROWS,
            "weights_sha256": EXPECTED_WEIGHTS_SHA256,
            "ledger_sha256": EXPECTED_LEDGER_SHA256,
            "sampler_seed": 42,
            "loader_seed": 1042,
        },
        label="preflight data and sampling",
    )


def _expected_formal_evidence() -> dict[str, dict[str, str]]:
    return {
        "metadata_preflight": {
            "path": str(EXPECTED_PREFLIGHT),
            "sha256": EXPECTED_PREFLIGHT_SHA256,
        },
        "strict_u50_probe_receipt": {
            "path": str(EXPECTED_PROBE_RECEIPT),
            "sha256": EXPECTED_PROBE_RECEIPT_SHA256,
        },
        "formal_gate_contract": {
            "path": str(EXPECTED_GATE),
            "sha256": EXPECTED_GATE_SHA256,
        },
    }


def _checkpoint_path_occurrences(value: Any, trail: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], str]]:
    found: list[tuple[tuple[str, ...], str]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            found.extend(_checkpoint_path_occurrences(item, trail + (str(key),)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_checkpoint_path_occurrences(item, trail + (str(index),)))
    elif isinstance(value, str) and value.endswith((".pth", ".pt", ".ckpt")):
        found.append((trail, value))
    return found


def _audit_saved_args(args: Mapping[str, Any]) -> None:
    expected = {
        "config_file": EXPECTED_RELATIVE_CONFIG,
        "datasets": EXPECTED_RELATIVE_DATASET,
        "output_dir": EXPECTED_RELATIVE_OUTPUT,
        "pretrain_model_path": EXPECTED_RELATIVE_INITIALIZER,
        "options": {"batch_size": 64, "epochs": 1},
        "stage_b_data_driven_score": True,
        "stage_b_data_driven_experiment_id": "DD1",
        "stage_b_data_driven_variant_id": HARDGAP3_VARIANT,
        "stage_b_data_driven_train_mode": "rank_patch_only",
        "stage_b_data_driven_category_complete": True,
        "stage_b_data_driven_confidence_trained": False,
        "stage_b_data_driven_rank_supervision": EXPECTED_RANK_SUPERVISION,
        "stage_b_data_driven_rank_weight": 0.0,
        "stage_b_data_driven_assignment_weight": 1.0,
        "stage_b_data_driven_deployment_weight": 1.0,
        "stage_b_data_driven_patch_weight": 1.0,
        "stage_b_data_driven_strict_sample_identity": True,
        "stage_b_data_driven_category_gate": False,
        "stage_b_data_driven_category_gate_max_gap": 3.0,
        "stage_b_data_driven_patch_score_clip": 5.0,
        "stage_b_data_driven_assignment_dataset_scope": EXPECTED_SCOPE,
        "stage_b_data_driven_assignment_expected_rows": EXPECTED_ROWS,
        "stage_b_data_driven_assignment_expected_valid_rows": EXPECTED_VALID_ROWS,
        "stage_b_data_driven_assignment_dataset_config_sha256": (
            EXPECTED_DATASET_CONFIG_SHA256
        ),
        "stage_b_data_driven_assignment_receipt_sha256": (
            EXPECTED_ASSIGNMENT_RECEIPT_SHA256
        ),
        "stage_b_data_driven_assignment_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "stage_b_data_driven_base_initializer_sha256": EXPECTED_INITIALIZER_SHA256,
        "stage_b_data_driven_initializer_pair_receipt_sha256": (
            EXPECTED_INITIALIZER_PAIR_SHA256
        ),
        "stage_b_data_driven_no_teacher_contract": (
            "b58_only_random_independent_heads_v1"
        ),
        "stage_b_gdino_score_adapter": False,
        "stage_b_u0_patch_rank": False,
        "stage_b_v7": False,
        "stage_b_v11_fixed_text": False,
        "stage_b_legacy_global_gate": False,
        "stage_b_data_driven_sampling_contract": "deterministic_epoch_ledger_v1",
        "stage_b_data_driven_sampler_seed": 42,
        "stage_b_data_driven_loader_seed": 1042,
        "stage_b_data_driven_grad_clip_contract": "per_optimizer_branch_v1",
        "stage_b_data_driven_required_allocator_env": "PYTORCH_CUDA_ALLOC_CONF",
        "stage_b_data_driven_required_allocator_conf": "expandable_segments:True",
        "stage_b_data_driven_execution_scope": EXPECTED_EXECUTION_SCOPE,
        "stage_b_data_driven_formal_fresh_start": True,
        "stage_b_data_driven_formal_expected_optimizer_updates": EXPECTED_UPDATES,
        "stage_b_data_driven_formal_config_path": str(EXPECTED_FORMAL_CONFIG),
        "stage_b_data_driven_formal_output_dir": str(EXPECTED_OUTPUT_DIR),
        "stage_b_data_driven_formal_preflight_path": str(EXPECTED_PREFLIGHT),
        "stage_b_data_driven_formal_preflight_sha256": EXPECTED_PREFLIGHT_SHA256,
        "stage_b_data_driven_formal_probe_receipt_path": str(EXPECTED_PROBE_RECEIPT),
        "stage_b_data_driven_formal_probe_receipt_sha256": (
            EXPECTED_PROBE_RECEIPT_SHA256
        ),
        "stage_b_data_driven_formal_gate_contract_path": str(EXPECTED_GATE),
        "stage_b_data_driven_formal_gate_contract_sha256": EXPECTED_GATE_SHA256,
        "stage_b_data_driven_formal_probe_code_manifest_sha256": (
            EXPECTED_TRAINING_CODE_MANIFEST_SHA256
        ),
        "stage_b_data_driven_formal_observed_code_manifest_sha256": (
            EXPECTED_TRAINING_CODE_MANIFEST_SHA256
        ),
        "stage_b_data_driven_formal_evidence": _expected_formal_evidence(),
        "seed": 42,
        "start_epoch": 0,
        "resume": "",
        "max_train_iters": EXPECTED_UPDATES,
        "iter_checkpoint_interval": 500,
        "num_workers": 4,
        "prefetch_factor": 1,
        "persistent_workers": False,
        "gradient_accumulation_steps": 1,
        "amp": True,
        "amp_init_scale": 8192.0,
        "save_log": True,
        "world_size": 1,
        "distributed": False,
        "batch_size": 64,
        "epochs": 1,
        "fix_size": True,
        "strong_aug": False,
        "data_aug_hflip_prob": 0.0,
        "skip_eval": True,
    }
    _require_exact_fields(args, expected, label="formal checkpoint saved args")
    forbidden_route_keys = [
        key
        for key in args
        if any(token in key.lower() for token in ("teacher", "distill", "r100", "p50"))
        and key != "stage_b_data_driven_no_teacher_contract"
    ]
    if forbidden_route_keys:
        raise PairTop1HardGap3FormalAuditError(
            f"saved args expose forbidden teacher routes: {forbidden_route_keys}"
        )
    expected_checkpoint_occurrences = {
        (("pretrain_model_path",), EXPECTED_RELATIVE_INITIALIZER),
        (("stage_b_data_driven_base_initializer_path",), str(EXPECTED_INITIALIZER)),
        (
            ("stage_b_data_driven_base_initializer", "path"),
            str(EXPECTED_INITIALIZER),
        ),
    }
    observed_checkpoint_occurrences = set(_checkpoint_path_occurrences(args))
    if observed_checkpoint_occurrences != expected_checkpoint_occurrences:
        raise PairTop1HardGap3FormalAuditError(
            "saved args checkpoint sources are not exactly the fresh A1 initializer"
        )


def _audit_config_chain(args: Mapping[str, Any]) -> dict[str, Any]:
    chain = config_import_chain(EXPECTED_FORMAL_CONFIG, root=REPO_ROOT)
    observed = [
        stable_file_record(path, label=f"formal config dependency {path.name}")
        for path in chain
    ]
    compact = [
        {"path": item["path"], "sha256": item["sha256"]} for item in observed
    ]
    if args.get("stage_b_data_driven_config_import_chain") != compact:
        raise PairTop1HardGap3FormalAuditError(
            "saved formal config import chain does not match current files"
        )
    leaf = stable_file_record(EXPECTED_FORMAL_CONFIG, label="formal config leaf")
    if leaf["sha256"] != EXPECTED_FORMAL_CONFIG_SHA256:
        raise PairTop1HardGap3FormalAuditError("formal config leaf SHA drifted")
    return {"leaf": leaf, "dependency_count": len(observed)}


def _audit_code_manifest(
    args: Mapping[str, Any],
    *,
    expected_manifest_sha256: str = EXPECTED_TRAINING_CODE_MANIFEST_SHA256,
    expected_file_count: int = EXPECTED_CODE_FILE_COUNT,
) -> dict[str, Any]:
    provenance = _require_mapping(
        args.get("stage_b_data_driven_training_provenance"),
        label="saved training provenance",
    )
    if provenance.get("schema") != PROVENANCE_SCHEMA:
        raise PairTop1HardGap3FormalAuditError("training provenance schema drifted")
    code_files = provenance.get("code_files")
    if not isinstance(code_files, list) or len(code_files) != expected_file_count:
        raise PairTop1HardGap3FormalAuditError(
            "training code file coverage drifted: "
            f"expected={expected_file_count}, "
            f"observed={len(code_files) if isinstance(code_files, list) else None}"
        )
    if canonical_json_sha256(code_files) != expected_manifest_sha256:
        raise PairTop1HardGap3FormalAuditError(
            "saved training code manifest does not match the sealed U50 receipt"
        )
    seen: set[str] = set()
    for index, value in enumerate(code_files):
        record = _require_mapping(value, label=f"training code file {index}")
        if set(record) != {"path", "size_bytes", "sha256"}:
            raise PairTop1HardGap3FormalAuditError(
                f"training code file {index} schema drifted"
            )
        path = record.get("path")
        if not isinstance(path, str) or not Path(path).is_absolute() or path in seen:
            raise PairTop1HardGap3FormalAuditError(
                f"training code file {index} path is invalid or duplicated"
            )
        seen.add(path)
        current = stable_file_record(Path(path), label=f"training code file {index}")
        if dict(record) != current:
            raise PairTop1HardGap3FormalAuditError(
                f"training code file changed after launch: {path}"
            )
    return {
        "training_code_file_count": len(code_files),
        "training_code_files_manifest_sha256": expected_manifest_sha256,
        "current_files_match_saved_manifest": True,
    }


def _audit_terminal_checkpoint(payload: Mapping[str, Any]) -> None:
    if set(payload) != EXPECTED_CHECKPOINT_KEYS:
        raise PairTop1HardGap3FormalAuditError(
            "formal checkpoint top-level schema drifted"
        )
    for key in ("model", "criterion", "optimizer", "scaler", "args"):
        _require_mapping(payload.get(key), label=f"formal checkpoint.{key}")
    _require_exact_fields(
        payload,
        {
            "epoch": 0,
            "iteration": 0,
            "optimizer_updates": EXPECTED_UPDATES,
            "epoch_finished": True,
            "checkpoint_reason": "max_train_iters",
        },
        label="formal terminal checkpoint",
    )
    sampling = _require_mapping(
        payload.get("stage_b_data_driven_sampling_state"),
        label="formal sampling state",
    )
    if not _deep_equal(sampling, EXPECTED_SAMPLING_STATE):
        raise PairTop1HardGap3FormalAuditError("formal sampling state drifted")
    for key in ("rng_state", "epoch_rng_state"):
        state = _require_mapping(payload.get(key), label=key)
        if set(state) != {"python", "numpy", "torch", "cuda"}:
            raise PairTop1HardGap3FormalAuditError(f"{key} schema drifted")
        _audit_finite_tree(state, label=key)
    scheduler = _require_mapping(payload.get("lr_scheduler"), label="lr scheduler")
    if not _deep_equal(scheduler, EXPECTED_SCHEDULER):
        raise PairTop1HardGap3FormalAuditError(
            "epoch-boundary scheduler state drifted"
        )


def _audit_initializer_metadata(metadata: Mapping[str, Any]) -> dict[str, list[str]]:
    _require_exact_fields(
        metadata,
        {
            "schema": "pivot.stageb.data_driven_relational_initializer/v2",
            "seed": 42,
            "role_key_counts": EXPECTED_INITIALIZER_ROLE_COUNTS,
        },
        label="A1 initializer metadata",
    )
    invariants = _require_mapping(
        metadata.get("invariants"), label="A1 initializer invariants"
    )
    required = {
        "b58_is_only_checkpoint_source",
        "no_u1000_u5020_tensor_source",
        "no_teacher_adapter_tensor_source",
        "rank_and_confidence_parameters_are_disjoint",
        "patch_backbone_aliases_b58",
    }
    if any(invariants.get(key) is not True for key in required):
        raise PairTop1HardGap3FormalAuditError(
            "A1 initializer no-teacher/frozen-role invariants drifted"
        )
    role_keys = _require_mapping(metadata.get("role_keys"), label="initializer roles")
    if set(role_keys) != set(EXPECTED_INITIALIZER_ROLE_COUNTS):
        raise PairTop1HardGap3FormalAuditError("initializer role schema drifted")
    normalized: dict[str, list[str]] = {}
    for role, expected_count in EXPECTED_INITIALIZER_ROLE_COUNTS.items():
        values = role_keys.get(role)
        if (
            not isinstance(values, list)
            or len(values) != expected_count
            or len(set(values)) != expected_count
            or any(not isinstance(value, str) for value in values)
        ):
            raise PairTop1HardGap3FormalAuditError(
                f"initializer role {role!r} coverage drifted"
            )
        normalized[role] = list(values)
    return normalized


def _audit_model_against_initializer(
    initializer_state: Mapping[str, Any],
    trained_state: Mapping[str, Any],
    role_keys: Mapping[str, Sequence[str]],
    *,
    expected_model_tensor_count: int = EXPECTED_MODEL_TENSOR_COUNT,
) -> dict[str, Any]:
    if set(initializer_state) != set(trained_state):
        raise PairTop1HardGap3FormalAuditError(
            "formal model key coverage differs from A1 initializer"
        )
    if len(trained_state) != expected_model_tensor_count:
        raise PairTop1HardGap3FormalAuditError(
            "formal model tensor count drifted: "
            f"expected={expected_model_tensor_count}, observed={len(trained_state)}"
        )
    non_tensors = [
        key
        for key in trained_state
        if not torch.is_tensor(trained_state[key])
        or not torch.is_tensor(initializer_state[key])
    ]
    if non_tensors:
        raise PairTop1HardGap3FormalAuditError(
            f"model state contains non-tensors: {non_tensors[:8]}"
        )
    role_union: set[str] = set()
    for role, names in role_keys.items():
        overlap = role_union.intersection(names)
        if overlap:
            raise PairTop1HardGap3FormalAuditError(
                f"initializer roles overlap at {sorted(overlap)[:8]}"
            )
        role_union.update(names)
    if role_union != set(initializer_state):
        raise PairTop1HardGap3FormalAuditError(
            "initializer roles do not partition the full model state"
        )
    rank_all = set(role_keys["random_relational_rank"])
    if RANK_FIXED_BUFFER not in rank_all:
        raise PairTop1HardGap3FormalAuditError("rank fixed Fourier buffer is missing")
    rank_trainable = rank_all - {RANK_FIXED_BUFFER}
    patch_trainable = set(role_keys["random_patch_projection"])
    confidence = set(role_keys["random_absolute_confidence"])
    if (
        len(rank_trainable) != EXPECTED_RANK_TRAINABLE_COUNT
        or patch_trainable != set(PATCH_PARAMETER_NAMES)
        or len(patch_trainable) != EXPECTED_PATCH_PARAMETER_COUNT
        or any(
            not any(name.startswith(prefix) for prefix in CONFIDENCE_PREFIXES)
            for name in confidence
        )
    ):
        raise PairTop1HardGap3FormalAuditError(
            "rank, patch, or confidence initializer roles drifted"
        )
    trainable = rank_trainable | patch_trainable
    frozen = set(trained_state) - trainable
    unchanged_trainable = sorted(
        key
        for key in trainable
        if _tensor_bitwise_equal(initializer_state[key], trained_state[key])
    )
    if unchanged_trainable:
        raise PairTop1HardGap3FormalAuditError(
            "not every rank+patch trainable tensor changed from A1: "
            f"{unchanged_trainable[:8]}"
        )
    changed_frozen = sorted(
        key
        for key in frozen
        if not _tensor_bitwise_equal(initializer_state[key], trained_state[key])
    )
    if changed_frozen:
        raise PairTop1HardGap3FormalAuditError(
            "frozen b58/confidence/buffer tensor changed from A1: "
            f"{changed_frozen[:8]}"
        )
    return {
        "model_tensors": len(trained_state),
        "frozen_tensors_bitwise_equal_to_initializer": len(frozen),
        "b58_base_tensors_bitwise_equal": len(role_keys["b58_base"]),
        "shared_backbone_alias_tensors_bitwise_equal": len(
            role_keys["shared_backbone_alias"]
        ),
        "confidence_tensors_bitwise_equal": len(confidence),
        "rank_trainable_tensors_changed": len(rank_trainable),
        "patch_trainable_tensors_changed": len(patch_trainable),
        "rank_fixed_buffers_bitwise_equal": 1,
    }


def _as_exact_step(value: Any, *, label: str, expected_updates: int) -> int:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise PairTop1HardGap3FormalAuditError(f"{label} is not scalar")
        numeric = float(value.detach().cpu().item())
    elif type(value) in (int, float):
        numeric = float(value)
    else:
        raise PairTop1HardGap3FormalAuditError(f"{label} is not numeric")
    if not math.isfinite(numeric) or numeric != expected_updates:
        raise PairTop1HardGap3FormalAuditError(
            f"{label} must equal {expected_updates}, got {numeric}"
        )
    return int(numeric)


def _audit_optimizer_state(
    optimizer: Mapping[str, Any],
    model_state: Mapping[str, Any],
    *,
    expected_updates: int = EXPECTED_UPDATES,
    expected_rank_count: int = EXPECTED_RANK_TRAINABLE_COUNT,
    patch_parameter_names: Sequence[str] = PATCH_PARAMETER_NAMES,
) -> dict[str, Any]:
    groups = optimizer.get("param_groups")
    states = optimizer.get("state")
    if not isinstance(groups, list) or len(groups) != 2 or not isinstance(states, Mapping):
        raise PairTop1HardGap3FormalAuditError(
            "optimizer must contain exactly two groups and a state mapping"
        )
    if [group.get("stage_b_data_driven_branch") for group in groups] != [
        "rank",
        "patch",
    ]:
        raise PairTop1HardGap3FormalAuditError(
            "optimizer branch labels/order drifted"
        )
    rank_names = [
        key
        for key in model_state
        if key.startswith(RANK_PREFIX) and key != RANK_FIXED_BUFFER
    ]
    if len(rank_names) != expected_rank_count:
        raise PairTop1HardGap3FormalAuditError(
            "optimizer rank parameter-name coverage drifted"
        )
    if any(name not in model_state for name in patch_parameter_names):
        raise PairTop1HardGap3FormalAuditError(
            "optimizer patch parameter-name coverage drifted"
        )
    names_by_role = {"rank": rank_names, "patch": list(patch_parameter_names)}
    expected_lrs = {"rank": 3e-5, "patch": 3e-4}
    all_ids: list[Any] = []
    for group in groups:
        role = group["stage_b_data_driven_branch"]
        parameter_ids = group.get("params")
        if (
            not isinstance(parameter_ids, list)
            or len(parameter_ids) != len(names_by_role[role])
            or len(set(parameter_ids)) != len(parameter_ids)
        ):
            raise PairTop1HardGap3FormalAuditError(
                f"optimizer {role} parameter coverage drifted"
            )
        all_ids.extend(parameter_ids)
        for name, parameter_id in zip(names_by_role[role], parameter_ids):
            state = states.get(parameter_id)
            if not isinstance(state, Mapping) or set(state) != {
                "step",
                "exp_avg",
                "exp_avg_sq",
            }:
                raise PairTop1HardGap3FormalAuditError(
                    f"optimizer state schema drifted at {name}"
                )
            _as_exact_step(
                state["step"],
                label=f"optimizer.{role}.{name}.step",
                expected_updates=expected_updates,
            )
            for moment in ("exp_avg", "exp_avg_sq"):
                tensor = state[moment]
                parameter = model_state[name]
                if (
                    not torch.is_tensor(tensor)
                    or tuple(tensor.shape) != tuple(parameter.shape)
                    or tensor.dtype != parameter.dtype
                ):
                    raise PairTop1HardGap3FormalAuditError(
                        f"optimizer {moment} shape/dtype drifted at {name}"
                    )
        expected_static = {
            "lr": expected_lrs[role],
            "stage_b_data_driven_branch": role,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.0001,
            "amsgrad": False,
            "maximize": False,
            "foreach": None,
            "capturable": False,
            "differentiable": False,
            "fused": None,
            "decoupled_weight_decay": True,
            "initial_lr": expected_lrs[role],
        }
        observed_static = {key: value for key, value in group.items() if key != "params"}
        if not _deep_equal(observed_static, expected_static):
            raise PairTop1HardGap3FormalAuditError(
                f"optimizer {role} static options drifted"
            )
    if len(set(all_ids)) != len(all_ids) or set(states) != set(all_ids):
        raise PairTop1HardGap3FormalAuditError(
            "optimizer states do not exactly cover both branches"
        )
    tensor_count = _audit_finite_tree(optimizer, label="formal optimizer")
    expected_state_count = expected_rank_count + len(patch_parameter_names)
    if len(states) != expected_state_count or tensor_count != 3 * expected_state_count:
        raise PairTop1HardGap3FormalAuditError(
            "optimizer state/tensor count drifted"
        )
    return {
        "optimizer_parameter_states": len(states),
        "optimizer_state_tensors": tensor_count,
        "optimizer_state_step": expected_updates,
        "rank_optimizer_parameters": expected_rank_count,
        "patch_optimizer_parameters": len(patch_parameter_names),
        "rank_optimizer_lr": 3e-5,
        "patch_optimizer_lr": 3e-4,
    }


def _criterion_scalar(criterion: Mapping[str, Any], key: str) -> int:
    value = criterion.get(key)
    if not torch.is_tensor(value) or value.numel() != 1 or value.dtype != torch.int64:
        raise PairTop1HardGap3FormalAuditError(f"criterion.{key} tensor drifted")
    return int(value.detach().cpu().item())


def _audit_criterion_state(criterion: Mapping[str, Any]) -> int:
    expected_keys = {
        "fpr_positive_queue",
        "fpr_positive_queue_count",
        "fpr_positive_queue_cursor",
        "criterion_contract_version",
        "rank_supervision_contract_id",
    }
    if set(criterion) != expected_keys:
        raise PairTop1HardGap3FormalAuditError("criterion state schema drifted")
    queue = criterion["fpr_positive_queue"]
    if (
        not torch.is_tensor(queue)
        or queue.dtype != torch.float32
        or tuple(queue.shape) != (4096,)
        or bool(torch.count_nonzero(queue).item())
    ):
        raise PairTop1HardGap3FormalAuditError("criterion confidence queue drifted")
    expected = {
        "fpr_positive_queue_count": 0,
        "fpr_positive_queue_cursor": 0,
        "criterion_contract_version": EXPECTED_CRITERION_VERSION,
        "rank_supervision_contract_id": EXPECTED_RANK_SUPERVISION_ID,
    }
    for key, value in expected.items():
        if _criterion_scalar(criterion, key) != value:
            raise PairTop1HardGap3FormalAuditError(
                f"criterion.{key} must equal {value}"
            )
    tensor_count = _audit_finite_tree(criterion, label="formal criterion")
    if tensor_count != EXPECTED_CRITERION_TENSOR_COUNT:
        raise PairTop1HardGap3FormalAuditError("criterion tensor count drifted")
    return tensor_count


def _audit_scaler_state(
    scaler: Mapping[str, Any], *, expected: Mapping[str, Any] = EXPECTED_SCALER
) -> int:
    _audit_finite_tree(scaler, label="formal scaler")
    if not _deep_equal(scaler, expected):
        raise PairTop1HardGap3FormalAuditError(
            "formal scaler does not prove the zero-skip U5020 trajectory"
        )
    return 0


_LOG_METRIC_RE = re.compile(
    r"Epoch: \[0\]\s+\[\s*(?P<iteration>\d+)/5020\].*"
    r"amp_step_skipped:\s+(?P<instant>[-+0-9.eE]+)\s+"
    r"\((?P<average>[-+0-9.eE]+)\).*"
    r"optimizer_step:\s+(?P<step>[-+0-9.eE]+)\s+"
    r"\((?P<step_average>[-+0-9.eE]+)\)"
)
_TERMINAL_LOG_MESSAGE = (
    "Reached max_train_iters=5020 optimizer updates; saved epoch-boundary "
    "checkpoint after advancing the epoch scheduler."
)


def _audit_training_log(
    path: Path,
    *,
    expected_iterations: Sequence[int] | None = None,
    expected_terminal_message: str = _TERMINAL_LOG_MESSAGE,
) -> dict[str, Any]:
    record = stable_file_record(path, label="formal info log")
    try:
        text = Path(record["path"]).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise PairTop1HardGap3FormalAuditError(
            f"could not read formal info log: {error}"
        ) from error
    after = stable_file_record(Path(record["path"]), label="formal info log")
    if after != record:
        raise PairTop1HardGap3FormalAuditError(
            "formal info log changed while it was audited"
        )
    matches = [_LOG_METRIC_RE.search(line) for line in text.splitlines()]
    matches = [match for match in matches if match is not None]
    expected = (
        list(range(0, EXPECTED_UPDATES, 10))
        if expected_iterations is None
        else list(expected_iterations)
    )
    observed = [int(match.group("iteration")) for match in matches]
    if observed != expected:
        raise PairTop1HardGap3FormalAuditError(
            "formal AMP log anchors drifted: "
            f"expected={len(expected)}, observed={len(observed)}"
        )
    for match in matches:
        values = tuple(
            float(match.group(name))
            for name in ("instant", "average", "step", "step_average")
        )
        if not all(math.isfinite(value) for value in values) or values != (
            0.0,
            0.0,
            1.0,
            1.0,
        ):
            raise PairTop1HardGap3FormalAuditError(
                "formal log records an AMP skip or missing optimizer step"
            )
    if text.count(expected_terminal_message) != 1:
        raise PairTop1HardGap3FormalAuditError(
            "formal log does not contain exactly one epoch-boundary terminal marker"
        )
    command_lines = [line for line in text.splitlines() if " | Command: " in line]
    if len(command_lines) != 1 or " --resume " in command_lines[0]:
        raise PairTop1HardGap3FormalAuditError(
            "formal log does not prove one fresh, non-resumed launch"
        )
    required_command_tokens = (
        f"-c {EXPECTED_RELATIVE_CONFIG}",
        f"--datasets {EXPECTED_RELATIVE_DATASET}",
        f"--output_dir {EXPECTED_RELATIVE_OUTPUT}",
        f"--pretrain_model_path {EXPECTED_RELATIVE_INITIALIZER}",
        "--seed 42",
        "--max_train_iters 5020",
        "--iter_checkpoint_interval 500",
        "--num_workers 4",
        "--prefetch_factor 1",
        "--no_persistent_workers",
        "--gradient_accumulation_steps 1",
        "--amp",
        "--save_log",
        "--options batch_size=64 epochs=1",
    )
    missing = [token for token in required_command_tokens if token not in command_lines[0]]
    if missing:
        raise PairTop1HardGap3FormalAuditError(
            f"formal launch command drifted; missing {missing}"
        )
    return {
        "info_log": record,
        "amp_log_anchors": len(matches),
        "max_amp_step_skipped": 0.0,
        "min_optimizer_step": 1.0,
    }


def _build_receipt(
    *,
    formal_checkpoint: Path,
    sealed_u50_probe_receipt: Path,
    formal_gate_contract: Path,
) -> dict[str, Any]:
    probe, probe_record = _require_input_record(
        sealed_u50_probe_receipt,
        expected_path=EXPECTED_PROBE_RECEIPT,
        expected_sha256=EXPECTED_PROBE_RECEIPT_SHA256,
        label="sealed U50 probe receipt",
    )
    gate, gate_record = _require_input_record(
        formal_gate_contract,
        expected_path=EXPECTED_GATE,
        expected_sha256=EXPECTED_GATE_SHA256,
        label="formal gate contract",
    )
    _audit_probe_receipt(probe)
    gate_training = _audit_gate(gate)
    preflight, preflight_record = _require_input_record(
        EXPECTED_PREFLIGHT,
        expected_path=EXPECTED_PREFLIGHT,
        expected_sha256=EXPECTED_PREFLIGHT_SHA256,
        label="formal metadata preflight",
    )
    _audit_preflight(preflight)

    try:
        checkpoint_path = formal_checkpoint.expanduser().resolve(strict=True)
        expected_checkpoint_path = (EXPECTED_OUTPUT_DIR / "checkpoint_iter.pth").resolve(
            strict=True
        )
    except OSError as error:
        raise PairTop1HardGap3FormalAuditError(
            f"could not resolve formal checkpoint: {error}"
        ) from error
    if checkpoint_path != expected_checkpoint_path:
        raise PairTop1HardGap3FormalAuditError("formal checkpoint path drifted")
    checkpoint_record = stable_file_record(
        checkpoint_path, label="formal U5020 checkpoint"
    )
    payload = _safe_load_checkpoint(checkpoint_path, label="formal U5020 checkpoint")
    _audit_terminal_checkpoint(payload)
    args = _require_mapping(payload["args"], label="formal checkpoint saved args")
    _audit_saved_args(args)
    config_audit = _audit_config_chain(args)
    code_audit = _audit_code_manifest(args)
    try:
        bindings = _validate_data_and_initializer_bindings(args, role="hardgap3")
    except PairTop1HardGap3ProbeAuditError as error:
        raise PairTop1HardGap3FormalAuditError(str(error)) from error

    initializer_record = stable_file_record(EXPECTED_INITIALIZER, label="A1 initializer")
    initializer_payload = _safe_load_checkpoint(EXPECTED_INITIALIZER, label="A1 initializer")
    if set(initializer_payload) != {"model", "data_driven_relational_initializer"}:
        raise PairTop1HardGap3FormalAuditError("A1 initializer schema drifted")
    initializer_metadata = _require_mapping(
        initializer_payload["data_driven_relational_initializer"],
        label="A1 initializer metadata",
    )
    role_keys = _audit_initializer_metadata(initializer_metadata)
    initializer_model = _require_mapping(
        initializer_payload["model"], label="A1 initializer model"
    )
    trained_model = _require_mapping(payload["model"], label="formal model")
    model_tensor_count = _audit_finite_tree(trained_model, label="formal model")
    initializer_tensor_count = _audit_finite_tree(
        initializer_model, label="A1 initializer model"
    )
    if (
        model_tensor_count != EXPECTED_MODEL_TENSOR_COUNT
        or initializer_tensor_count != EXPECTED_MODEL_TENSOR_COUNT
    ):
        raise PairTop1HardGap3FormalAuditError("model finite-tensor coverage drifted")
    model_audit = _audit_model_against_initializer(
        initializer_model, trained_model, role_keys
    )

    optimizer = _require_mapping(payload["optimizer"], label="formal optimizer")
    criterion = _require_mapping(payload["criterion"], label="formal criterion")
    scaler = _require_mapping(payload["scaler"], label="formal scaler")
    optimizer_audit = _audit_optimizer_state(optimizer, trained_model)
    criterion_tensor_count = _audit_criterion_state(criterion)
    scaler_tensor_count = _audit_scaler_state(scaler)
    log_audit = _audit_training_log(EXPECTED_OUTPUT_DIR / "info.txt")

    checkpoint_after = stable_file_record(
        checkpoint_path, label="formal U5020 checkpoint"
    )
    if checkpoint_after != checkpoint_record:
        raise PairTop1HardGap3FormalAuditError(
            "formal checkpoint changed while it was audited"
        )
    if initializer_record["sha256"] != EXPECTED_INITIALIZER_SHA256:
        raise PairTop1HardGap3FormalAuditError("A1 initializer SHA drifted")
    if gate_training["output_dir"] != str(checkpoint_path.parent):
        raise PairTop1HardGap3FormalAuditError(
            "formal gate output directory does not own the checkpoint"
        )

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "passed",
        "phase": "D1",
        "variant": HARDGAP3_VARIANT,
        "scope": "fresh_formal_training_terminal_state",
        "execution_scope": EXPECTED_EXECUTION_SCOPE,
        "training": {
            "checkpoint": checkpoint_record,
            "config": config_audit["leaf"],
            "initializer": initializer_record,
            "fresh_start": True,
            "resume": "",
            "seed": 42,
            "batch_size": 64,
            "optimizer_updates": EXPECTED_UPDATES,
            "num_workers": 4,
            "prefetch_factor": 1,
            "gradient_accumulation_steps": 1,
            "amp": True,
            "amp_init_scale": 8192.0,
        },
        "terminal_checkpoint": {
            "epoch": 0,
            "iteration": 0,
            "optimizer_updates": EXPECTED_UPDATES,
            "checkpoint_reason": "max_train_iters",
            "epoch_finished": True,
            "scheduler_last_epoch": 1,
        },
        "evidence": {
            "sealed_u50_probe_receipt": probe_record,
            "formal_gate_contract": gate_record,
            "metadata_preflight": preflight_record,
            **code_audit,
        },
        "data": {
            key: bindings[key]
            for key in (
                "scope",
                "rows",
                "valid_rows",
                "dataset_config",
                "assignment_receipt",
                "manifests",
            )
        },
        "criterion": {
            "criterion_contract_version": EXPECTED_CRITERION_VERSION,
            "rank_supervision_contract_id": EXPECTED_RANK_SUPERVISION_ID,
            "rank_supervision": EXPECTED_RANK_SUPERVISION,
            "assignment_weight": 1.0,
            "deployment_weight": 1.0,
        },
        "state_audit": {
            **model_audit,
            **optimizer_audit,
            "criterion_tensors": criterion_tensor_count,
            "scaler_tensors": scaler_tensor_count,
            "amp_scale": scaler["scale"],
            "amp_growth_tracker": scaler["_growth_tracker"],
            "amp_log_anchors": log_audit["amp_log_anchors"],
            "amp_step_skips": 0,
        },
        "artifacts": {
            "info_log": log_audit["info_log"],
            "config_dependency_count": config_audit["dependency_count"],
        },
        "invariants": {
            "all_5020_optimizer_updates_succeeded": True,
            "all_optimizer_states_are_step_5020": True,
            "amp_step_skips_zero": True,
            "all_model_optimizer_criterion_scaler_tensors_finite": True,
            "frozen_b58_and_confidence_equal_initializer": True,
            "all_rank_and_patch_trainable_tensors_changed": True,
            "training_code_matches_u50_and_current_files": True,
            "full_official_assignment_data_and_sampling_are_bound": True,
            "fresh_a1_start_without_resume": True,
            "no_teacher_checkpoint_logits_weights_or_loss_targets": True,
            "formal_checkpoint_is_terminal_and_must_not_be_resumed": True,
        },
        "decision": (
            "eligible_for_the_sealed_fixed_gap3_refcocog_val_headline_evaluation; "
            "training success alone does not establish the accuracy gate"
        ),
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    return receipt


def build_receipt(
    *,
    formal_checkpoint: Path,
    sealed_u50_probe_receipt: Path,
    formal_gate_contract: Path,
) -> dict[str, Any]:
    try:
        return _build_receipt(
            formal_checkpoint=formal_checkpoint,
            sealed_u50_probe_receipt=sealed_u50_probe_receipt,
            formal_gate_contract=formal_gate_contract,
        )
    except PairTop1HardGap3FormalAuditError:
        raise
    except PairTop1HardGap3ProbeAuditError as error:
        raise PairTop1HardGap3FormalAuditError(str(error)) from error


def _publish_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        _atomic_publish_fresh_json(path, payload)
    except FileExistsError as error:
        raise FileExistsError(
            f"formal training receipt output must be fresh: {path.resolve()}"
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-checkpoint", type=Path, required=True)
    parser.add_argument("--sealed-u50-probe-receipt", type=Path, required=True)
    parser.add_argument("--formal-gate-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt(
        formal_checkpoint=args.formal_checkpoint,
        sealed_u50_probe_receipt=args.sealed_u50_probe_receipt,
        formal_gate_contract=args.formal_gate_contract,
    )
    _publish_receipt(args.output, receipt)
    output_record = stable_file_record(args.output, label="formal training receipt")
    print(
        f"[OK] wrote {output_record['path']} size={output_record['size_bytes']} "
        f"sha256={output_record['sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
