#!/usr/bin/env python3
"""Preregister the sealed three-way LR probe for the new Stage-B rank head.

The artifact is deliberately created before any candidate output exists.  It
binds the training inputs, the three thin config leaves, the shared dev-screen
evaluation inputs, and the deterministic selection rule.  Publication is
atomic and create-new: an existing or concurrently-created destination is
never replaced.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pivot.stageb.data_driven.new_head_lr_preregistration/v1"
PARTITION_SCHEMA = "pivot.stageb.data_driven.new_head_partition_receipt/v1"
SUPPORT_SCHEMA = "pivot.stageb.data_driven.support_partition_receipt/v1"
PAIR_SCHEMA = "pivot.stageb.data_driven_initializer_pair/v1"

CANDIDATE_RANK_LRS = (3e-5, 1e-4, 3e-4)
OPTIMIZER_UPDATES_PER_CANDIDATE = 1000
SELECTION_PARTITION = "dev_screen"
SELECTION_METRIC = "macro_ref3_acc50"
SECONDARY_SELECTION_METRIC = "macro_ref3_mean_listwise_nll"
TIE_BREAK_RULE = (
    "max_macro_ref3_acc50",
    "min_macro_ref3_mean_listwise_nll",
    "min_rank_lr",
)
EXECUTION_SCOPE = "fresh_a0_new_head_lr_probe_u1000_v1"
EXECUTION_CONTRACT = "diagnostic_lr_probe_u1000_v1"
PROBE_BASE_MODULE = (
    "config.ablations.cfg_stageb_data_driven_dd1_new_head_lr_probe_20260723"
)
SOURCE_MANIFESTS = (
    "refcoco_stageb_phrase_v1.jsonl",
    "refcocoplus_stageb_phrase_v1.jsonl",
    "refcocog_stageb_phrase_v1.jsonl",
)

EXPECTED_DATASET_SHA256 = (
    "76de77705b897bcd3d6bf6fa4cc2a6baa82b7499812f13fe316572df2e194b77"
)
EXPECTED_PARTITION_SHA256 = (
    "56de31d883ed137f3f9332c34de846839d82c0724120f42a49c5c1c302f38506"
)
EXPECTED_SUPPORT_SHA256 = (
    "a0e6632182bc7c01ac6e6997b15f1f96e0fbb0bf6dd9d1e3fd8485ad39a6da62"
)
EXPECTED_A0_SHA256 = (
    "c2c4ba71656054d3afc3d219ca2f6d56839396d6258bac0201878566b1937034"
)
EXPECTED_PAIR_SHA256 = (
    "e304d2e8439f5714facf1b510795ba3a9874ec456433110afeef91f2d1dc7d8d"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data/ablations/stageb_data_driven_new_head_lr_selection_20260723/"
    "preregistration.json"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class NewHeadLRPreregistrationError(RuntimeError):
    """One or more preregistered inputs or semantics failed closed."""


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    label: str
    rank_lr: float
    config_path: Path
    training_output_dir: Path
    dev_screen_eval_dir: Path


@dataclass(frozen=True, slots=True)
class PreregistrationSpec:
    repo_root: Path
    dataset_path: Path
    dataset_sha256: str
    partition_receipt_path: Path
    partition_receipt_sha256: str
    support_receipt_path: Path
    support_receipt_sha256: str
    a0_initializer_path: Path
    a0_initializer_sha256: str
    pair_receipt_path: Path
    pair_receipt_sha256: str
    main_path: Path
    evaluator_path: Path
    selector_path: Path
    candidates: tuple[CandidateSpec, ...]
    probe_base_module: str = PROBE_BASE_MODULE


def default_spec(repo_root: Path = REPO_ROOT) -> PreregistrationSpec:
    root = repo_root.expanduser().resolve(strict=True)
    candidate_root = (
        root
        / "outputs/paper_cvpr_v1/data_driven_new_head_20260723/lr_probe_u1000"
    )
    candidate_values = (
        (
            "lr3e5",
            3e-5,
            "cfg_stageb_data_driven_dd1_new_head_lr3e5_probe_u1000_20260723.py",
        ),
        (
            "lr1e4",
            1e-4,
            "cfg_stageb_data_driven_dd1_new_head_lr1e4_probe_u1000_20260723.py",
        ),
        (
            "lr3e4",
            3e-4,
            "cfg_stageb_data_driven_dd1_new_head_lr3e4_probe_u1000_20260723.py",
        ),
    )
    candidates = tuple(
        CandidateSpec(
            label=label,
            rank_lr=rank_lr,
            config_path=root / "config/ablations" / filename,
            training_output_dir=candidate_root / label,
            dev_screen_eval_dir=(
                candidate_root / label / "evaluations/new_head_dev_screen"
            ),
        )
        for label, rank_lr, filename in candidate_values
    )
    initializer_root = (
        root
        / "outputs/paper_cvpr_v1/data_driven_initializers/fair_v2_seed42"
    )
    return PreregistrationSpec(
        repo_root=root,
        dataset_path=(
            root
            / "config/datasets_stageb_data_driven_dd1_new_head_train_20260723.json"
        ),
        dataset_sha256=EXPECTED_DATASET_SHA256,
        partition_receipt_path=(
            root
            / "data/ablations/stageb_data_driven_new_head_partition_20260723/"
            "receipt.json"
        ),
        partition_receipt_sha256=EXPECTED_PARTITION_SHA256,
        support_receipt_path=(
            root
            / "data/ablations/stageb_data_driven_support_partition_20260723/"
            "receipt.json"
        ),
        support_receipt_sha256=EXPECTED_SUPPORT_SHA256,
        a0_initializer_path=(
            initializer_root / "checkpoint_dd_a0_absolute_v2_init.pth"
        ),
        a0_initializer_sha256=EXPECTED_A0_SHA256,
        pair_receipt_path=initializer_root / "a0_a1_v2_pair_receipt.json",
        pair_receipt_sha256=EXPECTED_PAIR_SHA256,
        main_path=root / "main.py",
        evaluator_path=root / "tools/eval_stageb_data_driven_new_head_dev.py",
        selector_path=root / "tools/select_stageb_data_driven_new_head_lr.py",
        candidates=candidates,
    )


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise NewHeadLRPreregistrationError(
            f"value is not canonical JSON: {error}"
        ) from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _file_record(path: Path, *, label: str) -> dict[str, Any]:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise NewHeadLRPreregistrationError(
            f"could not resolve {label}: {path}: {error}"
        ) from error
    if not resolved.is_file():
        raise NewHeadLRPreregistrationError(f"{label} is not a file: {resolved}")
    try:
        before = resolved.stat()
        digest = _sha256_file(resolved)
        after = resolved.stat()
    except OSError as error:
        raise NewHeadLRPreregistrationError(
            f"could not hash {label}: {resolved}: {error}"
        ) from error
    if _stat_identity(before) != _stat_identity(after):
        raise NewHeadLRPreregistrationError(
            f"{label} changed while it was hashed: {resolved}"
        )
    return {
        "path": str(resolved),
        "size_bytes": int(after.st_size),
        "sha256": digest,
    }


def _require_sha(record: Mapping[str, Any], expected: str, *, label: str) -> None:
    if _SHA256_RE.fullmatch(str(expected)) is None:
        raise NewHeadLRPreregistrationError(
            f"{label} expected SHA256 is not lowercase hexadecimal"
        )
    if record.get("sha256") != expected:
        raise NewHeadLRPreregistrationError(
            f"{label} SHA256 drifted: expected={expected}, "
            f"observed={record.get('sha256')}"
        )


def _strict_json_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise NewHeadLRPreregistrationError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise NewHeadLRPreregistrationError(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _load_json_mapping(
    path: Path, *, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _file_record(path, label=label)
    try:
        value = json.loads(
            Path(record["path"]).read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except NewHeadLRPreregistrationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NewHeadLRPreregistrationError(
            f"could not parse {label}: {record['path']}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise NewHeadLRPreregistrationError(f"{label} must be a JSON mapping")
    if _file_record(Path(record["path"]), label=label) != record:
        raise NewHeadLRPreregistrationError(f"{label} changed while it was read")
    return value, record


def _require_canonical_receipt(
    value: Mapping[str, Any], *, label: str
) -> str:
    declared = value.get("canonical_payload_sha256")
    if not isinstance(declared, str) or _SHA256_RE.fullmatch(declared) is None:
        raise NewHeadLRPreregistrationError(
            f"{label} has no lowercase canonical payload SHA256"
        )
    payload = dict(value)
    del payload["canonical_payload_sha256"]
    observed = _sha256_bytes(_canonical_bytes(payload))
    if declared != observed:
        raise NewHeadLRPreregistrationError(
            f"{label} canonical payload SHA256 drifted"
        )
    return declared


def _require_all_true_invariants(value: Any, *, label: str) -> dict[str, bool]:
    if not isinstance(value, Mapping) or not value:
        raise NewHeadLRPreregistrationError(f"{label} invariants are missing")
    invariants = dict(value)
    if any(item is not True for item in invariants.values()):
        raise NewHeadLRPreregistrationError(
            f"{label} invariants are not all exactly true"
        )
    return invariants


def _declared_file_matches(
    value: Any, observed: Mapping[str, Any], *, label: str
) -> None:
    if not isinstance(value, Mapping):
        raise NewHeadLRPreregistrationError(f"{label} file record is missing")
    for key in ("path", "size_bytes", "sha256"):
        if value.get(key) != observed.get(key):
            raise NewHeadLRPreregistrationError(
                f"{label} file record drifted at {key}"
            )


def _resolve_existing_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise NewHeadLRPreregistrationError(f"{label} path is missing")
    try:
        return Path(value).expanduser().resolve(strict=True)
    except OSError as error:
        raise NewHeadLRPreregistrationError(
            f"could not resolve {label}: {value}: {error}"
        ) from error


def _same_path(value: Any, expected: Path, *, label: str) -> None:
    observed = _resolve_existing_path(value, label=label)
    if observed != expected.expanduser().resolve(strict=True):
        raise NewHeadLRPreregistrationError(
            f"{label} path drifted: {observed}"
        )


def _module_path(module: str, *, repo_root: Path) -> Path:
    if not module.startswith("config."):
        raise NewHeadLRPreregistrationError(
            f"config import is not repository-local: {module!r}"
        )
    path = repo_root.joinpath(*module.split(".")).with_suffix(".py")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(repo_root)
    except (OSError, ValueError) as error:
        raise NewHeadLRPreregistrationError(
            f"could not resolve config import {module!r}: {error}"
        ) from error
    if not resolved.is_file():
        raise NewHeadLRPreregistrationError(
            f"config import is not a file: {resolved}"
        )
    return resolved


def _parse_config_ast(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise NewHeadLRPreregistrationError(
            f"could not parse config {path}: {error}"
        ) from error


def _load_literal_config(
    path: Path,
    *,
    repo_root: Path,
    cache: dict[Path, tuple[dict[str, Any], tuple[Path, ...]]],
    active: set[Path],
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    try:
        resolved = path.expanduser().resolve(strict=True)
        resolved.relative_to(repo_root)
    except (OSError, ValueError) as error:
        raise NewHeadLRPreregistrationError(
            f"config escapes repository root: {path}: {error}"
        ) from error
    if resolved in cache:
        values, chain = cache[resolved]
        return dict(values), chain
    if resolved in active:
        raise NewHeadLRPreregistrationError(
            f"cyclic config import detected: {resolved}"
        )
    active.add(resolved)
    values: dict[str, Any] = {}
    chain: list[Path] = []
    for node in _parse_config_ast(resolved).body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(
            node.value.value, str
        ):
            continue
        if isinstance(node, ast.ImportFrom):
            if (
                node.level != 0
                or not node.module
                or len(node.names) != 1
                or node.names[0].name != "*"
            ):
                raise NewHeadLRPreregistrationError(
                    f"config uses a non-literal import contract: {resolved}:{node.lineno}"
                )
            imported_path = _module_path(node.module, repo_root=repo_root)
            imported, imported_chain = _load_literal_config(
                imported_path,
                repo_root=repo_root,
                cache=cache,
                active=active,
            )
            values.update(imported)
            for item in imported_chain:
                if item not in chain:
                    chain.append(item)
            continue
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise NewHeadLRPreregistrationError(
                    f"config assignment is not one literal name: "
                    f"{resolved}:{node.lineno}"
                )
            name = node.targets[0].id
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError) as error:
                raise NewHeadLRPreregistrationError(
                    f"config assignment is not literal: {resolved}:{node.lineno}"
                ) from error
            values[name] = value
            continue
        raise NewHeadLRPreregistrationError(
            f"unsupported executable config syntax: "
            f"{resolved}:{getattr(node, 'lineno', '?')}:{type(node).__name__}"
        )
    active.remove(resolved)
    chain.append(resolved)
    result = (dict(values), tuple(chain))
    cache[resolved] = result
    return dict(values), tuple(chain)


def _validate_thin_probe_leaf(
    path: Path,
    *,
    expected_rank_lr: float,
    expected_base_module: str,
) -> None:
    meaningful: list[ast.stmt] = []
    for node in _parse_config_ast(path).body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(
            node.value.value, str
        ):
            continue
        meaningful.append(node)
    if len(meaningful) != 2:
        raise NewHeadLRPreregistrationError(
            f"thin probe config must contain only one import and rank_lr: {path}"
        )
    imported, assignment = meaningful
    if not (
        isinstance(imported, ast.ImportFrom)
        and imported.level == 0
        and imported.module == expected_base_module
        and len(imported.names) == 1
        and imported.names[0].name == "*"
    ):
        raise NewHeadLRPreregistrationError(
            f"thin probe config imports the wrong shared leaf: {path}"
        )
    if not (
        isinstance(assignment, ast.Assign)
        and len(assignment.targets) == 1
        and isinstance(assignment.targets[0], ast.Name)
        and assignment.targets[0].id == "stage_b_data_driven_rank_lr"
    ):
        raise NewHeadLRPreregistrationError(
            f"thin probe config has a non-rank-LR override: {path}"
        )
    try:
        observed = ast.literal_eval(assignment.value)
    except (ValueError, TypeError) as error:
        raise NewHeadLRPreregistrationError(
            f"thin probe rank_lr is not literal: {path}"
        ) from error
    if type(observed) is not float or observed != expected_rank_lr:
        raise NewHeadLRPreregistrationError(
            f"thin probe rank_lr drifted: {path}: {observed!r}"
        )


def _exact_value(observed: Any, expected: Any) -> bool:
    return type(observed) is type(expected) and observed == expected


def _required_config_values(spec: PreregistrationSpec) -> dict[str, Any]:
    return {
        "stage_b_data_driven_score": True,
        "stage_b_data_driven_experiment_id": "DD1",
        "stage_b_data_driven_variant_id": "DD1-NEWHEAD-LR-PROBE",
        "stage_b_data_driven_category_complete": True,
        "stage_b_data_driven_execution_scope": EXECUTION_SCOPE,
        "stage_b_data_driven_new_head_formal_contract": EXECUTION_CONTRACT,
        "stage_b_data_driven_formal_fresh_start": False,
        "stage_b_data_driven_formal_expected_optimizer_updates": 1000,
        "stage_b_data_driven_formal_config_path": "",
        "stage_b_data_driven_formal_output_dir": "",
        "stage_b_data_driven_new_head_dataset_config_path": str(
            spec.dataset_path.expanduser().resolve(strict=True)
        ),
        "stage_b_data_driven_new_head_dataset_config_sha256": (
            spec.dataset_sha256
        ),
        "stage_b_data_driven_base_initializer_path": str(
            spec.a0_initializer_path.expanduser().resolve(strict=True)
        ),
        "stage_b_data_driven_base_initializer_sha256": (
            spec.a0_initializer_sha256
        ),
        "stage_b_data_driven_initializer_pair_receipt_path": str(
            spec.pair_receipt_path.expanduser().resolve(strict=True)
        ),
        "stage_b_data_driven_initializer_pair_receipt_sha256": (
            spec.pair_receipt_sha256
        ),
        "stage_b_data_driven_new_head_partition_receipt_path": str(
            spec.partition_receipt_path.expanduser().resolve(strict=True)
        ),
        "stage_b_data_driven_new_head_partition_receipt_sha256": (
            spec.partition_receipt_sha256
        ),
        "stage_b_data_driven_new_head_support_receipt_path": str(
            spec.support_receipt_path.expanduser().resolve(strict=True)
        ),
        "stage_b_data_driven_new_head_support_receipt_sha256": (
            spec.support_receipt_sha256
        ),
        "stage_b_data_driven_train_mode": "rank_patch_only",
        "stage_b_data_driven_confidence_trained": False,
        "stage_b_data_driven_rank_architecture": "absolute_token",
        "stage_b_data_driven_rank_supervision": "all_nonpositive_negative_v1",
        "stage_b_data_driven_strict_sample_identity": True,
        "stage_b_data_driven_rank_weight": 1.0,
        "stage_b_data_driven_patch_weight": 1.0,
        "stage_b_data_driven_patch_lr": 3e-4,
        "stage_b_data_driven_sampling_contract": (
            "deterministic_epoch_ledger_v1"
        ),
        "stage_b_data_driven_sampler_seed": 42,
        "stage_b_data_driven_loader_seed": 1042,
        "stage_b_data_driven_grad_clip_contract": "per_optimizer_branch_v1",
        "stage_b_data_driven_required_allocator_env": (
            "PYTORCH_CUDA_ALLOC_CONF"
        ),
        "stage_b_data_driven_required_allocator_conf": (
            "expandable_segments:True"
        ),
        "stage_b": False,
        "stage_b_gdino_score_adapter": False,
        "stage_b_u0_patch_rank": False,
        "stage_b_v7": False,
        "stage_b_v11_fixed_text": False,
        "stage_b_legacy_global_gate": False,
        "patch_only": False,
        "enable_patch_branch": True,
        "patch_gate_with_text": False,
        "stage_b_data_driven_category_gate": False,
        "batch_size": 64,
        "epochs": 1,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "clip_max_norm": 0.1,
        "lr_drop": 100,
        "onecyclelr": False,
        "multi_step_lr": False,
        "lr_drop_list": [4, 8],
        "fix_size": True,
        "strong_aug": False,
        "data_aug_hflip_prob": 0.0,
        "aux_loss": False,
        "use_checkpoint": False,
        "use_transformer_ckpt": False,
        "persistent_workers": False,
        "amp_init_scale": 8192.0,
        "save_checkpoint_interval": 1,
        "skip_eval": True,
        "use_coco_eval": False,
    }


_PAIRED_TRAINING_KEYS = (
    "stage_b_data_driven_train_mode",
    "stage_b_data_driven_rank_architecture",
    "stage_b_data_driven_rank_dim",
    "stage_b_data_driven_rank_num_heads",
    "stage_b_data_driven_rank_image_level_policy",
    "stage_b_data_driven_rank_image_levels",
    "stage_b_data_driven_rank_image_pool_size",
    "stage_b_data_driven_rank_image_pool_policy",
    "stage_b_data_driven_rank_box_fourier_bands",
    "stage_b_data_driven_rank_ffn_dim",
    "stage_b_data_driven_rank_dropout",
    "stage_b_data_driven_head_init_seed",
    "stage_b_data_driven_rank_supervision",
    "stage_b_data_driven_rank_negative_iou_threshold",
    "stage_b_data_driven_rank_weight",
    "stage_b_data_driven_patch_weight",
    "stage_b_data_driven_positive_iou_threshold",
    "stage_b_data_driven_patch_negative_iou_threshold",
    "stage_b_data_driven_temperature",
    "stage_b_data_driven_category_margin",
    "stage_b_data_driven_rank_lr",
    "stage_b_data_driven_patch_lr",
    "stage_b_data_driven_sampling_contract",
    "stage_b_data_driven_sampler_seed",
    "stage_b_data_driven_loader_seed",
    "stage_b_data_driven_grad_clip_contract",
    "batch_size",
    "epochs",
    "max_train_iters",
    "gradient_accumulation_steps",
    "weight_decay",
    "clip_max_norm",
    "lr_drop",
    "onecyclelr",
    "multi_step_lr",
    "lr_drop_list",
    "num_workers",
    "prefetch_factor",
    "pin_memory",
    "persistent_workers",
    "stage_b_data_driven_required_allocator_env",
    "stage_b_data_driven_required_allocator_conf",
    "stage_b_data_driven_execution_scope",
    "stage_b_data_driven_formal_fresh_start",
    "stage_b_data_driven_formal_expected_optimizer_updates",
    "stage_b_data_driven_new_head_formal_contract",
    "seed",
    "amp",
    "fix_size",
    "strong_aug",
    "data_aug_hflip_prob",
    "aux_loss",
    "use_checkpoint",
    "use_transformer_ckpt",
)


def _effective_training_contract(
    resolved: Mapping[str, Any], *, rank_lr: float
) -> dict[str, Any]:
    effective = dict(resolved)
    effective.update(
        {
            "stage_b_data_driven_rank_lr": rank_lr,
            "max_train_iters": 1000,
            "gradient_accumulation_steps": 1,
            "num_workers": 4,
            "prefetch_factor": 1,
            "pin_memory": True,
            "persistent_workers": False,
            "seed": 42,
            "amp": True,
        }
    )
    missing = [key for key in _PAIRED_TRAINING_KEYS if key not in effective]
    if missing:
        raise NewHeadLRPreregistrationError(
            f"resolved probe config lacks training semantics: {missing}"
        )
    contract = {key: effective[key] for key in _PAIRED_TRAINING_KEYS}
    if type(contract["stage_b_data_driven_rank_lr"]) is not float:
        raise NewHeadLRPreregistrationError("rank_lr must remain one exact float")
    return contract


def _receipt_manifest_records(
    value: Any,
    *,
    label: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != set(SOURCE_MANIFESTS):
        raise NewHeadLRPreregistrationError(
            f"{label} manifest set/order contract drifted"
        )
    records = []
    for manifest_name in SOURCE_MANIFESTS:
        declared = value.get(manifest_name)
        if not isinstance(declared, Mapping):
            raise NewHeadLRPreregistrationError(
                f"{label} record is missing: {manifest_name}"
            )
        rows = declared.get("rows")
        if type(rows) is not int or rows <= 0:
            raise NewHeadLRPreregistrationError(
                f"{label} row count is invalid: {manifest_name}"
            )
        path = _resolve_existing_path(
            declared.get("path"), label=f"{label} {manifest_name}"
        )
        observed = _file_record(path, label=f"{label} {manifest_name}")
        _declared_file_matches(
            declared,
            observed,
            label=f"{label} {manifest_name}",
        )
        records.append(
            {
                **observed,
                "manifest": manifest_name,
                "rows": rows,
                "ordered_identity_stream_sha256": declared.get(
                    "ordered_identity_stream_sha256"
                ),
            }
        )
    return records


def _validate_partition_receipt(
    spec: PreregistrationSpec,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    receipt, record = _load_json_mapping(
        spec.partition_receipt_path, label="new-head partition receipt"
    )
    _require_sha(record, spec.partition_receipt_sha256, label="partition receipt")
    if receipt.get("schema") != PARTITION_SCHEMA:
        raise NewHeadLRPreregistrationError("partition receipt schema drifted")
    canonical = _require_canonical_receipt(receipt, label="partition receipt")
    invariants = _require_all_true_invariants(
        receipt.get("invariants"), label="partition receipt"
    )
    if receipt.get("source_manifest_order") != list(SOURCE_MANIFESTS):
        raise NewHeadLRPreregistrationError(
            "partition source manifest order drifted"
        )
    outputs = receipt.get("outputs")
    if not isinstance(outputs, Mapping):
        raise NewHeadLRPreregistrationError("partition outputs are missing")
    d1 = outputs.get("d1_category_complete")
    d0 = outputs.get("d0_ordinary_primary")
    if not isinstance(d1, Mapping) or not isinstance(d0, Mapping):
        raise NewHeadLRPreregistrationError("partition D0/D1 outputs are missing")
    train_records = _receipt_manifest_records(
        d1.get("train"), label="D1 train partition"
    )
    eval_records = _receipt_manifest_records(
        d0.get("dev_screen"), label="shared D0 dev_screen partition"
    )
    summary = receipt.get("partition_summary")
    if not isinstance(summary, Mapping):
        raise NewHeadLRPreregistrationError("partition summary is missing")
    for partition, records in (
        ("train", train_records),
        ("dev_screen", eval_records),
    ):
        observed = summary.get(partition)
        if not isinstance(observed, Mapping):
            raise NewHeadLRPreregistrationError(
                f"partition summary is missing {partition}"
            )
        rows_by_manifest = observed.get("rows_by_manifest")
        expected_rows = {
            item["manifest"]: item["rows"] for item in records
        }
        if (
            rows_by_manifest != expected_rows
            or observed.get("rows") != sum(expected_rows.values())
        ):
            raise NewHeadLRPreregistrationError(
                f"partition {partition} row summary drifted"
            )
    binding = {
        **record,
        "schema": PARTITION_SCHEMA,
        "canonical_payload_sha256": canonical,
        "invariants": invariants,
    }
    return receipt, binding, train_records, eval_records


def _validate_support_receipt(
    spec: PreregistrationSpec,
    *,
    partition_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt, record = _load_json_mapping(
        spec.support_receipt_path, label="new-head support receipt"
    )
    _require_sha(record, spec.support_receipt_sha256, label="support receipt")
    if receipt.get("schema") != SUPPORT_SCHEMA:
        raise NewHeadLRPreregistrationError("support receipt schema drifted")
    canonical = _require_canonical_receipt(receipt, label="support receipt")
    invariants = _require_all_true_invariants(
        receipt.get("invariants"), label="support receipt"
    )
    inputs = receipt.get("inputs")
    partition = receipt.get("partition")
    if not isinstance(inputs, Mapping) or not isinstance(partition, Mapping):
        raise NewHeadLRPreregistrationError(
            "support receipt partition lineage is missing"
        )
    _declared_file_matches(
        inputs.get("partition_receipt"),
        partition_binding,
        label="support receipt partition input",
    )
    partition_receipt = partition.get("receipt")
    if partition_receipt is not None:
        _declared_file_matches(
            partition_receipt,
            partition_binding,
            label="support receipt partition summary",
        )
    if (
        partition.get("schema") != PARTITION_SCHEMA
        or partition.get("canonical_payload_sha256")
        != partition_binding["canonical_payload_sha256"]
    ):
        raise NewHeadLRPreregistrationError(
            "support receipt partition canonical lineage drifted"
        )
    filter_contract = receipt.get("filter_contract")
    expected_settings = {
        "patch_bank_cache": False,
        "patch_bank_cache_write": False,
        "support_patch_use_embedding": False,
        "support_patch_max_per_class": 200,
    }
    if not isinstance(filter_contract, Mapping) or (
        filter_contract.get("D0_and_D1_share_identical_runtime_bank") is not True
        or filter_contract.get("bank_consumers") != ["D0", "D1"]
        or filter_contract.get("required_dataset_settings") != expected_settings
    ):
        raise NewHeadLRPreregistrationError(
            "support receipt runtime-bank contract drifted"
        )
    outputs = receipt.get("outputs")
    runtime_declared = (
        outputs.get("runtime_support_tsv")
        if isinstance(outputs, Mapping)
        else None
    )
    if not isinstance(runtime_declared, Mapping):
        raise NewHeadLRPreregistrationError(
            "support receipt runtime TSV is missing"
        )
    runtime_path = _resolve_existing_path(
        runtime_declared.get("path"), label="runtime support TSV"
    )
    runtime_record = _file_record(runtime_path, label="runtime support TSV")
    _declared_file_matches(
        runtime_declared, runtime_record, label="runtime support TSV"
    )
    rows = runtime_declared.get("rows")
    runtime_bank = receipt.get("runtime_bank")
    coverage = receipt.get("training_class_coverage")
    if (
        type(rows) is not int
        or rows <= 0
        or not isinstance(runtime_bank, Mapping)
        or runtime_bank.get("candidate_rows") != rows
        or type(runtime_bank.get("class_count")) is not int
        or runtime_bank["class_count"] <= 0
        or not isinstance(coverage, Mapping)
        or type(coverage.get("required_class_count")) is not int
        or coverage.get("covered_class_count")
        != coverage.get("required_class_count")
        or coverage.get("missing_class_ids") != []
    ):
        raise NewHeadLRPreregistrationError(
            "support receipt coverage/count contract drifted"
        )
    binding = {
        **record,
        "schema": SUPPORT_SCHEMA,
        "canonical_payload_sha256": canonical,
        "runtime_support_tsv": {**runtime_record, "rows": rows},
        "runtime_class_count": runtime_bank["class_count"],
        "training_class_count": coverage["required_class_count"],
        "invariants": invariants,
    }
    return receipt, binding


def _validate_initializer_pair(
    spec: PreregistrationSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    a0_record = _file_record(spec.a0_initializer_path, label="A0 initializer")
    _require_sha(a0_record, spec.a0_initializer_sha256, label="A0 initializer")
    pair, pair_record = _load_json_mapping(
        spec.pair_receipt_path, label="A0/A1 initializer pair receipt"
    )
    _require_sha(pair_record, spec.pair_receipt_sha256, label="pair receipt")
    if pair.get("schema") != PAIR_SCHEMA or pair.get("status") != "passed":
        raise NewHeadLRPreregistrationError(
            "initializer pair receipt is not passed"
        )
    invariants = _require_all_true_invariants(
        pair.get("invariants"), label="initializer pair receipt"
    )
    _declared_file_matches(
        pair.get("absolute_initializer"),
        a0_record,
        label="pair receipt A0 initializer",
    )
    binding = {
        **pair_record,
        "schema": PAIR_SCHEMA,
        "status": "passed",
        "invariants": invariants,
    }
    return a0_record, binding


def _validate_training_dataset(
    spec: PreregistrationSpec,
    *,
    train_records: Sequence[Mapping[str, Any]],
    partition_binding: Mapping[str, Any],
    support_binding: Mapping[str, Any],
) -> dict[str, Any]:
    dataset, record = _load_json_mapping(
        spec.dataset_path, label="D1 new-head training dataset config"
    )
    _require_sha(record, spec.dataset_sha256, label="training dataset config")
    rows = dataset.get("train")
    if not isinstance(rows, list) or len(rows) != len(SOURCE_MANIFESTS):
        raise NewHeadLRPreregistrationError(
            "training dataset must contain exactly three rows"
        )
    if dataset.get("val") != []:
        raise NewHeadLRPreregistrationError("training dataset val must be empty")
    runtime_support = support_binding["runtime_support_tsv"]
    expected_rows = {item["manifest"]: item for item in train_records}
    bound_rows = []
    for index, (dataset_row, manifest_name) in enumerate(
        zip(rows, SOURCE_MANIFESTS)
    ):
        if not isinstance(dataset_row, Mapping):
            raise NewHeadLRPreregistrationError(
                f"training dataset row {index} is not a mapping"
            )
        manifest = expected_rows[manifest_name]
        exact = {
            "dataset_mode": "patch_episode",
            "root": "/",
            "stage_b_data_driven_variant": "dd1_category_complete",
            "stage_b_data_driven_partition": "train",
            "stage_b_data_driven_manifest_sha256": manifest["sha256"],
            "stage_b_data_driven_receipt_sha256": (
                spec.partition_receipt_sha256
            ),
            "stage_b_data_driven_support_receipt_sha256": (
                spec.support_receipt_sha256
            ),
            "patch_bank_cache": False,
            "patch_bank_cache_write": False,
            "support_patch_use_embedding": False,
            "support_patch_max_per_class": 200,
            "mix_weight": 2.0,
        }
        drift = {
            key: (dataset_row.get(key), expected)
            for key, expected in exact.items()
            if not _exact_value(dataset_row.get(key), expected)
        }
        if drift:
            raise NewHeadLRPreregistrationError(
                f"training dataset row {index} contract drifted: {drift}"
            )
        if str(dataset_row.get("patch_bank_cache_path", "") or "").strip():
            raise NewHeadLRPreregistrationError(
                f"training dataset row {index} retains a cache path"
            )
        _same_path(
            dataset_row.get("anno"),
            Path(manifest["path"]),
            label=f"training dataset row {index} annotation",
        )
        _same_path(
            dataset_row.get("stage_b_data_driven_receipt"),
            Path(partition_binding["path"]),
            label=f"training dataset row {index} partition receipt",
        )
        _same_path(
            dataset_row.get("stage_b_data_driven_support_receipt"),
            Path(support_binding["path"]),
            label=f"training dataset row {index} support receipt",
        )
        _same_path(
            dataset_row.get("support_patch_tsv"),
            Path(runtime_support["path"]),
            label=f"training dataset row {index} support TSV",
        )
        bound_rows.append(dict(manifest))
    return {
        **record,
        "variant": "d1_category_complete",
        "partition": "train",
        "rows": sum(item["rows"] for item in bound_rows),
        "manifests": bound_rows,
    }


def _resolved_absent_directory(path: Path, *, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.exists() or expanded.is_symlink():
        raise NewHeadLRPreregistrationError(
            f"{label} must not exist before preregistration: {expanded}"
        )
    return expanded.resolve(strict=False)


def _validate_candidates(
    spec: PreregistrationSpec,
) -> tuple[list[dict[str, Any]], str]:
    if len(spec.candidates) != len(CANDIDATE_RANK_LRS):
        raise NewHeadLRPreregistrationError(
            "exactly three LR probe candidates are required"
        )
    if tuple(candidate.rank_lr for candidate in spec.candidates) != (
        CANDIDATE_RANK_LRS
    ):
        raise NewHeadLRPreregistrationError(
            "candidate LRs/order must be exactly 3e-5, 1e-4, 3e-4"
        )
    if len({candidate.label for candidate in spec.candidates}) != 3:
        raise NewHeadLRPreregistrationError("candidate labels must be unique")

    repo_root = spec.repo_root.expanduser().resolve(strict=True)
    cache: dict[Path, tuple[dict[str, Any], tuple[Path, ...]]] = {}
    required_values = _required_config_values(spec)
    candidate_payloads = []
    resolved_without_rank_lr: dict[str, Any] | None = None
    shared_chain_records: list[dict[str, Any]] | None = None
    output_paths: list[Path] = []
    semantic_sha256: str | None = None

    for candidate in spec.candidates:
        config_path = candidate.config_path.expanduser().resolve(strict=True)
        _validate_thin_probe_leaf(
            config_path,
            expected_rank_lr=candidate.rank_lr,
            expected_base_module=spec.probe_base_module,
        )
        resolved, chain = _load_literal_config(
            config_path,
            repo_root=repo_root,
            cache=cache,
            active=set(),
        )
        drift = {
            key: (resolved.get(key), expected)
            for key, expected in required_values.items()
            if not _exact_value(resolved.get(key), expected)
        }
        if drift:
            raise NewHeadLRPreregistrationError(
                f"probe config fixed training contract drifted: "
                f"{candidate.label}: {drift}"
            )
        observed_rank_lr = resolved.get("stage_b_data_driven_rank_lr")
        if (
            type(observed_rank_lr) is not float
            or observed_rank_lr != candidate.rank_lr
        ):
            raise NewHeadLRPreregistrationError(
                f"resolved probe rank_lr drifted: {candidate.label}"
            )

        without_lr = dict(resolved)
        del without_lr["stage_b_data_driven_rank_lr"]
        if resolved_without_rank_lr is None:
            resolved_without_rank_lr = without_lr
            semantic_sha256 = _sha256_bytes(_canonical_bytes(without_lr))
        elif without_lr != resolved_without_rank_lr:
            raise NewHeadLRPreregistrationError(
                "resolved probe configs differ outside rank_lr"
            )

        chain_records = [
            _file_record(path, label=f"{candidate.label} config dependency")
            for path in chain
        ]
        shared = [
            record for record in chain_records if record["path"] != str(config_path)
        ]
        if shared_chain_records is None:
            shared_chain_records = shared
        elif shared != shared_chain_records:
            raise NewHeadLRPreregistrationError(
                "probe config import chains differ outside their thin leaves"
            )
        training_output = _resolved_absent_directory(
            candidate.training_output_dir,
            label=f"{candidate.label} training output directory",
        )
        evaluation_output = _resolved_absent_directory(
            candidate.dev_screen_eval_dir,
            label=f"{candidate.label} dev_screen evaluation directory",
        )
        output_paths.extend((training_output, evaluation_output))
        effective_contract = _effective_training_contract(
            resolved, rank_lr=candidate.rank_lr
        )
        effective_without_lr = dict(effective_contract)
        del effective_without_lr["stage_b_data_driven_rank_lr"]
        effective_sha = _sha256_bytes(_canonical_bytes(effective_without_lr))
        candidate_payloads.append(
            {
                "label": candidate.label,
                "rank_lr": candidate.rank_lr,
                "config_file": chain_records[-1],
                "config_import_chain": chain_records,
                "resolved_config_without_rank_lr_sha256": semantic_sha256,
                "effective_training_contract": effective_contract,
                "effective_training_contract_without_rank_lr_sha256": (
                    effective_sha
                ),
                "training_output_dir": str(training_output),
                "dev_screen_eval_dir": str(evaluation_output),
                "expected_checkpoint": str(
                    training_output / "checkpoint_iter.pth"
                ),
                "expected_evaluation_summary": str(
                    evaluation_output / "summary.json"
                ),
            }
        )

    if len(set(output_paths)) != len(output_paths):
        raise NewHeadLRPreregistrationError(
            "candidate training/evaluation output paths must be distinct"
        )
    effective_hashes = {
        item["effective_training_contract_without_rank_lr_sha256"]
        for item in candidate_payloads
    }
    if len(effective_hashes) != 1:
        raise NewHeadLRPreregistrationError(
            "effective training contracts differ outside rank_lr"
        )
    if semantic_sha256 is None:
        raise NewHeadLRPreregistrationError("no candidate semantics were resolved")
    return candidate_payloads, semantic_sha256


def build_preregistration(
    spec: PreregistrationSpec | None = None,
) -> dict[str, Any]:
    contract = default_spec() if spec is None else spec
    try:
        repo_root = contract.repo_root.expanduser().resolve(strict=True)
    except OSError as error:
        raise NewHeadLRPreregistrationError(
            f"repository root cannot be resolved: {contract.repo_root}"
        ) from error
    if not repo_root.is_dir():
        raise NewHeadLRPreregistrationError(
            f"repository root is not a directory: {repo_root}"
        )

    candidates, resolved_semantics_sha256 = _validate_candidates(contract)
    (
        _partition_receipt,
        partition_binding,
        train_records,
        dev_screen_records,
    ) = _validate_partition_receipt(contract)
    _support_receipt, support_binding = _validate_support_receipt(
        contract, partition_binding=partition_binding
    )
    a0_binding, pair_binding = _validate_initializer_pair(contract)
    dataset_binding = _validate_training_dataset(
        contract,
        train_records=train_records,
        partition_binding=partition_binding,
        support_binding=support_binding,
    )
    source_code = {
        "main": _file_record(contract.main_path, label="training main.py"),
        "new_head_dev_evaluator": _file_record(
            contract.evaluator_path, label="new-head dev evaluator"
        ),
        "lr_selector": _file_record(
            contract.selector_path, label="new-head LR selector"
        ),
    }

    training_contract = {
        "experiment_id": "DD1",
        "variant_id": "DD1-NEWHEAD-LR-PROBE",
        "execution_scope": EXECUTION_SCOPE,
        "execution_contract": EXECUTION_CONTRACT,
        "diagnostic_not_formal_headline": True,
        "fresh_start": {
            "resume": "",
            "pretrain_model_path": a0_binding["path"],
            "pretrain_model_sha256": a0_binding["sha256"],
        },
        "optimizer_updates": 1000,
        "max_train_iters": 1000,
        "iter_checkpoint_interval": 1000,
        "batch_size": 64,
        "epochs": 1,
        "gradient_accumulation_steps": 1,
        "rank_lr_source": "candidate",
        "patch_lr": 3e-4,
        "optimizer": {
            "name": "AdamW",
            "weight_decay": 1e-4,
            "clip_max_norm": 0.1,
            "gradient_clip_contract": "per_optimizer_branch_v1",
        },
        "scheduler": {
            "name": "StepLR",
            "lr_drop": 100,
            "onecyclelr": False,
            "multi_step_lr": False,
        },
        "seed": 42,
        "sampler_seed": 42,
        "loader_seed": 1042,
        "sampling_contract": "deterministic_epoch_ledger_v1",
        "amp": True,
        "amp_init_scale": 8192.0,
        "data_loader": {
            "num_workers": 4,
            "prefetch_factor": 1,
            "pin_memory": True,
            "persistent_workers": False,
            "drop_last": True,
        },
        "allocator": {
            "environment_variable": "PYTORCH_CUDA_ALLOC_CONF",
            "value": "expandable_segments:True",
        },
        "resolved_config_without_rank_lr_sha256": resolved_semantics_sha256,
    }
    evaluation_contract = {
        "checkpoint_variant": "d1_category_complete",
        "manifest_variant": "d0_ordinary_primary",
        "partition": "dev_screen",
        "rank_only": True,
        "category_gate": False,
        "source_order": list(SOURCE_MANIFESTS),
        "rows": sum(item["rows"] for item in dev_screen_records),
        "manifests": dev_screen_records,
        "selection_metric": SELECTION_METRIC,
        "secondary_selection_metric": SECONDARY_SELECTION_METRIC,
    }
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "preregistered",
        "candidate_rank_lrs": list(CANDIDATE_RANK_LRS),
        "optimizer_updates_per_candidate": OPTIMIZER_UPDATES_PER_CANDIDATE,
        "selection_partition": SELECTION_PARTITION,
        "selection_metric": SELECTION_METRIC,
        "secondary_selection_metric": SECONDARY_SELECTION_METRIC,
        "tie_break_rule": list(TIE_BREAK_RULE),
        "repository_root": str(repo_root),
        "sealed_inputs": {
            "training_dataset_config": dataset_binding,
            "new_head_partition_receipt": partition_binding,
            "support_partition_receipt": support_binding,
            "a0_initializer": a0_binding,
            "initializer_pair_receipt": pair_binding,
        },
        "source_code": source_code,
        "training_contract": training_contract,
        "evaluation_contract": evaluation_contract,
        "candidates": candidates,
        "invariants": {
            "candidate_set_was_fixed_before_observing_metrics": True,
            "selection_rule_was_fixed_before_observing_metrics": True,
            "all_candidate_output_directories_were_absent": True,
            "all_probe_configs_are_thin_rank_lr_only_leaves": True,
            "resolved_probe_configs_differ_only_in_rank_lr": True,
            "effective_runtime_contracts_differ_only_in_rank_lr": True,
            "all_candidates_start_fresh_from_the_same_sealed_a0": True,
            "all_candidates_use_exactly_1000_optimizer_updates": True,
            "all_candidates_share_batch_patch_seed_amp_and_loader_contract": True,
            "d1_training_dataset_binds_the_sealed_train_partition": True,
            "d1_training_dataset_binds_the_sealed_support_partition": True,
            "lr_selection_uses_shared_d0_ordinary_primary_dev_screen_manifests": True,
            "main_evaluator_and_selector_sources_are_content_bound": True,
            "preregistration_output_is_atomic_create_new": True,
        },
    }
    if any(value is not True for value in payload["invariants"].values()):
        raise NewHeadLRPreregistrationError(
            "one or more preregistration invariants failed"
        )
    payload["canonical_payload_sha256"] = _sha256_bytes(
        _canonical_bytes(payload)
    )
    return payload


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish ``source`` without replacing an existing path.

    The experiment lives on exFAT, where hard links are unsupported. Linux's
    renameat2 provides the same create-new publication guarantee without
    requiring hard-link support.
    """

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable") from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def write_preregistration_new(
    payload: Mapping[str, Any], output: Path
) -> Path:
    path = output.expanduser().resolve(strict=False)
    if path.exists() or path.is_symlink():
        raise NewHeadLRPreregistrationError(
            f"refusing to overwrite existing preregistration: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    descriptor, temporary_value = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_value)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            _rename_noreplace(temporary, path)
        except FileExistsError as error:
            raise NewHeadLRPreregistrationError(
                f"refusing concurrent overwrite of preregistration: {path}"
            ) from error
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def build_and_write(
    *,
    output: Path = DEFAULT_OUTPUT,
    spec: PreregistrationSpec | None = None,
) -> dict[str, Any]:
    path = output.expanduser().resolve(strict=False)
    if path.exists() or path.is_symlink():
        raise NewHeadLRPreregistrationError(
            f"refusing to overwrite existing preregistration: {path}"
        )
    payload = build_preregistration(spec)
    write_preregistration_new(payload, path)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="new preregistration JSON path; existing files are rejected",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = build_and_write(output=args.output)
    except (NewHeadLRPreregistrationError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


__all__ = [
    "CANDIDATE_RANK_LRS",
    "CandidateSpec",
    "DEFAULT_OUTPUT",
    "NewHeadLRPreregistrationError",
    "PreregistrationSpec",
    "SCHEMA",
    "build_and_write",
    "build_preregistration",
    "default_spec",
    "main",
    "write_preregistration_new",
]


if __name__ == "__main__":
    raise SystemExit(main())
