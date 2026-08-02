#!/usr/bin/env python3
"""Replay sealed U2 at U25/U50/U75/U100 and prove the U100 trajectory.

The formal U2 run only preserved U100.  This runner restarts from the exact U0
initializer, stops at 25-update boundaries, and resumes the complete training
state to preserve four evaluation milestones.  A replay is accepted only when
its U100 trainable-tensor SHA256 is bitwise equal to the sealed formal U100.

Planning is the default and never launches training.  Pass ``--execute`` to
run the four GPU segments, or ``--audit-only`` to re-audit existing milestones.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.GroundingDINO.stage_b_u0_patch_rank import (  # noqa: E402
    StageBU0PatchRankAdapter,
    stage_b_u0_tensor_state_sha256,
)
from tools.audit_stageb_u0_transition import audit_u0_transition  # noqa: E402
from tools.build_stageb_u0_training_receipt import (  # noqa: E402
    _safe_load_checkpoint,
    canonical_json_sha256,
    stable_file_record,
)
from util.slconfig import SLConfig  # noqa: E402


SCHEMA = "pivot.stageb.u2_trajectory_replay/v1"
MILESTONES = (25, 50, 75, 100)
CHECKPOINT_INTERVAL = 25
EXPECTED_SEED = 42
EXPECTED_BATCH_SIZE = 56
EXPECTED_AMP_SCALE = 8192.0
EXPECTED_DATA_ROOT = Path("/media/haoyi/T9/data")
EXPECTED_FORMAL_CHECKPOINT_SHA256 = (
    "44e3d70b164eff2bcefacc37081b7cbab184a9373720ef69713d47949d449b90"
)
EXPECTED_INITIALIZER_SHA256 = (
    "c89e5dfba795fd8074a044f0c09d81c871705c20a1dbf819b9f16c770a2cba43"
)
EXPECTED_FORMAL_TRAINABLE_SHA256 = (
    "3f2175306494931b85114f85e346cbfb7d45caa8ddab82f0c289f4f927b9f6b0"
)
EXPECTED_FROZEN_SHA256 = (
    "09f64e2f72c9b8ac11e6ce759a92fb9c1b58337bfcc5e573c61855fcf4893898"
)

FORMAL_ROOT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/u2_category_complete_seed42_b56_scale8192_v2"
)
FORMAL_CHECKPOINT = FORMAL_ROOT / "checkpoint_iter.pth"
FORMAL_RECEIPT = FORMAL_ROOT / "training_receipt.json"
FORMAL_TRANSITION_AUDIT = (
    FORMAL_ROOT / "audits/checkpoint_iter_000100.transition.json"
)
INITIALIZER = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/u0_single_network_seed42_b56_v1/initializer/checkpoint_u0_init.pth"
)
CONFIG = REPO_ROOT / "config/ablations/cfg_stageb_u2_category_complete_patch_rank.py"
DATASETS = REPO_ROOT / "config/datasets_stageb_u2_category_complete_three_ref.json"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/u2_category_complete_seed42_b56_scale8192_replay_interval25_v1"
)
DEFAULT_PYTHON = Path("/home/haoyi/miniconda/envs/gdino5090/bin/python")

# These two post-sealing changes add an inference-only category gate.  They are
# accepted only at these exact hashes and only after the disabled U2 path is
# checked functionally below.  All other sealed training sources stay bytewise
# bound to the formal receipt.
INACTIVE_CATEGORY_GATE_CODE_COMPATIBILITY = {
    "models/GroundingDINO/groundingdino.py": {
        "sealed_sha256": "f9fc08f9d01b6822fd48a5ed3cef6c914627854dc98bcb07daa4dd84911ac855",
        "compatible_sha256": "8be237d5afb7fae34c72ab038e51bf23c90dd88480b3b39507280bbf4c6a6d34",
    },
    "models/GroundingDINO/stage_b_u0_patch_rank.py": {
        "sealed_sha256": "ccda313145b5ec1468177e423f1e276f3a90c7cc2361e9f1cc4e5d6724e52e6e",
        "compatible_sha256": "e07ba12fc4e635dd6cc7f12c2af4e2c793f5ae8370420b50920682b383aab147",
    },
}


class U2ReplayError(RuntimeError):
    """The requested run is not an exact, safely isolated U2 replay."""


def _strict_json(path: Path, *, label: str) -> MutableMapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise U2ReplayError(f"could not parse {label}: {error}") from error
    if not isinstance(value, MutableMapping):
        raise U2ReplayError(f"{label} is not a JSON object")
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise U2ReplayError(f"{label} is not an object")
    return value


def _scalar_int(value: Any, *, label: str) -> int:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise U2ReplayError(f"{label} is not scalar")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise U2ReplayError(f"{label} is not an integer scalar")
    if int(value) != float(value):
        raise U2ReplayError(f"{label} is not an integer scalar")
    return int(value)


def _same_path(value: Any, expected: Path, *, label: str) -> None:
    if not isinstance(value, (str, os.PathLike)) or not os.fspath(value):
        raise U2ReplayError(f"{label} is not a path")
    observed = Path(os.fspath(value)).expanduser()
    if not observed.is_absolute():
        observed = REPO_ROOT / observed
    if observed.resolve() != expected.expanduser().resolve():
        raise U2ReplayError(
            f"{label} drifted: expected {expected.resolve()}, got {observed.resolve()}"
        )


def _record_matches(
    record: Mapping[str, Any],
    *,
    label: str,
    path_override: Path | None = None,
    cache: dict[Path, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    raw_path = path_override if path_override is not None else record.get("path")
    if not isinstance(raw_path, (str, os.PathLike)):
        raise U2ReplayError(f"{label} has no path")
    path = Path(raw_path).expanduser().resolve()
    if cache is not None and path in cache:
        observed = cache[path]
    else:
        try:
            observed = stable_file_record(path, label=label)
        except Exception as error:
            raise U2ReplayError(str(error)) from error
        if cache is not None:
            cache[path] = observed
    if observed["sha256"] != record.get("sha256"):
        raise U2ReplayError(
            f"{label} SHA256 drifted: expected {record.get('sha256')}, "
            f"got {observed['sha256']}"
        )
    if observed["size_bytes"] != record.get("size_bytes"):
        raise U2ReplayError(f"{label} size drifted")
    return observed


def _stable_record(path: Path, *, label: str) -> dict[str, Any]:
    try:
        return stable_file_record(path, label=label)
    except Exception as error:
        raise U2ReplayError(str(error)) from error


def _verify_inactive_category_gate_compatibility() -> dict[str, Any]:
    cfg = SLConfig.fromfile(str(CONFIG))
    if bool(getattr(cfg, "stage_b_u0_category_preserving_patch_gate", False)):
        raise U2ReplayError("formal U2 config unexpectedly enables the category gate")

    payload = _load(FORMAL_CHECKPOINT, label="formal U100 checkpoint")
    state = _mapping(payload.get("model"), label="formal U100 model state")
    prefix = "stage_b_u0_patch_rank_adapter."
    adapter_state = {
        str(key).removeprefix(prefix): value
        for key, value in state.items()
        if str(key).startswith(prefix)
    }
    adapter = StageBU0PatchRankAdapter(
        query_count=900,
        hidden_dim=64,
        score_clip=5.0,
    )
    if adapter.category_preserving_gate:
        raise U2ReplayError("default U2 adapter unexpectedly enables the category gate")
    try:
        adapter.load_state_dict(adapter_state, strict=True)
    except Exception as error:
        raise U2ReplayError(
            f"compatible adapter no longer strict-loads formal U2 state: {error}"
        ) from error

    adapter.train()
    patch = torch.linspace(-2.0, 2.0, steps=1800, dtype=torch.float32).reshape(2, 900)
    teacher = torch.linspace(1.0, -1.0, steps=1800, dtype=torch.float32).reshape(2, 900)
    mask = torch.ones((2, 900), dtype=torch.bool)
    mask[0, -7:] = False
    with torch.no_grad():
        patch_normalized = adapter._standardize(patch, mask, clip=adapter.score_clip)
        teacher_detached = teacher.detach()
        teacher_normalized = adapter._standardize(
            teacher_detached, mask, clip=adapter.score_clip
        )
        features = torch.stack(
            (
                patch_normalized,
                teacher_normalized,
                patch_normalized * teacher_normalized,
            ),
            dim=-1,
        )
        learned = adapter.output(adapter.trunk(features)).squeeze(-1)
        learned = learned.to(dtype=teacher.dtype).masked_fill(~mask, 0.0)
        expected_rank = teacher_detached + learned
        observed = adapter(patch, teacher, mask)
    if not torch.equal(observed["rank_score"], expected_rank):
        raise U2ReplayError("disabled category gate changed the formal U2 rank path")
    if not torch.equal(observed["patch_rank_residual"], learned):
        raise U2ReplayError("disabled category gate changed the formal U2 residual path")
    if any("category_gate" in key for key in observed):
        raise U2ReplayError("disabled category gate emitted inference-only outputs")
    del payload, state, adapter_state, adapter
    gc.collect()
    return {
        "mode": "exact_hash_whitelisted_inference_only_addition",
        "formal_u2_gate_enabled": False,
        "formal_adapter_strict_load": True,
        "disabled_training_forward_matches_pre_gate_equation_bitwise": True,
        "model_state_surface_unchanged": True,
        "files": INACTIVE_CATEGORY_GATE_CODE_COMPATIBILITY,
    }


def _verify_receipt_and_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = _strict_json(FORMAL_RECEIPT, label="formal U2 receipt")
    claimed = receipt.pop("receipt_sha256", None)
    if claimed != canonical_json_sha256(receipt):
        raise U2ReplayError("formal U2 receipt self-digest is invalid")
    if receipt.get("schema") != "pivot.stageb.u2_training_receipt/v1":
        raise U2ReplayError("formal U2 receipt schema drifted")

    cache: dict[Path, dict[str, Any]] = {}
    checkpoint_record = _mapping(
        _mapping(receipt.get("checkpoint"), label="receipt.checkpoint").get("file"),
        label="receipt.checkpoint.file",
    )
    initializer_record = _mapping(
        _mapping(receipt.get("initializer"), label="receipt.initializer").get("file"),
        label="receipt.initializer.file",
    )
    observed_checkpoint = _record_matches(
        checkpoint_record,
        label="formal U100 checkpoint",
        path_override=FORMAL_CHECKPOINT,
        cache=cache,
    )
    observed_initializer = _record_matches(
        initializer_record,
        label="formal U0 initializer",
        path_override=INITIALIZER,
        cache=cache,
    )
    if observed_checkpoint["sha256"] != EXPECTED_FORMAL_CHECKPOINT_SHA256:
        raise U2ReplayError("formal U100 checkpoint is not the sealed artifact")
    if observed_initializer["sha256"] != EXPECTED_INITIALIZER_SHA256:
        raise U2ReplayError("U0 initializer is not the sealed artifact")

    compatible_code_drift = []
    for index, raw in enumerate(receipt.get("core_sources_at_sealing", [])):
        record = _mapping(raw, label=f"core source record {index}")
        relative = record.get("relative_path")
        if not isinstance(relative, str):
            raise U2ReplayError(f"core source record {index} lacks relative_path")
        compatibility = INACTIVE_CATEGORY_GATE_CODE_COMPATIBILITY.get(relative)
        if compatibility is None:
            _record_matches(
                record,
                label=f"core source {relative}",
                path_override=REPO_ROOT / relative,
                cache=cache,
            )
            continue
        if record.get("sha256") != compatibility["sealed_sha256"]:
            raise U2ReplayError(f"sealed hash contract drifted for {relative}")
        observed = _stable_record(
            REPO_ROOT / relative, label=f"core source {relative}"
        )
        if observed["sha256"] != compatibility["compatible_sha256"]:
            raise U2ReplayError(
                f"unreviewed code drift in {relative}: got {observed['sha256']}"
            )
        cache[(REPO_ROOT / relative).resolve()] = observed
        compatible_code_drift.append(relative)
    if sorted(compatible_code_drift) != sorted(
        INACTIVE_CATEGORY_GATE_CODE_COMPATIBILITY
    ):
        raise U2ReplayError("inactive category-gate compatibility surface is incomplete")
    _verify_inactive_category_gate_compatibility()

    config_binding = _mapping(receipt.get("config"), label="receipt.config")
    for index, raw in enumerate(config_binding.get("import_chain", [])):
        record = _mapping(raw, label=f"config import record {index}")
        _record_matches(record, label=f"config import {index}", cache=cache)
    dataset_binding = _mapping(receipt.get("datasets"), label="receipt.datasets")
    _record_matches(
        _mapping(dataset_binding.get("file"), label="datasets.file"),
        label="dataset config",
        path_override=DATASETS,
        cache=cache,
    )
    _record_matches(
        _mapping(dataset_binding.get("canonical_classes_json"), label="canonical classes"),
        label="canonical classes",
        cache=cache,
    )
    _record_matches(
        _mapping(dataset_binding.get("support_patch_tsv"), label="support TSV"),
        label="support patch TSV",
        cache=cache,
    )
    for index, raw in enumerate(dataset_binding.get("train", [])):
        entry = _mapping(raw, label=f"train dataset {index}")
        _record_matches(
            _mapping(entry.get("annotation"), label=f"train annotation {index}"),
            label=f"train annotation {index}",
            cache=cache,
        )
    category_receipt = _mapping(
        receipt.get("category_complete_data_receipt"), label="category data receipt"
    )
    _record_matches(
        _mapping(category_receipt.get("file"), label="category receipt file"),
        label="category receipt file",
        cache=cache,
    )
    for split, raw in _mapping(
        category_receipt.get("coco_annotations"), label="COCO annotations"
    ).items():
        value = _mapping(raw, label=f"COCO annotation {split}")
        _record_matches(
            _mapping(value.get("file"), label=f"COCO annotation file {split}"),
            label=f"COCO annotation {split}",
            cache=cache,
        )

    transition_record = _mapping(
        _mapping(receipt.get("transition_audit"), label="transition audit").get("file"),
        label="transition audit file",
    )
    _record_matches(
        transition_record,
        label="formal transition audit",
        path_override=FORMAL_TRANSITION_AUDIT,
        cache=cache,
    )
    transition = _strict_json(
        FORMAL_TRANSITION_AUDIT, label="formal transition audit"
    )
    if transition.get("u0_trainable_tensor_sha256") != EXPECTED_FORMAL_TRAINABLE_SHA256:
        raise U2ReplayError("formal transition audit trainable SHA drifted")

    runtime = _mapping(receipt.get("runtime_at_sealing"), label="sealed runtime")
    observed_runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    if dict(runtime) != observed_runtime:
        raise U2ReplayError(
            f"runtime drifted: expected {dict(runtime)}, got {observed_runtime}"
        )
    if Path(receipt.get("data_root", "")).resolve() != EXPECTED_DATA_ROOT.resolve():
        raise U2ReplayError("formal receipt data root drifted")
    return dict(receipt), transition


def milestone_path(output_root: Path, target: int) -> Path:
    return output_root / "milestones" / f"checkpoint_iter_{target:06d}.pth"


def milestone_audit_path(output_root: Path, target: int) -> Path:
    return output_root / "audits" / f"checkpoint_iter_{target:06d}.replay.json"


def build_train_command(
    *,
    python_bin: Path,
    output_root: Path,
    target: int,
    source: Path,
    resume: bool,
) -> list[str]:
    command = [
        str(python_bin),
        "main.py",
        "--config_file",
        str(CONFIG.relative_to(REPO_ROOT)),
        "--datasets",
        str(DATASETS.relative_to(REPO_ROOT)),
        "--output_dir",
        str(output_root),
        "--seed",
        str(EXPECTED_SEED),
    ]
    command.extend(["--resume" if resume else "--pretrain_model_path", str(source)])
    command.extend(
        [
            "--options",
            f"batch_size={EXPECTED_BATCH_SIZE}",
            "epochs=1",
            "--max_train_iters",
            str(target),
            "--iter_checkpoint_interval",
            str(CHECKPOINT_INTERVAL),
            "--gradient_accumulation_steps",
            "1",
            "--num_workers",
            "8",
            "--prefetch_factor",
            "1",
            "--mp_sharing_strategy",
            "file_system",
            "--min_nofile",
            "65536",
            "--amp",
            "--save_log",
        ]
    )
    return command


def build_plan(*, output_root: Path, python_bin: Path) -> dict[str, Any]:
    segments = []
    for index, target in enumerate(MILESTONES):
        resume = index > 0
        source = INITIALIZER if not resume else milestone_path(output_root, MILESTONES[index - 1])
        segments.append(
            {
                "target": target,
                "initialization": "resume" if resume else "pretrain",
                "source": str(source),
                "milestone": str(milestone_path(output_root, target)),
                "command": build_train_command(
                    python_bin=python_bin,
                    output_root=output_root,
                    target=target,
                    source=source,
                    resume=resume,
                ),
            }
        )
    return {
        "schema": SCHEMA,
        "mode": "segmented_complete_state_resume",
        "output_root": str(output_root),
        "formal_root": str(FORMAL_ROOT),
        "formal_checkpoint_sha256": EXPECTED_FORMAL_CHECKPOINT_SHA256,
        "initializer_sha256": EXPECTED_INITIALIZER_SHA256,
        "expected_u100_trainable_tensor_sha256": EXPECTED_FORMAL_TRAINABLE_SHA256,
        "expected_frozen_tensor_sha256": EXPECTED_FROZEN_SHA256,
        "contract": {
            "seed": EXPECTED_SEED,
            "physical_batch_size": EXPECTED_BATCH_SIZE,
            "gradient_accumulation_steps": 1,
            "amp_initial_scale": EXPECTED_AMP_SCALE,
            "amp_skips_allowed": 0,
            "optimizer": "AdamW",
            "optimizer_group_lrs": [3e-4, 3e-4],
            "weight_decay": 1e-4,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "milestones": list(MILESTONES),
            "data_root": str(EXPECTED_DATA_ROOT),
        },
        "acceptance": {
            "all_milestones_transition_audited": True,
            "all_milestones_frozen_sha_exact": True,
            "u100_trainable_sha_bitwise_equal_formal": True,
            "full_checkpoint_file_sha_expected_to_differ_due_to_replay_args": True,
        },
        "source_compatibility": {
            "sealed_sources_bytewise_exact_except": sorted(
                INACTIVE_CATEGORY_GATE_CODE_COMPATIBILITY
            ),
            "exception": "reviewed_inference_only_category_gate_default_off",
            "exact_compatible_hashes": INACTIVE_CATEGORY_GATE_CODE_COMPATIBILITY,
            "u100_trainable_sha_remains_final_trajectory_gate": True,
        },
        "segments": segments,
    }


def _validate_optimizer(payload: Mapping[str, Any], *, target: int) -> dict[str, Any]:
    optimizer = _mapping(payload.get("optimizer"), label="optimizer")
    groups = optimizer.get("param_groups")
    states = _mapping(optimizer.get("state"), label="optimizer.state")
    if not isinstance(groups, list) or len(groups) != 2:
        raise U2ReplayError("optimizer must have exactly two parameter groups")
    expected = (("patch_rank_residual", 8), ("patch_projection", 8))
    parameter_ids: list[Any] = []
    summaries = []
    for index, ((branch, count), raw) in enumerate(zip(expected, groups)):
        group = _mapping(raw, label=f"optimizer group {index}")
        parameters = group.get("params")
        if not isinstance(parameters, list) or len(parameters) != count:
            raise U2ReplayError(f"optimizer group {branch} parameter count drifted")
        if group.get("stage_b_u0_branch") != branch:
            raise U2ReplayError(f"optimizer group {index} branch drifted")
        for field, value in (
            ("lr", 3e-4),
            ("initial_lr", 3e-4),
            ("weight_decay", 1e-4),
            ("eps", 1e-8),
        ):
            if group.get(field) != value:
                raise U2ReplayError(f"optimizer group {branch} {field} drifted")
        if list(group.get("betas", [])) != [0.9, 0.999]:
            raise U2ReplayError(f"optimizer group {branch} betas drifted")
        parameter_ids.extend(parameters)
        summaries.append({"branch": branch, "parameter_count": len(parameters)})
    if len(set(parameter_ids)) != 16 or set(states) != set(parameter_ids):
        raise U2ReplayError("optimizer state does not exactly cover 16 trainable tensors")
    steps = {
        _scalar_int(
            _mapping(states[key], label=f"optimizer state {key}").get("step"),
            label=f"optimizer state {key}.step",
        )
        for key in parameter_ids
    }
    if steps != {target}:
        raise U2ReplayError(f"optimizer state steps drifted: {sorted(steps)}")
    return {"group_count": 2, "groups": summaries, "state_steps": [target]}


def _validate_args(
    payload: Mapping[str, Any],
    *,
    target: int,
    output_root: Path,
    source: Path,
    resume: bool,
) -> None:
    args = _mapping(payload.get("args"), label="checkpoint args")
    exact = {
        "seed": EXPECTED_SEED,
        "batch_size": EXPECTED_BATCH_SIZE,
        "max_train_iters": target,
        "iter_checkpoint_interval": CHECKPOINT_INTERVAL,
        "gradient_accumulation_steps": 1,
        "num_workers": 8,
        "prefetch_factor": 1,
        "mp_sharing_strategy": "file_system",
        "min_nofile": 65536,
        "amp_init_scale": EXPECTED_AMP_SCALE,
        "stage_b_u0_patch_rank_lr": 3e-4,
        "stage_b_u0_patch_projection_lr": 3e-4,
        "weight_decay": 1e-4,
    }
    for key, expected in exact.items():
        if args.get(key) != expected:
            raise U2ReplayError(
                f"checkpoint args.{key} drifted: expected {expected!r}, got {args.get(key)!r}"
            )
    for key in (
        "amp",
        "stage_b_u0_patch_rank",
        "stage_b_u2_category_complete_supervision",
        "stage_b_gdino_score_adapter",
        "enable_patch_branch",
        "skip_eval",
    ):
        if args.get(key) is not True:
            raise U2ReplayError(f"checkpoint args.{key} must be true")
    if args.get("distributed") is not False or args.get("world_size") != 1:
        raise U2ReplayError("replay must remain single-process")
    _same_path(args.get("config_file"), CONFIG, label="checkpoint config")
    _same_path(args.get("datasets"), DATASETS, label="checkpoint datasets")
    _same_path(args.get("output_dir"), output_root, label="checkpoint output root")
    if resume:
        _same_path(args.get("resume"), source, label="checkpoint resume source")
        if args.get("pretrain_model_path") not in (None, ""):
            raise U2ReplayError("resumed segment also declares a pretrain initializer")
    else:
        if args.get("resume") not in (None, ""):
            raise U2ReplayError("first replay segment unexpectedly resumes")
        _same_path(
            args.get("pretrain_model_path"), source, label="checkpoint initializer"
        )


def audit_checkpoint_payload(
    *,
    initializer_payload: Mapping[str, Any],
    payload: Mapping[str, Any],
    target: int,
    output_root: Path,
    source: Path,
    resume: bool,
    trainable_keys: Sequence[str],
) -> dict[str, Any]:
    for key in ("iteration", "optimizer_updates"):
        if type(payload.get(key)) is not int or payload.get(key) != target:
            raise U2ReplayError(f"checkpoint {key} is not exact U{target}")
    if payload.get("epoch") != 0 or payload.get("epoch_finished") is not False:
        raise U2ReplayError(f"U{target} is not a mid-epoch checkpoint")
    if payload.get("checkpoint_reason") != "max_train_iters":
        raise U2ReplayError(f"U{target} checkpoint reason drifted")
    _validate_args(
        payload,
        target=target,
        output_root=output_root,
        source=source,
        resume=resume,
    )
    optimizer = _validate_optimizer(payload, target=target)
    scaler = _mapping(payload.get("scaler"), label="AMP scaler")
    expected_scaler = {
        "scale": EXPECTED_AMP_SCALE,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 2000,
        "_growth_tracker": target,
    }
    for key, expected in expected_scaler.items():
        if scaler.get(key) != expected:
            raise U2ReplayError(
                f"U{target} scaler.{key} drifted: expected {expected}, got {scaler.get(key)}"
            )
    for key in ("rng_state", "epoch_rng_state", "criterion", "lr_scheduler"):
        if not isinstance(payload.get(key), Mapping):
            raise U2ReplayError(f"U{target} lacks complete {key}")

    try:
        transition = audit_u0_transition(initializer_payload, payload)
    except Exception as error:
        raise U2ReplayError(f"U{target} transition audit failed: {error}") from error
    if transition.get("changed_keys") != sorted(trainable_keys):
        raise U2ReplayError(f"U{target} did not update the exact 16-tensor surface")
    state = _mapping(payload.get("model"), label="model state")
    trainable_sha = stage_b_u0_tensor_state_sha256(state, trainable_keys)
    frozen_keys = sorted(set(state).difference(trainable_keys))
    frozen_sha = stage_b_u0_tensor_state_sha256(state, frozen_keys)
    if trainable_sha != transition.get("u0_trainable_tensor_sha256"):
        raise U2ReplayError(f"U{target} trainable SHA audit disagrees internally")
    if len(frozen_keys) != 1149 or frozen_sha != EXPECTED_FROZEN_SHA256:
        raise U2ReplayError(f"U{target} frozen tensor state drifted")
    return {
        "schema": SCHEMA,
        "status": "verified",
        "target": target,
        "source": str(source.resolve()),
        "initialization": "resume" if resume else "pretrain",
        "optimizer": optimizer,
        "amp": {
            "initial_scale": EXPECTED_AMP_SCALE,
            "final_scale": scaler["scale"],
            "growth_tracker": scaler["_growth_tracker"],
            "skipped_steps": 0,
        },
        "transition": transition,
        "trainable_key_count": len(trainable_keys),
        "trainable_tensor_sha256": trainable_sha,
        "frozen_key_count": len(frozen_keys),
        "frozen_tensor_sha256": frozen_sha,
        "formal_u100_trainable_tensor_sha256": EXPECTED_FORMAL_TRAINABLE_SHA256,
        "formal_u100_trainable_exact_match": (
            target == 100 and trainable_sha == EXPECTED_FORMAL_TRAINABLE_SHA256
        ),
    }


def _load(path: Path, *, label: str) -> MutableMapping[str, Any]:
    try:
        return _safe_load_checkpoint(path, label=label)
    except Exception as error:
        raise U2ReplayError(str(error)) from error


def _atomic_copy_fresh(source: Path, destination: Path) -> None:
    if destination.exists():
        raise U2ReplayError(f"refusing to overwrite milestone: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(destination) + ".tmp")
    if temporary.exists():
        raise U2ReplayError(f"stale milestone temporary exists: {temporary}")
    try:
        with source.open("rb") as input_handle, temporary.open("xb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=16 * 1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _publish_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise U2ReplayError(f"existing JSON artifact disagrees: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    if temporary.exists():
        raise U2ReplayError(f"stale JSON temporary exists: {temporary}")
    temporary.write_text(encoded, encoding="ascii")
    os.replace(temporary, path)


def _validate_output_root(output_root: Path) -> None:
    output = output_root.resolve()
    formal = FORMAL_ROOT.resolve()
    if output == formal or formal in output.parents or output in formal.parents:
        raise U2ReplayError("replay output must be isolated from the formal U2 root")
    if output == REPO_ROOT.resolve():
        raise U2ReplayError("repository root cannot be used as replay output")


def _audit_path(
    *,
    checkpoint: Path,
    target: int,
    output_root: Path,
    source: Path,
    resume: bool,
    initializer_payload: Mapping[str, Any],
    trainable_keys: Sequence[str],
) -> dict[str, Any]:
    payload = _load(checkpoint, label=f"U{target} replay checkpoint")
    result = audit_checkpoint_payload(
        initializer_payload=initializer_payload,
        payload=payload,
        target=target,
        output_root=output_root,
        source=source,
        resume=resume,
        trainable_keys=trainable_keys,
    )
    result["checkpoint"] = _stable_record(
        checkpoint, label=f"U{target} replay checkpoint"
    )
    del payload
    gc.collect()
    return result


def _publish_replay_summary(
    output_root: Path, audits: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if [item.get("target") for item in audits] != list(MILESTONES):
        raise U2ReplayError("replay summary does not cover all four milestones")
    final_sha = audits[-1]["trainable_tensor_sha256"]
    if final_sha != EXPECTED_FORMAL_TRAINABLE_SHA256:
        raise U2ReplayError(
            "U100 replay is not the formal trajectory: "
            f"expected {EXPECTED_FORMAL_TRAINABLE_SHA256}, got {final_sha}"
        )
    summary = {
        "schema": SCHEMA,
        "status": "exact_u100_trainable_match",
        "output_root": str(output_root),
        "milestones": [
            {
                "target": item["target"],
                "checkpoint": item["checkpoint"],
                "trainable_tensor_sha256": item["trainable_tensor_sha256"],
                "frozen_tensor_sha256": item["frozen_tensor_sha256"],
            }
            for item in audits
        ],
        "formal_u100_trainable_tensor_sha256": EXPECTED_FORMAL_TRAINABLE_SHA256,
        "u100_trainable_bitwise_equal": True,
    }
    _publish_json(output_root / "replay_summary.json", summary)
    return summary


def audit_milestones(output_root: Path) -> dict[str, Any]:
    _, formal_transition = _verify_receipt_and_sources()
    trainable_keys = tuple(formal_transition.get("changed_keys", ()))
    if len(trainable_keys) != 16:
        raise U2ReplayError("formal trainable-key surface is not exactly 16 tensors")
    initializer_payload = _load(INITIALIZER, label="U0 initializer")
    audits = []
    previous = INITIALIZER
    for index, target in enumerate(MILESTONES):
        checkpoint = milestone_path(output_root, target)
        result = _audit_path(
            checkpoint=checkpoint,
            target=target,
            output_root=output_root,
            source=previous,
            resume=index > 0,
            initializer_payload=initializer_payload,
            trainable_keys=trainable_keys,
        )
        _publish_json(milestone_audit_path(output_root, target), result)
        audits.append(result)
        previous = checkpoint
    del initializer_payload
    gc.collect()
    return _publish_replay_summary(output_root, audits)


def execute_replay(
    *,
    output_root: Path,
    python_bin: Path,
    continue_run: bool,
) -> dict[str, Any]:
    _, formal_transition = _verify_receipt_and_sources()
    trainable_keys = tuple(formal_transition.get("changed_keys", ()))
    if len(trainable_keys) != 16:
        raise U2ReplayError("formal trainable-key surface is not exactly 16 tensors")
    if not python_bin.is_file():
        raise U2ReplayError(f"Python executable is missing: {python_bin}")
    existing = output_root.exists() and any(output_root.iterdir())
    if existing and not continue_run:
        raise U2ReplayError(
            f"replay output is non-empty; inspect it and pass --continue-run: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    plan = build_plan(output_root=output_root, python_bin=python_bin)
    _publish_json(output_root / "replay_plan.json", plan)

    audits = []
    previous = INITIALIZER
    for index, target in enumerate(MILESTONES):
        destination = milestone_path(output_root, target)
        if destination.exists():
            initializer_payload = _load(INITIALIZER, label="U0 initializer")
            result = _audit_path(
                checkpoint=destination,
                target=target,
                output_root=output_root,
                source=previous,
                resume=index > 0,
                initializer_payload=initializer_payload,
                trainable_keys=trainable_keys,
            )
            del initializer_payload
            gc.collect()
            _publish_json(milestone_audit_path(output_root, target), result)
            audits.append(result)
            previous = destination
            continue
        live = output_root / "checkpoint_iter.pth"
        command = build_train_command(
            python_bin=python_bin,
            output_root=output_root,
            target=target,
            source=previous,
            resume=index > 0,
        )
        launch = {
            "schema": SCHEMA,
            "target": target,
            "source": str(previous),
            "command": command,
        }
        _publish_json(
            output_root / "launches" / f"target_{target:06d}.json", launch
        )
        environment = os.environ.copy()
        environment.update(
            {
                "DATA_ROOT": str(EXPECTED_DATA_ROOT),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        segment_log = output_root / "segment_logs" / f"target_{target:06d}.log"
        segment_log.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"[INFO] replaying U{target}; child output -> {segment_log}",
            flush=True,
        )
        with segment_log.open("ab") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            while process.poll() is None:
                print(
                    f"[INFO] U{target} child pid={process.pid} still running",
                    flush=True,
                )
                time.sleep(10.0)
            returncode = int(process.returncode)
        if returncode != 0:
            tail = segment_log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
            raise U2ReplayError(
                f"U{target} training segment exited with {returncode}; "
                f"log tail:\n" + "\n".join(tail)
            )
        if not live.is_file():
            raise U2ReplayError(f"U{target} segment produced no live checkpoint")
        _atomic_copy_fresh(live, destination)
        initializer_payload = _load(INITIALIZER, label="U0 initializer")
        result = _audit_path(
            checkpoint=destination,
            target=target,
            output_root=output_root,
            source=previous,
            resume=index > 0,
            initializer_payload=initializer_payload,
            trainable_keys=trainable_keys,
        )
        del initializer_payload
        gc.collect()
        _publish_json(milestone_audit_path(output_root, target), result)
        audits.append(result)
        previous = destination
    return _publish_replay_summary(output_root, audits)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--python",
        dest="python_bin",
        type=Path,
        default=DEFAULT_PYTHON if DEFAULT_PYTHON.is_file() else Path(sys.executable),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--audit-only", action="store_true")
    parser.add_argument("--continue-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_root = output_root.resolve()
    python_bin = args.python_bin.expanduser().resolve()
    _validate_output_root(output_root)
    if args.execute:
        result = execute_replay(
            output_root=output_root,
            python_bin=python_bin,
            continue_run=bool(args.continue_run),
        )
    elif args.audit_only:
        result = audit_milestones(output_root)
    else:
        _verify_receipt_and_sources()
        result = build_plan(output_root=output_root, python_bin=python_bin)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
    return result


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, U2ReplayError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
