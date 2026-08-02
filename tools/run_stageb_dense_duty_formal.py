#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Fail-closed controller for the formal two-phase dense-duty Stage-B run.

The controller deliberately exposes no experiment overrides.  The atomic
``checkpoint_iter.pth`` in each phase directory is the only source of training
progress; process return codes are never treated as completion evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import enum
import fcntl
import hashlib
import json
import math
import os
import signal
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PYTHON = Path("/home/haoyi/miniconda/envs/gdino5090/bin/python")
if __name__ == "__main__" and Path(sys.executable).resolve() != PYTHON.resolve():
    if not PYTHON.is_file() or not os.access(PYTHON, os.X_OK):
        print(
            f"[FAIL] fixed controller interpreter is unavailable: {PYTHON}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    os.execv(
        str(PYTHON),
        [str(PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
    )
from util.stage_b_dense_duty_audit import (  # noqa: E402
    SOURCE_CLOSURE_ARG,
    TRAINING_CONTRACT_ARG,
    build_source_closure,
    build_training_contract,
    validate_source_closure,
    validate_strict_resume_checkpoint_payload,
)

MAIN = REPO_ROOT / "main.py"
FORMAL_ROOT = REPO_ROOT / "outputs/paper_cvpr_v1/dense_duty_20260728/formal"
LOCK_PATH = FORMAL_ROOT.parent / ".formal_dense_duty.lock"

RANK_CONFIG = REPO_ROOT / "config/ablations/cfg_stageb_dense_duty_rank_20260728.py"
CONFIDENCE_CONFIG = (
    REPO_ROOT / "config/ablations/cfg_stageb_dense_duty_confidence_20260728.py"
)
RANK_DATASET = REPO_ROOT / "config/datasets_stageb_dense_duty_rank_20260728.json"
CONFIDENCE_DATASET = (
    REPO_ROOT / "config/datasets_stageb_dense_duty_confidence_20260728.json"
)
STAGE_A_CHECKPOINT = Path(
    "/media/haoyi/T9/gdino/outputs/stageA_coco_multipatch/checkpoint0006.pth"
)
TEXT_CHECKPOINT = REPO_ROOT / "weights/groundingdino_swint_ogc.pth"
TN_MANIFEST = (
    REPO_ROOT
    / "data/ablations/stageb_gdino_adapter_semantic_partition_20260717/"
    "single_edit_train.jsonl"
)

RANK_UPDATES = 10_295
CONFIDENCE_UPDATES = 4_412
RANK_DATASET_SHA256 = (
    "6cc541d8347468c625ca0785a8a87c6a85ef9e85ac911a301feab4c25061ceba"
)
CONFIDENCE_DATASET_SHA256 = (
    "09ad8048e89e60243c6a1397a13a0f24356de81022b1e90a22855c0e61ad114e"
)
STAGE_A_SHA256 = (
    "a4f153c8cbd9b408b9479901e27ec486a10f393013193d44b0da1dcd1888cb91"
)
TEXT_CHECKPOINT_SHA256 = (
    "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799"
)
TN_MANIFEST_SHA256 = (
    "276dc5a67c6e7a6654d6daa6a88cb99b9c59b1c52f84ef93205a3d6326b1b529"
)

PHYSICAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4
EXPRESSION_MICROBATCH = 16
NUM_WORKERS = 0
PREFETCH_FACTOR = 1
PIN_MEMORY = True
PERSISTENT_WORKERS = False
ITER_CHECKPOINT_INTERVAL = 100
SEED = 42

CHECKPOINT_NAME = "checkpoint_iter.pth"
CHECKPOINT_AUDIT_SCHEMA = "pivot.stageb.dense_duty_checkpoint_audit/v1"
RUNTIME_AUDIT_SCHEMA = "pivot.stageb.dense_duty_runtime_audit/v1"


class FormalLaunchError(RuntimeError):
    """The formal run cannot continue from the observed evidence."""


class PhaseStatus(str, enum.Enum):
    FRESH = "fresh"
    PARTIAL = "partial"
    TERMINAL = "terminal"
    INVALID = "invalid"


class LaunchAction(str, enum.Enum):
    START_RANK = "start_rank"
    RESUME_RANK = "resume_rank"
    START_CONFIDENCE = "start_confidence"
    RESUME_CONFIDENCE = "resume_confidence"
    COMPLETE = "complete"
    INVALID = "invalid"


@dataclasses.dataclass(frozen=True)
class PhaseSpec:
    phase: str
    config: Path
    dataset: Path
    dataset_sha256: str
    output: Path
    expected_updates: int

    @property
    def checkpoint(self) -> Path:
        return self.output / CHECKPOINT_NAME


@dataclasses.dataclass(frozen=True)
class PhaseInspection:
    phase: str
    status: PhaseStatus
    checkpoint: Path | None = None
    optimizer_updates: int | None = None
    checkpoint_reason: str | None = None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "status": self.status.value,
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "optimizer_updates": self.optimizer_updates,
            "checkpoint_reason": self.checkpoint_reason,
            "detail": self.detail,
        }


@dataclasses.dataclass(frozen=True)
class ExperimentInspection:
    rank: PhaseInspection
    confidence: PhaseInspection
    action: LaunchAction
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "pivot.stageb.dense_duty_formal_controller_state/v1",
            "rank": self.rank.as_dict(),
            "confidence": self.confidence.as_dict(),
            "action": self.action.value,
            "detail": self.detail,
        }


@dataclasses.dataclass(frozen=True)
class ChildResult:
    returncode: int
    forwarded_signals: tuple[int, ...]


def formal_phase_specs(root: Path = FORMAL_ROOT) -> tuple[PhaseSpec, PhaseSpec]:
    root = Path(root)
    return (
        PhaseSpec(
            phase="rank",
            config=RANK_CONFIG,
            dataset=RANK_DATASET,
            dataset_sha256=RANK_DATASET_SHA256,
            output=root / "rank",
            expected_updates=RANK_UPDATES,
        ),
        PhaseSpec(
            phase="confidence",
            config=CONFIDENCE_CONFIG,
            dataset=CONFIDENCE_DATASET,
            dataset_sha256=CONFIDENCE_DATASET_SHA256,
            output=root / "confidence",
            expected_updates=CONFIDENCE_UPDATES,
        ),
    )


def _invalid(spec: PhaseSpec, detail: str) -> PhaseInspection:
    return PhaseInspection(spec.phase, PhaseStatus.INVALID, detail=detail)


def _is_exact_nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _required_saved_args(spec: PhaseSpec) -> dict[str, Any]:
    return {
        "config_file": str(spec.config.resolve()),
        "options": None,
        "datasets": str(spec.dataset.resolve()),
        "output_dir": str(spec.output.resolve()),
        "note": f"formal_dense_duty_{spec.phase}_seed{SEED}",
        "device": "cuda",
        "seed": SEED,
        "eval": False,
        "test": False,
        "debug": False,
        "save_results": False,
        "save_log": True,
        "finetune_ignore": None,
        "remove_difficult": False,
        "num_workers": NUM_WORKERS,
        "prefetch_factor": PREFETCH_FACTOR,
        "pin_memory": PIN_MEMORY,
        "persistent_workers": PERSISTENT_WORKERS,
        "mp_sharing_strategy": "file_system",
        "min_nofile": 65_536,
        "world_size": 1,
        "dist_url": "env://",
        "rank": 0,
        "local_rank": 0,
        "distributed": False,
        "find_unused_params": False,
        "amp": True,
        "iter_checkpoint_interval": ITER_CHECKPOINT_INTERVAL,
        "max_train_iters": spec.expected_updates,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "batch_size": PHYSICAL_BATCH_SIZE,
        "stage_b_v11_expression_microbatch": EXPRESSION_MICROBATCH,
        "stage_b_dense_duty": True,
        "stage_b_dense_duty_phase": spec.phase,
        "stage_b_v22_train_phase": spec.phase,
        "stage_b_dense_duty_no_stageb_teacher": True,
        "stage_b_dense_duty_execution_scope": "formal",
        "stage_b_v22_score_ownership": "independent_decoders_two_phase",
        "stage_b_dense_duty_dataset_config_path": str(spec.dataset.resolve()),
        "stage_b_dense_duty_dataset_config_sha256": spec.dataset_sha256,
        "stage_b_dense_duty_rank_dataset_config_sha256": RANK_DATASET_SHA256,
        "stage_b_dense_duty_base_checkpoint_sha256": STAGE_A_SHA256,
        "stage_b_dense_duty_text_checkpoint_sha256": TEXT_CHECKPOINT_SHA256,
        "stage_b_dense_duty_expected_physical_batch_size": PHYSICAL_BATCH_SIZE,
        "stage_b_dense_duty_expected_gradient_accumulation_steps": (
            GRADIENT_ACCUMULATION_STEPS
        ),
        "stage_b_dense_duty_expected_expression_microbatch": EXPRESSION_MICROBATCH,
        "stage_b_v11_candidate_topk": 50,
        "stage_b_v11_num_layers": 6,
        "stage_b_v15_patch_rank_fusion": False,
        "stage_b_v15_patch_rank_weight": 0.0,
        SOURCE_CLOSURE_ARG: build_source_closure(
            spec.config, repo_root=REPO_ROOT
        ),
    }


def _validate_runtime_audit(
    runtime: Any, *, optimizer_updates: int, terminal: bool
) -> str | None:
    if not isinstance(runtime, Mapping):
        return "saved args lack the dense-duty runtime audit"
    if runtime.get("schema") != RUNTIME_AUDIT_SCHEMA:
        return "runtime audit schema is invalid"
    successful_steps = runtime.get("successful_optimizer_steps")
    if type(successful_steps) is not int or successful_steps != optimizer_updates:
        return "runtime successful-step count differs from checkpoint progress"
    boundaries = runtime.get("optimizer_step_boundaries")
    if not _is_exact_nonnegative_int(boundaries) or boundaries != optimizer_updates:
        return "runtime optimizer-boundary count is invalid"
    if type(runtime.get("zero_gradient_successful_steps")) is not int or (
        runtime.get("zero_gradient_successful_steps") != 0
    ):
        return "runtime reports a zero-gradient successful step"
    if type(runtime.get("amp_skipped_optimizer_steps")) is not int or (
        runtime.get("amp_skipped_optimizer_steps") != 0
    ):
        return "runtime reports an AMP-skipped optimizer boundary"
    if type(runtime.get("nonfinite_gradient_boundaries")) is not int or (
        runtime.get("nonfinite_gradient_boundaries") != 0
    ):
        return "runtime reports a non-finite gradient boundary"
    if terminal:
        grad_norm = runtime.get("max_active_grad_norm_preclip")
        peak_reserved = runtime.get("peak_reserved_bytes")
        if (
            isinstance(grad_norm, bool)
            or not isinstance(grad_norm, (int, float))
            or not math.isfinite(float(grad_norm))
            or float(grad_norm) <= 0.0
        ):
            return "terminal runtime lacks a positive active gradient norm"
        if not _is_exact_nonnegative_int(peak_reserved) or peak_reserved <= 0:
            return "terminal runtime lacks measured CUDA reservation"
    return None


def classify_checkpoint_payload(
    payload: Any,
    spec: PhaseSpec,
    *,
    checkpoint_path: Path | None = None,
) -> PhaseInspection:
    """Classify a checkpoint from plain mappings, without loading a model/GPU."""
    path = Path(checkpoint_path) if checkpoint_path is not None else spec.checkpoint
    if path.name != CHECKPOINT_NAME or path.parent.resolve() != spec.output.resolve():
        return _invalid(spec, "checkpoint is not the canonical phase checkpoint_iter.pth")
    if not isinstance(payload, Mapping):
        return _invalid(spec, "checkpoint payload is not a mapping")

    required = {
        "model",
        "criterion",
        "optimizer",
        "lr_scheduler",
        "scaler",
        "epoch",
        "iteration",
        "optimizer_updates",
        "epoch_finished",
        "rng_state",
        "epoch_rng_state",
        "args",
        "checkpoint_reason",
    }
    missing = sorted(required.difference(payload))
    if missing:
        return _invalid(spec, f"checkpoint lacks required training state: {missing}")
    for key in (
        "model",
        "criterion",
        "optimizer",
        "lr_scheduler",
        "scaler",
        "rng_state",
        "epoch_rng_state",
        "args",
    ):
        if not isinstance(payload.get(key), Mapping):
            return _invalid(spec, f"checkpoint field {key!r} is not a mapping")
        if not payload[key]:
            return _invalid(spec, f"checkpoint field {key!r} is empty")

    epoch = payload["epoch"]
    iteration = payload["iteration"]
    updates = payload["optimizer_updates"]
    epoch_finished = payload["epoch_finished"]
    reason = payload["checkpoint_reason"]
    if not _is_exact_nonnegative_int(epoch):
        return _invalid(spec, "checkpoint epoch is not an exact non-negative integer")
    if not _is_exact_nonnegative_int(iteration):
        return _invalid(spec, "checkpoint iteration is not an exact non-negative integer")
    if iteration % GRADIENT_ACCUMULATION_STEPS != 0:
        return _invalid(spec, "checkpoint iteration is not an accumulation boundary")
    if type(updates) is not int or not 0 < updates <= spec.expected_updates:
        return _invalid(spec, "checkpoint optimizer_updates is outside the formal phase")
    if type(epoch_finished) is not bool:
        return _invalid(spec, "checkpoint epoch_finished is not an exact boolean")
    if epoch_finished != (iteration == 0):
        return _invalid(spec, "checkpoint epoch boundary fields are inconsistent")
    if not isinstance(reason, str):
        return _invalid(spec, "checkpoint reason is not a string")

    saved_args = payload["args"]
    expected_args = _required_saved_args(spec)
    drift = {
        key: (saved_args.get(key), expected)
        for key, expected in expected_args.items()
        if saved_args.get(key) != expected
    }
    if drift:
        return _invalid(spec, f"checkpoint formal arguments drifted: {drift}")

    resume_value = saved_args.get("resume")
    pretrain_value = saved_args.get("pretrain_model_path")
    start_epoch = saved_args.get("start_epoch")
    if resume_value == "":
        expected_pretrain = (
            str(STAGE_A_CHECKPOINT)
            if spec.phase == "rank"
            else str((spec.output.parent / "rank" / CHECKPOINT_NAME).resolve())
        )
        if pretrain_value != expected_pretrain or start_epoch != 0:
            return _invalid(
                spec,
                "fresh phase checkpoint violates its initializer/start-epoch contract",
            )
    elif resume_value == str(spec.checkpoint.resolve()):
        if pretrain_value not in {None, ""}:
            return _invalid(
                spec,
                "same-phase resume checkpoint also records a pretrain initializer",
            )
        if (
            type(start_epoch) is not int
            or start_epoch < 0
            or start_epoch > epoch
        ):
            return _invalid(
                spec,
                "same-phase resume checkpoint has an invalid restored start_epoch",
            )
    else:
        return _invalid(
            spec,
            "checkpoint was not produced by the canonical fresh/resume transition",
        )

    lineage = saved_args.get("stage_b_dense_duty_lineage_audit")
    dataset_record = lineage.get("dataset_config", {}) if isinstance(lineage, Mapping) else {}
    if (
        not isinstance(lineage, Mapping)
        or lineage.get("phase") != spec.phase
        or lineage.get("execution_scope") != "formal"
        or lineage.get("no_stage_b_teacher") is not True
        or not isinstance(dataset_record, Mapping)
        or dataset_record.get("sha256") != spec.dataset_sha256
    ):
        return _invalid(spec, "checkpoint dense-duty lineage is incomplete or drifted")

    terminal = updates == spec.expected_updates
    if terminal:
        if reason != "max_train_iters":
            return _invalid(spec, "terminal update count lacks max_train_iters reason")
        status = PhaseStatus.TERMINAL
    else:
        allowed = (
            {"signal", "interval_epoch", "signal_after_epoch"}
            if epoch_finished
            else {"interval", "signal"}
        )
        if reason not in allowed:
            return _invalid(spec, f"partial checkpoint reason {reason!r} is invalid")
        if reason.startswith("interval") and updates % ITER_CHECKPOINT_INTERVAL != 0:
            return _invalid(spec, "interval checkpoint is off the fixed update cadence")
        status = PhaseStatus.PARTIAL

    runtime_error = _validate_runtime_audit(
        saved_args.get("stage_b_dense_duty_runtime_audit"),
        optimizer_updates=updates,
        terminal=terminal,
    )
    if runtime_error:
        return _invalid(spec, runtime_error)

    return PhaseInspection(
        phase=spec.phase,
        status=status,
        checkpoint=path,
        optimizer_updates=updates,
        checkpoint_reason=reason,
    )


def _torch_load_checkpoint(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        message = str(exc)
        if "Weights only load failed" not in message and "weights_only" not in message:
            raise
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise FormalLaunchError("checkpoint loader returned a non-mapping payload")
    return payload


def _audit_checkpoint(payload: Mapping[str, Any], spec: PhaseSpec, path: Path) -> None:
    from util.stage_b_dense_duty_audit import (
        audit_checkpoint_payload,
        validate_rank_handoff_audit,
    )

    saved_args = payload.get("args")
    if not isinstance(saved_args, Mapping):
        raise FormalLaunchError("checkpoint lacks its saved argument mapping")
    recorded_contract = saved_args.get(TRAINING_CONTRACT_ARG)
    rebuilt_contract = build_training_contract(saved_args)
    if not isinstance(recorded_contract, Mapping) or dict(recorded_contract) != (
        rebuilt_contract
    ):
        raise FormalLaunchError(
            "checkpoint saved training contract is absent or inconsistent"
        )
    if int(payload.get("optimizer_updates", -1)) < spec.expected_updates:
        strict_resume = validate_strict_resume_checkpoint_payload(
            payload,
            saved_args,
            checkpoint_path=path,
        )
        if (
            strict_resume.get("phase") != spec.phase
            or strict_resume.get("optimizer_updates")
            != payload.get("optimizer_updates")
        ):
            raise FormalLaunchError(
                "partial checkpoint failed strict same-phase resume validation"
            )

    audit = audit_checkpoint_payload(payload, checkpoint_path=path)
    if (
        audit.get("schema") != CHECKPOINT_AUDIT_SCHEMA
        or audit.get("status") != "passed"
        or audit.get("phase") != spec.phase
        or audit.get("optimizer_updates") != payload.get("optimizer_updates")
    ):
        raise FormalLaunchError("checkpoint failed its dense-duty ownership audit")
    current_source = build_source_closure(spec.config, repo_root=REPO_ROOT)
    saved_source = validate_source_closure(
        saved_args.get(SOURCE_CLOSURE_ARG)
    )
    if saved_source != current_source:
        raise FormalLaunchError(
            "checkpoint source closure differs from the current formal source"
        )
    code_source_sha256 = current_source["code"]["sha256"]
    if (
        spec.phase == "rank"
        and payload.get("optimizer_updates") == spec.expected_updates
    ):
        scorer_init = saved_args.get("stage_b_v15_scorer_init_audit")
        if (
            not isinstance(scorer_init, Mapping)
            or scorer_init.get("schema") != "stage_b_v15_scorer_init/v1"
            or scorer_init.get("status") != "applied"
            or scorer_init.get("source_sha256") != TEXT_CHECKPOINT_SHA256
            or scorer_init.get("resolved_source_path")
            != str(TEXT_CHECKPOINT.resolve(strict=True))
            or scorer_init.get("source_decoder_num_layers") != 6
            or scorer_init.get("loaded_num_layers") != 6
        ):
            raise FormalLaunchError(
                "terminal rank checkpoint lacks its exact OGC scorer initialization"
            )
        validate_rank_handoff_audit(
            audit,
            execution_scope="formal",
            rank_dataset_sha256=RANK_DATASET_SHA256,
            required_optimizer_updates=RANK_UPDATES,
            code_source_sha256=code_source_sha256,
        )
    elif spec.phase == "confidence":
        validate_rank_handoff_audit(
            saved_args.get("stage_b_dense_duty_rank_source_checkpoint_audit"),
            execution_scope="formal",
            rank_dataset_sha256=RANK_DATASET_SHA256,
            required_optimizer_updates=RANK_UPDATES,
            code_source_sha256=code_source_sha256,
        )


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    value = path.stat()
    if not stat.S_ISREG(value.st_mode):
        raise FormalLaunchError(f"checkpoint is not a regular file: {path}")
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _stable_regular_file_sha256(path: Path) -> str:
    path = Path(path).expanduser()
    if path.is_symlink():
        raise FormalLaunchError(f"formal lineage file must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    identity_before = _file_identity(resolved)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if _file_identity(resolved) != identity_before:
        raise FormalLaunchError(
            f"formal lineage file changed while it was being hashed: {resolved}"
        )
    return digest.hexdigest()


def validate_launch_inputs(
    action: LaunchAction, root: Path = FORMAL_ROOT
) -> dict[str, Any]:
    rank_spec, confidence_spec = formal_phase_specs(root)
    if action in {LaunchAction.START_RANK, LaunchAction.RESUME_RANK}:
        spec = rank_spec
    elif action in {
        LaunchAction.START_CONFIDENCE,
        LaunchAction.RESUME_CONFIDENCE,
    }:
        spec = confidence_spec
    else:
        raise FormalLaunchError(
            f"action {action.value} has no formal training inputs"
        )
    if not PYTHON.is_file() or not os.access(PYTHON, os.X_OK):
        raise FormalLaunchError(f"fixed Python interpreter is unavailable: {PYTHON}")
    try:
        source_closure = validate_source_closure(
            build_source_closure(spec.config, repo_root=REPO_ROOT)
        )
    except (OSError, RuntimeError) as exc:
        raise FormalLaunchError(
            f"formal source closure preflight failed: {exc}"
        ) from exc
    lineage = (
        (spec.dataset, spec.dataset_sha256, f"{spec.phase} dataset config"),
        (STAGE_A_CHECKPOINT, STAGE_A_SHA256, "Stage-A initializer"),
        (TEXT_CHECKPOINT, TEXT_CHECKPOINT_SHA256, "OGC text initializer"),
        (TN_MANIFEST, TN_MANIFEST_SHA256, "traceable TN manifest"),
    )
    records = []
    for path, expected_sha256, label in lineage:
        observed_sha256 = _stable_regular_file_sha256(path)
        if observed_sha256 != expected_sha256:
            raise FormalLaunchError(
                f"{label} SHA256 drifted: expected={expected_sha256}, "
                f"observed={observed_sha256}"
            )
        records.append(
            {
                "label": label,
                "path": str(Path(path).resolve(strict=True)),
                "sha256": observed_sha256,
            }
        )
    return {"source_closure": source_closure, "lineage": records}


def inspect_phase_directory(
    spec: PhaseSpec,
    *,
    checkpoint_loader: Callable[[Path], Mapping[str, Any]] = _torch_load_checkpoint,
    checkpoint_auditor: Callable[[Mapping[str, Any], PhaseSpec, Path], None]
    | None = _audit_checkpoint,
) -> PhaseInspection:
    output = spec.output
    if not output.exists():
        return PhaseInspection(spec.phase, PhaseStatus.FRESH)
    if output.is_symlink() or not output.is_dir():
        return _invalid(spec, "phase output is not a real directory")
    entries = list(output.iterdir())
    if not entries:
        return PhaseInspection(spec.phase, PhaseStatus.FRESH)
    checkpoint = spec.checkpoint
    stale_temporary = [
        item.name
        for item in entries
        if item.name.startswith(f".{CHECKPOINT_NAME}.") and item.name.endswith(".tmp")
    ]
    if stale_temporary:
        return _invalid(spec, f"phase contains unpublished checkpoint bytes: {stale_temporary}")
    if not checkpoint.exists():
        return _invalid(
            spec,
            "phase output is non-empty but has no atomic checkpoint_iter.pth",
        )
    if checkpoint.is_symlink():
        return _invalid(spec, "canonical checkpoint must not be a symlink")
    try:
        identity_before = _file_identity(checkpoint)
        payload = checkpoint_loader(checkpoint)
        inspection = classify_checkpoint_payload(
            payload, spec, checkpoint_path=checkpoint
        )
        if inspection.status is PhaseStatus.INVALID:
            return inspection
        if checkpoint_auditor is not None:
            checkpoint_auditor(payload, spec, checkpoint)
        if _file_identity(checkpoint) != identity_before:
            return _invalid(spec, "checkpoint changed while it was being verified")
        return inspection
    except Exception as exc:
        return _invalid(spec, f"checkpoint verification failed: {type(exc).__name__}: {exc}")


def decide_action(
    rank: PhaseInspection, confidence: PhaseInspection
) -> tuple[LaunchAction, str]:
    if rank.status is PhaseStatus.INVALID or confidence.status is PhaseStatus.INVALID:
        return LaunchAction.INVALID, "at least one phase has invalid evidence"
    states = (rank.status, confidence.status)
    legal = {
        (PhaseStatus.FRESH, PhaseStatus.FRESH): LaunchAction.START_RANK,
        (PhaseStatus.PARTIAL, PhaseStatus.FRESH): LaunchAction.RESUME_RANK,
        (PhaseStatus.TERMINAL, PhaseStatus.FRESH): LaunchAction.START_CONFIDENCE,
        (PhaseStatus.TERMINAL, PhaseStatus.PARTIAL): LaunchAction.RESUME_CONFIDENCE,
        (PhaseStatus.TERMINAL, PhaseStatus.TERMINAL): LaunchAction.COMPLETE,
    }
    action = legal.get(states)
    if action is None:
        return (
            LaunchAction.INVALID,
            "illegal cross-phase state: "
            f"rank={states[0].value}, confidence={states[1].value}",
        )
    return action, ""


def inspect_experiment(
    root: Path = FORMAL_ROOT,
    *,
    checkpoint_loader: Callable[[Path], Mapping[str, Any]] = _torch_load_checkpoint,
    checkpoint_auditor: Callable[[Mapping[str, Any], PhaseSpec, Path], None]
    | None = _audit_checkpoint,
) -> ExperimentInspection:
    root = Path(root)
    rank_spec, confidence_spec = formal_phase_specs(root)
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            invalid = _invalid(rank_spec, "formal root is not a real directory")
            fresh = PhaseInspection("confidence", PhaseStatus.FRESH)
            return ExperimentInspection(invalid, fresh, LaunchAction.INVALID, invalid.detail)
        unexpected = sorted(
            child.name for child in root.iterdir() if child.name not in {"rank", "confidence"}
        )
        if unexpected:
            invalid = _invalid(rank_spec, f"formal root has unexpected entries: {unexpected}")
            fresh = PhaseInspection("confidence", PhaseStatus.FRESH)
            return ExperimentInspection(invalid, fresh, LaunchAction.INVALID, invalid.detail)
    rank = inspect_phase_directory(
        rank_spec,
        checkpoint_loader=checkpoint_loader,
        checkpoint_auditor=checkpoint_auditor,
    )
    confidence = inspect_phase_directory(
        confidence_spec,
        checkpoint_loader=checkpoint_loader,
        checkpoint_auditor=checkpoint_auditor,
    )
    action, detail = decide_action(rank, confidence)
    return ExperimentInspection(rank, confidence, action, detail)


def build_training_command(
    action: LaunchAction, root: Path = FORMAL_ROOT
) -> tuple[str, ...]:
    rank_spec, confidence_spec = formal_phase_specs(root)
    if action in {LaunchAction.START_RANK, LaunchAction.RESUME_RANK}:
        spec = rank_spec
    elif action in {LaunchAction.START_CONFIDENCE, LaunchAction.RESUME_CONFIDENCE}:
        spec = confidence_spec
    else:
        raise FormalLaunchError(f"action {action.value} does not launch training")

    command = [
        str(PYTHON),
        str(MAIN),
        "--config_file",
        str(spec.config.resolve()),
        "--datasets",
        str(spec.dataset.resolve()),
        "--output_dir",
        str(spec.output.resolve()),
        "--device",
        "cuda",
        "--seed",
        str(SEED),
        "--num_workers",
        str(NUM_WORKERS),
        "--prefetch_factor",
        str(PREFETCH_FACTOR),
        "--pin_memory",
        "--no_persistent_workers",
        "--mp_sharing_strategy",
        "file_system",
        "--min_nofile",
        "65536",
        "--world_size",
        "1",
        "--gradient_accumulation_steps",
        str(GRADIENT_ACCUMULATION_STEPS),
        "--iter_checkpoint_interval",
        str(ITER_CHECKPOINT_INTERVAL),
        "--max_train_iters",
        str(spec.expected_updates),
        "--amp",
        "--save_log",
        "--note",
        f"formal_dense_duty_{spec.phase}_seed{SEED}",
    ]
    if action is LaunchAction.START_RANK:
        command.extend(("--pretrain_model_path", str(STAGE_A_CHECKPOINT)))
    elif action is LaunchAction.RESUME_RANK:
        command.extend(("--resume", str(rank_spec.checkpoint.resolve())))
    elif action is LaunchAction.START_CONFIDENCE:
        command.extend(("--pretrain_model_path", str(rank_spec.checkpoint.resolve())))
    else:
        command.extend(("--resume", str(confidence_spec.checkpoint.resolve())))
    if "--options" in command or any(item.startswith("--options=") for item in command):
        raise AssertionError("formal dense-duty command must never contain --options")
    return tuple(command)


@contextlib.contextmanager
def exclusive_controller_lock(path: Path = LOCK_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="ascii") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FormalLaunchError(f"another formal dense-duty controller holds {path}") from exc
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid={os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
            yield handle.fileno()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(key, None)
    for key in tuple(environment):
        if key.startswith("SLURM_"):
            environment.pop(key)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    return environment


def run_training_child(
    command: Sequence[str], *, lock_fd: int | None = None
) -> ChildResult:
    if lock_fd is not None and (type(lock_fd) is not int or lock_fd < 0):
        raise FormalLaunchError("training child received an invalid controller lock FD")
    process: subprocess.Popen[Any] | None = None
    forwarded: list[int] = []
    previous: dict[int, Any] = {}

    def forward(signum: int, _frame: Any) -> None:
        if process is not None and process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signum)
            forwarded.append(int(signum))

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, forward)
    try:
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            env=_subprocess_environment(),
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
            pass_fds=(() if lock_fd is None else (lock_fd,)),
        )
        try:
            returncode = process.wait()
        except BaseException:
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)
            raise
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return ChildResult(int(returncode), tuple(forwarded))


def _print_state(state: ExperimentInspection, command: Sequence[str] | None = None) -> None:
    payload = state.as_dict()
    if command is not None:
        payload["command"] = list(command)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def run_controller(
    *,
    root: Path = FORMAL_ROOT,
    dry_run: bool = False,
    lock_fd: int | None = None,
) -> int:
    while True:
        state = inspect_experiment(root)
        if state.action is LaunchAction.INVALID:
            _print_state(state)
            raise FormalLaunchError(state.detail or "formal experiment state is invalid")
        if state.action is LaunchAction.COMPLETE:
            _print_state(state)
            return 0
        validate_launch_inputs(state.action, root)
        command = build_training_command(state.action, root)
        _print_state(state, command)
        if dry_run:
            return 0

        launched_action = state.action
        child = run_training_child(command, lock_fd=lock_fd)
        observed = inspect_experiment(root)
        _print_state(observed)
        expected_phase_terminal = (
            observed.rank.status is PhaseStatus.TERMINAL
            if launched_action in {LaunchAction.START_RANK, LaunchAction.RESUME_RANK}
            else observed.confidence.status is PhaseStatus.TERMINAL
        )
        if child.forwarded_signals:
            return 128 + child.forwarded_signals[-1]
        if not expected_phase_terminal or observed.action is LaunchAction.INVALID:
            raise FormalLaunchError(
                "training subprocess exited without a verified terminal phase "
                f"checkpoint (returncode={child.returncode})"
            )
        # A terminal atomic checkpoint is the fact source even if cleanup after
        # publishing it returned a non-zero process status.  Re-enter the state
        # machine to transition phases or report full completion.


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--status", action="store_true", help="inspect only")
    modes.add_argument("--dry-run", action="store_true", help="print the next fixed command")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if any(item == "--options" or item.startswith("--options=") for item in raw_argv):
        print("[FAIL] formal dense-duty launcher forbids --options", file=sys.stderr)
        return 2
    args = build_parser().parse_args(raw_argv)
    try:
        with exclusive_controller_lock() as lock_fd:
            if args.status:
                state = inspect_experiment()
                _print_state(state)
                return 2 if state.action is LaunchAction.INVALID else 0
            return run_controller(dry_run=args.dry_run, lock_fd=lock_fd)
    except (FileNotFoundError, OSError, ValueError, FormalLaunchError) as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
