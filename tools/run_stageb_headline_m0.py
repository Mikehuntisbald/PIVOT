#!/usr/bin/env python3
"""Run the fixed Stage-B M0 and compute-matched M0N training contracts.

The runner is an additive control plane around the unchanged ``main.py``.  It
implements the paper-runner CLI consumed by ``run_stageb_serial_matrix_queue``
and deliberately executes one training attempt at a time.  A graceful signal
may continue only through an explicitly authorized, complete, same-run,
mid-epoch checkpoint.  Epoch-boundary recovery is rejected because the current
training entry point does not replay its runtime RNG exactly on that branch.

Formal runs are fixed to B40/U23532, workers=2, prefetch=1, AMP, gradient
accumulation 1, 500-update checkpoints, and seeds 17/42/73.  M0 and M0N must be
placed in separate serial queues in that exact seed order.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import re
import runpy
import shlex
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_stageb_paper_ablation_matrices as paper_launcher
from tools import run_stageb_token_ablation_matrix as token_launcher
from tools import stageb_evaluation_source_contracts as source_contracts


RUN_IDS = (
    *source_contracts.M0_CONTRACT.dedicated_queue_run_ids,
    *source_contracts.M0N_CONTRACT.dedicated_queue_run_ids,
)
CONTRACTS = {
    source_contracts.M0_CONTRACT.id: source_contracts.M0_CONTRACT,
    source_contracts.M0N_CONTRACT.id: source_contracts.M0N_CONTRACT,
}

DEFAULT_STAGE_A_INIT = Path("/media/haoyi/T9/gdino/checkpoint0004.pth")
DEFAULT_STAGE_A_SHA256 = (
    "7f4cdd0ab94fc74d46fc7658b2014588a06d7de44be2c1d482ed073bbd7ca1b1"
)
FORBIDDEN_B58_SHA256 = (
    "b58e5209dc07dbffb2e5ed3d792e0db70c3306ec2ed24389693b4aeeebab1157"
)
DEFAULT_PYTHON = source_contracts.DEFAULT_PYTHON
DEFAULT_DATA_ROOT = source_contracts.DEFAULT_DATA_ROOT
DEFAULT_DATASET = REPO_ROOT / "config/datasets_stageb_v21_single_edit_train.json"
DEFAULT_OUTPUT_ROOT = source_contracts.FORMAL_TRAINING_ROOT
DEFAULT_ORCHESTRATION_ROOT = (
    REPO_ROOT / "outputs/paper_cvpr_v1/orchestration/headline_m0"
)

FORMAL_BATCH_SIZE = 40
FORMAL_UPDATES = 23_532
FORMAL_BATCH_SLOTS = 941_280
FORMAL_CHECKPOINT_INTERVAL = 500
FORMAL_NUM_WORKERS = 2
FORMAL_PREFETCH_FACTOR = 1
FORMAL_GRADIENT_DIAGNOSTIC_INTERVAL = 100
FORMAL_DATALOADER_MICROBATCHES = 8_388
FORMAL_TRAIN_ROWS = 335_523
FORMAL_FINAL_EPOCH = 2
FORMAL_FINAL_ITERATION = 6_756
FORMAL_OPTIMIZER_STATE_COUNT = 94
FORMAL_TELEMETRY_INTERVAL_SECONDS = 1
MILESTONE_UPDATES = (1_000, 4_000, 8_000, FORMAL_UPDATES)

TRAINING_SEQUENCE_SCHEMA = "pivot.stageb.paper_ablation_run_launch/v1"
TRAINING_PHASE_SCHEMA = "pivot.stageb.paper_ablation_phase_launch/v1"
POSTFLIGHT_SCHEMA = "pivot.stageb.paper_ablation_phase_postflight/v1"
DETACHED_SCHEMA = "pivot.stageb.headline_m0_detached_launch/v1"
ATTEMPT_SCHEMA = "pivot.stageb.headline_m0_training_attempt/v1"
ATTEMPT_TELEMETRY_SCHEMA = "pivot.stageb.headline_m0_attempt_telemetry/v1"
FULL_RUN_TELEMETRY_SCHEMA = "pivot.stageb.headline_m0_full_run_telemetry/v1"
ANCESTRY_SCHEMA = "pivot.stageb.headline_m0_model_state_ancestry/v1"
STABLE_CLOSURE_SCHEMA = "pivot.stageb.headline_m0_stable_input_closure/v1"
RESUME_REQUEST_SCHEMA = "pivot.stageb.headline_m0_resume_request/v1"
TRAINING_QUEUE_CONTRACT_SCHEMA = (
    "pivot.stageb.headline_m0_training_queue_contract/v1"
)
COMPLETED_TRAINING_VERIFICATION_SCHEMA = (
    "pivot.stageb.headline_m0_completed_training_verification/v1"
)
COMPLETED_TRAINING_SOURCE_SNAPSHOT_SCHEMA = (
    "pivot.stageb.headline_m0_completed_training_verifier_sources/v1"
)
COMPLETED_TRAINING_EVIDENCE_SNAPSHOT_SCHEMA = (
    "pivot.stageb.headline_m0_completed_training_evidence/v1"
)
TRAINING_QUEUE_EXTENSION_KEY = "headline_m0"

COMPLETE_STATE_COMPONENTS = {
    "model": True,
    "criterion": True,
    "optimizer": True,
    "lr_scheduler": True,
    "scaler": True,
    "epoch": True,
    "iteration": True,
    "optimizer_updates": True,
    "epoch_finished": True,
    "rng_state": True,
    "epoch_rng_state": True,
    "args": True,
}


class HeadlineM0Error(RuntimeError):
    """A formal M0/M0N launch or evidence contract was violated."""


@dataclass(frozen=True)
class Runtime:
    python: Path
    stage_a_init: Path
    dataset: Path
    output_root: Path
    data_root: Path
    batch_size: int
    max_train_iters: int
    iter_checkpoint_interval: int
    num_workers: int
    prefetch_factor: int
    omp_num_threads: int
    min_nofile: int
    cuda_visible_devices: str
    mp_sharing_strategy: str
    gradient_diagnostic_interval: int
    telemetry_interval_seconds: int
    pin_memory: bool
    persistent_workers: bool
    gradient_accumulation_steps: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(
    path: Path,
    cache: token_launcher.HashCache | None = None,
    *,
    role: str | None = None,
) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    stat = path.stat()
    record: dict[str, Any] = {
        "path": str(path),
        "sha256": cache.digest(path) if cache is not None else _sha256_file(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if role is not None:
        record["role"] = role
    return record


def _compact_file_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(Path(str(record["path"])).resolve(strict=True)),
        "sha256": str(record["sha256"]),
        "size_bytes": int(record["size_bytes"]),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    data = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"
    try:
        with temporary.open("x", encoding="ascii") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _write_json_no_replace(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"
    try:
        with path.open("x", encoding="ascii") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise HeadlineM0Error(f"refusing to overwrite immutable artifact: {path}") from exc
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HeadlineM0Error(f"{label} is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise HeadlineM0Error(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HeadlineM0Error(f"{label} must be a JSON object: {path}")
    return value


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise HeadlineM0Error(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise HeadlineM0Error(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_exact(name: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise HeadlineM0Error(
            f"formal M0/M0N requires {name}={expected!r}, got {observed!r}"
        )


def runtime_from_environment() -> Runtime:
    python = token_launcher._resolve_executable(
        os.environ.get("PIVOT_PYTHON", str(DEFAULT_PYTHON))
    )
    stage_a = Path(
        os.environ.get("PIVOT_STAGE_A_INIT", str(DEFAULT_STAGE_A_INIT))
    ).expanduser().resolve(strict=True)
    scorer = Path(
        os.environ.get("PIVOT_SCORER_WARMSTART", str(DEFAULT_STAGE_A_INIT))
    ).expanduser().resolve(strict=True)
    data_root = Path(
        os.environ.get(
            "PIVOT_DATA_ROOT", os.environ.get("DATA_ROOT", str(DEFAULT_DATA_ROOT))
        )
    ).expanduser().resolve(strict=True)
    visible = os.environ.get(
        "PIVOT_CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    ).strip()
    runtime = Runtime(
        python=python,
        stage_a_init=stage_a,
        dataset=DEFAULT_DATASET.resolve(strict=True),
        output_root=DEFAULT_OUTPUT_ROOT.resolve(strict=False),
        data_root=data_root,
        batch_size=_env_int("PIVOT_BATCH_SIZE", FORMAL_BATCH_SIZE),
        max_train_iters=_env_int("PIVOT_MAX_TRAIN_ITERS", FORMAL_UPDATES),
        iter_checkpoint_interval=_env_int(
            "PIVOT_ITER_CHECKPOINT_INTERVAL", FORMAL_CHECKPOINT_INTERVAL
        ),
        num_workers=_env_int("PIVOT_NUM_WORKERS", FORMAL_NUM_WORKERS, minimum=0),
        prefetch_factor=_env_int(
            "PIVOT_PREFETCH_FACTOR", FORMAL_PREFETCH_FACTOR
        ),
        omp_num_threads=_env_int("PIVOT_OMP_NUM_THREADS", 8),
        min_nofile=_env_int("PIVOT_MIN_NOFILE", 65_536, minimum=0),
        cuda_visible_devices=visible,
        mp_sharing_strategy=os.environ.get(
            "PIVOT_MP_SHARING_STRATEGY", "file_system"
        ),
        gradient_diagnostic_interval=_env_int(
            "PIVOT_GRADIENT_DIAGNOSTIC_INTERVAL",
            FORMAL_GRADIENT_DIAGNOSTIC_INTERVAL,
        ),
        telemetry_interval_seconds=FORMAL_TELEMETRY_INTERVAL_SECONDS,
        pin_memory=True,
        persistent_workers=False,
        gradient_accumulation_steps=1,
    )
    _require_exact(
        "controller Python",
        Path(sys.executable).resolve(strict=True),
        DEFAULT_PYTHON.resolve(strict=True),
    )
    _require_exact("python", runtime.python, DEFAULT_PYTHON.resolve(strict=True))
    _require_exact("Stage-A initializer", runtime.stage_a_init, DEFAULT_STAGE_A_INIT)
    _require_exact("scorer warm-start", scorer, DEFAULT_STAGE_A_INIT)
    _require_exact("data root", runtime.data_root, DEFAULT_DATA_ROOT.resolve(strict=True))
    _require_exact("batch size", runtime.batch_size, FORMAL_BATCH_SIZE)
    _require_exact("optimizer updates", runtime.max_train_iters, FORMAL_UPDATES)
    _require_exact(
        "iteration checkpoint interval",
        runtime.iter_checkpoint_interval,
        FORMAL_CHECKPOINT_INTERVAL,
    )
    _require_exact("num workers", runtime.num_workers, FORMAL_NUM_WORKERS)
    _require_exact(
        "prefetch factor", runtime.prefetch_factor, FORMAL_PREFETCH_FACTOR
    )
    _require_exact(
        "gradient diagnostic interval",
        runtime.gradient_diagnostic_interval,
        FORMAL_GRADIENT_DIAGNOSTIC_INTERVAL,
    )
    _require_exact("OMP thread count", runtime.omp_num_threads, 8)
    _require_exact("minimum open-file limit", runtime.min_nofile, 65_536)
    _require_exact("multiprocessing sharing", runtime.mp_sharing_strategy, "file_system")
    if not visible or "," in visible:
        raise HeadlineM0Error(
            "formal M0/M0N requires exactly one CUDA-visible device"
        )
    if _sha256_file(runtime.stage_a_init) != DEFAULT_STAGE_A_SHA256:
        raise HeadlineM0Error("Stage-A checkpoint0004 SHA-256 drifted")
    return runtime


def _contract(contract_id: str) -> source_contracts.FormalPaperRunContract:
    try:
        contract = CONTRACTS[contract_id]
    except KeyError as exc:
        raise HeadlineM0Error(f"unknown formal contract {contract_id!r}") from exc
    expected = {
        "runner": "tools/run_stageb_headline_m0.py",
        "dataset": "config/datasets_stageb_v21_single_edit_train.json",
        "architecture_objective": "S2F",
        "compute_contract": "b58_successful_update_batch_slot_matched",
        "phase_ids": ("joint",),
        "batch_size": FORMAL_BATCH_SIZE,
        "optimizer_updates": FORMAL_UPDATES,
        "successful_update_batch_slots": FORMAL_BATCH_SLOTS,
        "final_phase_updates": FORMAL_UPDATES,
        "iter_checkpoint_interval": FORMAL_CHECKPOINT_INTERVAL,
        "seeds": (17, 42, 73),
    }
    for key, value in expected.items():
        _require_exact(f"{contract.id} contract {key}", getattr(contract, key), value)
    _require_exact(
        f"{contract.id} contributing phases",
        contract.contributing_phase_updates,
        (("joint", FORMAL_UPDATES),),
    )
    return contract


def _parse_run_id(value: str) -> tuple[source_contracts.FormalPaperRunContract, int]:
    try:
        raw_id, raw_seed = value.split(":", 1)
        seed = int(raw_seed)
    except (ValueError, TypeError) as exc:
        raise HeadlineM0Error(f"invalid run ID {value!r}; expected M0:17") from exc
    contract = _contract(raw_id.upper())
    canonical = f"{contract.id}:{seed}"
    if canonical != value.upper() or seed not in contract.seeds:
        raise HeadlineM0Error(f"run ID is outside the formal registry: {value!r}")
    return contract, seed


def output_directory(
    runtime: Runtime,
    contract: source_contracts.FormalPaperRunContract,
    seed: int,
) -> Path:
    expected = contract.canonical_training_root(seed)
    observed = runtime.output_root / contract.id / f"seed{seed}"
    if observed.resolve(strict=False) != expected:
        raise HeadlineM0Error(
            f"formal output root drifted: expected {expected}, got {observed}"
        )
    return observed


def _config_path(contract: source_contracts.FormalPaperRunContract) -> Path:
    return (REPO_ROOT / contract.config).resolve(strict=True)


def _validate_config(contract: source_contracts.FormalPaperRunContract) -> dict[str, Any]:
    path = _config_path(contract)
    try:
        values = runpy.run_path(str(path))
    except (OSError, ImportError, RuntimeError, SyntaxError) as exc:
        raise HeadlineM0Error(f"cannot evaluate {contract.id} config: {exc}") from exc
    expected = {
        "stage_b_v25_main_id": contract.id,
        "stage_b_v25_compute_contract": "b58_successful_update_batch_slot_matched",
        "stage_b_v25_budget_unit": (
            "successful_optimizer_update_global_batch_slots"
        ),
        "stage_b_v25_successful_update_batch_slots": FORMAL_BATCH_SLOTS,
        "stage_b_v25_initializer_contract": "same_stage_a_model_and_scorer_no_b58",
        "stage_b_v25_strict_resume": True,
        "stage_b_v22_table_id": "S2F",
        "stage_b_v22_objective_fidelity": "full_v19_base_plus_gate_objective",
        "stage_b_v21_token_objective": (
            "edit_bce" if contract.id == "M0" else "targetlocal_allneg_bce"
        ),
        "stage_b_v11_predicate_tn_rank_weight": 1.0,
        "stage_b_v21_token_weight": 1.0,
        "stage_b_v21_token_positive_weight": 1.0,
        "stage_b_v21_token_shared_weight": 0.25,
        "stage_b_v21_token_edit_weight": 1.0,
        "stage_b_v21_allow_legacy_token_diff_fallback": False,
        "stage_b_v15_separate_grad_clip": True,
        "stage_b_v22_gradient_diagnostic_interval": 100,
        "batch_size": 40,
        "epochs": 8,
        "lr_drop": 4,
        "save_checkpoint_interval": 1,
        "onecyclelr": False,
        "multi_step_lr": False,
        "lr": 2e-5,
        "stage_b_v15_validity_lr": 5e-4,
        "weight_decay": 1e-4,
        "clip_max_norm": 0.1,
    }
    for key, expected_value in expected.items():
        if values.get(key) != expected_value:
            raise HeadlineM0Error(
                f"{contract.id} config {key} expected {expected_value!r}, "
                f"got {values.get(key)!r}"
            )
    scorer = Path(str(values.get("stage_b_v15_scorer_init_checkpoint", "")))
    if scorer.resolve(strict=True) != DEFAULT_STAGE_A_INIT:
        raise HeadlineM0Error(f"{contract.id} scorer source is not Stage-A checkpoint0004")
    if contract.id == "M0N":
        controls = {
            "stage_b_v25_control_of": "M0",
            "stage_b_v25_headline_eligible": False,
            "stage_b_v25_matrix_validation_only": True,
            "stage_b_v25_comparison_claim": (
                "full_token_objective_control_not_labels_only"
            ),
            "stage_b_v25_token_objective_scope": (
                "target_local_positive_and_all_negative_token_logits"
            ),
        }
        for key, expected_value in controls.items():
            if values.get(key) != expected_value:
                raise HeadlineM0Error(
                    f"M0N control metadata {key} drifted: {values.get(key)!r}"
                )
    return values


def _expand_dataset_path(raw: str, *, runtime: Runtime) -> Path:
    expanded = os.path.expandvars(raw.replace("${DATA_ROOT}", str(runtime.data_root)))
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = runtime.dataset.parent / path
    legacy_root = Path("/home/user/PIVOT")
    try:
        relative = path.relative_to(legacy_root)
    except ValueError:
        return path.resolve(strict=False)
    return (REPO_ROOT / relative).resolve(strict=False)


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _validate_dataset(runtime: Runtime) -> tuple[dict[str, Any], list[Path]]:
    payload = _read_json(runtime.dataset, label="formal M0 dataset manifest")
    train = payload.get("train")
    if not isinstance(train, list) or len(train) != 4 or payload.get("val") != []:
        raise HeadlineM0Error("M0 dataset requires four train sources and val=[]")
    weights = [float(value.get("mix_weight", 1.0)) for value in train]
    if weights != [1.0, 1.0, 1.0, 3.0]:
        raise HeadlineM0Error(f"M0 dataset mix weights drifted: {weights}")
    tn = train[-1]
    expected_tn = {
        "source": "sam3_tn_pair",
        "require_global_tn_verified": False,
        "require_single_edit_token_provenance": True,
        "paper_table_b_id": "D3",
        "paper_tn_scope": "proposal_covered_verified",
    }
    for key, expected in expected_tn.items():
        if tn.get(key) != expected:
            raise HeadlineM0Error(
                f"M0 TN source {key} expected {expected!r}, got {tn.get(key)!r}"
            )
    source_files: set[Path] = set()
    annotations: list[Path] = []
    source_paths: list[dict[str, Any]] = []
    for index, source in enumerate(train):
        if not isinstance(source, Mapping) or source.get("dataset_mode") != "patch_episode":
            raise HeadlineM0Error(f"M0 dataset source {index} is not patch_episode")
        for key in (
            "anno",
            "canonical_classes_json",
            "support_patch_tsv",
            "paper_contract_audit",
        ):
            raw = source.get(key)
            if not isinstance(raw, str) or not raw.strip():
                continue
            resolved = _expand_dataset_path(raw, runtime=runtime)
            if not resolved.is_file():
                raise HeadlineM0Error(
                    f"dataset source {index}/{key} is not a file: {resolved}"
                )
            source_paths.append(
                {
                    "dataset_index": index,
                    "field": key,
                    "declared": raw,
                    "resolved": str(resolved),
                }
            )
            source_files.add(resolved.resolve(strict=True))
            if key == "anno":
                annotations.append(resolved.resolve(strict=True))
    row_counts = [_line_count(path) for path in annotations]
    if sum(row_counts) != FORMAL_TRAIN_ROWS:
        raise HeadlineM0Error(
            f"M0 train rows expected {FORMAL_TRAIN_ROWS}, got {sum(row_counts)}"
        )
    if sum(row_counts) // FORMAL_BATCH_SIZE != FORMAL_DATALOADER_MICROBATCHES:
        raise HeadlineM0Error("M0 dataloader length no longer equals 8388")
    return (
        {
            "train_source_count": 4,
            "positive_source_count": 3,
            "tn_source_count": 1,
            "mix_weights": weights,
            "expected_tn_draw_fraction": 0.5,
            "tn_source": expected_tn,
            "annotation_row_counts": row_counts,
            "total_train_rows": sum(row_counts),
            "drop_last_microbatches_per_epoch": FORMAL_DATALOADER_MICROBATCHES,
            "source_paths": source_paths,
        },
        sorted(source_files, key=lambda value: str(value)),
    )


def _package_initializers(path: Path) -> set[Path]:
    path = path.resolve(strict=True)
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError:
        return set()
    result: set[Path] = set()
    current = REPO_ROOT
    for part in relative.parts[:-1]:
        current = current / part
        candidate = current / "__init__.py"
        if candidate.is_file():
            result.add(candidate.resolve(strict=True))
    return result


def _native_runtime_dependency_paths() -> list[Path]:
    """Bind the actual Python-ABI extension without claiming the cpython312 build."""

    native_root = (REPO_ROOT / "models/GroundingDINO/ops").resolve(strict=True)
    spec = importlib.util.find_spec("MultiScaleDeformableAttention")
    origin = getattr(spec, "origin", None) if spec is not None else None
    if not isinstance(origin, str) or not origin:
        raise HeadlineM0Error("MultiScaleDeformableAttention runtime is missing")
    actual = Path(origin).resolve(strict=True)
    ext_suffix = str(sysconfig.get_config_var("EXT_SUFFIX") or "")
    if not ext_suffix or not actual.name.endswith(ext_suffix):
        raise HeadlineM0Error(
            f"native extension {actual} does not match runtime ABI {ext_suffix!r}"
        )
    repo_extension = native_root / actual.name
    build_extension = native_root / "build" / "lib.linux-x86_64-cpython-311" / actual.name
    required = {
        actual,
        repo_extension.resolve(strict=True),
        build_extension.resolve(strict=True),
        (native_root / "setup.py").resolve(strict=True),
    }
    for pattern in ("src/**/*.cpp", "src/**/*.cu", "src/**/*.h"):
        required.update(path.resolve(strict=True) for path in native_root.glob(pattern))
    required.update(
        path.resolve(strict=True)
        for path in (native_root / "MultiScaleDeformableAttention.egg-info").glob("*")
        if path.is_file()
    )
    required.update(
        path.resolve(strict=True)
        for path in actual.parent.glob("MultiScaleDeformableAttention-*.egg-info/*")
        if path.is_file()
    )
    if any("cpython-312" in str(path) for path in required):
        raise HeadlineM0Error("cpython312 build metadata leaked into cpython311 closure")
    suffixes = {path.suffix for path in required}
    if not {".cpp", ".cu", ".h"}.issubset(suffixes):
        raise HeadlineM0Error("native source closure is incomplete")
    return sorted(required, key=lambda value: str(value))


def _repository_dependency_paths(
    contract: source_contracts.FormalPaperRunContract,
) -> list[Path]:
    from tools.stageb_dependency_audit import local_python_dependency_paths

    explicit = {
        Path(__file__).resolve(strict=True),
        (REPO_ROOT / "main.py").resolve(strict=True),
        (REPO_ROOT / "engine.py").resolve(strict=True),
        (REPO_ROOT / "datasets/patch_episode.py").resolve(strict=True),
        (REPO_ROOT / "models/GroundingDINO/groundingdino.py").resolve(strict=True),
        (REPO_ROOT / "models/GroundingDINO/stage_b_fixed_text_scorer.py").resolve(
            strict=True
        ),
        (REPO_ROOT / "models/GroundingDINO/stage_b_fixed_text_criterion.py").resolve(
            strict=True
        ),
        (REPO_ROOT / "tools/stageb_evaluation_source_contracts.py").resolve(
            strict=True
        ),
        (REPO_ROOT / "tools/run_stageb_paper_ablation_matrices.py").resolve(
            strict=True
        ),
        (REPO_ROOT / "tools/run_stageb_token_ablation_matrix.py").resolve(strict=True),
        _config_path(contract),
    }
    paths = set(
        local_python_dependency_paths(explicit, root=REPO_ROOT, include=explicit)
    )
    for path in tuple(paths):
        paths.update(_package_initializers(path))
    return sorted(paths, key=lambda value: str(value))


def _stable_input_closure(
    runtime: Runtime,
    contract: source_contracts.FormalPaperRunContract,
    cache: token_launcher.HashCache,
) -> dict[str, Any]:
    _validate_config(contract)
    dataset_contract, dataset_sources = _validate_dataset(runtime)
    records: list[dict[str, Any]] = [
        _file_record(runtime.stage_a_init, cache, role="stage_a_initializer"),
        _file_record(runtime.stage_a_init, cache, role="scorer_warmstart"),
        _file_record(runtime.python, cache, role="runtime_python"),
        _file_record(runtime.dataset, cache, role="dataset_manifest"),
    ]
    config_dependencies = token_launcher._config_dependencies(_config_path(contract))
    records.extend(
        _file_record(path, cache, role="config_dependency")
        for path in config_dependencies
    )
    records.extend(
        _file_record(path, cache, role="dataset_source") for path in dataset_sources
    )
    records.extend(
        _file_record(path, cache, role="repository_source")
        for path in _repository_dependency_paths(contract)
    )
    records.extend(
        _file_record(path, cache, role="native_runtime_dependency")
        for path in _native_runtime_dependency_paths()
    )
    protocol = REPO_ROOT / "docs/paper_cvpr_ablation_protocol.md"
    records.append(_file_record(protocol, cache, role="paper_protocol"))
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (str(record["path"]), str(record["role"]))
        previous = unique.setdefault(key, record)
        if previous != record:
            raise HeadlineM0Error(f"conflicting stable input record: {key}")
        if record["sha256"] == FORBIDDEN_B58_SHA256:
            raise HeadlineM0Error("b58 content is forbidden from M0/M0N ancestry")
    ordered = sorted(unique.values(), key=lambda value: (value["path"], value["role"]))
    normalized = [
        {**_compact_file_record(record), "roles": [str(record["role"])]}
        for record in ordered
    ]
    digest = _sha256_bytes(
        _canonical_json_bytes(
            {"schema": STABLE_CLOSURE_SCHEMA, "records": normalized}
        )
    )
    return {
        "records": ordered,
        "normalized_records": normalized,
        "digest": digest,
        "dataset_contract": dataset_contract,
        "config_dependency_count": len(config_dependencies),
    }


def _training_queue_contract_payload(
    runtime: Runtime,
    contract: source_contracts.FormalPaperRunContract,
    cache: token_launcher.HashCache,
) -> dict[str, Any]:
    closure = _stable_input_closure(runtime, contract, cache)
    return {
        "schema": TRAINING_QUEUE_CONTRACT_SCHEMA,
        "contract_id": contract.id,
        "ordered_run_ids": list(contract.dedicated_queue_run_ids),
        "runner": _compact_file_record(_file_record(Path(__file__), cache)),
        "controller_python": _compact_file_record(
            _file_record(runtime.python, cache)
        ),
        "stable_input_closure": {
            "schema": STABLE_CLOSURE_SCHEMA,
            "algorithm": "sha256_canonical_path_content_size_roles_v1",
            "digest": closure["digest"],
            "records": closure["normalized_records"],
        },
    }


def _validate_training_queue_contract_payload(
    value: Any,
    contract: source_contracts.FormalPaperRunContract,
) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "contract_id",
        "ordered_run_ids",
        "runner",
        "controller_python",
        "stable_input_closure",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise HeadlineM0Error("formal training queue contract field set drifted")
    if (
        value.get("schema") != TRAINING_QUEUE_CONTRACT_SCHEMA
        or value.get("contract_id") != contract.id
        or value.get("ordered_run_ids") != list(contract.dedicated_queue_run_ids)
    ):
        raise HeadlineM0Error("formal training queue contract identity drifted")
    expected_runner = _compact_file_record(_file_record(Path(__file__)))
    expected_python = _compact_file_record(_file_record(DEFAULT_PYTHON))
    if value.get("runner") != expected_runner:
        raise HeadlineM0Error("formal training queue runner record drifted")
    if value.get("controller_python") != expected_python:
        raise HeadlineM0Error("formal training queue Python record drifted")

    closure = value.get("stable_input_closure")
    if not isinstance(closure, Mapping) or set(closure) != {
        "schema",
        "algorithm",
        "digest",
        "records",
    }:
        raise HeadlineM0Error("formal training queue stable closure is invalid")
    records = closure.get("records")
    if not isinstance(records, list) or not records:
        raise HeadlineM0Error("formal training queue stable closure is empty")
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "sha256",
            "size_bytes",
            "roles",
        }:
            raise HeadlineM0Error(
                f"formal training queue stable input {index} is invalid"
            )
        roles = record.get("roles")
        if (
            not isinstance(roles, list)
            or not roles
            or not all(isinstance(role, str) and role for role in roles)
            or roles != sorted(set(roles))
        ):
            raise HeadlineM0Error(
                f"formal training queue stable input {index} roles drifted"
            )
        path = Path(str(record.get("path", ""))).expanduser().resolve(
            strict=False
        )
        sha256 = str(record.get("sha256", ""))
        size = record.get("size_bytes")
        if (
            not path.is_absolute()
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
            or type(size) is not int
            or size < 0
        ):
            raise HeadlineM0Error(
                f"formal training queue stable input {index} identity drifted"
            )
        normalized.append(
            {
                "path": str(path),
                "sha256": sha256,
                "size_bytes": size,
                "roles": list(roles),
            }
        )
    ordered = sorted(normalized, key=lambda record: (record["path"], record["roles"]))
    if normalized != ordered or len(
        {(record["path"], tuple(record["roles"])) for record in normalized}
    ) != len(normalized):
        raise HeadlineM0Error("formal training queue stable inputs are not canonical")
    digest = _sha256_bytes(
        _canonical_json_bytes(
            {"schema": STABLE_CLOSURE_SCHEMA, "records": normalized}
        )
    )
    if (
        closure.get("schema") != STABLE_CLOSURE_SCHEMA
        or closure.get("algorithm")
        != "sha256_canonical_path_content_size_roles_v1"
        or closure.get("digest") != digest
    ):
        raise HeadlineM0Error("formal training queue stable closure digest drifted")
    return dict(value)


def _require_manifest_matches_queue_contract(
    manifest: Mapping[str, Any], queue_contract: Mapping[str, Any]
) -> None:
    phases = manifest.get("phases")
    phase = phases[0] if isinstance(phases, list) and len(phases) == 1 else None
    closure = queue_contract.get("stable_input_closure")
    inputs = phase.get("inputs") if isinstance(phase, Mapping) else None
    if not isinstance(closure, Mapping) or not isinstance(inputs, Mapping):
        raise HeadlineM0Error("training plan lacks a queue-bound stable closure")
    normalized = _stable_closure_from_manifest(phase)
    if (
        manifest.get("stable_input_closure_digest") != closure.get("digest")
        or inputs.get("stable_closure_digest") != closure.get("digest")
        or normalized.get("digest") != closure.get("digest")
        or normalized.get("records") != closure.get("records")
    ):
        raise HeadlineM0Error(
            "training plan differs from the immutable three-seed queue closure"
        )


def _phase(contract: source_contracts.FormalPaperRunContract) -> dict[str, Any]:
    return {
        "phase_id": "joint",
        "config": contract.config,
        "updates": FORMAL_UPDATES,
        "diagnostic_interval": FORMAL_GRADIENT_DIAGNOSTIC_INTERVAL,
        "scorer_warmstart": True,
        "contributes_to_budget": True,
        "pretrain_source": "stage_a_initializer_or_same_run_resume",
    }


def build_command(
    runtime: Runtime,
    contract: source_contracts.FormalPaperRunContract,
    seed: int,
    output_dir: Path,
    *,
    resume_checkpoint: Path | None = None,
) -> list[str]:
    command = [
        str(runtime.python),
        str((REPO_ROOT / "main.py").resolve(strict=True)),
        "-c",
        str(_config_path(contract)),
        "--datasets",
        str(runtime.dataset),
        "--output_dir",
        str(output_dir.resolve(strict=False)),
    ]
    if resume_checkpoint is None:
        command.extend(("--pretrain_model_path", str(runtime.stage_a_init)))
    else:
        command.extend(("--resume", str(resume_checkpoint.resolve(strict=True))))
    command.extend(
        (
            "--seed",
            str(seed),
            "--num_workers",
            str(runtime.num_workers),
            "--prefetch_factor",
            str(runtime.prefetch_factor),
            "--mp_sharing_strategy",
            runtime.mp_sharing_strategy,
            "--min_nofile",
            str(runtime.min_nofile),
            "--max_train_iters",
            str(runtime.max_train_iters),
            "--iter_checkpoint_interval",
            str(runtime.iter_checkpoint_interval),
            "--gradient_accumulation_steps",
            "1",
            "--pin_memory",
            "--no_persistent_workers",
            "--world_size",
            "1",
            "--note",
            f"paper_cvpr_v1_{contract.id}_seed{seed}_joint",
            "--amp",
            "--save_log",
            "--options",
            f"batch_size={runtime.batch_size}",
            (
                "stage_b_v22_gradient_diagnostic_interval="
                f"{runtime.gradient_diagnostic_interval}"
            ),
            "skip_eval=True",
            f"stage_b_v15_scorer_init_checkpoint={runtime.stage_a_init}",
        )
    )
    if command.count("--resume") + command.count("--pretrain_model_path") != 1:
        raise AssertionError("M0 attempt must use exactly one initialization path")
    return command


def _runtime_payload(runtime: Runtime) -> dict[str, Any]:
    return {
        "python": str(runtime.python),
        "batch_size": runtime.batch_size,
        "phase_train_iters": runtime.max_train_iters,
        "total_paper_train_iters": runtime.max_train_iters,
        "max_train_iters": runtime.max_train_iters,
        "iter_checkpoint_interval": runtime.iter_checkpoint_interval,
        "num_workers": runtime.num_workers,
        "prefetch_factor": runtime.prefetch_factor,
        "omp_num_threads": runtime.omp_num_threads,
        "min_nofile": runtime.min_nofile,
        "cuda_visible_devices": runtime.cuda_visible_devices,
        "mp_sharing_strategy": runtime.mp_sharing_strategy,
        "amp": True,
        "gradient_accumulation_steps": 1,
        "gradient_diagnostic_interval": runtime.gradient_diagnostic_interval,
        "telemetry_interval_seconds": runtime.telemetry_interval_seconds,
        "pin_memory": True,
        "persistent_workers": False,
    }


def build_manifest(
    runtime: Runtime,
    contract: source_contracts.FormalPaperRunContract,
    seed: int,
    cache: token_launcher.HashCache,
) -> dict[str, Any]:
    if seed not in contract.seeds:
        raise HeadlineM0Error(f"unexpected {contract.id} seed {seed}")
    run_root = output_directory(runtime, contract, seed).resolve(strict=False)
    closure = _stable_input_closure(runtime, contract, cache)
    phase = _phase(contract)
    command = build_command(runtime, contract, seed, run_root)
    phase_manifest = {
        "schema": TRAINING_PHASE_SCHEMA,
        "status": "planned",
        "created_at_utc": _utc_now(),
        "run_id": f"{contract.id}:{seed}",
        "row": contract.expected_row(),
        "seed": seed,
        # The generic serial queue consumes this top-level key, while the paper
        # evaluator consumes the nested phase object.
        "phase_id": "joint",
        "phase": phase,
        "output_dir": str(run_root),
        "command": command,
        "command_shell": shlex.join(command),
        "runtime": _runtime_payload(runtime),
        "fixed_contract": {
            "architecture_objective": "S2F",
            "compute_contract": "b58_successful_update_batch_slot_matched",
            "successful_update_batch_slots": FORMAL_BATCH_SLOTS,
            "candidate_topk": 50,
            "positive_iou_threshold": 0.5,
            "negative_iou_threshold": 0.499,
            "token_objective": contract.expected_row().get(
                "token_objective", "edit_bce"
            ),
            "token_objective_scope": contract.token_objective_scope,
            "predicate_pair_rank_weight": 1.0,
            "stage_a_and_scorer_same_source": True,
            "b58_model_ancestry_forbidden": True,
            "dataset": closure["dataset_contract"],
            "optimizer_resume": "same_run_mid_epoch_signal_only",
        },
        "inputs": {
            "records": closure["records"],
            "stable_closure_digest": closure["digest"],
            "stable_closure_algorithm": (
                "sha256_canonical_path_content_size_roles_v1"
            ),
        },
    }
    return {
        "schema": TRAINING_SEQUENCE_SCHEMA,
        "status": "planned",
        "created_at_utc": _utc_now(),
        "repository_root": str(REPO_ROOT),
        "run_id": f"{contract.id}:{seed}",
        "row": contract.expected_row(),
        "seed": seed,
        "training_seeds_contract": list(contract.seeds),
        "output_dir": str(run_root),
        "output_dir_fresh_at_plan": not run_root.exists(),
        "equal_budget_contract": contract.expected_budget(),
        "stable_input_closure_digest": closure["digest"],
        "one_attempt_execution": True,
        "resume_policy": (
            "explicit_authorization_complete_same_run_mid_epoch_signal_only"
        ),
        "phases": [phase_manifest],
    }


def _iter_input_records(manifest: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    inputs = manifest.get("inputs")
    records = inputs.get("records") if isinstance(inputs, Mapping) else None
    if not isinstance(records, list) or not records:
        raise HeadlineM0Error("phase manifest has no stable input records")
    for record in records:
        if not isinstance(record, Mapping):
            raise HeadlineM0Error("phase manifest contains an invalid input record")
        yield record


def _verify_input_identities(manifest: Mapping[str, Any], *, rehash: bool) -> None:
    for record in _iter_input_records(manifest):
        path = Path(str(record.get("path", ""))).resolve(strict=True)
        stat = path.stat()
        if (
            int(record.get("size_bytes", -1)) != int(stat.st_size)
            or int(record.get("mtime_ns", -1)) != int(stat.st_mtime_ns)
        ):
            raise HeadlineM0Error(f"stable input identity changed: {path}")
        if rehash and _sha256_file(path) != record.get("sha256"):
            raise HeadlineM0Error(f"stable input SHA-256 changed: {path}")


def _stable_closure_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    normalized = []
    for record in _iter_input_records(manifest):
        role = record.get("role")
        if not isinstance(role, str) or not role:
            raise HeadlineM0Error("stable input record has no role")
        normalized.append({**_compact_file_record(record), "roles": [role]})
    normalized.sort(key=lambda value: (value["path"], value["roles"]))
    if len({(v["path"], tuple(v["roles"])) for v in normalized}) != len(normalized):
        raise HeadlineM0Error("stable input closure has duplicate identities")
    digest = _sha256_bytes(
        _canonical_json_bytes(
            {"schema": STABLE_CLOSURE_SCHEMA, "records": normalized}
        )
    )
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping) or inputs.get("stable_closure_digest") != digest:
        raise HeadlineM0Error("phase stable input closure digest drifted")
    return {"records": normalized, "digest": digest}


_SAFE_CHECKPOINT_INSPECT_SCRIPT = r"""
import json
import math
import sys

import numpy as np
import torch

path = sys.argv[1]
numpy_core = getattr(np, "_core", np.core)
safe_globals = [
    numpy_core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
    type(np.dtype(np.uint32)),
]
with torch.serialization.safe_globals(safe_globals):
    payload = torch.load(path, map_location="cpu", weights_only=True)
if not isinstance(payload, dict):
    raise TypeError("checkpoint payload must be a dict")
args = payload.get("args")
if hasattr(args, "__dict__"):
    args = vars(args)
if not isinstance(args, dict):
    raise TypeError("checkpoint args must be a dict")

components = {
    key: key in payload
    for key in (
        "model", "criterion", "optimizer", "lr_scheduler", "scaler",
        "epoch", "iteration", "optimizer_updates", "epoch_finished",
        "rng_state", "epoch_rng_state", "args",
    )
}
optimizer = payload.get("optimizer")
if not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict):
    raise TypeError("checkpoint optimizer state is missing")

steps = []
for state in optimizer["state"].values():
    if not isinstance(state, dict) or "step" not in state:
        continue
    value = state["step"]
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError("optimizer step tensor is not scalar")
        value = value.item()
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise TypeError("optimizer step is not a finite integer")
    steps.append(int(numeric))

wanted = [
    "seed", "batch_size", "max_train_iters", "iter_checkpoint_interval",
    "gradient_accumulation_steps", "num_workers", "prefetch_factor",
    "pin_memory", "persistent_workers", "world_size", "distributed",
    "config_file", "datasets", "output_dir", "pretrain_model_path", "resume",
    "stage_b_v15_scorer_init_checkpoint", "stage_b_v15_scorer_init_audit",
    "stage_b_v25_main_id", "stage_b_v25_compute_contract",
    "stage_b_v25_budget_unit", "stage_b_v25_successful_update_batch_slots",
    "stage_b_v25_initializer_contract", "stage_b_v25_strict_resume",
    "stage_b_v25_control_of", "stage_b_v25_headline_eligible",
    "stage_b_v25_matrix_validation_only", "stage_b_v25_comparison_claim",
    "stage_b_v25_token_objective_scope", "stage_b_v22_table_id",
    "stage_b_v22_objective_fidelity", "stage_b_v22_gradient_diagnostic_interval",
    "stage_b_v15_separate_grad_clip", "stage_b_v21_token_objective",
    "stage_b_v21_token_weight", "stage_b_v21_token_positive_weight",
    "stage_b_v21_token_shared_weight", "stage_b_v21_token_edit_weight",
    "stage_b_v11_predicate_tn_rank_weight",
    "stage_b_v21_allow_legacy_token_diff_fallback", "skip_eval", "amp",
]
result = {
    "top_level_keys": sorted(payload),
    "complete_state_components": components,
    "optimizer_updates": payload.get("optimizer_updates"),
    "optimizer_state_count": len(optimizer["state"]),
    "optimizer_step_values": sorted(set(steps)),
    "epoch": payload.get("epoch"),
    "iteration": payload.get("iteration"),
    "epoch_finished": payload.get("epoch_finished"),
    "checkpoint_reason": payload.get("checkpoint_reason"),
    "checkpoint_cuda_memory": {
        key: payload.get(key)
        for key in (
            "torch_cuda_max_memory_allocated_bytes",
            "torch_cuda_max_memory_reserved_bytes",
        )
    },
    "args": {key: args.get(key) for key in wanted},
}
print(json.dumps(result, sort_keys=True))
"""


def _subprocess_environment(runtime: Runtime) -> dict[str, str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = runtime.cuda_visible_devices
    environment["DATA_ROOT"] = str(runtime.data_root)
    environment["OMP_NUM_THREADS"] = str(runtime.omp_num_threads)
    environment["TORCH_MP_SHARING_STRATEGY"] = runtime.mp_sharing_strategy
    for key in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "SLURM_PROCID",
        "SLURM_NTASKS",
    ):
        environment.pop(key, None)
    old_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not old_pythonpath
        else str(REPO_ROOT) + os.pathsep + old_pythonpath
    )
    return environment


def _inspect_checkpoint_extended(
    path: Path,
    *,
    python: Path = DEFAULT_PYTHON,
    stable_fd: int | None = None,
) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    python = python.expanduser().resolve(strict=True)
    inspect_path = str(path)
    pass_fds: tuple[int, ...] = ()
    if stable_fd is not None:
        if type(stable_fd) is not int or stable_fd < 0:
            raise HeadlineM0Error("safe checkpoint descriptor is invalid")
        inspect_path = f"/proc/self/fd/{stable_fd}"
        pass_fds = (stable_fd,)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONPATH"] = str(REPO_ROOT)
    try:
        result = subprocess.run(
            [str(python), "-B", "-c", _SAFE_CHECKPOINT_INSPECT_SCRIPT, inspect_path],
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
            check=False,
            pass_fds=pass_fds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HeadlineM0Error(f"safe checkpoint inspection failed: {exc}") from exc
    if result.returncode != 0:
        raise HeadlineM0Error(
            "safe weights-only checkpoint inspection failed: "
            + result.stderr.strip()[-4000:]
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise HeadlineM0Error("safe checkpoint inspector returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise HeadlineM0Error("safe checkpoint inspector returned no mapping")
    return payload


def inspect_training_checkpoint_for_release(path: Path) -> dict[str, Any]:
    """Return the exact dependency-light checkpoint projection used by release."""

    inspected = _inspect_checkpoint_extended(Path(path))
    return {
        "optimizer_updates": inspected.get("optimizer_updates"),
        "optimizer_state_count": inspected.get("optimizer_state_count"),
        "optimizer_step_values": inspected.get("optimizer_step_values"),
        "complete_state_components": inspected.get("complete_state_components"),
        "checkpoint_reason": inspected.get("checkpoint_reason"),
    }


def _resolved_path(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise HeadlineM0Error(f"checkpoint path metadata is invalid: {value!r}")
    return Path(value).expanduser().resolve(strict=False)


def _validate_checkpoint_metadata(
    metadata: Mapping[str, Any],
    *,
    runtime: Runtime,
    contract: source_contracts.FormalPaperRunContract,
    seed: int,
    output_dir: Path,
    source_optimizer_updates: int,
    resume_checkpoint: Path | None,
) -> str:
    if metadata.get("complete_state_components") != COMPLETE_STATE_COMPONENTS:
        raise HeadlineM0Error("checkpoint does not contain the complete training state")
    updates = metadata.get("optimizer_updates")
    if type(updates) is not int or not source_optimizer_updates < updates <= FORMAL_UPDATES:
        raise HeadlineM0Error(
            f"checkpoint optimizer updates are not contiguous: {updates!r}"
        )
    if metadata.get("optimizer_state_count") != FORMAL_OPTIMIZER_STATE_COUNT:
        raise HeadlineM0Error(
            "checkpoint optimizer state count is not the sealed 94-parameter contract"
        )
    if metadata.get("optimizer_step_values") != [updates]:
        raise HeadlineM0Error("checkpoint optimizer step values do not equal progress")
    reason = metadata.get("checkpoint_reason")
    if reason not in {"signal", "max_train_iters"}:
        if reason == "signal_after_epoch":
            raise HeadlineM0Error(
                "epoch-boundary signal_after_epoch recovery is forbidden because "
                "runtime RNG replay is not exact"
            )
        raise HeadlineM0Error(f"checkpoint exit reason is not recoverable: {reason!r}")
    epoch = metadata.get("epoch")
    iteration = metadata.get("iteration")
    epoch_finished = metadata.get("epoch_finished")
    if type(epoch) is not int or type(iteration) is not int or type(epoch_finished) is not bool:
        raise HeadlineM0Error("checkpoint epoch/iteration metadata is invalid")
    if reason == "signal":
        if epoch_finished or not 0 < iteration < FORMAL_DATALOADER_MICROBATCHES:
            raise HeadlineM0Error(
                "signal recovery must be strictly inside one dataloader epoch"
            )
        if updates >= FORMAL_UPDATES:
            raise HeadlineM0Error("a target-complete checkpoint cannot be a recovery edge")
    else:
        expected = (FORMAL_FINAL_EPOCH, FORMAL_FINAL_ITERATION, False, FORMAL_UPDATES)
        observed = (epoch, iteration, epoch_finished, updates)
        if observed != expected:
            raise HeadlineM0Error(
                f"final checkpoint expected epoch/iteration/state {expected}, got {observed}"
            )

    args = metadata.get("args")
    if not isinstance(args, Mapping):
        raise HeadlineM0Error("checkpoint args metadata is missing")
    expected_scalars = {
        "seed": seed,
        "batch_size": FORMAL_BATCH_SIZE,
        "max_train_iters": FORMAL_UPDATES,
        "iter_checkpoint_interval": FORMAL_CHECKPOINT_INTERVAL,
        "gradient_accumulation_steps": 1,
        "num_workers": FORMAL_NUM_WORKERS,
        "prefetch_factor": FORMAL_PREFETCH_FACTOR,
        "pin_memory": True,
        "persistent_workers": False,
        "world_size": 1,
        "distributed": False,
        "stage_b_v25_main_id": contract.id,
        "stage_b_v25_compute_contract": "b58_successful_update_batch_slot_matched",
        "stage_b_v25_budget_unit": (
            "successful_optimizer_update_global_batch_slots"
        ),
        "stage_b_v25_successful_update_batch_slots": FORMAL_BATCH_SLOTS,
        "stage_b_v25_initializer_contract": "same_stage_a_model_and_scorer_no_b58",
        "stage_b_v25_strict_resume": True,
        "stage_b_v22_table_id": "S2F",
        "stage_b_v22_objective_fidelity": "full_v19_base_plus_gate_objective",
        "stage_b_v22_gradient_diagnostic_interval": 100,
        "stage_b_v15_separate_grad_clip": True,
        "stage_b_v21_token_objective": (
            "edit_bce" if contract.id == "M0" else "targetlocal_allneg_bce"
        ),
        "stage_b_v21_token_weight": 1.0,
        "stage_b_v21_token_positive_weight": 1.0,
        "stage_b_v21_token_shared_weight": 0.25,
        "stage_b_v21_token_edit_weight": 1.0,
        "stage_b_v11_predicate_tn_rank_weight": 1.0,
        "stage_b_v21_allow_legacy_token_diff_fallback": False,
        "skip_eval": True,
        "amp": True,
    }
    for key, expected in expected_scalars.items():
        if args.get(key) != expected:
            raise HeadlineM0Error(
                f"checkpoint args {key} expected {expected!r}, got {args.get(key)!r}"
            )
    if contract.id == "M0N":
        expected_control = {
            "stage_b_v25_control_of": "M0",
            "stage_b_v25_headline_eligible": False,
            "stage_b_v25_matrix_validation_only": True,
            "stage_b_v25_comparison_claim": (
                "full_token_objective_control_not_labels_only"
            ),
            "stage_b_v25_token_objective_scope": (
                "target_local_positive_and_all_negative_token_logits"
            ),
        }
        for key, expected in expected_control.items():
            if args.get(key) != expected:
                raise HeadlineM0Error(f"checkpoint M0N metadata {key} drifted")
    expected_paths = {
        "config_file": _config_path(contract),
        "datasets": runtime.dataset,
        "output_dir": output_dir,
        "stage_b_v15_scorer_init_checkpoint": runtime.stage_a_init,
    }
    for key, expected in expected_paths.items():
        if _resolved_path(args.get(key)) != expected.resolve(strict=False):
            raise HeadlineM0Error(f"checkpoint args path {key} drifted")
    if resume_checkpoint is None:
        if _resolved_path(args.get("pretrain_model_path")) != runtime.stage_a_init:
            raise HeadlineM0Error("fresh attempt did not initialize from Stage-A")
        if args.get("resume") not in (None, ""):
            raise HeadlineM0Error("fresh attempt unexpectedly contains --resume")
    else:
        if _resolved_path(args.get("resume")) != resume_checkpoint.resolve(strict=True):
            raise HeadlineM0Error("resume attempt checkpoint path drifted")
        if args.get("pretrain_model_path") not in (None, ""):
            raise HeadlineM0Error("resume attempt also contains pretrain_model_path")
    return str(reason)


def _validate_resume_log(
    path: Path, *, source_metadata: Mapping[str, Any]
) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    forbidden = (
        "continuing with fresh optimizer state",
        "loaded model weights only and will use fresh training state",
    )
    if any(value in text for value in forbidden):
        raise HeadlineM0Error("resume silently fell back to fresh training state")
    expected = (
        "Restored resume training state: "
        f"epoch={source_metadata['epoch']}, "
        f"iteration={source_metadata['iteration']}, "
        f"optimizer_updates={source_metadata['optimizer_updates']}, "
        "epoch_finished=False, scaler_restored=True"
    )
    if expected not in text or "Resuming mid-epoch from epoch=" not in text:
        raise HeadlineM0Error("resume log does not prove complete mid-epoch restoration")


def _checkpoint_metadata_for_attempt(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "optimizer_updates": metadata.get("optimizer_updates"),
        "optimizer_state_count": metadata.get("optimizer_state_count"),
        "optimizer_step_values": metadata.get("optimizer_step_values"),
        "complete_state_components": metadata.get("complete_state_components"),
        "checkpoint_reason": metadata.get("checkpoint_reason"),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_checkpoint_to_temporary(source: Path, directory: Path) -> tuple[Path, str]:
    source = source.resolve(strict=True)
    directory.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        temporary = directory / f".checkpoint-copy-{os.getpid()}-{uuid.uuid4().hex}.tmp"
        digest = hashlib.sha256()
        try:
            with source.open("rb") as source_handle:
                before = os.fstat(source_handle.fileno())
                with temporary.open("xb") as target_handle:
                    while True:
                        chunk = source_handle.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        target_handle.write(chunk)
                        digest.update(chunk)
                    target_handle.flush()
                    os.fsync(target_handle.fileno())
                after = os.fstat(source_handle.fileno())
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                temporary.unlink()
                if attempt < 2:
                    time.sleep(0.25)
                    continue
                raise HeadlineM0Error(
                    f"checkpoint changed while copying from stable descriptor: {source}"
                )
            if temporary.stat().st_size != before.st_size:
                raise HeadlineM0Error("stable checkpoint copy size mismatch")
            return temporary, digest.hexdigest()
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()
            raise
    raise AssertionError("unreachable stable-copy retry state")


def _publish_temporary_no_replace(temporary: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(temporary, target)
    except FileExistsError as exc:
        raise HeadlineM0Error(f"refusing to overwrite checkpoint artifact: {target}") from exc
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    _fsync_directory(target.parent)


def _validate_milestone_metadata(metadata: Mapping[str, Any], update: int) -> None:
    allowed_reasons = (
        {"max_train_iters"}
        if update == FORMAL_UPDATES
        else {"interval", "signal"}
    )
    if (
        metadata.get("complete_state_components") != COMPLETE_STATE_COMPONENTS
        or metadata.get("optimizer_updates") != update
        or metadata.get("optimizer_state_count") != FORMAL_OPTIMIZER_STATE_COUNT
        or metadata.get("optimizer_step_values") != [update]
        or metadata.get("epoch_finished") is not False
        or metadata.get("checkpoint_reason") not in allowed_reasons
    ):
        raise HeadlineM0Error(f"seed17 milestone U{update} checkpoint drifted")
    expected_epoch, expected_iteration = divmod(update, FORMAL_DATALOADER_MICROBATCHES)
    if (
        metadata.get("epoch") != expected_epoch
        or metadata.get("iteration") != expected_iteration
    ):
        raise HeadlineM0Error(f"seed17 milestone U{update} cursor drifted")


def _publish_milestone(
    source: Path,
    *,
    runtime: Runtime,
    run_root: Path,
    contract: source_contracts.FormalPaperRunContract,
    seed: int,
    update: int,
) -> dict[str, Any]:
    if seed != 17 or update not in MILESTONE_UPDATES:
        raise HeadlineM0Error("only formal seed17 milestones may be published")
    milestone_root = run_root / "milestones"
    target = milestone_root / f"checkpoint_iter_{update:06d}.pth"
    audit_path = milestone_root / f"checkpoint_iter_{update:06d}.json"
    if target.exists() or audit_path.exists():
        if not target.is_file() or not audit_path.is_file():
            raise HeadlineM0Error(f"partial seed17 milestone exists for U{update}")
        audit = _read_json(audit_path, label=f"seed17 U{update} milestone")
        observed = _file_record(target)
        if (
            audit.get("checkpoint") != _compact_file_record(observed)
            or audit.get("optimizer_updates") != update
            or audit.get("contract_id") != contract.id
        ):
            raise HeadlineM0Error(f"existing seed17 U{update} milestone drifted")
        return audit
    temporary, digest = _copy_checkpoint_to_temporary(source, milestone_root)
    try:
        metadata = _inspect_checkpoint_extended(temporary, python=runtime.python)
        _validate_milestone_metadata(metadata, update)
        _publish_temporary_no_replace(temporary, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise
    record = _compact_file_record(_file_record(target))
    if record["sha256"] != digest:
        raise HeadlineM0Error("published milestone digest differs from stable copy")
    payload = {
        "schema": "pivot.stageb.headline_m0_training_milestone/v1",
        "status": "sealed",
        "created_at_utc": _utc_now(),
        "contract_id": contract.id,
        "run_id": f"{contract.id}:{seed}",
        "seed": seed,
        "optimizer_updates": update,
        "role": "diagnostic_learning_curve_checkpoint_not_model_selection",
        "checkpoint": record,
        "checkpoint_metadata": _checkpoint_metadata_for_attempt(metadata),
    }
    _write_json_no_replace(audit_path, payload)
    return payload


def _archive_recovery_checkpoint(
    source: Path,
    *,
    runtime: Runtime,
    run_root: Path,
    next_attempt_ordinal: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    update = int(metadata["optimizer_updates"])
    recovery_root = run_root / "recovery"
    temporary, digest = _copy_checkpoint_to_temporary(source, recovery_root)
    target = recovery_root / (
        f"attempt_{next_attempt_ordinal:03d}_from_u{update:06d}_{digest[:12]}.pth"
    )
    try:
        copied = _inspect_checkpoint_extended(temporary, python=runtime.python)
        if copied != dict(metadata):
            raise HeadlineM0Error("recovery copy differs from source safe-load metadata")
        _publish_temporary_no_replace(temporary, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise
    record = _compact_file_record(_file_record(target))
    if record["sha256"] != digest:
        raise HeadlineM0Error("published recovery digest differs from stable copy")
    return record


_CHECKPOINT_LOG_RE = re.compile(
    r"Saved iteration checkpoint .*optimizer_updates=([0-9]+), reason=([^\)]+)"
)


def _run_training_process(
    command: Sequence[str],
    *,
    runtime: Runtime,
    run_root: Path,
    attempt_dir: Path,
    contract: source_contracts.FormalPaperRunContract,
    seed: int,
) -> dict[str, Any]:
    root_log = run_root / "train_console.log"
    attempt_log = attempt_dir / "train_console.log"
    process: subprocess.Popen[str] | None = None
    forwarded_signals: list[int] = []
    previous_handlers: dict[int, Any] = {}

    def forward(signum: int, _frame: Any) -> None:
        if process is not None and process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signum)
            forwarded_signals.append(int(signum))
            with contextlib.suppress(OSError):
                os.write(
                    2,
                    (
                        f"\n[M0 controller] forwarded signal {signum} to "
                        f"training process group {process.pid}\n"
                    ).encode("ascii"),
                )

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, forward)
    started_at = _utc_now()
    try:
        with (
            root_log.open("a", encoding="utf-8", buffering=1) as aggregate,
            attempt_log.open("x", encoding="utf-8", buffering=1) as attempt,
        ):
            process = subprocess.Popen(
                list(command),
                cwd=REPO_ROOT,
                env=_subprocess_environment(runtime),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=True,
                close_fds=True,
            )
            identity = paper_launcher._read_process_identity(int(process.pid))
            if process.stdout is None:
                raise HeadlineM0Error("training subprocess stdout was not captured")
            try:
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    aggregate.write(line)
                    attempt.write(line)
                    match = _CHECKPOINT_LOG_RE.search(line)
                    if match is None or seed != 17:
                        continue
                    update = int(match.group(1))
                    if update in MILESTONE_UPDATES:
                        _publish_milestone(
                            run_root / "checkpoint_iter.pth",
                            runtime=runtime,
                            run_root=run_root,
                            contract=contract,
                            seed=seed,
                            update=update,
                        )
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
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return {
        "pid": int(process.pid) if process is not None else None,
        "identity": identity if process is not None else None,
        "start_new_session": True,
        "stdin": "DEVNULL",
        "stdout_stderr": str(attempt_log),
        "returncode": int(returncode),
        "forwarded_signals": forwarded_signals,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
    }


def _attempt_runtime(runtime: Runtime) -> dict[str, Any]:
    return {
        "batch_size": runtime.batch_size,
        "num_workers": runtime.num_workers,
        "prefetch_factor": runtime.prefetch_factor,
        "amp": True,
        "gradient_accumulation_steps": 1,
        "max_train_iters": runtime.max_train_iters,
        "iter_checkpoint_interval": runtime.iter_checkpoint_interval,
        "gradient_diagnostic_interval": runtime.gradient_diagnostic_interval,
        "telemetry_interval_seconds": runtime.telemetry_interval_seconds,
        "pin_memory": True,
        "persistent_workers": False,
    }


def _attempt_paths(run_root: Path, ordinal: int) -> tuple[Path, Path, Path]:
    directory = run_root / "attempts" / f"{ordinal:03d}"
    return directory, directory / "input_closure.json", directory / "attempt_manifest.json"


def _attempt_telemetry_paths(run_root: Path, ordinal: int) -> dict[str, Path]:
    directory = _attempt_paths(run_root, ordinal)[0]
    return {
        "gpu_environment": directory / "gpu_environment.json",
        "gpu_telemetry": directory / "gpu_telemetry.csv",
        "gpu_telemetry_summary": directory / "gpu_telemetry_summary.json",
    }


def _gpu_telemetry_device_projection(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise HeadlineM0Error("GPU telemetry has no device identity projection")
    projection = []
    for device in value:
        if not isinstance(device, Mapping):
            raise HeadlineM0Error("GPU telemetry device identity is invalid")
        projected = {
            key: device.get(key)
            for key in (
                "physical_index",
                "uuid",
                "name",
                "driver_version",
                "total_memory_mib",
            )
        }
        if (
            type(projected["physical_index"]) is not int
            or not isinstance(projected["uuid"], str)
            or not projected["uuid"]
            or not isinstance(projected["name"], str)
            or not projected["name"]
            or not isinstance(projected["driver_version"], str)
            or not projected["driver_version"]
            or isinstance(projected["total_memory_mib"], bool)
            or not isinstance(projected["total_memory_mib"], (int, float))
        ):
            raise HeadlineM0Error("GPU telemetry device identity is incomplete")
        projection.append(projected)
    return projection


def _archive_attempt_telemetry(
    *,
    run_root: Path,
    ordinal: int,
    gpu_environment: Mapping[str, Any],
    gpu_summary: Mapping[str, Any],
) -> dict[str, Any]:
    root_paths = {
        "gpu_environment": run_root / "gpu_environment.json",
        "gpu_telemetry": run_root / "gpu_telemetry.csv",
        "gpu_telemetry_summary": run_root / "gpu_telemetry_summary.json",
    }
    target_paths = _attempt_telemetry_paths(run_root, ordinal)
    missing = [name for name, path in root_paths.items() if not path.is_file()]
    if missing:
        raise HeadlineM0Error(
            f"attempt {ordinal} telemetry archive is missing root artifacts: {missing}"
        )
    artifacts: dict[str, dict[str, Any]] = {}
    for name, source in root_paths.items():
        temporary, digest = _copy_checkpoint_to_temporary(
            source, target_paths[name].parent
        )
        _publish_temporary_no_replace(temporary, target_paths[name])
        source_record = _compact_file_record(_file_record(source))
        target_record = _compact_file_record(_file_record(target_paths[name]))
        if (
            digest != source_record["sha256"]
            or target_record["sha256"] != source_record["sha256"]
            or target_record["size_bytes"] != source_record["size_bytes"]
        ):
            raise HeadlineM0Error(
                f"attempt {ordinal} {name} immutable archive identity drifted"
            )
        artifacts[name] = target_record
    archived_environment = _read_json(
        target_paths["gpu_environment"],
        label=f"attempt {ordinal} archived GPU environment",
    )
    archived_summary = _read_json(
        target_paths["gpu_telemetry_summary"],
        label=f"attempt {ordinal} archived GPU telemetry summary",
    )
    try:
        paper_launcher._validate_gpu_telemetry_contract(
            archived_environment, archived_summary
        )
        replayed = paper_launcher._summarize_nvidia_csv(
            target_paths["gpu_telemetry"]
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise HeadlineM0Error(
            f"attempt {ordinal} archived GPU telemetry replay failed: {exc}"
        ) from exc
    if (
        archived_environment != dict(gpu_environment)
        or archived_summary != dict(gpu_summary)
        or archived_summary.get("sampling_interval_ms")
        != FORMAL_TELEMETRY_INTERVAL_SECONDS * 1000
        or any(
            replayed.get(key) != archived_summary.get(key)
            for key in ("schema", "sample_rows", "devices")
        )
    ):
        raise HeadlineM0Error(
            f"attempt {ordinal} archived GPU telemetry content drifted"
        )
    return {
        "schema": ATTEMPT_TELEMETRY_SCHEMA,
        "status": "sealed",
        "attempt_ordinal": ordinal,
        "sampling_interval_ms": FORMAL_TELEMETRY_INTERVAL_SECONDS * 1000,
        "sample_rows": archived_summary["sample_rows"],
        "devices": _gpu_telemetry_device_projection(archived_summary.get("devices")),
        "artifacts": artifacts,
    }


def _verify_attempt_telemetry(
    value: Any,
    *,
    run_root: Path,
    ordinal: int,
) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "status",
        "attempt_ordinal",
        "sampling_interval_ms",
        "sample_rows",
        "devices",
        "artifacts",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_keys
        or value.get("schema") != ATTEMPT_TELEMETRY_SCHEMA
        or value.get("status") != "sealed"
        or value.get("attempt_ordinal") != ordinal
        or value.get("sampling_interval_ms")
        != FORMAL_TELEMETRY_INTERVAL_SECONDS * 1000
    ):
        raise HeadlineM0Error(f"attempt {ordinal} telemetry manifest drifted")
    paths = _attempt_telemetry_paths(run_root, ordinal)
    raw_artifacts = value.get("artifacts")
    if not isinstance(raw_artifacts, Mapping) or set(raw_artifacts) != set(paths):
        raise HeadlineM0Error(f"attempt {ordinal} telemetry artifact set drifted")
    artifacts = {
        name: _verify_completed_file_record(
            raw_artifacts.get(name),
            label=f"attempt {ordinal} {name}",
            expected_path=path,
            compact=True,
        )
        for name, path in paths.items()
    }
    environment = _read_json(
        paths["gpu_environment"], label=f"attempt {ordinal} GPU environment"
    )
    summary = _read_json(
        paths["gpu_telemetry_summary"],
        label=f"attempt {ordinal} GPU telemetry summary",
    )
    try:
        paper_launcher._validate_gpu_telemetry_contract(environment, summary)
        replayed = paper_launcher._summarize_nvidia_csv(paths["gpu_telemetry"])
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise HeadlineM0Error(
            f"attempt {ordinal} GPU telemetry replay failed: {exc}"
        ) from exc
    if (
        summary.get("sampling_interval_ms")
        != FORMAL_TELEMETRY_INTERVAL_SECONDS * 1000
        or any(
            replayed.get(key) != summary.get(key)
            for key in ("schema", "sample_rows", "devices")
        )
    ):
        raise HeadlineM0Error(f"attempt {ordinal} raw GPU telemetry summary drifted")
    expected = {
        "schema": ATTEMPT_TELEMETRY_SCHEMA,
        "status": "sealed",
        "attempt_ordinal": ordinal,
        "sampling_interval_ms": FORMAL_TELEMETRY_INTERVAL_SECONDS * 1000,
        "sample_rows": summary.get("sample_rows"),
        "devices": _gpu_telemetry_device_projection(summary.get("devices")),
        "artifacts": artifacts,
    }
    if dict(value) != expected:
        raise HeadlineM0Error(f"attempt {ordinal} telemetry evidence forked")
    return expected


def _full_run_telemetry_projection(
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not attempts:
        raise HeadlineM0Error("full-run telemetry has no attempts")
    expected_devices = _gpu_telemetry_device_projection(attempts[0].get("devices"))
    total_rows = 0
    projected_attempts = []
    for ordinal, attempt in enumerate(attempts):
        rows = attempt.get("sample_rows")
        if (
            attempt.get("schema") != ATTEMPT_TELEMETRY_SCHEMA
            or attempt.get("status") != "sealed"
            or attempt.get("attempt_ordinal") != ordinal
            or attempt.get("sampling_interval_ms")
            != FORMAL_TELEMETRY_INTERVAL_SECONDS * 1000
            or type(rows) is not int
            or rows <= 0
            or attempt.get("devices") != expected_devices
            or not isinstance(attempt.get("artifacts"), Mapping)
            or set(attempt["artifacts"])
            != {"gpu_environment", "gpu_telemetry", "gpu_telemetry_summary"}
        ):
            raise HeadlineM0Error(
                f"full-run telemetry attempt {ordinal} is not contiguous"
            )
        total_rows += rows
        projected_attempts.append(
            {
                "attempt_ordinal": ordinal,
                "sample_rows": rows,
                "devices": attempt["devices"],
                "artifacts": attempt["artifacts"],
                "evidence_sha256": _sha256_bytes(
                    _canonical_json_bytes(dict(attempt))
                ),
            }
        )
    projection = {
        "schema": FULL_RUN_TELEMETRY_SCHEMA,
        "status": "passed",
        "attempt_count": len(attempts),
        "sampling_interval_ms": FORMAL_TELEMETRY_INTERVAL_SECONDS * 1000,
        "sample_rows": total_rows,
        "devices": expected_devices,
        "all_attempts_same_devices": True,
        "attempts": projected_attempts,
    }
    projection["semantic_sha256"] = _sha256_bytes(
        _canonical_json_bytes(projection)
    )
    return projection


def _write_attempt_input_closure(
    phase_manifest: Mapping[str, Any], *, run_root: Path, ordinal: int
) -> dict[str, Any]:
    closure = _stable_closure_from_manifest(phase_manifest)
    directory, path, _ = _attempt_paths(run_root, ordinal)
    directory.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema": STABLE_CLOSURE_SCHEMA,
        "status": "sealed",
        "algorithm": "sha256_canonical_path_content_size_roles_v1",
        "records": closure["records"],
        "digest": closure["digest"],
    }
    _write_json_no_replace(path, payload)
    return _compact_file_record(_file_record(path))


def _attempt_manifest_record(run_root: Path, ordinal: int) -> dict[str, Any]:
    _, _, path = _attempt_paths(run_root, ordinal)
    return _compact_file_record(_file_record(path))


def _load_attempt(run_root: Path, ordinal: int) -> dict[str, Any]:
    _, _, path = _attempt_paths(run_root, ordinal)
    return _read_json(path, label=f"M0 attempt {ordinal}")


def _verify_recovery_checkpoint(
    record: Mapping[str, Any],
    *,
    runtime: Runtime,
    run_root: Path,
    ordinal: int,
    expected_optimizer_updates: int,
) -> dict[str, Any]:
    path = Path(str(record.get("path", ""))).resolve(strict=True)
    expected_parent = (run_root / "recovery").resolve(strict=True)
    if path.parent != expected_parent or not path.name.startswith(
        f"attempt_{ordinal:03d}_from_u{expected_optimizer_updates:06d}_"
    ):
        raise HeadlineM0Error("resume checkpoint is outside its canonical recovery edge")
    observed = _compact_file_record(_file_record(path))
    if observed != dict(record):
        raise HeadlineM0Error("immutable recovery checkpoint identity changed")
    metadata = _inspect_checkpoint_extended(path, python=runtime.python)
    if (
        metadata.get("complete_state_components") != COMPLETE_STATE_COMPONENTS
        or metadata.get("checkpoint_reason") != "signal"
        or metadata.get("optimizer_updates") != expected_optimizer_updates
        or metadata.get("optimizer_state_count") != FORMAL_OPTIMIZER_STATE_COUNT
        or metadata.get("optimizer_step_values") != [expected_optimizer_updates]
        or metadata.get("epoch_finished") is not False
        or not 0 < int(metadata.get("iteration", 0)) < FORMAL_DATALOADER_MICROBATCHES
    ):
        raise HeadlineM0Error("recovery checkpoint is not a complete mid-epoch signal state")
    return metadata


def _write_attempt_manifest(
    *,
    runtime: Runtime,
    contract: source_contracts.FormalPaperRunContract,
    seed: int,
    run_root: Path,
    ordinal: int,
    command: Sequence[str],
    source_optimizer_updates: int,
    resume_checkpoint: dict[str, Any] | None,
    resume_authorization: dict[str, Any] | None,
    process: Mapping[str, Any],
    termination: Mapping[str, Any],
    checkpoint_at_exit: Mapping[str, Any],
    checkpoint_metadata: Mapping[str, Any],
    input_closure: Mapping[str, Any],
    telemetry: Mapping[str, Any],
) -> dict[str, Any]:
    parent = _attempt_manifest_record(run_root, ordinal - 1) if ordinal else None
    manifest = {
        "schema": ATTEMPT_SCHEMA,
        "status": "completed",
        "run_id": f"{contract.id}:{seed}",
        "seed": seed,
        "attempt_ordinal": ordinal,
        "initialization_mode": "fresh_stage_a" if ordinal == 0 else "same_run_resume",
        "parent_attempt_manifest": parent,
        "resume_checkpoint": resume_checkpoint,
        "resume_authorization": resume_authorization,
        "source_optimizer_updates": source_optimizer_updates,
        "target_optimizer_updates": FORMAL_UPDATES,
        "command": list(command),
        "command_shell": shlex.join(command),
        "runtime": _attempt_runtime(runtime),
        "input_closure_digest": _stable_closure_from_manifest(
            _read_json(run_root / "launch_manifest.json", label="M0 phase launch")
        )["digest"],
        "input_closure": dict(input_closure),
        "telemetry": dict(telemetry),
        "process": dict(process),
        "termination": dict(termination),
        "complete_state_components": dict(COMPLETE_STATE_COMPONENTS),
        "checkpoint_at_exit": dict(checkpoint_at_exit),
        "checkpoint_metadata": _checkpoint_metadata_for_attempt(checkpoint_metadata),
        "started_at_utc": process["started_at_utc"],
        "finished_at_utc": process["finished_at_utc"],
    }
    _, _, path = _attempt_paths(run_root, ordinal)
    _write_json_no_replace(path, manifest)
    return manifest


def _build_ancestry(
    *,
    phase_manifest: Mapping[str, Any],
    runtime: Runtime,
    contract: source_contracts.FormalPaperRunContract,
    seed: int,
    run_root: Path,
    final_ordinal: int,
) -> dict[str, Any]:
    records = list(_iter_input_records(phase_manifest))

    def unique_role(role: str) -> dict[str, Any]:
        selected = [record for record in records if record.get("role") == role]
        if len(selected) != 1:
            raise HeadlineM0Error(f"expected exactly one stable input role {role}")
        return _compact_file_record(selected[0])

    stage_a = unique_role("stage_a_initializer")
    scorer = unique_role("scorer_warmstart")
    if stage_a != scorer:
        raise HeadlineM0Error("model and scorer initializers do not share Stage-A content")
    attempts = [_load_attempt(run_root, ordinal) for ordinal in range(final_ordinal + 1)]
    attempt_records = [
        _attempt_manifest_record(run_root, ordinal)
        for ordinal in range(final_ordinal + 1)
    ]
    resume_edges = []
    for ordinal in range(1, final_ordinal + 1):
        previous = attempts[ordinal - 1]
        current = attempts[ordinal]
        source = previous.get("checkpoint_at_exit")
        metadata = previous.get("checkpoint_metadata")
        authorization = current.get("resume_authorization")
        if (
            not isinstance(source, Mapping)
            or not isinstance(metadata, Mapping)
            or not isinstance(authorization, Mapping)
        ):
            raise HeadlineM0Error("resume ancestry source attempt is incomplete")
        if metadata.get("checkpoint_reason") != "signal":
            raise HeadlineM0Error("only reason=signal may enter resume ancestry")
        resume_edges.append(
            {
                "ordinal": ordinal,
                "run_id": f"{contract.id}:{seed}",
                "source_checkpoint": dict(source),
                "source_optimizer_updates": metadata["optimizer_updates"],
                "source_checkpoint_reason": "signal",
                "complete_training_state": True,
                "same_run": True,
                "resume_authorization": dict(authorization),
                "attempt_manifest": attempt_records[ordinal],
            }
        )
    return {
        "schema": ANCESTRY_SCHEMA,
        "fresh_start": {
            "run_id": f"{contract.id}:{seed}",
            "initialization_mode": (
                "pretrain_model_path_plus_same_source_scorer_init"
            ),
            "pretrain": {**stage_a, "role": "stage_a_initializer"},
            "scorer": {**scorer, "role": "scorer_warmstart"},
            "same_source": True,
            "resume_argument": None,
            "attempt_manifest": attempt_records[0],
        },
        "resume_ancestry": resume_edges,
        "ultimate_pretrain": {
            "path": stage_a["path"],
            "sha256": stage_a["sha256"],
            "role": "stage_a_initializer",
        },
        "ultimate_scorer": {
            "path": scorer["path"],
            "sha256": scorer["sha256"],
            "role": "scorer_warmstart",
        },
        "ultimate_same_stage_a_source": True,
        "resume_chain_contiguous": True,
        "b58_ancestry_count": 0,
        "b58_ancestry_paths": [],
        "b58_ancestry_sha256s": [],
    }


def _scorer_initializer_wrapper(
    *, runtime: Runtime, artifacts: Mapping[str, Any]
) -> dict[str, Any]:
    path = Path(str(artifacts["scorer_init_audit"]["path"])).resolve(strict=True)
    payload = _read_json(path, label="scorer initialization audit")
    expected = {
        "schema": "stage_b_v15_scorer_init/v1",
        "status": "applied",
        "source_sha256": DEFAULT_STAGE_A_SHA256,
        "loaded_tensor_count": 90,
        "loaded_num_layers": 3,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise HeadlineM0Error(
                f"scorer initialization audit {key} expected {value!r}, "
                f"got {payload.get(key)!r}"
            )
    if _resolved_path(payload.get("resolved_source_path")) != runtime.stage_a_init:
        raise HeadlineM0Error("scorer initialization audit source path drifted")
    return {
        "status": "passed",
        "applied": True,
        "source_path": str(runtime.stage_a_init),
        "source_sha256": DEFAULT_STAGE_A_SHA256,
        "loaded_tensor_count": 90,
        "loaded_num_layers": 3,
        "artifact": dict(artifacts["scorer_init_audit"]),
        "same_as_stage_a_initializer": True,
        "b58_source": False,
    }


def _milestone_evidence(
    *, run_root: Path, contract: source_contracts.FormalPaperRunContract, seed: int
) -> dict[str, Any]:
    if seed != 17:
        return {
            "status": "not_required",
            "reason": "diagnostic learning-curve checkpoints are retained only for seed17",
            "updates": [],
        }
    audits = []
    for update in MILESTONE_UPDATES:
        path = run_root / "milestones" / f"checkpoint_iter_{update:06d}.json"
        payload = _read_json(path, label=f"seed17 U{update} milestone audit")
        checkpoint = payload.get("checkpoint")
        if (
            payload.get("status") != "sealed"
            or payload.get("contract_id") != contract.id
            or payload.get("run_id") != f"{contract.id}:17"
            or payload.get("optimizer_updates") != update
            or not isinstance(checkpoint, Mapping)
        ):
            raise HeadlineM0Error(f"seed17 U{update} milestone audit drifted")
        observed = _compact_file_record(_file_record(Path(str(checkpoint["path"]))))
        if dict(checkpoint) != observed:
            raise HeadlineM0Error(f"seed17 U{update} milestone content drifted")
        audits.append(_compact_file_record(_file_record(path)))
    return {
        "status": "passed",
        "role": "diagnostic_learning_curve_only_not_selection",
        "updates": list(MILESTONE_UPDATES),
        "audits": audits,
    }


def _perform_postflight(
    phase_manifest: Mapping[str, Any],
    *,
    runtime: Runtime,
    contract: source_contracts.FormalPaperRunContract,
    seed: int,
    run_root: Path,
    metadata: Mapping[str, Any],
    final_attempt_ordinal: int,
    cache: token_launcher.HashCache,
) -> dict[str, Any]:
    required = {
        "checkpoint": run_root / "checkpoint_iter.pth",
        "native_info_log": run_root / "info.txt",
        "train_console_log": run_root / "train_console.log",
        "gpu_environment": run_root / "gpu_environment.json",
        "gpu_telemetry": run_root / "gpu_telemetry.csv",
        "gpu_telemetry_summary": run_root / "gpu_telemetry_summary.json",
        "scorer_init_audit": run_root / "stage_b_v15_scorer_init_audit.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise HeadlineM0Error(f"postflight is missing required artifacts: {missing}")
    for key in ("native_info_log", "train_console_log"):
        if required[key].stat().st_size <= 0:
            raise HeadlineM0Error(f"postflight log is empty: {required[key]}")
    _verify_input_identities(phase_manifest, rehash=True)
    input_rehash = paper_launcher._rehash_inputs(phase_manifest)
    input_rehash_path = run_root / "input_rehash.json"
    _write_json_atomic(input_rehash_path, input_rehash)
    required["input_rehash"] = input_rehash_path

    gpu_environment = _read_json(
        required["gpu_environment"], label="GPU environment"
    )
    gpu_summary = _read_json(
        required["gpu_telemetry_summary"], label="GPU telemetry summary"
    )
    paper_launcher._validate_gpu_telemetry_contract(gpu_environment, gpu_summary)
    numerical = paper_launcher._training_numerical_status(
        required["native_info_log"], required["train_console_log"]
    )
    logs = (
        required["native_info_log"].read_text(encoding="utf-8", errors="replace")
        + "\n"
        + required["train_console_log"].read_text(
            encoding="utf-8", errors="replace"
        )
    )
    if "stage_b_v22_branch_isolation_pass" not in logs:
        raise HeadlineM0Error("M0/S2F branch-isolation diagnostic is absent")
    if "continuing with fresh optimizer state" in logs:
        raise HeadlineM0Error("training history contains a resume-state fallback")

    artifacts = {
        name: _file_record(path, cache)
        for name, path in required.items()
    }
    ancestry = _build_ancestry(
        phase_manifest=phase_manifest,
        runtime=runtime,
        contract=contract,
        seed=seed,
        run_root=run_root,
        final_ordinal=final_attempt_ordinal,
    )
    full_run_telemetry = _full_run_telemetry_projection(
        [
            _verify_attempt_telemetry(
                _load_attempt(run_root, ordinal).get("telemetry"),
                run_root=run_root,
                ordinal=ordinal,
            )
            for ordinal in range(final_attempt_ordinal + 1)
        ]
    )
    progress = {
        "status": "passed",
        "optimizer_updates": FORMAL_UPDATES,
        "consumed_microbatches": FORMAL_UPDATES,
        "gradient_accumulation_steps": 1,
        "data_loader_microbatches_per_epoch": FORMAL_DATALOADER_MICROBATCHES,
        "checkpoint_epoch": FORMAL_FINAL_EPOCH,
        "checkpoint_iteration": FORMAL_FINAL_ITERATION,
        "checkpoint_epoch_finished": False,
        "checkpoint_reason": "max_train_iters",
        "optimizer_state_count": FORMAL_OPTIMIZER_STATE_COUNT,
        "optimizer_step_values": [FORMAL_UPDATES],
        "checkpoint_optimizer_step": FORMAL_UPDATES,
        "successful_update_batch_slots": FORMAL_BATCH_SLOTS,
        "successful_updates_equal_consumed_microbatches": True,
    }
    checkpoint_memory = metadata.get("checkpoint_cuda_memory")
    return {
        "schema": POSTFLIGHT_SCHEMA,
        "status": "passed",
        "validated_at_utc": _utc_now(),
        "run_id": f"{contract.id}:{seed}",
        "seed": seed,
        "phase_id": "joint",
        "checkpoint_metadata": dict(metadata),
        "optimizer_progress": progress,
        "input_rehash": input_rehash,
        "gpu_environment": gpu_environment,
        "gpu_telemetry_summary": gpu_summary,
        "full_run_telemetry": full_run_telemetry,
        "numerical_status": numerical,
        "checkpoint_cuda_memory": {
            "available": bool(
                isinstance(checkpoint_memory, Mapping)
                and any(value is not None for value in checkpoint_memory.values())
            ),
            "values": (
                dict(checkpoint_memory)
                if isinstance(checkpoint_memory, Mapping)
                else {}
            ),
        },
        "artifacts": artifacts,
        "model_state_ancestry": ancestry,
        "scorer_initializer_audit": _scorer_initializer_wrapper(
            runtime=runtime, artifacts=artifacts
        ),
        "training_attempt_count": final_attempt_ordinal + 1,
        "same_run_resume_count": final_attempt_ordinal,
        "milestones": _milestone_evidence(
            run_root=run_root, contract=contract, seed=seed
        ),
        "formal_claim": (
            "successful_optimizer_update_batch_slot_matched_not_flop_or_wall_clock_matched"
        ),
    }


def _resume_required_path(run_root: Path) -> Path:
    return run_root / "control/resume_required.json"


def _resume_request_path(run_root: Path) -> Path:
    return run_root / "control/resume_request.json"


def _wait_for_resume_authorization(
    *,
    run_root: Path,
    run_id: str,
    next_ordinal: int,
    recovery_checkpoint: Mapping[str, Any],
    orchestration_status: Path | None,
) -> dict[str, Any]:
    required = {
        "schema": RESUME_REQUEST_SCHEMA,
        "status": "required",
        "created_at_utc": _utc_now(),
        "run_id": run_id,
        "next_attempt_ordinal": next_ordinal,
        "recovery_checkpoint": dict(recovery_checkpoint),
        "policy": "explicit_one_attempt_mid_epoch_signal_resume",
    }
    required_path = _resume_required_path(run_root)
    request_path = _resume_request_path(run_root)
    _write_json_atomic(required_path, required)
    paper_launcher._update_orchestration_status(
        orchestration_status,
        status="running",
        current_run_id=run_id,
        current_phase_id="joint",
        resume_required={
            "run_root": str(run_root),
            "next_attempt_ordinal": next_ordinal,
            "recovery_checkpoint": dict(recovery_checkpoint),
        },
    )
    print(
        json.dumps(
            {
                "status": "resume_required",
                "run_id": run_id,
                "run_root": str(run_root),
                "next_attempt_ordinal": next_ordinal,
                "authorize_with": (
                    f"{sys.executable} {Path(__file__).resolve()} resume {run_root}"
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    while True:
        if not request_path.exists():
            time.sleep(5.0)
            continue
        request = _read_json(request_path, label="M0 resume authorization")
        if (
            set(request)
            != {
                "schema",
                "status",
                "run_id",
                "next_attempt_ordinal",
                "recovery_checkpoint",
                "policy",
                "authorized_at_utc",
                "authorizer_pid",
                "detached_controller_identity",
            }
            or
            request.get("schema") != RESUME_REQUEST_SCHEMA
            or request.get("status") != "authorized"
            or request.get("run_id") != run_id
            or request.get("next_attempt_ordinal") != next_ordinal
            or request.get("recovery_checkpoint") != dict(recovery_checkpoint)
            or request.get("policy") != "explicit_one_attempt_mid_epoch_signal_resume"
            or type(request.get("authorizer_pid")) is not int
            or int(request["authorizer_pid"]) <= 0
            or (
                request.get("detached_controller_identity") is not None
                and not isinstance(request.get("detached_controller_identity"), Mapping)
            )
        ):
            raise HeadlineM0Error("resume authorization does not match the paused attempt")
        try:
            authorized_at = datetime.fromisoformat(str(request["authorized_at_utc"]))
        except (TypeError, ValueError) as exc:
            raise HeadlineM0Error("resume authorization timestamp is invalid") from exc
        if authorized_at.tzinfo is None:
            raise HeadlineM0Error("resume authorization timestamp lacks a timezone")
        archive = run_root / "control/resume_requests" / f"{next_ordinal:03d}.json"
        archive.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(request_path, archive)
        except FileExistsError as exc:
            raise HeadlineM0Error("resume authorization ordinal was already consumed") from exc
        request_path.unlink()
        required_path.unlink()
        _fsync_directory(archive.parent)
        _fsync_directory(required_path.parent)
        return _compact_file_record(_file_record(archive))


def _resolve_resume_target(target: Path) -> tuple[Path, dict[str, Any] | None]:
    target = target.expanduser().resolve(strict=True)
    launch_path = target / "launch.json"
    status_path = target / "status.json"
    if launch_path.is_file() and status_path.is_file():
        observation = paper_launcher._inspect_or_reconcile_detached_job(
            target, mutate=False
        )
        liveness = observation.get("pid_liveness")
        if (
            observation.get("observed_status") != "running"
            or not isinstance(liveness, Mapping)
            or liveness.get("running") is not True
        ):
            raise HeadlineM0Error("detached M0 controller is not live and waiting")
        launch = _read_json(launch_path, label="M0 detached launch")
        roots = launch.get("expected_run_roots")
        if not isinstance(roots, list) or len(roots) != 1:
            raise HeadlineM0Error("detached launch has no unique run root")
        return Path(str(roots[0])).resolve(strict=True), launch
    sequence = target / "sequence_manifest.json"
    if sequence.is_file():
        return target, None
    raise HeadlineM0Error(
        "resume target must be a live detached job directory or paused run root"
    )


def authorize_resume(target: Path) -> dict[str, Any]:
    run_root, detached = _resolve_resume_target(target)
    required_path = _resume_required_path(run_root)
    required = _read_json(required_path, label="M0 resume requirement")
    sequence = _read_json(
        run_root / "sequence_manifest.json", label="paused M0 sequence"
    )
    paused = sequence.get("paused_for_resume")
    if (
        sequence.get("status") != "running"
        or not isinstance(paused, Mapping)
        or paused.get("next_attempt_ordinal") != required.get("next_attempt_ordinal")
        or paused.get("recovery_checkpoint") != required.get("recovery_checkpoint")
    ):
        raise HeadlineM0Error("M0 sequence is not paused at the declared recovery edge")
    request = {
        **required,
        "status": "authorized",
        "authorized_at_utc": _utc_now(),
        "authorizer_pid": os.getpid(),
        "detached_controller_identity": (
            detached.get("child_process_identity") if detached else None
        ),
    }
    request.pop("created_at_utc", None)
    _write_json_no_replace(_resume_request_path(run_root), request)
    return {
        "status": "authorized",
        "run_id": required["run_id"],
        "run_root": str(run_root),
        "next_attempt_ordinal": required["next_attempt_ordinal"],
        "request": str(_resume_request_path(run_root)),
    }


def _run_body(
    contract: source_contracts.FormalPaperRunContract,
    seed: int,
    *,
    orchestration_status: Path | None,
    queue_binding: Mapping[str, Any] | None = None,
    queue_dir: Path | None = None,
    queue_orchestration_root: Path | None = None,
) -> int:
    runtime = runtime_from_environment()
    run_root = output_directory(runtime, contract, seed)
    if run_root.exists():
        raise FileExistsError(f"formal run root must be fresh: {run_root}")
    cache = token_launcher.HashCache()
    sequence = build_manifest(runtime, contract, seed, cache)
    if queue_dir is not None:
        if queue_orchestration_root is None:
            raise HeadlineM0Error("formal run lacks its queue orchestration root")
        queue_binding = verify_training_queue(
            queue_dir,
            contract.id,
            expected_run_id=f"{contract.id}:{seed}",
            expected_orchestration_root=queue_orchestration_root,
            expected_manifest=sequence,
        )
    if queue_binding is not None:
        sequence["training_queue_binding"] = dict(queue_binding)
        sequence["phases"][0]["training_queue_binding"] = dict(queue_binding)
    if sequence.get("output_dir_fresh_at_plan") is not True:
        raise HeadlineM0Error("formal run did not pass fresh-output preflight")
    phase_manifest = dict(sequence["phases"][0])
    _verify_input_identities(phase_manifest, rehash=False)
    run_root.mkdir(parents=True, exist_ok=False)
    sequence_path = run_root / "sequence_manifest.json"
    launch_path = run_root / "launch_manifest.json"
    now = _utc_now()
    sequence["status"] = "running"
    sequence["started_at_utc"] = now
    phase_manifest["status"] = "running"
    phase_manifest["started_at_utc"] = now
    _write_json_atomic(sequence_path, sequence)
    _write_json_atomic(launch_path, phase_manifest)
    paper_launcher._update_orchestration_status(
        orchestration_status,
        status="preflight_passed",
        run_ids=[f"{contract.id}:{seed}"],
        expected_run_roots=[str(run_root)],
        completed_run_ids=[],
    )

    ordinal = 0
    source_optimizer_updates = 0
    resume_record: dict[str, Any] | None = None
    resume_authorization: dict[str, Any] | None = None
    resume_metadata: dict[str, Any] | None = None
    while True:
        _verify_input_identities(phase_manifest, rehash=False)
        input_closure = _write_attempt_input_closure(
            phase_manifest, run_root=run_root, ordinal=ordinal
        )
        resume_path = (
            Path(str(resume_record["path"])).resolve(strict=True)
            if resume_record is not None
            else None
        )
        if resume_record is not None:
            resume_metadata = _verify_recovery_checkpoint(
                resume_record,
                runtime=runtime,
                run_root=run_root,
                ordinal=ordinal,
                expected_optimizer_updates=source_optimizer_updates,
            )
        command = build_command(
            runtime,
            contract,
            seed,
            run_root,
            resume_checkpoint=resume_path,
        )
        phase_manifest["command"] = command
        phase_manifest["command_shell"] = shlex.join(command)
        phase_manifest["status"] = "running"
        phase_manifest["current_attempt_ordinal"] = ordinal
        phase_manifest.pop("resume_required", None)
        _write_json_atomic(launch_path, phase_manifest)
        sequence.pop("paused_for_resume", None)
        sequence["current_attempt_ordinal"] = ordinal
        _write_json_atomic(sequence_path, sequence)
        paper_launcher._update_orchestration_status(
            orchestration_status,
            status="running",
            current_run_id=f"{contract.id}:{seed}",
            current_phase_id="joint",
            current_attempt_ordinal=ordinal,
            resume_required=None,
        )
        print(
            f"[{contract.id}:{seed}/joint/attempt{ordinal}] {shlex.join(command)}",
            flush=True,
        )
        gpu_environment = paper_launcher._capture_gpu_environment(runtime, run_root)
        sampler = paper_launcher._GpuTelemetrySampler(runtime, run_root)
        try:
            process = _run_training_process(
                command,
                runtime=runtime,
                run_root=run_root,
                attempt_dir=_attempt_paths(run_root, ordinal)[0],
                contract=contract,
                seed=seed,
            )
        finally:
            gpu_summary = sampler.stop()
        phase_manifest["gpu_environment"] = gpu_environment
        phase_manifest["gpu_telemetry_summary"] = gpu_summary
        if process["returncode"] != 0:
            raise HeadlineM0Error(
                f"{contract.id}:{seed} attempt {ordinal} exited {process['returncode']}"
            )
        attempt_telemetry = _archive_attempt_telemetry(
            run_root=run_root,
            ordinal=ordinal,
            gpu_environment=gpu_environment,
            gpu_summary=gpu_summary,
        )
        checkpoint = run_root / "checkpoint_iter.pth"
        if not checkpoint.is_file():
            raise HeadlineM0Error("training attempt exited without checkpoint_iter.pth")
        metadata = _inspect_checkpoint_extended(checkpoint, python=runtime.python)
        reason = _validate_checkpoint_metadata(
            metadata,
            runtime=runtime,
            contract=contract,
            seed=seed,
            output_dir=run_root,
            source_optimizer_updates=source_optimizer_updates,
            resume_checkpoint=resume_path,
        )
        if resume_path is not None:
            if resume_metadata is None:
                raise AssertionError("resume metadata was not retained")
            _validate_resume_log(
                _attempt_paths(run_root, ordinal)[0] / "train_console.log",
                source_metadata=resume_metadata,
            )
        if reason == "signal":
            recovery = _archive_recovery_checkpoint(
                checkpoint,
                runtime=runtime,
                run_root=run_root,
                next_attempt_ordinal=ordinal + 1,
                metadata=metadata,
            )
            _write_attempt_manifest(
                runtime=runtime,
                contract=contract,
                seed=seed,
                run_root=run_root,
                ordinal=ordinal,
                command=command,
                source_optimizer_updates=source_optimizer_updates,
                resume_checkpoint=resume_record,
                resume_authorization=resume_authorization,
                process=process,
                termination={
                    "kind": "graceful_signal_checkpoint",
                    "reason": "signal",
                },
                checkpoint_at_exit=recovery,
                checkpoint_metadata=metadata,
                input_closure=input_closure,
                telemetry=attempt_telemetry,
            )
            sequence["paused_for_resume"] = {
                "reason": "signal",
                "completed_attempt_ordinal": ordinal,
                "next_attempt_ordinal": ordinal + 1,
                "source_optimizer_updates": metadata["optimizer_updates"],
                "recovery_checkpoint": recovery,
            }
            phase_manifest["resume_required"] = dict(sequence["paused_for_resume"])
            _write_json_atomic(sequence_path, sequence)
            _write_json_atomic(launch_path, phase_manifest)
            next_authorization = _wait_for_resume_authorization(
                run_root=run_root,
                run_id=f"{contract.id}:{seed}",
                next_ordinal=ordinal + 1,
                recovery_checkpoint=recovery,
                orchestration_status=orchestration_status,
            )
            resume_record = recovery
            resume_authorization = next_authorization
            source_optimizer_updates = int(metadata["optimizer_updates"])
            ordinal += 1
            continue

        if seed == 17:
            final_milestone = run_root / "milestones" / (
                f"checkpoint_iter_{FORMAL_UPDATES:06d}.pth"
            )
            if not final_milestone.exists():
                _publish_milestone(
                    checkpoint,
                    runtime=runtime,
                    run_root=run_root,
                    contract=contract,
                    seed=seed,
                    update=FORMAL_UPDATES,
                )
        checkpoint_record = _compact_file_record(_file_record(checkpoint))
        _write_attempt_manifest(
            runtime=runtime,
            contract=contract,
            seed=seed,
            run_root=run_root,
            ordinal=ordinal,
            command=command,
            source_optimizer_updates=source_optimizer_updates,
            resume_checkpoint=resume_record,
            resume_authorization=resume_authorization,
            process=process,
            termination={"kind": "target_completed", "reason": "max_train_iters"},
            checkpoint_at_exit=checkpoint_record,
            checkpoint_metadata=metadata,
            input_closure=input_closure,
            telemetry=attempt_telemetry,
        )
        postflight = _perform_postflight(
            phase_manifest,
            runtime=runtime,
            contract=contract,
            seed=seed,
            run_root=run_root,
            metadata=metadata,
            final_attempt_ordinal=ordinal,
            cache=cache,
        )
        postflight_path = run_root / "postflight.json"
        _write_json_atomic(postflight_path, postflight)
        phase_manifest["postflight"] = postflight
        phase_manifest["postflight_artifact"] = _file_record(
            postflight_path, cache
        )
        phase_manifest["returncode"] = 0
        phase_manifest["status"] = "completed"
        phase_manifest["finished_at_utc"] = _utc_now()
        phase_manifest.pop("current_attempt_ordinal", None)
        phase_manifest.pop("resume_required", None)
        _write_json_atomic(launch_path, phase_manifest)
        checkpoint_full = _file_record(checkpoint, cache)
        sequence["status"] = "completed"
        sequence["finished_at_utc"] = phase_manifest["finished_at_utc"]
        sequence["completed_phases"] = [
            {
                "phase_id": "joint",
                "status": "completed",
                "output_dir": str(run_root),
                "checkpoint": checkpoint_full,
                "postflight": _file_record(postflight_path, cache),
            }
        ]
        sequence["training_attempt_count"] = ordinal + 1
        sequence["same_run_resume_count"] = ordinal
        sequence.pop("current_attempt_ordinal", None)
        sequence.pop("paused_for_resume", None)
        _write_json_atomic(sequence_path, sequence)
        return 0


def _run(contract: source_contracts.FormalPaperRunContract, seed: int) -> int:
    raw_status = os.environ.get("PIVOT_ORCHESTRATION_STATUS")
    status_path = (
        Path(raw_status).expanduser().resolve(strict=False) if raw_status else None
    )
    run_id = f"{contract.id}:{seed}"
    raw_queue_dir = os.environ.get("PIVOT_HEADLINE_M0_QUEUE_DIR")
    if not raw_queue_dir or status_path is None:
        raise HeadlineM0Error(
            "formal M0/M0N run requires a verified serial queue and detached job"
        )
    try:
        status_path = status_path.resolve(strict=True)
    except OSError as exc:
        raise HeadlineM0Error("formal detached status artifact is missing") from exc
    job_dir = status_path.parent
    queue_orchestration_root = job_dir.parent
    launch = _read_json(job_dir / "launch.json", label="formal detached launch")
    detached_status = _read_json(status_path, label="formal detached status")
    if (
        launch.get("schema") != DETACHED_SCHEMA
        or launch.get("status") not in {"prepared", "launched"}
        or launch.get("job_dir") != str(job_dir)
        or launch.get("orchestrator_status") != str(status_path)
        or launch.get("run_ids") != [run_id]
        or detached_status.get("run_ids") != [run_id]
        or detached_status.get("expected_run_roots")
        != launch.get("expected_run_roots")
    ):
        raise HeadlineM0Error("formal child is not bound to its detached launch")
    queue_binding = verify_training_queue(
        Path(raw_queue_dir),
        contract.id,
        expected_run_id=run_id,
        expected_orchestration_root=queue_orchestration_root,
    )
    launch_binding = launch.get("training_queue_binding")
    identity_keys = {
        "contract_id",
        "queue_id",
        "plan_sha256",
        "queue_contract_sha256",
        "stable_input_closure_digest",
    }
    if not isinstance(launch_binding, Mapping) or any(
        launch_binding.get(key) != queue_binding.get(key) for key in identity_keys
    ):
        raise HeadlineM0Error("detached launch queue identity differs from live replay")
    launch_active = launch_binding.get("active_item")
    live_active = queue_binding.get("active_item")
    if not isinstance(launch_active, Mapping) or not isinstance(live_active, Mapping):
        raise HeadlineM0Error("detached launch lacks an active queue item binding")
    for key in ("item_index", "run_id", "orchestration_root", "gpu_key", "lease_path"):
        if launch_active.get(key) != live_active.get(key):
            raise HeadlineM0Error("detached launch active queue item drifted")
    paper_launcher._update_orchestration_status(
        status_path,
        status="starting",
        run_ids=[run_id],
        started_at_utc=_utc_now(),
    )
    try:
        result = _run_body(
            contract,
            seed,
            orchestration_status=status_path,
            queue_binding=queue_binding,
            queue_dir=Path(raw_queue_dir),
            queue_orchestration_root=queue_orchestration_root,
        )
    except BaseException as exc:
        run_root = contract.canonical_training_root(seed)
        sequence_path = run_root / "sequence_manifest.json"
        launch_path = run_root / "launch_manifest.json"
        rendered = f"{type(exc).__name__}: {exc}"
        if sequence_path.is_file():
            with contextlib.suppress(Exception):
                sequence = _read_json(sequence_path, label="failed M0 sequence")
                sequence["status"] = "failed"
                sequence["finished_at_utc"] = _utc_now()
                sequence["error"] = rendered
                sequence.setdefault("completed_phases", [])
                _write_json_atomic(sequence_path, sequence)
        if launch_path.is_file():
            with contextlib.suppress(Exception):
                launch = _read_json(launch_path, label="failed M0 launch")
                launch["status"] = "failed"
                launch["finished_at_utc"] = _utc_now()
                launch["failure_error"] = rendered
                _write_json_atomic(launch_path, launch)
        paper_launcher._update_orchestration_status(
            status_path,
            status="failed",
            finished_at_utc=_utc_now(),
            error=rendered,
        )
        raise
    paper_launcher._update_orchestration_status(
        status_path,
        status="completed",
        finished_at_utc=_utc_now(),
        current_run_id=None,
        current_phase_id=None,
        completed_run_ids=[run_id],
        resume_required=None,
    )
    return result


def queue_spec(contract_id: str) -> dict[str, Any]:
    contract = _contract(contract_id)
    return {
        "schema": "pivot.stageb.headline_m0_queue_spec/v1",
        "contract_id": contract.id,
        "runner": str(Path(__file__).resolve(strict=True)),
        "runner_sha256": _sha256_file(Path(__file__).resolve(strict=True)),
        "ordered_run_ids": list(contract.dedicated_queue_run_ids),
        "separate_queue_required": True,
        "mixed_M0_M0N_queue_forbidden": True,
        "canonical_create_entrypoint": [
            str(DEFAULT_PYTHON.resolve(strict=True)),
            str(Path(__file__).resolve(strict=True)),
            "create-queue",
            "QUEUE_DIR",
            contract.id,
        ],
        "runtime": {
            "batch_size": FORMAL_BATCH_SIZE,
            "optimizer_updates": FORMAL_UPDATES,
            "successful_update_batch_slots": FORMAL_BATCH_SLOTS,
            "num_workers": FORMAL_NUM_WORKERS,
            "prefetch_factor": FORMAL_PREFETCH_FACTOR,
            "amp": True,
            "iter_checkpoint_interval": FORMAL_CHECKPOINT_INTERVAL,
            "gradient_diagnostic_interval": FORMAL_GRADIENT_DIAGNOSTIC_INTERVAL,
            "gradient_accumulation_steps": 1,
            "pin_memory": True,
            "persistent_workers": False,
        },
        "serial_queue_create_arguments": [
            "--runner-python",
            str(DEFAULT_PYTHON.resolve(strict=True)),
            "--paper-runner",
            str(Path(__file__).resolve(strict=True)),
            *[
                value
                for run_id in contract.dedicated_queue_run_ids
                for value in ("--run-id", run_id)
            ],
        ],
    }


def _validate_queue_runtime_snapshot(snapshot: Mapping[str, Any]) -> None:
    expected = {
        "PIVOT_PYTHON": str(DEFAULT_PYTHON),
        "PIVOT_STAGE_A_INIT": str(DEFAULT_STAGE_A_INIT),
        "PIVOT_SCORER_WARMSTART": str(DEFAULT_STAGE_A_INIT),
        "PIVOT_BATCH_SIZE": str(FORMAL_BATCH_SIZE),
        "PIVOT_MAX_TRAIN_ITERS": str(FORMAL_UPDATES),
        "PIVOT_ITER_CHECKPOINT_INTERVAL": str(FORMAL_CHECKPOINT_INTERVAL),
        "PIVOT_NUM_WORKERS": str(FORMAL_NUM_WORKERS),
        "PIVOT_PREFETCH_FACTOR": str(FORMAL_PREFETCH_FACTOR),
        "PIVOT_OMP_NUM_THREADS": "8",
        "PIVOT_MIN_NOFILE": "65536",
        "PIVOT_DATA_ROOT": str(DEFAULT_DATA_ROOT),
        "PIVOT_MP_SHARING_STRATEGY": "file_system",
        "PIVOT_GRADIENT_DIAGNOSTIC_INTERVAL": str(
            FORMAL_GRADIENT_DIAGNOSTIC_INTERVAL
        ),
    }
    for key, expected_value in expected.items():
        observed = snapshot.get(key)
        if observed is not None and observed != expected_value:
            raise HeadlineM0Error(
                f"training queue runtime {key} expected null/default or "
                f"{expected_value!r}, got {observed!r}"
            )
    data_root = snapshot.get("DATA_ROOT")
    if data_root is not None and data_root != str(DEFAULT_DATA_ROOT):
        raise HeadlineM0Error("training queue DATA_ROOT drifted")
    visible = snapshot.get("PIVOT_CUDA_VISIBLE_DEVICES")
    if visible is None:
        visible = snapshot.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and (not str(visible).strip() or "," in str(visible)):
        raise HeadlineM0Error("training queue must bind exactly one CUDA device")


def create_training_queue(
    queue_dir: Path,
    contract_id: str,
    *,
    lease_root: Path | None = None,
    gpu_key: str | None = None,
) -> dict[str, Any]:
    from tools import run_stageb_serial_matrix_queue as serial_queue

    contract = _contract(contract_id)
    runtime = runtime_from_environment()
    extension = _training_queue_contract_payload(
        runtime, contract, token_launcher.HashCache()
    )
    if lease_root is None:
        lease_root = serial_queue.DEFAULT_LEASE_ROOT
    try:
        queue = serial_queue.create_queue(
            queue_dir,
            run_ids=contract.dedicated_queue_run_ids,
            runner_python=DEFAULT_PYTHON,
            token_runner=serial_queue.DEFAULT_TOKEN_RUNNER,
            paper_runner=Path(__file__),
            lease_root=lease_root,
            gpu_key=gpu_key,
            plan_extensions={TRAINING_QUEUE_EXTENSION_KEY: extension},
        )
    except (OSError, ValueError, serial_queue.QueueContractError) as exc:
        raise HeadlineM0Error(f"cannot create formal training queue: {exc}") from exc
    verify_training_queue(queue_dir, contract.id)
    return queue


def _completed_stat_identity(stat: os.stat_result) -> tuple[int, int, int, int]:
    return (int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))


def _stable_completed_file_record(path: Path) -> dict[str, Any]:
    """Hash one path through a stable descriptor and prove the path still names it."""

    path = path.expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    if _completed_stat_identity(before) != _completed_stat_identity(after):
        raise HeadlineM0Error(f"completed evidence changed while hashing: {path}")
    current = path.stat()
    if _completed_stat_identity(current) != _completed_stat_identity(after):
        raise HeadlineM0Error(f"completed evidence path changed while hashing: {path}")
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
    }


def _read_completed_json_stably(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.expanduser().resolve(strict=True)
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        data = handle.read()
        after = os.fstat(handle.fileno())
    if _completed_stat_identity(before) != _completed_stat_identity(after):
        raise HeadlineM0Error(f"{label} changed while reading: {path}")
    current = path.stat()
    if _completed_stat_identity(current) != _completed_stat_identity(after):
        raise HeadlineM0Error(f"{label} path changed while reading: {path}")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HeadlineM0Error(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HeadlineM0Error(f"{label} must be a JSON object: {path}")
    return value, {
        "path": str(path),
        "sha256": _sha256_bytes(data),
        "size_bytes": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
    }


def _completed_evidence_snapshot(paths: Iterable[Path]) -> dict[str, Any]:
    ordered = sorted(
        {Path(path).expanduser().resolve(strict=True) for path in paths}, key=str
    )
    if not ordered:
        raise HeadlineM0Error("completed-training evidence snapshot is empty")
    records = [_stable_completed_file_record(path) for path in ordered]
    digest = _sha256_bytes(
        _canonical_json_bytes(
            {
                "schema": COMPLETED_TRAINING_EVIDENCE_SNAPSHOT_SCHEMA,
                "records": records,
            }
        )
    )
    return {
        "schema": COMPLETED_TRAINING_EVIDENCE_SNAPSHOT_SCHEMA,
        "algorithm": "sha256_stable_descriptor_path_content_size_mtime_v1",
        "records": records,
        "digest": digest,
    }


def _validate_completed_evidence_snapshot(
    value: Any, *, label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "algorithm",
        "records",
        "digest",
    }:
        raise HeadlineM0Error(f"{label} evidence snapshot is invalid")
    records = value.get("records")
    if not isinstance(records, list) or not records:
        raise HeadlineM0Error(f"{label} evidence snapshot is empty")
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "sha256",
            "size_bytes",
            "mtime_ns",
        }:
            raise HeadlineM0Error(f"{label} evidence record {index} is invalid")
        path = Path(str(record.get("path", ""))).expanduser().resolve(strict=False)
        sha256 = str(record.get("sha256", ""))
        size = record.get("size_bytes")
        mtime = record.get("mtime_ns")
        if (
            not path.is_absolute()
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
            or type(size) is not int
            or size < 0
            or type(mtime) is not int
            or mtime < 0
        ):
            raise HeadlineM0Error(f"{label} evidence record {index} drifted")
        normalized.append(
            {
                "path": str(path),
                "sha256": sha256,
                "size_bytes": size,
                "mtime_ns": mtime,
            }
        )
    if normalized != sorted(normalized, key=lambda record: record["path"]) or len(
        {record["path"] for record in normalized}
    ) != len(normalized):
        raise HeadlineM0Error(f"{label} evidence records are not canonical")
    digest = _sha256_bytes(
        _canonical_json_bytes(
            {
                "schema": COMPLETED_TRAINING_EVIDENCE_SNAPSHOT_SCHEMA,
                "records": normalized,
            }
        )
    )
    if (
        value.get("schema") != COMPLETED_TRAINING_EVIDENCE_SNAPSHOT_SCHEMA
        or value.get("algorithm")
        != "sha256_stable_descriptor_path_content_size_mtime_v1"
        or value.get("digest") != digest
    ):
        raise HeadlineM0Error(f"{label} evidence snapshot digest drifted")
    return {
        "schema": COMPLETED_TRAINING_EVIDENCE_SNAPSHOT_SCHEMA,
        "algorithm": "sha256_stable_descriptor_path_content_size_mtime_v1",
        "records": normalized,
        "digest": digest,
    }


def _require_same_completed_evidence(
    before: Mapping[str, Any], after: Mapping[str, Any], *, label: str
) -> None:
    if dict(before) != dict(after):
        raise HeadlineM0Error(f"{label} evidence identity changed during replay")


def _verify_completed_evidence_current(
    snapshot: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    expected = _validate_completed_evidence_snapshot(snapshot, label=label)
    observed = _completed_evidence_snapshot(
        Path(record["path"]) for record in expected["records"]
    )
    _require_same_completed_evidence(expected, observed, label=label)
    return observed


def _completed_evidence_record_map(
    snapshot: Mapping[str, Any], *, label: str
) -> dict[Path, dict[str, Any]]:
    validated = _validate_completed_evidence_snapshot(snapshot, label=label)
    return {
        Path(record["path"]).resolve(strict=False): dict(record)
        for record in validated["records"]
    }


def _completed_stable_input_snapshot(
    records: Iterable[Mapping[str, Any]], *, label: str
) -> dict[str, Any]:
    expected_by_path: dict[Path, dict[str, Any]] = {}
    for record in records:
        compact = _compact_file_record(record)
        path = Path(compact["path"]).resolve(strict=True)
        previous = expected_by_path.setdefault(path, compact)
        if previous != compact:
            raise HeadlineM0Error(f"{label} stable input identities conflict for {path}")
    if not expected_by_path:
        raise HeadlineM0Error(f"{label} stable input closure is empty")
    snapshot = _completed_evidence_snapshot(expected_by_path)
    observed = _completed_evidence_record_map(snapshot, label=label)
    for path, expected in expected_by_path.items():
        if _compact_file_record(observed[path]) != expected:
            raise HeadlineM0Error(f"{label} stable input identity changed: {path}")
    return snapshot


def _completed_run_evidence_paths(
    run_root: Path,
    *,
    seed: int,
    attempts: Sequence[Mapping[str, Any]],
) -> list[Path]:
    paths = {
        run_root / "sequence_manifest.json",
        run_root / "launch_manifest.json",
        run_root / "postflight.json",
        run_root / "checkpoint_iter.pth",
        run_root / "info.txt",
        run_root / "train_console.log",
        run_root / "gpu_environment.json",
        run_root / "gpu_telemetry.csv",
        run_root / "gpu_telemetry_summary.json",
        run_root / "stage_b_v15_scorer_init_audit.json",
        run_root / "input_rehash.json",
    }
    for ordinal, attempt in enumerate(attempts):
        attempt_dir, closure_path, attempt_path = _attempt_paths(run_root, ordinal)
        paths.update({attempt_path, closure_path, attempt_dir / "train_console.log"})
        paths.update(_attempt_telemetry_paths(run_root, ordinal).values())
        checkpoint = attempt.get("checkpoint_at_exit")
        if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("path"), str):
            paths.add(Path(str(checkpoint["path"])))
        authorization = attempt.get("resume_authorization")
        if isinstance(authorization, Mapping) and isinstance(
            authorization.get("path"), str
        ):
            paths.add(Path(str(authorization["path"])))
        if ordinal > 0:
            paths.add(run_root / "control" / "resume_requests" / f"{ordinal:03d}.json")
    if seed == 17:
        for update in MILESTONE_UPDATES:
            root = run_root / "milestones" / f"checkpoint_iter_{update:06d}"
            paths.add(root.with_suffix(".pth"))
            paths.add(root.with_suffix(".json"))
    return sorted(paths, key=lambda path: str(path.resolve(strict=False)))


def _inspect_completed_checkpoint_snapshot(
    path: Path,
    *,
    expected_record: Mapping[str, Any],
    python: Path,
    label: str,
) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    expected = _compact_file_record(expected_record)
    before = _compact_file_record(_stable_completed_file_record(path))
    if before != expected:
        raise HeadlineM0Error(f"{label} identity changed before safe inspection")
    with tempfile.TemporaryDirectory(prefix="pivot-m0-checkpoint-inspect-") as temporary:
        copied, digest = _copy_checkpoint_to_temporary(path, Path(temporary))
        copied_record = _compact_file_record(_stable_completed_file_record(copied))
        if (
            digest != expected["sha256"]
            or copied_record["sha256"] != expected["sha256"]
            or copied_record["size_bytes"] != expected["size_bytes"]
        ):
            raise HeadlineM0Error(
                f"{label} stable inspection snapshot differs from its sealed bytes"
            )
        copied.chmod(0o400)
        with copied.open("rb") as handle:
            descriptor_before = os.fstat(handle.fileno())
            metadata = _inspect_checkpoint_extended(
                copied, python=python, stable_fd=handle.fileno()
            )
            descriptor_after = os.fstat(handle.fileno())
        if _completed_stat_identity(descriptor_before) != _completed_stat_identity(
            descriptor_after
        ):
            raise HeadlineM0Error(f"{label} private descriptor changed during inspection")
        copied_after = _compact_file_record(_stable_completed_file_record(copied))
        if copied_after != copied_record:
            raise HeadlineM0Error(f"{label} private snapshot changed during inspection")
    after = _compact_file_record(_stable_completed_file_record(path))
    if after != before:
        raise HeadlineM0Error(f"{label} identity changed during safe inspection")
    return metadata


def _verify_completed_file_record(
    value: Any,
    *,
    label: str,
    expected_path: Path | None = None,
    compact: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HeadlineM0Error(f"{label} file record is invalid")
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise HeadlineM0Error(f"{label} file record has no path")
    try:
        path = Path(raw_path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise HeadlineM0Error(f"{label} artifact is missing: {raw_path}") from exc
    if expected_path is not None and path != expected_path.resolve(strict=True):
        raise HeadlineM0Error(f"{label} path drifted: {path}")
    observed = _file_record(path)
    if compact:
        observed = _compact_file_record(observed)
    if dict(value) != observed:
        raise HeadlineM0Error(f"{label} identity changed")
    return observed


def _parse_completed_timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise HeadlineM0Error(f"{label} is not an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HeadlineM0Error(f"{label} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise HeadlineM0Error(f"{label} timestamp lacks a timezone")
    return parsed


def _completed_training_verifier_source_paths(
    contract: source_contracts.FormalPaperRunContract,
) -> list[Path]:
    paths = set(_repository_dependency_paths(contract))
    paths.add(Path(__file__).resolve(strict=True))
    return sorted(paths, key=lambda path: str(path))


def _completed_training_verifier_source_snapshot(
    contract: source_contracts.FormalPaperRunContract,
) -> dict[str, Any]:
    records = [
        _compact_file_record(_file_record(path))
        for path in _completed_training_verifier_source_paths(contract)
    ]
    if not records:
        raise HeadlineM0Error("completed-training verifier source closure is empty")
    if records != sorted(records, key=lambda record: record["path"]):
        raise HeadlineM0Error("completed-training verifier sources are not canonical")
    if len({record["path"] for record in records}) != len(records):
        raise HeadlineM0Error("completed-training verifier source closure is duplicated")
    digest = _sha256_bytes(
        _canonical_json_bytes(
            {
                "schema": COMPLETED_TRAINING_SOURCE_SNAPSHOT_SCHEMA,
                "records": records,
            }
        )
    )
    return {
        "schema": COMPLETED_TRAINING_SOURCE_SNAPSHOT_SCHEMA,
        "algorithm": "sha256_canonical_path_content_size_v1",
        "records": records,
        "digest": digest,
    }


def _require_same_verifier_sources(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if dict(before) != dict(after):
        raise HeadlineM0Error(f"{label} source identity changed during replay")


def _runtime_from_completed_launch(
    launch: Mapping[str, Any],
    *,
    run_root: Path,
) -> Runtime:
    value = launch.get("runtime")
    expected_keys = {
        "python",
        "batch_size",
        "phase_train_iters",
        "total_paper_train_iters",
        "max_train_iters",
        "iter_checkpoint_interval",
        "num_workers",
        "prefetch_factor",
        "omp_num_threads",
        "min_nofile",
        "cuda_visible_devices",
        "mp_sharing_strategy",
        "amp",
        "gradient_accumulation_steps",
        "gradient_diagnostic_interval",
        "telemetry_interval_seconds",
        "pin_memory",
        "persistent_workers",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise HeadlineM0Error("completed launch runtime field set drifted")
    visible = value.get("cuda_visible_devices")
    if not isinstance(visible, str) or not visible.strip() or "," in visible:
        raise HeadlineM0Error("completed launch does not bind exactly one CUDA device")
    expected = {
        "python": str(DEFAULT_PYTHON.resolve(strict=True)),
        "batch_size": FORMAL_BATCH_SIZE,
        "phase_train_iters": FORMAL_UPDATES,
        "total_paper_train_iters": FORMAL_UPDATES,
        "max_train_iters": FORMAL_UPDATES,
        "iter_checkpoint_interval": FORMAL_CHECKPOINT_INTERVAL,
        "num_workers": FORMAL_NUM_WORKERS,
        "prefetch_factor": FORMAL_PREFETCH_FACTOR,
        "omp_num_threads": 8,
        "min_nofile": 65_536,
        "mp_sharing_strategy": "file_system",
        "amp": True,
        "gradient_accumulation_steps": 1,
        "gradient_diagnostic_interval": FORMAL_GRADIENT_DIAGNOSTIC_INTERVAL,
        "telemetry_interval_seconds": FORMAL_TELEMETRY_INTERVAL_SECONDS,
        "pin_memory": True,
        "persistent_workers": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise HeadlineM0Error(
                f"completed launch runtime {key} expected {expected_value!r}, "
                f"got {value.get(key)!r}"
            )
    runtime = Runtime(
        python=DEFAULT_PYTHON.resolve(strict=True),
        stage_a_init=DEFAULT_STAGE_A_INIT.resolve(strict=True),
        dataset=DEFAULT_DATASET.resolve(strict=True),
        output_root=DEFAULT_OUTPUT_ROOT.resolve(strict=False),
        data_root=DEFAULT_DATA_ROOT.resolve(strict=True),
        batch_size=FORMAL_BATCH_SIZE,
        max_train_iters=FORMAL_UPDATES,
        iter_checkpoint_interval=FORMAL_CHECKPOINT_INTERVAL,
        num_workers=FORMAL_NUM_WORKERS,
        prefetch_factor=FORMAL_PREFETCH_FACTOR,
        omp_num_threads=8,
        min_nofile=65_536,
        cuda_visible_devices=visible,
        mp_sharing_strategy="file_system",
        gradient_diagnostic_interval=FORMAL_GRADIENT_DIAGNOSTIC_INTERVAL,
        telemetry_interval_seconds=FORMAL_TELEMETRY_INTERVAL_SECONDS,
        pin_memory=True,
        persistent_workers=False,
        gradient_accumulation_steps=1,
    )
    if _resolved_path(launch.get("output_dir")) != run_root:
        raise HeadlineM0Error("completed launch output root drifted")
    return runtime


def _verify_completed_launch_contract(
    sequence: Mapping[str, Any],
    launch: Mapping[str, Any],
    *,
    contract: source_contracts.FormalPaperRunContract,
    seed: int,
    run_root: Path,
) -> tuple[Runtime, dict[str, Any]]:
    run_id = f"{contract.id}:{seed}"
    expected_root = contract.canonical_training_root(seed).resolve(strict=False)
    if run_root != expected_root:
        raise HeadlineM0Error(
            f"{run_id} completed root is not canonical: {run_root} != {expected_root}"
        )
    expected_sequence_keys = {
        "schema",
        "status",
        "created_at_utc",
        "repository_root",
        "run_id",
        "row",
        "seed",
        "training_seeds_contract",
        "output_dir",
        "output_dir_fresh_at_plan",
        "equal_budget_contract",
        "stable_input_closure_digest",
        "one_attempt_execution",
        "resume_policy",
        "phases",
        "training_queue_binding",
        "started_at_utc",
        "finished_at_utc",
        "completed_phases",
        "training_attempt_count",
        "same_run_resume_count",
    }
    expected_launch_keys = {
        "schema",
        "status",
        "created_at_utc",
        "run_id",
        "row",
        "seed",
        "phase_id",
        "phase",
        "output_dir",
        "command",
        "command_shell",
        "runtime",
        "fixed_contract",
        "inputs",
        "training_queue_binding",
        "started_at_utc",
        "gpu_environment",
        "gpu_telemetry_summary",
        "postflight",
        "postflight_artifact",
        "returncode",
        "finished_at_utc",
    }
    if (
        set(sequence) != expected_sequence_keys
        or sequence.get("schema") != TRAINING_SEQUENCE_SCHEMA
        or sequence.get("status") != "completed"
        or sequence.get("repository_root") != str(REPO_ROOT)
        or sequence.get("run_id") != run_id
        or sequence.get("row") != contract.expected_row()
        or sequence.get("seed") != seed
        or sequence.get("training_seeds_contract") != list(contract.seeds)
        or _resolved_path(sequence.get("output_dir")) != run_root
        or sequence.get("output_dir_fresh_at_plan") is not True
        or sequence.get("equal_budget_contract") != contract.expected_budget()
        or sequence.get("one_attempt_execution") is not True
        or sequence.get("resume_policy")
        != "explicit_authorization_complete_same_run_mid_epoch_signal_only"
    ):
        raise HeadlineM0Error(f"{run_id} completed sequence contract drifted")
    if (
        set(launch) != expected_launch_keys
        or launch.get("schema") != TRAINING_PHASE_SCHEMA
        or launch.get("status") != "completed"
        or launch.get("run_id") != run_id
        or launch.get("row") != contract.expected_row()
        or launch.get("seed") != seed
        or launch.get("phase_id") != "joint"
        or launch.get("phase") != _phase(contract)
        or launch.get("returncode") != 0
    ):
        raise HeadlineM0Error(f"{run_id} completed launch contract drifted")
    sequence_created = _parse_completed_timestamp(
        sequence.get("created_at_utc"), label=f"{run_id} sequence creation"
    )
    launch_created = _parse_completed_timestamp(
        launch.get("created_at_utc"), label=f"{run_id} launch creation"
    )
    sequence_started = _parse_completed_timestamp(
        sequence.get("started_at_utc"), label=f"{run_id} sequence start"
    )
    launch_started = _parse_completed_timestamp(
        launch.get("started_at_utc"), label=f"{run_id} launch start"
    )
    sequence_finished = _parse_completed_timestamp(
        sequence.get("finished_at_utc"), label=f"{run_id} sequence finish"
    )
    launch_finished = _parse_completed_timestamp(
        launch.get("finished_at_utc"), label=f"{run_id} launch finish"
    )
    if not (
        launch_created <= sequence_created <= sequence_started == launch_started
        <= sequence_finished == launch_finished
    ):
        raise HeadlineM0Error(f"{run_id} sequence/launch chronology forked")
    runtime = _runtime_from_completed_launch(launch, run_root=run_root)
    config = _validate_config(contract)
    dataset_contract, _ = _validate_dataset(runtime)
    expected_fixed = {
        "architecture_objective": "S2F",
        "compute_contract": "b58_successful_update_batch_slot_matched",
        "successful_update_batch_slots": FORMAL_BATCH_SLOTS,
        "candidate_topk": 50,
        "positive_iou_threshold": 0.5,
        "negative_iou_threshold": 0.499,
        "token_objective": (
            "edit_bce" if contract.id == "M0" else "targetlocal_allneg_bce"
        ),
        "token_objective_scope": contract.token_objective_scope,
        "predicate_pair_rank_weight": 1.0,
        "stage_a_and_scorer_same_source": True,
        "b58_model_ancestry_forbidden": True,
        "dataset": dataset_contract,
        "optimizer_resume": "same_run_mid_epoch_signal_only",
    }
    if launch.get("fixed_contract") != expected_fixed:
        raise HeadlineM0Error(f"{run_id} fixed training contract drifted")
    queue_binding = sequence.get("training_queue_binding")
    if not isinstance(queue_binding, Mapping) or not queue_binding:
        raise HeadlineM0Error(f"{run_id} lacks a formal training queue binding")
    if launch.get("training_queue_binding") != queue_binding:
        raise HeadlineM0Error(f"{run_id} launch/sequence queue binding forked")
    phases = sequence.get("phases")
    planned = phases[0] if isinstance(phases, list) and len(phases) == 1 else None
    if not isinstance(planned, Mapping):
        raise HeadlineM0Error(f"{run_id} sequence has no unique planned phase")
    expected_planned_keys = {
        "schema",
        "status",
        "created_at_utc",
        "run_id",
        "row",
        "seed",
        "phase_id",
        "phase",
        "output_dir",
        "command",
        "command_shell",
        "runtime",
        "fixed_contract",
        "inputs",
        "training_queue_binding",
    }
    if set(planned) != expected_planned_keys:
        raise HeadlineM0Error(f"{run_id} planned phase field set drifted")
    if planned.get("training_queue_binding") != queue_binding:
        raise HeadlineM0Error(f"{run_id} planned phase queue binding forked")
    expected_fresh_command = build_command(runtime, contract, seed, run_root)
    if (
        planned.get("schema") != TRAINING_PHASE_SCHEMA
        or planned.get("status") != "planned"
        or planned.get("created_at_utc") != launch.get("created_at_utc")
        or planned.get("run_id") != run_id
        or planned.get("row") != contract.expected_row()
        or planned.get("seed") != seed
        or planned.get("phase_id") != "joint"
        or planned.get("phase") != _phase(contract)
        or _resolved_path(planned.get("output_dir")) != run_root
        or planned.get("runtime") != launch.get("runtime")
        or planned.get("fixed_contract") != expected_fixed
        or planned.get("command") != expected_fresh_command
        or planned.get("command_shell") != shlex.join(expected_fresh_command)
    ):
        raise HeadlineM0Error(f"{run_id} immutable planned phase drifted")
    launch_closure = _stable_closure_from_manifest(launch)
    planned_closure = _stable_closure_from_manifest(planned)
    if (
        launch_closure != planned_closure
        or sequence.get("stable_input_closure_digest")
        != launch_closure["digest"]
    ):
        raise HeadlineM0Error(f"{run_id} stable input closure forked")
    _verify_input_identities(launch, rehash=True)
    source_paths = {
        str(path)
        for path in _completed_training_verifier_source_paths(contract)
    }
    sealed_source_paths = {
        str(Path(str(record.get("path", ""))).resolve(strict=True))
        for record in _iter_input_records(launch)
        if record.get("role") == "repository_source"
    }
    if source_paths != sealed_source_paths:
        raise HeadlineM0Error(f"{run_id} verifier/source closure differs from launch")
    stage_a_records = [
        record
        for record in _iter_input_records(launch)
        if record.get("role") in {"stage_a_initializer", "scorer_warmstart"}
    ]
    role_counts = {
        role: sum(record.get("role") == role for record in stage_a_records)
        for role in ("stage_a_initializer", "scorer_warmstart")
    }
    expected_stage_a = _compact_file_record(_file_record(runtime.stage_a_init))
    if (
        expected_stage_a["sha256"] != DEFAULT_STAGE_A_SHA256
        or role_counts != {"stage_a_initializer": 1, "scorer_warmstart": 1}
        or any(_compact_file_record(record) != expected_stage_a for record in stage_a_records)
        or any(
            record.get("sha256") == FORBIDDEN_B58_SHA256
            for record in _iter_input_records(launch)
        )
    ):
        raise HeadlineM0Error(f"{run_id} is not Stage-A-only/no-b58")
    if contract.id == "M0":
        forbidden_controls = {
            "stage_b_v25_control_of",
            "stage_b_v25_headline_eligible",
            "stage_b_v25_matrix_validation_only",
            "stage_b_v25_comparison_claim",
            "stage_b_v25_token_objective_scope",
        }
        if any(config.get(key) is not None for key in forbidden_controls):
            raise HeadlineM0Error("M0 config contains M0N-only control metadata")
    return runtime, launch_closure


def _verify_completed_input_rehash(
    value: Any,
    *,
    launch: Mapping[str, Any],
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "status",
        "algorithm",
        "verified_at_utc",
        "unique_input_count",
        "records",
    }:
        raise HeadlineM0Error("completed input rehash field set drifted")
    _parse_completed_timestamp(
        value.get("verified_at_utc"), label="completed input rehash time"
    )
    expected_by_path: dict[Path, dict[str, Any]] = {}
    roles_by_path: dict[Path, set[str]] = {}
    for record in _iter_input_records(launch):
        path = Path(str(record.get("path", ""))).resolve(strict=True)
        compact = _compact_file_record(record)
        previous = expected_by_path.setdefault(path, compact)
        if previous != compact:
            raise HeadlineM0Error(f"stable input identities conflict for {path}")
        roles_by_path.setdefault(path, set()).add(str(record.get("role")))
    expected_records = []
    for path in sorted(expected_by_path, key=lambda item: str(item)):
        observed = _compact_file_record(_file_record(path))
        expected = expected_by_path[path]
        stat = path.stat()
        expected_records.append(
            {
                "path": str(path),
                "roles": sorted(roles_by_path[path]),
                "expected_sha256": expected["sha256"],
                "observed_sha256": observed["sha256"],
                "observed_size_bytes": int(stat.st_size),
                "observed_mtime_ns": int(stat.st_mtime_ns),
                "passed": observed == expected,
            }
        )
    if (
        value.get("status") != "passed"
        or value.get("algorithm") != "sha256"
        or value.get("unique_input_count") != len(expected_records)
        or value.get("records") != expected_records
        or any(record["passed"] is not True for record in expected_records)
    ):
        raise HeadlineM0Error("completed input rehash does not replay the launch closure")


def _verify_completed_attempt_input_closure(
    value: Any,
    *,
    run_root: Path,
    ordinal: int,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    _, closure_path, _ = _attempt_paths(run_root, ordinal)
    record = _verify_completed_file_record(
        value,
        label=f"attempt {ordinal} input closure",
        expected_path=closure_path,
        compact=True,
    )
    payload = _read_json(closure_path, label=f"attempt {ordinal} input closure")
    if set(payload) != {"schema", "status", "algorithm", "records", "digest"}:
        raise HeadlineM0Error(f"attempt {ordinal} input closure field set drifted")
    if (
        payload.get("schema") != STABLE_CLOSURE_SCHEMA
        or payload.get("status") != "sealed"
        or payload.get("algorithm")
        != "sha256_canonical_path_content_size_roles_v1"
        or payload.get("records") != expected.get("records")
        or payload.get("digest") != expected.get("digest")
    ):
        raise HeadlineM0Error(f"attempt {ordinal} input closure forked")
    digest = _sha256_bytes(
        _canonical_json_bytes(
            {"schema": STABLE_CLOSURE_SCHEMA, "records": payload["records"]}
        )
    )
    if digest != payload["digest"]:
        raise HeadlineM0Error(f"attempt {ordinal} input closure digest drifted")
    return record


def _verify_completed_resume_authorization(
    value: Any,
    *,
    run_root: Path,
    run_id: str,
    ordinal: int,
    recovery_checkpoint: Mapping[str, Any],
) -> tuple[dict[str, Any], datetime]:
    path = run_root / "control" / "resume_requests" / f"{ordinal:03d}.json"
    record = _verify_completed_file_record(
        value,
        label=f"resume authorization {ordinal}",
        expected_path=path,
        compact=True,
    )
    payload = _read_json(path, label=f"resume authorization {ordinal}")
    expected_keys = {
        "schema",
        "status",
        "run_id",
        "next_attempt_ordinal",
        "recovery_checkpoint",
        "policy",
        "authorized_at_utc",
        "authorizer_pid",
        "detached_controller_identity",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema") != RESUME_REQUEST_SCHEMA
        or payload.get("status") != "authorized"
        or payload.get("run_id") != run_id
        or payload.get("next_attempt_ordinal") != ordinal
        or payload.get("recovery_checkpoint") != dict(recovery_checkpoint)
        or payload.get("policy") != "explicit_one_attempt_mid_epoch_signal_resume"
        or type(payload.get("authorizer_pid")) is not int
        or int(payload["authorizer_pid"]) <= 0
        or (
            payload.get("detached_controller_identity") is not None
            and not isinstance(payload.get("detached_controller_identity"), Mapping)
        )
    ):
        raise HeadlineM0Error(f"resume authorization {ordinal} drifted")
    return record, _parse_completed_timestamp(
        payload.get("authorized_at_utc"),
        label=f"resume authorization {ordinal} time",
    )


def _verify_completed_contract_checkpoint_args(
    metadata: Mapping[str, Any],
    *,
    runtime: Runtime,
    contract: source_contracts.FormalPaperRunContract,
    run_root: Path,
) -> None:
    args = metadata.get("args")
    if not isinstance(args, Mapping):
        raise HeadlineM0Error("completed checkpoint args metadata is missing")
    audit = args.get("stage_b_v15_scorer_init_audit")
    expected_audit_path = run_root / "stage_b_v15_scorer_init_audit.json"
    sidecar = _read_json(
        expected_audit_path, label="completed checkpoint scorer initializer audit"
    )
    audit_keys = {
        "schema",
        "status",
        "requested_source_path",
        "resolved_source_path",
        "source_sha256",
        "source_size_bytes",
        "source_decoder_num_layers",
        "selected_source_layer_indices",
        "loaded_num_layers",
        "loaded_tensor_count",
        "loaded_components",
    }
    stage_a = runtime.stage_a_init.resolve(strict=True)
    stage_a_record = _compact_file_record(_file_record(stage_a))
    if (
        not isinstance(audit, Mapping)
        or set(audit) != audit_keys
        or dict(audit) != sidecar
        or audit.get("schema") != "stage_b_v15_scorer_init/v1"
        or audit.get("status") != "applied"
        or _resolved_path(audit.get("requested_source_path")) != stage_a
        or _resolved_path(audit.get("resolved_source_path")) != stage_a
        or audit.get("source_sha256") != stage_a_record["sha256"]
        or audit.get("source_size_bytes") != stage_a_record["size_bytes"]
        or type(audit.get("source_decoder_num_layers")) is not int
        or int(audit["source_decoder_num_layers"]) < 3
        or audit.get("loaded_num_layers") != 3
        or audit.get("loaded_tensor_count") != 90
        or audit.get("selected_source_layer_indices")
        != list(
            range(
                int(audit["source_decoder_num_layers"]) - 3,
                int(audit["source_decoder_num_layers"]),
            )
        )
        or audit.get("loaded_components")
        != ["decoder.layers[-N:]", "decoder.ref_point_head", "decoder.norm"]
    ):
        raise HeadlineM0Error("completed checkpoint scorer initializer audit drifted")
    control_keys = {
        "stage_b_v25_control_of": "M0",
        "stage_b_v25_headline_eligible": False,
        "stage_b_v25_matrix_validation_only": True,
        "stage_b_v25_comparison_claim": (
            "full_token_objective_control_not_labels_only"
        ),
        "stage_b_v25_token_objective_scope": (
            "target_local_positive_and_all_negative_token_logits"
        ),
    }
    if contract.id == "M0N":
        if any(args.get(key) != expected for key, expected in control_keys.items()):
            raise HeadlineM0Error("completed M0N checkpoint control metadata drifted")
    elif any(args.get(key) is not None for key in control_keys):
        raise HeadlineM0Error("completed M0 checkpoint contains M0N-only metadata")


def _verify_completed_attempts(
    *,
    sequence: Mapping[str, Any],
    launch: Mapping[str, Any],
    postflight: Mapping[str, Any],
    runtime: Runtime,
    contract: source_contracts.FormalPaperRunContract,
    seed: int,
    run_root: Path,
    stable_closure: Mapping[str, Any],
    final_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    run_id = f"{contract.id}:{seed}"
    attempt_count = postflight.get("training_attempt_count")
    resume_count = postflight.get("same_run_resume_count")
    if (
        type(attempt_count) is not int
        or attempt_count <= 0
        or resume_count != attempt_count - 1
        or sequence.get("training_attempt_count") != attempt_count
        or sequence.get("same_run_resume_count") != resume_count
    ):
        raise HeadlineM0Error(f"{run_id} attempt/resume counts drifted")
    expected_attempt_keys = {
        "schema",
        "status",
        "run_id",
        "seed",
        "attempt_ordinal",
        "initialization_mode",
        "parent_attempt_manifest",
        "resume_checkpoint",
        "resume_authorization",
        "source_optimizer_updates",
        "target_optimizer_updates",
        "command",
        "command_shell",
        "runtime",
        "input_closure_digest",
        "input_closure",
        "telemetry",
        "process",
        "termination",
        "complete_state_components",
        "checkpoint_at_exit",
        "checkpoint_metadata",
        "started_at_utc",
        "finished_at_utc",
    }
    attempts: list[dict[str, Any]] = []
    attempt_records: list[dict[str, Any]] = []
    inspected: list[dict[str, Any]] = []
    attempt_telemetry: list[dict[str, Any]] = []
    previous_updates = 0
    previous_finish: datetime | None = None
    previous_checkpoint: dict[str, Any] | None = None
    previous_attempt_record: dict[str, Any] | None = None
    for ordinal in range(attempt_count):
        attempt_dir, _, attempt_path = _attempt_paths(run_root, ordinal)
        attempt_record = _compact_file_record(_file_record(attempt_path))
        attempt = _read_json(attempt_path, label=f"{run_id} attempt {ordinal}")
        if set(attempt) != expected_attempt_keys:
            raise HeadlineM0Error(f"{run_id} attempt {ordinal} field set drifted")
        process = attempt.get("process")
        process_keys = {
            "pid",
            "identity",
            "start_new_session",
            "stdin",
            "stdout_stderr",
            "returncode",
            "forwarded_signals",
            "started_at_utc",
            "finished_at_utc",
        }
        if (
            attempt.get("schema") != ATTEMPT_SCHEMA
            or attempt.get("status") != "completed"
            or attempt.get("run_id") != run_id
            or attempt.get("seed") != seed
            or attempt.get("attempt_ordinal") != ordinal
            or attempt.get("target_optimizer_updates") != FORMAL_UPDATES
            or attempt.get("runtime") != _attempt_runtime(runtime)
            or attempt.get("complete_state_components") != COMPLETE_STATE_COMPONENTS
            or attempt.get("input_closure_digest") != stable_closure["digest"]
            or not isinstance(process, Mapping)
            or set(process) != process_keys
            or type(process.get("pid")) is not int
            or int(process["pid"]) <= 0
            or not isinstance(process.get("identity"), Mapping)
            or process.get("start_new_session") is not True
            or process.get("stdin") != "DEVNULL"
            or process.get("returncode") != 0
            or not isinstance(process.get("forwarded_signals"), list)
            or not all(type(value) is int for value in process["forwarded_signals"])
        ):
            raise HeadlineM0Error(f"{run_id} attempt {ordinal} contract drifted")
        expected_log = (attempt_dir / "train_console.log").resolve(strict=True)
        if (
            _resolved_path(process.get("stdout_stderr")) != expected_log
            or expected_log.stat().st_size <= 0
        ):
            raise HeadlineM0Error(f"{run_id} attempt {ordinal} log drifted")
        started = _parse_completed_timestamp(
            attempt.get("started_at_utc"), label=f"{run_id} attempt {ordinal} start"
        )
        finished = _parse_completed_timestamp(
            attempt.get("finished_at_utc"), label=f"{run_id} attempt {ordinal} finish"
        )
        if (
            process.get("started_at_utc") != attempt.get("started_at_utc")
            or process.get("finished_at_utc") != attempt.get("finished_at_utc")
            or finished < started
            or (previous_finish is not None and started < previous_finish)
        ):
            raise HeadlineM0Error(f"{run_id} attempt {ordinal} chronology forked")
        _verify_completed_attempt_input_closure(
            attempt.get("input_closure"),
            run_root=run_root,
            ordinal=ordinal,
            expected=stable_closure,
        )
        attempt_telemetry.append(
            _verify_attempt_telemetry(
                attempt.get("telemetry"), run_root=run_root, ordinal=ordinal
            )
        )
        source_updates = attempt.get("source_optimizer_updates")
        if type(source_updates) is not int or source_updates != previous_updates:
            raise HeadlineM0Error(
                f"{run_id} attempt {ordinal} source updates are not contiguous"
            )
        if ordinal == 0:
            if (
                attempt.get("initialization_mode") != "fresh_stage_a"
                or attempt.get("parent_attempt_manifest") is not None
                or attempt.get("resume_checkpoint") is not None
                or attempt.get("resume_authorization") is not None
            ):
                raise HeadlineM0Error(f"{run_id} fresh attempt ancestry drifted")
            resume_path = None
        else:
            if previous_checkpoint is None or previous_attempt_record is None:
                raise AssertionError("resume replay lost its previous edge")
            authorization, authorized_at = _verify_completed_resume_authorization(
                attempt.get("resume_authorization"),
                run_root=run_root,
                run_id=run_id,
                ordinal=ordinal,
                recovery_checkpoint=previous_checkpoint,
            )
            if (
                attempt.get("initialization_mode") != "same_run_resume"
                or attempt.get("parent_attempt_manifest") != previous_attempt_record
                or attempt.get("resume_checkpoint") != previous_checkpoint
                or attempt.get("resume_authorization") != authorization
                or previous_finish is None
                or not previous_finish <= authorized_at <= started
            ):
                raise HeadlineM0Error(f"{run_id} resume edge {ordinal} forked")
            resume_path = Path(str(previous_checkpoint["path"])).resolve(strict=True)
            _validate_resume_log(expected_log, source_metadata=inspected[-1])
        expected_command = build_command(
            runtime,
            contract,
            seed,
            run_root,
            resume_checkpoint=resume_path,
        )
        if (
            attempt.get("command") != expected_command
            or attempt.get("command_shell") != shlex.join(expected_command)
        ):
            raise HeadlineM0Error(f"{run_id} attempt {ordinal} command drifted")
        termination = attempt.get("termination")
        if not isinstance(termination, Mapping):
            raise HeadlineM0Error(f"{run_id} attempt {ordinal} termination is invalid")
        is_final = ordinal == attempt_count - 1
        if is_final:
            checkpoint_record = _verify_completed_file_record(
                attempt.get("checkpoint_at_exit"),
                label=f"{run_id} final checkpoint",
                expected_path=run_root / "checkpoint_iter.pth",
                compact=True,
            )
            if (
                checkpoint_record != dict(final_checkpoint)
                or termination
                != {"kind": "target_completed", "reason": "max_train_iters"}
            ):
                raise HeadlineM0Error(f"{run_id} final attempt is not target-complete")
        else:
            checkpoint_record = _verify_completed_file_record(
                attempt.get("checkpoint_at_exit"),
                label=f"{run_id} recovery checkpoint {ordinal + 1}",
                compact=True,
            )
            if termination != {
                "kind": "graceful_signal_checkpoint",
                "reason": "signal",
            }:
                raise HeadlineM0Error(f"{run_id} recovery edge {ordinal + 1} drifted")
        metadata = _inspect_completed_checkpoint_snapshot(
            Path(checkpoint_record["path"]),
            expected_record=checkpoint_record,
            python=runtime.python,
            label=f"{run_id} attempt {ordinal} checkpoint",
        )
        reason = _validate_checkpoint_metadata(
            metadata,
            runtime=runtime,
            contract=contract,
            seed=seed,
            output_dir=run_root,
            source_optimizer_updates=previous_updates,
            resume_checkpoint=resume_path,
        )
        _verify_completed_contract_checkpoint_args(
            metadata, runtime=runtime, contract=contract, run_root=run_root
        )
        expected_reason = "max_train_iters" if is_final else "signal"
        if (
            reason != expected_reason
            or attempt.get("checkpoint_metadata")
            != _checkpoint_metadata_for_attempt(metadata)
        ):
            raise HeadlineM0Error(
                f"{run_id} attempt {ordinal} checkpoint metadata replay drifted"
            )
        checkpoint_after = _verify_completed_file_record(
            attempt.get("checkpoint_at_exit"),
            label=f"{run_id} attempt {ordinal} checkpoint post-inspection",
            expected_path=Path(checkpoint_record["path"]),
            compact=True,
        )
        if checkpoint_after != checkpoint_record:
            raise HeadlineM0Error(
                f"{run_id} attempt {ordinal} checkpoint changed during inspection"
            )
        updates = metadata.get("optimizer_updates")
        if type(updates) is not int or not previous_updates < updates:
            raise HeadlineM0Error(f"{run_id} attempt updates are not monotonic")
        if not is_final:
            path = Path(checkpoint_record["path"])
            expected_name = (
                f"attempt_{ordinal + 1:03d}_from_u{updates:06d}_"
                f"{checkpoint_record['sha256'][:12]}.pth"
            )
            if path.parent != (run_root / "recovery").resolve() or path.name != expected_name:
                raise HeadlineM0Error(f"{run_id} recovery checkpoint path drifted")
        attempts.append(attempt)
        attempt_records.append(attempt_record)
        inspected.append(metadata)
        previous_updates = updates
        previous_finish = finished
        previous_checkpoint = checkpoint_record
        previous_attempt_record = attempt_record
    final_metadata = inspected[-1]
    if previous_updates != FORMAL_UPDATES:
        raise HeadlineM0Error(f"{run_id} final optimizer updates are not U23532")
    if (
        launch.get("command") != attempts[-1].get("command")
        or launch.get("command_shell") != attempts[-1].get("command_shell")
    ):
        raise HeadlineM0Error(f"{run_id} final launch command differs from final attempt")
    first_started = _parse_completed_timestamp(
        attempts[0]["started_at_utc"], label=f"{run_id} first attempt start"
    )
    final_finished = _parse_completed_timestamp(
        attempts[-1]["finished_at_utc"], label=f"{run_id} final attempt finish"
    )
    launch_started = _parse_completed_timestamp(
        launch.get("started_at_utc"), label=f"{run_id} launch start"
    )
    launch_finished = _parse_completed_timestamp(
        launch.get("finished_at_utc"), label=f"{run_id} launch finish"
    )
    if first_started < launch_started or final_finished > launch_finished:
        raise HeadlineM0Error(f"{run_id} attempt chronology escapes its launch")
    rebuilt_ancestry = _build_ancestry(
        phase_manifest=launch,
        runtime=runtime,
        contract=contract,
        seed=seed,
        run_root=run_root,
        final_ordinal=attempt_count - 1,
    )
    ancestry = postflight.get("model_state_ancestry")
    rendered = json.dumps(ancestry, sort_keys=True)
    if (
        ancestry != rebuilt_ancestry
        or FORBIDDEN_B58_SHA256 in rendered
        or str(source_contracts.FIXED_BASELINE["checkpoint"]) in rendered
        or rebuilt_ancestry.get("b58_ancestry_count") != 0
        or rebuilt_ancestry.get("resume_chain_contiguous") is not True
        or len(rebuilt_ancestry.get("resume_ancestry", [])) != resume_count
    ):
        raise HeadlineM0Error(f"{run_id} Stage-A/resume ancestry replay drifted")
    return {
        "attempt_count": attempt_count,
        "resume_count": resume_count,
        "attempt_manifests": attempt_records,
        "attempt_telemetry": attempt_telemetry,
        "full_run_telemetry": _full_run_telemetry_projection(attempt_telemetry),
        "final_metadata": final_metadata,
        "ancestry": rebuilt_ancestry,
    }


def _verify_completed_postflight(
    *,
    sequence: Mapping[str, Any],
    launch: Mapping[str, Any],
    postflight: Mapping[str, Any],
    runtime: Runtime,
    contract: source_contracts.FormalPaperRunContract,
    seed: int,
    run_root: Path,
    stable_closure: Mapping[str, Any],
) -> dict[str, Any]:
    run_id = f"{contract.id}:{seed}"
    expected_postflight_keys = {
        "schema",
        "status",
        "validated_at_utc",
        "run_id",
        "seed",
        "phase_id",
        "checkpoint_metadata",
        "optimizer_progress",
        "input_rehash",
        "gpu_environment",
        "gpu_telemetry_summary",
        "full_run_telemetry",
        "numerical_status",
        "checkpoint_cuda_memory",
        "artifacts",
        "model_state_ancestry",
        "scorer_initializer_audit",
        "training_attempt_count",
        "same_run_resume_count",
        "milestones",
        "formal_claim",
    }
    if (
        set(postflight) != expected_postflight_keys
        or postflight.get("schema") != POSTFLIGHT_SCHEMA
        or postflight.get("status") != "passed"
        or postflight.get("run_id") != run_id
        or postflight.get("seed") != seed
        or postflight.get("phase_id") != "joint"
        or postflight.get("formal_claim")
        != "successful_optimizer_update_batch_slot_matched_not_flop_or_wall_clock_matched"
    ):
        raise HeadlineM0Error(f"{run_id} postflight contract drifted")
    _parse_completed_timestamp(
        postflight.get("validated_at_utc"), label=f"{run_id} postflight time"
    )
    artifact_paths = {
        "checkpoint": run_root / "checkpoint_iter.pth",
        "native_info_log": run_root / "info.txt",
        "train_console_log": run_root / "train_console.log",
        "gpu_environment": run_root / "gpu_environment.json",
        "gpu_telemetry": run_root / "gpu_telemetry.csv",
        "gpu_telemetry_summary": run_root / "gpu_telemetry_summary.json",
        "scorer_init_audit": run_root / "stage_b_v15_scorer_init_audit.json",
        "input_rehash": run_root / "input_rehash.json",
    }
    artifacts = postflight.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(artifact_paths):
        raise HeadlineM0Error(f"{run_id} postflight artifact set drifted")
    verified_artifacts = {
        name: _verify_completed_file_record(
            artifacts.get(name), label=f"{run_id} {name}", expected_path=path
        )
        for name, path in artifact_paths.items()
    }
    for name in ("native_info_log", "train_console_log", "gpu_telemetry"):
        if artifact_paths[name].stat().st_size <= 0:
            raise HeadlineM0Error(f"{run_id} {name} is empty")
    final_checkpoint = _compact_file_record(verified_artifacts["checkpoint"])
    attempts = _verify_completed_attempts(
        sequence=sequence,
        launch=launch,
        postflight=postflight,
        runtime=runtime,
        contract=contract,
        seed=seed,
        run_root=run_root,
        stable_closure=stable_closure,
        final_checkpoint=final_checkpoint,
    )
    if postflight.get("full_run_telemetry") != attempts["full_run_telemetry"]:
        raise HeadlineM0Error(f"{run_id} full-run GPU telemetry projection drifted")
    final_attempt_telemetry = attempts["attempt_telemetry"][-1]["artifacts"]
    for name in ("gpu_environment", "gpu_telemetry", "gpu_telemetry_summary"):
        archived_record = final_attempt_telemetry[name]
        root_record = _compact_file_record(verified_artifacts[name])
        if (
            archived_record["sha256"] != root_record["sha256"]
            or archived_record["size_bytes"] != root_record["size_bytes"]
        ):
            raise HeadlineM0Error(
                f"{run_id} canonical {name} is not the final attempt projection"
            )
    metadata = attempts["final_metadata"]
    if postflight.get("checkpoint_metadata") != metadata:
        raise HeadlineM0Error(f"{run_id} final checkpoint metadata is not safe-load replayed")
    expected_progress = {
        "status": "passed",
        "optimizer_updates": FORMAL_UPDATES,
        "consumed_microbatches": FORMAL_UPDATES,
        "gradient_accumulation_steps": 1,
        "data_loader_microbatches_per_epoch": FORMAL_DATALOADER_MICROBATCHES,
        "checkpoint_epoch": FORMAL_FINAL_EPOCH,
        "checkpoint_iteration": FORMAL_FINAL_ITERATION,
        "checkpoint_epoch_finished": False,
        "checkpoint_reason": "max_train_iters",
        "optimizer_state_count": FORMAL_OPTIMIZER_STATE_COUNT,
        "optimizer_step_values": [FORMAL_UPDATES],
        "checkpoint_optimizer_step": FORMAL_UPDATES,
        "successful_update_batch_slots": FORMAL_BATCH_SLOTS,
        "successful_updates_equal_consumed_microbatches": True,
    }
    if postflight.get("optimizer_progress") != expected_progress:
        raise HeadlineM0Error(f"{run_id} B40/U23532 optimizer progress drifted")
    checkpoint_memory = metadata.get("checkpoint_cuda_memory")
    expected_memory = {
        "available": bool(
            isinstance(checkpoint_memory, Mapping)
            and any(value is not None for value in checkpoint_memory.values())
        ),
        "values": dict(checkpoint_memory) if isinstance(checkpoint_memory, Mapping) else {},
    }
    if postflight.get("checkpoint_cuda_memory") != expected_memory:
        raise HeadlineM0Error(f"{run_id} checkpoint CUDA memory evidence drifted")

    input_rehash = _read_json(
        artifact_paths["input_rehash"], label=f"{run_id} input rehash"
    )
    if postflight.get("input_rehash") != input_rehash:
        raise HeadlineM0Error(f"{run_id} embedded input rehash forked")
    _verify_completed_input_rehash(input_rehash, launch=launch)

    gpu_environment = _read_json(
        artifact_paths["gpu_environment"], label=f"{run_id} GPU environment"
    )
    gpu_summary = _read_json(
        artifact_paths["gpu_telemetry_summary"],
        label=f"{run_id} GPU telemetry summary",
    )
    if (
        postflight.get("gpu_environment") != gpu_environment
        or postflight.get("gpu_telemetry_summary") != gpu_summary
        or launch.get("gpu_environment") != gpu_environment
        or launch.get("gpu_telemetry_summary") != gpu_summary
        or gpu_summary.get("sampling_interval_ms") != 1000
    ):
        raise HeadlineM0Error(f"{run_id} embedded GPU telemetry evidence forked")
    try:
        paper_launcher._validate_gpu_telemetry_contract(
            gpu_environment, gpu_summary
        )
        replayed_summary = paper_launcher._summarize_nvidia_csv(
            artifact_paths["gpu_telemetry"]
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise HeadlineM0Error(f"{run_id} GPU telemetry replay failed: {exc}") from exc
    for key in ("schema", "sample_rows", "devices"):
        if replayed_summary.get(key) != gpu_summary.get(key):
            raise HeadlineM0Error(f"{run_id} raw GPU telemetry summary drifted")

    try:
        numerical = paper_launcher._training_numerical_status(
            artifact_paths["native_info_log"], artifact_paths["train_console_log"]
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise HeadlineM0Error(f"{run_id} numerical/AMP replay failed: {exc}") from exc
    if (
        postflight.get("numerical_status") != numerical
        or numerical.get("status") != "passed"
        or numerical.get("amp_enabled") is not True
        or int(numerical.get("finite_loss_observations", 0)) <= 0
        or numerical.get("loss_values_all_finite") is not True
        or int(numerical.get("amp_skip_observations", 0)) <= 0
        or numerical.get("max_amp_step_skipped") != 0.0
    ):
        raise HeadlineM0Error(f"{run_id} finite/zero-AMP-skip evidence drifted")
    logs = (
        artifact_paths["native_info_log"].read_text(
            encoding="utf-8", errors="replace"
        )
        + "\n"
        + artifact_paths["train_console_log"].read_text(
            encoding="utf-8", errors="replace"
        )
    )
    if (
        "stage_b_v22_branch_isolation_pass" not in logs
        or "continuing with fresh optimizer state" in logs
        or "loaded model weights only and will use fresh training state" in logs
    ):
        raise HeadlineM0Error(f"{run_id} branch isolation/resume log contract failed")

    scorer = _scorer_initializer_wrapper(runtime=runtime, artifacts=artifacts)
    if postflight.get("scorer_initializer_audit") != scorer:
        raise HeadlineM0Error(f"{run_id} scorer Stage-A initializer evidence drifted")
    milestones = _milestone_evidence(run_root=run_root, contract=contract, seed=seed)
    if postflight.get("milestones") != milestones:
        raise HeadlineM0Error(f"{run_id} learning-curve milestone evidence drifted")
    postflight_path = run_root / "postflight.json"
    postflight_record = _verify_completed_file_record(
        launch.get("postflight_artifact"),
        label=f"{run_id} launch postflight",
        expected_path=postflight_path,
    )
    if launch.get("postflight") != dict(postflight):
        raise HeadlineM0Error(f"{run_id} launch embeds another postflight")
    completed_phases = sequence.get("completed_phases")
    expected_completed = [
        {
            "phase_id": "joint",
            "status": "completed",
            "output_dir": str(run_root),
            "checkpoint": verified_artifacts["checkpoint"],
            "postflight": postflight_record,
        }
    ]
    if completed_phases != expected_completed:
        raise HeadlineM0Error(f"{run_id} completed phase artifact binding drifted")
    for name, path in artifact_paths.items():
        replayed = _verify_completed_file_record(
            artifacts.get(name),
            label=f"{run_id} {name} post-replay",
            expected_path=path,
        )
        if replayed != verified_artifacts[name]:
            raise HeadlineM0Error(f"{run_id} {name} changed during postflight replay")
    return {
        "final_checkpoint": final_checkpoint,
        "attempts": attempts,
        "numerical": numerical,
        "gpu_environment": gpu_environment,
        "gpu_summary": gpu_summary,
        "full_run_telemetry": attempts["full_run_telemetry"],
        "artifacts": verified_artifacts,
        "postflight_record": postflight_record,
    }


def _completed_training_queue_binding_projection(
    value: Any,
    *,
    contract: source_contracts.FormalPaperRunContract,
    seed: int,
    stable_input_closure_digest: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HeadlineM0Error("completed training queue binding is invalid")
    run_id = f"{contract.id}:{seed}"
    active = value.get("active_item")
    if (
        value.get("schema") != "pivot.stageb.headline_m0_queue_verification/v1"
        or value.get("status") != "passed"
        or value.get("contract_id") != contract.id
        or value.get("queue_status") != "running"
        or not isinstance(value.get("queue_id"), str)
        or not value.get("queue_id")
        or re.fullmatch(r"[0-9a-f]{64}", str(value.get("plan_sha256", ""))) is None
        or value.get("ordered_run_ids") != list(contract.dedicated_queue_run_ids)
        or re.fullmatch(
            r"[0-9a-f]{64}", str(value.get("queue_contract_sha256", ""))
        )
        is None
        or value.get("stable_input_closure_digest")
        != stable_input_closure_digest
        or not isinstance(active, Mapping)
        or active.get("run_id") != run_id
        or type(active.get("item_index")) is not int
        or active.get("item_status") not in {"reserved", "launching", "launched"}
        or not isinstance(active.get("orchestration_root"), str)
        or not active.get("orchestration_root")
        or not isinstance(active.get("gpu_key"), str)
        or not active.get("gpu_key")
        or not isinstance(active.get("lease_path"), str)
        or not active.get("lease_path")
    ):
        raise HeadlineM0Error(f"{run_id} completed training queue binding drifted")
    return {
        "contract_id": contract.id,
        "queue_id": value["queue_id"],
        "plan_sha256": value["plan_sha256"],
        "queue_contract_sha256": value["queue_contract_sha256"],
        "stable_input_closure_digest": stable_input_closure_digest,
        "ordered_run_ids": list(contract.dedicated_queue_run_ids),
        "active_item": {
            key: active.get(key)
            for key in (
                "item_index",
                "run_id",
                "item_status",
                "orchestration_root",
                "gpu_key",
                "lease_path",
            )
        },
    }


def _verify_completed_training_run_replay(
    run_root: Path,
    contract: source_contracts.FormalPaperRunContract,
    seed: int,
    *,
    verifier_sources: Mapping[str, Any],
) -> dict[str, Any]:
    run_root = run_root.expanduser().resolve(strict=True)
    run_id = f"{contract.id}:{seed}"
    sequence_path = run_root / "sequence_manifest.json"
    launch_path = run_root / "launch_manifest.json"
    postflight_path = run_root / "postflight.json"
    sequence, sequence_discovery = _read_completed_json_stably(
        sequence_path, label=f"{run_id} sequence"
    )
    launch, launch_discovery = _read_completed_json_stably(
        launch_path, label=f"{run_id} launch"
    )
    postflight, postflight_discovery = _read_completed_json_stably(
        postflight_path, label=f"{run_id} postflight"
    )
    attempt_count = postflight.get("training_attempt_count")
    if type(attempt_count) is not int or attempt_count <= 0:
        raise HeadlineM0Error(f"{run_id} completed attempt count is invalid")
    discovered_records = {
        Path(record["path"]).resolve(strict=False): record
        for record in (sequence_discovery, launch_discovery, postflight_discovery)
    }
    discovered_attempts: list[dict[str, Any]] = []
    for ordinal in range(attempt_count):
        _, _, attempt_path = _attempt_paths(run_root, ordinal)
        attempt, attempt_record = _read_completed_json_stably(
            attempt_path, label=f"{run_id} attempt {ordinal} discovery"
        )
        discovered_attempts.append(attempt)
        discovered_records[attempt_path.resolve(strict=True)] = attempt_record
    evidence_paths = _completed_run_evidence_paths(
        run_root, seed=seed, attempts=discovered_attempts
    )
    evidence_before = _completed_evidence_snapshot(evidence_paths)
    evidence_before_by_path = _completed_evidence_record_map(
        evidence_before, label=f"{run_id} initial evidence"
    )
    for path, record in discovered_records.items():
        if evidence_before_by_path.get(path) != record:
            raise HeadlineM0Error(
                f"{run_id} evidence changed while discovering the replay closure"
            )
    stable_inputs_before = _completed_stable_input_snapshot(
        _iter_input_records(launch), label=f"{run_id} initial stable inputs"
    )
    runtime, stable_closure = _verify_completed_launch_contract(
        sequence,
        launch,
        contract=contract,
        seed=seed,
        run_root=run_root,
    )
    verified = _verify_completed_postflight(
        sequence=sequence,
        launch=launch,
        postflight=postflight,
        runtime=runtime,
        contract=contract,
        seed=seed,
        run_root=run_root,
        stable_closure=stable_closure,
    )
    _verify_input_identities(launch, rehash=True)
    final_sequence, _ = _read_completed_json_stably(
        sequence_path, label=f"{run_id} sequence post-replay"
    )
    final_launch, _ = _read_completed_json_stably(
        launch_path, label=f"{run_id} launch post-replay"
    )
    final_postflight, _ = _read_completed_json_stably(
        postflight_path, label=f"{run_id} postflight post-replay"
    )
    if (
        final_sequence != sequence
        or final_launch != launch
        or final_postflight != postflight
    ):
        raise HeadlineM0Error(f"{run_id} manifests changed during deep replay")
    evidence_after = _completed_evidence_snapshot(evidence_paths)
    _require_same_completed_evidence(
        evidence_before, evidence_after, label=f"{run_id} completed training"
    )
    stable_inputs_after = _completed_stable_input_snapshot(
        _iter_input_records(launch), label=f"{run_id} final stable inputs"
    )
    _require_same_completed_evidence(
        stable_inputs_before,
        stable_inputs_after,
        label=f"{run_id} stable inputs",
    )
    final_evidence = _completed_evidence_record_map(
        evidence_after, label=f"{run_id} final evidence"
    )
    final_inputs = _completed_evidence_record_map(
        stable_inputs_after, label=f"{run_id} final stable inputs"
    )
    stage_a = _compact_file_record(
        final_inputs[runtime.stage_a_init.resolve(strict=True)]
    )
    ancestry = verified["attempts"]["ancestry"]
    gpu_summary = verified["gpu_summary"]
    devices = gpu_summary.get("devices")
    device_projection = [
        {
            key: device.get(key)
            for key in (
                "physical_index",
                "uuid",
                "name",
                "driver_version",
                "total_memory_mib",
            )
        }
        for device in devices
    ] if isinstance(devices, list) else []
    numerical = verified["numerical"]
    queue_binding = _completed_training_queue_binding_projection(
        sequence.get("training_queue_binding"),
        contract=contract,
        seed=seed,
        stable_input_closure_digest=stable_closure["digest"],
    )
    final_checkpoint = _compact_file_record(
        final_evidence[(run_root / "checkpoint_iter.pth").resolve(strict=True)]
    )
    final_attempt_records = [
        _compact_file_record(
            final_evidence[_attempt_paths(run_root, ordinal)[2].resolve(strict=True)]
        )
        for ordinal in range(attempt_count)
    ]
    projection = {
        "schema": COMPLETED_TRAINING_VERIFICATION_SCHEMA,
        "status": "passed",
        "run_id": run_id,
        "contract_id": contract.id,
        "seed": seed,
        "run_root": str(run_root),
        "contract": {
            "row": contract.expected_row(),
            "headline": bool(contract.headline),
            "matrix_validation_only": bool(contract.matrix_validation_only),
            "token_objective": (
                "edit_bce" if contract.id == "M0" else contract.token_objective
            ),
            "token_objective_scope": contract.token_objective_scope,
        },
        "final_checkpoint": final_checkpoint,
        "training_queue_binding": queue_binding,
        "budget": {
            **contract.expected_budget(),
            "gradient_accumulation_steps": 1,
            "amp": True,
            "final_epoch": FORMAL_FINAL_EPOCH,
            "final_iteration": FORMAL_FINAL_ITERATION,
            "optimizer_state_count": FORMAL_OPTIMIZER_STATE_COUNT,
        },
        "ancestry": {
            "status": "passed",
            "stage_a_initializer": stage_a,
            "stage_a_and_scorer_same_source": True,
            "b58_ancestry_count": ancestry["b58_ancestry_count"],
            "resume_chain_contiguous": ancestry["resume_chain_contiguous"],
            "attempt_count": verified["attempts"]["attempt_count"],
            "resume_count": verified["attempts"]["resume_count"],
            "attempt_manifests": final_attempt_records,
        },
        "numerical": {
            "status": "passed",
            "amp_enabled": numerical["amp_enabled"],
            "finite_loss_observations": numerical["finite_loss_observations"],
            "loss_values_all_finite": numerical["loss_values_all_finite"],
            "amp_skip_observations": numerical["amp_skip_observations"],
            "max_amp_step_skipped": numerical["max_amp_step_skipped"],
            "evidence_sha256": _sha256_bytes(_canonical_json_bytes(numerical)),
        },
        "telemetry": {
            "status": "passed",
            "sampling_interval_ms": gpu_summary["sampling_interval_ms"],
            "sample_rows": gpu_summary["sample_rows"],
            "devices": device_projection,
            "evidence_sha256": _sha256_bytes(_canonical_json_bytes(gpu_summary)),
            "full_run": verified["full_run_telemetry"],
        },
        "input_closure": {
            "status": "passed",
            "digest": stable_closure["digest"],
            "record_count": len(stable_closure["records"]),
            "verifier_source_digest": verifier_sources["digest"],
            "verifier_source_count": len(verifier_sources["records"]),
            "identity_snapshot": stable_inputs_after,
        },
        "artifacts": {
            "sequence_manifest": _compact_file_record(
                final_evidence[sequence_path.resolve(strict=True)]
            ),
            "launch_manifest": _compact_file_record(
                final_evidence[launch_path.resolve(strict=True)]
            ),
            "postflight": _compact_file_record(
                final_evidence[postflight_path.resolve(strict=True)]
            ),
        },
        "evidence_snapshot": evidence_after,
    }
    projection["semantic_sha256"] = _sha256_bytes(_canonical_json_bytes(projection))
    return projection


def verify_completed_training_run(
    run_root: Path,
    contract_id: str,
    seed: int | None = None,
) -> dict[str, Any]:
    """Deeply replay one completed formal M0 or M0N training run.

    The returned projection contains only deterministic, serializable evidence.
    Source identities are captured before and after replay so a concurrent edit
    cannot create a mixed-verifier completion result.
    """

    contract = _contract(contract_id)
    run_root = Path(run_root).expanduser().resolve(strict=True)
    if seed is None:
        match = re.fullmatch(r"seed([0-9]+)", run_root.name)
        if match is None:
            raise HeadlineM0Error("completed training seed is not encoded by run_root")
        seed = int(match.group(1))
    if seed not in contract.seeds:
        raise HeadlineM0Error(f"unexpected {contract.id} completed seed {seed}")
    sources_before = _completed_training_verifier_source_snapshot(contract)
    replay_error: Exception | None = None
    projection: dict[str, Any] | None = None
    try:
        projection = _verify_completed_training_run_replay(
            run_root,
            contract,
            seed,
            verifier_sources=sources_before,
        )
    except Exception as exc:
        replay_error = exc
    try:
        sources_after = _completed_training_verifier_source_snapshot(contract)
    except Exception as exc:
        raise HeadlineM0Error(
            "completed-training verifier sources disappeared during replay"
        ) from exc
    _require_same_verifier_sources(
        sources_before,
        sources_after,
        label=f"{contract.id}:{seed} completed-training verifier",
    )
    if replay_error is not None:
        if isinstance(replay_error, HeadlineM0Error):
            raise replay_error
        raise HeadlineM0Error(
            f"{contract.id}:{seed} completed-training replay failed: {replay_error}"
        ) from replay_error
    if projection is None:
        raise AssertionError("completed-training replay returned no projection")
    _verify_completed_evidence_current(
        projection.get("evidence_snapshot"),
        label=f"{contract.id}:{seed} completed training",
    )
    input_closure = projection.get("input_closure")
    input_snapshot = (
        input_closure.get("identity_snapshot")
        if isinstance(input_closure, Mapping)
        else None
    )
    _verify_completed_evidence_current(
        input_snapshot,
        label=f"{contract.id}:{seed} stable inputs",
    )
    return projection


def _deterministic_serial_completion_evidence(
    value: Any,
    *,
    queue: Mapping[str, Any],
    ordered_run_ids: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HeadlineM0Error("serial queue completion evidence is invalid")
    verified_items = value.get("verified_items")
    if (
        value.get("schema") != "pivot.stageb.serial_matrix_queue_verification/v1"
        or value.get("status") != "passed"
        or value.get("queue_status") != "completed"
        or value.get("queue_id") != queue["plan"]["queue_id"]
        or value.get("plan_sha256") != queue["plan_sha256"]
        or value.get("errors") not in (None, [])
        or not isinstance(verified_items, list)
        or [
            item.get("run_id") if isinstance(item, Mapping) else None
            for item in verified_items
        ]
        != list(ordered_run_ids)
    ):
        raise HeadlineM0Error("serial queue completion evidence drifted")

    def strip_timestamps(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                str(key): strip_timestamps(nested)
                for key, nested in item.items()
                if key != "verified_at_utc"
            }
        if isinstance(item, list):
            return [strip_timestamps(nested) for nested in item]
        return item

    normalized = strip_timestamps(value)
    if not isinstance(normalized, dict):
        raise AssertionError("serial completion normalization lost its mapping")
    return normalized


def _require_completed_binding_matches_queue(
    completed: Mapping[str, Any],
    *,
    queue: Mapping[str, Any],
    queue_contract_sha256: str,
    stable_input_closure_digest: str,
    item: Mapping[str, Any],
    expected_run_id: str,
) -> None:
    binding = completed.get("training_queue_binding")
    if not isinstance(binding, Mapping):
        raise HeadlineM0Error(f"{expected_run_id} has no replayed queue binding")
    plan = queue["plan"]
    expected = {
        "contract_id": expected_run_id.split(":", 1)[0],
        "queue_id": plan.get("queue_id"),
        "plan_sha256": queue.get("plan_sha256"),
        "queue_contract_sha256": queue_contract_sha256,
        "stable_input_closure_digest": stable_input_closure_digest,
        "ordered_run_ids": [
            planned.get("run_id") for planned in plan.get("items", [])
        ],
    }
    for key, expected_value in expected.items():
        if binding.get(key) != expected_value:
            raise HeadlineM0Error(
                f"{expected_run_id} embedded training queue {key} drifted"
            )
    active = binding.get("active_item")
    raw_root = item.get("orchestration_root")
    if not isinstance(raw_root, str) or not raw_root:
        raise HeadlineM0Error(
            f"{expected_run_id} completed queue item lacks its orchestration root"
        )
    expected_active = {
        "item_index": item.get("index"),
        "run_id": expected_run_id,
        "orchestration_root": str(Path(raw_root).resolve(strict=False)),
        "gpu_key": plan.get("gpu_key"),
        "lease_path": plan.get("lease_path"),
    }
    if not isinstance(active, Mapping) or active.get("item_status") not in {
        "reserved",
        "launching",
        "launched",
    }:
        raise HeadlineM0Error(
            f"{expected_run_id} embedded active queue item is invalid"
        )
    for key, expected_value in expected_active.items():
        if active.get(key) != expected_value:
            raise HeadlineM0Error(
                f"{expected_run_id} embedded active queue item {key} drifted"
            )


def _load_training_queue_stably(
    serial_queue: Any,
    queue_dir: Path,
    *,
    max_attempts: int = 5,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    queue_path = queue_dir / "queue.json"
    for attempt in range(max_attempts):
        before = (
            _stable_completed_file_record(queue_path) if queue_path.is_file() else None
        )
        try:
            queue = serial_queue.load_queue(queue_dir)
        except (OSError, ValueError, serial_queue.QueueContractError) as exc:
            raise HeadlineM0Error(f"cannot load formal training queue: {exc}") from exc
        after = (
            _stable_completed_file_record(queue_path) if queue_path.is_file() else None
        )
        if before == after:
            return queue, after
        if attempt + 1 < max_attempts:
            time.sleep(0.01)
    raise HeadlineM0Error("formal training queue did not stabilize while loading")


def _validate_active_training_queue_transition(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    expected_run_id: str,
    expected_orchestration_root: Path,
) -> None:
    before_plan = before.get("plan")
    after_plan = after.get("plan")
    if (
        not isinstance(before_plan, Mapping)
        or not isinstance(after_plan, Mapping)
        or dict(after_plan) != dict(before_plan)
        or after.get("plan_sha256") != before.get("plan_sha256")
        or after_plan.get("queue_id") != before_plan.get("queue_id")
        or before.get("status") != "running"
        or after.get("status") != "running"
    ):
        raise HeadlineM0Error("active formal training queue immutable plan drifted")
    before_items = before.get("items")
    after_items = after.get("items")
    if (
        not isinstance(before_items, list)
        or not isinstance(after_items, list)
        or len(before_items) != len(after_items)
    ):
        raise HeadlineM0Error("active formal training queue item order drifted")
    identity_keys = ("index", "run_id", "runner")
    for index, (before_item, after_item) in enumerate(zip(before_items, after_items)):
        if (
            not isinstance(before_item, Mapping)
            or not isinstance(after_item, Mapping)
            or any(
                before_item.get(key) != after_item.get(key)
                for key in identity_keys
            )
            or before_item.get("index") != index
        ):
            raise HeadlineM0Error("active formal training queue item order drifted")
    before_active = next(
        (
            item
            for item in before_items
            if isinstance(item, Mapping) and item.get("status") != "completed"
        ),
        None,
    )
    after_active = next(
        (
            item
            for item in after_items
            if isinstance(item, Mapping) and item.get("status") != "completed"
        ),
        None,
    )
    if (
        not isinstance(before_active, Mapping)
        or not isinstance(after_active, Mapping)
        or type(before_active.get("index")) is not int
        or before_active.get("index") != after_active.get("index")
        or before_active.get("run_id") != expected_run_id
        or after_active.get("run_id") != expected_run_id
    ):
        raise HeadlineM0Error("active formal training queue item changed")
    active_index = int(before_active["index"])
    for index, (before_item, after_item) in enumerate(zip(before_items, after_items)):
        if index != active_index and before_item.get("status") != after_item.get("status"):
            raise HeadlineM0Error("active formal training queue advanced another item")
    status_order = {"reserved": 0, "launching": 1, "launched": 2}
    before_status = before_active.get("status")
    after_status = after_active.get("status")
    if (
        not isinstance(before_status, str)
        or not isinstance(after_status, str)
        or before_status not in status_order
        or after_status not in status_order
        or status_order[after_status] < status_order[before_status]
    ):
        raise HeadlineM0Error("active formal training queue status transition is invalid")
    expected_root = expected_orchestration_root.expanduser().resolve(strict=False)
    for item in (before_active, after_active):
        raw_root = item.get("orchestration_root")
        if (
            not isinstance(raw_root, str)
            or not raw_root
            or Path(raw_root).expanduser().resolve(strict=False) != expected_root
        ):
            raise HeadlineM0Error("active queue item orchestration root drifted")


def _active_training_queue_binding(
    queue: Mapping[str, Any],
    *,
    expected_run_id: str,
    expected_orchestration_root: Path | None,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    plan = queue.get("plan")
    mutable_items = queue.get("items")
    if not isinstance(plan, Mapping) or not isinstance(mutable_items, list):
        raise HeadlineM0Error("formal training queue mutable items are missing")
    active = next(
        (
            item
            for item in mutable_items
            if isinstance(item, Mapping) and item.get("status") != "completed"
        ),
        None,
    )
    if (
        queue.get("status") != "running"
        or not isinstance(active, Mapping)
        or active.get("run_id") != expected_run_id
        or active.get("runner") != "paper"
        or active.get("status") not in {"reserved", "launching", "launched"}
    ):
        raise HeadlineM0Error(
            "formal run is not the queue's single active ordered item"
        )
    raw_root = active.get("orchestration_root")
    if not isinstance(raw_root, str) or not raw_root:
        raise HeadlineM0Error("active queue item has no orchestration root")
    active_root = Path(raw_root).expanduser().resolve(strict=False)
    if (
        expected_orchestration_root is None
        or active_root
        != expected_orchestration_root.expanduser().resolve(strict=False)
    ):
        raise HeadlineM0Error(
            "formal run orchestration root differs from the active queue item"
        )
    return active, {
        "item_index": active.get("index"),
        "run_id": expected_run_id,
        "item_status": active.get("status"),
        "orchestration_root": str(active_root),
        "gpu_key": plan.get("gpu_key"),
        "lease_path": plan.get("lease_path"),
    }


def verify_training_queue(
    queue_dir: Path,
    contract_id: str,
    *,
    expected_run_id: str | None = None,
    expected_orchestration_root: Path | None = None,
    expected_manifest: Mapping[str, Any] | None = None,
    require_completed: bool = False,
) -> dict[str, Any]:
    from tools import run_stageb_serial_matrix_queue as serial_queue

    contract = _contract(contract_id)
    queue_dir = Path(queue_dir).expanduser().resolve(strict=False)
    queue_path = queue_dir / "queue.json"
    queue, queue_record_before = _load_training_queue_stably(
        serial_queue, queue_dir
    )
    plan = queue.get("plan")
    if not isinstance(plan, Mapping):
        raise HeadlineM0Error("formal training queue has no immutable plan")
    planned_python = Path(str(plan.get("runner_python", ""))).expanduser().resolve(
        strict=True
    )
    if planned_python != DEFAULT_PYTHON.resolve(strict=True):
        raise HeadlineM0Error(
            "formal training queue runner Python is not the sealed GDINO runtime"
        )
    items = plan.get("items")
    observed_ids = [
        item.get("run_id") if isinstance(item, Mapping) else None
        for item in items or []
    ]
    if observed_ids != list(contract.dedicated_queue_run_ids):
        raise HeadlineM0Error(
            f"{contract.id} queue order must be "
            f"{list(contract.dedicated_queue_run_ids)}, got {observed_ids}"
        )
    if any(
        not isinstance(item, Mapping) or item.get("runner") != "paper"
        for item in items or []
    ):
        raise HeadlineM0Error("M0/M0N queue items must use the paper runner slot")
    runners = plan.get("runners")
    paper = runners.get("paper") if isinstance(runners, Mapping) else None
    runner_path = Path(__file__).resolve(strict=True)
    if (
        not isinstance(paper, Mapping)
        or Path(str(paper.get("path", ""))).resolve(strict=True) != runner_path
        or paper.get("sha256") != _sha256_file(runner_path)
    ):
        raise HeadlineM0Error("formal queue does not seal the current M0 runner")
    extensions = plan.get("extensions")
    queue_contract = _validate_training_queue_contract_payload(
        extensions.get(TRAINING_QUEUE_EXTENSION_KEY)
        if isinstance(extensions, Mapping)
        else None,
        contract,
    )
    queue_contract_sha256 = _sha256_bytes(_canonical_json_bytes(queue_contract))
    if expected_manifest is not None:
        _require_manifest_matches_queue_contract(expected_manifest, queue_contract)
    snapshot = plan.get("runtime_environment")
    if not isinstance(snapshot, Mapping):
        raise HeadlineM0Error("formal queue has no runtime environment snapshot")
    _validate_queue_runtime_snapshot(snapshot)
    active_binding = None
    if expected_run_id is not None:
        if expected_run_id not in contract.dedicated_queue_run_ids:
            raise HeadlineM0Error("active queue binding names an unknown run")
        active, active_binding = _active_training_queue_binding(
            queue,
            expected_run_id=expected_run_id,
            expected_orchestration_root=expected_orchestration_root,
        )
        try:
            serial_queue._ensure_lease(queue, active, create=False)
        except (OSError, serial_queue.QueueContractError) as exc:
            raise HeadlineM0Error(f"formal training queue lease is invalid: {exc}") from exc

    verification = None
    serial_completion_evidence = None
    completed_runs: list[dict[str, Any]] = []
    completion_semantic_sha256 = None
    completed_input_snapshot = None
    if require_completed and queue.get("status") != "completed":
        raise HeadlineM0Error("formal M0/M0N training queue is not completed")
    if queue.get("status") == "completed":
        if queue_record_before is None:
            raise HeadlineM0Error("completed formal training queue manifest is missing")
        verifier_sources_before = _completed_training_verifier_source_snapshot(
            contract
        )
        closure = queue_contract["stable_input_closure"]
        completed_inputs_before = _completed_stable_input_snapshot(
            closure["records"], label=f"{contract.id} queue initial stable inputs"
        )
        verification = serial_queue.verify_queue(queue_dir)
        serial_completion_evidence = _deterministic_serial_completion_evidence(
            verification,
            queue=queue,
            ordered_run_ids=contract.dedicated_queue_run_ids,
        )
        for run_id, item in zip(contract.dedicated_queue_run_ids, queue["items"]):
            output_root = Path(str(item.get("output_root", ""))).resolve(strict=True)
            sequence, _ = _read_completed_json_stably(
                output_root / "sequence_manifest.json",
                label=f"{run_id} completed sequence",
            )
            if (
                sequence.get("run_id") != run_id
                or sequence.get("status") != "completed"
                or sequence.get("stable_input_closure_digest")
                != closure["digest"]
            ):
                raise HeadlineM0Error(
                    f"{run_id} completed under a different queue source closure"
                )
            launch, _ = _read_completed_json_stably(
                output_root / "launch_manifest.json",
                label=f"{run_id} completed launch",
            )
            normalized = _stable_closure_from_manifest(launch)
            if (
                normalized.get("digest") != closure["digest"]
                or normalized.get("records") != closure["records"]
            ):
                raise HeadlineM0Error(
                    f"{run_id} launch source closure differs from its queue"
                )
            _, seed = _parse_run_id(run_id)
            completed = verify_completed_training_run(
                output_root, contract.id, seed
            )
            if completed["input_closure"]["digest"] != closure["digest"]:
                raise HeadlineM0Error(
                    f"{run_id} deep replay differs from its queue source closure"
                )
            _require_completed_binding_matches_queue(
                completed,
                queue=queue,
                queue_contract_sha256=queue_contract_sha256,
                stable_input_closure_digest=closure["digest"],
                item=item,
                expected_run_id=run_id,
            )
            completed_runs.append(completed)
        for completed in completed_runs:
            run_id = str(completed["run_id"])
            _verify_completed_evidence_current(
                completed["evidence_snapshot"],
                label=f"{run_id} queue-final completed training",
            )
            _verify_completed_evidence_current(
                completed["input_closure"]["identity_snapshot"],
                label=f"{run_id} queue-final stable inputs",
            )
        completed_inputs_after = _completed_stable_input_snapshot(
            closure["records"], label=f"{contract.id} queue final stable inputs"
        )
        _require_same_completed_evidence(
            completed_inputs_before,
            completed_inputs_after,
            label=f"{contract.id} queue stable inputs",
        )
        completed_input_snapshot = completed_inputs_after
        verifier_sources_after = _completed_training_verifier_source_snapshot(
            contract
        )
        _require_same_verifier_sources(
            verifier_sources_before,
            verifier_sources_after,
            label=f"{contract.id} completed training queue verifier",
        )
        queue_record_after = _stable_completed_file_record(queue_path)
        if queue_record_after != queue_record_before:
            raise HeadlineM0Error(
                "formal completed training queue changed during completion replay"
            )
        completion_semantic_sha256 = _sha256_bytes(
            _canonical_json_bytes(
                {
                    "contract_id": contract.id,
                    "queue_id": plan.get("queue_id"),
                    "plan_sha256": queue.get("plan_sha256"),
                    "queue_contract_sha256": queue_contract_sha256,
                    "queue_manifest": queue_record_after,
                    "ordered_run_ids": list(contract.dedicated_queue_run_ids),
                    "run_semantic_sha256s": [
                        value["semantic_sha256"] for value in completed_runs
                    ],
                    "stable_input_snapshot": completed_input_snapshot,
                    "serial_completion_evidence": serial_completion_evidence,
                    "verifier_source_digest": verifier_sources_before["digest"],
                }
            )
        )
    queue_record_after = queue_record_before
    if active_binding is not None:
        if expected_run_id is None or expected_orchestration_root is None:
            raise AssertionError("active queue replay lost its expected binding")
        latest_queue, queue_record_after = _load_training_queue_stably(
            serial_queue, queue_dir
        )
        _validate_active_training_queue_transition(
            queue,
            latest_queue,
            expected_run_id=expected_run_id,
            expected_orchestration_root=expected_orchestration_root,
        )
        latest_active, active_binding = _active_training_queue_binding(
            latest_queue,
            expected_run_id=expected_run_id,
            expected_orchestration_root=expected_orchestration_root,
        )
        try:
            serial_queue._ensure_lease(latest_queue, latest_active, create=False)
        except (OSError, serial_queue.QueueContractError) as exc:
            raise HeadlineM0Error(f"formal training queue lease is invalid: {exc}") from exc
        queue = latest_queue
        plan = queue["plan"]
    elif queue_record_before is not None:
        queue_record_after = _stable_completed_file_record(queue_path)
        if queue_record_after != queue_record_before:
            raise HeadlineM0Error("formal training queue changed during verification")
    if completed_input_snapshot is not None:
        for completed in completed_runs:
            _verify_completed_evidence_current(
                completed["evidence_snapshot"],
                label=f"{completed['run_id']} queue return-bound evidence",
            )
            _verify_completed_evidence_current(
                completed["input_closure"]["identity_snapshot"],
                label=f"{completed['run_id']} queue return-bound stable inputs",
            )
        _verify_completed_evidence_current(
            completed_input_snapshot,
            label=f"{contract.id} queue return-bound stable inputs",
        )
    if active_binding is not None:
        if expected_run_id is None or expected_orchestration_root is None:
            raise AssertionError("active queue return replay lost its expected binding")
        latest_queue, latest_record = _load_training_queue_stably(
            serial_queue, queue_dir
        )
        _validate_active_training_queue_transition(
            queue,
            latest_queue,
            expected_run_id=expected_run_id,
            expected_orchestration_root=expected_orchestration_root,
        )
        latest_active, active_binding = _active_training_queue_binding(
            latest_queue,
            expected_run_id=expected_run_id,
            expected_orchestration_root=expected_orchestration_root,
        )
        try:
            serial_queue._ensure_lease(latest_queue, latest_active, create=False)
        except (OSError, serial_queue.QueueContractError) as exc:
            raise HeadlineM0Error(f"formal training queue lease is invalid: {exc}") from exc
        queue = latest_queue
        plan = queue["plan"]
        queue_record_after = latest_record
    elif queue_record_before is not None:
        queue_record_after = _stable_completed_file_record(queue_path)
        if queue_record_after != queue_record_before:
            raise HeadlineM0Error("formal training queue changed before verification return")
    return {
        "schema": "pivot.stageb.headline_m0_queue_verification/v1",
        "status": "passed",
        "contract_id": contract.id,
        "queue_status": queue.get("status"),
        "queue_id": plan.get("queue_id"),
        "plan_sha256": queue.get("plan_sha256"),
        "ordered_run_ids": observed_ids,
        "queue_contract_sha256": queue_contract_sha256,
        "queue_manifest": queue_record_after,
        "stable_input_closure_digest": queue_contract[
            "stable_input_closure"
        ]["digest"],
        "active_item": active_binding,
        "completion_verification": verification,
        "serial_completion_evidence": serial_completion_evidence,
        "completed_stable_input_snapshot": completed_input_snapshot,
        "completed_training_runs": completed_runs,
        "completion_semantic_sha256": completion_semantic_sha256,
    }


def _list_runs(as_json: bool) -> int:
    payload = {
        "rows": [contract.expected_row() for contract in CONTRACTS.values()],
        "seeds": [17, 42, 73],
        "run_ids": list(RUN_IDS),
        "queue_contracts": {
            contract_id: list(contract.dedicated_queue_run_ids)
            for contract_id, contract in CONTRACTS.items()
        },
        "separate_exact_queues_required": True,
    }
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for contract in CONTRACTS.values():
            print(
                f"{contract.id}: config={contract.config}, "
                f"queue={','.join(contract.dedicated_queue_run_ids)}"
            )
    return 0


def _dry_run(
    contract: source_contracts.FormalPaperRunContract,
    seed: int,
    manifest_path: Path | None,
) -> int:
    runtime = runtime_from_environment()
    manifest = build_manifest(
        runtime, contract, seed, token_launcher.HashCache()
    )
    print(
        f"[{manifest['run_id']}/joint] "
        f"{manifest['phases'][0]['command_shell']}"
    )
    if manifest_path is not None:
        _write_json_no_replace(manifest_path, manifest)
    return 0


def _serial_queue_dir_from_orchestration_root(root: Path) -> Path:
    root = root.expanduser().resolve(strict=True)
    if root.parent.name != "jobs":
        raise HeadlineM0Error(
            "formal detach must be invoked by run_stageb_serial_matrix_queue"
        )
    queue_dir = root.parent.parent
    if not (queue_dir / "queue.json").is_file():
        raise HeadlineM0Error("formal detach cannot locate its serial queue manifest")
    return queue_dir.resolve(strict=True)


def _detach(
    contract: source_contracts.FormalPaperRunContract,
    seed: int,
    orchestration_root: Path | None,
) -> int:
    runtime = runtime_from_environment()
    run_root = output_directory(runtime, contract, seed)
    if run_root.exists():
        raise FileExistsError(f"formal run root must be fresh: {run_root}")
    manifest = build_manifest(runtime, contract, seed, token_launcher.HashCache())
    if manifest.get("output_dir_fresh_at_plan") is not True:
        raise HeadlineM0Error("detached preflight did not prove a fresh run root")
    root = (
        orchestration_root
        if orchestration_root is not None
        else Path(
            os.environ.get("PIVOT_ORCHESTRATION_ROOT", str(DEFAULT_ORCHESTRATION_ROOT))
        )
    ).expanduser().resolve(strict=False)
    queue_dir = _serial_queue_dir_from_orchestration_root(root)
    queue_binding = verify_training_queue(
        queue_dir,
        contract.id,
        expected_run_id=f"{contract.id}:{seed}",
        expected_orchestration_root=root,
        expected_manifest=manifest,
    )
    job_name = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + f"-pid{os.getpid()}"
    )
    job_dir = root / job_name
    job_dir.mkdir(parents=True, exist_ok=False)
    plans_dir = job_dir / "plans" / contract.id
    _write_json_no_replace(plans_dir / f"seed{seed}.json", manifest)
    status_path = job_dir / "status.json"
    launch_path = job_dir / "launch.json"
    log_path = job_dir / "orchestrator.log"
    child_command = [
        sys.executable,
        str(Path(__file__).resolve(strict=True)),
        "run",
        "--run-id",
        f"{contract.id}:{seed}",
    ]
    launch: dict[str, Any] = {
        "schema": DETACHED_SCHEMA,
        "status": "prepared",
        "created_at_utc": _utc_now(),
        "job_dir": str(job_dir),
        "run_ids": [f"{contract.id}:{seed}"],
        "expected_run_roots": [str(run_root)],
        "command": child_command,
        "command_shell": shlex.join(child_command),
        "orchestrator_log": str(log_path),
        "orchestrator_status": str(status_path),
        "plans_dir": str(job_dir / "plans"),
        "one_attempt_execution": True,
        "resume_authorization": "external_explicit_request_only",
        "training_queue_binding": queue_binding,
        "runtime": {
            **_runtime_payload(runtime),
            "output_root": str(runtime.output_root),
        },
    }
    _write_json_atomic(launch_path, launch)
    paper_launcher._update_orchestration_status(
        status_path,
        status="prepared",
        job_dir=str(job_dir),
        run_ids=launch["run_ids"],
        expected_run_roots=launch["expected_run_roots"],
    )
    environment = dict(os.environ)
    environment["PIVOT_ORCHESTRATION_STATUS"] = str(status_path)
    environment["PIVOT_HEADLINE_M0_QUEUE_DIR"] = str(queue_dir)
    environment["PYTHONUNBUFFERED"] = "1"
    try:
        with log_path.open("ab", buffering=0) as handle:
            process = subprocess.Popen(
                child_command,
                cwd=REPO_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    except BaseException as exc:
        launch["status"] = "spawn_failed"
        launch["spawn_error"] = f"{type(exc).__name__}: {exc}"
        launch["finished_at_utc"] = _utc_now()
        _write_json_atomic(launch_path, launch)
        paper_launcher._update_orchestration_status(
            status_path, status="spawn_failed", error=launch["spawn_error"]
        )
        raise
    launch["status"] = "launched"
    launch["launched_at_utc"] = _utc_now()
    launch["child_pid"] = int(process.pid)
    launch["child_process_identity"] = paper_launcher._read_process_identity(
        int(process.pid)
    )
    launch["child_start_new_session"] = True
    launch["stdin"] = "DEVNULL"
    launch["stdout_stderr"] = str(log_path)
    _write_json_atomic(launch_path, launch)
    print(
        json.dumps(
            {
                "status": "launched",
                "pid": int(process.pid),
                "job_dir": str(job_dir),
                "status_file": str(status_path),
                "log_file": str(log_path),
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    listing = subparsers.add_parser("list", help="list the six registered runs")
    listing.add_argument("--json", action="store_true")
    dry = subparsers.add_parser("dry-run", help="seal one plan without training")
    dry.add_argument("--run-id", required=True)
    dry.add_argument("--manifest", type=Path)
    run = subparsers.add_parser("run", help="execute exactly one formal run")
    run.add_argument("--run-id", required=True)
    detach = subparsers.add_parser(
        "detach", help="preflight and detach exactly one formal run"
    )
    detach.add_argument("--run-id", required=True)
    detach.add_argument("--orchestration-root", type=Path)
    resume = subparsers.add_parser(
        "resume", help="authorize one attempt from a live mid-epoch signal pause"
    )
    resume.add_argument("target", type=Path)
    status = subparsers.add_parser("status", help="inspect a detached job")
    status.add_argument("job_dir", type=Path)
    reconcile = subparsers.add_parser(
        "reconcile", help="reconcile a detached job from PID/artifact evidence"
    )
    reconcile.add_argument("job_dir", type=Path)
    spec = subparsers.add_parser("queue-spec", help="print one exact queue contract")
    spec.add_argument("contract_id", choices=tuple(CONTRACTS))
    create_queue = subparsers.add_parser(
        "create-queue",
        help="create one source-sealed exact three-seed serial training queue",
    )
    create_queue.add_argument("queue_dir", type=Path)
    create_queue.add_argument("contract_id", choices=tuple(CONTRACTS))
    create_queue.add_argument("--lease-root", type=Path)
    create_queue.add_argument("--gpu-key")
    verify = subparsers.add_parser(
        "verify-queue", help="verify exact order/runtime/runner for one queue"
    )
    verify.add_argument("queue_dir", type=Path)
    verify.add_argument("contract_id", choices=tuple(CONTRACTS))
    verify.add_argument("--require-completed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.mode == "list":
            return _list_runs(args.json)
        if args.mode in {"dry-run", "run", "detach"}:
            contract, seed = _parse_run_id(args.run_id)
            if args.mode == "dry-run":
                return _dry_run(contract, seed, args.manifest)
            if args.mode == "run":
                return _run(contract, seed)
            return _detach(contract, seed, args.orchestration_root)
        if args.mode == "resume":
            print(json.dumps(authorize_resume(args.target), indent=2, sort_keys=True))
            return 0
        if args.mode == "status":
            print(
                json.dumps(
                    paper_launcher._inspect_or_reconcile_detached_job(
                        args.job_dir, mutate=False
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.mode == "reconcile":
            print(
                json.dumps(
                    paper_launcher._inspect_or_reconcile_detached_job(
                        args.job_dir, mutate=True
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.mode == "queue-spec":
            print(json.dumps(queue_spec(args.contract_id), indent=2, sort_keys=True))
            return 0
        if args.mode == "create-queue":
            print(
                json.dumps(
                    create_training_queue(
                        args.queue_dir,
                        args.contract_id,
                        lease_root=args.lease_root,
                        gpu_key=args.gpu_key,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.mode == "verify-queue":
            print(
                json.dumps(
                    verify_training_queue(
                        args.queue_dir,
                        args.contract_id,
                        require_completed=args.require_completed,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        parser.error(f"unknown mode: {args.mode}")
    except (
        FileExistsError,
        FileNotFoundError,
        HeadlineM0Error,
        NotADirectoryError,
        OSError,
        PermissionError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
