#!/usr/bin/env python3
"""Launch the sealed CVPR Table-C L0-L10 token ablation matrix.

The launcher is deliberately fail-closed for training: ``run`` requires an
explicit ``--run-id`` (for example ``L4:17``) or ``--all``, and every selected
output directory must not exist. ``list`` and ``dry-run`` never start
``main.py``. A launch manifest is written before training so an interrupted run
still retains its exact command and input hashes.

Runtime environment overrides:

* ``PIVOT_PYTHON`` (default: gdino5090 Python)
* ``PIVOT_STAGE_A_INIT`` (default: recovered Stage-A checkpoint0004)
* ``PIVOT_SCORER_WARMSTART`` (default: b58 scorer source)
* ``PIVOT_BATCH_SIZE`` (default: 16)
* ``PIVOT_MAX_TRAIN_ITERS`` (default: 1000)
* ``PIVOT_ITER_CHECKPOINT_INTERVAL`` (default: max-train-iters)
* ``PIVOT_NUM_WORKERS``, ``PIVOT_PREFETCH_FACTOR``, ``PIVOT_OMP_NUM_THREADS``
* ``PIVOT_CUDA_VISIBLE_DEVICES``, ``PIVOT_DATA_ROOT``
* ``PIVOT_TOKEN_OUTPUT_ROOT``
* ``PIVOT_ORCHESTRATION_ROOT`` (``detach`` control artifacts)

``PIVOT_TOKEN_DATASETS`` exists for launcher testing and creation of a new
explicit protocol block. Paper Table-C runs use the default sealed manifest.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATASET = REPO_ROOT / "config/datasets_stageb_v21_single_edit_train.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/paper_cvpr_v1/token_ablation"
DEFAULT_ORCHESTRATION_ROOT = (
    REPO_ROOT / "outputs/paper_cvpr_v1/orchestration/token_ablation"
)
DEFAULT_STAGE_A_INIT = Path("/media/haoyi/T9/gdino/checkpoint0004.pth")
DEFAULT_SCORER_WARMSTART = Path(
    "/media/haoyi/T9/gdino/outputs/"
    "gdino_ft_stageb_from_gdino_ft_e1_with_tn_bs19_nopatchbranch/"
    "checkpoint0001.pth"
)
DEFAULT_PYTHON = Path("/home/haoyi/miniconda/envs/gdino5090/bin/python")
DEFAULT_STAGE_A_SHA256 = (
    "7f4cdd0ab94fc74d46fc7658b2014588a06d7de44be2c1d482ed073bbd7ca1b1"
)
DEFAULT_SCORER_SHA256 = (
    "b58e5209dc07dbffb2e5ed3d792e0db70c3306ec2ed24389693b4aeeebab1157"
)
SEEDS = (17, 42, 73)


@dataclass(frozen=True)
class MatrixRow:
    row_id: str
    config: str
    token_objective: str
    predicate_pair_rank_weight: float
    positive_weight: float = 1.0
    shared_weight: float = 0.25
    edit_weight: float = 1.0


ROWS: tuple[MatrixRow, ...] = (
    MatrixRow(
        "L0",
        "config/ablations/cfg_stageb_v21_token_l0_off.py",
        "off",
        0.0,
    ),
    MatrixRow(
        "L1",
        "config/ablations/cfg_stageb_v21_token_l1_allquery_allneg_focal.py",
        "allquery_allneg_focal",
        0.0,
    ),
    MatrixRow(
        "L2",
        "config/ablations/cfg_stageb_v21_token_l2_targetlocal_allneg_focal.py",
        "targetlocal_allneg_focal",
        0.0,
    ),
    MatrixRow(
        "L3",
        "config/ablations/cfg_stageb_v21_token_l3_edit_bce.py",
        "edit_bce",
        0.0,
    ),
    MatrixRow(
        "L4",
        "config/ablations/cfg_stageb_v21_token_l4_edit_bce_pair_rank.py",
        "edit_bce",
        1.0,
    ),
    MatrixRow(
        "L5",
        "config/ablations/cfg_stageb_v21_token_l5_edit_bce_uniform_roles.py",
        "edit_bce",
        1.0,
        shared_weight=1.0,
    ),
    MatrixRow(
        "L6",
        "config/ablations/cfg_stageb_v21_token_l6_edit_bce_no_shared.py",
        "edit_bce",
        1.0,
        shared_weight=0.0,
    ),
    MatrixRow(
        "L7",
        "config/ablations/cfg_stageb_v21_token_l7_targetlocal_allneg_bce.py",
        "targetlocal_allneg_bce",
        0.0,
    ),
    MatrixRow(
        "L8",
        "config/ablations/cfg_stageb_v21_token_l8_edit_focal.py",
        "edit_focal",
        0.0,
    ),
    MatrixRow(
        "L9",
        "config/ablations/cfg_stageb_v21_token_l9_gdino_loss_form.py",
        "gdino_allquery_allneg_focal",
        0.0,
    ),
    MatrixRow(
        "L10",
        "config/ablations/cfg_stageb_v21_token_l10_edit_bce_group_balanced.py",
        "edit_bce_group_balanced",
        1.0,
    ),
)
ROW_BY_ID = {row.row_id: row for row in ROWS}


@dataclass(frozen=True)
class Runtime:
    python: Path
    stage_a_init: Path
    scorer_warmstart: Path
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


class HashCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, int], str] = {}

    def digest(self, path: Path) -> str:
        path = path.resolve(strict=True)
        stat = path.stat()
        key = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        stat_after = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (
            stat_after.st_size,
            stat_after.st_mtime_ns,
        ):
            raise RuntimeError(f"input changed while hashing: {path}")
        self._cache[key] = value
        return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _paper_runtime_evidence() -> Any:
    """Load the shared paper-run evidence implementation after this module.

    The paper matrix launcher imports this token launcher for its hash and
    subprocess helpers, so this dependency must stay lazy to avoid an import
    cycle. Table C deliberately shares its GPU and detached-job semantics.
    """

    from tools import run_stageb_paper_ablation_matrices as paper_launcher

    return paper_launcher


def _env_int(name: str, default: int, *, minimum: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _resolve_executable(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        resolved = candidate.resolve(strict=True)
    else:
        found = shutil.which(value)
        if found is None:
            raise FileNotFoundError(f"runtime executable not found: {value}")
        resolved = Path(found).resolve(strict=True)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise PermissionError(f"runtime executable is not executable: {resolved}")
    return resolved


def runtime_from_environment() -> Runtime:
    max_train_iters = _env_int("PIVOT_MAX_TRAIN_ITERS", 1000, minimum=1)
    interval_raw = os.environ.get("PIVOT_ITER_CHECKPOINT_INTERVAL")
    interval = (
        max_train_iters
        if interval_raw is None
        else _env_int("PIVOT_ITER_CHECKPOINT_INTERVAL", max_train_iters, minimum=1)
    )
    if interval > max_train_iters:
        raise ValueError(
            "PIVOT_ITER_CHECKPOINT_INTERVAL cannot exceed "
            "PIVOT_MAX_TRAIN_ITERS"
        )
    sharing = os.environ.get("PIVOT_MP_SHARING_STRATEGY", "file_system")
    if sharing not in {"file_system", "file_descriptor", "none"}:
        raise ValueError(
            "PIVOT_MP_SHARING_STRATEGY must be file_system, file_descriptor, or none"
        )
    runtime = Runtime(
        python=_resolve_executable(
            os.environ.get("PIVOT_PYTHON", str(DEFAULT_PYTHON))
        ),
        stage_a_init=Path(
            os.environ.get("PIVOT_STAGE_A_INIT", str(DEFAULT_STAGE_A_INIT))
        ).expanduser().resolve(strict=True),
        scorer_warmstart=Path(
            os.environ.get(
                "PIVOT_SCORER_WARMSTART", str(DEFAULT_SCORER_WARMSTART)
            )
        ).expanduser().resolve(strict=True),
        dataset=Path(
            os.environ.get("PIVOT_TOKEN_DATASETS", str(DEFAULT_DATASET))
        ).expanduser().resolve(strict=True),
        output_root=Path(
            os.environ.get("PIVOT_TOKEN_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT))
        ).expanduser().resolve(strict=False),
        data_root=Path(
            os.environ.get(
                "PIVOT_DATA_ROOT",
                os.environ.get("DATA_ROOT", "/media/haoyi/T9/data"),
            )
        ).expanduser().resolve(strict=False),
        batch_size=_env_int("PIVOT_BATCH_SIZE", 16, minimum=1),
        max_train_iters=max_train_iters,
        iter_checkpoint_interval=interval,
        num_workers=_env_int("PIVOT_NUM_WORKERS", 8, minimum=0),
        prefetch_factor=_env_int("PIVOT_PREFETCH_FACTOR", 1, minimum=1),
        omp_num_threads=_env_int("PIVOT_OMP_NUM_THREADS", 8, minimum=1),
        min_nofile=_env_int("PIVOT_MIN_NOFILE", 65536, minimum=0),
        cuda_visible_devices=os.environ.get(
            "PIVOT_CUDA_VISIBLE_DEVICES",
            os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        ),
        mp_sharing_strategy=sharing,
    )
    for label, path in (
        ("Stage-A initializer", runtime.stage_a_init),
        ("scorer warm-start", runtime.scorer_warmstart),
        ("dataset manifest", runtime.dataset),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is not a file: {path}")
    return runtime


def _file_record(path: Path, cache: HashCache) -> dict[str, Any]:
    path = path.resolve(strict=True)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": cache.digest(path),
    }


def _config_dependencies(config_path: Path) -> list[Path]:
    pending = [config_path.resolve(strict=True)]
    visited: set[Path] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        tree = ast.parse(current.read_text(encoding="utf-8"), filename=str(current))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            if not node.module.startswith("config."):
                continue
            imported = REPO_ROOT / (node.module.replace(".", "/") + ".py")
            if imported.is_file():
                pending.append(imported.resolve(strict=True))
    return sorted(visited, key=lambda path: str(path))


def _expand_dataset_path(raw: str, *, runtime: Runtime) -> Path:
    expanded = raw.replace("${DATA_ROOT}", str(runtime.data_root))
    expanded = os.path.expandvars(expanded)
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = runtime.dataset.parent / path
    if path.exists():
        return path.resolve(strict=True)
    legacy_repo = Path("/home/user/PIVOT")
    try:
        relative = path.relative_to(legacy_repo)
    except ValueError:
        return path.resolve(strict=False)
    return (REPO_ROOT / relative).resolve(strict=False)


def validate_dataset_contract(runtime: Runtime) -> tuple[dict[str, Any], list[Path]]:
    payload = json.loads(runtime.dataset.read_text(encoding="utf-8"))
    train = payload.get("train")
    if not isinstance(train, list) or len(train) != 4:
        raise RuntimeError("Table-C dataset manifest must contain exactly four train sources")
    weights = [float(source.get("mix_weight", 1.0)) for source in train]
    if weights != [1.0, 1.0, 1.0, 3.0]:
        raise RuntimeError(
            "Table-C dataset mix weights must be [1,1,1,3] so TN exposure is 50%; "
            f"got {weights}"
        )
    tn = train[-1]
    required_tn = {
        "source": "sam3_tn_pair",
        "require_global_tn_verified": False,
        "require_single_edit_token_provenance": True,
        "paper_table_b_id": "D3",
        "paper_tn_scope": "proposal_covered_verified",
    }
    for key, expected in required_tn.items():
        if tn.get(key) != expected:
            raise RuntimeError(
                f"Table-C TN source requires {key}={expected!r}, got {tn.get(key)!r}"
            )
    if payload.get("val") != []:
        raise RuntimeError("Table-C training manifest must keep val=[] and skip_eval=True")

    source_files: set[Path] = set()
    source_paths: list[dict[str, Any]] = []
    for index, source in enumerate(train):
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
            source_paths.append(
                {
                    "dataset_index": index,
                    "field": key,
                    "declared": raw,
                    "resolved": str(resolved),
                }
            )
            if not resolved.is_file():
                raise FileNotFoundError(
                    f"dataset source {index} field {key} is not a file: {resolved}"
                )
            source_files.add(resolved)
    contract = {
        "train_source_count": 4,
        "positive_source_count": 3,
        "tn_source_count": 1,
        "mix_weights": weights,
        "expected_tn_exposure_fraction": weights[-1] / sum(weights),
        "tn_source": required_tn,
        "source_paths": source_paths,
    }
    return contract, sorted(source_files, key=lambda path: str(path))


def _relevant_repository_sources() -> list[Path]:
    relative_paths = (
        "main.py",
        "engine.py",
        "datasets/patch_episode.py",
        "models/GroundingDINO/groundingdino.py",
        "models/GroundingDINO/stage_b_fixed_text_scorer.py",
        "models/GroundingDINO/stage_b_fixed_text_criterion.py",
        "docs/paper_cvpr_ablation_protocol.md",
        "tools/run_stageb_paper_ablation_matrices.py",
        "tools/run_stageb_token_ablation_matrix.py",
    )
    return [(REPO_ROOT / value).resolve(strict=True) for value in relative_paths]


def output_directory(runtime: Runtime, row: MatrixRow, seed: int) -> Path:
    return runtime.output_root / row.row_id / f"seed{seed}"


def build_command(
    runtime: Runtime,
    row: MatrixRow,
    seed: int,
    output_dir: Path,
) -> list[str]:
    config_path = (REPO_ROOT / row.config).resolve(strict=True)
    return [
        str(runtime.python),
        str((REPO_ROOT / "main.py").resolve(strict=True)),
        "-c",
        str(config_path),
        "--datasets",
        str(runtime.dataset),
        "--output_dir",
        str(output_dir),
        "--pretrain_model_path",
        str(runtime.stage_a_init),
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
        "--note",
        f"paper_cvpr_v1_token_{row.row_id}_seed{seed}",
        "--amp",
        "--save_log",
        "--options",
        f"batch_size={runtime.batch_size}",
        f"stage_b_v15_scorer_init_checkpoint={runtime.scorer_warmstart}",
    ]


def build_manifest(
    runtime: Runtime,
    row: MatrixRow,
    seed: int,
    cache: HashCache,
) -> dict[str, Any]:
    config_path = (REPO_ROOT / row.config).resolve(strict=True)
    output_dir = output_directory(runtime, row, seed).resolve(strict=False)
    command = build_command(runtime, row, seed, output_dir)
    dataset_contract, dataset_sources = validate_dataset_contract(runtime)

    stage_a = _file_record(runtime.stage_a_init, cache)
    scorer = _file_record(runtime.scorer_warmstart, cache)
    stage_a_is_default = runtime.stage_a_init == DEFAULT_STAGE_A_INIT.resolve(strict=False)
    scorer_is_default = (
        runtime.scorer_warmstart == DEFAULT_SCORER_WARMSTART.resolve(strict=False)
    )
    if stage_a_is_default and stage_a["sha256"] != DEFAULT_STAGE_A_SHA256:
        raise RuntimeError("default Stage-A checkpoint SHA-256 does not match protocol")
    if scorer_is_default and scorer["sha256"] != DEFAULT_SCORER_SHA256:
        raise RuntimeError("default scorer warm-start SHA-256 does not match protocol")
    stage_a.update(
        {
            "is_protocol_default": stage_a_is_default,
            "expected_sha256": DEFAULT_STAGE_A_SHA256 if stage_a_is_default else None,
        }
    )
    scorer.update(
        {
            "is_protocol_default": scorer_is_default,
            "expected_sha256": DEFAULT_SCORER_SHA256 if scorer_is_default else None,
            "load_scope": "scorer_only_last_three_decoder_layers_norm_refpoint",
        }
    )

    return {
        "schema": "pivot.stageb.token_ablation_launch/v2",
        "status": "planned",
        "created_at_utc": _utc_now(),
        "repository_root": str(REPO_ROOT),
        "run_id": f"{row.row_id}:{seed}",
        "row": asdict(row),
        "seed": seed,
        "training_seeds_contract": list(SEEDS),
        "output_dir": str(output_dir),
        "output_dir_fresh_at_plan": not output_dir.exists(),
        "command": command,
        "command_shell": shlex.join(command),
        "runtime": {
            "python": str(runtime.python),
            "batch_size": runtime.batch_size,
            "max_train_iters": runtime.max_train_iters,
            "iter_checkpoint_interval": runtime.iter_checkpoint_interval,
            "num_workers": runtime.num_workers,
            "prefetch_factor": runtime.prefetch_factor,
            "omp_num_threads": runtime.omp_num_threads,
            "min_nofile": runtime.min_nofile,
            "cuda_visible_devices": runtime.cuda_visible_devices,
            "mp_sharing_strategy": runtime.mp_sharing_strategy,
            "amp": True,
        },
        "runtime_evidence_contract": {
            "gpu_identity_captured_before_training": True,
            "gpu_telemetry_interval_ms": 1000,
            "gpu_identity_bound_in_postflight": True,
            "finite_loss_required": True,
            "positive_finite_amp_scale_required": True,
            "zero_amp_skipped_steps_required": True,
            "all_inputs_rehashed_after_training": True,
            "sequence_manifest_required": True,
            "shared_implementation": (
                "tools/run_stageb_paper_ablation_matrices.py"
            ),
        },
        "fixed_contract": {
            "architecture": "v19_base_plus_gate_with_acc50_hardneg_fixed_stage_a_top50",
            "candidate_topk": 50,
            "positive_iou_threshold": 0.5,
            "negative_iou_threshold": 0.499,
            "dataset_manifest": str(runtime.dataset),
            "dataset_is_protocol_default": runtime.dataset
            == DEFAULT_DATASET.resolve(strict=False),
            "dataset": dataset_contract,
        },
        "inputs": {
            "stage_a_initializer": stage_a,
            "scorer_warmstart": scorer,
            "dataset_manifest": _file_record(runtime.dataset, cache),
            "config_dependencies": [
                _file_record(path, cache)
                for path in _config_dependencies(config_path)
            ],
            "dataset_source_files": [
                _file_record(path, cache) for path in dataset_sources
            ],
            "repository_sources": [
                _file_record(path, cache)
                for path in _relevant_repository_sources()
            ],
        },
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _iter_file_records(manifest: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    inputs = manifest["inputs"]
    yield inputs["stage_a_initializer"]
    yield inputs["scorer_warmstart"]
    yield inputs["dataset_manifest"]
    yield from inputs["config_dependencies"]
    yield from inputs["dataset_source_files"]
    yield from inputs["repository_sources"]


def _verify_file_identities(manifest: Mapping[str, Any]) -> None:
    for record in _iter_file_records(manifest):
        path = Path(str(record["path"]))
        stat = path.stat()
        if (
            int(stat.st_size) != int(record["size_bytes"])
            or int(stat.st_mtime_ns) != int(record["mtime_ns"])
        ):
            raise RuntimeError(f"input changed after manifest hashing: {path}")


def _paper_input_view(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt the Table-C manifest to the shared flat input-record contract."""

    inputs = manifest["inputs"]
    records: list[dict[str, Any]] = []
    for role, value in (
        ("stage_a_initializer", inputs["stage_a_initializer"]),
        ("scorer_warmstart", inputs["scorer_warmstart"]),
        ("dataset_manifest", inputs["dataset_manifest"]),
    ):
        records.append({**dict(value), "role": role})
    for role, values in (
        ("config_dependency", inputs["config_dependencies"]),
        ("dataset_source", inputs["dataset_source_files"]),
        ("repository_source", inputs["repository_sources"]),
    ):
        records.extend({**dict(record), "role": role} for record in values)
    return {"inputs": {"records": records}}


def _rehash_inputs(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Rehash every unique input using the shared paper evidence contract."""

    return _paper_runtime_evidence()._rehash_inputs(_paper_input_view(manifest))


def _capture_gpu_environment(runtime: Runtime, output_dir: Path) -> dict[str, Any]:
    return _paper_runtime_evidence()._capture_gpu_environment(runtime, output_dir)


def _start_gpu_telemetry(runtime: Runtime, output_dir: Path) -> Any:
    return _paper_runtime_evidence()._GpuTelemetrySampler(runtime, output_dir)


def _validate_gpu_telemetry_contract(
    gpu_environment: Mapping[str, Any],
    gpu_summary: Mapping[str, Any],
) -> None:
    _paper_runtime_evidence()._validate_gpu_telemetry_contract(
        gpu_environment, gpu_summary
    )


def _training_numerical_status(
    info_log: Path, console_log: Path
) -> dict[str, Any]:
    return _paper_runtime_evidence()._training_numerical_status(
        info_log, console_log
    )


def _subprocess_environment(runtime: Runtime) -> dict[str, str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = runtime.cuda_visible_devices
    environment["DATA_ROOT"] = str(runtime.data_root)
    environment["OMP_NUM_THREADS"] = str(runtime.omp_num_threads)
    environment["TORCH_MP_SHARING_STRATEGY"] = runtime.mp_sharing_strategy
    old_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not old_pythonpath
        else str(REPO_ROOT) + os.pathsep + old_pythonpath
    )
    return environment


def _stream_subprocess(
    command: Sequence[str],
    *,
    runtime: Runtime,
    console_log: Path,
) -> int:
    """Stream merged stdout/stderr while retaining an exact persistent log."""

    interrupt_count = 0
    with console_log.open("w", encoding="utf-8", buffering=1) as log_handle:
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            env=_subprocess_environment(runtime),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
        if process.stdout is None:
            raise RuntimeError("failed to capture training stdout")
        while True:
            try:
                line = process.stdout.readline()
            except KeyboardInterrupt:
                interrupt_count += 1
                forwarded_signal = (
                    signal.SIGINT if interrupt_count == 1 else signal.SIGTERM
                )
                os.killpg(process.pid, forwarded_signal)
                notice = (
                    f"\n[launcher] forwarded {forwarded_signal.name} to training; "
                    "waiting for checkpoint/exit\n"
                )
                sys.stdout.write(notice)
                sys.stdout.flush()
                log_handle.write(notice)
                continue
            if line:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_handle.write(line)
                continue
            if process.poll() is not None:
                break
        return int(process.wait())


_SAFE_CHECKPOINT_INSPECT_SCRIPT = r"""
import json
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
wanted = [
    "seed",
    "batch_size",
    "max_train_iters",
    "iter_checkpoint_interval",
    "config_file",
    "datasets",
    "output_dir",
    "pretrain_model_path",
    "stage_b_v15_scorer_init_checkpoint",
    "stage_b_v15_scorer_init_audit",
    "stage_b_v20_acc50_aligned_hard_negatives",
    "stage_b_v11_candidate_topk",
    "stage_b_v11_positive_iou_threshold",
    "stage_b_v11_negative_iou_threshold",
    "stage_b_v21_token_objective",
    "stage_b_v21_token_weight",
    "stage_b_v21_token_positive_weight",
    "stage_b_v21_token_shared_weight",
    "stage_b_v21_token_edit_weight",
    "stage_b_v21_token_focal_alpha",
    "stage_b_v21_token_focal_gamma",
    "stage_b_v11_predicate_tn_rank_weight",
    "stage_b_v21_allow_legacy_token_diff_fallback",
    "stage_b_v19_allow_scope_labeled_tn_ablation",
    "stage_b_v19_table_b_id",
    "stage_b_v19_table_b_scope_allowlist",
    "stage_b_v19_table_b_audit",
    "stage_b_v19_table_b_audit_sha256",
    "stage_b_v19_table_b_allow_single_edit_token_provenance",
    "skip_eval",
    "amp",
]
result = {
    "top_level_keys": sorted(payload),
    "has_complete_training_state": all(
        key in payload
        for key in ("model", "criterion", "optimizer", "lr_scheduler", "scaler")
    ),
    "epoch": payload.get("epoch"),
    "iteration": payload.get("iteration"),
    "epoch_finished": payload.get("epoch_finished"),
    "checkpoint_reason": payload.get("checkpoint_reason"),
    "args": {key: args.get(key) for key in wanted},
}
print(json.dumps(result, sort_keys=True))
"""


def _inspect_checkpoint_safely(runtime: Runtime, checkpoint: Path) -> dict[str, Any]:
    environment = _subprocess_environment(runtime)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    try:
        completed = subprocess.run(
            [
                str(runtime.python),
                "-c",
                _SAFE_CHECKPOINT_INSPECT_SCRIPT,
                str(checkpoint),
            ],
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("safe checkpoint metadata inspection timed out") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-4000:]
        raise RuntimeError(
            "safe weights-only checkpoint metadata inspection failed: " + detail
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "safe checkpoint inspector returned invalid JSON: "
            + completed.stdout[-1000:]
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("safe checkpoint metadata must be a mapping")
    return payload


def _resolved_path(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"checkpoint path metadata is invalid: {value!r}")
    return Path(value).expanduser().resolve(strict=False)


def _validate_checkpoint_metadata(
    metadata: Mapping[str, Any],
    *,
    runtime: Runtime,
    row: MatrixRow,
    seed: int,
    output_dir: Path,
    scorer_sha256: str,
    scorer_audit: Mapping[str, Any],
) -> None:
    expected_top_level = {
        "model",
        "criterion",
        "optimizer",
        "lr_scheduler",
        "scaler",
        "args",
    }
    if not expected_top_level.issubset(set(metadata.get("top_level_keys", []))):
        raise RuntimeError("iteration checkpoint lacks complete training state")
    if metadata.get("has_complete_training_state") is not True:
        raise RuntimeError("iteration checkpoint training-state audit failed")
    if int(metadata.get("iteration", -1)) != runtime.max_train_iters:
        raise RuntimeError(
            "iteration checkpoint mismatch: expected "
            f"{runtime.max_train_iters}, got {metadata.get('iteration')}"
        )
    if metadata.get("checkpoint_reason") != "max_train_iters":
        raise RuntimeError(
            "iteration checkpoint reason must be max_train_iters, got "
            f"{metadata.get('checkpoint_reason')!r}"
        )
    if metadata.get("epoch_finished") is not False:
        raise RuntimeError("max_train_iters checkpoint must be a mid-epoch state")
    checkpoint_args = metadata.get("args")
    if not isinstance(checkpoint_args, Mapping):
        raise RuntimeError("iteration checkpoint args metadata is missing")

    expected_scalars = {
        "seed": seed,
        "batch_size": runtime.batch_size,
        "max_train_iters": runtime.max_train_iters,
        "iter_checkpoint_interval": runtime.iter_checkpoint_interval,
        "stage_b_v20_acc50_aligned_hard_negatives": True,
        "stage_b_v11_candidate_topk": 50,
        "stage_b_v11_positive_iou_threshold": 0.5,
        "stage_b_v11_negative_iou_threshold": 0.499,
        "stage_b_v21_token_objective": row.token_objective,
        "stage_b_v21_token_weight": 1.0,
        "stage_b_v21_token_positive_weight": row.positive_weight,
        "stage_b_v21_token_shared_weight": row.shared_weight,
        "stage_b_v21_token_edit_weight": row.edit_weight,
        "stage_b_v21_token_focal_alpha": 0.25,
        "stage_b_v21_token_focal_gamma": 2.0,
        "stage_b_v11_predicate_tn_rank_weight": row.predicate_pair_rank_weight,
        "stage_b_v21_allow_legacy_token_diff_fallback": False,
        "stage_b_v19_allow_scope_labeled_tn_ablation": True,
        "stage_b_v19_table_b_id": "D3",
        "stage_b_v19_table_b_scope_allowlist": ["proposal_covered_verified"],
        "stage_b_v19_table_b_audit": (
            "data/ablations/stageb_tn_table_b_equal_exposure_20260717/audit.json"
        ),
        "stage_b_v19_table_b_audit_sha256": (
            "7d74d541529a3e9abfbe84b192f2d0d3608d291bf46d19263c7c06a6ccb2291d"
        ),
        "stage_b_v19_table_b_allow_single_edit_token_provenance": True,
        "skip_eval": True,
        "amp": True,
    }
    for key, expected in expected_scalars.items():
        if checkpoint_args.get(key) != expected:
            raise RuntimeError(
                f"checkpoint args mismatch for {key}: expected {expected!r}, "
                f"got {checkpoint_args.get(key)!r}"
            )
    expected_paths = {
        "config_file": (REPO_ROOT / row.config).resolve(strict=True),
        "datasets": runtime.dataset,
        "output_dir": output_dir,
        "pretrain_model_path": runtime.stage_a_init,
        "stage_b_v15_scorer_init_checkpoint": runtime.scorer_warmstart,
    }
    for key, expected in expected_paths.items():
        if _resolved_path(checkpoint_args.get(key)) != expected.resolve(strict=False):
            raise RuntimeError(
                f"checkpoint args path mismatch for {key}: "
                f"expected {expected}, got {checkpoint_args.get(key)!r}"
            )

    embedded_audit = checkpoint_args.get("stage_b_v15_scorer_init_audit")
    if not isinstance(embedded_audit, Mapping):
        raise RuntimeError("checkpoint lacks embedded scorer warm-start audit")
    if dict(embedded_audit) != dict(scorer_audit):
        raise RuntimeError(
            "checkpoint scorer warm-start audit differs from persisted audit"
        )
    if embedded_audit.get("source_sha256") != scorer_sha256:
        raise RuntimeError("checkpoint scorer warm-start SHA-256 mismatch")


def _perform_postflight(
    manifest: Mapping[str, Any],
    *,
    runtime: Runtime,
    row: MatrixRow,
    seed: int,
    cache: HashCache,
) -> dict[str, Any]:
    output_dir = Path(str(manifest["output_dir"]))
    required = {
        "checkpoint": output_dir / "checkpoint_iter.pth",
        "scorer_init_audit": output_dir / "stage_b_v15_scorer_init_audit.json",
        "native_info_log": output_dir / "info.txt",
        "train_console_log": output_dir / "train_console.log",
        "launch_manifest": output_dir / "launch_manifest.json",
        "gpu_environment": output_dir / "gpu_environment.json",
        "gpu_telemetry": output_dir / "gpu_telemetry.csv",
        "gpu_telemetry_summary": output_dir / "gpu_telemetry_summary.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"postflight is missing required artifacts: {missing}")
    for name in ("native_info_log", "train_console_log"):
        if required[name].stat().st_size <= 0:
            raise RuntimeError(f"postflight artifact is empty: {required[name]}")

    # Persist immutable-input evidence first so it survives any later derived
    # artifact validation failure.
    input_rehash = _rehash_inputs(manifest)
    input_rehash_path = output_dir / "input_rehash.json"
    _write_json_atomic(input_rehash_path, input_rehash)
    required["input_rehash"] = input_rehash_path

    scorer_audit = json.loads(
        required["scorer_init_audit"].read_text(encoding="utf-8")
    )
    if not isinstance(scorer_audit, dict):
        raise RuntimeError("scorer initialization audit must be a mapping")
    expected_scorer_audit = {
        "schema": "stage_b_v15_scorer_init/v1",
        "status": "applied",
        "source_sha256": manifest["inputs"]["scorer_warmstart"]["sha256"],
        "loaded_num_layers": 3,
    }
    for key, expected in expected_scorer_audit.items():
        if scorer_audit.get(key) != expected:
            raise RuntimeError(
                f"scorer initialization audit mismatch for {key}: "
                f"expected {expected!r}, got {scorer_audit.get(key)!r}"
            )
    if _resolved_path(scorer_audit.get("resolved_source_path")) != (
        runtime.scorer_warmstart.resolve(strict=False)
    ):
        raise RuntimeError("scorer initialization audit source path mismatch")

    checkpoint_metadata = _inspect_checkpoint_safely(
        runtime, required["checkpoint"]
    )
    _validate_checkpoint_metadata(
        checkpoint_metadata,
        runtime=runtime,
        row=row,
        seed=seed,
        output_dir=output_dir,
        scorer_sha256=manifest["inputs"]["scorer_warmstart"]["sha256"],
        scorer_audit=scorer_audit,
    )
    gpu_environment = json.loads(
        required["gpu_environment"].read_text(encoding="utf-8")
    )
    gpu_summary = json.loads(
        required["gpu_telemetry_summary"].read_text(encoding="utf-8")
    )
    _validate_gpu_telemetry_contract(gpu_environment, gpu_summary)
    numerical_status = _training_numerical_status(
        required["native_info_log"], required["train_console_log"]
    )
    epoch_log = output_dir / "log.txt"
    return {
        "schema": "pivot.stageb.token_ablation_postflight/v2",
        "status": "passed",
        "validated_at_utc": _utc_now(),
        "run_id": manifest["run_id"],
        "checkpoint_metadata": checkpoint_metadata,
        "input_rehash": input_rehash,
        "gpu_environment": gpu_environment,
        "gpu_telemetry_summary": gpu_summary,
        "numerical_status": numerical_status,
        "artifacts": {
            name: _file_record(path, cache)
            for name, path in required.items()
            if name != "launch_manifest"
        },
        "launch_manifest": {
            "path": str(required["launch_manifest"].resolve(strict=True)),
            "present": True,
            "hash_omitted": "manifest embeds postflight and is updated after validation",
        },
        "native_epoch_log": (
            _file_record(epoch_log, cache)
            if epoch_log.is_file()
            else {
                "path": str(epoch_log.resolve(strict=False)),
                "present": False,
                "expected_absence_reason": (
                    "max_train_iters exits before native epoch aggregation"
                ),
            }
        ),
    }


def _orchestration_status_path() -> Path | None:
    raw = os.environ.get("PIVOT_ORCHESTRATION_STATUS", "").strip()
    return Path(raw).expanduser().resolve(strict=False) if raw else None


def _update_orchestration_status(
    path: Path | None,
    *,
    status: str,
    **fields: Any,
) -> None:
    if path is None:
        return
    payload: dict[str, Any] = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict):
            payload.update(existing)
    payload.update(fields)
    payload["schema"] = "pivot.stageb.token_ablation_orchestration_status/v1"
    payload["status"] = status
    payload["updated_at_utc"] = _utc_now()
    payload["pid"] = os.getpid()
    _write_json_atomic(path, payload)


def _read_process_identity(pid: int) -> dict[str, Any]:
    return _paper_runtime_evidence()._read_process_identity(pid)


def _inspect_or_reconcile_detached_job(
    job_dir: Path, *, mutate: bool
) -> dict[str, Any]:
    return _paper_runtime_evidence()._inspect_or_reconcile_detached_job(
        job_dir, mutate=mutate
    )


def _build_sequence_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    runtime = manifest["runtime"]
    return {
        "schema": "pivot.stageb.token_ablation_sequence/v1",
        "status": "planned",
        "created_at_utc": _utc_now(),
        "repository_root": str(REPO_ROOT),
        "run_id": manifest["run_id"],
        "row": dict(manifest["row"]),
        "seed": manifest["seed"],
        "training_seeds_contract": list(SEEDS),
        "output_dir": manifest["output_dir"],
        "equal_budget_contract": {
            "batch_size": int(runtime["batch_size"]),
            "optimizer_updates": int(runtime["max_train_iters"]),
            "contributing_phase_updates": {
                "joint": int(runtime["max_train_iters"])
            },
        },
        "phases": [
            {
                "phase_id": "joint",
                "output_dir": manifest["output_dir"],
                "optimizer_updates": int(runtime["max_train_iters"]),
                "contributes_to_budget": True,
            }
        ],
    }


def _parse_run_id(value: str) -> tuple[MatrixRow, int]:
    try:
        row_value, seed_value = value.upper().split(":", 1)
        row = ROW_BY_ID[row_value]
        seed = int(seed_value)
    except (KeyError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"run id must be ROW:SEED with ROW=L0..L10 and SEED in {SEEDS}; "
            f"got {value!r}"
        ) from exc
    if seed not in SEEDS:
        raise argparse.ArgumentTypeError(
            f"training seed must be one of {SEEDS}, got {seed}"
        )
    return row, seed


def _all_runs() -> list[tuple[MatrixRow, int]]:
    return [(row, seed) for row in ROWS for seed in SEEDS]


def _selected_runs(args: argparse.Namespace) -> list[tuple[MatrixRow, int]]:
    if args.run_id:
        values = list(dict.fromkeys(args.run_id))
    elif args.mode == "dry-run" or args.all:
        values = _all_runs()
    else:
        raise ValueError("run requires at least one --run-id ROW:SEED or --all")
    return values


def _add_selection_arguments(parser: argparse.ArgumentParser, *, run: bool) -> None:
    parser.add_argument(
        "--run-id",
        action="append",
        type=_parse_run_id,
        help="select one ROW:SEED; repeat to select multiple runs",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "run the complete 33-run matrix" if run else "show all runs (the default)"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    list_parser = subparsers.add_parser("list", help="list rows, seeds, and run ids")
    list_parser.add_argument("--json", action="store_true", help="emit JSON")

    dry_parser = subparsers.add_parser(
        "dry-run", help="hash inputs and print commands without creating train outputs"
    )
    _add_selection_arguments(dry_parser, run=False)
    dry_parser.add_argument(
        "--manifest",
        type=Path,
        help="write one selected planned manifest to this path",
    )
    dry_parser.add_argument(
        "--manifest-dir",
        type=Path,
        help="write planned manifests for every selection under this directory",
    )

    run_parser = subparsers.add_parser(
        "run", help="launch explicitly selected fresh training runs"
    )
    _add_selection_arguments(run_parser, run=True)
    detach_parser = subparsers.add_parser(
        "detach",
        help=(
            "preflight then launch an independent background orchestrator with "
            "persistent log/status artifacts"
        ),
    )
    _add_selection_arguments(detach_parser, run=True)
    detach_parser.add_argument(
        "--orchestration-root",
        type=Path,
        default=None,
        help=(
            "control-artifact root (default: PIVOT_ORCHESTRATION_ROOT or "
            "outputs/paper_cvpr_v1/orchestration/token_ablation)"
        ),
    )
    status_parser = subparsers.add_parser(
        "status",
        help="inspect one detached job without modifying its persisted status",
    )
    status_parser.add_argument(
        "job_dir",
        type=Path,
        help="job directory containing launch.json and status.json",
    )
    reconcile_parser = subparsers.add_parser(
        "reconcile",
        help=(
            "atomically repair a stale detached status using PID liveness and "
            "explicit terminal artifacts"
        ),
    )
    reconcile_parser.add_argument(
        "job_dir",
        type=Path,
        help="job directory containing launch.json and status.json",
    )
    return parser


def _list_rows(as_json: bool) -> int:
    payload = {
        "rows": [asdict(row) for row in ROWS],
        "seeds": list(SEEDS),
        "run_ids": [f"{row.row_id}:{seed}" for row, seed in _all_runs()],
    }
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for row in ROWS:
            print(
                f"{row.row_id}: objective={row.token_objective}, "
                f"pair_rank={row.predicate_pair_rank_weight:g}, "
                f"roles={row.positive_weight:g}/{row.shared_weight:g}/{row.edit_weight:g}, "
                f"config={row.config}"
            )
        print("seeds: " + ",".join(str(seed) for seed in SEEDS))
    return 0


def _dry_run(args: argparse.Namespace) -> int:
    selections = _selected_runs(args)
    if args.manifest is not None and len(selections) != 1:
        raise ValueError("--manifest requires exactly one --run-id")
    if args.manifest is not None and args.manifest_dir is not None:
        raise ValueError("use only one of --manifest and --manifest-dir")
    runtime = runtime_from_environment()
    cache = HashCache()
    for row, seed in selections:
        manifest = build_manifest(runtime, row, seed, cache)
        print(f"[{manifest['run_id']}] {manifest['command_shell']}")
        if args.manifest is not None:
            _write_json_atomic(args.manifest.resolve(strict=False), manifest)
        elif args.manifest_dir is not None:
            target = (
                args.manifest_dir.resolve(strict=False)
                / row.row_id
                / f"seed{seed}.launch.json"
            )
            _write_json_atomic(target, manifest)
    return 0


def _detach(args: argparse.Namespace) -> int:
    """Preflight synchronously, then hand selected runs to a new OS session."""

    selections = _selected_runs(args)
    runtime = runtime_from_environment()
    outputs = [output_directory(runtime, row, seed) for row, seed in selections]
    conflicts = [path for path in outputs if path.exists()]
    if conflicts:
        rendered = "\n".join(f"  {path}" for path in conflicts)
        raise FileExistsError(
            "every selected output must be fresh; existing paths:\n" + rendered
        )

    cache = HashCache()
    planned = [build_manifest(runtime, row, seed, cache) for row, seed in selections]
    root_value = args.orchestration_root
    if root_value is None:
        root_value = Path(
            os.environ.get(
                "PIVOT_ORCHESTRATION_ROOT", str(DEFAULT_ORCHESTRATION_ROOT)
            )
        )
    orchestration_root = root_value.expanduser().resolve(strict=False)
    job_name = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + f"-pid{os.getpid()}"
    )
    job_dir = orchestration_root / job_name
    job_dir.mkdir(parents=True, exist_ok=False)
    plans_dir = job_dir / "plans"
    for (row, seed), manifest in zip(selections, planned):
        _write_json_atomic(plans_dir / row.row_id / f"seed{seed}.json", manifest)

    child_command = [
        sys.executable,
        str(Path(__file__).resolve(strict=True)),
        "run",
    ]
    for row, seed in selections:
        child_command.extend(("--run-id", f"{row.row_id}:{seed}"))
    log_path = job_dir / "orchestrator.log"
    status_path = job_dir / "status.json"
    launch_path = job_dir / "launch.json"
    launch: dict[str, Any] = {
        "schema": "pivot.stageb.token_ablation_detached_launch/v1",
        "status": "prepared",
        "created_at_utc": _utc_now(),
        "job_dir": str(job_dir),
        "run_ids": [f"{row.row_id}:{seed}" for row, seed in selections],
        "expected_run_roots": [str(path) for path in outputs],
        "command": child_command,
        "command_shell": shlex.join(child_command),
        "orchestrator_log": str(log_path),
        "orchestrator_status": str(status_path),
        "plans_dir": str(plans_dir),
        "runtime": {
            "python": str(runtime.python),
            "batch_size": runtime.batch_size,
            "max_train_iters": runtime.max_train_iters,
            "cuda_visible_devices": runtime.cuda_visible_devices,
            "token_output_root": str(runtime.output_root),
        },
    }
    _write_json_atomic(launch_path, launch)
    _update_orchestration_status(
        status_path,
        status="prepared",
        job_dir=str(job_dir),
        run_ids=launch["run_ids"],
        expected_run_roots=launch["expected_run_roots"],
    )
    environment = dict(os.environ)
    environment["PIVOT_ORCHESTRATION_STATUS"] = str(status_path)
    environment["PYTHONUNBUFFERED"] = "1"
    try:
        with log_path.open("ab", buffering=0) as log_handle:
            process = subprocess.Popen(
                child_command,
                cwd=REPO_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    except BaseException as exc:
        launch["status"] = "spawn_failed"
        launch["spawn_error"] = f"{type(exc).__name__}: {exc}"
        launch["finished_at_utc"] = _utc_now()
        _write_json_atomic(launch_path, launch)
        _update_orchestration_status(
            status_path,
            status="spawn_failed",
            error=launch["spawn_error"],
        )
        raise
    launch["status"] = "launched"
    launch["launched_at_utc"] = _utc_now()
    launch["child_pid"] = int(process.pid)
    launch["child_process_identity"] = _read_process_identity(int(process.pid))
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


def _run_body(
    args: argparse.Namespace,
    *,
    orchestration_status: Path | None,
) -> int:
    selections = _selected_runs(args)
    runtime = runtime_from_environment()
    outputs = [output_directory(runtime, row, seed) for row, seed in selections]
    conflicts = [path for path in outputs if path.exists()]
    if conflicts:
        rendered = "\n".join(f"  {path}" for path in conflicts)
        raise FileExistsError(
            "every selected output must be fresh; existing paths:\n" + rendered
        )

    cache = HashCache()
    # Validate and hash the entire selection before creating the first output.
    planned = [build_manifest(runtime, row, seed, cache) for row, seed in selections]
    completed_run_ids: list[str] = []
    _update_orchestration_status(
        orchestration_status,
        status="preflight_passed",
        run_ids=[manifest["run_id"] for manifest in planned],
        expected_run_roots=[str(path) for path in outputs],
        completed_run_ids=completed_run_ids,
    )

    for (row, seed), manifest in zip(selections, planned):
        output_dir = Path(manifest["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = output_dir / "launch_manifest.json"
        sequence_path = output_dir / "sequence_manifest.json"
        sequence = _build_sequence_manifest(manifest)
        manifest["status"] = "running"
        manifest["started_at_utc"] = _utc_now()
        sequence["status"] = "running"
        sequence["started_at_utc"] = manifest["started_at_utc"]
        _write_json_atomic(manifest_path, manifest)
        _write_json_atomic(sequence_path, sequence)
        _update_orchestration_status(
            orchestration_status,
            status="running",
            current_run_id=manifest["run_id"],
            current_phase_id="joint",
            completed_run_ids=completed_run_ids,
        )
        try:
            _verify_file_identities(manifest)
            print(f"[{manifest['run_id']}] {manifest['command_shell']}", flush=True)
            gpu_environment = _capture_gpu_environment(runtime, output_dir)
            sampler = _start_gpu_telemetry(runtime, output_dir)
            try:
                returncode = _stream_subprocess(
                    manifest["command"],
                    runtime=runtime,
                    console_log=output_dir / "train_console.log",
                )
            finally:
                gpu_summary = sampler.stop()
            manifest["gpu_environment"] = gpu_environment
            manifest["gpu_telemetry_summary"] = gpu_summary
        except BaseException as exc:
            manifest["status"] = "failed"
            manifest["failure_phase"] = "gpu_telemetry_or_training_process"
            manifest["failure_error"] = f"{type(exc).__name__}: {exc}"
            manifest["finished_at_utc"] = _utc_now()
            sequence["status"] = "failed"
            sequence["finished_at_utc"] = manifest["finished_at_utc"]
            sequence["completed_phases"] = []
            sequence["error"] = manifest["failure_error"]
            _write_json_atomic(manifest_path, manifest)
            _write_json_atomic(sequence_path, sequence)
            raise
        manifest["returncode"] = int(returncode)
        manifest["finished_at_utc"] = _utc_now()
        manifest["training_finished_at_utc"] = manifest["finished_at_utc"]
        if returncode != 0:
            manifest["status"] = "failed"
            manifest["failure_phase"] = "training_process"
            sequence["status"] = "failed"
            sequence["finished_at_utc"] = manifest["finished_at_utc"]
            sequence["completed_phases"] = []
            sequence["error"] = f"training process exited {returncode}"
            _write_json_atomic(manifest_path, manifest)
            _write_json_atomic(sequence_path, sequence)
            return int(returncode) or 1
        try:
            postflight = _perform_postflight(
                manifest,
                runtime=runtime,
                row=row,
                seed=seed,
                cache=cache,
            )
            postflight_path = output_dir / "postflight.json"
            _write_json_atomic(postflight_path, postflight)
            manifest["postflight"] = postflight
            manifest["postflight_artifact"] = _file_record(
                postflight_path, cache
            )
            manifest["status"] = "completed"
            manifest["finished_at_utc"] = _utc_now()
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["failure_phase"] = "postflight"
            manifest["postflight_error"] = f"{type(exc).__name__}: {exc}"
            manifest["finished_at_utc"] = _utc_now()
            sequence["status"] = "failed"
            sequence["finished_at_utc"] = _utc_now()
            sequence["completed_phases"] = []
            sequence["error"] = manifest["postflight_error"]
            _write_json_atomic(manifest_path, manifest)
            _write_json_atomic(sequence_path, sequence)
            print(
                f"[{manifest['run_id']}] postflight failed: {exc}",
                file=sys.stderr,
            )
            return 1
        _write_json_atomic(manifest_path, manifest)
        checkpoint = output_dir / "checkpoint_iter.pth"
        sequence["status"] = "completed"
        sequence["finished_at_utc"] = manifest["finished_at_utc"]
        sequence["completed_phases"] = [
            {
                "phase_id": "joint",
                "status": "completed",
                "output_dir": str(output_dir),
                "checkpoint": _file_record(checkpoint, cache),
                "postflight": _file_record(postflight_path, cache),
            }
        ]
        _write_json_atomic(sequence_path, sequence)
        completed_run_ids.append(manifest["run_id"])
        _update_orchestration_status(
            orchestration_status,
            status="running",
            current_run_id=None,
            current_phase_id=None,
            completed_run_ids=completed_run_ids,
        )
    return 0


def _run(args: argparse.Namespace) -> int:
    status_path = _orchestration_status_path()
    try:
        selections = _selected_runs(args)
        _update_orchestration_status(
            status_path,
            status="starting",
            run_ids=[f"{row.row_id}:{seed}" for row, seed in selections],
            started_at_utc=_utc_now(),
        )
        result = _run_body(args, orchestration_status=status_path)
    except BaseException as exc:
        _update_orchestration_status(
            status_path,
            status="failed",
            finished_at_utc=_utc_now(),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    if result == 0:
        _update_orchestration_status(
            status_path,
            status="completed",
            finished_at_utc=_utc_now(),
            current_run_id=None,
            current_phase_id=None,
        )
    else:
        _update_orchestration_status(
            status_path,
            status="failed",
            finished_at_utc=_utc_now(),
            returncode=int(result),
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.mode == "list":
            return _list_rows(args.json)
        if args.mode == "dry-run":
            return _dry_run(args)
        if args.mode == "run":
            return _run(args)
        if args.mode == "detach":
            return _detach(args)
        if args.mode == "status":
            print(
                json.dumps(
                    _inspect_or_reconcile_detached_job(
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
                    _inspect_or_reconcile_detached_job(
                        args.job_dir, mutate=True
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
        NotADirectoryError,
        PermissionError,
        RuntimeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
