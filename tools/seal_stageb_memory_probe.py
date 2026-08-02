#!/usr/bin/env python3
"""Seal a Stage-B batch-size ladder and soak into a paper runtime contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "pivot.stageb.paper_memory_probe_seal/v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PAPER_LAUNCH_SCHEMA = "pivot.stageb.paper_ablation_phase_launch/v1"
PAPER_POSTFLIGHT_SCHEMA = "pivot.stageb.paper_ablation_phase_postflight/v1"
PAPER_SEQUENCE_SCHEMA = "pivot.stageb.paper_ablation_run_launch/v1"
TABLE_D_ROW_IDS = frozenset({"S0", "S1", "S2", "S3", "S2F"})
MATCHED_TABLE_B_ROW_IDS = frozenset({"D2m", "D3m"})
TABLE_B_MEMORY_ROW_ID = "D3m"
TABLE_B_MEMORY_BATCH_SIZE = 40
TABLE_B_MEMORY_SEED = 17
TABLE_B_MEMORY_SOAK_UPDATES = 50
TABLE_B_MEMORY_MIN_HEADROOM_MIB = 1024.0


class MemoryProbeError(ValueError):
    pass


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MemoryProbeError(f"{label}: cannot read {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise MemoryProbeError(f"{label}: JSON root must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _require_file_record(record: Any, path: Path, *, label: str) -> None:
    if not isinstance(record, Mapping):
        raise MemoryProbeError(f"{label}: file record is missing")
    path = path.resolve(strict=True)
    recorded = Path(str(record.get("path", ""))).expanduser().resolve(strict=False)
    if recorded != path:
        raise MemoryProbeError(
            f"{label}: recorded path {recorded} differs from {path}"
        )
    observed = _file_record(path)
    for key in ("size_bytes", "sha256"):
        if record.get(key) != observed[key]:
            raise MemoryProbeError(
                f"{label}: {key} drifted; expected {record.get(key)!r}, "
                f"observed {observed[key]!r}"
            )


def _paper_input_records(launch: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    inputs = launch.get("inputs")
    records = inputs.get("records") if isinstance(inputs, Mapping) else None
    if not isinstance(records, list) or not records:
        raise MemoryProbeError("paper probe launch has no input records")
    if any(not isinstance(record, Mapping) for record in records):
        raise MemoryProbeError("paper probe launch contains an invalid input record")
    return tuple(records)


def _current_paper_training_closure(config_path: Path) -> tuple[Path, ...]:
    try:
        from tools.stageb_dependency_audit import (
            DependencyAuditError,
            config_import_chain,
            local_python_dependency_paths,
        )

        closure = set(
            local_python_dependency_paths(
                [
                    REPO_ROOT / "main.py",
                    REPO_ROOT / "tools/run_stageb_paper_ablation_matrices.py",
                    REPO_ROOT / "tools/run_stageb_token_ablation_matrix.py",
                ],
                root=REPO_ROOT,
            )
        )
        closure.update(config_import_chain(config_path, root=REPO_ROOT))
    except (DependencyAuditError, OSError, ValueError) as error:
        raise MemoryProbeError(
            f"cannot compute current recursive paper-training closure: {error}"
        ) from error
    closure.update(_native_training_dependency_paths())
    closure.add((REPO_ROOT / "docs/paper_cvpr_ablation_protocol.md").resolve(strict=True))
    return tuple(sorted(closure, key=lambda value: str(value)))


def _native_training_dependency_paths() -> tuple[Path, ...]:
    """Bind the imported deformable-attention binary to source/build lineage."""

    native_root = (REPO_ROOT / "models/GroundingDINO/ops").resolve(strict=True)
    patterns = (
        "MultiScaleDeformableAttention*.so",
        "build/**/MultiScaleDeformableAttention*.so",
        "src/**/*.cpp",
        "src/**/*.cu",
        "src/**/*.h",
        "build/**/build.ninja",
        "MultiScaleDeformableAttention.egg-info/*",
    )
    paths = {
        path.resolve(strict=True)
        for pattern in patterns
        for path in native_root.glob(pattern)
        if path.is_file()
    }
    spec = importlib.util.find_spec("MultiScaleDeformableAttention")
    origin = getattr(spec, "origin", None) if spec is not None else None
    if not isinstance(origin, str) or not origin:
        raise MemoryProbeError(
            "current recursive paper-training closure cannot resolve "
            "MultiScaleDeformableAttention"
        )
    actual = Path(origin).expanduser().resolve(strict=True)
    if not actual.is_file() or actual.suffix != ".so":
        raise MemoryProbeError(
            "current MultiScaleDeformableAttention runtime is not a shared object"
        )
    paths.add(actual)
    for metadata in actual.parent.glob(
        "MultiScaleDeformableAttention-*.egg-info/*"
    ):
        if metadata.is_file():
            paths.add(metadata.resolve(strict=True))
    root_extensions = [
        path
        for path in paths
        if path.parent == native_root
        and path.name.startswith("MultiScaleDeformableAttention")
        and path.suffix == ".so"
    ]
    suffixes = {path.suffix for path in paths}
    if len(root_extensions) != 1:
        raise MemoryProbeError(
            "current recursive paper-training closure requires exactly one "
            "importable root MultiScaleDeformableAttention extension"
        )
    if not {".cpp", ".cu", ".h"}.issubset(suffixes):
        raise MemoryProbeError(
            "current recursive paper-training closure lacks native source lineage"
        )
    if not any(path.name == "build.ninja" for path in paths):
        raise MemoryProbeError(
            "current recursive paper-training closure lacks native build metadata"
        )
    return tuple(sorted(paths, key=lambda value: str(value)))


def _verify_current_input_closure(
    launch: Mapping[str, Any], closure: Iterable[Path]
) -> None:
    records = _paper_input_records(launch)
    by_path: dict[Path, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise MemoryProbeError(f"paper probe input {index} has no path")
        path = Path(raw_path).expanduser().resolve(strict=True)
        previous = by_path.setdefault(path, record)
        if previous.get("sha256") != record.get("sha256"):
            raise MemoryProbeError(
                f"paper probe input records disagree on the digest for {path}"
            )
    missing = [path for path in closure if path.resolve(strict=True) not in by_path]
    if missing:
        rendered = [str(path) for path in missing[:12]]
        raise MemoryProbeError(
            "paper probe launch omitted current recursive training-source closure: "
            f"{rendered}"
        )
    for path in closure:
        resolved = path.resolve(strict=True)
        _require_file_record(
            by_path[resolved], resolved, label=f"paper training closure {resolved}"
        )


def _table_d_expected_diagnostic_interval(
    row_id: str, *, updates: int, minimum_soak_updates: int
) -> int | None:
    if row_id not in {"S2", "S2F"}:
        return None
    return 10 if updates >= minimum_soak_updates else 1


def _exact_int(value: Any, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MemoryProbeError(f"{label}: expected an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise MemoryProbeError(
            f"{label}: expected an integer >= {minimum}, got {value!r}"
        )
    return value


def _single_input_record(
    launch: Mapping[str, Any], *, role: str
) -> Mapping[str, Any]:
    records = [
        record
        for record in _paper_input_records(launch)
        if record.get("role") == role
    ]
    if len(records) != 1:
        raise MemoryProbeError(
            f"paper probe requires exactly one {role!r} input record, "
            f"got {len(records)}"
        )
    return records[0]


def _record_path(record: Mapping[str, Any], *, label: str) -> Path:
    raw = record.get("path")
    if not isinstance(raw, str) or not raw.strip():
        raise MemoryProbeError(f"{label}: input record has no path")
    try:
        return Path(raw).expanduser().resolve(strict=True)
    except OSError as error:
        raise MemoryProbeError(f"{label}: cannot resolve {raw!r}: {error}") from error


def _infer_data_root(launch: Mapping[str, Any]) -> Path:
    """Recover the launch-time DATA_ROOT from the persisted dataset contract."""

    fixed = launch.get("fixed_contract")
    dataset = fixed.get("dataset") if isinstance(fixed, Mapping) else None
    sources = dataset.get("source_paths") if isinstance(dataset, Mapping) else None
    if not isinstance(sources, list):
        raise MemoryProbeError("paper probe lacks dataset source-path evidence")
    candidates: set[Path] = set()
    marker = "${DATA_ROOT}"
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise MemoryProbeError(
                f"paper probe dataset source path {index} is not an object"
            )
        declared = source.get("declared")
        resolved_raw = source.get("resolved")
        if not isinstance(declared, str) or marker not in declared:
            continue
        if not isinstance(resolved_raw, str) or not resolved_raw.strip():
            raise MemoryProbeError(
                f"paper probe dataset source path {index} has no resolved path"
            )
        suffix = Path(declared.split(marker, 1)[1].lstrip("/"))
        resolved = Path(resolved_raw).expanduser().resolve(strict=True)
        candidate = resolved
        for part in reversed(suffix.parts):
            if candidate.name != part:
                raise MemoryProbeError(
                    "paper probe dataset DATA_ROOT binding is internally "
                    f"inconsistent for {declared!r} -> {resolved}"
                )
            candidate = candidate.parent
        candidates.add(candidate.resolve(strict=True))
    if len(candidates) != 1:
        raise MemoryProbeError(
            "paper probe must prove exactly one DATA_ROOT through its resolved "
            f"dataset sources, got {sorted(str(value) for value in candidates)}"
        )
    return next(iter(candidates))


def _runtime_from_paper_launch(
    launch: Mapping[str, Any], *, root: Path, canonical: Any
) -> Any:
    """Reconstruct the runner Runtime solely from persisted, hash-bound evidence."""

    from tools import run_stageb_paper_ablation_matrices as paper_launcher

    runtime = launch.get("runtime")
    fixed = launch.get("fixed_contract")
    if not isinstance(runtime, Mapping) or not isinstance(fixed, Mapping):
        raise MemoryProbeError(f"{root}: launch runtime/fixed contract is missing")
    python_raw = runtime.get("python")
    if not isinstance(python_raw, str) or not python_raw.strip():
        raise MemoryProbeError(f"{root}: launch runtime Python is missing")
    python = Path(python_raw).expanduser().resolve(strict=True)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise MemoryProbeError(f"{root}: launch runtime Python is not executable")
    stage_a = _record_path(
        _single_input_record(launch, role="stage_a_initializer"),
        label="stage_a_initializer",
    )
    scorer = _record_path(
        _single_input_record(launch, role="scorer_warmstart"),
        label="scorer_warmstart",
    )
    run_family_root = root.parent.parent.resolve(strict=False)
    sharing = runtime.get("mp_sharing_strategy")
    if sharing not in {"file_system", "file_descriptor", "none"}:
        raise MemoryProbeError(
            f"{root}: invalid multiprocessing sharing strategy {sharing!r}"
        )
    cuda_devices = runtime.get("cuda_visible_devices")
    if not isinstance(cuda_devices, str) or not cuda_devices.strip():
        raise MemoryProbeError(f"{root}: CUDA_VISIBLE_DEVICES contract is missing")
    if runtime.get("amp") is not True:
        raise MemoryProbeError(f"{root}: paper memory probes require AMP")
    diagnostic = fixed.get("gradient_diagnostic_interval")
    return paper_launcher.Runtime(
        python=python,
        stage_a_init=stage_a,
        scorer_warmstart=scorer,
        tn_output_root=(
            run_family_root
            if canonical.table == "B"
            else (root / "unused_tn_output_root").resolve(strict=False)
        ),
        score_output_root=(
            run_family_root
            if canonical.table == "D"
            else (root / "unused_score_output_root").resolve(strict=False)
        ),
        data_root=_infer_data_root(launch),
        batch_size=_exact_int(
            runtime.get("batch_size"), label=f"{root} runtime batch_size", minimum=1
        ),
        total_train_iters=_exact_int(
            runtime.get("total_paper_train_iters"),
            label=f"{root} runtime total_paper_train_iters",
            minimum=1,
        ),
        iter_checkpoint_interval=_exact_int(
            runtime.get("iter_checkpoint_interval"),
            label=f"{root} runtime iter_checkpoint_interval",
            minimum=1,
        ),
        num_workers=_exact_int(
            runtime.get("num_workers"),
            label=f"{root} runtime num_workers",
            minimum=0,
        ),
        prefetch_factor=_exact_int(
            runtime.get("prefetch_factor"),
            label=f"{root} runtime prefetch_factor",
            minimum=1,
        ),
        omp_num_threads=_exact_int(
            runtime.get("omp_num_threads"),
            label=f"{root} runtime omp_num_threads",
            minimum=1,
        ),
        min_nofile=_exact_int(
            runtime.get("min_nofile"),
            label=f"{root} runtime min_nofile",
            minimum=0,
        ),
        cuda_visible_devices=cuda_devices,
        mp_sharing_strategy=sharing,
        gradient_diagnostic_interval=_exact_int(
            diagnostic,
            label=f"{root} fixed gradient_diagnostic_interval",
            minimum=0,
        ),
    )


def _compare_rehash_replay(
    persisted: Mapping[str, Any], fresh: Mapping[str, Any], *, root: Path
) -> None:
    for key in ("status", "algorithm", "unique_input_count", "records"):
        if persisted.get(key) != fresh.get(key):
            raise MemoryProbeError(
                f"{root}: persisted input rehash differs from fresh replay for {key}"
            )


def _compare_telemetry_replay(
    persisted: Mapping[str, Any], fresh: Mapping[str, Any], *, root: Path
) -> None:
    for key in ("schema", "sample_rows", "devices"):
        if persisted.get(key) != fresh.get(key):
            raise MemoryProbeError(
                f"{root}: GPU telemetry summary differs from CSV replay for {key}"
            )


def _verify_paper_probe_current(
    root: Path,
    *,
    expected_row_id: str | None,
    minimum_soak_updates: int,
) -> None:
    """Replay one formal paper probe against current code and raw evidence."""

    from tools import run_stageb_paper_ablation_matrices as paper_launcher

    root = root.expanduser().resolve(strict=True)
    sequence_path = root / "sequence_manifest.json"
    launch_path = root / "launch_manifest.json"
    postflight_path = root / "postflight.json"
    telemetry_path = root / "gpu_telemetry_summary.json"
    input_rehash_path = root / "input_rehash.json"
    gpu_environment_path = root / "gpu_environment.json"
    gpu_telemetry_path = root / "gpu_telemetry.csv"
    info_log_path = root / "info.txt"
    console_log_path = root / "train_console.log"
    scorer_audit_path = root / "stage_b_v15_scorer_init_audit.json"
    checkpoint_path = root / "checkpoint_iter.pth"
    sequence = _read_json(sequence_path, label="sequence manifest")
    launch = _read_json(launch_path, label="paper launch manifest")
    postflight = _read_json(postflight_path, label="paper postflight")
    telemetry = _read_json(telemetry_path, label="GPU telemetry summary")
    persisted_rehash = _read_json(input_rehash_path, label="input rehash")
    gpu_environment = _read_json(
        gpu_environment_path, label="GPU environment"
    )
    scorer_audit = _read_json(scorer_audit_path, label="scorer initialization audit")

    if (
        sequence.get("schema") != PAPER_SEQUENCE_SCHEMA
        or sequence.get("status") != "completed"
        or sequence.get("repository_root") != str(REPO_ROOT)
        or sequence.get("training_seeds_contract") != list(paper_launcher.SEEDS)
        or sequence.get("output_dir_fresh_at_plan") is not True
    ):
        raise MemoryProbeError(f"{root}: paper sequence contract is not canonical")
    if launch.get("schema") != PAPER_LAUNCH_SCHEMA or launch.get("status") != "completed":
        raise MemoryProbeError(f"{root}: paper launch is not completed")
    if (
        postflight.get("schema") != PAPER_POSTFLIGHT_SCHEMA
        or postflight.get("status") != "passed"
    ):
        raise MemoryProbeError(f"{root}: paper postflight is not passed")
    if launch.get("postflight") != postflight:
        raise MemoryProbeError(f"{root}: embedded and persisted postflight differ")
    _require_file_record(
        launch.get("postflight_artifact"),
        postflight_path,
        label="paper launch postflight",
    )

    artifacts = postflight.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise MemoryProbeError(f"{root}: postflight artifacts are missing")
    for key, path in (
        ("gpu_telemetry_summary", telemetry_path),
        ("input_rehash", input_rehash_path),
        ("checkpoint", checkpoint_path),
        ("gpu_environment", gpu_environment_path),
        ("gpu_telemetry", gpu_telemetry_path),
        ("native_info_log", info_log_path),
        ("train_console_log", console_log_path),
        ("scorer_init_audit", scorer_audit_path),
    ):
        _require_file_record(artifacts.get(key), path, label=f"postflight {key}")
    if postflight.get("gpu_telemetry_summary") != telemetry:
        raise MemoryProbeError(f"{root}: embedded GPU telemetry summary drifted")
    if postflight.get("gpu_environment") != gpu_environment:
        raise MemoryProbeError(f"{root}: embedded GPU environment drifted")
    if postflight.get("input_rehash") != persisted_rehash:
        raise MemoryProbeError(f"{root}: embedded input rehash drifted")
    if persisted_rehash.get("status") != "passed":
        raise MemoryProbeError(f"{root}: persisted input rehash did not pass")
    if launch.get("gpu_environment") != gpu_environment:
        raise MemoryProbeError(f"{root}: launch GPU environment drifted")
    if launch.get("gpu_telemetry_summary") != telemetry:
        raise MemoryProbeError(f"{root}: launch GPU telemetry summary drifted")
    if launch.get("returncode") != 0:
        raise MemoryProbeError(f"{root}: launch did not persist returncode=0")

    try:
        fresh_rehash = paper_launcher._rehash_inputs(launch)
        _compare_rehash_replay(persisted_rehash, fresh_rehash, root=root)
        fresh_telemetry = paper_launcher._summarize_nvidia_csv(gpu_telemetry_path)
        _compare_telemetry_replay(telemetry, fresh_telemetry, root=root)
        paper_launcher._validate_gpu_telemetry_contract(
            gpu_environment, telemetry
        )
        fresh_numerical = paper_launcher._training_numerical_status(
            info_log_path, console_log_path
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise MemoryProbeError(
            f"{root}: postflight evidence replay failed: {error}"
        ) from error
    if postflight.get("numerical_status") != fresh_numerical:
        raise MemoryProbeError(
            f"{root}: persisted numerical status differs from log replay"
        )

    row = sequence.get("row")
    launch_row = launch.get("row")
    if not isinstance(row, Mapping) or row != launch_row:
        raise MemoryProbeError(f"{root}: sequence/launch row contracts differ")
    row_id = str(row.get("row_id", ""))
    if expected_row_id is not None and row_id != expected_row_id:
        raise MemoryProbeError(
            f"{root}: expected row {expected_row_id!r}, got {row_id!r}"
        )
    canonical = paper_launcher.ROW_BY_ID.get(row_id)
    allowed = canonical is not None and (
        canonical.table == "D" or canonical.row_id in MATCHED_TABLE_B_ROW_IDS
    )
    if not allowed or dict(row) != asdict(canonical):
        raise MemoryProbeError(
            f"{root}: probe is not a canonical Table-D or matched Table-B row"
        )
    run_id = str(sequence.get("run_id", ""))
    if (
        launch.get("run_id") != run_id
        or postflight.get("run_id") != run_id
        or run_id != f"{row_id}:{TABLE_B_MEMORY_SEED}"
    ):
        raise MemoryProbeError(f"{root}: run identity differs across artifacts")
    if Path(str(sequence.get("output_dir", ""))).resolve(strict=False) != root:
        raise MemoryProbeError(f"{root}: sequence output root differs")
    if Path(str(launch.get("output_dir", ""))).resolve(strict=False) != root:
        raise MemoryProbeError(f"{root}: launch output root differs")

    planned = sequence.get("phases")
    completed = sequence.get("completed_phases")
    if (
        not isinstance(planned, list)
        or len(planned) != 1
        or not isinstance(completed, list)
        or len(completed) != 1
        or not isinstance(planned[0], Mapping)
        or not isinstance(completed[0], Mapping)
    ):
        raise MemoryProbeError(f"{root}: memory probe must have exactly one phase")
    phase = launch.get("phase")
    if (
        not isinstance(phase, Mapping)
        or completed[0].get("phase_id") != "joint"
        or completed[0].get("status") != "completed"
        or postflight.get("phase_id") != "joint"
    ):
        raise MemoryProbeError(f"{root}: probe phase identity is inconsistent")
    _require_file_record(
        completed[0].get("checkpoint"),
        checkpoint_path,
        label="sequence checkpoint",
    )

    budget = sequence.get("equal_budget_contract")
    metadata = postflight.get("checkpoint_metadata")
    if not isinstance(budget, Mapping) or not isinstance(metadata, Mapping):
        raise MemoryProbeError(f"{root}: runtime/checkpoint contracts are incomplete")
    batch_size = _exact_int(
        budget.get("batch_size"), label=f"{root} budget batch_size", minimum=1
    )
    updates = _exact_int(
        budget.get("optimizer_updates"),
        label=f"{root} budget optimizer_updates",
        minimum=1,
    )
    expected_budget = {
        "batch_size": batch_size,
        "optimizer_updates": updates,
        "s3_probe_updates_excluded": 0,
        "contributing_phase_updates": {"joint": updates},
    }
    if dict(budget) != expected_budget:
        raise MemoryProbeError(f"{root}: sequence equal-budget contract drifted")
    runtime_contract = _runtime_from_paper_launch(
        launch, root=root, canonical=canonical
    )
    canonical_phases = paper_launcher._phases(runtime_contract, canonical)
    if len(canonical_phases) != 1:
        raise MemoryProbeError(f"{root}: memory probe row is not single-phase")
    phase_contract = canonical_phases[0]
    if dict(phase) != asdict(phase_contract):
        raise MemoryProbeError(f"{root}: launch phase differs from current canonical row")
    if planned[0].get("phase") != dict(phase):
        raise MemoryProbeError(f"{root}: sequence and launch phase contracts differ")
    diagnostic_interval = phase_contract.diagnostic_interval
    expected_diagnostic = _table_d_expected_diagnostic_interval(
        row_id,
        updates=updates,
        minimum_soak_updates=minimum_soak_updates,
    )
    expected_values = {
        "runtime batch_size": (runtime_contract.batch_size, batch_size),
        "runtime total updates": (runtime_contract.total_train_iters, updates),
        "runtime checkpoint interval": (
            runtime_contract.iter_checkpoint_interval,
            updates,
        ),
        "runtime num_workers": (runtime_contract.num_workers, 2),
        "phase updates": (phase_contract.updates, updates),
        "sequence seed": (sequence.get("seed"), TABLE_B_MEMORY_SEED),
        "launch seed": (launch.get("seed"), TABLE_B_MEMORY_SEED),
    }
    for label, (observed, expected) in expected_values.items():
        if observed != expected:
            raise MemoryProbeError(
                f"{root}: {label} mismatch: expected {expected!r}, got {observed!r}"
            )
    if expected_diagnostic is not None and diagnostic_interval != expected_diagnostic:
        raise MemoryProbeError(
            f"{root}: {row_id} {updates}-update probe requires diagnostic interval "
            f"{expected_diagnostic}, got {diagnostic_interval}"
        )
    try:
        current_manifest = paper_launcher._phase_manifest(
            runtime_contract,
            canonical,
            TABLE_B_MEMORY_SEED,
            phase_contract,
            root,
            paper_launcher.token_launcher.HashCache(),
            rank_checkpoint=None,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise MemoryProbeError(
            f"{root}: current canonical phase reconstruction failed: {error}"
        ) from error
    immutable_keys = (
        "schema",
        "run_id",
        "row",
        "seed",
        "phase",
        "output_dir",
        "command",
        "command_shell",
        "runtime",
        "fixed_contract",
        "generated_dependency",
        "inputs",
    )
    for key in immutable_keys:
        expected = current_manifest.get(key)
        if launch.get(key) != expected:
            raise MemoryProbeError(
                f"{root}: launch {key} differs from current canonical reconstruction"
            )
        if planned[0].get(key) != expected:
            raise MemoryProbeError(
                f"{root}: sequence phase {key} differs from current canonical reconstruction"
            )
    if launch.get("command_shell") != shlex.join(list(launch.get("command", []))):
        raise MemoryProbeError(f"{root}: command_shell does not encode command")

    try:
        observed_metadata = paper_launcher._inspect_checkpoint_safely(
            runtime_contract, checkpoint_path
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise MemoryProbeError(
            f"{root}: checkpoint metadata replay failed: {error}"
        ) from error
    if observed_metadata != metadata:
        raise MemoryProbeError(
            f"{root}: persisted checkpoint metadata differs from checkpoint replay"
        )
    expected_scorer_audit = {
        "schema": "stage_b_v15_scorer_init/v1",
        "status": "applied",
        "source_sha256": _single_input_record(
            launch, role="scorer_warmstart"
        ).get("sha256"),
        "loaded_num_layers": 3,
    }
    for key, expected in expected_scorer_audit.items():
        if scorer_audit.get(key) != expected:
            raise MemoryProbeError(
                f"{root}: scorer initialization audit mismatch for {key}"
            )
    if Path(str(scorer_audit.get("resolved_source_path", ""))).resolve(
        strict=False
    ) != runtime_contract.scorer_warmstart:
        raise MemoryProbeError(f"{root}: scorer audit source path drifted")
    try:
        paper_launcher._validate_checkpoint_metadata(
            observed_metadata,
            runtime=runtime_contract,
            row=canonical,
            seed=TABLE_B_MEMORY_SEED,
            phase=phase_contract,
            output_dir=root,
            pretrain_path=runtime_contract.stage_a_init,
            scorer_audit=scorer_audit,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise MemoryProbeError(
            f"{root}: current checkpoint contract failed: {error}"
        ) from error
    ancestry = postflight.get("model_state_ancestry")
    expected_ancestry = {
        "pretrain_path": str(runtime_contract.stage_a_init.resolve(strict=False)),
        "pretrain_sha256": _single_input_record(
            launch, role="stage_a_initializer"
        ).get("sha256"),
        "pretrain_manifest_role": "stage_a_initializer",
        "pretrain_mode": "model_state_only_no_optimizer_resume",
        "checkpoint_resume_argument": None,
        "scorer_warmstart_applied": True,
        "generated_dependency": None,
    }
    if ancestry != expected_ancestry:
        raise MemoryProbeError(f"{root}: model-state ancestry contract drifted")

    config_path = (REPO_ROOT / phase_contract.config).resolve(strict=True)
    closure = _current_paper_training_closure(config_path)
    _verify_current_input_closure(launch, closure)
    try:
        final_rehash = paper_launcher._rehash_inputs(launch)
        _compare_rehash_replay(persisted_rehash, final_rehash, root=root)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise MemoryProbeError(
            f"{root}: final launch-input replay failed: {error}"
        ) from error
    _require_file_record(
        launch.get("postflight_artifact"),
        postflight_path,
        label="paper launch postflight final replay",
    )
    _require_file_record(
        artifacts.get("checkpoint"),
        checkpoint_path,
        label="postflight checkpoint final replay",
    )


def _probe_declared_row_ids(path: str | Path, *, label: str) -> frozenset[str]:
    root = Path(path).expanduser().resolve(strict=True)
    sequence = _read_json(root / "sequence_manifest.json", label=f"{label} sequence")
    values: set[str] = set()
    row = sequence.get("row")
    if isinstance(row, Mapping):
        values.add(str(row.get("row_id", "")))
    launch_path = root / "launch_manifest.json"
    if launch_path.is_file():
        launch = _read_json(launch_path, label=f"{label} launch")
        launch_row = launch.get("row")
        if isinstance(launch_row, Mapping):
            values.add(str(launch_row.get("row_id", "")))
    return frozenset(values)


def _matched_table_b_contract_requested(
    probes: Mapping[str, str | Path], *, expected_row_id: str | None
) -> bool:
    if expected_row_id in MATCHED_TABLE_B_ROW_IDS:
        return True
    return any(
        bool(
            _probe_declared_row_ids(path, label=name)
            & MATCHED_TABLE_B_ROW_IDS
        )
        for name, path in sorted(probes.items())
    )


def _replay_matched_table_b_probes(
    probes: Mapping[str, str | Path],
    *,
    expected_row_id: str | None,
    minimum_soak_updates: int,
) -> None:
    for _, path in sorted(probes.items()):
        _verify_paper_probe_current(
            Path(path).expanduser().resolve(strict=True),
            expected_row_id=expected_row_id,
            minimum_soak_updates=minimum_soak_updates,
        )
    if expected_row_id != TABLE_B_MEMORY_ROW_ID:
        raise MemoryProbeError(
            "formal matched Table-B memory seals require "
            f"--expected-row-id {TABLE_B_MEMORY_ROW_ID}; D2m and implicit row "
            "selection are ineligible"
        )
    if minimum_soak_updates != TABLE_B_MEMORY_SOAK_UPDATES:
        raise MemoryProbeError(
            "formal D3m memory seals require exactly "
            f"minimum_soak_updates={TABLE_B_MEMORY_SOAK_UPDATES}"
        )


def verify_existing_seal(
    output: Path,
    probes: Mapping[str, str | Path],
    *,
    selected: str,
    minimum_headroom_mib: float,
    minimum_soak_updates: int,
    expected_row_id: str | None,
) -> Mapping[str, Any]:
    seal = build_seal(
        probes,
        selected=selected,
        minimum_headroom_mib=minimum_headroom_mib,
        minimum_soak_updates=minimum_soak_updates,
        expected_row_id=expected_row_id,
    )
    existing = _read_json(output, label="existing seal")
    if existing != seal:
        raise MemoryProbeError("existing seal differs from current probe evidence")
    for name, path in sorted(probes.items()):
        root = Path(path).expanduser().resolve(strict=True)
        sequence = _read_json(root / "sequence_manifest.json", label=f"{name} sequence")
        row = sequence.get("row")
        row_id = str(row.get("row_id", "")) if isinstance(row, Mapping) else ""
        if row_id in TABLE_D_ROW_IDS or expected_row_id in TABLE_D_ROW_IDS:
            _verify_paper_probe_current(
                root,
                expected_row_id=expected_row_id,
                minimum_soak_updates=minimum_soak_updates,
            )
    return seal


def inspect_probe(
    root: str | Path, *, expected_row_id: str | None = None
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve(strict=True)
    sequence_path = root / "sequence_manifest.json"
    postflight_path = root / "postflight.json"
    telemetry_path = root / "gpu_telemetry_summary.json"
    sequence = _read_json(sequence_path, label="sequence manifest")
    postflight = _read_json(postflight_path, label="postflight")
    telemetry = _read_json(telemetry_path, label="GPU telemetry summary")
    if sequence.get("status") != "completed":
        raise MemoryProbeError(f"{root}: sequence status is not completed")
    if postflight.get("status") != "passed":
        raise MemoryProbeError(f"{root}: postflight status is not passed")
    budget = sequence.get("equal_budget_contract")
    if not isinstance(budget, Mapping):
        raise MemoryProbeError(f"{root}: sequence has no equal-budget contract")
    batch_size = int(budget.get("batch_size", 0))
    updates = int(budget.get("optimizer_updates", 0))
    if batch_size <= 0 or updates <= 0:
        raise MemoryProbeError(f"{root}: invalid batch size or update count")
    devices = telemetry.get("devices")
    if not isinstance(devices, list) or len(devices) != 1:
        raise MemoryProbeError(f"{root}: exactly one telemetry device is required")
    device = devices[0]
    if not isinstance(device, Mapping):
        raise MemoryProbeError(f"{root}: telemetry device is not an object")
    required = (
        "uuid",
        "name",
        "driver_version",
        "total_memory_mib",
        "peak_used_memory_mib",
        "min_free_memory_mib",
        "sample_count",
    )
    missing = [key for key in required if key not in device]
    if missing:
        raise MemoryProbeError(f"{root}: missing telemetry fields {missing}")
    total = float(device["total_memory_mib"])
    peak = float(device["peak_used_memory_mib"])
    headroom = float(device["min_free_memory_mib"])
    if min(total, peak, headroom) < 0.0 or peak > total or headroom > total:
        raise MemoryProbeError(f"{root}: invalid telemetry memory values")
    validation = postflight.get("numerical_status")
    if not isinstance(validation, Mapping) or validation.get("status") != "passed":
        raise MemoryProbeError(f"{root}: training-log validation did not pass")
    if validation.get("loss_values_all_finite") is not True:
        raise MemoryProbeError(f"{root}: losses were not proven finite")
    if float(validation.get("max_amp_step_skipped", 1.0)) != 0.0:
        raise MemoryProbeError(f"{root}: AMP skipped an optimizer step")
    experiment: dict[str, Any] | None = None
    if expected_row_id is not None:
        row = sequence.get("row")
        if not isinstance(row, Mapping):
            raise MemoryProbeError(f"{root}: sequence has no row contract")
        row_id = str(row.get("row_id", ""))
        run_id = str(sequence.get("run_id", ""))
        if row_id != expected_row_id or not run_id.startswith(f"{expected_row_id}:"):
            raise MemoryProbeError(
                f"{root}: expected row {expected_row_id!r}, got "
                f"row_id={row_id!r}, run_id={run_id!r}"
            )
        experiment = {
            "run_id": run_id,
            "row_id": row_id,
            "table": row.get("table"),
            "score_ownership": row.get("score_ownership"),
            "objective_fidelity": row.get("objective_fidelity"),
        }
    result = {
        "root": str(root),
        "batch_size": batch_size,
        "optimizer_updates": updates,
        "gpu": {
            key: device[key]
            for key in required
        },
        "artifacts": {
            "sequence_manifest": _file_record(sequence_path),
            "postflight": _file_record(postflight_path),
            "gpu_telemetry_summary": _file_record(telemetry_path),
        },
        "finite_losses": True,
        "amp_skipped_steps": 0,
    }
    if experiment is not None:
        result["experiment"] = experiment
    return result


def build_seal(
    probes: Mapping[str, str | Path],
    *,
    selected: str,
    minimum_headroom_mib: float = 1024.0,
    minimum_soak_updates: int = 50,
    expected_row_id: str | None = None,
) -> dict[str, Any]:
    if not probes:
        raise MemoryProbeError("at least one probe is required")
    if selected not in probes:
        raise MemoryProbeError(f"selected probe {selected!r} is not declared")
    if minimum_headroom_mib < 0.0 or minimum_soak_updates <= 0:
        raise MemoryProbeError("headroom and soak requirements are invalid")
    matched_table_b = _matched_table_b_contract_requested(
        probes, expected_row_id=expected_row_id
    )
    if matched_table_b:
        _replay_matched_table_b_probes(
            probes,
            expected_row_id=expected_row_id,
            minimum_soak_updates=minimum_soak_updates,
        )
        if minimum_headroom_mib < TABLE_B_MEMORY_MIN_HEADROOM_MIB:
            raise MemoryProbeError(
                "formal D3m memory seals cannot relax minimum headroom below "
                f"{TABLE_B_MEMORY_MIN_HEADROOM_MIB} MiB"
            )
    rows = {
        name: inspect_probe(path, expected_row_id=expected_row_id)
        for name, path in sorted(probes.items())
    }
    identities = {
        (
            row["gpu"]["uuid"],
            row["gpu"]["name"],
            row["gpu"]["driver_version"],
            float(row["gpu"]["total_memory_mib"]),
        )
        for row in rows.values()
    }
    if len(identities) != 1:
        raise MemoryProbeError("probe GPU identity or total memory drifted")
    for row in rows.values():
        row["headroom_pass"] = (
            float(row["gpu"]["min_free_memory_mib"]) >= minimum_headroom_mib
        )
        row["soak_pass"] = row["optimizer_updates"] >= minimum_soak_updates
    winner = rows[selected]
    if not winner["headroom_pass"]:
        raise MemoryProbeError(
            f"selected probe has {winner['gpu']['min_free_memory_mib']} MiB headroom; "
            f"requires at least {minimum_headroom_mib} MiB"
        )
    if not winner["soak_pass"]:
        raise MemoryProbeError(
            f"selected probe has {winner['optimizer_updates']} updates; "
            f"requires at least {minimum_soak_updates}"
        )
    if matched_table_b:
        if winner["batch_size"] != TABLE_B_MEMORY_BATCH_SIZE:
            raise MemoryProbeError(
                "formal D3m memory soak must select batch size "
                f"{TABLE_B_MEMORY_BATCH_SIZE}, got {winner['batch_size']}"
            )
        if winner["optimizer_updates"] != TABLE_B_MEMORY_SOAK_UPDATES:
            raise MemoryProbeError(
                "formal D3m memory soak must contain exactly "
                f"{TABLE_B_MEMORY_SOAK_UPDATES} optimizer updates, got "
                f"{winner['optimizer_updates']}"
            )
    result = {
        "schema": SCHEMA,
        "status": "sealed",
        "selection": {
            "probe": selected,
            "batch_size": winner["batch_size"],
            "minimum_headroom_mib": float(minimum_headroom_mib),
            "minimum_soak_updates": int(minimum_soak_updates),
            "observed_minimum_free_memory_mib": float(
                winner["gpu"]["min_free_memory_mib"]
            ),
        },
        "gpu_identity": {
            "uuid": winner["gpu"]["uuid"],
            "name": winner["gpu"]["name"],
            "driver_version": winner["gpu"]["driver_version"],
            "total_memory_mib": float(winner["gpu"]["total_memory_mib"]),
        },
        "probes": rows,
        "policy": {
            "hard_terminated_or_incomplete_jobs_are_ineligible": True,
            "oom_is_not_inferred_from_missing_processes_or_log_text": True,
            "selection_requires_completed_sequence_and_passed_postflight": True,
        },
    }
    if expected_row_id is not None:
        result["experiment_contract"] = {
            "expected_row_id": expected_row_id,
            "all_probes_match": True,
        }
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _parse_probe(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("probe must use NAME=RUN_ROOT")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("probe must use non-empty NAME=RUN_ROOT")
    return name.strip(), Path(raw_path).expanduser()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="append", type=_parse_probe, required=True)
    parser.add_argument("--selected", required=True)
    parser.add_argument("--minimum-headroom-mib", type=float, default=1024.0)
    parser.add_argument("--minimum-soak-updates", type=int, default=50)
    parser.add_argument(
        "--expected-row-id",
        help="require every probe sequence manifest to belong to this exact row",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    probes = dict(args.probe)
    if len(probes) != len(args.probe):
        print("ERROR: duplicate probe name", file=sys.stderr)
        return 2
    try:
        if args.verify_only:
            seal = verify_existing_seal(
                args.output,
                probes,
                selected=args.selected,
                minimum_headroom_mib=args.minimum_headroom_mib,
                minimum_soak_updates=args.minimum_soak_updates,
                expected_row_id=args.expected_row_id,
            )
        else:
            seal = build_seal(
                probes,
                selected=args.selected,
                minimum_headroom_mib=args.minimum_headroom_mib,
                minimum_soak_updates=args.minimum_soak_updates,
                expected_row_id=args.expected_row_id,
            )
            if args.output.exists():
                raise MemoryProbeError(f"refuse to overwrite existing seal: {args.output}")
            _atomic_json(args.output, seal)
    except (MemoryProbeError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "verified" if args.verify_only else "sealed", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
