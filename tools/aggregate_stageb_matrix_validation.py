#!/usr/bin/env python3
"""Aggregate validation-only Stage-B ablation matrices with record replay.

The input specification is deliberately small and names only completed
``matrix_validation`` evaluation roots::

    {
      "schema": "pivot.stageb.matrix_validation_input/v3",
      "expected_train_seeds": [17, 42, 73],
      "evaluation_queue_dir": ".../table_c_matrix_validation_v1",
      "evaluation_queue_id": "...",
      "evaluation_plan_sha256": "...",
      "reference_experiment": "L0",
      "experiments": [
        {
          "id": "L0",
          "label": "no token supervision",
          "evaluation_roots": {
            "17": ".../L0/seed17",
            "42": ".../L0/seed42",
            "73": ".../L0/seed73"
          }
        }
      ]
    }

The report can contain only the three locked Ref validation splits and the
1,570-row proposal-covered calibration surface.  Ref test and strict-TN
artifacts are rejected, not ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_stageb_matrix_validation_queue as evaluation_queue  # noqa: E402
from tools import run_stageb_paper_evaluations as evaluator  # noqa: E402
from tools.compare_stageb_fpr95_records import (  # noqa: E402
    RecordComparisonError,
    exact_binary_auroc,
    exact_fpr95,
    load_manifest,
    load_tn_records,
)
from tools.stageb_ref_split_contract import (  # noqa: E402
    REF_SPLIT_CONTRACT,
    REF_SPLIT_MANIFEST_FILES,
)
from tools.stageb_eval_records import (  # noqa: E402
    RefRecordContractError,
    load_formal_ref_records,
)
from tools.stageb_screen_calibration import (  # noqa: E402
    DERIVATION_ALGORITHM,
    EVAL_SPLIT as CALIBRATION_SPLIT,
    SCHEMA as CALIBRATION_BINDING_SCHEMA,
    ScreenCalibrationError,
    load_binding,
)


INPUT_SCHEMA = evaluation_queue.AGGREGATION_INPUT_SCHEMA
REPORT_SCHEMA = "pivot.stageb.matrix_validation_report/v2"
PROFILE = evaluator.MATRIX_PROFILE
REF_VALIDATION_SPLITS = (
    "refcoco_val",
    "refcocop_val",
    "refcocog_val",
)
REF_VALIDATION_CONTRACT = {
    split: dict(REF_SPLIT_CONTRACT[split]) for split in REF_VALIDATION_SPLITS
}
CALIBRATION_ROWS = 1570
FORMAL_TRAIN_SEEDS = (17, 42, 73)
FORMAL_EXPERIMENT_IDS = tuple(f"L{index}" for index in range(11))
FORMAL_REFERENCE_EXPERIMENT: str | None = "L0"
FORMAL_EVALUATION_QUEUE_DIR = evaluation_queue.DEFAULT_QUEUE_DIR
# Canonical validation queue created from the immutable Table-C execution
# snapshot.  Aggregation fails closed on any queue or plan mismatch.
FORMAL_EVALUATION_QUEUE_ID: str | None = "68360aac-cf82-4a9e-a357-04a0c5ccd3b3"
FORMAL_EVALUATION_PLAN_SHA256: str | None = (
    "b238f24b0090323c6a52294592c276b7706662ff9c4b965e84dc775034632078"
)
DEFAULT_INPUT_SPEC = evaluation_queue.DEFAULT_AGGREGATION_INPUT_SPEC
FORMAL_TRAINING_OUTPUT_ROOT = (
    REPO_ROOT / "outputs/paper_cvpr_v1/token_ablation_frozen_v2"
)
FORMAL_TRAINING_QUEUE_BINDINGS = {
    "completed_l0_l4_seed17": {
        "queue_id": "3e5a961a-f2da-45ba-8e44-94740f4baee9",
        "plan_sha256": (
            "63619de10c9e41d2ecc5177242b4b3bbf175d57c3c9cdcd7013b1185a53e6cde"
        ),
        "run_ids": tuple(f"L{index}:17" for index in range(5)),
    },
    "remaining_table_c": {
        "queue_id": "ffcc3e46-ca1d-45d0-9fbd-22e5db14ac9f",
        "plan_sha256": (
            "b4b8ef280fcbd67dbf82fc59d6c90f63c9c3573976b8950c06f1e84dbb31c2cc"
        ),
        "run_ids": (
            *(f"L{index}:17" for index in range(5, 11)),
            *(f"L{index}:{seed}" for seed in (42, 73) for index in range(11)),
        ),
    },
}
CALIBRATION_METRICS = (
    "fpr95",
    "positive_q05",
    "auroc",
    "positive_over_negative_pair_win_rate",
)
REF_METRICS = ("acc50", "mean_iou")
DEFAULT_BOOTSTRAP_ITERATIONS = 5000
DEFAULT_BOOTSTRAP_SEED = 20260718
DEFAULT_CONFIDENCE = 0.95
FORMAL_BOOTSTRAP_ITERATIONS = 5000
FORMAL_BOOTSTRAP_SEED = 20260718
FORMAL_CONFIDENCE = 0.95
ACCESS_LABEL = (
    "matrix validation/calibration only; no Ref test or strict-TN access"
)
_ID_RE = re.compile(r"[A-Za-z0-9_.-]+")
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_PATH_RE = re.compile(
    r"(?:testa|testb|refcocog_test|strict2031|strict1607|ref8_strict)",
    re.IGNORECASE,
)
_FORBIDDEN_COMMAND_VALUES = {
    "refcoco_testa",
    "refcoco_testb",
    "refcocop_testa",
    "refcocop_testb",
    "refcocog_test",
    "strict2031",
    "strict1607",
}
AGGREGATION_SOURCE_PRUNED_EDGES = (
    (
        "tools/run_stageb_paper_evaluations.py",
        "tools/run_stageb_paper_ablation_matrices.py",
    ),
    (
        "tools/run_stageb_paper_evaluations.py",
        "tools/stageb_headline_release_contract.py",
    ),
    (
        "tools/run_stageb_paper_evaluations.py",
        "tools/stageb_profile_dependency_audit.py",
    ),
    (
        "tools/run_stageb_matrix_validation_queue.py",
        "tools/stageb_profile_dependency_audit.py",
    ),
    (
        "tools/run_stageb_token_ablation_matrix.py",
        "tools/run_stageb_paper_ablation_matrices.py",
    ),
)
AGGREGATION_EXPECTED_SOURCE_PATHS = (
    "tools/aggregate_stageb_matrix_validation.py",
    "tools/compare_stageb_fpr95_records.py",
    "tools/recover_stageb_serial_matrix_pretraining_failure.py",
    "tools/run_stageb_matrix_validation_queue.py",
    "tools/run_stageb_paper_evaluations.py",
    "tools/run_stageb_serial_matrix_queue.py",
    "tools/run_stageb_token_ablation_matrix.py",
    "tools/stageb_dependency_audit.py",
    "tools/stageb_eval_records.py",
    "tools/stageb_evaluation_source_contracts.py",
    "tools/stageb_profile_dependency_audit.py",
    "tools/stageb_ref_split_contract.py",
    "tools/stageb_screen_calibration.py",
)


class MatrixValidationError(ValueError):
    """Raised when validation-only matrix evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class RefSurface:
    split: str
    rows: tuple[Mapping[str, Any], ...]
    identities: tuple[tuple[Any, ...], ...]
    image_ids: np.ndarray
    correct50: np.ndarray
    top1_iou: np.ndarray
    metrics: Mapping[str, float]
    manifest_sha256: str
    evaluation_manifest: Path
    records: Mapping[str, Any]


@dataclass(frozen=True)
class CalibrationSurface:
    rows: tuple[Mapping[str, Any], ...]
    identities: tuple[tuple[Any, ...], ...]
    image_ids: np.ndarray
    positive: np.ndarray
    negative: np.ndarray
    metrics: Mapping[str, float]
    source_manifest_sha256: str
    source_audit_sha256: str
    derived_manifest_sha256: str
    row_mapping_sha256: str
    records: Mapping[str, Any]


@dataclass(frozen=True)
class LoadedRun:
    experiment_id: str
    seed: int
    root: Path
    evaluation_id: str
    training_run_id: str
    training_run_root: Path
    checkpoint: Path
    checkpoint_sha256: str
    checkpoint_run_id: str
    ref: Mapping[str, RefSurface]
    calibration: CalibrationSurface
    code_fingerprint: Mapping[str, Mapping[str, Any]]
    data_fingerprint: Mapping[str, Mapping[str, Any]]
    runtime_fingerprint: Mapping[str, Any]
    surface_fingerprint: Mapping[str, Any]
    evidence: Mapping[str, Any]
    sealed_files: tuple[Mapping[str, Any], ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _object_without_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MatrixValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except MatrixValidationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MatrixValidationError(f"{label} is not readable JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise MatrixValidationError(f"{label} must be a JSON object")
    return value


def _read_jsonl(path: Path, *, label: str) -> tuple[Mapping[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise MatrixValidationError(f"{label} is not readable JSONL: {exc}") from exc
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise MatrixValidationError(f"{label}:{line_number}: blank JSONL row")
        try:
            value = json.loads(line, object_pairs_hook=_object_without_duplicate_keys)
        except MatrixValidationError:
            raise
        except json.JSONDecodeError as exc:
            raise MatrixValidationError(
                f"{label}:{line_number}: invalid JSON"
            ) from exc
        if not isinstance(value, Mapping):
            raise MatrixValidationError(f"{label}:{line_number}: row is not an object")
        rows.append(value)
    if not rows:
        raise MatrixValidationError(f"{label} is empty")
    return tuple(rows)


def _resolve_path(
    value: Any, *, base: Path, label: str, directory: bool
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise MatrixValidationError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    try:
        path = path.resolve(strict=True)
    except (OSError, FileNotFoundError) as exc:
        raise MatrixValidationError(f"{label} does not exist: {path}") from exc
    if path.is_dir() is not directory:
        kind = "directory" if directory else "file"
        raise MatrixValidationError(f"{label} is not a {kind}: {path}")
    return path


def _exact_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise MatrixValidationError(
            f"{label} must be an exact integer >= {minimum}"
        )
    return value


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise MatrixValidationError(f"{label} must be finite numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MatrixValidationError(f"{label} must be finite numeric") from exc
    if not math.isfinite(result):
        raise MatrixValidationError(f"{label} must be finite numeric")
    return result


def _same_float(left: Any, right: Any, *, label: str) -> None:
    a = _finite(left, label=f"{label}.left")
    b = _finite(right, label=f"{label}.right")
    if not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12):
        raise MatrixValidationError(f"{label} mismatch: {a} != {b}")


def _verify_file_record(
    record: Any,
    *,
    label: str,
    cache: evaluator.HashCache,
    expected_path: Path | None = None,
) -> Path:
    if not isinstance(record, Mapping):
        raise MatrixValidationError(f"{label} has no file record")
    try:
        path = evaluator._verify_declared_file(record, label=label, cache=cache)
    except (evaluator.PaperEvaluationError, OSError, ValueError) as exc:
        raise MatrixValidationError(str(exc)) from exc
    if expected_path is not None and path != expected_path.resolve(strict=True):
        raise MatrixValidationError(f"{label} path is not canonical")
    return path


def _compact_file_record(path: Path, cache: evaluator.HashCache) -> dict[str, Any]:
    path = path.resolve(strict=True)
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": cache.digest(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _snapshot_files(
    paths: Iterable[Path], cache: evaluator.HashCache
) -> tuple[Mapping[str, Any], ...]:
    unique = {path.expanduser().resolve(strict=True) for path in paths}
    return tuple(
        _compact_file_record(path, cache) for path in sorted(unique, key=str)
    )


def _verify_snapshot_files(
    records: Iterable[Mapping[str, Any]], *, label: str
) -> None:
    cache = evaluator.HashCache()
    for index, expected in enumerate(records):
        if not isinstance(expected, Mapping):
            raise MatrixValidationError(f"{label} snapshot {index} is invalid")
        path = Path(str(expected.get("path", "")))
        try:
            observed = _compact_file_record(path, cache)
        except (OSError, FileNotFoundError) as exc:
            raise MatrixValidationError(
                f"{label} changed during aggregation: {path}"
            ) from exc
        if observed != dict(expected):
            raise MatrixValidationError(
                f"{label} changed during aggregation: {path}"
            )


def _verify_content_record(
    record: Any,
    *,
    label: str,
    cache: evaluator.HashCache,
) -> Path:
    """Verify builder-style records that intentionally omit mtime/roles."""

    if not isinstance(record, Mapping):
        raise MatrixValidationError(f"{label} has no file record")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise MatrixValidationError(f"{label} has no path")
    try:
        path = Path(raw_path).expanduser().resolve(strict=True)
    except (OSError, FileNotFoundError) as exc:
        raise MatrixValidationError(f"{label} path is missing") from exc
    if not path.is_file():
        raise MatrixValidationError(f"{label} is not a file")
    sha256 = str(record.get("sha256", "")).lower()
    if _SHA_RE.fullmatch(sha256) is None or cache.digest(path) != sha256:
        raise MatrixValidationError(f"{label} SHA-256 mismatch")
    if _exact_int(record.get("size_bytes"), label=f"{label}.size_bytes") != int(
        path.stat().st_size
    ):
        raise MatrixValidationError(f"{label} size mismatch")
    if "mtime_ns" in record and _exact_int(
        record.get("mtime_ns"), label=f"{label}.mtime_ns"
    ) != int(path.stat().st_mtime_ns):
        raise MatrixValidationError(f"{label} mtime mismatch")
    return path


def _input_role_fingerprint(
    launch: Mapping[str, Any], role: str
) -> dict[str, dict[str, Any]]:
    inputs = launch.get("inputs")
    records = inputs.get("records") if isinstance(inputs, Mapping) else None
    if not isinstance(records, list) or not records:
        raise MatrixValidationError("launch has no input records")
    result: dict[str, dict[str, Any]] = {}
    seen_paths: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise MatrixValidationError(f"launch input record {index} is invalid")
        path = str(record.get("path", ""))
        roles = record.get("roles")
        if not path or not isinstance(roles, list) or any(
            not isinstance(value, str) or not value for value in roles
        ):
            raise MatrixValidationError(f"launch input record {index} is malformed")
        if path in seen_paths:
            raise MatrixValidationError(f"launch input path is duplicated: {path}")
        seen_paths.add(path)
        sha256 = str(record.get("sha256", "")).lower()
        if _SHA_RE.fullmatch(sha256) is None:
            raise MatrixValidationError(f"launch input record {index} has invalid SHA")
        if role in roles:
            result[path] = {
                "sha256": sha256,
                "size_bytes": _exact_int(
                    record.get("size_bytes"), label=f"input {index}.size_bytes"
                ),
            }
    if not result:
        raise MatrixValidationError(f"launch has no {role} inputs")
    return dict(sorted(result.items()))


def _verify_input_rehash(
    launch: Mapping[str, Any],
    postflight: Mapping[str, Any],
    *,
    root: Path,
    cache: evaluator.HashCache,
) -> Mapping[str, Any]:
    artifact = launch.get("input_rehash_artifact")
    path = _verify_file_record(
        artifact,
        label="matrix input rehash",
        cache=cache,
        expected_path=root / "input_rehash.json",
    )
    payload = _read_json(path, label="matrix input rehash")
    if (
        payload.get("schema") != evaluator.INPUT_REHASH_SCHEMA
        or payload.get("status") != "passed"
        or postflight.get("input_rehash") != payload
    ):
        raise MatrixValidationError("matrix input rehash did not pass or is detached")
    try:
        replay = evaluator._rehash_inputs(launch)
    except (evaluator.PaperEvaluationError, OSError, ValueError) as exc:
        raise MatrixValidationError(
            f"matrix input rehash replay failed: {exc}"
        ) from exc
    for key in ("schema", "status", "records"):
        if replay.get(key) != payload.get(key):
            raise MatrixValidationError(f"matrix input rehash {key} drifted")
    return payload


def _assert_validation_only_root(root: Path, launch: Mapping[str, Any]) -> None:
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if _FORBIDDEN_PATH_RE.search(relative):
            raise MatrixValidationError(
                f"Ref test/strict artifact is forbidden in matrix root: {relative}"
            )
    commands = launch.get("commands")
    if not isinstance(commands, list) or len(commands) != 1:
        raise MatrixValidationError("matrix launch must contain exactly one command")
    command = commands[0]
    tokens = command.get("command") if isinstance(command, Mapping) else None
    if not isinstance(tokens, list) or any(
        not isinstance(value, str) for value in tokens
    ):
        raise MatrixValidationError("matrix launch command is malformed")
    lowered = {value.lower() for value in tokens}
    forbidden = sorted(lowered & _FORBIDDEN_COMMAND_VALUES)
    forbidden_tokens = sorted(
        value
        for value in tokens
        if _FORBIDDEN_PATH_RE.search(value)
        or value.lower().startswith("--strict")
        or value.lower().startswith("--ref_test")
    )
    if forbidden or forbidden_tokens or "--skip_ref" in lowered:
        raise MatrixValidationError(
            "matrix launch command accesses Ref test/strict surface: "
            f"{forbidden or forbidden_tokens}"
        )


def _source_path(value: Any, *, label: str, directory: bool = False) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise MatrixValidationError(f"matrix source {label} is missing")
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, FileNotFoundError) as exc:
        raise MatrixValidationError(f"matrix source {label} is missing") from exc
    if path.is_dir() is not directory:
        expected = "directory" if directory else "file"
        raise MatrixValidationError(f"matrix source {label} is not a {expected}")
    return path


def _evaluation_source_from_launch(
    raw: Mapping[str, Any],
) -> evaluator.EvaluationSource:
    fields = evaluator.EvaluationSource.__dataclass_fields__
    unsupported = set(raw) - set(fields)
    if unsupported:
        raise MatrixValidationError(
            f"matrix source has unsupported fields: {sorted(unsupported)}"
        )
    required = {
        "kind",
        "evaluation_id",
        "config",
        "checkpoint",
        "checkpoint_sha256",
        "training_run_id",
        "training_seed",
        "training_run_root",
        "sequence_manifest",
        "training_postflight",
        "training_queue_manifest",
        "training_queue_detached_launch",
        "training_queue_detached_status",
        "training_queue_id",
        "training_queue_plan_sha256",
        "artifact_repository_root",
    }
    missing = sorted(key for key in required if raw.get(key) is None)
    if missing:
        raise MatrixValidationError(
            f"matrix source has incomplete formal training/queue evidence: {missing}"
        )
    values = dict(raw)
    for key in ("config", "checkpoint", "sequence_manifest"):
        values[key] = _source_path(raw.get(key), label=key)
    values["training_run_root"] = _source_path(
        raw.get("training_run_root"),
        label="training_run_root",
        directory=True,
    )
    values["artifact_repository_root"] = _source_path(
        raw.get("artifact_repository_root"),
        label="artifact_repository_root",
        directory=True,
    )
    for key in (
        "final_phase_manifest",
        "training_postflight",
        "selected_phase_manifest",
        "selected_training_postflight",
        "training_queue_manifest",
        "training_queue_detached_launch",
        "training_queue_detached_status",
    ):
        value = raw.get(key)
        values[key] = None if value is None else _source_path(value, label=key)
    training_data = raw.get("training_data", [])
    if not isinstance(training_data, list) or not training_data:
        raise MatrixValidationError("matrix source training_data is missing")
    values["training_data"] = tuple(
        _source_path(value, label=f"training_data[{index}]")
        for index, value in enumerate(training_data)
    )
    try:
        return evaluator.EvaluationSource(**values)
    except (TypeError, ValueError) as exc:
        raise MatrixValidationError(f"matrix source fields are invalid: {exc}") from exc


def _runtime_from_launch(raw: Any) -> evaluator.Runtime:
    expected_keys = {
        "python",
        "data_root",
        "device",
        "batch_size",
        "num_workers",
        "amp",
        "log_every",
        "eval_seed",
        "max_ref_batches",
        "max_tn_batches",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_keys:
        raise MatrixValidationError("matrix launch runtime contract is not exact")
    python = _source_path(raw.get("python"), label="runtime.python")
    if not os.access(python, os.X_OK):
        raise MatrixValidationError("matrix runtime Python is not executable")
    data_root = _source_path(
        raw.get("data_root"), label="runtime.data_root", directory=True
    )
    device = raw.get("device")
    if not isinstance(device, str) or not device:
        raise MatrixValidationError("matrix runtime device is invalid")
    batch_size = _exact_int(
        raw.get("batch_size"), label="runtime.batch_size", minimum=1
    )
    num_workers = _exact_int(raw.get("num_workers"), label="runtime.num_workers")
    log_every = _exact_int(raw.get("log_every"), label="runtime.log_every", minimum=1)
    if type(raw.get("amp")) is not bool:
        raise MatrixValidationError("matrix runtime amp must be boolean")
    if (
        raw.get("eval_seed") != evaluator.EVAL_SEED
        or raw.get("max_ref_batches") != 0
        or raw.get("max_tn_batches") != 0
    ):
        raise MatrixValidationError("matrix runtime evaluation limits drifted")
    return evaluator.Runtime(
        python=python,
        data_root=data_root,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        amp=raw["amp"],
        log_every=log_every,
    )


def _validate_locked_training_identity(
    source: evaluator.EvaluationSource,
    *,
    experiment_id: str,
    seed: int,
) -> None:
    run_id = f"{experiment_id}:{seed}"
    matches = [
        binding
        for binding in FORMAL_TRAINING_QUEUE_BINDINGS.values()
        if run_id in binding["run_ids"]
    ]
    if len(matches) != 1:
        raise MatrixValidationError(
            f"training run is outside the locked Table-C queues: {run_id}"
        )
    binding = matches[0]
    expected_root = (
        FORMAL_TRAINING_OUTPUT_ROOT / experiment_id / f"seed{seed}"
    ).resolve(strict=False)
    if (
        source.training_run_id != run_id
        or source.training_seed != seed
        or source.training_run_root != expected_root
        or source.training_queue_id != binding["queue_id"]
        or source.training_queue_plan_sha256 != binding["plan_sha256"]
    ):
        raise MatrixValidationError(
            f"training source is not the locked Table-C run/queue identity: {run_id}"
        )


def _aggregation_source_paths() -> tuple[Path, ...]:
    from tools.stageb_profile_dependency_audit import (
        ProfileDependencyAuditError,
        recursive_local_python_dependencies,
    )

    try:
        paths = recursive_local_python_dependencies(
            [Path(__file__).resolve()],
            repository_root=REPO_ROOT,
            pruned_edges=AGGREGATION_SOURCE_PRUNED_EDGES,
        )
    except ProfileDependencyAuditError as exc:
        raise MatrixValidationError(
            f"aggregation dependency closure failed: {exc}"
        ) from exc
    expected = {
        (REPO_ROOT / relative).resolve(strict=True)
        for relative in AGGREGATION_EXPECTED_SOURCE_PATHS
    }
    observed = {path.resolve(strict=True) for path in paths}
    if observed != expected:
        missing = sorted(path.relative_to(REPO_ROOT).as_posix() for path in expected - observed)
        extra = sorted(path.relative_to(REPO_ROOT).as_posix() for path in observed - expected)
        raise MatrixValidationError(
            "aggregation dependency closure differs from the exact 13-source "
            f"profile; missing={missing}, extra={extra}"
        )
    return tuple(sorted(observed, key=lambda path: path.relative_to(REPO_ROOT).as_posix()))


def _validate_evaluation_queue_binding(
    queue_dir: Path,
    *,
    spec_path: Path,
    evaluation_queue_id: str,
    evaluation_plan_sha256: str,
    experiments: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    if queue_dir != FORMAL_EVALUATION_QUEUE_DIR.resolve(strict=False):
        raise MatrixValidationError(
            "evaluation_queue_dir is not the canonical predeclared Table-C queue"
        )
    if spec_path.expanduser().resolve(strict=True) != DEFAULT_INPUT_SPEC.resolve(
        strict=False
    ):
        raise MatrixValidationError(
            "matrix aggregation spec is not the canonical predeclared spec path"
        )
    if (
        FORMAL_EVALUATION_QUEUE_ID is None
        or FORMAL_EVALUATION_PLAN_SHA256 is None
    ):
        raise MatrixValidationError(
            "formal matrix evaluation queue identity has not been sealed"
        )
    if (
        evaluation_queue_id != FORMAL_EVALUATION_QUEUE_ID
        or evaluation_plan_sha256 != FORMAL_EVALUATION_PLAN_SHA256
    ):
        raise MatrixValidationError(
            "matrix aggregation spec is not the authorized evaluation queue identity"
        )
    try:
        queue = evaluation_queue.load_queue(queue_dir)
        verification = evaluation_queue.verify_queue(queue_dir)
    except (
        evaluation_queue.MatrixQueueError,
        OSError,
        ValueError,
    ) as exc:
        raise MatrixValidationError(
            f"canonical matrix evaluation queue verification failed: {exc}"
        ) from exc
    if (
        queue.get("status") != "completed"
        or verification.get("status") != "passed"
        or verification.get("errors")
        or queue["plan"].get("queue_id") != evaluation_queue_id
        or queue.get("plan_sha256") != evaluation_plan_sha256
        or queue["plan"].get("queue_id") != FORMAL_EVALUATION_QUEUE_ID
        or queue.get("plan_sha256") != FORMAL_EVALUATION_PLAN_SHA256
        or queue["plan"].get("provenance_scope")
        != evaluation_queue.FORMAL_PROVENANCE_SCOPE
        or verification.get("queue_id") != evaluation_queue_id
        or verification.get("plan_sha256") != evaluation_plan_sha256
        or verification.get("queue_id") != FORMAL_EVALUATION_QUEUE_ID
        or verification.get("plan_sha256") != FORMAL_EVALUATION_PLAN_SHA256
        or verification.get("provenance_scope")
        != evaluation_queue.FORMAL_PROVENANCE_SCOPE
        or queue["plan"].get("profile") != PROFILE
    ):
        raise MatrixValidationError(
            "canonical matrix evaluation queue is not completed/verified exactly"
        )
    expected_ids = [
        f"{experiment_id}:{seed}"
        for seed in seeds
        for experiment_id in FORMAL_EXPERIMENT_IDS
    ]
    plan_items = queue["plan"].get("items")
    if not isinstance(plan_items, list) or [
        item.get("run_id") if isinstance(item, Mapping) else None
        for item in plan_items
    ] != expected_ids:
        raise MatrixValidationError(
            "canonical matrix evaluation queue run order is not exact"
        )
    declared_roots = {
        f"{experiment['id']}:{seed}": experiment["roots"][seed]
        for experiment in experiments
        for seed in seeds
    }
    planned_roots = {
        item["run_id"]: Path(str(item["evaluation_root"])).resolve(strict=True)
        for item in plan_items
    }
    if declared_roots != planned_roots:
        raise MatrixValidationError(
            "aggregate spec roots differ from the predeclared evaluation queue"
        )
    spec_record = queue.get("aggregation_input_spec")
    cache = evaluator.HashCache()
    observed_spec_record = _compact_file_record(spec_path, cache)
    observed_spec_content = {
        key: observed_spec_record[key]
        for key in ("path", "sha256", "size_bytes")
    }
    if (
        not isinstance(spec_record, Mapping)
        or dict(spec_record) != observed_spec_content
        or queue["plan"].get("aggregation_input_spec")
        != evaluation_queue._aggregation_input_spec_binding(spec_path)
    ):
        raise MatrixValidationError(
            "aggregate spec is not the queue-bound predeclared input"
        )
    sealed_paths = (
        queue_dir / "queue.json",
        queue_dir / "predeclared_contract.json",
        queue_dir / "training_attestation.json",
    )
    return {
        "queue_dir": str(queue_dir),
        "queue_id": evaluation_queue_id,
        "plan_sha256": evaluation_plan_sha256,
        "provenance_scope": evaluation_queue.FORMAL_PROVENANCE_SCOPE,
        "predeclared_contract_sha256": queue[
            "predeclared_contract_sha256"
        ],
        "verification_schema": verification.get("schema"),
        "verification_status": "passed",
        "verified_item_count": len(verification.get("verified_items", [])),
        "sealed_files": _snapshot_files(sealed_paths, cache),
    }


def _replay_matrix_plan_contract(
    launch: Mapping[str, Any],
    *,
    source: evaluator.EvaluationSource,
    runtime: evaluator.Runtime,
    output_root: Path,
    cache: evaluator.HashCache,
) -> dict[str, Any]:
    fixed_runtime = {
        "batch_size": 16,
        "num_workers": 4,
        "amp": True,
        "log_every": 50,
    }
    for key, expected in fixed_runtime.items():
        if getattr(runtime, key) != expected:
            raise MatrixValidationError(
                f"matrix runtime {key} must be exactly {expected!r}"
            )
    try:
        calibration = evaluator._screen_calibration_contract(cache)
        canonical_commands = evaluator._commands(
            runtime,
            source,
            output_root,
            profile=PROFILE,
            screen_contract=calibration,
        )
        entries: list[tuple[Path, str]] = [
            (source.config, "evaluation_config"),
            (source.checkpoint, "evaluation_checkpoint"),
        ]
        entries.extend(
            (path, "config_dependency")
            for path in evaluator._config_paths(
                source.config,
                repository_root=(
                    source.artifact_repository_root
                    or evaluator.ARTIFACT_REPOSITORY_ROOT
                ),
            )
        )
        entries.extend(
            (path, "evaluation_code_dependency")
            for path in evaluator._evaluation_code_paths()
        )
        entries.extend(
            (path, "source_provenance_dependency")
            for path in evaluator._evaluation_source_provenance_paths(source)
        )
        entries.extend(
            (path, "evaluation_data_input")
            for path in evaluator._data_input_paths(runtime.data_root)
        )
        entries.extend(
            (
                (
                    Path(calibration["source_manifest"]["path"]),
                    "matrix_calibration_source",
                ),
                (
                    Path(calibration["source_audit"]["path"]),
                    "matrix_calibration_audit",
                ),
            )
        )
        entries.extend((path, "training_data") for path in source.training_data)
        for path, role in (
            (source.sequence_manifest, "training_sequence_manifest"),
            (source.final_phase_manifest, "training_final_phase_manifest"),
            (source.training_postflight, "training_final_phase_postflight"),
            (source.selected_phase_manifest, "training_selected_phase_manifest"),
            (
                source.selected_training_postflight,
                "training_selected_phase_postflight",
            ),
            (source.training_queue_manifest, "training_queue_manifest"),
            (
                source.training_queue_detached_launch,
                "training_queue_detached_launch",
            ),
            (
                source.training_queue_detached_status,
                "training_queue_detached_status",
            ),
        ):
            if path is not None:
                entries.append((path, role))
        expected_inputs = {
            "algorithm": "sha256",
            "records": evaluator._merge_input_records(entries, cache),
        }
    except (evaluator.PaperEvaluationError, OSError, ValueError) as exc:
        raise MatrixValidationError(
            f"matrix canonical plan reconstruction failed: {exc}"
        ) from exc
    protocol = launch.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get(
        "screen_calibration"
    ) != calibration:
        raise MatrixValidationError("matrix calibration preflight contract drifted")
    if launch.get("commands") != canonical_commands:
        raise MatrixValidationError(
            "matrix launch command differs from the canonical matrix profile"
        )
    if launch.get("inputs") != expected_inputs:
        raise MatrixValidationError(
            "matrix launch inputs differ from the canonical formal plan"
        )
    return {
        "python": str(runtime.python),
        "data_root": str(runtime.data_root),
        "device": runtime.device,
        "batch_size": runtime.batch_size,
        "num_workers": runtime.num_workers,
        "amp": runtime.amp,
        "log_every": runtime.log_every,
        "eval_seed": evaluator.EVAL_SEED,
        "max_ref_batches": 0,
        "max_tn_batches": 0,
    }


def _validate_launch_contract(
    launch: Mapping[str, Any],
    *,
    cache: evaluator.HashCache,
    experiment_id: str,
    root: Path,
    seed: int,
) -> tuple[str, str, Path, Path, str, Mapping[str, Any]]:
    if launch.get("schema") != evaluator.SCHEMA or launch.get("status") != "completed":
        raise MatrixValidationError(
            "evaluation launch is not completed under the paper schema"
        )
    try:
        declared_root = Path(str(launch.get("output_dir", ""))).resolve(strict=True)
    except (OSError, FileNotFoundError) as exc:
        raise MatrixValidationError("evaluation launch output_dir is invalid") from exc
    if declared_root != root:
        raise MatrixValidationError("evaluation root differs from launch output_dir")
    protocol = launch.get("protocol")
    expected_protocol = {
        "profile": PROFILE,
        "ref_splits": list(REF_VALIDATION_SPLITS),
        "strict_manifests": {},
        "processes": ["validation_calibration"],
        "strict1607_skip_ref": False,
        "per_example_records": True,
        "release_policy": (
            "ablation_matrix_validation_only_no_ref_test_or_strict_access"
        ),
    }
    if not isinstance(protocol, Mapping):
        raise MatrixValidationError("evaluation launch protocol is missing")
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise MatrixValidationError(
                f"matrix protocol {key} mismatch: {protocol.get(key)!r}"
            )
    if not isinstance(protocol.get("screen_calibration"), Mapping):
        raise MatrixValidationError("matrix launch lacks calibration source contract")
    completed = launch.get("completed_phases")
    if not (
        isinstance(completed, list)
        and len(completed) == 1
        and isinstance(completed[0], Mapping)
        and completed[0].get("phase_id") == "validation_calibration"
        and completed[0].get("status") == "completed"
        and completed[0].get("returncode") == 0
    ):
        raise MatrixValidationError("matrix launch phase is not exactly completed")
    source = launch.get("source")
    if not isinstance(source, Mapping):
        raise MatrixValidationError("matrix launch source is missing")
    if source.get("kind") != "pivot_token_ablation_training_run":
        raise MatrixValidationError("matrix launch source is not a formal token run")
    if source.get("training_phase", "final") != "final" or source.get(
        "diagnostic_only", False
    ) is not False:
        raise MatrixValidationError(
            "matrix launch source is not the final training phase"
        )
    if _exact_int(source.get("training_seed"), label="training_seed") != seed:
        raise MatrixValidationError(
            "matrix launch training seed differs from input spec"
        )
    evaluation_id = str(launch.get("evaluation_id") or "")
    training_run_id = str(source.get("training_run_id") or "")
    if not evaluation_id or source.get("evaluation_id") != evaluation_id:
        raise MatrixValidationError("matrix evaluation_id is missing or inconsistent")
    if not training_run_id:
        raise MatrixValidationError("matrix training_run_id is missing")
    expected_training_run_id = f"{experiment_id}:{seed}"
    if training_run_id != expected_training_run_id:
        raise MatrixValidationError(
            "matrix training run_id does not match the declared experiment/seed: "
            f"{training_run_id!r} != {expected_training_run_id!r}"
        )
    formal_source = _evaluation_source_from_launch(source)
    artifact_repository_root = formal_source.artifact_repository_root
    assert artifact_repository_root is not None
    try:
        _, artifact_outputs_root = evaluator._attested_artifact_repository_root(
            {"repository_root": str(artifact_repository_root)},
            label="matrix launch artifact",
        )
    except evaluator.PaperEvaluationError as exc:
        raise MatrixValidationError(str(exc)) from exc
    if (
        Path(str(launch.get("artifact_repository_root", ""))).resolve(
            strict=False
        )
        != artifact_repository_root
        or Path(str(launch.get("artifact_outputs_root", ""))).resolve(
            strict=False
        )
        != artifact_outputs_root
    ):
        raise MatrixValidationError("matrix launch artifact-root binding drifted")
    _validate_locked_training_identity(
        formal_source,
        experiment_id=experiment_id,
        seed=seed,
    )
    try:
        evaluator._revalidate_matrix_source(formal_source, cache)
    except (evaluator.PaperEvaluationError, OSError, ValueError) as exc:
        raise MatrixValidationError(
            f"matrix formal training source revalidation failed: {exc}"
        ) from exc
    runtime = _runtime_from_launch(launch.get("runtime"))
    runtime_fingerprint = _replay_matrix_plan_contract(
        launch,
        source=formal_source,
        runtime=runtime,
        output_root=root,
        cache=cache,
    )
    training_root = formal_source.training_run_root
    assert training_root is not None
    checkpoint = formal_source.checkpoint
    checkpoint_sha = str(source.get("checkpoint_sha256", "")).lower()
    if _SHA_RE.fullmatch(checkpoint_sha) is None:
        raise MatrixValidationError("matrix checkpoint SHA is invalid")
    return (
        evaluation_id,
        training_run_id,
        training_root,
        checkpoint,
        checkpoint_sha,
        runtime_fingerprint,
    )


def _validate_postflight(
    launch: Mapping[str, Any],
    postflight: Mapping[str, Any],
    input_rehash: Mapping[str, Any],
) -> None:
    if (
        postflight.get("schema") != evaluator.POSTFLIGHT_SCHEMA
        or postflight.get("status") != "passed"
        or postflight.get("profile") != PROFILE
        or postflight.get("evaluation_id") != launch.get("evaluation_id")
    ):
        raise MatrixValidationError("matrix postflight is not a passed matrix profile")
    expected_contracts = (
        "ref_validation_split_set_exact",
        "ref_test_splits_not_run",
        "strict2031_not_run",
        "strict1607_not_run",
        "full_per_example_records",
        "zero_invalid_records",
        "calibration_source_to_derived_binding",
        "proposal_covered_scope_preserved",
        "single_edit_calibration_only",
        "checkpoint_consistent_across_all_rows",
    )
    contracts = postflight.get("contracts")
    if not isinstance(contracts, Mapping) or any(
        contracts.get(key) is not True for key in expected_contracts
    ):
        raise MatrixValidationError("matrix postflight contracts are incomplete")
    artifacts = postflight.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "summary",
        "ref_validation",
        "matrix_calibration",
    }:
        raise MatrixValidationError(
            "matrix postflight artifact set is not validation-only"
        )
    try:
        replay = evaluator._postflight_screen(launch, input_rehash)
    except (evaluator.PaperEvaluationError, OSError, ValueError) as exc:
        raise MatrixValidationError(f"matrix postflight replay failed: {exc}") from exc
    observed = dict(postflight)
    expected = dict(replay)
    observed.pop("validated_at_utc", None)
    expected.pop("validated_at_utc", None)
    if observed != expected:
        raise MatrixValidationError("persisted matrix postflight differs from replay")


def _record_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("task"),
        row.get("manifest_key"),
        row.get("manifest_sha256"),
        row.get("manifest_n"),
        row.get("manifest_index"),
        row.get("sample_id"),
        row.get("image_id"),
        row.get("ann_id"),
        row.get("ref_id"),
        row.get("sent_id"),
        row.get("split"),
    )


def _load_ref_surface(
    *,
    split: str,
    summary_row: Mapping[str, Any],
    artifact: Mapping[str, Any],
    summary_path: Path,
    section_dir: Path,
    run_id: str,
    cache: evaluator.HashCache,
) -> RefSurface:
    expected = REF_VALIDATION_CONTRACT[split]
    if (
        _exact_int(summary_row.get("manifest_n"), label=f"{split}.manifest_n")
        != int(expected["rows"])
        or str(summary_row.get("manifest_sha256", "")).lower()
        != str(expected["sha256"])
    ):
        raise MatrixValidationError(f"{split} count/manifest contract mismatch")
    records_record = artifact.get("records")
    records_path = _verify_file_record(
        records_record,
        label=f"{split} records",
        cache=cache,
    )
    try:
        records_path.relative_to(section_dir)
    except ValueError as exc:
        raise MatrixValidationError(f"{split} records escape matrix output") from exc
    try:
        replay_artifact = {
            key: records_record[key]
            for key in ("path", "sha256", "size_bytes")
        }
        loaded = load_formal_ref_records(
            replay_artifact,
            base_dir=section_dir,
            label=f"matrix {split}",
            split=split,
            summary_row=summary_row,
            summary_path=summary_path,
            split_contract=REF_SPLIT_CONTRACT,
        )
    except (RefRecordContractError, OSError, ValueError) as exc:
        raise MatrixValidationError(f"{split} record replay failed: {exc}") from exc
    rows = _read_jsonl(records_path, label=f"{split} records")
    manifest_path = (
        section_dir
        / "refcoco_eval_inputs"
        / REF_SPLIT_MANIFEST_FILES[split]
    ).resolve(strict=True)
    manifest_rows = _read_jsonl(
        manifest_path, label=f"{split} locked evaluation manifest"
    )
    if (
        len(manifest_rows) != int(expected["rows"])
        or cache.digest(manifest_path) != str(expected["sha256"])
    ):
        raise MatrixValidationError(
            f"{split} generated evaluation manifest violates the locked contract"
        )
    if len(rows) != len(manifest_rows):
        raise MatrixValidationError(
            f"{split} records/evaluation manifest length drifted"
        )
    if any(row.get("run_id") != run_id for row in rows):
        raise MatrixValidationError(f"{split} records mix checkpoint run IDs")
    iou_values: list[float] = []
    identities: list[tuple[Any, ...]] = []
    identity_fields = ("image_id", "ann_id", "ref_id", "sent_id")
    for index, (row, manifest_row) in enumerate(zip(rows, manifest_rows)):
        expected_sample_id = "ref:{}:{}:{}:{}:{}".format(
            split, *(manifest_row.get(field) for field in identity_fields)
        )
        if (
            row.get("manifest_index") != index
            or any(
                row.get(field) != manifest_row.get(field)
                for field in identity_fields
            )
            or row.get("sample_id") != expected_sample_id
        ):
            raise MatrixValidationError(
                f"{split}[{index}] identity differs from locked evaluation manifest"
            )
        iou = _finite(row.get("top1_iou"), label=f"{split}[{index}].top1_iou")
        if not 0.0 <= iou <= 1.0:
            raise MatrixValidationError(f"{split}[{index}] top1_iou is outside [0,1]")
        if row.get("correct50") is not (iou >= 0.5):
            raise MatrixValidationError(f"{split}[{index}] correct50 contradicts IoU")
        iou_values.append(iou)
        identities.append(_record_identity(row))
    iou_array = np.asarray(iou_values, dtype=np.float64)
    measured = {
        "acc50": float(loaded.correct50.astype(np.float64).mean()),
        "mean_iou": float(iou_array.mean()),
    }
    for metric, value in measured.items():
        _same_float(
            summary_row.get(metric), value, label=f"{split}.summary.{metric}"
        )
    if (
        artifact.get("manifest_n") != int(expected["rows"])
        or artifact.get("manifest_sha256") != str(expected["sha256"])
    ):
        raise MatrixValidationError(f"{split} postflight manifest contract drifted")
    _same_float(
        artifact.get("summary_acc50"),
        measured["acc50"],
        label=f"{split}.postflight.acc50",
    )
    return RefSurface(
        split=split,
        rows=rows,
        identities=tuple(identities),
        image_ids=np.asarray(loaded.image_ids, dtype=np.int64),
        correct50=np.asarray(loaded.correct50, dtype=np.bool_),
        top1_iou=iou_array,
        metrics=measured,
        manifest_sha256=str(expected["sha256"]),
        evaluation_manifest=manifest_path,
        records=dict(records_record),
    )


def _calibration_metrics(
    positive: np.ndarray, negative: np.ndarray
) -> dict[str, float]:
    if positive.shape != negative.shape or positive.size == 0:
        raise MatrixValidationError(
            "calibration scores must be non-empty paired arrays"
        )
    fpr = exact_fpr95(positive, negative)
    return {
        "fpr95": float(fpr["fpr"]),
        "positive_q05": float(fpr["threshold"]),
        "auroc": float(exact_binary_auroc(positive, negative)),
        "positive_over_negative_pair_win_rate": float(
            np.mean(positive > negative)
        ),
    }


def _load_calibration_surface(
    *,
    summary_row: Mapping[str, Any],
    artifact: Mapping[str, Any],
    section_dir: Path,
    run_id: str,
    cache: evaluator.HashCache,
) -> CalibrationSurface:
    if artifact.get("manifest_n") != CALIBRATION_ROWS:
        raise MatrixValidationError("matrix calibration is not exactly 1,570 rows")
    for key in (
        "source_manifest",
        "source_audit",
        "derived_manifest",
        "binding",
        "records",
    ):
        if not isinstance(artifact.get(key), Mapping):
            raise MatrixValidationError(f"matrix calibration lacks {key} artifact")
    source_path = _verify_content_record(
        artifact["source_manifest"], label="calibration source", cache=cache
    )
    audit_path = _verify_content_record(
        artifact["source_audit"], label="calibration audit", cache=cache
    )
    derived_path = _verify_content_record(
        artifact["derived_manifest"], label="calibration derived manifest", cache=cache
    )
    binding_path = _verify_file_record(
        artifact["binding"], label="calibration binding", cache=cache
    )
    records_path = _verify_file_record(
        artifact["records"], label="calibration records", cache=cache
    )
    for path, label in (
        (derived_path, "derived manifest"),
        (binding_path, "binding"),
        (records_path, "records"),
    ):
        try:
            path.relative_to(section_dir)
        except ValueError as exc:
            raise MatrixValidationError(
                f"calibration {label} escapes matrix output"
            ) from exc
    try:
        binding = load_binding(binding_path, expected_derived=derived_path)
    except (ScreenCalibrationError, OSError, ValueError) as exc:
        raise MatrixValidationError(
            f"calibration binding replay failed: {exc}"
        ) from exc
    if (
        Path(str(binding.source_manifest["path"])).resolve(strict=True) != source_path
        or Path(str(binding.source_audit["path"])).resolve(strict=True) != audit_path
        or int(binding.source_manifest["rows"]) != CALIBRATION_ROWS
        or int(binding.derived_manifest["rows"]) != CALIBRATION_ROWS
        or binding.eval_split != CALIBRATION_SPLIT
    ):
        raise MatrixValidationError("calibration binding source/count/split drifted")
    try:
        manifest = load_manifest(derived_path)
        loaded = load_tn_records(records_path, manifest, label="matrix calibration")
    except (RecordComparisonError, OSError, ValueError) as exc:
        raise MatrixValidationError(
            f"calibration record binding failed: {exc}"
        ) from exc
    if (
        loaded.manifest_binding_mode != "legacy_direct_source_v1"
        or loaded.run_ids != (run_id,)
        or not bool(np.all(loaded.valid))
        or len(loaded.rows) != CALIBRATION_ROWS
        or set(manifest.splits) != {CALIBRATION_SPLIT}
    ):
        raise MatrixValidationError(
            "calibration records are invalid, mixed, or incomplete"
        )
    metrics = _calibration_metrics(loaded.positive, loaded.negative)
    expected_summary = {
        "manifest_n": CALIBRATION_ROWS,
        "num_pairs": CALIBRATION_ROWS,
        "screen_calibration_source_n": CALIBRATION_ROWS,
        "screen_calibration_binding_schema": CALIBRATION_BINDING_SCHEMA,
        "screen_calibration_derivation_algorithm": DERIVATION_ALGORITHM,
        "screen_calibration_scope": "proposal_covered_verified",
        "screen_calibration_single_edit": True,
    }
    for key, expected in expected_summary.items():
        if summary_row.get(key) != expected:
            raise MatrixValidationError(f"calibration summary {key} drifted")
    for field, metric in (
        ("fpr95tpr", "fpr95"),
        ("threshold_at_95tpr", "positive_q05"),
        ("pair_win_rate", "positive_over_negative_pair_win_rate"),
    ):
        _same_float(
            summary_row.get(field), metrics[metric], label=f"calibration.{field}"
        )
    _same_float(
        artifact.get("summary_fpr95"),
        metrics["fpr95"],
        label="postflight.calibration.fpr95",
    )
    identities = tuple(_record_identity(row) for row in loaded.rows)
    image_ids = np.asarray(
        [
            _exact_int(row.get("image_id"), label="calibration.image_id")
            for row in loaded.rows
        ],
        dtype=np.int64,
    )
    return CalibrationSurface(
        rows=tuple(loaded.rows),
        identities=identities,
        image_ids=image_ids,
        positive=np.asarray(loaded.positive, dtype=np.float64),
        negative=np.asarray(loaded.negative, dtype=np.float64),
        metrics=metrics,
        source_manifest_sha256=str(binding.source_manifest["sha256"]),
        source_audit_sha256=str(binding.source_audit["sha256"]),
        derived_manifest_sha256=str(binding.derived_manifest["sha256"]),
        row_mapping_sha256=binding.row_mapping_sha256,
        records=dict(artifact["records"]),
    )


def _load_run(
    *,
    experiment_id: str,
    seed: int,
    root: Path,
) -> LoadedRun:
    root = root.resolve(strict=True)
    cache = evaluator.HashCache()
    launch_path = (root / "launch_manifest.json").resolve(strict=True)
    postflight_path = (root / "postflight.json").resolve(strict=True)
    launch = _read_json(launch_path, label="matrix launch")
    postflight = _read_json(postflight_path, label="matrix postflight")
    _assert_validation_only_root(root, launch)
    (
        evaluation_id,
        training_run_id,
        training_run_root,
        checkpoint,
        checkpoint_sha,
        runtime_fingerprint,
    ) = _validate_launch_contract(
        launch,
        cache=cache,
        experiment_id=experiment_id,
        root=root,
        seed=seed,
    )
    postflight_artifact = launch.get("postflight_artifact")
    _verify_file_record(
        postflight_artifact,
        label="matrix postflight artifact",
        cache=cache,
        expected_path=postflight_path,
    )
    if launch.get("postflight") != postflight:
        raise MatrixValidationError("matrix launch embeds a different postflight")
    input_rehash = _verify_input_rehash(
        launch, postflight, root=root, cache=cache
    )
    _validate_postflight(launch, postflight, input_rehash)
    checkpoint_contract = postflight.get("checkpoint")
    if not isinstance(checkpoint_contract, Mapping):
        raise MatrixValidationError("matrix postflight checkpoint is missing")
    checkpoint_run_id = str(checkpoint_contract.get("run_id") or "")
    if (
        Path(str(checkpoint_contract.get("path", ""))).resolve(strict=True)
        != checkpoint
        or str(checkpoint_contract.get("sha256", "")).lower() != checkpoint_sha
        or not checkpoint_run_id
    ):
        raise MatrixValidationError("matrix checkpoint launch/postflight drifted")
    if cache.digest(checkpoint) != checkpoint_sha:
        raise MatrixValidationError("matrix checkpoint changed after evaluation")

    artifacts = postflight["artifacts"]
    section_dir = (root / "validation_calibration").resolve(strict=True)
    summary_path = _verify_file_record(
        artifacts["summary"],
        label="matrix summary",
        cache=cache,
        expected_path=section_dir / "summary.json",
    )
    summary = _read_json(summary_path, label="matrix summary")
    if set(summary) != {"refcoco", "tn"}:
        raise MatrixValidationError(
            "matrix summary must contain exactly refcoco and tn"
        )
    ref_rows = summary.get("refcoco")
    tn_rows = summary.get("tn")
    if not isinstance(ref_rows, list) or not isinstance(tn_rows, list):
        raise MatrixValidationError("matrix summary sections must be lists")
    by_split: dict[str, Mapping[str, Any]] = {}
    for row in ref_rows:
        if not isinstance(row, Mapping):
            raise MatrixValidationError("matrix Ref summary contains a non-object")
        split = str(row.get("dataset") or "")
        if split in by_split:
            raise MatrixValidationError(f"matrix Ref split is duplicated: {split}")
        by_split[split] = row
    if tuple(by_split) != REF_VALIDATION_SPLITS or set(by_split) != set(
        REF_VALIDATION_SPLITS
    ):
        raise MatrixValidationError(
            "matrix Ref summary is not the exact ordered three validation splits"
        )
    ref_artifacts = artifacts["ref_validation"]
    if not isinstance(ref_artifacts, Mapping) or set(ref_artifacts) != set(
        REF_VALIDATION_SPLITS
    ):
        raise MatrixValidationError("matrix Ref postflight split set drifted")
    ref = {
        split: _load_ref_surface(
            split=split,
            summary_row=by_split[split],
            artifact=ref_artifacts[split],
            summary_path=summary_path,
            section_dir=section_dir,
            run_id=checkpoint_run_id,
            cache=cache,
        )
        for split in REF_VALIDATION_SPLITS
    }
    if len(tn_rows) != 1 or not isinstance(tn_rows[0], Mapping):
        raise MatrixValidationError("matrix calibration summary must contain one row")
    calibration = _load_calibration_surface(
        summary_row=tn_rows[0],
        artifact=artifacts["matrix_calibration"],
        section_dir=section_dir,
        run_id=checkpoint_run_id,
        cache=cache,
    )
    expected_record_paths = {
        Path(str(surface.records["path"])).resolve(strict=True)
        for surface in ref.values()
    } | {Path(str(calibration.records["path"])).resolve(strict=True)}
    observed_record_paths = {
        path.resolve(strict=True) for path in root.rglob("*.records.jsonl")
    }
    if observed_record_paths != expected_record_paths:
        extra = sorted(
            str(path) for path in observed_record_paths - expected_record_paths
        )
        missing = sorted(
            str(path) for path in expected_record_paths - observed_record_paths
        )
        raise MatrixValidationError(
            f"matrix per-example artifact set drifted; extra={extra}, missing={missing}"
        )
    code_fingerprint = _input_role_fingerprint(
        launch, "evaluation_code_dependency"
    )
    data_fingerprint = _input_role_fingerprint(launch, "evaluation_data_input")
    surface_fingerprint = {
        "ref_validation": {
            split: {
                "rows": int(REF_VALIDATION_CONTRACT[split]["rows"]),
                "sha256": str(REF_VALIDATION_CONTRACT[split]["sha256"]),
            }
            for split in REF_VALIDATION_SPLITS
        },
        "calibration": {
            "rows": CALIBRATION_ROWS,
            "source_manifest_sha256": calibration.source_manifest_sha256,
            "source_audit_sha256": calibration.source_audit_sha256,
            "derived_manifest_sha256": calibration.derived_manifest_sha256,
            "row_mapping_sha256": calibration.row_mapping_sha256,
        },
    }
    sealed_paths = {
        launch_path,
        root / "input_rehash.json",
        postflight_path,
        summary_path,
        checkpoint,
        *(surface.evaluation_manifest for surface in ref.values()),
        *(Path(str(surface.records["path"])) for surface in ref.values()),
        Path(str(calibration.records["path"])),
    }
    inputs = launch.get("inputs")
    input_records = inputs.get("records") if isinstance(inputs, Mapping) else None
    if not isinstance(input_records, list):
        raise MatrixValidationError("matrix launch inputs disappeared")
    for record in input_records:
        if not isinstance(record, Mapping):
            raise MatrixValidationError("matrix launch input record is invalid")
        sealed_paths.add(Path(str(record.get("path", ""))))
    matrix_artifact = artifacts["matrix_calibration"]
    if not isinstance(matrix_artifact, Mapping):
        raise MatrixValidationError("matrix calibration artifact disappeared")
    for key in (
        "source_manifest",
        "source_audit",
        "derived_manifest",
        "binding",
        "records",
    ):
        record = matrix_artifact.get(key)
        if not isinstance(record, Mapping):
            raise MatrixValidationError(f"matrix calibration lacks {key}")
        sealed_paths.add(Path(str(record.get("path", ""))))
    return LoadedRun(
        experiment_id=experiment_id,
        seed=seed,
        root=root,
        evaluation_id=evaluation_id,
        training_run_id=training_run_id,
        training_run_root=training_run_root,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_run_id=checkpoint_run_id,
        ref=ref,
        calibration=calibration,
        code_fingerprint=code_fingerprint,
        data_fingerprint=data_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        surface_fingerprint=surface_fingerprint,
        evidence={
            "launch": _compact_file_record(launch_path, cache),
            "input_rehash": _compact_file_record(root / "input_rehash.json", cache),
            "postflight": _compact_file_record(postflight_path, cache),
            "summary": _compact_file_record(summary_path, cache),
            "ref_records": {
                split: _compact_file_record(
                    Path(str(ref[split].records["path"])), cache
                )
                for split in REF_VALIDATION_SPLITS
            },
            "ref_evaluation_manifests": {
                split: _compact_file_record(
                    ref[split].evaluation_manifest, cache
                )
                for split in REF_VALIDATION_SPLITS
            },
            "calibration_records": _compact_file_record(
                Path(str(calibration.records["path"])), cache
            ),
        },
        sealed_files=_snapshot_files(sealed_paths, cache),
    )


def _sample_summary(values: Iterable[float]) -> dict[str, Any]:
    rendered = [_finite(value, label="seed metric") for value in values]
    if len(rendered) < 2:
        raise MatrixValidationError("sample standard deviation requires >=2 seeds")
    return {
        "n": len(rendered),
        "mean": float(statistics.fmean(rendered)),
        "sample_std": float(statistics.stdev(rendered)),
        "ddof": 1,
    }


def _run_metrics(run: LoadedRun) -> dict[str, Any]:
    ref = {
        split: {
            "n": int(run.ref[split].correct50.size),
            **dict(run.ref[split].metrics),
        }
        for split in REF_VALIDATION_SPLITS
    }
    ref["val_macro"] = {
        metric: float(
            np.mean([run.ref[split].metrics[metric] for split in REF_VALIDATION_SPLITS])
        )
        for metric in REF_METRICS
    }
    return {
        "train_seed": run.seed,
        "evaluation_root": str(run.root),
        "evaluation_id": run.evaluation_id,
        "training_run_id": run.training_run_id,
        "training_run_root": str(run.training_run_root),
        "checkpoint": {
            "path": str(run.checkpoint),
            "sha256": run.checkpoint_sha256,
            "record_run_id": run.checkpoint_run_id,
        },
        "access_label": ACCESS_LABEL,
        "ref_validation": ref,
        "calibration": {
            "n": CALIBRATION_ROWS,
            **dict(run.calibration.metrics),
        },
        "evidence": dict(run.evidence),
    }


def _aggregate_experiment(runs: Mapping[int, LoadedRun]) -> dict[str, Any]:
    seeds = sorted(runs)
    per_seed = {str(seed): _run_metrics(runs[seed]) for seed in seeds}
    ref: dict[str, Any] = {}
    for split in (*REF_VALIDATION_SPLITS, "val_macro"):
        ref[split] = {
            metric: _sample_summary(
                per_seed[str(seed)]["ref_validation"][split][metric]
                for seed in seeds
            )
            for metric in REF_METRICS
        }
    calibration = {
        metric: _sample_summary(
            per_seed[str(seed)]["calibration"][metric] for seed in seeds
        )
        for metric in CALIBRATION_METRICS
    }
    return {
        "per_seed": per_seed,
        "aggregate": {
            "estimator": "per-training-seed metric, then equal-seed mean/sample std",
            "ref_validation": ref,
            "calibration": calibration,
        },
    }


def _clusters(image_ids: np.ndarray) -> tuple[np.ndarray, ...]:
    grouped: dict[int, list[int]] = {}
    order: list[int] = []
    for index, raw in enumerate(image_ids.tolist()):
        image_id = int(raw)
        if image_id not in grouped:
            grouped[image_id] = []
            order.append(image_id)
        grouped[image_id].append(index)
    return tuple(
        np.asarray(grouped[image_id], dtype=np.int64) for image_id in order
    )


def _derived_seed(seed: int, *parts: Any) -> int:
    payload = ":".join([str(seed), *(str(value) for value in parts)]).encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _assert_aligned(reference: LoadedRun, candidate: LoadedRun) -> None:
    for split in REF_VALIDATION_SPLITS:
        left = reference.ref[split]
        right = candidate.ref[split]
        if left.identities != right.identities or not np.array_equal(
            left.image_ids, right.image_ids
        ):
            raise MatrixValidationError(
                f"declared comparison {split} record identities are not aligned"
            )
    left_tn = reference.calibration
    right_tn = candidate.calibration
    if left_tn.identities != right_tn.identities or not np.array_equal(
        left_tn.image_ids, right_tn.image_ids
    ):
        raise MatrixValidationError(
            "declared comparison calibration record identities are not aligned"
        )


def _delta_summary(
    draws: np.ndarray,
    *,
    observed: float,
    per_seed: Sequence[float],
    iterations: int,
    confidence: float,
    seed: int,
    clusters_n: int,
) -> dict[str, Any]:
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return {
        "direction": "candidate_minus_reference",
        "observed_seed_mean_delta": float(observed),
        "per_seed_delta": _sample_summary(per_seed),
        "bootstrap": {
            "paired": True,
            "unit": "image_cluster_within_training_seed",
            "seed_weighting": "equal",
            "seed_first": True,
            "iterations": int(iterations),
            "confidence": float(confidence),
            "ci_method": "percentile",
            "seed": int(seed),
            "image_clusters_across_seeds": int(clusters_n),
            "delta_mean": float(draws.mean()),
            "delta_std_ddof0": float(draws.std(ddof=0)),
            "delta_ci_low": float(low),
            "delta_ci_high": float(high),
            "probability_delta_below_zero": float(np.mean(draws < 0.0)),
        },
    }


def _comparison(
    *,
    reference_id: str,
    candidate_id: str,
    reference_runs: Mapping[int, LoadedRun],
    candidate_runs: Mapping[int, LoadedRun],
    iterations: int,
    confidence: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    seeds = sorted(reference_runs)
    if seeds != sorted(candidate_runs):
        raise MatrixValidationError("declared comparison seed sets differ")
    for seed in seeds:
        _assert_aligned(reference_runs[seed], candidate_runs[seed])
    per_seed: dict[str, Any] = {}
    for seed in seeds:
        left = _run_metrics(reference_runs[seed])
        right = _run_metrics(candidate_runs[seed])
        per_seed[str(seed)] = {
            "ref_validation": {
                split: {
                    metric: float(
                        right["ref_validation"][split][metric]
                        - left["ref_validation"][split][metric]
                    )
                    for metric in REF_METRICS
                }
                for split in (*REF_VALIDATION_SPLITS, "val_macro")
            },
            "calibration": {
                metric: float(
                    right["calibration"][metric]
                    - left["calibration"][metric]
                )
                for metric in CALIBRATION_METRICS
            },
        }

    derived = _derived_seed(
        bootstrap_seed, reference_id, candidate_id, "matrix_validation"
    )
    rng = np.random.default_rng(derived)
    draw_keys = [
        *(
            f"ref:{split}:{metric}"
            for split in REF_VALIDATION_SPLITS
            for metric in REF_METRICS
        ),
        *(f"ref:val_macro:{metric}" for metric in REF_METRICS),
        *(f"calibration:{metric}" for metric in CALIBRATION_METRICS),
    ]
    draws = {
        key: np.empty(iterations, dtype=np.float64) for key in draw_keys
    }
    cluster_counts = {key: 0 for key in draw_keys}
    ref_clusters: dict[tuple[int, str], tuple[np.ndarray, ...]] = {}
    tn_clusters: dict[int, tuple[np.ndarray, ...]] = {}
    for seed in seeds:
        for split in REF_VALIDATION_SPLITS:
            clusters = _clusters(reference_runs[seed].ref[split].image_ids)
            if not clusters:
                raise MatrixValidationError("cannot bootstrap empty Ref surface")
            ref_clusters[(seed, split)] = clusters
            for metric in REF_METRICS:
                cluster_counts[f"ref:{split}:{metric}"] += len(clusters)
                cluster_counts[f"ref:val_macro:{metric}"] += len(clusters)
        clusters = _clusters(reference_runs[seed].calibration.image_ids)
        if not clusters:
            raise MatrixValidationError("cannot bootstrap empty calibration surface")
        tn_clusters[seed] = clusters
        for metric in CALIBRATION_METRICS:
            cluster_counts[f"calibration:{metric}"] += len(clusters)

    for iteration in range(iterations):
        seed_draws: dict[str, list[float]] = {key: [] for key in draw_keys}
        for seed in seeds:
            macro: dict[str, list[float]] = {metric: [] for metric in REF_METRICS}
            for split in REF_VALIDATION_SPLITS:
                clusters = ref_clusters[(seed, split)]
                chosen = rng.integers(0, len(clusters), size=len(clusters))
                indices = np.concatenate([clusters[index] for index in chosen])
                left = reference_runs[seed].ref[split]
                right = candidate_runs[seed].ref[split]
                values = {
                    "acc50": float(
                        right.correct50[indices].mean()
                        - left.correct50[indices].mean()
                    ),
                    "mean_iou": float(
                        right.top1_iou[indices].mean()
                        - left.top1_iou[indices].mean()
                    ),
                }
                for metric, value in values.items():
                    seed_draws[f"ref:{split}:{metric}"].append(value)
                    macro[metric].append(value)
            for metric in REF_METRICS:
                seed_draws[f"ref:val_macro:{metric}"].append(
                    float(np.mean(macro[metric]))
                )
            clusters = tn_clusters[seed]
            chosen = rng.integers(0, len(clusters), size=len(clusters))
            indices = np.concatenate([clusters[index] for index in chosen])
            left_tn = reference_runs[seed].calibration
            right_tn = candidate_runs[seed].calibration
            left_metrics = _calibration_metrics(
                left_tn.positive[indices], left_tn.negative[indices]
            )
            right_metrics = _calibration_metrics(
                right_tn.positive[indices], right_tn.negative[indices]
            )
            for metric in CALIBRATION_METRICS:
                seed_draws[f"calibration:{metric}"].append(
                    right_metrics[metric] - left_metrics[metric]
                )
        for key in draw_keys:
            if len(seed_draws[key]) != len(seeds):
                raise MatrixValidationError(
                    f"bootstrap seed-first coverage failed: {key}"
                )
            draws[key][iteration] = float(np.mean(seed_draws[key]))

    ref_report: dict[str, Any] = {}
    for split in (*REF_VALIDATION_SPLITS, "val_macro"):
        ref_report[split] = {}
        for metric in REF_METRICS:
            values = [
                per_seed[str(seed)]["ref_validation"][split][metric]
                for seed in seeds
            ]
            ref_report[split][metric] = _delta_summary(
                draws[f"ref:{split}:{metric}"],
                observed=float(statistics.fmean(values)),
                per_seed=values,
                iterations=iterations,
                confidence=confidence,
                seed=derived,
                clusters_n=cluster_counts[f"ref:{split}:{metric}"],
            )
    calibration_report = {}
    for metric in CALIBRATION_METRICS:
        values = [
            per_seed[str(seed)]["calibration"][metric] for seed in seeds
        ]
        calibration_report[metric] = _delta_summary(
            draws[f"calibration:{metric}"],
            observed=float(statistics.fmean(values)),
            per_seed=values,
            iterations=iterations,
            confidence=confidence,
            seed=derived,
            clusters_n=cluster_counts[f"calibration:{metric}"],
        )
    return {
        "reference_experiment": reference_id,
        "candidate_experiment": candidate_id,
        "direction": "candidate_minus_reference",
        "record_identities_aligned": True,
        "train_seeds": seeds,
        "per_seed": per_seed,
        "seed_first_paired_bootstrap": {
            "iterations": iterations,
            "confidence": confidence,
            "base_seed": bootstrap_seed,
            "derived_seed": derived,
        },
        "ref_validation": ref_report,
        "calibration": calibration_report,
    }


def _parse_spec(
    spec_path: Path,
) -> tuple[
    tuple[int, ...],
    str | None,
    Path,
    str,
    str,
    list[dict[str, Any]],
]:
    spec = _read_json(spec_path, label="matrix validation input spec")
    allowed = {
        "schema",
        "expected_train_seeds",
        "evaluation_queue_dir",
        "evaluation_queue_id",
        "evaluation_plan_sha256",
        "evaluation_provenance_scope",
        "reference_experiment",
        "experiments",
    }
    if set(spec) - allowed:
        raise MatrixValidationError(
            f"input spec has unsupported fields: {sorted(set(spec) - allowed)}"
        )
    if spec.get("schema") != INPUT_SCHEMA:
        raise MatrixValidationError(
            f"input spec schema must be exactly {INPUT_SCHEMA!r}"
        )
    evaluation_queue_id = spec.get("evaluation_queue_id")
    evaluation_plan_sha256 = spec.get("evaluation_plan_sha256")
    evaluation_provenance_scope = spec.get("evaluation_provenance_scope")
    if not isinstance(evaluation_queue_id, str) or not evaluation_queue_id:
        raise MatrixValidationError(
            "input spec evaluation_queue_id must be a non-empty string"
        )
    if _SHA_RE.fullmatch(str(evaluation_plan_sha256 or "")) is None:
        raise MatrixValidationError(
            "input spec evaluation_plan_sha256 must be 64 lowercase hex characters"
        )
    if evaluation_provenance_scope != evaluation_queue.FORMAL_PROVENANCE_SCOPE:
        raise MatrixValidationError(
            "input spec evaluation_provenance_scope must be exactly formal"
        )
    if (
        FORMAL_EVALUATION_QUEUE_ID is None
        or FORMAL_EVALUATION_PLAN_SHA256 is None
    ):
        raise MatrixValidationError(
            "formal matrix evaluation queue identity has not been sealed"
        )
    if (
        evaluation_queue_id != FORMAL_EVALUATION_QUEUE_ID
        or evaluation_plan_sha256 != FORMAL_EVALUATION_PLAN_SHA256
    ):
        raise MatrixValidationError(
            "input spec does not name the authorized matrix evaluation queue"
        )
    evaluation_queue_dir = _resolve_path(
        spec.get("evaluation_queue_dir"),
        base=spec_path.parent,
        label="evaluation_queue_dir",
        directory=True,
    )
    raw_seeds = spec.get("expected_train_seeds")
    if not isinstance(raw_seeds, list):
        raise MatrixValidationError("expected_train_seeds must be a list")
    seeds = tuple(
        _exact_int(value, label=f"expected_train_seeds[{index}]")
        for index, value in enumerate(raw_seeds)
    )
    if seeds != FORMAL_TRAIN_SEEDS:
        raise MatrixValidationError(
            "expected_train_seeds must be exactly "
            f"{list(FORMAL_TRAIN_SEEDS)!r} in canonical order"
        )
    raw_experiments = spec.get("experiments")
    if not isinstance(raw_experiments, list) or not raw_experiments:
        raise MatrixValidationError("experiments must be a non-empty list")
    experiments: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_experiments):
        if not isinstance(raw, Mapping) or set(raw) != {
            "id",
            "label",
            "evaluation_roots",
        }:
            raise MatrixValidationError(
                f"experiment {index} must contain exactly id/label/evaluation_roots"
            )
        experiment_id = str(raw.get("id") or "")
        label = str(raw.get("label") or "")
        if _ID_RE.fullmatch(experiment_id) is None or experiment_id in seen_ids:
            raise MatrixValidationError(
                f"experiment {index} has invalid/duplicate id {experiment_id!r}"
            )
        if not label.strip():
            raise MatrixValidationError(f"experiment {experiment_id} has no label")
        if label != experiment_id:
            raise MatrixValidationError(
                f"experiment {experiment_id} label must equal its canonical id"
            )
        seen_ids.add(experiment_id)
        roots = raw.get("evaluation_roots")
        if not isinstance(roots, Mapping):
            raise MatrixValidationError(
                f"experiment {experiment_id} evaluation_roots must be an object"
            )
        expected_keys = {str(seed) for seed in seeds}
        if set(roots) != expected_keys:
            missing = sorted(expected_keys - set(roots))
            extra = sorted(set(roots) - expected_keys)
            raise MatrixValidationError(
                f"experiment {experiment_id} seed set mismatch; "
                f"missing={missing}, extra={extra}"
            )
        resolved = {
            seed: _resolve_path(
                roots[str(seed)],
                base=spec_path.parent,
                label=f"{experiment_id} seed {seed} evaluation root",
                directory=True,
            )
            for seed in seeds
        }
        experiments.append(
            {"id": experiment_id, "label": label.strip(), "roots": resolved}
        )
    raw_reference = spec.get("reference_experiment")
    reference = None if raw_reference is None else str(raw_reference)
    if reference is not None and reference not in seen_ids:
        raise MatrixValidationError(
            f"reference_experiment {reference!r} is not declared"
        )
    observed_ids = tuple(experiment["id"] for experiment in experiments)
    if observed_ids != FORMAL_EXPERIMENT_IDS:
        raise MatrixValidationError(
            "experiments must be exactly the canonical Table-C rows in order: "
            f"{list(FORMAL_EXPERIMENT_IDS)!r}"
        )
    if reference != FORMAL_REFERENCE_EXPERIMENT:
        raise MatrixValidationError(
            "reference_experiment must be exactly "
            f"{FORMAL_REFERENCE_EXPERIMENT!r}"
        )
    return (
        seeds,
        reference,
        evaluation_queue_dir,
        evaluation_queue_id,
        str(evaluation_plan_sha256),
        experiments,
    )


def _canonical_spec_payload() -> dict[str, Any]:
    queue_dir = FORMAL_EVALUATION_QUEUE_DIR.expanduser().resolve(strict=True)
    if queue_dir != evaluation_queue.DEFAULT_QUEUE_DIR.resolve(strict=False):
        raise MatrixValidationError(
            "canonical matrix evaluation queue directory constant drifted"
        )
    try:
        queue = evaluation_queue.load_queue(queue_dir)
        verification = evaluation_queue.verify_queue(queue_dir)
    except (evaluation_queue.MatrixQueueError, OSError, ValueError) as exc:
        raise MatrixValidationError(
            f"canonical matrix evaluation queue verification failed: {exc}"
        ) from exc
    plan = queue.get("plan")
    if not isinstance(plan, Mapping):
        raise MatrixValidationError("canonical matrix evaluation queue plan is missing")
    if (
        queue.get("status") != "completed"
        or verification.get("status") != "passed"
        or verification.get("errors")
        or verification.get("queue_id") != plan.get("queue_id")
        or verification.get("plan_sha256") != queue.get("plan_sha256")
        or plan.get("queue_id") != FORMAL_EVALUATION_QUEUE_ID
        or queue.get("plan_sha256") != FORMAL_EVALUATION_PLAN_SHA256
        or plan.get("provenance_scope")
        != evaluation_queue.FORMAL_PROVENANCE_SCOPE
        or verification.get("provenance_scope")
        != evaluation_queue.FORMAL_PROVENANCE_SCOPE
        or plan.get("profile") != PROFILE
    ):
        raise MatrixValidationError(
            "canonical matrix evaluation queue is not completed/verified exactly"
        )
    plan_items = plan.get("items")
    if not isinstance(plan_items, list):
        raise MatrixValidationError("canonical matrix evaluation queue items are missing")
    expected_ids = [
        f"{experiment_id}:{seed}"
        for seed in FORMAL_TRAIN_SEEDS
        for experiment_id in FORMAL_EXPERIMENT_IDS
    ]
    observed_ids = [
        item.get("run_id") if isinstance(item, Mapping) else None
        for item in plan_items
    ]
    if observed_ids != expected_ids:
        raise MatrixValidationError(
            "canonical matrix evaluation queue run order is not exact"
        )
    if len(verification.get("verified_items", [])) != len(expected_ids):
        raise MatrixValidationError(
            "canonical matrix evaluation queue verification count is not exact"
        )

    roots: dict[tuple[str, int], Path] = {}
    for item in plan_items:
        run_id = str(item["run_id"])
        experiment_id, raw_seed = run_id.split(":", 1)
        seed = int(raw_seed)
        expected_root = (
            evaluation_queue.DEFAULT_OUTPUT_ROOT / experiment_id / f"seed{seed}"
        ).resolve(strict=False)
        observed_root = Path(str(item.get("evaluation_root", ""))).resolve(strict=True)
        if observed_root != expected_root or not observed_root.is_dir():
            raise MatrixValidationError(
                f"canonical evaluation root drifted or is missing: {run_id}"
            )
        roots[(experiment_id, seed)] = observed_root

    try:
        payload = evaluation_queue._aggregation_input_spec_payload(
            plan, str(queue["plan_sha256"])
        )
    except (evaluation_queue.MatrixQueueError, KeyError, ValueError) as exc:
        raise MatrixValidationError(
            f"canonical aggregation input projection failed: {exc}"
        ) from exc
    if payload.get("schema") != INPUT_SCHEMA:
        raise MatrixValidationError("canonical aggregation input schema drifted")
    return payload


def build_canonical_spec(output_path: Path | None = None) -> dict[str, Any]:
    if output_path is None:
        output_path = DEFAULT_INPUT_SPEC
    output_path = output_path.expanduser().resolve(strict=True)
    if output_path != DEFAULT_INPUT_SPEC.resolve(strict=True):
        raise MatrixValidationError(
            f"canonical input spec path must be exactly {DEFAULT_INPUT_SPEC}"
        )
    payload = _canonical_spec_payload()
    if _read_json(output_path, label="canonical matrix input spec") != payload:
        raise MatrixValidationError(
            "canonical matrix input spec differs from its completed queue"
        )
    return payload


def aggregate_spec(
    spec_path: Path | str,
    *,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    confidence: float = DEFAULT_CONFIDENCE,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    spec_path = Path(spec_path).expanduser().resolve(strict=True)
    entry_cache = evaluator.HashCache()
    spec_snapshot = _compact_file_record(spec_path, entry_cache)
    aggregation_sources = _aggregation_source_paths()
    aggregation_source_snapshot = _snapshot_files(
        aggregation_sources, entry_cache
    )
    bootstrap_iterations = _exact_int(
        bootstrap_iterations, label="bootstrap_iterations", minimum=1
    )
    bootstrap_seed = _exact_int(bootstrap_seed, label="bootstrap_seed")
    confidence = _finite(confidence, label="confidence")
    if not 0.0 < confidence < 1.0:
        raise MatrixValidationError("confidence must be in (0,1)")
    if (
        bootstrap_iterations != FORMAL_BOOTSTRAP_ITERATIONS
        or confidence != FORMAL_CONFIDENCE
        or bootstrap_seed != FORMAL_BOOTSTRAP_SEED
    ):
        raise MatrixValidationError(
            "formal bootstrap protocol must be exactly "
            f"iterations={FORMAL_BOOTSTRAP_ITERATIONS}, "
            f"confidence={FORMAL_CONFIDENCE}, seed={FORMAL_BOOTSTRAP_SEED}"
        )
    (
        seeds,
        reference_id,
        evaluation_queue_dir,
        evaluation_queue_id,
        evaluation_plan_sha256,
        experiment_specs,
    ) = _parse_spec(spec_path)
    evaluation_queue_binding = _validate_evaluation_queue_binding(
        evaluation_queue_dir,
        spec_path=spec_path,
        evaluation_queue_id=evaluation_queue_id,
        evaluation_plan_sha256=evaluation_plan_sha256,
        experiments=experiment_specs,
        seeds=seeds,
    )

    loaded: dict[str, dict[int, LoadedRun]] = {}
    labels: dict[str, str] = {}
    seen_roots: set[Path] = set()
    seen_training_run_ids: set[str] = set()
    seen_training_roots: set[Path] = set()
    seen_checkpoint_paths: set[Path] = set()
    seen_checkpoint_hashes: set[str] = set()
    seen_evaluation_ids: set[str] = set()
    common_code: Mapping[str, Mapping[str, Any]] | None = None
    common_data: Mapping[str, Mapping[str, Any]] | None = None
    common_runtime: Mapping[str, Any] | None = None
    common_surface: Mapping[str, Any] | None = None
    canonical_ref_identities: dict[str, tuple[tuple[Any, ...], ...]] = {}
    canonical_calibration_identities: tuple[tuple[Any, ...], ...] | None = None

    for experiment in experiment_specs:
        experiment_id = str(experiment["id"])
        labels[experiment_id] = str(experiment["label"])
        runs: dict[int, LoadedRun] = {}
        for seed in seeds:
            root = experiment["roots"][seed]
            if root in seen_roots:
                raise MatrixValidationError(f"evaluation root is duplicated: {root}")
            seen_roots.add(root)
            run = _load_run(
                experiment_id=experiment_id,
                seed=seed,
                root=root,
            )
            for value, seen, label in (
                (run.training_run_id, seen_training_run_ids, "training run_id"),
                (run.training_run_root, seen_training_roots, "training root"),
                (run.checkpoint, seen_checkpoint_paths, "checkpoint path"),
                (run.checkpoint_sha256, seen_checkpoint_hashes, "checkpoint SHA"),
                (run.evaluation_id, seen_evaluation_ids, "evaluation_id"),
            ):
                if value in seen:
                    raise MatrixValidationError(f"duplicate {label}: {value}")
                seen.add(value)
            if common_code is None:
                common_code = run.code_fingerprint
                common_data = run.data_fingerprint
                common_runtime = run.runtime_fingerprint
                common_surface = run.surface_fingerprint
            elif (
                run.code_fingerprint != common_code
                or run.data_fingerprint != common_data
                or run.runtime_fingerprint != common_runtime
                or run.surface_fingerprint != common_surface
            ):
                raise MatrixValidationError(
                    "matrix runs have inconsistent runtime/code/data/surface hashes"
                )
            for split in REF_VALIDATION_SPLITS:
                identities = run.ref[split].identities
                if split not in canonical_ref_identities:
                    canonical_ref_identities[split] = identities
                elif identities != canonical_ref_identities[split]:
                    raise MatrixValidationError(
                        f"matrix runs have inconsistent {split} record identities"
                    )
            if canonical_calibration_identities is None:
                canonical_calibration_identities = run.calibration.identities
            elif run.calibration.identities != canonical_calibration_identities:
                raise MatrixValidationError(
                    "matrix runs have inconsistent calibration record identities"
                )
            runs[seed] = run
        loaded[experiment_id] = runs

    rendered_experiments = {
        experiment_id: {
            "id": experiment_id,
            "label": labels[experiment_id],
            **_aggregate_experiment(runs),
        }
        for experiment_id, runs in loaded.items()
    }
    comparisons: dict[str, Any] = {}
    if reference_id is not None:
        for candidate_id, candidate_runs in loaded.items():
            if candidate_id == reference_id:
                continue
            comparisons[candidate_id] = _comparison(
                reference_id=reference_id,
                candidate_id=candidate_id,
                reference_runs=loaded[reference_id],
                candidate_runs=candidate_runs,
                iterations=bootstrap_iterations,
                confidence=confidence,
                bootstrap_seed=bootstrap_seed,
            )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "validated_matrix_validation_only",
        "created_at_utc": _utc_now(),
        "access_label": ACCESS_LABEL,
        "formal_test_or_strict_result": False,
        "protocol": {
            "profile": PROFILE,
            "expected_train_seeds": list(seeds),
            "ref_validation_splits": list(REF_VALIDATION_SPLITS),
            "ref_test_access": False,
            "strict2031_access": False,
            "strict1607_access": False,
            "calibration_scope": "proposal_covered_verified",
            "calibration_rows": CALIBRATION_ROWS,
            "seed_estimator": (
                "compute each metric per training seed, then equal-seed mean and "
                "sample standard deviation"
            ),
            "val_macro_weighting": "equal weight over the three Ref validation splits",
            "sample_std_ddof": 1,
            "paired_bootstrap": {
                "iterations": bootstrap_iterations,
                "confidence": confidence,
                "seed": bootstrap_seed,
                "unit": "image cluster within training seed",
                "seed_first": True,
            },
        },
        "validation": {
            "pass": True,
            "exact_seed_sets": True,
            "unique_evaluation_roots": True,
            "unique_training_runs": True,
            "unique_checkpoints": True,
            "input_rehash_replayed": True,
            "postflight_replayed": True,
            "evaluation_code_hashes_consistent": True,
            "evaluation_data_hashes_consistent": True,
            "evaluation_runtime_consistent": True,
            "record_surfaces_consistent": True,
            "no_ref_test_or_strict_artifacts": True,
        },
        "inputs": {
            "spec": dict(spec_snapshot),
            "aggregator_source": next(
                dict(record)
                for record in aggregation_source_snapshot
                if Path(str(record["path"])) == Path(__file__).resolve()
            ),
            "aggregation_source_closure": [
                dict(record) for record in aggregation_source_snapshot
            ],
            "runtime_dependencies": {
                "python": sys.version,
                "numpy": np.__version__,
            },
            "evaluation_queue": {
                key: value
                for key, value in evaluation_queue_binding.items()
                if key != "sealed_files"
            },
            "common_evaluation_code": dict(common_code or {}),
            "common_evaluation_data": dict(common_data or {}),
            "common_runtime": dict(common_runtime or {}),
            "common_surfaces": dict(common_surface or {}),
        },
        "reference_experiment": reference_id,
        "experiments": rendered_experiments,
        "comparisons_to_reference": comparisons,
    }
    _verify_snapshot_files((spec_snapshot,), label="input spec")
    _verify_snapshot_files(
        aggregation_source_snapshot, label="aggregation source"
    )
    _verify_snapshot_files(
        evaluation_queue_binding["sealed_files"],
        label="canonical evaluation queue",
    )
    for experiment_id, runs in loaded.items():
        for seed, run in runs.items():
            _verify_snapshot_files(
                run.sealed_files,
                label=f"{experiment_id}:{seed} evaluation evidence",
            )
    return report


def _write_atomic(path: Path, content: str) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"output artifact must be fresh: {path}")
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"output artifact must be fresh: {path}"
            ) from exc
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    temporary.unlink()


def _assert_output_path_isolated(
    path: Path, report: Mapping[str, Any]
) -> None:
    output = path.expanduser().resolve(strict=False)
    experiments = report.get("experiments")
    if not isinstance(experiments, Mapping):
        raise MatrixValidationError("report experiments are missing")
    roots: set[Path] = set()
    for experiment in experiments.values():
        per_seed = (
            experiment.get("per_seed")
            if isinstance(experiment, Mapping)
            else None
        )
        if not isinstance(per_seed, Mapping):
            raise MatrixValidationError("report per-seed evidence is missing")
        for run in per_seed.values():
            if not isinstance(run, Mapping):
                raise MatrixValidationError("report run evidence is invalid")
            for key in ("evaluation_root", "training_run_root"):
                roots.add(Path(str(run.get(key, ""))).resolve(strict=True))
    for root in roots:
        try:
            output.relative_to(root)
        except ValueError:
            continue
        raise MatrixValidationError(
            f"output artifact must not be inside evaluated evidence root: {root}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--spec", type=Path)
    mode.add_argument(
        "--build-canonical-spec",
        action="store_true",
        help=f"verify the queue-predeclared input at {DEFAULT_INPUT_SPEC}",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=DEFAULT_BOOTSTRAP_ITERATIONS,
    )
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument(
        "--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.build_canonical_spec:
            if (
                args.output_json is not None
                or args.bootstrap_iterations != DEFAULT_BOOTSTRAP_ITERATIONS
                or args.confidence != DEFAULT_CONFIDENCE
                or args.bootstrap_seed != DEFAULT_BOOTSTRAP_SEED
            ):
                parser.error(
                    "--build-canonical-spec cannot be combined with aggregation options"
                )
            payload = build_canonical_spec()
            print(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
            )
            return 0
        assert args.spec is not None
        report = aggregate_spec(
            args.spec,
            bootstrap_iterations=args.bootstrap_iterations,
            confidence=args.confidence,
            bootstrap_seed=args.bootstrap_seed,
        )
    except (MatrixValidationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output_json is not None:
        _assert_output_path_isolated(args.output_json, report)
        _write_atomic(args.output_json, rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
