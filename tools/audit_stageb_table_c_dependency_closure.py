#!/usr/bin/env python3
"""Create or verify the supplemental Table-C dependency-closure attestation.

This audit does not alter or strengthen the historical launch manifests.  It
records a deterministic, present-day closure of repository-local Python
imports, all L0-L10 config import chains, and the filesystem evidence that the
supplemental files predate the first completed L0-L4 seed-17 launch.  The
attestation is intentionally explicit that this is retrospective evidence,
not a retroactive cryptographic launch binding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_stageb_serial_matrix_queue as queue_runner
from tools.stageb_dependency_audit import (
    DependencyAuditError,
    config_import_chain,
    local_python_dependency_paths,
)


SCHEMA = "pivot.stageb.table_c_dependency_closure_attestation/v1"
VERIFICATION_SCHEMA = (
    "pivot.stageb.table_c_dependency_closure_attestation_verification/v1"
)
CLOSURE_DIGEST_SCHEMA = "pivot.stageb.table_c_dependency_closure_digest/v1"
ATTESTATION_DIGEST_SCHEMA = (
    "pivot.stageb.table_c_dependency_closure_attestation_digest/v1"
)

COMPLETED_RUN_IDS = tuple(f"L{index}:17" for index in range(5))
REMAINING_RUN_IDS = tuple(
    [f"L{index}:17" for index in range(5, 11)]
    + [f"L{index}:42" for index in range(11)]
    + [f"L{index}:73" for index in range(11)]
)

DEFAULT_COMPLETED_QUEUE = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/queues/"
    "table_c_screen_l0_l4_seed17_b40_u1000_frozen_v2"
)
DEFAULT_REMAINING_QUEUE = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/queues/"
    "table_c_remaining_28_b40_u1000_frozen_v2"
)
DEFAULT_TRAINING_ROOT = (
    REPO_ROOT / "outputs/paper_cvpr_v1/token_ablation_frozen_v2"
)


class TableCDependencyClosureError(RuntimeError):
    """The supplemental dependency-closure contract failed closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise TableCDependencyClosureError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _mtime_utc(mtime_ns: int) -> str:
    seconds, nanoseconds = divmod(int(mtime_ns), 1_000_000_000)
    base = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return f"{base.strftime('%Y-%m-%dT%H:%M:%S')}.{nanoseconds:09d}+00:00"


def _parse_utc_ns(value: Any, *, label: str) -> int:
    if not isinstance(value, str) or not value:
        raise TableCDependencyClosureError(f"{label} is not a UTC timestamp")
    rendered = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(rendered)
    except ValueError as exc:
        raise TableCDependencyClosureError(
            f"{label} is not an ISO timestamp: {value!r}"
        ) from exc
    if parsed.tzinfo is None:
        raise TableCDependencyClosureError(f"{label} has no timezone: {value!r}")
    parsed = parsed.astimezone(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = parsed - epoch
    return (
        (delta.days * 86400 + delta.seconds) * 1_000_000_000
        + delta.microseconds * 1000
    )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TableCDependencyClosureError(f"{label} is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise TableCDependencyClosureError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TableCDependencyClosureError(f"{label} must be a JSON object: {path}")
    return value


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve(strict=False)
    if path.exists():
        raise FileExistsError(f"attestation output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    try:
        with temporary.open("x", encoding="ascii") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"attestation output appeared concurrently: {path}")
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _relative_path(path: Path, root: Path, *, label: str) -> str:
    resolved = path.expanduser().resolve(strict=True)
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise TableCDependencyClosureError(
            f"{label} escapes repository root {root}: {resolved}"
        ) from exc


def _file_record(
    path: Path,
    *,
    root: Path | None = None,
    memberships: Iterable[str] = (),
    static_repository_source_bound: bool | None = None,
    historical_repository_rows: Iterable[str] = (),
    historical_config_rows: Iterable[str] = (),
) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    if not path.is_file():
        raise TableCDependencyClosureError(f"evidence path is not a file: {path}")
    stat = path.stat()
    record: dict[str, Any] = {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "mtime_utc": _mtime_utc(int(stat.st_mtime_ns)),
        "sha256": _sha256_file(path),
    }
    if root is not None:
        record["relative_path"] = _relative_path(path, root, label="closure file")
    if memberships:
        record["memberships"] = sorted(set(memberships))
    if static_repository_source_bound is not None:
        record["static_repository_source_bound"] = bool(
            static_repository_source_bound
        )
        record["supplemental_only_relative_to_repository_sources"] = not bool(
            static_repository_source_bound
        )
        repository_rows = sorted(set(historical_repository_rows))
        config_rows = sorted(set(historical_config_rows))
        record["historical_launch_bindings"] = {
            "repository_source_rows": repository_rows,
            "config_dependency_rows": config_rows,
            "bound_by_any_l0_l4_seed17_launch": bool(
                repository_rows or config_rows
            ),
        }
    return record


def _record_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "relative_path": record["relative_path"],
        "sha256": record["sha256"],
        "size_bytes": int(record["size_bytes"]),
        "mtime_ns": int(record["mtime_ns"]),
        "memberships": list(record["memberships"]),
        "static_repository_source_bound": bool(
            record["static_repository_source_bound"]
        ),
        "supplemental_only_relative_to_repository_sources": bool(
            record["supplemental_only_relative_to_repository_sources"]
        ),
        "historical_launch_bindings": dict(record["historical_launch_bindings"]),
    }


def _validate_expected_file_record(
    expected: Any,
    path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(expected, Mapping):
        raise TableCDependencyClosureError(f"{label} has no file record")
    path = path.resolve(strict=True)
    recorded_path = Path(str(expected.get("path", ""))).resolve(strict=False)
    if recorded_path != path:
        raise TableCDependencyClosureError(
            f"{label} path differs: expected {path}, recorded {recorded_path}"
        )
    observed = _file_record(path)
    for key in ("sha256", "size_bytes", "mtime_ns"):
        if expected.get(key) != observed[key]:
            raise TableCDependencyClosureError(
                f"{label} {key} drift for {path}: "
                f"expected {expected.get(key)!r}, observed {observed[key]!r}"
            )
    return observed


def _default_config_entries(repository_root: Path) -> dict[str, str]:
    if repository_root != REPO_ROOT:
        raise TableCDependencyClosureError(
            "non-default repository roots require explicit config_entries"
        )
    from tools import run_stageb_token_ablation_matrix as token_runner

    entries = {row.row_id: row.config for row in token_runner.ROWS}
    expected_rows = {f"L{index}" for index in range(11)}
    if set(entries) != expected_rows:
        raise TableCDependencyClosureError(
            "token runner ROWS no longer defines exactly L0-L10"
        )
    return dict(sorted(entries.items(), key=lambda item: int(item[0][1:])))


def _normalize_config_entries(
    repository_root: Path,
    config_entries: Mapping[str, str | Path] | None,
) -> dict[str, str]:
    raw = (
        _default_config_entries(repository_root)
        if config_entries is None
        else dict(config_entries)
    )
    expected_rows = {f"L{index}" for index in range(11)}
    if set(raw) != expected_rows:
        raise TableCDependencyClosureError(
            "config entries must define exactly L0-L10"
        )
    normalized: dict[str, str] = {}
    for index in range(11):
        row_id = f"L{index}"
        path = Path(raw[row_id])
        if not path.is_absolute():
            path = repository_root / path
        relative = _relative_path(path, repository_root, label=f"{row_id} config")
        if not relative.startswith("config/") or not relative.endswith(".py"):
            raise TableCDependencyClosureError(
                f"{row_id} config is not a repository config Python file: {relative}"
            )
        normalized[row_id] = relative
    return normalized


def _config_chains(
    repository_root: Path,
    entries: Mapping[str, str],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for index in range(11):
        row_id = f"L{index}"
        try:
            paths = config_import_chain(
                repository_root / entries[row_id], root=repository_root
            )
        except DependencyAuditError as exc:
            raise TableCDependencyClosureError(
                f"cannot compute {row_id} config import chain: {exc}"
            ) from exc
        relative = [
            _relative_path(path, repository_root, label=f"{row_id} config dependency")
            for path in paths
        ]
        if entries[row_id] not in relative:
            raise TableCDependencyClosureError(
                f"{row_id} config chain omitted its entry {entries[row_id]}"
            )
        result[row_id] = relative
    return result


def _artifact_record(path: Path) -> dict[str, Any]:
    return _file_record(path)


def _validate_training_evidence(
    repository_root: Path,
    training_root: Path,
    config_chains: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    training_root = training_root.expanduser().resolve(strict=True)
    if not training_root.is_dir():
        raise TableCDependencyClosureError(
            f"training root is not a directory: {training_root}"
        )
    runs: list[dict[str, Any]] = []
    earliest_start_ns: int | None = None
    earliest_start_utc: str | None = None
    static_records: dict[str, dict[str, Any]] | None = None
    repository_rows: dict[str, list[str]] = {}
    config_rows: dict[str, list[str]] = {}

    for index in range(5):
        row_id = f"L{index}"
        run_id = f"{row_id}:17"
        run_root = (training_root / row_id / "seed17").resolve(strict=True)
        launch_path = run_root / "launch_manifest.json"
        sequence_path = run_root / "sequence_manifest.json"
        postflight_path = run_root / "postflight.json"
        launch = _read_json(launch_path, label=f"{run_id} launch manifest")
        sequence = _read_json(sequence_path, label=f"{run_id} sequence manifest")
        postflight = _read_json(postflight_path, label=f"{run_id} postflight")

        if (
            launch.get("schema") != "pivot.stageb.token_ablation_launch/v2"
            or launch.get("status") != "completed"
            or launch.get("run_id") != run_id
        ):
            raise TableCDependencyClosureError(
                f"{run_id} launch is not an explicitly completed Table-C launch"
            )
        if Path(str(launch.get("repository_root", ""))).resolve(strict=False) != repository_root:
            raise TableCDependencyClosureError(
                f"{run_id} launch repository root differs from this audit"
            )
        if Path(str(launch.get("output_dir", ""))).resolve(strict=False) != run_root:
            raise TableCDependencyClosureError(
                f"{run_id} launch output_dir differs from the required run root"
            )

        started_utc = launch.get("started_at_utc")
        started_ns = _parse_utc_ns(started_utc, label=f"{run_id} started_at_utc")
        if sequence.get("started_at_utc") != started_utc:
            raise TableCDependencyClosureError(
                f"{run_id} launch and sequence start timestamps differ"
            )
        if earliest_start_ns is None or started_ns < earliest_start_ns:
            earliest_start_ns = started_ns
            earliest_start_utc = str(started_utc)

        if (
            sequence.get("schema") != "pivot.stageb.token_ablation_sequence/v1"
            or sequence.get("status") != "completed"
            or sequence.get("run_id") != run_id
            or Path(str(sequence.get("output_dir", ""))).resolve(strict=False)
            != run_root
        ):
            raise TableCDependencyClosureError(
                f"{run_id} sequence is not explicitly completed for this run root"
            )
        phases = sequence.get("phases")
        completed_phases = sequence.get("completed_phases")
        if (
            not isinstance(phases, list)
            or len(phases) != 1
            or not isinstance(phases[0], Mapping)
            or phases[0].get("phase_id") != "joint"
            or not isinstance(completed_phases, list)
            or len(completed_phases) != 1
            or not isinstance(completed_phases[0], Mapping)
            or completed_phases[0].get("phase_id") != "joint"
            or completed_phases[0].get("status") != "completed"
        ):
            raise TableCDependencyClosureError(
                f"{run_id} sequence does not have one completed joint phase"
            )
        if (
            Path(str(phases[0].get("output_dir", ""))).resolve(strict=False)
            != run_root
            or Path(
                str(completed_phases[0].get("output_dir", ""))
            ).resolve(strict=False)
            != run_root
        ):
            raise TableCDependencyClosureError(
                f"{run_id} sequence phase output differs from the run root"
            )

        if (
            postflight.get("schema") != "pivot.stageb.token_ablation_postflight/v2"
            or postflight.get("status") != "passed"
            or postflight.get("run_id") != run_id
        ):
            raise TableCDependencyClosureError(
                f"{run_id} postflight did not explicitly pass"
            )
        input_rehash = postflight.get("input_rehash")
        if not isinstance(input_rehash, Mapping) or input_rehash.get("status") != "passed":
            raise TableCDependencyClosureError(
                f"{run_id} postflight input rehash did not pass"
            )
        observed_postflight = _validate_expected_file_record(
            launch.get("postflight_artifact"),
            postflight_path,
            label=f"{run_id} launch postflight artifact",
        )
        _validate_expected_file_record(
            completed_phases[0].get("postflight"),
            postflight_path,
            label=f"{run_id} sequence postflight artifact",
        )

        inputs = launch.get("inputs")
        if not isinstance(inputs, Mapping):
            raise TableCDependencyClosureError(f"{run_id} launch inputs are missing")
        raw_repository = inputs.get("repository_sources")
        raw_configs = inputs.get("config_dependencies")
        if not isinstance(raw_repository, list) or not raw_repository:
            raise TableCDependencyClosureError(
                f"{run_id} launch has no repository_sources"
            )
        if not isinstance(raw_configs, list) or not raw_configs:
            raise TableCDependencyClosureError(
                f"{run_id} launch has no config_dependencies"
            )

        observed_static: dict[str, dict[str, Any]] = {}
        for record in raw_repository:
            if not isinstance(record, Mapping):
                raise TableCDependencyClosureError(
                    f"{run_id} repository_sources contains a non-object"
                )
            path = Path(str(record.get("path", ""))).resolve(strict=True)
            relative = _relative_path(
                path, repository_root, label=f"{run_id} repository source"
            )
            if relative in observed_static:
                raise TableCDependencyClosureError(
                    f"{run_id} has duplicate repository source {relative}"
                )
            observed = _validate_expected_file_record(
                record, path, label=f"{run_id} repository source"
            )
            observed_static[relative] = {
                "path": str(path),
                "sha256": observed["sha256"],
                "size_bytes": observed["size_bytes"],
                "mtime_ns": observed["mtime_ns"],
            }
            repository_rows.setdefault(relative, []).append(row_id)
        if static_records is None:
            static_records = observed_static
        elif observed_static != static_records:
            raise TableCDependencyClosureError(
                f"{run_id} repository_sources differ from the other L0-L4 launches"
            )

        observed_config_paths: list[str] = []
        for record in raw_configs:
            if not isinstance(record, Mapping):
                raise TableCDependencyClosureError(
                    f"{run_id} config_dependencies contains a non-object"
                )
            path = Path(str(record.get("path", ""))).resolve(strict=True)
            relative = _relative_path(
                path, repository_root, label=f"{run_id} config dependency"
            )
            _validate_expected_file_record(
                record, path, label=f"{run_id} config dependency"
            )
            observed_config_paths.append(relative)
            config_rows.setdefault(relative, []).append(row_id)
        if sorted(observed_config_paths) != sorted(config_chains[row_id]):
            raise TableCDependencyClosureError(
                f"{run_id} launch config_dependencies differ from the recomputed chain"
            )

        runs.append(
            {
                "row_id": row_id,
                "run_id": run_id,
                "run_root": str(run_root),
                "started_at_utc": started_utc,
                "started_at_ns": started_ns,
                "launch_manifest": _artifact_record(launch_path),
                "sequence_manifest": _artifact_record(sequence_path),
                "postflight": observed_postflight,
            }
        )

    if earliest_start_ns is None or earliest_start_utc is None or static_records is None:
        raise TableCDependencyClosureError("no completed L0-L4 training evidence found")
    return {
        "training_root": str(training_root),
        "required_run_ids": list(COMPLETED_RUN_IDS),
        "earliest_training_start_utc": earliest_start_utc,
        "earliest_training_start_ns": earliest_start_ns,
        "runs": runs,
        "static_repository_sources": [
            {"relative_path": relative, **static_records[relative]}
            for relative in sorted(static_records)
        ],
        "historical_repository_rows": {
            path: sorted(set(rows)) for path, rows in sorted(repository_rows.items())
        },
        "historical_config_rows": {
            path: sorted(set(rows)) for path, rows in sorted(config_rows.items())
        },
    }


def _queue_binding(
    queue_dir: Path,
    *,
    repository_root: Path,
    expected_run_ids: Sequence[str],
    role: str,
    policy: str,
) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    try:
        queue = queue_runner.load_queue(queue_dir)
    except (OSError, ValueError, queue_runner.QueueContractError) as exc:
        raise TableCDependencyClosureError(
            f"{role} queue canonical plan verification failed: {exc}"
        ) from exc
    plan = queue["plan"]
    observed_plan_sha = _canonical_sha256(plan)
    if queue.get("plan_sha256") != observed_plan_sha:
        raise TableCDependencyClosureError(
            f"{role} queue canonical plan SHA-256 mismatch"
        )
    if Path(str(plan.get("repository_root", ""))).resolve(strict=False) != repository_root:
        raise TableCDependencyClosureError(
            f"{role} queue repository root differs from the attested repository"
        )
    plan_items = plan.get("items")
    if not isinstance(plan_items, list):
        raise TableCDependencyClosureError(f"{role} queue plan items are missing")
    observed_ids = [
        item.get("run_id") if isinstance(item, Mapping) else None
        for item in plan_items
    ]
    if observed_ids != list(expected_run_ids):
        raise TableCDependencyClosureError(
            f"{role} queue run order differs from the required Table-C plan"
        )
    if any(
        not isinstance(item, Mapping) or item.get("runner") != "token"
        for item in plan_items
    ):
        raise TableCDependencyClosureError(
            f"{role} queue contains a non-token runner item"
        )

    runners = plan.get("runners")
    token_record = runners.get("token") if isinstance(runners, Mapping) else None
    if not isinstance(token_record, Mapping):
        raise TableCDependencyClosureError(
            f"{role} queue has no token runner identity"
        )
    token_path = Path(str(token_record.get("path", ""))).resolve(strict=True)
    _relative_path(token_path, repository_root, label=f"{role} queue token runner")
    observed_runner_sha = _sha256_file(token_path)
    if token_record.get("sha256") != observed_runner_sha:
        raise TableCDependencyClosureError(
            f"{role} queue token runner SHA-256 drift"
        )

    status = queue.get("status")
    completion_verification: dict[str, Any] | None = None
    if role == "completed_l0_l4":
        allowed = {"completed"}
    elif policy == "preflight":
        allowed = {"running", "completed"}
    elif policy == "final":
        allowed = {"completed"}
    else:
        raise TableCDependencyClosureError(f"unsupported verification policy: {policy}")
    if status not in allowed:
        raise TableCDependencyClosureError(
            f"{role} queue status {status!r} is not allowed by {policy} policy"
        )
    if status == "running" and not any(
        item.get("status") in {"reserved", "launching", "launched", "completed"}
        for item in queue["items"]
    ):
        raise TableCDependencyClosureError(
            f"{role} running queue has no preflight-advanced item"
        )
    if status == "completed":
        verification = queue_runner.verify_queue(queue_dir)
        if verification.get("status") != "passed" or verification.get("errors"):
            raise TableCDependencyClosureError(
                f"{role} completed queue verification did not pass: "
                f"{verification.get('errors')!r}"
            )
        if (
            verification.get("queue_id") != plan.get("queue_id")
            or verification.get("plan_sha256") != observed_plan_sha
        ):
            raise TableCDependencyClosureError(
                f"{role} completion verification identity mismatch"
            )
        completion_verification = {
            "schema": verification.get("schema"),
            "status": "passed",
            "verified_item_count": len(verification.get("verified_items", [])),
        }

    return {
        "role": role,
        "queue_dir": str(queue_dir),
        "queue_id": plan.get("queue_id"),
        "plan_sha256": observed_plan_sha,
        "plan_sha256_recomputed_from_canonical_json": True,
        "ordered_run_ids": list(observed_ids),
        "token_runner": {
            "path": str(token_path),
            "sha256": observed_runner_sha,
        },
        "observed_status": status,
        "status_policy": (
            "completed_required"
            if role == "completed_l0_l4" or policy == "final"
            else "running_or_completed"
        ),
        "completion_verification": completion_verification,
        "mutable_queue_json_hash_bound": False,
        "identity_source": "canonical immutable plan embedded in mutable queue.json",
    }


def _queue_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "role": record["role"],
        "queue_dir": record["queue_dir"],
        "queue_id": record["queue_id"],
        "plan_sha256": record["plan_sha256"],
        "ordered_run_ids": list(record["ordered_run_ids"]),
        "token_runner": dict(record["token_runner"]),
    }


def _build_closure(
    repository_root: Path,
    training: Mapping[str, Any],
    config_chains: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    static_paths = {
        record["relative_path"] for record in training["static_repository_sources"]
    }
    python_entries = sorted(path for path in static_paths if path.endswith(".py"))
    if not python_entries:
        raise TableCDependencyClosureError(
            "historical repository_sources contain no Python closure entries"
        )
    try:
        recursive_paths = local_python_dependency_paths(
            python_entries, root=repository_root
        )
    except DependencyAuditError as exc:
        raise TableCDependencyClosureError(
            f"cannot compute recursive local-Python closure: {exc}"
        ) from exc
    recursive_relative = [
        _relative_path(path, repository_root, label="recursive Python dependency")
        for path in recursive_paths
    ]
    config_memberships: dict[str, list[str]] = {}
    for row_id, paths in config_chains.items():
        for relative in paths:
            config_memberships.setdefault(relative, []).append(f"config_chain:{row_id}")
    combined = sorted(set(recursive_relative) | set(config_memberships))
    repository_rows = training["historical_repository_rows"]
    config_rows = training["historical_config_rows"]
    records: list[dict[str, Any]] = []
    for relative in combined:
        memberships = list(config_memberships.get(relative, []))
        if relative in recursive_relative:
            memberships.append("recursive_local_python")
        records.append(
            _file_record(
                repository_root / relative,
                root=repository_root,
                memberships=memberships,
                static_repository_source_bound=relative in static_paths,
                historical_repository_rows=repository_rows.get(relative, []),
                historical_config_rows=config_rows.get(relative, []),
            )
        )
    digest_payload = {
        "schema": CLOSURE_DIGEST_SCHEMA,
        "records": [_record_identity(record) for record in records],
    }
    recursive_static = sorted(set(recursive_relative) & static_paths)
    recursive_supplemental = sorted(set(recursive_relative) - static_paths)
    combined_static = sorted(set(combined) & static_paths)
    combined_supplemental = sorted(set(combined) - static_paths)
    return {
        "algorithm": {
            "implementation": "tools/stageb_dependency_audit.py",
            "parser": "Python ast static Import/ImportFrom traversal",
            "recursive_entry_paths": python_entries,
            "config_entry_paths": dict(config_chains),
            "ordering": "repository-relative POSIX path ascending",
            "digest_payload_schema": CLOSURE_DIGEST_SCHEMA,
        },
        "recursive_local_python": {
            "paths": recursive_relative,
            "path_count": len(recursive_relative),
            "static_repository_source_bound_paths": recursive_static,
            "static_repository_source_bound_count": len(recursive_static),
            "supplemental_only_paths": recursive_supplemental,
            "supplemental_only_count": len(recursive_supplemental),
        },
        "combined_with_l0_l10_config_chains": {
            "paths": combined,
            "path_count": len(combined),
            "static_repository_source_bound_paths": combined_static,
            "static_repository_source_bound_count": len(combined_static),
            "supplemental_only_paths": combined_supplemental,
            "supplemental_only_count": len(combined_supplemental),
        },
        "file_records": records,
        "canonical_closure_sha256": _canonical_sha256(digest_payload),
    }


def _require_supplemental_mtimes_predate_training(
    closure: Mapping[str, Any],
    *,
    earliest_start_ns: int,
) -> None:
    late: list[dict[str, Any]] = []
    for record in closure["file_records"]:
        if (
            record["supplemental_only_relative_to_repository_sources"]
            and int(record["mtime_ns"]) >= int(earliest_start_ns)
        ):
            late.append(
                {
                    "relative_path": record["relative_path"],
                    "mtime_ns": int(record["mtime_ns"]),
                }
            )
    if late:
        rendered = ", ".join(item["relative_path"] for item in late[:10])
        raise TableCDependencyClosureError(
            "supplemental file mtime does not predate the earliest L0-L4 "
            f"training start: {rendered}"
        )


def _auditor_source_records() -> list[dict[str, Any]]:
    paths = (
        Path(__file__).resolve(strict=True),
        (REPO_ROOT / "tools/stageb_dependency_audit.py").resolve(strict=True),
        (REPO_ROOT / "tools/run_stageb_serial_matrix_queue.py").resolve(strict=True),
    )
    return [_file_record(path) for path in paths]


def _attestation_digest(payload: Mapping[str, Any]) -> str:
    view = dict(payload)
    view.pop("attestation_sha256", None)
    return _canonical_sha256(
        {"schema": ATTESTATION_DIGEST_SCHEMA, "attestation": view}
    )


def create_attestation(
    output_path: Path,
    *,
    repository_root: Path = REPO_ROOT,
    completed_queue_dir: Path = DEFAULT_COMPLETED_QUEUE,
    remaining_queue_dir: Path = DEFAULT_REMAINING_QUEUE,
    training_root: Path = DEFAULT_TRAINING_ROOT,
    config_entries: Mapping[str, str | Path] | None = None,
    policy: str = "preflight",
) -> dict[str, Any]:
    if policy not in {"preflight", "final"}:
        raise TableCDependencyClosureError(
            f"unsupported attestation creation policy: {policy}"
        )
    repository_root = repository_root.expanduser().resolve(strict=True)
    entries = _normalize_config_entries(repository_root, config_entries)
    chains = _config_chains(repository_root, entries)
    training = _validate_training_evidence(repository_root, training_root, chains)
    closure = _build_closure(repository_root, training, chains)
    _require_supplemental_mtimes_predate_training(
        closure, earliest_start_ns=int(training["earliest_training_start_ns"])
    )
    completed_queue = _queue_binding(
        completed_queue_dir,
        repository_root=repository_root,
        expected_run_ids=COMPLETED_RUN_IDS,
        role="completed_l0_l4",
        policy="final",
    )
    remaining_queue = _queue_binding(
        remaining_queue_dir,
        repository_root=repository_root,
        expected_run_ids=REMAINING_RUN_IDS,
        role="remaining_table_c",
        policy=policy,
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at_utc": _utc_now(),
        "repository_root": str(repository_root),
        "evidence_class": "supplemental_retrospective_filesystem_attestation",
        "claim_scope": {
            "retroactively_launch_binds_omitted_files": False,
            "historical_launch_manifests_modified": False,
            "queue_identity_binds_mutable_queue_json_bytes": False,
            "queue_identity_binds_canonical_immutable_plan": True,
        },
        "limitations": {
            "primary": (
                "This supplemental attestation does not retroactively make omitted "
                "dependencies launch-bound."
            ),
            "mtime_evidence": (
                "For files omitted from historical repository_sources, a present-day "
                "hash plus a pre-training filesystem mtime is retrospective evidence, "
                "not cryptographic proof of the exact bytes read during training."
            ),
            "static_import_analysis": (
                "The closure covers repository-local Python imports visible to static "
                "AST Import/ImportFrom analysis; runtime-generated or plugin imports are "
                "outside this claim."
            ),
        },
        "config_entries": entries,
        "config_import_chains": chains,
        "training_evidence": training,
        "dependency_closure": closure,
        "queues": {
            "completed_l0_l4": completed_queue,
            "remaining_table_c": remaining_queue,
        },
        "auditor_sources": _auditor_source_records(),
    }
    payload["attestation_sha256"] = _attestation_digest(payload)
    _write_json_exclusive(output_path, payload)
    return payload


def _compare_artifact_record(
    expected: Mapping[str, Any], observed: Mapping[str, Any], *, label: str
) -> None:
    for key in ("path", "sha256", "size_bytes", "mtime_ns"):
        if expected.get(key) != observed.get(key):
            raise TableCDependencyClosureError(
                f"{label} artifact identity drift in {key}"
            )


def _compare_training_evidence(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> None:
    for key in (
        "training_root",
        "required_run_ids",
        "earliest_training_start_utc",
        "earliest_training_start_ns",
        "static_repository_sources",
        "historical_repository_rows",
        "historical_config_rows",
    ):
        if expected.get(key) != observed.get(key):
            raise TableCDependencyClosureError(
                f"training evidence drift in {key}"
            )
    expected_runs = expected.get("runs")
    observed_runs = observed.get("runs")
    if not isinstance(expected_runs, list) or not isinstance(observed_runs, list):
        raise TableCDependencyClosureError("training evidence runs are missing")
    if len(expected_runs) != len(observed_runs):
        raise TableCDependencyClosureError("training evidence run count drift")
    for old, new in zip(expected_runs, observed_runs):
        for key in ("row_id", "run_id", "run_root", "started_at_utc", "started_at_ns"):
            if old.get(key) != new.get(key):
                raise TableCDependencyClosureError(
                    f"training evidence drift for {old.get('run_id')} in {key}"
                )
        for key in ("launch_manifest", "sequence_manifest", "postflight"):
            _compare_artifact_record(
                old[key], new[key], label=f"{old.get('run_id')} {key}"
            )


def _verify_auditor_sources(raw: Any) -> None:
    if not isinstance(raw, list) or not raw:
        raise TableCDependencyClosureError("attestation auditor_sources are missing")
    for record in raw:
        if not isinstance(record, Mapping):
            raise TableCDependencyClosureError("auditor_sources contains a non-object")
        path = Path(str(record.get("path", ""))).resolve(strict=True)
        observed = _file_record(path)
        for key in ("sha256", "size_bytes", "mtime_ns"):
            if record.get(key) != observed[key]:
                raise TableCDependencyClosureError(
                    f"auditor source identity drift for {path} in {key}"
                )


def verify_attestation(
    attestation_path: Path,
    *,
    policy: str = "preflight",
    config_entries: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    attestation_path = attestation_path.expanduser().resolve(strict=True)
    payload = _read_json(attestation_path, label="Table-C dependency attestation")
    if payload.get("schema") != SCHEMA:
        raise TableCDependencyClosureError(
            f"unsupported attestation schema: {payload.get('schema')!r}"
        )
    if payload.get("attestation_sha256") != _attestation_digest(payload):
        raise TableCDependencyClosureError("attestation canonical SHA-256 mismatch")
    claim_scope = payload.get("claim_scope")
    limitations = payload.get("limitations")
    if (
        not isinstance(claim_scope, Mapping)
        or claim_scope.get("retroactively_launch_binds_omitted_files") is not False
        or claim_scope.get("historical_launch_manifests_modified") is not False
        or not isinstance(limitations, Mapping)
        or "does not retroactively make omitted dependencies launch-bound"
        not in str(limitations.get("primary", ""))
    ):
        raise TableCDependencyClosureError(
            "attestation no longer states its mandatory supplemental-evidence limitation"
        )
    _verify_auditor_sources(payload.get("auditor_sources"))

    repository_root = Path(str(payload.get("repository_root", ""))).resolve(strict=True)
    stored_entries = payload.get("config_entries")
    if not isinstance(stored_entries, Mapping):
        raise TableCDependencyClosureError("attestation config_entries are missing")
    current_entries = _normalize_config_entries(repository_root, config_entries)
    if current_entries != dict(stored_entries):
        raise TableCDependencyClosureError("Table-C config entry mapping drift")
    chains = _config_chains(repository_root, current_entries)
    if chains != payload.get("config_import_chains"):
        raise TableCDependencyClosureError("Table-C config import-chain set drift")

    expected_training = payload.get("training_evidence")
    if not isinstance(expected_training, Mapping):
        raise TableCDependencyClosureError("attestation training_evidence is missing")
    observed_training = _validate_training_evidence(
        repository_root,
        Path(str(expected_training.get("training_root", ""))),
        chains,
    )
    _compare_training_evidence(expected_training, observed_training)

    expected_closure = payload.get("dependency_closure")
    if not isinstance(expected_closure, Mapping):
        raise TableCDependencyClosureError("attestation dependency_closure is missing")
    observed_closure = _build_closure(repository_root, observed_training, chains)
    expected_paths = expected_closure.get("combined_with_l0_l10_config_chains", {}).get(
        "paths"
    )
    observed_paths = observed_closure["combined_with_l0_l10_config_chains"]["paths"]
    if expected_paths != observed_paths:
        raise TableCDependencyClosureError(
            "dependency closure-set drift: recomputed paths differ from attestation"
        )
    expected_recursive = expected_closure.get("recursive_local_python", {}).get("paths")
    if expected_recursive != observed_closure["recursive_local_python"]["paths"]:
        raise TableCDependencyClosureError(
            "recursive local-Python closure-set drift"
        )
    expected_records = expected_closure.get("file_records")
    observed_records = observed_closure["file_records"]
    if not isinstance(expected_records, list) or len(expected_records) != len(
        observed_records
    ):
        raise TableCDependencyClosureError("dependency closure record-count drift")
    for expected, observed in zip(expected_records, observed_records):
        if not isinstance(expected, Mapping):
            raise TableCDependencyClosureError(
                "dependency closure contains a non-object record"
            )
        for key in (
            "relative_path",
            "path",
            "sha256",
            "size_bytes",
            "mtime_ns",
            "memberships",
            "static_repository_source_bound",
            "supplemental_only_relative_to_repository_sources",
            "historical_launch_bindings",
        ):
            if expected.get(key) != observed.get(key):
                raise TableCDependencyClosureError(
                    "dependency file identity or membership drift for "
                    f"{expected.get('relative_path', '<unknown>')} in {key}"
                )
    if (
        expected_closure.get("canonical_closure_sha256")
        != observed_closure["canonical_closure_sha256"]
    ):
        raise TableCDependencyClosureError("canonical closure SHA-256 drift")
    _require_supplemental_mtimes_predate_training(
        observed_closure,
        earliest_start_ns=int(observed_training["earliest_training_start_ns"]),
    )

    queues = payload.get("queues")
    if not isinstance(queues, Mapping):
        raise TableCDependencyClosureError("attestation queue bindings are missing")
    expected_completed = queues.get("completed_l0_l4")
    expected_remaining = queues.get("remaining_table_c")
    if not isinstance(expected_completed, Mapping) or not isinstance(
        expected_remaining, Mapping
    ):
        raise TableCDependencyClosureError("attestation queue identity is incomplete")
    observed_completed = _queue_binding(
        Path(str(expected_completed.get("queue_dir", ""))),
        repository_root=repository_root,
        expected_run_ids=COMPLETED_RUN_IDS,
        role="completed_l0_l4",
        policy="final",
    )
    observed_remaining = _queue_binding(
        Path(str(expected_remaining.get("queue_dir", ""))),
        repository_root=repository_root,
        expected_run_ids=REMAINING_RUN_IDS,
        role="remaining_table_c",
        policy=policy,
    )
    if _queue_identity(expected_completed) != _queue_identity(observed_completed):
        raise TableCDependencyClosureError("completed L0-L4 queue plan identity drift")
    if _queue_identity(expected_remaining) != _queue_identity(observed_remaining):
        raise TableCDependencyClosureError("remaining Table-C queue plan identity drift")

    return {
        "schema": VERIFICATION_SCHEMA,
        "status": "passed",
        "verified_at_utc": _utc_now(),
        "policy": policy,
        "attestation": str(attestation_path),
        "attestation_sha256": payload["attestation_sha256"],
        "canonical_closure_sha256": observed_closure[
            "canonical_closure_sha256"
        ],
        "closure_path_count": len(observed_records),
        "completed_queue_status": observed_completed["observed_status"],
        "remaining_queue_status": observed_remaining["observed_status"],
        "supplemental_limitation_reaffirmed": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    create = subparsers.add_parser("create", help="write one fresh attestation")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    create.add_argument(
        "--completed-queue-dir", type=Path, default=DEFAULT_COMPLETED_QUEUE
    )
    create.add_argument(
        "--remaining-queue-dir", type=Path, default=DEFAULT_REMAINING_QUEUE
    )
    create.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    create.add_argument(
        "--policy",
        choices=("preflight", "final"),
        default="preflight",
        help="preflight permits a running remaining queue; final requires completion",
    )

    verify = subparsers.add_parser("verify", help="rehash and revalidate an attestation")
    verify.add_argument("attestation", type=Path)
    verify.add_argument(
        "--policy",
        choices=("preflight", "final"),
        default="preflight",
        help="preflight permits a running remaining queue; final requires completion",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "create":
            payload = create_attestation(
                args.output,
                repository_root=args.repository_root,
                completed_queue_dir=args.completed_queue_dir,
                remaining_queue_dir=args.remaining_queue_dir,
                training_root=args.training_root,
                policy=args.policy,
            )
            result = {
                "status": "created",
                "output": str(args.output.expanduser().resolve(strict=True)),
                "attestation_sha256": payload["attestation_sha256"],
                "canonical_closure_sha256": payload["dependency_closure"][
                    "canonical_closure_sha256"
                ],
                "closure_path_count": payload["dependency_closure"][
                    "combined_with_l0_l10_config_chains"
                ]["path_count"],
            }
        else:
            result = verify_attestation(args.attestation, policy=args.policy)
    except (OSError, ValueError, DependencyAuditError, TableCDependencyClosureError) as exc:
        print(
            json.dumps(
                {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
