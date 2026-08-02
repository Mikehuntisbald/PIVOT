#!/usr/bin/env python3
"""Build and verify an immutable execution tree for Table-C validation.

The formal matrix queue normally fingerprints files in the editable repository.
For a long GPU run that is unnecessarily fragile: an authorized source edit made
after launch invalidates the whole queue.  This builder copies the exact 77-file
controller/evaluator union into a content-addressed, read-only repository tree.
The tree exposes the canonical artifact directory through one checked ``outputs``
symlink, so queue artifacts still land at their normal absolute paths.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PARENT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/source_snapshots/table_c_matrix_validation_v1"
)
EXECUTION_PARENT = Path(
    "/tmp/pivot-stageb-execution/table_c_matrix_validation_v1"
)
SNAPSHOT_SCHEMA = "pivot.stageb.table_c_matrix_execution_snapshot/v1"
SNAPSHOT_DIGEST_SCHEMA = (
    "pivot.stageb.table_c_matrix_execution_snapshot_digest/v1"
)
VERIFICATION_SCHEMA = (
    "pivot.stageb.table_c_matrix_execution_snapshot_verification/v1"
)
BINDING_SCHEMA = "pivot.stageb.table_c_matrix_execution_snapshot_binding/v1"
BINDING_DIGEST_SCHEMA = (
    "pivot.stageb.table_c_matrix_execution_snapshot_binding_digest/v1"
)
INVENTORY_SCHEMA = "pivot.stageb.table_c_matrix_execution_inventory/v1"

EVALUATION_SOURCE_COUNT = 75
CONTROLLER_SOURCE_COUNT = 12
SOURCE_UNION_COUNT = 77
SOURCE_OVERLAP_COUNT = 10
PROFILE_SUPPORT_SOURCE_COUNT = 2
SNAPSHOT_FILE_COUNT = SOURCE_UNION_COUNT + PROFILE_SUPPORT_SOURCE_COUNT


class ExecutionSnapshotError(RuntimeError):
    """The immutable Table-C execution snapshot failed closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ExecutionSnapshotError(f"value is not canonical JSON: {exc}") from exc
    return rendered.encode("ascii")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _snapshot_digest(payload: Mapping[str, Any]) -> str:
    view = dict(payload)
    view.pop("snapshot_sha256", None)
    return _canonical_sha(
        {"schema": SNAPSHOT_DIGEST_SCHEMA, "snapshot": view}
    )


def _binding_digest(payload: Mapping[str, Any]) -> str:
    view = dict(payload)
    view.pop("binding_sha256", None)
    return _canonical_sha(
        {"schema": BINDING_DIGEST_SCHEMA, "binding": view}
    )


def _strict_json(data: bytes, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("ascii"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ExecutionSnapshotError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExecutionSnapshotError(f"{label} is not a JSON object")
    return value


def _file_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    raw = os.lstat(path)
    if stat.S_ISLNK(raw.st_mode) or not stat.S_ISREG(raw.st_mode):
        raise ExecutionSnapshotError(f"source is not a regular file: {path}")
    digest = hashlib.sha256()
    size = 0
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.stat(path, follow_symlinks=False)
    identity = lambda value: (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )
    if identity(before) != identity(after) or identity(after) != identity(current):
        raise ExecutionSnapshotError(f"source changed while hashing: {path}")
    if size != int(after.st_size):
        raise ExecutionSnapshotError(f"source size changed while hashing: {path}")
    return {
        "path": str(path),
        "size_bytes": size,
        "mtime_ns": int(after.st_mtime_ns),
        "sha256": digest.hexdigest(),
        "executable": bool(after.st_mode & 0o111),
    }


_INVENTORY_PROGRAM = r'''
import hashlib
import json
from pathlib import Path
from tools import run_stageb_matrix_validation_queue as queue

root = queue.REPO_ROOT.resolve(strict=True)
evaluation = queue._child_evaluation_source_paths(
    evaluation_runner=queue.DEFAULT_EVALUATION_RUNNER.resolve(strict=True),
    injected_paths=None,
)
controller = queue._controller_source_paths()
profile_support = queue._profile_support_source_paths()
evaluation_set = set(evaluation)
controller_set = set(controller)
union = sorted(evaluation_set | controller_set, key=str)
records = []
for path in union:
    resolved = path.resolve(strict=True)
    data = resolved.read_bytes()
    info = resolved.stat()
    records.append({
        "relative_path": resolved.relative_to(root).as_posix(),
        "size_bytes": len(data),
        "mtime_ns": info.st_mtime_ns,
        "sha256": hashlib.sha256(data).hexdigest(),
        "executable": bool(info.st_mode & 0o111),
        "roles": [
            role for role, present in (
                ("evaluation", resolved in evaluation_set),
                ("controller", resolved in controller_set),
            ) if present
        ],
    })
print(json.dumps({
    "schema": "pivot.stageb.table_c_matrix_execution_inventory/v1",
    "repository_root": str(root),
    "evaluation_source_count": len(evaluation_set),
    "controller_source_count": len(controller_set),
    "source_union_count": len(union),
    "source_overlap_count": len(evaluation_set & controller_set),
    "profile_support_source_count": len(profile_support),
    "sources": records,
    "profile_support_sources": [
        {
            "relative_path": path.resolve(strict=True).relative_to(root).as_posix(),
            "size_bytes": path.resolve(strict=True).stat().st_size,
            "mtime_ns": path.resolve(strict=True).stat().st_mtime_ns,
            "sha256": hashlib.sha256(path.resolve(strict=True).read_bytes()).hexdigest(),
            "executable": bool(path.resolve(strict=True).stat().st_mode & 0o111),
            "roles": ["profile_support"],
        }
        for path in profile_support
    ],
}, sort_keys=True, separators=(",", ":")))
'''


def _inventory_from_repository(repository_root: Path) -> dict[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", _INVENTORY_PROGRAM],
        cwd=root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0:
        raise ExecutionSnapshotError(
            "cannot inventory execution repository: "
            + (completed.stderr.strip() or f"exit {completed.returncode}")
        )
    payload = _strict_json(
        completed.stdout.encode("ascii"), label="execution inventory"
    )
    if (
        payload.get("schema") != INVENTORY_SCHEMA
        or payload.get("repository_root") != str(root)
        or payload.get("evaluation_source_count") != EVALUATION_SOURCE_COUNT
        or payload.get("controller_source_count") != CONTROLLER_SOURCE_COUNT
        or payload.get("source_union_count") != SOURCE_UNION_COUNT
        or payload.get("source_overlap_count") != SOURCE_OVERLAP_COUNT
        or payload.get("profile_support_source_count")
        != PROFILE_SUPPORT_SOURCE_COUNT
    ):
        raise ExecutionSnapshotError("execution inventory cardinality drifted")
    sources = payload.get("sources")
    if not isinstance(sources, list) or len(sources) != SOURCE_UNION_COUNT:
        raise ExecutionSnapshotError("execution inventory source list drifted")
    relative = [record.get("relative_path") for record in sources]
    if relative != sorted(relative) or len(relative) != len(set(relative)):
        raise ExecutionSnapshotError("execution inventory is not deterministic")
    for record in sources:
        if (
            not isinstance(record, Mapping)
            or set(record)
            != {
                "relative_path",
                "size_bytes",
                "mtime_ns",
                "sha256",
                "executable",
                "roles",
            }
            or record.get("roles")
            not in (["evaluation"], ["controller"], ["evaluation", "controller"])
        ):
            raise ExecutionSnapshotError("execution inventory record is invalid")
    support = payload.get("profile_support_sources")
    if (
        not isinstance(support, list)
        or len(support) != PROFILE_SUPPORT_SOURCE_COUNT
        or [record.get("relative_path") for record in support]
        != sorted(record.get("relative_path") for record in support)
    ):
        raise ExecutionSnapshotError("profile-support inventory drifted")
    for record in support:
        if (
            not isinstance(record, Mapping)
            or set(record)
            != {
                "relative_path",
                "size_bytes",
                "mtime_ns",
                "sha256",
                "executable",
                "roles",
            }
            or record.get("roles") != ["profile_support"]
        ):
            raise ExecutionSnapshotError(
                "profile-support inventory record is invalid"
            )
    return payload


def _content_identity(inventory: Mapping[str, Any]) -> str:
    return _canonical_sha(
        {
            "schema": INVENTORY_SCHEMA,
            "sources": [
                {
                    key: record[key]
                    for key in (
                        "relative_path",
                        "size_bytes",
                        "mtime_ns",
                        "sha256",
                        "executable",
                        "roles",
                    )
                }
                for record in inventory["sources"]
            ],
            "profile_support_sources": [
                {
                    key: record[key]
                    for key in (
                        "relative_path",
                        "size_bytes",
                        "mtime_ns",
                        "sha256",
                        "executable",
                        "roles",
                    )
                }
                for record in inventory["profile_support_sources"]
            ],
        }
    )


def _copy_stable_source(
    source: Path, destination: Path, expected: Mapping[str, Any]
) -> None:
    observed = _file_record(source)
    if any(
        observed[key] != expected[key]
        for key in ("size_bytes", "mtime_ns", "sha256", "executable")
    ):
        raise ExecutionSnapshotError(f"live source drifted before copy: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        before = os.fstat(descriptor)
        with destination.open("xb") as output:
            while True:
                chunk = os.read(descriptor, 4 * 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = _file_record(source)
    if (
        int(before.st_size) != int(after.st_size)
        or int(before.st_mtime_ns) != int(after.st_mtime_ns)
        or any(
            current[key] != expected[key]
            for key in ("size_bytes", "mtime_ns", "sha256", "executable")
        )
    ):
        raise ExecutionSnapshotError(f"live source drifted during copy: {source}")
    os.chmod(destination, 0o755 if expected["executable"] else 0o644)
    os.utime(
        destination,
        ns=(int(expected["mtime_ns"]), int(expected["mtime_ns"])),
        follow_symlinks=False,
    )
    copied = _file_record(destination)
    if any(
        copied[key] != expected[key]
        for key in ("size_bytes", "mtime_ns", "sha256", "executable")
    ):
        raise ExecutionSnapshotError(f"copied source identity drifted: {destination}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:  # pragma: no cover - required Linux runtime
        raise ExecutionSnapshotError(
            "atomic execution snapshot publication requires Linux renameat2"
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) == 0:
        return
    number = ctypes.get_errno()
    if number == errno.EEXIST:
        raise ExecutionSnapshotError(
            f"execution snapshot appeared concurrently: {destination}"
        )
    raise OSError(number, os.strerror(number), str(destination))


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    rendered = _canonical_bytes(payload) + b"\n"
    with path.open("xb") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": str(path),
        "size_bytes": len(rendered),
        "sha256": hashlib.sha256(rendered).hexdigest(),
    }


def _lock_tree(root: Path, sources: Sequence[Mapping[str, Any]]) -> None:
    for record in sources:
        path = root / str(record["relative_path"])
        os.chmod(path, 0o555 if record["executable"] else 0o444)
    directories = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_dir() and not path.is_symlink()
        ),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        os.chmod(directory, 0o555)
    manifest = root / "snapshot.json"
    if manifest.exists():
        os.chmod(manifest, 0o444)
    os.chmod(root, 0o555)


def _manifest_sources_from_inventory(
    inventory: Mapping[str, Any],
    *,
    key: str,
    snapshot_root: Path,
    execution_root: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in inventory[key]:
        records.append(
            {
                **dict(source),
                "live_path": str(REPO_ROOT / str(source["relative_path"])),
                "archive_path": str(snapshot_root / str(source["relative_path"])),
                "execution_path": str(execution_root / str(source["relative_path"])),
            }
        )
    return records


def _read_manifest(snapshot_root: Path) -> tuple[bytes, dict[str, Any]]:
    path = snapshot_root / "snapshot.json"
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ExecutionSnapshotError(f"cannot read execution snapshot: {exc}") from exc
    payload = _strict_json(data, label="execution snapshot manifest")
    if data != _canonical_bytes(payload) + b"\n":
        raise ExecutionSnapshotError("execution snapshot manifest is not canonical")
    return data, payload


def _project_inventory_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            key: record[key]
            for key in (
                "relative_path",
                "size_bytes",
                "mtime_ns",
                "sha256",
                "executable",
                "roles",
            )
        }
        for record in records
    ]


def _verify_execution_tree(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    execution_root = Path(str(manifest.get("execution_root", "")))
    if not execution_root.is_absolute():
        raise ExecutionSnapshotError("execution tree path is not absolute")
    try:
        root = execution_root.resolve(strict=True)
    except OSError as exc:
        raise ExecutionSnapshotError("execution tree is unavailable") from exc
    if root != execution_root or root.name != manifest.get("content_identity_sha256"):
        raise ExecutionSnapshotError("execution tree path drifted")
    outputs = root / "outputs"
    expected_outputs = (REPO_ROOT / "outputs").resolve(strict=True)
    if (
        not outputs.is_symlink()
        or os.readlink(outputs) != str(REPO_ROOT / "outputs")
        or outputs.resolve(strict=True) != expected_outputs
    ):
        raise ExecutionSnapshotError("execution tree outputs link drifted")
    records = [
        *manifest["sources"],
        *manifest["profile_support_sources"],
    ]
    for record in records:
        path = root / str(record["relative_path"])
        observed = _file_record(path)
        if str(path) != record.get("execution_path") or any(
            observed[key] != record.get(key)
            for key in ("size_bytes", "mtime_ns", "sha256", "executable")
        ):
            raise ExecutionSnapshotError(
                f"materialized execution source drifted: {record.get('relative_path')}"
            )
    inventory = _inventory_from_repository(root)
    if (
        inventory["sources"] != _project_inventory_records(manifest["sources"])
        or inventory["profile_support_sources"]
        != _project_inventory_records(manifest["profile_support_sources"])
    ):
        raise ExecutionSnapshotError("materialized dependency replay drifted")
    return inventory


def _materialize_execution_tree(manifest: Mapping[str, Any]) -> dict[str, Any]:
    destination = Path(str(manifest.get("execution_root", "")))
    if not destination.is_absolute():
        raise ExecutionSnapshotError("execution tree path is not absolute")
    if destination.exists():
        return _verify_execution_tree(manifest)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = parent / f".{destination.name}.creating-{uuid.uuid4().hex}"
    stage.mkdir(mode=0o755)
    records = [
        *manifest["sources"],
        *manifest["profile_support_sources"],
    ]
    try:
        for record in records:
            relative = Path(str(record["relative_path"]))
            _copy_stable_source(
                Path(str(record["archive_path"])), stage / relative, record
            )
        os.symlink(
            str(REPO_ROOT / "outputs"),
            stage / "outputs",
            target_is_directory=True,
        )
        staged_inventory = _inventory_from_repository(stage)
        if (
            staged_inventory["sources"]
            != _project_inventory_records(manifest["sources"])
            or staged_inventory["profile_support_sources"]
            != _project_inventory_records(manifest["profile_support_sources"])
        ):
            raise ExecutionSnapshotError("staged materialization replay drifted")
        _lock_tree(stage, records)
        _fsync_directory(stage)
        _rename_noreplace(stage, destination)
        _fsync_directory(parent)
    except BaseException:
        if stage.exists():
            for path in stage.rglob("*"):
                if not path.is_symlink():
                    try:
                        os.chmod(path, 0o755 if path.is_dir() else 0o644)
                    except OSError:
                        pass
            os.chmod(stage, 0o755)
            shutil.rmtree(stage)
        raise
    return _verify_execution_tree(manifest)


def verify_snapshot(
    snapshot_root: Path, *, require_live_parity: bool = False
) -> dict[str, Any]:
    root = snapshot_root.expanduser().resolve(strict=True)
    data, manifest = _read_manifest(root)
    if (
        manifest.get("schema") != SNAPSHOT_SCHEMA
        or manifest.get("status") != "sealed"
        or manifest.get("snapshot_root") != str(root)
        or manifest.get("snapshot_sha256") != _snapshot_digest(manifest)
        or manifest.get("source_count") != SOURCE_UNION_COUNT
        or manifest.get("profile_support_source_count")
        != PROFILE_SUPPORT_SOURCE_COUNT
        or manifest.get("snapshot_file_count") != SNAPSHOT_FILE_COUNT
        or root.name != manifest.get("content_identity_sha256")
        or Path(str(manifest.get("execution_root", ""))).name
        != manifest.get("content_identity_sha256")
    ):
        raise ExecutionSnapshotError("execution snapshot manifest drifted")
    if (
        manifest.get("outputs_symlink_target") != str(REPO_ROOT / "outputs")
        or manifest.get("outputs_resolved_target")
        != str((REPO_ROOT / "outputs").resolve(strict=True))
    ):
        raise ExecutionSnapshotError("execution snapshot outputs target drifted")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or len(sources) != SOURCE_UNION_COUNT:
        raise ExecutionSnapshotError("execution snapshot sources drifted")
    relative = [record.get("relative_path") for record in sources]
    if relative != sorted(relative) or len(relative) != len(set(relative)):
        raise ExecutionSnapshotError("execution snapshot ordering drifted")
    for record in sources:
        path = root / str(record["relative_path"])
        observed = _file_record(path)
        if str(path) != record.get("archive_path") or any(
            observed[key] != record.get(key)
            for key in ("size_bytes", "mtime_ns", "sha256", "executable")
        ):
            raise ExecutionSnapshotError(
                f"execution snapshot source drifted: {record.get('relative_path')}"
            )
    support = manifest.get("profile_support_sources")
    if not isinstance(support, list) or len(support) != PROFILE_SUPPORT_SOURCE_COUNT:
        raise ExecutionSnapshotError("execution snapshot profile support drifted")
    support_relative = [record.get("relative_path") for record in support]
    if (
        support_relative != sorted(support_relative)
        or set(support_relative) & set(relative)
    ):
        raise ExecutionSnapshotError("profile-support snapshot ordering drifted")
    for record in support:
        path = root / str(record["relative_path"])
        observed = _file_record(path)
        if str(path) != record.get("archive_path") or any(
            observed[key] != record.get(key)
            for key in ("size_bytes", "mtime_ns", "sha256", "executable")
        ):
            raise ExecutionSnapshotError(
                "execution snapshot profile-support source drifted: "
                f"{record.get('relative_path')}"
            )
    archive_inventory = _inventory_from_repository(root)
    if _content_identity(archive_inventory) != manifest.get(
        "content_identity_sha256"
    ):
        raise ExecutionSnapshotError(
            "execution snapshot content identity differs from dependency replay"
        )
    projected = _project_inventory_records(sources)
    if archive_inventory["sources"] != projected:
        raise ExecutionSnapshotError("snapshot dependency replay drifted")
    projected_support = _project_inventory_records(support)
    if archive_inventory["profile_support_sources"] != projected_support:
        raise ExecutionSnapshotError("snapshot profile-support replay drifted")
    inventory = _materialize_execution_tree(manifest)
    live_parity_count = None
    if require_live_parity:
        live = _inventory_from_repository(REPO_ROOT)
        if (
            live["sources"] != projected
            or live["profile_support_sources"] != projected_support
        ):
            raise ExecutionSnapshotError("live source closure differs from snapshot")
        live_parity_count = len(projected)
    return {
        "schema": VERIFICATION_SCHEMA,
        "status": "passed",
        "verified_at_utc": _utc_now(),
        "snapshot_root": str(root),
        "execution_root": manifest["execution_root"],
        "content_identity_sha256": manifest["content_identity_sha256"],
        "snapshot_sha256": manifest["snapshot_sha256"],
        "snapshot_file_sha256": hashlib.sha256(data).hexdigest(),
        "source_count": len(sources),
        "evaluation_source_count": inventory["evaluation_source_count"],
        "controller_source_count": inventory["controller_source_count"],
        "source_overlap_count": inventory["source_overlap_count"],
        "profile_support_source_count": len(support),
        "snapshot_file_count": len(sources) + len(support),
        "live_parity_required": require_live_parity,
        "live_parity_count": live_parity_count,
    }


def _read_json_path(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ExecutionSnapshotError(f"cannot read {label}: {exc}") from exc
    return data, _strict_json(data, label=label)


def _queue_snapshot_contract(
    manifest: Mapping[str, Any], queue_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = queue_dir.expanduser().resolve(strict=True)
    queue_bytes, queue = _read_json_path(
        directory / "queue.json", label="Table-C matrix queue"
    )
    plan = queue.get("plan")
    if (
        queue.get("schema") != "pivot.stageb.matrix_validation_queue/v1"
        or not isinstance(plan, Mapping)
        or plan.get("schema") != "pivot.stageb.matrix_validation_queue_plan/v1"
        or queue.get("plan_sha256") != _canonical_sha(plan)
        or Path(str(plan.get("queue_dir", ""))).resolve(strict=False) != directory
        or plan.get("repository_root") != manifest.get("execution_root")
        or plan.get("provenance_scope") != "formal"
    ):
        raise ExecutionSnapshotError("matrix queue is not bound to this execution root")

    def expected(role: str) -> list[dict[str, Any]]:
        return [
            {
                "path": record["execution_path"],
                "size_bytes": record["size_bytes"],
                "mtime_ns": record["mtime_ns"],
                "sha256": record["sha256"],
            }
            for record in manifest["sources"]
            if role in record["roles"]
        ]

    expected_evaluation = expected("evaluation")
    expected_controller = expected("controller")
    expected_support = [
        {
            "path": record["execution_path"],
            "size_bytes": record["size_bytes"],
            "mtime_ns": record["mtime_ns"],
            "sha256": record["sha256"],
        }
        for record in manifest["profile_support_sources"]
    ]
    runner = next(
        record
        for record in expected_evaluation
        if record["path"].endswith("/tools/run_stageb_paper_evaluations.py")
    )
    if (
        plan.get("evaluation_sources") != expected_evaluation
        or plan.get("controller_sources") != expected_controller
        or plan.get("profile_support_sources") != expected_support
        or plan.get("evaluation_runner") != runner
    ):
        raise ExecutionSnapshotError("matrix queue source plan differs from snapshot")
    return queue, {
        "path": str(directory / "queue.json"),
        "sha256_at_binding": hashlib.sha256(queue_bytes).hexdigest(),
        "size_bytes_at_binding": len(queue_bytes),
    }


def bind_queue(snapshot_root: Path, queue_dir: Path) -> dict[str, Any]:
    snapshot_report = verify_snapshot(snapshot_root)
    root = Path(snapshot_report["snapshot_root"])
    manifest_bytes, manifest = _read_manifest(root)
    queue, queue_record = _queue_snapshot_contract(manifest, queue_dir)
    items = queue.get("items")
    if (
        queue.get("status") != "waiting_training"
        or queue.get("revision") != 0
        or queue.get("training_attestation") is not None
        or queue.get("final_verification") is not None
        or queue.get("failure") is not None
        or not isinstance(items, list)
        or len(items) != 33
        or any(
            not isinstance(item, Mapping) or item.get("status") != "pending"
            for item in items
        )
    ):
        raise ExecutionSnapshotError(
            "execution snapshot binding requires a pristine revision-0 queue"
        )
    directory = queue_dir.expanduser().resolve(strict=True)
    binding_path = directory / "execution_snapshot_binding.json"
    binding = {
        "schema": BINDING_SCHEMA,
        "status": "bound",
        "created_at_utc": _utc_now(),
        "queue_id": queue["plan"]["queue_id"],
        "plan_sha256": queue["plan_sha256"],
        "predeclared_contract_sha256": queue["predeclared_contract_sha256"],
        "queue": queue_record,
        "snapshot": {
            "path": str(root / "snapshot.json"),
            "file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "file_size_bytes": len(manifest_bytes),
            "snapshot_sha256": manifest["snapshot_sha256"],
            "content_identity_sha256": manifest["content_identity_sha256"],
            "execution_root": manifest["execution_root"],
            "source_count": SOURCE_UNION_COUNT,
            "profile_support_source_count": PROFILE_SUPPORT_SOURCE_COUNT,
        },
    }
    binding["binding_sha256"] = _binding_digest(binding)
    _write_manifest(binding_path, binding)
    return verify_queue_binding(directory)


def verify_queue_binding(queue_dir: Path) -> dict[str, Any]:
    directory = queue_dir.expanduser().resolve(strict=True)
    binding_path = directory / "execution_snapshot_binding.json"
    binding_bytes, binding = _read_json_path(
        binding_path, label="execution snapshot binding"
    )
    snapshot_record = binding.get("snapshot")
    queue_record = binding.get("queue")
    expected_binding_keys = {
        "schema",
        "status",
        "created_at_utc",
        "queue_id",
        "plan_sha256",
        "predeclared_contract_sha256",
        "queue",
        "snapshot",
        "binding_sha256",
    }
    expected_snapshot_keys = {
        "path",
        "file_sha256",
        "file_size_bytes",
        "snapshot_sha256",
        "content_identity_sha256",
        "execution_root",
        "source_count",
        "profile_support_source_count",
    }
    if (
        set(binding) != expected_binding_keys
        or binding.get("schema") != BINDING_SCHEMA
        or binding.get("status") != "bound"
        or binding.get("binding_sha256") != _binding_digest(binding)
        or not isinstance(snapshot_record, Mapping)
        or set(snapshot_record) != expected_snapshot_keys
        or not isinstance(queue_record, Mapping)
        or set(queue_record)
        != {"path", "sha256_at_binding", "size_bytes_at_binding"}
    ):
        raise ExecutionSnapshotError("execution snapshot binding drifted")
    snapshot_path = Path(str(snapshot_record.get("path", "")))
    snapshot_root = snapshot_path.parent
    snapshot_report = verify_snapshot(snapshot_root)
    manifest_bytes, manifest = _read_manifest(snapshot_root)
    queue, _ = _queue_snapshot_contract(manifest, directory)
    if (
        queue_record.get("path") != str(directory / "queue.json")
        or not isinstance(queue_record.get("sha256_at_binding"), str)
        or len(str(queue_record.get("sha256_at_binding"))) != 64
        or not isinstance(queue_record.get("size_bytes_at_binding"), int)
        or queue_record.get("size_bytes_at_binding") <= 0
        or snapshot_path != snapshot_root / "snapshot.json"
        or snapshot_record.get("file_sha256")
        != hashlib.sha256(manifest_bytes).hexdigest()
        or snapshot_record.get("file_size_bytes") != len(manifest_bytes)
        or snapshot_record.get("snapshot_sha256") != manifest["snapshot_sha256"]
        or snapshot_record.get("content_identity_sha256")
        != manifest["content_identity_sha256"]
        or snapshot_record.get("execution_root") != manifest["execution_root"]
        or snapshot_record.get("source_count") != SOURCE_UNION_COUNT
        or snapshot_record.get("profile_support_source_count")
        != PROFILE_SUPPORT_SOURCE_COUNT
        or binding.get("queue_id") != queue["plan"]["queue_id"]
        or binding.get("plan_sha256") != queue["plan_sha256"]
        or binding.get("predeclared_contract_sha256")
        != queue["predeclared_contract_sha256"]
    ):
        raise ExecutionSnapshotError("queue/snapshot binding semantics drifted")
    return {
        "schema": BINDING_SCHEMA,
        "status": "passed",
        "verified_at_utc": _utc_now(),
        "binding_path": str(binding_path),
        "binding_file_sha256": hashlib.sha256(binding_bytes).hexdigest(),
        "binding_sha256": binding["binding_sha256"],
        "queue_id": binding["queue_id"],
        "plan_sha256": binding["plan_sha256"],
        "snapshot_root": snapshot_report["snapshot_root"],
        "execution_root": snapshot_report["execution_root"],
        "content_identity_sha256": snapshot_report["content_identity_sha256"],
    }


def build_snapshot(
    *,
    snapshot_parent: Path = SNAPSHOT_PARENT,
    execution_parent: Path = EXECUTION_PARENT,
) -> dict[str, Any]:
    parent = snapshot_parent.expanduser().resolve(strict=False)
    materialization_parent = execution_parent.expanduser().resolve(strict=False)
    live_inventory = _inventory_from_repository(REPO_ROOT)
    identity = _content_identity(live_inventory)
    destination = parent / identity
    execution_root = materialization_parent / identity
    if destination.exists():
        return verify_snapshot(destination, require_live_parity=True)
    parent.mkdir(parents=True, exist_ok=True)
    stage = parent / f".{identity}.creating-{uuid.uuid4().hex}"
    stage.mkdir(mode=0o755)
    try:
        target_sources = _manifest_sources_from_inventory(
            live_inventory,
            key="sources",
            snapshot_root=destination,
            execution_root=execution_root,
        )
        target_profile_support = _manifest_sources_from_inventory(
            live_inventory,
            key="profile_support_sources",
            snapshot_root=destination,
            execution_root=execution_root,
        )
        for record in [*target_sources, *target_profile_support]:
            relative = Path(str(record["relative_path"]))
            _copy_stable_source(REPO_ROOT / relative, stage / relative, record)
        manifest = {
            "schema": SNAPSHOT_SCHEMA,
            "status": "sealed",
            "created_at_utc": _utc_now(),
            "source_repository_root": str(REPO_ROOT),
            "snapshot_root": str(destination),
            "execution_root": str(execution_root),
            "outputs_symlink_target": str(REPO_ROOT / "outputs"),
            "outputs_resolved_target": str(
                (REPO_ROOT / "outputs").resolve(strict=True)
            ),
            "content_identity_sha256": identity,
            "source_count": SOURCE_UNION_COUNT,
            "profile_support_source_count": PROFILE_SUPPORT_SOURCE_COUNT,
            "snapshot_file_count": SNAPSHOT_FILE_COUNT,
            "evaluation_source_count": EVALUATION_SOURCE_COUNT,
            "controller_source_count": CONTROLLER_SOURCE_COUNT,
            "source_overlap_count": SOURCE_OVERLAP_COUNT,
            "sources": target_sources,
            "profile_support_sources": target_profile_support,
        }
        manifest["snapshot_sha256"] = _snapshot_digest(manifest)
        _write_manifest(stage / "snapshot.json", manifest)
        staged_inventory = _inventory_from_repository(stage)
        if (
            staged_inventory["sources"] != live_inventory["sources"]
            or staged_inventory["profile_support_sources"]
            != live_inventory["profile_support_sources"]
        ):
            raise ExecutionSnapshotError("staged execution dependency replay drifted")
        final_live_inventory = _inventory_from_repository(REPO_ROOT)
        if final_live_inventory != live_inventory:
            raise ExecutionSnapshotError(
                "live execution closure changed during snapshot publication"
            )
        _lock_tree(stage, [*target_sources, *target_profile_support])
        _fsync_directory(stage)
        _rename_noreplace(stage, destination)
        _fsync_directory(parent)
    except BaseException:
        if stage.exists():
            for path in stage.rglob("*"):
                if not path.is_symlink():
                    try:
                        os.chmod(path, 0o755 if path.is_dir() else 0o644)
                    except OSError:
                        pass
            os.chmod(stage, 0o755)
            shutil.rmtree(stage)
        raise
    return verify_snapshot(destination, require_live_parity=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("build", help="publish or reuse the current content snapshot")
    verify = subparsers.add_parser("verify", help="replay one published snapshot")
    verify.add_argument("snapshot_root", type=Path)
    verify.add_argument("--require-live-parity", action="store_true")
    bind = subparsers.add_parser(
        "bind-queue", help="bind one predeclared queue to a durable snapshot"
    )
    bind.add_argument("snapshot_root", type=Path)
    bind.add_argument("queue_dir", type=Path)
    verify_binding = subparsers.add_parser(
        "verify-binding", help="replay a queue/snapshot binding"
    )
    verify_binding.add_argument("queue_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.mode == "build":
            report = build_snapshot()
        elif args.mode == "verify":
            report = verify_snapshot(
                args.snapshot_root,
                require_live_parity=args.require_live_parity,
            )
        elif args.mode == "bind-queue":
            report = bind_queue(args.snapshot_root, args.queue_dir)
        elif args.mode == "verify-binding":
            report = verify_queue_binding(args.queue_dir)
        else:  # pragma: no cover
            parser.error(f"unsupported mode: {args.mode}")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (
        ExecutionSnapshotError,
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
