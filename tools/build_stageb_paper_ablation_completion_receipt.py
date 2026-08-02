#!/usr/bin/env python3
"""Build or verify the fail-closed paper-ablation completion receipt.

Each paper block must have a code-registered semantic adapter.  A block with
an unsealed adapter cannot be represented by arbitrary file records and keeps
the headline final gate closed.  The receipt is deterministic, fresh-only,
and replayed from the canonical artifacts on every verification.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pivot.stageb.paper_ablation_completion_receipt/v1"
REGISTRY_SCHEMA = "pivot.stageb.paper_ablation_completion_adapter_registry/v1"
BLOCKS = ("A", "B", "C", "D", "G0c")
CANONICAL_RECEIPT_PATH = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/attestations/paper_ablation_completion_receipt.json"
)

TABLE_C_FINAL_DEPENDENCY_ATTESTATION = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/attestations/table_c_dependency_closure_final.json"
)
TABLE_C_PREFLIGHT_DEPENDENCY_ATTESTATION = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/attestations/"
    "table_c_dependency_closure_preflight_20260718.json"
)
TABLE_C_CURRENT_POLICY_FINAL_DEPENDENCY_ATTESTATION = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/attestations/"
    "table_c_dependency_closure_final_policy_20260719.json"
)
TABLE_C_AGGREGATION_SPEC = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/attestations/table_c_matrix_validation_input.json"
)
TABLE_C_AGGREGATION_REPORT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/aggregates/table_c_matrix_validation_report.json"
)
TABLE_C_PRETRAINING_RECOVERY_RECEIPT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/recovery/table_c_remaining_28/"
    "L2_seed42_attempt000/recovery_receipt.json"
)
TABLE_C_VALIDATION_RECOVERY_RECEIPT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/recovery/table_c_matrix_validation_v1/"
    "L3_seed17_attempt000/recovery_receipt.json"
)
TABLE_C_VALIDATION_RECOVERY_FILE_SHA256 = (
    "6891ba95523a6dfb6a583bdb12263253419dd51c84a9548f19ed1e8daf626520"
)
TABLE_C_VALIDATION_RECOVERY_RECEIPT_SHA256 = (
    "70f3abc4305cf1b3edf08b12043e82eb4f1f9690311dcc6312a828a9c936ba08"
)
TABLE_C_TRAINING_SNAPSHOT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/attestations/table_c_u1000_training_snapshot_v1"
)
TABLE_C_TRAINING_SNAPSHOT_IDENTITY: Mapping[str, str] = {
    "completion_subreceipt_file_sha256": (
        "b7e467d94d9632dc2f5e44b0d5045b6da9241a0a3420ff9e9a4513e78b045d8c"
    ),
    "completion_subreceipt_sha256": (
        "461397059b3fc87256719926b611c2a8a499f44c40598caaba6565c819f0f48f"
    ),
    "source_snapshot_file_sha256": (
        "11e39a431ad73014ad8ca8308b835a21493c45639ae61e2eb90ca401b0b5d8a2"
    ),
    "source_snapshot_sha256": (
        "b28bcd208c6388e4b305db99a3eb2d4c206607b4b31af0759423937938b10538"
    ),
}
TABLE_C_DEPENDENCY_ATTESTATION_SCHEMA = (
    "pivot.stageb.table_c_dependency_closure_attestation/v1"
)
TABLE_C_DEPENDENCY_ATTESTATION_DIGEST_SCHEMA = (
    "pivot.stageb.table_c_dependency_closure_attestation_digest/v1"
)
TABLE_C_FINALIZATION_SCHEMA = (
    "pivot.stageb.table_c_dependency_closure_attestation_finalization/v1"
)
TABLE_C_FINALIZATION_LINEAGE_EVIDENCE_SCHEMA = (
    "pivot.stageb.table_c_archived_finalization_lineage_evidence/v1"
)
TABLE_C_HISTORICAL_FINALIZER_SOURCE_IDENTITY: Mapping[str, Any] = {
    "path": str(REPO_ROOT / "tools/finalize_stageb_table_c_dependency_closure.py"),
    "size_bytes": 24786,
    "mtime_ns": 1784424954180000000,
    "mtime_utc": "2026-07-19T01:35:54.180000000+00:00",
    "sha256": "b1d281d44d4e8544bac944bfa2f31fb263060e178f7e2f1426dba064413a7856",
}
TABLE_C_DEPENDENCY_CLOSURE_SHA256 = (
    "2f9175c85d3ff061e25c4802265dce885ea5cd5038c17ceaee85aa442a75e966"
)
TABLE_C_DEPENDENCY_ATTESTATION_IDENTITIES: Mapping[str, Mapping[str, str]] = {
    "preflight": {
        "file_sha256": (
            "79d1cb09563a4cd21be0ffec2a3d9cc0193f9fe4a59b645fc765a5b4dd3cd509"
        ),
        "semantic_sha256": (
            "1811a3b3dbb03eb51e8d6129395845cc461c59d02f03235ed1479bff42b8652c"
        ),
    },
    "final": {
        "file_sha256": (
            "5ffac291c2d2036b5fcc710aae773d41b2342f87240a975836049e1e909d6e55"
        ),
        "semantic_sha256": (
            "506875044e34f81c5a3045d7647056ac4caed8c7cf8b0388c0e18488d161b823"
        ),
    },
    "supplemental_current_policy": {
        "file_sha256": (
            "036eca11dcb8df6d7918d2684fedf6371640c5c86e080bada55eeace0997b266"
        ),
        "semantic_sha256": (
            "19449b789786fa1d5bbcd6c76d5fc08c9e0a3116bb21b1dcc7ced340da7bb3ae"
        ),
    },
}
TABLE_C_ARCHIVED_SOURCE_COUNTS: Mapping[str, int] = {
    "dependency_closure": 85,
    "static_repository_source": 9,
    "auditor_source": 3,
    "deduplicated_source": 89,
}
TABLE_C_OLD_ATTESTATION_KEYS = frozenset(
    {
        "schema",
        "created_at_utc",
        "repository_root",
        "evidence_class",
        "claim_scope",
        "limitations",
        "config_entries",
        "config_import_chains",
        "training_evidence",
        "dependency_closure",
        "queues",
        "auditor_sources",
        "attestation_sha256",
    }
)
TABLE_C_STATIC_IDENTITY_KEYS = (
    "schema",
    "repository_root",
    "evidence_class",
    "claim_scope",
    "limitations",
    "config_entries",
    "config_import_chains",
    "training_evidence",
    "dependency_closure",
)
TABLE_C_ADAPTER_ID = "table_c_training_validation_aggregate/v5"
HEADLINE_M0_ADAPTER_ID = "headline_m0_m0n_training_validation_aggregate/v1"
HEADLINE_M0_TRAINING_QUEUE = (
    REPO_ROOT / "outputs/paper_cvpr_v1/queues/headline_m0_training_u23532_v1"
)
HEADLINE_M0N_TRAINING_QUEUE = (
    REPO_ROOT / "outputs/paper_cvpr_v1/queues/headline_m0n_training_u23532_v1"
)
HEADLINE_M0_VALIDATION_QUEUE = (
    REPO_ROOT / "outputs/paper_cvpr_v1/queues/headline_m0_m0n_validation_v1"
)
HEADLINE_M0_VALIDATION_AGGREGATE = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/aggregates/headline_m0_m0n_validation_report.json"
)
TABLE_B_V2_TRAINING_QUEUE = (
    REPO_ROOT / "outputs/paper_cvpr_v1/queues/table_b_v2_training_u1000_v1"
)
TABLE_B_V2_VALIDATION_QUEUE = (
    REPO_ROOT / "outputs/paper_cvpr_v1/queues/table_b_v2_validation_v1"
)
TABLE_B_V2_VALIDATION_AGGREGATE = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/aggregates/table_b_v2_validation_report.json"
)
TABLE_B_V2_ADAPTER_ID = "table_b_v2_formal_training_validation_aggregate/v1"
TABLE_D_TRAINING_QUEUE = (
    REPO_ROOT / "outputs/paper_cvpr_v1/queues/table_d_formal_training_u1000_v1"
)
TABLE_D_VALIDATION_QUEUE = (
    REPO_ROOT / "outputs/paper_cvpr_v1/queues/table_d_matrix_validation_v1"
)
TABLE_D_VALIDATION_AGGREGATE = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/aggregates/table_d_formal_matrix_report.json"
)
TABLE_D_ADAPTER_ID = "table_d_formal_training_validation_aggregate/v1"
G0C_SOAK_QUEUE = (
    REPO_ROOT / "outputs/paper_cvpr_v1/queues/table_a_g0c_soak_u50_v1"
)
G0C_TRAINING_QUEUE = (
    REPO_ROOT / "outputs/paper_cvpr_v1/queues/table_a_g0c_training_u1000_v1"
)
G0C_VALIDATION_QUEUE = (
    REPO_ROOT / "outputs/paper_cvpr_v1/queues/table_a_g0c_validation_v1"
)
G0C_VALIDATION_AGGREGATE = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/table_a/aggregates/validation/"
    "table_a_three_seed.json"
)
G0C_ADAPTER_ID = "table_a_g0c_soak_training_validation_aggregate/v2"


class CompletionReceiptError(RuntimeError):
    """A block adapter or completion receipt failed closed."""


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish source without replacing an existing receipt."""

    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:  # pragma: no cover - required Linux runtime
        raise CompletionReceiptError(
            "atomic receipt publication requires Linux renameat2"
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
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
        raise CompletionReceiptError(
            f"completion receipt appeared concurrently: {destination}"
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


@dataclass(frozen=True)
class BlockAdapter:
    adapter_id: str
    verifier: Callable[[], Mapping[str, Any]] | None
    unsealed_reason: str | None = None


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CompletionReceiptError(
            f"completion payload is not canonical JSON: {exc}"
        ) from exc
    return rendered.encode("ascii")


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    try:
        path = path.expanduser().resolve(strict=True)
        before = path.stat()
        if not path.is_file():
            raise CompletionReceiptError(f"completion artifact is not a file: {path}")
        digest = _sha256_file(path)
        after = path.stat()
    except OSError as exc:
        raise CompletionReceiptError(
            f"cannot bind completion artifact {path}: {exc}"
        ) from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise CompletionReceiptError(
            f"completion artifact changed while hashing: {path}"
        )
    return {
        "path": str(path),
        "sha256": digest,
        "size_bytes": int(after.st_size),
    }


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite constant {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CompletionReceiptError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CompletionReceiptError(f"{label} is not a JSON object: {path}")
    return value


def _strip_volatile(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_volatile(item)
            for key, item in value.items()
            if key
            not in {
                "created_at_utc",
                "validated_at_utc",
                "verified_at_utc",
            }
        }
    if isinstance(value, list):
        return [_strip_volatile(item) for item in value]
    return value


def _artifact_records(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    """Bind a named artifact set so semantic replay can detect concurrent drift."""

    return {name: file_record(path) for name, path in paths.items()}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_stable_artifacts(
    before: Mapping[str, Mapping[str, Any]],
    paths: Mapping[str, Path],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    after = _artifact_records(paths)
    if dict(before) != after:
        raise CompletionReceiptError(
            f"{label} artifacts changed during semantic replay"
        )
    return after


def _table_c_attestation_digest(payload: Mapping[str, Any]) -> str:
    view = copy.deepcopy(dict(payload))
    view.pop("attestation_sha256", None)
    return canonical_json_sha256(
        {
            "schema": TABLE_C_DEPENDENCY_ATTESTATION_DIGEST_SCHEMA,
            "attestation": view,
        }
    )


def _table_c_mtime_utc(mtime_ns: int) -> str:
    seconds, nanoseconds = divmod(int(mtime_ns), 1_000_000_000)
    base = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return f"{base.strftime('%Y-%m-%dT%H:%M:%S')}.{nanoseconds:09d}+00:00"


def _table_c_full_file_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    before = path.stat()
    record = file_record(path)
    after = path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise CompletionReceiptError(
            f"Table-C file changed while recording full identity: {path}"
        )
    return {
        "path": str(path),
        "size_bytes": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "mtime_utc": _table_c_mtime_utc(int(after.st_mtime_ns)),
        "sha256": record["sha256"],
    }


def _table_c_mtime_file_record(path: Path) -> dict[str, Any]:
    record = _table_c_full_file_record(path)
    record.pop("mtime_utc")
    return record


def _table_c_source_identity(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CompletionReceiptError(f"{label} is not a file identity")
    path = str(value.get("path", ""))
    sha256 = str(value.get("sha256", ""))
    size = value.get("size_bytes")
    mtime = value.get("mtime_ns")
    if (
        not path
        or not Path(path).is_absolute()
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or isinstance(mtime, bool)
        or not isinstance(mtime, int)
        or mtime <= 0
    ):
        raise CompletionReceiptError(f"{label} file identity is invalid")
    return {
        "path": str(Path(path).resolve(strict=False)),
        "sha256": sha256,
        "size_bytes": size,
        "mtime_ns": mtime,
    }


def _table_c_archived_object_path(
    snapshot_root: Path, archive_object: str
) -> Path:
    relative = Path(archive_object)
    parts = relative.parts
    if (
        relative.is_absolute()
        or relative.as_posix() != archive_object
        or len(parts) != 4
        or parts[:2] != ("objects", "sha256")
        or len(parts[2]) != 2
        or len(parts[3]) != 64
        or parts[2] != parts[3][:2]
        or any(character not in "0123456789abcdef" for character in parts[3])
    ):
        raise CompletionReceiptError(
            "Table-C archived source object path is invalid"
        )
    configured = snapshot_root / relative
    if configured.is_symlink():
        raise CompletionReceiptError("Table-C archived source object is a symlink")
    resolved_root = snapshot_root.resolve(strict=True)
    resolved = configured.resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise CompletionReceiptError(
            "Table-C archived source object escapes snapshot root"
        ) from exc
    return configured


def _require_table_c_archived_object_inventory(
    object_paths: Mapping[str, Path], snapshot_root: Path
) -> None:
    snapshot_root = snapshot_root.expanduser().resolve(strict=True)
    expected_objects = {
        path.resolve(strict=False) for path in object_paths.values()
    }
    object_root = snapshot_root / "objects"
    observed_objects: set[Path] = set()
    for path in object_root.rglob("*"):
        if path.is_symlink():
            raise CompletionReceiptError(
                "Table-C archived object inventory contains a symlink"
            )
        if path.is_file():
            observed_objects.add(path.resolve(strict=True))
    if observed_objects != expected_objects:
        raise CompletionReceiptError(
            "Table-C archived object inventory is not exact"
        )


def _verify_table_c_attestation_identity(
    path: Path,
    payload: Mapping[str, Any],
    *,
    identity_name: str,
    label: str,
) -> dict[str, Any]:
    expected = TABLE_C_DEPENDENCY_ATTESTATION_IDENTITIES[identity_name]
    record = file_record(path)
    if (
        payload.get("schema") != TABLE_C_DEPENDENCY_ATTESTATION_SCHEMA
        or frozenset(payload)
        not in {
            TABLE_C_OLD_ATTESTATION_KEYS,
            TABLE_C_OLD_ATTESTATION_KEYS | {"finalization"},
        }
        or payload.get("attestation_sha256")
        != _table_c_attestation_digest(payload)
        or payload.get("attestation_sha256") != expected["semantic_sha256"]
        or record["sha256"] != expected["file_sha256"]
    ):
        raise CompletionReceiptError(f"{label} fixed identity or self-hash drifted")
    return record


def _verify_table_c_archived_source_objects(
    preflight: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
    *,
    snapshot_root: Path | None = None,
) -> dict[str, Any]:
    if (
        source_snapshot.get("schema")
        != "pivot.stageb.table_c_u1000_training_source_snapshot/v1"
        or source_snapshot.get("status")
        != "retrospective_training_source_snapshot"
        or source_snapshot.get("source_snapshot_sha256")
        != TABLE_C_TRAINING_SNAPSHOT_IDENTITY["source_snapshot_sha256"]
    ):
        raise CompletionReceiptError("Table-C archived source snapshot identity drifted")
    repository_root = Path(str(preflight.get("repository_root", ""))).resolve(
        strict=False
    )
    categories = (
        (
            "dependency_closure",
            preflight.get("dependency_closure", {}).get("file_records"),
        ),
        (
            "static_repository_source",
            preflight.get("training_evidence", {}).get(
                "static_repository_sources"
            ),
        ),
        ("auditor_source", preflight.get("auditor_sources")),
    )
    expected: dict[str, dict[str, Any]] = {}
    for membership, records in categories:
        expected_count = TABLE_C_ARCHIVED_SOURCE_COUNTS[membership]
        if not isinstance(records, list) or len(records) != expected_count:
            raise CompletionReceiptError(
                f"Table-C archived {membership} record count drifted"
            )
        for index, record in enumerate(records):
            identity = _table_c_source_identity(
                record, label=f"Table-C {membership} record {index}"
            )
            try:
                relative = Path(identity["path"]).relative_to(
                    repository_root
                ).as_posix()
            except ValueError as exc:
                raise CompletionReceiptError(
                    f"Table-C {membership} source escapes repository root"
                ) from exc
            existing = expected.get(identity["path"])
            if existing is None:
                expected[identity["path"]] = {
                    **identity,
                    "relative_path": relative,
                    "memberships": [membership],
                }
            else:
                if any(
                    existing[key] != identity[key]
                    for key in ("sha256", "size_bytes", "mtime_ns")
                ):
                    raise CompletionReceiptError(
                        "Table-C archived source categories conflict"
                    )
                existing["memberships"].append(membership)
    if len(expected) != TABLE_C_ARCHIVED_SOURCE_COUNTS["deduplicated_source"]:
        raise CompletionReceiptError(
            "Table-C authoritative source categories do not deduplicate to 89"
        )
    for record in expected.values():
        record["memberships"] = sorted(set(record["memberships"]))
        record["archive_object"] = (
            f"objects/sha256/{record['sha256'][:2]}/{record['sha256']}"
        )

    sources = source_snapshot.get("sources")
    if not isinstance(sources, list) or len(sources) != len(expected):
        raise CompletionReceiptError("Table-C archived source record set is incomplete")
    observed: list[dict[str, Any]] = []
    object_paths: dict[str, Path] = {}
    source_by_object: dict[str, dict[str, Any]] = {}
    if snapshot_root is None:
        snapshot_root = TABLE_C_TRAINING_SNAPSHOT
    snapshot_root = snapshot_root.expanduser().resolve(strict=True)
    for index, record in enumerate(sources):
        identity = _table_c_source_identity(
            record, label=f"Table-C archived source record {index}"
        )
        normalized = {
            **identity,
            "relative_path": record.get("relative_path"),
            "memberships": record.get("memberships"),
            "archive_object": record.get("archive_object"),
        }
        observed.append(normalized)
        expected_record = expected.get(identity["path"])
        if expected_record != normalized:
            raise CompletionReceiptError(
                f"Table-C archived source record drifted: {identity['path']}"
            )
        archive_object = str(normalized["archive_object"])
        archive_path = _table_c_archived_object_path(
            snapshot_root, archive_object
        )
        if archive_object in object_paths:
            raise CompletionReceiptError(
                "Table-C archived source object identity is not unique"
            )
        object_paths[archive_object] = archive_path
        source_by_object[archive_object] = identity

    expected_order = sorted(
        expected.values(), key=lambda value: value["relative_path"]
    )
    if observed != expected_order:
        raise CompletionReceiptError("Table-C archived source ordering drifted")
    _require_table_c_archived_object_inventory(object_paths, snapshot_root)
    object_records_before = _artifact_records(object_paths)
    for archive_object, archive_record in object_records_before.items():
        identity = source_by_object[archive_object]
        if (
            archive_record["sha256"] != identity["sha256"]
            or archive_record["size_bytes"] != identity["size_bytes"]
        ):
            raise CompletionReceiptError(
                "Table-C archived object bytes drifted: "
                f"{object_paths[archive_object]}"
            )
    object_records_after = _require_stable_artifacts(
        object_records_before,
        object_paths,
        label="Table-C archived source objects",
    )
    _require_table_c_archived_object_inventory(object_paths, snapshot_root)
    manifest = {
        "schema": "pivot.stageb.table_c_archived_dependency_sources/v1",
        "sources": observed,
    }
    return {
        "dependency_record_count": TABLE_C_ARCHIVED_SOURCE_COUNTS[
            "dependency_closure"
        ],
        "static_source_record_count": TABLE_C_ARCHIVED_SOURCE_COUNTS[
            "static_repository_source"
        ],
        "auditor_record_count": TABLE_C_ARCHIVED_SOURCE_COUNTS["auditor_source"],
        "source_count": len(observed),
        "object_count": len(object_paths),
        "archived_source_manifest_sha256": canonical_json_sha256(manifest),
        "archived_object_records_sha256": canonical_json_sha256(
            {"objects": object_records_after}
        ),
        "_archived_object_records": object_records_after,
    }


def _validate_table_c_sequences() -> tuple[list[str], list[dict[str, Any]]]:
    from tools import run_stageb_paper_evaluations as paper_evaluator
    from tools import run_stageb_matrix_validation_queue as evaluation_queue

    expected_ids = list(evaluation_queue.EXPECTED_RUN_IDS)
    roots = []
    for run_id in expected_ids:
        row_id, seed_text = run_id.split(":", 1)
        seed = int(seed_text)
        root = (
            evaluation_queue.DEFAULT_TRAINING_OUTPUT_ROOT
            / row_id
            / f"seed{seed}"
        ).resolve(strict=True)
        sequence_path = (root / "sequence_manifest.json").resolve(strict=True)
        sequence = _read_json(
            sequence_path, label=f"Table-C {run_id} training sequence"
        )
        expected_budget = {
            "batch_size": 40,
            "optimizer_updates": 1000,
        }
        observed_budget = sequence.get("equal_budget_contract")
        if (
            sequence.get("schema") != paper_evaluator.TOKEN_TRAINING_SEQUENCE_SCHEMA
            or sequence.get("status") != "completed"
            or sequence.get("run_id") != run_id
            or sequence.get("row", {}).get("row_id") != row_id
            or sequence.get("seed") != seed
            or not isinstance(observed_budget, Mapping)
            or observed_budget.get("batch_size") != expected_budget["batch_size"]
            or observed_budget.get("optimizer_updates")
            != expected_budget["optimizer_updates"]
        ):
            raise CompletionReceiptError(
                f"Table-C {run_id} is not a completed B40/U1000 formal sequence"
            )
        roots.append(file_record(sequence_path))
    return expected_ids, roots


def _verify_table_c_pretraining_recovery(
    queue: Mapping[str, Any], specification: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from tools import recover_stageb_serial_matrix_pretraining_failure as recovery
    from tools import run_stageb_matrix_validation_queue as evaluation_queue

    queue_dir = evaluation_queue.DEFAULT_TRAINING_QUEUE_DIRS[1]
    result = recovery.verify_recovery(
        queue_dir,
        TABLE_C_PRETRAINING_RECOVERY_RECEIPT,
    )
    expected_receipt = _table_c_mtime_file_record(
        TABLE_C_PRETRAINING_RECOVERY_RECEIPT
    )
    expected_verifier = _table_c_mtime_file_record(Path(recovery.__file__))
    if (
        result.get("status") != "passed"
        or result.get("queue_id") != specification["queue_id"]
        or result.get("plan_sha256") != specification["plan_sha256"]
        or result.get("run_id") != "L2:42"
        or result.get("current_item_status") != "completed"
        or result.get("archived_evidence_verified") is not True
        or result.get("semantic_replay") != recovery.SEMANTIC_REPLAY_PROOF
        or result.get("verifier_source") != expected_verifier
    ):
        raise CompletionReceiptError(
            "Table-C pretraining recovery receipt did not pass exact replay"
        )
    events = [
        event
        for event in queue.get("events", [])
        if isinstance(event, Mapping)
        and event.get("event") == recovery.RECOVERY_EVENT
    ]
    recovered_items = [
        item
        for item in queue.get("items", [])
        if isinstance(item, Mapping)
        and item.get("pretraining_recovery_receipts") is not None
    ]
    if (
        len(events) != 1
        or events[0].get("run_id") != "L2:42"
        or events[0].get("failed_revision") != 590
        or events[0].get("receipt") != expected_receipt
        or len(recovered_items) != 1
        or recovered_items[0].get("run_id") != "L2:42"
        or recovered_items[0].get("pretraining_recovery_receipts")
        != [expected_receipt]
    ):
        raise CompletionReceiptError(
            "Table-C queue recovery history is not the single sealed pretraining event"
        )
    return expected_receipt, expected_verifier, result


def _verify_table_c_validation_recovery(
    queue: Mapping[str, Any], evaluation_queue: Any
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from tools import recover_stageb_matrix_validation_interruption as recovery

    try:
        result = recovery.verify_recovery(
            evaluation_queue.DEFAULT_QUEUE_DIR,
            TABLE_C_VALIDATION_RECOVERY_RECEIPT,
        )
        expected_receipt = _table_c_mtime_file_record(
            TABLE_C_VALIDATION_RECOVERY_RECEIPT
        )
        expected_verifier = _table_c_mtime_file_record(Path(recovery.__file__))
    except (OSError, ValueError, recovery.RecoveryError) as exc:
        raise CompletionReceiptError(
            f"Table-C validation recovery replay failed: {exc}"
        ) from exc

    plan = queue.get("plan")
    expected_result_keys = {
        "schema",
        "status",
        "queue_id",
        "plan_sha256",
        "run_id",
        "receipt_sha256",
        "semantic_replay",
    }
    if (
        not isinstance(plan, Mapping)
        or set(result) != expected_result_keys
        or result.get("schema") != recovery.VERIFICATION_SCHEMA
        or result.get("status") != "passed"
        or result.get("queue_id") != plan.get("queue_id")
        or result.get("plan_sha256") != queue.get("plan_sha256")
        or result.get("run_id") != recovery.EXPECTED_RUN_ID
        or result.get("receipt_sha256")
        != TABLE_C_VALIDATION_RECOVERY_RECEIPT_SHA256
        or result.get("semantic_replay") != recovery.SEMANTIC_REPLAY_PROOF
        or expected_receipt.get("sha256")
        != TABLE_C_VALIDATION_RECOVERY_FILE_SHA256
    ):
        raise CompletionReceiptError(
            "Table-C validation recovery receipt did not pass exact replay"
        )

    items = queue.get("items")
    events = queue.get("events")
    if not isinstance(items, list) or not isinstance(events, list):
        raise CompletionReceiptError(
            "Table-C validation recovery queue history is invalid"
        )
    recovered_items = [
        item
        for item in items
        if isinstance(item, Mapping) and "evaluation_recovery_receipts" in item
    ]
    recovery_events = [
        event
        for event in events
        if isinstance(event, Mapping)
        and event.get("event") == recovery.RECOVERY_EVENT
    ]
    if (
        len(recovered_items) != 1
        or recovered_items[0].get("run_id") != recovery.EXPECTED_RUN_ID
        or recovered_items[0].get("index") != recovery.EXPECTED_INDEX
        or recovered_items[0].get("status") != "completed"
        or recovered_items[0].get("evaluation_recovery_receipts")
        != [expected_receipt]
        or len(recovery_events) != 1
        or recovery_events[0].get("run_id") != recovery.EXPECTED_RUN_ID
        or recovery_events[0].get("index") != recovery.EXPECTED_INDEX
        or recovery_events[0].get("interrupted_revision")
        != recovery.EXPECTED_INTERRUPTED_REVISION
        or recovery_events[0].get("receipt") != expected_receipt
    ):
        raise CompletionReceiptError(
            "Table-C validation recovery history is not the single sealed event"
        )
    return expected_receipt, expected_verifier, result


def _verify_table_c_supplemental_final_dependency_attestation() -> tuple[
    dict[str, Any], dict[str, Any]
]:
    """Validate the later attestation without treating live source bytes as proof."""

    current = _read_json(
        TABLE_C_CURRENT_POLICY_FINAL_DEPENDENCY_ATTESTATION,
        label="Table-C supplemental current-policy attestation",
    )
    current_record = _verify_table_c_attestation_identity(
        TABLE_C_CURRENT_POLICY_FINAL_DEPENDENCY_ATTESTATION,
        current,
        identity_name="supplemental_current_policy",
        label="Table-C supplemental current-policy attestation",
    )
    archived_final = _read_json(
        TABLE_C_FINAL_DEPENDENCY_ATTESTATION,
        label="Table-C archived final dependency attestation",
    )
    _verify_table_c_attestation_identity(
        TABLE_C_FINAL_DEPENDENCY_ATTESTATION,
        archived_final,
        identity_name="final",
        label="Table-C archived final dependency attestation",
    )
    source_snapshot_path = TABLE_C_TRAINING_SNAPSHOT / "source_snapshot.json"
    source_snapshot = _read_json(
        source_snapshot_path,
        label="Table-C archived training source snapshot",
    )
    source_snapshot_record = file_record(source_snapshot_path)
    if (
        source_snapshot_record["sha256"]
        != TABLE_C_TRAINING_SNAPSHOT_IDENTITY["source_snapshot_file_sha256"]
        or source_snapshot.get("source_snapshot_sha256")
        != TABLE_C_TRAINING_SNAPSHOT_IDENTITY["source_snapshot_sha256"]
    ):
        raise CompletionReceiptError(
            "Table-C supplemental proof source snapshot identity drifted"
        )

    for key in (*TABLE_C_STATIC_IDENTITY_KEYS, "queues"):
        if current.get(key) != archived_final.get(key):
            raise CompletionReceiptError(
                f"Table-C supplemental attestation core drifted at {key}"
            )
    closure = current.get("dependency_closure")
    if (
        not isinstance(closure, Mapping)
        or closure.get("canonical_closure_sha256")
        != TABLE_C_DEPENDENCY_CLOSURE_SHA256
        or not isinstance(closure.get("file_records"), list)
        or len(closure["file_records"])
        != TABLE_C_ARCHIVED_SOURCE_COUNTS["dependency_closure"]
    ):
        raise CompletionReceiptError(
            "Table-C supplemental dependency closure identity drifted"
        )

    historical_auditors = archived_final.get("auditor_sources")
    current_auditors = current.get("auditor_sources")
    sources = source_snapshot.get("sources")
    if (
        not isinstance(historical_auditors, list)
        or not isinstance(current_auditors, list)
        or len(historical_auditors)
        != TABLE_C_ARCHIVED_SOURCE_COUNTS["auditor_source"]
        or len(current_auditors) != len(historical_auditors)
        or not isinstance(sources, list)
    ):
        raise CompletionReceiptError(
            "Table-C supplemental auditor evidence is incomplete"
        )
    historical_by_path = {
        str(record.get("path", "")): record
        for record in historical_auditors
        if isinstance(record, Mapping)
    }
    current_by_path = {
        str(record.get("path", "")): record
        for record in current_auditors
        if isinstance(record, Mapping)
    }
    if (
        len(historical_by_path) != len(historical_auditors)
        or set(current_by_path) != set(historical_by_path)
    ):
        raise CompletionReceiptError("Table-C supplemental auditor path set drifted")
    archived_identities = {
        (
            str(record.get("path", "")),
            str(record.get("sha256", "")),
            record.get("size_bytes"),
            record.get("mtime_ns"),
        )
        for record in sources
        if isinstance(record, Mapping)
    }
    archived_count = 0
    newer_unarchived: list[dict[str, Any]] = []
    for path, current_record_value in current_by_path.items():
        historical = historical_by_path[path]
        if current_record_value == historical:
            identity = _table_c_source_identity(
                current_record_value,
                label="Table-C supplemental archived auditor",
            )
            if (
                identity["path"],
                identity["sha256"],
                identity["size_bytes"],
                identity["mtime_ns"],
            ) not in archived_identities:
                raise CompletionReceiptError(
                    "Table-C supplemental auditor lacks archived bytes"
                )
            archived_count += 1
            continue
        current_identity = _table_c_source_identity(
            current_record_value,
            label="Table-C supplemental newer auditor",
        )
        historical_identity = _table_c_source_identity(
            historical,
            label="Table-C historical auditor",
        )
        if (
            Path(path).name != "audit_stageb_table_c_dependency_closure.py"
            or current_identity["sha256"] == historical_identity["sha256"]
            or current_identity["mtime_ns"] <= historical_identity["mtime_ns"]
            or (
                current_identity["path"],
                current_identity["sha256"],
                current_identity["size_bytes"],
                current_identity["mtime_ns"],
            )
            in archived_identities
        ):
            raise CompletionReceiptError(
                "Table-C supplemental unarchived auditor exception drifted"
            )
        unarchived_object = (
            TABLE_C_TRAINING_SNAPSHOT
            / "objects/sha256"
            / current_identity["sha256"][:2]
            / current_identity["sha256"]
        )
        if unarchived_object.exists():
            raise CompletionReceiptError(
                "Table-C newer supplemental auditor unexpectedly has archived bytes"
            )
        newer_unarchived.append(current_identity)
    if archived_count != 2 or len(newer_unarchived) != 1:
        raise CompletionReceiptError(
            "Table-C supplemental auditor classification is not exactly 2+1"
        )
    return (
        {
            "status": "passed_supplemental_self_hashed",
            "policy": "final",
            "semantic_attestation_sha256": current["attestation_sha256"],
            "canonical_closure_sha256": closure["canonical_closure_sha256"],
            "closure_path_count": len(closure["file_records"]),
            "archived_auditor_record_count": archived_count,
            "unarchived_newer_auditor_record_count": len(newer_unarchived),
            "unarchived_newer_auditor_sha256": newer_unarchived[0]["sha256"],
            "authoritative_dependency_proof": False,
            "live_source_parity_required": False,
        },
        current_record,
    )


def _verify_table_c_archived_finalization_lineage() -> tuple[dict[str, Any], dict[str, Any]]:
    source_snapshot_path = TABLE_C_TRAINING_SNAPSHOT / "source_snapshot.json"
    lineage_paths = {
        "final_attestation": TABLE_C_FINAL_DEPENDENCY_ATTESTATION,
        "preflight_attestation": TABLE_C_PREFLIGHT_DEPENDENCY_ATTESTATION,
        "archived_source_snapshot": source_snapshot_path,
    }
    lineage_before = _artifact_records(lineage_paths)

    payload = _read_json(
        TABLE_C_FINAL_DEPENDENCY_ATTESTATION,
        label="Table-C historical finalization attestation",
    )
    final_attestation_record = _verify_table_c_attestation_identity(
        TABLE_C_FINAL_DEPENDENCY_ATTESTATION,
        payload,
        identity_name="final",
        label="Table-C archived finalization attestation",
    )
    preflight_path = TABLE_C_PREFLIGHT_DEPENDENCY_ATTESTATION.resolve(strict=True)
    preflight = _read_json(
        preflight_path,
        label="Table-C dependency preflight attestation",
    )
    preflight_attestation_record = _verify_table_c_attestation_identity(
        preflight_path,
        preflight,
        identity_name="preflight",
        label="Table-C dependency preflight attestation",
    )
    finalization = payload.get("finalization")
    finalization_preflight = (
        finalization.get("preflight")
        if isinstance(finalization, Mapping)
        else None
    )
    if (
        not isinstance(finalization, Mapping)
        or not isinstance(finalization_preflight, Mapping)
        or set(finalization)
        != {
            "schema",
            "policy",
            "preflight",
            "staged_old_schema_attestation_sha256",
            "finalizer_source",
            "auditor_preservation",
            "completion_verification_counts",
            "transformation",
        }
        or finalization.get("schema") != TABLE_C_FINALIZATION_SCHEMA
        or finalization.get("policy") != "final"
        or finalization.get("completion_verification_counts")
        != {"completed_l0_l4": 5, "remaining_table_c": 28}
        or finalization_preflight.get("semantic_attestation_sha256")
        != preflight.get("attestation_sha256")
    ):
        raise CompletionReceiptError(
            "Table-C archived finalization lineage is incomplete"
        )
    if finalization_preflight.get("file_record") != _table_c_full_file_record(
        preflight_path
    ):
        raise CompletionReceiptError(
            "Table-C archived finalization preflight identity drifted"
        )
    if finalization.get("finalizer_source") != dict(
        TABLE_C_HISTORICAL_FINALIZER_SOURCE_IDENTITY
    ):
        raise CompletionReceiptError(
            "Table-C archived finalizer source identity drifted"
        )
    for key in (*TABLE_C_STATIC_IDENTITY_KEYS, "auditor_sources"):
        if payload.get(key) != preflight.get(key):
            raise CompletionReceiptError(
                f"Table-C archived finalization static identity drifted at {key}"
            )
    expected_auditor_preservation = {
        "historical_auditor_sources_unchanged": True,
        "record_count": TABLE_C_ARCHIVED_SOURCE_COUNTS["auditor_source"],
        "canonical_sha256": canonical_json_sha256(preflight["auditor_sources"]),
    }
    if (
        finalization.get("auditor_preservation")
        != expected_auditor_preservation
        or finalization.get("transformation")
        != {
            "only_queue_field_changed": (
                "queues.remaining_table_c.status_policy"
            ),
            "from": "running_or_completed",
            "to": "completed_required",
            "historical_auditor_sources_modified": False,
        }
    ):
        raise CompletionReceiptError(
            "Table-C archived auditor/final transformation lineage drifted"
        )
    queues = payload.get("queues")
    preflight_queues = preflight.get("queues")
    expected_remaining = copy.deepcopy(
        preflight_queues.get("remaining_table_c")
        if isinstance(preflight_queues, Mapping)
        else None
    )
    if not isinstance(expected_remaining, dict):
        raise CompletionReceiptError(
            "Table-C archived preflight remaining queue binding is missing"
        )
    expected_remaining.update(
        {
            "observed_status": "completed",
            "completion_verification": {
                "schema": "pivot.stageb.serial_matrix_queue_verification/v1",
                "status": "passed",
                "verified_item_count": 28,
            },
            "status_policy": "completed_required",
        }
    )
    if (
        not isinstance(queues, Mapping)
        or not isinstance(preflight_queues, Mapping)
        or queues.get("completed_l0_l4")
        != preflight_queues.get("completed_l0_l4")
        or queues.get("remaining_table_c") != expected_remaining
    ):
        raise CompletionReceiptError(
            "Table-C archived finalization queue transformation drifted"
        )
    for role, count in (("completed_l0_l4", 5), ("remaining_table_c", 28)):
        queue_record = queues.get(role)
        if (
            not isinstance(queue_record, Mapping)
            or queue_record.get("observed_status") != "completed"
            or queue_record.get("status_policy") != "completed_required"
            or queue_record.get("completion_verification")
            != {
                "schema": "pivot.stageb.serial_matrix_queue_verification/v1",
                "status": "passed",
                "verified_item_count": count,
            }
        ):
            raise CompletionReceiptError(
                f"Table-C archived finalization {role} completion drifted"
            )
    staged_sha = str(finalization.get("staged_old_schema_attestation_sha256", ""))
    reconstructed = copy.deepcopy(dict(payload))
    reconstructed.pop("finalization", None)
    reconstructed["queues"]["remaining_table_c"][
        "status_policy"
    ] = "running_or_completed"
    reconstructed["attestation_sha256"] = staged_sha
    if (
        len(staged_sha) != 64
        or frozenset(reconstructed) != TABLE_C_OLD_ATTESTATION_KEYS
        or _table_c_attestation_digest(reconstructed) != staged_sha
    ):
        raise CompletionReceiptError(
            "Table-C archived finalization is not the declared one-field upgrade"
        )

    source_snapshot = _read_json(
        source_snapshot_path,
        label="Table-C archived training source snapshot",
    )
    source_snapshot_record = file_record(source_snapshot_path)
    if source_snapshot_record["sha256"] != TABLE_C_TRAINING_SNAPSHOT_IDENTITY[
        "source_snapshot_file_sha256"
    ]:
        raise CompletionReceiptError(
            "Table-C archived training source snapshot file identity drifted"
        )
    archived_sources = _verify_table_c_archived_source_objects(
        preflight,
        source_snapshot,
    )
    archived_object_records = archived_sources.pop(
        "_archived_object_records", None
    )
    if (
        not isinstance(archived_object_records, Mapping)
        or len(archived_object_records)
        != TABLE_C_ARCHIVED_SOURCE_COUNTS["deduplicated_source"]
    ):
        raise CompletionReceiptError(
            "Table-C archived object stability closure is incomplete"
        )
    historical_auditor = next(
        record
        for record in preflight["auditor_sources"]
        if Path(str(record["path"])).name
        == "audit_stageb_table_c_dependency_closure.py"
    )
    lineage_after = _require_stable_artifacts(
        lineage_before,
        lineage_paths,
        label="Table-C archived finalization lineage",
    )
    if (
        final_attestation_record != lineage_after["final_attestation"]
        or preflight_attestation_record != lineage_after["preflight_attestation"]
        or source_snapshot_record != lineage_after["archived_source_snapshot"]
    ):
        raise CompletionReceiptError(
            "Table-C archived finalization lineage identity changed during replay"
        )
    lineage_evidence: dict[str, Any] = {
        "schema": TABLE_C_FINALIZATION_LINEAGE_EVIDENCE_SCHEMA,
        "status": "passed",
        "final_attestation": lineage_after["final_attestation"],
        "preflight_attestation": lineage_after["preflight_attestation"],
        "archived_source_snapshot": lineage_after["archived_source_snapshot"],
        "historical_finalizer_source_record": dict(
            TABLE_C_HISTORICAL_FINALIZER_SOURCE_IDENTITY
        ),
        "historical_finalizer_live_bytes_required": False,
        "semantic_transformation_replayed": True,
        "archived_source_manifest_sha256": archived_sources[
            "archived_source_manifest_sha256"
        ],
        "archived_object_records_sha256": archived_sources[
            "archived_object_records_sha256"
        ],
    }
    lineage_evidence["evidence_sha256"] = canonical_json_sha256(lineage_evidence)
    return (
        {
            "status": "passed",
            "preflight_attestation_sha256": preflight["attestation_sha256"],
            "final_attestation_sha256": payload["attestation_sha256"],
            "historical_auditor_sha256": historical_auditor["sha256"],
            "historical_auditor_archived": True,
            "historical_finalizer_evidence_bound": True,
            "historical_finalizer_live_bytes_required": False,
            "staged_upgrade_replayed": True,
            "all_authoritative_sources_archived": True,
            **archived_sources,
            "_archived_object_records": dict(archived_object_records),
        },
        lineage_evidence,
    )


def _verify_table_c_training_snapshot() -> tuple[dict[str, Any], dict[str, Any]]:
    from tools import build_stageb_table_c_u1000_training_snapshot as snapshot

    try:
        result = snapshot.verify_snapshot(
            TABLE_C_TRAINING_SNAPSHOT,
            require_live_source_parity=False,
        )
    except (OSError, ValueError, snapshot.TrainingSnapshotError) as exc:
        raise CompletionReceiptError(
            f"Table-C immutable training snapshot replay failed: {exc}"
        ) from exc
    if (
        result.get("status") != "passed"
        or result.get("run_count") != 33
        or result.get("source_count") != 89
        or result.get("object_count") != 89
        or result.get("live_completion_record_count") != 443
        or result.get("live_source_parity_required") is not False
        or result.get("live_parity_record_count") is not None
        or result.get("strict_live_final_gates") is not None
    ):
        raise CompletionReceiptError(
            "Table-C immutable training snapshot contract is incomplete"
        )
    observed_identity = {
        key: result.get(key) for key in TABLE_C_TRAINING_SNAPSHOT_IDENTITY
    }
    if observed_identity != dict(TABLE_C_TRAINING_SNAPSHOT_IDENTITY):
        raise CompletionReceiptError(
            "Table-C immutable training snapshot identity drifted"
        )
    return result, file_record(Path(snapshot.__file__))


def _verify_table_c_matrix_aggregate(aggregate: Any) -> Mapping[str, Any]:
    persisted_report = _read_json(
        TABLE_C_AGGREGATION_REPORT,
        label="Table-C matrix validation aggregate",
    )
    try:
        replayed_report = aggregate.aggregate_spec(TABLE_C_AGGREGATION_SPEC)
    except (aggregate.MatrixValidationError, OSError, ValueError) as exc:
        raise CompletionReceiptError(
            f"Table-C matrix validation aggregate replay failed: {exc}"
        ) from exc
    if _strip_volatile(persisted_report) != _strip_volatile(replayed_report):
        raise CompletionReceiptError(
            "Table-C matrix validation aggregate differs from semantic replay"
        )
    if (
        persisted_report.get("schema") != aggregate.REPORT_SCHEMA
        or persisted_report.get("status") != "validated_matrix_validation_only"
        or persisted_report.get("validation", {}).get("pass") is not True
        or set(persisted_report.get("experiments", {}))
        != set(aggregate.FORMAL_EXPERIMENT_IDS)
    ):
        raise CompletionReceiptError("Table-C aggregate contract is incomplete")
    return persisted_report


def _table_c_canonical_artifact_paths(
    evaluation_queue: Any,
    recovery: Any,
    validation_recovery: Any,
    snapshot: Any,
) -> dict[str, Path]:
    training_queue_dirs = tuple(evaluation_queue.DEFAULT_TRAINING_QUEUE_DIRS)
    if len(training_queue_dirs) != 2:
        raise CompletionReceiptError(
            "Table-C canonical training queue set is not exactly two queues"
        )
    return {
        "training_queue_completed_l0_l4": training_queue_dirs[0] / "queue.json",
        "training_queue_remaining_28": training_queue_dirs[1] / "queue.json",
        "pretraining_recovery_receipt": TABLE_C_PRETRAINING_RECOVERY_RECEIPT,
        "pretraining_recovery_verifier": Path(recovery.__file__),
        "validation_recovery_receipt": TABLE_C_VALIDATION_RECOVERY_RECEIPT,
        "validation_recovery_verifier": Path(validation_recovery.__file__),
        "archived_finalization_attestation": (
            TABLE_C_FINAL_DEPENDENCY_ATTESTATION
        ),
        "archived_preflight_dependency_attestation": (
            TABLE_C_PREFLIGHT_DEPENDENCY_ATTESTATION
        ),
        "supplemental_current_policy_attestation": (
            TABLE_C_CURRENT_POLICY_FINAL_DEPENDENCY_ATTESTATION
        ),
        "training_snapshot_completion_subreceipt": (
            TABLE_C_TRAINING_SNAPSHOT / "completion_subreceipt.json"
        ),
        "training_snapshot_source_manifest": (
            TABLE_C_TRAINING_SNAPSHOT / "source_snapshot.json"
        ),
        "training_snapshot_verifier": Path(snapshot.__file__),
        "validation_queue": evaluation_queue.DEFAULT_QUEUE_DIR / "queue.json",
        "aggregation_spec": TABLE_C_AGGREGATION_SPEC,
        "aggregation_report": TABLE_C_AGGREGATION_REPORT,
    }


def _table_c_object_paths_from_records(
    records: Any,
    *,
    snapshot_root: Path,
) -> dict[str, Path]:
    if (
        not isinstance(records, Mapping)
        or len(records)
        != TABLE_C_ARCHIVED_SOURCE_COUNTS["deduplicated_source"]
    ):
        raise CompletionReceiptError(
            "Table-C archived object stability closure is incomplete"
        )
    snapshot_root = snapshot_root.expanduser().resolve(strict=True)
    object_paths: dict[str, Path] = {}
    for archive_object, value in records.items():
        archive_object = str(archive_object)
        path = _table_c_archived_object_path(snapshot_root, archive_object)
        if (
            not isinstance(value, Mapping)
            or set(value) != {"path", "sha256", "size_bytes"}
            or value.get("path") != str(path.resolve(strict=False))
            or value.get("sha256") != Path(archive_object).name
            or isinstance(value.get("size_bytes"), bool)
            or not isinstance(value.get("size_bytes"), int)
            or value.get("size_bytes") < 0
        ):
            raise CompletionReceiptError(
                "Table-C archived object stability record drifted"
            )
        object_paths[archive_object] = path
    _require_table_c_archived_object_inventory(object_paths, snapshot_root)
    return object_paths


def _table_c_artifacts_from_records(
    records: Mapping[str, Mapping[str, Any]],
    finalization_lineage_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "training_queue_completed_l0_l4": dict(
            records["training_queue_completed_l0_l4"]
        ),
        "training_queue_remaining_28": dict(
            records["training_queue_remaining_28"]
        ),
        "pretraining_recovery_receipt": dict(
            records["pretraining_recovery_receipt"]
        ),
        "pretraining_recovery_verifier": dict(
            records["pretraining_recovery_verifier"]
        ),
        "validation_recovery_receipt": dict(
            records["validation_recovery_receipt"]
        ),
        "validation_recovery_verifier": dict(
            records["validation_recovery_verifier"]
        ),
        "archived_finalization_attestation": dict(
            records["archived_finalization_attestation"]
        ),
        "archived_preflight_dependency_attestation": dict(
            records["archived_preflight_dependency_attestation"]
        ),
        "archived_finalization_lineage_evidence": copy.deepcopy(
            dict(finalization_lineage_evidence)
        ),
        "supplemental_current_policy_attestation": dict(
            records["supplemental_current_policy_attestation"]
        ),
        "training_snapshot_completion_subreceipt": dict(
            records["training_snapshot_completion_subreceipt"]
        ),
        "training_snapshot_source_manifest": dict(
            records["training_snapshot_source_manifest"]
        ),
        "training_snapshot_verifier": dict(
            records["training_snapshot_verifier"]
        ),
        "validation_queue": dict(records["validation_queue"]),
        "aggregation_spec": dict(records["aggregation_spec"]),
        "aggregation_report": dict(records["aggregation_report"]),
    }


def _verify_table_c() -> Mapping[str, Any]:
    from tools import build_stageb_table_c_u1000_training_snapshot as snapshot
    from tools import recover_stageb_serial_matrix_pretraining_failure as recovery
    from tools import recover_stageb_matrix_validation_interruption as validation_recovery
    from tools import aggregate_stageb_matrix_validation as aggregate
    from tools import run_stageb_matrix_validation_queue as evaluation_queue
    from tools import run_stageb_serial_matrix_queue as training_queue

    canonical_artifact_paths = _table_c_canonical_artifact_paths(
        evaluation_queue,
        recovery,
        validation_recovery,
        snapshot,
    )
    canonical_artifacts_before = _artifact_records(canonical_artifact_paths)

    expected_ids, sequence_records = _validate_table_c_sequences()
    training_snapshot_result, training_snapshot_verifier = (
        _verify_table_c_training_snapshot()
    )
    finalization_result, finalization_lineage_evidence = (
        _verify_table_c_archived_finalization_lineage()
    )
    archived_object_records = finalization_result.get(
        "_archived_object_records"
    )
    archived_object_paths = _table_c_object_paths_from_records(
        archived_object_records,
        snapshot_root=TABLE_C_TRAINING_SNAPSHOT,
    )
    supplemental_result, supplemental_attestation = (
        _verify_table_c_supplemental_final_dependency_attestation()
    )

    training_queue_records: list[dict[str, Any]] = []
    observed_training_ids: set[str] = set()
    remaining_training_queue: Mapping[str, Any] | None = None
    for queue_dir, specification in zip(
        evaluation_queue.DEFAULT_TRAINING_QUEUE_DIRS,
        evaluation_queue.LOCKED_TRAINING_QUEUES.values(),
    ):
        queue = training_queue.load_queue(queue_dir)
        verification = training_queue.verify_queue(queue_dir)
        planned = queue.get("plan", {}).get("items")
        run_ids = {
            str(item.get("run_id"))
            for item in planned or []
            if isinstance(item, Mapping)
        }
        expected_queue_ids = set(specification["run_ids"])
        if (
            queue.get("status") != "completed"
            or verification.get("status") != "passed"
            or queue.get("plan", {}).get("queue_id")
            != specification["queue_id"]
            or queue.get("plan_sha256") != specification["plan_sha256"]
            or run_ids != expected_queue_ids
        ):
            raise CompletionReceiptError(
                f"Table-C training queue contract drifted: {queue_dir}"
            )
        observed_training_ids.update(run_ids)
        training_queue_records.append(file_record(queue_dir / "queue.json"))
        if (
            specification["queue_id"]
            == evaluation_queue.LOCKED_TRAINING_QUEUES["remaining_table_c"]["queue_id"]
        ):
            remaining_training_queue = queue
    if observed_training_ids != set(expected_ids):
        raise CompletionReceiptError("Table-C training queue run set is not exact")
    if remaining_training_queue is None:
        raise CompletionReceiptError("Table-C remaining training queue was not replayed")
    recovery_receipt, recovery_verifier, recovery_result = (
        _verify_table_c_pretraining_recovery(
            remaining_training_queue,
            evaluation_queue.LOCKED_TRAINING_QUEUES["remaining_table_c"],
        )
    )

    evaluation = evaluation_queue.load_queue(evaluation_queue.DEFAULT_QUEUE_DIR)
    evaluation_verification = evaluation_queue.verify_queue(
        evaluation_queue.DEFAULT_QUEUE_DIR
    )
    evaluation_run_ids = [
        str(item.get("run_id"))
        for item in evaluation.get("plan", {}).get("items", [])
        if isinstance(item, Mapping)
    ]
    aggregation_spec = _read_json(
        TABLE_C_AGGREGATION_SPEC,
        label="Table-C predeclared matrix aggregation input",
    )
    aggregation_spec_record = file_record(TABLE_C_AGGREGATION_SPEC)
    if (
        evaluation.get("status") != "completed"
        or evaluation_verification.get("status") != "passed"
        or evaluation_run_ids != expected_ids
        or aggregation_spec.get("schema") != aggregate.INPUT_SCHEMA
        or aggregation_spec.get("evaluation_queue_id")
        != evaluation.get("plan", {}).get("queue_id")
        or aggregation_spec.get("evaluation_plan_sha256")
        != evaluation.get("plan_sha256")
        or evaluation.get("aggregation_input_spec")
        != aggregation_spec_record
    ):
        raise CompletionReceiptError(
            "Table-C canonical validation queue is not sealed and completed"
        )

    validation_recovery_receipt, validation_recovery_verifier, (
        validation_recovery_result
    ) = _verify_table_c_validation_recovery(evaluation, evaluation_queue)

    _verify_table_c_matrix_aggregate(aggregate)

    canonical_artifacts_postflight = _require_stable_artifacts(
        canonical_artifacts_before,
        canonical_artifact_paths,
        label="Table-C canonical",
    )
    archived_object_records_postflight = _require_stable_artifacts(
        archived_object_records,
        archived_object_paths,
        label="Table-C archived source objects outer closure",
    )
    _require_table_c_archived_object_inventory(
        archived_object_paths,
        TABLE_C_TRAINING_SNAPSHOT,
    )

    lineage_without_hash = copy.deepcopy(dict(finalization_lineage_evidence))
    lineage_sha256 = lineage_without_hash.pop("evidence_sha256", None)
    expected_object_records_sha256 = canonical_json_sha256(
        {"objects": archived_object_records_postflight}
    )
    helper_records = {
        "training_queue_completed_l0_l4": training_queue_records[0],
        "training_queue_remaining_28": training_queue_records[1],
        "pretraining_recovery_receipt": {
            key: recovery_receipt.get(key)
            for key in ("path", "sha256", "size_bytes")
        },
        "pretraining_recovery_verifier": {
            key: recovery_verifier.get(key)
            for key in ("path", "sha256", "size_bytes")
        },
        "validation_recovery_receipt": {
            key: validation_recovery_receipt.get(key)
            for key in ("path", "sha256", "size_bytes")
        },
        "validation_recovery_verifier": {
            key: validation_recovery_verifier.get(key)
            for key in ("path", "sha256", "size_bytes")
        },
        "supplemental_current_policy_attestation": supplemental_attestation,
        "training_snapshot_verifier": training_snapshot_verifier,
        "aggregation_spec": aggregation_spec_record,
    }
    if any(
        helper_records[name] != canonical_artifacts_postflight[name]
        for name in helper_records
    ):
        raise CompletionReceiptError(
            "Table-C helper artifact identity differs from canonical postflight"
        )
    if (
        lineage_sha256 != canonical_json_sha256(lineage_without_hash)
        or finalization_lineage_evidence.get("final_attestation")
        != canonical_artifacts_postflight["archived_finalization_attestation"]
        or finalization_lineage_evidence.get("preflight_attestation")
        != canonical_artifacts_postflight[
            "archived_preflight_dependency_attestation"
        ]
        or finalization_lineage_evidence.get("archived_source_snapshot")
        != canonical_artifacts_postflight["training_snapshot_source_manifest"]
        or finalization_lineage_evidence.get(
            "archived_object_records_sha256"
        )
        != expected_object_records_sha256
        or finalization_lineage_evidence.get(
            "archived_source_manifest_sha256"
        )
        != finalization_result.get("archived_source_manifest_sha256")
        or finalization_result.get("archived_object_records_sha256")
        != expected_object_records_sha256
        or canonical_artifacts_postflight[
            "training_snapshot_completion_subreceipt"
        ]["sha256"]
        != TABLE_C_TRAINING_SNAPSHOT_IDENTITY[
            "completion_subreceipt_file_sha256"
        ]
        or canonical_artifacts_postflight["training_snapshot_source_manifest"][
            "sha256"
        ]
        != TABLE_C_TRAINING_SNAPSHOT_IDENTITY[
            "source_snapshot_file_sha256"
        ]
    ):
        raise CompletionReceiptError(
            "Table-C helper evidence differs from canonical postflight"
        )

    # Exercise artifact assembly inside the closure, then publish only the
    # records from the final postflight below.
    _table_c_artifacts_from_records(
        canonical_artifacts_postflight,
        finalization_lineage_evidence,
    )
    canonical_artifacts_final = _require_stable_artifacts(
        canonical_artifacts_postflight,
        canonical_artifact_paths,
        label="Table-C canonical artifact assembly",
    )
    archived_object_records_final = _require_stable_artifacts(
        archived_object_records_postflight,
        archived_object_paths,
        label="Table-C archived source object artifact assembly",
    )
    _require_table_c_archived_object_inventory(
        archived_object_paths,
        TABLE_C_TRAINING_SNAPSHOT,
    )
    if canonical_json_sha256(
        {"objects": archived_object_records_final}
    ) != expected_object_records_sha256:
        raise CompletionReceiptError(
            "Table-C archived object evidence changed during artifact assembly"
        )
    artifacts = _table_c_artifacts_from_records(
        canonical_artifacts_final,
        finalization_lineage_evidence,
    )
    return {
        "status": "completed",
        "adapter_id": TABLE_C_ADAPTER_ID,
        "contract": {
            "rows": list(aggregate.FORMAL_EXPERIMENT_IDS),
            "seeds": list(aggregate.FORMAL_TRAIN_SEEDS),
            "run_ids": expected_ids,
            "batch_size": 40,
            "optimizer_updates": 1000,
            "training_run_count": 33,
            "validation_run_count": 33,
            "profile": "matrix_validation",
            "ref_surface": ["refcoco_val", "refcocop_val", "refcocog_val"],
            "tn_surface": "sealed_calibration_only",
            "dependency_proof": (
                "archived_89_source_objects_plus_preflight_final_lineage"
            ),
            "archived_source_count": 89,
            "archived_object_count": 89,
            "live_completion_record_count": 443,
        },
        "artifacts": artifacts,
        "semantic_replay": {
            "training_queues_verified": True,
            "single_pretraining_recovery_verified": (
                recovery_result.get("status") == "passed"
            ),
            "single_validation_recovery_verified": (
                validation_recovery_result.get("status") == "passed"
            ),
            "training_sequences_verified": len(sequence_records) == 33,
            "immutable_training_snapshot_verified": (
                training_snapshot_result.get("status") == "passed"
            ),
            "archived_dependency_final_proof_verified": (
                finalization_result.get("staged_upgrade_replayed") is True
                and finalization_result.get("historical_auditor_archived") is True
                and finalization_result.get("historical_finalizer_evidence_bound")
                is True
                and finalization_result.get(
                    "historical_finalizer_live_bytes_required"
                )
                is False
                and finalization_result.get("all_authoritative_sources_archived")
                is True
                and finalization_result.get("source_count") == 89
                and finalization_result.get("object_count") == 89
            ),
            "live_completion_records_verified": (
                training_snapshot_result.get("live_completion_record_count") == 443
            ),
            "supplemental_current_policy_self_hash_verified": (
                supplemental_result.get("status")
                == "passed_supplemental_self_hashed"
                and supplemental_result.get("authoritative_dependency_proof")
                is False
            ),
            "authorized_live_source_evolution_tolerated": True,
            "canonical_artifact_closure_verified": True,
            "archived_object_closure_verified": True,
            "validation_queue_verified": True,
            "aggregate_recomputed": True,
            "no_ref_test_or_strict_access": True,
        },
    }


def _verify_headline_evidence_snapshot(
    training_runner: Any, value: Any, *, label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CompletionReceiptError(f"canonical {label} snapshot is invalid")
    records = value.get("records")
    if not isinstance(records, list) or not records:
        raise CompletionReceiptError(f"canonical {label} snapshot is empty")
    normalized = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "sha256",
            "size_bytes",
            "mtime_ns",
        }:
            raise CompletionReceiptError(
                f"canonical {label} snapshot record is invalid"
            )
        path = Path(str(record.get("path", ""))).resolve(strict=False)
        if (
            not path.is_absolute()
            or not _is_sha256(record.get("sha256"))
            or type(record.get("size_bytes")) is not int
            or record.get("size_bytes", -1) < 0
            or type(record.get("mtime_ns")) is not int
            or record.get("mtime_ns", -1) < 0
        ):
            raise CompletionReceiptError(
                f"canonical {label} snapshot record drifted"
            )
        normalized.append(
            {
                "path": str(path),
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
                "mtime_ns": record["mtime_ns"],
            }
        )
    if (
        normalized != sorted(normalized, key=lambda record: record["path"])
        or len({record["path"] for record in normalized}) != len(normalized)
    ):
        raise CompletionReceiptError(
            f"canonical {label} snapshot records are not canonical"
        )
    expected_digest = canonical_json_sha256(
        {
            "schema": training_runner.COMPLETED_TRAINING_EVIDENCE_SNAPSHOT_SCHEMA,
            "records": normalized,
        }
    )
    if (
        value.get("schema")
        != training_runner.COMPLETED_TRAINING_EVIDENCE_SNAPSHOT_SCHEMA
        or value.get("algorithm")
        != "sha256_stable_descriptor_path_content_size_mtime_v1"
        or value.get("digest") != expected_digest
    ):
        raise CompletionReceiptError(f"canonical {label} snapshot digest drifted")
    return {
        "schema": training_runner.COMPLETED_TRAINING_EVIDENCE_SNAPSHOT_SCHEMA,
        "algorithm": "sha256_stable_descriptor_path_content_size_mtime_v1",
        "records": normalized,
        "digest": expected_digest,
    }


def _verify_headline_full_run_telemetry(
    training_runner: Any,
    *,
    run_id: str,
    telemetry: Mapping[str, Any],
    ancestry: Mapping[str, Any],
    evidence_snapshot: Mapping[str, Any],
) -> None:
    """Validate and evidence-bind the all-attempt telemetry projection."""

    full_run = telemetry.get("full_run")
    expected_full_keys = {
        "schema",
        "status",
        "attempt_count",
        "sampling_interval_ms",
        "sample_rows",
        "devices",
        "all_attempts_same_devices",
        "attempts",
        "semantic_sha256",
    }
    attempts = full_run.get("attempts") if isinstance(full_run, Mapping) else None
    expected_attempt_count = ancestry.get("attempt_count")
    sampling_interval_ms = training_runner.FORMAL_TELEMETRY_INTERVAL_SECONDS * 1000
    if (
        not isinstance(full_run, Mapping)
        or set(full_run) != expected_full_keys
        or full_run.get("schema") != training_runner.FULL_RUN_TELEMETRY_SCHEMA
        or full_run.get("status") != "passed"
        or type(expected_attempt_count) is not int
        or expected_attempt_count <= 0
        or full_run.get("attempt_count") != expected_attempt_count
        or full_run.get("sampling_interval_ms") != sampling_interval_ms
        or type(full_run.get("sample_rows")) is not int
        or full_run.get("sample_rows", 0) <= 0
        or full_run.get("devices") != telemetry.get("devices")
        or full_run.get("all_attempts_same_devices") is not True
        or not isinstance(attempts, list)
        or len(attempts) != expected_attempt_count
    ):
        raise CompletionReceiptError(
            f"canonical {run_id} full-run telemetry contract drifted"
        )

    evidence_by_path = {
        str(Path(str(record["path"])).resolve(strict=False)): record
        for record in evidence_snapshot["records"]
    }
    expected_attempt_keys = {
        "attempt_ordinal",
        "sample_rows",
        "devices",
        "artifacts",
        "evidence_sha256",
    }
    expected_artifact_names = {
        "gpu_environment",
        "gpu_telemetry",
        "gpu_telemetry_summary",
    }
    total_rows = 0
    seen_artifact_paths: set[str] = set()
    for ordinal, attempt in enumerate(attempts):
        artifacts = attempt.get("artifacts") if isinstance(attempt, Mapping) else None
        rows = attempt.get("sample_rows") if isinstance(attempt, Mapping) else None
        if (
            not isinstance(attempt, Mapping)
            or set(attempt) != expected_attempt_keys
            or attempt.get("attempt_ordinal") != ordinal
            or type(rows) is not int
            or rows <= 0
            or attempt.get("devices") != full_run.get("devices")
            or not isinstance(artifacts, Mapping)
            or set(artifacts) != expected_artifact_names
        ):
            raise CompletionReceiptError(
                f"canonical {run_id} full-run telemetry attempt {ordinal} drifted"
            )
        total_rows += rows
        normalized_artifacts: dict[str, dict[str, Any]] = {}
        for name in sorted(expected_artifact_names):
            record = artifacts[name]
            if (
                not isinstance(record, Mapping)
                or set(record) != {"path", "sha256", "size_bytes"}
                or not _is_sha256(record.get("sha256"))
                or type(record.get("size_bytes")) is not int
                or record.get("size_bytes", -1) < 0
            ):
                raise CompletionReceiptError(
                    f"canonical {run_id} attempt {ordinal} {name} telemetry record drifted"
                )
            path = Path(str(record.get("path", ""))).resolve(strict=False)
            if not path.is_absolute() or str(path) in seen_artifact_paths:
                raise CompletionReceiptError(
                    f"canonical {run_id} attempt telemetry paths are not distinct"
                )
            seen_artifact_paths.add(str(path))
            normalized = {
                "path": str(path),
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
            }
            evidence_record = evidence_by_path.get(str(path))
            if (
                not isinstance(evidence_record, Mapping)
                or any(
                    evidence_record.get(key) != normalized[key]
                    for key in ("path", "sha256", "size_bytes")
                )
            ):
                raise CompletionReceiptError(
                    f"canonical {run_id} attempt telemetry is outside completed evidence"
                )
            normalized_artifacts[name] = normalized
        expected_attempt_evidence_sha256 = canonical_json_sha256(
            {
                "schema": training_runner.ATTEMPT_TELEMETRY_SCHEMA,
                "status": "sealed",
                "attempt_ordinal": ordinal,
                "sampling_interval_ms": sampling_interval_ms,
                "sample_rows": rows,
                "devices": attempt["devices"],
                "artifacts": normalized_artifacts,
            }
        )
        if attempt.get("evidence_sha256") != expected_attempt_evidence_sha256:
            raise CompletionReceiptError(
                f"canonical {run_id} attempt {ordinal} telemetry SHA-256 drifted"
            )

    full_run_payload = dict(full_run)
    semantic_sha256 = full_run_payload.pop("semantic_sha256", None)
    if (
        total_rows != full_run.get("sample_rows")
        or attempts[-1].get("sample_rows") != telemetry.get("sample_rows")
        or not _is_sha256(semantic_sha256)
        or semantic_sha256 != canonical_json_sha256(full_run_payload)
    ):
        raise CompletionReceiptError(
            f"canonical {run_id} full-run telemetry total or semantic SHA-256 drifted"
        )


def _verify_headline_training_projection(
    training_runner: Any,
    *,
    contract_id: str,
    queue_dir: Path,
    queue_artifact: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the deterministic projection returned by the deep M0 verifier."""

    if not isinstance(verification, Mapping):
        raise CompletionReceiptError(
            f"canonical {contract_id} training verification is invalid"
        )
    contract = training_runner.CONTRACTS[contract_id]
    expected_ids = list(contract.dedicated_queue_run_ids)
    completed_runs = verification.get("completed_training_runs")
    completion_verification = verification.get("completion_verification")
    serial_completion = verification.get("serial_completion_evidence")
    queue_manifest = verification.get("queue_manifest")
    queue_id = verification.get("queue_id")
    try:
        canonical_queue_id = str(uuid.UUID(str(queue_id)))
    except (AttributeError, ValueError) as exc:
        raise CompletionReceiptError(
            f"canonical {contract_id} training queue ID is invalid"
        ) from exc
    if (
        verification.get("schema")
        != "pivot.stageb.headline_m0_queue_verification/v1"
        or verification.get("status") != "passed"
        or verification.get("contract_id") != contract_id
        or verification.get("queue_status") != "completed"
        or canonical_queue_id != queue_id
        or not _is_sha256(verification.get("plan_sha256"))
        or verification.get("ordered_run_ids") != expected_ids
        or not _is_sha256(verification.get("queue_contract_sha256"))
        or not _is_sha256(verification.get("stable_input_closure_digest"))
        or verification.get("active_item") is not None
        or not isinstance(completion_verification, Mapping)
        or completion_verification.get("status") != "passed"
        or not isinstance(serial_completion, Mapping)
        or serial_completion.get("schema")
        != "pivot.stageb.serial_matrix_queue_verification/v1"
        or serial_completion.get("status") != "passed"
        or serial_completion.get("queue_status") != "completed"
        or serial_completion.get("queue_id") != queue_id
        or serial_completion.get("plan_sha256")
        != verification.get("plan_sha256")
        or serial_completion.get("errors") not in (None, [])
        or [
            item.get("run_id") if isinstance(item, Mapping) else None
            for item in serial_completion.get("verified_items", [])
        ]
        != expected_ids
        or not isinstance(queue_manifest, Mapping)
        or queue_manifest.get("path") != queue_artifact.get("path")
        or queue_manifest.get("sha256") != queue_artifact.get("sha256")
        or queue_manifest.get("size_bytes") != queue_artifact.get("size_bytes")
        or type(queue_manifest.get("mtime_ns")) is not int
        or queue_manifest.get("mtime_ns", -1) < 0
        or not isinstance(completed_runs, list)
        or len(completed_runs) != len(expected_ids)
        or not _is_sha256(verification.get("completion_semantic_sha256"))
    ):
        raise CompletionReceiptError(
            f"canonical {contract_id} training queue completion is incomplete"
        )

    completed_input_snapshot = _verify_headline_evidence_snapshot(
        training_runner,
        verification.get("completed_stable_input_snapshot"),
        label=f"{contract_id} queue stable-input",
    )

    expected_contract = {
        "row": contract.expected_row(),
        "headline": bool(contract.headline),
        "matrix_validation_only": bool(contract.matrix_validation_only),
        "token_objective": (
            "edit_bce" if contract_id == "M0" else contract.token_objective
        ),
        "token_objective_scope": contract.token_objective_scope,
    }
    expected_budget = {
        **contract.expected_budget(),
        "gradient_accumulation_steps": 1,
        "amp": True,
        "final_epoch": training_runner.FORMAL_FINAL_EPOCH,
        "final_iteration": training_runner.FORMAL_FINAL_ITERATION,
        "optimizer_state_count": training_runner.FORMAL_OPTIMIZER_STATE_COUNT,
    }
    run_semantic_sha256s: list[str] = []
    verifier_source_digests: set[str] = set()
    for expected_id, expected_seed, run in zip(
        expected_ids, contract.seeds, completed_runs
    ):
        if not isinstance(run, Mapping):
            raise CompletionReceiptError(
                f"canonical {contract_id} deep training evidence is invalid"
            )
        expected_root = contract.canonical_training_root(expected_seed).resolve(
            strict=False
        )
        final_checkpoint = run.get("final_checkpoint")
        ancestry = run.get("ancestry")
        numerical = run.get("numerical")
        telemetry = run.get("telemetry")
        input_closure = run.get("input_closure")
        artifacts = run.get("artifacts")
        queue_binding = run.get("training_queue_binding")
        if (
            run.get("schema")
            != training_runner.COMPLETED_TRAINING_VERIFICATION_SCHEMA
            or run.get("status") != "passed"
            or run.get("run_id") != expected_id
            or run.get("contract_id") != contract_id
            or run.get("seed") != expected_seed
            or Path(str(run.get("run_root", ""))).resolve(strict=False)
            != expected_root
            or run.get("contract") != expected_contract
            or run.get("budget") != expected_budget
            or not isinstance(queue_binding, Mapping)
            or queue_binding.get("contract_id") != contract_id
            or queue_binding.get("queue_id") != queue_id
            or queue_binding.get("plan_sha256")
            != verification.get("plan_sha256")
            or queue_binding.get("queue_contract_sha256")
            != verification.get("queue_contract_sha256")
            or queue_binding.get("stable_input_closure_digest")
            != verification.get("stable_input_closure_digest")
            or queue_binding.get("ordered_run_ids") != expected_ids
            or not isinstance(queue_binding.get("active_item"), Mapping)
            or queue_binding["active_item"].get("item_index")
            != expected_ids.index(expected_id)
            or queue_binding["active_item"].get("run_id") != expected_id
            or queue_binding["active_item"].get("item_status")
            not in {"reserved", "launching", "launched"}
            or not isinstance(
                queue_binding["active_item"].get("orchestration_root"), str
            )
            or not queue_binding["active_item"].get("orchestration_root")
            or not isinstance(queue_binding["active_item"].get("gpu_key"), str)
            or not queue_binding["active_item"].get("gpu_key")
            or not isinstance(queue_binding["active_item"].get("lease_path"), str)
            or not queue_binding["active_item"].get("lease_path")
            or not isinstance(final_checkpoint, Mapping)
            or Path(str(final_checkpoint.get("path", ""))).resolve(strict=False)
            != expected_root / "checkpoint_iter.pth"
            or not _is_sha256(final_checkpoint.get("sha256"))
            or not isinstance(ancestry, Mapping)
            or ancestry.get("status") != "passed"
            or ancestry.get("stage_a_and_scorer_same_source") is not True
            or ancestry.get("b58_ancestry_count") != 0
            or ancestry.get("resume_chain_contiguous") is not True
            or type(ancestry.get("attempt_count")) is not int
            or ancestry.get("attempt_count", 0) < 1
            or type(ancestry.get("resume_count")) is not int
            or ancestry.get("resume_count", -1) < 0
            or not isinstance(ancestry.get("attempt_manifests"), list)
            or len(ancestry.get("attempt_manifests", []))
            != ancestry.get("attempt_count")
            or not isinstance(numerical, Mapping)
            or numerical.get("status") != "passed"
            or numerical.get("amp_enabled") is not True
            or numerical.get("loss_values_all_finite") is not True
            or type(numerical.get("finite_loss_observations")) is not int
            or numerical.get("finite_loss_observations", 0) <= 0
            or numerical.get("max_amp_step_skipped") != 0.0
            or not _is_sha256(numerical.get("evidence_sha256"))
            or not isinstance(telemetry, Mapping)
            or telemetry.get("status") != "passed"
            or type(telemetry.get("sample_rows")) is not int
            or telemetry.get("sample_rows", 0) <= 0
            or not isinstance(telemetry.get("devices"), list)
            or not telemetry.get("devices")
            or not _is_sha256(telemetry.get("evidence_sha256"))
            or not isinstance(input_closure, Mapping)
            or input_closure.get("status") != "passed"
            or input_closure.get("digest")
            != verification.get("stable_input_closure_digest")
            or type(input_closure.get("record_count")) is not int
            or input_closure.get("record_count", 0) <= 0
            or not _is_sha256(input_closure.get("verifier_source_digest"))
            or type(input_closure.get("verifier_source_count")) is not int
            or input_closure.get("verifier_source_count", 0) <= 0
            or not isinstance(artifacts, Mapping)
            or set(artifacts) != {
                "sequence_manifest",
                "launch_manifest",
                "postflight",
            }
        ):
            raise CompletionReceiptError(
                f"canonical {expected_id} deep completed-training evidence drifted"
            )
        _verify_headline_evidence_snapshot(
            training_runner,
            input_closure.get("identity_snapshot"),
            label=f"{expected_id} stable-input",
        )
        run_evidence_snapshot = _verify_headline_evidence_snapshot(
            training_runner,
            run.get("evidence_snapshot"),
            label=f"{expected_id} completed evidence",
        )
        _verify_headline_full_run_telemetry(
            training_runner,
            run_id=expected_id,
            telemetry=telemetry,
            ancestry=ancestry,
            evidence_snapshot=run_evidence_snapshot,
        )
        for name, filename in (
            ("sequence_manifest", "sequence_manifest.json"),
            ("launch_manifest", "launch_manifest.json"),
            ("postflight", "postflight.json"),
        ):
            record = artifacts[name]
            if (
                not isinstance(record, Mapping)
                or Path(str(record.get("path", ""))).resolve(strict=False)
                != expected_root / filename
                or not _is_sha256(record.get("sha256"))
                or type(record.get("size_bytes")) is not int
                or record.get("size_bytes", -1) < 0
            ):
                raise CompletionReceiptError(
                    f"canonical {expected_id} {name} evidence drifted"
                )
        semantic_sha256 = run.get("semantic_sha256")
        semantic_payload = dict(run)
        semantic_payload.pop("semantic_sha256", None)
        if (
            not _is_sha256(semantic_sha256)
            or semantic_sha256 != canonical_json_sha256(semantic_payload)
        ):
            raise CompletionReceiptError(
                f"canonical {expected_id} training semantic SHA-256 drifted"
            )
        run_semantic_sha256s.append(semantic_sha256)
        verifier_source_digests.add(input_closure["verifier_source_digest"])

    if len(set(run_semantic_sha256s)) != len(expected_ids):
        raise CompletionReceiptError(
            f"canonical {contract_id} deep training semantics are not distinct"
        )
    if len(verifier_source_digests) != 1:
        raise CompletionReceiptError(
            f"canonical {contract_id} verifier source digest changed across seeds"
        )
    expected_completion_sha256 = canonical_json_sha256(
        {
            "contract_id": contract_id,
            "queue_id": queue_id,
            "plan_sha256": verification["plan_sha256"],
            "queue_contract_sha256": verification["queue_contract_sha256"],
            "queue_manifest": dict(queue_manifest),
            "ordered_run_ids": expected_ids,
            "run_semantic_sha256s": run_semantic_sha256s,
            "stable_input_snapshot": completed_input_snapshot,
            "serial_completion_evidence": dict(serial_completion),
            "verifier_source_digest": next(iter(verifier_source_digests)),
        }
    )
    if verification.get("completion_semantic_sha256") != expected_completion_sha256:
        raise CompletionReceiptError(
            f"canonical {contract_id} completion semantic SHA-256 drifted"
        )
    return {
        "queue_dir": str(queue_dir.resolve(strict=True)),
        "queue_id": queue_id,
        "plan_sha256": verification["plan_sha256"],
        "queue_contract_sha256": verification["queue_contract_sha256"],
        "stable_input_closure_digest": verification[
            "stable_input_closure_digest"
        ],
        "completion_semantic_sha256": verification[
            "completion_semantic_sha256"
        ],
        "ordered_run_ids": expected_ids,
        "run_semantic_sha256s": run_semantic_sha256s,
    }


def _verify_headline_m0_adapter() -> Mapping[str, Any]:
    """Replay the complete M0/M0N training, validation, and report contract."""

    from tools import aggregate_stageb_headline_m0_validation as aggregate
    from tools import run_stageb_headline_m0 as training_runner
    from tools import run_stageb_headline_m0_validation_queue as validation_queue

    if (
        HEADLINE_M0_VALIDATION_QUEUE.resolve(strict=False)
        != validation_queue.DEFAULT_QUEUE_DIR.resolve(strict=False)
        or HEADLINE_M0_VALIDATION_QUEUE.resolve(strict=False)
        != aggregate.DEFAULT_QUEUE_DIR.resolve(strict=False)
        or HEADLINE_M0_VALIDATION_AGGREGATE.resolve(strict=False)
        != aggregate.DEFAULT_REPORT_PATH.resolve(strict=False)
    ):
        raise CompletionReceiptError(
            "canonical M0/M0N validation or aggregate path constants drifted"
        )
    paths = {
        "m0_training_queue": HEADLINE_M0_TRAINING_QUEUE / "queue.json",
        "m0n_training_queue": HEADLINE_M0N_TRAINING_QUEUE / "queue.json",
        "validation_queue": HEADLINE_M0_VALIDATION_QUEUE / "queue.json",
        "validation_aggregation_input_spec": (
            HEADLINE_M0_VALIDATION_QUEUE / validation_queue.AGGREGATION_SPEC_NAME
        ),
        "validation_aggregate": HEADLINE_M0_VALIDATION_AGGREGATE,
        "training_verifier": Path(training_runner.__file__),
        "training_contract_source": Path(training_runner.source_contracts.__file__),
        "validation_verifier": Path(validation_queue.__file__),
        "aggregate_verifier": Path(aggregate.__file__),
    }
    try:
        artifacts_before = _artifact_records(paths)
        m0_verification = training_runner.verify_training_queue(
            HEADLINE_M0_TRAINING_QUEUE, "M0", require_completed=True
        )
        m0n_verification = training_runner.verify_training_queue(
            HEADLINE_M0N_TRAINING_QUEUE, "M0N", require_completed=True
        )
        validation_manifest = validation_queue.load_queue(
            HEADLINE_M0_VALIDATION_QUEUE
        )
        validation_verification = validation_queue.verify_queue(
            HEADLINE_M0_VALIDATION_QUEUE
        )
        aggregation_spec = _read_json(
            paths["validation_aggregation_input_spec"],
            label="M0/M0N validation aggregation input spec",
        )
        report = aggregate.verify_report(HEADLINE_M0_VALIDATION_AGGREGATE)
    except CompletionReceiptError:
        raise
    except Exception as exc:
        raise CompletionReceiptError(
            f"M0/M0N training/validation semantic replay failed: {exc}"
        ) from exc
    if (
        not isinstance(m0_verification, Mapping)
        or not isinstance(m0n_verification, Mapping)
        or not isinstance(validation_manifest, Mapping)
        or not isinstance(validation_verification, Mapping)
        or not isinstance(report, Mapping)
    ):
        raise CompletionReceiptError(
            "M0/M0N semantic replay returned a non-object result"
        )

    training = {
        "M0": _verify_headline_training_projection(
            training_runner,
            contract_id="M0",
            queue_dir=HEADLINE_M0_TRAINING_QUEUE,
            queue_artifact=artifacts_before["m0_training_queue"],
            verification=m0_verification,
        ),
        "M0N": _verify_headline_training_projection(
            training_runner,
            contract_id="M0N",
            queue_dir=HEADLINE_M0N_TRAINING_QUEUE,
            queue_artifact=artifacts_before["m0n_training_queue"],
            verification=m0n_verification,
        ),
    }
    if (
        HEADLINE_M0_TRAINING_QUEUE.resolve(strict=True)
        == HEADLINE_M0N_TRAINING_QUEUE.resolve(strict=True)
        or training["M0"]["queue_id"] == training["M0N"]["queue_id"]
        or set(training["M0"]["run_semantic_sha256s"])
        & set(training["M0N"]["run_semantic_sha256s"])
    ):
        raise CompletionReceiptError(
            "canonical M0 and M0N training queues or run semantics are not distinct"
        )

    expected_validation_ids = list(validation_queue.RUN_IDS)
    plan = validation_manifest.get("plan")
    plan_training = plan.get("training_queues") if isinstance(plan, Mapping) else None
    plan_items = plan.get("items") if isinstance(plan, Mapping) else None
    if (
        validation_manifest.get("schema") != validation_queue.QUEUE_SCHEMA
        or validation_manifest.get("status") != "completed"
        or not isinstance(plan, Mapping)
        or plan.get("schema") != validation_queue.PLAN_SCHEMA
        or plan.get("queue_id") != validation_verification.get("queue_id")
        or validation_manifest.get("plan_sha256")
        != validation_verification.get("plan_sha256")
        or not _is_sha256(validation_manifest.get("plan_sha256"))
        or Path(str(plan.get("queue_dir", ""))).resolve(strict=False)
        != HEADLINE_M0_VALIDATION_QUEUE.resolve(strict=True)
        or Path(str(plan.get("output_root", ""))).resolve(strict=False)
        != validation_queue.DEFAULT_OUTPUT_ROOT.resolve(strict=False)
        or plan.get("profile") != validation_queue.PROFILE
        or plan.get("ordered_run_ids") != expected_validation_ids
        or plan.get("aggregation_input_spec")
        != {
            "schema": validation_queue.SPEC_SCHEMA,
            "path": str(paths["validation_aggregation_input_spec"].resolve(strict=True)),
        }
        or not isinstance(plan_training, list)
        or len(plan_training) != len(validation_queue.CONTRACT_IDS)
        or not isinstance(plan_items, list)
        or len(plan_items) != len(expected_validation_ids)
    ):
        raise CompletionReceiptError(
            "canonical M0/M0N validation plan is not exact or canonical"
        )

    for contract_id, record in zip(validation_queue.CONTRACT_IDS, plan_training):
        expected = training[contract_id]
        bound_manifest = record.get("manifest_at_creation") if isinstance(record, Mapping) else None
        queue_artifact = artifacts_before[
            "m0_training_queue" if contract_id == "M0" else "m0n_training_queue"
        ]
        if (
            not isinstance(record, Mapping)
            or record.get("contract_id") != contract_id
            or Path(str(record.get("queue_dir", ""))).resolve(strict=False)
            != Path(expected["queue_dir"]).resolve(strict=True)
            or any(
                record.get(key) != expected[key]
                for key in (
                    "queue_id",
                    "plan_sha256",
                    "queue_contract_sha256",
                    "stable_input_closure_digest",
                    "ordered_run_ids",
                )
            )
            or not isinstance(bound_manifest, Mapping)
            or bound_manifest.get("path") != queue_artifact["path"]
            or bound_manifest.get("sha256") != queue_artifact["sha256"]
            or bound_manifest.get("size_bytes") != queue_artifact["size_bytes"]
        ):
            raise CompletionReceiptError(
                f"canonical validation queue is not bound to {contract_id} training"
            )

    for expected_id, item in zip(expected_validation_ids, plan_items):
        contract_id, raw_seed = expected_id.split(":", 1)
        seed = int(raw_seed)
        expected_training = training[contract_id]
        expected_evaluation_root = validation_queue._evaluation_root(
            validation_queue.DEFAULT_OUTPUT_ROOT, expected_id
        )
        if (
            not isinstance(item, Mapping)
            or item.get("run_id") != expected_id
            or item.get("contract_id") != contract_id
            or item.get("train_seed") != seed
            or Path(str(item.get("training_root", ""))).resolve(strict=False)
            != training_runner.CONTRACTS[contract_id].canonical_training_root(seed)
            or Path(str(item.get("training_queue_dir", ""))).resolve(strict=False)
            != Path(expected_training["queue_dir"]).resolve(strict=True)
            or item.get("training_queue_id") != expected_training["queue_id"]
            or item.get("training_queue_plan_sha256")
            != expected_training["plan_sha256"]
            or Path(str(item.get("evaluation_root", ""))).resolve(strict=False)
            != expected_evaluation_root
        ):
            raise CompletionReceiptError(
                f"canonical validation item {expected_id} is not training-bound"
            )

    expected_spec = validation_queue._aggregation_spec_payload(
        plan, str(validation_manifest["plan_sha256"])
    )
    verified_items = validation_verification.get("verified_items")
    verified_spec = validation_verification.get("aggregation_input_spec")
    spec_artifact = artifacts_before["validation_aggregation_input_spec"]
    try:
        validation_queue_id = str(
            uuid.UUID(str(validation_verification.get("queue_id")))
        )
    except (AttributeError, ValueError) as exc:
        raise CompletionReceiptError(
            "canonical M0/M0N validation queue ID is invalid"
        ) from exc
    if (
        aggregation_spec != expected_spec
        or validation_verification.get("schema")
        != validation_queue.VERIFICATION_SCHEMA
        or validation_verification.get("status") != "passed"
        or validation_verification.get("queue_status") != "completed"
        or validation_queue_id != validation_verification.get("queue_id")
        or validation_verification.get("plan_sha256")
        != validation_manifest.get("plan_sha256")
        or validation_verification.get("ordered_run_ids")
        != expected_validation_ids
        or validation_verification.get("errors") not in (None, [])
        or not isinstance(verified_items, list)
        or [
            item.get("run_id") if isinstance(item, Mapping) else None
            for item in verified_items
        ]
        != expected_validation_ids
        or not isinstance(verified_spec, Mapping)
        or verified_spec.get("path") != spec_artifact["path"]
        or verified_spec.get("sha256") != spec_artifact["sha256"]
        or verified_spec.get("size_bytes") != spec_artifact["size_bytes"]
    ):
        raise CompletionReceiptError(
            "canonical M0/M0N validation queue is not exactly completed"
        )

    report_protocol = report.get("protocol")
    report_validation = report.get("validation")
    report_inputs = report.get("inputs")
    report_queue = (
        report_inputs.get("evaluation_queue")
        if isinstance(report_inputs, Mapping)
        else None
    )
    report_spec = (
        report_inputs.get("aggregation_spec")
        if isinstance(report_inputs, Mapping)
        else None
    )
    checkpoint_shas = (
        report_inputs.get("checkpoint_sha256s")
        if isinstance(report_inputs, Mapping)
        else None
    )
    report_digest_payload = dict(report)
    report_digest_payload.pop("created_at_utc", None)
    report_digest = report_digest_payload.pop("report_sha256", None)
    expected_checkpoint_keys = {str(seed) for seed in validation_queue.SEEDS}
    flattened_checkpoint_shas = [
        digest
        for contract_id in validation_queue.CONTRACT_IDS
        for digest in (
            checkpoint_shas.get(contract_id, {}).values()
            if isinstance(checkpoint_shas, Mapping)
            and isinstance(checkpoint_shas.get(contract_id), Mapping)
            else []
        )
    ]
    report_experiments = report.get("experiments")
    if (
        report.get("schema") != aggregate.REPORT_SCHEMA
        or report.get("status") != aggregate.REPORT_STATUS
        or report.get("formal_test_or_strict_result") is not False
        or report.get("reference_experiment") != "M0"
        or report.get("candidate_experiment") != "M0N"
        or report.get("direction") != "M0N_minus_M0"
        or report.get("comparison_claim")
        != "full_token_objective_control_not_labels_only"
        or not _is_sha256(report_digest)
        or report_digest != canonical_json_sha256(report_digest_payload)
        or not isinstance(report_protocol, Mapping)
        or report_protocol.get("profile") != validation_queue.PROFILE
        or report_protocol.get("train_seeds") != list(validation_queue.SEEDS)
        or report_protocol.get("ref_test_access") is not False
        or report_protocol.get("strict_tn_access") is not False
        or report_protocol.get("paired_bootstrap")
        != {
            "iterations": aggregate.FORMAL_BOOTSTRAP_ITERATIONS,
            "confidence": aggregate.FORMAL_BOOTSTRAP_CONFIDENCE,
            "seed": aggregate.FORMAL_BOOTSTRAP_SEED,
            "unit": "image cluster within training seed",
            "seed_first": True,
        }
        or not isinstance(report_validation, Mapping)
        or any(
            report_validation.get(key) is not True
            for key in (
                "pass",
                "training_queues_separate",
                "exact_six_evaluations",
                "record_identities_aligned",
                "runtime_code_data_surface_equal",
                "input_rehash_and_postflight_replayed",
            )
        )
        or not isinstance(report_queue, Mapping)
        or report_queue.get("queue_id") != validation_verification.get("queue_id")
        or report_queue.get("plan_sha256")
        != validation_verification.get("plan_sha256")
        or report_queue.get("verification_schema")
        != validation_queue.VERIFICATION_SCHEMA
        or not isinstance(report_spec, Mapping)
        or report_spec.get("path") != spec_artifact["path"]
        or report_spec.get("sha256") != spec_artifact["sha256"]
        or report_spec.get("size_bytes") != spec_artifact["size_bytes"]
        or dict(report_spec) != dict(verified_spec)
        or not isinstance(checkpoint_shas, Mapping)
        or set(checkpoint_shas) != set(validation_queue.CONTRACT_IDS)
        or any(
            not isinstance(checkpoint_shas.get(contract_id), Mapping)
            or set(checkpoint_shas[contract_id]) != expected_checkpoint_keys
            for contract_id in validation_queue.CONTRACT_IDS
        )
        or len(flattened_checkpoint_shas) != len(expected_validation_ids)
        or any(not _is_sha256(value) for value in flattened_checkpoint_shas)
        or len(set(flattened_checkpoint_shas)) != len(expected_validation_ids)
        or not isinstance(report_experiments, Mapping)
        or set(report_experiments) != set(validation_queue.CONTRACT_IDS)
        or not isinstance(report.get("comparison"), Mapping)
    ):
        raise CompletionReceiptError(
            "canonical M0/M0N aggregate report contract is incomplete"
        )

    try:
        for contract_id, verification in (
            ("M0", m0_verification),
            ("M0N", m0n_verification),
        ):
            training_runner._verify_completed_evidence_current(
                verification["completed_stable_input_snapshot"],
                label=f"{contract_id} adapter-final queue stable inputs",
            )
            for run in verification["completed_training_runs"]:
                run_id = str(run["run_id"])
                training_runner._verify_completed_evidence_current(
                    run["evidence_snapshot"],
                    label=f"{run_id} adapter-final completed evidence",
                )
                training_runner._verify_completed_evidence_current(
                    run["input_closure"]["identity_snapshot"],
                    label=f"{run_id} adapter-final stable inputs",
                )
    except (OSError, ValueError, training_runner.HeadlineM0Error) as exc:
        raise CompletionReceiptError(
            f"M0/M0N completed-training evidence changed after semantic replay: {exc}"
        ) from exc

    artifacts_after = _require_stable_artifacts(
        artifacts_before, paths, label="M0/M0N training/validation/aggregate"
    )
    return {
        "status": "completed",
        "adapter_id": HEADLINE_M0_ADAPTER_ID,
        "contract": {
            "rows": list(validation_queue.CONTRACT_IDS),
            "seeds": list(validation_queue.SEEDS),
            "training_run_ids": [
                *training["M0"]["ordered_run_ids"],
                *training["M0N"]["ordered_run_ids"],
            ],
            "validation_run_ids": expected_validation_ids,
            "training_run_count": 6,
            "validation_run_count": 6,
            "batch_size": training_runner.FORMAL_BATCH_SIZE,
            "optimizer_updates": training_runner.FORMAL_UPDATES,
            "successful_update_batch_slots": training_runner.FORMAL_BATCH_SLOTS,
            "iter_checkpoint_interval": training_runner.FORMAL_CHECKPOINT_INTERVAL,
            "profile": validation_queue.PROFILE,
            "direction": "M0N_minus_M0",
            "bootstrap": {
                "iterations": aggregate.FORMAL_BOOTSTRAP_ITERATIONS,
                "confidence": aggregate.FORMAL_BOOTSTRAP_CONFIDENCE,
                "seed": aggregate.FORMAL_BOOTSTRAP_SEED,
            },
            "training_queues": {
                contract_id: {
                    key: training[contract_id][key]
                    for key in (
                        "queue_id",
                        "plan_sha256",
                        "completion_semantic_sha256",
                    )
                }
                for contract_id in validation_queue.CONTRACT_IDS
            },
            "validation_queue_id": validation_verification["queue_id"],
            "validation_queue_plan_sha256": validation_verification["plan_sha256"],
            "aggregate_report_sha256": report_digest,
        },
        "artifacts": artifacts_after,
        "semantic_replay": {
            "separate_completed_training_queues_verified": True,
            "six_deep_completed_training_runs_verified": True,
            "checkpoint_ancestry_optimizer_resume_numerical_telemetry_verified": True,
            "exact_six_item_validation_queue_verified": True,
            "validation_to_training_binding_verified": True,
            "canonical_validation_output_and_spec_verified": True,
            "queue_bound_aggregate_recomputed": True,
            "aggregate_self_digest_verified": True,
            "no_ref_test_or_strict_access": True,
        },
    }


def _verify_table_b_v2_adapter() -> Mapping[str, Any]:
    """Replay formal Table-B v2 training, validation, and matched aggregate."""

    from tools import aggregate_stageb_table_b_v2_validation as aggregate
    from tools import run_stageb_table_b_v2_queue as training_queue
    from tools import run_stageb_table_b_v2_validation_queue as validation_queue
    from util import stage_b_table_b_v2_contract as contract

    paths = {
        "training_queue": TABLE_B_V2_TRAINING_QUEUE / "queue.json",
        "training_source_plan": (
            TABLE_B_V2_TRAINING_QUEUE / training_queue.SOURCE_PLAN_NAME
        ),
        "training_scope_plan": (
            TABLE_B_V2_TRAINING_QUEUE / training_queue.SCOPE_PLAN_NAME
        ),
        "training_completion_attestation": (
            TABLE_B_V2_TRAINING_QUEUE / training_queue.COMPLETION_NAME
        ),
        "validation_queue": TABLE_B_V2_VALIDATION_QUEUE / "queue.json",
        "validation_input_spec": (
            TABLE_B_V2_VALIDATION_QUEUE / validation_queue.VALIDATION_SPEC_NAME
        ),
        "validation_aggregate": TABLE_B_V2_VALIDATION_AGGREGATE,
        "training_verifier": Path(training_queue.__file__),
        "validation_verifier": Path(validation_queue.__file__),
        "aggregate_verifier": Path(aggregate.__file__),
        "training_contract_source": Path(contract.__file__),
    }
    try:
        artifacts_before = _artifact_records(paths)
        training = training_queue.verify_formal_queue(
            TABLE_B_V2_TRAINING_QUEUE, persist=False
        )
        validation_manifest = validation_queue.load_queue(
            TABLE_B_V2_VALIDATION_QUEUE
        )
        validation = validation_queue.verify_queue(TABLE_B_V2_VALIDATION_QUEUE)
        persisted_report = _read_json(
            TABLE_B_V2_VALIDATION_AGGREGATE,
            label="Table-B v2 validation aggregate",
        )
        replayed_report = aggregate.aggregate(TABLE_B_V2_VALIDATION_QUEUE)
    except CompletionReceiptError:
        raise
    except Exception as exc:
        raise CompletionReceiptError(
            f"Table-B v2 training/validation semantic replay failed: {exc}"
        ) from exc

    expected_training_ids = list(contract.FORMAL_RUN_IDS)
    training_identity = training.get("queue")
    common_inputs = training.get("common_input_replay")
    if (
        training.get("schema") != training_queue.COMPLETION_SCHEMA
        or training.get("status") != "passed"
        or training.get("profile") != contract.FORMAL_PROFILE
        or training.get("ordered_run_ids") != expected_training_ids
        or set(training.get("runs", {})) != set(expected_training_ids)
        or training.get("semantic_sha256")
        != training_queue._semantic_sha256(training)
        or not isinstance(training_identity, Mapping)
        or not isinstance(training_identity.get("queue_id"), str)
        or not training_identity.get("queue_id")
        or len(str(training_identity.get("plan_sha256", ""))) != 64
        or not isinstance(common_inputs, Mapping)
        or common_inputs.get("status") != "passed"
        or common_inputs.get("all_six_runs_share_identical_common_inputs") is not True
        or common_inputs.get("only_declared_condition_inputs_differ") is not True
    ):
        raise CompletionReceiptError(
            "canonical Table-B v2 training completion contract is incomplete"
        )

    expected_validation_ids = list(validation_queue.RUN_IDS)
    verified_items = validation.get("verified_items")
    bound_training = validation_manifest.get("plan", {}).get("training_queue")
    if (
        validation_manifest.get("status") != "completed"
        or validation.get("schema") != validation_queue.VERIFICATION_SCHEMA
        or validation.get("status") != "passed"
        or validation.get("queue_status") != "completed"
        or validation.get("ordered_seeds") != list(validation_queue.SEEDS)
        or validation.get("phase_order_per_seed")
        != list(validation_queue.PHASE_ORDER)
        or validation.get("total_phase_count")
        != len(validation_queue.SEEDS) * len(validation_queue.PHASE_ORDER)
        or validation.get("errors") not in (None, [])
        or not isinstance(verified_items, list)
        or [item.get("run_id") for item in verified_items]
        != expected_validation_ids
        or not isinstance(bound_training, Mapping)
        or Path(str(bound_training.get("queue_dir", ""))).resolve(strict=False)
        != TABLE_B_V2_TRAINING_QUEUE.resolve(strict=True)
        or bound_training.get("queue_id") != training_identity.get("queue_id")
        or bound_training.get("plan_sha256")
        != training_identity.get("plan_sha256")
        or bound_training.get("completion_semantic_sha256")
        != training.get("semantic_sha256")
    ):
        raise CompletionReceiptError(
            "canonical Table-B v2 validation queue is not exact or training-bound"
        )

    if _strip_volatile(persisted_report) != _strip_volatile(replayed_report):
        raise CompletionReceiptError(
            "Table-B v2 aggregate differs from semantic replay"
        )
    validation_claims = persisted_report.get("validation")
    report_queue = persisted_report.get("inputs", {}).get(
        "formal_v2_validation_queue"
    )
    if (
        persisted_report.get("schema") != aggregate.REPORT_SCHEMA
        or persisted_report.get("status")
        != "validated_formal_v2_supplemental_diagnostic"
        or persisted_report.get("formal_global_fpr_eligible") is not False
        or persisted_report.get("formal_evaluation_protocol", {}).get(
            "training_source_contract"
        )
        != "table_b_v2_formal"
        or not isinstance(validation_claims, Mapping)
        or any(
            validation_claims.get(key) is not True
            for key in (
                "formal_v2_training_resolver_replayed",
                "exact_three_seed_six_phase_queue_replayed",
                "validation_queue_spec_replayed",
                "shared_gpu_lease_queue_verified",
            )
        )
        or not isinstance(report_queue, Mapping)
        or report_queue.get("queue_id") != validation["queue_id"]
        or report_queue.get("plan_sha256") != validation["plan_sha256"]
        or report_queue.get("training_queue_id")
        != training_identity.get("queue_id")
        or report_queue.get("training_queue_plan_sha256")
        != training_identity.get("plan_sha256")
        or report_queue.get("ordered_seeds") != list(validation_queue.SEEDS)
        or report_queue.get("phase_order_per_seed")
        != list(validation_queue.PHASE_ORDER)
        or report_queue.get("total_phase_count")
        != len(validation_queue.SEEDS) * len(validation_queue.PHASE_ORDER)
    ):
        raise CompletionReceiptError("Table-B v2 aggregate contract is incomplete")
    artifacts_after = _require_stable_artifacts(
        artifacts_before, paths, label="Table-B v2"
    )
    return {
        "status": "completed",
        "adapter_id": TABLE_B_V2_ADAPTER_ID,
        "contract": {
            "rows": list(contract.TABLE_B_SCOPE_BY_ID),
            "seeds": list(contract.SEEDS),
            "training_run_ids": expected_training_ids,
            "validation_run_ids": expected_validation_ids,
            "training_run_count": len(expected_training_ids),
            "validation_job_count": len(expected_validation_ids),
            "validation_phase_count": (
                len(validation_queue.SEEDS) * len(validation_queue.PHASE_ORDER)
            ),
            "batch_size": contract.FORMAL_BATCH_SIZE,
            "optimizer_updates": contract.FORMAL_TRAIN_UPDATES,
            "iter_checkpoint_interval": contract.FORMAL_CHECKPOINT_INTERVAL,
            "successful_update_batch_slots_per_run": (
                contract.FORMAL_SUCCESSFUL_UPDATE_BATCH_SLOTS
            ),
            "profile": contract.FORMAL_PROFILE,
            "formal_global_fpr_eligible": False,
        },
        "artifacts": {
            **artifacts_after,
        },
        "semantic_replay": {
            "exact_six_training_runs_verified": True,
            "common_input_contract_verified": True,
            "exact_three_seed_six_phase_validation_verified": True,
            "validation_to_training_binding_verified": True,
            "queue_bound_aggregate_recomputed": True,
            "supplemental_diagnostic_scope_preserved": True,
        },
    }


def _verify_table_d_adapter() -> Mapping[str, Any]:
    """Replay the exact Table-D training/validation matrix and aggregate."""

    from tools import aggregate_stageb_table_d_formal_matrix as aggregate
    from tools import run_stageb_table_d_formal_queue as training_queue
    from tools import run_stageb_table_d_matrix_validation_queue as validation_queue

    paths = {
        "training_queue": TABLE_D_TRAINING_QUEUE / "queue.json",
        "training_source_plan": TABLE_D_TRAINING_QUEUE / training_queue.SOURCE_PLAN_NAME,
        "training_scope_plan": TABLE_D_TRAINING_QUEUE / training_queue.SCOPE_PLAN_NAME,
        "training_completion_attestation": (
            TABLE_D_TRAINING_QUEUE / training_queue.COMPLETION_NAME
        ),
        "validation_queue": TABLE_D_VALIDATION_QUEUE / "queue.json",
        "aggregation_spec": (
            TABLE_D_VALIDATION_QUEUE / validation_queue.AGGREGATION_SPEC_NAME
        ),
        "evaluation_scope_plan": (
            TABLE_D_VALIDATION_QUEUE / validation_queue.EVALUATION_SCOPE_PLAN_NAME
        ),
        "validation_aggregate": TABLE_D_VALIDATION_AGGREGATE,
        "training_verifier": Path(training_queue.__file__),
        "validation_verifier": Path(validation_queue.__file__),
        "aggregate_verifier": Path(aggregate.__file__),
    }
    try:
        artifacts_before = _artifact_records(paths)
        training = training_queue.verify_training_queue(
            TABLE_D_TRAINING_QUEUE, persist=False
        )
        validation_manifest = validation_queue.load_queue(TABLE_D_VALIDATION_QUEUE)
        validation = validation_queue.verify_queue(TABLE_D_VALIDATION_QUEUE)
        persisted_report = _read_json(
            TABLE_D_VALIDATION_AGGREGATE,
            label="Table-D formal matrix aggregate",
        )
        diagnostics_binding = persisted_report.get("inputs", {}).get(
            "final_diagnostics"
        )
        if not isinstance(diagnostics_binding, Mapping):
            raise CompletionReceiptError(
                "Table-D aggregate final-diagnostics binding is missing"
            )
        final_diagnostics_path = None
        if diagnostics_binding.get("status") == "bound_and_replayed":
            report_record = diagnostics_binding.get("report")
            if not isinstance(report_record, Mapping):
                raise CompletionReceiptError(
                    "Table-D bound final-diagnostics report is missing"
                )
            final_diagnostics_path = Path(str(report_record.get("path", "")))
            paths["final_diagnostics_report"] = final_diagnostics_path
            artifacts_before["final_diagnostics_report"] = file_record(
                final_diagnostics_path
            )
        elif diagnostics_binding.get("status") != "not_bound":
            raise CompletionReceiptError(
                "Table-D aggregate final-diagnostics status is invalid"
            )
        replayed_report = aggregate.aggregate(
            TABLE_D_VALIDATION_QUEUE,
            final_diagnostics_report=final_diagnostics_path,
        )
    except CompletionReceiptError:
        raise
    except Exception as exc:
        raise CompletionReceiptError(
            f"Table-D training/validation semantic replay failed: {exc}"
        ) from exc

    expected_training_ids = list(training_queue.RUN_IDS)
    training_identity = training.get("queue")
    if (
        training.get("schema") != training_queue.COMPLETION_SCHEMA
        or training.get("status") != "passed"
        or training.get("profile") != training_queue.PROFILE
        or training.get("ordered_run_ids") != expected_training_ids
        or training.get("formal_training_contract")
        != training_queue.FORMAL_TRAINING_CONTRACT
        or training.get("active_item_identity_replayed") is not True
        or set(training.get("runs", {})) != set(expected_training_ids)
        or training.get("semantic_sha256") != training_queue._semantic_sha(training)
        or not isinstance(training_identity, Mapping)
        or not isinstance(training_identity.get("queue_id"), str)
        or not training_identity.get("queue_id")
        or len(str(training_identity.get("plan_sha256", ""))) != 64
    ):
        raise CompletionReceiptError(
            "canonical Table-D training completion contract is incomplete"
        )

    expected_validation_ids = list(validation_queue.JOB_IDS)
    verified_items = validation.get("verified_items")
    bound_training = validation_manifest.get("plan", {}).get("training_queue")
    final_verification = validation.get("final_verification")
    if (
        validation_manifest.get("status") != "completed"
        or validation.get("schema") != validation_queue.VERIFICATION_SCHEMA
        or validation.get("status") != "passed"
        or validation.get("queue_status") != "completed"
        or validation.get("profile") != validation_queue.PROFILE
        or validation.get("ordered_job_ids") != expected_validation_ids
        or validation.get("errors") not in (None, [])
        or not isinstance(verified_items, list)
        or len(verified_items) != len(expected_validation_ids)
        or [item.get("job_id") for item in verified_items]
        != expected_validation_ids
        or not isinstance(bound_training, Mapping)
        or Path(str(bound_training.get("queue_dir", ""))).resolve(strict=False)
        != TABLE_D_TRAINING_QUEUE.resolve(strict=True)
        or bound_training.get("queue_id") != training_identity.get("queue_id")
        or bound_training.get("plan_sha256")
        != training_identity.get("plan_sha256")
        or bound_training.get("completion_semantic_sha256")
        != training.get("semantic_sha256")
        or not isinstance(final_verification, Mapping)
        or final_verification.get("schema")
        != validation_queue.FINAL_VERIFICATION_SCHEMA
        or final_verification.get("queue_id") != validation["queue_id"]
        or final_verification.get("plan_sha256") != validation["plan_sha256"]
        or final_verification.get("ordered_job_ids") != expected_validation_ids
        or final_verification.get("training_completion_semantic_sha256")
        != training.get("semantic_sha256")
    ):
        raise CompletionReceiptError(
            "canonical Table-D validation queue is not exact or training-bound"
        )

    if _strip_volatile(persisted_report) != _strip_volatile(replayed_report):
        raise CompletionReceiptError(
            "Table-D formal matrix aggregate differs from semantic replay"
        )
    validation_claims = persisted_report.get("validation")
    report_queue = persisted_report.get("inputs", {}).get("evaluation_queue")
    if (
        persisted_report.get("schema") != aggregate.REPORT_SCHEMA
        or persisted_report.get("status") != "validated_matrix_validation_only"
        or persisted_report.get("formal_test_or_strict_result") is not False
        or set(persisted_report.get("experiments", {}))
        != {*training_queue.ROWS, "S3_rank"}
        or set(persisted_report.get("comparisons", {}))
        != {
            "clean_ownership_vs_S0",
            "S2F_minus_S2_full_objective_control",
            "S3_confidence_minus_rank_diagnostic",
        }
        or not isinstance(validation_claims, Mapping)
        or any(
            validation_claims.get(key) is not True
            for key in (
                "pass",
                "exact_fifteen_final_jobs",
                "exact_three_s3_rank_jobs",
                "record_identities_aligned",
                "runtime_code_data_surface_equal",
                "input_rehash_and_postflight_replayed",
                "training_authority_replayed",
            )
        )
        or not isinstance(report_queue, Mapping)
        or report_queue.get("queue_id") != validation["queue_id"]
        or report_queue.get("plan_sha256") != validation["plan_sha256"]
        or report_queue.get("final_verification") != final_verification
        or diagnostics_binding.get("separate_from_matrix_metrics") is not True
        or diagnostics_binding.get("pooled_into_matrix_results") is not False
        or persisted_report.get("protocol", {}).get("paired_bootstrap")
        != {
            "iterations": aggregate.FORMAL_BOOTSTRAP_ITERATIONS,
            "confidence": aggregate.FORMAL_BOOTSTRAP_CONFIDENCE,
            "seed": aggregate.FORMAL_BOOTSTRAP_SEED,
            "unit": "image cluster within training seed",
            "seed_first": True,
        }
    ):
        raise CompletionReceiptError("Table-D formal aggregate contract is incomplete")
    artifacts_after = _require_stable_artifacts(
        artifacts_before, paths, label="Table-D"
    )
    return {
        "status": "completed",
        "adapter_id": TABLE_D_ADAPTER_ID,
        "contract": {
            "rows": list(training_queue.ROWS),
            "seeds": list(training_queue.SEEDS),
            "training_run_ids": expected_training_ids,
            "validation_job_ids": expected_validation_ids,
            "training_run_count": len(expected_training_ids),
            "validation_job_count": len(expected_validation_ids),
            **dict(training_queue.FORMAL_TRAINING_CONTRACT),
            "profile": training_queue.PROFILE,
            "bootstrap": {
                "iterations": aggregate.FORMAL_BOOTSTRAP_ITERATIONS,
                "confidence": aggregate.FORMAL_BOOTSTRAP_CONFIDENCE,
                "seed": aggregate.FORMAL_BOOTSTRAP_SEED,
            },
        },
        "artifacts": {
            **artifacts_after,
        },
        "semantic_replay": {
            "exact_fifteen_training_runs_verified": True,
            "fixed_training_runtime_contract_verified": True,
            "exact_eighteen_validation_jobs_verified": True,
            "durable_final_validation_receipt_verified": True,
            "validation_to_training_binding_verified": True,
            "queue_bound_aggregate_recomputed": True,
            "no_ref_test_or_strict_access": True,
        },
    }


def _verify_g0c_queue_adapter() -> Mapping[str, Any]:
    """Replay the canonical soak, G0c queues, and six-item Table-A aggregate."""

    from tools import aggregate_stageb_table_a_results as aggregate
    from tools import run_stageb_table_a_g0c_soak_queue as soak_queue
    from tools import run_stageb_table_a_g0c_queues as queues

    expected_training_ids = [f"G0c:{seed}" for seed in queues.FORMAL_SEEDS]
    expected_validation_ids = [
        *[f"candidate:{seed}" for seed in queues.FORMAL_SEEDS],
        *[f"g0c:{seed}" for seed in queues.FORMAL_SEEDS],
    ]
    paths = {
        "soak_queue": G0C_SOAK_QUEUE / "queue.json",
        "training_queue": G0C_TRAINING_QUEUE / "queue.json",
        "validation_queue": G0C_VALIDATION_QUEUE / "queue.json",
        "validation_aggregate": G0C_VALIDATION_AGGREGATE,
        "soak_queue_verifier": Path(soak_queue.__file__),
        "queue_verifier": Path(queues.__file__),
        "aggregate_verifier": Path(aggregate.__file__),
        "g0c_training_runner_source": Path(queues.training_runner.__file__),
    }
    try:
        artifacts_before = _artifact_records(paths)
        soak = soak_queue.verify_queue(G0C_SOAK_QUEUE)
        raw_soak_evidence = soak.get("completion_evidence")
        if not isinstance(raw_soak_evidence, Mapping):
            raise CompletionReceiptError(
                "canonical G0c soak completion evidence is missing"
            )
        soak_paths = {
            label: Path(str(raw_soak_evidence[label]["path"]))
            for label in ("soak_plan", "postflight", "checkpoint", "soak_seal")
        }
        soak_artifacts_before = _artifact_records(soak_paths)
        for label, observed in soak_artifacts_before.items():
            attested = raw_soak_evidence[label]
            if (
                not isinstance(attested, Mapping)
                or attested.get("path") != observed["path"]
                or attested.get("sha256") != observed["sha256"]
                or attested.get("size_bytes") != observed["size_bytes"]
            ):
                raise CompletionReceiptError(
                    f"canonical G0c soak {label} changed after native replay"
                )
        training = queues.verify_queue(G0C_TRAINING_QUEUE)
        training_manifest = queues.load_queue(G0C_TRAINING_QUEUE)
        validation = queues.verify_queue(G0C_VALIDATION_QUEUE)
        report = aggregate.verify_report(G0C_VALIDATION_AGGREGATE)
    except CompletionReceiptError:
        raise
    except (
        soak_queue.G0cSoakQueueError,
        queues.G0cQueueError,
        aggregate.TableAAggregationError,
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise CompletionReceiptError(
            f"G0c training/validation semantic replay failed: {exc}"
        ) from exc

    soak_evidence = soak.get("completion_evidence")
    if (
        soak.get("schema") != soak_queue.VERIFICATION_SCHEMA
        or soak.get("status") != "passed"
        or soak.get("queue_status") != "completed"
        or soak.get("errors") not in (None, [])
        or soak.get("run_id") != soak_queue.RUN_ID
        or not isinstance(soak_evidence, Mapping)
        or soak_evidence.get("schema") != soak_queue.COMPLETION_SCHEMA
        or soak_evidence.get("run_id") != soak_queue.RUN_ID
        or soak_evidence.get("seed") != soak_queue.SEED
        or soak_evidence.get("micro_batch_size") != soak_queue.MICRO_BATCH_SIZE
        or soak_evidence.get("gradient_accumulation_steps")
        != soak_queue.GRADIENT_ACCUMULATION_STEPS
        or soak_evidence.get("effective_global_batch")
        != soak_queue.EFFECTIVE_GLOBAL_BATCH
        or soak_evidence.get("optimizer_updates") != soak_queue.OPTIMIZER_UPDATES
        or soak_evidence.get("fresh_only") is not True
        or soak_evidence.get("lease_release_gate")
        != "durable_completion_reload_plus_full_native_replay"
    ):
        raise CompletionReceiptError(
            "canonical G0c U50 soak queue is not exactly completed and replayed"
        )
    soak_seal = soak_evidence.get("soak_seal")
    if not isinstance(soak_seal, Mapping):
        raise CompletionReceiptError("canonical G0c soak completion lacks its seal")
    expected_seal = {
        "path": str(Path(str(soak_seal.get("path", ""))).resolve(strict=True)),
        "sha256": soak_seal.get("sha256"),
    }
    planned_training_items = training_manifest.get("plan", {}).get("items")
    if (
        not isinstance(planned_training_items, list)
        or len(planned_training_items) != len(expected_training_ids)
    ):
        raise CompletionReceiptError("canonical G0c training plan is missing")
    for item in planned_training_items:
        formal_plan = item.get("expected_plan") if isinstance(item, Mapping) else None
        bound_input = (
            formal_plan.get("inputs", {}).get("g0c_soak_seal")
            if isinstance(formal_plan, Mapping)
            else None
        )
        declared_seal = (
            formal_plan.get("soak_seal")
            if isinstance(formal_plan, Mapping)
            else None
        )
        if (
            not isinstance(bound_input, Mapping)
            or not isinstance(declared_seal, Mapping)
            or {
                "path": str(
                    Path(str(bound_input.get("path", ""))).resolve(strict=False)
                ),
                "sha256": bound_input.get("sha256"),
            }
            != expected_seal
            or {
                "path": str(
                    Path(str(declared_seal.get("path", ""))).resolve(strict=False)
                ),
                "sha256": declared_seal.get("sha256"),
            }
            != expected_seal
            or declared_seal.get("schema")
            != queues.training_runner.SOAK_SEAL_SCHEMA
        ):
            raise CompletionReceiptError(
                "canonical G0c training queue is not bound to the queue-owned U50 soak seal"
            )

    queue_contracts = (
        (
            "training",
            training,
            queues.TRAINING_KIND,
            expected_training_ids,
        ),
        (
            "validation",
            validation,
            queues.VALIDATION_KIND,
            expected_validation_ids,
        ),
    )
    for label, verification, queue_kind, expected_ids in queue_contracts:
        verified_items = verification.get("verified_items")
        if (
            verification.get("status") != "passed"
            or verification.get("queue_status") != "completed"
            or verification.get("queue_kind") != queue_kind
            or verification.get("ordered_run_ids") != expected_ids
            or verification.get("errors") not in (None, [])
            or not isinstance(verification.get("queue_id"), str)
            or not verification.get("queue_id")
            or len(str(verification.get("plan_sha256", ""))) != 64
            or not isinstance(verified_items, list)
            or [item.get("run_id") for item in verified_items] != expected_ids
        ):
            raise CompletionReceiptError(
                f"canonical G0c {label} queue is not exactly completed and replayed"
            )
    if (
        report.get("schema") != aggregate.SCHEMA
        or report.get("status") != "passed"
        or report.get("profile") != aggregate.table_a.VALIDATION_PROFILE
        or report.get("formal_seeds") != list(queues.FORMAL_SEEDS)
        or len(str(report.get("report_sha256", ""))) != 64
    ):
        raise CompletionReceiptError(
            "canonical G0c/Table-A validation aggregate contract is incomplete"
        )
    artifacts_after = _require_stable_artifacts(
        artifacts_before, paths, label="G0c soak/queue/aggregate"
    )
    soak_artifacts_after = _require_stable_artifacts(
        soak_artifacts_before, soak_paths, label="G0c native soak"
    )

    return {
        "status": "completed",
        "adapter_id": G0C_ADAPTER_ID,
        "contract": {
            "soak_run_id": soak_queue.RUN_ID,
            "soak_seed": soak_queue.SEED,
            "soak_micro_batch_size": soak_queue.MICRO_BATCH_SIZE,
            "soak_gradient_accumulation_steps": (
                soak_queue.GRADIENT_ACCUMULATION_STEPS
            ),
            "soak_effective_global_batch": soak_queue.EFFECTIVE_GLOBAL_BATCH,
            "soak_optimizer_updates": soak_queue.OPTIMIZER_UPDATES,
            "seeds": list(queues.FORMAL_SEEDS),
            "training_run_ids": expected_training_ids,
            "validation_run_ids": expected_validation_ids,
            "training_run_count": len(expected_training_ids),
            "validation_run_count": len(expected_validation_ids),
            "batch_size": queues.training_runner.REQUIRED_EFFECTIVE_GLOBAL_BATCH,
            "optimizer_updates": queues.training_runner.FORMAL_OPTIMIZER_UPDATES,
            "profile": aggregate.table_a.VALIDATION_PROFILE,
            "training_queue_id": training["queue_id"],
            "training_queue_plan_sha256": training["plan_sha256"],
            "validation_queue_id": validation["queue_id"],
            "validation_queue_plan_sha256": validation["plan_sha256"],
            "aggregate_report_sha256": report["report_sha256"],
        },
        "artifacts": {
            **artifacts_after,
            "soak_plan": soak_artifacts_after["soak_plan"],
            "soak_postflight": soak_artifacts_after["postflight"],
            "soak_checkpoint": soak_artifacts_after["checkpoint"],
            "soak_seal": soak_artifacts_after["soak_seal"],
        },
        "semantic_replay": {
            "queue_owned_u50_soak_verified": True,
            "formal_training_bound_to_soak_seal": True,
            "training_queue_verified": True,
            "validation_queue_verified": True,
            "exact_three_seed_training_surface": True,
            "exact_six_item_validation_surface": True,
            "aggregate_recomputed": True,
        },
    }


BLOCK_ADAPTER_REGISTRY: Mapping[str, BlockAdapter] = {
    "A": BlockAdapter(
        adapter_id=HEADLINE_M0_ADAPTER_ID,
        verifier=_verify_headline_m0_adapter,
    ),
    "B": BlockAdapter(
        adapter_id=TABLE_B_V2_ADAPTER_ID,
        verifier=_verify_table_b_v2_adapter,
    ),
    "C": BlockAdapter(
        adapter_id=TABLE_C_ADAPTER_ID,
        verifier=_verify_table_c,
    ),
    "D": BlockAdapter(
        adapter_id=TABLE_D_ADAPTER_ID,
        verifier=_verify_table_d_adapter,
    ),
    "G0c": BlockAdapter(
        adapter_id=G0C_ADAPTER_ID,
        verifier=_verify_g0c_queue_adapter,
    ),
}


def _registry_projection() -> dict[str, Any]:
    if tuple(BLOCK_ADAPTER_REGISTRY) != BLOCKS:
        raise CompletionReceiptError(
            f"completion adapter registry must be ordered exactly as {BLOCKS}"
        )
    return {
        block: {
            "adapter_id": BLOCK_ADAPTER_REGISTRY[block].adapter_id,
            "state": (
                "sealed" if BLOCK_ADAPTER_REGISTRY[block].verifier else "unsealed"
            ),
        }
        for block in BLOCKS
    }


def _derive_receipt_payload() -> dict[str, Any]:
    registry = _registry_projection()
    completed: dict[str, Any] = {}
    for block in BLOCKS:
        adapter = BLOCK_ADAPTER_REGISTRY[block]
        if adapter.verifier is None:
            raise CompletionReceiptError(
                f"paper block {block} adapter is unsealed: {adapter.unsealed_reason}"
            )
        try:
            evidence = copy.deepcopy(dict(adapter.verifier()))
        except CompletionReceiptError:
            raise
        except Exception as exc:
            raise CompletionReceiptError(
                f"paper block {block} semantic replay failed: {exc}"
            ) from exc
        if (
            set(evidence) != {
                "status",
                "adapter_id",
                "contract",
                "artifacts",
                "semantic_replay",
            }
            or evidence.get("status") != "completed"
            or evidence.get("adapter_id") != adapter.adapter_id
            or not isinstance(evidence.get("contract"), Mapping)
            or not isinstance(evidence.get("artifacts"), Mapping)
            or not evidence["artifacts"]
            or not isinstance(evidence.get("semantic_replay"), Mapping)
            or any(value is not True for value in evidence["semantic_replay"].values())
        ):
            raise CompletionReceiptError(
                f"paper block {block} adapter returned an incomplete contract"
            )
        completed[block] = evidence

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "completed",
        "completed_before_final_gate": True,
        "all_training_validation_diagnostics_completed": True,
        "adapter_registry": {
            "schema": REGISTRY_SCHEMA,
            "blocks": registry,
            "registry_source": file_record(Path(__file__)),
        },
        "completed_blocks": completed,
    }
    payload["receipt_sha256"] = canonical_json_sha256(payload)
    return payload


def _canonical_receipt_path(path: Path, *, must_exist: bool) -> Path:
    expected = CANONICAL_RECEIPT_PATH.resolve(strict=False)
    observed = path.expanduser().resolve(strict=False)
    if observed != expected:
        raise CompletionReceiptError(
            f"completion receipt path is not canonical: {observed}"
        )
    if must_exist:
        observed = path.expanduser().resolve(strict=True)
        if observed != expected:
            raise CompletionReceiptError("completion receipt resolves non-canonically")
    return observed


def verify_receipt(path: Path | None = None) -> dict[str, Any]:
    receipt_path = _canonical_receipt_path(
        CANONICAL_RECEIPT_PATH if path is None else path,
        must_exist=True,
    )
    observed = _read_json(receipt_path, label="paper ablation completion receipt")
    without_hash = copy.deepcopy(observed)
    receipt_sha = without_hash.pop("receipt_sha256", None)
    if receipt_sha != canonical_json_sha256(without_hash):
        raise CompletionReceiptError("completion receipt self SHA-256 mismatch")
    expected = _derive_receipt_payload()
    if observed != expected:
        raise CompletionReceiptError(
            "completion receipt differs from fresh adapter semantic replay"
        )
    return observed


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path = _canonical_receipt_path(path, must_exist=False)
    if path.exists():
        raise CompletionReceiptError(f"completion receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    rendered = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    try:
        with temporary.open("x", encoding="ascii") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        _rename_noreplace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def build_receipt() -> dict[str, Any]:
    path = _canonical_receipt_path(CANONICAL_RECEIPT_PATH, must_exist=False)
    if path.exists():
        raise CompletionReceiptError(f"completion receipt already exists: {path}")
    payload = _derive_receipt_payload()
    _write_exclusive(path, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "dry-run", "verify", "registry"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            payload = build_receipt()
        elif args.command == "verify":
            payload = verify_receipt()
        elif args.command == "registry":
            payload = {
                "schema": REGISTRY_SCHEMA,
                "blocks": _registry_projection(),
            }
        else:
            payload = _derive_receipt_payload()
    except (CompletionReceiptError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
