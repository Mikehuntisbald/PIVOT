#!/usr/bin/env python3
"""Finalize the supplemental Table-C dependency-closure attestation.

The historical dependency auditor remains the authority for the old
attestation schema.  This tool creates a fresh old-schema replay after both
training queues complete, proves that its static evidence is identical to the
sealed preflight attestation, upgrades only the remaining queue's completion
policy, and adds an explicit finalization lineage record.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import audit_stageb_table_c_dependency_closure as audit
from tools.stageb_dependency_audit import DependencyAuditError


FINALIZATION_SCHEMA = (
    "pivot.stageb.table_c_dependency_closure_attestation_finalization/v1"
)
FINAL_VERIFICATION_SCHEMA = (
    "pivot.stageb.table_c_dependency_closure_final_verification/v1"
)

DEFAULT_PREFLIGHT_ATTESTATION = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/attestations/"
    "table_c_dependency_closure_preflight_20260718.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/attestations/"
    "table_c_dependency_closure_final.json"
)

CANONICAL_PREFLIGHT_ATTESTATION_SHA256 = (
    "1811a3b3dbb03eb51e8d6129395845cc461c59d02f03235ed1479bff42b8652c"
)
CANONICAL_PREFLIGHT_FILE_SHA256 = (
    "79d1cb09563a4cd21be0ffec2a3d9cc0193f9fe4a59b645fc765a5b4dd3cd509"
)

COMPLETION_COUNTS = {
    "completed_l0_l4": len(audit.COMPLETED_RUN_IDS),
    "remaining_table_c": len(audit.REMAINING_RUN_IDS),
}

_OLD_ATTESTATION_KEYS = frozenset(
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
_FINAL_ATTESTATION_KEYS = _OLD_ATTESTATION_KEYS | {"finalization"}
_STATIC_IDENTITY_KEYS = (
    "schema",
    "repository_root",
    "evidence_class",
    "claim_scope",
    "limitations",
    "config_entries",
    "config_import_chains",
    "training_evidence",
    "dependency_closure",
    "auditor_sources",
)


class TableCFinalizationError(RuntimeError):
    """The final Table-C dependency attestation failed closed."""


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TableCFinalizationError(f"{label} is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise TableCFinalizationError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TableCFinalizationError(f"{label} must be a JSON object: {path}")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise TableCFinalizationError(
            f"{label} keys differ: missing={missing!r}, extra={extra!r}"
        )


def _config_entries(payload: Mapping[str, Any]) -> Mapping[str, str | Path]:
    value = payload.get("config_entries")
    if not isinstance(value, Mapping):
        raise TableCFinalizationError("attestation config_entries are missing")
    return value


def _completion_verification(count: int) -> dict[str, Any]:
    return {
        "schema": "pivot.stageb.serial_matrix_queue_verification/v1",
        "status": "passed",
        "verified_item_count": count,
    }


def _require_completed_binding(
    record: Any,
    *,
    role: str,
    count: int,
    status_policy: str,
) -> None:
    if not isinstance(record, Mapping):
        raise TableCFinalizationError(f"{role} queue binding is missing")
    if record.get("observed_status") != "completed":
        raise TableCFinalizationError(f"{role} queue is not recorded as completed")
    if record.get("status_policy") != status_policy:
        raise TableCFinalizationError(
            f"{role} status_policy must be {status_policy!r}"
        )
    expected = _completion_verification(count)
    if record.get("completion_verification") != expected:
        raise TableCFinalizationError(
            f"{role} completion verification must prove exactly {count} items"
        )


def _require_canonical_preflight_identity(
    path: Path, payload: Mapping[str, Any]
) -> None:
    if path != DEFAULT_PREFLIGHT_ATTESTATION.resolve(strict=False):
        return
    if payload.get("attestation_sha256") != CANONICAL_PREFLIGHT_ATTESTATION_SHA256:
        raise TableCFinalizationError(
            "canonical preflight semantic attestation SHA-256 differs"
        )
    observed_file_sha = audit._sha256_file(path)
    if observed_file_sha != CANONICAL_PREFLIGHT_FILE_SHA256:
        raise TableCFinalizationError("canonical preflight file SHA-256 differs")


def _load_verified_preflight(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    payload = _read_json_object(path, label="Table-C preflight attestation")
    _require_exact_keys(payload, _OLD_ATTESTATION_KEYS, label="preflight attestation")
    if payload.get("schema") != audit.SCHEMA:
        raise TableCFinalizationError(
            f"unsupported preflight schema: {payload.get('schema')!r}"
        )
    if payload.get("attestation_sha256") != audit._attestation_digest(payload):
        raise TableCFinalizationError("preflight semantic attestation SHA-256 differs")
    _require_canonical_preflight_identity(path, payload)
    result = audit.verify_attestation(
        path,
        policy="final",
        config_entries=_config_entries(payload),
    )
    if result.get("status") != "passed" or result.get("policy") != "final":
        raise TableCFinalizationError(
            "preflight did not pass the historical auditor under final policy"
        )

    queues = payload.get("queues")
    if not isinstance(queues, Mapping):
        raise TableCFinalizationError("preflight queue bindings are missing")
    _require_completed_binding(
        queues.get("completed_l0_l4"),
        role="completed_l0_l4",
        count=COMPLETION_COUNTS["completed_l0_l4"],
        status_policy="completed_required",
    )
    remaining = queues.get("remaining_table_c")
    if not isinstance(remaining, Mapping):
        raise TableCFinalizationError("preflight remaining queue binding is missing")
    if (
        remaining.get("observed_status") != "running"
        or remaining.get("status_policy") != "running_or_completed"
        or remaining.get("completion_verification") is not None
    ):
        raise TableCFinalizationError(
            "preflight remaining queue is not the original running-policy binding"
        )
    return payload


def _expected_remaining_binding(
    preflight: Mapping[str, Any], *, final_policy: bool
) -> dict[str, Any]:
    queues = preflight.get("queues")
    if not isinstance(queues, Mapping) or not isinstance(
        queues.get("remaining_table_c"), Mapping
    ):
        raise TableCFinalizationError("preflight remaining queue binding is missing")
    expected = copy.deepcopy(dict(queues["remaining_table_c"]))
    expected["observed_status"] = "completed"
    expected["completion_verification"] = _completion_verification(
        COMPLETION_COUNTS["remaining_table_c"]
    )
    expected["status_policy"] = (
        "completed_required" if final_policy else "running_or_completed"
    )
    return expected


def _assert_static_identity(
    candidate: Mapping[str, Any], preflight: Mapping[str, Any], *, label: str
) -> None:
    for key in _STATIC_IDENTITY_KEYS:
        if candidate.get(key) != preflight.get(key):
            raise TableCFinalizationError(
                f"{label} differs from preflight in static identity field {key}"
            )


def _assert_staged_replay(
    staged: Mapping[str, Any], preflight: Mapping[str, Any]
) -> None:
    _require_exact_keys(staged, _OLD_ATTESTATION_KEYS, label="staged attestation")
    if staged.get("attestation_sha256") != audit._attestation_digest(staged):
        raise TableCFinalizationError("staged old-schema attestation digest differs")
    _assert_static_identity(staged, preflight, label="staged attestation")

    staged_queues = staged.get("queues")
    preflight_queues = preflight.get("queues")
    if not isinstance(staged_queues, Mapping) or not isinstance(
        preflight_queues, Mapping
    ):
        raise TableCFinalizationError("staged or preflight queues are missing")
    if staged_queues.get("completed_l0_l4") != preflight_queues.get(
        "completed_l0_l4"
    ):
        raise TableCFinalizationError(
            "staged completed queue binding differs from preflight"
        )
    expected_remaining = _expected_remaining_binding(preflight, final_policy=False)
    if staged_queues.get("remaining_table_c") != expected_remaining:
        raise TableCFinalizationError(
            "staged remaining queue binding is not the exact completed replay"
        )
    _require_completed_binding(
        staged_queues.get("completed_l0_l4"),
        role="completed_l0_l4",
        count=COMPLETION_COUNTS["completed_l0_l4"],
        status_policy="completed_required",
    )
    _require_completed_binding(
        staged_queues.get("remaining_table_c"),
        role="remaining_table_c",
        count=COMPLETION_COUNTS["remaining_table_c"],
        status_policy="running_or_completed",
    )


def _finalization_record(
    *,
    preflight_path: Path,
    preflight: Mapping[str, Any],
    staged_attestation_sha256: str,
) -> dict[str, Any]:
    auditor_sources = preflight.get("auditor_sources")
    if not isinstance(auditor_sources, list):
        raise TableCFinalizationError("preflight auditor_sources are missing")
    return {
        "schema": FINALIZATION_SCHEMA,
        "policy": "final",
        "preflight": {
            "file_record": audit._file_record(preflight_path),
            "semantic_attestation_sha256": preflight["attestation_sha256"],
        },
        "staged_old_schema_attestation_sha256": staged_attestation_sha256,
        "finalizer_source": audit._file_record(Path(__file__).resolve(strict=True)),
        "auditor_preservation": {
            "historical_auditor_sources_unchanged": True,
            "record_count": len(auditor_sources),
            "canonical_sha256": audit._canonical_sha256(auditor_sources),
        },
        "completion_verification_counts": dict(COMPLETION_COUNTS),
        "transformation": {
            "only_queue_field_changed": (
                "queues.remaining_table_c.status_policy"
            ),
            "from": "running_or_completed",
            "to": "completed_required",
            "historical_auditor_sources_modified": False,
        },
    }


def _verify_file_record(
    raw: Any, expected_path: Path, *, label: str
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TableCFinalizationError(f"{label} file record is missing")
    expected_path = expected_path.expanduser().resolve(strict=True)
    observed = audit._file_record(expected_path)
    if dict(raw) != observed:
        raise TableCFinalizationError(f"{label} file record identity differs")
    return observed


def _verify_reconstructed_staged_digest(
    payload: Mapping[str, Any], finalization: Mapping[str, Any]
) -> None:
    staged_sha = finalization.get("staged_old_schema_attestation_sha256")
    if not isinstance(staged_sha, str) or len(staged_sha) != 64:
        raise TableCFinalizationError(
            "staged old-schema attestation SHA-256 is missing"
        )
    reconstructed = copy.deepcopy(dict(payload))
    reconstructed.pop("finalization", None)
    queues = reconstructed.get("queues")
    if not isinstance(queues, dict) or not isinstance(
        queues.get("remaining_table_c"), dict
    ):
        raise TableCFinalizationError(
            "cannot reconstruct staged remaining queue binding"
        )
    queues["remaining_table_c"]["status_policy"] = "running_or_completed"
    reconstructed["attestation_sha256"] = staged_sha
    _require_exact_keys(
        reconstructed,
        _OLD_ATTESTATION_KEYS,
        label="reconstructed staged attestation",
    )
    if audit._attestation_digest(reconstructed) != staged_sha:
        raise TableCFinalizationError(
            "final attestation is not the declared one-field staged upgrade"
        )


def verify_final_attestation(
    attestation_path: Path,
    *,
    preflight_path: Path = DEFAULT_PREFLIGHT_ATTESTATION,
) -> dict[str, Any]:
    """Replay the historical final policy and the added finalization lineage."""

    attestation_path = attestation_path.expanduser().resolve(strict=True)
    payload = _read_json_object(
        attestation_path, label="final Table-C dependency attestation"
    )
    _require_exact_keys(
        payload, _FINAL_ATTESTATION_KEYS, label="final Table-C attestation"
    )
    old_result = audit.verify_attestation(
        attestation_path,
        policy="final",
        config_entries=_config_entries(payload),
    )
    if old_result.get("status") != "passed" or old_result.get("policy") != "final":
        raise TableCFinalizationError(
            "final attestation did not pass the historical final-policy verifier"
        )

    raw_finalization = payload.get("finalization")
    if not isinstance(raw_finalization, Mapping):
        raise TableCFinalizationError("finalization record is missing")
    expected_finalization_keys = frozenset(
        {
            "schema",
            "policy",
            "preflight",
            "staged_old_schema_attestation_sha256",
            "finalizer_source",
            "auditor_preservation",
            "completion_verification_counts",
            "transformation",
        }
    )
    _require_exact_keys(
        raw_finalization,
        expected_finalization_keys,
        label="finalization record",
    )
    if (
        raw_finalization.get("schema") != FINALIZATION_SCHEMA
        or raw_finalization.get("policy") != "final"
    ):
        raise TableCFinalizationError("finalization schema or policy differs")

    raw_preflight = raw_finalization.get("preflight")
    if not isinstance(raw_preflight, Mapping) or set(raw_preflight) != {
        "file_record",
        "semantic_attestation_sha256",
    }:
        raise TableCFinalizationError("finalization preflight lineage is incomplete")
    preflight_record = raw_preflight.get("file_record")
    if not isinstance(preflight_record, Mapping):
        raise TableCFinalizationError("finalization preflight file record is missing")
    recorded_preflight_path = Path(str(preflight_record.get("path", ""))).resolve(
        strict=False
    )
    required_preflight_path = preflight_path.expanduser().resolve(strict=True)
    if recorded_preflight_path != required_preflight_path:
        raise TableCFinalizationError(
            "finalization references a different preflight attestation"
        )
    _verify_file_record(
        preflight_record,
        required_preflight_path,
        label="preflight attestation",
    )
    preflight = _load_verified_preflight(required_preflight_path)
    if (
        raw_preflight.get("semantic_attestation_sha256")
        != preflight.get("attestation_sha256")
    ):
        raise TableCFinalizationError(
            "preflight semantic attestation SHA-256 lineage differs"
        )

    _verify_file_record(
        raw_finalization.get("finalizer_source"),
        Path(__file__),
        label="finalizer source",
    )
    _assert_static_identity(payload, preflight, label="final attestation")

    expected_auditor_preservation = {
        "historical_auditor_sources_unchanged": True,
        "record_count": len(preflight["auditor_sources"]),
        "canonical_sha256": audit._canonical_sha256(
            preflight["auditor_sources"]
        ),
    }
    if raw_finalization.get("auditor_preservation") != expected_auditor_preservation:
        raise TableCFinalizationError(
            "historical auditor-source preservation lineage differs"
        )
    if payload.get("auditor_sources") != preflight.get("auditor_sources"):
        raise TableCFinalizationError(
            "historical auditor_sources were not preserved exactly"
        )
    if raw_finalization.get("completion_verification_counts") != COMPLETION_COUNTS:
        raise TableCFinalizationError("finalization completion counts differ")
    if raw_finalization.get("transformation") != {
        "only_queue_field_changed": "queues.remaining_table_c.status_policy",
        "from": "running_or_completed",
        "to": "completed_required",
        "historical_auditor_sources_modified": False,
    }:
        raise TableCFinalizationError("finalization transformation declaration differs")

    queues = payload.get("queues")
    preflight_queues = preflight.get("queues")
    if not isinstance(queues, Mapping) or not isinstance(preflight_queues, Mapping):
        raise TableCFinalizationError("final or preflight queue bindings are missing")
    if queues.get("completed_l0_l4") != preflight_queues.get("completed_l0_l4"):
        raise TableCFinalizationError(
            "final completed queue binding differs from preflight"
        )
    expected_remaining = _expected_remaining_binding(preflight, final_policy=True)
    if queues.get("remaining_table_c") != expected_remaining:
        raise TableCFinalizationError(
            "final remaining queue binding is not the exact completed-policy upgrade"
        )
    _require_completed_binding(
        queues.get("completed_l0_l4"),
        role="completed_l0_l4",
        count=COMPLETION_COUNTS["completed_l0_l4"],
        status_policy="completed_required",
    )
    _require_completed_binding(
        queues.get("remaining_table_c"),
        role="remaining_table_c",
        count=COMPLETION_COUNTS["remaining_table_c"],
        status_policy="completed_required",
    )
    _verify_reconstructed_staged_digest(payload, raw_finalization)

    return {
        "schema": FINAL_VERIFICATION_SCHEMA,
        "status": "passed",
        "policy": "final",
        "verified_at_utc": old_result.get("verified_at_utc"),
        "attestation": str(attestation_path),
        "attestation_sha256": payload["attestation_sha256"],
        "preflight_attestation": str(required_preflight_path),
        "preflight_attestation_sha256": preflight["attestation_sha256"],
        "canonical_closure_sha256": old_result.get(
            "canonical_closure_sha256"
        ),
        "closure_path_count": old_result.get("closure_path_count"),
        "completion_verification_counts": dict(COMPLETION_COUNTS),
        "historical_auditor_sources_preserved": True,
        "staged_upgrade_replayed": True,
    }


def _rename_noreplace(source: Path, destination: Path) -> None:
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:  # pragma: no cover - required Linux runtime
        raise TableCFinalizationError(
            "atomic final-attestation publication requires Linux renameat2"
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
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            f"final attestation appeared concurrently: {destination}"
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _temporary_path(output_path: Path, *, role: str) -> Path:
    return output_path.with_name(
        f".{output_path.name}.{role}-{os.getpid()}-{uuid.uuid4().hex}.json"
    )


def finalize_attestation(
    output_path: Path = DEFAULT_OUTPUT,
    *,
    preflight_path: Path = DEFAULT_PREFLIGHT_ATTESTATION,
) -> dict[str, Any]:
    """Create, replay, and atomically publish one fresh final attestation."""

    preflight_path = preflight_path.expanduser().resolve(strict=True)
    output_path = output_path.expanduser().resolve(strict=False)
    if output_path.exists():
        raise FileExistsError(f"final attestation already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staged_path = _temporary_path(output_path, role="old-schema-stage")
    candidate_path = _temporary_path(output_path, role="final-candidate")
    try:
        preflight = _load_verified_preflight(preflight_path)
        queues = preflight["queues"]
        training = preflight["training_evidence"]
        staged = audit.create_attestation(
            staged_path,
            repository_root=Path(str(preflight["repository_root"])),
            completed_queue_dir=Path(str(queues["completed_l0_l4"]["queue_dir"])),
            remaining_queue_dir=Path(str(queues["remaining_table_c"]["queue_dir"])),
            training_root=Path(str(training["training_root"])),
            config_entries=_config_entries(preflight),
        )
        _assert_staged_replay(staged, preflight)

        candidate = copy.deepcopy(staged)
        remaining = candidate["queues"]["remaining_table_c"]
        if remaining.get("status_policy") != "running_or_completed":
            raise TableCFinalizationError(
                "staged remaining queue policy is not eligible for final upgrade"
            )
        remaining["status_policy"] = "completed_required"
        candidate["finalization"] = _finalization_record(
            preflight_path=preflight_path,
            preflight=preflight,
            staged_attestation_sha256=str(staged["attestation_sha256"]),
        )
        candidate["attestation_sha256"] = audit._attestation_digest(candidate)
        audit._write_json_exclusive(candidate_path, candidate)
        verification = verify_final_attestation(
            candidate_path,
            preflight_path=preflight_path,
        )

        _rename_noreplace(candidate_path, output_path)
        directory_fd = os.open(output_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        verification["attestation"] = str(output_path)
        verification["publication_status"] = "published"
        return verification
    finally:
        staged_path.unlink(missing_ok=True)
        candidate_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    finalize = subparsers.add_parser(
        "finalize", help="create and atomically publish the fresh final attestation"
    )
    finalize.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    finalize.add_argument(
        "--preflight", type=Path, default=DEFAULT_PREFLIGHT_ATTESTATION
    )

    verify = subparsers.add_parser(
        "verify", help="replay old-policy and finalization-lineage verification"
    )
    verify.add_argument("attestation", type=Path)
    verify.add_argument(
        "--preflight", type=Path, default=DEFAULT_PREFLIGHT_ATTESTATION
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "finalize":
            result = finalize_attestation(
                args.output,
                preflight_path=args.preflight,
            )
        else:
            result = verify_final_attestation(
                args.attestation,
                preflight_path=args.preflight,
            )
    except (
        DependencyAuditError,
        OSError,
        TableCFinalizationError,
        audit.TableCDependencyClosureError,
        TypeError,
        ValueError,
    ) as exc:
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
