#!/usr/bin/env python3
"""Create, supervise, and replay the exact formal Table-D training matrix."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shlex
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_stageb_paper_ablation_matrices as paper  # noqa: E402
from tools import run_stageb_paper_evaluations as evaluator  # noqa: E402
from tools import run_stageb_serial_matrix_queue as serial_queue  # noqa: E402
from tools import seal_stageb_memory_probe as probe_sealer  # noqa: E402
from tools import stageb_profile_dependency_audit as dependency_audit  # noqa: E402
from tools import aggregate_stageb_table_d_diagnostics as diagnostics  # noqa: E402


PROFILE = "table_d_formal_b40_u1000_i1000_w2_d100"
ROWS = ("S0", "S1", "S2", "S3", "S2F")
SEEDS = (17, 42, 73)
RUN_IDS = tuple(f"{row}:{seed}" for row in ROWS for seed in SEEDS)
FORMAL_TRAINING_WRAPPER = (
    REPO_ROOT / "tools/run_stageb_table_d_formal_training.py"
)
SOURCE_PLAN_NAME = "formal_source_plan.json"
SCOPE_PLAN_NAME = "formal_scope_plan.json"
COMPLETION_NAME = "formal_completion_attestation.json"

SOURCE_PLAN_SCHEMA = "pivot.stageb.table_d_formal_source_plan/v1"
SCOPE_PLAN_SCHEMA = "pivot.stageb.table_d_formal_scope_plan/v1"
COMPLETION_SCHEMA = "pivot.stageb.table_d_formal_completion/v1"
EXTENSION_SCHEMA = "pivot.stageb.table_d_generic_queue_extension/v1"
SOURCE_ROLE = "table_d_formal_source_plan"
SCOPE_ROLE = "table_d_formal_scope_plan"

FORMAL_TRAINING_CONTRACT = {
    "batch_size": 40,
    "optimizer_updates": 1_000,
    "iter_checkpoint_interval": 1_000,
    "num_workers": 2,
    "gradient_diagnostic_interval": 100,
    "successful_update_batch_slots_per_run": 40_000,
    "successful_update_batch_slots_total": 600_000,
    "s3": {
        "isolation_probe_updates_excluded": 1,
        "rank_updates": 500,
        "confidence_updates": 500,
    },
}
MINIMUM_HEADROOM_MIB = 1_024.0
ITEM_STATUSES = frozenset(
    {"pending", "reserved", "launching", "launched", "completed", "failed"}
)

_SEQUENCE_KEYS = (
    "schema",
    "repository_root",
    "run_id",
    "row",
    "seed",
    "training_seeds_contract",
    "output_dir",
    "equal_budget_contract",
)
_PHASE_KEYS = (
    "schema",
    "run_id",
    "row",
    "seed",
    "phase",
    "output_dir",
    "command",
    "runtime",
    "fixed_contract",
    "generated_dependency",
    "inputs",
)


class TableDFormalQueueError(RuntimeError):
    """The formal Table-D plan, execution, or replay contract drifted."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise TableDFormalQueueError(f"value is not canonical JSON: {exc}") from exc


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _semantic_sha(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("semantic_sha256", None)
    return _canonical_sha(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, roles: Iterable[str] = ()) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    stat = path.stat()
    record: dict[str, Any] = {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    rendered = sorted(set(roles))
    if rendered:
        record["roles"] = rendered
    return record


def _verify_file_record(record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise TableDFormalQueueError(f"{label} file record is missing")
    path = Path(str(record.get("path", ""))).expanduser().resolve(strict=True)
    expected = _file_record(path, roles=record.get("roles", ()))
    if expected != dict(record):
        raise TableDFormalQueueError(f"{label} file identity changed: {path}")
    return path


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TableDFormalQueueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TableDFormalQueueError(f"{label} must be a JSON object")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    rendered = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"
    try:
        with temporary.open("x", encoding="ascii") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


@contextlib.contextmanager
def _environment(values: Mapping[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _runtime_environment(*, output_root: Path, runner_python: Path) -> dict[str, str]:
    return {
        "PIVOT_PYTHON": str(runner_python),
        "PIVOT_SCORE_OUTPUT_ROOT": str(output_root),
        "PIVOT_BATCH_SIZE": "40",
        "PIVOT_MAX_TRAIN_ITERS": "1000",
        "PIVOT_ITER_CHECKPOINT_INTERVAL": "1000",
        "PIVOT_NUM_WORKERS": "2",
        "PIVOT_GRADIENT_DIAGNOSTIC_INTERVAL": "100",
    }


def _validate_runner_python(runner_python: Path) -> Path:
    runner_python = runner_python.expanduser().resolve(strict=True)
    expected = paper.DEFAULT_PYTHON.expanduser().resolve(strict=True)
    if runner_python != expected:
        raise TableDFormalQueueError(
            f"formal Table-D training requires the sealed GDINO Python: {expected}"
        )
    if not runner_python.is_file() or not os.access(runner_python, os.X_OK):
        raise TableDFormalQueueError(
            f"runner Python is not executable: {runner_python}"
        )
    return runner_python


def _validate_runtime(
    runtime: paper.Runtime, *, output_root: Path, runner_python: Path
) -> None:
    runner_python = _validate_runner_python(runner_python)
    expected = {
        "python": runner_python.resolve(strict=True),
        "score_output_root": output_root.resolve(strict=False),
        "batch_size": 40,
        "total_train_iters": 1_000,
        "iter_checkpoint_interval": 1_000,
        "num_workers": 2,
        "gradient_diagnostic_interval": 100,
    }
    observed = {key: getattr(runtime, key) for key in expected}
    if observed != expected:
        raise TableDFormalQueueError(
            f"formal Table-D runtime mismatch: expected {expected}, got {observed}"
        )


def _normalize_input_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TableDFormalQueueError(f"input record {index} is not an object")
        path = Path(str(record.get("path", ""))).expanduser().resolve(strict=True)
        roles: set[str] = set()
        role = record.get("role")
        if isinstance(role, str) and role:
            roles.add(role)
        raw_roles = record.get("roles")
        if isinstance(raw_roles, list):
            roles.update(str(value) for value in raw_roles if str(value))
        identity = {
            "path": str(path),
            "sha256": str(record.get("sha256", "")),
            "size_bytes": int(record.get("size_bytes", -1)),
            "mtime_ns": int(record.get("mtime_ns", -1)),
        }
        previous = result.get(str(path))
        if previous is not None:
            if {key: previous[key] for key in identity} != identity:
                raise TableDFormalQueueError(f"input identities conflict for {path}")
            previous["roles"] = sorted(set(previous["roles"]) | roles)
        else:
            result[str(path)] = {**identity, "roles": sorted(roles)}
    return result


def _input_identity_sha(
    identity: Mapping[str, Mapping[str, Any]],
    *,
    excluded_roles: frozenset[str] = frozenset(),
) -> str:
    values = [
        dict(identity[path])
        for path in sorted(identity)
        if not excluded_roles.intersection(identity[path].get("roles", ()))
    ]
    return _canonical_sha(values)


def _immutable_phase(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: manifest.get(key) for key in _PHASE_KEYS}


def _immutable_sequence(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **{key: manifest.get(key) for key in _SEQUENCE_KEYS},
        "phases": [_immutable_phase(phase) for phase in manifest.get("phases", [])],
    }


def _phase_scope_record(phase: Mapping[str, Any]) -> dict[str, Any]:
    phase_contract = phase.get("phase")
    if not isinstance(phase_contract, Mapping):
        raise TableDFormalQueueError("planned phase contract is missing")
    inputs = phase.get("inputs")
    records = inputs.get("records") if isinstance(inputs, Mapping) else None
    if not isinstance(records, list) or not records:
        raise TableDFormalQueueError("planned phase inputs are missing")
    identity = _normalize_input_records(records)
    command = phase.get("command")
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise TableDFormalQueueError("planned phase command is invalid")
    return {
        "phase_id": str(phase_contract.get("phase_id", "")),
        "input_identity_sha256": _input_identity_sha(identity),
        "command_sha256": _canonical_sha(command),
        "immutable_manifest_sha256": _canonical_sha(_immutable_phase(phase)),
        "generated_dependency": phase.get("generated_dependency"),
    }


def _probe_roots(payload: Mapping[str, Any]) -> dict[str, Path]:
    probes = payload.get("probes")
    if not isinstance(probes, Mapping) or not probes:
        raise TableDFormalQueueError("readiness seal has no probes")
    roots: dict[str, Path] = {}
    for name, value in probes.items():
        if not isinstance(name, str) or not isinstance(value, Mapping):
            raise TableDFormalQueueError("readiness seal probe inventory is invalid")
        roots[name] = Path(str(value.get("root", ""))).expanduser().resolve(strict=True)
    return roots


def verify_s2_soak_seal(path: Path) -> dict[str, Any]:
    row_id = "S2"
    optimizer_updates = 50
    path = path.expanduser().resolve(strict=True)
    payload = _read_json(path, label=f"{row_id} readiness seal")
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise TableDFormalQueueError(f"{row_id} readiness selection is missing")
    selected = selection.get("probe")
    if not isinstance(selected, str) or not selected:
        raise TableDFormalQueueError(f"{row_id} readiness selected probe is invalid")
    roots = _probe_roots(payload)
    try:
        replay = probe_sealer.verify_existing_seal(
            path,
            roots,
            selected=selected,
            minimum_headroom_mib=MINIMUM_HEADROOM_MIB,
            minimum_soak_updates=optimizer_updates,
            expected_row_id=row_id,
        )
    except (OSError, ValueError, probe_sealer.MemoryProbeError) as exc:
        raise TableDFormalQueueError(
            f"{row_id} final-code readiness replay failed: {exc}"
        ) from exc
    selected_probe = replay["probes"][selected]
    expected_diagnostic = 10
    root = Path(str(selected_probe["root"])).resolve(strict=True)
    launch = _read_json(root / "launch_manifest.json", label=f"{row_id} probe launch")
    phase = launch.get("phase")
    runtime = launch.get("runtime")
    if not (
        selected_probe.get("batch_size") == 40
        and selected_probe.get("optimizer_updates") == optimizer_updates
        and selection.get("batch_size") == 40
        and selection.get("minimum_soak_updates") == optimizer_updates
        and selection.get("minimum_headroom_mib") == MINIMUM_HEADROOM_MIB
        and isinstance(phase, Mapping)
        and phase.get("diagnostic_interval") == expected_diagnostic
        and isinstance(runtime, Mapping)
        and runtime.get("num_workers") == 2
    ):
        raise TableDFormalQueueError(
            f"{row_id} readiness must be exact B40/U{optimizer_updates}/W2/D{expected_diagnostic}"
        )
    return {
        "row_id": row_id,
        "batch_size": 40,
        "optimizer_updates": optimizer_updates,
        "num_workers": 2,
        "diagnostic_interval": expected_diagnostic,
        "selected_probe": selected,
        "selected_root": str(root),
        "gpu_identity": dict(replay["gpu_identity"]),
        "seal": _file_record(path, roles=(f"{row_id}_readiness_seal",)),
        "seal_semantic_sha256": _canonical_sha(payload),
    }


def verify_s2f_confirmation(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    try:
        probe_sealer._verify_paper_probe_current(
            root,
            expected_row_id="S2F",
            minimum_soak_updates=50,
        )
        observed = probe_sealer.inspect_probe(root, expected_row_id="S2F")
    except (OSError, ValueError, probe_sealer.MemoryProbeError) as exc:
        raise TableDFormalQueueError(
            f"S2F final-code confirmation replay failed: {exc}"
        ) from exc
    launch = _read_json(root / "launch_manifest.json", label="S2F confirmation launch")
    phase = launch.get("phase")
    runtime = launch.get("runtime")
    gpu = observed.get("gpu")
    try:
        minimum_free_memory_mib = float(
            gpu.get("min_free_memory_mib") if isinstance(gpu, Mapping) else None
        )
    except (TypeError, ValueError) as exc:
        raise TableDFormalQueueError(
            "S2F confirmation GPU headroom is missing"
        ) from exc
    if not (
        observed.get("batch_size") == 40
        and observed.get("optimizer_updates") == 2
        and isinstance(phase, Mapping)
        and phase.get("diagnostic_interval") == 1
        and isinstance(runtime, Mapping)
        and runtime.get("num_workers") == 2
        and runtime.get("iter_checkpoint_interval") == 2
        and math.isfinite(minimum_free_memory_mib)
        and minimum_free_memory_mib >= MINIMUM_HEADROOM_MIB
    ):
        raise TableDFormalQueueError(
            "S2F confirmation must be exact B40/U2/I2/W2/D1 with at least "
            f"{MINIMUM_HEADROOM_MIB} MiB headroom"
        )
    artifacts = observed.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise TableDFormalQueueError("S2F confirmation artifact inventory is missing")
    records = []
    for name in ("sequence_manifest", "postflight", "gpu_telemetry_summary"):
        record = artifacts.get(name)
        if not isinstance(record, Mapping):
            raise TableDFormalQueueError(f"S2F confirmation lacks {name}")
        records.append({**dict(record), "roles": [f"S2F_confirmation_{name}"]})
    return {
        "row_id": "S2F",
        "batch_size": 40,
        "optimizer_updates": 2,
        "iter_checkpoint_interval": 2,
        "num_workers": 2,
        "diagnostic_interval": 1,
        "minimum_free_memory_mib": minimum_free_memory_mib,
        "minimum_headroom_mib": MINIMUM_HEADROOM_MIB,
        "root": str(root),
        "gpu_identity": {
            key: observed["gpu"][key]
            for key in ("uuid", "name", "driver_version", "total_memory_mib")
        },
        "artifacts": records,
        "semantic_sha256": _canonical_sha(observed),
    }


def _validate_readiness_pair(readiness: Mapping[str, Any]) -> None:
    s2 = readiness.get("s2_b40_u50_soak")
    s2f = readiness.get("s2f_b40_u2_confirmation")
    if not (
        isinstance(s2, Mapping)
        and isinstance(s2f, Mapping)
        and s2.get("gpu_identity") == s2f.get("gpu_identity")
        and isinstance(s2.get("gpu_identity"), Mapping)
        and bool(s2["gpu_identity"].get("uuid"))
    ):
        raise TableDFormalQueueError(
            "S2 soak and S2F confirmation must share one exact GPU identity"
        )


def _planned_manifests(
    runtime: paper.Runtime,
) -> dict[str, dict[str, Any]]:
    cache = paper.token_launcher.HashCache()
    result: dict[str, dict[str, Any]] = {}
    for run_id in RUN_IDS:
        row_id, raw_seed = run_id.split(":", 1)
        result[run_id] = paper.build_manifest(
            runtime, paper.ROW_BY_ID[row_id], int(raw_seed), cache
        )
    return result


def _source_closure(
    manifests: Mapping[str, Mapping[str, Any]], *, runner_python: Path
) -> dict[str, Any]:
    paths: dict[Path, set[str]] = {}
    for manifest in manifests.values():
        for phase in manifest["phases"]:
            for record in phase["inputs"]["records"]:
                path = Path(str(record["path"])).resolve(strict=True)
                roles = paths.setdefault(path, set())
                role = record.get("role")
                if isinstance(role, str):
                    roles.add(role)
    entrypoints = (Path(__file__).resolve(), FORMAL_TRAINING_WRAPPER.resolve(strict=True))
    try:
        recursive = dependency_audit.recursive_local_python_dependencies(
            tuple(path.relative_to(REPO_ROOT).as_posix() for path in entrypoints),
            repository_root=REPO_ROOT,
        )
    except dependency_audit.ProfileDependencyAuditError as exc:
        raise TableDFormalQueueError(f"formal controller closure failed: {exc}") from exc
    for path in recursive:
        paths.setdefault(path.resolve(strict=True), set()).add("formal_controller_source")
    for path, role in (
        (runner_python, "runner_python"),
        (Path(paper.__file__), "frozen_paper_launcher"),
        (Path(serial_queue.__file__), "shared_durable_gpu_queue"),
        (Path(probe_sealer.__file__), "readiness_seal_verifier"),
        (Path(evaluator.__file__), "completion_replay_verifier"),
        (Path(diagnostics.__file__), "s3_atomic_lineage_verifier"),
    ):
        paths.setdefault(Path(path).resolve(strict=True), set()).add(role)
    records = [
        _file_record(path, roles=roles)
        for path, roles in sorted(paths.items(), key=lambda item: str(item[0]))
    ]
    return {
        "status": "sealed",
        "records": records,
        "semantic_sha256": _canonical_sha(records),
    }


def _common_input_contract(
    manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    normalized: list[tuple[str, str, dict[str, dict[str, Any]]]] = []
    for run_id in RUN_IDS:
        for phase in manifests[run_id]["phases"]:
            phase_id = phase["phase"]["phase_id"]
            identity = _normalize_input_records(phase["inputs"]["records"])
            common = {
                path: record
                for path, record in identity.items()
                if "config_dependency" not in record["roles"]
                and "rank_phase_model_state_pretrain" not in record["roles"]
            }
            normalized.append((run_id, phase_id, common))
    reference = normalized[0][2]
    drifted = [
        f"{run_id}/{phase_id}"
        for run_id, phase_id, identity in normalized
        if identity != reference
    ]
    if drifted:
        raise TableDFormalQueueError(
            "Table-D phases differ outside their fully bound config dependencies: "
            + ", ".join(drifted)
        )
    return {
        "status": "passed",
        "all_phases_share_exact_nonconfig_inputs": True,
        "config_dependencies_are_phase_specific_and_fully_bound": True,
        "phase_count": len(normalized),
        "common_input_identity_sha256": _input_identity_sha(reference),
    }


def _scope_runs(
    manifests: Mapping[str, Mapping[str, Any]],
    *,
    queue_id: str,
    queue_plan_sha256: str,
    source_plan_semantic_sha256: str,
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for run_id in RUN_IDS:
        row_id, raw_seed = run_id.split(":", 1)
        manifest = manifests[run_id]
        phases = [_phase_scope_record(phase) for phase in manifest["phases"]]
        scope = {
            "profile": PROFILE,
            "run_id": run_id,
            "row_id": row_id,
            "seed": int(raw_seed),
            "queue_id": queue_id,
            "queue_plan_sha256": queue_plan_sha256,
            "source_plan_semantic_sha256": source_plan_semantic_sha256,
            "output_root": manifest["output_dir"],
            "training_contract": FORMAL_TRAINING_CONTRACT,
            "sequence_contract_sha256": _canonical_sha(
                _immutable_sequence(manifest)
            ),
            "phases": phases,
        }
        records[run_id] = {
            **scope,
            "scope_sha256": _canonical_sha(scope),
        }
    return records


def create_training_queue(
    queue_dir: Path,
    *,
    output_root: Path,
    s2_soak_seal: Path,
    s2f_confirmation_root: Path,
    runner_python: Path = paper.DEFAULT_PYTHON,
    token_runner: Path = serial_queue.DEFAULT_TOKEN_RUNNER,
    lease_root: Path = serial_queue.DEFAULT_LEASE_ROOT,
    gpu_key: str | None = None,
) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=False)
    output_root = output_root.expanduser().resolve(strict=False)
    runner_python = _validate_runner_python(runner_python)
    if queue_dir.exists():
        raise FileExistsError(f"formal training queue must be fresh: {queue_dir}")
    if output_root.exists():
        raise FileExistsError(f"formal Table-D output root must be fresh: {output_root}")
    readiness = {
        "s2_b40_u50_soak": verify_s2_soak_seal(s2_soak_seal),
        "s2f_b40_u2_confirmation": verify_s2f_confirmation(
            s2f_confirmation_root
        ),
    }
    _validate_readiness_pair(readiness)
    environment = _runtime_environment(
        output_root=output_root, runner_python=runner_python
    )
    with _environment(environment):
        runtime = paper.runtime_from_environment()
        _validate_runtime(
            runtime, output_root=output_root, runner_python=runner_python
        )
        manifests = _planned_manifests(runtime)
    common = _common_input_contract(manifests)
    closure = _source_closure(manifests, runner_python=runner_python)
    controller = _file_record(Path(__file__))
    wrapper = _file_record(FORMAL_TRAINING_WRAPPER)
    frozen_launcher = _file_record(Path(paper.__file__))
    readiness_identity = _canonical_sha(readiness)
    extension: dict[str, Any] = {
        "schema": EXTENSION_SCHEMA,
        "profile": PROFILE,
        "ordered_run_ids": list(RUN_IDS),
        "formal_training_contract": FORMAL_TRAINING_CONTRACT,
        "explicit_output_root": str(output_root),
        "source_closure_semantic_sha256": closure["semantic_sha256"],
        "common_input_identity_sha256": common[
            "common_input_identity_sha256"
        ],
        "readiness_identity_sha256": readiness_identity,
        "dedicated_controller": controller,
        "formal_training_wrapper": wrapper,
        "frozen_paper_launcher": frozen_launcher,
    }
    extension["semantic_sha256"] = _semantic_sha(extension)
    with _environment(environment):
        queue = serial_queue.create_queue(
            queue_dir,
            run_ids=RUN_IDS,
            runner_python=runner_python,
            token_runner=token_runner,
            paper_runner=FORMAL_TRAINING_WRAPPER,
            lease_root=lease_root,
            gpu_key=gpu_key,
            plan_extensions=extension,
        )
    plan = queue["plan"]
    source_plan: dict[str, Any] = {
        "schema": SOURCE_PLAN_SCHEMA,
        "status": "sealed",
        "profile": PROFILE,
        "ordered_run_ids": list(RUN_IDS),
        "formal_training_contract": FORMAL_TRAINING_CONTRACT,
        "runtime": {
            "python": str(runner_python),
            "output_root": str(output_root),
            "environment": environment,
        },
        "queue": {
            "queue_dir": str(queue_dir),
            "queue_id": plan["queue_id"],
            "plan_sha256": queue["plan_sha256"],
            "gpu_key": plan["gpu_key"],
            "lease_path": plan["lease_path"],
            "extension_semantic_sha256": extension["semantic_sha256"],
        },
        "readiness": readiness,
        "readiness_identity_sha256": readiness_identity,
        "source_closure": closure,
        "common_input_contract": common,
    }
    source_plan["semantic_sha256"] = _semantic_sha(source_plan)
    source_path = queue_dir / SOURCE_PLAN_NAME
    _write_json_atomic(source_path, source_plan)
    runs = _scope_runs(
        manifests,
        queue_id=plan["queue_id"],
        queue_plan_sha256=queue["plan_sha256"],
        source_plan_semantic_sha256=source_plan["semantic_sha256"],
    )
    scope_plan: dict[str, Any] = {
        "schema": SCOPE_PLAN_SCHEMA,
        "status": "sealed",
        "profile": PROFILE,
        "ordered_run_ids": list(RUN_IDS),
        "source_plan": _file_record(source_path, roles=(SOURCE_ROLE,)),
        "source_plan_semantic_sha256": source_plan["semantic_sha256"],
        "queue": dict(source_plan["queue"]),
        "runs": runs,
    }
    scope_plan["semantic_sha256"] = _semantic_sha(scope_plan)
    scope_path = queue_dir / SCOPE_PLAN_NAME
    _write_json_atomic(scope_path, scope_plan)
    _load_plans(queue_dir)
    return {
        "status": "planned",
        "profile": PROFILE,
        "queue_dir": str(queue_dir),
        "queue_id": plan["queue_id"],
        "queue_plan_sha256": queue["plan_sha256"],
        "ordered_run_ids": list(RUN_IDS),
        "source_plan": _file_record(source_path, roles=(SOURCE_ROLE,)),
        "scope_plan": _file_record(scope_path, roles=(SCOPE_ROLE,)),
        "readiness": readiness,
    }


def _expected_extension(source: Mapping[str, Any]) -> dict[str, Any]:
    extension: dict[str, Any] = {
        "schema": EXTENSION_SCHEMA,
        "profile": PROFILE,
        "ordered_run_ids": list(RUN_IDS),
        "formal_training_contract": FORMAL_TRAINING_CONTRACT,
        "explicit_output_root": source["runtime"]["output_root"],
        "source_closure_semantic_sha256": source["source_closure"][
            "semantic_sha256"
        ],
        "common_input_identity_sha256": source["common_input_contract"][
            "common_input_identity_sha256"
        ],
        "readiness_identity_sha256": source["readiness_identity_sha256"],
        "dedicated_controller": _file_record(Path(__file__)),
        "formal_training_wrapper": _file_record(FORMAL_TRAINING_WRAPPER),
        "frozen_paper_launcher": _file_record(Path(paper.__file__)),
    }
    extension["semantic_sha256"] = _semantic_sha(extension)
    return extension


def _validate_source_plan(
    queue_dir: Path, source: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    if not (
        source.get("schema") == SOURCE_PLAN_SCHEMA
        and source.get("status") == "sealed"
        and source.get("profile") == PROFILE
        and source.get("ordered_run_ids") == list(RUN_IDS)
        and source.get("formal_training_contract") == FORMAL_TRAINING_CONTRACT
        and source.get("semantic_sha256") == _semantic_sha(source)
    ):
        raise TableDFormalQueueError("formal source plan identity drifted")
    runtime = source.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("environment") != _runtime_environment(
        output_root=Path(str(runtime.get("output_root", ""))),
        runner_python=Path(str(runtime.get("python", ""))),
    ):
        raise TableDFormalQueueError("formal source runtime environment drifted")
    readiness = source.get("readiness")
    if not isinstance(readiness, Mapping) or set(readiness) != {
        "s2_b40_u50_soak",
        "s2f_b40_u2_confirmation",
    }:
        raise TableDFormalQueueError("formal readiness inventory drifted")
    s2_seal = _verify_file_record(
        readiness["s2_b40_u50_soak"].get("seal"), label="S2 readiness seal"
    )
    replayed_readiness = {
        "s2_b40_u50_soak": verify_s2_soak_seal(s2_seal),
        "s2f_b40_u2_confirmation": verify_s2f_confirmation(
            Path(readiness["s2f_b40_u2_confirmation"]["root"])
        ),
    }
    _validate_readiness_pair(replayed_readiness)
    if (
        replayed_readiness != readiness
        or source.get("readiness_identity_sha256")
        != _canonical_sha(replayed_readiness)
    ):
        raise TableDFormalQueueError("formal readiness replay drifted")
    closure = source.get("source_closure")
    records = closure.get("records") if isinstance(closure, Mapping) else None
    if not (
        isinstance(records, list)
        and records
        and closure.get("status") == "sealed"
        and closure.get("semantic_sha256") == _canonical_sha(records)
    ):
        raise TableDFormalQueueError("formal source closure is incomplete")
    for index, record in enumerate(records):
        _verify_file_record(record, label=f"source closure record {index}")
    queue = serial_queue.load_queue(queue_dir)
    plan = queue.get("plan")
    queue_binding = source.get("queue")
    if not (
        isinstance(plan, Mapping)
        and isinstance(queue_binding, Mapping)
        and plan.get("queue_id") == queue_binding.get("queue_id")
        and queue.get("plan_sha256") == queue_binding.get("plan_sha256")
        and Path(str(plan.get("queue_dir", ""))).resolve(strict=False) == queue_dir
        and [item.get("run_id") for item in plan.get("items", [])]
        == list(RUN_IDS)
        and all(item.get("runner") == "paper" for item in plan.get("items", []))
        and plan.get("runtime_environment") == runtime["environment"]
        and plan.get("extensions") == _expected_extension(source)
        and queue_binding.get("extension_semantic_sha256")
        == plan["extensions"]["semantic_sha256"]
    ):
        raise TableDFormalQueueError("generic training queue/source plan drifted")
    runner = plan.get("runners", {}).get("paper")
    if not (
        isinstance(runner, Mapping)
        and Path(str(runner.get("path", ""))).resolve(strict=True)
        == FORMAL_TRAINING_WRAPPER.resolve(strict=True)
        and runner.get("sha256") == _sha256_file(FORMAL_TRAINING_WRAPPER)
    ):
        raise TableDFormalQueueError("generic queue does not bind the formal wrapper")
    return queue


def _validate_scope_plan(
    queue_dir: Path,
    source: Mapping[str, Any],
    scope: Mapping[str, Any],
) -> None:
    if not (
        scope.get("schema") == SCOPE_PLAN_SCHEMA
        and scope.get("status") == "sealed"
        and scope.get("profile") == PROFILE
        and scope.get("ordered_run_ids") == list(RUN_IDS)
        and scope.get("semantic_sha256") == _semantic_sha(scope)
        and scope.get("source_plan_semantic_sha256") == source["semantic_sha256"]
        and scope.get("queue") == source["queue"]
    ):
        raise TableDFormalQueueError("formal scope plan identity drifted")
    source_path = _verify_file_record(
        scope.get("source_plan"), label="formal source plan"
    )
    if source_path != (queue_dir / SOURCE_PLAN_NAME).resolve(strict=True):
        raise TableDFormalQueueError("scope plan source path drifted")
    environment = source["runtime"]["environment"]
    with _environment(environment):
        runtime = paper.runtime_from_environment()
        _validate_runtime(
            runtime,
            output_root=Path(source["runtime"]["output_root"]),
            runner_python=Path(source["runtime"]["python"]),
        )
        manifests = _planned_manifests(runtime)
    expected = _scope_runs(
        manifests,
        queue_id=source["queue"]["queue_id"],
        queue_plan_sha256=source["queue"]["plan_sha256"],
        source_plan_semantic_sha256=source["semantic_sha256"],
    )
    if scope.get("runs") != expected:
        raise TableDFormalQueueError("formal scope/input/command plan drifted")
    if _common_input_contract(manifests) != source["common_input_contract"]:
        raise TableDFormalQueueError("formal common-input plan drifted")


def _load_plans(
    queue_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], MutableMapping[str, Any]]:
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    source = _read_json(queue_dir / SOURCE_PLAN_NAME, label="formal source plan")
    scope = _read_json(queue_dir / SCOPE_PLAN_NAME, label="formal scope plan")
    queue = _validate_source_plan(queue_dir, source)
    _validate_scope_plan(queue_dir, source, scope)
    return source, scope, queue


def _active_index(queue: Mapping[str, Any]) -> int | None:
    return next(
        (
            index
            for index, item in enumerate(queue.get("items", []))
            if item.get("status") != "completed"
        ),
        None,
    )


def authorize_wrapper_operation(
    queue_dir: Path,
    *,
    run_id: str | None = None,
    orchestration_root: Path | None = None,
    job_dir: Path | None = None,
) -> None:
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    _source, _scope, queue = _load_plans(queue_dir)
    if (run_id is None) == (job_dir is None):
        raise TableDFormalQueueError(
            "wrapper authorization requires exactly one detach or job operation"
        )
    if run_id is not None:
        if orchestration_root is None:
            raise TableDFormalQueueError("detach authorization lacks orchestration root")
        index = _active_index(queue)
        if index is None:
            raise TableDFormalQueueError("completed queue cannot authorize another detach")
        item = queue["items"][index]
        planned = queue["plan"]["items"][index]
        expected_root = serial_queue._item_orchestration_root(queue, item)
        if not (
            run_id == item.get("run_id") == planned.get("run_id")
            and item.get("runner") == planned.get("runner") == "paper"
            and item.get("status") in {"reserved", "launching"}
            and orchestration_root.expanduser().resolve(strict=False) == expected_root
        ):
            raise TableDFormalQueueError(
                "wrapper detach does not match the exact active queue item"
            )
        serial_queue._ensure_lease(queue, item, create=False)
        return

    assert job_dir is not None
    job_dir = job_dir.expanduser().resolve(strict=True)
    matches = []
    for index, item in enumerate(queue["items"]):
        root = serial_queue._item_orchestration_root(queue, item)
        if job_dir.parent == root:
            matches.append((index, item))
    if len(matches) != 1:
        raise TableDFormalQueueError("detached job is not owned by one queue item")
    index, item = matches[0]
    if not (
        item.get("run_id") == queue["plan"]["items"][index].get("run_id")
        and item.get("status") in {"launching", "launched", "completed"}
    ):
        raise TableDFormalQueueError("detached job item identity/status drifted")
    launch = _read_json(job_dir / "launch.json", label="detached launch")
    status = _read_json(job_dir / "status.json", label="detached status")
    if launch.get("run_ids") != [item["run_id"]] or status.get("run_ids") != [
        item["run_id"]
    ]:
        raise TableDFormalQueueError("detached job run identity differs from queue")
    if queue.get("status") != "completed":
        serial_queue._ensure_lease(queue, item, create=False)


def run_training_queue(
    queue_dir: Path, *, poll_seconds: float, once: bool = False
) -> MutableMapping[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    _load_plans(queue_dir)
    return serial_queue.run_queue(
        queue_dir, poll_seconds=poll_seconds, once=once
    )


def queue_status(queue_dir: Path) -> dict[str, Any]:
    source, scope, _queue = _load_plans(queue_dir)
    return {
        "schema": "pivot.stageb.table_d_formal_queue_status/v1",
        "profile": PROFILE,
        "source_plan_semantic_sha256": source["semantic_sha256"],
        "scope_plan_semantic_sha256": scope["semantic_sha256"],
        "completion_attestation_present": (
            Path(queue_dir) / COMPLETION_NAME
        ).is_file(),
        "generic_queue": serial_queue.queue_status(queue_dir),
    }


def _compare_input_rehash(
    launch: Mapping[str, Any], postflight: Mapping[str, Any], *, label: str
) -> Path:
    output = Path(str(launch.get("output_dir", ""))).resolve(strict=True)
    path = (output / "input_rehash.json").resolve(strict=True)
    persisted = _read_json(path, label=f"{label} input rehash")
    try:
        replay = paper._rehash_inputs(launch)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise TableDFormalQueueError(f"{label} input rehash replay failed: {exc}") from exc
    for key in ("schema", "status", "records"):
        if persisted.get(key) != replay.get(key):
            raise TableDFormalQueueError(f"{label} input rehash replay drifted")
    if postflight.get("input_rehash") != persisted:
        raise TableDFormalQueueError(f"{label} postflight input rehash drifted")
    return path


def _replay_phase(
    *,
    runtime: paper.Runtime,
    row: paper.MatrixRow,
    seed: int,
    run_root: Path,
    phase: paper.Phase,
    sequence_phase: Mapping[str, Any],
    completed_phase: Mapping[str, Any],
    scope_phase: Mapping[str, Any],
    rank_checkpoint: Path | None,
) -> tuple[dict[str, Any], Path]:
    output = paper._phase_output(run_root, row, phase).resolve(strict=True)
    launch_path = (output / "launch_manifest.json").resolve(strict=True)
    postflight_path = (output / "postflight.json").resolve(strict=True)
    launch = _read_json(launch_path, label=f"{row.row_id}:{seed}/{phase.phase_id} launch")
    postflight = _read_json(
        postflight_path, label=f"{row.row_id}:{seed}/{phase.phase_id} postflight"
    )
    label = f"{row.row_id}:{seed}/{phase.phase_id}"
    expected_planned = paper._phase_manifest(
        runtime,
        row,
        seed,
        phase,
        output,
        paper.token_launcher.HashCache(),
        rank_checkpoint=None,
    )
    expected_actual = paper._phase_manifest(
        runtime,
        row,
        seed,
        phase,
        output,
        paper.token_launcher.HashCache(),
        rank_checkpoint=(
            rank_checkpoint
            if phase.pretrain_source == "rank_phase_checkpoint"
            else None
        ),
    )
    if _immutable_phase(sequence_phase) != _immutable_phase(expected_planned):
        raise TableDFormalQueueError(f"{label} sequence plan drifted")
    if _immutable_phase(launch) != _immutable_phase(expected_actual):
        raise TableDFormalQueueError(f"{label} launch differs from current code plan")
    if not (
        launch.get("status") == "completed"
        and launch.get("returncode") == 0
        and completed_phase.get("phase_id") == phase.phase_id
        and completed_phase.get("status") == "completed"
        and Path(str(completed_phase.get("output_dir", ""))).resolve(strict=True)
        == output
        and launch.get("postflight") == postflight
        and postflight.get("status") == "passed"
        and postflight.get("run_id") == f"{row.row_id}:{seed}"
        and postflight.get("phase_id") == phase.phase_id
    ):
        raise TableDFormalQueueError(f"{label} completion identity drifted")
    planned_identity = _normalize_input_records(
        expected_planned["inputs"]["records"]
    )
    actual_identity = _normalize_input_records(launch["inputs"]["records"])
    base_actual = {
        path: record
        for path, record in actual_identity.items()
        if "rank_phase_model_state_pretrain" not in record["roles"]
    }
    if _input_identity_sha(base_actual) != scope_phase.get(
        "input_identity_sha256"
    ) or _input_identity_sha(planned_identity) != scope_phase.get(
        "input_identity_sha256"
    ):
        raise TableDFormalQueueError(f"{label} predeclared input identity drifted")
    if (
        _canonical_sha(launch.get("command"))
        != scope_phase.get("command_sha256")
        or _canonical_sha(expected_planned.get("command"))
        != scope_phase.get("command_sha256")
    ):
        raise TableDFormalQueueError(f"{label} predeclared command identity drifted")
    _compare_input_rehash(launch, postflight, label=label)
    artifacts = postflight.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise TableDFormalQueueError(f"{label} postflight artifacts are missing")
    checkpoint = evaluator._verify_declared_file(
        artifacts.get("checkpoint"),
        label=f"{label} checkpoint",
        cache=evaluator.HashCache(),
    )
    completed_checkpoint = evaluator._verify_declared_file(
        completed_phase.get("checkpoint"),
        label=f"{label} sequence checkpoint",
        cache=evaluator.HashCache(),
    )
    if checkpoint != completed_checkpoint or checkpoint != output / "checkpoint_iter.pth":
        raise TableDFormalQueueError(f"{label} checkpoint identity drifted")
    try:
        metadata = paper._inspect_checkpoint_safely(runtime, checkpoint)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise TableDFormalQueueError(f"{label} checkpoint replay failed: {exc}") from exc
    if postflight.get("checkpoint_metadata") != metadata:
        raise TableDFormalQueueError(f"{label} checkpoint metadata drifted")
    scorer_audit = None
    if phase.scorer_warmstart:
        scorer_audit = _read_json(
            output / "stage_b_v15_scorer_init_audit.json",
            label=f"{label} scorer audit",
        )
    pretrain = (
        rank_checkpoint
        if phase.pretrain_source == "rank_phase_checkpoint"
        else runtime.stage_a_init
    )
    if pretrain is None:
        raise TableDFormalQueueError(f"{label} rank pretrain is missing")
    try:
        paper._validate_checkpoint_metadata(
            metadata,
            runtime=runtime,
            row=row,
            seed=seed,
            phase=phase,
            output_dir=output,
            pretrain_path=pretrain,
            scorer_audit=scorer_audit,
        )
        numerical = paper._training_numerical_status(
            output / "info.txt", output / "train_console.log"
        )
        telemetry = paper._summarize_nvidia_csv(output / "gpu_telemetry.csv")
        gpu_environment = _read_json(
            output / "gpu_environment.json", label=f"{label} GPU environment"
        )
        persisted_telemetry = _read_json(
            output / "gpu_telemetry_summary.json",
            label=f"{label} GPU telemetry summary",
        )
        paper._validate_gpu_telemetry_contract(gpu_environment, persisted_telemetry)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise TableDFormalQueueError(f"{label} postflight replay failed: {exc}") from exc
    if numerical != postflight.get("numerical_status"):
        raise TableDFormalQueueError(f"{label} numerical replay drifted")
    for key in ("schema", "status", "devices"):
        if telemetry.get(key) != persisted_telemetry.get(key):
            raise TableDFormalQueueError(f"{label} telemetry replay drifted")
    if phase.diagnostic_interval > 0:
        logs = (output / "info.txt").read_text(
            encoding="utf-8", errors="replace"
        ) + (output / "train_console.log").read_text(
            encoding="utf-8", errors="replace"
        )
        fragment = (
            "stage_b_v22_grad_cosine"
            if row.row_id in {"S0", "S1"}
            else "stage_b_v22_branch_isolation_pass"
        )
        if fragment not in logs:
            raise TableDFormalQueueError(f"{label} diagnostic replay is absent")
    return (
        {
            "phase_id": phase.phase_id,
            "launch_manifest": _file_record(
                launch_path, roles=("training_phase_launch",)
            ),
            "postflight": _file_record(
                postflight_path, roles=("training_phase_postflight",)
            ),
            "checkpoint": _file_record(
                checkpoint, roles=("training_phase_checkpoint",)
            ),
            "input_identity_sha256": _input_identity_sha(actual_identity),
            "base_input_identity_sha256": _input_identity_sha(base_actual),
            "command_sha256": _canonical_sha(launch["command"]),
        },
        checkpoint,
    )


def _verify_completed_training_run(
    *,
    queue_dir: Path,
    source_plan: Mapping[str, Any],
    scope_plan: Mapping[str, Any],
    generic_item: Mapping[str, Any],
    generic_evidence: Mapping[str, Any],
    runtime: paper.Runtime,
    run_id: str,
) -> dict[str, Any]:
    row_id, raw_seed = run_id.split(":", 1)
    seed = int(raw_seed)
    row = paper.ROW_BY_ID[row_id]
    run_root = (
        Path(source_plan["runtime"]["output_root"]) / row_id / f"seed{seed}"
    ).resolve(strict=True)
    if not (
        generic_item.get("run_id") == run_id
        and generic_item.get("runner") == "paper"
        and generic_item.get("status") == "completed"
        and generic_evidence.get("run_id") == run_id
        and generic_evidence.get("runner") == "paper"
        and Path(str(generic_evidence.get("output_root", ""))).resolve(
            strict=True
        )
        == run_root
    ):
        raise TableDFormalQueueError(f"{run_id} generic active-item evidence drifted")
    sequence_path = (run_root / "sequence_manifest.json").resolve(strict=True)
    sequence = _read_json(sequence_path, label=f"{run_id} sequence")
    current_plan = paper.build_manifest(
        runtime, row, seed, paper.token_launcher.HashCache()
    )
    if not (
        sequence.get("status") == "completed"
        and sequence.get("run_id") == run_id
        and _immutable_sequence(sequence) == _immutable_sequence(current_plan)
        and _canonical_sha(_immutable_sequence(current_plan))
        == scope_plan["runs"][run_id]["sequence_contract_sha256"]
    ):
        raise TableDFormalQueueError(f"{run_id} completed sequence differs from scope plan")
    planned_phases = sequence.get("phases")
    completed_phases = sequence.get("completed_phases")
    phases = paper._phases(runtime, row)
    expected_ids = [phase.phase_id for phase in phases]
    if not (
        isinstance(planned_phases, list)
        and isinstance(completed_phases, list)
        and [value.get("phase", {}).get("phase_id") for value in planned_phases]
        == expected_ids
        and [value.get("phase_id") for value in completed_phases] == expected_ids
        and [value.get("phase_id") for value in scope_plan["runs"][run_id]["phases"]]
        == expected_ids
    ):
        raise TableDFormalQueueError(f"{run_id} phase order is not exact")
    phase_reports: list[dict[str, Any]] = []
    rank_checkpoint: Path | None = None
    for phase, planned, completed, scope_phase in zip(
        phases,
        planned_phases,
        completed_phases,
        scope_plan["runs"][run_id]["phases"],
    ):
        report, checkpoint = _replay_phase(
            runtime=runtime,
            row=row,
            seed=seed,
            run_root=run_root,
            phase=phase,
            sequence_phase=planned,
            completed_phase=completed,
            scope_phase=scope_phase,
            rank_checkpoint=rank_checkpoint,
        )
        phase_reports.append(report)
        if phase.phase_id == "rank":
            rank_checkpoint = checkpoint
    try:
        final_source = evaluator._resolve_paper_source(
            run_root,
            evaluator.HashCache(),
            training_phase="final",
            training_queue_dir=queue_dir,
            allow_nonformal_fixture=True,
        )
    except (OSError, ValueError, evaluator.PaperEvaluationError) as exc:
        raise TableDFormalQueueError(f"{run_id} final source replay failed: {exc}") from exc
    if final_source.training_run_id != run_id or final_source.training_seed != seed:
        raise TableDFormalQueueError(f"{run_id} final evaluation source drifted")
    s3_atomic = None
    if row_id == "S3":
        try:
            rank_source = evaluator._resolve_paper_source(
                run_root,
                evaluator.HashCache(),
                training_phase="rank",
                training_queue_dir=queue_dir,
                allow_nonformal_fixture=True,
            )
            lineage = diagnostics._verify_s3_training_lineage(
                rank_source, final_source
            )
            allowlist = diagnostics.checkpoint_allowlist(
                rank_source.checkpoint, final_source.checkpoint
            )
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            evaluator.PaperEvaluationError,
            diagnostics.TableDDiagnosticsError,
        ) as exc:
            raise TableDFormalQueueError(f"{run_id} atomic S3 replay failed: {exc}") from exc
        if not (
            rank_source.training_phase == "rank"
            and rank_source.diagnostic_only is True
            and final_source.training_phase == "final"
            and final_source.final_phase_id == "confidence"
        ):
            raise TableDFormalQueueError(f"{run_id} rank/confidence source roles drifted")
        s3_atomic = {
            "status": "passed",
            "phase_order": ["isolation_probe", "rank", "confidence"],
            "isolation_probe_replayed": True,
            "rank_source_checkpoint": _file_record(
                rank_source.checkpoint, roles=("S3_rank_checkpoint",)
            ),
            "confidence_source_checkpoint": _file_record(
                final_source.checkpoint, roles=("S3_confidence_checkpoint",)
            ),
            "rank_to_confidence_lineage": lineage,
            "confidence_only_checkpoint_diff": allowlist,
        }
    job_dir = Path(str(generic_evidence.get("job_dir", ""))).resolve(strict=True)
    return {
        "schema": "pivot.stageb.table_d_formal_run_verification/v1",
        "status": "passed",
        "profile": PROFILE,
        "run_id": run_id,
        "row_id": row_id,
        "seed": seed,
        "run_root": str(run_root),
        "scope_sha256": scope_plan["runs"][run_id]["scope_sha256"],
        "sequence_manifest": _file_record(
            sequence_path, roles=("training_sequence",)
        ),
        "generic_queue_item": {
            "index": generic_item["index"],
            "runner": "paper",
            "job_dir": str(job_dir),
            "detached_launch": _file_record(
                job_dir / "launch.json", roles=("detached_launch",)
            ),
            "detached_status": _file_record(
                job_dir / "status.json", roles=("detached_status",)
            ),
        },
        "phases": phase_reports,
        "selected_final_checkpoint": _file_record(
            final_source.checkpoint, roles=("formal_final_checkpoint",)
        ),
        "s3_atomic_replay": s3_atomic,
    }


def _completion_payload(queue_dir: Path) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    source, scope, queue = _load_plans(queue_dir)
    generic = serial_queue.verify_queue(queue_dir)
    if queue.get("status") != "completed" or generic.get("status") != "passed":
        raise TableDFormalQueueError(
            "formal Table-D generic training queue is not completed and verified"
        )
    verified_by_id = {
        value.get("run_id"): value
        for value in generic.get("verified_items", [])
        if isinstance(value, Mapping)
    }
    if set(verified_by_id) != set(RUN_IDS):
        raise TableDFormalQueueError("generic queue verification inventory drifted")
    environment = source["runtime"]["environment"]
    with _environment(environment):
        runtime = paper.runtime_from_environment()
        _validate_runtime(
            runtime,
            output_root=Path(source["runtime"]["output_root"]),
            runner_python=Path(source["runtime"]["python"]),
        )
        reports = {
            run_id: _verify_completed_training_run(
                queue_dir=queue_dir,
                source_plan=source,
                scope_plan=scope,
                generic_item=queue["items"][index],
                generic_evidence=verified_by_id[run_id],
                runtime=runtime,
                run_id=run_id,
            )
            for index, run_id in enumerate(RUN_IDS)
        }
    payload: dict[str, Any] = {
        "schema": COMPLETION_SCHEMA,
        "status": "passed",
        "profile": PROFILE,
        "ordered_run_ids": list(RUN_IDS),
        "formal_training_contract": FORMAL_TRAINING_CONTRACT,
        "queue": dict(source["queue"]),
        "source_plan": _file_record(
            queue_dir / SOURCE_PLAN_NAME, roles=(SOURCE_ROLE,)
        ),
        "scope_plan": _file_record(
            queue_dir / SCOPE_PLAN_NAME, roles=(SCOPE_ROLE,)
        ),
        "readiness": dict(source["readiness"]),
        "common_input_contract": dict(source["common_input_contract"]),
        "active_item_identity_replayed": True,
        "runs": reports,
    }
    payload["semantic_sha256"] = _semantic_sha(payload)
    return payload


def verify_training_queue(
    queue_dir: Path, *, persist: bool = True
) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    try:
        payload = _completion_payload(queue_dir)
    except TableDFormalQueueError:
        raise
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        serial_queue.QueueContractError,
    ) as exc:
        raise TableDFormalQueueError(
            f"formal Table-D completion replay failed: {exc}"
        ) from exc
    path = queue_dir / COMPLETION_NAME
    if path.exists():
        existing = _read_json(path, label="formal completion attestation")
        if existing != payload:
            raise TableDFormalQueueError(
                "persisted formal Table-D completion attestation drifted"
            )
    elif persist:
        _write_json_atomic(path, payload)
    return payload


def formal_evaluation_evidence(
    queue_dir: Path,
    *,
    run_id: str,
    run_root: Path,
    training_phase: str,
) -> dict[str, Any]:
    if training_phase not in {"final", "rank"}:
        raise TableDFormalQueueError("formal evaluation phase must be final or rank")
    if training_phase == "rank" and not run_id.startswith("S3:"):
        raise TableDFormalQueueError("rank evaluation is restricted to S3")
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    attestation_path = (queue_dir / COMPLETION_NAME).resolve(strict=True)
    persisted = _read_json(attestation_path, label="formal completion attestation")
    replayed = verify_training_queue(queue_dir, persist=False)
    if persisted != replayed or run_id not in persisted["runs"]:
        raise TableDFormalQueueError("formal completion attestation replay drifted")
    receipt = persisted["runs"][run_id]
    if Path(receipt["run_root"]).resolve(strict=True) != run_root.expanduser().resolve(
        strict=True
    ):
        raise TableDFormalQueueError("evaluation run root differs from attestation")
    return {
        "profile": PROFILE,
        "run_id": run_id,
        "training_phase": training_phase,
        "queue_id": persisted["queue"]["queue_id"],
        "queue_plan_sha256": persisted["queue"]["plan_sha256"],
        "completion_semantic_sha256": persisted["semantic_sha256"],
        "source_plan": persisted["source_plan"],
        "scope_plan": persisted["scope_plan"],
        "completion_attestation": _file_record(
            attestation_path, roles=("table_d_formal_completion_attestation",)
        ),
        "run_verification": receipt,
    }


def preflight_training_queue(
    queue_dir: Path,
    *,
    output_root: Path,
    s2_soak_seal: Path,
    s2f_confirmation_root: Path,
    runner_python: Path = paper.DEFAULT_PYTHON,
) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=False)
    output_root = output_root.expanduser().resolve(strict=False)
    runner_python = _validate_runner_python(runner_python)
    if queue_dir.exists() or output_root.exists():
        raise FileExistsError("preflight requires fresh queue and output roots")
    readiness = {
        "s2_b40_u50_soak": verify_s2_soak_seal(s2_soak_seal),
        "s2f_b40_u2_confirmation": verify_s2f_confirmation(
            s2f_confirmation_root
        ),
    }
    _validate_readiness_pair(readiness)
    environment = _runtime_environment(
        output_root=output_root, runner_python=runner_python
    )
    with _environment(environment):
        runtime = paper.runtime_from_environment()
        _validate_runtime(
            runtime, output_root=output_root, runner_python=runner_python
        )
        manifests = _planned_manifests(runtime)
    return {
        "schema": "pivot.stageb.table_d_formal_preflight/v1",
        "status": "ready",
        "mutated": False,
        "profile": PROFILE,
        "queue_dir": str(queue_dir),
        "output_root": str(output_root),
        "ordered_run_ids": list(RUN_IDS),
        "formal_training_contract": FORMAL_TRAINING_CONTRACT,
        "readiness": readiness,
        "common_input_contract": _common_input_contract(manifests),
        "source_closure": _source_closure(
            manifests, runner_python=runner_python
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("preflight", "create"):
        child = subparsers.add_parser(mode)
        child.add_argument("queue_dir", type=Path)
        child.add_argument("--output-root", type=Path, required=True)
        child.add_argument("--s2-soak-seal", type=Path, required=True)
        child.add_argument("--s2f-confirmation-root", type=Path, required=True)
        child.add_argument(
            "--runner-python", type=Path, default=paper.DEFAULT_PYTHON
        )
        if mode == "create":
            child.add_argument(
                "--token-runner", type=Path, default=serial_queue.DEFAULT_TOKEN_RUNNER
            )
            child.add_argument(
                "--lease-root", type=Path, default=serial_queue.DEFAULT_LEASE_ROOT
            )
            child.add_argument("--gpu-key")
    run = subparsers.add_parser("run")
    run.add_argument("queue_dir", type=Path)
    run.add_argument("--poll-seconds", type=float, default=30.0)
    run.add_argument("--once", action="store_true")
    reconcile = subparsers.add_parser("reconcile")
    reconcile.add_argument("queue_dir", type=Path)
    for mode in ("status", "verify"):
        child = subparsers.add_parser(mode)
        child.add_argument("queue_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.mode == "preflight":
            report = preflight_training_queue(
                args.queue_dir,
                output_root=args.output_root,
                s2_soak_seal=args.s2_soak_seal,
                s2f_confirmation_root=args.s2f_confirmation_root,
                runner_python=args.runner_python,
            )
        elif args.mode == "create":
            report = create_training_queue(
                args.queue_dir,
                output_root=args.output_root,
                s2_soak_seal=args.s2_soak_seal,
                s2f_confirmation_root=args.s2f_confirmation_root,
                runner_python=args.runner_python,
                token_runner=args.token_runner,
                lease_root=args.lease_root,
                gpu_key=args.gpu_key,
            )
        elif args.mode == "run":
            queue = run_training_queue(
                args.queue_dir,
                poll_seconds=args.poll_seconds,
                once=args.once,
            )
            report = queue_status(args.queue_dir)
            if queue.get("status") == "failed":
                print(json.dumps(report, indent=2, sort_keys=True))
                return 1
        elif args.mode == "reconcile":
            run_training_queue(args.queue_dir, poll_seconds=1.0, once=True)
            report = queue_status(args.queue_dir)
        elif args.mode == "status":
            report = queue_status(args.queue_dir)
        elif args.mode == "verify":
            report = verify_training_queue(args.queue_dir, persist=True)
        else:
            parser.error(f"unsupported mode {args.mode!r}")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        PermissionError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        TableDFormalQueueError,
        serial_queue.QueueContractError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
