#!/usr/bin/env python3
"""Create, supervise, and attest the formal six-run Table-B v2 panel."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_stageb_serial_matrix_queue as generic_queue  # noqa: E402
from tools import run_stageb_table_b_v2 as v2_runner  # noqa: E402
from util import stage_b_table_b_v2_contract as contract  # noqa: E402


SOURCE_PLAN_NAME = v2_runner.FORMAL_SOURCE_PLAN_NAME
SCOPE_PLAN_NAME = v2_runner.FORMAL_SCOPE_PLAN_NAME
COMPLETION_NAME = v2_runner.FORMAL_COMPLETION_ATTESTATION_NAME
COMPLETION_SCHEMA = "pivot.stageb.table_b_v2_formal_completion/v1"
GENERIC_EXTENSION_SCHEMA = "pivot.stageb.table_b_v2_generic_queue_extension/v1"
FORMAL_SOURCE_ROLE = "table_b_v2_formal_source_plan"


class FormalQueueError(RuntimeError):
    """Raised when the dedicated Table-B v2 queue contract is not exact."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _semantic_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("semantic_sha256", None)
    return _canonical_sha256(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, roles: Sequence[str]) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": int(path.stat().st_size),
        "roles": sorted(set(roles)),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="ascii") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalQueueError(f"invalid {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise FormalQueueError(f"{label} must be a JSON object")
    return value


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


def _formal_runtime_environment(
    *, output_root: Path, runner_python: Path
) -> dict[str, str]:
    return {
        "PIVOT_PYTHON": str(runner_python),
        "PIVOT_TN_OUTPUT_ROOT": str(output_root),
        "PIVOT_BATCH_SIZE": str(contract.FORMAL_BATCH_SIZE),
        "PIVOT_MAX_TRAIN_ITERS": str(contract.FORMAL_TRAIN_UPDATES),
        "PIVOT_ITER_CHECKPOINT_INTERVAL": str(contract.FORMAL_CHECKPOINT_INTERVAL),
    }


def _normalize_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    roles: dict[str, set[str]] = {}
    for record in records:
        path = str(Path(str(record["path"])).resolve(strict=True))
        identity = {
            "path": path,
            "sha256": str(record["sha256"]),
            "size_bytes": int(record["size_bytes"]),
        }
        previous = normalized.setdefault(path, identity)
        if previous != identity:
            raise FormalQueueError(f"conflicting source identities for {path}")
        role = record.get("role")
        if isinstance(role, str) and role:
            roles.setdefault(path, set()).add(role)
        raw_roles = record.get("roles")
        if isinstance(raw_roles, list):
            roles.setdefault(path, set()).update(str(value) for value in raw_roles)
    return {
        path: {**identity, "roles": sorted(roles.get(path, {"input"}))}
        for path, identity in normalized.items()
    }


def _input_identity_sha256(identity: Mapping[str, Mapping[str, Any]]) -> str:
    values = [dict(identity[path]) for path in sorted(identity)]
    for value in values:
        value.pop("roles", None)
    return _canonical_sha256(values)


def _collect_condition_inputs(launcher: Any, runtime: Any) -> dict[str, dict[str, Any]]:
    conditions: dict[str, dict[str, Any]] = {}
    for table_b_id in ("D2m", "D3m"):
        row = launcher.ROW_BY_ID[table_b_id]
        phase = launcher._phases(runtime, row)[0]
        config = (REPO_ROOT / phase.config).resolve(strict=True)
        launcher._validate_phase_config(row, phase)
        _dataset_contract, dataset_sources = launcher._validate_dataset(row, runtime)
        paths: dict[Path, set[str]] = {
            Path(runtime.stage_a_init).resolve(strict=True): {"stage_a_initializer"},
            Path(runtime.scorer_warmstart).resolve(strict=True): {"scorer_warmstart"},
            (REPO_ROOT / row.dataset).resolve(strict=True): {"dataset_manifest"},
        }
        for path in launcher.token_launcher._config_dependencies(config):
            paths.setdefault(Path(path).resolve(strict=True), set()).add(
                "config_dependency"
            )
        for path in dataset_sources:
            paths.setdefault(Path(path).resolve(strict=True), set()).add("dataset_source")
        for path in launcher._relevant_repository_sources():
            paths.setdefault(Path(path).resolve(strict=True), set()).add(
                "repository_source"
            )
        records = [
            _file_record(path, roles=sorted(roles))
            for path, roles in sorted(paths.items(), key=lambda item: str(item[0]))
        ]
        conditions[table_b_id] = {
            "records": records,
            "identity": _normalize_records(records),
        }
    return conditions


def _common_input_contract(
    conditions: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    d2 = conditions["D2m"]["identity"]
    d3 = conditions["D3m"]["identity"]
    common_paths = sorted(set(d2) & set(d3))
    for path in common_paths:
        left = {key: value for key, value in d2[path].items() if key != "roles"}
        right = {key: value for key, value in d3[path].items() if key != "roles"}
        if left != right:
            raise FormalQueueError(f"common D2m/D3m input identity drifted: {path}")
    specific = {
        "D2m": sorted(set(d2) - set(d3)),
        "D3m": sorted(set(d3) - set(d2)),
    }
    audit = contract.validate_v2_audit()
    expected_specific = {
        "D2m": sorted(
            str(path.resolve(strict=True))
            for path in (
                contract.DATASET_PATH_BY_ID["D2m"],
                REPO_ROOT / v2_runner.CONFIG_BY_ID["D2m"],
                Path(audit["outputs"]["d2m_train"]["path"]),
            )
        ),
        "D3m": sorted(
            str(path.resolve(strict=True))
            for path in (
                contract.DATASET_PATH_BY_ID["D3m"],
                REPO_ROOT / v2_runner.CONFIG_BY_ID["D3m"],
                Path(audit["outputs"]["d3m_train"]["path"]),
            )
        ),
    }
    if specific != expected_specific:
        raise FormalQueueError(
            "D2m/D3m differ outside the declared config/manifest/TN-source paths"
        )
    common_records = [d2[path] for path in common_paths]
    return {
        "status": "passed",
        "common_inputs_identical": True,
        "only_declared_condition_inputs_differ": True,
        "common_records": common_records,
        "common_identity_sha256": _input_identity_sha256(
            {record["path"]: record for record in common_records}
        ),
        "declared_condition_specific_paths": specific,
        "condition_specific_records": {
            condition: [conditions[condition]["identity"][path] for path in paths]
            for condition, paths in specific.items()
        },
    }


def _source_closure(
    conditions: Mapping[str, Mapping[str, Any]], *, runner_python: Path
) -> dict[str, Any]:
    records = []
    for condition in ("D2m", "D3m"):
        records.extend(conditions[condition]["records"])
    records.extend(
        [
            _file_record(runner_python, roles=["runner_python"]),
            _file_record(Path(__file__), roles=["formal_queue_controller"]),
            _file_record(
                Path(generic_queue.__file__), roles=["shared_gpu_lease_queue"]
            ),
        ]
    )
    normalized = _normalize_records(records)
    values = [normalized[path] for path in sorted(normalized)]
    return {
        "status": "sealed",
        "records": values,
        "semantic_sha256": _canonical_sha256(values),
    }


def _validate_runtime(runtime: Any, *, runner_python: Path, output_root: Path) -> None:
    expected = {
        "python": runner_python.resolve(strict=True),
        "batch_size": contract.FORMAL_BATCH_SIZE,
        "total_train_iters": contract.FORMAL_TRAIN_UPDATES,
        "iter_checkpoint_interval": contract.FORMAL_CHECKPOINT_INTERVAL,
        "tn_output_root": output_root.resolve(),
    }
    observed = {
        "python": Path(runtime.python).resolve(strict=True),
        "batch_size": runtime.batch_size,
        "total_train_iters": runtime.total_train_iters,
        "iter_checkpoint_interval": runtime.iter_checkpoint_interval,
        "tn_output_root": Path(runtime.tn_output_root).resolve(),
    }
    if observed != expected:
        raise FormalQueueError(f"formal runtime mismatch: expected {expected}, got {observed}")


def create_formal_queue(
    queue_dir: Path,
    *,
    output_root: Path,
    runner_python: Path,
    token_runner: Path = generic_queue.DEFAULT_TOKEN_RUNNER,
    lease_root: Path = generic_queue.DEFAULT_LEASE_ROOT,
    gpu_key: str | None = None,
) -> dict[str, Any]:
    queue_dir = Path(queue_dir).expanduser().resolve(strict=False)
    output_root = Path(output_root).expanduser().resolve(strict=False)
    runner_python = Path(runner_python).expanduser().resolve(strict=True)
    if not runner_python.is_file() or not os.access(runner_python, os.X_OK):
        raise FormalQueueError(f"runner Python is not executable: {runner_python}")
    environment = _formal_runtime_environment(
        output_root=output_root, runner_python=runner_python
    )
    with _environment(environment):
        launcher = v2_runner._launcher()
        runtime = launcher.runtime_from_environment()
        _validate_runtime(
            runtime, runner_python=runner_python, output_root=output_root
        )
        conflicts = [
            launcher.output_directory(runtime, launcher.ROW_BY_ID[run_id.split(":")[0]], int(run_id.split(":")[1]))
            for run_id in contract.FORMAL_RUN_IDS
            if launcher.output_directory(runtime, launcher.ROW_BY_ID[run_id.split(":")[0]], int(run_id.split(":")[1])).exists()
        ]
        if conflicts:
            raise FileExistsError(
                "formal Table-B v2 run roots must be fresh: "
                + ", ".join(str(path) for path in conflicts)
            )
        conditions = _collect_condition_inputs(launcher, runtime)
        common = _common_input_contract(conditions)
        closure = _source_closure(conditions, runner_python=runner_python)
        controller = _file_record(
            Path(__file__), roles=["formal_queue_controller"]
        )
        controller.pop("roles", None)
        plan_extension = {
            "schema": GENERIC_EXTENSION_SCHEMA,
            "profile": contract.FORMAL_PROFILE,
            "ordered_run_ids": list(contract.FORMAL_RUN_IDS),
            "formal_training_contract": dict(contract.FORMAL_TRAINING_CONTRACT),
            "explicit_output_root": str(output_root),
            "source_closure_semantic_sha256": closure["semantic_sha256"],
            "common_input_identity_sha256": common["common_identity_sha256"],
            "dedicated_controller": controller,
        }
        plan_extension["semantic_sha256"] = _semantic_sha256(plan_extension)
        queue = generic_queue.create_queue(
            queue_dir,
            run_ids=contract.FORMAL_RUN_IDS,
            runner_python=runner_python,
            token_runner=Path(token_runner),
            paper_runner=Path(v2_runner.__file__),
            lease_root=Path(lease_root),
            gpu_key=gpu_key,
            plan_extensions=plan_extension,
        )
        plan = queue["plan"]
        source_plan: dict[str, Any] = {
            "schema": contract.FORMAL_SOURCE_PLAN_SCHEMA,
            "status": "sealed",
            "profile": contract.FORMAL_PROFILE,
            "formal_training_contract": dict(contract.FORMAL_TRAINING_CONTRACT),
            "ordered_run_ids": list(contract.FORMAL_RUN_IDS),
            "queue": {
                "queue_dir": str(queue_dir),
                "queue_id": plan["queue_id"],
                "plan_sha256": queue["plan_sha256"],
                "gpu_key": plan["gpu_key"],
                "lease_path": plan["lease_path"],
                "extension_semantic_sha256": plan_extension["semantic_sha256"],
            },
            "runtime": {
                "python": str(Path(runtime.python).resolve(strict=True)),
                "batch_size": runtime.batch_size,
                "total_train_iters": runtime.total_train_iters,
                "iter_checkpoint_interval": runtime.iter_checkpoint_interval,
                "output_root": str(output_root),
                "stage_a_init": str(Path(runtime.stage_a_init).resolve(strict=True)),
                "stage_a_init_sha256": _sha256_file(Path(runtime.stage_a_init)),
                "scorer_warmstart": str(
                    Path(runtime.scorer_warmstart).resolve(strict=True)
                ),
                "scorer_warmstart_sha256": _sha256_file(
                    Path(runtime.scorer_warmstart)
                ),
            },
            "source_closure": closure,
            "common_input_contract": common,
        }
        source_plan["semantic_sha256"] = _semantic_sha256(source_plan)
        source_path = queue_dir / SOURCE_PLAN_NAME
        _write_json_atomic(source_path, source_plan)
        contract.validate_formal_source_plan(source_path)

        cache = launcher.token_launcher.HashCache()
        run_records: dict[str, Any] = {}
        for run_id in contract.FORMAL_RUN_IDS:
            table_b_id, raw_seed = run_id.split(":", 1)
            seed = int(raw_seed)
            manifest = v2_runner.build_formal_planned_manifest(
                runtime,
                launcher.ROW_BY_ID[table_b_id],
                seed,
                cache,
                source_plan_path=source_path,
            )
            phase = manifest["phases"][0]
            identity = _normalize_records(phase["inputs"]["records"])
            expected_paths = set(conditions[table_b_id]["identity"]) | {
                str(source_path.resolve(strict=True))
            }
            if set(identity) != expected_paths:
                raise FormalQueueError(
                    f"{run_id} planned inputs differ from the sealed closure"
                )
            run_records[run_id] = {
                "run_id": run_id,
                "table_b_id": table_b_id,
                "seed": seed,
                "scope_sha256": manifest["table_b_v2_scope_sha256"],
                "input_identity_sha256": _input_identity_sha256(identity),
                "command_sha256": hashlib.sha256(
                    "\0".join(phase["command"]).encode("utf-8")
                ).hexdigest(),
                "output_root": manifest["output_dir"],
            }
        scope_plan: dict[str, Any] = {
            "schema": contract.FORMAL_SCOPE_PLAN_SCHEMA,
            "status": "sealed",
            "profile": contract.FORMAL_PROFILE,
            "ordered_run_ids": list(contract.FORMAL_RUN_IDS),
            "source_plan": contract.file_record(
                source_path, role=FORMAL_SOURCE_ROLE
            ),
            "source_plan_semantic_sha256": source_plan["semantic_sha256"],
            "queue": dict(source_plan["queue"]),
            "runs": run_records,
        }
        scope_plan["semantic_sha256"] = _semantic_sha256(scope_plan)
        scope_path = queue_dir / SCOPE_PLAN_NAME
        _write_json_atomic(scope_path, scope_plan)
        contract.validate_formal_scope_plan(scope_path)
    return {
        "status": "planned",
        "profile": contract.FORMAL_PROFILE,
        "queue_dir": str(queue_dir),
        "queue_id": queue["plan"]["queue_id"],
        "queue_plan_sha256": queue["plan_sha256"],
        "source_plan": _file_record(source_path, roles=[FORMAL_SOURCE_ROLE]),
        "scope_plan": _file_record(scope_path, roles=["table_b_v2_formal_scope_plan"]),
        "ordered_run_ids": list(contract.FORMAL_RUN_IDS),
    }


def _load_plans(queue_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    queue_dir = Path(queue_dir).expanduser().resolve(strict=True)
    try:
        source = contract.validate_formal_source_plan(queue_dir / SOURCE_PLAN_NAME)
        scope = contract.validate_formal_scope_plan(queue_dir / SCOPE_PLAN_NAME)
    except (OSError, ValueError, contract.TableBContractError) as error:
        raise FormalQueueError(f"formal Table-B v2 plan verification failed: {error}") from error
    return source, scope


def _execution_environment(queue_dir: Path) -> dict[str, str]:
    source_path = (Path(queue_dir) / SOURCE_PLAN_NAME).resolve(strict=True)
    scope_path = (Path(queue_dir) / SCOPE_PLAN_NAME).resolve(strict=True)
    source, _scope = _load_plans(queue_dir)
    return {
        **_formal_runtime_environment(
            output_root=Path(source["runtime"]["output_root"]),
            runner_python=Path(source["runtime"]["python"]),
        ),
        contract.FORMAL_SOURCE_PLAN_ENV: str(source_path),
        contract.FORMAL_SOURCE_PLAN_SHA_ENV: _sha256_file(source_path),
        contract.FORMAL_SCOPE_PLAN_ENV: str(scope_path),
        contract.FORMAL_SCOPE_PLAN_SHA_ENV: _sha256_file(scope_path),
    }


def run_formal_queue(
    queue_dir: Path, *, poll_seconds: float, once: bool
) -> dict[str, Any]:
    queue_dir = Path(queue_dir).expanduser().resolve(strict=True)
    environment = _execution_environment(queue_dir)
    with _environment(environment):
        return generic_queue.run_queue(
            queue_dir, poll_seconds=poll_seconds, once=once
        )


def _normalized_actual_common_inputs(
    queue_dir: Path, source_plan: Mapping[str, Any]
) -> tuple[str, dict[str, str]]:
    specific = source_plan["common_input_contract"][
        "declared_condition_specific_paths"
    ]
    normalized_by_run: dict[str, dict[str, Any]] = {}
    full_hashes: dict[str, str] = {}
    for run_id in contract.FORMAL_RUN_IDS:
        table_b_id, raw_seed = run_id.split(":", 1)
        run_root = (
            Path(source_plan["runtime"]["output_root"])
            / table_b_id
            / f"seed{int(raw_seed)}"
        ).resolve(strict=True)
        phase = _read_json(run_root / "launch_manifest.json", label=f"{run_id} launch")
        identity = _normalize_records(phase["inputs"]["records"])
        full_hashes[run_id] = _input_identity_sha256(identity)
        excluded = set(specific[table_b_id])
        normalized_by_run[run_id] = {
            path: value for path, value in identity.items() if path not in excluded
        }
    first = normalized_by_run[contract.FORMAL_RUN_IDS[0]]
    if any(value != first for value in normalized_by_run.values()):
        raise FormalQueueError("completed D2m/D3m runs do not share exact common inputs")
    return _input_identity_sha256(first), full_hashes


def _build_completion_payload(queue_dir: Path) -> dict[str, Any]:
    queue_dir = Path(queue_dir).expanduser().resolve(strict=True)
    source, scope = _load_plans(queue_dir)
    generic = generic_queue.verify_queue(queue_dir)
    if generic.get("status") != "passed":
        raise FormalQueueError("generic shared-lease queue is not completed/verified")
    reports: dict[str, Any] = {}
    for run_id in contract.FORMAL_RUN_IDS:
        table_b_id, raw_seed = run_id.split(":", 1)
        run_root = (
            Path(source["runtime"]["output_root"])
            / table_b_id
            / f"seed{int(raw_seed)}"
        )
        reports[run_id] = v2_runner.verify_completed_run(
            run_root,
            training_queue_dir=queue_dir,
            require_queue=True,
            require_formal=True,
        )
    common_sha, actual_hashes = _normalized_actual_common_inputs(queue_dir, source)
    if any(
        actual_hashes[run_id] != scope["runs"][run_id]["input_identity_sha256"]
        for run_id in contract.FORMAL_RUN_IDS
    ):
        raise FormalQueueError("completed input identities differ from the scope plan")
    payload: dict[str, Any] = {
        "schema": COMPLETION_SCHEMA,
        "status": "passed",
        "profile": contract.FORMAL_PROFILE,
        "ordered_run_ids": list(contract.FORMAL_RUN_IDS),
        "queue": dict(source["queue"]),
        "source_plan": contract.file_record(
            queue_dir / SOURCE_PLAN_NAME, role=FORMAL_SOURCE_ROLE
        ),
        "scope_plan": contract.file_record(
            queue_dir / SCOPE_PLAN_NAME, role="table_b_v2_formal_scope_plan"
        ),
        "common_input_replay": {
            "status": "passed",
            "all_six_runs_share_identical_common_inputs": True,
            "only_declared_condition_inputs_differ": True,
            "common_input_identity_sha256": common_sha,
            "per_run_input_identity_sha256": actual_hashes,
        },
        "runs": reports,
    }
    payload["semantic_sha256"] = _semantic_sha256(payload)
    return payload


def verify_formal_queue(
    queue_dir: Path, *, persist: bool = True
) -> dict[str, Any]:
    queue_dir = Path(queue_dir).expanduser().resolve(strict=True)
    try:
        payload = _build_completion_payload(queue_dir)
    except FormalQueueError:
        raise
    except (
        OSError,
        ValueError,
        generic_queue.QueueContractError,
        contract.TableBContractError,
        v2_runner.TableBV2RunnerError,
    ) as error:
        raise FormalQueueError(
            f"formal Table-B v2 completion replay failed: {error}"
        ) from error
    path = queue_dir / COMPLETION_NAME
    if path.exists():
        existing = _read_json(path, label="formal completion attestation")
        if existing != payload:
            raise FormalQueueError("persisted formal completion attestation drifted")
    elif persist:
        _write_json_atomic(path, payload)
    return payload


def formal_evaluation_evidence(
    queue_dir: Path, *, run_id: str, run_root: Path
) -> dict[str, Any]:
    queue_dir = Path(queue_dir).expanduser().resolve(strict=True)
    attestation_path = (queue_dir / COMPLETION_NAME).resolve(strict=True)
    persisted = _read_json(attestation_path, label="formal completion attestation")
    replayed = verify_formal_queue(queue_dir, persist=False)
    if persisted != replayed or run_id not in persisted["runs"]:
        raise FormalQueueError("formal completion attestation replay drifted")
    receipt = persisted["runs"][run_id]
    if Path(receipt["run_root"]).resolve(strict=True) != Path(run_root).resolve(
        strict=True
    ):
        raise FormalQueueError("formal evaluation run root differs from attestation")
    return {
        "profile": contract.FORMAL_PROFILE,
        "run_id": run_id,
        "queue_id": persisted["queue"]["queue_id"],
        "queue_plan_sha256": persisted["queue"]["plan_sha256"],
        "completion_semantic_sha256": persisted["semantic_sha256"],
        "source_plan": persisted["source_plan"],
        "scope_plan": persisted["scope_plan"],
        "completion_attestation": contract.file_record(
            attestation_path, role="table_b_v2_formal_completion_attestation"
        ),
        "run_verification": receipt,
    }


def queue_status(queue_dir: Path) -> dict[str, Any]:
    source, scope = _load_plans(queue_dir)
    return {
        "schema": "pivot.stageb.table_b_v2_formal_queue_status/v1",
        "profile": contract.FORMAL_PROFILE,
        "source_plan_semantic_sha256": source["semantic_sha256"],
        "scope_plan_semantic_sha256": scope["semantic_sha256"],
        "generic_queue": generic_queue.queue_status(queue_dir),
        "completion_attestation_present": (
            Path(queue_dir) / COMPLETION_NAME
        ).is_file(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("queue_dir", type=Path)
    create.add_argument("--output-root", type=Path, required=True)
    create.add_argument("--runner-python", type=Path, default=Path(sys.executable))
    create.add_argument("--token-runner", type=Path, default=generic_queue.DEFAULT_TOKEN_RUNNER)
    create.add_argument("--lease-root", type=Path, default=generic_queue.DEFAULT_LEASE_ROOT)
    create.add_argument("--gpu-key")
    run = subparsers.add_parser("run")
    run.add_argument("queue_dir", type=Path)
    run.add_argument("--poll-seconds", type=float, default=30.0)
    run.add_argument("--once", action="store_true")
    reconcile = subparsers.add_parser("reconcile")
    reconcile.add_argument("queue_dir", type=Path)
    status = subparsers.add_parser("status")
    status.add_argument("queue_dir", type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("queue_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.mode == "create":
            report = create_formal_queue(
                args.queue_dir,
                output_root=args.output_root,
                runner_python=args.runner_python,
                token_runner=args.token_runner,
                lease_root=args.lease_root,
                gpu_key=args.gpu_key,
            )
        elif args.mode == "run":
            queue = run_formal_queue(
                args.queue_dir,
                poll_seconds=args.poll_seconds,
                once=args.once,
            )
            report = queue_status(args.queue_dir)
            if queue["status"] == "failed":
                print(json.dumps(report, indent=2, sort_keys=True))
                return 1
        elif args.mode == "reconcile":
            run_formal_queue(args.queue_dir, poll_seconds=1.0, once=True)
            report = queue_status(args.queue_dir)
        elif args.mode == "status":
            report = queue_status(args.queue_dir)
        elif args.mode == "verify":
            report = verify_formal_queue(args.queue_dir, persist=True)
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
        ValueError,
        FormalQueueError,
        generic_queue.QueueContractError,
        contract.TableBContractError,
        v2_runner.TableBV2RunnerError,
    ) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
