#!/usr/bin/env python3
"""Run explicit Stage-B ablation run IDs through one fail-closed GPU queue.

The queue is a durable control plane around the existing token and paper
matrix launchers.  It never invokes ``main.py`` itself.  Every queue item is
one existing-launcher ``detach --run-id`` job, and the next item is released
only after the detached job is no longer alive and its sequence plus every
phase postflight are explicitly complete.

The mutable queue state is stored in ``QUEUE_DIR/queue.json``.  Its immutable
plan (ordered run IDs, launcher hashes, selected GPU, and relevant runtime
environment) is embedded in that file and protected by a canonical SHA-256.
A durable per-GPU lease prevents another queue created by this tool from
launching concurrently.  Interrupted supervisors can be restarted with the
same ``run QUEUE_DIR`` command; detached launcher jobs remain authoritative.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import signal
import shlex
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKEN_RUNNER = REPO_ROOT / "tools/run_stageb_token_ablation_matrix.py"
DEFAULT_PAPER_RUNNER = REPO_ROOT / "tools/run_stageb_paper_ablation_matrices.py"
DEFAULT_LEASE_ROOT = (
    REPO_ROOT / "outputs/paper_cvpr_v1/orchestration/serial_matrix_queue/leases"
)

QUEUE_SCHEMA = "pivot.stageb.serial_matrix_queue/v1"
PLAN_SCHEMA = "pivot.stageb.serial_matrix_queue_plan/v1"
LEASE_SCHEMA = "pivot.stageb.serial_matrix_gpu_lease/v1"
COMPLETION_SCHEMA = "pivot.stageb.serial_matrix_completion_evidence/v1"

ITEM_STATUSES = frozenset(
    {"pending", "reserved", "launching", "launched", "completed", "failed"}
)
RUNNER_NONTERMINAL_STATUSES = frozenset(
    {"prepared", "launched", "starting", "preflight_passed", "running"}
)
RUNNER_FAILURE_STATUSES = frozenset(
    {"failed", "spawn_failed", "hard_terminated_unknown"}
)

# Only runtime controls consumed by the two sealed launchers are persisted.
# This deliberately excludes arbitrary environment values and secrets.
RUNTIME_ENV_KEYS = (
    "PIVOT_PYTHON",
    "PIVOT_STAGE_A_INIT",
    "PIVOT_SCORER_WARMSTART",
    "PIVOT_BATCH_SIZE",
    "PIVOT_MAX_TRAIN_ITERS",
    "PIVOT_ITER_CHECKPOINT_INTERVAL",
    "PIVOT_NUM_WORKERS",
    "PIVOT_PREFETCH_FACTOR",
    "PIVOT_OMP_NUM_THREADS",
    "PIVOT_MIN_NOFILE",
    "PIVOT_CUDA_VISIBLE_DEVICES",
    "PIVOT_DATA_ROOT",
    "PIVOT_TOKEN_DATASETS",
    "PIVOT_TOKEN_OUTPUT_ROOT",
    "PIVOT_TN_OUTPUT_ROOT",
    "PIVOT_SCORE_OUTPUT_ROOT",
    "PIVOT_MP_SHARING_STRATEGY",
    "PIVOT_GRADIENT_DIAGNOSTIC_INTERVAL",
    "DATA_ROOT",
    "CUDA_VISIBLE_DEVICES",
)

# Keep locally spawned detach launchers reachable until they exit.  The actual
# training orchestrator is started in a new session by the existing runner;
# this registry only prevents unreaped short-lived launcher processes.
_LOCAL_DETACH_LAUNCHERS: dict[int, subprocess.Popen[Any]] = {}
CHILD_TERMINATION_GRACE_SECONDS = 5.0
CHILD_TERMINATION_POLL_SECONDS = 0.05


class QueueContractError(RuntimeError):
    """The persisted queue or child evidence violates a required contract."""


class QueueBusyError(QueueContractError):
    """Another supervisor or queue currently owns the requested resource."""


class QueueLeaseOwnershipError(QueueBusyError):
    """The durable GPU lease exists but belongs to a different queue."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise QueueContractError(f"{description} is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise QueueContractError(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QueueContractError(f"{description} must be a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
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


def _resolve_executable(value: Path) -> Path:
    path = value.expanduser().resolve(strict=True)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise QueueContractError(f"runner Python is not executable: {path}")
    return path


def _runner_environment(snapshot: Mapping[str, Any]) -> dict[str, str]:
    environment = dict(os.environ)
    for key in (*RUNTIME_ENV_KEYS, "PIVOT_ORCHESTRATION_ROOT", "PIVOT_ORCHESTRATION_STATUS"):
        environment.pop(key, None)
    for key in RUNTIME_ENV_KEYS:
        value = snapshot.get(key)
        if value is not None:
            if not isinstance(value, str):
                raise QueueContractError(f"runtime environment {key} is not a string/null")
            environment[key] = value
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def _parse_json_stdout(
    result: subprocess.CompletedProcess[str], *, description: str
) -> dict[str, Any]:
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise QueueContractError(
            f"{description} exited {result.returncode}: {detail[-4000:]}"
        )
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise QueueContractError(
            f"{description} did not emit one JSON object: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise QueueContractError(f"{description} JSON is not an object")
    return value


def _run_json_command(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    description: str,
    timeout: float = 120.0,
) -> dict[str, Any]:
    try:
        result = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            env=dict(environment),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise QueueContractError(f"cannot execute {description}: {exc}") from exc
    return _parse_json_stdout(result, description=description)


def _runner_inventory(
    python: Path, runner: Path, environment: Mapping[str, str]
) -> tuple[str, ...]:
    payload = _run_json_command(
        [str(python), str(runner), "list", "--json"],
        environment=environment,
        description=f"{runner.name} list",
    )
    raw = payload.get("run_ids")
    if not isinstance(raw, list) or not raw:
        raise QueueContractError(f"{runner} list returned no run_ids")
    values: list[str] = []
    for value in raw:
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9]+:[0-9]+", value):
            raise QueueContractError(f"{runner} returned invalid run_id {value!r}")
        values.append(value)
    if len(values) != len(set(value.casefold() for value in values)):
        raise QueueContractError(f"{runner} returned duplicate run IDs")
    return tuple(values)


def _gpu_key_from_environment(snapshot: Mapping[str, Any], requested: str | None) -> str:
    visible = snapshot.get("PIVOT_CUDA_VISIBLE_DEVICES")
    if visible is None:
        visible = snapshot.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        visible = "0"
    if not isinstance(visible, str) or not visible.strip():
        raise QueueContractError("the selected CUDA device must be one non-empty value")
    visible = visible.strip()
    if "," in visible:
        raise QueueContractError(
            "serial queue requires exactly one CUDA-visible device, got " + visible
        )
    gpu_key = visible if requested is None else requested.strip()
    if not gpu_key or "," in gpu_key:
        raise QueueContractError("--gpu-key must identify exactly one GPU")
    if requested is not None and gpu_key != visible:
        raise QueueContractError(
            f"--gpu-key {gpu_key!r} differs from captured CUDA device {visible!r}"
        )
    return gpu_key


def _lease_path(lease_root: Path, gpu_key: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", gpu_key).strip("._-") or "gpu"
    digest = _sha256_bytes(gpu_key.encode("utf-8"))[:12]
    return lease_root / f"gpu-{slug[:40]}-{digest}.lease.json"


def _snapshot_environment() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in RUNTIME_ENV_KEYS}


def create_queue(
    queue_dir: Path,
    *,
    run_ids: Sequence[str],
    runner_python: Path,
    token_runner: Path,
    paper_runner: Path,
    lease_root: Path,
    gpu_key: str | None,
    plan_extensions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not run_ids:
        raise QueueContractError("create requires at least one explicit --run-id")
    queue_dir = queue_dir.expanduser().resolve(strict=False)
    if queue_dir.exists():
        raise FileExistsError(f"queue directory must be fresh: {queue_dir}")
    runner_python = _resolve_executable(runner_python)
    token_runner = token_runner.expanduser().resolve(strict=True)
    paper_runner = paper_runner.expanduser().resolve(strict=True)
    for label, path in (("token", token_runner), ("paper", paper_runner)):
        if not path.is_file():
            raise QueueContractError(f"{label} runner is not a file: {path}")

    snapshot = _snapshot_environment()
    selected_gpu = _gpu_key_from_environment(snapshot, gpu_key)
    environment = _runner_environment(snapshot)
    inventories = {
        "token": _runner_inventory(runner_python, token_runner, environment),
        "paper": _runner_inventory(runner_python, paper_runner, environment),
    }
    routes: dict[str, tuple[str, str]] = {}
    for runner_name, values in inventories.items():
        for canonical in values:
            folded = canonical.casefold()
            if folded in routes:
                raise QueueContractError(
                    f"run_id {canonical} is ambiguous across runner inventories"
                )
            routes[folded] = (runner_name, canonical)

    selections: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in run_ids:
        route = routes.get(raw.casefold())
        if route is None:
            raise QueueContractError(f"run_id is not registered by either runner: {raw}")
        runner_name, canonical = route
        if canonical.casefold() in seen:
            raise QueueContractError(f"duplicate queue run_id: {canonical}")
        seen.add(canonical.casefold())
        selections.append({"run_id": canonical, "runner": runner_name})

    lease_root = lease_root.expanduser().resolve(strict=False)
    runners = {
        "token": {
            "path": str(token_runner),
            "sha256": _sha256_file(token_runner),
            "supported_run_ids": list(inventories["token"]),
        },
        "paper": {
            "path": str(paper_runner),
            "sha256": _sha256_file(paper_runner),
            "supported_run_ids": list(inventories["paper"]),
        },
    }
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "queue_id": str(uuid.uuid4()),
        "created_at_utc": _utc_now(),
        "queue_dir": str(queue_dir),
        "repository_root": str(REPO_ROOT),
        "runner_python": str(runner_python),
        "runners": runners,
        "runtime_environment": snapshot,
        "gpu_key": selected_gpu,
        "lease_root": str(lease_root),
        "lease_path": str(_lease_path(lease_root, selected_gpu)),
        "items": selections,
    }
    if plan_extensions is not None:
        if not isinstance(plan_extensions, Mapping) or not plan_extensions:
            raise QueueContractError("plan_extensions must be a non-empty mapping")
        try:
            normalized_extensions = json.loads(
                json.dumps(
                    dict(plan_extensions),
                    sort_keys=True,
                    ensure_ascii=True,
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError) as exc:
            raise QueueContractError(
                f"plan_extensions is not canonical JSON: {exc}"
            ) from exc
        if not isinstance(normalized_extensions, dict):
            raise QueueContractError("plan_extensions did not normalize to an object")
        plan["extensions"] = normalized_extensions
    plan_sha256 = _sha256_bytes(_canonical_json_bytes(plan))
    now = _utc_now()
    queue: dict[str, Any] = {
        "schema": QUEUE_SCHEMA,
        "status": "planned",
        "created_at_utc": now,
        "updated_at_utc": now,
        "revision": 0,
        "plan": plan,
        "plan_sha256": plan_sha256,
        "items": [
            {
                "index": index,
                "run_id": item["run_id"],
                "runner": item["runner"],
                "status": "pending",
            }
            for index, item in enumerate(selections)
        ],
        "events": [
            {
                "at_utc": now,
                "event": "queue_created",
                "ordered_run_ids": [item["run_id"] for item in selections],
            }
        ],
    }
    queue_dir.mkdir(parents=True, exist_ok=False)
    _write_json_atomic(queue_dir / "queue.json", queue)
    return queue


def _validate_queue(queue: Mapping[str, Any], queue_dir: Path) -> None:
    if queue.get("schema") != QUEUE_SCHEMA:
        raise QueueContractError(f"unsupported queue schema: {queue.get('schema')!r}")
    plan = queue.get("plan")
    if not isinstance(plan, Mapping) or plan.get("schema") != PLAN_SCHEMA:
        raise QueueContractError("queue has no valid immutable plan")
    expected_hash = _sha256_bytes(_canonical_json_bytes(plan))
    if queue.get("plan_sha256") != expected_hash:
        raise QueueContractError("immutable queue plan SHA-256 mismatch")
    if Path(str(plan.get("queue_dir"))).resolve(strict=False) != queue_dir:
        raise QueueContractError("queue was opened through a path different from its plan")
    if queue.get("status") not in {"planned", "running", "completed", "failed"}:
        raise QueueContractError(f"invalid queue status: {queue.get('status')!r}")
    plan_items = plan.get("items")
    items = queue.get("items")
    if not isinstance(plan_items, list) or not isinstance(items, list):
        raise QueueContractError("queue items are missing")
    if len(plan_items) != len(items) or not items:
        raise QueueContractError("mutable items do not match immutable queue plan")

    active_count = 0
    completed_prefix = True
    failed_count = 0
    for index, (planned, item) in enumerate(zip(plan_items, items)):
        if not isinstance(planned, Mapping) or not isinstance(item, Mapping):
            raise QueueContractError("queue item is not an object")
        for key in ("run_id", "runner"):
            if item.get(key) != planned.get(key):
                raise QueueContractError(f"item {index} changed immutable field {key}")
        if item.get("index") != index:
            raise QueueContractError(f"item {index} has invalid index")
        status = item.get("status")
        if status not in ITEM_STATUSES:
            raise QueueContractError(f"item {index} has invalid status {status!r}")
        if status == "completed":
            if not completed_prefix:
                raise QueueContractError("completed queue items must form one prefix")
        else:
            completed_prefix = False
        if status in {"reserved", "launching", "launched", "failed"}:
            active_count += 1
        if status == "failed":
            failed_count += 1
        if status == "pending" and any(
            later.get("status") != "pending" for later in items[index + 1 :]
        ):
            raise QueueContractError("a later item advanced past a pending predecessor")
    if active_count > 1 or failed_count > 1:
        raise QueueContractError("queue contains more than one active/failed item")
    status = queue["status"]
    if status == "planned" and any(item["status"] != "pending" for item in items):
        raise QueueContractError("planned queue contains a started item")
    if status == "completed" and any(item["status"] != "completed" for item in items):
        raise QueueContractError("completed queue contains an incomplete item")
    if status == "failed" and failed_count != 1:
        raise QueueContractError("failed queue has no single failed item")
    if status == "running" and failed_count:
        raise QueueContractError("running queue contains a failed item")


def load_queue(queue_dir: Path) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    if not queue_dir.is_dir():
        raise NotADirectoryError(f"queue path is not a directory: {queue_dir}")
    queue = _json_object(queue_dir / "queue.json", description="queue state")
    _validate_queue(queue, queue_dir)
    return queue


def _save_queue(queue: MutableMapping[str, Any]) -> None:
    queue_dir = Path(str(queue["plan"]["queue_dir"])).resolve(strict=True)
    queue["revision"] = int(queue.get("revision", 0)) + 1
    queue["updated_at_utc"] = _utc_now()
    _validate_queue(queue, queue_dir)
    _write_json_atomic(queue_dir / "queue.json", queue)


def _event(queue: MutableMapping[str, Any], name: str, **fields: Any) -> None:
    events = queue.setdefault("events", [])
    if not isinstance(events, list):
        raise QueueContractError("queue events field is not a list")
    events.append({"at_utc": _utc_now(), "event": name, **fields})


@contextlib.contextmanager
def _exclusive_file_lock(path: Path, *, busy_message: str) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise QueueBusyError(busy_message) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _lease_record(queue: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, Any]:
    plan = queue["plan"]
    return {
        "schema": LEASE_SCHEMA,
        "status": "owned",
        "created_at_utc": _utc_now(),
        "queue_id": plan["queue_id"],
        "queue_dir": plan["queue_dir"],
        "plan_sha256": queue["plan_sha256"],
        "gpu_key": plan["gpu_key"],
        "first_run_id": item["run_id"],
        "policy": "retained_across_items_until_verified_queue_completion",
    }


def _lease_identity_mismatches(
    queue: Mapping[str, Any], lease: Mapping[str, Any]
) -> dict[str, Any]:
    first_item = queue["plan"].get("items", [None])[0]
    first_run_id = (
        first_item.get("run_id") if isinstance(first_item, Mapping) else None
    )
    expected = {
        "schema": LEASE_SCHEMA,
        "status": "owned",
        "queue_id": queue["plan"]["queue_id"],
        "queue_dir": queue["plan"]["queue_dir"],
        "plan_sha256": queue["plan_sha256"],
        "gpu_key": queue["plan"]["gpu_key"],
        "first_run_id": first_run_id,
        "policy": "retained_across_items_until_verified_queue_completion",
    }
    mismatches = {
        key: {"expected": value, "observed": lease.get(key)}
        for key, value in expected.items()
        if lease.get(key) != value
    }
    expected_keys = {*expected, "created_at_utc"}
    if set(lease) != expected_keys:
        mismatches["keys"] = {
            "expected": sorted(expected_keys),
            "observed": sorted(str(key) for key in lease),
        }
    created_at = lease.get("created_at_utc")
    try:
        parsed_created_at = datetime.fromisoformat(created_at)
    except (TypeError, ValueError):
        parsed_created_at = None
    if parsed_created_at is None or parsed_created_at.tzinfo is None:
        mismatches["created_at_utc"] = {
            "expected": "ISO-8601 timezone-aware timestamp",
            "observed": created_at,
        }
    return mismatches


def _ensure_lease(queue: Mapping[str, Any], item: Mapping[str, Any], *, create: bool) -> None:
    lease_path = Path(str(queue["plan"]["lease_path"]))
    lock_path = lease_path.with_suffix(lease_path.suffix + ".lock")
    with _exclusive_file_lock(lock_path, busy_message=f"GPU lease lock is busy: {lock_path}"):
        if lease_path.exists():
            lease = _json_object(lease_path, description="GPU lease")
            mismatches = _lease_identity_mismatches(queue, lease)
            if mismatches:
                raise QueueLeaseOwnershipError(
                    f"GPU {queue['plan']['gpu_key']} is leased by another queue: "
                    f"{lease_path}; mismatches={mismatches}"
                )
            return
        if not create:
            raise QueueContractError(
                f"active queue lost its durable GPU lease: {lease_path}"
            )
        _write_json_atomic(lease_path, _lease_record(queue, item))


def _clear_owned_lease(queue: Mapping[str, Any]) -> None:
    lease_path = Path(str(queue["plan"]["lease_path"]))
    lock_path = lease_path.with_suffix(lease_path.suffix + ".lock")
    with _exclusive_file_lock(lock_path, busy_message=f"GPU lease lock is busy: {lock_path}"):
        if not lease_path.exists():
            return
        lease = _json_object(lease_path, description="GPU lease")
        if _lease_identity_mismatches(queue, lease):
            raise QueueLeaseOwnershipError(
                f"refusing to clear a lease owned by another queue: {lease_path}"
            )
        lease_path.unlink()


def _runner_record(queue: Mapping[str, Any], item: Mapping[str, Any]) -> Mapping[str, Any]:
    runners = queue["plan"].get("runners")
    if not isinstance(runners, Mapping):
        raise QueueContractError("queue plan has no runner records")
    record = runners.get(item["runner"])
    if not isinstance(record, Mapping):
        raise QueueContractError(f"queue item names unknown runner {item['runner']!r}")
    return record


def _verify_runner_source(queue: Mapping[str, Any], item: Mapping[str, Any]) -> Path:
    record = _runner_record(queue, item)
    path = Path(str(record.get("path", ""))).resolve(strict=True)
    observed = _sha256_file(path)
    if observed != record.get("sha256"):
        raise QueueContractError(
            f"runner source changed after queue creation: {path}; "
            f"expected {record.get('sha256')}, got {observed}"
        )
    supported = record.get("supported_run_ids")
    if not isinstance(supported, list) or item["run_id"] not in supported:
        raise QueueContractError(f"runner plan does not register {item['run_id']}")
    return path


def _read_process_identity(pid: int) -> dict[str, Any]:
    result: dict[str, Any] = {"pid": pid, "available": False}
    try:
        stat_text = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
        right = stat_text.rfind(")")
        fields = stat_text[right + 2 :].split()
        if right < 0 or len(fields) < 20:
            raise ValueError("unexpected /proc stat layout")
        result.update(
            {
                "available": True,
                "state": fields[0],
                "start_time_ticks": int(fields[19]),
            }
        )
        boot = Path("/proc/sys/kernel/random/boot_id")
        if boot.is_file():
            result["boot_id"] = boot.read_text(encoding="utf-8").strip()
    except (OSError, ValueError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _process_running(pid: Any, expected: Any = None) -> bool | None:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    local = _LOCAL_DETACH_LAUNCHERS.get(pid)
    if local is not None:
        if local.poll() is None:
            return True
        _LOCAL_DETACH_LAUNCHERS.pop(pid, None)
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return None
    observed = _read_process_identity(pid)
    if observed.get("available") and observed.get("state") == "Z":
        return False
    if isinstance(expected, Mapping) and expected.get("available"):
        if not observed.get("available"):
            return None
        if expected.get("start_time_ticks") != observed.get("start_time_ticks"):
            return False
        expected_boot = expected.get("boot_id")
        if expected_boot and observed.get("boot_id") and expected_boot != observed.get("boot_id"):
            return False
    return True


def _validated_child_identity(pid: Any, identity: Any) -> dict[str, Any]:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise QueueContractError("bound child PID is invalid")
    if not isinstance(identity, Mapping):
        raise QueueContractError("bound child process identity is missing")
    start_time = identity.get("start_time_ticks")
    boot_id = identity.get("boot_id")
    if (
        identity.get("pid") != pid
        or identity.get("available") is not True
        or isinstance(start_time, bool)
        or not isinstance(start_time, int)
        or start_time <= 0
        or not isinstance(boot_id, str)
        or not boot_id.strip()
    ):
        raise QueueContractError(
            "bound child identity lacks exact PID/start-time/boot binding"
        )
    return dict(identity)


def _process_group_exists(process_group_id: int) -> bool | None:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return None
    return True


def _wait_for_bound_group_exit(
    pid: int,
    identity: Mapping[str, Any],
    *,
    timeout_seconds: float,
) -> bool | None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        running = _process_running(pid, identity)
        group_exists = _process_group_exists(pid)
        if running is False and group_exists is False:
            return True
        if running is None or group_exists is None:
            return None
        if time.monotonic() >= deadline:
            return False
        time.sleep(CHILD_TERMINATION_POLL_SECONDS)


def _terminate_exact_process_group(
    pid: Any,
    identity: Any,
    *,
    label: str,
    resume_stopped_group: bool = False,
) -> dict[str, Any]:
    expected = _validated_child_identity(pid, identity)
    running = _process_running(pid, expected)
    if running is None:
        raise QueueContractError(
            f"{label} identity is unobservable during lease-loss shutdown"
        )
    if running is False:
        if _process_group_exists(pid) is not False:
            raise QueueContractError(
                f"{label} leader exited but its exact process group is not gone"
            )
        return {
            "status": "already_exited",
            "pid": pid,
            "process_group_id": pid,
            "terminated_at_utc": _utc_now(),
        }
    observed = _validated_child_identity(pid, _read_process_identity(pid))
    if (
        observed["start_time_ticks"] != expected["start_time_ticks"]
        or observed["boot_id"] != expected["boot_id"]
    ):
        raise QueueContractError(
            f"{label} identity changed before process-group termination"
        )
    try:
        process_group_id = os.getpgid(pid)
        session_id = os.getsid(pid)
    except ProcessLookupError as exc:
        raise QueueContractError(
            f"{label} disappeared before its process group was proven gone"
        ) from exc
    if process_group_id != pid or session_id != pid:
        raise QueueContractError(
            f"{label} is not its sealed session/process-group leader"
        )
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process_group_id, signal.SIGTERM)
    if resume_stopped_group:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process_group_id, signal.SIGCONT)
    exited = _wait_for_bound_group_exit(
        pid,
        expected,
        timeout_seconds=CHILD_TERMINATION_GRACE_SECONDS,
    )
    escalated = False
    if exited is not True:
        if exited is None:
            raise QueueContractError(
                f"{label} shutdown became unobservable after SIGTERM"
            )
        before_kill = _read_process_identity(pid)
        if before_kill.get("available") is True and (
            before_kill.get("start_time_ticks") != expected["start_time_ticks"]
            or before_kill.get("boot_id") != expected["boot_id"]
        ):
            raise QueueContractError(
                f"refusing SIGKILL because the {label} PID was reused"
            )
        escalated = True
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process_group_id, signal.SIGKILL)
        exited = _wait_for_bound_group_exit(
            pid,
            expected,
            timeout_seconds=CHILD_TERMINATION_GRACE_SECONDS,
        )
    if exited is not True:
        raise QueueContractError(
            f"{label} process group was not proven gone after termination"
        )
    return {
        "status": "terminated",
        "pid": pid,
        "process_group_id": process_group_id,
        "session_id": session_id,
        "signal": "SIGKILL" if escalated else "SIGTERM",
        "terminated_at_utc": _utc_now(),
    }


def _process_relationships() -> dict[int, dict[str, int | str]]:
    proc = Path("/proc")
    if not proc.is_dir():
        raise QueueContractError("process topology is unavailable")
    relationships: dict[int, dict[str, int | str]] = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat_text = (entry / "stat").read_text(encoding="utf-8")
            right = stat_text.rfind(")")
            fields = stat_text[right + 2 :].split()
            if right < 0 or len(fields) < 4:
                continue
            relationships[int(entry.name)] = {
                "state": fields[0],
                "parent_pid": int(fields[1]),
                "process_group_id": int(fields[2]),
                "session_id": int(fields[3]),
            }
        except (OSError, ValueError):
            continue
    return relationships


def _stop_and_prove_spawn_window_group_closed(
    pid: int,
    identity: Mapping[str, Any],
    status_path: Path,
    launch_epoch_identity: Mapping[str, Any] | None = None,
) -> list[int]:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGSTOP)
    deadline = time.monotonic() + CHILD_TERMINATION_GRACE_SECONDS
    while True:
        observed = _read_process_identity(pid)
        if observed.get("state") in {"T", "t"}:
            break
        if _process_running(pid, identity) is not True:
            raise QueueContractError(
                "spawn-window child could not be stopped for topology proof"
            )
        if time.monotonic() >= deadline:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGCONT)
            raise QueueContractError(
                "spawn-window child did not stop for topology proof"
            )
        time.sleep(CHILD_TERMINATION_POLL_SECONDS)
    try:
        _sealed_session_leader_identity(pid, identity)
        relationships = _process_relationships()
        if pid not in relationships:
            raise QueueContractError(
                "spawn-window child vanished from the process topology"
            )
        descendants: set[int] = set()
        frontier = [pid]
        while frontier:
            parent = frontier.pop()
            children = [
                candidate
                for candidate, relation in relationships.items()
                if relation["parent_pid"] == parent
                and candidate not in descendants
            ]
            descendants.update(children)
            frontier.extend(children)
        escaped = sorted(
            descendant
            for descendant in descendants
            if relationships[descendant]["process_group_id"] != pid
            or relationships[descendant]["session_id"] != pid
        )
        if escaped:
            raise QueueContractError(
                "spawn-window child has descendants outside its sealed group"
            )
        marker_processes: set[int] = set()
        expected_status = str(status_path.resolve(strict=True))
        launch_epoch_start = identity.get("start_time_ticks")
        if isinstance(launch_epoch_identity, Mapping):
            candidate_start = launch_epoch_identity.get("start_time_ticks")
            if (
                isinstance(candidate_start, int)
                and not isinstance(candidate_start, bool)
                and candidate_start > 0
                and launch_epoch_identity.get("boot_id") == identity.get("boot_id")
            ):
                launch_epoch_start = candidate_start
        for candidate in relationships:
            try:
                owner_uid = (Path("/proc") / str(candidate)).stat().st_uid
            except OSError:
                continue
            if owner_uid != os.geteuid():
                continue
            candidate_identity = _read_process_identity(candidate)
            candidate_start = candidate_identity.get("start_time_ticks")
            if (
                isinstance(launch_epoch_start, int)
                and isinstance(candidate_start, int)
                and not isinstance(candidate_start, bool)
                and candidate_start < launch_epoch_start
            ):
                continue
            running = _process_running(candidate, candidate_identity)
            if running is False:
                continue
            if running is None:
                raise QueueContractError(
                    "same-host process liveness is unobservable during marker scan"
                )
            try:
                environment = _process_environment(candidate)
            except UnicodeDecodeError:
                environment = None
            if environment is None:
                if _process_running(candidate, candidate_identity) is False:
                    continue
                try:
                    still_same_user = (
                        (Path("/proc") / str(candidate)).stat().st_uid
                        == owner_uid
                    )
                except OSError:
                    continue
                if still_same_user:
                    raise QueueContractError(
                        "same-user process environment is unreadable during "
                        "job-marker topology proof"
                    )
                continue
            if environment.get("PIVOT_ORCHESTRATION_STATUS") != expected_status:
                continue
            try:
                candidate_group = os.getpgid(candidate)
                candidate_session = os.getsid(candidate)
            except ProcessLookupError:
                continue
            marker_processes.add(candidate)
            if candidate_group != pid or candidate_session != pid:
                raise QueueContractError(
                    "spawn-window job marker escaped the sealed child group"
                )
        if pid not in marker_processes:
            raise QueueContractError(
                "sealed child lost its exact job-status environment binding"
            )
        return sorted(descendants | marker_processes - {pid})
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGCONT)
        raise


def _terminate_bound_child_process_group(
    queue: Mapping[str, Any], index: int
) -> dict[str, Any]:
    item = queue["items"][index]
    pid = item.get("child_pid")
    expected_group = item.get("child_process_group_id", pid)
    expected_session = item.get("child_session_id", pid)
    if expected_group != pid or expected_session != pid:
        raise QueueContractError(
            "bound child durable PGID/SID differs from its sealed PID"
        )
    identity = item.get("child_process_identity")
    stopped_for_topology = bool(item.get("spawn_window_bound_at_utc"))
    descendants: list[int] = []
    if stopped_for_topology:
        raw_status_path = item.get("spawn_window_status_path")
        if not isinstance(raw_status_path, str) or not raw_status_path:
            raise QueueContractError(
                "spawn-window child lacks its durable job-status binding"
            )
        descendants = _stop_and_prove_spawn_window_group_closed(
            pid,
            _validated_child_identity(pid, identity),
            Path(raw_status_path),
            item.get("detach_launcher_identity"),
        )
    try:
        termination = _terminate_exact_process_group(
            pid,
            identity,
            label="bound child",
            resume_stopped_group=stopped_for_topology,
        )
    except BaseException:
        if stopped_for_topology:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGCONT)
        raise
    termination["sealed_descendant_pids"] = descendants
    return termination


def _terminate_detach_launcher_process_group(
    queue: Mapping[str, Any], index: int
) -> dict[str, Any]:
    item = queue["items"][index]
    pid = item.get("detach_launcher_pid")
    identity = item.get("detach_launcher_identity")
    running = _process_running(pid, identity)
    if running is False and isinstance(pid, int):
        if _process_group_exists(pid) is False:
            return {
                "status": "already_exited",
                "pid": pid,
                "process_group_id": pid,
                "terminated_at_utc": _utc_now(),
            }
    return _terminate_exact_process_group(
        pid,
        identity,
        label="detach launcher",
    )


def _launch_record_group_is_gone(launch: Mapping[str, Any]) -> bool:
    pid = launch.get("child_pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    if _process_running(pid, launch.get("child_process_identity")) is not False:
        return False
    return _process_group_exists(pid) is False


def _item_orchestration_root(queue: Mapping[str, Any], item: Mapping[str, Any]) -> Path:
    raw = item.get("orchestration_root")
    if isinstance(raw, str):
        return Path(raw).resolve(strict=False)
    queue_dir = Path(str(queue["plan"]["queue_dir"]))
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", item["run_id"])
    return (queue_dir / "jobs" / f"{int(item['index']):03d}-{slug}").resolve(
        strict=False
    )


def _detached_job_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        child.resolve()
        for child in root.iterdir()
        if child.is_dir()
        and (child / "launch.json").is_file()
        and (child / "status.json").is_file()
    )


def _matching_detach_launchers(
    *, runner: Path, root: Path, run_id: str
) -> list[tuple[int, dict[str, Any]]]:
    matches: list[tuple[int, dict[str, Any]]] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return matches
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
            values = [value.decode("utf-8", errors="replace") for value in args if value]
        except OSError:
            continue
        required = {str(runner), "detach", "--orchestration-root", str(root), "--run-id", run_id}
        if required.issubset(set(values)):
            identity = _read_process_identity(pid)
            if _process_running(pid, identity) is True:
                matches.append((pid, identity))
    return matches


def _reserve_next(queue: MutableMapping[str, Any], index: int) -> None:
    item = queue["items"][index]
    _verify_runner_source(queue, item)
    _ensure_lease(queue, item, create=True)
    root = _item_orchestration_root(queue, item)
    if root.exists():
        raise QueueContractError(f"fresh item orchestration root already exists: {root}")
    item["status"] = "reserved"
    item["reserved_at_utc"] = _utc_now()
    item["orchestration_root"] = str(root)
    queue["status"] = "running"
    _event(queue, "item_reserved", index=index, run_id=item["run_id"])
    _save_queue(queue)


def _launch_detach(queue: MutableMapping[str, Any], index: int) -> None:
    item = queue["items"][index]
    root = _item_orchestration_root(queue, item)
    root.mkdir(parents=True, exist_ok=False)
    runner = _verify_runner_source(queue, item)
    python = Path(str(queue["plan"]["runner_python"])).resolve(strict=True)
    command = [
        str(python),
        str(runner),
        "detach",
        "--orchestration-root",
        str(root),
        "--run-id",
        item["run_id"],
    ]
    log_path = root / "detach_launcher.log"
    environment = _runner_environment(queue["plan"]["runtime_environment"])
    try:
        with log_path.open("ab", buffering=0) as handle:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    except BaseException as exc:
        raise QueueContractError(f"cannot spawn detached runner launcher: {exc}") from exc
    item["status"] = "launching"
    item["detach_command"] = command
    item["detach_command_shell"] = shlex.join(command)
    item["detach_log"] = str(log_path)
    item["detach_launcher_pid"] = int(process.pid)
    item["detach_launcher_identity"] = _read_process_identity(int(process.pid))
    item["detach_started_at_utc"] = _utc_now()
    _LOCAL_DETACH_LAUNCHERS[int(process.pid)] = process
    _event(
        queue,
        "detach_launcher_spawned",
        index=index,
        run_id=item["run_id"],
        pid=int(process.pid),
    )
    _save_queue(queue)


def _validate_and_bind_job(
    queue: MutableMapping[str, Any], index: int, job_dir: Path
) -> None:
    item = queue["items"][index]
    root = _item_orchestration_root(queue, item)
    try:
        job_dir.relative_to(root)
    except ValueError as exc:
        raise QueueContractError(f"detached job escaped its orchestration root: {job_dir}") from exc
    launch = _json_object(job_dir / "launch.json", description="detached launch")
    status = _json_object(job_dir / "status.json", description="detached status")
    if launch.get("run_ids") != [item["run_id"]]:
        raise QueueContractError("detached launch run_ids differ from the queue item")
    if status.get("run_ids") != [item["run_id"]]:
        raise QueueContractError("detached status run_ids differ from the queue item")
    roots = launch.get("expected_run_roots")
    if not isinstance(roots, list) or len(roots) != 1 or not isinstance(roots[0], str):
        raise QueueContractError("detached launch must name exactly one expected run root")
    output_root = Path(roots[0]).resolve(strict=False)
    status_roots = status.get("expected_run_roots")
    if status_roots != roots:
        raise QueueContractError("detached launch/status expected roots differ")
    runtime = launch.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("cuda_visible_devices") != queue["plan"]["gpu_key"]
    ):
        raise QueueContractError("detached launch GPU differs from the queue lease key")
    plans = sorted((job_dir / "plans").rglob("*.json")) if (job_dir / "plans").is_dir() else []
    if len(plans) != 1:
        raise QueueContractError("single-run detached job must retain exactly one preflight plan")
    child_plan = _json_object(plans[0], description="detached child preflight plan")
    if child_plan.get("run_id") != item["run_id"]:
        raise QueueContractError("detached child plan run_id differs from queue item")
    if child_plan.get("output_dir_fresh_at_plan") is not True:
        raise QueueContractError("detached runner did not prove a fresh output at preflight")
    if Path(str(child_plan.get("output_dir"))).resolve(strict=False) != output_root:
        raise QueueContractError("detached child plan output differs from launch output")
    launch_status = launch.get("status")
    if launch_status == "spawn_failed":
        raise QueueContractError("detached runner explicitly reports spawn_failed")
    if launch_status != "launched":
        raise QueueContractError(f"detached launch is not launched: {launch_status!r}")
    item["status"] = "launched"
    item["job_dir"] = str(job_dir)
    item["output_root"] = str(output_root)
    item["child_pid"] = launch.get("child_pid")
    item["child_process_identity"] = launch.get("child_process_identity")
    item["job_bound_at_utc"] = _utc_now()
    _event(queue, "detached_job_bound", index=index, run_id=item["run_id"], job_dir=str(job_dir))
    _save_queue(queue)


def _process_environment(pid: int) -> dict[str, str] | None:
    try:
        entries = (Path("/proc") / str(pid) / "environ").read_bytes().split(
            b"\0"
        )
    except OSError:
        return None
    environment: dict[str, str] = {}
    for entry in entries:
        if not entry:
            continue
        raw_key, separator, raw_value = entry.partition(b"=")
        if not separator:
            return None
        key = raw_key.decode("utf-8", errors="strict")
        if key in environment:
            return None
        environment[key] = raw_value.decode("utf-8", errors="strict")
    return environment


def _expected_spawn_window_child_command(
    queue: Mapping[str, Any], item: Mapping[str, Any]
) -> list[str]:
    runner = _verify_runner_source(queue, item)
    python = Path(str(queue["plan"]["runner_python"])).resolve(strict=True)
    return [
        str(python),
        str(runner),
        "run",
        "--run-id",
        str(item["run_id"]),
    ]


def _validated_prepared_spawn_window(
    queue: Mapping[str, Any], index: int, job_dir: Path
) -> tuple[list[str], Path, Path]:
    item = queue["items"][index]
    root = _item_orchestration_root(queue, item)
    try:
        job_dir.relative_to(root)
    except ValueError as exc:
        raise QueueContractError(
            f"spawn-window job escaped its orchestration root: {job_dir}"
        ) from exc
    launch_path = job_dir / "launch.json"
    status_path = (job_dir / "status.json").resolve(strict=True)
    launch = _json_object(launch_path, description="prepared detached launch")
    status = _json_object(status_path, description="prepared detached status")
    if launch.get("status") != "prepared":
        raise QueueContractError("spawn-window launch is not exactly prepared")
    if launch.get("run_ids") != [item["run_id"]]:
        raise QueueContractError("prepared launch run_ids differ from the queue item")
    if status.get("run_ids") != [item["run_id"]]:
        raise QueueContractError("prepared status run_ids differ from the queue item")
    roots = launch.get("expected_run_roots")
    if not (
        isinstance(roots, list)
        and len(roots) == 1
        and isinstance(roots[0], str)
        and status.get("expected_run_roots") == roots
    ):
        raise QueueContractError("prepared launch/status roots are not exact")
    output_root = Path(roots[0]).resolve(strict=False)
    runtime = launch.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("cuda_visible_devices") != queue["plan"]["gpu_key"]
    ):
        raise QueueContractError("prepared launch GPU differs from the queue lease")
    plans_dir = job_dir / "plans"
    plans = sorted(plans_dir.rglob("*.json")) if plans_dir.is_dir() else []
    if len(plans) != 1:
        raise QueueContractError(
            "prepared single-run job must retain exactly one preflight plan"
        )
    child_plan = _json_object(
        plans[0], description="prepared detached child preflight plan"
    )
    if (
        child_plan.get("run_id") != item["run_id"]
        or child_plan.get("output_dir_fresh_at_plan") is not True
        or Path(str(child_plan.get("output_dir"))).resolve(strict=False)
        != output_root
    ):
        raise QueueContractError("prepared child preflight binding drifted")
    expected_command = _expected_spawn_window_child_command(queue, item)
    command = launch.get("command")
    if command != expected_command or launch.get("command_shell") != shlex.join(
        expected_command
    ):
        raise QueueContractError("prepared child command differs from the sealed queue")
    if (
        Path(str(launch.get("job_dir", ""))).resolve(strict=False) != job_dir
        or Path(str(launch.get("orchestrator_status", ""))).resolve(strict=False)
        != status_path
    ):
        raise QueueContractError("prepared launch control paths drifted")
    return expected_command, status_path, output_root


def _sealed_session_leader_identity(
    pid: int, expected: Mapping[str, Any]
) -> dict[str, Any]:
    expected_identity = _validated_child_identity(pid, expected)
    if _process_running(pid, expected_identity) is not True:
        raise QueueContractError("spawn-window child is not provably running")
    observed = _validated_child_identity(pid, _read_process_identity(pid))
    if (
        observed["start_time_ticks"] != expected_identity["start_time_ticks"]
        or observed["boot_id"] != expected_identity["boot_id"]
    ):
        raise QueueContractError("spawn-window child identity changed")
    try:
        process_group_id = os.getpgid(pid)
        session_id = os.getsid(pid)
    except ProcessLookupError as exc:
        raise QueueContractError(
            "spawn-window child disappeared during identity verification"
        ) from exc
    if process_group_id != pid or session_id != pid:
        raise QueueContractError(
            "spawn-window child is not its sealed session/process-group leader"
        )
    return observed


def _matching_spawn_window_children(
    command: Sequence[str], status_path: Path
) -> list[tuple[int, dict[str, Any]]]:
    proc = Path("/proc")
    if not proc.is_dir():
        raise QueueContractError("spawn-window process inventory is unavailable")
    expected_command = list(command)
    expected_status = str(status_path.resolve(strict=True))
    matches: list[tuple[int, dict[str, Any]]] = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            values = [
                value.decode("utf-8", errors="strict")
                for value in (entry / "cmdline").read_bytes().split(b"\0")
                if value
            ]
        except (OSError, UnicodeDecodeError):
            continue
        if values != expected_command:
            continue
        pid = int(entry.name)
        try:
            environment = _process_environment(pid)
        except UnicodeDecodeError as exc:
            raise QueueContractError(
                "spawn-window child environment is not observable"
            ) from exc
        if environment is None:
            raise QueueContractError(
                "spawn-window child environment is not observable"
            )
        if environment.get("PIVOT_ORCHESTRATION_STATUS") != expected_status:
            continue
        identity = _read_process_identity(pid)
        matches.append((pid, _sealed_session_leader_identity(pid, identity)))
    return matches


def _seal_spawn_window_group_binding(
    queue: MutableMapping[str, Any], index: int, *, launch_status: str
) -> None:
    item = queue["items"][index]
    pid = item.get("child_pid")
    identity = _sealed_session_leader_identity(
        pid, item.get("child_process_identity")
    )
    item["child_process_identity"] = identity
    item["child_process_group_id"] = pid
    item["child_session_id"] = pid
    item["spawn_window_status_path"] = str(
        (Path(str(item["job_dir"])) / "status.json").resolve(strict=True)
    )
    item["spawn_window_bound_at_utc"] = _utc_now()
    item["spawn_window_launch_status"] = launch_status
    _event(
        queue,
        "spawn_window_child_group_sealed",
        index=index,
        run_id=item["run_id"],
        pid=pid,
        process_group_id=pid,
        session_id=pid,
    )
    _save_queue(queue)


def _bind_spawn_window_child(
    queue: MutableMapping[str, Any], index: int
) -> None:
    item = queue["items"][index]
    root = _item_orchestration_root(queue, item)
    jobs = _detached_job_dirs(root)
    if len(jobs) != 1:
        raise QueueContractError(
            "spawn-window recovery requires exactly one detached job"
        )
    job_dir = jobs[0]
    launch = _json_object(job_dir / "launch.json", description="detached launch")
    if launch.get("status") == "launched":
        _validate_and_bind_job(queue, index, job_dir)
        _seal_spawn_window_group_binding(
            queue, index, launch_status="launched"
        )
        return
    command, status_path, output_root = _validated_prepared_spawn_window(
        queue, index, job_dir
    )
    matches = _matching_spawn_window_children(command, status_path)
    if len(matches) != 1:
        raise QueueContractError(
            "spawn-window child identity is absent or non-unique"
        )
    pid, identity = matches[0]
    identity = _sealed_session_leader_identity(pid, identity)
    latest_launch = _json_object(
        job_dir / "launch.json", description="rechecked detached launch"
    )
    if latest_launch.get("status") == "launched":
        published_pid = latest_launch.get("child_pid")
        published_identity = _validated_child_identity(
            published_pid, latest_launch.get("child_process_identity")
        )
        if (
            published_pid != pid
            or published_identity["start_time_ticks"]
            != identity["start_time_ticks"]
            or published_identity["boot_id"] != identity["boot_id"]
        ):
            raise QueueContractError(
                "published child identity differs from spawn-window recovery"
            )
        _validate_and_bind_job(queue, index, job_dir)
        _seal_spawn_window_group_binding(
            queue, index, launch_status="launched"
        )
        return
    if latest_launch.get("status") != "prepared":
        raise QueueContractError(
            "spawn-window launch changed to an unrecognized state"
        )
    item["status"] = "launched"
    item["job_dir"] = str(job_dir)
    item["output_root"] = str(output_root)
    item["child_pid"] = pid
    item["child_process_identity"] = identity
    item["child_process_group_id"] = pid
    item["child_session_id"] = pid
    item["spawn_window_status_path"] = str(status_path)
    item["spawn_window_bound_at_utc"] = _utc_now()
    item["spawn_window_launch_status"] = "prepared"
    _event(
        queue,
        "spawn_window_child_bound",
        index=index,
        run_id=item["run_id"],
        pid=pid,
    )
    _save_queue(queue)


def _recover_or_launch_reserved(queue: MutableMapping[str, Any], index: int) -> None:
    item = queue["items"][index]
    _ensure_lease(queue, item, create=False)
    root = _item_orchestration_root(queue, item)
    if not root.exists():
        _launch_detach(queue, index)
        return
    jobs = _detached_job_dirs(root)
    if len(jobs) > 1:
        raise QueueContractError(f"multiple detached jobs exist for one queue item: {jobs}")
    if len(jobs) == 1:
        launch = _json_object(jobs[0] / "launch.json", description="detached launch")
        if launch.get("status") == "launched":
            _validate_and_bind_job(queue, index, jobs[0])
            return
        if launch.get("status") == "spawn_failed":
            raise QueueContractError("recovered detach launch reports spawn_failed")
    runner = _verify_runner_source(queue, item)
    live = _matching_detach_launchers(runner=runner, root=root, run_id=item["run_id"])
    if len(live) > 1:
        raise QueueContractError("multiple live detach launchers match one queue item")
    if len(live) == 1:
        pid, identity = live[0]
        item["status"] = "launching"
        item["detach_launcher_pid"] = pid
        item["detach_launcher_identity"] = identity
        item["detach_recovered_at_utc"] = _utc_now()
        _event(queue, "detach_launcher_recovered", index=index, run_id=item["run_id"], pid=pid)
        _save_queue(queue)
        return
    if jobs:
        raise QueueContractError("detach launcher died before its launch record became terminal")
    # The durable reservation existed, but neither a launcher nor any child job
    # exists.  Re-running detach is safe: the existing runner repeats its own
    # fresh-output preflight before spawning training.
    root.rmdir()
    _launch_detach(queue, index)


def _advance_launching(queue: MutableMapping[str, Any], index: int) -> None:
    item = queue["items"][index]
    root = _item_orchestration_root(queue, item)
    jobs = _detached_job_dirs(root)
    if len(jobs) > 1:
        raise QueueContractError(f"multiple detached jobs exist for one queue item: {jobs}")
    if jobs:
        launch = _json_object(jobs[0] / "launch.json", description="detached launch")
        if launch.get("status") == "launched":
            try:
                _validate_and_bind_job(queue, index, jobs[0])
            except KeyboardInterrupt:
                raise
            except BaseException as exc:
                if _launch_record_group_is_gone(launch):
                    raise
                # The child may still be publishing its preflight evidence.
                # Keep the durable launching state and retry without allowing
                # a later item to start or releasing the queue's GPU lease.
                return
            _ensure_lease(queue, queue["items"][index], create=False)
            return
        if launch.get("status") == "spawn_failed":
            raise QueueContractError("detach launch reports spawn_failed")
    _ensure_lease(queue, item, create=False)
    running = _process_running(
        item.get("detach_launcher_pid"), item.get("detach_launcher_identity")
    )
    if running is True or running is None:
        return
    if jobs:
        raise QueueContractError("detach launcher terminated with a nonterminal launch record")
    raise QueueContractError("detach launcher terminated without creating a child job")


def _runner_observation(
    queue: Mapping[str, Any], item: Mapping[str, Any], *, reconcile: bool
) -> dict[str, Any]:
    runner = _verify_runner_source(queue, item)
    job_dir = item.get("job_dir")
    if not isinstance(job_dir, str):
        raise QueueContractError("launched item has no detached job directory")
    mode = "reconcile" if reconcile else "status"
    python = Path(str(queue["plan"]["runner_python"])).resolve(strict=True)
    return _run_json_command(
        [str(python), str(runner), mode, job_dir],
        environment=_runner_environment(queue["plan"]["runtime_environment"]),
        description=f"{item['runner']} runner {mode} for {item['run_id']}",
    )


def _validate_file_record(record: Any, expected_path: Path, *, label: str) -> None:
    if not isinstance(record, Mapping):
        return
    raw_path = record.get("path")
    if raw_path is not None and Path(str(raw_path)).resolve(strict=False) != expected_path:
        raise QueueContractError(f"{label} record points to a different path")
    expected_sha = record.get("sha256")
    if expected_sha is not None and _sha256_file(expected_path) != expected_sha:
        raise QueueContractError(f"{label} SHA-256 differs from its sequence record")


def _phase_map(sequence: Mapping[str, Any], key: str) -> dict[str, Mapping[str, Any]]:
    raw = sequence.get(key)
    if not isinstance(raw, list) or not raw:
        raise QueueContractError(f"completed sequence has no {key}")
    result: dict[str, Mapping[str, Any]] = {}
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise QueueContractError(f"sequence {key} contains a non-object")
        phase_id = entry.get("phase_id")
        if not isinstance(phase_id, str) or not phase_id or phase_id in result:
            raise QueueContractError(f"sequence {key} has invalid/duplicate phase_id")
        result[phase_id] = entry
    return result


def _verify_completed_item(queue: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, Any]:
    _verify_runner_source(queue, item)
    job_dir = Path(str(item.get("job_dir", ""))).resolve(strict=True)
    launch = _json_object(job_dir / "launch.json", description="detached launch")
    status = _json_object(job_dir / "status.json", description="detached status")
    if launch.get("run_ids") != [item["run_id"]] or status.get("run_ids") != [item["run_id"]]:
        raise QueueContractError("terminal detached job run IDs differ from queue item")
    if status.get("status") != "completed":
        raise QueueContractError(f"detached status is not completed: {status.get('status')!r}")
    completed_ids = status.get("completed_run_ids")
    if not isinstance(completed_ids, list) or item["run_id"] not in completed_ids:
        raise QueueContractError("detached status does not explicitly list the completed run")
    roots = launch.get("expected_run_roots")
    if not isinstance(roots, list) or len(roots) != 1:
        raise QueueContractError("terminal detached launch has invalid expected roots")
    output_root = Path(str(roots[0])).resolve(strict=True)
    if output_root != Path(str(item.get("output_root", ""))).resolve(strict=False):
        raise QueueContractError("terminal output root differs from the bound queue item")
    sequence_path = output_root / "sequence_manifest.json"
    sequence = _json_object(sequence_path, description="completed sequence manifest")
    if sequence.get("status") != "completed" or sequence.get("run_id") != item["run_id"]:
        raise QueueContractError("sequence is not explicitly completed for this run ID")
    if Path(str(sequence.get("output_dir"))).resolve(strict=False) != output_root:
        raise QueueContractError("sequence output_dir differs from detached run root")
    planned = _phase_map(sequence, "phases")
    completed = _phase_map(sequence, "completed_phases")
    if list(planned) != list(completed):
        raise QueueContractError(
            "completed phase order differs from planned phases: "
            f"{list(completed)} vs {list(planned)}"
        )
    phase_evidence: list[dict[str, Any]] = []
    for phase_id in planned:
        planned_phase = planned[phase_id]
        completed_phase = completed[phase_id]
        if completed_phase.get("status") != "completed":
            raise QueueContractError(f"phase {phase_id} is not explicitly completed")
        planned_dir = Path(str(planned_phase.get("output_dir"))).resolve(strict=False)
        completed_dir = Path(str(completed_phase.get("output_dir"))).resolve(strict=True)
        if completed_dir != planned_dir:
            raise QueueContractError(f"phase {phase_id} output changed from the plan")
        try:
            completed_dir.relative_to(output_root)
        except ValueError as exc:
            raise QueueContractError(f"phase {phase_id} output escaped the run root") from exc
        postflight_path = completed_dir / "postflight.json"
        postflight = _json_object(postflight_path, description=f"{phase_id} postflight")
        if postflight.get("status") != "passed" or postflight.get("run_id") != item["run_id"]:
            raise QueueContractError(f"phase {phase_id} postflight did not pass for this run")
        if item["runner"] == "paper" and postflight.get("phase_id") != phase_id:
            raise QueueContractError(f"paper phase {phase_id} postflight phase_id mismatch")
        launch_manifest = _json_object(
            completed_dir / "launch_manifest.json",
            description=f"{phase_id} launch manifest",
        )
        if (
            launch_manifest.get("status") != "completed"
            or launch_manifest.get("run_id") != item["run_id"]
        ):
            raise QueueContractError(f"phase {phase_id} launch manifest is not completed")
        _validate_file_record(
            completed_phase.get("postflight"), postflight_path, label=f"{phase_id} postflight"
        )
        _validate_file_record(
            launch_manifest.get("postflight_artifact"),
            postflight_path,
            label=f"{phase_id} launch postflight",
        )
        phase_evidence.append(
            {
                "phase_id": phase_id,
                "output_dir": str(completed_dir),
                "postflight": str(postflight_path),
                "postflight_sha256": _sha256_file(postflight_path),
                "launch_manifest": str(completed_dir / "launch_manifest.json"),
            }
        )
    return {
        "schema": COMPLETION_SCHEMA,
        "verified_at_utc": _utc_now(),
        "run_id": item["run_id"],
        "runner": item["runner"],
        "job_dir": str(job_dir),
        "output_root": str(output_root),
        "sequence_manifest": str(sequence_path),
        "sequence_sha256": _sha256_file(sequence_path),
        "phases": phase_evidence,
        "advance_gate": (
            "dead_detached_orchestrator_plus_completed_sequence_plus_"
            "passed_postflights"
        ),
    }


def _fail_queue(
    queue: MutableMapping[str, Any], index: int, *, phase: str, error: BaseException | str
) -> None:
    item = queue["items"][index]
    rendered = str(error) if isinstance(error, str) else f"{type(error).__name__}: {error}"
    item["status"] = "failed"
    item["failed_at_utc"] = _utc_now()
    item["failure_phase"] = phase
    item["failure_error"] = rendered
    queue["status"] = "failed"
    queue["failure"] = {
        "index": index,
        "run_id": item["run_id"],
        "phase": phase,
        "error": rendered,
        "lease_retained_fail_closed": _owned_lease_present(queue),
    }
    _event(queue, "queue_failed", index=index, run_id=item["run_id"], phase=phase, error=rendered)
    _save_queue(queue)


def _owned_lease_present(queue: Mapping[str, Any]) -> bool:
    lease_path = Path(str(queue["plan"]["lease_path"]))
    if not lease_path.is_file():
        return False
    try:
        lease = _json_object(lease_path, description="GPU lease")
    except QueueContractError:
        return False
    return not _lease_identity_mismatches(queue, lease)


def _advance_launched(queue: MutableMapping[str, Any], index: int) -> None:
    item = queue["items"][index]
    launcher_pid = item.get("detach_launcher_pid")
    if isinstance(launcher_pid, int):
        _process_running(launcher_pid, item.get("detach_launcher_identity"))
    _ensure_lease(queue, item, create=False)
    observation = _runner_observation(queue, item, reconcile=True)
    observed = observation.get("observed_status")
    liveness = observation.get("pid_liveness")
    running = liveness.get("running") if isinstance(liveness, Mapping) else None
    item["last_observation"] = {
        "observed_at_utc": observation.get("observed_at_utc", _utc_now()),
        "persisted_status": observation.get("persisted_status"),
        "observed_status": observed,
        "reason": observation.get("reason"),
        "pid_running": running,
    }
    # Persist the authoritative child observation before interpreting it.  A
    # later completion-artifact failure must not erase the terminal evidence
    # that led the queue to perform that verification.
    _save_queue(queue)
    if observed == "completed":
        if running is not False:
            return
        evidence = _verify_completed_item(queue, item)
        item["status"] = "completed"
        item["completed_at_utc"] = _utc_now()
        item["completion_evidence"] = evidence
        _event(queue, "item_completed", index=index, run_id=item["run_id"])
        if all(candidate["status"] == "completed" for candidate in queue["items"]):
            queue["status"] = "completed"
            queue["completed_at_utc"] = _utc_now()
            _event(queue, "queue_completed")
        _save_queue(queue)
        if queue["status"] == "completed":
            _clear_owned_lease(queue)
        return
    if observed in RUNNER_FAILURE_STATUSES:
        raise QueueContractError(
            f"detached runner reached terminal failure {observed}: {observation.get('reason')}"
        )
    if observed in RUNNER_NONTERMINAL_STATUSES:
        if running is False:
            raise QueueContractError(
                "reconcile retained a nonterminal status for a dead detached runner"
            )
        return
    raise QueueContractError(f"unrecognized detached runner status: {observed!r}")


def advance_once(queue_dir: Path) -> dict[str, Any]:
    queue = load_queue(queue_dir)
    if queue["status"] == "completed":
        _clear_owned_lease(queue)
        return queue
    if queue["status"] == "failed":
        return queue
    index = next(
        (i for i, item in enumerate(queue["items"]) if item["status"] != "completed"),
        None,
    )
    if index is None:
        raise QueueContractError("queue has no incomplete item but is not completed")
    item = queue["items"][index]
    try:
        if item["status"] == "pending":
            _reserve_next(queue, index)
        elif item["status"] == "reserved":
            _recover_or_launch_reserved(queue, index)
        elif item["status"] == "launching":
            _advance_launching(queue, index)
        elif item["status"] == "launched":
            _advance_launched(queue, index)
        else:
            raise QueueContractError(f"cannot advance item in status {item['status']!r}")
    except QueueLeaseOwnershipError as exc:
        current = load_queue(queue_dir)
        current_index = next(
            (
                i
                for i, candidate in enumerate(current["items"])
                if candidate["status"] != "completed"
            ),
            index,
        )
        current_item = current["items"][current_index]
        current_status = current_item["status"]
        if current_status == "pending":
            raise
        if current_status == "launching":
            try:
                launcher_termination = _terminate_detach_launcher_process_group(
                    current, current_index
                )
            except BaseException as launcher_error:
                current_item["spawn_window_reconciliation_blocked"] = {
                    "at_utc": _utc_now(),
                    "original_error": f"{type(exc).__name__}: {exc}",
                    "reconciliation_error": (
                        f"{type(launcher_error).__name__}: {launcher_error}"
                    ),
                    "stage": "detach_launcher_fence",
                }
                _save_queue(current)
                raise QueueContractError(
                    "queue remains launching because the detach launcher could "
                    "not be fenced after GPU lease ownership loss"
                ) from launcher_error
            current_item["detach_launcher_termination"] = launcher_termination
            _save_queue(current)
            try:
                _bind_spawn_window_child(current, current_index)
            except KeyboardInterrupt:
                raise
            except BaseException as reconciliation_error:
                recovered = load_queue(queue_dir)
                recovered_item = recovered["items"][current_index]
                if recovered_item["status"] != "launched":
                    recovered_item["spawn_window_reconciliation_blocked"] = {
                        "at_utc": _utc_now(),
                        "original_error": f"{type(exc).__name__}: {exc}",
                        "reconciliation_error": (
                            f"{type(reconciliation_error).__name__}: "
                            f"{reconciliation_error}"
                        ),
                        "stage": "child_identity_reconciliation",
                    }
                    _save_queue(recovered)
                    raise QueueContractError(
                        "queue remains launching because the spawn-window child "
                        "identity could not be proven after GPU lease ownership loss"
                    ) from reconciliation_error
                current = recovered
                current_item = recovered_item
            current_status = current_item["status"]
        if (
            current_status == "launched"
            and not current_item.get("spawn_window_bound_at_utc")
            and isinstance(current_item.get("detach_launcher_pid"), int)
        ):
            try:
                if not current_item.get("detach_launcher_termination"):
                    current_item["detach_launcher_termination"] = (
                        _terminate_detach_launcher_process_group(
                            current, current_index
                        )
                    )
                    _save_queue(current)
                _seal_spawn_window_group_binding(
                    current, current_index, launch_status="launched"
                )
            except BaseException as seal_error:
                pid = current_item.get("child_pid")
                child_is_gone = (
                    _process_running(
                        pid, current_item.get("child_process_identity")
                    )
                    is False
                    and isinstance(pid, int)
                    and _process_group_exists(pid) is False
                )
                if not child_is_gone:
                    current_item["spawn_window_reconciliation_blocked"] = {
                        "at_utc": _utc_now(),
                        "original_error": f"{type(exc).__name__}: {exc}",
                        "reconciliation_error": (
                            f"{type(seal_error).__name__}: {seal_error}"
                        ),
                        "stage": "published_child_group_seal",
                    }
                    _save_queue(current)
                    raise QueueContractError(
                        "queue remains launched because the concurrently published "
                        "child group could not be sealed after lease ownership loss"
                    ) from seal_error
        if current_status == "launched":
            try:
                termination = _terminate_bound_child_process_group(
                    current, current_index
                )
            except BaseException as termination_error:
                current_item["child_termination_blocked"] = {
                    "at_utc": _utc_now(),
                    "original_error": f"{type(exc).__name__}: {exc}",
                    "termination_error": (
                        f"{type(termination_error).__name__}: {termination_error}"
                    ),
                }
                _save_queue(current)
                raise QueueContractError(
                    "queue remains launched because its child process group could "
                    "not be proven terminated after GPU lease ownership loss"
                ) from termination_error
            current_item["child_termination"] = termination
        elif current_status == "reserved" and _item_orchestration_root(
            current, current_item
        ).exists():
            raise
        _fail_queue(
            current,
            current_index,
            phase="lease_ownership_loss",
            error=exc,
        )
        return load_queue(queue_dir)
    except (QueueBusyError, KeyboardInterrupt):
        raise
    except BaseException as exc:
        # Reload before failing so a transition atomically persisted inside the
        # attempted step is never overwritten by an older in-memory revision.
        current = load_queue(queue_dir)
        if current["status"] == "completed":
            # Completion evidence was already committed.  A transient lease
            # cleanup error remains recoverable on the next invocation and is
            # not a training/postflight failure.
            return current
        current_index = next(
            (
                i
                for i, candidate in enumerate(current["items"])
                if candidate["status"] != "completed"
            ),
            index,
        )
        if current["items"][current_index]["status"] != "failed":
            _fail_queue(current, current_index, phase="advance", error=exc)
        return load_queue(queue_dir)
    return load_queue(queue_dir)


def run_queue(queue_dir: Path, *, poll_seconds: float, once: bool) -> dict[str, Any]:
    if poll_seconds < 0.05:
        raise QueueContractError("--poll-seconds must be at least 0.05")
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    lock_path = queue_dir / "supervisor.lock"
    with _exclusive_file_lock(
        lock_path, busy_message=f"another queue supervisor is active: {queue_dir}"
    ):
        while True:
            queue = advance_once(queue_dir)
            if once or queue["status"] in {"completed", "failed"}:
                return queue
            time.sleep(poll_seconds)


def queue_status(queue_dir: Path) -> dict[str, Any]:
    queue = load_queue(queue_dir)
    counts = {
        status: sum(item["status"] == status for item in queue["items"])
        for status in ITEM_STATUSES
    }
    current = next(
        (item for item in queue["items"] if item["status"] not in {"completed", "pending"}),
        None,
    )
    report: dict[str, Any] = {
        "schema": "pivot.stageb.serial_matrix_queue_status/v1",
        "observed_at_utc": _utc_now(),
        "queue_dir": queue["plan"]["queue_dir"],
        "queue_id": queue["plan"]["queue_id"],
        "status": queue["status"],
        "revision": queue["revision"],
        "gpu_key": queue["plan"]["gpu_key"],
        "lease_path": queue["plan"]["lease_path"],
        "counts": counts,
        "ordered_run_ids": [item["run_id"] for item in queue["items"]],
        "current_item": dict(current) if current is not None else None,
        "failure": queue.get("failure"),
    }
    lease_path = Path(queue["plan"]["lease_path"])
    report["lease"] = (
        _json_object(lease_path, description="GPU lease")
        if lease_path.is_file()
        else {"present": False}
    )
    runner_drift: dict[str, Any] = {}
    for name, record in queue["plan"]["runners"].items():
        path = Path(record["path"])
        try:
            observed = _sha256_file(path)
            runner_drift[name] = {
                "path": str(path),
                "expected_sha256": record["sha256"],
                "observed_sha256": observed,
                "matches": observed == record["sha256"],
            }
        except OSError as exc:
            runner_drift[name] = {"path": str(path), "error": str(exc), "matches": False}
    report["runner_source_identity"] = runner_drift
    if current is not None and current.get("status") == "launched":
        try:
            report["detached_job_observation"] = _runner_observation(
                queue, current, reconcile=False
            )
        except QueueContractError as exc:
            report["detached_job_observation_error"] = str(exc)
    return report


def verify_queue(queue_dir: Path) -> dict[str, Any]:
    queue = load_queue(queue_dir)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for item in queue["items"]:
        if item["status"] != "completed":
            errors.append({"run_id": item["run_id"], "error": f"status={item['status']}"})
            continue
        try:
            results.append(_verify_completed_item(queue, item))
        except QueueContractError as exc:
            errors.append({"run_id": item["run_id"], "error": str(exc)})
    return {
        "schema": "pivot.stageb.serial_matrix_queue_verification/v1",
        "verified_at_utc": _utc_now(),
        "status": "passed" if queue["status"] == "completed" and not errors else "failed",
        "queue_status": queue["status"],
        "queue_id": queue["plan"]["queue_id"],
        "plan_sha256": queue["plan_sha256"],
        "verified_items": results,
        "errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    create = subparsers.add_parser("create", help="create one fresh explicit queue")
    create.add_argument("queue_dir", type=Path)
    create.add_argument(
        "--run-id",
        action="append",
        required=True,
        help="ordered ROW:SEED selection; repeat for every queue item",
    )
    create.add_argument("--runner-python", type=Path, default=Path(sys.executable))
    create.add_argument("--token-runner", type=Path, default=DEFAULT_TOKEN_RUNNER)
    create.add_argument("--paper-runner", type=Path, default=DEFAULT_PAPER_RUNNER)
    create.add_argument("--lease-root", type=Path, default=DEFAULT_LEASE_ROOT)
    create.add_argument(
        "--gpu-key",
        help="canonical single-GPU lease key; must match captured CUDA visibility",
    )
    run = subparsers.add_parser("run", help="resume and supervise until terminal")
    run.add_argument("queue_dir", type=Path)
    run.add_argument("--poll-seconds", type=float, default=30.0)
    run.add_argument(
        "--once",
        action="store_true",
        help="perform one durable state transition and return",
    )
    status = subparsers.add_parser("status", help="read queue and child status without mutation")
    status.add_argument("queue_dir", type=Path)
    verify = subparsers.add_parser("verify", help="recheck every completed sequence/postflight")
    verify.add_argument("queue_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.mode == "create":
            queue = create_queue(
                args.queue_dir,
                run_ids=args.run_id,
                runner_python=args.runner_python,
                token_runner=args.token_runner,
                paper_runner=args.paper_runner,
                lease_root=args.lease_root,
                gpu_key=args.gpu_key,
            )
            print(json.dumps(queue, indent=2, sort_keys=True))
            return 0
        if args.mode == "run":
            queue = run_queue(
                args.queue_dir, poll_seconds=args.poll_seconds, once=args.once
            )
            print(json.dumps(queue_status(args.queue_dir), indent=2, sort_keys=True))
            return 0 if queue["status"] != "failed" else 1
        if args.mode == "status":
            print(json.dumps(queue_status(args.queue_dir), indent=2, sort_keys=True))
            return 0
        if args.mode == "verify":
            report = verify_queue(args.queue_dir)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["status"] == "passed" else 1
        parser.error(f"unknown mode: {args.mode}")
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        PermissionError,
        QueueContractError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
