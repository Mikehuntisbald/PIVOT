#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Fail-closed controller for the CVPR confidence-adapter Stage-B phase."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/haoyi/miniconda/envs/gdino5090/bin/python")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if __name__ == "__main__" and Path(sys.executable).resolve() != PYTHON.resolve():
    os.execv(str(PYTHON), [str(PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

from util.stage_b_confidence_adapter_migration import (  # noqa: E402
    validate_confidence_adapter_migration_audit,
)
from util.stage_b_dense_duty_audit import (  # noqa: E402
    SOURCE_CLOSURE_ARG,
    STRICT_RESUME_REQUIRED_KEYS,
    TRAINING_CONTRACT_ARG,
    _validate_resume_rng_state,
    audit_checkpoint_payload,
    build_source_closure,
    build_training_contract,
    fingerprint_named_tensors,
    validate_evaluation_checkpoint_payload,
    validate_formal_invocation,
    validate_resume_training_contract,
    validate_strict_resume_checkpoint_payload,
)
from util.slconfig import SLConfig  # noqa: E402


CONFIG = REPO_ROOT / "config/ablations/cfg_stageb_dense_duty_confidence_adapter_20260730.py"
DATASET = REPO_ROOT / "config/datasets_stageb_dense_duty_confidence_20260728.json"
RANK_SOURCE = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/dense_duty_20260728/formal/rank/checkpoint_iter.pth"
)
OUTPUT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/dense_duty_adapter_20260730/formal/confidence"
)
CHECKPOINT = OUTPUT / "checkpoint_iter.pth"
LOCK = OUTPUT.parent.parent / ".formal_confidence_adapter.lock"
LOG = OUTPUT.parent / "controller.log"

SOURCE_SHA256 = "50e60a1314f7f2908bee5eea84ede5549b908177b367609efdec1682caa67ed3"
RANK_SHA256 = "e03219d5868004aa5cb9ff4fe68f1aa94d33f1f0f6e1290cb251d12f9c914045"
TRANSFERRED_SHA256 = "5300b52061b2f441346fb81334268bc7d192881c819773e2076d96a36070fe96"
UPDATES = 4_412
SOURCE_UPDATES = 6_551
SOURCE_REASON = "signal"
FORMAL_ADMISSION_VALIDATOR: Callable[[], Mapping[str, Any]] | None = None


class ControllerError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ControllerError(f"checkpoint is not a mapping: {path}")
    return payload


def _validate_migration(saved_args: Mapping[str, Any]) -> dict[str, Any]:
    return validate_confidence_adapter_migration_audit(
        saved_args.get("stage_b_dense_duty_confidence_adapter_migration_audit"),
        source_checkpoint_sha256=SOURCE_SHA256,
        source_optimizer_updates=SOURCE_UPDATES,
        source_checkpoint_reason=SOURCE_REASON,
        rank_sha256=RANK_SHA256,
        transferred_sha256=TRANSFERRED_SHA256,
    )


def _formal_admission_audit() -> dict[str, Any] | None:
    if FORMAL_ADMISSION_VALIDATOR is None:
        return None
    value = FORMAL_ADMISSION_VALIDATOR()
    if not isinstance(value, Mapping):
        raise ControllerError("formal admission validator returned no audit mapping")
    expected = {
        "status": "verified",
        "decision": "admit_to_formal_training",
        "diagnostic_only": True,
        "formal_training_admitted": True,
    }
    for field, required in expected.items():
        if value.get(field) != required:
            raise ControllerError(
                f"formal admission audit requires {field}={required!r}, "
                f"got {value.get(field)!r}"
            )
    return dict(value)


def _formal_current_args() -> dict[str, Any]:
    values = SLConfig.fromfile(str(CONFIG))._cfg_dict.to_dict()
    accumulation_steps = int(
        values["stage_b_dense_duty_expected_gradient_accumulation_steps"]
    )
    values.update(
        {
            "config_file": str(CONFIG),
            "datasets": str(DATASET),
            "output_dir": str(OUTPUT),
            "device": "cuda",
            "seed": 42,
            "resume": str(CHECKPOINT),
            "pretrain_model_path": None,
            "options": None,
            "remove_difficult": False,
            "num_workers": 0,
            "prefetch_factor": 1,
            "pin_memory": True,
            "persistent_workers": False,
            "world_size": 1,
            "distributed": False,
            "find_unused_params": False,
            "gradient_accumulation_steps": accumulation_steps,
            "iter_checkpoint_interval": 100,
            "max_train_iters": UPDATES,
            "amp": True,
        }
    )
    values[SOURCE_CLOSURE_ARG] = build_source_closure(
        CONFIG, repo_root=REPO_ROOT
    )
    admission = _formal_admission_audit()
    if admission is not None:
        values[
            "stage_b_dense_duty_confidence_probe_admission_audit"
        ] = admission
    build_training_contract(values)
    return values


def _validate_terminal_training_state(
    payload: Mapping[str, Any], current_args: Mapping[str, Any]
) -> None:
    import torch

    missing = sorted(STRICT_RESUME_REQUIRED_KEYS.difference(payload))
    if missing:
        raise ControllerError(
            f"terminal checkpoint lacks complete training state: {missing}"
        )
    for key in ("model", "criterion", "optimizer", "lr_scheduler", "scaler"):
        if not isinstance(payload[key], Mapping):
            raise ControllerError(f"terminal checkpoint has invalid {key} state")

    queue_size = int(current_args["stage_b_v14_tail_queue_size"])
    criterion = payload["criterion"]
    queue_keys = {
        "tail_positive_queue",
        "tail_negative_queue",
        "tail_positive_ptr",
        "tail_negative_ptr",
        "tail_positive_count",
        "tail_negative_count",
    }
    if set(criterion) != queue_keys:
        raise ControllerError("terminal checkpoint criterion tail-queue schema drifted")
    for name in ("tail_positive_queue", "tail_negative_queue"):
        value = criterion[name]
        if (
            not torch.is_tensor(value)
            or tuple(value.shape) != (queue_size,)
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all().item())
        ):
            raise ControllerError(f"terminal checkpoint has invalid {name}")
    for name in (
        "tail_positive_ptr",
        "tail_negative_ptr",
        "tail_positive_count",
        "tail_negative_count",
    ):
        value = criterion[name]
        if (
            not torch.is_tensor(value)
            or value.dtype != torch.int64
            or value.numel() != 1
        ):
            raise ControllerError(f"terminal checkpoint has invalid {name}")

    optimizer = payload["optimizer"]
    if (
        not isinstance(optimizer.get("state"), Mapping)
        or not optimizer["state"]
        or not isinstance(optimizer.get("param_groups"), list)
        or not optimizer["param_groups"]
    ):
        raise ControllerError("terminal checkpoint optimizer state is incomplete")
    if not payload["lr_scheduler"]:
        raise ControllerError("terminal checkpoint scheduler state is empty")
    scaler_keys = {
        "scale",
        "growth_factor",
        "backoff_factor",
        "growth_interval",
        "_growth_tracker",
    }
    if bool(current_args["amp"]) and not scaler_keys.issubset(payload["scaler"]):
        raise ControllerError("terminal checkpoint AMP scaler state is incomplete")
    _validate_resume_rng_state(payload["rng_state"], label="rng_state")
    _validate_resume_rng_state(payload["epoch_rng_state"], label="epoch_rng_state")

    for key in ("epoch", "iteration", "optimizer_updates"):
        if type(payload[key]) is not int or payload[key] < 0:
            raise ControllerError(f"terminal checkpoint has invalid {key}")
    if type(payload["epoch_finished"]) is not bool:
        raise ControllerError("terminal checkpoint epoch_finished is not a bool")
    if payload["optimizer_updates"] != UPDATES:
        raise ControllerError("terminal checkpoint update count drifted")
    if payload["checkpoint_reason"] != "max_train_iters":
        raise ControllerError("terminal checkpoint reason is not max_train_iters")
    iteration = payload["iteration"]
    accumulation = int(current_args["gradient_accumulation_steps"])
    if payload["epoch_finished"]:
        if iteration != 0:
            raise ControllerError("terminal epoch-boundary cursor is invalid")
    else:
        expected_physical_forwards = int(
            current_args.get(
                "stage_b_dense_duty_expected_physical_forwards_per_epoch",
                0,
            )
        )
        if (
            iteration <= 0
            or iteration % accumulation != 0
            or expected_physical_forwards <= 0
            or iteration >= expected_physical_forwards
        ):
            raise ControllerError(
                "terminal checkpoint is not at an in-epoch update boundary"
            )


def inspect() -> dict[str, Any]:
    if not OUTPUT.exists():
        return {"status": "fresh", "action": "start"}
    if OUTPUT.is_symlink() or not OUTPUT.is_dir():
        return {"status": "invalid", "reason": "output is not a real directory"}
    if not CHECKPOINT.exists():
        entries = sorted(path.name for path in OUTPUT.iterdir())
        if entries:
            return {
                "status": "invalid",
                "reason": "non-empty output has no atomic checkpoint",
                "entries": entries,
            }
        return {"status": "fresh", "action": "start"}
    try:
        payload = _load(CHECKPOINT)
        saved_args = payload.get("args")
        if not isinstance(saved_args, Mapping):
            raise ControllerError("checkpoint lacks saved args")
        if saved_args.get("stage_b_v22_score_ownership") != (
            "rank_tower_stopgrad_token_adapter_two_phase"
        ):
            raise ControllerError("checkpoint ownership is not the adapter contract")
        _validate_migration(saved_args)
        current_args = _formal_current_args()
        updates = payload.get("optimizer_updates")
        if type(updates) is not int or not 0 < updates <= UPDATES:
            raise ControllerError("checkpoint update count is outside the formal phase")
        if updates < UPDATES:
            validate_strict_resume_checkpoint_payload(
                payload, current_args, checkpoint_path=CHECKPOINT
            )
            checkpoint_audit = audit_checkpoint_payload(
                payload, checkpoint_path=CHECKPOINT
            )
            rank_names = sorted(
                name
                for name in payload["model"]
                if str(name).startswith(
                    "stage_b_fixed_text_scorer.rank_tower."
                )
            )
            rank = fingerprint_named_tensors(payload["model"], rank_names)
            rank_adaptation = int(
                current_args.get(
                    "stage_b_dense_duty_confidence_rank_decoder_unfreeze_last_n",
                    0,
                )
                or 0
            )
            if rank_adaptation == 0 and rank["sha256"] != RANK_SHA256:
                raise ControllerError("partial checkpoint changed the selected rank tower")
            return {"status": "partial", "action": "resume", "updates": updates}
        _validate_terminal_training_state(payload, current_args)
        saved_contract = saved_args.get(TRAINING_CONTRACT_ARG)
        if saved_contract != build_training_contract(saved_args):
            raise ControllerError("terminal training contract is invalid")
        validate_resume_training_contract(current_args, saved_args)
        evaluation_cfg = SimpleNamespace(**current_args)
        checkpoint_audit = validate_evaluation_checkpoint_payload(
            payload,
            evaluation_cfg,
            checkpoint_path=CHECKPOINT,
            current_code_source_closure=current_args[SOURCE_CLOSURE_ARG]["code"],
        )
        rank_names = sorted(
            name
            for name in payload["model"]
            if str(name).startswith("stage_b_fixed_text_scorer.rank_tower.")
        )
        rank = fingerprint_named_tensors(payload["model"], rank_names)
        rank_adaptation = int(
            current_args.get(
                "stage_b_dense_duty_confidence_rank_decoder_unfreeze_last_n",
                0,
            )
            or 0
        )
        if rank_adaptation == 0 and rank["sha256"] != RANK_SHA256:
            raise ControllerError("terminal checkpoint changed the selected rank tower")
        return {
            "status": "terminal",
            "action": "complete",
            "updates": updates,
            "rank_sha256": rank["sha256"],
            "checkpoint_audit": checkpoint_audit,
        }
    except Exception as exc:
        return {"status": "invalid", "reason": f"{type(exc).__name__}: {exc}"}


def command(action: str) -> list[str]:
    config_values = SLConfig.fromfile(str(CONFIG))._cfg_dict.to_dict()
    accumulation_steps = int(
        config_values["stage_b_dense_duty_expected_gradient_accumulation_steps"]
    )
    base = [
        str(PYTHON),
        str(REPO_ROOT / "main.py"),
        "--config_file",
        str(CONFIG),
        "--datasets",
        str(DATASET),
        "--output_dir",
        str(OUTPUT),
        "--device",
        "cuda",
        "--seed",
        "42",
        "--num_workers",
        "0",
        "--prefetch_factor",
        "1",
        "--pin_memory",
        "--no_persistent_workers",
        "--mp_sharing_strategy",
        "file_system",
        "--min_nofile",
        "65536",
        "--world_size",
        "1",
        "--gradient_accumulation_steps",
        str(accumulation_steps),
        "--iter_checkpoint_interval",
        "100",
        "--max_train_iters",
        str(UPDATES),
        "--amp",
        "--save_log",
        "--note",
        "formal_dense_duty_confidence_adapter_seed42",
    ]
    if action == "start":
        base.extend(("--pretrain_model_path", str(RANK_SOURCE)))
    elif action == "resume":
        base.extend(("--resume", str(CHECKPOINT)))
    else:
        raise ControllerError(f"unsupported launch action: {action}")
    return base


def validate_inputs() -> None:
    if not PYTHON.is_file() or not os.access(PYTHON, os.X_OK):
        raise ControllerError(f"fixed Python interpreter is unavailable: {PYTHON}")
    for path in (CONFIG, DATASET, RANK_SOURCE):
        if not path.is_file() or path.is_symlink():
            raise ControllerError(f"required input is missing or symlinked: {path}")
    if _sha256(RANK_SOURCE) != SOURCE_SHA256:
        raise ControllerError("selected U6551 checkpoint SHA256 drifted")
    validate_formal_invocation(_formal_current_args(), repo_root=REPO_ROOT)


def cuda_ready() -> bool:
    import torch

    return bool(torch.cuda.is_available() and torch.cuda.device_count() == 1)


def wait_for_cuda() -> None:
    while not cuda_ready():
        print(
            json.dumps(
                {
                    "status": "waiting_for_cuda",
                    "checked_unix_time": int(time.time()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(30)


@contextlib.contextmanager
def lock_controller():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+", encoding="ascii") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ControllerError("another confidence-adapter controller is active") from exc
        yield handle.fileno()


def run_child(argv: Sequence[str], lock_fd: int) -> tuple[int, list[int]]:
    environment = dict(os.environ)
    for key in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_NTASKS",
        "SLURM_NODELIST",
    ):
        environment.pop(key, None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "MPLCONFIGDIR": "/tmp/matplotlib-stageb-adapter",
        }
    )
    process: subprocess.Popen[Any] | None = None
    forwarded: list[int] = []

    def forward(signum: int, _frame: Any) -> None:
        if process is not None and process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signum)
            forwarded.append(signum)

    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in old_handlers:
        signal.signal(sig, forward)
    try:
        process = subprocess.Popen(
            list(argv),
            cwd=REPO_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
            pass_fds=(lock_fd,),
        )
        return process.wait(), forwarded
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--wait-for-gpu", action="store_true")
    args = parser.parse_args(argv)
    try:
        with lock_controller() as lock_fd:
            state = inspect()
            print(json.dumps(state, indent=2, sort_keys=True), flush=True)
            if state["status"] == "invalid":
                raise ControllerError(str(state["reason"]))
            if args.status:
                if state["status"] != "terminal":
                    validate_inputs()
                return 0
            if state["status"] == "terminal":
                return 0
            validate_inputs()
            argv = command(str(state["action"]))
            print(json.dumps({"command": argv}, indent=2), flush=True)
            if args.dry_run:
                return 0
            if args.wait_for_gpu:
                wait_for_cuda()
            elif not cuda_ready():
                raise ControllerError(
                    "CUDA is unavailable; use --wait-for-gpu without touching the "
                    "formal output directory"
                )
            returncode, forwarded = run_child(argv, lock_fd)
            observed = inspect()
            print(json.dumps(observed, indent=2, sort_keys=True), flush=True)
            if forwarded:
                return 128 + forwarded[-1]
            if observed.get("status") != "terminal":
                raise ControllerError(
                    "training exited without a verified terminal checkpoint "
                    f"(returncode={returncode})"
                )
            return 0
    except (ControllerError, OSError, RuntimeError, ValueError) as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
