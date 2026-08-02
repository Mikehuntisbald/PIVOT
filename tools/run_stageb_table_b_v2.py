#!/usr/bin/env python3
"""Run the additive class-aligned D2m/D3m Table-B v2 panel.

The public CLI is compatible with ``run_stageb_serial_matrix_queue.py``.  All
heavy launcher imports are lazy.  The private ``_bootstrap-main`` entry point
validates and installs the v2 data/scope contract before importing ``main`` or
any training module.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RUNNER_SCHEMA = "pivot.stageb.table_b_v2_runner/v1"
DETACH_SCHEMA = "pivot.stageb.table_b_v2_detached_launch/v1"
STATUS_SCHEMA = "pivot.stageb.table_b_v2_orchestration_status/v1"
BOOTSTRAP_SCHEMA = "pivot.stageb.table_b_v2_training_bootstrap/v1"
VERIFICATION_SCHEMA = "pivot.stageb.table_b_v2_training_verification/v1"
FORMAL_SOURCE_PLAN_NAME = "formal_source_plan.json"
FORMAL_SCOPE_PLAN_NAME = "formal_scope_plan.json"
FORMAL_COMPLETION_ATTESTATION_NAME = "formal_completion_attestation.json"
PHASE_ID = "joint"
SEEDS = (17, 42, 73)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "outputs/paper_cvpr_v1/tn_data_ablation_matched_class_aligned_v2"
)
DEFAULT_ORCHESTRATION_ROOT = (
    REPO_ROOT / "outputs/paper_cvpr_v1/orchestration/table_b_matched_class_aligned_v2"
)
V2_AUDIT = (
    REPO_ROOT
    / "data/ablations/stageb_tn_c2_parent_matched_class_aligned_20260718_v2/audit.json"
)
V2_AUDIT_SHA256 = "5ff62a838a5123d580a72e353147b97bb69e9d7967348b55cba4ccb9ca36cb96"
V1_AUDIT = REPO_ROOT / "data/ablations/stageb_tn_c2_parent_matched_20260717/audit.json"
V1_AUDIT_SHA256 = "ca1c9c581fd78f1fe026397cc127d9b7448c60227b31c5e83148c91e9c61861e"
DATASET_BY_ID = {
    "D2m": "config/datasets_stageb_table_b_d2m_matched_class_aligned_v2_traceable.json",
    "D3m": "config/datasets_stageb_table_b_d3m_matched_class_aligned_v2_proposal_covered.json",
}
CONFIG_BY_ID = {
    "D2m": "config/ablations/cfg_stageb_v24_table_b_d2m_matched.py",
    "D3m": "config/ablations/cfg_stageb_v24_table_b_d3m_matched.py",
}
SCOPE_BY_ID = {
    "D2m": "traceable_counterfactual_edit",
    "D3m": "proposal_covered_verified",
}
_LAUNCHER: Any | None = None
_FORMAL_CONTEXT_OVERRIDE: tuple[dict[str, Any], Mapping[str, Any] | None] | None = None


class TableBV2RunnerError(RuntimeError):
    """Raised when v2 runner evidence is incomplete or inconsistent."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _contract() -> Any:
    return importlib.import_module("util.stage_b_table_b_v2_contract")


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=2048)
def _sha256_identity_cached(
    path_text: str, size_bytes: int, mtime_ns: int, ctime_ns: int
) -> str:
    del size_bytes, mtime_ns, ctime_ns
    return _sha256_file(Path(path_text))


def _stable_sha256(path: Path) -> str:
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return _sha256_identity_cached(
        str(path),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            encoding="ascii",
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TableBV2RunnerError(f"invalid {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise TableBV2RunnerError(f"{label} must be a JSON object")
    return value


def _file_record(path: Path, *, role: str | None = None) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    record: dict[str, Any] = {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if role is not None:
        record["role"] = role
    return record


def _verify_file_record(record: Any, expected: Path | None, *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise TableBV2RunnerError(f"{label} record is missing")
    raw = record.get("path")
    if not isinstance(raw, str) or not raw.strip():
        raise TableBV2RunnerError(f"{label} path is invalid")
    path = Path(raw).expanduser().resolve(strict=True)
    if expected is not None and path != expected.resolve(strict=True):
        raise TableBV2RunnerError(f"{label} path drifted")
    if record.get("sha256") != _stable_sha256(path):
        raise TableBV2RunnerError(f"{label} SHA-256 drifted")
    if record.get("size_bytes") != path.stat().st_size:
        raise TableBV2RunnerError(f"{label} size drifted")
    return path


def _read_process_identity(pid: int) -> dict[str, Any]:
    """Capture Linux PID identity so a later PID reuse cannot look alive."""

    proc = Path("/proc") / str(pid)
    try:
        stat_text = (proc / "stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"available": False, "pid": pid, "reason": "proc_entry_missing"}
    except OSError as error:
        return {
            "available": False,
            "pid": pid,
            "reason": f"proc_stat_unreadable:{type(error).__name__}",
        }
    closing = stat_text.rfind(")")
    fields = stat_text[closing + 2 :].split() if closing >= 0 else []
    if len(fields) <= 19:
        return {"available": False, "pid": pid, "reason": "proc_stat_malformed"}
    try:
        start_ticks = int(fields[19])
    except ValueError:
        return {"available": False, "pid": pid, "reason": "proc_start_malformed"}
    try:
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        ).strip()
    except OSError:
        command = ""
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
    except OSError:
        boot_id = ""
    return {
        "available": True,
        "pid": pid,
        "state": fields[0],
        "start_time_ticks": start_ticks,
        "boot_id": boot_id or None,
        "command": command,
    }


def _same_process_identity(expected: Any, observed: Any) -> bool:
    return bool(
        isinstance(expected, Mapping)
        and isinstance(observed, Mapping)
        and expected.get("available") is True
        and observed.get("available") is True
        and expected.get("pid") == observed.get("pid")
        and expected.get("start_time_ticks") == observed.get("start_time_ticks")
        and (
            not expected.get("boot_id")
            or not observed.get("boot_id")
            or expected.get("boot_id") == observed.get("boot_id")
        )
    )


def _formal_execution_context(
    runtime: Any,
) -> tuple[dict[str, Any], Mapping[str, Any]] | None:
    contract = _contract()
    if _FORMAL_CONTEXT_OVERRIDE is not None:
        context, scope_plan = _FORMAL_CONTEXT_OVERRIDE
        return dict(context), {} if scope_plan is None else scope_plan
    source_raw = os.environ.get(contract.FORMAL_SOURCE_PLAN_ENV, "").strip()
    scope_raw = os.environ.get(contract.FORMAL_SCOPE_PLAN_ENV, "").strip()
    if not source_raw and not scope_raw:
        return None
    if not source_raw or not scope_raw:
        raise TableBV2RunnerError("formal v2 execution requires both source and scope plans")
    source_sha = os.environ.get(contract.FORMAL_SOURCE_PLAN_SHA_ENV, "").strip()
    scope_sha = os.environ.get(contract.FORMAL_SCOPE_PLAN_SHA_ENV, "").strip()
    try:
        source_path = Path(source_raw).expanduser().resolve(strict=True)
        scope_path = Path(scope_raw).expanduser().resolve(strict=True)
        source_plan = contract.validate_formal_source_plan(
            source_path, expected_file_sha256=source_sha
        )
        scope_plan = contract.validate_formal_scope_plan(
            scope_path, expected_file_sha256=scope_sha
        )
    except (OSError, ValueError, contract.TableBContractError) as error:
        raise TableBV2RunnerError(f"formal v2 plan validation failed: {error}") from error
    planned_runtime = source_plan["runtime"]
    runtime_checks = {
        "python": str(Path(runtime.python).resolve(strict=True)),
        "batch_size": runtime.batch_size,
        "total_train_iters": runtime.total_train_iters,
        "iter_checkpoint_interval": runtime.iter_checkpoint_interval,
        "output_root": str(Path(runtime.tn_output_root).resolve()),
        "stage_a_init": str(Path(runtime.stage_a_init).resolve(strict=True)),
        "stage_a_init_sha256": _sha256_file(Path(runtime.stage_a_init)),
        "scorer_warmstart": str(
            Path(runtime.scorer_warmstart).resolve(strict=True)
        ),
        "scorer_warmstart_sha256": _sha256_file(Path(runtime.scorer_warmstart)),
    }
    if any(planned_runtime.get(key) != value for key, value in runtime_checks.items()):
        raise TableBV2RunnerError("runtime differs from the sealed formal v2 source plan")
    context = contract.formal_context_from_source_plan(
        source_plan, source_plan_path=source_path
    )
    return context, scope_plan


def _scope_for(
    runtime: Any, row: Any, seed: int, phase: Any
) -> tuple[dict[str, Any], str]:
    contract = _contract()
    formal = _formal_execution_context(runtime)
    context = formal[0] if formal is not None else None
    binding = contract.build_scope_binding(
        table_b_id=row.row_id,
        seed=seed,
        phase_id=phase.phase_id,
        dataset_path=(REPO_ROOT / row.dataset).resolve(strict=True),
        config_path=(REPO_ROOT / phase.config).resolve(strict=True),
        runner_path=Path(__file__).resolve(strict=True),
        execution_context=context,
    )
    binding_sha = contract.canonical_sha256(binding)
    if formal is not None and formal[1]:
        expected = formal[1].get("runs", {}).get(f"{row.row_id}:{seed}")
        if not isinstance(expected, Mapping) or expected.get("scope_sha256") != binding_sha:
            raise TableBV2RunnerError("computed v2 scope differs from the predeclared scope plan")
    return binding, binding_sha


def _validate_base_config(launcher: Any, row: Any, phase: Any) -> dict[str, Any]:
    """Validate the immutable v1 base leaf before applying explicit v2 overrides."""

    path = (REPO_ROOT / phase.config).resolve(strict=True)
    config = launcher.runpy.run_path(str(path))
    launcher._validate_common_config(config, label=f"{row.row_id}/{phase.phase_id}")
    dependencies = launcher.token_launcher._config_dependencies(path)
    if "cfg_stageb_v19_full_text_base_plus_gate.py" not in {
        dependency.name for dependency in dependencies
    }:
        raise TableBV2RunnerError("v2 base config lost the v19 objective implementation")
    expected = {
        "stage_b_v23_ablation_table": "B",
        "stage_b_v23_table_id": row.row_id,
        "stage_b_v23_objective_contract": "v19_base_plus_gate_acc50_hardneg_v21_l4",
        "stage_b_v23_tn_token_provenance_contract": "disabled_uniformly_D2m_D3m",
        "stage_b_v19_allow_scope_labeled_tn_ablation": True,
        "stage_b_v19_table_b_id": row.row_id,
        "stage_b_v19_table_b_scope_allowlist": [row.tn_scope],
        "stage_b_v19_table_b_audit_sha256": V1_AUDIT_SHA256,
        "stage_b_v24_matched_causal_panel": True,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise TableBV2RunnerError(
            f"{row.row_id} historical base config drifted before the additive v2 override"
        )
    configured_audit = config.get("stage_b_v19_table_b_audit")
    if not isinstance(configured_audit, str) or (
        REPO_ROOT / configured_audit
    ).resolve(strict=True) != V1_AUDIT.resolve(strict=True):
        raise TableBV2RunnerError("historical base config no longer binds the reviewed v1 audit")
    return config


def _configure_launcher(launcher: Any) -> None:
    if getattr(launcher, "_TABLE_B_V2_PATCHED", False):
        return
    launcher.MATCHED_TABLE_B_AUDIT = V2_AUDIT
    launcher.MATCHED_TABLE_B_AUDIT_SHA256 = V2_AUDIT_SHA256
    launcher.DEFAULT_TN_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    launcher.DEFAULT_ORCHESTRATION_ROOT = DEFAULT_ORCHESTRATION_ROOT
    rows = tuple(
        launcher.MatrixRow(
            row_id,
            "B",
            CONFIG_BY_ID[row_id],
            DATASET_BY_ID[row_id],
            tn_scope=SCOPE_BY_ID[row_id],
        )
        for row_id in ("D2m", "D3m")
    )
    launcher.ROWS = rows
    launcher.ROW_BY_ID = {row.row_id: row for row in rows}
    launcher.ROW_BY_ID_CASEFOLD = {row.row_id.casefold(): row for row in rows}

    original_build_command = launcher._build_command
    original_relevant_sources = launcher._relevant_repository_sources
    original_phase_manifest = launcher._phase_manifest
    original_postflight = launcher._perform_postflight
    original_validate_checkpoint = launcher._validate_checkpoint_metadata
    original_build_manifest = launcher.build_manifest

    def validate_phase_config(row: Any, phase: Any) -> dict[str, Any]:
        if row.row_id not in SCOPE_BY_ID or phase.phase_id != PHASE_ID:
            raise TableBV2RunnerError("v2 runner accepts only D2m/D3m joint phases")
        return _validate_base_config(launcher, row, phase)

    def build_command(
        runtime: Any,
        row: Any,
        seed: int,
        phase: Any,
        output_dir: Path,
        *,
        rank_checkpoint: Path | None,
    ) -> list[str]:
        command = original_build_command(
            runtime,
            row,
            seed,
            phase,
            output_dir,
            rank_checkpoint=rank_checkpoint,
        )
        binding, binding_sha = _scope_for(runtime, row, seed, phase)
        launch_manifest = output_dir / "launch_manifest.json"
        main_args = command[2:]
        main_args.extend(
            [
                f"stage_b_v19_table_b_audit={V2_AUDIT.relative_to(REPO_ROOT)}",
                f"stage_b_v19_table_b_audit_sha256={V2_AUDIT_SHA256}",
                f"stage_b_v2_scope_contract_sha256={binding_sha}",
                f"stage_b_v2_phase_id={PHASE_ID}",
                f"stage_b_v2_profile={binding['profile']}",
                "stage_b_v2_training_queue_id="
                + str((binding.get("queue") or {}).get("queue_id", "none")),
                "stage_b_v2_training_queue_plan_sha256="
                + str((binding.get("queue") or {}).get("plan_sha256", "none")),
                "stage_b_v2_formal_source_plan_sha256="
                + str(binding.get("source_plan_semantic_sha256") or "none"),
            ]
        )
        return [
            command[0],
            str(Path(__file__).resolve(strict=True)),
            "_bootstrap-main",
            "--launch-manifest",
            str(launch_manifest),
            "--scope-sha256",
            binding_sha,
            "--",
            *main_args,
        ]

    def relevant_sources() -> list[Path]:
        values = list(original_relevant_sources())
        values.extend(
            [
                Path(__file__).resolve(strict=True),
                (REPO_ROOT / "util/stage_b_table_b_v2_contract.py").resolve(strict=True),
            ]
        )
        return list(dict.fromkeys(values))

    def phase_manifest(*args: Any, **kwargs: Any) -> dict[str, Any]:
        manifest = original_phase_manifest(*args, **kwargs)
        runtime, row, seed, phase, output_dir = args[:5]
        binding, binding_sha = _scope_for(runtime, row, seed, phase)
        owner = _read_process_identity(os.getpid())
        if owner.get("available") is not True:
            raise TableBV2RunnerError("runner process identity is unavailable")
        manifest.update(
            {
                "phase_id": phase.phase_id,
                "profile": binding["profile"],
                "formal_queue": binding.get("queue"),
                "table_b_v2_scope": binding,
                "table_b_v2_scope_sha256": binding_sha,
                "scope_bootstrap_path": str(Path(output_dir) / "scope_bootstrap.json"),
                "runner_owner": {
                    "phase_id": phase.phase_id,
                    "process_identity": owner,
                    "runner": _file_record(Path(__file__), role="table_b_v2_runner_owner"),
                },
                "v2_provenance": {
                    "phase_id": phase.phase_id,
                    "scope_sha256": binding_sha,
                    "profile": binding["profile"],
                    "queue": binding.get("queue"),
                    "source_plan_semantic_sha256": binding.get(
                        "source_plan_semantic_sha256"
                    ),
                    "base_config_is_historical_v1": True,
                    "runtime_scope_override_is_v2": True,
                    "claim_class": binding["claim_class"],
                },
            }
        )
        if binding["profile"] == _contract().FORMAL_PROFILE:
            source_plan_path = Path(str(binding["source_plan"]["path"]))
            manifest["inputs"]["records"].append(
                _file_record(
                    source_plan_path, role="table_b_v2_formal_source_plan"
                )
            )
        return manifest

    def validate_checkpoint(metadata: Mapping[str, Any], **kwargs: Any) -> None:
        original_validate_checkpoint(metadata, **kwargs)
        row = kwargs["row"]
        phase = kwargs["phase"]
        binding, binding_sha = _scope_for(
            kwargs["runtime"], row, kwargs["seed"], phase
        )
        checkpoint_args = metadata.get("args")
        if not isinstance(checkpoint_args, Mapping) or not (
            checkpoint_args.get("stage_b_v2_scope_contract_sha256") == binding_sha
            and checkpoint_args.get("stage_b_v2_phase_id") == phase.phase_id
            and checkpoint_args.get("stage_b_v2_profile") == binding["profile"]
            and checkpoint_args.get("stage_b_v2_training_queue_id")
            == str((binding.get("queue") or {}).get("queue_id", "none"))
            and checkpoint_args.get("stage_b_v2_training_queue_plan_sha256")
            == str((binding.get("queue") or {}).get("plan_sha256", "none"))
            and checkpoint_args.get("stage_b_v2_formal_source_plan_sha256")
            == str(binding.get("source_plan_semantic_sha256") or "none")
            and Path(str(checkpoint_args.get("stage_b_v19_table_b_audit", ""))).as_posix()
            == V2_AUDIT.relative_to(REPO_ROOT).as_posix()
            and checkpoint_args.get("stage_b_v19_table_b_audit_sha256") == V2_AUDIT_SHA256
        ):
            raise TableBV2RunnerError("checkpoint metadata lost its v2 scope/phase binding")

    def postflight(manifest: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        result = original_postflight(manifest, **kwargs)
        scope = manifest.get("table_b_v2_scope")
        scope_sha = manifest.get("table_b_v2_scope_sha256")
        _contract().validate_scope_binding(scope, expected_sha256=scope_sha)
        receipt_path = Path(str(manifest.get("scope_bootstrap_path", ""))).resolve(
            strict=True
        )
        receipt = _read_json(receipt_path, label="v2 bootstrap receipt")
        command = manifest.get("command")
        if not isinstance(command, list) or "--" not in command:
            raise TableBV2RunnerError("v2 launch command lost its bootstrap boundary")
        main_args = command[command.index("--") + 1 :]
        owner = manifest.get("runner_owner")
        owner_identity = (
            owner.get("process_identity") if isinstance(owner, Mapping) else None
        )
        if not isinstance(owner_identity, Mapping):
            raise TableBV2RunnerError("v2 launch lost its runner owner identity")
        _validate_bootstrap_receipt(
            receipt,
            scope_sha256=str(scope_sha),
            phase_id=str(manifest.get("phase_id")),
            launch_manifest=Path(str(manifest["output_dir"])) / "launch_manifest.json",
            scope=scope,
            main_args=main_args,
            runner_owner_identity=owner_identity,
        )
        result.update(
            {
                "table_b_v2_scope": scope,
                "table_b_v2_scope_sha256": scope_sha,
                "profile": scope["profile"],
                "formal_queue": scope.get("queue"),
                "v2_provenance": {
                    "phase_id": manifest.get("phase_id"),
                    "scope_sha256": scope_sha,
                    "profile": scope["profile"],
                    "queue": scope.get("queue"),
                    "source_plan_semantic_sha256": scope.get(
                        "source_plan_semantic_sha256"
                    ),
                    "bootstrap": _file_record(receipt_path, role="table_b_v2_bootstrap"),
                    "scope_established_before_training_imports": True,
                },
            }
        )
        result.setdefault("artifacts", {})["table_b_v2_bootstrap"] = _file_record(
            receipt_path
        )
        return result

    def build_manifest(*args: Any, **kwargs: Any) -> dict[str, Any]:
        manifest = original_build_manifest(*args, **kwargs)
        phase_manifest = manifest["phases"][0]
        scope = phase_manifest["table_b_v2_scope"]
        scope_sha = phase_manifest["table_b_v2_scope_sha256"]
        manifest.update(
            {
                "phase_id": PHASE_ID,
                "profile": scope["profile"],
                "formal_queue": scope.get("queue"),
                "table_b_v2_scope": scope,
                "table_b_v2_scope_sha256": scope_sha,
                "v2_provenance": {
                    "phase_id": PHASE_ID,
                    "scope_sha256": scope_sha,
                    "profile": scope["profile"],
                    "queue": scope.get("queue"),
                    "source_plan_semantic_sha256": scope.get(
                        "source_plan_semantic_sha256"
                    ),
                    "runner_schema": RUNNER_SCHEMA,
                    "queue_compatible_runner": True,
                    "matched_evaluation_adapter": "resolve_for_matched_evaluation",
                },
            }
        )
        return manifest

    marker = '"stage_b_v19_table_b_allow_single_edit_token_provenance",'
    addition = marker + (
        '\n    "stage_b_v2_scope_contract_sha256", "stage_b_v2_phase_id",'
        '\n    "stage_b_v2_profile", "stage_b_v2_training_queue_id",'
        '\n    "stage_b_v2_training_queue_plan_sha256",'
        ' "stage_b_v2_formal_source_plan_sha256",'
    )
    if marker not in launcher._SAFE_CHECKPOINT_INSPECT_SCRIPT:
        raise TableBV2RunnerError("checkpoint inspector field anchor drifted")
    launcher._SAFE_CHECKPOINT_INSPECT_SCRIPT = launcher._SAFE_CHECKPOINT_INSPECT_SCRIPT.replace(
        marker, addition, 1
    )
    launcher._validate_phase_config = validate_phase_config
    launcher._build_command = build_command
    launcher._relevant_repository_sources = relevant_sources
    launcher._phase_manifest = phase_manifest
    launcher._validate_checkpoint_metadata = validate_checkpoint
    launcher._perform_postflight = postflight
    launcher.build_manifest = build_manifest
    launcher._TABLE_B_V2_PATCHED = True


def _launcher() -> Any:
    global _LAUNCHER
    if _LAUNCHER is None:
        os.environ.setdefault(
            "PIVOT_TABLE_B_V2_RUNNER_SCOPE",
            "supplemental_matched_source_policy_sensitivity",
        )
        module_name = "tools._stageb_table_b_v2_legacy_runtime"
        module_path = REPO_ROOT / "tools/run_stageb_paper_ablation_matrices.py"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise TableBV2RunnerError("cannot create isolated v2 launcher runtime")
        launcher = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = launcher
        try:
            spec.loader.exec_module(launcher)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
        _configure_launcher(launcher)
        _LAUNCHER = launcher
    return _LAUNCHER


def _validate_phase_layers(value: Mapping[str, Any], *, scope_sha256: str) -> None:
    if value.get("phase_id") != PHASE_ID:
        raise TableBV2RunnerError("top-level phase_id is not joint")
    if value.get("table_b_v2_scope_sha256") != scope_sha256:
        raise TableBV2RunnerError("top-level v2 scope SHA-256 drifted")
    scope = value.get("table_b_v2_scope")
    _contract().validate_scope_binding(scope, expected_sha256=scope_sha256)
    if not (
        value.get("profile") == scope.get("profile")
        and value.get("formal_queue") == scope.get("queue")
    ):
        raise TableBV2RunnerError("top-level v2 profile/queue binding drifted")
    if scope.get("phase_id") != PHASE_ID or scope.get("evidence", {}).get("phase_id") != PHASE_ID:
        raise TableBV2RunnerError("scope phase_id is inconsistent")
    provenance = value.get("v2_provenance")
    if not (
        isinstance(provenance, Mapping)
        and provenance.get("phase_id") == PHASE_ID
        and provenance.get("scope_sha256") == scope_sha256
        and provenance.get("profile") == scope.get("profile")
        and provenance.get("queue") == scope.get("queue")
        and provenance.get("source_plan_semantic_sha256")
        == scope.get("source_plan_semantic_sha256")
    ):
        raise TableBV2RunnerError("nested v2 provenance phase/scope drifted")


def build_manifest(runtime: Any, row: Any, seed: int, cache: Any) -> dict[str, Any]:
    manifest = _launcher().build_manifest(runtime, row, seed, cache)
    _validate_phase_layers(
        manifest, scope_sha256=str(manifest.get("table_b_v2_scope_sha256"))
    )
    if len(manifest.get("phases", [])) != 1:
        raise TableBV2RunnerError("Table-B v2 requires exactly one phase")
    _validate_phase_layers(
        manifest["phases"][0],
        scope_sha256=str(manifest.get("table_b_v2_scope_sha256")),
    )
    formal = _formal_execution_context(runtime)
    if formal is not None and formal[1]:
        run_id = f"{row.row_id}:{seed}"
        expected = formal[1].get("runs", {}).get(run_id)
        phase = manifest["phases"][0]
        identity = _input_identity(phase.get("inputs", {}).get("records"))
        command = phase.get("command")
        if not (
            isinstance(expected, Mapping)
            and expected.get("scope_sha256")
            == manifest["table_b_v2_scope_sha256"]
            and expected.get("input_identity_sha256")
            == _input_identity_sha256(identity)
            and isinstance(command, list)
            and expected.get("command_sha256")
            == hashlib.sha256("\0".join(command).encode("utf-8")).hexdigest()
            and Path(str(expected.get("output_root", ""))).resolve()
            == Path(manifest["output_dir"]).resolve()
        ):
            raise TableBV2RunnerError(
                "formal v2 manifest differs from its predeclared scope/input/command plan"
            )
    return manifest


def build_formal_planned_manifest(
    runtime: Any,
    row: Any,
    seed: int,
    cache: Any,
    *,
    source_plan_path: Path,
) -> dict[str, Any]:
    """Build one queue-creation manifest before the derived scope plan exists."""

    contract = _contract()
    source_plan_path = Path(source_plan_path).resolve(strict=True)
    source_plan = contract.validate_formal_source_plan(source_plan_path)
    context = contract.formal_context_from_source_plan(
        source_plan, source_plan_path=source_plan_path
    )
    global _FORMAL_CONTEXT_OVERRIDE
    if _FORMAL_CONTEXT_OVERRIDE is not None:
        raise TableBV2RunnerError("nested formal v2 planning contexts are forbidden")
    _FORMAL_CONTEXT_OVERRIDE = (context, None)
    try:
        return build_manifest(runtime, row, seed, cache)
    finally:
        _FORMAL_CONTEXT_OVERRIDE = None


def _require_formal_runtime(runtime: Any) -> dict[str, Any]:
    formal = _formal_execution_context(runtime)
    if formal is None:
        raise TableBV2RunnerError(
            "public v2 training requires the dedicated formal source/scope plans"
        )
    if not formal[1]:
        raise TableBV2RunnerError("formal v2 scope plan is unavailable")
    return formal[0]


def _validate_generic_queue_execution(
    args: argparse.Namespace,
    *,
    context: Mapping[str, Any],
    stage: str,
) -> None:
    queue_runner = importlib.import_module("tools.run_stageb_serial_matrix_queue")
    queue_dir = Path(str(context["queue"]["queue_dir"])).resolve(strict=True)
    queue = queue_runner.load_queue(queue_dir)
    selections = _launcher()._selected_runs(args)
    if len(selections) != 1:
        raise TableBV2RunnerError("formal shared-lease execution accepts one queue item")
    row, seed = selections[0]
    run_id = f"{row.row_id}:{seed}"
    matches = [item for item in queue["items"] if item.get("run_id") == run_id]
    if len(matches) != 1 or matches[0].get("status") not in {
        "reserved",
        "launching",
        "launched",
    }:
        raise TableBV2RunnerError("formal run is not the active shared-lease queue item")
    item = matches[0]
    try:
        queue_runner._ensure_lease(queue, item, create=False)
    except queue_runner.QueueContractError as error:
        raise TableBV2RunnerError(
            f"formal run does not own the shared GPU lease: {error}"
        ) from error
    item_root = queue_runner._item_orchestration_root(queue, item)
    if stage == "detach":
        requested = getattr(args, "orchestration_root", None)
        if requested is None or Path(requested).resolve() != item_root.resolve():
            raise TableBV2RunnerError("formal detach escaped its queue orchestration root")
    elif stage == "run":
        status = _status_path()
        if status is None or status.parent.parent.resolve() != item_root.resolve():
            raise TableBV2RunnerError("formal training child is not owned by its queue item")
    else:
        raise TableBV2RunnerError(f"unknown formal execution stage {stage!r}")


def _status_path() -> Path | None:
    raw = os.environ.get("PIVOT_ORCHESTRATION_STATUS", "").strip()
    return Path(raw).expanduser().resolve(strict=False) if raw else None


def _update_status(path: Path | None, *, status: str, **fields: Any) -> None:
    if path is None:
        return
    payload: dict[str, Any] = {}
    if path.is_file():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = None
        if isinstance(current, dict):
            payload.update(current)
    payload.update(fields)
    phase_value = fields.get("current_phase_id")
    if not isinstance(phase_value, str) or not phase_value:
        phase_value = PHASE_ID
    payload.update(
        {
            "schema": STATUS_SCHEMA,
            "runner_schema": RUNNER_SCHEMA,
            "status": status,
            "phase_id": phase_value,
            "updated_at_utc": _utc_now(),
            "pid": os.getpid(),
            "process_identity": _read_process_identity(os.getpid()),
        }
    )
    _write_json_atomic(path, payload)


def _completed_phase_record(
    launcher: Any,
    phase_manifest: Mapping[str, Any],
    checkpoint: Path,
    output_dir: Path,
    cache: Any,
) -> dict[str, Any]:
    scope_sha = str(phase_manifest["table_b_v2_scope_sha256"])
    return {
        "phase_id": PHASE_ID,
        "status": phase_manifest["status"],
        "output_dir": str(output_dir),
        "checkpoint": launcher.token_launcher._file_record(checkpoint, cache),
        "postflight": launcher.token_launcher._file_record(
            output_dir / "postflight.json", cache
        ),
        "table_b_v2_scope_sha256": scope_sha,
        "profile": phase_manifest["profile"],
        "formal_queue": phase_manifest["formal_queue"],
        "v2_provenance": {
            "phase_id": PHASE_ID,
            "scope_sha256": scope_sha,
            "profile": phase_manifest["profile"],
            "queue": phase_manifest["formal_queue"],
            "source_plan_semantic_sha256": phase_manifest[
                "table_b_v2_scope"
            ].get("source_plan_semantic_sha256"),
        },
    }


def _run_body(args: argparse.Namespace, *, orchestration_status: Path | None) -> int:
    launcher = _launcher()
    selections = launcher._selected_runs(args)
    runtime = launcher.runtime_from_environment()
    context = _require_formal_runtime(runtime)
    _validate_generic_queue_execution(args, context=context, stage="run")
    roots = [launcher.output_directory(runtime, row, seed) for row, seed in selections]
    conflicts = [path for path in roots if path.exists()]
    if conflicts:
        raise FileExistsError(
            "every selected v2 run root must be fresh; existing paths:\n"
            + "\n".join(f"  {path}" for path in conflicts)
        )
    cache = launcher.token_launcher.HashCache()
    planned = [build_manifest(runtime, row, seed, cache) for row, seed in selections]
    run_ids = [f"{row.row_id}:{seed}" for row, seed in selections]
    _update_status(
        orchestration_status,
        status="preflight_passed",
        run_ids=run_ids,
        expected_run_roots=[str(path) for path in roots],
        completed_run_ids=[],
    )
    completed_run_ids: list[str] = []
    for (row, seed), sequence in zip(selections, planned):
        run_id = f"{row.row_id}:{seed}"
        run_root = launcher.output_directory(runtime, row, seed)
        sequence_path = run_root / "sequence_manifest.json"
        sequence.update(status="running", started_at_utc=_utc_now())
        _update_status(
            orchestration_status,
            status="running",
            current_run_id=run_id,
            current_phase_id=PHASE_ID,
            completed_run_ids=completed_run_ids,
        )
        completed_phases: list[dict[str, Any]] = []
        try:
            phase = launcher._phases(runtime, row)[0]
            output_dir = launcher._phase_output(run_root, row, phase)
            phase_manifest, checkpoint = launcher._run_phase(
                runtime=runtime,
                row=row,
                seed=seed,
                phase=phase,
                output_dir=output_dir,
                cache=cache,
                rank_checkpoint=None,
            )
            _validate_phase_layers(
                phase_manifest,
                scope_sha256=str(sequence["table_b_v2_scope_sha256"]),
            )
            completed_phases.append(
                _completed_phase_record(
                    launcher, phase_manifest, checkpoint, output_dir, cache
                )
            )
        except BaseException as error:
            sequence.update(
                status="failed",
                finished_at_utc=_utc_now(),
                completed_phases=completed_phases,
                error=f"{type(error).__name__}: {error}",
            )
            _write_json_atomic(sequence_path, sequence)
            _update_status(
                orchestration_status,
                status="failed",
                current_run_id=run_id,
                current_phase_id=PHASE_ID,
                completed_run_ids=completed_run_ids,
                error=sequence["error"],
            )
            raise
        sequence.update(
            status="completed",
            finished_at_utc=_utc_now(),
            completed_phases=completed_phases,
        )
        _validate_completed_sequence(sequence)
        _write_json_atomic(sequence_path, sequence)
        completed_run_ids.append(run_id)
        _update_status(
            orchestration_status,
            status="running",
            current_run_id=None,
            current_phase_id=None,
            completed_run_ids=completed_run_ids,
        )
    return 0


def _run(args: argparse.Namespace) -> int:
    status_path = _status_path()
    launcher = _launcher()
    selections = launcher._selected_runs(args)
    run_ids = [f"{row.row_id}:{seed}" for row, seed in selections]
    _update_status(
        status_path,
        status="starting",
        run_ids=run_ids,
        started_at_utc=_utc_now(),
    )
    try:
        result = _run_body(args, orchestration_status=status_path)
    except BaseException as error:
        _update_status(
            status_path,
            status="failed",
            finished_at_utc=_utc_now(),
            error=f"{type(error).__name__}: {error}",
        )
        raise
    _update_status(
        status_path,
        status="completed",
        finished_at_utc=_utc_now(),
        current_run_id=None,
        current_phase_id=None,
        completed_run_ids=run_ids,
    )
    return result


def _detach(args: argparse.Namespace) -> int:
    launcher = _launcher()
    selections = launcher._selected_runs(args)
    runtime = launcher.runtime_from_environment()
    context = _require_formal_runtime(runtime)
    _validate_generic_queue_execution(args, context=context, stage="detach")
    run_roots = [launcher.output_directory(runtime, row, seed) for row, seed in selections]
    conflicts = [path for path in run_roots if path.exists()]
    if conflicts:
        raise FileExistsError(
            "every selected v2 run root must be fresh; existing paths:\n"
            + "\n".join(f"  {path}" for path in conflicts)
        )
    cache = launcher.token_launcher.HashCache()
    planned = [build_manifest(runtime, row, seed, cache) for row, seed in selections]
    root = (
        args.orchestration_root
        or Path(os.environ.get("PIVOT_ORCHESTRATION_ROOT", DEFAULT_ORCHESTRATION_ROOT))
    ).expanduser().resolve(strict=False)
    job_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"-pid{os.getpid()}"
    job_dir = root / job_name
    job_dir.mkdir(parents=True, exist_ok=False)
    plans = job_dir / "plans"
    for (row, seed), manifest in zip(selections, planned):
        _write_json_atomic(plans / row.row_id / f"seed{seed}.json", manifest)
    run_ids = [f"{row.row_id}:{seed}" for row, seed in selections]
    child_command = [sys.executable, str(Path(__file__).resolve(strict=True)), "run"]
    for run_id in run_ids:
        child_command.extend(("--run-id", run_id))
    log_path = job_dir / "orchestrator.log"
    status_path = job_dir / "status.json"
    launch_path = job_dir / "launch.json"
    launch: dict[str, Any] = {
        "schema": DETACH_SCHEMA,
        "runner_schema": RUNNER_SCHEMA,
        "status": "prepared",
        "created_at_utc": _utc_now(),
        "job_dir": str(job_dir),
        "run_ids": run_ids,
        "phase_id": PHASE_ID,
        "expected_run_roots": [str(path) for path in run_roots],
        "command": child_command,
        "command_shell": shlex.join(child_command),
        "orchestrator_log": str(log_path),
        "orchestrator_status": str(status_path),
        "plans_dir": str(plans),
        "runner": _file_record(Path(__file__), role="detached_runner"),
        "contract": _file_record(
            REPO_ROOT / "util/stage_b_table_b_v2_contract.py",
            role="table_b_v2_contract",
        ),
        "runtime": {
            "python": str(runtime.python),
            "batch_size": runtime.batch_size,
            "total_train_iters": runtime.total_train_iters,
            "cuda_visible_devices": runtime.cuda_visible_devices,
            "tn_output_root": str(runtime.tn_output_root),
        },
    }
    _write_json_atomic(launch_path, launch)
    _update_status(
        status_path,
        status="prepared",
        job_dir=str(job_dir),
        run_ids=run_ids,
        expected_run_roots=launch["expected_run_roots"],
        completed_run_ids=[],
    )
    environment = dict(os.environ)
    environment["PIVOT_ORCHESTRATION_STATUS"] = str(status_path)
    environment["PYTHONUNBUFFERED"] = "1"
    try:
        with log_path.open("ab", buffering=0) as log_handle:
            process = subprocess.Popen(
                child_command,
                cwd=REPO_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    except BaseException as error:
        launch.update(
            status="spawn_failed",
            spawn_error=f"{type(error).__name__}: {error}",
            finished_at_utc=_utc_now(),
        )
        _write_json_atomic(launch_path, launch)
        _update_status(status_path, status="spawn_failed", error=launch["spawn_error"])
        raise
    child_identity = _read_process_identity(int(process.pid))
    if child_identity.get("available") is not True:
        process.terminate()
        raise TableBV2RunnerError("detached child process identity is unavailable")
    launch.update(
        status="launched",
        launched_at_utc=_utc_now(),
        child_pid=int(process.pid),
        child_process_identity=child_identity,
        child_start_new_session=True,
        stdin="DEVNULL",
        stdout_stderr=str(log_path),
    )
    _write_json_atomic(launch_path, launch)
    print(
        json.dumps(
            {
                "status": "launched",
                "pid": int(process.pid),
                "job_dir": str(job_dir),
                "status_file": str(status_path),
                "log_file": str(log_path),
            },
            sort_keys=True,
        )
    )
    return 0


def _validate_bootstrap_receipt(
    receipt: Mapping[str, Any],
    *,
    scope_sha256: str,
    phase_id: str,
    launch_manifest: Path,
    scope: Mapping[str, Any],
    main_args: Sequence[str],
    runner_owner_identity: Mapping[str, Any],
) -> None:
    if not (
        receipt.get("schema") == BOOTSTRAP_SCHEMA
        and receipt.get("status") == "scope_established_before_training_imports"
        and receipt.get("phase_id") == phase_id == PHASE_ID
        and receipt.get("scope_sha256") == scope_sha256
        and Path(str(receipt.get("launch_manifest", ""))).resolve(strict=True)
        == launch_manifest.resolve(strict=True)
        and receipt.get("imports_absent_before_scope")
        == list(_contract().TRAINING_IMPORT_NAMES)
        and receipt.get("contract_installed_as_legacy_import") is True
        and receipt.get("profile") == scope.get("profile")
        and receipt.get("formal_queue") == scope.get("queue")
        and receipt.get("source_plan_semantic_sha256")
        == scope.get("source_plan_semantic_sha256")
        and receipt.get("main_argv_sha256")
        == hashlib.sha256("\0".join(main_args).encode("utf-8")).hexdigest()
        and receipt.get("runner_owner_process_identity")
        == dict(runner_owner_identity)
        and isinstance(receipt.get("bootstrap_process_identity"), Mapping)
        and receipt["bootstrap_process_identity"].get("available") is True
        and receipt.get("evidence")
        == {
            "phase_id": PHASE_ID,
            "scope_sha256": scope_sha256,
            "profile": scope.get("profile"),
            "queue": scope.get("queue"),
            "source_plan_semantic_sha256": scope.get(
                "source_plan_semantic_sha256"
            ),
        }
    ):
        raise TableBV2RunnerError("v2 bootstrap receipt contract failed")


def _split_bootstrap_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    try:
        boundary = list(argv).index("--")
    except ValueError as error:
        raise TableBV2RunnerError("_bootstrap-main requires a '--' argument boundary") from error
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--scope-sha256", required=True)
    special = parser.parse_args(list(argv[:boundary]))
    main_args = list(argv[boundary + 1 :])
    if not main_args:
        raise TableBV2RunnerError("_bootstrap-main received no main.py arguments")
    return special, main_args


def _bootstrap_main(argv: Sequence[str]) -> int:
    special, main_args = _split_bootstrap_args(argv)
    launch_path = special.launch_manifest.expanduser().resolve(strict=True)
    launch = _read_json(launch_path, label="v2 phase launch manifest")
    scope = launch.get("table_b_v2_scope")
    scope_sha = launch.get("table_b_v2_scope_sha256")
    if not (
        launch.get("status") == "running"
        and launch.get("phase_id") == PHASE_ID
        and scope_sha == special.scope_sha256
    ):
        raise TableBV2RunnerError("v2 bootstrap launch/scope identity mismatch")
    contract = _contract()
    contract.validate_scope_binding(scope, expected_sha256=special.scope_sha256)
    owner = launch.get("runner_owner")
    expected_owner = owner.get("process_identity") if isinstance(owner, Mapping) else None
    if not (
        isinstance(owner, Mapping)
        and owner.get("phase_id") == PHASE_ID
        and isinstance(expected_owner, Mapping)
        and expected_owner.get("pid") == os.getppid()
        and _same_process_identity(expected_owner, _read_process_identity(os.getppid()))
    ):
        raise TableBV2RunnerError("training child is not owned by the bound v2 runner process")
    absent = [name for name in contract.TRAINING_IMPORT_NAMES if name not in sys.modules]
    if absent != list(contract.TRAINING_IMPORT_NAMES):
        raise TableBV2RunnerError("training stack was imported before the v2 scope guard")
    contract.establish_process_scope(scope, special.scope_sha256)
    contract.install_as_training_contract()
    receipt = {
        "schema": BOOTSTRAP_SCHEMA,
        "status": "scope_established_before_training_imports",
        "created_at_utc": _utc_now(),
        "phase_id": PHASE_ID,
        "scope_sha256": special.scope_sha256,
        "profile": scope["profile"],
        "formal_queue": scope.get("queue"),
        "source_plan_semantic_sha256": scope.get(
            "source_plan_semantic_sha256"
        ),
        "launch_manifest": str(launch_path),
        "bootstrap_process_identity": _read_process_identity(os.getpid()),
        "runner_owner_process_identity": dict(expected_owner),
        "imports_absent_before_scope": absent,
        "contract_installed_as_legacy_import": (
            sys.modules.get("util.stage_b_table_b_contract") is contract
        ),
        "main_argv_sha256": hashlib.sha256("\0".join(main_args).encode("utf-8")).hexdigest(),
        "evidence": {
            "phase_id": PHASE_ID,
            "scope_sha256": special.scope_sha256,
            "profile": scope["profile"],
            "queue": scope.get("queue"),
            "source_plan_semantic_sha256": scope.get(
                "source_plan_semantic_sha256"
            ),
        },
    }
    receipt_path = launch_path.parent / "scope_bootstrap.json"
    _write_json_atomic(receipt_path, receipt)
    _validate_bootstrap_receipt(
        receipt,
        scope_sha256=special.scope_sha256,
        phase_id=PHASE_ID,
        launch_manifest=launch_path,
        scope=scope,
        main_args=main_args,
        runner_owner_identity=expected_owner,
    )

    training_main = importlib.import_module("main")
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[training_main.get_args_parser()]
    )
    args = parser.parse_args(main_args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    result = training_main.main(args)
    return 0 if result is None else int(result)


def _validate_completed_sequence(sequence: Mapping[str, Any]) -> None:
    if not (
        sequence.get("schema") == "pivot.stageb.paper_ablation_run_launch/v1"
        and sequence.get("status") == "completed"
        and sequence.get("phase_id") == PHASE_ID
    ):
        raise TableBV2RunnerError("v2 sequence is not explicitly completed/joint")
    scope_sha = sequence.get("table_b_v2_scope_sha256")
    if not isinstance(scope_sha, str):
        raise TableBV2RunnerError("v2 sequence lacks scope SHA-256")
    _validate_phase_layers(sequence, scope_sha256=scope_sha)
    scope = sequence["table_b_v2_scope"]
    row = sequence.get("row")
    run_id = sequence.get("run_id")
    seed = sequence.get("seed")
    if not (
        isinstance(row, Mapping)
        and row.get("row_id") == scope.get("table_b_id")
        and seed == scope.get("seed")
        and run_id == f"{scope.get('table_b_id')}:{scope.get('seed')}"
    ):
        raise TableBV2RunnerError("sequence row/seed/run ID differs from its v2 scope")
    if scope.get("profile") == _contract().FORMAL_PROFILE:
        expected_budget = {
            "batch_size": _contract().FORMAL_BATCH_SIZE,
            "optimizer_updates": _contract().FORMAL_TRAIN_UPDATES,
            "s3_probe_updates_excluded": 0,
            "contributing_phase_updates": {
                PHASE_ID: _contract().FORMAL_TRAIN_UPDATES
            },
        }
        if sequence.get("equal_budget_contract") != expected_budget:
            raise TableBV2RunnerError("formal v2 sequence is not B40/U1000")
    planned = sequence.get("phases")
    completed = sequence.get("completed_phases")
    if not (isinstance(planned, list) and len(planned) == 1 and isinstance(completed, list) and len(completed) == 1):
        raise TableBV2RunnerError("v2 sequence must have one planned/completed phase")
    _validate_phase_layers(planned[0], scope_sha256=scope_sha)
    completed_phase = completed[0]
    provenance = completed_phase.get("v2_provenance") if isinstance(completed_phase, Mapping) else None
    if not (
        isinstance(completed_phase, Mapping)
        and completed_phase.get("phase_id") == PHASE_ID
        and completed_phase.get("status") == "completed"
        and completed_phase.get("table_b_v2_scope_sha256") == scope_sha
        and completed_phase.get("profile") == scope.get("profile")
        and completed_phase.get("formal_queue") == scope.get("queue")
        and isinstance(provenance, Mapping)
        and provenance.get("phase_id") == PHASE_ID
        and provenance.get("scope_sha256") == scope_sha
        and provenance.get("profile") == scope.get("profile")
        and provenance.get("queue") == scope.get("queue")
        and provenance.get("source_plan_semantic_sha256")
        == scope.get("source_plan_semantic_sha256")
    ):
        raise TableBV2RunnerError("completed phase lost its joint v2 provenance")


def _verify_queue_binding(queue_dir: Path, *, run_id: str, run_root: Path) -> dict[str, Any]:
    queue_runner = importlib.import_module("tools.run_stageb_serial_matrix_queue")
    try:
        queue = queue_runner.load_queue(queue_dir)
        report = queue_runner.verify_queue(queue_dir)
    except Exception as error:
        raise TableBV2RunnerError(f"v2 queue verification failed: {error}") from error
    plan = queue.get("plan")
    if not (
        queue.get("status") == "completed"
        and report.get("status") == "passed"
        and isinstance(plan, Mapping)
        and queue.get("plan_sha256") == report.get("plan_sha256")
        and plan.get("queue_id") == report.get("queue_id")
    ):
        raise TableBV2RunnerError("v2 queue identity/completion is not exact")
    paper_runner = plan.get("runners", {}).get("paper")
    if not (
        isinstance(paper_runner, Mapping)
        and Path(str(paper_runner.get("path", ""))).resolve(strict=True)
        == Path(__file__).resolve(strict=True)
        and paper_runner.get("sha256") == _sha256_file(Path(__file__))
    ):
        raise TableBV2RunnerError("queue does not bind this exact v2 runner source")

    def selected(values: Any) -> list[Mapping[str, Any]]:
        return [
            value
            for value in values or []
            if isinstance(value, Mapping)
            and value.get("run_id") == run_id
            and value.get("runner") == "paper"
        ]

    planned, observed, verified = (
        selected(plan.get("items")),
        selected(queue.get("items")),
        selected(report.get("verified_items")),
    )
    if not len(planned) == len(observed) == len(verified) == 1:
        raise TableBV2RunnerError("queue does not uniquely bind the v2 run")
    if observed[0].get("status") != "completed" or Path(
        str(verified[0].get("output_root", ""))
    ).resolve(strict=True) != run_root.resolve(strict=True):
        raise TableBV2RunnerError("queue completed item/output root drifted")
    return {
        "queue_id": plan["queue_id"],
        "plan_sha256": queue["plan_sha256"],
        "manifest": _file_record(queue_dir / "queue.json", role="table_b_v2_training_queue"),
    }


def _input_identity(records: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(records, list) or not records:
        raise TableBV2RunnerError("v2 phase has no input records")
    result: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TableBV2RunnerError(f"v2 input record {index} is malformed")
        path = _verify_file_record(record, None, label=f"v2 input record {index}")
        key = str(path)
        identity = {
            "path": key,
            "sha256": record.get("sha256"),
            "size_bytes": record.get("size_bytes"),
        }
        previous = result.setdefault(key, identity)
        if previous != identity:
            raise TableBV2RunnerError("v2 inputs contain conflicting file identities")
    return result


def _input_identity_sha256(identity: Mapping[str, Mapping[str, Any]]) -> str:
    return _canonical_sha256([dict(identity[path]) for path in sorted(identity)])


def _verify_input_rehash(
    phase_manifest: Mapping[str, Any], postflight: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    inputs = phase_manifest.get("inputs")
    identity = _input_identity(
        inputs.get("records") if isinstance(inputs, Mapping) else None
    )
    rehash = postflight.get("input_rehash")
    records = rehash.get("records") if isinstance(rehash, Mapping) else None
    if not (
        isinstance(rehash, Mapping)
        and rehash.get("status") == "passed"
        and rehash.get("algorithm") == "sha256"
        and rehash.get("unique_input_count") == len(identity)
        and isinstance(records, list)
        and len(records) == len(identity)
    ):
        raise TableBV2RunnerError("v2 input rehash summary is incomplete")
    observed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TableBV2RunnerError("v2 input rehash record is malformed")
        path = str(Path(str(record.get("path", ""))).resolve(strict=True))
        expected = identity.get(path)
        if not (
            expected is not None
            and record.get("passed") is True
            and record.get("expected_sha256") == expected["sha256"]
            and record.get("observed_sha256") == expected["sha256"]
            and record.get("observed_size_bytes") == expected["size_bytes"]
        ):
            raise TableBV2RunnerError(f"v2 input rehash drifted for {path}")
        observed[path] = expected
    if observed != identity:
        raise TableBV2RunnerError("v2 input rehash path set differs from launch inputs")
    return identity


def _verify_formal_checkpoint_metadata(
    *,
    sequence: Mapping[str, Any],
    phase_manifest: Mapping[str, Any],
    postflight: Mapping[str, Any],
    scope: Mapping[str, Any],
    run_root: Path,
) -> None:
    metadata = postflight.get("checkpoint_metadata")
    args = metadata.get("args") if isinstance(metadata, Mapping) else None
    queue = scope.get("queue")
    expected = {
        "seed": scope["seed"],
        "batch_size": _contract().FORMAL_BATCH_SIZE,
        "max_train_iters": _contract().FORMAL_TRAIN_UPDATES,
        "iter_checkpoint_interval": _contract().FORMAL_CHECKPOINT_INTERVAL,
        "stage_b_v2_scope_contract_sha256": phase_manifest[
            "table_b_v2_scope_sha256"
        ],
        "stage_b_v2_phase_id": PHASE_ID,
        "stage_b_v2_profile": _contract().FORMAL_PROFILE,
        "stage_b_v2_training_queue_id": queue["queue_id"],
        "stage_b_v2_training_queue_plan_sha256": queue["plan_sha256"],
        "stage_b_v2_formal_source_plan_sha256": scope[
            "source_plan_semantic_sha256"
        ],
        "stage_b_v19_table_b_audit_sha256": V2_AUDIT_SHA256,
    }
    if not isinstance(args, Mapping) or any(args.get(key) != value for key, value in expected.items()):
        raise TableBV2RunnerError("formal v2 checkpoint metadata drifted")
    expected_config = Path(str(phase_manifest["phase"]["config"]))
    if not expected_config.is_absolute():
        expected_config = REPO_ROOT / expected_config
    configured_audit = Path(str(args.get("stage_b_v19_table_b_audit", "")))
    if not configured_audit.is_absolute():
        configured_audit = REPO_ROOT / configured_audit
    configured_config = Path(str(args.get("config_file", "")))
    if not configured_config.is_absolute():
        configured_config = REPO_ROOT / configured_config
    if not (
        Path(str(args.get("output_dir", ""))).resolve(strict=True) == run_root
        and configured_config.resolve(strict=True) == expected_config.resolve(strict=True)
        and configured_audit.resolve(strict=True) == V2_AUDIT.resolve(strict=True)
        and sequence.get("seed") == scope["seed"]
    ):
        raise TableBV2RunnerError("formal v2 checkpoint path/seed binding drifted")


def _verify_bootstrap_replay(
    *,
    run_root: Path,
    phase_manifest: Mapping[str, Any],
    postflight: Mapping[str, Any],
    scope: Mapping[str, Any],
) -> dict[str, Any]:
    bootstrap_path = (run_root / "scope_bootstrap.json").resolve(strict=True)
    if Path(str(phase_manifest.get("scope_bootstrap_path", ""))).resolve(
        strict=True
    ) != bootstrap_path:
        raise TableBV2RunnerError("v2 bootstrap path is not canonical")
    receipt = _read_json(bootstrap_path, label="v2 bootstrap receipt")
    command = phase_manifest.get("command")
    if not isinstance(command, list) or "--" not in command:
        raise TableBV2RunnerError("v2 bootstrap command boundary is missing")
    owner = phase_manifest.get("runner_owner")
    owner_identity = owner.get("process_identity") if isinstance(owner, Mapping) else None
    if not isinstance(owner_identity, Mapping):
        raise TableBV2RunnerError("v2 bootstrap runner owner is missing")
    _validate_bootstrap_receipt(
        receipt,
        scope_sha256=str(phase_manifest["table_b_v2_scope_sha256"]),
        phase_id=PHASE_ID,
        launch_manifest=run_root / "launch_manifest.json",
        scope=scope,
        main_args=command[command.index("--") + 1 :],
        runner_owner_identity=owner_identity,
    )
    provenance = postflight.get("v2_provenance")
    artifacts = postflight.get("artifacts")
    bootstrap_record = provenance.get("bootstrap") if isinstance(provenance, Mapping) else None
    artifact_record = (
        artifacts.get("table_b_v2_bootstrap") if isinstance(artifacts, Mapping) else None
    )
    _verify_file_record(bootstrap_record, bootstrap_path, label="v2 bootstrap provenance")
    _verify_file_record(artifact_record, bootstrap_path, label="v2 bootstrap artifact")
    return receipt


def verify_completed_run(
    run_root: Path,
    *,
    training_queue_dir: Path | None,
    require_queue: bool = True,
    require_formal: bool = True,
) -> dict[str, Any]:
    run_root = Path(run_root).expanduser().resolve(strict=True)
    sequence_path = (run_root / "sequence_manifest.json").resolve(strict=True)
    sequence = _read_json(sequence_path, label="v2 sequence manifest")
    _validate_completed_sequence(sequence)
    run_id = sequence.get("run_id")
    if not isinstance(run_id, str) or run_id != f"{sequence.get('row', {}).get('row_id')}:{sequence.get('seed')}":
        raise TableBV2RunnerError("v2 sequence run ID/row/seed mismatch")
    scope_sha = str(sequence["table_b_v2_scope_sha256"])
    scope = sequence["table_b_v2_scope"]
    if require_formal and scope.get("profile") != _contract().FORMAL_PROFILE:
        raise TableBV2RunnerError("formal v2 verification rejects nonformal training")
    phase_manifest_path = run_root / "launch_manifest.json"
    phase_manifest = _read_json(phase_manifest_path, label="v2 phase launch")
    if not (
        phase_manifest.get("schema") == "pivot.stageb.paper_ablation_phase_launch/v1"
        and phase_manifest.get("status") == "completed"
        and phase_manifest.get("returncode") == 0
        and phase_manifest.get("run_id") == run_id
    ):
        raise TableBV2RunnerError("v2 phase launch is not completed")
    _validate_phase_layers(phase_manifest, scope_sha256=scope_sha)
    if phase_manifest.get("table_b_v2_scope") != scope:
        raise TableBV2RunnerError("sequence and phase v2 scopes differ")
    postflight_path = run_root / "postflight.json"
    postflight = _read_json(postflight_path, label="v2 postflight")
    if not (
        postflight.get("schema") == "pivot.stageb.paper_ablation_phase_postflight/v1"
        and postflight.get("status") == "passed"
        and postflight.get("run_id") == run_id
    ):
        raise TableBV2RunnerError("v2 postflight did not pass")
    _validate_phase_layers(postflight, scope_sha256=scope_sha)
    if postflight.get("table_b_v2_scope") != scope:
        raise TableBV2RunnerError("sequence and postflight v2 scopes differ")
    embedded = phase_manifest.get("postflight")
    if not isinstance(embedded, Mapping) or dict(embedded) != postflight:
        raise TableBV2RunnerError("embedded and persisted v2 postflight differ")
    _verify_file_record(
        phase_manifest.get("postflight_artifact"), postflight_path, label="v2 postflight"
    )
    checkpoint = run_root / "checkpoint_iter.pth"
    completed = sequence["completed_phases"][0]
    _verify_file_record(completed.get("checkpoint"), checkpoint, label="v2 checkpoint")
    _verify_file_record(
        completed.get("postflight"), postflight_path, label="v2 sequence postflight"
    )
    bootstrap = _verify_bootstrap_replay(
        run_root=run_root,
        phase_manifest=phase_manifest,
        postflight=postflight,
        scope=scope,
    )
    input_identity = _verify_input_rehash(phase_manifest, postflight)
    if scope.get("profile") == _contract().FORMAL_PROFILE:
        _verify_formal_checkpoint_metadata(
            sequence=sequence,
            phase_manifest=phase_manifest,
            postflight=postflight,
            scope=scope,
            run_root=run_root,
        )
    queue = None
    if training_queue_dir is not None:
        queue = _verify_queue_binding(
            Path(training_queue_dir).expanduser().resolve(strict=True),
            run_id=run_id,
            run_root=run_root,
        )
    elif require_queue:
        raise TableBV2RunnerError("formal v2 resolution requires an explicit training queue")
    source_plan_record = None
    scope_plan_record = None
    if scope.get("profile") == _contract().FORMAL_PROFILE:
        if training_queue_dir is None:
            raise TableBV2RunnerError("formal v2 run lacks its dedicated queue directory")
        queue_dir = Path(training_queue_dir).expanduser().resolve(strict=True)
        source_plan_path = (queue_dir / FORMAL_SOURCE_PLAN_NAME).resolve(strict=True)
        scope_plan_path = (queue_dir / FORMAL_SCOPE_PLAN_NAME).resolve(strict=True)
        source_plan = _contract().validate_formal_source_plan(source_plan_path)
        scope_plan = _contract().validate_formal_scope_plan(scope_plan_path)
        expected_run = scope_plan["runs"].get(run_id)
        if not (
            scope["source_plan"]["path"] == str(source_plan_path)
            and scope["source_plan_semantic_sha256"]
            == source_plan["semantic_sha256"]
            and scope["queue"]["queue_id"] == source_plan["queue"]["queue_id"]
            and scope["queue"]["plan_sha256"]
            == source_plan["queue"]["plan_sha256"]
            and isinstance(expected_run, Mapping)
            and expected_run.get("scope_sha256") == scope_sha
            and expected_run.get("input_identity_sha256")
            == _input_identity_sha256(input_identity)
            and expected_run.get("command_sha256")
            == hashlib.sha256(
                "\0".join(phase_manifest["command"]).encode("utf-8")
            ).hexdigest()
        ):
            raise TableBV2RunnerError("formal v2 run differs from its predeclared plans")
        source_plan_record = _file_record(
            source_plan_path, role="table_b_v2_formal_source_plan"
        )
        scope_plan_record = _file_record(
            scope_plan_path, role="table_b_v2_formal_scope_plan"
        )
    return {
        "schema": VERIFICATION_SCHEMA,
        "status": "passed",
        "run_id": run_id,
        "phase_id": PHASE_ID,
        "scope_sha256": scope_sha,
        "profile": scope["profile"],
        "run_root": str(run_root),
        "sequence": _file_record(sequence_path, role="table_b_v2_sequence"),
        "phase_launch": _file_record(phase_manifest_path, role="table_b_v2_phase_launch"),
        "postflight": _file_record(postflight_path, role="table_b_v2_postflight"),
        "checkpoint": _file_record(checkpoint, role="table_b_v2_checkpoint"),
        "bootstrap": _file_record(
            run_root / "scope_bootstrap.json", role="table_b_v2_bootstrap"
        ),
        "input_identity_sha256": _input_identity_sha256(input_identity),
        "input_count": len(input_identity),
        "formal_source_plan": source_plan_record,
        "formal_scope_plan": scope_plan_record,
        "training_queue": queue,
    }


def resolve_for_matched_evaluation(
    root: Path,
    cache: Any,
    *,
    training_phase: str = "final",
    training_queue_dir: Path,
) -> Any:
    """Strict adapter for ``run_stageb_table_b_matched_evaluations``."""

    if training_phase != "final":
        raise TableBV2RunnerError("Table-B v2 matched evaluation accepts final phase only")
    verification = verify_completed_run(
        root, training_queue_dir=training_queue_dir, require_queue=True
    )
    formal_queue = importlib.import_module("tools.run_stageb_table_b_v2_queue")
    formal_evidence = formal_queue.formal_evaluation_evidence(
        training_queue_dir,
        run_id=str(verification["run_id"]),
        run_root=Path(root),
    )
    evaluator = importlib.import_module("tools.run_stageb_paper_evaluations")
    source = evaluator._resolve_paper_source(
        root,
        cache,
        training_phase="final",
        training_queue_dir=training_queue_dir,
        allow_nonformal_fixture=True,
    )
    evidence_paths = tuple(
        Path(str(formal_evidence[key]["path"])).resolve(strict=True)
        for key in ("source_plan", "scope_plan", "completion_attestation")
    )
    return replace(
        source,
        training_data=tuple(dict.fromkeys((*source.training_data, *evidence_paths))),
        formal_contract_id=_contract().FORMAL_PROFILE,
        matrix_validation_only=True,
    )


def matched_evaluation_resolver(training_queue_dir: Path) -> Callable[..., Any]:
    queue = Path(training_queue_dir).expanduser().resolve(strict=True)

    def resolve(root: Path, cache: Any, *, training_phase: str = "final") -> Any:
        return resolve_for_matched_evaluation(
            root,
            cache,
            training_phase=training_phase,
            training_queue_dir=queue,
        )

    return resolve


def _list_rows(*, as_json: bool) -> int:
    launcher = _launcher()
    payload = {
        "schema": RUNNER_SCHEMA,
        "rows": [asdict(row) for row in launcher.ROWS],
        "seeds": list(SEEDS),
        "run_ids": [f"{row.row_id}:{seed}" for row in launcher.ROWS for seed in SEEDS],
        "phases": [PHASE_ID],
        "claim_class": "supplemental_matched_source_policy_sensitivity",
    }
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for row in launcher.ROWS:
            print(
                f"{row.row_id}: phase={PHASE_ID}, scope={row.tn_scope}, "
                f"config={row.config}, dataset={row.dataset}"
            )
        print("seeds: " + ",".join(str(seed) for seed in SEEDS))
    return 0


def _dry_run(args: argparse.Namespace) -> int:
    launcher = _launcher()
    selections = launcher._selected_runs(args)
    if args.manifest is not None and len(selections) != 1:
        raise ValueError("--manifest requires exactly one --run-id")
    if args.manifest is not None and args.manifest_dir is not None:
        raise ValueError("use only one of --manifest and --manifest-dir")
    runtime = launcher.runtime_from_environment()
    cache = launcher.token_launcher.HashCache()
    for row, seed in selections:
        manifest = build_manifest(runtime, row, seed, cache)
        phase = manifest["phases"][0]
        print(f"[{manifest['run_id']}/{PHASE_ID}] {phase['command_shell']}")
        if args.manifest is not None:
            _write_json_atomic(args.manifest.resolve(strict=False), manifest)
        elif args.manifest_dir is not None:
            target = args.manifest_dir.resolve(strict=False) / row.row_id / f"seed{seed}.launch.json"
            _write_json_atomic(target, manifest)
    return 0


def _add_selection(parser: argparse.ArgumentParser, *, running: bool) -> None:
    launcher = _launcher()
    parser.add_argument("--table", choices=("all", "B", "b"), default="all")
    parser.add_argument("--run-id", action="append", type=launcher._parse_run_id)
    parser.add_argument(
        "--all",
        action="store_true",
        help="run all six v2 condition/seed jobs" if running else "select all jobs",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    listing = subparsers.add_parser("list")
    listing.add_argument("--table", choices=("all", "B", "b"), default="all")
    listing.add_argument("--json", action="store_true")
    dry = subparsers.add_parser("dry-run")
    _add_selection(dry, running=False)
    dry.add_argument("--manifest", type=Path)
    dry.add_argument("--manifest-dir", type=Path)
    run = subparsers.add_parser("run")
    _add_selection(run, running=True)
    detach = subparsers.add_parser("detach")
    _add_selection(detach, running=True)
    detach.add_argument("--orchestration-root", type=Path)
    status = subparsers.add_parser("status")
    status.add_argument("job_dir", type=Path)
    reconcile = subparsers.add_parser("reconcile")
    reconcile.add_argument("job_dir", type=Path)
    verify = subparsers.add_parser("verify-run")
    verify.add_argument("run_root", type=Path)
    verify.add_argument("--training-queue-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "_bootstrap-main":
        return _bootstrap_main(raw[1:])
    parser = build_parser()
    args = parser.parse_args(raw)
    try:
        if args.mode == "list":
            return _list_rows(as_json=args.json)
        if args.mode == "dry-run":
            return _dry_run(args)
        if args.mode == "run":
            return _run(args)
        if args.mode == "detach":
            return _detach(args)
        if args.mode in {"status", "reconcile"}:
            report = _launcher()._inspect_or_reconcile_detached_job(
                args.job_dir, mutate=args.mode == "reconcile"
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.mode == "verify-run":
            report = verify_completed_run(
                args.run_root,
                training_queue_dir=args.training_queue_dir,
                require_queue=True,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        parser.error(f"unknown mode: {args.mode}")
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        PermissionError,
        TableBV2RunnerError,
        ValueError,
    ) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
