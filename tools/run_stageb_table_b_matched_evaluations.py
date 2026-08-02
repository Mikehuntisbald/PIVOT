#!/usr/bin/env python3
"""Run the formal D2m/D3m evaluation on one canonical matched TN surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.aggregate_stageb_table_b_matched_panel import (  # noqa: E402
    PROVENANCE_SCHEMA,
    aggregate_matched_panel,
)
from tools.build_stageb_tn_matched_causal_panel import (  # noqa: E402
    MatchedPanelError,
    verify_panel,
)
from tools.compare_stageb_fpr95_records import (  # noqa: E402
    exact_fpr95,
)
from tools.run_stageb_paper_evaluations import (  # noqa: E402
    DATA_INPUT_RELATIVE_PATHS,
    EvaluationSource,
    HashCache,
    PaperEvaluationError,
    _config_paths,
    _resolve_pivot_source,
)
from tools.stageb_eval_records import RECORD_SCHEMA  # noqa: E402
from tools.stageb_table_b_matched_eval_surface import (  # noqa: E402
    DECLARED_SCOPE,
    DEFAULT_AUDIT,
    DEFAULT_D3M_SOURCE,
    DEFAULT_LEDGER,
    EVAL_SPLIT,
    MatchedEvalSurfaceBinding,
    MatchedEvalSurfaceError,
    build_surface,
    iter_rows as iter_surface_rows,
    load_binding,
    sha256_file,
    summary_fields as surface_summary_fields,
)


LAUNCH_SCHEMA = "pivot.stageb.table_b_matched_evaluations_launch/v1"
POSTFLIGHT_SCHEMA = "pivot.stageb.table_b_matched_evaluations_postflight/v1"
EVAL_SEED = 42
EVAL_PROFILE = "table_b_matched_calibration_v1"
CONDITIONS = ("D2m", "D3m")
LEGACY_TRAINING_SOURCE_CONTRACT = "legacy_table_b"
FORMAL_V2_TRAINING_SOURCE_CONTRACT = "table_b_v2_formal"
TRAINING_SOURCE_CONTRACTS = (
    LEGACY_TRAINING_SOURCE_CONTRACT,
    FORMAL_V2_TRAINING_SOURCE_CONTRACT,
)
CODE_ENTRY = "tools/run_stageb_table_b_matched_evaluations.py"
EVALUATOR_ENTRY = "tools/eval_text_groundingdino_refcoco_tn.py"
CODE_ENTRIES = (CODE_ENTRY, EVALUATOR_ENTRY)
CODE_INCLUDE = (
    "datasets/patch_episode.py",
    "models/GroundingDINO/groundingdino.py",
    "models/GroundingDINO/transformer.py",
)
VALIDATION_QUEUE_SPEC_ROLE = "table_b_v2_validation_queue_spec"


class MatchedEvaluationError(RuntimeError):
    """Raised when a matched formal evaluation contract fails."""


def _sha256(path: Path) -> str:
    return sha256_file(Path(path))


def _file_record(
    path: Path, *, role: str | None = None, rows: int | None = None
) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    record: dict[str, Any] = {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }
    if role is not None:
        record["role"] = role
    if rows is not None:
        record["rows"] = int(rows)
    return record


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MatchedEvaluationError(f"{label}: invalid JSON: {error}") from error
    if not isinstance(value, Mapping):
        raise MatchedEvaluationError(f"{label}: expected an object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise MatchedEvaluationError(f"{label}: invalid JSONL: {error}") from error
    rows = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise MatchedEvaluationError(f"{label}:{line_number}: blank row")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise MatchedEvaluationError(
                f"{label}:{line_number}: invalid JSON: {error}"
            ) from error
        if not isinstance(row, dict):
            raise MatchedEvaluationError(f"{label}:{line_number}: expected object")
        rows.append(row)
    if not rows:
        raise MatchedEvaluationError(f"{label}: empty JSONL")
    return rows


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            encoding="ascii",
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="ascii") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _audited_output_record(
    audit: Mapping[str, Any], *, key: str, audit_path: Path
) -> dict[str, Any]:
    outputs = audit.get("outputs")
    declared = outputs.get(key) if isinstance(outputs, Mapping) else None
    if not isinstance(declared, Mapping):
        raise MatchedEvaluationError(f"audit.outputs.{key} is missing")
    path = Path(str(declared.get("path", ""))).expanduser()
    if not path.is_absolute():
        path = audit_path.parent / path
    path = path.resolve(strict=True)
    observed = _file_record(path, rows=int(declared.get("rows", -1)))
    if any(declared.get(field) != observed[field] for field in observed):
        raise MatchedEvaluationError(f"audit.outputs.{key} drifted")
    return dict(declared)


def _source_evidence_paths(source: EvaluationSource) -> tuple[Path, ...]:
    candidates = (
        source.config,
        source.checkpoint,
        source.sequence_manifest,
        source.final_phase_manifest,
        source.training_postflight,
        source.selected_phase_manifest,
        source.selected_training_postflight,
        source.training_queue_manifest,
        source.training_queue_detached_launch,
        source.training_queue_detached_status,
        source.training_run_root,
    )
    paths = [Path(value).resolve(strict=True) for value in candidates if value]
    paths.extend(Path(value).resolve(strict=True) for value in source.training_data)
    return tuple(dict.fromkeys(path for path in paths if path.is_file()))


def validate_condition_source(
    *,
    condition: str,
    seed: int,
    source: EvaluationSource,
    training_source_record: Mapping[str, Any],
    formal_v2: bool = False,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise MatchedEvaluationError(f"unknown Table-B condition: {condition}")
    expected_run_id = f"{condition}:{seed}"
    if not (
        source.kind == "pivot_paper_training_run"
        and source.training_run_id == expected_run_id
        and source.training_seed == seed
        and source.training_phase == "final"
        and source.diagnostic_only is False
        and source.training_run_root is not None
        and source.sequence_manifest is not None
        and source.final_phase_manifest is not None
        and source.training_postflight is not None
    ):
        raise MatchedEvaluationError(
            f"{condition}: paper resolver did not return completed {expected_run_id}"
        )
    if formal_v2 and not (
        source.formal_contract_id
        == "table_b_v2_formal_b40_u1000_i1000"
        and source.matrix_validation_only is True
    ):
        raise MatchedEvaluationError(
            f"{condition}: resolver did not return a formal Table-B v2 source"
        )
    checkpoint = Path(source.checkpoint).resolve(strict=True)
    if _sha256(checkpoint) != source.checkpoint_sha256:
        raise MatchedEvaluationError(f"{condition}: checkpoint SHA-256 drift")
    expected_training_path = Path(str(training_source_record["path"])).resolve(
        strict=True
    )
    resolved_training_data = {
        Path(value).resolve(strict=True) for value in source.training_data
    }
    if expected_training_path not in resolved_training_data:
        raise MatchedEvaluationError(
            f"{condition}: formal training evidence lacks its audited matched source"
        )
    evidence = [
        _file_record(path, role=f"{condition}:training_evidence")
        for path in _source_evidence_paths(source)
    ]
    return {
        "condition": condition,
        "training_run_id": expected_run_id,
        "train_seed": seed,
        "training_run_root": str(Path(source.training_run_root).resolve()),
        "config": _file_record(Path(source.config), role=f"{condition}:config"),
        "checkpoint": _file_record(
            checkpoint, role=f"{condition}:checkpoint"
        ),
        "training_source": dict(training_source_record),
        "training_evidence": evidence,
    }


def _resolve_sources(
    *,
    d2m_root: Path,
    d3m_root: Path,
    seed: int,
    audit: Mapping[str, Any],
    audit_path: Path,
    training_queue_dir: Path | None = None,
    resolver: Callable[..., EvaluationSource] | None = None,
    formal_v2: bool = False,
) -> tuple[dict[str, EvaluationSource], dict[str, Any]]:
    if formal_v2:
        if training_queue_dir is None:
            raise MatchedEvaluationError(
                "formal Table-B v2 evaluation requires --training-queue-dir"
            )
        if resolver is None:
            from tools import run_stageb_table_b_v2 as v2_training

            resolver = v2_training.matched_evaluation_resolver(training_queue_dir)
    resolver = resolver or _resolve_pivot_source
    cache = HashCache()
    sources = {
        "D2m": resolver(Path(d2m_root), cache, training_phase="final"),
        "D3m": resolver(Path(d3m_root), cache, training_phase="final"),
    }
    evidence = {}
    for condition in CONDITIONS:
        training_record = _audited_output_record(
            audit, key=f"{condition.lower()}_train", audit_path=audit_path
        )
        evidence[condition] = validate_condition_source(
            condition=condition,
            seed=seed,
            source=sources[condition],
            training_source_record=training_record,
            formal_v2=formal_v2,
        )
        if training_queue_dir is not None:
            evidence[condition]["training_queue"] = _paper_queue_attestation(
                training_queue_dir,
                condition=condition,
                seed=seed,
                run_root=Path(str(evidence[condition]["training_run_root"])),
            )
        if formal_v2:
            from tools import run_stageb_table_b_v2_queue as v2_queue

            try:
                evidence[condition]["formal_v2"] = v2_queue.formal_evaluation_evidence(
                    training_queue_dir,
                    run_id=f"{condition}:{seed}",
                    run_root=Path(str(evidence[condition]["training_run_root"])),
                )
            except (OSError, ValueError, v2_queue.FormalQueueError) as error:
                raise MatchedEvaluationError(
                    f"{condition}: formal v2 attestation failed: {error}"
                ) from error
    if sources["D2m"].checkpoint_sha256 == sources["D3m"].checkpoint_sha256:
        raise MatchedEvaluationError("D2m and D3m resolved to one checkpoint")
    return sources, evidence


def _paper_queue_attestation(
    queue_dir: Path, *, condition: str, seed: int, run_root: Path
) -> dict[str, Any]:
    from tools import run_stageb_serial_matrix_queue as queue_runner

    try:
        queue_dir = Path(queue_dir).expanduser().resolve(strict=True)
        queue = queue_runner.load_queue(queue_dir)
        verification = queue_runner.verify_queue(queue_dir)
    except (
        queue_runner.QueueContractError,
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        ValueError,
    ) as error:
        raise MatchedEvaluationError(
            f"paper training queue attestation failed: {error}"
        ) from error
    plan = queue.get("plan")
    if not (
        queue.get("status") == "completed"
        and verification.get("status") == "passed"
        and isinstance(plan, Mapping)
    ):
        raise MatchedEvaluationError("paper training queue is not completed/verified")
    queue_id = plan.get("queue_id")
    plan_sha256 = queue.get("plan_sha256")
    if not (
        isinstance(queue_id, str)
        and queue_id
        and verification.get("queue_id") == queue_id
        and isinstance(plan_sha256, str)
        and verification.get("plan_sha256") == plan_sha256
    ):
        raise MatchedEvaluationError("paper training queue identity drifted")
    repository_root = Path(str(plan.get("repository_root", ""))).resolve(strict=False)
    if repository_root != REPO_ROOT:
        raise MatchedEvaluationError("paper training queue repository root mismatch")
    environment = plan.get("runtime_environment")
    raw_root = (
        environment.get("PIVOT_TN_OUTPUT_ROOT")
        if isinstance(environment, Mapping)
        else None
    )
    if not isinstance(raw_root, str) or not raw_root.strip():
        raise MatchedEvaluationError("paper training queue lacks PIVOT_TN_OUTPUT_ROOT")
    output_root = Path(raw_root).expanduser()
    if not output_root.is_absolute():
        output_root = repository_root / output_root
    expected_root = (output_root / condition / f"seed{seed}").resolve(strict=False)
    run_root = Path(run_root).resolve(strict=True)
    if run_root != expected_root:
        raise MatchedEvaluationError("paper queue output root differs from training root")
    run_id = f"{condition}:{seed}"

    def select(values: Any) -> list[Mapping[str, Any]]:
        return [
            value
            for value in values or []
            if isinstance(value, Mapping)
            and value.get("run_id") == run_id
            and value.get("runner") == "paper"
        ]

    planned, observed, verified = (
        select(plan.get("items")),
        select(queue.get("items")),
        select(verification.get("verified_items")),
    )
    if not len(planned) == len(observed) == len(verified) == 1:
        raise MatchedEvaluationError(
            "paper training queue does not uniquely attest the condition/seed"
        )
    item = observed[0]
    verified_item = verified[0]
    if item.get("status") != "completed" or Path(
        str(verified_item.get("output_root", ""))
    ).resolve(strict=True) != run_root:
        raise MatchedEvaluationError("paper training queue item is not completed/canonical")
    job_dir = Path(str(verified_item.get("job_dir", ""))).resolve(strict=True)
    return {
        "queue_id": queue_id,
        "plan_sha256": plan_sha256,
        "runner": "paper",
        "run_id": run_id,
        "output_root": str(run_root),
        "manifest": _file_record(queue_dir / "queue.json", role="training_queue"),
        "detached_launch": _file_record(
            job_dir / "launch.json", role="training_queue_detached_launch"
        ),
        "detached_status": _file_record(
            job_dir / "status.json", role="training_queue_detached_status"
        ),
    }


def _code_records() -> list[dict[str, Any]]:
    from tools.stageb_dependency_audit import (
        DependencyAuditError,
        local_python_dependency_paths,
    )

    try:
        paths = local_python_dependency_paths(
            CODE_ENTRIES, root=REPO_ROOT, include=CODE_INCLUDE
        )
    except DependencyAuditError as error:
        raise MatchedEvaluationError(
            f"matched evaluation dependency audit failed: {error}"
        ) from error
    return [
        _file_record(path, role="evaluation_code") for path in paths
    ]


def _flatten_input_records(
    *,
    audit_path: Path,
    binding: MatchedEvalSurfaceBinding,
    training_evidence: Mapping[str, Any],
    data_root: Path,
    python: Path,
    validation_queue_spec: Path | None = None,
) -> list[dict[str, Any]]:
    records = [
        _file_record(audit_path, role="matched_panel_audit"),
        _file_record(
            Path(str(binding.pair_ledger["path"])), role="matched_pair_ledger"
        ),
        _file_record(
            Path(str(binding.source_manifest["path"])), role="D3m_eval_source"
        ),
        _file_record(
            Path(str(binding.derived_manifest["path"])), role="derived_eval_surface"
        ),
        _file_record(binding.path, role="derived_eval_surface_binding"),
        *[
            _file_record(
                Path(str(record["path"])), role="matched_eval_query_image"
            )
            for record in binding.image_files
        ],
        *[
            _file_record(
                Path(str(record["path"])), role="matched_eval_support_patch"
            )
            for record in binding.support_pool_files
        ],
        *_code_records(),
        _file_record(python, role="evaluation_runtime_python"),
        *[
            _file_record(
                Path(data_root) / relative, role="evaluation_data_dependency"
            )
            for relative in DATA_INPUT_RELATIVE_PATHS
        ],
    ]
    if validation_queue_spec is not None:
        records.append(
            _file_record(
                Path(validation_queue_spec), role=VALIDATION_QUEUE_SPEC_ROLE
            )
        )
    for condition in CONDITIONS:
        value = training_evidence[condition]
        records.extend(
            [
                dict(value["config"]),
                dict(value["checkpoint"]),
                *[dict(record) for record in value["training_evidence"]],
                *[
                    _file_record(path, role=f"{condition}:config_dependency")
                    for path in _config_paths(Path(value["config"]["path"]))
                ],
            ]
        )
        queue = value.get("training_queue")
        if isinstance(queue, Mapping):
            records.extend(
                dict(queue[key])
                for key in ("manifest", "detached_launch", "detached_status")
            )
    deduplicated = {}
    roles: dict[str, set[str]] = {}
    for record in records:
        path = str(Path(record["path"]).resolve())
        roles.setdefault(path, set()).add(str(record.get("role", "input")))
        canonical = {key: value for key, value in record.items() if key != "role"}
        previous = deduplicated.get(path)
        if previous is not None and previous != canonical:
            raise MatchedEvaluationError(f"input identity collision: {path}")
        deduplicated[path] = canonical
    return [
        {**deduplicated[path], "roles": sorted(roles[path])}
        for path in sorted(deduplicated)
    ]


def _evaluation_code_closure_identity(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected = {
        str(Path(record["path"]).resolve(strict=True)): {
            key: record[key] for key in ("path", "sha256", "size_bytes")
        }
        for record in _code_records()
    }
    declared = {}
    for record in records:
        roles = record.get("roles")
        if not isinstance(roles, list) or "evaluation_code" not in roles:
            continue
        path = str(Path(str(record.get("path", ""))).resolve(strict=True))
        if path in declared:
            raise MatchedEvaluationError(
                f"evaluation code closure repeats a path: {path}"
            )
        declared[path] = {
            key: record.get(key) for key in ("path", "sha256", "size_bytes")
        }
    if declared != expected:
        missing = sorted(set(expected).difference(declared))
        extra = sorted(set(declared).difference(expected))
        changed = sorted(
            path
            for path in set(expected).intersection(declared)
            if expected[path] != declared[path]
        )
        raise MatchedEvaluationError(
            "evaluation code closure identity drifted: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    return [
        {
            "path": str(Path(path).relative_to(REPO_ROOT)),
            "sha256": expected[path]["sha256"],
            "size_bytes": expected[path]["size_bytes"],
        }
        for path in sorted(expected)
    ]


def _input_rehash(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    replay = []
    for index, declared in enumerate(records):
        path = Path(str(declared.get("path", ""))).resolve(strict=True)
        observed = _file_record(path)
        expected = {
            key: declared[key]
            for key in ("path", "sha256", "size_bytes")
        }
        if observed != expected:
            raise MatchedEvaluationError(f"input record {index} changed: {path}")
        replay.append({**observed, "roles": list(declared.get("roles", []))})
    return {
        "status": "passed",
        "records": replay,
        "records_sha256": _canonical_sha256(replay),
    }


def _condition_command(
    *,
    python: Path,
    condition: str,
    source: EvaluationSource,
    binding: MatchedEvalSurfaceBinding,
    output_dir: Path,
    data_root: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    amp: bool,
    log_every: int,
) -> list[str]:
    command = [
        str(Path(python).resolve(strict=True)),
        str((REPO_ROOT / "tools/eval_text_groundingdino_refcoco_tn.py").resolve()),
        "--config",
        str(Path(source.config).resolve(strict=True)),
        "--ckpts",
        str(Path(source.checkpoint).resolve(strict=True)),
        "--output_dir",
        str(output_dir.resolve()),
        "--data_root",
        str(Path(data_root).resolve(strict=True)),
        "--tn_jsonl",
        str(binding.derived_manifest["path"]),
        "--direct_prebuilt_tn",
        "--direct_prebuilt_tn_binding",
        str(binding.path),
        "--skip_ref",
        "--device",
        str(device),
        "--batch_size",
        str(int(batch_size)),
        "--num_workers",
        str(int(num_workers)),
        "--seed",
        str(EVAL_SEED),
        "--max_tn_batches",
        "0",
        "--log_every",
        str(int(log_every)),
    ]
    if amp:
        command.append("--amp")
    return command


def _lexical_absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _require_canonical_artifact_path(
    declared: Path | str, *, expected: Path, output_dir: Path, label: str
) -> None:
    lexical = _lexical_absolute(declared)
    expected_lexical = _lexical_absolute(expected)
    if lexical != expected_lexical:
        raise MatchedEvaluationError(f"{label}: path is not canonical")
    resolved = lexical.resolve(strict=False)
    try:
        resolved.relative_to(output_dir.resolve(strict=True))
    except ValueError as error:
        raise MatchedEvaluationError(f"{label}: path escaped output_dir") from error


def _formal_runtime_protocol(contract: Mapping[str, Any]) -> dict[str, Any]:
    runtime = contract.get("runtime")
    expected_runtime_fields = {
        "python",
        "data_root",
        "device",
        "batch_size",
        "num_workers",
        "amp",
        "log_every",
    }
    if not isinstance(runtime, Mapping) or set(runtime) != expected_runtime_fields:
        raise MatchedEvaluationError("matched evaluation runtime field set drifted")
    if not (
        contract.get("evaluation_profile") == EVAL_PROFILE
        and contract.get("evaluation_seed") == EVAL_SEED
    ):
        raise MatchedEvaluationError("matched evaluation profile/seed drifted")
    python_record = runtime.get("python")
    if not isinstance(python_record, Mapping) or set(python_record) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise MatchedEvaluationError("matched evaluation Python identity is incomplete")
    python_path = Path(str(python_record.get("path", ""))).resolve(strict=True)
    if _lexical_absolute(python_record.get("path", "")) != python_path:
        raise MatchedEvaluationError("matched evaluation Python path is not canonical")
    if _file_record(python_path) != dict(python_record):
        raise MatchedEvaluationError("matched evaluation Python identity drifted")
    data_root = Path(str(runtime.get("data_root", ""))).resolve(strict=True)
    if _lexical_absolute(runtime.get("data_root", "")) != data_root:
        raise MatchedEvaluationError("matched evaluation data_root is not canonical")
    if not (
        isinstance(runtime.get("device"), str)
        and bool(str(runtime["device"]).strip())
        and type(runtime.get("batch_size")) is int
        and int(runtime["batch_size"]) > 0
        and type(runtime.get("num_workers")) is int
        and int(runtime["num_workers"]) >= 0
        and type(runtime.get("amp")) is bool
        and type(runtime.get("log_every")) is int
        and int(runtime["log_every"]) >= 0
    ):
        raise MatchedEvaluationError("matched evaluation runtime values are invalid")
    return {
        "profile": EVAL_PROFILE,
        "evaluation_seed": EVAL_SEED,
        "python": dict(python_record),
        "data_root": str(data_root),
        "device": str(runtime["device"]),
        "batch_size": int(runtime["batch_size"]),
        "num_workers": int(runtime["num_workers"]),
        "amp": bool(runtime["amp"]),
        "log_every": int(runtime["log_every"]),
    }


def _normalized_phase_command_template(
    *,
    phase: Mapping[str, Any],
    condition: str,
    contract: Mapping[str, Any],
) -> list[str]:
    training = contract["training"][condition]
    surface = contract["surface"]
    expected_values = {
        "--config": str(Path(training["config"]["path"]).resolve(strict=True)),
        "--ckpts": str(Path(training["checkpoint"]["path"]).resolve(strict=True)),
        "--output_dir": str(Path(phase["raw_output_dir"]).resolve()),
        "--tn_jsonl": str(
            Path(surface["matched_eval_surface_derived_path"]).resolve(strict=True)
        ),
        "--direct_prebuilt_tn_binding": str(
            Path(surface["matched_eval_surface_binding_path"]).resolve(strict=True)
        ),
    }
    placeholders = {
        "--config": "<TRAINING_CONFIG>",
        "--ckpts": "<TRAINING_CHECKPOINT>",
        "--output_dir": "<EVALUATION_OUTPUT>",
        "--tn_jsonl": "<MATCHED_EVAL_SURFACE>",
        "--direct_prebuilt_tn_binding": "<MATCHED_EVAL_SURFACE_BINDING>",
    }
    command = phase.get("command")
    if not isinstance(command, list) or not all(
        isinstance(token, str) and token for token in command
    ):
        raise MatchedEvaluationError(f"{condition}: phase command is invalid")
    normalized = list(command)
    for flag, expected in expected_values.items():
        positions = [index for index, token in enumerate(command) if token == flag]
        if len(positions) != 1 or positions[0] + 1 >= len(command):
            raise MatchedEvaluationError(
                f"{condition}: phase command does not contain exactly one {flag}"
            )
        value_index = positions[0] + 1
        if command[value_index] != expected:
            raise MatchedEvaluationError(
                f"{condition}: phase command {flag} value is not canonical"
            )
        normalized[value_index] = placeholders[flag]
    return normalized


def formal_protocol_identity(launch: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the cross-seed invariant protocol from one verified launch."""

    contract = launch.get("contract")
    if not isinstance(contract, Mapping):
        raise MatchedEvaluationError("matched evaluation launch lacks a contract")
    runtime = _formal_runtime_protocol(contract)
    records = contract.get("input_records")
    if not isinstance(records, list):
        raise MatchedEvaluationError("matched evaluation launch lacks input records")
    closure = _evaluation_code_closure_identity(records)
    python_records = [
        record
        for record in records
        if "evaluation_runtime_python" in record.get("roles", [])
    ]
    if len(python_records) != 1 or any(
        python_records[0].get(field) != runtime["python"][field]
        for field in ("path", "sha256", "size_bytes")
    ):
        raise MatchedEvaluationError(
            "matched evaluation inputs do not bind the runtime Python"
        )
    phases = contract.get("phases")
    if not isinstance(phases, list) or [
        phase.get("condition") for phase in phases
    ] != list(CONDITIONS):
        raise MatchedEvaluationError("matched evaluation phase set drifted")
    templates = {
        condition: _normalized_phase_command_template(
            phase=phase, condition=condition, contract=contract
        )
        for condition, phase in zip(CONDITIONS, phases)
    }
    if templates["D2m"] != templates["D3m"]:
        raise MatchedEvaluationError(
            "D2m/D3m phase commands differ outside normalized identities"
        )
    return {
        "schema": "pivot.stageb.table_b_matched_formal_protocol/v1",
        "common_runtime": runtime,
        "phase_command_templates": templates,
        "command_template_sha256": _canonical_sha256(templates),
        "evaluation_code_closure": closure,
        "evaluation_code_closure_sha256": _canonical_sha256(closure),
    }


def _validate_artifact_layout(
    launch: Mapping[str, Any], *, binding: MatchedEvalSurfaceBinding
) -> None:
    """Rebuild every phase path and command from the immutable contract."""

    output_dir = Path(str(launch.get("output_dir", ""))).resolve(strict=True)
    if _lexical_absolute(launch.get("output_dir", "")) != output_dir:
        raise MatchedEvaluationError("matched evaluation output_dir is not canonical")
    contract = launch.get("contract")
    if not isinstance(contract, Mapping):
        raise MatchedEvaluationError("matched evaluation contract is missing")
    formal_protocol_identity(launch)
    runtime = contract.get("runtime")
    training = contract.get("training")
    phases = contract.get("phases")
    if not (
        isinstance(runtime, Mapping)
        and isinstance(training, Mapping)
        and isinstance(phases, list)
        and len(phases) == len(CONDITIONS)
    ):
        raise MatchedEvaluationError("matched evaluation layout contract is incomplete")
    seed = int(contract["seed"])
    expected_phases = []
    for condition in CONDITIONS:
        raw_output = output_dir / "raw" / condition
        console_log = output_dir / "logs" / f"{condition}.log"
        final_records = output_dir / "records" / f"{condition}_seed{seed}.records.jsonl"
        for value, expected, label in (
            (raw_output, raw_output, f"{condition} raw output"),
            (console_log, console_log, f"{condition} console log"),
            (final_records, final_records, f"{condition} final records"),
        ):
            _require_canonical_artifact_path(
                value, expected=expected, output_dir=output_dir, label=label
            )
        source_record = training.get(condition)
        if not isinstance(source_record, Mapping):
            raise MatchedEvaluationError(f"{condition}: training contract is missing")
        source = SimpleNamespace(
            config=Path(str(source_record["config"]["path"])),
            checkpoint=Path(str(source_record["checkpoint"]["path"])),
        )
        command = _condition_command(
            python=Path(str(runtime["python"]["path"])),
            condition=condition,
            source=source,
            binding=binding,
            output_dir=raw_output,
            data_root=Path(str(runtime["data_root"])),
            device=str(runtime["device"]),
            batch_size=int(runtime["batch_size"]),
            num_workers=int(runtime["num_workers"]),
            amp=bool(runtime["amp"]),
            log_every=int(runtime["log_every"]),
        )
        expected_phases.append(
            {
                "condition": condition,
                "command": command,
                "command_shell": shlex.join(command),
                "raw_output_dir": str(raw_output),
                "console_log": str(console_log),
                "final_records": str(final_records),
            }
        )
    if phases != expected_phases:
        raise MatchedEvaluationError(
            "matched evaluation phase paths/commands are not canonical"
        )
    runtime_phases = launch.get("phases")
    if not isinstance(runtime_phases, list) or len(runtime_phases) != len(
        expected_phases
    ):
        raise MatchedEvaluationError("matched runtime phase list is incomplete")
    for expected, observed in zip(expected_phases, runtime_phases):
        if any(observed.get(key) != value for key, value in expected.items()):
            raise MatchedEvaluationError(
                "matched runtime phase paths/commands are not canonical"
            )
        _require_canonical_artifact_path(
            observed["raw_output_dir"],
            expected=Path(expected["raw_output_dir"]),
            output_dir=output_dir,
            label=f"{expected['condition']} raw output",
        )
        _require_canonical_artifact_path(
            observed["console_log"],
            expected=Path(expected["console_log"]),
            output_dir=output_dir,
            label=f"{expected['condition']} console log",
        )
        _require_canonical_artifact_path(
            observed["final_records"],
            expected=Path(expected["final_records"]),
            output_dir=output_dir,
            label=f"{expected['condition']} final records",
        )


def prepare_evaluation(
    *,
    output_dir: Path,
    audit_path: Path,
    ledger_path: Path,
    d3m_source_path: Path,
    data_root: Path,
    d2m_training_run_root: Path,
    d3m_training_run_root: Path,
    seed: int,
    python: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    amp: bool,
    log_every: int,
    training_queue_dir: Path | None = None,
    resolver: Callable[..., EvaluationSource] | None = None,
    training_source_contract: str = LEGACY_TRAINING_SOURCE_CONTRACT,
    validation_queue_spec: Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"matched evaluation output must be fresh: {output_dir}")
    audit_path = Path(audit_path).resolve(strict=True)
    ledger_path = Path(ledger_path).resolve(strict=True)
    d3m_source_path = Path(d3m_source_path).resolve(strict=True)
    data_root = Path(data_root).resolve(strict=True)
    python = Path(python).resolve(strict=True)
    if validation_queue_spec is not None:
        validation_queue_spec = Path(validation_queue_spec).resolve(strict=True)
    if training_source_contract not in TRAINING_SOURCE_CONTRACTS:
        raise MatchedEvaluationError(
            f"unsupported training source contract: {training_source_contract!r}"
        )
    formal_v2 = training_source_contract == FORMAL_V2_TRAINING_SOURCE_CONTRACT
    try:
        audit = verify_panel(audit_path)
    except (OSError, KeyError, TypeError, MatchedPanelError) as error:
        raise MatchedEvaluationError(f"matched v2 audit failed: {error}") from error
    sources, training_evidence = _resolve_sources(
        d2m_root=d2m_training_run_root,
        d3m_root=d3m_training_run_root,
        seed=seed,
        audit=audit,
        audit_path=audit_path,
        training_queue_dir=training_queue_dir,
        resolver=resolver,
        formal_v2=formal_v2,
    )
    output_dir.mkdir(parents=True)
    derived = output_dir / "surface/matched_calibration.jsonl"
    binding = build_surface(
        audit_path=audit_path,
        ledger_path=ledger_path,
        source_path=d3m_source_path,
        derived_path=derived,
        data_root=data_root,
    )
    inputs = _flatten_input_records(
        audit_path=audit_path,
        binding=binding,
        training_evidence=training_evidence,
        data_root=data_root,
        python=python,
        validation_queue_spec=validation_queue_spec,
    )
    phases = []
    for condition in CONDITIONS:
        raw_output = output_dir / "raw" / condition
        command = _condition_command(
            python=python,
            condition=condition,
            source=sources[condition],
            binding=binding,
            output_dir=raw_output,
            data_root=data_root,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            amp=amp,
            log_every=log_every,
        )
        phases.append(
            {
                "condition": condition,
                "command": command,
                "command_shell": shlex.join(command),
                "raw_output_dir": str(raw_output),
                "console_log": str(output_dir / "logs" / f"{condition}.log"),
                "final_records": str(
                    output_dir / "records" / f"{condition}_seed{seed}.records.jsonl"
                ),
            }
        )
    contract = {
        "seed": int(seed),
        "evaluation_profile": EVAL_PROFILE,
        "evaluation_seed": EVAL_SEED,
        "conditions": list(CONDITIONS),
        "training_source_contract": training_source_contract,
        "declared_scope": DECLARED_SCOPE,
        "formal_global_fpr_eligible": False,
        "same_surface_for_both_conditions": True,
        "surface": surface_summary_fields(binding),
        "final_record_binding": {
            "logical_manifest": dict(binding.source_manifest),
            "physical_manifest": dict(binding.derived_manifest),
            "physical_binding_sha256": _sha256(binding.path),
            "row_mapping_sha256": binding.row_mapping_sha256,
            "score_fields_preserved_exactly": [
                "pos_score",
                "neg_score",
                "pos_iou",
                "neg_iou",
            ],
            "purpose": (
                "rebind the evaluator representation to the audited D3m logical "
                "source after exact one-to-one surface replay"
            ),
        },
        "training": training_evidence,
        "phases": phases,
        "input_records": inputs,
        "runtime": {
            "python": _file_record(python),
            "data_root": str(data_root),
            "device": str(device),
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "amp": bool(amp),
            "log_every": int(log_every),
        },
        "validation_queue_spec": (
            _file_record(
                validation_queue_spec, role=VALIDATION_QUEUE_SPEC_ROLE
            )
            if validation_queue_spec is not None
            else None
        ),
    }
    launch = {
        "schema": LAUNCH_SCHEMA,
        "status": "planned",
        "output_dir": str(output_dir),
        "contract_sha256": _canonical_sha256(contract),
        "contract": contract,
        "phases": [{**phase, "status": "pending"} for phase in phases],
        "completed_conditions": [],
    }
    _validate_artifact_layout(launch, binding=binding)
    _write_json_atomic(output_dir / "launch.json", launch)
    return launch


def _surface_rows(binding: MatchedEvalSurfaceBinding) -> list[dict[str, Any]]:
    return list(iter_surface_rows(Path(str(binding.derived_manifest["path"]))))


def _raw_result(
    *, condition: str, phase: Mapping[str, Any], binding: MatchedEvalSurfaceBinding,
    source: EvaluationSource,
) -> tuple[Mapping[str, Any], Path, list[dict[str, Any]]]:
    raw_dir = Path(str(phase["raw_output_dir"])).resolve(strict=True)
    summary_path = (raw_dir / "summary.json").resolve(strict=True)
    summary = _read_json(summary_path, label=f"{condition} raw summary")
    if summary.get("refcoco") != []:
        raise MatchedEvaluationError(f"{condition}: raw evaluator was not TN-only")
    tn_rows = summary.get("tn")
    if not (
        isinstance(tn_rows, list)
        and len(tn_rows) == 1
        and isinstance(tn_rows[0], Mapping)
    ):
        raise MatchedEvaluationError(f"{condition}: raw summary TN row is not unique")
    row = tn_rows[0]
    expected_surface = surface_summary_fields(binding)
    if any(row.get(field) != value for field, value in expected_surface.items()):
        raise MatchedEvaluationError(f"{condition}: raw summary surface binding drift")
    if not (
        Path(str(row.get("checkpoint", ""))).resolve(strict=True)
        == Path(source.checkpoint).resolve(strict=True)
        and row.get("seed") == EVAL_SEED
        and row.get("max_batches") == 0
        and row.get("eval_scope") == DECLARED_SCOPE
        and row.get("manifest_sha256") == binding.derived_manifest["sha256"]
        and row.get("manifest_n") == binding.derived_manifest["rows"]
        and row.get("num_pairs") == binding.derived_manifest["rows"]
        and row.get("invalid_records") == 0
        and row.get("invalid_positive_pairs") == 0
        and row.get("invalid_negative_pairs") == 0
    ):
        raise MatchedEvaluationError(f"{condition}: raw summary formal contract failed")
    records_path = Path(str(row.get("records_jsonl", ""))).expanduser()
    if not records_path.is_absolute():
        records_path = REPO_ROOT / records_path
    records_path = records_path.resolve(strict=True)
    try:
        records_path.relative_to(raw_dir)
    except ValueError as error:
        raise MatchedEvaluationError(
            f"{condition}: raw records escape raw output"
        ) from error
    records = _read_jsonl(records_path, label=f"{condition} raw records")
    surface_rows = _surface_rows(binding)
    if len(records) != len(surface_rows):
        raise MatchedEvaluationError(f"{condition}: raw records dropped surface rows")
    run_ids = set()
    positives = []
    negatives = []
    for index, (record, surface_row) in enumerate(zip(records, surface_rows)):
        if not (
            record.get("schema") == RECORD_SCHEMA
            and record.get("task") == "tn"
            and record.get("manifest_key") == "tn_global"
            and record.get("manifest_sha256") == binding.derived_manifest["sha256"]
            and record.get("manifest_n") == len(records)
            and record.get("manifest_index") == index
            and record.get("sample_id") == surface_row.get("sample_id")
            and record.get("image_id") == surface_row.get("image_id")
            and record.get("split") == EVAL_SPLIT
            and record.get("valid") is True
            and record.get("eval_scope") == DECLARED_SCOPE
            and record.get("global_tn_verified") is not True
        ):
            raise MatchedEvaluationError(
                f"{condition}: raw record identity/scope drift at {index}"
            )
        support_sha = str(record.get("support_input_sha256") or "")
        support_classes = record.get("support_class_ids")
        if not (
            len(support_sha) == 64
            and all(character in "0123456789abcdef" for character in support_sha)
            and record.get("support_input_kind") in {"patch", "patches", "patch_global"}
            and support_classes == [int(surface_row["class_id"])]
        ):
            raise MatchedEvaluationError(
                f"{condition}: raw support identity drift at {index}"
            )
        try:
            positive = float(record["pos_score"])
            negative = float(record["neg_score"])
        except (KeyError, TypeError, ValueError) as error:
            raise MatchedEvaluationError(
                f"{condition}: raw score missing at {index}"
            ) from error
        if not math.isfinite(positive) or not math.isfinite(negative):
            raise MatchedEvaluationError(f"{condition}: non-finite raw score at {index}")
        positives.append(positive)
        negatives.append(negative)
        run_ids.add(str(record.get("run_id") or ""))
    if len(run_ids) != 1 or not next(iter(run_ids)):
        raise MatchedEvaluationError(f"{condition}: raw run_id is not unique")
    if row.get("run_id") != next(iter(run_ids)):
        raise MatchedEvaluationError(
            f"{condition}: raw summary/record run_id mismatch"
        )
    fpr = exact_fpr95(positives, negatives)
    measured = {
        "fpr95tpr": float(fpr["fpr"]),
        "threshold_at_95tpr": float(fpr["threshold"]),
        "pair_win_rate": sum(
            positive > negative for positive, negative in zip(positives, negatives)
        )
        / len(positives),
    }
    for field, value in measured.items():
        if not math.isclose(float(row.get(field, float("nan"))), value, abs_tol=1e-7):
            raise MatchedEvaluationError(
                f"{condition}: raw summary {field} does not replay from records"
            )
    return row, records_path, records


def postprocess_condition_records(
    *,
    condition: str,
    seed: int,
    source: EvaluationSource,
    binding: MatchedEvalSurfaceBinding,
    audit_record: Mapping[str, Any],
    training_source_record: Mapping[str, Any],
    raw_records_path: Path,
    raw_records: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    if not (
        condition in CONDITIONS
        and source.training_run_id == f"{condition}:{seed}"
        and source.training_seed == seed
        and _sha256(Path(source.checkpoint)) == source.checkpoint_sha256
    ):
        raise MatchedEvaluationError(
            f"{condition}: postprocess training source identity mismatch"
        )
    logical_rows = _read_jsonl(
        Path(str(binding.source_manifest["path"])), label="logical D3m eval source"
    )
    physical_rows = _surface_rows(binding)
    if not len(raw_records) == len(logical_rows) == len(physical_rows):
        raise MatchedEvaluationError(f"{condition}: postprocess row counts differ")
    raw_record = _file_record(raw_records_path, rows=len(raw_records))
    binding_record = _file_record(binding.path)
    finalized = []
    for index, (raw, logical, physical, mapping) in enumerate(
        zip(raw_records, logical_rows, physical_rows, binding.row_mapping)
    ):
        if not (
            mapping.get("source_index") == index
            and mapping.get("derived_index") == index
            and raw.get("manifest_index") == index
            and raw.get("sample_id") == physical.get("sample_id")
            and logical.get("sample_id") == physical.get("sample_id")
        ):
            raise MatchedEvaluationError(
                f"{condition}: raw/physical/logical mapping drift at {index}"
            )
        final = {
            "schema": RECORD_SCHEMA,
            "task": "tn",
            "manifest_key": "tn_global",
            "manifest_sha256": binding.source_manifest["sha256"],
            "manifest_n": len(logical_rows),
            "manifest_index": index,
            "sample_id": str(logical["sample_id"]),
            "image_id": int(logical["image_id"]),
            "ann_id": int(logical["ann_id"]),
            "ref_id": int(logical["ref_id"]),
            "sent_id": int(logical["sent_id"]),
            "split": str(logical.get("eval_split") or logical.get("split")),
            "run_id": f"{condition}:{seed}",
            "valid": True,
            "pos_score": float(raw["pos_score"]),
            "neg_score": float(raw["neg_score"]),
            "pos_iou": float(raw["pos_iou"]),
            "neg_iou": float(raw["neg_iou"]),
            "train_scope": (
                "traceable_counterfactual_edit"
                if condition == "D2m"
                else DECLARED_SCOPE
            ),
            "eval_scope": DECLARED_SCOPE,
            "global_tn_verified": False,
            "formal_global_fpr_eligible": False,
            "provenance_schema": PROVENANCE_SCHEMA,
            "table_b_id": condition,
            "train_seed": seed,
            "checkpoint_sha256": source.checkpoint_sha256,
            "training_source_sha256": training_source_record["sha256"],
            "training_source_n": training_source_record["rows"],
            "matched_panel_audit_sha256": audit_record["sha256"],
            "evaluation_source_sha256": binding.source_manifest["sha256"],
            "evaluation_source_n": binding.source_manifest["rows"],
            "evaluation_manifest_sha256": binding.source_manifest["sha256"],
            "declared_evaluation_surface": DECLARED_SCOPE,
            "matched_eval_surface_sha256": binding.derived_manifest["sha256"],
            "matched_eval_surface_binding_sha256": binding_record["sha256"],
            "matched_eval_surface_row_mapping_sha256": binding.row_mapping_sha256,
            "raw_records_sha256": raw_record["sha256"],
            "raw_manifest_index": index,
            "raw_manifest_sha256": binding.derived_manifest["sha256"],
            "matched_pair_id": str(logical["matched_pair_id"]),
            "matched_parent_key_sha256": str(
                logical["matched_parent_key_sha256"]
            ),
            "negative_text_relation": str(
                logical["matched_stratum"]["negative_text_relation"]
            ),
            "canonical_class_id_match": bool(
                logical["canonical_class_id_match"]
            ),
            "support_input_kind": str(raw["support_input_kind"]),
            "support_input_sha256": str(raw["support_input_sha256"]),
            "support_class_ids": list(raw["support_class_ids"]),
        }
        if any(
            not math.isfinite(float(final[field]))
            for field in ("pos_score", "neg_score", "pos_iou", "neg_iou")
        ):
            raise MatchedEvaluationError(
                f"{condition}: non-finite finalized value at {index}"
            )
        finalized.append(final)
    _write_jsonl_atomic(output_path, finalized)
    return _file_record(output_path, role=f"{condition}:final_records", rows=len(finalized))


def _compact_aggregator_result(report: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    strata = {}
    for name, value in report["per_seed"][str(seed)]["strata"].items():
        strata[name] = {
            key: nested
            for key, nested in value.items()
            if key != "paired_records"
        }
    return {
        "report_sha256": _canonical_sha256(report),
        "validation": report["validation"],
        "strata": strata,
    }


def _validate_runtime_sources(
    launch: Mapping[str, Any], sources: Mapping[str, EvaluationSource]
) -> None:
    contract = launch["contract"]
    seed = int(contract["seed"])
    training_source_contract = contract.get(
        "training_source_contract", LEGACY_TRAINING_SOURCE_CONTRACT
    )
    formal_v2 = training_source_contract == FORMAL_V2_TRAINING_SOURCE_CONTRACT
    if not (
        set(sources) == set(CONDITIONS)
        and contract.get("conditions") == list(CONDITIONS)
        and contract.get("evaluation_seed") == EVAL_SEED
        and contract.get("declared_scope") == DECLARED_SCOPE
        and contract.get("formal_global_fpr_eligible") is False
        and contract.get("same_surface_for_both_conditions") is True
        and training_source_contract in TRAINING_SOURCE_CONTRACTS
        and isinstance(contract.get("training"), Mapping)
        and set(contract["training"]) == set(CONDITIONS)
        and [phase.get("condition") for phase in contract.get("phases", [])]
        == list(CONDITIONS)
    ):
        raise MatchedEvaluationError("runtime condition contract drifted")
    for condition in CONDITIONS:
        source = sources[condition]
        planned = contract["training"][condition]
        planned_training_source = Path(
            str(planned["training_source"]["path"])
        ).resolve(strict=True)
        if not (
            planned.get("condition") == condition
            and planned.get("training_run_id") == f"{condition}:{seed}"
            and planned.get("train_seed") == seed
            and source.kind == "pivot_paper_training_run"
            and source.training_run_id == f"{condition}:{seed}"
            and source.training_seed == seed
            and source.training_phase == "final"
            and source.diagnostic_only is False
            and (
                not formal_v2
                or (
                    source.formal_contract_id
                    == "table_b_v2_formal_b40_u1000_i1000"
                    and source.matrix_validation_only is True
                )
            )
            and Path(source.training_run_root).resolve(strict=True)
            == Path(planned["training_run_root"]).resolve(strict=True)
            and Path(source.config).resolve(strict=True)
            == Path(planned["config"]["path"]).resolve(strict=True)
            and _sha256(Path(source.config)) == planned["config"]["sha256"]
            and source.checkpoint_sha256 == planned["checkpoint"]["sha256"]
            and Path(source.checkpoint).resolve(strict=True)
            == Path(planned["checkpoint"]["path"]).resolve(strict=True)
            and _sha256(Path(source.checkpoint)) == source.checkpoint_sha256
            and planned_training_source
            in {Path(path).resolve(strict=True) for path in source.training_data}
        ):
            raise MatchedEvaluationError(
                f"{condition}: runtime source differs from launch contract"
            )
        queue = planned.get("training_queue")
        if isinstance(queue, Mapping):
            observed_queue = _paper_queue_attestation(
                Path(str(queue["manifest"]["path"])).parent,
                condition=condition,
                seed=seed,
                run_root=Path(str(planned["training_run_root"])),
            )
            if observed_queue != queue:
                raise MatchedEvaluationError(
                    f"{condition}: runtime training queue attestation drifted"
                )
        if formal_v2:
            from tools import run_stageb_table_b_v2_queue as v2_queue

            planned_v2 = planned.get("formal_v2")
            if not isinstance(planned_v2, Mapping):
                raise MatchedEvaluationError(
                    f"{condition}: launch lacks formal v2 training evidence"
                )
            try:
                observed_v2 = v2_queue.formal_evaluation_evidence(
                    Path(str(planned_v2["completion_attestation"]["path"])).parent,
                    run_id=f"{condition}:{seed}",
                    run_root=Path(str(planned["training_run_root"])),
                )
            except (OSError, ValueError, v2_queue.FormalQueueError) as error:
                raise MatchedEvaluationError(
                    f"{condition}: formal v2 runtime replay failed: {error}"
                ) from error
            if observed_v2 != planned_v2:
                raise MatchedEvaluationError(
                    f"{condition}: formal v2 runtime attestation drifted"
                )


def _execution_phases_match_contract(launch: Mapping[str, Any]) -> bool:
    planned = launch.get("contract", {}).get("phases", [])
    executed = launch.get("phases")
    return bool(
        isinstance(planned, list)
        and isinstance(executed, list)
        and len(planned) == len(executed) == len(CONDITIONS)
        and all(
            all(runtime.get(key) == value for key, value in phase.items())
            for phase, runtime in zip(planned, executed)
        )
    )


def _build_postflight(
    *,
    launch: Mapping[str, Any],
    sources: Mapping[str, EvaluationSource],
    binding: MatchedEvalSurfaceBinding,
) -> dict[str, Any]:
    contract = launch["contract"]
    if _canonical_sha256(contract) != launch.get("contract_sha256"):
        raise MatchedEvaluationError("launch contract SHA-256 drift")
    _validate_artifact_layout(launch, binding=binding)
    _validate_runtime_sources(launch, sources)
    if not (
        launch.get("completed_conditions") == list(CONDITIONS)
        and _execution_phases_match_contract(launch)
        and [phase.get("condition") for phase in launch.get("phases", [])]
        == list(CONDITIONS)
        and all(
            phase.get("status") == "completed"
            and phase.get("returncode") == 0
            for phase in launch.get("phases", [])
        )
    ):
        raise MatchedEvaluationError("evaluation phases are not exactly completed")
    input_rehash = _input_rehash(contract["input_records"])
    audit_record = next(
        record
        for record in contract["input_records"]
        if "matched_panel_audit" in record.get("roles", [])
    )
    seed = int(contract["seed"])
    condition_artifacts = {}
    final_paths = {}
    for phase in contract["phases"]:
        condition = str(phase["condition"])
        raw_summary, raw_path, raw_records = _raw_result(
            condition=condition,
            phase=phase,
            binding=binding,
            source=sources[condition],
        )
        training_source = contract["training"][condition]["training_source"]
        final_path = Path(str(phase["final_records"])).resolve()
        final_record = postprocess_condition_records(
            condition=condition,
            seed=seed,
            source=sources[condition],
            binding=binding,
            audit_record=audit_record,
            training_source_record=training_source,
            raw_records_path=raw_path,
            raw_records=raw_records,
            output_path=final_path,
        )
        finalized = _read_jsonl(final_path, label=f"{condition} final records")
        if any(
            float(raw[field]) != float(final[field])
            for raw, final in zip(raw_records, finalized)
            for field in ("pos_score", "neg_score", "pos_iou", "neg_iou")
        ):
            raise MatchedEvaluationError(f"{condition}: score changed in postprocess")
        condition_artifacts[condition] = {
            "raw_summary": _file_record(
                Path(str(phase["raw_output_dir"])) / "summary.json",
                role=f"{condition}:raw_summary",
            ),
            "raw_records": _file_record(
                raw_path, role=f"{condition}:raw_records", rows=len(raw_records)
            ),
            "final_records": final_record,
            "raw_summary_run_id": raw_summary["run_id"],
            "score_replay_exact": True,
        }
        final_paths[condition] = final_path
    report = aggregate_matched_panel(
        audit_path=Path(str(audit_record["path"])),
        pair_ledger_path=Path(str(binding.pair_ledger["path"])),
        d2m_source_path=Path(
            str(
                verify_panel(Path(str(audit_record["path"])))
                ["outputs"]["d2m_calibration"]["path"]
            )
        ),
        d3m_source_path=Path(str(binding.source_manifest["path"])),
        evaluation_manifest_path=Path(str(binding.source_manifest["path"])),
        d2m_records={seed: final_paths["D2m"]},
        d3m_records={seed: final_paths["D3m"]},
        expected_seeds=[seed],
    )
    return {
        "schema": POSTFLIGHT_SCHEMA,
        "status": "passed",
        "contract_sha256": launch["contract_sha256"],
        "formal_global_fpr_eligible": False,
        "input_rehash": input_rehash,
        "surface": {
            **surface_summary_fields(binding),
            "binding_replay_passed": True,
        },
        "conditions": condition_artifacts,
        "aggregator_replay": _compact_aggregator_result(report, seed=seed),
        "all_records_and_scores_replayed": True,
        "scope_upgrade_forbidden": True,
    }


def run_evaluation(
    launch: dict[str, Any], *, sources: Mapping[str, EvaluationSource]
) -> dict[str, Any]:
    output_dir = Path(str(launch["output_dir"])).resolve(strict=True)
    launch_path = output_dir / "launch.json"
    contract = launch["contract"]
    if _canonical_sha256(contract) != launch.get("contract_sha256"):
        raise MatchedEvaluationError("launch contract changed before execution")
    _validate_runtime_sources(launch, sources)
    if not _execution_phases_match_contract(launch):
        raise MatchedEvaluationError("execution phases differ from launch contract")
    binding = load_binding(
        Path(contract["surface"]["matched_eval_surface_binding_path"]),
        expected_derived=Path(
            contract["surface"]["matched_eval_surface_derived_path"]
        ),
    )
    _validate_artifact_layout(launch, binding=binding)
    for phase in launch["phases"]:
        condition = str(phase["condition"])
        phase["status"] = "running"
        launch["status"] = "running"
        launch["current_condition"] = condition
        _write_json_atomic(launch_path, launch)
        log_path = Path(str(phase["console_log"]))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                list(phase["command"]),
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        phase["returncode"] = int(completed.returncode)
        phase["console_log_record"] = _file_record(
            log_path, role=f"{condition}:console_log"
        )
        if completed.returncode != 0:
            phase["status"] = "failed"
            launch["status"] = "failed"
            launch["failure_condition"] = condition
            _write_json_atomic(launch_path, launch)
            raise MatchedEvaluationError(
                f"{condition} evaluator failed with code {completed.returncode}"
            )
        phase["status"] = "completed"
        launch["completed_conditions"].append(condition)
        _write_json_atomic(launch_path, launch)
    postflight = _build_postflight(launch=launch, sources=sources, binding=binding)
    postflight_path = output_dir / "postflight.json"
    _write_json_atomic(postflight_path, postflight)
    launch["status"] = "completed"
    launch["current_condition"] = None
    launch["postflight"] = postflight
    launch["postflight_artifact"] = _file_record(
        postflight_path, role="postflight"
    )
    _write_json_atomic(launch_path, launch)
    return launch


def _verify_declared_artifact(
    declared: Mapping[str, Any], *, label: str
) -> Path:
    path = Path(str(declared.get("path", ""))).resolve(strict=True)
    observed = _file_record(path, rows=declared.get("rows"))
    if any(declared.get(field) != value for field, value in observed.items()):
        raise MatchedEvaluationError(f"{label} artifact drift")
    return path


def verify_completed_output(output_dir: Path) -> Mapping[str, Any]:
    """Replay the persisted postflight without trusting its claimed status."""

    output_dir = Path(output_dir).resolve(strict=True)
    launch = _read_json(output_dir / "launch.json", label="matched launch")
    if launch.get("schema") != LAUNCH_SCHEMA or launch.get("status") != "completed":
        raise MatchedEvaluationError("matched launch is not completed")
    contract = launch.get("contract")
    if not isinstance(contract, Mapping) or _canonical_sha256(contract) != launch.get(
        "contract_sha256"
    ):
        raise MatchedEvaluationError("persisted launch contract drift")
    training_source_contract = contract.get(
        "training_source_contract", LEGACY_TRAINING_SOURCE_CONTRACT
    )
    if training_source_contract not in TRAINING_SOURCE_CONTRACTS:
        raise MatchedEvaluationError("persisted training source contract drifted")
    formal_v2 = training_source_contract == FORMAL_V2_TRAINING_SOURCE_CONTRACT
    postflight_declared = launch.get("postflight_artifact")
    if not isinstance(postflight_declared, Mapping):
        raise MatchedEvaluationError("launch lacks postflight artifact")
    postflight_path = _verify_declared_artifact(
        postflight_declared, label="postflight"
    )
    if postflight_path != (output_dir / "postflight.json").resolve(strict=True):
        raise MatchedEvaluationError("postflight path is not canonical")
    postflight = _read_json(postflight_path, label="matched postflight")
    if not (
        postflight.get("schema") == POSTFLIGHT_SCHEMA
        and postflight.get("status") == "passed"
        and postflight.get("contract_sha256") == launch.get("contract_sha256")
        and postflight.get("formal_global_fpr_eligible") is False
        and postflight.get("scope_upgrade_forbidden") is True
        and postflight.get("all_records_and_scores_replayed") is True
        and dict(launch.get("postflight", {})) == dict(postflight)
    ):
        raise MatchedEvaluationError("persisted postflight contract failed")
    execution_phases = launch.get("phases")
    if not (
        launch.get("completed_conditions") == list(CONDITIONS)
        and isinstance(execution_phases, list)
        and len(execution_phases) == len(contract["phases"])
    ):
        raise MatchedEvaluationError("persisted execution phase set drift")
    for planned, executed in zip(contract["phases"], execution_phases):
        if not (
            all(executed.get(key) == value for key, value in planned.items())
            and executed.get("status") == "completed"
            and executed.get("returncode") == 0
            and isinstance(executed.get("console_log_record"), Mapping)
        ):
            raise MatchedEvaluationError("persisted execution phase contract failed")
        _verify_declared_artifact(
            executed["console_log_record"],
            label=f"{executed.get('condition')} console log",
        )
    input_rehash = _input_rehash(contract["input_records"])
    if input_rehash != postflight.get("input_rehash"):
        raise MatchedEvaluationError("postflight input replay drift")
    for condition in CONDITIONS:
        planned_training = contract["training"][condition]
        queue = planned_training.get("training_queue")
        if isinstance(queue, Mapping):
            observed_queue = _paper_queue_attestation(
                Path(str(queue["manifest"]["path"])).parent,
                condition=condition,
                seed=int(contract["seed"]),
                run_root=Path(str(planned_training["training_run_root"])),
            )
            if observed_queue != queue:
                raise MatchedEvaluationError(
                    f"{condition}: persisted training queue attestation drifted"
                )
        if formal_v2:
            from tools import run_stageb_table_b_v2_queue as v2_queue

            planned_v2 = planned_training.get("formal_v2")
            if not isinstance(planned_v2, Mapping):
                raise MatchedEvaluationError(
                    f"{condition}: persisted formal v2 evidence is missing"
                )
            try:
                observed_v2 = v2_queue.formal_evaluation_evidence(
                    Path(str(planned_v2["completion_attestation"]["path"])).parent,
                    run_id=f"{condition}:{contract['seed']}",
                    run_root=Path(str(planned_training["training_run_root"])),
                )
            except (OSError, ValueError, v2_queue.FormalQueueError) as error:
                raise MatchedEvaluationError(
                    f"{condition}: persisted formal v2 replay failed: {error}"
                ) from error
            if observed_v2 != planned_v2:
                raise MatchedEvaluationError(
                    f"{condition}: persisted formal v2 evidence drifted"
                )
    binding = load_binding(
        Path(contract["surface"]["matched_eval_surface_binding_path"]),
        expected_derived=Path(
            contract["surface"]["matched_eval_surface_derived_path"]
        ),
    )
    _validate_artifact_layout(launch, binding=binding)
    expected_surface = {
        **surface_summary_fields(binding),
        "binding_replay_passed": True,
    }
    if postflight.get("surface") != expected_surface:
        raise MatchedEvaluationError("postflight surface replay drift")
    conditions = postflight.get("conditions")
    if not isinstance(conditions, Mapping) or set(conditions) != set(CONDITIONS):
        raise MatchedEvaluationError("postflight condition set drift")
    final_paths = {}
    logical_rows = _read_jsonl(
        Path(binding.source_manifest["path"]), label="logical D3m source"
    )
    audit_records = [
        record
        for record in contract["input_records"]
        if "matched_panel_audit" in record.get("roles", [])
    ]
    if len(audit_records) != 1:
        raise MatchedEvaluationError("persisted contract lacks one matched audit")
    audit_record = audit_records[0]
    for phase in contract["phases"]:
        condition = str(phase["condition"])
        artifacts = conditions[condition]
        if not isinstance(artifacts, Mapping):
            raise MatchedEvaluationError(f"{condition}: artifact set missing")
        raw_path = _verify_declared_artifact(
            artifacts["raw_records"], label=f"{condition} raw records"
        )
        final_path = _verify_declared_artifact(
            artifacts["final_records"], label=f"{condition} final records"
        )
        _verify_declared_artifact(
            artifacts["raw_summary"], label=f"{condition} raw summary"
        )
        raw_rows = _read_jsonl(raw_path, label=f"{condition} raw records")
        final_rows = _read_jsonl(final_path, label=f"{condition} final records")
        training = contract["training"][condition]
        training_source = training["training_source"]
        if len(raw_rows) != len(final_rows):
            raise MatchedEvaluationError(f"{condition}: replay row-count mismatch")
        for index, (raw, final, logical) in enumerate(
            zip(raw_rows, final_rows, logical_rows)
        ):
            if not (
                final.get("manifest_index") == index
                and final.get("raw_manifest_index") == index
                and final.get("run_id") == f"{condition}:{contract['seed']}"
                and final.get("table_b_id") == condition
                and final.get("train_seed") == contract["seed"]
                and final.get("checkpoint_sha256")
                == training["checkpoint"]["sha256"]
                and final.get("training_source_sha256")
                == training_source["sha256"]
                and final.get("training_source_n") == training_source["rows"]
                and final.get("matched_panel_audit_sha256")
                == audit_record["sha256"]
                and final.get("evaluation_source_sha256")
                == binding.source_manifest["sha256"]
                and final.get("evaluation_source_n")
                == binding.source_manifest["rows"]
                and final.get("evaluation_manifest_sha256")
                == binding.source_manifest["sha256"]
                and final.get("eval_scope") == DECLARED_SCOPE
                and final.get("global_tn_verified") is False
                and final.get("formal_global_fpr_eligible") is False
                and final.get("matched_eval_surface_sha256")
                == binding.derived_manifest["sha256"]
                and final.get("matched_eval_surface_row_mapping_sha256")
                == binding.row_mapping_sha256
                and final.get("matched_pair_id") == logical.get("matched_pair_id")
                and final.get("matched_parent_key_sha256")
                == logical.get("matched_parent_key_sha256")
                and final.get("support_input_kind")
                == raw.get("support_input_kind")
                and final.get("support_input_sha256")
                == raw.get("support_input_sha256")
                and final.get("support_class_ids")
                == raw.get("support_class_ids")
                and all(
                    float(raw[field]) == float(final[field])
                    for field in ("pos_score", "neg_score", "pos_iou", "neg_iou")
                )
            ):
                raise MatchedEvaluationError(
                    f"{condition}: finalized replay drift at row {index}"
                )
        final_paths[condition] = final_path
    audit = verify_panel(Path(audit_record["path"]))
    seed = int(contract["seed"])
    report = aggregate_matched_panel(
        audit_path=Path(audit_record["path"]),
        pair_ledger_path=Path(binding.pair_ledger["path"]),
        d2m_source_path=Path(audit["outputs"]["d2m_calibration"]["path"]),
        d3m_source_path=Path(binding.source_manifest["path"]),
        evaluation_manifest_path=Path(binding.source_manifest["path"]),
        d2m_records={seed: final_paths["D2m"]},
        d3m_records={seed: final_paths["D3m"]},
        expected_seeds=[seed],
    )
    if _compact_aggregator_result(report, seed=seed) != postflight.get(
        "aggregator_replay"
    ):
        raise MatchedEvaluationError("postflight aggregator replay drift")
    return postflight


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dry-run", "run"))
    parser.add_argument("--d2m-training-run-root", type=Path, required=True)
    parser.add_argument("--d3m-training-run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--pair-ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--d3m-source", type=Path, default=DEFAULT_D3M_SOURCE)
    parser.add_argument("--data-root", type=Path, default=Path("/media/haoyi/T9/data"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--training-queue-dir", type=Path, required=True)
    parser.add_argument(
        "--training-source-contract",
        choices=TRAINING_SOURCE_CONTRACTS,
        default=LEGACY_TRAINING_SOURCE_CONTRACT,
        help="select the legacy resolver or the attested formal Table-B v2 source",
    )
    parser.add_argument(
        "--validation-queue-spec",
        type=Path,
        help="optional immutable Table-B v2 validation plan bound into inputs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        launch = prepare_evaluation(
            output_dir=args.output_dir,
            audit_path=args.audit,
            ledger_path=args.pair_ledger,
            d3m_source_path=args.d3m_source,
            data_root=args.data_root,
            d2m_training_run_root=args.d2m_training_run_root,
            d3m_training_run_root=args.d3m_training_run_root,
            seed=args.seed,
            python=args.python,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            amp=args.amp,
            log_every=args.log_every,
            training_queue_dir=args.training_queue_dir,
            training_source_contract=args.training_source_contract,
            validation_queue_spec=args.validation_queue_spec,
        )
        if args.mode == "dry-run":
            launch["status"] = "dry_run_completed"
            _write_json_atomic(Path(launch["output_dir"]) / "launch.json", launch)
            for phase in launch["contract"]["phases"]:
                print(phase["command_shell"])
            return 0
        sources, _evidence = _resolve_sources(
            d2m_root=args.d2m_training_run_root,
            d3m_root=args.d3m_training_run_root,
            seed=args.seed,
            audit=verify_panel(Path(args.audit).resolve(strict=True)),
            audit_path=Path(args.audit).resolve(strict=True),
            training_queue_dir=args.training_queue_dir,
            formal_v2=(
                args.training_source_contract
                == FORMAL_V2_TRAINING_SOURCE_CONTRACT
            ),
        )
        run_evaluation(launch, sources=sources)
        print(Path(launch["output_dir"]) / "postflight.json")
        return 0
    except (
        FileExistsError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        PaperEvaluationError,
        MatchedPanelError,
        MatchedEvalSurfaceError,
        MatchedEvaluationError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
