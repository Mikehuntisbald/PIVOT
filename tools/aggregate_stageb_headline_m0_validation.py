#!/usr/bin/env python3
"""Aggregate the exact M0 versus M0N three-seed validation comparison."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import aggregate_stageb_matrix_validation as matrix  # noqa: E402
from tools import run_stageb_headline_m0_validation_queue as queue_runner  # noqa: E402


REPORT_SCHEMA = "pivot.stageb.headline_m0_validation_report/v1"
REPORT_STATUS = "validated_matrix_validation_only"
FORMAL_BOOTSTRAP_ITERATIONS = 5_000
FORMAL_BOOTSTRAP_CONFIDENCE = 0.95
FORMAL_BOOTSTRAP_SEED = 20260719
DEFAULT_QUEUE_DIR = queue_runner.DEFAULT_QUEUE_DIR
DEFAULT_REPORT_PATH = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/aggregates/headline_m0_m0n_validation_report.json"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class HeadlineM0AggregationError(RuntimeError):
    """The M0/M0N validation evidence or aggregation contract drifted."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(value: Any) -> str:
    rendered = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(rendered).hexdigest()


def _report_semantic_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("created_at_utc", None)
    payload.pop("report_sha256", None)
    return payload


def _report_sha256(value: Mapping[str, Any]) -> str:
    return _canonical_sha256(_report_semantic_payload(value))


def _validate_report_envelope(
    value: Mapping[str, Any], *, label: str
) -> None:
    if value.get("schema") != REPORT_SCHEMA or value.get("status") != REPORT_STATUS:
        raise HeadlineM0AggregationError(
            f"{label} schema/status is not the exact formal M0/M0N contract"
        )
    created_at = value.get("created_at_utc")
    if not isinstance(created_at, str):
        raise HeadlineM0AggregationError(f"{label} created_at_utc is invalid")
    try:
        parsed = datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise HeadlineM0AggregationError(
            f"{label} created_at_utc is invalid"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise HeadlineM0AggregationError(f"{label} created_at_utc is not UTC")
    digest = value.get("report_sha256")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise HeadlineM0AggregationError(f"{label} report SHA-256 is invalid")
    if digest != _report_sha256(value):
        raise HeadlineM0AggregationError(f"{label} self SHA-256 mismatch")


def _write_json_no_replace(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    rendered = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"
    try:
        with temporary.open("x", encoding="ascii") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise HeadlineM0AggregationError(
                f"refusing to overwrite aggregation report: {path}"
            ) from exc
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _verify_predeclared_sources(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = plan.get("aggregation_sources")
    if not isinstance(records, list) or not records:
        raise HeadlineM0AggregationError("aggregation source closure is missing")
    rendered = []
    for index, record in enumerate(records):
        try:
            queue_runner._verify_file_record(
                record, label=f"aggregation source {index}"
            )
        except (OSError, ValueError, queue_runner.HeadlineValidationQueueError) as exc:
            raise HeadlineM0AggregationError(str(exc)) from exc
        rendered.append(dict(record))
    own = str(Path(__file__).resolve(strict=True))
    if sum(str(record.get("path")) == own for record in rendered) != 1:
        raise HeadlineM0AggregationError(
            "aggregation source closure does not uniquely bind this aggregator"
        )
    return rendered


def _queue_bound_file_snapshot(plan: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Collect the queue-owned files that must remain stable through aggregation."""

    records: list[Any] = [plan.get("runner_python"), plan.get("evaluation_runner")]
    for key in ("evaluation_sources", "controller_sources", "aggregation_sources"):
        values = plan.get(key)
        if not isinstance(values, list) or not values:
            raise HeadlineM0AggregationError(f"validation queue {key} is missing")
        records.extend(values)
    training_queues = plan.get("training_queues")
    if not isinstance(training_queues, list) or len(training_queues) != len(
        queue_runner.CONTRACT_IDS
    ):
        raise HeadlineM0AggregationError(
            "validation queue training bindings are incomplete"
        )
    records.extend(
        record.get("manifest_at_creation")
        if isinstance(record, Mapping)
        else None
        for record in training_queues
    )

    by_path: dict[str, dict[str, Any]] = {}
    expected_keys = {"path", "sha256", "size_bytes", "mtime_ns"}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != expected_keys:
            raise HeadlineM0AggregationError(
                f"validation queue bound file record {index} is invalid"
            )
        normalized = dict(record)
        path = str(Path(str(normalized["path"])).expanduser().resolve(strict=False))
        normalized["path"] = path
        previous = by_path.setdefault(path, normalized)
        if previous != normalized:
            raise HeadlineM0AggregationError(
                f"validation queue bound file identities conflict for {path}"
            )
    return tuple(by_path[path] for path in sorted(by_path))


def _verify_final_aggregation_evidence(
    *,
    spec_snapshot: tuple[Mapping[str, Any], ...],
    queue_snapshot: tuple[Mapping[str, Any], ...],
    queue_bound_files: tuple[Mapping[str, Any], ...],
    loaded: Mapping[str, Mapping[int, matrix.LoadedRun]],
) -> None:
    """Rehash every input after statistics are computed and before returning."""

    try:
        matrix._verify_snapshot_files(spec_snapshot, label="M0/M0N aggregation spec")
        matrix._verify_snapshot_files(
            queue_snapshot, label="M0/M0N validation queue"
        )
        matrix._verify_snapshot_files(
            queue_bound_files, label="M0/M0N validation queue source closure"
        )
        for contract_id, runs in loaded.items():
            for seed, run in runs.items():
                matrix._verify_snapshot_files(
                    run.sealed_files,
                    label=f"{contract_id}:{seed} evaluation evidence",
                )
    except (OSError, ValueError, matrix.MatrixValidationError) as exc:
        raise HeadlineM0AggregationError(
            f"M0/M0N evidence changed during aggregation: {exc}"
        ) from exc


def _replay_headline_plan_contract(
    launch: Mapping[str, Any],
    *,
    source: queue_runner.evaluator.EvaluationSource,
    runtime: queue_runner.evaluator.Runtime,
    output_root: Path,
    spec_path: Path,
    cache: queue_runner.evaluator.HashCache,
) -> dict[str, Any]:
    fixed_runtime = {
        "batch_size": 16,
        "num_workers": 4,
        "amp": True,
        "log_every": 50,
    }
    for key, expected in fixed_runtime.items():
        if getattr(runtime, key) != expected:
            raise HeadlineM0AggregationError(
                f"M0/M0N matrix runtime {key} must be exactly {expected!r}"
            )
    evaluator = queue_runner.evaluator
    try:
        commands = evaluator._commands(
            runtime, source, output_root, profile=queue_runner.PROFILE
        )
        calibration = evaluator._screen_calibration_contract(cache)
        entries: list[tuple[Path, str]] = [
            (source.config, "evaluation_config"),
            (source.checkpoint, "evaluation_checkpoint"),
            (spec_path, "matrix_validation_queue_spec"),
        ]
        entries.extend(
            (path, "config_dependency")
            for path in evaluator._config_paths(source.config)
        )
        entries.extend(
            (path, "evaluation_code_dependency")
            for path in evaluator._evaluation_code_paths()
        )
        entries.extend(
            (path, "source_provenance_dependency")
            for path in evaluator._evaluation_source_provenance_paths(source)
        )
        entries.extend(
            (path, "evaluation_data_input")
            for path in evaluator._data_input_paths(runtime.data_root)
        )
        entries.append(
            (
                Path(calibration["source_manifest"]["path"]),
                "matrix_calibration_source",
            )
        )
        entries.append(
            (
                Path(calibration["source_audit"]["path"]),
                "matrix_calibration_audit",
            )
        )
        entries.extend((path, "training_data") for path in source.training_data)
        for path, role in (
            (source.sequence_manifest, "training_sequence_manifest"),
            (source.final_phase_manifest, "training_final_phase_manifest"),
            (source.training_postflight, "training_final_phase_postflight"),
            (source.selected_phase_manifest, "training_selected_phase_manifest"),
            (
                source.selected_training_postflight,
                "training_selected_phase_postflight",
            ),
            (source.training_queue_manifest, "training_queue_manifest"),
            (source.training_queue_detached_launch, "training_queue_detached_launch"),
            (source.training_queue_detached_status, "training_queue_detached_status"),
        ):
            if path is not None:
                entries.append((path, role))
        expected_inputs = {
            "algorithm": "sha256",
            "records": evaluator._merge_input_records(entries, cache),
        }
        expected_spec = evaluator._file_record(
            spec_path, cache, roles=("matrix_validation_queue_spec",)
        )
    except (evaluator.PaperEvaluationError, OSError, ValueError) as exc:
        raise HeadlineM0AggregationError(
            f"M0/M0N canonical evaluation replay failed: {exc}"
        ) from exc
    protocol = launch.get("protocol")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("screen_calibration") != calibration
        or launch.get("commands") != commands
        or launch.get("inputs") != expected_inputs
        or launch.get("matrix_validation_queue_spec") != expected_spec
    ):
        raise HeadlineM0AggregationError(
            "M0/M0N launch differs from its canonical queue-bound matrix plan"
        )
    return {
        "python": str(runtime.python),
        "data_root": str(runtime.data_root),
        "device": runtime.device,
        "batch_size": runtime.batch_size,
        "num_workers": runtime.num_workers,
        "amp": runtime.amp,
        "log_every": runtime.log_every,
        "eval_seed": evaluator.EVAL_SEED,
        "max_ref_batches": 0,
        "max_tn_batches": 0,
    }


def _validate_headline_launch(
    launch: Mapping[str, Any],
    *,
    contract_id: str,
    seed: int,
    root: Path,
    training_queue: Mapping[str, Any],
    spec_path: Path,
    cache: queue_runner.evaluator.HashCache,
) -> tuple[str, str, Path, Path, str, Mapping[str, Any]]:
    evaluator = queue_runner.evaluator
    if launch.get("schema") != evaluator.SCHEMA or launch.get("status") != "completed":
        raise HeadlineM0AggregationError("M0/M0N evaluation launch is not completed")
    if Path(str(launch.get("output_dir", ""))).resolve(strict=True) != root:
        raise HeadlineM0AggregationError("M0/M0N evaluation output root drifted")
    protocol = launch.get("protocol")
    expected_protocol = {
        "profile": queue_runner.PROFILE,
        "ref_splits": list(matrix.REF_VALIDATION_SPLITS),
        "strict_manifests": {},
        "processes": ["validation_calibration"],
        "strict1607_skip_ref": False,
        "per_example_records": True,
        "release_policy": (
            "ablation_matrix_validation_only_no_ref_test_or_strict_access"
        ),
    }
    if not isinstance(protocol, Mapping) or any(
        protocol.get(key) != expected for key, expected in expected_protocol.items()
    ):
        raise HeadlineM0AggregationError("M0/M0N matrix protocol drifted")
    if not isinstance(protocol.get("screen_calibration"), Mapping):
        raise HeadlineM0AggregationError("M0/M0N calibration contract is missing")
    completed = launch.get("completed_phases")
    if not (
        isinstance(completed, list)
        and len(completed) == 1
        and isinstance(completed[0], Mapping)
        and completed[0].get("phase_id") == "validation_calibration"
        and completed[0].get("status") == "completed"
        and completed[0].get("returncode") == 0
    ):
        raise HeadlineM0AggregationError("M0/M0N evaluation phase is not exact")
    raw_source = launch.get("source")
    run_id = f"{contract_id}:{seed}"
    evaluation_id = f"{contract_id}_seed{seed}"
    contract = queue_runner.training_runner.CONTRACTS[contract_id]
    if (
        not isinstance(raw_source, Mapping)
        or raw_source.get("kind") != "pivot_paper_training_run"
        or raw_source.get("formal_contract_id") != contract_id
        or raw_source.get("evaluation_id") != evaluation_id
        or launch.get("evaluation_id") != evaluation_id
        or raw_source.get("training_run_id") != run_id
        or raw_source.get("training_seed") != seed
        or raw_source.get("training_phase", "final") != "final"
        or raw_source.get("diagnostic_only", False) is not False
        or raw_source.get("matrix_validation_only")
        is not contract.matrix_validation_only
        or raw_source.get("training_queue_id") != training_queue["queue_id"]
        or raw_source.get("training_queue_plan_sha256")
        != training_queue["plan_sha256"]
    ):
        raise HeadlineM0AggregationError(
            f"{run_id} is not the exact formal queue-attested source"
        )
    try:
        source = matrix._evaluation_source_from_launch(raw_source)
    except (OSError, ValueError, matrix.MatrixValidationError) as exc:
        raise HeadlineM0AggregationError(f"cannot reconstruct {run_id} source: {exc}") from exc
    expected_training_root = contract.canonical_training_root(seed)
    expected_queue_manifest = (
        Path(str(training_queue["queue_dir"])) / "queue.json"
    ).resolve(strict=True)
    if (
        source.training_run_root != expected_training_root
        or source.training_queue_manifest != expected_queue_manifest
    ):
        raise HeadlineM0AggregationError(f"{run_id} training root/queue drifted")
    try:
        evaluator._revalidate_matrix_source(source, cache)
        runtime = matrix._runtime_from_launch(launch.get("runtime"))
    except (OSError, ValueError, evaluator.PaperEvaluationError, matrix.MatrixValidationError) as exc:
        raise HeadlineM0AggregationError(
            f"{run_id} formal source/runtime replay failed: {exc}"
        ) from exc
    runtime_fingerprint = _replay_headline_plan_contract(
        launch,
        source=source,
        runtime=runtime,
        output_root=root,
        spec_path=spec_path,
        cache=cache,
    )
    checkpoint_sha = str(raw_source.get("checkpoint_sha256", "")).lower()
    if matrix._SHA_RE.fullmatch(checkpoint_sha) is None:
        raise HeadlineM0AggregationError(f"{run_id} checkpoint SHA-256 is invalid")
    return (
        evaluation_id,
        run_id,
        expected_training_root,
        source.checkpoint,
        checkpoint_sha,
        runtime_fingerprint,
    )


def _load_headline_run(
    *,
    contract_id: str,
    seed: int,
    root: Path,
    training_queue: Mapping[str, Any],
    spec_path: Path,
) -> matrix.LoadedRun:
    root = root.resolve(strict=True)
    evaluator = queue_runner.evaluator
    cache = evaluator.HashCache()
    launch_path = (root / "launch_manifest.json").resolve(strict=True)
    postflight_path = (root / "postflight.json").resolve(strict=True)
    launch = matrix._read_json(launch_path, label="M0/M0N matrix launch")
    postflight = matrix._read_json(postflight_path, label="M0/M0N matrix postflight")
    matrix._assert_validation_only_root(root, launch)
    (
        evaluation_id,
        training_run_id,
        training_run_root,
        checkpoint,
        checkpoint_sha,
        runtime_fingerprint,
    ) = _validate_headline_launch(
        launch,
        contract_id=contract_id,
        seed=seed,
        root=root,
        training_queue=training_queue,
        spec_path=spec_path,
        cache=cache,
    )
    matrix._verify_file_record(
        launch.get("postflight_artifact"),
        label="M0/M0N matrix postflight artifact",
        cache=cache,
        expected_path=postflight_path,
    )
    if launch.get("postflight") != postflight:
        raise HeadlineM0AggregationError("M0/M0N launch embeds another postflight")
    input_rehash = matrix._verify_input_rehash(
        launch, postflight, root=root, cache=cache
    )
    matrix._validate_postflight(launch, postflight, input_rehash)
    checkpoint_contract = postflight.get("checkpoint")
    checkpoint_run_id = (
        str(checkpoint_contract.get("run_id") or "")
        if isinstance(checkpoint_contract, Mapping)
        else ""
    )
    if (
        not isinstance(checkpoint_contract, Mapping)
        or Path(str(checkpoint_contract.get("path", ""))).resolve(strict=True)
        != checkpoint
        or str(checkpoint_contract.get("sha256", "")).lower() != checkpoint_sha
        or not checkpoint_run_id
        or cache.digest(checkpoint) != checkpoint_sha
    ):
        raise HeadlineM0AggregationError("M0/M0N checkpoint evidence drifted")

    artifacts = postflight["artifacts"]
    section_dir = (root / "validation_calibration").resolve(strict=True)
    summary_path = matrix._verify_file_record(
        artifacts["summary"],
        label="M0/M0N matrix summary",
        cache=cache,
        expected_path=section_dir / "summary.json",
    )
    summary = matrix._read_json(summary_path, label="M0/M0N matrix summary")
    if set(summary) != {"refcoco", "tn"}:
        raise HeadlineM0AggregationError("M0/M0N summary surface drifted")
    ref_rows = summary.get("refcoco")
    tn_rows = summary.get("tn")
    if not isinstance(ref_rows, list) or not isinstance(tn_rows, list):
        raise HeadlineM0AggregationError("M0/M0N summary sections are invalid")
    by_split: dict[str, Mapping[str, Any]] = {}
    for row in ref_rows:
        if not isinstance(row, Mapping):
            raise HeadlineM0AggregationError("M0/M0N Ref summary row is invalid")
        split = str(row.get("dataset") or "")
        if split in by_split:
            raise HeadlineM0AggregationError(f"duplicate M0/M0N Ref split: {split}")
        by_split[split] = row
    if tuple(by_split) != matrix.REF_VALIDATION_SPLITS:
        raise HeadlineM0AggregationError("M0/M0N Ref split order drifted")
    ref_artifacts = artifacts.get("ref_validation")
    if not isinstance(ref_artifacts, Mapping) or set(ref_artifacts) != set(
        matrix.REF_VALIDATION_SPLITS
    ):
        raise HeadlineM0AggregationError("M0/M0N Ref artifact set drifted")
    ref = {
        split: matrix._load_ref_surface(
            split=split,
            summary_row=by_split[split],
            artifact=ref_artifacts[split],
            summary_path=summary_path,
            section_dir=section_dir,
            run_id=checkpoint_run_id,
            cache=cache,
        )
        for split in matrix.REF_VALIDATION_SPLITS
    }
    if len(tn_rows) != 1 or not isinstance(tn_rows[0], Mapping):
        raise HeadlineM0AggregationError("M0/M0N calibration row set drifted")
    calibration = matrix._load_calibration_surface(
        summary_row=tn_rows[0],
        artifact=artifacts["matrix_calibration"],
        section_dir=section_dir,
        run_id=checkpoint_run_id,
        cache=cache,
    )
    expected_record_paths = {
        Path(str(surface.records["path"])).resolve(strict=True)
        for surface in ref.values()
    } | {Path(str(calibration.records["path"])).resolve(strict=True)}
    observed_record_paths = {
        path.resolve(strict=True) for path in root.rglob("*.records.jsonl")
    }
    if observed_record_paths != expected_record_paths:
        raise HeadlineM0AggregationError("M0/M0N per-example artifact set drifted")
    code_fingerprint = matrix._input_role_fingerprint(
        launch, "evaluation_code_dependency"
    )
    data_fingerprint = matrix._input_role_fingerprint(
        launch, "evaluation_data_input"
    )
    surface_fingerprint = {
        "ref_validation": {
            split: {
                "rows": int(matrix.REF_VALIDATION_CONTRACT[split]["rows"]),
                "sha256": str(matrix.REF_VALIDATION_CONTRACT[split]["sha256"]),
            }
            for split in matrix.REF_VALIDATION_SPLITS
        },
        "calibration": {
            "rows": matrix.CALIBRATION_ROWS,
            "source_manifest_sha256": calibration.source_manifest_sha256,
            "source_audit_sha256": calibration.source_audit_sha256,
            "derived_manifest_sha256": calibration.derived_manifest_sha256,
            "row_mapping_sha256": calibration.row_mapping_sha256,
        },
    }
    sealed_paths = {
        launch_path,
        root / "input_rehash.json",
        postflight_path,
        summary_path,
        checkpoint,
        *(surface.evaluation_manifest for surface in ref.values()),
        *(Path(str(surface.records["path"])) for surface in ref.values()),
        Path(str(calibration.records["path"])),
    }
    inputs = launch.get("inputs")
    input_records = inputs.get("records") if isinstance(inputs, Mapping) else None
    if not isinstance(input_records, list):
        raise HeadlineM0AggregationError("M0/M0N launch inputs disappeared")
    for record in input_records:
        if not isinstance(record, Mapping):
            raise HeadlineM0AggregationError("M0/M0N input record is invalid")
        sealed_paths.add(Path(str(record.get("path", ""))))
    calibration_artifact = artifacts.get("matrix_calibration")
    if not isinstance(calibration_artifact, Mapping):
        raise HeadlineM0AggregationError("M0/M0N calibration artifact disappeared")
    for key in ("source_manifest", "source_audit", "derived_manifest", "binding", "records"):
        record = calibration_artifact.get(key)
        if not isinstance(record, Mapping):
            raise HeadlineM0AggregationError(f"M0/M0N calibration lacks {key}")
        sealed_paths.add(Path(str(record.get("path", ""))))
    return matrix.LoadedRun(
        experiment_id=contract_id,
        seed=seed,
        root=root,
        evaluation_id=evaluation_id,
        training_run_id=training_run_id,
        training_run_root=training_run_root,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_run_id=checkpoint_run_id,
        ref=ref,
        calibration=calibration,
        code_fingerprint=code_fingerprint,
        data_fingerprint=data_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        surface_fingerprint=surface_fingerprint,
        evidence={
            "launch": matrix._compact_file_record(launch_path, cache),
            "input_rehash": matrix._compact_file_record(
                root / "input_rehash.json", cache
            ),
            "postflight": matrix._compact_file_record(postflight_path, cache),
            "summary": matrix._compact_file_record(summary_path, cache),
            "ref_records": {
                split: matrix._compact_file_record(
                    Path(str(ref[split].records["path"])), cache
                )
                for split in matrix.REF_VALIDATION_SPLITS
            },
            "ref_evaluation_manifests": {
                split: matrix._compact_file_record(
                    ref[split].evaluation_manifest, cache
                )
                for split in matrix.REF_VALIDATION_SPLITS
            },
            "calibration_records": matrix._compact_file_record(
                Path(str(calibration.records["path"])), cache
            ),
        },
        sealed_files=matrix._snapshot_files(sealed_paths, cache),
    )


def _load_all_runs(
    spec: Mapping[str, Any], queue: Mapping[str, Any]
) -> dict[str, dict[int, matrix.LoadedRun]]:
    experiments = spec.get("experiments")
    if not isinstance(experiments, list) or [
        value.get("id") if isinstance(value, Mapping) else None
        for value in experiments
    ] != list(queue_runner.CONTRACT_IDS):
        raise HeadlineM0AggregationError("aggregation experiments must be exactly M0/M0N")
    loaded: dict[str, dict[int, matrix.LoadedRun]] = {}
    seen_roots: set[Path] = set()
    seen_training_ids: set[str] = set()
    seen_checkpoints: set[Path] = set()
    seen_checkpoint_shas: set[str] = set()
    common_code: Mapping[str, Mapping[str, Any]] | None = None
    common_data: Mapping[str, Mapping[str, Any]] | None = None
    common_runtime: Mapping[str, Any] | None = None
    common_surface: Mapping[str, Any] | None = None
    ref_identities: dict[str, tuple[tuple[Any, ...], ...]] = {}
    calibration_identities: tuple[tuple[Any, ...], ...] | None = None
    training_queues = {
        str(record["contract_id"]): record
        for record in queue["plan"]["training_queues"]
    }
    spec_path = (
        Path(str(queue["plan"]["queue_dir"])) / queue_runner.AGGREGATION_SPEC_NAME
    ).resolve(strict=True)

    for experiment in experiments:
        assert isinstance(experiment, Mapping)
        contract_id = str(experiment["id"])
        roots = experiment.get("evaluation_roots")
        if not isinstance(roots, Mapping) or set(roots) != {
            str(seed) for seed in queue_runner.SEEDS
        }:
            raise HeadlineM0AggregationError(
                f"{contract_id} evaluation root seed set drifted"
            )
        runs: dict[int, matrix.LoadedRun] = {}
        for seed in queue_runner.SEEDS:
            root = Path(str(roots[str(seed)])).expanduser().resolve(strict=True)
            if root in seen_roots:
                raise HeadlineM0AggregationError(f"duplicate evaluation root: {root}")
            seen_roots.add(root)
            try:
                run = _load_headline_run(
                    contract_id=contract_id,
                    seed=seed,
                    root=root,
                    training_queue=training_queues[contract_id],
                    spec_path=spec_path,
                )
            except (
                OSError,
                ValueError,
                matrix.MatrixValidationError,
                HeadlineM0AggregationError,
            ) as exc:
                raise HeadlineM0AggregationError(
                    f"cannot load {contract_id}:{seed}: {exc}"
                ) from exc
            if run.training_run_id != f"{contract_id}:{seed}":
                raise HeadlineM0AggregationError(
                    f"{contract_id}:{seed} training identity drifted"
                )
            if run.training_run_id in seen_training_ids:
                raise HeadlineM0AggregationError("duplicate training run identity")
            if run.checkpoint in seen_checkpoints:
                raise HeadlineM0AggregationError("duplicate checkpoint path")
            if run.checkpoint_sha256 in seen_checkpoint_shas:
                raise HeadlineM0AggregationError("duplicate checkpoint content")
            seen_training_ids.add(run.training_run_id)
            seen_checkpoints.add(run.checkpoint)
            seen_checkpoint_shas.add(run.checkpoint_sha256)
            if common_code is None:
                common_code = run.code_fingerprint
                common_data = run.data_fingerprint
                common_runtime = run.runtime_fingerprint
                common_surface = run.surface_fingerprint
            elif (
                run.code_fingerprint != common_code
                or run.data_fingerprint != common_data
                or run.runtime_fingerprint != common_runtime
                or run.surface_fingerprint != common_surface
            ):
                raise HeadlineM0AggregationError(
                    "M0/M0N evaluations differ in code, data, runtime, or surface"
                )
            for split in matrix.REF_VALIDATION_SPLITS:
                identities = run.ref[split].identities
                previous = ref_identities.setdefault(split, identities)
                if identities != previous:
                    raise HeadlineM0AggregationError(
                        f"M0/M0N {split} record identities are not aligned"
                    )
            identities = run.calibration.identities
            if calibration_identities is None:
                calibration_identities = identities
            elif identities != calibration_identities:
                raise HeadlineM0AggregationError(
                    "M0/M0N calibration record identities are not aligned"
                )
            runs[seed] = run
        loaded[contract_id] = runs
    return loaded


def aggregate(
    queue_dir: Path,
    *,
    bootstrap_iterations: int = FORMAL_BOOTSTRAP_ITERATIONS,
    confidence: float = FORMAL_BOOTSTRAP_CONFIDENCE,
    bootstrap_seed: int = FORMAL_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if (
        type(bootstrap_iterations) is not int
        or bootstrap_iterations != FORMAL_BOOTSTRAP_ITERATIONS
        or type(bootstrap_seed) is not int
        or bootstrap_seed != FORMAL_BOOTSTRAP_SEED
        or confidence != FORMAL_BOOTSTRAP_CONFIDENCE
    ):
        raise HeadlineM0AggregationError(
            "formal M0/M0N bootstrap must be exactly 5000/0.95/20260719"
        )
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    entry_cache = queue_runner.evaluator.HashCache()
    spec_path = (queue_dir / queue_runner.AGGREGATION_SPEC_NAME).resolve(strict=True)
    queue_snapshot = matrix._snapshot_files((queue_dir / "queue.json",), entry_cache)
    spec_snapshot = matrix._snapshot_files((spec_path,), entry_cache)
    queue = queue_runner.load_queue(queue_dir)
    verification = queue_runner.verify_queue(queue_dir)
    if queue.get("status") != "completed" or verification.get("status") != "passed":
        raise HeadlineM0AggregationError(
            "M0/M0N validation queue is not completed and verified"
        )
    plan = queue["plan"]
    queue_bound_files = _queue_bound_file_snapshot(plan)
    sources = _verify_predeclared_sources(plan)
    spec = queue_runner._read_json(spec_path, label="M0/M0N aggregation input")
    expected_spec = queue_runner._aggregation_spec_payload(
        plan, str(queue["plan_sha256"])
    )
    if spec != expected_spec:
        raise HeadlineM0AggregationError(
            "M0/M0N aggregation input differs from the immutable queue"
        )
    loaded = _load_all_runs(spec, queue)
    try:
        comparison = matrix._comparison(
            reference_id="M0",
            candidate_id="M0N",
            reference_runs=loaded["M0"],
            candidate_runs=loaded["M0N"],
            iterations=bootstrap_iterations,
            confidence=confidence,
            bootstrap_seed=bootstrap_seed,
        )
        experiments = {
            contract_id: {
                "id": contract_id,
                **matrix._aggregate_experiment(loaded[contract_id]),
            }
            for contract_id in queue_runner.CONTRACT_IDS
        }
    except (ValueError, matrix.MatrixValidationError) as exc:
        raise HeadlineM0AggregationError(f"paired aggregation failed: {exc}") from exc
    checkpoint_shas = {
        contract_id: {
            str(seed): loaded[contract_id][seed].checkpoint_sha256
            for seed in queue_runner.SEEDS
        }
        for contract_id in queue_runner.CONTRACT_IDS
    }
    if any(
        len(set(values.values())) != len(queue_runner.SEEDS)
        for values in checkpoint_shas.values()
    ):
        raise HeadlineM0AggregationError("a contract reused a checkpoint across seeds")
    report = {
        "schema": REPORT_SCHEMA,
        "status": REPORT_STATUS,
        "created_at_utc": _utc_now(),
        "formal_test_or_strict_result": False,
        "comparison_claim": "full_token_objective_control_not_labels_only",
        "reference_experiment": "M0",
        "candidate_experiment": "M0N",
        "direction": "M0N_minus_M0",
        "protocol": {
            "profile": evaluator_profile(),
            "train_seeds": list(queue_runner.SEEDS),
            "seed_estimator": "equal-seed mean and sample standard deviation",
            "paired_bootstrap": {
                "iterations": bootstrap_iterations,
                "confidence": confidence,
                "seed": bootstrap_seed,
                "unit": "image cluster within training seed",
                "seed_first": True,
            },
            "ref_test_access": False,
            "strict_tn_access": False,
        },
        "validation": {
            "pass": True,
            "training_queues_separate": True,
            "exact_six_evaluations": True,
            "record_identities_aligned": True,
            "runtime_code_data_surface_equal": True,
            "input_rehash_and_postflight_replayed": True,
        },
        "inputs": {
            "aggregation_spec": dict(spec_snapshot[0]),
            "evaluation_queue": {
                "queue_id": plan["queue_id"],
                "plan_sha256": queue["plan_sha256"],
                "verification_schema": verification["schema"],
            },
            "aggregation_source_closure": sources,
            "checkpoint_sha256s": checkpoint_shas,
        },
        "experiments": experiments,
        "comparison": comparison,
    }
    report["report_sha256"] = _report_sha256(report)
    _verify_final_aggregation_evidence(
        spec_snapshot=spec_snapshot,
        queue_snapshot=queue_snapshot,
        queue_bound_files=queue_bound_files,
        loaded=loaded,
    )
    return report


def verify_report(path: Path = DEFAULT_REPORT_PATH) -> dict[str, Any]:
    canonical = DEFAULT_REPORT_PATH.expanduser().resolve(strict=False)
    path = Path(path).expanduser().resolve(strict=False)
    if path != canonical:
        raise HeadlineM0AggregationError(
            "M0/M0N aggregate report path is not canonical"
        )
    path = path.resolve(strict=True)
    try:
        before_bytes = path.read_bytes()
        value = json.loads(before_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HeadlineM0AggregationError(
            f"cannot read M0/M0N aggregate report {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise HeadlineM0AggregationError(
            "M0/M0N aggregate report must be a JSON object"
        )
    _validate_report_envelope(value, label="M0/M0N aggregate report")

    replayed = aggregate(DEFAULT_QUEUE_DIR)
    _validate_report_envelope(replayed, label="replayed M0/M0N aggregate report")
    if _report_semantic_payload(value) != _report_semantic_payload(replayed):
        raise HeadlineM0AggregationError(
            "M0/M0N aggregate report differs from full canonical replay"
        )
    try:
        after_bytes = path.read_bytes()
    except OSError as exc:
        raise HeadlineM0AggregationError(
            f"cannot re-read M0/M0N aggregate report {path}: {exc}"
        ) from exc
    if after_bytes != before_bytes:
        raise HeadlineM0AggregationError(
            "M0/M0N aggregate report changed during full replay"
        )
    return value


def evaluator_profile() -> str:
    if queue_runner.PROFILE != matrix.PROFILE:
        raise HeadlineM0AggregationError("matrix profile constants drifted")
    return queue_runner.PROFILE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue_dir", nargs="?", type=Path, default=DEFAULT_QUEUE_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--bootstrap-iterations", type=int, default=FORMAL_BOOTSTRAP_ITERATIONS
    )
    parser.add_argument("--confidence", type=float, default=FORMAL_BOOTSTRAP_CONFIDENCE)
    parser.add_argument("--bootstrap-seed", type=int, default=FORMAL_BOOTSTRAP_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    queue_dir = args.queue_dir.expanduser().resolve(strict=False)
    canonical_queue = DEFAULT_QUEUE_DIR.expanduser().resolve(strict=False)
    if queue_dir != canonical_queue:
        parser.error("queue_dir must be the canonical M0/M0N validation queue")
    output = (
        args.output.expanduser().resolve(strict=False)
        if args.output is not None
        else DEFAULT_REPORT_PATH.expanduser().resolve(strict=False)
    )
    canonical_output = DEFAULT_REPORT_PATH.expanduser().resolve(strict=False)
    if output != canonical_output:
        parser.error("--output must be the canonical M0/M0N aggregate report path")
    try:
        report = aggregate(
            canonical_queue,
            bootstrap_iterations=args.bootstrap_iterations,
            confidence=args.confidence,
            bootstrap_seed=args.bootstrap_seed,
        )
        _write_json_no_replace(output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (
        FileNotFoundError,
        OSError,
        ValueError,
        HeadlineM0AggregationError,
        queue_runner.HeadlineValidationQueueError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
