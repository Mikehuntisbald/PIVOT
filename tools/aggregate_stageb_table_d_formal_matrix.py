#!/usr/bin/env python3
"""Verify and aggregate the exact 18-job formal Table-D validation matrix."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import aggregate_stageb_matrix_validation as matrix  # noqa: E402
from tools import aggregate_stageb_table_d_diagnostics as diagnostics  # noqa: E402
from tools import run_stageb_table_d_formal_evaluations as formal_eval  # noqa: E402
from tools import run_stageb_table_d_matrix_validation_queue as queue_runner  # noqa: E402


REPORT_SCHEMA = "pivot.stageb.table_d_formal_matrix_report/v1"
FORMAL_BOOTSTRAP_ITERATIONS = 5_000
FORMAL_BOOTSTRAP_CONFIDENCE = 0.95
FORMAL_BOOTSTRAP_SEED = 20260719


class TableDFormalAggregationError(RuntimeError):
    """The Table-D validation matrix or aggregate contract drifted."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_no_replace(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve(strict=False)
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
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise TableDFormalAggregationError(
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


def _verify_aggregation_sources(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = plan.get("aggregation_sources")
    if not isinstance(records, list) or not records:
        raise TableDFormalAggregationError("aggregation source closure is missing")
    rendered = []
    for index, record in enumerate(records):
        try:
            queue_runner._verify_file_record(
                record, label=f"aggregation source {index}"
            )
        except (OSError, ValueError, queue_runner.TableDValidationQueueError) as exc:
            raise TableDFormalAggregationError(str(exc)) from exc
        rendered.append(dict(record))
    own = str(Path(__file__).resolve(strict=True))
    if sum(record.get("path") == own for record in rendered) != 1:
        raise TableDFormalAggregationError(
            "aggregation closure does not uniquely bind this source"
        )
    return rendered


def _expected_source(
    *,
    queue: Mapping[str, Any],
    planned: Mapping[str, Any],
) -> formal_eval.evaluator.EvaluationSource:
    try:
        source, _binding = formal_eval.resolve_formal_source(
            training_queue_dir=Path(queue["plan"]["training_queue"]["queue_dir"]),
            training_run_root=Path(planned["training_root"]),
            training_phase=planned["training_phase"],
        )
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        formal_eval.TableDFormalEvaluationError,
    ) as exc:
        raise TableDFormalAggregationError(
            f"{planned['job_id']} formal source replay failed: {exc}"
        ) from exc
    return source


def _load_run(
    *,
    queue: Mapping[str, Any],
    planned: Mapping[str, Any],
) -> matrix.LoadedRun:
    root = Path(planned["evaluation_root"]).resolve(strict=True)
    cache = formal_eval.evaluator.HashCache()
    launch_path = (root / "launch_manifest.json").resolve(strict=True)
    postflight_path = (root / "postflight.json").resolve(strict=True)
    launch = matrix._read_json(launch_path, label="Table-D matrix launch")
    postflight = matrix._read_json(postflight_path, label="Table-D matrix postflight")
    matrix._assert_validation_only_root(root, launch)
    evidence = queue_runner._verify_completed_item(
        queue, int(planned["index"])
    )
    if evidence.get("job_id") != planned["job_id"]:
        raise TableDFormalAggregationError("validation queue job evidence drifted")
    source = _expected_source(queue=queue, planned=planned)
    raw_source = launch.get("source")
    if not isinstance(raw_source, Mapping):
        raise TableDFormalAggregationError("matrix launch source is missing")
    try:
        observed_source = matrix._evaluation_source_from_launch(raw_source)
    except (OSError, ValueError, matrix.MatrixValidationError) as exc:
        raise TableDFormalAggregationError(f"cannot reconstruct source: {exc}") from exc
    if observed_source != source:
        raise TableDFormalAggregationError(
            f"{planned['job_id']} launch source differs from formal replay"
        )
    if not (
        launch.get("evaluation_id") == source.evaluation_id
        and raw_source.get("training_run_id") == planned["training_run_id"]
        and raw_source.get("training_seed") == planned["train_seed"]
        and raw_source.get("training_phase") == planned["training_phase"]
    ):
        raise TableDFormalAggregationError(
            f"{planned['job_id']} evaluation/source identity drifted"
        )
    matrix._verify_file_record(
        launch.get("postflight_artifact"),
        label="Table-D matrix postflight artifact",
        cache=cache,
        expected_path=postflight_path,
    )
    if launch.get("postflight") != postflight:
        raise TableDFormalAggregationError("matrix launch embeds another postflight")
    input_rehash = matrix._verify_input_rehash(
        launch, postflight, root=root, cache=cache
    )
    matrix._validate_postflight(launch, postflight, input_rehash)
    checkpoint = source.checkpoint.resolve(strict=True)
    checkpoint_sha = source.checkpoint_sha256
    checkpoint_contract = postflight.get("checkpoint")
    checkpoint_run_id = (
        str(checkpoint_contract.get("run_id") or "")
        if isinstance(checkpoint_contract, Mapping)
        else ""
    )
    if not (
        isinstance(checkpoint_contract, Mapping)
        and Path(str(checkpoint_contract.get("path", ""))).resolve(strict=True)
        == checkpoint
        and str(checkpoint_contract.get("sha256", "")).lower()
        == checkpoint_sha
        and checkpoint_run_id
        and cache.digest(checkpoint) == checkpoint_sha
    ):
        raise TableDFormalAggregationError("matrix checkpoint evidence drifted")

    artifacts = postflight.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise TableDFormalAggregationError("matrix postflight artifacts are missing")
    section = (root / "validation_calibration").resolve(strict=True)
    summary_path = matrix._verify_file_record(
        artifacts.get("summary"),
        label="Table-D matrix summary",
        cache=cache,
        expected_path=section / "summary.json",
    )
    summary = matrix._read_json(summary_path, label="Table-D matrix summary")
    if set(summary) != {"refcoco", "tn"}:
        raise TableDFormalAggregationError("matrix summary surface drifted")
    ref_rows = summary.get("refcoco")
    tn_rows = summary.get("tn")
    if not isinstance(ref_rows, list) or not isinstance(tn_rows, list):
        raise TableDFormalAggregationError("matrix summary sections are invalid")
    by_split: dict[str, Mapping[str, Any]] = {}
    for row in ref_rows:
        if not isinstance(row, Mapping):
            raise TableDFormalAggregationError("matrix Ref summary row is invalid")
        split = str(row.get("dataset") or "")
        if split in by_split:
            raise TableDFormalAggregationError(f"duplicate matrix Ref split: {split}")
        by_split[split] = row
    if tuple(by_split) != matrix.REF_VALIDATION_SPLITS:
        raise TableDFormalAggregationError("matrix Ref split order drifted")
    ref_artifacts = artifacts.get("ref_validation")
    if not isinstance(ref_artifacts, Mapping) or set(ref_artifacts) != set(
        matrix.REF_VALIDATION_SPLITS
    ):
        raise TableDFormalAggregationError("matrix Ref artifact set drifted")
    ref = {
        split: matrix._load_ref_surface(
            split=split,
            summary_row=by_split[split],
            artifact=ref_artifacts[split],
            summary_path=summary_path,
            section_dir=section,
            run_id=checkpoint_run_id,
            cache=cache,
        )
        for split in matrix.REF_VALIDATION_SPLITS
    }
    if len(tn_rows) != 1 or not isinstance(tn_rows[0], Mapping):
        raise TableDFormalAggregationError("matrix calibration row set drifted")
    calibration = matrix._load_calibration_surface(
        summary_row=tn_rows[0],
        artifact=artifacts["matrix_calibration"],
        section_dir=section,
        run_id=checkpoint_run_id,
        cache=cache,
    )
    expected_records = {
        Path(str(surface.records["path"])).resolve(strict=True)
        for surface in ref.values()
    } | {Path(str(calibration.records["path"])).resolve(strict=True)}
    observed_records = {
        path.resolve(strict=True) for path in root.rglob("*.records.jsonl")
    }
    if observed_records != expected_records:
        raise TableDFormalAggregationError("matrix per-example artifact set drifted")
    code_fingerprint = matrix._input_role_fingerprint(
        launch, "evaluation_code_dependency"
    )
    data_fingerprint = matrix._input_role_fingerprint(
        launch, "evaluation_data_input"
    )
    runtime_fingerprint = dict(launch["runtime"])
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
    return matrix.LoadedRun(
        experiment_id=(
            planned["row_id"]
            if planned["training_phase"] == "final"
            else "S3_rank"
        ),
        seed=int(planned["train_seed"]),
        root=root,
        evaluation_id=source.evaluation_id,
        training_run_id=str(source.training_run_id),
        training_run_root=source.training_run_root,
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
            "formal_evaluation": evidence["formal_evaluation"],
        },
        sealed_files=matrix._snapshot_files(sealed_paths, cache),
    )


def _load_all(
    queue: Mapping[str, Any], spec: Mapping[str, Any]
) -> tuple[
    dict[str, dict[int, matrix.LoadedRun]],
    dict[int, matrix.LoadedRun],
]:
    if not (
        spec.get("schema") == queue_runner.SPEC_SCHEMA
        and spec.get("profile") == queue_runner.PROFILE
        and spec.get("ordered_job_ids") == list(queue_runner.JOB_IDS)
        and spec.get("expected_train_seeds")
        == list(queue_runner.training_queue.SEEDS)
    ):
        raise TableDFormalAggregationError("aggregation input identity drifted")
    finals: dict[str, dict[int, matrix.LoadedRun]] = {
        row_id: {} for row_id in queue_runner.training_queue.ROWS
    }
    ranks: dict[int, matrix.LoadedRun] = {}
    seen_roots: set[Path] = set()
    seen_source_phases: set[tuple[str, str]] = set()
    seen_checkpoints: set[Path] = set()
    seen_checkpoint_shas: set[str] = set()
    common_code: Mapping[str, Mapping[str, Any]] | None = None
    common_data: Mapping[str, Mapping[str, Any]] | None = None
    common_runtime: Mapping[str, Any] | None = None
    common_surface: Mapping[str, Any] | None = None
    ref_identities: dict[str, tuple[tuple[Any, ...], ...]] = {}
    calibration_identities: tuple[tuple[Any, ...], ...] | None = None
    for planned in queue["plan"]["items"]:
        root = Path(planned["evaluation_root"]).resolve(strict=True)
        if root in seen_roots:
            raise TableDFormalAggregationError(f"duplicate evaluation root: {root}")
        seen_roots.add(root)
        try:
            run = _load_run(queue=queue, planned=planned)
        except (
            FileNotFoundError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            matrix.MatrixValidationError,
            TableDFormalAggregationError,
        ) as exc:
            raise TableDFormalAggregationError(
                f"cannot load {planned['job_id']}: {exc}"
            ) from exc
        source_phase = (run.training_run_id, planned["training_phase"])
        if source_phase in seen_source_phases:
            raise TableDFormalAggregationError("duplicate training source/phase identity")
        if run.checkpoint in seen_checkpoints or run.checkpoint_sha256 in seen_checkpoint_shas:
            raise TableDFormalAggregationError("matrix reused a checkpoint across jobs")
        seen_source_phases.add(source_phase)
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
            raise TableDFormalAggregationError(
                "Table-D evaluations differ in code, data, runtime, or surface"
            )
        for split in matrix.REF_VALIDATION_SPLITS:
            identities = run.ref[split].identities
            previous = ref_identities.setdefault(split, identities)
            if identities != previous:
                raise TableDFormalAggregationError(
                    f"Table-D {split} record identities are not aligned"
                )
        identities = run.calibration.identities
        if calibration_identities is None:
            calibration_identities = identities
        elif identities != calibration_identities:
            raise TableDFormalAggregationError(
                "Table-D calibration record identities are not aligned"
            )
        seed = int(planned["train_seed"])
        if planned["training_phase"] == "rank":
            ranks[seed] = run
        else:
            finals[str(planned["row_id"])][seed] = run
    expected_seeds = set(queue_runner.training_queue.SEEDS)
    if any(set(values) != expected_seeds for values in finals.values()) or set(
        ranks
    ) != expected_seeds:
        raise TableDFormalAggregationError("Table-D aggregate seed inventory drifted")
    return finals, ranks


def _bind_final_diagnostics(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "status": "not_bound",
            "separate_from_matrix_metrics": True,
            "pooled_into_matrix_results": False,
        }
    path = path.expanduser().resolve(strict=True)
    payload = queue_runner._read_json(path, label="Table-D final diagnostics")
    if not (
        payload.get("schema") == diagnostics.REPORT_SCHEMA
        and payload.get("status") == "passed"
        and payload.get("expected_train_seeds") == list(diagnostics.EXPECTED_SEEDS)
    ):
        raise TableDFormalAggregationError("final diagnostics report identity drifted")
    input_manifest = payload.get("input_manifest")
    if not isinstance(input_manifest, Mapping):
        raise TableDFormalAggregationError("final diagnostics input binding is missing")
    manifest_path = Path(str(input_manifest.get("path", ""))).resolve(strict=True)
    try:
        replay = diagnostics.aggregate(manifest_path)
    except (OSError, ValueError, diagnostics.TableDDiagnosticsError) as exc:
        raise TableDFormalAggregationError(
            f"final diagnostics replay failed: {exc}"
        ) from exc
    observed = dict(payload)
    expected = dict(replay)
    observed.pop("created_at_utc", None)
    expected.pop("created_at_utc", None)
    if observed != expected:
        raise TableDFormalAggregationError("final diagnostics report replay drifted")
    return {
        "status": "bound_and_replayed",
        "report": queue_runner._file_record(path),
        "schema": diagnostics.REPORT_SCHEMA,
        "separate_from_matrix_metrics": True,
        "pooled_into_matrix_results": False,
    }


def aggregate(
    queue_dir: Path,
    *,
    bootstrap_iterations: int = FORMAL_BOOTSTRAP_ITERATIONS,
    confidence: float = FORMAL_BOOTSTRAP_CONFIDENCE,
    bootstrap_seed: int = FORMAL_BOOTSTRAP_SEED,
    final_diagnostics_report: Path | None = None,
) -> dict[str, Any]:
    if not (
        type(bootstrap_iterations) is int
        and bootstrap_iterations == FORMAL_BOOTSTRAP_ITERATIONS
        and confidence == FORMAL_BOOTSTRAP_CONFIDENCE
        and type(bootstrap_seed) is int
        and bootstrap_seed == FORMAL_BOOTSTRAP_SEED
    ):
        raise TableDFormalAggregationError(
            "formal Table-D bootstrap must be exactly 5000/0.95/20260719"
        )
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    queue = queue_runner.load_queue(queue_dir)
    verification = queue_runner.verify_queue(queue_dir)
    if queue.get("status") != "completed" or verification.get("status") != "passed":
        raise TableDFormalAggregationError(
            "Table-D validation queue is not completed and verified"
        )
    plan = queue["plan"]
    sources = _verify_aggregation_sources(plan)
    spec_path = (queue_dir / queue_runner.AGGREGATION_SPEC_NAME).resolve(strict=True)
    spec = queue_runner._read_json(spec_path, label="Table-D aggregation input")
    if spec != queue_runner._aggregation_spec_payload(
        plan, str(queue["plan_sha256"])
    ):
        raise TableDFormalAggregationError(
            "Table-D aggregation input differs from the immutable queue"
        )
    finals, ranks = _load_all(queue, spec)
    try:
        experiments = {
            row_id: {
                "id": row_id,
                "comparison_class": (
                    "clean_score_ownership"
                    if row_id in {"S0", "S1", "S2", "S3"}
                    else "full_v19_objective_control"
                ),
                **matrix._aggregate_experiment(finals[row_id]),
            }
            for row_id in queue_runner.training_queue.ROWS
        }
        rank_experiment = {
            "id": "S3_rank",
            "comparison_class": "diagnostic_rank_checkpoint",
            **matrix._aggregate_experiment(ranks),
        }
        ownership = {
            candidate: matrix._comparison(
                reference_id="S0",
                candidate_id=candidate,
                reference_runs=finals["S0"],
                candidate_runs=finals[candidate],
                iterations=bootstrap_iterations,
                confidence=confidence,
                bootstrap_seed=bootstrap_seed,
            )
            for candidate in ("S1", "S2", "S3")
        }
        full_objective = matrix._comparison(
            reference_id="S2",
            candidate_id="S2F",
            reference_runs=finals["S2"],
            candidate_runs=finals["S2F"],
            iterations=bootstrap_iterations,
            confidence=confidence,
            bootstrap_seed=bootstrap_seed,
        )
        s3_schedule = matrix._comparison(
            reference_id="S3_rank",
            candidate_id="S3",
            reference_runs=ranks,
            candidate_runs=finals["S3"],
            iterations=bootstrap_iterations,
            confidence=confidence,
            bootstrap_seed=bootstrap_seed,
        )
    except (ValueError, matrix.MatrixValidationError) as exc:
        raise TableDFormalAggregationError(f"paired aggregation failed: {exc}") from exc
    checkpoint_shas = {
        **{
            row_id: {
                str(seed): finals[row_id][seed].checkpoint_sha256
                for seed in queue_runner.training_queue.SEEDS
            }
            for row_id in queue_runner.training_queue.ROWS
        },
        "S3_rank": {
            str(seed): ranks[seed].checkpoint_sha256
            for seed in queue_runner.training_queue.SEEDS
        },
    }
    return {
        "schema": REPORT_SCHEMA,
        "status": "validated_matrix_validation_only",
        "created_at_utc": _utc_now(),
        "formal_test_or_strict_result": False,
        "protocol": {
            "profile": queue_runner.PROFILE,
            "ordered_job_ids": list(queue_runner.JOB_IDS),
            "train_seeds": list(queue_runner.training_queue.SEEDS),
            "clean_ownership_rows": ["S0", "S1", "S2", "S3"],
            "full_objective_control": "S2F",
            "s3_rank_is_diagnostic_only": True,
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
            "exact_fifteen_final_jobs": True,
            "exact_three_s3_rank_jobs": True,
            "record_identities_aligned": True,
            "runtime_code_data_surface_equal": True,
            "input_rehash_and_postflight_replayed": True,
            "training_authority_replayed": True,
        },
        "inputs": {
            "aggregation_spec": queue_runner._file_record(spec_path),
            "evaluation_queue": {
                "queue_id": plan["queue_id"],
                "plan_sha256": queue["plan_sha256"],
                "verification_schema": verification["schema"],
                "final_verification": verification["final_verification"],
                "evaluation_scope_plan": verification[
                    "evaluation_scope_plan"
                ],
            },
            "training_queue": dict(plan["training_queue"]),
            "aggregation_source_closure": sources,
            "checkpoint_sha256s": checkpoint_shas,
            "final_diagnostics": _bind_final_diagnostics(
                final_diagnostics_report
            ),
        },
        "experiments": {**experiments, "S3_rank": rank_experiment},
        "comparisons": {
            "clean_ownership_vs_S0": ownership,
            "S2F_minus_S2_full_objective_control": full_objective,
            "S3_confidence_minus_rank_diagnostic": s3_schedule,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--final-diagnostics-report", type=Path)
    parser.add_argument(
        "--bootstrap-iterations", type=int, default=FORMAL_BOOTSTRAP_ITERATIONS
    )
    parser.add_argument("--confidence", type=float, default=FORMAL_BOOTSTRAP_CONFIDENCE)
    parser.add_argument("--bootstrap-seed", type=int, default=FORMAL_BOOTSTRAP_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = aggregate(
            args.queue_dir,
            bootstrap_iterations=args.bootstrap_iterations,
            confidence=args.confidence,
            bootstrap_seed=args.bootstrap_seed,
            final_diagnostics_report=args.final_diagnostics_report,
        )
        if args.output is not None:
            output = args.output.expanduser().resolve(strict=False)
            queue = queue_runner.load_queue(args.queue_dir)
            queue_root = args.queue_dir.expanduser().resolve(strict=True)
            evaluation_root = Path(queue["plan"]["output_root"]).resolve(
                strict=False
            )
            if (
                output == evaluation_root
                or evaluation_root in output.parents
                or output == queue_root
                or queue_root in output.parents
            ):
                raise TableDFormalAggregationError(
                    "aggregate output cannot be inside queue or evaluation evidence"
                )
            _write_json_no_replace(output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        matrix.MatrixValidationError,
        TableDFormalAggregationError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
