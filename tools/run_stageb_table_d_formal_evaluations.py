#!/usr/bin/env python3
"""Run matrix-validation evaluations from strictly replayed formal Table-D runs."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_stageb_paper_evaluations as evaluator  # noqa: E402
from tools import run_stageb_table_d_formal_queue as training_queue  # noqa: E402


PROFILE = evaluator.MATRIX_PROFILE
EVAL_BATCH_SIZE = 16
EVAL_NUM_WORKERS = 4
EVAL_LOG_EVERY = 50


class TableDFormalEvaluationError(RuntimeError):
    """The formal training source or matrix evaluation contract drifted."""


def _formal_binding(evidence: Mapping[str, Any]) -> dict[str, Any]:
    run = evidence.get("run_verification")
    if not isinstance(run, Mapping):
        raise TableDFormalEvaluationError("formal run verification is missing")
    return {
        "profile": evidence["profile"],
        "run_id": evidence["run_id"],
        "training_phase": evidence["training_phase"],
        "queue_id": evidence["queue_id"],
        "queue_plan_sha256": evidence["queue_plan_sha256"],
        "completion_semantic_sha256": evidence["completion_semantic_sha256"],
        "scope_sha256": run["scope_sha256"],
        "source_plan": evidence["source_plan"],
        "scope_plan": evidence["scope_plan"],
        "completion_attestation": evidence["completion_attestation"],
    }


def _rendered_source(source: evaluator.EvaluationSource) -> dict[str, Any]:
    raw = asdict(source)
    for key in (
        "config",
        "checkpoint",
        "training_run_root",
        "sequence_manifest",
        "final_phase_manifest",
        "training_postflight",
        "selected_phase_manifest",
        "selected_training_postflight",
        "training_queue_manifest",
        "training_queue_detached_launch",
        "training_queue_detached_status",
    ):
        value = raw.get(key)
        raw[key] = str(value) if value is not None else None
    raw["training_data"] = [str(path) for path in source.training_data]
    return raw


def resolve_formal_source(
    *,
    training_queue_dir: Path,
    training_run_root: Path,
    training_phase: str,
) -> tuple[evaluator.EvaluationSource, dict[str, Any]]:
    training_queue_dir = training_queue_dir.expanduser().resolve(strict=True)
    training_run_root = training_run_root.expanduser().resolve(strict=True)
    sequence = training_queue._read_json(
        training_run_root / "sequence_manifest.json", label="formal training sequence"
    )
    run_id = sequence.get("run_id")
    if not isinstance(run_id, str) or run_id not in training_queue.RUN_IDS:
        raise TableDFormalEvaluationError("training root is not a formal Table-D run")
    try:
        evidence = training_queue.formal_evaluation_evidence(
            training_queue_dir,
            run_id=run_id,
            run_root=training_run_root,
            training_phase=training_phase,
        )
        source = evaluator._resolve_paper_source(
            training_run_root,
            evaluator.HashCache(),
            training_phase=training_phase,
            training_queue_dir=training_queue_dir,
            allow_nonformal_fixture=True,
        )
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        evaluator.PaperEvaluationError,
        training_queue.TableDFormalQueueError,
    ) as exc:
        raise TableDFormalEvaluationError(
            f"formal Table-D source replay failed: {exc}"
        ) from exc
    if not (
        source.training_run_id == run_id
        and source.training_seed == int(run_id.split(":", 1)[1])
        and source.training_phase == training_phase
        and (training_phase == "rank")
        == (source.kind == "pivot_paper_training_run_rank_diagnostic")
        and (training_phase == "rank") == source.diagnostic_only
    ):
        raise TableDFormalEvaluationError("formal Table-D source identity drifted")
    authority_paths = tuple(
        Path(record["path"]).resolve(strict=True)
        for record in (
            evidence["source_plan"],
            evidence["scope_plan"],
            evidence["completion_attestation"],
        )
    )
    source = replace(
        source,
        training_data=tuple(dict.fromkeys((*source.training_data, *authority_paths))),
    )
    return source, _formal_binding(evidence)


@contextlib.contextmanager
def _scoped_formal_resolver(
    source: evaluator.EvaluationSource,
) -> Iterator[None]:
    """Install one process-local revalidation result while building its plan."""

    original = evaluator._resolve_pivot_source

    def resolve(
        run_root: Path,
        cache: evaluator.HashCache,
        *,
        training_phase: str = "final",
        training_queue_dir: Path | None = None,
        allow_nonformal_fixture: bool = False,
    ) -> evaluator.EvaluationSource:
        del cache, allow_nonformal_fixture
        expected_queue = (
            source.training_queue_manifest.parent
            if source.training_queue_manifest is not None
            else None
        )
        if (
            source.training_run_root is not None
            and Path(run_root).resolve(strict=True)
            == source.training_run_root.resolve(strict=True)
            and training_phase == source.training_phase
            and training_queue_dir is not None
            and expected_queue is not None
            and Path(training_queue_dir).resolve(strict=True)
            == expected_queue.resolve(strict=True)
        ):
            return source
        return original(
            run_root,
            evaluator.HashCache(),
            training_phase=training_phase,
            training_queue_dir=training_queue_dir,
        )

    evaluator._resolve_pivot_source = resolve
    try:
        yield
    finally:
        evaluator._resolve_pivot_source = original


def _fixed_runtime(
    *,
    python: Path = evaluator.DEFAULT_PYTHON,
    data_root: Path = evaluator.DEFAULT_DATA_ROOT,
    device: str = "cuda:0",
) -> evaluator.Runtime:
    python = python.expanduser().resolve(strict=True)
    data_root = data_root.expanduser().resolve(strict=True)
    if python != evaluator.DEFAULT_PYTHON.resolve(strict=True):
        raise TableDFormalEvaluationError("matrix evaluation Python drifted")
    if data_root != evaluator.DEFAULT_DATA_ROOT.resolve(strict=True):
        raise TableDFormalEvaluationError("matrix evaluation data root drifted")
    if device != "cuda:0":
        raise TableDFormalEvaluationError("matrix evaluation device must be cuda:0")
    return evaluator.Runtime(
        python=python,
        data_root=data_root,
        device=device,
        batch_size=EVAL_BATCH_SIZE,
        num_workers=EVAL_NUM_WORKERS,
        amp=True,
        log_every=EVAL_LOG_EVERY,
    )


def _merge_wrapper_source(plan: dict[str, Any]) -> None:
    records = plan.get("inputs", {}).get("records")
    if not isinstance(records, list):
        raise TableDFormalEvaluationError("evaluation plan inputs are missing")
    entries = [
        (Path(record["path"]), role)
        for record in records
        for role in record.get("roles", [])
    ]
    entries.extend(
        (
            (Path(__file__).resolve(strict=True), "evaluation_code_dependency"),
            (
                Path(training_queue.__file__).resolve(strict=True),
                "source_provenance_dependency",
            ),
        )
    )
    plan["inputs"]["records"] = evaluator._merge_input_records(
        entries, evaluator.HashCache()
    )


def _expected_inputs(
    *,
    runtime: evaluator.Runtime,
    source: evaluator.EvaluationSource,
    matrix_queue_spec: Path,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    cache = evaluator.HashCache()
    calibration = evaluator._screen_calibration_contract(cache)
    entries: list[tuple[Path, str]] = [
        (source.config, "evaluation_config"),
        (source.checkpoint, "evaluation_checkpoint"),
        (matrix_queue_spec, "matrix_validation_queue_spec"),
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
    entries.extend(
        (
            (
                Path(calibration["source_manifest"]["path"]),
                "matrix_calibration_source",
            ),
            (
                Path(calibration["source_audit"]["path"]),
                "matrix_calibration_audit",
            ),
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
    entries.extend(
        (
            (Path(__file__).resolve(strict=True), "evaluation_code_dependency"),
            (
                Path(training_queue.__file__).resolve(strict=True),
                "source_provenance_dependency",
            ),
        )
    )
    return (
        {
            "algorithm": "sha256",
            "records": evaluator._merge_input_records(entries, cache),
        },
        calibration,
    )


def build_formal_plan(
    *,
    training_queue_dir: Path,
    training_run_root: Path,
    training_phase: str,
    output_dir: Path,
    matrix_queue_spec: Path,
    python: Path = evaluator.DEFAULT_PYTHON,
    data_root: Path = evaluator.DEFAULT_DATA_ROOT,
    device: str = "cuda:0",
) -> tuple[dict[str, Any], evaluator.Runtime]:
    source, binding = resolve_formal_source(
        training_queue_dir=training_queue_dir,
        training_run_root=training_run_root,
        training_phase=training_phase,
    )
    runtime = _fixed_runtime(python=python, data_root=data_root, device=device)
    matrix_queue_spec = matrix_queue_spec.expanduser().resolve(strict=True)
    try:
        with _scoped_formal_resolver(source):
            plan = evaluator.build_plan(
                runtime,
                source,
                output_dir,
                evaluator.HashCache(),
                profile=PROFILE,
                matrix_queue_spec=matrix_queue_spec,
            )
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        ValueError,
        evaluator.PaperEvaluationError,
    ) as exc:
        raise TableDFormalEvaluationError(
            f"formal matrix plan construction failed: {exc}"
        ) from exc
    _merge_wrapper_source(plan)
    plan["table_d_formal"] = binding
    return plan, runtime


def _runtime_from_launch(raw: Any) -> evaluator.Runtime:
    expected = {
        "python": str(evaluator.DEFAULT_PYTHON.resolve(strict=True)),
        "data_root": str(evaluator.DEFAULT_DATA_ROOT.resolve(strict=True)),
        "device": "cuda:0",
        "batch_size": EVAL_BATCH_SIZE,
        "num_workers": EVAL_NUM_WORKERS,
        "amp": True,
        "log_every": EVAL_LOG_EVERY,
        "eval_seed": evaluator.EVAL_SEED,
        "max_ref_batches": 0,
        "max_tn_batches": 0,
    }
    if not isinstance(raw, Mapping) or dict(raw) != expected:
        raise TableDFormalEvaluationError("matrix evaluation runtime drifted")
    return _fixed_runtime()


def replay_completed_evaluation(
    *,
    training_queue_dir: Path,
    training_run_root: Path,
    training_phase: str,
    evaluation_root: Path,
    matrix_queue_spec: Path,
) -> dict[str, Any]:
    evaluation_root = evaluation_root.expanduser().resolve(strict=True)
    matrix_queue_spec = matrix_queue_spec.expanduser().resolve(strict=True)
    launch_path = (evaluation_root / "launch_manifest.json").resolve(strict=True)
    rehash_path = (evaluation_root / "input_rehash.json").resolve(strict=True)
    postflight_path = (evaluation_root / "postflight.json").resolve(strict=True)
    launch = training_queue._read_json(launch_path, label="formal matrix launch")
    postflight = training_queue._read_json(
        postflight_path, label="formal matrix postflight"
    )
    source, binding = resolve_formal_source(
        training_queue_dir=training_queue_dir,
        training_run_root=training_run_root,
        training_phase=training_phase,
    )
    runtime = _runtime_from_launch(launch.get("runtime"))
    expected_inputs, calibration = _expected_inputs(
        runtime=runtime,
        source=source,
        matrix_queue_spec=matrix_queue_spec,
    )
    if not (
        launch.get("schema") == evaluator.SCHEMA
        and launch.get("status") == "completed"
        and Path(str(launch.get("output_dir", ""))).resolve(strict=True)
        == evaluation_root
        and launch.get("source") == _rendered_source(source)
        and launch.get("table_d_formal") == binding
        and launch.get("commands")
        == evaluator._commands(runtime, source, evaluation_root, profile=PROFILE)
        and launch.get("inputs") == expected_inputs
    ):
        raise TableDFormalEvaluationError("completed matrix launch identity drifted")
    protocol = launch.get("protocol")
    if not (
        isinstance(protocol, Mapping)
        and protocol.get("profile") == PROFILE
        and protocol.get("processes") == ["validation_calibration"]
        and protocol.get("strict_manifests") == {}
        and protocol.get("strict1607_skip_ref") is False
        and protocol.get("screen_calibration") == calibration
    ):
        raise TableDFormalEvaluationError("matrix validation protocol drifted")
    expected_spec = evaluator._file_record(
        matrix_queue_spec,
        evaluator.HashCache(),
        roles=("matrix_validation_queue_spec",),
    )
    if launch.get("matrix_validation_queue_spec") != expected_spec:
        raise TableDFormalEvaluationError("matrix queue specification binding drifted")
    persisted_rehash = training_queue._read_json(
        rehash_path, label="formal matrix input rehash"
    )
    replay = evaluator._rehash_inputs(launch)
    for key in ("schema", "status", "records"):
        if replay.get(key) != persisted_rehash.get(key):
            raise TableDFormalEvaluationError("matrix input rehash replay drifted")
    if not (
        launch.get("postflight") == postflight
        and postflight.get("schema") == evaluator.POSTFLIGHT_SCHEMA
        and postflight.get("status") == "passed"
        and postflight.get("profile") == PROFILE
        and postflight.get("input_rehash") == persisted_rehash
    ):
        raise TableDFormalEvaluationError("matrix postflight identity drifted")
    replayed_postflight = evaluator._postflight_screen(launch, persisted_rehash)
    observed = dict(postflight)
    expected = dict(replayed_postflight)
    observed.pop("validated_at_utc", None)
    expected.pop("validated_at_utc", None)
    if observed != expected:
        raise TableDFormalEvaluationError("matrix postflight replay drifted")
    return {
        "status": "passed",
        "run_id": binding["run_id"],
        "training_phase": training_phase,
        "evaluation_id": launch["evaluation_id"],
        "evaluation_root": str(evaluation_root),
        "launch_manifest": training_queue._file_record(
            launch_path, roles=("matrix_evaluation_launch",)
        ),
        "input_rehash": training_queue._file_record(
            rehash_path, roles=("matrix_evaluation_input_rehash",)
        ),
        "postflight": training_queue._file_record(
            postflight_path, roles=("matrix_evaluation_postflight",)
        ),
        "formal_binding": binding,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("dry-run", "run"):
        child = subparsers.add_parser(mode)
        child.add_argument("--training-queue-dir", type=Path, required=True)
        child.add_argument("--training-run-root", type=Path, required=True)
        child.add_argument(
            "--training-phase", choices=("final", "rank"), default="final"
        )
        child.add_argument("--matrix-queue-spec", type=Path, required=True)
        child.add_argument("--output-dir", type=Path, required=True)
        child.add_argument("--python", type=Path, default=evaluator.DEFAULT_PYTHON)
        child.add_argument("--data-root", type=Path, default=evaluator.DEFAULT_DATA_ROOT)
        child.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan, runtime = build_formal_plan(
            training_queue_dir=args.training_queue_dir,
            training_run_root=args.training_run_root,
            training_phase=args.training_phase,
            output_dir=args.output_dir,
            matrix_queue_spec=args.matrix_queue_spec,
            python=args.python,
            data_root=args.data_root,
            device=args.device,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        TableDFormalEvaluationError,
    ) as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.mode == "dry-run":
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    return evaluator._execute(plan, runtime)


if __name__ == "__main__":
    raise SystemExit(main())
