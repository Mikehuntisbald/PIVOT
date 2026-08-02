#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Fail-closed formal evaluation for the dense-duty Stage-B checkpoint.

This controller has one model source and one evaluation protocol.  It accepts
no checkpoint, config, runtime, seed, split, or sample-limit overrides.  A run
is admitted only after replaying the terminal dense-duty checkpoint audit
against the current formal config and source closure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FIXED_PYTHON = Path("/home/haoyi/miniconda/envs/gdino5090/bin/python")
if (
    __name__ == "__main__"
    and Path(sys.executable).resolve() != FIXED_PYTHON.resolve()
):
    if not FIXED_PYTHON.is_file() or not os.access(FIXED_PYTHON, os.X_OK):
        print(
            f"[FAIL] fixed evaluation interpreter is unavailable: {FIXED_PYTHON}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    os.execv(
        str(FIXED_PYTHON),
        [str(FIXED_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
    )

from tools import run_stageb_dense_duty_formal as training  # noqa: E402
from tools import run_stageb_paper_evaluations as paper  # noqa: E402
from tools import stageb_headline_release_contract as headline  # noqa: E402
from tools.compare_stageb_fpr95_records import (  # noqa: E402
    RecordComparisonError,
    compare_record_files,
    render_markdown,
)
from util.slconfig import SLConfig  # noqa: E402
from util.stage_b_dense_duty_audit import (  # noqa: E402
    build_code_source_closure,
    validate_evaluation_checkpoint_payload,
)


SCHEMA = "pivot.stageb.dense_duty_formal_evaluation_launch/v1"
POSTFLIGHT_SCHEMA = "pivot.stageb.dense_duty_formal_evaluation_postflight/v1"
COMPARISON_SCHEMA = "pivot.stageb.dense_duty_formal_comparison/v1"
EVALUATION_ID = "dense_duty_20260728_formal"

CONFIG = REPO_ROOT / "config/ablations/cfg_stageb_dense_duty_confidence_20260728.py"
CHECKPOINT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/dense_duty_20260728/formal/confidence/checkpoint_iter.pth"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs/paper_cvpr_v1/dense_duty_20260728/formal_evaluation"
)
BASELINE_MANIFEST = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/manifests/baseline_b58_seed42_results_manifest.json"
)
BASELINE_MANIFEST_SHA256 = (
    "70b0752359419615793ec7e9d8009065cb3f66ddcfd081d2ac30c0fff6134b4b"
)
BASELINE_CHECKPOINT_SHA256 = (
    "b58e5209dc07dbffb2e5ed3d792e0db70c3306ec2ed24389693b4aeeebab1157"
)
BOOTSTRAP_ITERATIONS = 5000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_SEED = 20260717
REF_SCORE_KEY = "stage_b_v15_dense_rank_score"
TN_SCORE_KEY = "stage_b_v7_final_score"
SCORE_OWNERSHIP = "independent_decoders_two_phase"
BASELINE_REF_CORRECT = {
    "refcoco_val": 6993,
    "refcoco_testA": 4102,
    "refcoco_testB": 2957,
    "refcocop_val": 7475,
    "refcocop_testA": 4305,
    "refcocop_testB": 3108,
    "refcocog_val": 3942,
    "refcocog_test": 7876,
}
BASELINE_TN_FALSE_ACCEPTS = {
    "strict2031": 1040,
    "strict1607": 801,
}


class DenseDutyFormalEvaluationError(RuntimeError):
    """The fixed formal evaluation contract cannot be proven."""


def _exact_binary_count(rate: Any, total: int, *, label: str) -> int:
    try:
        value = float(rate)
    except (TypeError, ValueError) as exc:
        raise DenseDutyFormalEvaluationError(
            f"{label} is not a finite binary rate"
        ) from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0 or total <= 0:
        raise DenseDutyFormalEvaluationError(
            f"{label} is not a finite binary rate"
        )
    count = int(round(value * total))
    if not math.isclose(value, count / total, rel_tol=0.0, abs_tol=1e-12):
        raise DenseDutyFormalEvaluationError(
            f"{label} does not replay to an exact integer count"
        )
    return count


def _direct_controller_dependencies() -> tuple[Path, ...]:
    return (
        Path(training.__file__).resolve(strict=True),
        Path(headline.__file__).resolve(strict=True),
    )


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    import torch

    kwargs = {"map_location": "cpu", "weights_only": False}
    try:
        payload = torch.load(path, mmap=True, **kwargs)
    except TypeError:
        payload = torch.load(path, **kwargs)
    if not isinstance(payload, Mapping):
        raise DenseDutyFormalEvaluationError("checkpoint payload is not a mapping")
    return payload


def _fixed_runtime() -> paper.Runtime:
    python = paper.DEFAULT_PYTHON.expanduser().resolve(strict=True)
    data_root = paper.DEFAULT_DATA_ROOT.expanduser().resolve(strict=True)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise DenseDutyFormalEvaluationError(
            f"fixed evaluation Python is unavailable: {python}"
        )
    if not data_root.is_dir():
        raise DenseDutyFormalEvaluationError(
            f"fixed evaluation data root is unavailable: {data_root}"
        )
    return paper.Runtime(
        python=python,
        data_root=data_root,
        device="cuda:0",
        batch_size=16,
        num_workers=4,
        amp=True,
        log_every=50,
    )


def _resolve_formal_source(
    cache: paper.HashCache,
    *,
    checkpoint_loader=_load_checkpoint,
) -> tuple[paper.EvaluationSource, dict[str, Any]]:
    checkpoint = CHECKPOINT.resolve(strict=True)
    config = CONFIG.resolve(strict=True)
    confidence_spec = training.formal_phase_specs()[1]
    if checkpoint != confidence_spec.checkpoint.resolve(strict=False):
        raise DenseDutyFormalEvaluationError(
            "dense-duty evaluation checkpoint is not the canonical confidence artifact"
        )
    checkpoint_sha_before = cache.digest(checkpoint)
    payload = checkpoint_loader(checkpoint)
    inspection = training.classify_checkpoint_payload(
        payload, confidence_spec, checkpoint_path=checkpoint
    )
    if inspection.status is not training.PhaseStatus.TERMINAL:
        raise DenseDutyFormalEvaluationError(
            "dense-duty confidence checkpoint is not terminal: " + inspection.detail
        )
    try:
        training._audit_checkpoint(payload, confidence_spec, checkpoint)
        cfg = SLConfig.fromfile(str(config))
        audit = validate_evaluation_checkpoint_payload(
            payload,
            cfg,
            checkpoint_path=checkpoint,
            current_code_source_closure=build_code_source_closure(),
        )
    except (RuntimeError, ValueError, KeyError, TypeError) as exc:
        raise DenseDutyFormalEvaluationError(
            f"dense-duty terminal checkpoint audit failed: {exc}"
        ) from exc
    rank_handoff = audit.get("rank_handoff")
    if (
        audit.get("status") != "passed"
        or audit.get("phase") != "confidence"
        or audit.get("optimizer_updates") != training.CONFIDENCE_UPDATES
        or audit.get("evaluation_scope") != "formal"
        or not isinstance(rank_handoff, Mapping)
        or rank_handoff.get("status") != "passed"
        or rank_handoff.get("phase") != "rank"
        or rank_handoff.get("optimizer_updates") != training.RANK_UPDATES
    ):
        raise DenseDutyFormalEvaluationError(
            "dense-duty audit lacks the exact confidence-4412/rank-10295 lineage"
        )
    saved_args = payload.get("args")
    exact_contract = {
        "batch_size": 16,
        "gradient_accumulation_steps": 4,
        "stage_b_v11_expression_microbatch": 16,
        "stage_b_dense_duty_no_stageb_teacher": True,
        "stage_b_dense_duty_execution_scope": "formal",
        "stage_b_dense_duty_phase": "confidence",
        "stage_b_v22_train_phase": "confidence",
        "stage_b_v22_score_ownership": "independent_decoders_two_phase",
    }
    if not isinstance(saved_args, Mapping) or any(
        saved_args.get(key) != value for key, value in exact_contract.items()
    ):
        raise DenseDutyFormalEvaluationError(
            "dense-duty checkpoint violates the fixed B16/acc4/E16/no-teacher contract"
        )
    checkpoint_sha_after = paper.HashCache().digest(checkpoint)
    if checkpoint_sha_after != checkpoint_sha_before:
        raise DenseDutyFormalEvaluationError(
            "dense-duty checkpoint changed while its terminal audit was running"
        )
    source = paper.EvaluationSource(
        kind="dense_duty_formal_terminal_confidence",
        evaluation_id=EVALUATION_ID,
        config=config,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha_after,
        training_run_id="dense_duty_20260728:42",
        training_seed=42,
        training_run_root=training.FORMAL_ROOT.resolve(strict=False),
        training_phase="confidence",
        diagnostic_only=False,
        final_phase_id="confidence",
        selected_phase_id="confidence",
        artifact_repository_root=paper.ARTIFACT_REPOSITORY_ROOT,
    )
    return source, audit


def _declared_path(record: Mapping[str, Any], *, label: str) -> Path:
    raw = record.get("path")
    if not isinstance(raw, str) or not raw:
        raise DenseDutyFormalEvaluationError(f"baseline {label} has no path")
    path = Path(raw).expanduser().resolve(strict=True)
    if not path.is_file():
        raise DenseDutyFormalEvaluationError(f"baseline {label} is not a file")
    try:
        expected_size = int(record.get("size_bytes"))
    except (TypeError, ValueError) as exc:
        raise DenseDutyFormalEvaluationError(
            f"baseline {label} has invalid size"
        ) from exc
    if path.stat().st_size != expected_size:
        raise DenseDutyFormalEvaluationError(f"baseline {label} size changed")
    declared_sha = str(record.get("sha256", "")).lower()
    if len(declared_sha) != 64:
        raise DenseDutyFormalEvaluationError(f"baseline {label} SHA-256 is invalid")
    return path


def _baseline_contract(cache: paper.HashCache) -> dict[str, Any]:
    manifest_path = BASELINE_MANIFEST.resolve(strict=True)
    if cache.digest(manifest_path) != BASELINE_MANIFEST_SHA256:
        raise DenseDutyFormalEvaluationError("canonical baseline manifest changed")
    manifest = paper._read_json(manifest_path, label="canonical baseline manifest")
    if (
        manifest.get("schema") != "stageb-paper-results-manifest-v1"
        or manifest.get("baseline_experiment") != headline.BASELINE_ID
        or manifest.get("expected_train_seeds") != [42]
    ):
        raise DenseDutyFormalEvaluationError("canonical baseline manifest identity drifted")
    experiments = manifest.get("experiments")
    if not isinstance(experiments, list) or len(experiments) != 1:
        raise DenseDutyFormalEvaluationError("canonical baseline experiment set drifted")
    experiment = experiments[0]
    if not isinstance(experiment, Mapping):
        raise DenseDutyFormalEvaluationError(
            "canonical baseline experiment record is invalid"
        )
    runs = experiment.get("runs")
    if (
        experiment.get("id") != headline.BASELINE_ID
        or not isinstance(runs, list)
        or len(runs) != 1
    ):
        raise DenseDutyFormalEvaluationError("canonical baseline run set drifted")
    run = runs[0]
    if not isinstance(run, Mapping) or run.get("train_seed") != 42:
        raise DenseDutyFormalEvaluationError("canonical baseline seed drifted")
    artifacts = run.get("artifacts")
    results = run.get("results")
    if not isinstance(artifacts, Mapping) or not isinstance(results, Mapping):
        raise DenseDutyFormalEvaluationError("canonical baseline artifacts are incomplete")
    checkpoint_record = artifacts.get("checkpoint")
    config_record = artifacts.get("config")
    if not isinstance(checkpoint_record, Mapping) or not isinstance(config_record, Mapping):
        raise DenseDutyFormalEvaluationError("canonical baseline identity is incomplete")
    checkpoint = _declared_path(checkpoint_record, label="checkpoint")
    config = _declared_path(config_record, label="config")
    if (
        checkpoint.resolve() != Path(str(headline.FIXED_BASELINE["checkpoint"])).resolve()
        or str(checkpoint_record.get("sha256", "")).lower()
        != BASELINE_CHECKPOINT_SHA256
        or cache.digest(checkpoint) != BASELINE_CHECKPOINT_SHA256
        or config.resolve() != Path(str(headline.FIXED_BASELINE["config"])).resolve()
        or cache.digest(config) != str(config_record.get("sha256", "")).lower()
    ):
        raise DenseDutyFormalEvaluationError("canonical b58 checkpoint/config identity changed")
    declared_data = artifacts.get("data")
    if not isinstance(declared_data, list):
        raise DenseDutyFormalEvaluationError("canonical baseline data inventory is missing")
    data_paths = []
    for index, record in enumerate(declared_data):
        if not isinstance(record, Mapping):
            raise DenseDutyFormalEvaluationError(
                f"canonical baseline data record {index} is invalid"
            )
        path = _declared_path(record, label=f"data record {index}")
        if cache.digest(path) != str(record.get("sha256", "")).lower():
            raise DenseDutyFormalEvaluationError(
                f"canonical baseline data record {index} changed"
            )
        data_paths.append(path)

    protocol = manifest.get("protocol")
    expected_bootstrap = {
        "confidence": BOOTSTRAP_CONFIDENCE,
        "iterations": BOOTSTRAP_ITERATIONS,
        "seed": BOOTSTRAP_SEED,
    }
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("bootstrap") != expected_bootstrap
        or protocol.get("ref_splits") != list(paper.REF_SPLITS)
    ):
        raise DenseDutyFormalEvaluationError("canonical baseline protocol drifted")
    tn_protocol = protocol.get("tn_splits")
    if not isinstance(tn_protocol, Mapping):
        raise DenseDutyFormalEvaluationError("canonical baseline TN protocol is missing")
    for label, specification in paper.STRICT_SPECS.items():
        observed = tn_protocol.get(label)
        if (
            not isinstance(observed, Mapping)
            or observed.get("expected_n") != specification["rows"]
            or not isinstance(observed.get("manifest"), Mapping)
            or observed["manifest"].get("sha256") != specification["sha256"]
        ):
            raise DenseDutyFormalEvaluationError(
                f"canonical baseline {label} protocol drifted"
            )

    ref_result = results.get("ref")
    tn_result = results.get("tn")
    if not isinstance(ref_result, Mapping) or not isinstance(tn_result, Mapping):
        raise DenseDutyFormalEvaluationError("canonical baseline results are incomplete")
    ref_summary = _declared_path(ref_result.get("summary", {}), label="Ref8 summary")
    ref_records_raw = ref_result.get("records")
    if not isinstance(ref_records_raw, Mapping) or set(ref_records_raw) != set(paper.REF_SPLITS):
        raise DenseDutyFormalEvaluationError("canonical baseline Ref8 records drifted")
    ref_records = {
        split: _declared_path(ref_records_raw[split], label=f"{split} records")
        for split in paper.REF_SPLITS
    }
    tn: dict[str, dict[str, Path]] = {}
    for label in ("strict2031", "strict1607"):
        item = tn_result.get(label)
        if not isinstance(item, Mapping):
            raise DenseDutyFormalEvaluationError(f"canonical baseline {label} is missing")
        tn[label] = {
            "summary": _declared_path(item.get("summary", {}), label=f"{label} summary"),
            "records": _declared_path(item.get("records", {}), label=f"{label} records"),
        }
        if item.get("run_id") != ref_result.get("run_id"):
            raise DenseDutyFormalEvaluationError(
                f"canonical baseline {label} run identity drifted"
            )
        for role, path in tn[label].items():
            if cache.digest(path) != str(item[role].get("sha256", "")).lower():
                raise DenseDutyFormalEvaluationError(
                    f"canonical baseline {label} {role} changed"
                )
    for split, path in ref_records.items():
        if cache.digest(path) != str(ref_records_raw[split].get("sha256", "")).lower():
            raise DenseDutyFormalEvaluationError(
                f"canonical baseline {split} records changed"
            )
    if cache.digest(ref_summary) != str(ref_result["summary"].get("sha256", "")).lower():
        raise DenseDutyFormalEvaluationError("canonical baseline Ref8 summary changed")
    return {
        "manifest": manifest_path,
        "checkpoint": checkpoint,
        "config": config,
        "data": tuple(data_paths),
        "run_id": str(ref_result.get("run_id", "")),
        "ref_summary": ref_summary,
        "ref_records": ref_records,
        "tn": tn,
    }


def build_plan(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve(strict=False)
    if output_dir.exists():
        raise FileExistsError(f"evaluation output root must be fresh: {output_dir}")
    cache = paper.HashCache()
    runtime = _fixed_runtime()
    source, checkpoint_audit = _resolve_formal_source(cache)
    baseline = _baseline_contract(cache)
    strict_records = {
        label: paper._strict_manifest_record(label, cache)
        for label in paper.STRICT_SPECS
    }
    entries: list[tuple[Path, str]] = [
        (source.config, "evaluation_config"),
        (source.checkpoint, "evaluation_checkpoint"),
        (Path(__file__).resolve(), "evaluation_controller"),
        (baseline["manifest"], "baseline_results_manifest"),
        (baseline["checkpoint"], "baseline_identity_checkpoint"),
        (baseline["config"], "baseline_identity_config"),
        (baseline["ref_summary"], "baseline_ref8_summary"),
    ]
    entries.extend(
        (path, "baseline_declared_training_data") for path in baseline["data"]
    )
    entries.extend(
        (path, "config_dependency")
        for path in paper._config_paths(source.config)
    )
    entries.extend(
        (path, "evaluation_code_dependency")
        for path in paper._evaluation_code_paths()
    )
    entries.extend(
        (path, "evaluation_controller_direct_dependency")
        for path in _direct_controller_dependencies()
    )
    entries.extend(
        (path, "evaluation_data_input")
        for path in paper._data_input_paths(runtime.data_root)
    )
    entries.extend(
        (Path(record["path"]), label) for label, record in strict_records.items()
    )
    entries.extend(
        (path, f"baseline_ref_records:{split}")
        for split, path in baseline["ref_records"].items()
    )
    for label, artifacts in baseline["tn"].items():
        entries.extend(
            (path, f"baseline_{label}_{role}")
            for role, path in artifacts.items()
        )
    input_records = paper._merge_input_records(entries, cache)
    source_record = {
        "kind": source.kind,
        "evaluation_id": source.evaluation_id,
        "config": str(source.config),
        "config_sha256": cache.digest(source.config),
        "checkpoint": str(source.checkpoint),
        "checkpoint_sha256": source.checkpoint_sha256,
        "training_run_root": str(source.training_run_root),
        "training_phase": source.training_phase,
        "training_seed": source.training_seed,
    }
    return {
        "schema": SCHEMA,
        "status": "planned",
        "created_at_utc": paper._utc_now(),
        "repository_root": str(REPO_ROOT),
        "evaluation_id": EVALUATION_ID,
        "output_dir": str(output_dir),
        "output_dir_fresh_at_plan": True,
        "source": source_record,
        "checkpoint_audit": checkpoint_audit,
        "baseline": {
            "id": headline.BASELINE_ID,
            "manifest": str(baseline["manifest"]),
            "manifest_sha256": BASELINE_MANIFEST_SHA256,
            "checkpoint_sha256": BASELINE_CHECKPOINT_SHA256,
            "run_id": baseline["run_id"],
            "ref_summary": str(baseline["ref_summary"]),
            "tn_records": {
                label: str(artifacts["records"])
                for label, artifacts in baseline["tn"].items()
            },
            "tn_summaries": {
                label: str(artifacts["summary"])
                for label, artifacts in baseline["tn"].items()
            },
        },
        "runtime": {
            "python": str(runtime.python),
            "data_root": str(runtime.data_root),
            "device": runtime.device,
            "batch_size": runtime.batch_size,
            "num_workers": runtime.num_workers,
            "amp": runtime.amp,
            "log_every": runtime.log_every,
            "eval_seed": paper.EVAL_SEED,
            "topk": 1,
            "max_ref_batches": 0,
            "max_tn_batches": 0,
        },
        "protocol": {
            "ref_splits": list(paper.REF_SPLITS),
            "strict_manifests": strict_records,
            "processes": ["ref8_strict2031", "strict1607"],
            "strict1607_skip_ref": True,
            "full_per_example_records": True,
            "bootstrap": {
                "iterations": BOOTSTRAP_ITERATIONS,
                "confidence": BOOTSTRAP_CONFIDENCE,
                "seed": BOOTSTRAP_SEED,
                "unit": "paired_image_cluster",
            },
        },
        "commands": paper._commands(runtime, source, output_dir),
        "inputs": {"algorithm": "sha256", "records": input_records},
    }


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_candidate_summary_provenance(
    plan: Mapping[str, Any],
    primary: Mapping[str, Any],
    supplemental: Mapping[str, Any],
) -> None:
    runtime = plan["runtime"]
    source = plan["source"]
    expected = {
        "config": str(Path(str(source["config"])).resolve(strict=True)),
        "config_sha256": source["config_sha256"],
        "checkpoint_sha256": source["checkpoint_sha256"],
        "amp": True,
        "device": "cuda:0",
        "data_root": str(Path(str(runtime["data_root"])).resolve(strict=True)),
        "batch_size": 16,
        "num_workers": 4,
        "ref_score_key": REF_SCORE_KEY,
        "tn_score_key": TN_SCORE_KEY,
        "score_ownership": SCORE_OWNERSHIP,
    }
    rows = [
        *(primary.get("refcoco") or []),
        *(primary.get("tn") or []),
        *(supplemental.get("tn") or []),
    ]
    if len(rows) != len(paper.REF_SPLITS) + 2 or any(
        not isinstance(row, Mapping) for row in rows
    ):
        raise DenseDutyFormalEvaluationError(
            "candidate summaries do not contain the exact ten formal rows"
        )
    for index, row in enumerate(rows):
        drift = {
            key: (row.get(key), value)
            for key, value in expected.items()
            if row.get(key) != value
        }
        if drift:
            raise DenseDutyFormalEvaluationError(
                f"candidate summary row {index} provenance drifted: {drift}"
            )


def _validate_paired_report(
    report: Mapping[str, Any],
    *,
    label: str,
    manifest: Path,
    baseline_records: Path,
    candidate_records: Path,
    baseline_run_id: str,
    candidate_run_id: str,
    cache: paper.HashCache,
) -> None:
    validation = report.get("validation")
    if not isinstance(validation, Mapping):
        raise DenseDutyFormalEvaluationError(
            f"{label} comparison lacks validation evidence"
        )
    expected_n = int(paper.STRICT_SPECS[label]["rows"])
    exact_validation = {
        "pass": True,
        "manifest_n": expected_n,
        "valid_n": expected_n,
        "invalid_n": 0,
        "manifest_index_order_match": True,
        "sample_id_order_match": True,
        "image_id_order_match": True,
        "split_order_match": True,
        "valid_mask_match": True,
        "baseline_run_ids": [baseline_run_id],
        "candidate_run_ids": [candidate_run_id],
        "baseline_manifest_binding_mode": "source_to_derived_v1",
        "candidate_manifest_binding_mode": "source_to_derived_v1",
    }
    drift = {
        key: (validation.get(key), value)
        for key, value in exact_validation.items()
        if validation.get(key) != value
    }
    if drift:
        raise DenseDutyFormalEvaluationError(
            f"{label} paired validation drifted: {drift}"
        )

    inputs = report.get("input_files")
    if not isinstance(inputs, Mapping):
        raise DenseDutyFormalEvaluationError(
            f"{label} comparison lacks byte-identity evidence"
        )
    expected_inputs = {
        "manifest": (manifest, paper.STRICT_SPECS[label]["sha256"]),
        "baseline_records": (baseline_records, cache.digest(baseline_records)),
        "candidate_records": (candidate_records, cache.digest(candidate_records)),
    }
    for role, (path, sha256) in expected_inputs.items():
        record = inputs.get(role)
        if (
            not isinstance(record, Mapping)
            or Path(str(record.get("path", ""))).resolve(strict=True)
            != path.resolve(strict=True)
            or record.get("sha256") != sha256
        ):
            raise DenseDutyFormalEvaluationError(
                f"{label} comparison {role} byte identity drifted"
            )
    if inputs.get("identity_is_from_the_same_bytes_used_for_metrics") is not True:
        raise DenseDutyFormalEvaluationError(
            f"{label} comparison did not bind metric bytes"
        )

    bootstrap = report.get("paired_bootstrap")
    expected_bootstrap = {
        "unit": "image_cluster",
        "paired": True,
        "recomputes_each_model_q05_per_resample": True,
        "iterations": BOOTSTRAP_ITERATIONS,
        "confidence": BOOTSTRAP_CONFIDENCE,
        "seed": BOOTSTRAP_SEED,
        "valid_records_n": expected_n,
    }
    if not isinstance(bootstrap, Mapping) or any(
        bootstrap.get(key) != value for key, value in expected_bootstrap.items()
    ) or int(bootstrap.get("image_clusters_n", 0)) <= 0:
        raise DenseDutyFormalEvaluationError(
            f"{label} paired bootstrap contract drifted"
        )


def _comparison_decision(
    ref_comparison: Mapping[str, Mapping[str, Any]],
    fpr_comparisons: Mapping[str, Mapping[str, Any]],
) -> dict[str, bool]:
    if set(ref_comparison) != set(paper.REF_SPLITS):
        raise DenseDutyFormalEvaluationError("formal comparison lacks exact Ref8")
    if set(fpr_comparisons) != {"strict2031", "strict1607"}:
        raise DenseDutyFormalEvaluationError("formal comparison lacks both strict TN sets")
    all_ref_win = all(row.get("strict_win") is True for row in ref_comparison.values())
    all_fpr_win = all(row.get("strict_win") is True for row in fpr_comparisons.values())
    return {
        "all_ref8_splits_strictly_higher": all_ref_win,
        "both_strict_fpr95_strictly_lower": all_fpr_win,
        "all_requested_point_estimates_strictly_better": all_ref_win and all_fpr_win,
        "overall_target_met": all_ref_win and all_fpr_win,
    }


def _postflight(
    plan: Mapping[str, Any], input_rehash: Mapping[str, Any]
) -> dict[str, Any]:
    output_dir = Path(str(plan["output_dir"])).resolve(strict=True)
    checkpoint = Path(str(plan["source"]["checkpoint"])).resolve(strict=True)
    cache = paper.HashCache()
    checkpoint_sha = cache.digest(checkpoint)
    if checkpoint_sha != plan["source"]["checkpoint_sha256"]:
        raise DenseDutyFormalEvaluationError("checkpoint changed during evaluation")
    # Reparse the authoritative manifest and rehash every declared artifact at
    # postflight.  The launch input rehash is necessary but is not a substitute
    # for replaying the baseline's semantic identity contract.
    observed_baseline = _baseline_contract(cache)
    if (
        str(observed_baseline["manifest"]) != str(plan["baseline"]["manifest"])
        or observed_baseline["run_id"] != plan["baseline"]["run_id"]
        or any(
            str(observed_baseline["tn"][label]["records"])
            != str(plan["baseline"]["tn_records"][label])
            for label in ("strict2031", "strict1607")
        )
        or any(
            str(observed_baseline["tn"][label]["summary"])
            != str(plan["baseline"]["tn_summaries"][label])
            for label in ("strict2031", "strict1607")
        )
    ):
        raise DenseDutyFormalEvaluationError(
            "canonical baseline binding changed between launch and postflight"
        )
    run_id = paper._checkpoint_run_id(checkpoint)
    primary_dir = (output_dir / "ref8_strict2031").resolve(strict=True)
    supplemental_dir = (output_dir / "strict1607").resolve(strict=True)
    primary_summary_path = (primary_dir / "summary.json").resolve(strict=True)
    supplemental_summary_path = (supplemental_dir / "summary.json").resolve(strict=True)
    primary = paper._load_summary(primary_summary_path, label="dense-duty Ref8+strict2031")
    supplemental = paper._load_summary(supplemental_summary_path, label="dense-duty strict1607")
    if supplemental["refcoco"]:
        raise DenseDutyFormalEvaluationError("strict1607 --skip_ref produced Ref rows")
    _validate_candidate_summary_provenance(plan, primary, supplemental)
    candidate_ref = paper._verify_ref_rows(
        primary,
        summary_path=primary_summary_path,
        section_dir=primary_dir,
        checkpoint=checkpoint,
        run_id=run_id,
        cache=cache,
    )
    candidate_tn = {
        "strict2031": paper._verify_tn_row(
            primary,
            label="strict2031",
            summary_path=primary_summary_path,
            section_dir=primary_dir,
            checkpoint=checkpoint,
            run_id=run_id,
            cache=cache,
        ),
        "strict1607": paper._verify_tn_row(
            supplemental,
            label="strict1607",
            summary_path=supplemental_summary_path,
            section_dir=supplemental_dir,
            checkpoint=checkpoint,
            run_id=run_id,
            cache=cache,
        ),
    }

    baseline_summary_path = Path(str(plan["baseline"]["ref_summary"])).resolve(strict=True)
    baseline_summary = paper._load_summary(baseline_summary_path, label="canonical b58 Ref8")
    baseline_ref = paper._verify_ref_rows(
        baseline_summary,
        summary_path=baseline_summary_path,
        section_dir=baseline_summary_path.parent,
        checkpoint=Path(str(headline.FIXED_BASELINE["checkpoint"])).resolve(strict=True),
        run_id=str(plan["baseline"]["run_id"]),
        cache=cache,
    )
    baseline_tn = {}
    baseline_tn_rows = {}
    for label in ("strict2031", "strict1607"):
        summary_path = Path(
            str(plan["baseline"]["tn_summaries"][label])
        ).resolve(strict=True)
        summary = paper._load_summary(
            summary_path, label=f"canonical b58 {label}"
        )
        if summary["refcoco"] or len(summary["tn"]) != 1:
            raise DenseDutyFormalEvaluationError(
                f"canonical b58 {label} summary shape drifted"
            )
        baseline_tn[label] = paper._verify_tn_row(
            summary,
            label=label,
            summary_path=summary_path,
            section_dir=summary_path.parent,
            checkpoint=Path(
                str(headline.FIXED_BASELINE["checkpoint"])
            ).resolve(strict=True),
            run_id=str(plan["baseline"]["run_id"]),
            cache=cache,
        )
        baseline_tn_rows[label] = summary["tn"][0]

    ref_comparison = {}
    for split in paper.REF_SPLITS:
        baseline_acc = float(baseline_ref[split]["summary_acc50"])
        candidate_acc = float(candidate_ref[split]["summary_acc50"])
        total = int(candidate_ref[split]["manifest_n"])
        baseline_correct = _exact_binary_count(
            baseline_acc, total, label=f"{split} baseline Acc@0.5"
        )
        candidate_correct = _exact_binary_count(
            candidate_acc, total, label=f"{split} candidate Acc@0.5"
        )
        if baseline_correct != BASELINE_REF_CORRECT[split]:
            raise DenseDutyFormalEvaluationError(
                f"{split} canonical baseline correct-count drifted"
            )
        minimum_correct = baseline_correct + 1
        strict_win = candidate_correct >= minimum_correct
        ref_comparison[split] = {
            "num_expressions": total,
            "baseline_acc50": baseline_acc,
            "candidate_acc50": candidate_acc,
            "candidate_minus_baseline": candidate_acc - baseline_acc,
            "baseline_correct": baseline_correct,
            "candidate_correct": candidate_correct,
            "minimum_candidate_correct_for_strict_win": minimum_correct,
            "strict_win": strict_win,
            "candidate_strictly_higher": strict_win,
        }

    comparison_dir = output_dir / "comparisons"
    fpr_comparisons = {}
    candidate_tn_rows = {
        "strict2031": primary["tn"][0],
        "strict1607": supplemental["tn"][0],
    }
    for label in ("strict2031", "strict1607"):
        manifest_path = Path(
            str(plan["protocol"]["strict_manifests"][label]["path"])
        ).resolve(strict=True)
        baseline_records = Path(
            str(plan["baseline"]["tn_records"][label])
        ).resolve(strict=True)
        candidate_records = Path(
            str(candidate_tn[label]["records"]["path"])
        ).resolve(strict=True)
        try:
            report = compare_record_files(
                baseline_records=baseline_records,
                candidate_records=candidate_records,
                manifest_path=manifest_path,
                bootstrap_iterations=BOOTSTRAP_ITERATIONS,
                confidence=BOOTSTRAP_CONFIDENCE,
                seed=BOOTSTRAP_SEED,
            )
        except (RecordComparisonError, OSError, ValueError) as exc:
            raise DenseDutyFormalEvaluationError(
                f"{label} paired comparison failed: {exc}"
            ) from exc
        _validate_paired_report(
            report,
            label=label,
            manifest=manifest_path,
            baseline_records=baseline_records,
            candidate_records=candidate_records,
            baseline_run_id=str(plan["baseline"]["run_id"]),
            candidate_run_id=run_id,
            cache=cache,
        )
        global_metrics = report.get("global")
        if not isinstance(global_metrics, Mapping):
            raise DenseDutyFormalEvaluationError(
                f"{label} paired comparison lacks global metrics"
            )
        baseline_metrics = global_metrics.get("baseline")
        candidate_metrics = global_metrics.get("candidate")
        if not isinstance(baseline_metrics, Mapping) or not isinstance(
            candidate_metrics, Mapping
        ):
            raise DenseDutyFormalEvaluationError(
                f"{label} paired comparison lacks model metrics"
            )
        baseline_fpr95 = float(baseline_metrics["fpr95"]["fpr"])
        candidate_fpr95 = float(candidate_metrics["fpr95"]["fpr"])
        baseline_q05 = float(baseline_metrics["fpr95"]["threshold"])
        candidate_q05 = float(candidate_metrics["fpr95"]["threshold"])
        expected_n = int(paper.STRICT_SPECS[label]["rows"])
        baseline_false_accepts = _exact_binary_count(
            baseline_fpr95,
            expected_n,
            label=f"{label} baseline FPR95",
        )
        candidate_false_accepts = _exact_binary_count(
            candidate_fpr95,
            expected_n,
            label=f"{label} candidate FPR95",
        )
        if baseline_false_accepts != BASELINE_TN_FALSE_ACCEPTS[label]:
            raise DenseDutyFormalEvaluationError(
                f"{label} canonical baseline false-accept count drifted"
            )
        expected_scalars = (
            (
                baseline_fpr95,
                float(baseline_tn[label]["summary_fpr95"]),
                "baseline FPR95",
            ),
            (
                candidate_fpr95,
                float(candidate_tn[label]["summary_fpr95"]),
                "candidate FPR95",
            ),
            (
                baseline_q05,
                float(baseline_tn_rows[label]["threshold_at_95tpr"]),
                "baseline q05",
            ),
            (
                candidate_q05,
                float(candidate_tn_rows[label]["threshold_at_95tpr"]),
                "candidate q05",
            ),
        )
        for observed, expected, scalar_label in expected_scalars:
            if not math.isclose(
                observed, expected, rel_tol=0.0, abs_tol=1e-12
            ):
                raise DenseDutyFormalEvaluationError(
                    f"{label} {scalar_label} differs from record replay"
                )
        maximum_false_accepts = baseline_false_accepts - 1
        strict_win = candidate_false_accepts <= maximum_false_accepts
        json_path = comparison_dir / f"{label}_paired_fpr95.json"
        markdown_path = comparison_dir / f"{label}_paired_fpr95.md"
        paper._write_json_atomic(json_path, report)
        _write_text_atomic(markdown_path, render_markdown(report))
        fpr_comparisons[label] = {
            "num_pairs": expected_n,
            "tie_policy": "negative_score >= positive_q05_order_statistic",
            "baseline_q05": baseline_q05,
            "candidate_q05": candidate_q05,
            "baseline_fpr95": baseline_fpr95,
            "candidate_fpr95": candidate_fpr95,
            "candidate_minus_baseline": global_metrics[
                "candidate_minus_baseline_fpr95"
            ],
            "baseline_false_accepts": baseline_false_accepts,
            "candidate_false_accepts": candidate_false_accepts,
            "maximum_candidate_false_accepts_for_strict_win": (
                maximum_false_accepts
            ),
            "strict_win": strict_win,
            "candidate_strictly_lower": strict_win,
            "paired_bootstrap": report["paired_bootstrap"],
            "report": paper._file_record(
                json_path, cache, roles=(label, "paired_fpr95")
            ),
            "markdown": paper._file_record(
                markdown_path, cache, roles=(label, "comparison_markdown")
            ),
        }

    decision = _comparison_decision(ref_comparison, fpr_comparisons)
    comparison = {
        "schema": COMPARISON_SCHEMA,
        "baseline_id": headline.BASELINE_ID,
        "candidate_id": EVALUATION_ID,
        "ref8": ref_comparison,
        "tn": fpr_comparisons,
        "decision": decision,
    }
    comparison_path = comparison_dir / "formal_comparison.json"
    paper._write_json_atomic(comparison_path, comparison)
    return {
        "schema": POSTFLIGHT_SCHEMA,
        "status": "passed",
        "outcome": (
            "target_met" if decision["overall_target_met"] else "target_not_met"
        ),
        "target_met": decision["overall_target_met"],
        "validated_at_utc": paper._utc_now(),
        "evaluation_id": EVALUATION_ID,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "run_id": run_id,
        },
        "input_rehash": dict(input_rehash),
        "artifacts": {
            "primary_summary": paper._file_record(
                primary_summary_path, cache, roles=("ref8", "strict2031")
            ),
            "supplemental_summary": paper._file_record(
                supplemental_summary_path, cache, roles=("strict1607",)
            ),
            "ref8": candidate_ref,
            "strict2031": candidate_tn["strict2031"],
            "strict1607": candidate_tn["strict1607"],
            "comparison": paper._file_record(
                comparison_path, cache, roles=("formal_comparison",)
            ),
        },
        "comparison": comparison,
        "contracts": {
            "terminal_confidence_checkpoint": True,
            "rank_10295_confidence_4412_lineage": True,
            "b16_acc4_e16_no_stage_b_teacher": True,
            "current_source_and_config_closure": True,
            "runtime_audit_passed": True,
            "exact_two_process_protocol": True,
            "same_checkpoint_across_all_rows": True,
            "summary_runtime_and_score_routes_exact": True,
            "full_per_example_records": True,
            "zero_invalid_records": True,
            "canonical_b58_baseline_identity": True,
            "canonical_b58_tn_record_replay": True,
            "integer_strict_win_boundaries": True,
            "paired_fpr95_bootstrap": True,
            "direct_controller_dependencies_rehashed": True,
        },
    }


def _execute(plan: dict[str, Any]) -> int:
    output_dir = Path(str(plan["output_dir"]))
    runtime = _fixed_runtime()
    output_dir.mkdir(parents=True, exist_ok=False)
    launch_path = output_dir / "launch_manifest.json"
    plan["status"] = "running"
    plan["started_at_utc"] = paper._utc_now()
    plan["completed_phases"] = []
    paper._write_json_atomic(launch_path, plan)
    try:
        for command_spec in plan["commands"]:
            phase_id = command_spec["phase_id"]
            plan["current_phase"] = phase_id
            paper._write_json_atomic(launch_path, plan)
            paper._verify_input_identities(plan)
            returncode = paper._stream_command(
                command_spec["command"],
                runtime=runtime,
                log_path=Path(command_spec["console_log"]),
            )
            if returncode != 0:
                raise DenseDutyFormalEvaluationError(
                    f"evaluation phase {phase_id} exited with code {returncode}"
                )
            plan["completed_phases"].append(
                {
                    "phase_id": phase_id,
                    "status": "completed",
                    "returncode": 0,
                    "finished_at_utc": paper._utc_now(),
                }
            )
            paper._write_json_atomic(launch_path, plan)
        input_rehash = paper._rehash_inputs(plan)
        input_rehash_path = output_dir / "input_rehash.json"
        paper._write_json_atomic(input_rehash_path, input_rehash)
        postflight = _postflight(plan, input_rehash)
        postflight_path = output_dir / "postflight.json"
        paper._write_json_atomic(postflight_path, postflight)
        cache = paper.HashCache()
        plan["input_rehash_artifact"] = paper._file_record(
            input_rehash_path, cache, roles=("input_rehash",)
        )
        plan["postflight_artifact"] = paper._file_record(
            postflight_path, cache, roles=("postflight",)
        )
        plan["postflight"] = postflight
        plan["status"] = "completed"
        plan["current_phase"] = None
        plan["finished_at_utc"] = paper._utc_now()
        paper._write_json_atomic(launch_path, plan)
        print(f"[OK] dense-duty formal evaluation completed: {output_dir}")
        return 0
    except BaseException as exc:
        plan["status"] = "failed"
        plan["failure_phase"] = plan.get("current_phase") or "postflight"
        plan["error"] = f"{type(exc).__name__}: {exc}"
        plan["finished_at_utc"] = paper._utc_now()
        paper._write_json_atomic(launch_path, plan)
        print(f"[FAIL] {plan['error']}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed dense-duty terminal formal evaluation."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("dry-run", "run"):
        child = subparsers.add_parser(mode)
        child.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = build_plan(args.output_dir)
    except (
        DenseDutyFormalEvaluationError,
        paper.PaperEvaluationError,
        FileNotFoundError,
        FileExistsError,
        OSError,
    ) as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.mode == "dry-run":
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    return _execute(plan)


if __name__ == "__main__":
    raise SystemExit(main())
