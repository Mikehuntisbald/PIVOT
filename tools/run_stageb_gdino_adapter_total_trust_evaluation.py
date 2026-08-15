#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Evaluate one audited total-trust checkpoint against historical B58.

The historical B58 model is not rerun. Its sealed paper-results manifest and
per-example records are replayed before planning and again after candidate
evaluation. The candidate is evaluated with the same canonical runtime and
manifests used by the paper evaluator.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import eval_stageb_tn_val as tn_eval  # noqa: E402
from tools import eval_text_groundingdino_refcoco_tn as joint_eval  # noqa: E402
from tools import run_stageb_dense_duty_formal_evaluation as dense  # noqa: E402
from tools import run_stageb_paper_evaluations as paper  # noqa: E402
from tools.compare_stageb_fpr95_records import (  # noqa: E402
    RecordComparisonError,
    compare_record_files,
    render_markdown,
)
from util.slconfig import SLConfig  # noqa: E402


FIXED_PYTHON = Path("/home/haoyi/miniconda/envs/gdino5090/bin/python")
if __name__ == "__main__" and Path(sys.executable).resolve() != FIXED_PYTHON.resolve():
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


SCHEMA = "pivot.stageb.gdino_adapter_total_trust_evaluation_launch/v1"
POSTFLIGHT_SCHEMA = (
    "pivot.stageb.gdino_adapter_total_trust_evaluation_postflight/v1"
)
COMPARISON_SCHEMA = (
    "pivot.stageb.gdino_adapter_total_trust_historical_b58_comparison/v1"
)
LINEAGE_EQUALITY_SCHEMA = (
    "pivot.stageb.gdino_adapter_total_trust_lineage_equality/v1"
)
TOTAL_TRUST_SCHEMA = "stageb-gdino-adapter-total-trust-probe-v1"
LEGACY_RECEIPT_SCHEMA = "pivot.stageb.legacy_replay_receipt/v1"
STAGEA_R100_RECEIPT_SCHEMA = "pivot.stagea_b58_r100_receipt/v3"
STAGEA_LAUNCH_SOURCE_SCHEMA = "pivot.stagea.launch_source_manifest/v1"
LEGACY_R100_INITIALIZATION = (
    "sealed_legacy_b58_r100_to_total_trust_confidence_pretrain_model_path"
)
STAGEA_R100_INITIALIZATION = (
    "sealed_stagea_b58_patch_realign_r100_to_total_trust_confidence_pretrain_model_path"
)
TOTAL_AUDITOR = (
    REPO_ROOT / "tools/stageb_gdino_adapter_total_trust_probe_audit.py"
)
REF_SCORE_KEY = "stage_b_gdino_rank_score"
REF_SAFE_SCORE_KEY = "stage_b_gdino_ref_safe_rank_score"
TN_SCORE_KEY = "stage_b_gdino_confidence_score"
SCORE_OWNERSHIP = (
    "shared_frozen_gdino_trunk_independent_rank_confidence_adapters"
)
REF_SAFE_SCORE_OWNERSHIP = SCORE_OWNERSHIP + "_b58_top1_anchored_rank_tail"
BOOTSTRAP_ITERATIONS = dense.BOOTSTRAP_ITERATIONS
BOOTSTRAP_CONFIDENCE = dense.BOOTSTRAP_CONFIDENCE
BOOTSTRAP_SEED = dense.BOOTSTRAP_SEED


class TotalTrustEvaluationError(RuntimeError):
    """The total-trust historical-B58 evaluation contract cannot be proven."""


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TotalTrustEvaluationError(f"could not read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise TotalTrustEvaluationError(f"{label} must be a JSON object")
    return value


def _same_file_record(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(
        left.get(key) == right.get(key)
        for key in ("path", "size_bytes", "sha256")
    )


def _plain_file_record(path: Path, cache: paper.HashCache) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    if not path.is_file():
        raise TotalTrustEvaluationError(f"required artifact is not a file: {path}")
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": cache.digest(path),
    }


def _validate_declared_file(
    value: Any,
    *,
    label: str,
    cache: paper.HashCache,
) -> Path:
    if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
        raise TotalTrustEvaluationError(f"{label} has no file record")
    path = Path(str(value["path"])).expanduser().resolve(strict=True)
    observed = _plain_file_record(path, cache)
    if not _same_file_record(value, observed):
        raise TotalTrustEvaluationError(f"{label} file record drifted: {path}")
    return path


def _lineage_artifact_paths(
    audit_path: Path,
    *,
    cache: paper.HashCache,
) -> tuple[Path, ...]:
    """Rehash every file record reachable through the JSON lineage graph."""

    root = audit_path.expanduser().resolve(strict=True)
    paths: set[Path] = {root}
    visited_json: set[Path] = set()

    def visit_value(value: Any, *, label: str) -> None:
        if isinstance(value, Mapping):
            if {"path", "size_bytes", "sha256"}.issubset(value):
                path = _validate_declared_file(value, label=label, cache=cache)
                paths.add(path)
                if path.suffix == ".json":
                    visit_json(path, label=label)
                return
            for key, nested in value.items():
                visit_value(nested, label=f"{label}.{key}")
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                visit_value(nested, label=f"{label}[{index}]")

    def visit_json(path: Path, *, label: str) -> None:
        path = path.resolve(strict=True)
        if path in visited_json:
            return
        visited_json.add(path)
        payload = _read_json(path, label=label)
        if payload.get("schema") == STAGEA_LAUNCH_SOURCE_SCHEMA:
            # This manifest describes the historical worktree that launched
            # Stage A. Its source-file records are archival evidence, not a
            # requirement that the current evaluation worktree be rolled back
            # to that historical state. Rehashing those original paths here
            # makes a valid forensic seal impossible to replay after normal
            # development. The dedicated verifier instead authenticates the
            # manifest, launch artifacts, and reconstructible git patch; the
            # Stage-A/R100 receipt is rebuilt independently before evaluation.
            from tools.seal_stagea_launch_source import (
                LaunchSourceSealError,
                verify_manifest,
            )

            try:
                verified = verify_manifest(path)
            except (LaunchSourceSealError, OSError) as error:
                raise TotalTrustEvaluationError(
                    f"{label} forensic launch-source verification failed: {error}"
                ) from error
            if verified != payload:
                raise TotalTrustEvaluationError(
                    f"{label} forensic launch-source payload changed during verification"
                )
            return
        visit_value(payload, label=label)

    visit_json(root, label="candidate milestone audit")
    return tuple(sorted(paths, key=str))


def _lineage_command(checkpoint: Path, audit: Path, output: Path) -> list[str]:
    return [
        str(FIXED_PYTHON.resolve(strict=True)),
        str(TOTAL_AUDITOR.resolve(strict=True)),
        "verify-evaluation",
        "--checkpoint",
        str(checkpoint),
        "--audit",
        str(audit),
        "--output",
        str(output),
    ]


def _validate_lineage_payload(
    payload: Mapping[str, Any],
    *,
    checkpoint: Path,
    audit: Path,
    cache: paper.HashCache,
) -> dict[str, Any]:
    expected_scalars = {
        "schema": TOTAL_TRUST_SCHEMA,
        "kind": "evaluation_checkpoint_verified",
        "phase": "confidence",
        "train_mode": "confidence_only",
        "tn_scope": "benchmark_dataft_alltn",
    }
    drift = {
        key: (payload.get(key), expected)
        for key, expected in expected_scalars.items()
        if payload.get(key) != expected
    }
    if drift:
        raise TotalTrustEvaluationError(
            f"candidate lineage is not total-trust confidence-only: {drift}"
        )
    checkpoint_path = _validate_declared_file(
        payload.get("checkpoint"), label="verified candidate checkpoint", cache=cache
    )
    if checkpoint_path != checkpoint:
        raise TotalTrustEvaluationError(
            "verified lineage checkpoint differs from the requested candidate"
        )
    audit_path = _validate_declared_file(
        payload.get("audit"), label="verified milestone audit", cache=cache
    )
    if audit_path != audit:
        raise TotalTrustEvaluationError(
            "verified lineage audit differs from the requested milestone audit"
        )
    config = _validate_declared_file(
        payload.get("config"), label="verified candidate config", cache=cache
    )
    datasets = _validate_declared_file(
        payload.get("datasets"), label="verified candidate datasets", cache=cache
    )

    milestone = _read_json(audit, label="candidate milestone audit")
    if (
        milestone.get("schema") != TOTAL_TRUST_SCHEMA
        or milestone.get("kind") != "milestone_checkpoint"
        or milestone.get("phase") != "confidence"
        or milestone.get("iteration") != payload.get("iteration")
    ):
        raise TotalTrustEvaluationError("candidate milestone audit identity drifted")
    preflight_path = _validate_declared_file(
        milestone.get("preflight"), label="candidate phase preflight", cache=cache
    )
    preflight = _read_json(preflight_path, label="candidate phase preflight")
    initialization = preflight.get("launch", {}).get("initialization")
    if (
        preflight.get("schema") != TOTAL_TRUST_SCHEMA
        or preflight.get("kind") != "phase_preflight"
        or preflight.get("phase") != "confidence"
        or initialization not in {LEGACY_R100_INITIALIZATION, STAGEA_R100_INITIALIZATION}
    ):
        raise TotalTrustEvaluationError(
            "candidate preflight is not a supported sealed R100 continuation"
        )
    receipt_path = _validate_declared_file(
        preflight.get("initial_audit"),
        label="sealed R100 receipt",
        cache=cache,
    )
    receipt = _read_json(receipt_path, label="sealed R100 receipt")
    candidate_record = payload.get("checkpoint")
    if not isinstance(candidate_record, Mapping):
        raise TotalTrustEvaluationError("candidate/R100 tensor records are incomplete")
    receipt_schema = receipt.get("schema")
    root_stagea = None
    if receipt_schema == LEGACY_RECEIPT_SCHEMA:
        if initialization != LEGACY_R100_INITIALIZATION:
            raise TotalTrustEvaluationError("legacy receipt uses the wrong initialization mode")
        checkpoints = receipt.get("checkpoints")
        if not isinstance(checkpoints, Mapping):
            raise TotalTrustEvaluationError("legacy replay receipt has no checkpoints")
        rank = checkpoints.get("rank_r100")
        baseline = checkpoints.get("b58")
        if not isinstance(rank, Mapping) or not isinstance(baseline, Mapping):
            raise TotalTrustEvaluationError("legacy replay receipt lacks B58/R100 records")
        rank_model = rank.get("model")
        baseline_file = baseline.get("file")
        if not isinstance(rank_model, Mapping) or not isinstance(baseline_file, Mapping):
            raise TotalTrustEvaluationError("legacy replay receipt records are incomplete")
        expected_rank_sha = rank_model.get("rank_tensor_sha256")
    elif receipt_schema == STAGEA_R100_RECEIPT_SCHEMA:
        if initialization != STAGEA_R100_INITIALIZATION:
            raise TotalTrustEvaluationError("Stage-A receipt uses the wrong initialization mode")
        from tools.build_stagea_b58_r100_receipt import (
            StageAR100ReceiptError,
            verify_receipt,
        )

        try:
            verified_receipt = verify_receipt(receipt_path)
        except (StageAR100ReceiptError, OSError, KeyError, TypeError) as error:
            raise TotalTrustEvaluationError(
                f"Stage-A/R100 receipt replay failed: {error}"
            ) from error
        if verified_receipt != receipt:
            raise TotalTrustEvaluationError("Stage-A/R100 receipt replay changed its payload")
        rank_model = receipt.get("rank_r100")
        root_stagea = receipt.get("stagea")
        if not isinstance(rank_model, Mapping) or not isinstance(root_stagea, Mapping):
            raise TotalTrustEvaluationError("Stage-A/R100 receipt records are incomplete")
        expected_rank_sha = rank_model.get("rank_tensor_sha256")
        baseline_file = root_stagea.get("b58_source")
        if not isinstance(baseline_file, Mapping):
            raise TotalTrustEvaluationError("Stage-A/R100 receipt has no B58 source")
    else:
        raise TotalTrustEvaluationError(f"unsupported R100 receipt schema: {receipt_schema!r}")
    if (
        not isinstance(expected_rank_sha, str)
        or candidate_record.get("rank_sha256") != expected_rank_sha
    ):
        raise TotalTrustEvaluationError(
            "candidate rank tensors are not bitwise-identical to sealed R100"
        )
    historical_b58 = dense._baseline_contract(cache)["checkpoint"]
    if not _same_file_record(
        baseline_file, _plain_file_record(historical_b58, cache)
    ):
        raise TotalTrustEvaluationError(
            "candidate lineage root differs from the historical B58 manifest"
        )
    return {
        **dict(payload),
        "config_path": str(config),
        "datasets_path": str(datasets),
        "preflight_path": str(preflight_path),
        "receipt_path": str(receipt_path),
        "r100_rank_sha256": expected_rank_sha,
        "rank_branch_unchanged_from_r100": True,
        "lineage_root_schema": receipt_schema,
        "root_stagea": dict(root_stagea) if isinstance(root_stagea, Mapping) else None,
        "root_historical_b58_checkpoint": dict(baseline_file),
    }


def _run_candidate_lineage_verification(
    *,
    checkpoint: Path,
    audit: Path,
    output: Path,
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    audit = audit.expanduser().resolve(strict=True)
    output = output.expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"lineage output must be fresh: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = paper._subprocess_environment(dense._fixed_runtime())
    process = subprocess.run(
        _lineage_command(checkpoint, audit, output),
        cwd=REPO_ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise TotalTrustEvaluationError(
            "total-trust checkpoint replay failed: " + process.stdout.strip()
        )
    if not output.is_file():
        raise TotalTrustEvaluationError(
            "total-trust auditor succeeded without writing its verification output"
        )
    payload = _read_json(output, label="total-trust verification output")
    return _validate_lineage_payload(
        payload,
        checkpoint=checkpoint,
        audit=audit,
        cache=paper.HashCache(),
    )


def _temporary_lineage_verification(
    checkpoint: Path, audit: Path
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="pivot-total-trust-eval-lineage-") as raw:
        return _run_candidate_lineage_verification(
            checkpoint=checkpoint,
            audit=audit,
            output=Path(raw) / "verified.json",
        )


def _validate_score_routes(
    *, config: Path, checkpoint: Path, runtime: paper.Runtime
) -> dict[str, Any]:
    import torch

    cfg = SLConfig.fromfile(str(config))
    expected_config = {
        "stage_b_gdino_score_adapter": True,
        "stage_b_gdino_adapter_train_mode": "confidence_only",
        "stage_b_gdino_tn_scope": "benchmark_dataft_alltn",
        "stage_b_gdino_confidence_objective": "detached_recent_q05_total_trust",
        "stage_b_gdino_rank_weight": 0.0,
        "stage_b_gdino_confidence_weight": 1.0,
    }
    drift = {
        key: (getattr(cfg, key, None), expected)
        for key, expected in expected_config.items()
        if getattr(cfg, key, None) != expected
    }
    if drift:
        raise TotalTrustEvaluationError(
            f"verified candidate config does not expose total-trust routes: {drift}"
        )
    ref_key = joint_eval._adapter_ref_score_key(cfg)
    guarded = bool(getattr(cfg, "stage_b_gdino_ref_top1_guard", False))
    expected_ref_key = REF_SAFE_SCORE_KEY if guarded else REF_SCORE_KEY
    expected_ownership = REF_SAFE_SCORE_OWNERSHIP if guarded else SCORE_OWNERSHIP
    if ref_key != expected_ref_key:
        raise TotalTrustEvaluationError(
            f"adapter Ref route drifted: expected {expected_ref_key}, got {ref_key}"
        )
    confidence = torch.tensor([[0.125, 0.75]], dtype=torch.float32)
    decoy = torch.tensor([[99.0, 99.0]], dtype=torch.float32)
    observed = tn_eval._slot_scores(
        {
            TN_SCORE_KEY: confidence,
            expected_ref_key: decoy,
            "stage_b_v7_final_score": decoy,
        },
        cfg,
        beta=1.0,
    )
    if tuple(observed.shape) != (1, 2, 1) or not torch.equal(
        observed.squeeze(-1), confidence
    ):
        raise TotalTrustEvaluationError(
            "adapter TN route no longer consumes stage_b_gdino_confidence_score"
        )
    provenance = joint_eval._evaluation_summary_provenance(
        cfg=cfg,
        args=SimpleNamespace(config=str(config), amp=True, device="cuda:0"),
        checkpoint=checkpoint,
        data_root=runtime.data_root,
    )
    expected_provenance_routes = {
        "ref_score_key": expected_ref_key,
        "tn_score_key": TN_SCORE_KEY,
        "score_ownership": expected_ownership,
    }
    provenance_drift = {
        key: (provenance.get(key), expected)
        for key, expected in expected_provenance_routes.items()
        if provenance.get(key) != expected
    }
    if provenance_drift:
        raise TotalTrustEvaluationError(
            "adapter summary score-route provenance drifted: "
            f"{provenance_drift}"
        )
    return {
        "ref_score_key": expected_ref_key,
        "tn_score_key": TN_SCORE_KEY,
        "score_ownership": expected_ownership,
        "ref_route_function": "eval_text_groundingdino_refcoco_tn._adapter_ref_score_key",
        "tn_route_function": "eval_stageb_tn_val._slot_scores",
        "route_functions_executed": True,
        "summary_route_fields_currently_emitted": True,
        "summary_route_binding": (
            "static route replay plus exact per-row evaluator provenance"
        ),
    }


def _validate_baseline_records(
    baseline: Mapping[str, Any], cache: paper.HashCache
) -> dict[str, Any]:
    checkpoint = Path(str(dense.headline.FIXED_BASELINE["checkpoint"])).resolve(
        strict=True
    )
    ref_summary_path = Path(str(baseline["ref_summary"])).resolve(strict=True)
    ref_summary = paper._load_summary(ref_summary_path, label="canonical B58 Ref8")
    ref = paper._verify_ref_rows(
        ref_summary,
        summary_path=ref_summary_path,
        section_dir=ref_summary_path.parent,
        checkpoint=checkpoint,
        run_id=str(baseline["run_id"]),
        cache=cache,
    )
    for split, expected_correct in dense.BASELINE_REF_CORRECT.items():
        total = int(ref[split]["manifest_n"])
        observed = dense._exact_binary_count(
            ref[split]["summary_acc50"], total, label=f"B58 {split} Acc@0.5"
        )
        if observed != expected_correct:
            raise TotalTrustEvaluationError(
                f"canonical B58 {split} correct-count drifted"
            )

    tn: dict[str, Any] = {}
    rows: dict[str, Mapping[str, Any]] = {}
    for label in ("strict2031", "strict1607"):
        summary_path = Path(str(baseline["tn"][label]["summary"])).resolve(
            strict=True
        )
        summary = paper._load_summary(summary_path, label=f"canonical B58 {label}")
        if summary["refcoco"] or len(summary["tn"]) != 1:
            raise TotalTrustEvaluationError(
                f"canonical B58 {label} summary shape drifted"
            )
        tn[label] = paper._verify_tn_row(
            summary,
            label=label,
            summary_path=summary_path,
            section_dir=summary_path.parent,
            checkpoint=checkpoint,
            run_id=str(baseline["run_id"]),
            cache=cache,
        )
        rows[label] = summary["tn"][0]
        expected_n = int(paper.STRICT_SPECS[label]["rows"])
        false_accepts = dense._exact_binary_count(
            tn[label]["summary_fpr95"],
            expected_n,
            label=f"B58 {label} FPR95",
        )
        if false_accepts != dense.BASELINE_TN_FALSE_ACCEPTS[label]:
            raise TotalTrustEvaluationError(
                f"canonical B58 {label} false-accept count drifted"
            )
    return {"ref": ref, "tn": tn, "tn_rows": rows}


def _fixed_runtime() -> paper.Runtime:
    runtime = dense._fixed_runtime()
    try:
        paper.source_contracts.validate_canonical_runtime(
            {
                "python": str(runtime.python),
                "data_root": str(runtime.data_root),
                "device": runtime.device,
                "batch_size": runtime.batch_size,
                "num_workers": runtime.num_workers,
                "amp": runtime.amp,
                "log_every": runtime.log_every,
                "eval_seed": paper.EVAL_SEED,
                "max_ref_batches": 0,
                "max_tn_batches": 0,
            }
        )
    except paper.source_contracts.EvaluationSourceContractError as exc:
        raise TotalTrustEvaluationError(
            f"canonical evaluation runtime drifted: {exc}"
        ) from exc
    return runtime


def _baseline_plan_record(baseline: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": dense.headline.BASELINE_ID,
        "manifest": str(baseline["manifest"]),
        "manifest_sha256": dense.BASELINE_MANIFEST_SHA256,
        "checkpoint": str(baseline["checkpoint"]),
        "checkpoint_sha256": dense.BASELINE_CHECKPOINT_SHA256,
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
    }


def _input_entries(
    *,
    checkpoint: Path,
    audit: Path,
    config: Path,
    runtime: paper.Runtime,
    baseline: Mapping[str, Any],
    strict_records: Mapping[str, Mapping[str, Any]],
    lineage_paths: Iterable[Path],
) -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = [
        (checkpoint, "evaluation_checkpoint"),
        (audit, "candidate_milestone_audit"),
        (config, "evaluation_config"),
        (Path(__file__).resolve(strict=True), "evaluation_controller"),
        (TOTAL_AUDITOR.resolve(strict=True), "candidate_lineage_auditor"),
        (Path(dense.__file__).resolve(strict=True), "historical_b58_contract"),
        (
            Path(dense.headline.__file__).resolve(strict=True),
            "historical_b58_identity_contract",
        ),
        (baseline["manifest"], "baseline_results_manifest"),
        (baseline["checkpoint"], "baseline_identity_checkpoint"),
        (baseline["config"], "baseline_identity_config"),
        (baseline["ref_summary"], "baseline_ref8_summary"),
    ]
    entries.extend((path, "candidate_lineage_artifact") for path in lineage_paths)
    entries.extend(
        (path, "baseline_declared_training_data") for path in baseline["data"]
    )
    entries.extend(
        (path, "config_dependency") for path in paper._config_paths(config)
    )
    entries.extend(
        (path, "evaluation_code_dependency")
        for path in paper._evaluation_code_paths()
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
    return entries


def build_plan(
    *, checkpoint: Path, audit: Path, output_dir: Path
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    audit = audit.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().resolve(strict=False)
    if output_dir.exists():
        raise FileExistsError(f"evaluation output root must be fresh: {output_dir}")
    runtime = _fixed_runtime()
    cache = paper.HashCache()
    lineage = _temporary_lineage_verification(checkpoint, audit)
    config = Path(lineage["config_path"]).resolve(strict=True)
    score_routes = _validate_score_routes(
        config=config, checkpoint=checkpoint, runtime=runtime
    )
    baseline = dense._baseline_contract(cache)
    baseline_metrics = _validate_baseline_records(baseline, cache)
    strict_records = {
        label: paper._strict_manifest_record(label, cache)
        for label in paper.STRICT_SPECS
    }
    lineage_paths = _lineage_artifact_paths(audit, cache=cache)
    entries = _input_entries(
        checkpoint=checkpoint,
        audit=audit,
        config=config,
        runtime=runtime,
        baseline=baseline,
        strict_records=strict_records,
        lineage_paths=lineage_paths,
    )
    input_records = paper._merge_input_records(entries, cache)
    source = paper.EvaluationSource(
        kind="gdino_adapter_total_trust_audited_milestone",
        evaluation_id=f"total_trust_{paper._checkpoint_run_id(checkpoint)}",
        config=config,
        checkpoint=checkpoint,
        checkpoint_sha256=cache.digest(checkpoint),
        training_phase="confidence",
        diagnostic_only=True,
        selected_phase_id="confidence",
        artifact_repository_root=paper.ARTIFACT_REPOSITORY_ROOT,
    )
    return {
        "schema": SCHEMA,
        "status": "planned",
        "created_at_utc": paper._utc_now(),
        "repository_root": str(REPO_ROOT),
        "evaluation_id": source.evaluation_id,
        "evaluation_scope": "historical_b58_full_diagnostic",
        "output_dir": str(output_dir),
        "output_dir_fresh_at_plan": True,
        "source": {
            "kind": source.kind,
            "evaluation_id": source.evaluation_id,
            "config": str(config),
            "config_sha256": cache.digest(config),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": source.checkpoint_sha256,
            "checkpoint_audit": str(audit),
            "checkpoint_audit_sha256": cache.digest(audit),
            "training_phase": "confidence",
            "iteration": lineage["iteration"],
            "rank_sha256": lineage["r100_rank_sha256"],
        },
        "candidate_lineage": lineage,
        "candidate_lineage_artifact_count": len(lineage_paths),
        "score_routes": score_routes,
        "baseline": _baseline_plan_record(baseline),
        "baseline_metrics": {
            "ref8_splits": list(baseline_metrics["ref"]),
            "strict2031_fpr95": baseline_metrics["tn"]["strict2031"][
                "summary_fpr95"
            ],
            "strict1607_fpr95": baseline_metrics["tn"]["strict1607"][
                "summary_fpr95"
            ],
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
            "baseline_reused_not_rerun": True,
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
    source = plan["source"]
    runtime = plan["runtime"]
    expected = {
        "config": str(Path(str(source["config"])).resolve(strict=True)),
        "config_sha256": source["config_sha256"],
        "checkpoint_sha256": source["checkpoint_sha256"],
        "amp": True,
        "device": "cuda:0",
        "data_root": str(Path(str(runtime["data_root"])).resolve(strict=True)),
        "batch_size": 16,
        "num_workers": 4,
    }
    rows = [
        *(primary.get("refcoco") or []),
        *(primary.get("tn") or []),
        *(supplemental.get("tn") or []),
    ]
    if len(rows) != len(paper.REF_SPLITS) + 2 or any(
        not isinstance(row, Mapping) for row in rows
    ):
        raise TotalTrustEvaluationError(
            "candidate summaries do not contain the exact ten canonical rows"
        )
    expected.update(
        {
            "ref_score_key": plan["score_routes"]["ref_score_key"],
            "tn_score_key": plan["score_routes"]["tn_score_key"],
            "score_ownership": plan["score_routes"]["score_ownership"],
        }
    )
    for index, row in enumerate(rows):
        drift = {
            key: (row.get(key), value)
            for key, value in expected.items()
            if row.get(key) != value
        }
        if drift:
            raise TotalTrustEvaluationError(
                f"candidate summary row {index} provenance drifted: {drift}"
            )


def _strict_fpr_result(
    *,
    label: str,
    report: Mapping[str, Any],
    baseline_summary_fpr95: float,
    candidate_summary_fpr95: float,
    baseline_summary_q05: float,
    candidate_summary_q05: float,
) -> dict[str, Any]:
    global_metrics = report.get("global")
    if not isinstance(global_metrics, Mapping):
        raise TotalTrustEvaluationError(f"{label} comparison lacks global metrics")
    baseline_metrics = global_metrics.get("baseline")
    candidate_metrics = global_metrics.get("candidate")
    if not isinstance(baseline_metrics, Mapping) or not isinstance(
        candidate_metrics, Mapping
    ):
        raise TotalTrustEvaluationError(f"{label} comparison lacks model metrics")
    baseline_fpr95 = float(baseline_metrics["fpr95"]["fpr"])
    candidate_fpr95 = float(candidate_metrics["fpr95"]["fpr"])
    baseline_q05 = float(baseline_metrics["fpr95"]["threshold"])
    candidate_q05 = float(candidate_metrics["fpr95"]["threshold"])
    for observed, expected, scalar in (
        (baseline_fpr95, baseline_summary_fpr95, "baseline FPR95"),
        (candidate_fpr95, candidate_summary_fpr95, "candidate FPR95"),
        (baseline_q05, baseline_summary_q05, "baseline q05"),
        (candidate_q05, candidate_summary_q05, "candidate q05"),
    ):
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise TotalTrustEvaluationError(
                f"{label} {scalar} differs from record replay"
            )
    total = int(paper.STRICT_SPECS[label]["rows"])
    baseline_false_accepts = dense._exact_binary_count(
        baseline_fpr95, total, label=f"{label} baseline FPR95"
    )
    candidate_false_accepts = dense._exact_binary_count(
        candidate_fpr95, total, label=f"{label} candidate FPR95"
    )
    if baseline_false_accepts != dense.BASELINE_TN_FALSE_ACCEPTS[label]:
        raise TotalTrustEvaluationError(
            f"{label} canonical B58 false-accept count drifted"
        )
    strict_win = candidate_false_accepts < baseline_false_accepts
    return {
        "num_pairs": total,
        "tie_policy": "negative_score >= positive_q05_order_statistic",
        "baseline_q05": baseline_q05,
        "candidate_q05": candidate_q05,
        "baseline_fpr95": baseline_fpr95,
        "candidate_fpr95": candidate_fpr95,
        "candidate_minus_baseline": candidate_fpr95 - baseline_fpr95,
        "baseline_false_accepts": baseline_false_accepts,
        "candidate_false_accepts": candidate_false_accepts,
        "maximum_candidate_false_accepts_for_strict_win": (
            baseline_false_accepts - 1
        ),
        "strict_win": strict_win,
        "candidate_strictly_lower": strict_win,
        "paired_bootstrap": report["paired_bootstrap"],
    }


def _postflight(
    plan: Mapping[str, Any],
    input_rehash: Mapping[str, Any],
    *,
    pre_lineage_path: Path,
    post_lineage_path: Path,
) -> dict[str, Any]:
    output_dir = Path(str(plan["output_dir"])).resolve(strict=True)
    checkpoint = Path(str(plan["source"]["checkpoint"])).resolve(strict=True)
    cache = paper.HashCache()
    if cache.digest(checkpoint) != plan["source"]["checkpoint_sha256"]:
        raise TotalTrustEvaluationError("candidate checkpoint changed during evaluation")
    audit = Path(str(plan["source"]["checkpoint_audit"])).resolve(strict=True)
    pre_payload = _read_json(
        pre_lineage_path, label="pre-eval candidate lineage"
    )
    post_payload = _read_json(
        post_lineage_path, label="post-eval candidate lineage"
    )
    pre_lineage = _validate_lineage_payload(
        pre_payload,
        checkpoint=checkpoint,
        audit=audit,
        cache=cache,
    )
    post_lineage = _validate_lineage_payload(
        post_payload,
        checkpoint=checkpoint,
        audit=audit,
        cache=cache,
    )
    if (
        pre_lineage != post_lineage
        or pre_lineage != plan["candidate_lineage"]
        or pre_lineage_path.read_bytes() != post_lineage_path.read_bytes()
    ):
        raise TotalTrustEvaluationError(
            "candidate lineage differs before and after evaluation"
        )

    baseline = dense._baseline_contract(cache)
    baseline_metrics = _validate_baseline_records(baseline, cache)
    planned_baseline = plan["baseline"]
    if (
        str(baseline["manifest"]) != planned_baseline["manifest"]
        or str(baseline["checkpoint"]) != planned_baseline["checkpoint"]
        or baseline["run_id"] != planned_baseline["run_id"]
    ):
        raise TotalTrustEvaluationError(
            "historical B58 binding changed between launch and postflight"
        )

    run_id = paper._checkpoint_run_id(checkpoint)
    primary_dir = (output_dir / "ref8_strict2031").resolve(strict=True)
    supplemental_dir = (output_dir / "strict1607").resolve(strict=True)
    primary_summary_path = (primary_dir / "summary.json").resolve(strict=True)
    supplemental_summary_path = (supplemental_dir / "summary.json").resolve(
        strict=True
    )
    primary = paper._load_summary(
        primary_summary_path, label="total-trust Ref8+strict2031"
    )
    supplemental = paper._load_summary(
        supplemental_summary_path, label="total-trust strict1607"
    )
    if supplemental["refcoco"]:
        raise TotalTrustEvaluationError(
            "strict1607 --skip_ref output unexpectedly contains Ref rows"
        )
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

    ref_comparison: dict[str, Any] = {}
    for split in paper.REF_SPLITS:
        total = int(candidate_ref[split]["manifest_n"])
        baseline_acc = float(baseline_metrics["ref"][split]["summary_acc50"])
        candidate_acc = float(candidate_ref[split]["summary_acc50"])
        baseline_correct = dense._exact_binary_count(
            baseline_acc, total, label=f"{split} B58 Acc@0.5"
        )
        candidate_correct = dense._exact_binary_count(
            candidate_acc, total, label=f"{split} candidate Acc@0.5"
        )
        ref_comparison[split] = {
            "num_expressions": total,
            "baseline_acc50": baseline_acc,
            "candidate_acc50": candidate_acc,
            "candidate_minus_baseline": candidate_acc - baseline_acc,
            "baseline_correct": baseline_correct,
            "candidate_correct": candidate_correct,
            "no_regression": candidate_correct >= baseline_correct,
            "strictly_higher": candidate_correct > baseline_correct,
        }

    comparison_dir = output_dir / "comparisons"
    fpr_comparisons: dict[str, Any] = {}
    candidate_rows = {
        "strict2031": primary["tn"][0],
        "strict1607": supplemental["tn"][0],
    }
    for label in ("strict2031", "strict1607"):
        manifest_path = Path(
            str(plan["protocol"]["strict_manifests"][label]["path"])
        ).resolve(strict=True)
        baseline_records = Path(
            str(planned_baseline["tn_records"][label])
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
            raise TotalTrustEvaluationError(
                f"{label} paired comparison failed: {exc}"
            ) from exc
        dense._validate_paired_report(
            report,
            label=label,
            manifest=manifest_path,
            baseline_records=baseline_records,
            candidate_records=candidate_records,
            baseline_run_id=str(planned_baseline["run_id"]),
            candidate_run_id=run_id,
            cache=cache,
        )
        result = _strict_fpr_result(
            label=label,
            report=report,
            baseline_summary_fpr95=baseline_metrics["tn"][label]["summary_fpr95"],
            candidate_summary_fpr95=candidate_tn[label]["summary_fpr95"],
            baseline_summary_q05=float(
                baseline_metrics["tn_rows"][label]["threshold_at_95tpr"]
            ),
            candidate_summary_q05=float(
                candidate_rows[label]["threshold_at_95tpr"]
            ),
        )
        json_path = comparison_dir / f"{label}_paired_fpr95.json"
        markdown_path = comparison_dir / f"{label}_paired_fpr95.md"
        paper._write_json_atomic(json_path, report)
        _write_text_atomic(markdown_path, render_markdown(report))
        result["report"] = paper._file_record(
            json_path, cache, roles=(label, "paired_fpr95")
        )
        result["markdown"] = paper._file_record(
            markdown_path, cache, roles=(label, "comparison_markdown")
        )
        fpr_comparisons[label] = result

    both_fpr_win = all(row["strict_win"] for row in fpr_comparisons.values())
    all_ref_no_regression = all(
        row["no_regression"] for row in ref_comparison.values()
    )
    stagea_lineage = (
        plan.get("candidate_lineage", {}).get("lineage_root_schema")
        == STAGEA_R100_RECEIPT_SCHEMA
    )
    all_stagea_sealed_gates = both_fpr_win and all_ref_no_regression
    comparison = {
        "schema": COMPARISON_SCHEMA,
        "baseline_id": dense.headline.BASELINE_ID,
        "candidate_id": plan["evaluation_id"],
        "evaluation_scope": "historical_b58_full_diagnostic",
        "ref8": ref_comparison,
        "tn": fpr_comparisons,
        "decision": {
            "both_strict_fpr95_strictly_lower": both_fpr_win,
            "fpr95_goal_met": both_fpr_win,
            "all_ref8_splits_no_regression": all_ref_no_regression,
            "ref8_is_reported_not_an_fpr_acceptance_gate": True,
            "stagea_r100_c100_requires_ref8_no_regression": stagea_lineage,
            "stagea_r100_c100_all_sealed_gates_passed": (
                all_stagea_sealed_gates if stagea_lineage else None
            ),
            "rank_branch_unchanged_from_r100": True,
        },
    }
    comparison_path = comparison_dir / "historical_b58_comparison.json"
    paper._write_json_atomic(comparison_path, comparison)
    lineage_equality = {
        "schema": LINEAGE_EQUALITY_SCHEMA,
        "status": "passed",
        "same_json": True,
        "same_bytes": True,
        "pre": paper._file_record(
            pre_lineage_path, cache, roles=("candidate_lineage_pre",)
        ),
        "post": paper._file_record(
            post_lineage_path, cache, roles=("candidate_lineage_post",)
        ),
    }
    lineage_equality_path = output_dir / "checkpoint_lineage.equality.json"
    paper._write_json_atomic(lineage_equality_path, lineage_equality)
    return {
        "schema": POSTFLIGHT_SCHEMA,
        "status": "passed",
        "outcome": (
            "sealed_replay_goal_met"
            if stagea_lineage and all_stagea_sealed_gates
            else "sealed_replay_goal_not_met"
            if stagea_lineage
            else "fpr95_goal_met"
            if both_fpr_win
            else "fpr95_goal_not_met"
        ),
        "fpr95_goal_met": both_fpr_win,
        "stagea_r100_c100_all_sealed_gates_passed": (
            all_stagea_sealed_gates if stagea_lineage else None
        ),
        "validated_at_utc": paper._utc_now(),
        "evaluation_id": plan["evaluation_id"],
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": cache.digest(checkpoint),
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
                comparison_path, cache, roles=("historical_b58_comparison",)
            ),
            "lineage_equality": paper._file_record(
                lineage_equality_path, cache, roles=("lineage_equality",)
            ),
        },
        "comparison": comparison,
        "contracts": {
            "candidate_total_trust_lineage_replayed_pre_and_post": True,
            "candidate_rank_bitwise_equal_to_sealed_r100": True,
            "canonical_runtime": True,
            "exact_two_process_protocol": True,
            "static_ref_rank_and_tn_confidence_routes_replayed": True,
            "same_checkpoint_across_all_rows": True,
            "full_per_example_records": True,
            "zero_invalid_records": True,
            "historical_b58_not_rerun": True,
            "historical_b58_manifest_and_records_replayed": True,
            "paired_fpr95_bootstrap": True,
        },
    }


def _append_fresh_lineage_input(
    plan: dict[str, Any], lineage_path: Path
) -> None:
    records = plan.get("inputs", {}).get("records")
    if not isinstance(records, list) or not records:
        raise TotalTrustEvaluationError("launch plan has no sealed input records")
    lineage_path = lineage_path.resolve(strict=True)
    if any(
        Path(str(record.get("path", ""))).resolve(strict=True) == lineage_path
        for record in records
        if isinstance(record, Mapping)
    ):
        raise TotalTrustEvaluationError(
            "fresh pre-evaluation lineage unexpectedly duplicates a planned input"
        )
    lineage_record = paper._file_record(
        lineage_path,
        paper.HashCache(),
        roles=("candidate_lineage_pre_eval",),
    )
    records.append(lineage_record)
    records.sort(key=lambda record: str(record["path"]))


def _execute(plan: dict[str, Any]) -> int:
    output_dir = Path(str(plan["output_dir"]))
    runtime = _fixed_runtime()
    output_dir.mkdir(parents=True, exist_ok=False)
    launch_path = output_dir / "launch_manifest.json"
    pre_lineage_path = output_dir / "checkpoint_lineage.pre.json"
    post_lineage_path = output_dir / "checkpoint_lineage.post.json"
    plan["status"] = "preflight"
    plan["started_at_utc"] = paper._utc_now()
    plan["completed_phases"] = []
    try:
        pre_lineage = _run_candidate_lineage_verification(
            checkpoint=Path(plan["source"]["checkpoint"]),
            audit=Path(plan["source"]["checkpoint_audit"]),
            output=pre_lineage_path,
        )
        if pre_lineage != plan["candidate_lineage"]:
            raise TotalTrustEvaluationError(
                "candidate lineage changed between plan and execution"
            )
        paper._verify_input_identities(plan)
        _append_fresh_lineage_input(plan, pre_lineage_path)
        paper._rehash_inputs(plan)
        plan["candidate_lineage_pre_eval"] = paper._file_record(
            pre_lineage_path,
            paper.HashCache(),
            roles=("candidate_lineage_pre_eval",),
        )
        plan["status"] = "running"
        paper._write_json_atomic(launch_path, plan)
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
                raise TotalTrustEvaluationError(
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

        post_lineage = _run_candidate_lineage_verification(
            checkpoint=Path(plan["source"]["checkpoint"]),
            audit=Path(plan["source"]["checkpoint_audit"]),
            output=post_lineage_path,
        )
        if post_lineage != pre_lineage:
            raise TotalTrustEvaluationError(
                "candidate lineage changed during evaluation"
            )
        input_rehash = paper._rehash_inputs(plan)
        input_rehash_path = output_dir / "input_rehash.json"
        paper._write_json_atomic(input_rehash_path, input_rehash)
        postflight = _postflight(
            plan,
            input_rehash,
            pre_lineage_path=pre_lineage_path,
            post_lineage_path=post_lineage_path,
        )
        postflight_path = output_dir / "postflight.json"
        paper._write_json_atomic(postflight_path, postflight)
        cache = paper.HashCache()
        plan["candidate_lineage_post_eval"] = paper._file_record(
            post_lineage_path, cache, roles=("candidate_lineage_post_eval",)
        )
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
        print(f"[OK] total-trust historical-B58 evaluation completed: {output_dir}")
        return 0
    except BaseException as exc:
        plan["status"] = "failed"
        plan["failure_phase"] = plan.get("current_phase") or "preflight_or_postflight"
        plan["error"] = f"{type(exc).__name__}: {exc}"
        plan["finished_at_utc"] = paper._utc_now()
        paper._write_json_atomic(launch_path, plan)
        print(f"[FAIL] {plan['error']}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("dry-run", "run"):
        child = subparsers.add_parser(mode)
        child.add_argument("--checkpoint", type=Path, required=True)
        child.add_argument("--checkpoint-audit", type=Path, required=True)
        child.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = build_plan(
            checkpoint=args.checkpoint,
            audit=args.checkpoint_audit,
            output_dir=args.output_dir,
        )
    except (
        TotalTrustEvaluationError,
        dense.DenseDutyFormalEvaluationError,
        paper.PaperEvaluationError,
        FileNotFoundError,
        FileExistsError,
        OSError,
    ) as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.mode == "dry-run":
        print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
        return 0
    return _execute(plan)


if __name__ == "__main__":
    raise SystemExit(main())
