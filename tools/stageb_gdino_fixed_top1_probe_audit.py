#!/usr/bin/env python3
"""Fail-closed audit for checkpoint-specific frozen-GDINO top-1 confidence training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch

from tools.stageb_dependency_audit import (
    DependencyAuditError,
    config_import_chain,
    local_python_dependency_paths,
)

from tools.stageb_gdino_adapter_probe_audit import (
    RANK_MILESTONES,
    SCHEMA as TWO_PHASE_SCHEMA,
    ProbeAuditError,
    checkpoint_args,
    checkpoint_record,
    count_nonempty_lines,
    file_record,
    load_checkpoint,
    read_json,
    resolve_path,
    write_json,
    _verify_milestone_checkpoint as _verify_two_phase_milestone_checkpoint,
)
from tools.judge_stageb_fixed_gdino_top1_qwen import (
    ASSET_POLICY_SHA256 as P3_ASSET_POLICY_SHA256,
    EXTRACTION_SCHEMA as P3_EXTRACTION_SCHEMA,
    GENERATION_CONFIG_SHA256 as P3_GENERATION_CONFIG_SHA256,
    INFERENCE_BATCH_SIZE as P3_INFERENCE_BATCH_SIZE,
    JUDGE_RUNTIME_POLICY as P3_JUDGE_RUNTIME_POLICY,
    JUDGE_RUNTIME_POLICY_SHA256 as P3_JUDGE_RUNTIME_POLICY_SHA256,
    JUDGMENT_SCHEMA as P3_JUDGMENT_SCHEMA,
    MODEL_ID as P3_MODEL_ID,
    MODEL_REVISION as P3_MODEL_REVISION,
    PROMPT_TEMPLATE_SHA256 as P3_PROMPT_TEMPLATE_SHA256,
    QwenJudgeError,
    VISION_PROCESSOR_CONFIG_SHA256 as P3_VISION_PROCESSOR_CONFIG_SHA256,
    validate_extraction_row as _p3_validate_extraction_row,
)
from tools.verify_stageb_fixed_gdino_top1_vlm_results import (
    DECISION_SCHEMA as P3_DECISION_SCHEMA,
    INHERIT_CONFIDENCE_THRESHOLD as P3_INHERIT_CONFIDENCE_THRESHOLD,
    INHERIT_IOU_THRESHOLD as P3_INHERIT_IOU_THRESHOLD,
    QWEN_CONFIDENCE_THRESHOLD as P3_QWEN_CONFIDENCE_THRESHOLD,
    VerificationError as P3VerificationError,
    _SourceLines as _P3SourceLines,
    _accepted_pair as _p3_accepted_pair,
    _claims as _p3_validate_claims,
    _identity as _p3_identity,
    _load_judgments as _p3_load_judgments,
    _region_decision as _p3_region_decision,
    _source_pair as _p3_source_pair,
    _validate_extraction_audit as _p3_validate_extraction_audit,
    _validate_provenance as _p3_validate_provenance,
    _validate_region_union as _p3_validate_region_union,
    _validate_source_lineage as _p3_validate_source_lineage,
)
SCHEMA = "stageb-gdino-adapter-fixed-top1-confidence-probe-v1"
PARTITION_SCHEMA = "stageb-gdino-fixed-top1-image-partition-v1"
SELECTION_SCHEMA = "stageb-gdino-fixed-top1-milestone-selection-v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_SCHEMA = "stage-b-gdino-adapter-fixed-top1-verified-pair-v1"
CONFIG = "config/ablations/cfg_stageb_gdino_score_adapter_fixed_top1_verified.py"
DATASETS = "config/datasets_stageb_gdino_adapter_fixed_top1_verified_pairs.json"
DATA_AUDIT = "data/ablations/stageb_gdino_adapter_fixed_top1_verified_20260712/verification_audit.json"
PARTITION_AUDIT = "data/ablations/stageb_gdino_adapter_fixed_top1_verified_20260712/partition_audit.json"
SELECTION_AUDIT_NAME = "selection/selected_milestone.json"
RESULT_AUDIT_SCHEMA = "stage-b-fixed-gdino-top1-vlm-results-audit-v1"
RESULT_AUDIT_KIND = "completed_fixed_gdino_top1_vlm_results_verification"
EXPECTED_EXTRACTION_ROWS = 17_738
P3_LOCKED_CONTRACT = {
    "extraction_schema": P3_EXTRACTION_SCHEMA,
    "judgment_schema": P3_JUDGMENT_SCHEMA,
    "model_id": P3_MODEL_ID,
    "model_revision": P3_MODEL_REVISION,
    "prompt_template_sha256": P3_PROMPT_TEMPLATE_SHA256,
    "asset_policy_sha256": P3_ASSET_POLICY_SHA256,
    "generation_config_sha256": P3_GENERATION_CONFIG_SHA256,
    "vision_processor_config_sha256": P3_VISION_PROCESSOR_CONFIG_SHA256,
    "inference_batch_size": P3_INFERENCE_BATCH_SIZE,
    "judge_runtime_policy": P3_JUDGE_RUNTIME_POLICY,
    "judge_runtime_policy_sha256": P3_JUDGE_RUNTIME_POLICY_SHA256,
    "source_inherit_iou_threshold": P3_INHERIT_IOU_THRESHOLD,
    "source_inherit_confidence_threshold": P3_INHERIT_CONFIDENCE_THRESHOLD,
    "qwen_no_confidence_threshold": P3_QWEN_CONFIDENCE_THRESHOLD,
}
FIXED_MAX_SCOPE = (
    "checkpoint_train_and_deploy_transform_specific_top1_union_verified"
)
STRICT2031_SHA256 = "0e47763c01178d63ee22430a6c93d4fc6a210848d43f32aafbb2e6cd7243e918"
STRICT1607_SHA256 = "f2dc97d58884b8de3ae2c8b4cefd281432e15c8952b23b5e0252eb8e5be36d25"
MODE = "confidence_only"
MODE_CODE = 2
SCOPE = "image_global_topk_verified"
SCOPE_CODE = 1
LEARNING_RATE = 3.0e-4
FIXED_TOP1_MILESTONES = (50, 100, 250, 500, 1000)


def verify_partition(*args, **kwargs):
    from tools.stageb_gdino_fixed_top1_selection import verify_partition as implementation

    return implementation(*args, **kwargs)


def verify_selection(*args, **kwargs):
    from tools.stageb_gdino_fixed_top1_selection import verify_selection as implementation

    return implementation(*args, **kwargs)
OBJECTIVE_CONTRACT = {
    "mode": "detached_recent_q05_trust",
    "mode_code": 2,
    "threshold_gradient": "zero_value_global_positive_gate_mean_translation_proxy",
    "positive_trust_loss": "mean_relu(-margin-positive_gate)",
    "positive_trust_margin": 0.02,
    "positive_trust_weight": 1.0,
    "queue_size": 512,
    "queue_min_count": 256,
    "paired_margin_weight": 0.25,
    "paired_margin": 0.05,
    "gate_pool_temperature": 0.01,
    "gate_score_topk": 3,
}
TRAIN_CODE_ENTRIES = (
    "main.py",
    "engine.py",
    "datasets/__init__.py",
    "datasets/patch_episode.py",
    "models/__init__.py",
    "models/GroundingDINO/groundingdino.py",
    "models/GroundingDINO/stage_b_gdino_score_adapter.py",
)
TRAIN_CODE_INCLUDE = (
    "tools/verify_stageb_fixed_gdino_top1_vlm_results.py",
    "tools/stageb_gdino_fixed_top1_probe_audit.py",
    "tools/stageb_gdino_fixed_top1_selection.py",
    "tools/eval_stageb_gdino_fixed_top1_calibration.py",
    "tools/stageb_dependency_audit.py",
)
ORCHESTRATION_PATHS = (
    "tools/run_stageb_gdino_fixed_top1_confidence_probe.sh",
    "tools/run_stageb_gdino_fixed_top1_calibration.sh",
)


class FixedTop1ProbeError(ProbeAuditError):
    pass


def _load_cfg(path: Path):
    from util.slconfig import SLConfig

    return SLConfig.fromfile(str(path))


def _matches(observed: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return isinstance(observed, (int, float)) and math.isclose(
            float(observed), expected, rel_tol=0.0, abs_tol=1e-12
        )
    return observed == expected


def _resolve_fixed_top1_find_unused_params(cfg: Any) -> bool:
    """Apply main.py's CLI default without redeclaring an argparse-owned key."""
    configured = getattr(cfg, "find_unused_params", None)
    if configured is None:
        return False
    if configured is not False:
        raise FixedTop1ProbeError(
            "semantic config mismatch for find_unused_params: expected the "
            "main argparse default False (or explicit False), got "
            f"{configured!r}"
        )
    return False


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _iter_jsonl(path: Path) -> Iterable[tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise FixedTop1ProbeError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise FixedTop1ProbeError(
                    f"expected a JSON object at {path}:{line_number}"
                )
            yield line_number, value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _audited_jsonl_record(
    value: Any,
    *,
    label: str,
    expected_path: Path | None = None,
    expected_sha256: str | None = None,
    expected_rows: int | None = None,
) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or not value.get("path"):
        raise FixedTop1ProbeError(f"verification audit has no {label} file record")
    path = resolve_path(str(value["path"]))
    if expected_path is not None and path != expected_path.resolve():
        raise FixedTop1ProbeError(
            f"verification audit {label} path mismatch: {path} != {expected_path}"
        )
    current = file_record(path)
    for key in ("path", "size_bytes", "sha256"):
        if value.get(key) != current.get(key):
            raise FixedTop1ProbeError(
                f"verification audit {label} {key} drifted"
            )
    rows = count_nonempty_lines(path)
    if type(value.get("rows")) is not int or int(value["rows"]) != rows:
        raise FixedTop1ProbeError(
            f"verification audit {label} row count drifted: "
            f"audited={value.get('rows')!r}, current={rows}"
        )
    if expected_rows is not None and rows != expected_rows:
        raise FixedTop1ProbeError(
            f"verification audit {label} rows mismatch: {rows} != {expected_rows}"
        )
    if expected_sha256 is not None and current["sha256"] != expected_sha256:
        raise FixedTop1ProbeError(
            f"verification audit {label} hash mismatch"
        )
    current["rows"] = rows
    return current


def _audited_file_record(value: Any, *, label: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or not value.get("path"):
        raise FixedTop1ProbeError(f"verification audit has no {label} file record")
    current = file_record(resolve_path(str(value["path"])))
    for key in ("path", "size_bytes", "sha256"):
        if value.get(key) != current.get(key):
            raise FixedTop1ProbeError(f"verification audit {label} {key} drifted")
    return current


def _bbox_xywh(value: Any, *, context: str) -> list[float]:
    if not (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 4
    ):
        raise FixedTop1ProbeError(f"{context} has no xywh[4] target box")
    try:
        box = [float(item) for item in value]
    except (TypeError, ValueError) as error:
        raise FixedTop1ProbeError(f"{context} target box is not numeric") from error
    if any(not math.isfinite(item) for item in box) or box[2] <= 0.0 or box[3] <= 0.0:
        raise FixedTop1ProbeError(f"{context} target box is invalid")
    return box


def _replay_p3_verifier_outputs(
    *,
    extraction_path: Path,
    judgment_path: Path,
    checkpoint_sha256: str,
    train_transform_contract_sha256: str,
    deploy_transform_contract_sha256: str,
    extraction_contract: Mapping[str, Any],
    strict_records: Mapping[str, Mapping[str, Any]],
    strict_images: Mapping[str, set[int]],
) -> Dict[str, Any]:
    """Rebuild every P3 output row with the verifier's authoritative logic."""
    try:
        judgments = _p3_load_judgments(judgment_path)
    except (P3VerificationError, QwenJudgeError, OSError, ValueError) as error:
        raise FixedTop1ProbeError(f"P3 judgment replay failed: {error}") from error

    source_lines = _P3SourceLines()
    expected_outputs: Dict[str, list[Dict[str, Any]]] = {
        "accepted": [],
        "rejected": [],
        "quarantine": [],
    }
    extraction_by_sample: Dict[str, Dict[str, Any]] = {}
    extraction_transform_rows: list[list[str]] = []
    used_judgments: set[tuple[str, str]] = set()
    checkpoint_hashes: set[str] = set()
    train_contract_hashes: set[str] = set()
    deploy_contract_hashes: set[str] = set()
    row_reason_counts: Counter[str] = Counter()
    region_reason_counts: Counter[str] = Counter()
    region_method_counts: Counter[str] = Counter()
    comparison_counts: Counter[str] = Counter()
    total_regions = 0

    for line_number, extraction in _iter_jsonl(extraction_path):
        context = f"extraction row {line_number}"
        try:
            _p3_validate_extraction_row(extraction)
            _p3_validate_claims(extraction)
            _p3_validate_region_union(extraction)
            identity = _p3_identity(extraction)
            sample_id = str(identity["sample_id"])
            if sample_id in extraction_by_sample:
                raise FixedTop1ProbeError(
                    f"duplicate extraction sample_id {sample_id}"
                )

            for name in ("strict2031", "strict1607"):
                if int(identity["image_id"]) in strict_images[name]:
                    raise FixedTop1ProbeError(
                        f"{context} overlaps {name} on image {identity['image_id']}"
                    )
            holdout = extraction.get("holdout")
            if (
                not isinstance(holdout, Mapping)
                or holdout.get("image_disjoint") is not True
                or holdout.get("strict2031_manifest_sha256")
                != strict_records["strict2031"]["sha256"]
                or holdout.get("strict1607_manifest_sha256")
                != strict_records["strict1607"]["sha256"]
            ):
                raise FixedTop1ProbeError(f"{context} holdout provenance drifted")

            provenance = _p3_validate_provenance(extraction)
            if (
                provenance["checkpoint_sha256"]
                != extraction_contract["checkpoint_sha256"]
                or provenance["train_transform_contract_sha256"]
                != extraction_contract["train_transform_contract_sha256"]
                or provenance["deploy_transform_contract_sha256"]
                != extraction_contract["deploy_transform_contract_sha256"]
            ):
                raise FixedTop1ProbeError(
                    f"{context} checkpoint/transform binding drifted from extraction audit"
                )
            if extraction.get("checkpoint") != extraction_contract["checkpoint"]:
                raise FixedTop1ProbeError(
                    f"{context} checkpoint drifted from extraction audit"
                )
            if extraction.get("config") != extraction_contract["model_config"]:
                raise FixedTop1ProbeError(
                    f"{context} model config drifted from extraction audit"
                )
            if extraction.get("data_config") != extraction_contract["data_config"]:
                raise FixedTop1ProbeError(
                    f"{context} data config drifted from extraction audit"
                )
            if extraction.get("code_sha256") != extraction_contract["code_sha256"]:
                raise FixedTop1ProbeError(
                    f"{context} code closure drifted from extraction audit"
                )
            checkpoint_hashes.add(str(provenance["checkpoint_sha256"]))
            train_contract_hashes.add(
                str(provenance["train_transform_contract_sha256"])
            )
            deploy_contract_hashes.add(
                str(provenance["deploy_transform_contract_sha256"])
            )
            extraction_transform_rows.append(
                [
                    sample_id,
                    str(provenance["train_transform_row_sha256"]),
                    str(provenance["deploy_transform_row_sha256"]),
                ]
            )
            extraction_by_sample[sample_id] = {
                "row": extraction,
                "sha256": _canonical_sha256(extraction),
            }

            source_pair = _p3_source_pair(extraction, source_lines)
            _p3_validate_source_lineage(extraction, source_pair, source_lines)
            region_decisions = []
            for region in extraction["regions"]:
                total_regions += 1
                judgment_key = (sample_id, str(region["region_id"]))
                judgment = judgments.get(judgment_key)
                if judgment is not None:
                    used_judgments.add(judgment_key)
                decision = _p3_region_decision(extraction, region, judgment)
                region_decisions.append(decision)
                region_reason_counts[str(decision["reason"])] += 1
                region_method_counts[str(decision["method"])] += 1
                comparison = decision.get("source_qwen_comparison")
                if comparison is not None:
                    comparison_counts[str(comparison)] += 1

            if any(item["decision"] == "rejected" for item in region_decisions):
                row_decision = "rejected"
                reason = next(
                    item["reason"]
                    for item in region_decisions
                    if item["decision"] == "rejected"
                )
            elif any(
                item["decision"] == "quarantine" for item in region_decisions
            ):
                row_decision = "quarantine"
                reason = next(
                    item["reason"]
                    for item in region_decisions
                    if item["decision"] == "quarantine"
                )
            else:
                row_decision = "accepted"
                reason = "all_union_regions_verified_no"
            row_reason_counts[str(reason)] += 1

            if row_decision == "accepted":
                expected_outputs["accepted"].append(
                    _p3_accepted_pair(
                        source_pair, extraction, region_decisions, provenance
                    )
                )
            else:
                expected_outputs[row_decision].append(
                    {
                        "schema": P3_DECISION_SCHEMA,
                        "identity": identity,
                        "decision": row_decision,
                        "reason": reason,
                        "region_decisions": region_decisions,
                        "extraction_row_sha256": _canonical_sha256(extraction),
                        "checkpoint_sha256": provenance["checkpoint_sha256"],
                        "train_transform_contract_sha256": provenance[
                            "train_transform_contract_sha256"
                        ],
                        "deploy_transform_contract_sha256": provenance[
                            "deploy_transform_contract_sha256"
                        ],
                        "train_transform_row_sha256": provenance[
                            "train_transform_row_sha256"
                        ],
                        "deploy_transform_row_sha256": provenance[
                            "deploy_transform_row_sha256"
                        ],
                    }
                )
        except FixedTop1ProbeError:
            raise
        except (
            P3VerificationError,
            QwenJudgeError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise FixedTop1ProbeError(f"{context} P3 replay failed: {error}") from error

    if len(extraction_by_sample) != EXPECTED_EXTRACTION_ROWS:
        raise FixedTop1ProbeError(
            "P3 extraction replay row count mismatch: "
            f"{len(extraction_by_sample)} != {EXPECTED_EXTRACTION_ROWS}"
        )
    unused_judgments = set(judgments) - used_judgments
    if unused_judgments:
        raise FixedTop1ProbeError(
            f"P3 replay found {len(unused_judgments)} unused judgment rows"
        )
    expected_singletons = (
        ("checkpoint", checkpoint_hashes, checkpoint_sha256),
        ("train transform contract", train_contract_hashes, train_transform_contract_sha256),
        ("deploy transform contract", deploy_contract_hashes, deploy_transform_contract_sha256),
    )
    for label, values, expected in expected_singletons:
        if values != {expected}:
            raise FixedTop1ProbeError(
                f"P3 replay {label} binding drifted: {sorted(values)}"
            )
    extraction_transform_rows.sort(key=lambda item: tuple(item))
    return {
        "outputs": expected_outputs,
        "extraction_by_sample": extraction_by_sample,
        "extraction_transform_rows": extraction_transform_rows,
        "regions": total_regions,
        "row_reason_counts": dict(sorted(row_reason_counts.items())),
        "region_reason_counts": dict(sorted(region_reason_counts.items())),
        "region_method_counts": dict(sorted(region_method_counts.items())),
        "source_qwen_comparison_counts": dict(sorted(comparison_counts.items())),
    }


def validate_verified_pairs(
    annotation: Path,
    data_audit_path: Path,
) -> Dict[str, Any]:
    annotation = annotation.resolve()
    data_audit_path = data_audit_path.resolve()
    data_audit = read_json(data_audit_path)
    if (
        data_audit.get("schema") != RESULT_AUDIT_SCHEMA
        or data_audit.get("kind") != RESULT_AUDIT_KIND
    ):
        raise FixedTop1ProbeError(
            "fixed-top1 verification audit schema/kind is invalid"
        )
    if data_audit.get("locked_contract") != P3_LOCKED_CONTRACT:
        raise FixedTop1ProbeError("fixed-top1 P3 locked contract drifted")

    inputs = data_audit.get("inputs")
    outputs = data_audit.get("outputs")
    if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
        raise FixedTop1ProbeError("fixed-top1 verification audit has no input/output records")
    input_records = {
        "extractions": _audited_jsonl_record(
            inputs.get("extractions"),
            label="input extractions",
            expected_rows=EXPECTED_EXTRACTION_ROWS,
        ),
        "judgments": _audited_jsonl_record(
            inputs.get("judgments"), label="input judgments"
        ),
    }
    input_records["strict2031"] = _audited_jsonl_record(
        inputs.get("strict2031"),
        label="input strict2031",
        expected_sha256=STRICT2031_SHA256,
        expected_rows=2031,
    )
    input_records["strict1607"] = _audited_jsonl_record(
        inputs.get("strict1607"),
        label="input strict1607",
        expected_sha256=STRICT1607_SHA256,
        expected_rows=1607,
    )
    extraction_audit_value = inputs.get("extraction_audit")
    extraction_audit_record = _audited_file_record(
        extraction_audit_value, label="input extraction_audit"
    )
    if (
        not isinstance(extraction_audit_value, Mapping)
        or extraction_audit_value.get("schema")
        != "stage-b-fixed-gdino-top1-vlm-extraction-audit-v1"
        or extraction_audit_value.get("kind")
        != "completed_fixed_gdino_top1_vlm_extraction"
        or int(extraction_audit_value.get("rows", -1))
        != EXPECTED_EXTRACTION_ROWS
    ):
        raise FixedTop1ProbeError("extraction completion audit contract is invalid")
    extraction_completion = read_json(Path(extraction_audit_record["path"]))
    if (
        extraction_completion.get("schema")
        != "stage-b-fixed-gdino-top1-vlm-extraction-audit-v1"
        or extraction_completion.get("kind")
        != "completed_fixed_gdino_top1_vlm_extraction"
        or int(extraction_completion.get("rows", -1))
        != EXPECTED_EXTRACTION_ROWS
    ):
        raise FixedTop1ProbeError(
            "current extraction completion audit is not the exact 17738-row P3 run"
        )
    try:
        extraction_contract = _p3_validate_extraction_audit(
            Path(extraction_audit_record["path"]),
            extraction_path=Path(input_records["extractions"]["path"]),
            extraction_record=input_records["extractions"],
            strict2031_record=input_records["strict2031"],
            strict1607_record=input_records["strict1607"],
        )
    except (P3VerificationError, OSError, KeyError, TypeError, ValueError) as error:
        raise FixedTop1ProbeError(
            f"P3 extraction completion audit replay failed: {error}"
        ) from error
    extraction_audit_record["rows"] = EXPECTED_EXTRACTION_ROWS
    input_records["extraction_audit"] = extraction_audit_record
    output_records = {
        name: _audited_jsonl_record(
            outputs.get(name),
            label=f"output {name}",
            expected_path=annotation if name == "accepted" else None,
        )
        for name in ("accepted", "rejected", "quarantine")
    }
    accepted_rows = int(output_records["accepted"]["rows"])
    if accepted_rows <= 0:
        raise FixedTop1ProbeError("fixed-top1 accepted pair file is empty")
    decisions = data_audit.get("decisions")
    if not isinstance(decisions, Mapping) or any(
        type(decisions.get(name)) is not int
        or int(decisions[name]) != int(output_records[name]["rows"])
        for name in ("accepted", "rejected", "quarantine")
    ):
        raise FixedTop1ProbeError("fixed-top1 decision counts do not match outputs")
    if int(data_audit.get("rows", -1)) != sum(
        int(output_records[name]["rows"])
        for name in ("accepted", "rejected", "quarantine")
    ):
        raise FixedTop1ProbeError("fixed-top1 audit row partition is incomplete")
    if data_audit.get("strict_image_overlap") != {
        "strict2031": 0,
        "strict1607": 0,
    }:
        raise FixedTop1ProbeError("fixed-top1 audit lost strict image disjointness")

    checkpoint_sha = data_audit.get("checkpoint_sha256")
    train_transform_sha = data_audit.get("train_transform_contract_sha256")
    deploy_transform_sha = data_audit.get("deploy_transform_contract_sha256")
    transform_rows_sha = data_audit.get("transform_rows_sha256")
    extraction_transform_rows_sha = data_audit.get(
        "extraction_transform_rows_sha256"
    )
    for label, value in (
        ("checkpoint", checkpoint_sha),
        ("train transform contract", train_transform_sha),
        ("deploy transform contract", deploy_transform_sha),
        ("ordered transform rows", transform_rows_sha),
        ("ordered extraction transform rows", extraction_transform_rows_sha),
    ):
        if not _is_sha256(value):
            raise FixedTop1ProbeError(f"fixed-top1 {label} hash is malformed")
    if data_audit.get("transform_sha256") != train_transform_sha:
        raise FixedTop1ProbeError("fixed-top1 train transform compatibility hash drifted")
    expected_hash_contract = {
        "schema": "stage-b-transform-row-hash-list-v1",
        "payload": "[[sample_id,train_row_sha256,deploy_row_sha256],...]",
        "ordering": "lexicographic_by_all_three_string_fields",
        "canonicalization": (
            "json.ensure_ascii=true,sort_keys=true,separators=(',',':'),"
            "allow_nan=false;sha256(utf8)"
        ),
        "transform_rows_scope": "accepted_output_rows",
        "extraction_transform_rows_scope": "all_extraction_rows",
    }
    if data_audit.get("transform_rows_hash_contract") != expected_hash_contract:
        raise FixedTop1ProbeError("transform-row hash contract drifted")
    scope = data_audit.get("scope")
    expected_scope = {
        "tn_scope_compatibility": SCOPE,
        "fixed_gdino_max_scope": FIXED_MAX_SCOPE,
        "global_max_label_is_semantic_extrapolation": False,
        "all_900_gdino_queries_verified": False,
        "image_global_semantic_absence_proven": False,
        "portable_to_other_checkpoint_or_transform": False,
    }
    if not isinstance(scope, Mapping) or any(
        scope.get(key) != expected for key, expected in expected_scope.items()
    ):
        raise FixedTop1ProbeError("fixed-top1 audit scope claims are malformed")

    strict_images: Dict[str, set[int]] = {}
    for name in ("strict2031", "strict1607"):
        images = set()
        for line_number, row in _iter_jsonl(Path(input_records[name]["path"])):
            try:
                images.add(int(row["image_id"]))
            except (KeyError, TypeError, ValueError) as error:
                raise FixedTop1ProbeError(
                    f"invalid image_id in {name} row {line_number}"
                ) from error
        strict_images[name] = images

    replay = _replay_p3_verifier_outputs(
        extraction_path=Path(input_records["extractions"]["path"]),
        judgment_path=Path(input_records["judgments"]["path"]),
        checkpoint_sha256=str(checkpoint_sha),
        train_transform_contract_sha256=str(train_transform_sha),
        deploy_transform_contract_sha256=str(deploy_transform_sha),
        extraction_contract=extraction_contract,
        strict_records={
            "strict2031": input_records["strict2031"],
            "strict1607": input_records["strict1607"],
        },
        strict_images=strict_images,
    )
    actual_outputs = {
        name: [
            row
            for _line_number, row in _iter_jsonl(Path(output_records[name]["path"]))
        ]
        for name in ("accepted", "rejected", "quarantine")
    }
    for line_number, row in enumerate(actual_outputs["accepted"], start=1):
        sample_id = row.get("sample_id")
        extraction_entry = replay["extraction_by_sample"].get(sample_id)
        if extraction_entry is None:
            raise FixedTop1ProbeError(
                f"accepted row {line_number} has no matching extraction sample_id"
            )
        if row.get("fixed_gdino_extraction_row_sha256") != extraction_entry[
            "sha256"
        ]:
            raise FixedTop1ProbeError(
                f"accepted row {line_number} extraction row hash drifted"
            )
    for name in ("accepted", "rejected", "quarantine"):
        expected_rows = replay["outputs"][name]
        if actual_outputs[name] != expected_rows:
            mismatch = next(
                (
                    index
                    for index, (actual, expected) in enumerate(
                        zip(actual_outputs[name], expected_rows), start=1
                    )
                    if actual != expected
                ),
                min(len(actual_outputs[name]), len(expected_rows)) + 1,
            )
            raise FixedTop1ProbeError(
                f"{name} output failed exact P3 verifier replay at row {mismatch}: "
                f"actual_rows={len(actual_outputs[name])}, "
                f"expected_rows={len(expected_rows)}"
            )
    replay_audit_fields = {
        "regions": replay["regions"],
        "row_reason_counts": replay["row_reason_counts"],
        "region_reason_counts": replay["region_reason_counts"],
        "region_method_counts": replay["region_method_counts"],
        "source_qwen_comparison_counts": replay[
            "source_qwen_comparison_counts"
        ],
    }
    for key, expected in replay_audit_fields.items():
        if data_audit.get(key) != expected:
            raise FixedTop1ProbeError(f"P3 replay audit field {key} drifted")

    sample_ids = set()
    accepted_images = set()
    transform_rows: list[list[str]] = []
    for line_number, row in _iter_jsonl(annotation):
        context = f"accepted row {line_number}"
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise FixedTop1ProbeError(f"{context} has no sample_id")
        if sample_id in sample_ids:
            raise FixedTop1ProbeError(f"duplicate accepted sample_id {sample_id}")
        sample_ids.add(sample_id)
        try:
            image_id = int(row["image_id"])
            int(row["class_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise FixedTop1ProbeError(f"{context} has invalid identity/class") from error
        accepted_images.add(image_id)
        _bbox_xywh(row.get("target_bbox_used"), context=context)
        for text_key in ("sent", "try_tn"):
            if not isinstance(row.get(text_key), str) or not row[text_key].strip():
                raise FixedTop1ProbeError(f"{context} has invalid {text_key}")

        expected_claims = {
            "adapter_pair_schema": DATA_SCHEMA,
            "source": "stage_b_gdino_adapter_fixed_top1_verified",
            "tn_scope": SCOPE,
            "global_tn_verified": True,
            "proposalset_proxy_verified": False,
            "visual_verified_negative": True,
            "semantic_verified_negative": True,
            "fixed_gdino_max_scope": FIXED_MAX_SCOPE,
            "fixed_gdino_global_max_verified": True,
            "fixed_gdino_global_max_verification_contract": (
                "target_plus_cached_sam3_all_no_and_all_frozen_gdino_"
                "train_primary_shadow_deploy_neartie_union_regions_no"
            ),
            "global_max_label_is_semantic_extrapolation": False,
            "all_900_gdino_queries_verified": False,
            "image_global_semantic_absence_proven": False,
            "portable_to_other_checkpoint_or_transform": False,
            "frozen_gdino_checkpoint_sha256": checkpoint_sha,
            "frozen_gdino_train_transform_contract_sha256": train_transform_sha,
            "frozen_gdino_deploy_transform_contract_sha256": deploy_transform_sha,
            "frozen_gdino_transform_sha256": train_transform_sha,
        }
        if any(row.get(key) != expected for key, expected in expected_claims.items()):
            raise FixedTop1ProbeError(f"{context} fixed-max claims drifted")
        train_row_sha = row.get("frozen_gdino_train_transform_row_sha256")
        deploy_row_sha = row.get("frozen_gdino_deploy_transform_row_sha256")
        extraction_row_sha = row.get("fixed_gdino_extraction_row_sha256")
        if not all(
            _is_sha256(value)
            for value in (train_row_sha, deploy_row_sha, extraction_row_sha)
        ):
            raise FixedTop1ProbeError(f"{context} row provenance hashes are malformed")
        transform_rows.append([sample_id, train_row_sha, deploy_row_sha])

        regions = row.get("fixed_gdino_region_verifications")
        if not isinstance(regions, list) or not regions:
            raise FixedTop1ProbeError(f"{context} has no verified fixed-max regions")
        region_ids = set()
        origins = set()
        for region in regions:
            if not isinstance(region, Mapping):
                raise FixedTop1ProbeError(f"{context} contains a non-object region")
            region_id = region.get("region_id")
            if not _is_sha256(region_id) or region_id in region_ids:
                raise FixedTop1ProbeError(f"{context} has invalid/duplicate region_id")
            region_ids.add(region_id)
            region_origins = region.get("origins")
            query_ids = region.get("query_ids")
            if not isinstance(region_origins, list) or not region_origins:
                raise FixedTop1ProbeError(f"{context} region has no origins")
            if not isinstance(query_ids, list) or not query_ids:
                raise FixedTop1ProbeError(f"{context} region has no query ids")
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < 900
                for value in query_ids
            ):
                raise FixedTop1ProbeError(f"{context} region query ids are invalid")
            origins.update(str(value) for value in region_origins)
            if region.get("method") not in {"source", "qwen"} or region.get(
                "reason"
            ) not in {"source_inherited_no", "qwen_no"}:
                raise FixedTop1ProbeError(f"{context} region is not an accepted NO")
        if not {"primary", "shadow", "deploy"}.issubset(origins):
            raise FixedTop1ProbeError(
                f"{context} does not cover train primary/shadow and deploy maxima"
            )

    for name, images in strict_images.items():
        overlap = accepted_images & images
        if overlap:
            raise FixedTop1ProbeError(
                f"accepted pairs overlap {name} on {len(overlap)} images"
            )
    if len(sample_ids) != accepted_rows:
        raise FixedTop1ProbeError("accepted row count changed while validating")
    transform_rows.sort(key=lambda item: tuple(item))
    if _canonical_sha256(transform_rows) != transform_rows_sha:
        raise FixedTop1ProbeError("ordered accepted transform-row hash drifted")

    extraction_transform_rows = replay["extraction_transform_rows"]
    if _canonical_sha256(extraction_transform_rows) != extraction_transform_rows_sha:
        raise FixedTop1ProbeError("ordered extraction transform-row hash drifted")

    annotation_record = file_record(annotation)
    annotation_record["rows"] = accepted_rows
    return {
        "data_audit": file_record(data_audit_path),
        "annotation": annotation_record,
        "inputs": input_records,
        "outputs": output_records,
        "frozen_gdino_checkpoint_sha256": checkpoint_sha,
        "train_transform_contract_sha256": train_transform_sha,
        "deploy_transform_contract_sha256": deploy_transform_sha,
        "transform_rows_sha256": transform_rows_sha,
        "extraction_transform_rows_sha256": extraction_transform_rows_sha,
        "transform_rows_hash_contract": expected_hash_contract,
        "strict_image_overlap": {"strict2031": 0, "strict1607": 0},
        "scope": dict(scope),
    }


def validate_static() -> Dict[str, Any]:
    config_path = resolve_path(CONFIG)
    datasets_path = resolve_path(DATASETS)
    data_audit_path = resolve_path(DATA_AUDIT)
    cfg = _load_cfg(config_path)
    expected_cfg = {
        "stage_b_gdino_score_adapter": True,
        "stage_b_gdino_adapter_train_mode": MODE,
        "stage_b_gdino_tn_scope": SCOPE,
        "patch_only": False,
        "stage_b": False,
        "enable_patch_branch": False,
        "stage_b_gdino_rank_weight": 0.0,
        "stage_b_gdino_confidence_weight": 1.0,
        "stage_b_gdino_confidence_objective": "detached_recent_q05_trust",
        "stage_b_gdino_paired_margin_weight": 0.25,
        "stage_b_gdino_paired_margin": 0.05,
        "stage_b_gdino_positive_trust_margin": 0.02,
        "stage_b_gdino_positive_trust_weight": 1.0,
        "stage_b_gdino_gate_lr": LEARNING_RATE,
        "stage_b_gdino_gate_pool_temperature": 0.01,
        "stage_b_gdino_gate_topk": 3,
        "stage_b_gdino_queue_size": 512,
        "stage_b_gdino_queue_min_count": 256,
        "data_aug_hflip_prob": 0.0,
        "batch_size": 4,
        "epochs": 1,
        "skip_eval": True,
    }
    for key, expected in expected_cfg.items():
        observed = getattr(cfg, key, None)
        if not _matches(observed, expected):
            raise FixedTop1ProbeError(
                f"semantic config mismatch for {key}: expected {expected!r}, "
                f"got {observed!r}"
            )
    expected_cfg["find_unused_params"] = _resolve_fixed_top1_find_unused_params(cfg)

    dataset_config = read_json(datasets_path)
    train = dataset_config.get("train")
    if not isinstance(train, list) or len(train) != 1 or dataset_config.get("val") != []:
        raise FixedTop1ProbeError("semantic dataset must have one train source and no val")
    entry = train[0]
    required_entry = {
        "dataset_mode": "patch_episode",
        "source": "sam3_tn_pair",
        "box_format": "xywh",
        "sam3_tn_bbox_key": "target_bbox_used",
        "neg_episode_prob": 0.0,
        "tn_balance_sampling": False,
        "require_global_tn_verified": True,
        "stage_b_gdino_adapter_no_support": True,
        "mix_weight": 1.0,
    }
    for key, expected in required_entry.items():
        if entry.get(key) != expected:
            raise FixedTop1ProbeError(
                f"semantic dataset mismatch for {key}: expected {expected!r}, "
                f"got {entry.get(key)!r}"
            )
    annotation = resolve_path(entry.get("anno", ""))
    data_audit = read_json(data_audit_path)
    outputs = data_audit.get("outputs")
    accepted_value = outputs.get("accepted") if isinstance(outputs, Mapping) else None
    if not isinstance(accepted_value, Mapping) or not accepted_value.get("path"):
        raise FixedTop1ProbeError("fixed-top1 verification audit has no accepted output")
    accepted_annotation = resolve_path(str(accepted_value["path"]))
    verified_pairs = validate_verified_pairs(accepted_annotation, data_audit_path)
    try:
        partition = verify_partition(
            resolve_path(PARTITION_AUDIT),
            expected_accepted=accepted_annotation,
            expected_verification_audit=data_audit_path,
            expected_train=annotation,
        )
    except RuntimeError as error:
        raise FixedTop1ProbeError(f"fixed-top1 partition audit failed: {error}") from error
    if partition["selection_readiness"].get("pass") is not True:
        raise FixedTop1ProbeError(
            "fixed-top1 partition does not meet calibration selection readiness: "
            f"{partition['selection_readiness'].get('errors')}"
        )
    from tools.extract_stageb_fixed_gdino_top1_vlm_manifest import (
        deploy_transform_contract_from_cfg,
        transform_contract_from_cfg,
    )

    current_transform_contracts = {
        "train": transform_contract_from_cfg(cfg),
        "deploy": deploy_transform_contract_from_cfg(cfg),
    }
    for name, audit_key in (
        ("train", "train_transform_contract_sha256"),
        ("deploy", "deploy_transform_contract_sha256"),
    ):
        if (
            current_transform_contracts[name].get("sha256")
            != verified_pairs[audit_key]
        ):
            raise FixedTop1ProbeError(
                f"fixed-top1 {name} transform contract does not match the "
                "current training/evaluation config"
            )

    try:
        config_paths = config_import_chain(config_path, root=REPO_ROOT)
        code_paths = local_python_dependency_paths(
            TRAIN_CODE_ENTRIES,
            root=REPO_ROOT,
            include=TRAIN_CODE_INCLUDE,
        )
    except DependencyAuditError as error:
        raise FixedTop1ProbeError(str(error)) from error
    if not config_paths or config_path not in config_paths:
        raise FixedTop1ProbeError("semantic config import chain is incomplete")

    return {
        "config": file_record(config_path),
        "config_import_chain": [file_record(path) for path in config_paths],
        "datasets": file_record(datasets_path),
        "data_audit": verified_pairs["data_audit"],
        "source_annotation": verified_pairs["annotation"],
        "annotation": partition["train"],
        "partition": {
            "schema": PARTITION_SCHEMA,
            "audit": partition["audit"],
            "accepted": partition["accepted"],
            "train": partition["train"],
            "calibration": partition["calibration"],
            "recommended_max_target": partition["recommended_max_target"],
            "selection_readiness": partition["selection_readiness"],
        },
        "verified_pair_contract": {
            key: verified_pairs[key]
            for key in (
                "inputs",
                "outputs",
                "frozen_gdino_checkpoint_sha256",
                "train_transform_contract_sha256",
                "deploy_transform_contract_sha256",
                "transform_rows_sha256",
                "extraction_transform_rows_sha256",
                "transform_rows_hash_contract",
                "strict_image_overlap",
                "scope",
            )
        },
        "tn_scope": SCOPE,
        "resolved_contract": expected_cfg,
        "objective_contract": dict(OBJECTIVE_CONTRACT),
        "current_transform_contracts": current_transform_contracts,
        "code": [file_record(path) for path in code_paths],
        "orchestration": [
            file_record(resolve_path(path)) for path in ORCHESTRATION_PATHS
        ],
    }


def _require_audited_hash(value: Any, *, label: str) -> Path:
    if not isinstance(value, Mapping) or not value.get("path") or not value.get("sha256"):
        raise FixedTop1ProbeError(f"missing path/hash for {label}")
    path = resolve_path(str(value["path"]))
    current = file_record(path)
    if current["sha256"] != value.get("sha256"):
        raise FixedTop1ProbeError(f"hash drift for {label}: {path}")
    return path


def _validate_source(
    source_kind: str, checkpoint_path: Path, audit_path: Path
) -> Dict[str, Any]:
    if source_kind != "rank":
        raise FixedTop1ProbeError(
            "fixed-top1 confidence must initialize from an audited rank milestone"
        )
    audit = read_json(audit_path)
    if (
        audit.get("schema") != TWO_PHASE_SCHEMA
        or audit.get("kind") != "milestone_checkpoint"
        or audit.get("phase") != "rank"
        or int(audit.get("iteration", -1)) not in RANK_MILESTONES
    ):
        raise FixedTop1ProbeError(
            "source audit is not an accepted rank milestone"
        )
    try:
        verified = _verify_two_phase_milestone_checkpoint(
            checkpoint_path.resolve(), audit_path.resolve()
        )
    except ProbeAuditError as error:
        raise FixedTop1ProbeError(
            f"rank source failed complete lineage replay: {error}"
        ) from error
    if verified.get("phase") != "rank" or int(verified.get("iteration", -1)) not in RANK_MILESTONES:
        raise FixedTop1ProbeError("deep source replay did not return a rank milestone")
    record = dict(verified["checkpoint"])
    if int(record.get("adapter_state_keys", 0)) <= 0:
        raise FixedTop1ProbeError("fixed-top1 phase requires a complete adapter checkpoint")
    if record.get("rank_final_zero") is True:
        raise FixedTop1ProbeError("rank source has an untrained rank output")
    return record


def _rank_source_baseline_sha256(audit_path: Path) -> str:
    audit = read_json(audit_path)
    preflight_value = audit.get("preflight")
    if not isinstance(preflight_value, Mapping) or not preflight_value.get("path"):
        raise FixedTop1ProbeError("rank source audit has no preflight record")
    preflight_path = resolve_path(str(preflight_value["path"]))
    if file_record(preflight_path) != dict(preflight_value):
        raise FixedTop1ProbeError("rank source preflight file drifted")
    preflight = read_json(preflight_path)
    baseline = preflight.get("initial_checkpoint")
    sha256 = baseline.get("sha256") if isinstance(baseline, Mapping) else None
    if not _is_sha256(sha256):
        raise FixedTop1ProbeError("rank source preflight lost its fixed baseline hash")
    return str(sha256)


def make_preflight(
    *,
    source_kind: str,
    source_checkpoint: Path,
    source_audit: Path,
    world_size: int,
    per_gpu_batch: int,
    max_target: int = 500,
) -> Dict[str, Any]:
    if world_size != 2 or per_gpu_batch != 4:
        raise FixedTop1ProbeError(
            "semantic probe requires two DDP ranks and per-GPU batch 4"
        )
    selected_milestones = _selected_milestones(max_target)
    initial_checkpoint = _validate_source(
        source_kind, source_checkpoint, source_audit
    )
    static = validate_static()
    required_rows = _validate_target_readiness(
        static,
        max_target=max_target,
        world_size=world_size,
        per_gpu_batch=per_gpu_batch,
    )
    baseline_sha256 = _rank_source_baseline_sha256(source_audit)
    fixed_checkpoint_sha256 = static["verified_pair_contract"][
        "frozen_gdino_checkpoint_sha256"
    ]
    if baseline_sha256 != fixed_checkpoint_sha256:
        raise FixedTop1ProbeError(
            "verified fixed-max pairs were extracted from a different baseline "
            "checkpoint than the selected rank lineage"
        )
    return {
        "schema": SCHEMA,
        "kind": "phase_preflight",
        "phase": "fixed-top1-confidence",
        "initialization": (
            f"audited_{source_kind}_to_fixed_top1_confidence_pretrain_model_path"
        ),
        "source_kind": source_kind,
        "initial_checkpoint": initial_checkpoint,
        "initial_audit": file_record(source_audit),
        "fixed_gdino_source_binding": {
            "checkpoint_sha256": baseline_sha256,
            "matches_rank_initial_baseline": True,
        },
        "static": static,
        "launch": {
            "world_size": world_size,
            "per_gpu_batch": per_gpu_batch,
            "global_batch": world_size * per_gpu_batch,
            "max_target": max_target,
            "milestones": list(selected_milestones),
            "minimum_accepted_rows": required_rows,
            "first_segment_initialization": "pretrain_model_path",
            "same_scope_continuation": "resume",
            "cross_scope_resume_forbidden": True,
        },
    }


def _selected_milestones(max_target: int) -> tuple[int, ...]:
    max_target = int(max_target)
    if max_target not in FIXED_TOP1_MILESTONES:
        raise FixedTop1ProbeError(
            f"max target must be one of {FIXED_TOP1_MILESTONES}"
        )
    return tuple(target for target in FIXED_TOP1_MILESTONES if target <= max_target)


def _validate_target_readiness(
    static: Mapping[str, Any],
    *,
    max_target: int,
    world_size: int = 2,
    per_gpu_batch: int = 4,
) -> int:
    _selected_milestones(max_target)
    required_rows = int(world_size) * int(per_gpu_batch) * int(max_target)
    annotation = static.get("annotation")
    train_rows = int(annotation.get("rows", -1)) if isinstance(annotation, Mapping) else -1
    partition = static.get("partition")
    if not isinstance(partition, Mapping):
        raise FixedTop1ProbeError("fixed-top1 static audit has no image partition")
    readiness = partition.get("selection_readiness")
    if not isinstance(readiness, Mapping) or readiness.get("pass") is not True:
        raise FixedTop1ProbeError("fixed-top1 calibration partition is not selection-ready")
    recommended = partition.get("recommended_max_target")
    if type(recommended) is not int or int(recommended) != int(max_target):
        raise FixedTop1ProbeError(
            "fixed-top1 max target must equal the partition's pre-registered "
            f"largest supported milestone: requested={max_target}, recommended={recommended}"
        )
    if train_rows < required_rows:
        raise FixedTop1ProbeError(
            "fixed-top1 train partition cannot reach the selected final milestone "
            f"within one epoch: rows={train_rows}, required>={required_rows} "
            f"for max_target={max_target}"
        )
    return required_rows


def _checkpoint_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return resolve_path(str(value))


def _scalar(state: Mapping[str, Any], key: str) -> int:
    value = state.get(key)
    if not torch.is_tensor(value) or value.numel() != 1:
        raise FixedTop1ProbeError(f"criterion state is missing scalar {key}")
    return int(value.detach().reshape(-1)[0].item())


def _scalar_float(state: Mapping[str, Any], key: str) -> float:
    value = state.get(key)
    if not torch.is_tensor(value) or value.numel() != 1 or not value.is_floating_point():
        raise FixedTop1ProbeError(f"criterion state is missing float scalar {key}")
    return float(value.detach().reshape(-1)[0].item())


def _validate_queue(criterion: Mapping[str, Any], *, require_warm: bool) -> Dict[str, int]:
    positive = criterion.get("fpr_positive_queue")
    negative = criterion.get("fpr_negative_queue")
    count_value = criterion.get("fpr_queue_count")
    pointer_value = criterion.get("fpr_queue_ptr")
    if (
        not torch.is_tensor(positive)
        or not torch.is_tensor(negative)
        or tuple(positive.shape) != (512,)
        or tuple(negative.shape) != (512,)
        or not positive.is_floating_point()
        or not negative.is_floating_point()
        or not torch.is_tensor(count_value)
        or count_value.numel() != 1
        or count_value.dtype != torch.int64
        or not torch.is_tensor(pointer_value)
        or pointer_value.numel() != 1
        or pointer_value.dtype != torch.int64
    ):
        raise FixedTop1ProbeError("semantic criterion queue state is malformed")
    count = int(count_value.reshape(-1)[0].item())
    pointer = int(pointer_value.reshape(-1)[0].item())
    if not 0 <= count <= 512 or not 0 <= pointer < 512:
        raise FixedTop1ProbeError("semantic criterion queue counter/pointer is invalid")
    if count < 512 and pointer != count:
        raise FixedTop1ProbeError(
            "semantic criterion partial queue pointer must equal its active count"
        )
    for label, queue in (("positive", positive), ("negative", negative)):
        active = queue.detach().reshape(-1)[:count]
        if active.numel() and not bool(torch.isfinite(active).all().item()):
            raise FixedTop1ProbeError(
                f"semantic criterion {label} queue has non-finite active entries"
            )
    if require_warm and count < 256:
        raise FixedTop1ProbeError(
            f"semantic q05 queue did not reach its 256-example warmup: {count}"
        )
    return {
        "count": count,
        "pointer": pointer,
        "capacity": 512,
        "minimum_for_q05": 256,
    }


def validate_checkpoint(
    *,
    checkpoint: Path,
    preflight: Mapping[str, Any],
    expected_target: int,
    source_checkpoint: Path,
    exact: bool,
) -> Dict[str, Any]:
    if expected_target not in FIXED_TOP1_MILESTONES:
        raise FixedTop1ProbeError(
            f"target must be one of {FIXED_TOP1_MILESTONES}"
        )
    launch = preflight.get("launch")
    selected = launch.get("milestones") if isinstance(launch, Mapping) else None
    if not isinstance(selected, list) or expected_target not in selected:
        raise FixedTop1ProbeError(
            f"target {expected_target} is outside the preflight milestone prefix"
        )
    payload = load_checkpoint(checkpoint)
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise FixedTop1ProbeError("semantic checkpoint has no model state")
    iteration = int(payload.get("iteration", 0) or 0)
    if exact:
        if iteration != expected_target or payload.get("checkpoint_reason") != "max_train_iters":
            raise FixedTop1ProbeError(
                f"semantic milestone expected max_train_iters at {expected_target}, "
                f"got iteration={iteration}, reason={payload.get('checkpoint_reason')!r}"
            )
    elif not 0 < iteration <= expected_target:
        raise FixedTop1ProbeError(
            f"semantic live iteration must be in [1,{expected_target}], got {iteration}"
        )
    if int(payload.get("epoch", -1)) != 0 or payload.get("epoch_finished") is not False:
        raise FixedTop1ProbeError("semantic probe must remain a mid-epoch epoch=0 run")
    for key in (
        "criterion",
        "optimizer",
        "lr_scheduler",
        "scaler",
        "rng_state",
        "epoch_rng_state",
    ):
        if not isinstance(payload.get(key), Mapping):
            raise FixedTop1ProbeError(f"semantic checkpoint is missing resumable {key}")

    args = checkpoint_args(payload)
    static = preflight.get("static")
    if not isinstance(static, Mapping):
        raise FixedTop1ProbeError("semantic preflight has no static section")
    expected_args = {
        "config_file": Path(static["config"]["path"]),
        "datasets": Path(static["datasets"]["path"]),
    }
    for key, expected in expected_args.items():
        if _checkpoint_path(args.get(key)) != expected:
            raise FixedTop1ProbeError(f"semantic checkpoint {key} mismatch")
    scalar_args = {
        "world_size": 2,
        "batch_size": 4,
        "distributed": True,
        "amp": True,
        "max_train_iters": expected_target,
        "stage_b_gdino_adapter_train_mode": MODE,
        "stage_b_gdino_tn_scope": SCOPE,
        "stage_b_gdino_confidence_objective": "detached_recent_q05_trust",
        "stage_b_gdino_gate_pool_temperature": 0.01,
        "stage_b_gdino_gate_topk": 3,
        "stage_b_gdino_positive_trust_margin": 0.02,
        "stage_b_gdino_positive_trust_weight": 1.0,
        "stage_b_gdino_queue_size": 512,
        "stage_b_gdino_queue_min_count": 256,
    }
    for key, expected in scalar_args.items():
        if args.get(key) != expected:
            raise FixedTop1ProbeError(
                f"semantic checkpoint arg {key}: expected {expected!r}, got {args.get(key)!r}"
            )
    source_checkpoint = source_checkpoint.resolve()
    pretrain = _checkpoint_path(args.get("pretrain_model_path"))
    resume = _checkpoint_path(args.get("resume"))
    if (pretrain == source_checkpoint) == (resume == source_checkpoint):
        raise FixedTop1ProbeError(
            "semantic lineage must identify its source through exactly one initialization flag"
        )
    initial_path = Path(preflight["initial_checkpoint"]["path"])
    if source_checkpoint == initial_path:
        if pretrain != source_checkpoint or resume is not None:
            raise FixedTop1ProbeError(
                "cross-scope semantic initialization must use --pretrain_model_path"
            )
        initialization_mode = "pretrain_model_path"
    else:
        if resume != source_checkpoint or pretrain is not None:
            raise FixedTop1ProbeError(
                "same-scope semantic continuation must use --resume"
            )
        initialization_mode = "resume"

    criterion = payload["criterion"]
    if _scalar(criterion, "criterion_train_mode_code") != MODE_CODE:
        raise FixedTop1ProbeError("semantic criterion train-mode code mismatch")
    if _scalar(criterion, "criterion_scope_code") != SCOPE_CODE:
        raise FixedTop1ProbeError("semantic criterion scope code mismatch")
    if _scalar(criterion, "criterion_confidence_objective_code") != 2:
        raise FixedTop1ProbeError("semantic criterion P3 objective code mismatch")
    if _scalar(criterion, "criterion_queue_size") != 512:
        raise FixedTop1ProbeError("semantic criterion queue-size contract mismatch")
    if _scalar(criterion, "criterion_queue_min_count") != 256:
        raise FixedTop1ProbeError("semantic criterion queue warmup contract mismatch")
    if not math.isclose(
        _scalar_float(criterion, "criterion_positive_trust_margin"),
        0.02,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise FixedTop1ProbeError("semantic criterion positive-trust margin mismatch")
    if not math.isclose(
        _scalar_float(criterion, "criterion_positive_trust_weight"),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise FixedTop1ProbeError("semantic criterion positive-trust weight mismatch")
    queue = _validate_queue(criterion, require_warm=bool(exact and iteration >= 50))

    groups = payload["optimizer"].get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1:
        raise FixedTop1ProbeError("semantic optimizer must have one parameter group")
    group = groups[0]
    if group.get("stage_b_gdino_branch") != "confidence" or not math.isclose(
        float(group.get("lr", math.nan)), LEARNING_RATE, rel_tol=0.0, abs_tol=1e-12
    ):
        raise FixedTop1ProbeError("semantic optimizer does not exclusively own the gate at LR 3e-4")

    record = checkpoint_record(checkpoint)
    initial = preflight.get("initial_checkpoint")
    if not isinstance(initial, Mapping):
        raise FixedTop1ProbeError("semantic preflight has no initial checkpoint")
    if record.get("base_model_sha256") != initial.get("base_model_sha256"):
        raise FixedTop1ProbeError("semantic phase changed frozen GDINO parameters")
    if record.get("rank_sha256") != initial.get("rank_sha256"):
        raise FixedTop1ProbeError("semantic phase changed the frozen rank branch")
    if record.get("confidence_sha256") == initial.get("confidence_sha256"):
        raise FixedTop1ProbeError("semantic phase did not change the confidence branch")
    if record.get("confidence_final_zero") is True:
        raise FixedTop1ProbeError("semantic confidence output layer remains zero")
    del payload
    return {
        "record": record,
        "iteration": iteration,
        "initialization_mode": initialization_mode,
        "queue": queue,
    }


def _previous(path: Path | None, *, iteration: int) -> Dict[str, Any] | None:
    if path is None:
        return None
    value = read_json(path)
    if (
        value.get("schema") != SCHEMA
        or value.get("kind") != "milestone_checkpoint"
        or int(value.get("iteration", -1)) >= iteration
    ):
        raise FixedTop1ProbeError("previous semantic milestone audit is invalid")
    return value


def _validate_previous_progress(
    previous: Mapping[str, Any] | None, record: Mapping[str, Any]
) -> None:
    if previous is None:
        return
    previous_record = previous.get("checkpoint")
    if not isinstance(previous_record, Mapping):
        raise FixedTop1ProbeError("previous semantic audit has no checkpoint record")
    if previous_record.get("rank_sha256") != record.get("rank_sha256"):
        raise FixedTop1ProbeError("rank branch drifted between semantic milestones")
    if previous_record.get("confidence_sha256") == record.get("confidence_sha256"):
        raise FixedTop1ProbeError("confidence branch did not update between milestones")


def _same_file_identity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(
        left.get(key) == right.get(key)
        for key in ("path", "size_bytes", "sha256")
    )


def _bound_previous_path(
    lineage: Mapping[str, Any], requested: Path | None
) -> Path | None:
    recorded = lineage.get("previous_audit")
    if requested is None:
        if recorded is not None:
            raise FixedTop1ProbeError(
                "semantic segment records a previous milestone but none was supplied"
            )
        return None
    if not isinstance(recorded, Mapping) or not _same_file_identity(
        file_record(requested), recorded
    ):
        raise FixedTop1ProbeError(
            "semantic previous milestone does not match current-segment lineage"
        )
    return requested


def make_segment_lineage(
    *,
    preflight_path: Path,
    expected_target: int,
    source_checkpoint: Path,
    initialization_mode: str,
    previous_audit_path: Path | None,
    recovery_inspection_path: Path | None,
) -> Dict[str, Any]:
    target = int(expected_target)
    if target not in FIXED_TOP1_MILESTONES:
        raise FixedTop1ProbeError(
            f"segment target must be one of {FIXED_TOP1_MILESTONES}"
        )
    preflight = read_json(preflight_path)
    if (
        preflight.get("schema") != SCHEMA
        or preflight.get("kind") != "phase_preflight"
        or preflight.get("phase") != "fixed-top1-confidence"
    ):
        raise FixedTop1ProbeError("segment lineage preflight is invalid")
    launch = preflight.get("launch")
    selected = launch.get("milestones") if isinstance(launch, Mapping) else None
    if not isinstance(selected, list) or target not in selected:
        raise FixedTop1ProbeError(
            f"segment target {target} is outside the preflight milestone prefix"
        )
    source_record = file_record(source_checkpoint)
    ancestry = ""
    if recovery_inspection_path is not None:
        if initialization_mode != "resume":
            raise FixedTop1ProbeError("semantic recovery segment must use --resume")
        inspection = read_json(recovery_inspection_path)
        inspected_checkpoint = inspection.get("checkpoint")
        inspected_lineage_record = inspection.get("segment_lineage")
        if (
            inspection.get("schema") != SCHEMA
            or inspection.get("kind") != "live_checkpoint_inspection"
            or inspection.get("phase") != "confidence"
            or inspection.get("confidence_protocol")
            != "fixed_gdino_top1_verified_v1"
            or inspection.get("tn_scope") != SCOPE
            or int(inspection.get("expected_target", -1)) != target
            or not isinstance(inspected_checkpoint, Mapping)
            or not isinstance(inspected_lineage_record, Mapping)
            or source_record.get("sha256") != inspected_checkpoint.get("sha256")
            or source_record.get("size_bytes")
            != inspected_checkpoint.get("size_bytes")
        ):
            raise FixedTop1ProbeError(
                "semantic recovery source does not match the audited live checkpoint"
            )
        inspected_path = resolve_path(str(inspected_checkpoint.get("path", "")))
        inspected_lineage_path = resolve_path(
            str(inspected_lineage_record.get("path", ""))
        )
        if not _same_file_identity(file_record(inspected_path), inspected_checkpoint):
            raise FixedTop1ProbeError("audited live checkpoint drifted before recovery")
        if not _same_file_identity(
            file_record(inspected_lineage_path), inspected_lineage_record
        ):
            raise FixedTop1ProbeError("recovery inspection segment lineage drifted")
        inspected_lineage = read_json(inspected_lineage_path)
        if (
            inspected_lineage.get("schema") != SCHEMA
            or inspected_lineage.get("kind") != "segment_lineage"
            or inspected_lineage.get("phase") != "confidence"
            or inspected_lineage.get("confidence_protocol")
            != "fixed_gdino_top1_verified_v1"
            or inspected_lineage.get("tn_scope") != SCOPE
            or int(inspected_lineage.get("expected_target", -1)) != target
        ):
            raise FixedTop1ProbeError("recovery inspection ancestry is invalid")
        inspected_previous = inspected_lineage.get("previous_audit")
        if previous_audit_path is None:
            if inspected_previous is not None:
                raise FixedTop1ProbeError("recovery dropped its previous-milestone anchor")
        elif not isinstance(inspected_previous, Mapping) or not _same_file_identity(
            file_record(previous_audit_path), inspected_previous
        ):
            raise FixedTop1ProbeError("recovery changed its previous-milestone anchor")
        ancestry = "audited_live_recovery"
    elif previous_audit_path is not None:
        if initialization_mode != "resume":
            raise FixedTop1ProbeError("post-milestone semantic segment must use --resume")
        previous = read_json(previous_audit_path)
        previous_checkpoint = previous.get("checkpoint")
        expected_previous = FIXED_TOP1_MILESTONES[
            FIXED_TOP1_MILESTONES.index(target) - 1
        ]
        if (
            target == FIXED_TOP1_MILESTONES[0]
            or previous.get("schema") != SCHEMA
            or previous.get("kind") != "milestone_checkpoint"
            or previous.get("phase") != "confidence"
            or int(previous.get("iteration", -1)) != expected_previous
            or not isinstance(previous_checkpoint, Mapping)
            or not _same_file_identity(source_record, previous_checkpoint)
        ):
            raise FixedTop1ProbeError(
                "semantic segment source is not the exact previous milestone"
            )
        ancestry = "previous_milestone"
    else:
        initial = preflight.get("initial_checkpoint")
        if (
            target != FIXED_TOP1_MILESTONES[0]
            or initialization_mode != "pretrain"
            or not isinstance(initial, Mapping)
            or not _same_file_identity(source_record, initial)
        ):
            raise FixedTop1ProbeError(
                "first semantic segment must pretrain from the exact audited R/C source"
            )
        ancestry = "phase_initial"
    return {
        "schema": SCHEMA,
        "kind": "segment_lineage",
        "phase": "confidence",
        "confidence_protocol": "fixed_gdino_top1_verified_v1",
        "tn_scope": SCOPE,
        "expected_target": target,
        "initialization_mode": initialization_mode,
        "ancestry": ancestry,
        "source_checkpoint": source_record,
        "preflight": file_record(preflight_path),
        "previous_audit": (
            file_record(previous_audit_path) if previous_audit_path else None
        ),
        "recovery_inspection": (
            file_record(recovery_inspection_path)
            if recovery_inspection_path
            else None
        ),
    }


def _cmd_static(args: argparse.Namespace) -> None:
    static = validate_static()
    _validate_target_readiness(static, max_target=int(args.max_target))
    print(json.dumps(static, indent=2, sort_keys=True, ensure_ascii=True))


def _cmd_preflight(args: argparse.Namespace) -> None:
    source_checkpoint = resolve_path(args.source_checkpoint)
    source_audit = resolve_path(args.source_audit)
    payload = make_preflight(
        source_kind=args.source_kind,
        source_checkpoint=source_checkpoint,
        source_audit=source_audit,
        world_size=int(args.world_size),
        per_gpu_batch=int(args.per_gpu_batch),
        max_target=int(args.max_target),
    )
    output = resolve_path(args.output)
    if output.exists():
        if not args.continue_run:
            raise FixedTop1ProbeError(f"semantic preflight already exists: {output}")
        if read_json(output) != payload:
            raise FixedTop1ProbeError(f"semantic preflight drifted: {output}")
        print(f"[OK] unchanged semantic preflight: {output}")
        return
    if args.continue_run:
        raise FixedTop1ProbeError(f"cannot continue without semantic preflight: {output}")
    write_json(output, payload)
    print(f"[OK] wrote semantic preflight: {output}")


def _cmd_segment_lineage(args: argparse.Namespace) -> None:
    payload = make_segment_lineage(
        preflight_path=resolve_path(args.preflight),
        expected_target=int(args.expected_target),
        source_checkpoint=resolve_path(args.source_checkpoint),
        initialization_mode=args.initialization_mode,
        previous_audit_path=(
            resolve_path(args.previous_audit) if args.previous_audit else None
        ),
        recovery_inspection_path=(
            resolve_path(args.recovery_inspection)
            if args.recovery_inspection
            else None
        ),
    )
    output = resolve_path(args.output)
    if output.exists():
        raise FixedTop1ProbeError(f"refusing to overwrite segment lineage: {output}")
    write_json(output, payload)
    print(
        f"[OK] recorded semantic segment ancestry for target "
        f"{args.expected_target}: {output}"
    )


def _cmd_inspect(args: argparse.Namespace) -> None:
    preflight = read_json(resolve_path(args.preflight))
    lineage_path = resolve_path(args.segment_lineage)
    lineage = read_json(lineage_path)
    source_record = lineage.get("source_checkpoint")
    if (
        lineage.get("schema") != SCHEMA
        or lineage.get("kind") != "segment_lineage"
        or lineage.get("phase") != "confidence"
        or lineage.get("confidence_protocol") != "fixed_gdino_top1_verified_v1"
        or lineage.get("tn_scope") != SCOPE
        or int(lineage.get("expected_target", -1)) != int(args.expected_target)
        or not isinstance(source_record, Mapping)
    ):
        raise FixedTop1ProbeError(
            "live semantic checkpoint has no matching current-segment lineage"
        )
    source = resolve_path(str(source_record.get("path", "")))
    if not _same_file_identity(file_record(source), source_record):
        raise FixedTop1ProbeError("semantic segment source checkpoint drifted")
    result = validate_checkpoint(
        checkpoint=resolve_path(args.checkpoint),
        preflight=preflight,
        expected_target=int(args.expected_target),
        source_checkpoint=source,
        exact=False,
    )
    requested_previous = (
        resolve_path(args.previous_audit) if args.previous_audit else None
    )
    previous_path = _bound_previous_path(lineage, requested_previous)
    previous = _previous(
        previous_path,
        iteration=result["iteration"],
    )
    _validate_previous_progress(previous, result["record"])
    payload = {
        "schema": SCHEMA,
        "kind": "live_checkpoint_inspection",
        "phase": "confidence",
        "confidence_protocol": "fixed_gdino_top1_verified_v1",
        "tn_scope": SCOPE,
        "iteration": result["iteration"],
        "expected_target": int(args.expected_target),
        "initialization_mode": result["initialization_mode"],
        "queue": result["queue"],
        "segment_lineage": file_record(lineage_path),
        "checkpoint": result["record"],
    }
    if args.output:
        output = resolve_path(args.output)
        if output.exists():
            if read_json(output) != payload:
                raise FixedTop1ProbeError(f"live checkpoint inspection drifted: {output}")
        else:
            write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))


def _cmd_milestone(args: argparse.Namespace) -> None:
    iteration = int(args.expected_iteration)
    preflight_path = resolve_path(args.preflight)
    preflight = read_json(preflight_path)
    lineage_path = resolve_path(args.segment_lineage)
    lineage = read_json(lineage_path)
    source_record = lineage.get("source_checkpoint")
    if (
        lineage.get("schema") != SCHEMA
        or lineage.get("kind") != "segment_lineage"
        or lineage.get("phase") != "confidence"
        or lineage.get("confidence_protocol") != "fixed_gdino_top1_verified_v1"
        or lineage.get("tn_scope") != SCOPE
        or int(lineage.get("expected_target", -1)) != iteration
        or not isinstance(source_record, Mapping)
    ):
        raise FixedTop1ProbeError("semantic milestone segment lineage is invalid")
    source = resolve_path(str(source_record.get("path", "")))
    if not _same_file_identity(file_record(source), source_record):
        raise FixedTop1ProbeError("semantic milestone source checkpoint drifted")
    result = validate_checkpoint(
        checkpoint=resolve_path(args.checkpoint),
        preflight=preflight,
        expected_target=iteration,
        source_checkpoint=source,
        exact=True,
    )
    requested_previous = (
        resolve_path(args.previous_audit) if args.previous_audit else None
    )
    previous_path = _bound_previous_path(lineage, requested_previous)
    previous = _previous(previous_path, iteration=iteration)
    _validate_previous_progress(previous, result["record"])
    payload = {
        "schema": SCHEMA,
        "kind": "milestone_checkpoint",
        # Keep the common evaluator phase vocabulary while making the distinct
        # label protocol and scope explicit below.
        "phase": "confidence",
        "confidence_protocol": "fixed_gdino_top1_verified_v1",
        "tn_scope": SCOPE,
        "iteration": iteration,
        "global_batch": 8,
        "initialization_mode": result["initialization_mode"],
        "queue": result["queue"],
        "segment_lineage": file_record(lineage_path),
        "source_checkpoint": file_record(source),
        "preflight": file_record(preflight_path),
        "previous_audit": file_record(previous_path) if previous_path else None,
        "objective_contract": preflight["static"]["objective_contract"],
        "config": preflight["static"]["config"],
        "config_import_chain": preflight["static"]["config_import_chain"],
        "datasets": preflight["static"]["datasets"],
        "data_audit": preflight["static"]["data_audit"],
        "annotation": preflight["static"]["annotation"],
        "verified_pair_contract": preflight["static"]["verified_pair_contract"],
        "fixed_gdino_source_binding": preflight["fixed_gdino_source_binding"],
        "code": preflight["static"]["code"],
        "orchestration": preflight["static"]["orchestration"],
        "checkpoint": result["record"],
    }
    output = resolve_path(args.output)
    if args.verify_only:
        if read_json(output) != payload:
            raise FixedTop1ProbeError(f"semantic milestone audit drifted: {output}")
        print(f"[OK] unchanged semantic milestone {iteration}: {output}")
        return
    if output.exists():
        raise FixedTop1ProbeError(f"refusing to overwrite semantic milestone audit: {output}")
    write_json(output, payload)
    print(f"[OK] audited semantic milestone {iteration}: {output}")


def _cmd_metadata(args: argparse.Namespace) -> None:
    payload = load_checkpoint(resolve_path(args.checkpoint))
    checkpoint_cli = checkpoint_args(payload)
    sources = [
        value
        for value in (
            checkpoint_cli.get("pretrain_model_path"),
            checkpoint_cli.get("resume"),
        )
        if value not in (None, "")
    ]
    value = {
        "iteration": int(payload.get("iteration", 0) or 0),
        "source_checkpoint": (
            str(resolve_path(sources[0])) if len(sources) == 1 else None
        ),
    }
    if args.field:
        if value.get(args.field) is None:
            raise FixedTop1ProbeError(f"null metadata field: {args.field}")
        print(value[args.field])
    else:
        print(json.dumps(value, sort_keys=True))


def _require_current_file_record(
    value: Any, *, label: str, expected_rows: int | None = None
) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or not value.get("path"):
        raise FixedTop1ProbeError(f"evaluation audit is missing {label} lineage")
    current = file_record(resolve_path(str(value["path"])))
    for key in ("path", "size_bytes", "sha256"):
        if value.get(key) != current.get(key):
            raise FixedTop1ProbeError(
                f"evaluation audit {label} {key} drifted: "
                f"audited={value.get(key)!r}, current={current.get(key)!r}"
            )
    if expected_rows is not None:
        rows = count_nonempty_lines(Path(current["path"]))
        if rows != expected_rows or int(value.get("rows", -1)) != expected_rows:
            raise FixedTop1ProbeError(
                f"evaluation audit {label} row drift: expected {expected_rows}, "
                f"audited={value.get('rows')!r}, current={rows}"
            )
        current["rows"] = rows
    return current


def _checkpoint_record_with_path(
    record: Mapping[str, Any], path: str
) -> Dict[str, Any]:
    value = dict(record)
    value["path"] = path
    return value


def _verify_two_phase_source_milestone(
    *, checkpoint_path: Path, audit_path: Path, expected_phase: str
) -> Dict[str, Any]:
    try:
        verified = _verify_two_phase_milestone_checkpoint(
            checkpoint_path,
            audit_path,
        )
    except ProbeAuditError as error:
        raise FixedTop1ProbeError(
            f"semantic initial two-phase milestone failed deep replay: {error}"
        ) from error
    if verified.get("phase") != expected_phase:
        raise FixedTop1ProbeError(
            f"semantic initial source is not an audited {expected_phase} milestone"
        )
    return dict(verified["checkpoint"])


class _FixedTop1LineageReplay:
    def __init__(self, preflight_path: Path) -> None:
        self.preflight_path = preflight_path.resolve()
        self.preflight_record = file_record(self.preflight_path)
        self.preflight = read_json(self.preflight_path)
        self.static = validate_static()
        self.milestone_cache: Dict[Path, Dict[str, Any]] = {}
        self.segment_cache: Dict[Path, Dict[str, Any]] = {}
        self.inspection_cache: Dict[Path, Dict[str, Any]] = {}
        self._milestone_stack: set[Path] = set()
        self._segment_stack: set[Path] = set()
        self._inspection_stack: set[Path] = set()
        self._verify_preflight()

    def _verify_preflight(self) -> None:
        initial = self.preflight.get("initial_checkpoint")
        initial_audit = self.preflight.get("initial_audit")
        launch = self.preflight.get("launch")
        source_kind = self.preflight.get("source_kind")
        if (
            self.preflight.get("schema") != SCHEMA
            or self.preflight.get("kind") != "phase_preflight"
            or self.preflight.get("phase") != "fixed-top1-confidence"
            or source_kind != "rank"
            or not isinstance(initial, Mapping)
            or not initial.get("path")
            or not isinstance(initial_audit, Mapping)
            or not initial_audit.get("path")
            or not isinstance(launch, Mapping)
        ):
            raise FixedTop1ProbeError("semantic formal preflight is incomplete")
        initial_path = resolve_path(str(initial["path"]))
        initial_audit_path = resolve_path(str(initial_audit["path"]))
        generic_record = _verify_two_phase_source_milestone(
            checkpoint_path=initial_path,
            audit_path=initial_audit_path,
            expected_phase="rank",
        )
        if generic_record != initial:
            raise FixedTop1ProbeError(
                "semantic preflight initial checkpoint differs from its deep audit"
            )
        current = make_preflight(
            source_kind=str(source_kind),
            source_checkpoint=initial_path,
            source_audit=initial_audit_path,
            world_size=int(launch.get("world_size", -1)),
            per_gpu_batch=int(launch.get("per_gpu_batch", -1)),
            max_target=int(launch.get("max_target", -1)),
        )
        if current != self.preflight or self.preflight.get("static") != self.static:
            raise FixedTop1ProbeError(
                "semantic config/data/code/initial lineage drifted since preflight"
            )

    @staticmethod
    def _previous_iteration(target: int) -> int | None:
        index = FIXED_TOP1_MILESTONES.index(target)
        return None if index == 0 else int(FIXED_TOP1_MILESTONES[index - 1])

    def verify_milestone(
        self, audit_path: Path, *, expected_checkpoint: Path | None = None
    ) -> Dict[str, Any]:
        audit_path = audit_path.resolve()
        if audit_path in self.milestone_cache:
            result = self.milestone_cache[audit_path]
            if expected_checkpoint is not None and Path(
                result["checkpoint"]["path"]
            ).resolve() != expected_checkpoint.resolve():
                raise FixedTop1ProbeError("semantic milestone checkpoint path forked")
            return result
        if audit_path in self._milestone_stack:
            raise FixedTop1ProbeError("semantic previous-milestone lineage contains a cycle")
        self._milestone_stack.add(audit_path)
        try:
            audit = read_json(audit_path)
            iteration = int(audit.get("iteration", -1))
            if (
                audit.get("schema") != SCHEMA
                or audit.get("kind") != "milestone_checkpoint"
                or audit.get("phase") != "confidence"
                or audit.get("confidence_protocol")
                != "fixed_gdino_top1_verified_v1"
                or audit.get("tn_scope") != SCOPE
                or iteration not in FIXED_TOP1_MILESTONES
                or iteration not in self.preflight.get("launch", {}).get(
                    "milestones", []
                )
                or int(audit.get("global_batch", -1)) != 8
            ):
                raise FixedTop1ProbeError("semantic milestone protocol is invalid")
            if audit.get("preflight") != self.preflight_record:
                raise FixedTop1ProbeError("semantic milestone changed its preflight")

            previous_value = audit.get("previous_audit")
            expected_previous_iteration = self._previous_iteration(iteration)
            previous_path: Path | None = None
            previous_result: Dict[str, Any] | None = None
            if expected_previous_iteration is None:
                if previous_value is not None:
                    raise FixedTop1ProbeError("first semantic milestone has a previous audit")
            else:
                previous_record = _require_current_file_record(
                    previous_value, label="previous semantic milestone"
                )
                previous_path = Path(previous_record["path"])
                previous_result = self.verify_milestone(previous_path)
                if int(previous_result["audit"]["iteration"]) != expected_previous_iteration:
                    raise FixedTop1ProbeError(
                        "semantic previous milestone chain skipped an iteration"
                    )

            segment_record = _require_current_file_record(
                audit.get("segment_lineage"), label="semantic segment lineage"
            )
            segment_path = Path(segment_record["path"])
            segment = self.verify_segment(
                segment_path, target=iteration, previous_path=previous_path
            )
            source_path = segment["source_path"]
            source_record = file_record(source_path)
            if audit.get("source_checkpoint") != source_record:
                raise FixedTop1ProbeError("semantic milestone source differs from its segment")

            checkpoint_value = audit.get("checkpoint")
            if not isinstance(checkpoint_value, Mapping) or not checkpoint_value.get("path"):
                raise FixedTop1ProbeError("semantic milestone has no checkpoint record")
            checkpoint_path = resolve_path(str(checkpoint_value["path"]))
            if expected_checkpoint is not None and checkpoint_path != expected_checkpoint.resolve():
                raise FixedTop1ProbeError(
                    "semantic evaluation checkpoint path differs from milestone audit"
                )
            result = validate_checkpoint(
                checkpoint=checkpoint_path,
                preflight=self.preflight,
                expected_target=iteration,
                source_checkpoint=source_path,
                exact=True,
            )
            if result["record"] != checkpoint_value:
                raise FixedTop1ProbeError("semantic milestone checkpoint content drifted")
            previous_audit = previous_result["audit"] if previous_result else None
            _validate_previous_progress(previous_audit, result["record"])
            expected = {
                "schema": SCHEMA,
                "kind": "milestone_checkpoint",
                "phase": "confidence",
                "confidence_protocol": "fixed_gdino_top1_verified_v1",
                "tn_scope": SCOPE,
                "iteration": iteration,
                "global_batch": 8,
                "initialization_mode": result["initialization_mode"],
                "queue": result["queue"],
                "segment_lineage": file_record(segment_path),
                "source_checkpoint": source_record,
                "preflight": self.preflight_record,
                "previous_audit": file_record(previous_path) if previous_path else None,
                "objective_contract": self.static["objective_contract"],
                "config": self.static["config"],
                "config_import_chain": self.static["config_import_chain"],
                "datasets": self.static["datasets"],
                "data_audit": self.static["data_audit"],
                "annotation": self.static["annotation"],
                "verified_pair_contract": self.static["verified_pair_contract"],
                "fixed_gdino_source_binding": self.preflight[
                    "fixed_gdino_source_binding"
                ],
                "code": self.static["code"],
                "orchestration": self.static["orchestration"],
                "checkpoint": result["record"],
            }
            if audit != expected:
                raise FixedTop1ProbeError("semantic milestone payload failed replay")
            verified = {
                "audit": audit,
                "checkpoint": result["record"],
                "segment": segment,
            }
            self.milestone_cache[audit_path] = verified
            return verified
        finally:
            self._milestone_stack.remove(audit_path)

    def verify_segment(
        self, lineage_path: Path, *, target: int, previous_path: Path | None
    ) -> Dict[str, Any]:
        lineage_path = lineage_path.resolve()
        if lineage_path in self.segment_cache:
            cached = self.segment_cache[lineage_path]
            if cached["target"] != target or cached["previous_path"] != previous_path:
                raise FixedTop1ProbeError("semantic segment was reused with different ancestry")
            return cached
        if lineage_path in self._segment_stack:
            raise FixedTop1ProbeError("semantic recovery segment lineage contains a cycle")
        self._segment_stack.add(lineage_path)
        try:
            lineage = read_json(lineage_path)
            source_value = lineage.get("source_checkpoint")
            if (
                lineage.get("schema") != SCHEMA
                or lineage.get("kind") != "segment_lineage"
                or lineage.get("phase") != "confidence"
                or lineage.get("confidence_protocol")
                != "fixed_gdino_top1_verified_v1"
                or lineage.get("tn_scope") != SCOPE
                or int(lineage.get("expected_target", -1)) != target
                or lineage.get("preflight") != self.preflight_record
                or not isinstance(source_value, Mapping)
            ):
                raise FixedTop1ProbeError("semantic segment protocol is invalid")
            expected_previous_record = (
                file_record(previous_path) if previous_path is not None else None
            )
            if lineage.get("previous_audit") != expected_previous_record:
                raise FixedTop1ProbeError("semantic segment previous milestone is disconnected")
            source_record = _require_current_file_record(
                source_value, label="semantic segment source"
            )
            source_path = Path(source_record["path"])
            recovery_value = lineage.get("recovery_inspection")
            mode = lineage.get("initialization_mode")
            ancestry = lineage.get("ancestry")

            if recovery_value is not None:
                if mode != "resume" or ancestry != "audited_live_recovery":
                    raise FixedTop1ProbeError("semantic recovery segment mode is invalid")
                recovery_record = _require_current_file_record(
                    recovery_value, label="semantic recovery inspection"
                )
                self.verify_inspection(
                    Path(recovery_record["path"]),
                    target=target,
                    previous_path=previous_path,
                    recovery_checkpoint=source_path,
                )
            elif previous_path is not None:
                previous = self.verify_milestone(previous_path)
                if (
                    mode != "resume"
                    or ancestry != "previous_milestone"
                    or not _same_file_identity(source_record, previous["checkpoint"])
                ):
                    raise FixedTop1ProbeError(
                        "semantic segment did not resume from the previous milestone"
                    )
            else:
                initial = self.preflight["initial_checkpoint"]
                if (
                    target != FIXED_TOP1_MILESTONES[0]
                    or mode != "pretrain"
                    or ancestry != "phase_initial"
                    or not _same_file_identity(source_record, initial)
                ):
                    raise FixedTop1ProbeError(
                        "first semantic segment did not pretrain from preflight initial"
                    )

            expected = {
                "schema": SCHEMA,
                "kind": "segment_lineage",
                "phase": "confidence",
                "confidence_protocol": "fixed_gdino_top1_verified_v1",
                "tn_scope": SCOPE,
                "expected_target": target,
                "initialization_mode": mode,
                "ancestry": ancestry,
                "source_checkpoint": source_record,
                "preflight": self.preflight_record,
                "previous_audit": expected_previous_record,
                "recovery_inspection": (
                    file_record(resolve_path(str(recovery_value["path"])))
                    if isinstance(recovery_value, Mapping)
                    else None
                ),
            }
            if lineage != expected:
                raise FixedTop1ProbeError("semantic segment payload failed replay")
            result = {
                "lineage": lineage,
                "source_path": source_path,
                "source_record": source_record,
                "target": target,
                "previous_path": previous_path,
            }
            self.segment_cache[lineage_path] = result
            return result
        finally:
            self._segment_stack.remove(lineage_path)

    def verify_inspection(
        self,
        inspection_path: Path,
        *,
        target: int,
        previous_path: Path | None,
        recovery_checkpoint: Path,
    ) -> Dict[str, Any]:
        inspection_path = inspection_path.resolve()
        if inspection_path in self.inspection_cache:
            cached = self.inspection_cache[inspection_path]
            if (
                cached["target"] != target
                or cached["previous_path"] != previous_path
                or cached["recovery_checkpoint"] != recovery_checkpoint
            ):
                raise FixedTop1ProbeError("semantic live inspection was reused inconsistently")
            return cached
        if inspection_path in self._inspection_stack:
            raise FixedTop1ProbeError("semantic recovery inspection lineage contains a cycle")
        self._inspection_stack.add(inspection_path)
        try:
            inspection = read_json(inspection_path)
            prior_segment_value = inspection.get("segment_lineage")
            audited_checkpoint = inspection.get("checkpoint")
            if (
                inspection.get("schema") != SCHEMA
                or inspection.get("kind") != "live_checkpoint_inspection"
                or inspection.get("phase") != "confidence"
                or inspection.get("confidence_protocol")
                != "fixed_gdino_top1_verified_v1"
                or inspection.get("tn_scope") != SCOPE
                or int(inspection.get("expected_target", -1)) != target
                or not isinstance(prior_segment_value, Mapping)
                or not isinstance(audited_checkpoint, Mapping)
            ):
                raise FixedTop1ProbeError("semantic recovery inspection protocol is invalid")
            prior_segment_record = _require_current_file_record(
                prior_segment_value, label="recovery prior segment"
            )
            prior_segment = self.verify_segment(
                Path(prior_segment_record["path"]),
                target=target,
                previous_path=previous_path,
            )
            result = validate_checkpoint(
                checkpoint=recovery_checkpoint,
                preflight=self.preflight,
                expected_target=target,
                source_checkpoint=prior_segment["source_path"],
                exact=False,
            )
            aliased_record = _checkpoint_record_with_path(
                result["record"], str(audited_checkpoint.get("path", ""))
            )
            if aliased_record != audited_checkpoint:
                raise FixedTop1ProbeError(
                    "semantic recovery checkpoint content does not match live inspection"
                )
            previous = (
                self.verify_milestone(previous_path)["audit"]
                if previous_path is not None
                else None
            )
            _validate_previous_progress(previous, result["record"])
            expected = {
                "schema": SCHEMA,
                "kind": "live_checkpoint_inspection",
                "phase": "confidence",
                "confidence_protocol": "fixed_gdino_top1_verified_v1",
                "tn_scope": SCOPE,
                "iteration": result["iteration"],
                "expected_target": target,
                "initialization_mode": result["initialization_mode"],
                "queue": result["queue"],
                "segment_lineage": prior_segment_record,
                "checkpoint": aliased_record,
            }
            if inspection != expected:
                raise FixedTop1ProbeError("semantic recovery inspection payload failed replay")
            verified = {
                "inspection": inspection,
                "target": target,
                "previous_path": previous_path,
                "recovery_checkpoint": recovery_checkpoint,
            }
            self.inspection_cache[inspection_path] = verified
            return verified
        finally:
            self._inspection_stack.remove(inspection_path)


def _replay_evaluation_milestone(
    *, checkpoint_path: Path, audit_path: Path
) -> tuple["_FixedTop1LineageReplay", Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    checkpoint_path = checkpoint_path.resolve()
    audit_path = audit_path.resolve()
    audit = read_json(audit_path)
    preflight_value = audit.get("preflight")
    if not isinstance(preflight_value, Mapping) or not preflight_value.get("path"):
        raise FixedTop1ProbeError("semantic evaluation audit has no preflight")
    preflight_record = _require_current_file_record(
        preflight_value, label="semantic preflight"
    )
    replay = _FixedTop1LineageReplay(Path(preflight_record["path"]))
    verified = replay.verify_milestone(
        audit_path, expected_checkpoint=checkpoint_path
    )
    audit = verified["audit"]
    segment = verified["segment"]
    return replay, verified, audit, segment


def verify_calibration_checkpoint(
    *, checkpoint_path: Path, audit_path: Path, expected_iteration: int
) -> Dict[str, Any]:
    replay, verified, audit, _segment = _replay_evaluation_milestone(
        checkpoint_path=checkpoint_path,
        audit_path=audit_path,
    )
    if int(audit["iteration"]) != int(expected_iteration):
        raise FixedTop1ProbeError(
            "calibration checkpoint iteration differs from its pre-registered milestone"
        )
    return {
        "schema": SCHEMA,
        "kind": "calibration_checkpoint_verification",
        "phase": "confidence",
        "iteration": int(audit["iteration"]),
        "checkpoint": verified["checkpoint"],
        "checkpoint_audit": file_record(audit_path.resolve()),
        "config": replay.static["config"],
        "partition": replay.static["partition"],
        "preflight": replay.preflight_record,
        "input_scope": "sealed_calibration_only",
        "formal_strict_authorization": False,
        "verified": True,
    }


def verify_evaluation_checkpoint(
    *, checkpoint_path: Path, audit_path: Path
) -> Dict[str, Any]:
    checkpoint_path = checkpoint_path.resolve()
    audit_path = audit_path.resolve()
    replay, verified, audit, segment = _replay_evaluation_milestone(
        checkpoint_path=checkpoint_path,
        audit_path=audit_path,
    )
    selection_path = replay.preflight_record["path"]
    selection_path = Path(str(selection_path)).resolve().parent / SELECTION_AUDIT_NAME
    try:
        selection = verify_selection(
            selection_path,
            expected_checkpoint=checkpoint_path,
            expected_milestone_audit=audit_path,
            expected_calibration_root=(
                Path(str(replay.preflight_record["path"])).resolve().parent
                / "calibration_selection"
            ),
        )
    except RuntimeError as error:
        raise FixedTop1ProbeError(
            f"fixed-top1 formal evaluation is not authorized by held-out selection: {error}"
        ) from error
    recovery_value = segment["lineage"].get("recovery_inspection")
    return {
        "schema": SCHEMA,
        "kind": "evaluation_checkpoint_verification",
        "phase": "confidence",
        "confidence_protocol": "fixed_gdino_top1_verified_v1",
        "tn_scope": SCOPE,
        "iteration": int(audit["iteration"]),
        "checkpoint": verified["checkpoint"],
        "checkpoint_audit": file_record(audit_path),
        "objective_contract": replay.static["objective_contract"],
        "config": replay.static["config"],
        "config_import_chain": replay.static["config_import_chain"],
        "datasets": replay.static["datasets"],
        "data_audit": replay.static["data_audit"],
        "annotation": replay.static["annotation"],
        "source_annotation": replay.static["source_annotation"],
        "partition": replay.static["partition"],
        "selection_authorization": {
            "schema": SELECTION_SCHEMA,
            "audit": selection["audit"],
            "selected_checkpoint": selection["selected_checkpoint"],
            "selected_milestone_audit": selection["selected_milestone_audit"],
            "selected_iteration": selection["selected_iteration"],
            "calibration_root": selection["calibration_root"],
            "input_scope": selection["input_scope"],
            "strict_paths_consumed_for_scoring": selection[
                "strict_paths_consumed_for_scoring"
            ],
            "formal_strict_authorization": True,
        },
        "verified_pair_contract": replay.static["verified_pair_contract"],
        "fixed_gdino_source_binding": replay.preflight[
            "fixed_gdino_source_binding"
        ],
        "code": replay.static["code"],
        "orchestration": replay.static["orchestration"],
        "preflight": replay.preflight_record,
        "segment_lineage": audit["segment_lineage"],
        "source_checkpoint": audit["source_checkpoint"],
        "previous_audit": audit["previous_audit"],
        "recovery_inspection": recovery_value,
        "lineage_replay": {
            "milestones": len(replay.milestone_cache),
            "segments": len(replay.segment_cache),
            "recovery_inspections": len(replay.inspection_cache),
            "initial_two_phase_milestone": True,
        },
        "verified": True,
    }


def _cmd_verify_calibration(args: argparse.Namespace) -> None:
    result = verify_calibration_checkpoint(
        checkpoint_path=resolve_path(args.checkpoint),
        audit_path=resolve_path(args.audit),
        expected_iteration=int(args.expected_iteration),
    )
    if args.output:
        write_json(resolve_path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))


def _cmd_verify_evaluation(args: argparse.Namespace) -> None:
    result = verify_evaluation_checkpoint(
        checkpoint_path=resolve_path(args.checkpoint),
        audit_path=resolve_path(args.audit),
    )
    if args.output:
        write_json(resolve_path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    static = subparsers.add_parser("static")
    static.add_argument("--max-target", type=int, default=500)
    static.set_defaults(func=_cmd_static)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--source-kind", choices=("rank",), required=True)
    preflight.add_argument("--source-checkpoint", required=True)
    preflight.add_argument("--source-audit", required=True)
    preflight.add_argument("--world-size", type=int, default=2)
    preflight.add_argument("--per-gpu-batch", type=int, default=4)
    preflight.add_argument("--max-target", type=int, default=500)
    preflight.add_argument("--output", required=True)
    preflight.add_argument("--continue-run", action="store_true")
    preflight.set_defaults(func=_cmd_preflight)

    segment = subparsers.add_parser("segment-lineage")
    segment.add_argument("--preflight", required=True)
    segment.add_argument("--expected-target", type=int, required=True)
    segment.add_argument("--source-checkpoint", required=True)
    segment.add_argument(
        "--initialization-mode", choices=("pretrain", "resume"), required=True
    )
    segment.add_argument("--previous-audit")
    segment.add_argument("--recovery-inspection")
    segment.add_argument("--output", required=True)
    segment.set_defaults(func=_cmd_segment_lineage)

    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--checkpoint", required=True)
    inspect.add_argument("--preflight", required=True)
    inspect.add_argument("--expected-target", type=int, required=True)
    inspect.add_argument("--segment-lineage", required=True)
    inspect.add_argument("--previous-audit")
    inspect.add_argument("--output")
    inspect.set_defaults(func=_cmd_inspect)

    milestone = subparsers.add_parser("milestone")
    milestone.add_argument("--checkpoint", required=True)
    milestone.add_argument("--preflight", required=True)
    milestone.add_argument("--expected-iteration", type=int, required=True)
    milestone.add_argument("--segment-lineage", required=True)
    milestone.add_argument("--previous-audit")
    milestone.add_argument("--output", required=True)
    milestone.add_argument("--verify-only", action="store_true")
    milestone.set_defaults(func=_cmd_milestone)

    metadata = subparsers.add_parser("metadata")
    metadata.add_argument("--checkpoint", required=True)
    metadata.add_argument("--field", choices=("iteration", "source_checkpoint"))
    metadata.set_defaults(func=_cmd_metadata)

    evaluation = subparsers.add_parser("verify-evaluation")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--audit", required=True)
    evaluation.add_argument("--output")
    evaluation.set_defaults(func=_cmd_verify_evaluation)

    calibration = subparsers.add_parser("verify-calibration")
    calibration.add_argument("--checkpoint", required=True)
    calibration.add_argument("--audit", required=True)
    calibration.add_argument("--expected-iteration", type=int, required=True)
    calibration.add_argument("--output")
    calibration.set_defaults(func=_cmd_verify_calibration)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (FixedTop1ProbeError, ProbeAuditError, OSError, ValueError, KeyError) as error:
        raise SystemExit(f"[FAIL] {error}") from error


if __name__ == "__main__":
    main()
