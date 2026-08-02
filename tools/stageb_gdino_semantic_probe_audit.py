#!/usr/bin/env python3
"""Fail-closed audit for the independent semantic confidence probe."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

try:
    import torch
except ModuleNotFoundError:  # Static/dry-run audits do not need PyTorch.
    torch = None

from tools.stageb_dependency_audit import (
    DependencyAuditError,
    config_import_chain,
    local_python_dependency_paths,
)

from tools.stageb_gdino_adapter_probe_audit import (
    CONFIDENCE_MILESTONES,
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


SCHEMA = "stageb-gdino-adapter-semantic-confidence-probe-v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_SCHEMA = "stage-b-gdino-adapter-semantic-verified-pairs-v1"
CONFIG = "config/ablations/cfg_stageb_gdino_score_adapter_semantic_verified.py"
DATASETS = "config/datasets_stageb_gdino_adapter_semantic_verified_pairs.json"
DATA_AUDIT = "data/ablations/stageb_gdino_adapter_semantic_verified_20260711/audit.json"
MODE = "confidence_only"
MODE_CODE = 2
SCOPE = "image_global_topk_verified"
SCOPE_CODE = 1
LEARNING_RATE = 3.0e-4
SEMANTIC_MILESTONES = CONFIDENCE_MILESTONES
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
    "tools/build_stageb_gdino_adapter_semantic_verified_pairs.py",
    "tools/stageb_gdino_semantic_probe_audit.py",
    "tools/stageb_dependency_audit.py",
)
ORCHESTRATION_PATHS = (
    "tools/run_stageb_gdino_semantic_confidence_probe.sh",
)


class SemanticProbeError(ProbeAuditError):
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
        "find_unused_params": False,
    }
    for key, expected in expected_cfg.items():
        observed = getattr(cfg, key, None)
        if not _matches(observed, expected):
            raise SemanticProbeError(
                f"semantic config mismatch for {key}: expected {expected!r}, "
                f"got {observed!r}"
            )

    dataset_config = read_json(datasets_path)
    train = dataset_config.get("train")
    if not isinstance(train, list) or len(train) != 1 or dataset_config.get("val") != []:
        raise SemanticProbeError("semantic dataset must have one train source and no val")
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
            raise SemanticProbeError(
                f"semantic dataset mismatch for {key}: expected {expected!r}, "
                f"got {entry.get(key)!r}"
            )
    annotation = resolve_path(entry.get("anno", ""))
    data_audit = read_json(data_audit_path)
    if (
        data_audit.get("schema") != DATA_SCHEMA
        or data_audit.get("tn_scope") != SCOPE
        or data_audit.get("proposalset_proxy_verified") is not False
        or data_audit.get("cached_proposal_coverage_only") is not True
        or data_audit.get("all_900_gdino_queries_verified") is not False
        or data_audit.get("global_max_label_is_semantic_extrapolation") is not True
        or int(data_audit.get("rows", -1)) != 17_829
        or resolve_path(data_audit.get("output", "")) != annotation
    ):
        raise SemanticProbeError("semantic pair audit contract is malformed")
    annotation_record = file_record(annotation)
    if annotation_record["sha256"] != data_audit.get("output_sha256"):
        raise SemanticProbeError("semantic pair file hash does not match its audit")
    annotation_rows = count_nonempty_lines(annotation)
    if annotation_rows != 17_829:
        raise SemanticProbeError(
            f"semantic pair row drift: {annotation_rows} != 17829"
        )
    overlap = data_audit.get("overlap_audit", {})
    strict1607 = overlap.get("strict1607", {}) if isinstance(overlap, Mapping) else {}
    if int(strict1607.get("train_image_overlap", -1)) != 0:
        raise SemanticProbeError("semantic pair source is no longer strict1607 image-disjoint")
    strict2031 = overlap.get("strict2031", {}) if isinstance(overlap, Mapping) else {}
    if int(strict2031.get("train_image_overlap", -1)) != 59:
        raise SemanticProbeError("strict2031 overlap disclosure drifted")
    if int(overlap.get("cross_dataset_image_overlap", -1)) != 1547:
        raise SemanticProbeError("cross-dataset image-overlap disclosure drifted")
    claims = data_audit.get("claims")
    if not isinstance(claims, Mapping) or "900" not in str(claims.get("scope_limit", "")):
        raise SemanticProbeError("semantic pair audit lost its 900-query scope limit")

    source_audit = data_audit.get("source_audit")
    if not isinstance(source_audit, Mapping):
        raise SemanticProbeError("semantic pair audit lost its frozen-source audit")
    _require_audited_hash(source_audit, label="frozen source audit")
    sources = data_audit.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {
        "refcocoplus",
        "refcocog",
    }:
        raise SemanticProbeError("semantic pair source lineage is incomplete")
    for dataset, expected_rows in (("refcocoplus", 10_855), ("refcocog", 6_974)):
        record = sources[dataset]
        path = _require_audited_hash(record, label=f"{dataset} frozen rows")
        source_rows = count_nonempty_lines(path)
        if source_rows != expected_rows or int(record.get("rows", -1)) != expected_rows:
            raise SemanticProbeError(f"{dataset} frozen row count drifted")
        split_counts = record.get("official_split_counts")
        if not isinstance(split_counts, Mapping) or int(
            split_counts.get("rows", -1)
        ) != expected_rows:
            raise SemanticProbeError(f"{dataset} official split audit is incomplete")
    official_refs = data_audit.get("official_refs")
    if not isinstance(official_refs, Mapping) or set(official_refs) != {
        "refcocoplus_unc",
        "refcocog_umd",
        "refcocog_google",
    }:
        raise SemanticProbeError("semantic official-ref lineage is incomplete")
    for name, record in official_refs.items():
        _require_audited_hash(record, label=f"official refs {name}")
    for name, expected_rows in (("strict2031", 2031), ("strict1607", 1607)):
        record = overlap.get(name)
        path = _require_audited_hash(record, label=name)
        manifest_rows = count_nonempty_lines(path)
        if manifest_rows != expected_rows or int(record.get("rows", -1)) != expected_rows:
            raise SemanticProbeError(f"{name} manifest row/hash lineage drifted")

    try:
        config_paths = config_import_chain(config_path, root=REPO_ROOT)
        code_paths = local_python_dependency_paths(
            TRAIN_CODE_ENTRIES,
            root=REPO_ROOT,
            include=TRAIN_CODE_INCLUDE,
        )
    except DependencyAuditError as error:
        raise SemanticProbeError(str(error)) from error
    if not config_paths or config_path not in config_paths:
        raise SemanticProbeError("semantic config import chain is incomplete")

    annotation_record["rows"] = annotation_rows
    return {
        "config": file_record(config_path),
        "config_import_chain": [file_record(path) for path in config_paths],
        "datasets": file_record(datasets_path),
        "data_audit": file_record(data_audit_path),
        "annotation": annotation_record,
        "tn_scope": SCOPE,
        "resolved_contract": expected_cfg,
        "objective_contract": dict(OBJECTIVE_CONTRACT),
        "code": [file_record(path) for path in code_paths],
        "orchestration": [
            file_record(resolve_path(path)) for path in ORCHESTRATION_PATHS
        ],
    }


def _require_audited_hash(value: Any, *, label: str) -> Path:
    if not isinstance(value, Mapping) or not value.get("path") or not value.get("sha256"):
        raise SemanticProbeError(f"missing path/hash for {label}")
    path = resolve_path(str(value["path"]))
    current = file_record(path)
    if current["sha256"] != value.get("sha256"):
        raise SemanticProbeError(f"hash drift for {label}: {path}")
    return path


def _validate_source(
    source_kind: str, checkpoint_path: Path, audit_path: Path
) -> Dict[str, Any]:
    expected_phase = {
        "rank": "rank",
        "dataft-confidence": "confidence",
    }.get(source_kind)
    if expected_phase is None:
        raise SemanticProbeError(f"unsupported source kind: {source_kind!r}")
    source_milestones = (
        RANK_MILESTONES
        if source_kind == "rank"
        else CONFIDENCE_MILESTONES
    )
    audit = read_json(audit_path)
    if (
        audit.get("schema") != TWO_PHASE_SCHEMA
        or audit.get("kind") != "milestone_checkpoint"
        or audit.get("phase") != expected_phase
        or int(audit.get("iteration", -1)) not in source_milestones
    ):
        raise SemanticProbeError(
            f"source audit is not an audited {expected_phase} milestone"
        )
    record = checkpoint_record(checkpoint_path)
    audited = audit.get("checkpoint")
    if not isinstance(audited, Mapping) or audited.get("sha256") != record.get("sha256"):
        raise SemanticProbeError("source checkpoint hash does not match its milestone audit")
    for key in ("base_model_sha256", "rank_sha256", "confidence_sha256"):
        if audited.get(key) != record.get(key):
            raise SemanticProbeError(f"source checkpoint {key} drifted from its audit")
    if int(record.get("adapter_state_keys", 0)) <= 0:
        raise SemanticProbeError("semantic phase requires a complete adapter checkpoint")
    if source_kind == "rank" and record.get("rank_final_zero") is True:
        raise SemanticProbeError("rank source has an untrained rank output")
    if source_kind == "dataft-confidence" and record.get("confidence_final_zero") is True:
        raise SemanticProbeError("data-FT confidence source has an untrained gate output")
    return record


def make_preflight(
    *,
    source_kind: str,
    source_checkpoint: Path,
    source_audit: Path,
    world_size: int,
    per_gpu_batch: int,
) -> Dict[str, Any]:
    if world_size != 2 or per_gpu_batch != 4:
        raise SemanticProbeError(
            "semantic probe requires two DDP ranks and per-GPU batch 4"
        )
    return {
        "schema": SCHEMA,
        "kind": "phase_preflight",
        "phase": "semantic-confidence",
        "initialization": (
            f"audited_{source_kind}_to_semantic_confidence_pretrain_model_path"
        ),
        "source_kind": source_kind,
        "initial_checkpoint": _validate_source(
            source_kind, source_checkpoint, source_audit
        ),
        "initial_audit": file_record(source_audit),
        "static": validate_static(),
        "launch": {
            "world_size": world_size,
            "per_gpu_batch": per_gpu_batch,
            "global_batch": world_size * per_gpu_batch,
            "first_segment_initialization": "pretrain_model_path",
            "same_scope_continuation": "resume",
            "cross_scope_resume_forbidden": True,
        },
    }


def _checkpoint_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return resolve_path(str(value))


def _scalar(state: Mapping[str, Any], key: str) -> int:
    value = state.get(key)
    if not torch.is_tensor(value) or value.numel() != 1:
        raise SemanticProbeError(f"criterion state is missing scalar {key}")
    return int(value.detach().reshape(-1)[0].item())


def _scalar_float(state: Mapping[str, Any], key: str) -> float:
    value = state.get(key)
    if not torch.is_tensor(value) or value.numel() != 1 or not value.is_floating_point():
        raise SemanticProbeError(f"criterion state is missing float scalar {key}")
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
        or not torch.is_tensor(count_value)
        or count_value.numel() != 1
        or not torch.is_tensor(pointer_value)
        or pointer_value.numel() != 1
    ):
        raise SemanticProbeError("semantic criterion queue state is malformed")
    count = int(count_value.reshape(-1)[0].item())
    pointer = int(pointer_value.reshape(-1)[0].item())
    if not 0 <= count <= 512 or not 0 <= pointer < 512:
        raise SemanticProbeError("semantic criterion queue counter/pointer is invalid")
    if require_warm and count < 256:
        raise SemanticProbeError(
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
    if expected_target not in SEMANTIC_MILESTONES:
        raise SemanticProbeError(
            f"target must be one of {SEMANTIC_MILESTONES}"
        )
    payload = load_checkpoint(checkpoint)
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise SemanticProbeError("semantic checkpoint has no model state")
    iteration = int(payload.get("iteration", 0) or 0)
    if exact:
        if iteration != expected_target or payload.get("checkpoint_reason") != "max_train_iters":
            raise SemanticProbeError(
                f"semantic milestone expected max_train_iters at {expected_target}, "
                f"got iteration={iteration}, reason={payload.get('checkpoint_reason')!r}"
            )
    elif not 0 < iteration <= expected_target:
        raise SemanticProbeError(
            f"semantic live iteration must be in [1,{expected_target}], got {iteration}"
        )
    if int(payload.get("epoch", -1)) != 0 or payload.get("epoch_finished") is not False:
        raise SemanticProbeError("semantic probe must remain a mid-epoch epoch=0 run")
    for key in (
        "criterion",
        "optimizer",
        "lr_scheduler",
        "scaler",
        "rng_state",
        "epoch_rng_state",
    ):
        if not isinstance(payload.get(key), Mapping):
            raise SemanticProbeError(f"semantic checkpoint is missing resumable {key}")

    args = checkpoint_args(payload)
    static = preflight.get("static")
    if not isinstance(static, Mapping):
        raise SemanticProbeError("semantic preflight has no static section")
    expected_args = {
        "config_file": Path(static["config"]["path"]),
        "datasets": Path(static["datasets"]["path"]),
    }
    for key, expected in expected_args.items():
        if _checkpoint_path(args.get(key)) != expected:
            raise SemanticProbeError(f"semantic checkpoint {key} mismatch")
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
            raise SemanticProbeError(
                f"semantic checkpoint arg {key}: expected {expected!r}, got {args.get(key)!r}"
            )
    source_checkpoint = source_checkpoint.resolve()
    pretrain = _checkpoint_path(args.get("pretrain_model_path"))
    resume = _checkpoint_path(args.get("resume"))
    if (pretrain == source_checkpoint) == (resume == source_checkpoint):
        raise SemanticProbeError(
            "semantic lineage must identify its source through exactly one initialization flag"
        )
    initial_path = Path(preflight["initial_checkpoint"]["path"])
    if source_checkpoint == initial_path:
        if pretrain != source_checkpoint or resume is not None:
            raise SemanticProbeError(
                "cross-scope semantic initialization must use --pretrain_model_path"
            )
        initialization_mode = "pretrain_model_path"
    else:
        if resume != source_checkpoint or pretrain is not None:
            raise SemanticProbeError(
                "same-scope semantic continuation must use --resume"
            )
        initialization_mode = "resume"

    criterion = payload["criterion"]
    if _scalar(criterion, "criterion_train_mode_code") != MODE_CODE:
        raise SemanticProbeError("semantic criterion train-mode code mismatch")
    if _scalar(criterion, "criterion_scope_code") != SCOPE_CODE:
        raise SemanticProbeError("semantic criterion scope code mismatch")
    if _scalar(criterion, "criterion_confidence_objective_code") != 2:
        raise SemanticProbeError("semantic criterion P3 objective code mismatch")
    if _scalar(criterion, "criterion_queue_size") != 512:
        raise SemanticProbeError("semantic criterion queue-size contract mismatch")
    if _scalar(criterion, "criterion_queue_min_count") != 256:
        raise SemanticProbeError("semantic criterion queue warmup contract mismatch")
    if not math.isclose(
        _scalar_float(criterion, "criterion_positive_trust_margin"),
        0.02,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise SemanticProbeError("semantic criterion positive-trust margin mismatch")
    if not math.isclose(
        _scalar_float(criterion, "criterion_positive_trust_weight"),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise SemanticProbeError("semantic criterion positive-trust weight mismatch")
    queue = _validate_queue(criterion, require_warm=bool(exact and iteration >= 50))

    groups = payload["optimizer"].get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1:
        raise SemanticProbeError("semantic optimizer must have one parameter group")
    group = groups[0]
    if group.get("stage_b_gdino_branch") != "confidence" or not math.isclose(
        float(group.get("lr", math.nan)), LEARNING_RATE, rel_tol=0.0, abs_tol=1e-12
    ):
        raise SemanticProbeError("semantic optimizer does not exclusively own the gate at LR 3e-4")

    record = checkpoint_record(checkpoint)
    initial = preflight.get("initial_checkpoint")
    if not isinstance(initial, Mapping):
        raise SemanticProbeError("semantic preflight has no initial checkpoint")
    if record.get("base_model_sha256") != initial.get("base_model_sha256"):
        raise SemanticProbeError("semantic phase changed frozen GDINO parameters")
    if record.get("rank_sha256") != initial.get("rank_sha256"):
        raise SemanticProbeError("semantic phase changed the frozen rank branch")
    if record.get("confidence_sha256") == initial.get("confidence_sha256"):
        raise SemanticProbeError("semantic phase did not change the confidence branch")
    if record.get("confidence_final_zero") is True:
        raise SemanticProbeError("semantic confidence output layer remains zero")
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
        raise SemanticProbeError("previous semantic milestone audit is invalid")
    return value


def _validate_previous_progress(
    previous: Mapping[str, Any] | None, record: Mapping[str, Any]
) -> None:
    if previous is None:
        return
    previous_record = previous.get("checkpoint")
    if not isinstance(previous_record, Mapping):
        raise SemanticProbeError("previous semantic audit has no checkpoint record")
    if previous_record.get("rank_sha256") != record.get("rank_sha256"):
        raise SemanticProbeError("rank branch drifted between semantic milestones")
    if previous_record.get("confidence_sha256") == record.get("confidence_sha256"):
        raise SemanticProbeError("confidence branch did not update between milestones")


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
            raise SemanticProbeError(
                "semantic segment records a previous milestone but none was supplied"
            )
        return None
    if not isinstance(recorded, Mapping) or not _same_file_identity(
        file_record(requested), recorded
    ):
        raise SemanticProbeError(
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
    if target not in SEMANTIC_MILESTONES:
        raise SemanticProbeError(
            f"segment target must be one of {SEMANTIC_MILESTONES}"
        )
    preflight = read_json(preflight_path)
    if (
        preflight.get("schema") != SCHEMA
        or preflight.get("kind") != "phase_preflight"
        or preflight.get("phase") != "semantic-confidence"
    ):
        raise SemanticProbeError("segment lineage preflight is invalid")
    source_record = file_record(source_checkpoint)
    ancestry = ""
    if recovery_inspection_path is not None:
        if initialization_mode != "resume":
            raise SemanticProbeError("semantic recovery segment must use --resume")
        inspection = read_json(recovery_inspection_path)
        inspected_checkpoint = inspection.get("checkpoint")
        inspected_lineage_record = inspection.get("segment_lineage")
        if (
            inspection.get("schema") != SCHEMA
            or inspection.get("kind") != "live_checkpoint_inspection"
            or inspection.get("phase") != "confidence"
            or inspection.get("confidence_protocol")
            != "semantic_verified_topk_v1"
            or inspection.get("tn_scope") != SCOPE
            or int(inspection.get("expected_target", -1)) != target
            or not isinstance(inspected_checkpoint, Mapping)
            or not isinstance(inspected_lineage_record, Mapping)
            or source_record.get("sha256") != inspected_checkpoint.get("sha256")
            or source_record.get("size_bytes")
            != inspected_checkpoint.get("size_bytes")
        ):
            raise SemanticProbeError(
                "semantic recovery source does not match the audited live checkpoint"
            )
        inspected_path = resolve_path(str(inspected_checkpoint.get("path", "")))
        inspected_lineage_path = resolve_path(
            str(inspected_lineage_record.get("path", ""))
        )
        if not _same_file_identity(file_record(inspected_path), inspected_checkpoint):
            raise SemanticProbeError("audited live checkpoint drifted before recovery")
        if not _same_file_identity(
            file_record(inspected_lineage_path), inspected_lineage_record
        ):
            raise SemanticProbeError("recovery inspection segment lineage drifted")
        inspected_lineage = read_json(inspected_lineage_path)
        if (
            inspected_lineage.get("schema") != SCHEMA
            or inspected_lineage.get("kind") != "segment_lineage"
            or inspected_lineage.get("phase") != "confidence"
            or inspected_lineage.get("confidence_protocol")
            != "semantic_verified_topk_v1"
            or inspected_lineage.get("tn_scope") != SCOPE
            or int(inspected_lineage.get("expected_target", -1)) != target
        ):
            raise SemanticProbeError("recovery inspection ancestry is invalid")
        inspected_previous = inspected_lineage.get("previous_audit")
        if previous_audit_path is None:
            if inspected_previous is not None:
                raise SemanticProbeError("recovery dropped its previous-milestone anchor")
        elif not isinstance(inspected_previous, Mapping) or not _same_file_identity(
            file_record(previous_audit_path), inspected_previous
        ):
            raise SemanticProbeError("recovery changed its previous-milestone anchor")
        ancestry = "audited_live_recovery"
    elif previous_audit_path is not None:
        if initialization_mode != "resume":
            raise SemanticProbeError("post-milestone semantic segment must use --resume")
        previous = read_json(previous_audit_path)
        previous_checkpoint = previous.get("checkpoint")
        expected_previous = SEMANTIC_MILESTONES[
            SEMANTIC_MILESTONES.index(target) - 1
        ]
        if (
            target == SEMANTIC_MILESTONES[0]
            or previous.get("schema") != SCHEMA
            or previous.get("kind") != "milestone_checkpoint"
            or previous.get("phase") != "confidence"
            or int(previous.get("iteration", -1)) != expected_previous
            or not isinstance(previous_checkpoint, Mapping)
            or not _same_file_identity(source_record, previous_checkpoint)
        ):
            raise SemanticProbeError(
                "semantic segment source is not the exact previous milestone"
            )
        ancestry = "previous_milestone"
    else:
        initial = preflight.get("initial_checkpoint")
        if (
            target != SEMANTIC_MILESTONES[0]
            or initialization_mode != "pretrain"
            or not isinstance(initial, Mapping)
            or not _same_file_identity(source_record, initial)
        ):
            raise SemanticProbeError(
                "first semantic segment must pretrain from the exact audited R/C source"
            )
        ancestry = "phase_initial"
    return {
        "schema": SCHEMA,
        "kind": "segment_lineage",
        "phase": "confidence",
        "confidence_protocol": "semantic_verified_topk_v1",
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


def _cmd_static(_: argparse.Namespace) -> None:
    print(json.dumps(validate_static(), indent=2, sort_keys=True, ensure_ascii=True))


def _cmd_preflight(args: argparse.Namespace) -> None:
    source_checkpoint = resolve_path(args.source_checkpoint)
    source_audit = resolve_path(args.source_audit)
    payload = make_preflight(
        source_kind=args.source_kind,
        source_checkpoint=source_checkpoint,
        source_audit=source_audit,
        world_size=int(args.world_size),
        per_gpu_batch=int(args.per_gpu_batch),
    )
    output = resolve_path(args.output)
    if output.exists():
        if not args.continue_run:
            raise SemanticProbeError(f"semantic preflight already exists: {output}")
        if read_json(output) != payload:
            raise SemanticProbeError(f"semantic preflight drifted: {output}")
        print(f"[OK] unchanged semantic preflight: {output}")
        return
    if args.continue_run:
        raise SemanticProbeError(f"cannot continue without semantic preflight: {output}")
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
        raise SemanticProbeError(f"refusing to overwrite segment lineage: {output}")
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
        or lineage.get("confidence_protocol") != "semantic_verified_topk_v1"
        or lineage.get("tn_scope") != SCOPE
        or int(lineage.get("expected_target", -1)) != int(args.expected_target)
        or not isinstance(source_record, Mapping)
    ):
        raise SemanticProbeError(
            "live semantic checkpoint has no matching current-segment lineage"
        )
    source = resolve_path(str(source_record.get("path", "")))
    if not _same_file_identity(file_record(source), source_record):
        raise SemanticProbeError("semantic segment source checkpoint drifted")
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
        "confidence_protocol": "semantic_verified_topk_v1",
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
                raise SemanticProbeError(f"live checkpoint inspection drifted: {output}")
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
        or lineage.get("confidence_protocol") != "semantic_verified_topk_v1"
        or lineage.get("tn_scope") != SCOPE
        or int(lineage.get("expected_target", -1)) != iteration
        or not isinstance(source_record, Mapping)
    ):
        raise SemanticProbeError("semantic milestone segment lineage is invalid")
    source = resolve_path(str(source_record.get("path", "")))
    if not _same_file_identity(file_record(source), source_record):
        raise SemanticProbeError("semantic milestone source checkpoint drifted")
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
        "confidence_protocol": "semantic_verified_topk_v1",
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
        "code": preflight["static"]["code"],
        "orchestration": preflight["static"]["orchestration"],
        "checkpoint": result["record"],
    }
    output = resolve_path(args.output)
    if args.verify_only:
        if read_json(output) != payload:
            raise SemanticProbeError(f"semantic milestone audit drifted: {output}")
        print(f"[OK] unchanged semantic milestone {iteration}: {output}")
        return
    if output.exists():
        raise SemanticProbeError(f"refusing to overwrite semantic milestone audit: {output}")
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
            raise SemanticProbeError(f"null metadata field: {args.field}")
        print(value[args.field])
    else:
        print(json.dumps(value, sort_keys=True))


def _require_current_file_record(
    value: Any, *, label: str, expected_rows: int | None = None
) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or not value.get("path"):
        raise SemanticProbeError(f"evaluation audit is missing {label} lineage")
    current = file_record(resolve_path(str(value["path"])))
    for key in ("path", "size_bytes", "sha256"):
        if value.get(key) != current.get(key):
            raise SemanticProbeError(
                f"evaluation audit {label} {key} drifted: "
                f"audited={value.get(key)!r}, current={current.get(key)!r}"
            )
    if expected_rows is not None:
        rows = count_nonempty_lines(Path(current["path"]))
        if rows != expected_rows or int(value.get("rows", -1)) != expected_rows:
            raise SemanticProbeError(
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
        raise SemanticProbeError(
            f"semantic initial two-phase milestone failed deep replay: {error}"
        ) from error
    if verified.get("phase") != expected_phase:
        raise SemanticProbeError(
            f"semantic initial source is not an audited {expected_phase} milestone"
        )
    return dict(verified["checkpoint"])


class _SemanticLineageReplay:
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
            or self.preflight.get("phase") != "semantic-confidence"
            or source_kind not in {"rank", "dataft-confidence"}
            or not isinstance(initial, Mapping)
            or not initial.get("path")
            or not isinstance(initial_audit, Mapping)
            or not initial_audit.get("path")
            or not isinstance(launch, Mapping)
        ):
            raise SemanticProbeError("semantic formal preflight is incomplete")
        initial_path = resolve_path(str(initial["path"]))
        initial_audit_path = resolve_path(str(initial_audit["path"]))
        expected_phase = "rank" if source_kind == "rank" else "confidence"
        generic_record = _verify_two_phase_source_milestone(
            checkpoint_path=initial_path,
            audit_path=initial_audit_path,
            expected_phase=expected_phase,
        )
        if generic_record != initial:
            raise SemanticProbeError(
                "semantic preflight initial checkpoint differs from its deep audit"
            )
        current = make_preflight(
            source_kind=str(source_kind),
            source_checkpoint=initial_path,
            source_audit=initial_audit_path,
            world_size=int(launch.get("world_size", -1)),
            per_gpu_batch=int(launch.get("per_gpu_batch", -1)),
        )
        if current != self.preflight or self.preflight.get("static") != self.static:
            raise SemanticProbeError(
                "semantic config/data/code/initial lineage drifted since preflight"
            )

    @staticmethod
    def _previous_iteration(target: int) -> int | None:
        index = SEMANTIC_MILESTONES.index(target)
        return None if index == 0 else int(SEMANTIC_MILESTONES[index - 1])

    def verify_milestone(
        self, audit_path: Path, *, expected_checkpoint: Path | None = None
    ) -> Dict[str, Any]:
        audit_path = audit_path.resolve()
        if audit_path in self.milestone_cache:
            result = self.milestone_cache[audit_path]
            if expected_checkpoint is not None and Path(
                result["checkpoint"]["path"]
            ).resolve() != expected_checkpoint.resolve():
                raise SemanticProbeError("semantic milestone checkpoint path forked")
            return result
        if audit_path in self._milestone_stack:
            raise SemanticProbeError("semantic previous-milestone lineage contains a cycle")
        self._milestone_stack.add(audit_path)
        try:
            audit = read_json(audit_path)
            iteration = int(audit.get("iteration", -1))
            if (
                audit.get("schema") != SCHEMA
                or audit.get("kind") != "milestone_checkpoint"
                or audit.get("phase") != "confidence"
                or audit.get("confidence_protocol")
                != "semantic_verified_topk_v1"
                or audit.get("tn_scope") != SCOPE
                or iteration not in SEMANTIC_MILESTONES
                or int(audit.get("global_batch", -1)) != 8
            ):
                raise SemanticProbeError("semantic milestone protocol is invalid")
            if audit.get("preflight") != self.preflight_record:
                raise SemanticProbeError("semantic milestone changed its preflight")

            previous_value = audit.get("previous_audit")
            expected_previous_iteration = self._previous_iteration(iteration)
            previous_path: Path | None = None
            previous_result: Dict[str, Any] | None = None
            if expected_previous_iteration is None:
                if previous_value is not None:
                    raise SemanticProbeError("first semantic milestone has a previous audit")
            else:
                previous_record = _require_current_file_record(
                    previous_value, label="previous semantic milestone"
                )
                previous_path = Path(previous_record["path"])
                previous_result = self.verify_milestone(previous_path)
                if int(previous_result["audit"]["iteration"]) != expected_previous_iteration:
                    raise SemanticProbeError(
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
                raise SemanticProbeError("semantic milestone source differs from its segment")

            checkpoint_value = audit.get("checkpoint")
            if not isinstance(checkpoint_value, Mapping) or not checkpoint_value.get("path"):
                raise SemanticProbeError("semantic milestone has no checkpoint record")
            checkpoint_path = resolve_path(str(checkpoint_value["path"]))
            if expected_checkpoint is not None and checkpoint_path != expected_checkpoint.resolve():
                raise SemanticProbeError(
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
                raise SemanticProbeError("semantic milestone checkpoint content drifted")
            previous_audit = previous_result["audit"] if previous_result else None
            _validate_previous_progress(previous_audit, result["record"])
            expected = {
                "schema": SCHEMA,
                "kind": "milestone_checkpoint",
                "phase": "confidence",
                "confidence_protocol": "semantic_verified_topk_v1",
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
                "code": self.static["code"],
                "orchestration": self.static["orchestration"],
                "checkpoint": result["record"],
            }
            if audit != expected:
                raise SemanticProbeError("semantic milestone payload failed replay")
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
                raise SemanticProbeError("semantic segment was reused with different ancestry")
            return cached
        if lineage_path in self._segment_stack:
            raise SemanticProbeError("semantic recovery segment lineage contains a cycle")
        self._segment_stack.add(lineage_path)
        try:
            lineage = read_json(lineage_path)
            source_value = lineage.get("source_checkpoint")
            if (
                lineage.get("schema") != SCHEMA
                or lineage.get("kind") != "segment_lineage"
                or lineage.get("phase") != "confidence"
                or lineage.get("confidence_protocol")
                != "semantic_verified_topk_v1"
                or lineage.get("tn_scope") != SCOPE
                or int(lineage.get("expected_target", -1)) != target
                or lineage.get("preflight") != self.preflight_record
                or not isinstance(source_value, Mapping)
            ):
                raise SemanticProbeError("semantic segment protocol is invalid")
            expected_previous_record = (
                file_record(previous_path) if previous_path is not None else None
            )
            if lineage.get("previous_audit") != expected_previous_record:
                raise SemanticProbeError("semantic segment previous milestone is disconnected")
            source_record = _require_current_file_record(
                source_value, label="semantic segment source"
            )
            source_path = Path(source_record["path"])
            recovery_value = lineage.get("recovery_inspection")
            mode = lineage.get("initialization_mode")
            ancestry = lineage.get("ancestry")

            if recovery_value is not None:
                if mode != "resume" or ancestry != "audited_live_recovery":
                    raise SemanticProbeError("semantic recovery segment mode is invalid")
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
                    raise SemanticProbeError(
                        "semantic segment did not resume from the previous milestone"
                    )
            else:
                initial = self.preflight["initial_checkpoint"]
                if (
                    target != SEMANTIC_MILESTONES[0]
                    or mode != "pretrain"
                    or ancestry != "phase_initial"
                    or not _same_file_identity(source_record, initial)
                ):
                    raise SemanticProbeError(
                        "first semantic segment did not pretrain from preflight initial"
                    )

            expected = {
                "schema": SCHEMA,
                "kind": "segment_lineage",
                "phase": "confidence",
                "confidence_protocol": "semantic_verified_topk_v1",
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
                raise SemanticProbeError("semantic segment payload failed replay")
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
                raise SemanticProbeError("semantic live inspection was reused inconsistently")
            return cached
        if inspection_path in self._inspection_stack:
            raise SemanticProbeError("semantic recovery inspection lineage contains a cycle")
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
                != "semantic_verified_topk_v1"
                or inspection.get("tn_scope") != SCOPE
                or int(inspection.get("expected_target", -1)) != target
                or not isinstance(prior_segment_value, Mapping)
                or not isinstance(audited_checkpoint, Mapping)
            ):
                raise SemanticProbeError("semantic recovery inspection protocol is invalid")
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
                raise SemanticProbeError(
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
                "confidence_protocol": "semantic_verified_topk_v1",
                "tn_scope": SCOPE,
                "iteration": result["iteration"],
                "expected_target": target,
                "initialization_mode": result["initialization_mode"],
                "queue": result["queue"],
                "segment_lineage": prior_segment_record,
                "checkpoint": aliased_record,
            }
            if inspection != expected:
                raise SemanticProbeError("semantic recovery inspection payload failed replay")
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


def verify_evaluation_checkpoint(
    *, checkpoint_path: Path, audit_path: Path
) -> Dict[str, Any]:
    checkpoint_path = checkpoint_path.resolve()
    audit_path = audit_path.resolve()
    audit = read_json(audit_path)
    preflight_value = audit.get("preflight")
    if not isinstance(preflight_value, Mapping) or not preflight_value.get("path"):
        raise SemanticProbeError("semantic evaluation audit has no preflight")
    preflight_record = _require_current_file_record(
        preflight_value, label="semantic preflight"
    )
    replay = _SemanticLineageReplay(Path(preflight_record["path"]))
    verified = replay.verify_milestone(
        audit_path, expected_checkpoint=checkpoint_path
    )
    audit = verified["audit"]
    segment = verified["segment"]
    recovery_value = segment["lineage"].get("recovery_inspection")
    return {
        "schema": SCHEMA,
        "kind": "evaluation_checkpoint_verification",
        "phase": "confidence",
        "confidence_protocol": "semantic_verified_topk_v1",
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
    static.set_defaults(func=_cmd_static)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument(
        "--source-kind", choices=("rank", "dataft-confidence"), required=True
    )
    preflight.add_argument("--source-checkpoint", required=True)
    preflight.add_argument("--source-audit", required=True)
    preflight.add_argument("--world-size", type=int, default=2)
    preflight.add_argument("--per-gpu-batch", type=int, default=4)
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (SemanticProbeError, ProbeAuditError, OSError, ValueError, KeyError) as error:
        raise SystemExit(f"[FAIL] {error}") from error


if __name__ == "__main__":
    main()
