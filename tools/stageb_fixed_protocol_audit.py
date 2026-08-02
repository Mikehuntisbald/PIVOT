#!/usr/bin/env python3
"""Fail-closed audits for the fixed Stage-B data-FT comparison protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.stageb_dependency_audit import (  # noqa: E402
    DependencyAuditError,
    config_import_chain,
    local_python_dependency_paths,
)
from tools.stageb_eval_records import (  # noqa: E402
    TN_DERIVATION_ALGORITHM,
    TN_DERIVED_MANIFEST_BINDING_SCHEMA,
    load_tn_derived_manifest_binding,
    tn_manifest_derivation_contract,
)
from util.path_compat import remap_legacy_path  # noqa: E402


BASELINE_CONFIG = Path(
    "config/ablations/cfg_stageb_from_gdino_ft_with_tn_alltn_tau05605_w036.py"
)
BASELINE_DATASETS = Path(
    "config/ablations/gdino_ft_stage_b_rebuild_20260711/"
    "datasets_gdino_ft_stageb_with_tn_local.json"
)
BASELINE_CONFIG_SHA256 = "62fb4fa21d6db11827f8729291ebb6f9b856ff696f0d9ed2c11cec05f13f2659"
BASELINE_DATASETS_SHA256 = "906c0f6962d719ffca2c7e8d41e5dc2b6383473a4ea3aafc1f2b61ccb838cbb7"
BASELINE_CONFIG_CHAIN = {
    "config/ablations/cfg_stageb_from_gdino_ft_with_tn_alltn00625_tnneg10.py":
        "33641454865618d12757bb8a333ad207b2ec9f994d5daf88ac57b3c78b771beb",
    "config/ablations/cfg_stageb_from_gdino_ft_with_tn_alltn00625.py":
        "063e3d3c8c154fd1f6500bdbea7160f833f6d5c804259257470339fb21175d68",
    "config/ablations/cfg_stageb_from_gdino_ft_with_tn.py":
        "08bfa55af9badf39162e5a9bbf943966fbe30af424681ea5872bfc40918194cb",
    "config/cfg_odvg.py":
        "894329d1a7c0f88ef05467baec0b53954c430a5042dd035d96c300def1a17495",
}
BASELINE_LABEL_MAP = {
    "path": "data/ablations/ogc_original_finetune_stage_a_20260711/"
    "stagea_odvg_canonical_label_map.json",
    "sha256": "56bc4800ead2ad97dc7d27fee9ac515f89748ce17acaf5acb4c6664d6dffedf4",
}
FINAL_STAGEA_CHECKPOINT = Path(
    "outputs/ogc_original_finetune_stage_a_rebuild_20260711/checkpoint0001.pth"
)

EXPECTED_TRAIN_DATA = [
    {
        "path": "data/ablations/ogc_original_finetune_stage_a_20260711/stagea_odvg_train_0_lvis.jsonl",
        "rows": 99388,
        "mix_weight": 2.0,
        "sha256": "3177e55d55cda0ebde3b2905b89ddc9d26d7fd2990fbafd8229c181f22c0f136",
    },
    {
        "path": "data/ablations/ogc_original_finetune_stage_a_20260711/stagea_odvg_train_1_coco.jsonl",
        "rows": 117266,
        "mix_weight": 2.0,
        "sha256": "8c8d84b32e64e57e629d0899dfaa65ed4cda4341c7fdb25c8d74cf0aa7cfd5b5",
    },
    {
        "path": "data/ablations/gdino_ft_stage_b_rebuild_20260711/"
        "stageb_gdino_ft_0_refcocoplus_stageb_phrase_v1_vg.jsonl",
        "rows": 120191,
        "mix_weight": 2.0,
        "sha256": "015e68210d798d250e88e12d57779d56b5d47b233d2bfb6ec3582def6c379562",
    },
    {
        "path": "data/ablations/gdino_ft_stage_b_rebuild_20260711/"
        "stageb_gdino_ft_1_refcocog_stageb_phrase_v1_vg.jsonl",
        "rows": 80512,
        "mix_weight": 2.0,
        "sha256": "cd4eda88128acd1799ef707c8c31011088f7ddfe3c34c70c5fff2c2594b08c0e",
    },
    {
        "path": "data/ablations/gdino_ft_stage_b_rebuild_20260711/"
        "stageb_gdino_ft_refexp_tn_stageb_v1_vg_empty.jsonl",
        "rows": 60000,
        "mix_weight": 1.0,
        "sha256": "5275cb224bba0d81dc3657e704b5bea63544c300fb28ea9725052e68695cede5",
    },
]

STRICT_MANIFESTS = {
    "strict2031": {
        "path": "data/eval_manifests/stageb_vlm_verified_strict_ann_umd_val_20260711/eval_manifest.jsonl",
        "rows": 2031,
        "sha256": "0e47763c01178d63ee22430a6c93d4fc6a210848d43f32aafbb2e6cd7243e918",
        "source_counts": {"refcoco+_unc": 1249, "refcocog_umd": 782},
    },
    "strict1607": {
        "path": "data/eval_manifests/stageb_vlm_verified_strict_ann_umd_val_20260711/"
        "semantic_stageb_union_image_disjoint_manifest.jsonl",
        "rows": 1607,
        "sha256": "f2dc97d58884b8de3ae2c8b4cefd281432e15c8952b23b5e0252eb8e5be36d25",
        "source_counts": {"refcoco+_unc": 965, "refcocog_umd": 642},
    },
}

REF_SPLITS = [
    "refcoco_val",
    "refcoco_testA",
    "refcoco_testB",
    "refcocop_val",
    "refcocop_testA",
    "refcocop_testB",
    "refcocog_val",
    "refcocog_test",
]

EVAL_CODE_ENTRIES = (
    "tools/eval_text_groundingdino_refcoco_tn.py",
    "tools/eval_refcoco_stageb.py",
    "tools/eval_stageb_tn_val.py",
    "tools/stageb_eval_records.py",
)
EVAL_CODE_INCLUDE = (
    "datasets/__init__.py",
    "datasets/patch_episode.py",
    "models/__init__.py",
    "models/GroundingDINO/groundingdino.py",
    "models/GroundingDINO/transformer.py",
    "models/GroundingDINO/stage_b_gdino_score_adapter.py",
    "tools/stageb_fixed_protocol_audit.py",
    "tools/stageb_dependency_audit.py",
)
EVAL_ORCHESTRATION = (
    "tools/run_stageb_fixed_protocol_eval.sh",
)
ACCEPTANCE_CODE_ENTRIES = (
    "tools/make_stageb_gdino_adapter_p0.py",
    "tools/verify_stageb_dual_gate.py",
    "tools/verify_stageb_p0_record_parity.py",
    "tools/compare_stageb_fpr95_records.py",
)
ACCEPTANCE_CODE_INCLUDE = (
    "tools/stageb_eval_records.py",
)
ACCEPTANCE_ORCHESTRATION = (
    "tools/run_stageb_gdino_adapter_probe_eval.sh",
    "tools/run_stageb_fixed_dual_gate.sh",
)

TRAIN_CODE_ENTRIES = (
    "main.py",
    "engine.py",
)
TRAIN_CODE_INCLUDE = (
    "datasets/__init__.py",
    "datasets/patch_episode.py",
    "models/__init__.py",
    "models/GroundingDINO/groundingdino.py",
    "models/GroundingDINO/transformer.py",
    "models/GroundingDINO/stage_b_gdino_score_adapter.py",
    "tools/stageb_fixed_protocol_audit.py",
    "tools/stageb_dependency_audit.py",
)
TRAIN_ORCHESTRATION = (
    "tools/run_stageb_fixed_baseline.sh",
)

MODEL_ARCHITECTURE_FIELDS = (
    "modelname",
    "backbone",
    "position_embedding",
    "return_interm_indices",
    "hidden_dim",
    "enc_layers",
    "dec_layers",
    "dim_feedforward",
    "dropout",
    "nheads",
    "num_queries",
    "query_dim",
    "num_patterns",
    "num_feature_levels",
    "enc_n_points",
    "dec_n_points",
    "two_stage_type",
    "transformer_activation",
    "max_text_len",
    "text_encoder_type",
    "use_text_enhancer",
    "use_fusion_layer",
    "use_text_cross_attention",
    "sub_sentence_present",
)
ADAPTER_PROTOCOL_FIELDS = (
    "stage_b_gdino_adapter_train_mode",
    "stage_b_gdino_tn_scope",
    "stage_b_gdino_adapter_dim",
    "stage_b_gdino_gate_hidden_dim",
    "stage_b_gdino_gate_pool_temperature",
    "stage_b_gdino_gate_topk",
    "stage_b_gdino_confidence_objective",
    "stage_b_gdino_fpr_temperature",
    "stage_b_gdino_fpr_margin",
    "stage_b_gdino_paired_margin_weight",
    "stage_b_gdino_paired_margin",
    "stage_b_gdino_positive_trust_margin",
    "stage_b_gdino_positive_trust_weight",
    "stage_b_gdino_queue_size",
    "stage_b_gdino_queue_min_count",
)


class ProtocolError(RuntimeError):
    pass


def _path(value: str | Path) -> Path:
    path = remap_legacy_path(value, repo_root=REPO_ROOT)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ProtocolError(f"{label} is missing or is not a file: {path}")


def _jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ProtocolError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ProtocolError(f"expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _count_nonempty_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _file_record(path: Path) -> Dict[str, Any]:
    _require_file(path, "protocol input")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _completed_stagea_checkpoint_record(path: Path) -> Dict[str, Any]:
    """Read only checkpoint metadata; mmap avoids materializing tensor storage."""
    import torch

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ProtocolError(f"Stage-A checkpoint payload is not a mapping: {path}")
    if int(checkpoint.get("epoch", -1)) != 1:
        raise ProtocolError(
            f"Stage-A checkpoint must be the completed second epoch (epoch=1), "
            f"got epoch={checkpoint.get('epoch')!r}"
        )
    if checkpoint.get("epoch_finished") is not True:
        raise ProtocolError(
            "Stage-A checkpoint must have exact boolean epoch_finished=true, "
            f"got {checkpoint.get('epoch_finished')!r}"
        )
    if int(checkpoint.get("iteration", 0) or 0) != 0:
        raise ProtocolError(
            f"completed Stage-A checkpoint must have iteration=0, got {checkpoint.get('iteration')!r}"
        )
    model = checkpoint.get("model")
    if not isinstance(model, Mapping) or not model:
        raise ProtocolError("Stage-A checkpoint has no non-empty model state")
    record = _file_record(path)
    record.update(
        {
            "epoch": 1,
            "epoch_finished": True,
            "iteration": 0,
            "checkpoint_reason": checkpoint.get("checkpoint_reason"),
            "model_state_keys": len(model),
            "has_optimizer": isinstance(checkpoint.get("optimizer"), Mapping),
            "has_lr_scheduler": isinstance(checkpoint.get("lr_scheduler"), Mapping),
            "has_scaler": isinstance(checkpoint.get("scaler"), Mapping),
        }
    )
    del checkpoint
    return record


def _checkpoint_args(payload: Mapping[str, Any]) -> Dict[str, Any]:
    value = payload.get("args")
    if isinstance(value, Mapping):
        return dict(value)
    if value is not None and hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _completed_stageb_baseline_checkpoint_record(path: Path) -> Dict[str, Any]:
    import torch

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ProtocolError(f"Stage-B baseline checkpoint payload is not a mapping: {path}")
    if int(checkpoint.get("epoch", -1)) != 0:
        raise ProtocolError(
            f"fixed one-epoch Stage-B checkpoint must have epoch=0, got {checkpoint.get('epoch')!r}"
        )
    if checkpoint.get("epoch_finished") is not True:
        raise ProtocolError("fixed Stage-B checkpoint must have epoch_finished=true")
    if int(checkpoint.get("iteration", 0) or 0) != 0:
        raise ProtocolError("fixed Stage-B completed checkpoint must have iteration=0")
    model = checkpoint.get("model")
    if not isinstance(model, Mapping) or not model:
        raise ProtocolError("fixed Stage-B checkpoint has no non-empty model state")
    if any(str(key).startswith("stage_b_gdino_score_adapter.") for key in model):
        raise ProtocolError("fixed pure baseline checkpoint unexpectedly contains adapter parameters")
    checkpoint_args = _checkpoint_args(checkpoint)
    expected_args = {
        "batch_size": 9,
        "world_size": 2,
        "distributed": True,
        "amp": True,
        "seed": 42,
        "epochs": 1,
        "stage_b": False,
        "patch_only": False,
    }
    for key, expected in expected_args.items():
        observed = checkpoint_args.get(key)
        if observed != expected:
            raise ProtocolError(
                f"fixed Stage-B checkpoint args mismatch for {key}: expected {expected!r}, got {observed!r}"
            )
    if checkpoint_args.get("resume") not in (None, ""):
        raise ProtocolError("fixed Stage-B baseline must not be initialized with --resume")
    if not math.isclose(
        float(checkpoint_args.get("data_aug_hflip_prob", 0.5)),
        0.5,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ProtocolError(
            "fixed pure baseline must retain the original data_aug_hflip_prob=0.5"
        )
    pretrain_path = checkpoint_args.get("pretrain_model_path")
    if not pretrain_path or _path(str(pretrain_path)) != _path(FINAL_STAGEA_CHECKPOINT):
        raise ProtocolError(
            "fixed Stage-B baseline checkpoint is not linked to the final Stage-A checkpoint"
        )
    config_path = checkpoint_args.get("config_file")
    datasets_path = checkpoint_args.get("datasets")
    if not config_path or _path(str(config_path)) != _path(BASELINE_CONFIG):
        raise ProtocolError("fixed Stage-B checkpoint was produced with the wrong config")
    if not datasets_path or _path(str(datasets_path)) != _path(BASELINE_DATASETS):
        raise ProtocolError("fixed Stage-B checkpoint was produced with the wrong dataset config")
    output_path = checkpoint_args.get("output_dir")
    if not output_path or _path(str(output_path)) != path.parent.resolve():
        raise ProtocolError(
            "fixed Stage-B checkpoint output_dir does not match its checkpoint directory"
        )
    record = _file_record(path)
    record.update(
        {
            "epoch": 0,
            "epoch_finished": True,
            "iteration": 0,
            "checkpoint_reason": checkpoint.get("checkpoint_reason"),
            "model_state_keys": len(model),
            "has_optimizer": isinstance(checkpoint.get("optimizer"), Mapping),
            "has_lr_scheduler": isinstance(checkpoint.get("lr_scheduler"), Mapping),
            "has_scaler": isinstance(checkpoint.get("scaler"), Mapping),
            "checkpoint_args": {key: checkpoint_args.get(key) for key in sorted(expected_args)},
        }
    )
    del checkpoint
    return record


def _load_cfg(path: Path):
    from util.slconfig import SLConfig

    return SLConfig.fromfile(str(path))


def _config_import_records(config: Path) -> List[Dict[str, Any]]:
    try:
        paths = config_import_chain(config, root=REPO_ROOT)
    except DependencyAuditError as error:
        raise ProtocolError(str(error)) from error
    if not paths:
        raise ProtocolError(f"configuration import chain is empty: {config}")
    return [_file_record(path) for path in paths]


def _dependency_records(
    entries: Iterable[str | Path],
    *,
    include: Iterable[str | Path],
) -> List[Dict[str, Any]]:
    try:
        paths = local_python_dependency_paths(
            entries,
            root=REPO_ROOT,
            include=include,
        )
    except DependencyAuditError as error:
        raise ProtocolError(str(error)) from error
    return [_file_record(path) for path in paths]


def _acceptance_code_records() -> List[Dict[str, Any]]:
    return _dependency_records(
        ACCEPTANCE_CODE_ENTRIES,
        include=ACCEPTANCE_CODE_INCLUDE,
    )


def _acceptance_orchestration_records() -> List[Dict[str, Any]]:
    return [
        _file_record(_path(path)) for path in ACCEPTANCE_ORCHESTRATION
    ]


def _resolved_eval_contract(cfg: Any) -> Dict[str, Any]:
    architecture = {
        key: getattr(cfg, key, None) for key in MODEL_ARCHITECTURE_FIELDS
    }
    model_protocol = {
        "stage_b_gdino_score_adapter": bool(
            getattr(cfg, "stage_b_gdino_score_adapter", False)
        ),
        **{key: getattr(cfg, key, None) for key in ADAPTER_PROTOCOL_FIELDS},
        "config_patch_only": bool(getattr(cfg, "patch_only", False)),
        "config_stage_b": bool(getattr(cfg, "stage_b", False)),
        "config_enable_patch_branch": bool(
            getattr(cfg, "enable_patch_branch", False)
        ),
        "patch_only_at_eval": False,
    }
    return {
        "resolved_model_architecture": architecture,
        "model_protocol": model_protocol,
    }


def _record_hash_map(rows: Any, label: str) -> Dict[str, str]:
    if not isinstance(rows, list) or not rows:
        raise ProtocolError(f"{label} must be a non-empty file-record list")
    result: Dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ProtocolError(f"{label} row {index} is not an object")
        path = row.get("path")
        sha256 = row.get("sha256")
        if not isinstance(path, str) or not path:
            raise ProtocolError(f"{label} row {index} has no path")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ProtocolError(f"{label} row {index} has no SHA-256")
        resolved = str(_path(path))
        if resolved in result:
            raise ProtocolError(f"{label} contains duplicate path {resolved}")
        result[resolved] = sha256
    return result


def _validate_eval_runtime_contract(runtime: Any) -> None:
    if not isinstance(runtime, Mapping):
        raise ProtocolError("evaluation preflight has no runtime contract")
    fixed = {
        "topk": [1],
        "threshold_tprs": [0.75, 0.9, 0.95],
        "score_thresholds": [0.5],
        "tn_splits": ["refcocop_val", "refcocog_umd_val"],
        "ref_splits": REF_SPLITS,
    }
    for key, expected in fixed.items():
        if runtime.get(key) != expected:
            raise ProtocolError(
                f"evaluation runtime mismatch for {key}: "
                f"expected {expected!r}, got {runtime.get(key)!r}"
            )
    if not isinstance(runtime.get("device"), str) or not runtime.get("device"):
        raise ProtocolError("evaluation runtime device must be a non-empty string")
    if type(runtime.get("batch_size")) is not int or int(runtime["batch_size"]) <= 0:
        raise ProtocolError("evaluation runtime batch_size must be a positive integer")
    if type(runtime.get("num_workers")) is not int or int(runtime["num_workers"]) < 0:
        raise ProtocolError("evaluation runtime num_workers must be a non-negative integer")
    if type(runtime.get("seed")) is not int:
        raise ProtocolError("evaluation runtime seed must be an integer")
    if type(runtime.get("amp")) is not bool:
        raise ProtocolError("evaluation runtime amp must be a boolean")


def _validate_eval_preflight_dependencies(preflight: Mapping[str, Any]) -> None:
    config_record = preflight.get("config")
    if not isinstance(config_record, Mapping) or not isinstance(
        config_record.get("path"), str
    ):
        raise ProtocolError("evaluation preflight has no canonical config record")
    config = _path(str(config_record["path"]))
    if dict(config_record) != _file_record(config):
        raise ProtocolError("evaluation config changed after its preflight was written")
    checkpoint_record = preflight.get("checkpoint")
    if not isinstance(checkpoint_record, Mapping) or not isinstance(
        checkpoint_record.get("path"), str
    ):
        raise ProtocolError("evaluation preflight has no canonical checkpoint record")
    checkpoint = _path(str(checkpoint_record["path"]))
    if dict(checkpoint_record) != _file_record(checkpoint):
        raise ProtocolError("evaluation checkpoint changed after its preflight was written")

    current_chain = _config_import_records(config)
    if preflight.get("config_import_chain") != current_chain:
        raise ProtocolError(
            "evaluation configuration import chain changed after its preflight was written"
        )
    current_code = _dependency_records(
        EVAL_CODE_ENTRIES,
        include=EVAL_CODE_INCLUDE,
    )
    if preflight.get("code") != current_code:
        raise ProtocolError(
            "evaluation code dependency closure changed after its preflight was written"
        )
    current_orchestration = [
        _file_record(_path(path)) for path in EVAL_ORCHESTRATION
    ]
    if preflight.get("orchestration") != current_orchestration:
        raise ProtocolError(
            "evaluation orchestration changed after its preflight was written"
        )
    if preflight.get("acceptance_code") != _acceptance_code_records():
        raise ProtocolError(
            "final acceptance code dependency closure changed after evaluation preflight"
        )
    if preflight.get(
        "acceptance_orchestration"
    ) != _acceptance_orchestration_records():
        raise ProtocolError(
            "final acceptance orchestration changed after evaluation preflight"
        )

    current_contract = _resolved_eval_contract(_load_cfg(config))
    for key, expected in current_contract.items():
        if preflight.get(key) != expected:
            raise ProtocolError(
                f"evaluation resolved configuration drifted in {key}"
            )
    _validate_eval_runtime_contract(preflight.get("runtime"))
    if preflight.get(
        "tn_manifest_derivation_contract"
    ) != tn_manifest_derivation_contract():
        raise ProtocolError(
            "evaluation preflight has no current two-layer TN manifest derivation "
            "contract; legacy derived evaluation artifacts must be rerun"
        )
    if preflight.get("strict_manifests") != _validate_strict_manifests():
        raise ProtocolError(
            "locked TN source manifests changed after evaluation preflight"
        )


def _shared_config_parent_hashes(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> Dict[str, str]:
    baseline_map = _record_hash_map(
        baseline.get("config_import_chain"), "baseline config import chain"
    )
    candidate_map = _record_hash_map(
        candidate.get("config_import_chain"), "candidate config import chain"
    )
    leaf_paths: set[str] = set()
    for label, payload in (("baseline", baseline), ("candidate", candidate)):
        config_record = payload.get("config")
        if not isinstance(config_record, Mapping) or not isinstance(
            config_record.get("path"), str
        ):
            raise ProtocolError(f"{label} evaluation has no config record")
        leaf_paths.add(str(_path(str(config_record["path"]))))
    shared = sorted(set(baseline_map).intersection(candidate_map).difference(leaf_paths))
    if not shared:
        raise ProtocolError(
            "baseline/candidate configurations have no shared imported parent config"
        )
    result: Dict[str, str] = {}
    for path in shared:
        if baseline_map[path] != candidate_map[path]:
            raise ProtocolError(
                "baseline/candidate shared parent config hash differs for "
                f"{path}: {baseline_map[path]} != {candidate_map[path]}"
            )
        result[path] = baseline_map[path]
    return result


def _validate_baseline_config() -> Dict[str, Any]:
    config_path = _path(BASELINE_CONFIG)
    datasets_path = _path(BASELINE_DATASETS)
    _require_file(config_path, "fixed baseline config")
    _require_file(datasets_path, "fixed baseline dataset config")
    config_sha = _sha256(config_path)
    datasets_sha = _sha256(datasets_path)
    if config_sha != BASELINE_CONFIG_SHA256:
        raise ProtocolError(
            "fixed baseline config drifted: "
            f"expected {BASELINE_CONFIG_SHA256}, got {config_sha} ({config_path})"
        )
    if datasets_sha != BASELINE_DATASETS_SHA256:
        raise ProtocolError(
            "fixed baseline dataset config drifted: "
            f"expected {BASELINE_DATASETS_SHA256}, got {datasets_sha} ({datasets_path})"
        )

    config_chain: List[Dict[str, Any]] = []
    for relative, expected_sha in BASELINE_CONFIG_CHAIN.items():
        path = _path(relative)
        _require_file(path, "fixed baseline parent config")
        actual_sha = _sha256(path)
        if actual_sha != expected_sha:
            raise ProtocolError(
                f"fixed baseline parent config drifted: expected {expected_sha}, "
                f"got {actual_sha} ({path})"
            )
        config_chain.append(_file_record(path))

    cfg = _load_cfg(config_path)
    expected_cfg = {
        "patch_only": False,
        "stage_b": False,
        "enable_patch_branch": False,
        "epochs": 1,
        "batch_size": 18,
        "lr": 1.0e-4,
        "lr_backbone": 1.0e-5,
        "freeze_keywords": ["bert"],
        "skip_eval": True,
        "gdino_tn_loss_type": "alltn00625",
        "gdino_tn_alltn_weight": 0.36,
        "gdino_tn_alltn_tau_neg": 0.5605,
        "gdino_tn_token_neg_weight_mode": "token_count",
    }
    observed_cfg: Dict[str, Any] = {}
    for key, expected in expected_cfg.items():
        observed = getattr(cfg, key, None)
        observed_cfg[key] = observed
        if observed != expected:
            raise ProtocolError(
                f"fixed baseline config mismatch for {key}: expected {expected!r}, got {observed!r}"
            )
    resolved_hflip = float(getattr(cfg, "data_aug_hflip_prob", 0.5))
    if not math.isclose(resolved_hflip, 0.5, rel_tol=0.0, abs_tol=1e-12):
        raise ProtocolError(
            "fixed pure baseline must retain the original data_aug_hflip_prob=0.5"
        )
    observed_cfg["data_aug_hflip_prob"] = resolved_hflip
    if bool(getattr(cfg, "stage_b_gdino_score_adapter", False)):
        raise ProtocolError("fixed pure baseline must not enable the GDINO score adapter")

    dataset_payload = json.loads(datasets_path.read_text(encoding="utf-8"))
    train_rows = dataset_payload.get("train")
    if not isinstance(train_rows, list) or len(train_rows) != len(EXPECTED_TRAIN_DATA):
        raise ProtocolError(
            f"fixed baseline must contain exactly {len(EXPECTED_TRAIN_DATA)} train sources"
        )
    source_records: List[Dict[str, Any]] = []
    for index, (row, expected) in enumerate(zip(train_rows, EXPECTED_TRAIN_DATA)):
        if not isinstance(row, dict):
            raise ProtocolError(f"train source {index} is not an object")
        actual_path = _path(str(row.get("anno", "")))
        expected_path = _path(expected["path"])
        if actual_path != expected_path:
            raise ProtocolError(
                f"train source {index} mismatch: expected {expected_path}, got {actual_path}"
            )
        actual_weight = float(row.get("mix_weight", 1.0))
        if actual_weight != float(expected["mix_weight"]):
            raise ProtocolError(
                f"train source {index} mix_weight mismatch: "
                f"expected {expected['mix_weight']}, got {actual_weight}"
            )
        _require_file(actual_path, f"train source {index}")
        actual_rows = _count_nonempty_lines(actual_path)
        actual_sha = _sha256(actual_path)
        if actual_rows != int(expected["rows"]):
            raise ProtocolError(
                f"train source {index} row count mismatch: expected {expected['rows']}, got {actual_rows}"
            )
        if actual_sha != str(expected["sha256"]):
            raise ProtocolError(
                f"train source {index} SHA-256 mismatch: expected {expected['sha256']}, got {actual_sha}"
            )
        source_records.append(
            {
                "index": index,
                "path": str(actual_path),
                "rows": actual_rows,
                "mix_weight": actual_weight,
                "sha256": actual_sha,
            }
        )

    label_map_path = _path(BASELINE_LABEL_MAP["path"])
    _require_file(label_map_path, "fixed baseline label map")
    label_map_sha = _sha256(label_map_path)
    if label_map_sha != BASELINE_LABEL_MAP["sha256"]:
        raise ProtocolError(
            "fixed baseline label map drifted: "
            f"expected {BASELINE_LABEL_MAP['sha256']}, got {label_map_sha}"
        )
    for index in (0, 1):
        row_label_map = _path(str(train_rows[index].get("label_map", "")))
        if row_label_map != label_map_path:
            raise ProtocolError(
                f"train source {index} must use the locked canonical label map {label_map_path}"
            )

    total_rows = sum(int(row["rows"]) for row in source_records)
    return {
        "config": _file_record(config_path),
        "config_chain": config_chain,
        "datasets": _file_record(datasets_path),
        "label_map": _file_record(label_map_path),
        "resolved_config": observed_cfg,
        "train_sources": source_records,
        "total_train_rows": total_rows,
        "mix_weight_sum": sum(float(row["mix_weight"]) for row in source_records),
    }


def _validate_strict_manifests() -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    sample_sets: Dict[str, set[str]] = {}
    for label, expected in STRICT_MANIFESTS.items():
        path = _path(expected["path"])
        _require_file(path, label)
        actual_sha = _sha256(path)
        rows = _jsonl_rows(path)
        if actual_sha != expected["sha256"]:
            raise ProtocolError(
                f"{label} SHA-256 mismatch: expected {expected['sha256']}, got {actual_sha}"
            )
        if len(rows) != int(expected["rows"]):
            raise ProtocolError(
                f"{label} row count mismatch: expected {expected['rows']}, got {len(rows)}"
            )
        source_counts = Counter(str(row.get("tn_eval_pair_source")) for row in rows)
        if dict(source_counts) != dict(expected["source_counts"]):
            raise ProtocolError(
                f"{label} source counts mismatch: expected {expected['source_counts']}, got {dict(source_counts)}"
            )
        sample_ids = [str(row.get("sample_id", "")) for row in rows]
        if any(not sample_id for sample_id in sample_ids) or len(set(sample_ids)) != len(sample_ids):
            raise ProtocolError(f"{label} has missing or duplicate sample_id values")
        for index, row in enumerate(rows):
            if row.get("manifest_schema") != "stageb_vlm_verified_strict_tn_v2":
                raise ProtocolError(f"{label} row {index} has the wrong manifest_schema")
            if row.get("visual_verified_negative") is not True or row.get("coverage_pass") is not True:
                raise ProtocolError(f"{label} row {index} is not an exact verified negative")
            if str(row.get("tn_eval_pair_source")) not in {"refcoco+_unc", "refcocog_umd"}:
                raise ProtocolError(f"{label} row {index} has an unsupported TN source")
        sample_sets[label] = set(sample_ids)
        result[label] = {
            "path": str(path),
            "rows": len(rows),
            "size_bytes": int(path.stat().st_size),
            "sha256": actual_sha,
            "source_counts": dict(source_counts),
        }
    if not sample_sets["strict1607"].issubset(sample_sets["strict2031"]):
        raise ProtocolError("strict1607 must be a sample-id subset of strict2031")
    return result


def _validate_eval_data_root(data_root: Path) -> Dict[str, Any]:
    required = [
        "canonical_classes_with_aliases.json",
        "COCO/refcoco/instances.json",
        "COCO/refcoco/refs(unc).p",
        "COCO/refcoco+/instances.json",
        "COCO/refcoco+/refs(unc).p",
        "COCO/refcocog/instances.json",
        "COCO/refcocog/refs(umd).p",
        "patches_quality_emb/emb_index_from_quality.tsv",
    ]
    records: List[Dict[str, Any]] = []
    for relative in required:
        path = data_root / relative
        _require_file(path, f"evaluation data prerequisite {relative}")
        records.append({"path": str(path.resolve()), "size_bytes": path.stat().st_size})
    return {"root": str(data_root.resolve()), "required_files": records}


def _record_file_audit(path: Path, *, expected_task: str, expected_key: str | None = None) -> Dict[str, Any]:
    rows = _jsonl_rows(path)
    if not rows:
        raise ProtocolError(f"evaluation record file is empty: {path}")
    keys = {str(row.get("manifest_key", "")) for row in rows}
    tasks = {str(row.get("task", "")) for row in rows}
    hashes = {str(row.get("manifest_sha256", "")) for row in rows}
    sizes = {int(row.get("manifest_n", -1)) for row in rows}
    indices = [int(row.get("manifest_index", -1)) for row in rows]
    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    invalid = sum(row.get("valid") is not True for row in rows)
    if any(row.get("schema") != "stageb-eval-record-v1" for row in rows):
        raise ProtocolError(f"{path} contains a non-canonical evaluation record schema")
    if tasks != {expected_task}:
        raise ProtocolError(f"{path} task mismatch: expected {expected_task}, got {sorted(tasks)}")
    if len(keys) != 1 or (expected_key is not None and keys != {expected_key}):
        raise ProtocolError(f"{path} manifest_key mismatch: got {sorted(keys)}")
    if len(hashes) != 1 or any(len(value) != 64 for value in hashes):
        raise ProtocolError(f"{path} has inconsistent manifest hashes")
    if sizes != {len(rows)}:
        raise ProtocolError(f"{path} record count does not match manifest_n: {sorted(sizes)}")
    if indices != list(range(len(rows))):
        raise ProtocolError(f"{path} manifest indices are not exact 0..N-1 order")
    if any(not value for value in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ProtocolError(f"{path} has missing or duplicate sample IDs")
    if invalid:
        raise ProtocolError(f"{path} has {invalid} invalid records; final comparison requires zero")
    for index, row in enumerate(rows):
        if expected_task == "tn":
            values = (row.get("pos_score"), row.get("neg_score"))
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in values
            ):
                raise ProtocolError(f"{path} row {index} has invalid TN scores")
        else:
            iou = row.get("top1_iou")
            if (
                isinstance(iou, bool)
                or not isinstance(iou, (int, float))
                or not math.isfinite(float(iou))
                or type(row.get("correct50")) is not bool
            ):
                raise ProtocolError(f"{path} row {index} has invalid Ref metrics")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "task": expected_task,
        "manifest_key": next(iter(keys)),
        "manifest_sha256": next(iter(hashes)),
        "rows": len(rows),
        "invalid": invalid,
        "sample_ids": sample_ids,
        "image_ids": [int(row.get("image_id", -1)) for row in rows],
    }


def _audit_fixed_tn_manifest_binding(
    *,
    output_dir: Path,
    label: str,
    records_path: Path,
    record_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    rows = _jsonl_rows(records_path)
    if not rows:
        raise ProtocolError(f"{label} TN records are empty")
    binding_paths = {str(row.get("manifest_binding_path", "")) for row in rows}
    binding_hashes = {str(row.get("manifest_binding_sha256", "")) for row in rows}
    if len(binding_paths) != 1 or not next(iter(binding_paths)):
        raise ProtocolError(
            f"{label} records have no unique derived-manifest binding path; "
            "legacy derived artifacts must be rerun"
        )
    binding_path = _path(next(iter(binding_paths)))
    expected_derived = (
        output_dir
        / label
        / "tn_eval_inputs"
        / "tn_refcocop_val_refcocog_umd_val.jsonl"
    ).resolve()
    expected_binding_path = expected_derived.with_suffix(
        expected_derived.suffix + ".binding.json"
    )
    if binding_path != expected_binding_path:
        raise ProtocolError(
            f"{label} TN binding path is not the fixed evaluator output: {binding_path}"
        )
    try:
        binding = load_tn_derived_manifest_binding(
            binding_path, expected_derived_manifest=expected_derived
        )
    except (OSError, TypeError, ValueError) as error:
        raise ProtocolError(f"{label} TN manifest binding failed: {error}") from error
    if binding_hashes != {binding.sha256}:
        raise ProtocolError(f"{label} TN records declare the wrong binding hash")

    expected_source_path = _path(STRICT_MANIFESTS[label]["path"])
    expected_source_rows = _jsonl_rows(expected_source_path)
    expected_source = {
        **_file_record(expected_source_path),
        "rows": len(expected_source_rows),
    }
    if dict(binding.source_manifest) != expected_source:
        raise ProtocolError(
            f"{label} TN binding does not reference the current locked source manifest"
        )
    if str(binding.source_manifest["sha256"]) != str(
        STRICT_MANIFESTS[label]["sha256"]
    ):
        raise ProtocolError(f"{label} locked source manifest SHA-256 mismatch")
    if int(binding.source_manifest["rows"]) != int(STRICT_MANIFESTS[label]["rows"]):
        raise ProtocolError(f"{label} locked source manifest row count mismatch")
    if int(binding.derived_manifest["rows"]) != int(STRICT_MANIFESTS[label]["rows"]):
        raise ProtocolError(f"{label} derived TN manifest row count mismatch")
    derivation = binding.derivation
    expected_runtime_derivation = {
        "requested_splits": ["refcocop_val", "refcocog_umd_val"],
        "max_pairs": 0,
        "max_pairs_per_split": 0,
        "holdout_level": "none",
    }
    for key, expected in expected_runtime_derivation.items():
        if derivation.get(key) != expected:
            raise ProtocolError(
                f"{label} TN derivation changed {key}: expected {expected!r}, "
                f"got {derivation.get(key)!r}"
            )
    source_indices = [int(row["source_index"]) for row in binding.row_mapping]
    if source_indices != list(range(len(expected_source_rows))):
        raise ProtocolError(
            f"{label} fixed TN derivation must preserve every locked source row in order"
        )
    derived_hash = str(binding.derived_manifest["sha256"])
    derived_size = int(binding.derived_manifest["size_bytes"])
    for index, (row, mapping) in enumerate(zip(rows, binding.row_mapping)):
        source_row = expected_source_rows[int(mapping["source_index"])]
        expected_fields = {
            "manifest_sha256": derived_hash,
            "manifest_path": str(binding.derived_manifest["path"]),
            "manifest_size_bytes": derived_size,
            "manifest_binding_schema": TN_DERIVED_MANIFEST_BINDING_SCHEMA,
            "manifest_derivation_algorithm": TN_DERIVATION_ALGORITHM,
            "manifest_binding_path": str(binding.path),
            "manifest_binding_sha256": binding.sha256,
            "manifest_binding_size_bytes": binding.size_bytes,
            "manifest_row_mapping_sha256": binding.row_mapping_sha256,
            "source_manifest_path": str(binding.source_manifest["path"]),
            "source_manifest_sha256": str(binding.source_manifest["sha256"]),
            "source_manifest_size_bytes": int(binding.source_manifest["size_bytes"]),
            "source_manifest_n": int(binding.source_manifest["rows"]),
            "source_manifest_index": int(mapping["source_index"]),
            "sample_id": str(mapping["sample_id"]),
            "split": str(mapping["eval_split"]),
            "image_id": source_row.get("image_id"),
            "ann_id": source_row.get("ann_id"),
            "ref_id": source_row.get("ref_id"),
            "sent_id": source_row.get("sent_id"),
        }
        for field, expected in expected_fields.items():
            if row.get(field) != expected:
                raise ProtocolError(
                    f"{label} TN record {index}.{field} does not match its "
                    "source/derived manifest binding"
                )
    if str(record_audit["manifest_sha256"]) != derived_hash:
        raise ProtocolError(f"{label} TN records do not bind the derived data manifest")
    return {
        "schema": TN_DERIVED_MANIFEST_BINDING_SCHEMA,
        "algorithm": TN_DERIVATION_ALGORITHM,
        "source_manifest": dict(binding.source_manifest),
        "derived_manifest": dict(binding.derived_manifest),
        "binding": _file_record(binding.path),
        "row_mapping_sha256": binding.row_mapping_sha256,
        "row_mapping_verified_against_source_and_derived": True,
        "all_locked_source_rows_preserved_in_order": True,
    }


def _audit_eval_outputs(output_dir: Path) -> Dict[str, Any]:
    summaries = {}
    for name in ("ref8", "strict2031", "strict1607"):
        summary_path = output_dir / name / "summary.json"
        _require_file(summary_path, f"{name} summary")
        summary_payload = _read_json_object(summary_path, f"{name} summary")
        if not summary_payload:
            raise ProtocolError(f"{name} summary must be a non-empty JSON object")
        summaries[name] = _file_record(summary_path)

    ref_record_dir = output_dir / "ref8" / "per_example_records"
    ref_files = sorted(ref_record_dir.glob("*.records.jsonl"))
    if len(ref_files) != len(REF_SPLITS):
        raise ProtocolError(
            f"ref8 must contain exactly {len(REF_SPLITS)} record files, found {len(ref_files)}"
        )
    ref_audits: Dict[str, Any] = {}
    for path in ref_files:
        first = _jsonl_rows(path)[:1]
        if not first:
            raise ProtocolError(f"empty Ref record file: {path}")
        key = str(first[0].get("manifest_key", ""))
        split = key.removeprefix("ref:") if key.startswith("ref:") else ""
        if split not in REF_SPLITS:
            raise ProtocolError(f"unexpected Ref manifest key {key!r} in {path}")
        if split in ref_audits:
            raise ProtocolError(f"duplicate Ref record group for {split}")
        ref_audits[split] = _record_file_audit(
            path, expected_task="ref", expected_key=f"ref:{split}"
        )
    if set(ref_audits) != set(REF_SPLITS):
        raise ProtocolError(
            f"Ref record groups mismatch: expected {REF_SPLITS}, got {sorted(ref_audits)}"
        )

    tn_audits: Dict[str, Any] = {}
    for label in ("strict2031", "strict1607"):
        record_files = sorted((output_dir / label / "per_example_records").glob("*.records.jsonl"))
        if len(record_files) != 1:
            raise ProtocolError(f"{label} must contain exactly one TN record file, found {len(record_files)}")
        audit = _record_file_audit(
            record_files[0], expected_task="tn", expected_key="tn_global"
        )
        source_rows = _jsonl_rows(_path(STRICT_MANIFESTS[label]["path"]))
        expected_ids = [str(row["sample_id"]) for row in source_rows]
        expected_images = [int(row["image_id"]) for row in source_rows]
        if audit["sample_ids"] != expected_ids or audit["image_ids"] != expected_images:
            raise ProtocolError(f"{label} output records drifted from the locked input manifest order")
        if audit["rows"] != int(STRICT_MANIFESTS[label]["rows"]):
            raise ProtocolError(f"{label} output record count mismatch")
        audit["manifest_binding"] = _audit_fixed_tn_manifest_binding(
            output_dir=output_dir,
            label=label,
            records_path=record_files[0],
            record_audit=audit,
        )
        summary_payload = _read_json_object(
            output_dir / label / "summary.json", f"{label} summary"
        )
        tn_rows = summary_payload.get("tn")
        ref_rows = summary_payload.get("refcoco")
        if (
            not isinstance(ref_rows, list)
            or ref_rows
            or not isinstance(tn_rows, list)
            or len(tn_rows) != 1
            or not isinstance(tn_rows[0], Mapping)
        ):
            raise ProtocolError(
                f"{label} summary must contain exactly one TN row and no Ref rows"
            )
        summary_row = tn_rows[0]
        binding_audit = audit["manifest_binding"]
        source_manifest = binding_audit["source_manifest"]
        derived_manifest = binding_audit["derived_manifest"]
        binding_file = binding_audit["binding"]
        expected_summary_fields = {
            "manifest_n": int(derived_manifest["rows"]),
            "manifest_path": str(derived_manifest["path"]),
            "manifest_sha256": str(derived_manifest["sha256"]),
            "manifest_size_bytes": int(derived_manifest["size_bytes"]),
            "manifest_binding_schema": TN_DERIVED_MANIFEST_BINDING_SCHEMA,
            "manifest_derivation_algorithm": TN_DERIVATION_ALGORITHM,
            "manifest_binding_path": str(binding_file["path"]),
            "manifest_binding_sha256": str(binding_file["sha256"]),
            "manifest_binding_size_bytes": int(binding_file["size_bytes"]),
            "manifest_row_mapping_sha256": str(
                binding_audit["row_mapping_sha256"]
            ),
            "source_manifest_path": str(source_manifest["path"]),
            "source_manifest_sha256": str(source_manifest["sha256"]),
            "source_manifest_size_bytes": int(source_manifest["size_bytes"]),
            "source_manifest_n": int(source_manifest["rows"]),
            "records_jsonl": str(record_files[0].resolve()),
        }
        for field, expected in expected_summary_fields.items():
            if summary_row.get(field) != expected:
                raise ProtocolError(
                    f"{label} summary {field} does not match its two-layer "
                    "manifest binding"
                )
        audit.pop("sample_ids")
        audit.pop("image_ids")
        tn_audits[label] = audit

    for audit in ref_audits.values():
        audit.pop("sample_ids")
        audit.pop("image_ids")
    return {"summaries": summaries, "ref_records": ref_audits, "tn_records": tn_audits}


def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    _require_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError(f"could not read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ProtocolError(f"{label} must be a JSON object: {path}")
    return payload


def _verify_eval_completion(
    output_dir: Path,
    *,
    checkpoint: Path | None = None,
    config: Path | None = None,
) -> Dict[str, Any]:
    preflight_path = output_dir / "protocol_eval_preflight.json"
    complete_path = output_dir / "protocol_eval_complete.json"
    preflight = _read_json_object(preflight_path, "evaluation preflight")
    complete = _read_json_object(complete_path, "evaluation completion audit")
    if preflight.get("schema") != "stageb-fixed-protocol-v1":
        raise ProtocolError("evaluation directory has the wrong preflight schema")
    if preflight.get("kind") != "fixed_stageb_eval_preflight":
        raise ProtocolError("evaluation directory has the wrong preflight kind")
    if complete.get("schema") != "stageb-fixed-protocol-v1":
        raise ProtocolError("evaluation directory has the wrong completion schema")
    if complete.get("kind") != "fixed_stageb_eval_complete":
        raise ProtocolError("evaluation directory has the wrong completion kind")
    current_preflight_record = _file_record(preflight_path)
    if complete.get("preflight") != current_preflight_record:
        raise ProtocolError(
            "evaluation completion audit is not exactly linked to its current preflight"
        )
    _validate_eval_preflight_dependencies(preflight)
    recomputed_outputs = _audit_eval_outputs(output_dir)
    if complete.get("outputs") != recomputed_outputs:
        raise ProtocolError(
            "evaluation outputs changed after protocol_eval_complete.json was written"
        )
    for label, current, recorded in (
        ("checkpoint", checkpoint, preflight.get("checkpoint")),
        ("config", config, preflight.get("config")),
    ):
        if current is None:
            continue
        if not isinstance(recorded, Mapping) or recorded != _file_record(current):
            raise ProtocolError(
                f"reused evaluation {label} does not exactly match its preflight"
            )
    return {
        "schema": "stageb-fixed-protocol-v1",
        "kind": "fixed_stageb_eval_verified",
        "eval_dir": str(output_dir.resolve()),
        "preflight": preflight,
        "completion": _file_record(complete_path),
        "outputs": recomputed_outputs,
    }


def _static_payload(data_root: Path) -> Dict[str, Any]:
    return {
        "schema": "stageb-fixed-protocol-v1",
        "baseline": _validate_baseline_config(),
        "strict_manifests": _validate_strict_manifests(),
        "tn_manifest_derivation_contract": tn_manifest_derivation_contract(),
        "evaluation_data": _validate_eval_data_root(data_root),
        "required_ref_splits": REF_SPLITS,
    }


def _cmd_static(args: argparse.Namespace) -> None:
    payload = _static_payload(_path(args.data_root))
    if args.output:
        _write_json(_path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))


def _make_train_preflight_payload(
    *,
    stagea_checkpoint: Path,
    data_root: Path,
    world_size: int,
    per_gpu_batch: int,
    allow_nonfinal_stagea_name: bool,
) -> Dict[str, Any]:
    if int(world_size) != 2 or int(per_gpu_batch) != 9:
        raise ProtocolError(
            "fixed baseline training requires two ranks with per-GPU batch 9 "
            "(global batch 18)"
        )
    stagea_checkpoint = _path(stagea_checkpoint)
    _require_file(stagea_checkpoint, "final Stage-A checkpoint")
    expected_stagea_checkpoint = _path(FINAL_STAGEA_CHECKPOINT)
    if (
        stagea_checkpoint != expected_stagea_checkpoint
        and not allow_nonfinal_stagea_name
    ):
        raise ProtocolError(
            f"fixed baseline must initialize from {expected_stagea_checkpoint}; "
            "pass --allow_nonfinal_stagea_name only for a named diagnostic"
        )
    if stagea_checkpoint.name != "checkpoint0001.pth" and not allow_nonfinal_stagea_name:
        raise ProtocolError("fixed baseline Stage-A checkpoint must be named checkpoint0001.pth")
    payload = _static_payload(_path(data_root))
    total_rows = int(payload["baseline"]["total_train_rows"])
    global_batch = int(world_size) * int(per_gpu_batch)
    payload.update(
        {
            "kind": "fixed_pure_stageb_dataft_train_preflight",
            "initial_checkpoint": _completed_stagea_checkpoint_record(stagea_checkpoint),
            "launch": {
                "world_size": int(world_size),
                "per_gpu_batch": int(per_gpu_batch),
                "global_batch": global_batch,
                "epochs": 1,
                "expected_steps": total_rows // global_batch,
                "expected_consumed_samples": (total_rows // global_batch) * global_batch,
                "learning_rate": 1.0e-4,
                "initialization_mode": "pretrain_model_path_fresh_optimizer",
                "allow_nonfinal_stagea_name": bool(allow_nonfinal_stagea_name),
            },
            "config_import_chain": _config_import_records(_path(BASELINE_CONFIG)),
            "code": _dependency_records(
                TRAIN_CODE_ENTRIES,
                include=TRAIN_CODE_INCLUDE,
            ),
            "orchestration": [
                _file_record(_path(path)) for path in TRAIN_ORCHESTRATION
            ],
        }
    )
    return payload


def _verify_train_preflight(preflight_path: Path) -> Dict[str, Any]:
    recorded = _read_json_object(preflight_path, "fixed baseline training preflight")
    if recorded.get("schema") != "stageb-fixed-protocol-v1":
        raise ProtocolError("fixed baseline training preflight has the wrong schema")
    if recorded.get("kind") != "fixed_pure_stageb_dataft_train_preflight":
        raise ProtocolError("fixed baseline training preflight has the wrong kind")
    initial = recorded.get("initial_checkpoint")
    evaluation_data = recorded.get("evaluation_data")
    launch = recorded.get("launch")
    if not isinstance(initial, Mapping) or not isinstance(initial.get("path"), str):
        raise ProtocolError("training preflight has no initial checkpoint record")
    if not isinstance(evaluation_data, Mapping) or not isinstance(
        evaluation_data.get("root"), str
    ):
        raise ProtocolError("training preflight has no evaluation data root")
    if not isinstance(launch, Mapping):
        raise ProtocolError("training preflight has no launch contract")
    for key in ("world_size", "per_gpu_batch"):
        if type(launch.get(key)) is not int:
            raise ProtocolError(f"training preflight launch has invalid {key}")
    if type(launch.get("allow_nonfinal_stagea_name")) is not bool:
        raise ProtocolError(
            "training preflight launch has no exact allow_nonfinal_stagea_name flag"
        )

    replayed = _make_train_preflight_payload(
        stagea_checkpoint=_path(str(initial["path"])),
        data_root=_path(str(evaluation_data["root"])),
        world_size=int(launch["world_size"]),
        per_gpu_batch=int(launch["per_gpu_batch"]),
        allow_nonfinal_stagea_name=bool(launch["allow_nonfinal_stagea_name"]),
    )
    if recorded != replayed:
        raise ProtocolError(
            "fixed baseline training preflight no longer matches its current "
            "code, config, data, checkpoint, or launch contract"
        )
    return recorded


def _verify_train_completion(
    output_dir: Path,
    *,
    checkpoint: Path | None = None,
) -> Dict[str, Any]:
    preflight_path = output_dir / "protocol_train_preflight.json"
    complete_path = output_dir / "protocol_train_complete.json"
    complete = _read_json_object(complete_path, "fixed baseline training completion")
    if complete.get("schema") != "stageb-fixed-protocol-v1":
        raise ProtocolError("fixed baseline training completion has the wrong schema")
    if complete.get("kind") != "fixed_pure_stageb_dataft_train_complete":
        raise ProtocolError("fixed baseline training completion has the wrong kind")
    if complete.get("preflight") != _file_record(preflight_path):
        raise ProtocolError(
            "fixed baseline training completion is not linked to its current preflight"
        )
    _verify_train_preflight(preflight_path)
    checkpoints = {
        name: _completed_stageb_baseline_checkpoint_record(output_dir / name)
        for name in ("checkpoint.pth", "checkpoint0000.pth")
    }
    if complete.get("checkpoints") != checkpoints:
        raise ProtocolError(
            "fixed baseline checkpoint files changed after training completion"
        )
    if complete.get("authoritative_checkpoint") != checkpoints["checkpoint0000.pth"]:
        raise ProtocolError("fixed baseline authoritative checkpoint record is invalid")
    if checkpoint is not None and _file_record(_path(checkpoint)) != _file_record(
        output_dir / "checkpoint0000.pth"
    ):
        raise ProtocolError(
            "baseline evaluation checkpoint is not the authoritative completed checkpoint"
        )
    return complete


def _cmd_train_preflight(args: argparse.Namespace) -> None:
    payload = _make_train_preflight_payload(
        stagea_checkpoint=_path(args.stagea_checkpoint),
        data_root=_path(args.data_root),
        world_size=int(args.world_size),
        per_gpu_batch=int(args.per_gpu_batch),
        allow_nonfinal_stagea_name=bool(args.allow_nonfinal_stagea_name),
    )
    _write_json(_path(args.output), payload)
    print(f"[OK] wrote fixed baseline training preflight: {_path(args.output)}")


def _cmd_train_postflight(args: argparse.Namespace) -> None:
    output_dir = _path(args.output_dir)
    preflight_path = output_dir / "protocol_train_preflight.json"
    _require_file(preflight_path, "fixed baseline training preflight")
    _verify_train_preflight(preflight_path)
    checkpoints = {}
    for name in ("checkpoint.pth", "checkpoint0000.pth"):
        checkpoints[name] = _completed_stageb_baseline_checkpoint_record(output_dir / name)
    payload = {
        "schema": "stageb-fixed-protocol-v1",
        "kind": "fixed_pure_stageb_dataft_train_complete",
        "preflight": _file_record(preflight_path),
        "authoritative_checkpoint": checkpoints["checkpoint0000.pth"],
        "checkpoints": checkpoints,
    }
    _write_json(_path(args.output), payload)
    print(f"[OK] wrote fixed baseline training completion audit: {_path(args.output)}")


def _cmd_eval_preflight(args: argparse.Namespace) -> None:
    config = _path(args.config)
    checkpoint = _path(args.checkpoint)
    _require_file(config, "evaluation config")
    _require_file(checkpoint, "evaluation checkpoint")
    cfg = _load_cfg(config)
    config_records = _config_import_records(config)
    eval_code_records = _dependency_records(
        EVAL_CODE_ENTRIES,
        include=EVAL_CODE_INCLUDE,
    )
    resolved_contract = _resolved_eval_contract(cfg)
    checkpoint_record = _file_record(checkpoint)
    baseline_train_link = None
    if _sha256(config) == BASELINE_CONFIG_SHA256:
        train_complete_path = checkpoint.parent / "protocol_train_complete.json"
        _verify_train_completion(checkpoint.parent, checkpoint=checkpoint)
        baseline_train_link = _file_record(train_complete_path)
    payload = _static_payload(_path(args.data_root))
    payload.update(
        {
            "kind": "fixed_stageb_eval_preflight",
            "config": _file_record(config),
            "config_import_chain": config_records,
            "checkpoint": checkpoint_record,
            "baseline_train_complete": baseline_train_link,
            **resolved_contract,
            "runtime": {
                "device": str(args.device),
                "batch_size": int(args.batch_size),
                "num_workers": int(args.num_workers),
                "seed": int(args.seed),
                "amp": bool(args.amp),
                "topk": [1],
                "threshold_tprs": [0.75, 0.9, 0.95],
                "score_thresholds": [0.5],
                "tn_splits": ["refcocop_val", "refcocog_umd_val"],
                "ref_splits": REF_SPLITS,
            },
            "code": eval_code_records,
            "orchestration": [
                _file_record(_path(path)) for path in EVAL_ORCHESTRATION
            ],
            "acceptance_code": _acceptance_code_records(),
            "acceptance_orchestration": _acceptance_orchestration_records(),
        }
    )
    _write_json(_path(args.output), payload)
    print(f"[OK] wrote fixed evaluation preflight: {_path(args.output)}")


def _cmd_eval_postflight(args: argparse.Namespace) -> None:
    output_dir = _path(args.output_dir)
    preflight = output_dir / "protocol_eval_preflight.json"
    _require_file(preflight, "evaluation preflight metadata")
    preflight_payload = _read_json_object(preflight, "evaluation preflight")
    if (
        preflight_payload.get("schema") != "stageb-fixed-protocol-v1"
        or preflight_payload.get("kind") != "fixed_stageb_eval_preflight"
    ):
        raise ProtocolError("evaluation preflight schema or kind is invalid")
    _validate_eval_preflight_dependencies(preflight_payload)
    payload = {
        "schema": "stageb-fixed-protocol-v1",
        "kind": "fixed_stageb_eval_complete",
        "preflight": _file_record(preflight),
        "outputs": _audit_eval_outputs(output_dir),
    }
    _write_json(_path(args.output), payload)
    print(f"[OK] wrote fixed evaluation completion audit: {_path(args.output)}")


def _cmd_verify_eval(args: argparse.Namespace) -> None:
    result = _verify_eval_completion(
        _path(args.output_dir),
        checkpoint=_path(args.checkpoint) if args.checkpoint else None,
        config=_path(args.config) if args.config else None,
    )
    if args.output:
        _write_json(_path(args.output), result)
    print(f"[OK] verified completed fixed evaluation: {_path(args.output_dir)}")


def _cmd_compare_evals(args: argparse.Namespace) -> None:
    baseline_dir = _path(args.baseline_dir)
    candidate_dir = _path(args.candidate_dir)
    payloads: Dict[str, Dict[str, Any]] = {}
    for label, directory in (("baseline", baseline_dir), ("candidate", candidate_dir)):
        verified = _verify_eval_completion(directory)
        payloads[label] = verified["preflight"]

    baseline = payloads["baseline"]
    candidate = payloads["candidate"]
    if baseline.get("config", {}).get("sha256") != BASELINE_CONFIG_SHA256:
        raise ProtocolError("the baseline evaluation did not use the locked pure data-FT config")
    if not baseline.get("baseline_train_complete"):
        raise ProtocolError("the baseline evaluation is not linked to a completed fixed baseline training run")
    baseline_checkpoint = _path(str(baseline["checkpoint"]["path"]))
    train_complete_path = baseline_checkpoint.parent / "protocol_train_complete.json"
    if baseline.get("baseline_train_complete") != _file_record(train_complete_path):
        raise ProtocolError(
            "the baseline evaluation training-completion link changed after preflight"
        )
    _verify_train_completion(
        baseline_checkpoint.parent,
        checkpoint=baseline_checkpoint,
    )
    for key in (
        "runtime",
        "strict_manifests",
        "evaluation_data",
        "required_ref_splits",
    ):
        if baseline.get(key) != candidate.get(key):
            raise ProtocolError(f"baseline/candidate paired protocol mismatch in {key}")
    if baseline.get("resolved_model_architecture") != candidate.get(
        "resolved_model_architecture"
    ):
        raise ProtocolError(
            "baseline/candidate resolved model architectures differ"
        )
    for label, preflight in (("baseline", baseline), ("candidate", candidate)):
        model_protocol = preflight.get("model_protocol")
        if not isinstance(model_protocol, Mapping):
            raise ProtocolError(f"{label} evaluation has no model protocol")
        for key in (
            "config_patch_only",
            "config_stage_b",
            "config_enable_patch_branch",
            "patch_only_at_eval",
        ):
            if model_protocol.get(key) is not False:
                raise ProtocolError(
                    f"{label} evaluation violates the pure-GDINO contract in {key}"
                )
    shared_config_parents = _shared_config_parent_hashes(baseline, candidate)
    baseline_code = {
        str(row.get("path")): str(row.get("sha256")) for row in baseline.get("code", [])
    }
    candidate_code = {
        str(row.get("path")): str(row.get("sha256")) for row in candidate.get("code", [])
    }
    if baseline_code != candidate_code:
        raise ProtocolError("baseline/candidate evaluator code hashes differ")
    if baseline.get("orchestration") != candidate.get("orchestration"):
        raise ProtocolError("baseline/candidate evaluator orchestration hashes differ")
    if baseline.get("acceptance_code") != candidate.get("acceptance_code"):
        raise ProtocolError("baseline/candidate final acceptance code hashes differ")
    if baseline.get("acceptance_orchestration") != candidate.get(
        "acceptance_orchestration"
    ):
        raise ProtocolError(
            "baseline/candidate final acceptance orchestration hashes differ"
        )
    result = {
        "schema": "stageb-fixed-protocol-v1",
        "kind": "fixed_stageb_paired_eval_protocol",
        "runtime": baseline["runtime"],
        "strict_manifests": baseline["strict_manifests"],
        "evaluation_data": baseline["evaluation_data"],
        "required_ref_splits": baseline["required_ref_splits"],
        "code": baseline_code,
        "orchestration": baseline["orchestration"],
        "acceptance_code": baseline["acceptance_code"],
        "acceptance_orchestration": baseline["acceptance_orchestration"],
        "resolved_model_architecture": baseline["resolved_model_architecture"],
        "shared_config_parents": shared_config_parents,
        "baseline": {
            "config": baseline["config"],
            "model_protocol": baseline["model_protocol"],
            "checkpoint": baseline["checkpoint"],
            "eval_dir": str(baseline_dir),
        },
        "candidate": {
            "config": candidate["config"],
            "model_protocol": candidate["model_protocol"],
            "checkpoint": candidate["checkpoint"],
            "eval_dir": str(candidate_dir),
        },
    }
    _write_json(_path(args.output), result)
    print(f"[OK] paired baseline/candidate protocol audit passed: {_path(args.output)}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    static_parser = subparsers.add_parser("static", help="Validate all immutable protocol inputs")
    static_parser.add_argument(
        "--data_root", default=os.environ.get("DATA_ROOT", "/home/user/datasets/pivot_data")
    )
    static_parser.add_argument("--output", default=None)
    static_parser.set_defaults(func=_cmd_static)

    train_preflight = subparsers.add_parser("train-preflight")
    train_preflight.add_argument("--stagea_checkpoint", required=True)
    train_preflight.add_argument("--data_root", required=True)
    train_preflight.add_argument("--world_size", type=int, required=True)
    train_preflight.add_argument("--per_gpu_batch", type=int, required=True)
    train_preflight.add_argument("--output", required=True)
    train_preflight.add_argument("--allow_nonfinal_stagea_name", action="store_true")
    train_preflight.set_defaults(func=_cmd_train_preflight)

    train_postflight = subparsers.add_parser("train-postflight")
    train_postflight.add_argument("--output_dir", required=True)
    train_postflight.add_argument("--output", required=True)
    train_postflight.set_defaults(func=_cmd_train_postflight)

    eval_preflight = subparsers.add_parser("eval-preflight")
    eval_preflight.add_argument("--config", required=True)
    eval_preflight.add_argument("--checkpoint", required=True)
    eval_preflight.add_argument("--data_root", required=True)
    eval_preflight.add_argument("--device", required=True)
    eval_preflight.add_argument("--batch_size", type=int, required=True)
    eval_preflight.add_argument("--num_workers", type=int, required=True)
    eval_preflight.add_argument("--seed", type=int, required=True)
    eval_preflight.add_argument("--amp", action="store_true")
    eval_preflight.add_argument("--output", required=True)
    eval_preflight.set_defaults(func=_cmd_eval_preflight)

    eval_postflight = subparsers.add_parser("eval-postflight")
    eval_postflight.add_argument("--output_dir", required=True)
    eval_postflight.add_argument("--output", required=True)
    eval_postflight.set_defaults(func=_cmd_eval_postflight)

    verify_eval = subparsers.add_parser("verify-eval")
    verify_eval.add_argument("--output_dir", required=True)
    verify_eval.add_argument("--checkpoint")
    verify_eval.add_argument("--config")
    verify_eval.add_argument("--output")
    verify_eval.set_defaults(func=_cmd_verify_eval)

    compare_evals = subparsers.add_parser("compare-evals")
    compare_evals.add_argument("--baseline_dir", required=True)
    compare_evals.add_argument("--candidate_dir", required=True)
    compare_evals.add_argument("--output", required=True)
    compare_evals.set_defaults(func=_cmd_compare_evals)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        args.func(args)
    except ProtocolError as error:
        raise SystemExit(f"[FAIL] {error}") from error


if __name__ == "__main__":
    main()
