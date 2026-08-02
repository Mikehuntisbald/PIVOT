#!/usr/bin/env python3
"""Create and verify one formal external-GDINO rank-transfer contract."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.merge_stageb_gdino_adapter_eval import (  # noqa: E402
    CONTRACT_SCHEMA as MERGED_CONTRACT_SCHEMA,
    EXPECTED_ARCHITECTURE,
    LINEAGE_SCHEMA as MERGED_LINEAGE_SCHEMA,
    MergedEvalCheckpointError,
    verify_merged_eval_checkpoint,
)
from tools.stageb_dependency_audit import (  # noqa: E402
    DependencyAuditError,
    config_import_chain,
)
from tools.stageb_canonical_caption_route_artifact import (  # noqa: E402
    CANDIDATE_DESCRIPTOR_IDS as ROUTE_CANDIDATE_DESCRIPTOR_IDS,
    CAPTION_CONTRACT as ROUTE_CAPTION_CONTRACT,
    DEFAULT_DESCRIPTOR_ID as ROUTE_DEFAULT_DESCRIPTOR_ID,
    DESCRIPTOR_REGISTRY as ROUTE_DESCRIPTOR_REGISTRY,
    KIND as ROUTE_ARTIFACT_KIND,
    SCHEMA as ROUTE_ARTIFACT_SCHEMA,
    SELECTION_CONTRACT as ROUTE_SELECTION_CONTRACT,
    CaptionRouteArtifactError,
    _norm_text as normalize_route_caption,
    load_and_verify_caption_route_artifact,
)
from tools.stageb_gdino_adapter_probe_audit import (  # noqa: E402
    ADAPTER_PREFIX,
    ProbeAuditError,
    checkpoint_args,
    file_record,
    load_checkpoint,
    model_hash_record,
    tensor_state_sha256,
)
from tools.stageb_fulltext_route_gate_artifact import (  # noqa: E402
    KIND as FULLTEXT_ROUTE_GATE_KIND,
    SCHEMA as FULLTEXT_ROUTE_GATE_SCHEMA,
    SELECTION_CONTRACT as FULLTEXT_ROUTE_GATE_SELECTION_CONTRACT,
    TOKEN_COUNT_CONTRACT as FULLTEXT_ROUTE_GATE_TOKEN_COUNT_CONTRACT,
    FullTextRouteGateArtifactError,
    load_and_verify_fulltext_route_gate_artifact,
)
from util.path_compat import remap_legacy_path  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402
from util.stageb_exact_topk_contract import canonical_sha256  # noqa: E402


SCHEMA = "stageb-external-gdino-rank-transfer-formal-artifact-v1"
KIND = "formal_external_gdino_rank_transfer"
ROUTED_V2_SCHEMA = "stageb-external-gdino-rank-transfer-formal-artifact-v2"
ROUTED_V2_KIND = "formal_external_gdino_canonical_caption_routed_transfer"
ROUTED_V2_POLICY_SCHEMA = "stageb-formal-canonical-caption-route-policy-v2"
ROUTED_V3_SCHEMA = "stageb-external-gdino-rank-transfer-formal-artifact-v3"
ROUTED_V3_KIND = "formal_external_gdino_fulltext_gated_caption_routed_transfer"
ROUTED_V3_POLICY_SCHEMA = "stageb-formal-fulltext-gated-caption-route-policy-v3"
IDENTITY_ALGORITHM = "sha256-canonical-json-excluding-artifact-identity-v1"
FORMAL_TRANSFER_MODES = (
    "max_score_iou_power",
    "top_query_nearest_candidate",
)
REF_SPLIT_ORDER = (
    "refcoco_val",
    "refcoco_testA",
    "refcoco_testB",
    "refcocop_val",
    "refcocop_testA",
    "refcocop_testB",
    "refcocog_val",
    "refcocog_test",
)
SPLIT_SEED_PROTOCOL = "canonical_ref8_split_index_stride100000_v1"
SPLIT_SEED_STRIDE = 100_000
CAPTION_PROVENANCE_CONTRACT = {
    "schema": "stageb-ref-full-expression-caption-provenance-v1",
    "target_key": "caption",
    "manifest_expression_path": "instances[0].raw_phrase",
    "normalization": "clean_phrase_then_space_period_v1",
    "required_expression_slots": 1,
    "canonical_caption_source": "patch_outputs.stage_a_captions",
}


class ExternalRankTransferArtifactError(RuntimeError):
    pass


def stable_ref_split_seed_map(base_seed: int) -> Dict[str, int]:
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ExternalRankTransferArtifactError(
            "formal evaluation base seed must be a non-negative integer"
        )
    if len(set(REF_SPLIT_ORDER)) != len(REF_SPLIT_ORDER):
        raise ExternalRankTransferArtifactError("canonical Ref split order is not unique")
    return {
        name: int(base_seed + index * SPLIT_SEED_STRIDE)
        for index, name in enumerate(REF_SPLIT_ORDER)
    }


def _resolve(value: str | Path) -> Path:
    path = remap_legacy_path(value, repo_root=REPO_ROOT)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.expanduser().resolve()


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExternalRankTransferArtifactError(f"{label} must be a mapping")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], keys: Sequence[str] | set[str], *, label: str
) -> None:
    expected = set(keys)
    if set(value) != expected:
        raise ExternalRankTransferArtifactError(
            f"{label} has unexpected or missing fields: "
            f"expected={sorted(expected)}, observed={sorted(map(str, value))}"
        )


def _config_binding(path: Path) -> Dict[str, Any]:
    try:
        chain = config_import_chain(path, root=REPO_ROOT)
        records = [file_record(item) for item in chain]
    except (DependencyAuditError, ProbeAuditError) as error:
        raise ExternalRankTransferArtifactError(str(error)) from error
    if not chain or path not in chain:
        raise ExternalRankTransferArtifactError(
            f"config import chain does not contain leaf {path}"
        )
    return {
        "leaf": next(row for row in records if row["path"] == str(path)),
        "import_chain": records,
    }


def _verify_file_record(record: Mapping[str, Any], *, label: str) -> Dict[str, Any]:
    if not isinstance(record.get("path"), str):
        raise ExternalRankTransferArtifactError(f"{label} has no path")
    try:
        observed = file_record(_resolve(record["path"]))
    except ProbeAuditError as error:
        raise ExternalRankTransferArtifactError(str(error)) from error
    expected = {
        key: record.get(key) for key in ("path", "size_bytes", "sha256")
    }
    expected["path"] = str(_resolve(expected["path"]))
    if observed != expected:
        raise ExternalRankTransferArtifactError(f"{label} file identity drifted")
    return observed


def _require_file_unchanged(
    path: Path, record: Mapping[str, Any], *, label: str
) -> None:
    observed = file_record(path)
    expected = {
        "path": str(path.resolve()),
        "size_bytes": record.get("size_bytes"),
        "sha256": record.get("sha256"),
    }
    if observed != expected:
        raise ExternalRankTransferArtifactError(f"{label} changed during validation")


def _verify_config_binding(binding: Mapping[str, Any], *, label: str) -> None:
    _require_exact_keys(binding, {"leaf", "import_chain"}, label=label)
    rows = binding["import_chain"]
    if not isinstance(rows, list) or not rows:
        raise ExternalRankTransferArtifactError(f"{label} import chain is empty")
    observed = [_verify_file_record(row, label=f"{label} import") for row in rows]
    leaf = _verify_file_record(binding["leaf"], label=f"{label} leaf")
    if sum(row == leaf for row in observed) != 1:
        raise ExternalRankTransferArtifactError(f"{label} leaf is not unique in chain")


def _model_state(payload: Mapping[str, Any], *, label: str) -> Mapping[str, torch.Tensor]:
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise ExternalRankTransferArtifactError(f"{label} has no model state")
    invalid = [
        key
        for key, value in state.items()
        if not isinstance(key, str) or not torch.is_tensor(value)
    ]
    if invalid:
        raise ExternalRankTransferArtifactError(
            f"{label} contains non-tensor model entries: {invalid[:8]}"
        )
    return state


def _load(path: Path, *, label: str) -> MutableMapping[str, Any]:
    try:
        return load_checkpoint(path)
    except ProbeAuditError as error:
        raise ExternalRankTransferArtifactError(f"{label}: {error}") from error


def _matches(observed: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return isinstance(observed, (int, float)) and math.isclose(
            float(observed), expected, rel_tol=0.0, abs_tol=1e-12
        )
    return observed == expected


def _require_config_values(cfg: Any, values: Mapping[str, Any], *, label: str) -> None:
    for key, expected in values.items():
        observed = getattr(cfg, key, None)
        if not _matches(observed, expected):
            raise ExternalRankTransferArtifactError(
                f"{label} config {key}: expected {expected!r}, got {observed!r}"
            )


def _patch_component(config: Path, checkpoint: Path) -> Dict[str, Any]:
    binding = _config_binding(config)
    try:
        cfg = SLConfig.fromfile(str(config))
    except Exception as error:
        raise ExternalRankTransferArtifactError(
            f"could not load patch config {config}: {error}"
        ) from error
    _verify_config_binding(binding, label="patch config")
    _require_config_values(
        cfg,
        {
            "stage_b_v11_fixed_text": True,
            "stage_b_v15_decoupled_confidence": True,
            "stage_b_v15_patch_rank_fusion": True,
            "stage_b_v11_candidate_topk": 50,
        },
        label="patch",
    )
    if bool(getattr(cfg, "stage_b_gdino_score_adapter", False)):
        raise ExternalRankTransferArtifactError(
            "patch config cannot enable the pure-GDINO score adapter"
        )
    contract_weight = float(
        getattr(cfg, "stage_b_v15_patch_rank_weight", float("nan"))
    )
    if not math.isfinite(contract_weight) or contract_weight < 0.0:
        raise ExternalRankTransferArtifactError(
            "patch config stage_b_v15_patch_rank_weight must be finite and non-negative"
        )
    checkpoint_record = file_record(checkpoint)
    payload = _load(checkpoint, label="patch checkpoint")
    state = _model_state(payload, label="patch checkpoint")
    args = checkpoint_args(payload)
    recorded_config = args.get("config_file")
    if recorded_config in (None, "") or _resolve(str(recorded_config)) != config:
        raise ExternalRankTransferArtifactError(
            "patch checkpoint config_file does not match the bound patch config"
        )
    topk_key = "stage_b_fixed_text_scorer._score_contract_candidate_topk"
    topk_value = state.get(topk_key)
    if (
        not torch.is_tensor(topk_value)
        or topk_value.numel() != 1
        or int(topk_value.detach().reshape(-1)[0].item()) != 50
    ):
        raise ExternalRankTransferArtifactError(
            "patch checkpoint does not carry the exact Top50 scorer contract"
        )
    _require_file_unchanged(
        checkpoint, checkpoint_record, label="patch checkpoint"
    )
    _verify_config_binding(binding, label="patch config")
    return {
        "config": binding,
        "checkpoint": {
            **checkpoint_record,
            "full_model_tensor_sha256": tensor_state_sha256(state, state.keys()),
            "model_state_keys": len(state),
        },
        "runtime_contract": {
            "candidate_topk": 50,
            "contract_patch_rank_weight": contract_weight,
            "candidate_index_key": "stage_b_v11_candidate_idx",
            "candidate_patch_score_key": "stage_b_v15_candidate_patch_logits",
            "candidate_box_key": "pred_boxes",
        },
    }


def _source_lineage_hashes(
    source: Mapping[str, Any],
    *,
    explicit_checkpoint: Path,
    role: str,
) -> Dict[str, Any]:
    _require_exact_keys(source, {"checkpoint", "config", "model"}, label=role)
    checkpoint_record = _require_mapping(source["checkpoint"], label=f"{role} checkpoint")
    if _resolve(str(checkpoint_record.get("path", ""))) != explicit_checkpoint:
        raise ExternalRankTransferArtifactError(f"{role} checkpoint path mismatch")
    verified_checkpoint = _verify_file_record(
        checkpoint_record, label=f"{role} checkpoint"
    )
    config_binding = _require_mapping(source["config"], label=f"{role} config")
    _verify_config_binding(
        config_binding,
        label=f"{role} config",
    )
    state = _model_state(_load(explicit_checkpoint, label=role), label=role)
    try:
        hashes = model_hash_record(state)
    except ProbeAuditError as error:
        raise ExternalRankTransferArtifactError(str(error)) from error
    model = _require_mapping(source["model"], label=f"{role} model lineage")
    hash_key = "rank_tensor_sha256" if role == "rank_source" else "confidence_tensor_sha256"
    if model.get("base_tensor_sha256") != hashes["base_model_sha256"]:
        raise ExternalRankTransferArtifactError(f"{role} base tensor hash drifted")
    observed_branch = hashes["rank_sha256" if role == "rank_source" else "confidence_sha256"]
    if model.get(hash_key) != observed_branch:
        raise ExternalRankTransferArtifactError(f"{role} branch tensor hash drifted")
    _require_file_unchanged(
        explicit_checkpoint, verified_checkpoint, label=f"{role} checkpoint"
    )
    _verify_config_binding(config_binding, label=f"{role} config")
    return hashes


def _validate_merged_component(
    *,
    config: Path,
    checkpoint: Path,
    source_rank_checkpoint: Path,
    source_confidence_checkpoint: Path,
    base_baseline_checkpoint: Path,
) -> Dict[str, Any]:
    config_binding = _config_binding(config)
    try:
        cfg = SLConfig.fromfile(str(config))
    except Exception as error:
        raise ExternalRankTransferArtifactError(
            f"could not load merged external config {config}: {error}"
        ) from error
    _require_config_values(
        cfg,
        {
            **EXPECTED_ARCHITECTURE,
            "stage_b_gdino_adapter_merged_eval_only": True,
            "stage_b_gdino_adapter_merged_eval_contract_version": 1,
            "stage_b_gdino_rank_weight": 0.0,
            "stage_b_gdino_confidence_weight": 0.0,
        },
        label="merged external",
    )
    checkpoint_record = file_record(checkpoint)
    try:
        merged_receipt = verify_merged_eval_checkpoint(checkpoint)
    except MergedEvalCheckpointError as error:
        raise ExternalRankTransferArtifactError(
            f"canonical merged checkpoint verification failed: {error}"
        ) from error
    if merged_receipt.get("checkpoint") != checkpoint_record:
        raise ExternalRankTransferArtifactError(
            "canonical merged verifier checkpoint identity drifted"
        )
    payload = _load(checkpoint, label="merged external checkpoint")
    _require_exact_keys(payload, {"model", "lineage", "contract"}, label="merged checkpoint")
    lineage = _require_mapping(payload["lineage"], label="merged lineage")
    contract = _require_mapping(payload["contract"], label="merged contract")
    _require_exact_keys(
        lineage,
        {"schema", "rank_source", "confidence_source", "baseline", "eval_config"},
        label="merged lineage",
    )
    if lineage.get("schema") != MERGED_LINEAGE_SCHEMA:
        raise ExternalRankTransferArtifactError("merged lineage schema mismatch")
    contract_keys = {
        "schema",
        "eval_only",
        "resumable",
        "architecture",
        "adapter_key_whitelist",
        "rank_key_whitelist",
        "confidence_key_whitelist",
        "branch_selection",
        "model_state_keys",
        "base_state_keys",
        "adapter_state_keys",
        "base_tensor_sha256",
        "rank_tensor_sha256",
        "confidence_tensor_sha256",
        "adapter_tensor_sha256",
        "full_model_tensor_sha256",
        "rank_final_zero",
        "confidence_final_zero",
        "functional_bitwise",
        "synthetic_input_contract",
    }
    _require_exact_keys(contract, contract_keys, label="merged contract")
    if (
        contract.get("schema") != MERGED_CONTRACT_SCHEMA
        or contract.get("eval_only") is not True
        or contract.get("resumable") is not False
        or contract.get("architecture") != EXPECTED_ARCHITECTURE
    ):
        raise ExternalRankTransferArtifactError("merged eval-only contract drifted")
    if lineage["eval_config"] != config_binding:
        raise ExternalRankTransferArtifactError(
            "merged checkpoint eval-config lineage does not match bound config"
        )
    _verify_config_binding(config_binding, label="merged external config")

    state = _model_state(payload, label="merged external checkpoint")
    try:
        hashes = model_hash_record(state)
        full_hash = tensor_state_sha256(state, state.keys())
    except ProbeAuditError as error:
        raise ExternalRankTransferArtifactError(str(error)) from error
    hash_checks = {
        "base_tensor_sha256": hashes["base_model_sha256"],
        "rank_tensor_sha256": hashes["rank_sha256"],
        "confidence_tensor_sha256": hashes["confidence_sha256"],
        "adapter_tensor_sha256": hashes["adapter_sha256"],
        "full_model_tensor_sha256": full_hash,
        "model_state_keys": len(state),
        "base_state_keys": hashes["base_state_keys"],
        "adapter_state_keys": hashes["adapter_state_keys"],
        "rank_final_zero": False,
        "confidence_final_zero": False,
    }
    for key, expected in hash_checks.items():
        if contract.get(key) != expected:
            raise ExternalRankTransferArtifactError(
                f"merged checkpoint contract {key} drifted"
            )
    adapter_keys = sorted(key for key in state if key.startswith(ADAPTER_PREFIX))
    rank_keys = list(contract["rank_key_whitelist"])
    confidence_keys = list(contract["confidence_key_whitelist"])
    if (
        len(adapter_keys) != 20
        or len(rank_keys) != 8
        or len(confidence_keys) != 12
        or list(contract["adapter_key_whitelist"]) != adapter_keys
        or sorted(rank_keys + confidence_keys) != adapter_keys
    ):
        raise ExternalRankTransferArtifactError("merged adapter whitelist drifted")
    functional = _require_mapping(
        contract["functional_bitwise"], label="merged functional contract"
    )
    if len(functional) != 6 or not all(value is True for value in functional.values()):
        raise ExternalRankTransferArtifactError(
            "merged functional bitwise contract is incomplete"
        )

    rank_hashes = _source_lineage_hashes(
        _require_mapping(lineage["rank_source"], label="rank source"),
        explicit_checkpoint=source_rank_checkpoint,
        role="rank_source",
    )
    confidence_hashes = _source_lineage_hashes(
        _require_mapping(lineage["confidence_source"], label="confidence source"),
        explicit_checkpoint=source_confidence_checkpoint,
        role="confidence_source",
    )
    baseline = _require_mapping(lineage["baseline"], label="baseline lineage")
    _require_exact_keys(
        baseline, {"checkpoint", "base_tensor_sha256"}, label="baseline lineage"
    )
    if _resolve(str(baseline["checkpoint"].get("path", ""))) != base_baseline_checkpoint:
        raise ExternalRankTransferArtifactError("baseline checkpoint path mismatch")
    baseline_record = _verify_file_record(
        baseline["checkpoint"], label="baseline checkpoint"
    )
    baseline_state = _model_state(
        _load(base_baseline_checkpoint, label="baseline checkpoint"),
        label="baseline checkpoint",
    )
    try:
        baseline_hashes = model_hash_record(baseline_state)
    except ProbeAuditError as error:
        raise ExternalRankTransferArtifactError(str(error)) from error
    if (
        baseline_hashes["adapter_state_keys"] != 0
        or baseline["base_tensor_sha256"] != baseline_hashes["base_model_sha256"]
        or hashes["base_model_sha256"] != baseline_hashes["base_model_sha256"]
        or rank_hashes["base_model_sha256"] != baseline_hashes["base_model_sha256"]
        or confidence_hashes["base_model_sha256"]
        != baseline_hashes["base_model_sha256"]
    ):
        raise ExternalRankTransferArtifactError(
            "merged/source base tensors do not match the bound baseline"
        )
    if (
        hashes["rank_sha256"] != rank_hashes["rank_sha256"]
        or hashes["confidence_sha256"] != confidence_hashes["confidence_sha256"]
    ):
        raise ExternalRankTransferArtifactError(
            "merged branch tensors do not match source lineage"
        )
    _require_file_unchanged(
        checkpoint, checkpoint_record, label="merged external checkpoint"
    )
    _verify_config_binding(config_binding, label="merged external config")
    _require_file_unchanged(
        base_baseline_checkpoint,
        baseline_record,
        label="baseline checkpoint",
    )
    return {
        "config": config_binding,
        "checkpoint": {
            **checkpoint_record,
            "full_model_tensor_sha256": full_hash,
        },
        "source_lineage": dict(lineage),
        "merged_contract": dict(contract),
        "source_lineage_sha256": canonical_sha256(lineage),
        "merged_contract_sha256": canonical_sha256(contract),
        "base_baseline_checkpoint": baseline_record,
    }


def _transfer_contract(
    *,
    mode: str,
    iou_power: float | None,
    patch_weight: float,
    text_weight: float,
) -> Dict[str, Any]:
    mode = str(mode)
    if mode not in FORMAL_TRANSFER_MODES:
        raise ExternalRankTransferArtifactError(
            f"formal transfer mode must be one of {FORMAL_TRANSFER_MODES}"
        )
    if mode == "max_score_iou_power":
        if iou_power is None or not math.isfinite(float(iou_power)) or float(iou_power) <= 0:
            raise ExternalRankTransferArtifactError(
                "max_score_iou_power requires a finite positive IoU power"
            )
        power: float | None = float(iou_power)
        formula = "max(rank_score * IoU**p) over strictly-positive-IoU queries"
    else:
        if iou_power is not None:
            raise ExternalRankTransferArtifactError(
                "top_query_nearest_candidate does not accept an IoU power"
            )
        power = None
        formula = (
            "first argmax-IoU candidate per external query, then candidate scatter amax "
            "of raw rank score"
        )
    patch_weight = float(patch_weight)
    text_weight = float(text_weight)
    if not math.isfinite(patch_weight) or patch_weight < 0:
        raise ExternalRankTransferArtifactError(
            "formal patch weight must be finite and non-negative"
        )
    if not math.isfinite(text_weight) or text_weight <= 0:
        raise ExternalRankTransferArtifactError(
            "formal text weight must be finite and strictly positive"
        )
    return {
        "mode": mode,
        "iou_power": power,
        "patch_weight": patch_weight,
        "text_weight": text_weight,
        "transfer_formula": formula,
        "fusion_formula": (
            "patch_weight * stage_b_v15_candidate_patch_logits + "
            "text_weight * transferred_stage_b_gdino_rank_score"
        ),
        "candidate_admission": "unchanged_exact_stage_a_top50",
        "candidate_topk": 50,
        "external_query_count": 900,
        "external_rank_score_key": "stage_b_gdino_rank_score",
        "external_confidence_score_key": "stage_b_gdino_confidence_score",
        "box_format": "normalized_cxcywh",
        "top_query_unassigned_candidate_policy": (
            "fallback_to_sample_min_external_rank_score"
        ),
        "top_query_global_rank_tie_policy": (
            "first external argmax query's nearest candidate wins at patch_weight=0"
        ),
    }


def build_formal_transfer_artifact(
    *,
    patch_config: str | Path,
    patch_checkpoint: str | Path,
    merged_external_config: str | Path,
    merged_external_checkpoint: str | Path,
    source_rank_checkpoint: str | Path,
    source_confidence_checkpoint: str | Path,
    base_baseline_checkpoint: str | Path,
    mode: str,
    iou_power: float | None,
    patch_weight: float,
    text_weight: float,
    base_seed: int = 42,
) -> Dict[str, Any]:
    paths = {
        "patch_config": _resolve(patch_config),
        "patch_checkpoint": _resolve(patch_checkpoint),
        "merged_external_config": _resolve(merged_external_config),
        "merged_external_checkpoint": _resolve(merged_external_checkpoint),
        "source_rank_checkpoint": _resolve(source_rank_checkpoint),
        "source_confidence_checkpoint": _resolve(source_confidence_checkpoint),
        "base_baseline_checkpoint": _resolve(base_baseline_checkpoint),
    }
    if len(set(paths.values())) != len(paths):
        raise ExternalRankTransferArtifactError(
            "formal transfer component paths must all be distinct"
        )
    patch = _patch_component(paths["patch_config"], paths["patch_checkpoint"])
    external = _validate_merged_component(
        config=paths["merged_external_config"],
        checkpoint=paths["merged_external_checkpoint"],
        source_rank_checkpoint=paths["source_rank_checkpoint"],
        source_confidence_checkpoint=paths["source_confidence_checkpoint"],
        base_baseline_checkpoint=paths["base_baseline_checkpoint"],
    )
    transfer = _transfer_contract(
        mode=mode,
        iou_power=iou_power,
        patch_weight=patch_weight,
        text_weight=text_weight,
    )
    evaluation_protocol = {
        "seed_protocol": SPLIT_SEED_PROTOCOL,
        "base_seed": int(base_seed),
        "split_seed_stride": SPLIT_SEED_STRIDE,
        "canonical_split_order": list(REF_SPLIT_ORDER),
        "split_seeds": stable_ref_split_seed_map(base_seed),
    }
    payload = {
        "schema": SCHEMA,
        "kind": KIND,
        "components": {"patch": patch, "merged_external": external},
        "inference_contract": {
            "transfer": transfer,
            "caption_provenance": dict(CAPTION_PROVENANCE_CONTRACT),
            "rank_confidence_policy": (
                "rank uses merged rank branch; absolute confidence uses the independently "
                "merged confidence branch; no shared trainable score"
            ),
        },
        "evaluation_protocol": evaluation_protocol,
    }
    payload["artifact_identity"] = {
        "algorithm": IDENTITY_ALGORITHM,
        "sha256": canonical_sha256(payload),
    }
    return payload


def _verified_caption_route_component(
    path: Path,
    *,
    patch: Mapping[str, Any],
    external: Mapping[str, Any],
) -> Dict[str, Any]:
    try:
        artifact_record = file_record(path)
        selection = load_and_verify_caption_route_artifact(path)
    except (ProbeAuditError, CaptionRouteArtifactError) as error:
        raise ExternalRankTransferArtifactError(
            f"caption route selection verification failed: {error}"
        ) from error
    _require_file_unchanged(path, artifact_record, label="caption route artifact")
    if (
        selection.get("schema") != ROUTE_ARTIFACT_SCHEMA
        or selection.get("kind") != ROUTE_ARTIFACT_KIND
    ):
        raise ExternalRankTransferArtifactError(
            "caption route selection schema or kind drifted"
        )
    if selection.get("caption_contract") != ROUTE_CAPTION_CONTRACT:
        raise ExternalRankTransferArtifactError(
            "caption route normalization contract drifted"
        )
    if (
        selection.get("descriptor_registry") != ROUTE_DESCRIPTOR_REGISTRY
        or selection.get("descriptor_registry_sha256")
        != canonical_sha256(ROUTE_DESCRIPTOR_REGISTRY)
    ):
        raise ExternalRankTransferArtifactError(
            "caption route descriptor registry drifted"
        )
    if (
        selection.get("selection_contract") != ROUTE_SELECTION_CONTRACT
        or selection.get("selection_contract_sha256")
        != canonical_sha256(ROUTE_SELECTION_CONTRACT)
    ):
        raise ExternalRankTransferArtifactError(
            "caption route selection policy drifted"
        )

    source_identity = _require_mapping(
        selection.get("source_identity"), label="caption route source identity"
    )
    expected_source_identity = {
        "checkpoint": patch["checkpoint"]["path"],
        "patch_checkpoint_sha256": patch["checkpoint"]["sha256"],
        "patch_config": patch["config"]["leaf"]["path"],
        "patch_config_sha256": patch["config"]["leaf"]["sha256"],
        "external_gdino_checkpoint": external["checkpoint"]["path"],
        "external_gdino_checkpoint_sha256": external["checkpoint"]["sha256"],
        "external_gdino_config": external["config"]["leaf"]["path"],
        "external_gdino_config_sha256": external["config"]["leaf"]["sha256"],
        "external_gdino_query_count": 900,
        "batch_size": 32,
        "num_workers": 4,
    }
    if dict(source_identity) != expected_source_identity:
        raise ExternalRankTransferArtifactError(
            "caption route evidence is not bound to the formal model components"
        )
    source_component_files = _require_mapping(
        selection.get("source_component_files"),
        label="caption route source component files",
    )
    expected_component_files = {
        "patch_config": {
            key: patch["config"]["leaf"][key]
            for key in ("path", "size_bytes", "sha256")
        },
        "patch_checkpoint": {
            key: patch["checkpoint"][key]
            for key in ("path", "size_bytes", "sha256")
        },
        "external_config": {
            key: external["config"]["leaf"][key]
            for key in ("path", "size_bytes", "sha256")
        },
        "external_checkpoint": {
            key: external["checkpoint"][key]
            for key in ("path", "size_bytes", "sha256")
        },
    }
    if dict(source_component_files) != expected_component_files:
        raise ExternalRankTransferArtifactError(
            "caption route source component file bindings drifted"
        )

    route = _require_mapping(selection.get("route"), label="caption route mapping")
    _require_exact_keys(
        route,
        {"default_descriptor_id", "overrides", "override_count", "caption_count"},
        label="caption route mapping",
    )
    if route.get("default_descriptor_id") != ROUTE_DEFAULT_DESCRIPTOR_ID:
        raise ExternalRankTransferArtifactError(
            "caption route default descriptor must be external base direct"
        )
    overrides = _require_mapping(route.get("overrides"), label="caption overrides")
    normalized_overrides: Dict[str, str] = {}
    allowed = set(ROUTE_CANDIDATE_DESCRIPTOR_IDS)
    for raw_caption, raw_descriptor in overrides.items():
        if not isinstance(raw_caption, str) or not raw_caption:
            raise ExternalRankTransferArtifactError(
                "caption route override keys must be non-empty strings"
            )
        caption = normalize_route_caption(raw_caption)
        descriptor_id = str(raw_descriptor)
        if caption != raw_caption or descriptor_id not in allowed:
            raise ExternalRankTransferArtifactError(
                "caption route override mapping escaped its frozen registry"
            )
        normalized_overrides[caption] = descriptor_id
    if (
        normalized_overrides != dict(overrides)
        or int(route.get("override_count", -1)) != len(normalized_overrides)
    ):
        raise ExternalRankTransferArtifactError(
            "caption route override mapping or count drifted"
        )

    selection_identity = _require_mapping(
        selection.get("artifact_identity"), label="caption route artifact identity"
    )
    canonical_classes = _require_mapping(
        selection.get("canonical_classes"),
        label="caption route canonical classes",
    )
    manifest_caption_audit_contract = _require_mapping(
        selection.get("manifest_caption_audit_contract"),
        label="manifest caption audit contract",
    )
    policy = {
        "schema": ROUTED_V2_POLICY_SCHEMA,
        "routing_source": "patch_outputs.stage_a_captions",
        "normalization": "_norm_text",
        "forbidden_routing_inputs": list(
            ROUTE_SELECTION_CONTRACT["forbidden_routing_inputs"]
        ),
        "default_descriptor_id": ROUTE_DEFAULT_DESCRIPTOR_ID,
        "unknown_caption_descriptor_id": ROUTE_DEFAULT_DESCRIPTOR_ID,
        "overrides": dict(sorted(normalized_overrides.items())),
        "allowed_override_descriptor_ids": list(ROUTE_CANDIDATE_DESCRIPTOR_IDS),
        "descriptor_registry_sha256": canonical_sha256(ROUTE_DESCRIPTOR_REGISTRY),
        "selection_artifact_identity_sha256": selection_identity["sha256"],
        "selection_contract_sha256": canonical_sha256(ROUTE_SELECTION_CONTRACT),
    }
    return {
        "artifact": artifact_record,
        "artifact_identity": dict(selection_identity),
        "artifact_schema": ROUTE_ARTIFACT_SCHEMA,
        "artifact_kind": ROUTE_ARTIFACT_KIND,
        "caption_contract": dict(ROUTE_CAPTION_CONTRACT),
        "canonical_classes": dict(canonical_classes),
        "manifest_caption_audit_contract": dict(
            manifest_caption_audit_contract
        ),
        "descriptor_registry": dict(ROUTE_DESCRIPTOR_REGISTRY),
        "descriptor_registry_sha256": canonical_sha256(ROUTE_DESCRIPTOR_REGISTRY),
        "selection_contract": dict(ROUTE_SELECTION_CONTRACT),
        "selection_contract_sha256": canonical_sha256(ROUTE_SELECTION_CONTRACT),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
    }


def build_formal_routed_transfer_artifact(
    *,
    patch_config: str | Path,
    patch_checkpoint: str | Path,
    merged_external_config: str | Path,
    merged_external_checkpoint: str | Path,
    source_rank_checkpoint: str | Path,
    source_confidence_checkpoint: str | Path,
    base_baseline_checkpoint: str | Path,
    caption_route_artifact: str | Path,
    base_seed: int = 42,
) -> Dict[str, Any]:
    if type(base_seed) is not int or base_seed != 42:
        raise ExternalRankTransferArtifactError(
            "formal routed evaluation requires the frozen base seed 42"
        )
    paths = {
        "patch_config": _resolve(patch_config),
        "patch_checkpoint": _resolve(patch_checkpoint),
        "merged_external_config": _resolve(merged_external_config),
        "merged_external_checkpoint": _resolve(merged_external_checkpoint),
        "source_rank_checkpoint": _resolve(source_rank_checkpoint),
        "source_confidence_checkpoint": _resolve(source_confidence_checkpoint),
        "base_baseline_checkpoint": _resolve(base_baseline_checkpoint),
        "caption_route_artifact": _resolve(caption_route_artifact),
    }
    if len(set(paths.values())) != len(paths):
        raise ExternalRankTransferArtifactError(
            "formal routed component paths must all be distinct"
        )
    patch = _patch_component(paths["patch_config"], paths["patch_checkpoint"])
    external = _validate_merged_component(
        config=paths["merged_external_config"],
        checkpoint=paths["merged_external_checkpoint"],
        source_rank_checkpoint=paths["source_rank_checkpoint"],
        source_confidence_checkpoint=paths["source_confidence_checkpoint"],
        base_baseline_checkpoint=paths["base_baseline_checkpoint"],
    )
    routing = _verified_caption_route_component(
        paths["caption_route_artifact"], patch=patch, external=external
    )
    evaluation_protocol = {
        "seed_protocol": SPLIT_SEED_PROTOCOL,
        "base_seed": int(base_seed),
        "split_seed_stride": SPLIT_SEED_STRIDE,
        "canonical_split_order": list(REF_SPLIT_ORDER),
        "split_seeds": stable_ref_split_seed_map(base_seed),
    }
    payload = {
        "schema": ROUTED_V2_SCHEMA,
        "kind": ROUTED_V2_KIND,
        "components": {"patch": patch, "merged_external": external},
        "inference_contract": {
            "routing": routing,
            "caption_provenance": dict(CAPTION_PROVENANCE_CONTRACT),
            "rank_confidence_policy": (
                "caption routing selects either frozen pure-GDINO base identity or "
                "one frozen rank-transfer descriptor; absolute confidence remains "
                "the independently merged confidence branch"
            ),
        },
        "evaluation_protocol": evaluation_protocol,
    }
    payload["artifact_identity"] = {
        "algorithm": IDENTITY_ALGORITHM,
        "sha256": canonical_sha256(payload),
    }
    return payload


def _verified_routed_v2_artifact_component(
    path: Path,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        artifact_record = file_record(path)
        payload = load_and_verify_formal_transfer_artifact(path)
    except ProbeAuditError as error:
        raise ExternalRankTransferArtifactError(
            f"routed v2 artifact verification failed: {error}"
        ) from error
    _require_file_unchanged(path, artifact_record, label="routed v2 artifact")
    if (
        payload.get("schema") != ROUTED_V2_SCHEMA
        or payload.get("kind") != ROUTED_V2_KIND
    ):
        raise ExternalRankTransferArtifactError(
            "full-text routed v3 requires a formal routed v2 artifact"
        )
    identity = _require_mapping(
        payload.get("artifact_identity"), label="routed v2 artifact identity"
    )
    component = {
        "artifact": artifact_record,
        "artifact_identity": dict(identity),
        "artifact_schema": ROUTED_V2_SCHEMA,
        "artifact_kind": ROUTED_V2_KIND,
    }
    return component, dict(payload)


def _normalized_fulltext_route(
    route: Mapping[str, Any],
    *,
    routed_v2_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    _require_exact_keys(
        route,
        {
            "default_descriptor_id",
            "unconditional_overrides",
            "conditional_overrides",
        },
        label="full-text gate route",
    )
    upstream_routing = _require_mapping(
        routed_v2_payload["inference_contract"]["routing"],
        label="routed v2 routing contract",
    )
    upstream_policy = _require_mapping(
        upstream_routing.get("policy"), label="routed v2 route policy"
    )
    registry = _require_mapping(
        upstream_routing.get("descriptor_registry"),
        label="routed v2 descriptor registry",
    )
    default_descriptor_id = str(route.get("default_descriptor_id"))
    if (
        default_descriptor_id != upstream_policy.get("default_descriptor_id")
        or default_descriptor_id not in registry
    ):
        raise ExternalRankTransferArtifactError(
            "full-text gate default descriptor drifted from routed v2"
        )

    unconditional = _require_mapping(
        route.get("unconditional_overrides"),
        label="full-text unconditional overrides",
    )
    normalized_unconditional: Dict[str, str] = {}
    for raw_caption, raw_descriptor_id in unconditional.items():
        if not isinstance(raw_caption, str) or not raw_caption:
            raise ExternalRankTransferArtifactError(
                "full-text unconditional caption keys must be non-empty strings"
            )
        caption = normalize_route_caption(raw_caption)
        descriptor_id = str(raw_descriptor_id)
        if caption != raw_caption or descriptor_id not in registry:
            raise ExternalRankTransferArtifactError(
                "full-text unconditional route escaped the routed v2 registry"
            )
        normalized_unconditional[caption] = descriptor_id
    if normalized_unconditional != dict(unconditional):
        raise ExternalRankTransferArtifactError(
            "full-text unconditional route normalization drifted"
        )

    conditional = _require_mapping(
        route.get("conditional_overrides"),
        label="full-text conditional overrides",
    )
    normalized_conditional: Dict[str, Dict[str, Any]] = {}
    for raw_caption, raw_entry in conditional.items():
        if not isinstance(raw_caption, str) or not raw_caption:
            raise ExternalRankTransferArtifactError(
                "full-text conditional caption keys must be non-empty strings"
            )
        caption = normalize_route_caption(raw_caption)
        if caption != raw_caption or caption in normalized_unconditional:
            raise ExternalRankTransferArtifactError(
                "full-text conditional route caption is invalid or duplicated"
            )
        entry = _require_mapping(
            raw_entry, label=f"full-text conditional route {caption}"
        )
        _require_exact_keys(
            entry,
            {"descriptor_id", "fallback_descriptor_id", "predicate"},
            label=f"full-text conditional route {caption}",
        )
        descriptor_id = str(entry.get("descriptor_id"))
        fallback_descriptor_id = str(entry.get("fallback_descriptor_id"))
        if descriptor_id not in registry or fallback_descriptor_id not in registry:
            raise ExternalRankTransferArtifactError(
                "full-text conditional route escaped the routed v2 registry"
            )
        predicate = _require_mapping(
            entry.get("predicate"),
            label=f"full-text conditional predicate {caption}",
        )
        _require_exact_keys(
            predicate,
            {"kind", "max_tokens", "token_count_contract"},
            label=f"full-text conditional predicate {caption}",
        )
        max_tokens = predicate.get("max_tokens")
        if (
            predicate.get("kind") != "full_expression_lexical_token_count_lte"
            or type(max_tokens) is not int
            or max_tokens
            not in FULLTEXT_ROUTE_GATE_SELECTION_CONTRACT["threshold_candidates"]
            or predicate.get("token_count_contract")
            != FULLTEXT_ROUTE_GATE_TOKEN_COUNT_CONTRACT
        ):
            raise ExternalRankTransferArtifactError(
                "full-text conditional predicate contract drifted"
            )
        normalized_conditional[caption] = {
            "descriptor_id": descriptor_id,
            "fallback_descriptor_id": fallback_descriptor_id,
            "predicate": {
                "kind": "full_expression_lexical_token_count_lte",
                "max_tokens": int(max_tokens),
                "token_count_contract": dict(
                    FULLTEXT_ROUTE_GATE_TOKEN_COUNT_CONTRACT
                ),
            },
        }

    expected_upstream_overrides = dict(normalized_unconditional)
    expected_upstream_overrides.update(
        {
            caption: entry["descriptor_id"]
            for caption, entry in normalized_conditional.items()
        }
    )
    if expected_upstream_overrides != dict(upstream_policy.get("overrides", {})):
        raise ExternalRankTransferArtifactError(
            "full-text gate does not exactly refine the routed v2 overrides"
        )
    return {
        "default_descriptor_id": default_descriptor_id,
        "unconditional_overrides": dict(sorted(normalized_unconditional.items())),
        "conditional_overrides": dict(sorted(normalized_conditional.items())),
    }


def _verified_fulltext_route_gate_component(
    path: Path,
    *,
    routed_v2_path: Path,
    routed_v2_component: Mapping[str, Any],
    routed_v2_payload: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        artifact_record = file_record(path)
        gate = load_and_verify_fulltext_route_gate_artifact(path)
    except (ProbeAuditError, FullTextRouteGateArtifactError) as error:
        raise ExternalRankTransferArtifactError(
            f"full-text route gate verification failed: {error}"
        ) from error
    _require_file_unchanged(path, artifact_record, label="full-text route gate artifact")
    if (
        gate.get("schema") != FULLTEXT_ROUTE_GATE_SCHEMA
        or gate.get("kind") != FULLTEXT_ROUTE_GATE_KIND
    ):
        raise ExternalRankTransferArtifactError(
            "full-text route gate schema or kind drifted"
        )
    gate_identity = _require_mapping(
        gate.get("artifact_identity"), label="full-text route gate identity"
    )
    if gate.get("selection_contract") != FULLTEXT_ROUTE_GATE_SELECTION_CONTRACT:
        raise ExternalRankTransferArtifactError(
            "full-text route gate selection contract drifted"
        )
    expected_selection_hash = canonical_sha256(
        FULLTEXT_ROUTE_GATE_SELECTION_CONTRACT
    )
    if gate.get("selection_contract_sha256") != expected_selection_hash:
        raise ExternalRankTransferArtifactError(
            "full-text route gate selection contract hash drifted"
        )

    formal_binding = _require_mapping(
        gate.get("formal_routed_artifact"),
        label="full-text gate routed v2 file binding",
    )
    expected_formal_binding = dict(routed_v2_component["artifact"])
    if (
        dict(formal_binding) != expected_formal_binding
        or _resolve(str(formal_binding.get("path", ""))) != routed_v2_path
        or gate.get("formal_routed_artifact_identity")
        != routed_v2_component["artifact_identity"]
    ):
        raise ExternalRankTransferArtifactError(
            "full-text route gate is not bound to the supplied routed v2 artifact"
        )

    upstream_routing = routed_v2_payload["inference_contract"]["routing"]
    if (
        gate.get("caption_route_artifact") != upstream_routing["artifact"]
        or gate.get("caption_route_artifact_identity")
        != upstream_routing["artifact_identity"]
        or gate.get("baseline_checkpoint")
        != routed_v2_payload["components"]["merged_external"][
            "base_baseline_checkpoint"
        ]
    ):
        raise ExternalRankTransferArtifactError(
            "full-text route gate upstream bindings drifted from routed v2"
        )
    route = _normalized_fulltext_route(
        _require_mapping(gate.get("route"), label="full-text route gate policy"),
        routed_v2_payload=routed_v2_payload,
    )
    component = {
        "artifact": artifact_record,
        "artifact_identity": dict(gate_identity),
        "artifact_schema": FULLTEXT_ROUTE_GATE_SCHEMA,
        "artifact_kind": FULLTEXT_ROUTE_GATE_KIND,
        "caption_route_artifact": dict(gate["caption_route_artifact"]),
        "caption_route_artifact_identity": dict(
            gate["caption_route_artifact_identity"]
        ),
        "formal_routed_artifact": dict(formal_binding),
        "formal_routed_artifact_identity": dict(
            gate["formal_routed_artifact_identity"]
        ),
        "baseline_checkpoint": dict(gate["baseline_checkpoint"]),
        "selection_contract": dict(FULLTEXT_ROUTE_GATE_SELECTION_CONTRACT),
        "selection_contract_sha256": expected_selection_hash,
        "route": route,
        "validation_route_summary": dict(gate["validation_route_summary"]),
    }
    return component, dict(gate)


def build_formal_fulltext_routed_transfer_artifact(
    *,
    routed_v2_artifact: str | Path,
    fulltext_route_gate_artifact: str | Path,
) -> Dict[str, Any]:
    routed_v2_path = _resolve(routed_v2_artifact)
    fulltext_gate_path = _resolve(fulltext_route_gate_artifact)
    if routed_v2_path == fulltext_gate_path:
        raise ExternalRankTransferArtifactError(
            "routed v2 and full-text gate artifacts must be distinct"
        )
    routed_v2_component, routed_v2_payload = (
        _verified_routed_v2_artifact_component(routed_v2_path)
    )
    fulltext_component, _ = _verified_fulltext_route_gate_component(
        fulltext_gate_path,
        routed_v2_path=routed_v2_path,
        routed_v2_component=routed_v2_component,
        routed_v2_payload=routed_v2_payload,
    )
    upstream_routing = routed_v2_payload["inference_contract"]["routing"]
    route = fulltext_component["route"]
    policy = {
        "schema": ROUTED_V3_POLICY_SCHEMA,
        "routing_sources": list(
            FULLTEXT_ROUTE_GATE_SELECTION_CONTRACT["runtime_routing_inputs"]
        ),
        "forbidden_routing_inputs": list(
            FULLTEXT_ROUTE_GATE_SELECTION_CONTRACT[
                "forbidden_runtime_routing_inputs"
            ]
        ),
        "canonical_caption_normalization": upstream_routing["policy"][
            "normalization"
        ],
        "default_descriptor_id": route["default_descriptor_id"],
        "unknown_caption_descriptor_id": route["default_descriptor_id"],
        "unconditional_overrides": dict(route["unconditional_overrides"]),
        "conditional_overrides": dict(route["conditional_overrides"]),
        "allowed_descriptor_ids": list(upstream_routing["descriptor_registry"]),
        "descriptor_registry_sha256": upstream_routing[
            "descriptor_registry_sha256"
        ],
        "routed_v2_artifact_identity_sha256": routed_v2_component[
            "artifact_identity"
        ]["sha256"],
        "routed_v2_policy_sha256": upstream_routing["policy_sha256"],
        "fulltext_route_gate_artifact_identity_sha256": fulltext_component[
            "artifact_identity"
        ]["sha256"],
        "fulltext_route_gate_selection_contract_sha256": fulltext_component[
            "selection_contract_sha256"
        ],
    }
    evaluation_protocol = dict(routed_v2_payload["evaluation_protocol"])
    evaluation_protocol.update(
        {"batch_size": 32, "num_workers": 4, "amp": True}
    )
    payload = {
        "schema": ROUTED_V3_SCHEMA,
        "kind": ROUTED_V3_KIND,
        "components": {
            "routed_v2": routed_v2_component,
            "fulltext_route_gate": fulltext_component,
        },
        "inference_contract": {
            "routing": {
                "descriptor_registry": dict(
                    upstream_routing["descriptor_registry"]
                ),
                "descriptor_registry_sha256": upstream_routing[
                    "descriptor_registry_sha256"
                ],
                "policy": policy,
                "policy_sha256": canonical_sha256(policy),
            },
            "caption_provenance": dict(
                routed_v2_payload["inference_contract"]["caption_provenance"]
            ),
            "rank_confidence_policy": (
                "canonical-caption routing is refined only by the frozen full-expression "
                "gate; absolute confidence remains the independently merged confidence "
                "branch"
            ),
        },
        "evaluation_protocol": evaluation_protocol,
    }
    payload["artifact_identity"] = {
        "algorithm": IDENTITY_ALGORITHM,
        "sha256": canonical_sha256(payload),
    }
    _require_file_unchanged(
        routed_v2_path,
        routed_v2_component["artifact"],
        label="routed v2 artifact",
    )
    _require_file_unchanged(
        fulltext_gate_path,
        fulltext_component["artifact"],
        label="full-text route gate artifact",
    )
    return payload


def create_formal_transfer_artifact(
    output: str | Path, **kwargs: Any
) -> Dict[str, Any]:
    output_path = _resolve(output)
    if output_path.exists():
        raise ExternalRankTransferArtifactError(
            f"refusing to overwrite formal transfer artifact: {output_path}"
        )
    payload = build_formal_transfer_artifact(**kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    try:
        verified = load_and_verify_formal_transfer_artifact(output_path)
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    if verified != payload:
        output_path.unlink(missing_ok=True)
        raise ExternalRankTransferArtifactError(
            "serialized formal transfer artifact failed exact verification"
        )
    return payload


def create_formal_routed_transfer_artifact(
    output: str | Path, **kwargs: Any
) -> Dict[str, Any]:
    output_path = _resolve(output)
    if output_path.exists():
        raise ExternalRankTransferArtifactError(
            f"refusing to overwrite formal routed transfer artifact: {output_path}"
        )
    payload = build_formal_routed_transfer_artifact(**kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    try:
        verified = load_and_verify_formal_transfer_artifact(output_path)
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    if verified != payload:
        output_path.unlink(missing_ok=True)
        raise ExternalRankTransferArtifactError(
            "serialized formal routed transfer artifact failed exact verification"
        )
    return payload


def create_formal_fulltext_routed_transfer_artifact(
    output: str | Path, **kwargs: Any
) -> Dict[str, Any]:
    output_path = _resolve(output)
    if output_path.exists():
        raise ExternalRankTransferArtifactError(
            f"refusing to overwrite formal full-text routed artifact: {output_path}"
        )
    payload = build_formal_fulltext_routed_transfer_artifact(**kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    try:
        verified = load_and_verify_formal_transfer_artifact(output_path)
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    if verified != payload:
        output_path.unlink(missing_ok=True)
        raise ExternalRankTransferArtifactError(
            "serialized formal full-text routed artifact failed exact verification"
        )
    return payload


def _load_and_verify_formal_routed_transfer_payload(
    payload: Mapping[str, Any],
    *,
    artifact_path: Path,
    artifact_record: Mapping[str, Any],
) -> Dict[str, Any]:
    _require_exact_keys(
        payload,
        {
            "schema",
            "kind",
            "components",
            "inference_contract",
            "evaluation_protocol",
            "artifact_identity",
        },
        label="formal routed transfer artifact",
    )
    if (
        payload.get("schema") != ROUTED_V2_SCHEMA
        or payload.get("kind") != ROUTED_V2_KIND
    ):
        raise ExternalRankTransferArtifactError(
            "unsupported formal routed transfer artifact"
        )
    identity = _require_mapping(payload["artifact_identity"], label="artifact identity")
    unsigned = {key: value for key, value in payload.items() if key != "artifact_identity"}
    if identity != {
        "algorithm": IDENTITY_ALGORITHM,
        "sha256": canonical_sha256(unsigned),
    }:
        raise ExternalRankTransferArtifactError(
            "formal routed transfer artifact identity mismatch"
        )
    components = _require_mapping(payload["components"], label="artifact components")
    patch = _require_mapping(components.get("patch"), label="patch component")
    external = _require_mapping(
        components.get("merged_external"), label="merged external component"
    )
    inference = _require_mapping(
        payload["inference_contract"], label="inference contract"
    )
    routing = _require_mapping(inference.get("routing"), label="routing contract")
    route_artifact = _require_mapping(
        routing.get("artifact"), label="caption route artifact binding"
    )
    protocol = _require_mapping(
        payload["evaluation_protocol"], label="evaluation protocol"
    )
    rebuilt = build_formal_routed_transfer_artifact(
        patch_config=patch["config"]["leaf"]["path"],
        patch_checkpoint=patch["checkpoint"]["path"],
        merged_external_config=external["config"]["leaf"]["path"],
        merged_external_checkpoint=external["checkpoint"]["path"],
        source_rank_checkpoint=external["source_lineage"]["rank_source"][
            "checkpoint"
        ]["path"],
        source_confidence_checkpoint=external["source_lineage"][
            "confidence_source"
        ]["checkpoint"]["path"],
        base_baseline_checkpoint=external["base_baseline_checkpoint"]["path"],
        caption_route_artifact=route_artifact["path"],
        base_seed=protocol["base_seed"],
    )
    if dict(payload) != rebuilt:
        raise ExternalRankTransferArtifactError(
            "formal routed transfer artifact no longer exactly matches its bound inputs"
        )
    _require_file_unchanged(
        artifact_path, artifact_record, label="formal routed transfer artifact"
    )
    return dict(payload)


def _load_and_verify_formal_fulltext_routed_transfer_payload(
    payload: Mapping[str, Any],
    *,
    artifact_path: Path,
    artifact_record: Mapping[str, Any],
) -> Dict[str, Any]:
    _require_exact_keys(
        payload,
        {
            "schema",
            "kind",
            "components",
            "inference_contract",
            "evaluation_protocol",
            "artifact_identity",
        },
        label="formal full-text routed transfer artifact",
    )
    if (
        payload.get("schema") != ROUTED_V3_SCHEMA
        or payload.get("kind") != ROUTED_V3_KIND
    ):
        raise ExternalRankTransferArtifactError(
            "unsupported formal full-text routed transfer artifact"
        )
    identity = _require_mapping(payload["artifact_identity"], label="artifact identity")
    unsigned = {key: value for key, value in payload.items() if key != "artifact_identity"}
    if identity != {
        "algorithm": IDENTITY_ALGORITHM,
        "sha256": canonical_sha256(unsigned),
    }:
        raise ExternalRankTransferArtifactError(
            "formal full-text routed transfer artifact identity mismatch"
        )
    components = _require_mapping(payload["components"], label="artifact components")
    _require_exact_keys(
        components,
        {"routed_v2", "fulltext_route_gate"},
        label="formal full-text routed components",
    )
    routed_v2 = _require_mapping(
        components["routed_v2"], label="routed v2 component"
    )
    fulltext_gate = _require_mapping(
        components["fulltext_route_gate"], label="full-text route gate component"
    )
    routed_v2_artifact = _require_mapping(
        routed_v2.get("artifact"), label="routed v2 file binding"
    )
    gate_artifact = _require_mapping(
        fulltext_gate.get("artifact"), label="full-text route gate file binding"
    )
    rebuilt = build_formal_fulltext_routed_transfer_artifact(
        routed_v2_artifact=routed_v2_artifact["path"],
        fulltext_route_gate_artifact=gate_artifact["path"],
    )
    if dict(payload) != rebuilt:
        raise ExternalRankTransferArtifactError(
            "formal full-text routed transfer artifact no longer exactly matches "
            "its bound inputs"
        )
    _require_file_unchanged(
        artifact_path,
        artifact_record,
        label="formal full-text routed transfer artifact",
    )
    return dict(payload)


def load_and_verify_formal_transfer_artifact(
    artifact: str | Path,
) -> Dict[str, Any]:
    artifact_path = _resolve(artifact)
    try:
        artifact_record = file_record(artifact_path)
    except ProbeAuditError as error:
        raise ExternalRankTransferArtifactError(str(error)) from error
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExternalRankTransferArtifactError(
            f"could not read formal transfer artifact {artifact_path}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ExternalRankTransferArtifactError("formal transfer artifact must be an object")
    if payload.get("schema") == ROUTED_V3_SCHEMA:
        return _load_and_verify_formal_fulltext_routed_transfer_payload(
            payload,
            artifact_path=artifact_path,
            artifact_record=artifact_record,
        )
    if payload.get("schema") == ROUTED_V2_SCHEMA:
        return _load_and_verify_formal_routed_transfer_payload(
            payload,
            artifact_path=artifact_path,
            artifact_record=artifact_record,
        )
    _require_exact_keys(
        payload,
        {
            "schema",
            "kind",
            "components",
            "inference_contract",
            "evaluation_protocol",
            "artifact_identity",
        },
        label="formal transfer artifact",
    )
    if payload.get("schema") != SCHEMA or payload.get("kind") != KIND:
        raise ExternalRankTransferArtifactError("unsupported formal transfer artifact")
    identity = _require_mapping(payload["artifact_identity"], label="artifact identity")
    unsigned = {key: value for key, value in payload.items() if key != "artifact_identity"}
    if identity != {
        "algorithm": IDENTITY_ALGORITHM,
        "sha256": canonical_sha256(unsigned),
    }:
        raise ExternalRankTransferArtifactError("formal transfer artifact identity mismatch")
    components = _require_mapping(payload["components"], label="artifact components")
    patch = _require_mapping(components.get("patch"), label="patch component")
    external = _require_mapping(
        components.get("merged_external"), label="merged external component"
    )
    inference = _require_mapping(
        payload["inference_contract"], label="inference contract"
    )
    transfer = _require_mapping(inference.get("transfer"), label="transfer contract")
    protocol = _require_mapping(payload["evaluation_protocol"], label="evaluation protocol")
    rebuilt = build_formal_transfer_artifact(
        patch_config=patch["config"]["leaf"]["path"],
        patch_checkpoint=patch["checkpoint"]["path"],
        merged_external_config=external["config"]["leaf"]["path"],
        merged_external_checkpoint=external["checkpoint"]["path"],
        source_rank_checkpoint=external["source_lineage"]["rank_source"]["checkpoint"]["path"],
        source_confidence_checkpoint=external["source_lineage"]["confidence_source"]["checkpoint"]["path"],
        base_baseline_checkpoint=external["base_baseline_checkpoint"]["path"],
        mode=transfer["mode"],
        iou_power=transfer["iou_power"],
        patch_weight=transfer["patch_weight"],
        text_weight=transfer["text_weight"],
        base_seed=protocol["base_seed"],
    )
    if dict(payload) != rebuilt:
        raise ExternalRankTransferArtifactError(
            "formal transfer artifact no longer exactly matches its bound inputs"
        )
    _require_file_unchanged(
        artifact_path, artifact_record, label="formal transfer artifact"
    )
    return dict(payload)


def _routed_evaluator_settings(
    payload: Mapping[str, Any],
    *,
    artifact_record: Mapping[str, Any],
) -> Dict[str, Any]:
    patch = payload["components"]["patch"]
    external = payload["components"]["merged_external"]
    routing = payload["inference_contract"]["routing"]
    policy = routing["policy"]
    registry = routing["descriptor_registry"]
    overrides = dict(policy["overrides"])
    descriptor_ids = list(ROUTE_CANDIDATE_DESCRIPTOR_IDS)
    descriptor_grid_by_id: Dict[str, Dict[str, Any]] = {}
    for descriptor_id in descriptor_ids:
        descriptor = registry[descriptor_id]
        descriptor_grid_by_id[descriptor_id] = {
            "transfer_mode": str(descriptor["transfer_mode"]),
            "iou_power": (
                None
                if descriptor["iou_power"] is None
                else float(descriptor["iou_power"])
            ),
            "patch_weight": float(descriptor["patch_weight"]),
            "text_weight": float(descriptor["text_weight"]),
        }
    fixed_grid = [descriptor_grid_by_id[key] for key in descriptor_ids]
    transfer_modes = list(
        dict.fromkeys(row["transfer_mode"] for row in fixed_grid)
    )
    iou_powers = list(
        dict.fromkeys(
            float(row["iou_power"])
            for row in fixed_grid
            if row["iou_power"] is not None
        )
    )
    patch_weights = list(
        dict.fromkeys(float(row["patch_weight"]) for row in fixed_grid)
    )
    text_weights = list(
        dict.fromkeys(float(row["text_weight"]) for row in fixed_grid)
    )
    return {
        "formal_artifact_version": 2,
        "transfer_modes": transfer_modes,
        "iou_powers": iou_powers,
        "patch_weights": patch_weights,
        "text_weights": text_weights,
        "candidate_topk": 50,
        "contract_patch_rank_weight": float(
            patch["runtime_contract"]["contract_patch_rank_weight"]
        ),
        "external_query_count": 900,
        "patch_config": dict(patch["config"]["leaf"]),
        "patch_checkpoints": [
            {
                key: patch["checkpoint"][key]
                for key in ("path", "size_bytes", "sha256")
            }
        ],
        "external_config": dict(external["config"]["leaf"]),
        "external_checkpoint": {
            key: external["checkpoint"][key]
            for key in ("path", "size_bytes", "sha256")
        },
        "caption_provenance": dict(
            payload["inference_contract"]["caption_provenance"]
        ),
        "evaluation_protocol": dict(payload["evaluation_protocol"]),
        "artifact": dict(artifact_record),
        "artifact_identity": dict(payload["artifact_identity"]),
        "artifact_payload": dict(payload),
        "route_selection": dict(routing),
        "route_policy": dict(policy),
        "route_policy_sha256": str(routing["policy_sha256"]),
        "route_default_descriptor_id": str(policy["default_descriptor_id"]),
        "route_overrides": overrides,
        "descriptor_registry": dict(registry),
        "descriptor_grid_by_id": descriptor_grid_by_id,
        "fixed_grid": fixed_grid,
    }


def _fulltext_routed_evaluator_settings(
    payload: Mapping[str, Any],
    *,
    artifact_record: Mapping[str, Any],
) -> Dict[str, Any]:
    components = payload["components"]
    routed_v2_component = components["routed_v2"]
    fulltext_component = components["fulltext_route_gate"]
    routed_v2_record = routed_v2_component["artifact"]
    routed_v2_path = _resolve(routed_v2_record["path"])
    observed_routed_v2_record = file_record(routed_v2_path)
    if observed_routed_v2_record != routed_v2_record:
        raise ExternalRankTransferArtifactError(
            "routed v2 artifact file identity drifted while deriving settings"
        )
    routed_v2_payload = load_and_verify_formal_transfer_artifact(routed_v2_path)
    _require_file_unchanged(
        routed_v2_path,
        routed_v2_record,
        label="routed v2 artifact",
    )
    settings = _routed_evaluator_settings(
        routed_v2_payload,
        artifact_record=routed_v2_record,
    )
    canonical_routing = routed_v2_payload["inference_contract"]["routing"]
    routing = payload["inference_contract"]["routing"]
    policy = routing["policy"]
    unconditional_overrides = dict(policy["unconditional_overrides"])
    conditional_overrides = dict(policy["conditional_overrides"])
    settings.update(
        {
            "formal_artifact_version": 3,
            "caption_provenance": dict(
                payload["inference_contract"]["caption_provenance"]
            ),
            "evaluation_protocol": dict(payload["evaluation_protocol"]),
            "artifact": dict(artifact_record),
            "artifact_identity": dict(payload["artifact_identity"]),
            "artifact_payload": dict(payload),
            "route_selection": dict(routing),
            "route_policy": dict(policy),
            "route_policy_sha256": str(routing["policy_sha256"]),
            "route_default_descriptor_id": str(
                policy["default_descriptor_id"]
            ),
            "route_overrides": unconditional_overrides,
            "route_unconditional_overrides": unconditional_overrides,
            "route_conditional_overrides": conditional_overrides,
            "canonical_route_selection": dict(canonical_routing),
            "canonical_route_policy": dict(canonical_routing["policy"]),
            "canonical_route_policy_sha256": str(
                canonical_routing["policy_sha256"]
            ),
            "routed_v2_artifact": dict(routed_v2_component),
            "routed_v2_artifact_identity": dict(
                routed_v2_component["artifact_identity"]
            ),
            "fulltext_route_gate": dict(fulltext_component),
            "fulltext_route_gate_artifact": dict(
                fulltext_component["artifact"]
            ),
            "fulltext_route_gate_artifact_identity": dict(
                fulltext_component["artifact_identity"]
            ),
        }
    )
    return settings


def evaluator_settings_from_artifact(
    artifact: str | Path,
) -> Dict[str, Any]:
    artifact_path = _resolve(artifact)
    artifact_record = file_record(artifact_path)
    payload = load_and_verify_formal_transfer_artifact(artifact_path)
    _require_file_unchanged(
        artifact_path, artifact_record, label="formal transfer artifact"
    )
    if payload.get("schema") == ROUTED_V3_SCHEMA:
        return _fulltext_routed_evaluator_settings(
            payload, artifact_record=artifact_record
        )
    if payload.get("schema") == ROUTED_V2_SCHEMA:
        return _routed_evaluator_settings(
            payload, artifact_record=artifact_record
        )
    patch = payload["components"]["patch"]
    external = payload["components"]["merged_external"]
    transfer = payload["inference_contract"]["transfer"]
    mode = str(transfer["mode"])
    power = transfer["iou_power"]
    settings = {
        "transfer_modes": [mode],
        "iou_powers": [] if power is None else [float(power)],
        "patch_weights": [float(transfer["patch_weight"])],
        "text_weights": [float(transfer["text_weight"])],
        "candidate_topk": 50,
        "contract_patch_rank_weight": float(
            patch["runtime_contract"]["contract_patch_rank_weight"]
        ),
        "external_query_count": 900,
        "patch_config": dict(patch["config"]["leaf"]),
        "patch_checkpoints": [
            {
                key: patch["checkpoint"][key]
                for key in ("path", "size_bytes", "sha256")
            }
        ],
        "external_config": dict(external["config"]["leaf"]),
        "external_checkpoint": {
            key: external["checkpoint"][key]
            for key in ("path", "size_bytes", "sha256")
        },
        "caption_provenance": dict(
            payload["inference_contract"]["caption_provenance"]
        ),
        "evaluation_protocol": dict(payload["evaluation_protocol"]),
        "artifact": artifact_record,
        "artifact_identity": dict(payload["artifact_identity"]),
        "transfer_contract": dict(transfer),
        "artifact_payload": payload,
    }
    variants = []
    if mode in {"nearest_iou", "top_query_nearest_candidate"}:
        variants.append((mode, None))
    else:
        variants.append((mode, float(power)))
    settings["fixed_grid"] = [
        {
            "transfer_mode": transfer_mode,
            "iou_power": transfer_power,
            "patch_weight": float(transfer["patch_weight"]),
            "text_weight": float(transfer["text_weight"]),
        }
        for transfer_mode, transfer_power in variants
    ]
    return settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or verify a formal external rank-transfer artifact"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--output", required=True)
    create.add_argument("--patch-config", required=True)
    create.add_argument("--patch-checkpoint", required=True)
    create.add_argument("--merged-external-config", required=True)
    create.add_argument("--merged-external-checkpoint", required=True)
    create.add_argument("--source-rank-checkpoint", required=True)
    create.add_argument("--source-confidence-checkpoint", required=True)
    create.add_argument("--base-baseline-checkpoint", required=True)
    create.add_argument("--mode", choices=FORMAL_TRANSFER_MODES, required=True)
    create.add_argument("--iou-power", type=float, default=None)
    create.add_argument("--patch-weight", type=float, required=True)
    create.add_argument("--text-weight", type=float, required=True)
    create.add_argument("--base-seed", type=int, default=42)
    routed = subparsers.add_parser("create-routed-v2")
    routed.add_argument("--output", required=True)
    routed.add_argument("--patch-config", required=True)
    routed.add_argument("--patch-checkpoint", required=True)
    routed.add_argument("--merged-external-config", required=True)
    routed.add_argument("--merged-external-checkpoint", required=True)
    routed.add_argument("--source-rank-checkpoint", required=True)
    routed.add_argument("--source-confidence-checkpoint", required=True)
    routed.add_argument("--base-baseline-checkpoint", required=True)
    routed.add_argument("--caption-route-artifact", required=True)
    routed.add_argument("--base-seed", type=int, default=42)
    fulltext_routed = subparsers.add_parser("create-routed-v3")
    fulltext_routed.add_argument("--output", required=True)
    fulltext_routed.add_argument("--routed-v2-artifact", required=True)
    fulltext_routed.add_argument("--fulltext-route-gate-artifact", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--artifact", required=True)
    args = parser.parse_args()
    if args.command == "create":
        kwargs = vars(args).copy()
        kwargs.pop("command")
        output = kwargs.pop("output")
        payload = create_formal_transfer_artifact(output, **kwargs)
        print(json.dumps(payload["artifact_identity"], sort_keys=True))
    elif args.command == "create-routed-v2":
        kwargs = vars(args).copy()
        kwargs.pop("command")
        output = kwargs.pop("output")
        payload = create_formal_routed_transfer_artifact(output, **kwargs)
        print(json.dumps(payload["artifact_identity"], sort_keys=True))
    elif args.command == "create-routed-v3":
        kwargs = vars(args).copy()
        kwargs.pop("command")
        output = kwargs.pop("output")
        payload = create_formal_fulltext_routed_transfer_artifact(output, **kwargs)
        print(json.dumps(payload["artifact_identity"], sort_keys=True))
    else:
        payload = load_and_verify_formal_transfer_artifact(args.artifact)
        print(json.dumps(payload["artifact_identity"], sort_keys=True))


if __name__ == "__main__":
    main()
