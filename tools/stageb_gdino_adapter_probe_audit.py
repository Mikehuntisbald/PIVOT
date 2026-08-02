#!/usr/bin/env python3
"""Fail-closed lineage and checkpoint audits for two-phase GDINO adapter probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

try:
    import torch
except ModuleNotFoundError:  # Static/dry-run audits do not need PyTorch.
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.stageb_dependency_audit import (  # noqa: E402
    DependencyAuditError,
    config_import_chain,
    local_python_dependency_paths,
)
from util.path_compat import remap_legacy_path  # noqa: E402


SCHEMA = "stageb-gdino-adapter-two-phase-probe-v1"
RANK_MILESTONES = (50, 100, 250, 500, 1000, 2000, 5000)
CONFIDENCE_MILESTONES = (50, 100, 250, 500)
MILESTONES_BY_PHASE = {
    "rank": RANK_MILESTONES,
    "confidence": CONFIDENCE_MILESTONES,
}
ADAPTER_PREFIX = "stage_b_gdino_score_adapter."
RANK_PARTS = ("rank_norm.", "rank_trunk.", "rank_output.")
CONFIDENCE_PARTS = (
    "confidence_norm.",
    "confidence_trunk.",
    "confidence_gate.",
)

PHASE_SPECS = {
    "rank": {
        "mode": "rank_only",
        "mode_code": 1,
        "scope": "",
        "scope_code": 0,
        "optimizer_branch": "rank",
        "learning_rate": 3.0e-5,
        "confidence_objective_code": 0,
        "criterion_positive_trust_margin": 0.0,
        "criterion_positive_trust_weight": 0.0,
        "paired_margin_weight": 0.0,
        "queue_size": 0,
        "queue_min_count": 0,
        "config": "config/ablations/cfg_stageb_gdino_score_adapter_rank_three_ref.py",
        "datasets": "config/datasets_stageb_gdino_adapter_rank_three_ref.json",
        "sources": (
            {
                "path": "data/ablations/gdino_ft_stage_b_rebuild_three_ref_20260711/"
                "stageb_gdino_ft_refcoco_stageb_phrase_v1_vg.jsonl",
                "rows": 120624,
                "sha256": "9578a59c37d7fff4477db923457e2ad9b3119a28ec7277c901517a6f27063ea1",
                "dataset_mode": "odvg",
                "mix_weight": 2.0,
            },
            {
                "path": "data/ablations/gdino_ft_stage_b_rebuild_20260711/"
                "stageb_gdino_ft_0_refcocoplus_stageb_phrase_v1_vg.jsonl",
                "rows": 120191,
                "sha256": "015e68210d798d250e88e12d57779d56b5d47b233d2bfb6ec3582def6c379562",
                "dataset_mode": "odvg",
                "mix_weight": 2.0,
            },
            {
                "path": "data/ablations/gdino_ft_stage_b_rebuild_20260711/"
                "stageb_gdino_ft_1_refcocog_stageb_phrase_v1_vg.jsonl",
                "rows": 80512,
                "sha256": "cd4eda88128acd1799ef707c8c31011088f7ddfe3c34c70c5fff2c2594b08c0e",
                "dataset_mode": "odvg",
                "mix_weight": 2.0,
            },
        ),
    },
    "confidence": {
        "mode": "confidence_only",
        "mode_code": 2,
        "scope": "benchmark_dataft_alltn",
        "scope_code": 2,
        "optimizer_branch": "confidence",
        "learning_rate": 3.0e-4,
        "confidence_objective_code": 2,
        "criterion_positive_trust_margin": 0.02,
        "criterion_positive_trust_weight": 1.0,
        "paired_margin_weight": 0.25,
        "queue_size": 512,
        "queue_min_count": 256,
        "config": "config/ablations/cfg_stageb_gdino_score_adapter_dataft.py",
        "datasets": "config/datasets_stageb_gdino_adapter_dataft_pairs.json",
        "sources": (
            {
                "path": "data/ablations/stageb_gdino_adapter_dataft_20260711/"
                "benchmark_dataft_alltn_pairs.jsonl",
                "rows": 60000,
                "sha256": "90bb07027b93e0e4a399960b34dff7dbc44be065b8e0080517b130f379ee8b14",
                "dataset_mode": "patch_episode",
                "mix_weight": 1.0,
            },
        ),
    },
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
    "tools/stageb_gdino_adapter_probe_audit.py",
    "tools/stageb_dependency_audit.py",
)
ORCHESTRATION_PATHS = (
    "tools/run_stageb_gdino_adapter_two_phase_probe.sh",
)


class ProbeAuditError(RuntimeError):
    pass


def resolve_path(value: str | Path) -> Path:
    path = remap_legacy_path(value, repo_root=REPO_ROOT)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise ProbeAuditError(f"required file is missing: {path}")
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def count_nonempty_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProbeAuditError(f"could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ProbeAuditError(f"expected a JSON object: {path}")
    return value


def load_checkpoint(path: Path) -> MutableMapping[str, Any]:
    if torch is None:
        raise ProbeAuditError("PyTorch is required for checkpoint audits")
    if not path.is_file():
        raise ProbeAuditError(f"checkpoint is missing: {path}")
    try:
        value = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, MutableMapping):
        raise ProbeAuditError(f"checkpoint payload is not a mapping: {path}")
    return value


def checkpoint_args(payload: Mapping[str, Any]) -> Dict[str, Any]:
    value = payload.get("args")
    if isinstance(value, Mapping):
        return dict(value)
    if value is not None and hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _tensor_bytes(value: torch.Tensor) -> memoryview:
    tensor = value.detach().cpu().contiguous()
    if tensor.numel() == 0:
        return memoryview(b"")
    # PyTorch rejects dtype-changing ``view`` directly on a 0-D tensor.  The
    # checkpoint contracts legitimately persist scalar buffers, so flatten
    # first while leaving the separately hashed dtype/shape header unchanged.
    return memoryview(tensor.reshape(-1).view(torch.uint8).numpy())


def tensor_state_sha256(
    state: Mapping[str, Any],
    keys: Iterable[str],
) -> str:
    selected = sorted(set(str(key) for key in keys))
    if not selected:
        raise ProbeAuditError("cannot hash an empty tensor-state selection")
    digest = hashlib.sha256()
    for key in selected:
        value = state.get(key)
        if not torch.is_tensor(value):
            raise ProbeAuditError(f"model state {key!r} is missing or is not a tensor")
        header = json.dumps(
            [key, str(value.dtype), list(value.shape)],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _adapter_key_groups(state: Mapping[str, Any]) -> Dict[str, Sequence[str]]:
    adapter_keys = sorted(
        str(key) for key in state if str(key).startswith(ADAPTER_PREFIX)
    )
    rank_keys = [
        key
        for key in adapter_keys
        if key.removeprefix(ADAPTER_PREFIX).startswith(RANK_PARTS)
    ]
    confidence_keys = [
        key
        for key in adapter_keys
        if key.removeprefix(ADAPTER_PREFIX).startswith(CONFIDENCE_PARTS)
    ]
    unknown = sorted(set(adapter_keys).difference(rank_keys).difference(confidence_keys))
    return {
        "all": adapter_keys,
        "rank": rank_keys,
        "confidence": confidence_keys,
        "unknown": unknown,
    }


def _all_zero(state: Mapping[str, Any], keys: Sequence[str]) -> bool:
    return bool(keys) and all(
        torch.is_tensor(state[key])
        and int(torch.count_nonzero(state[key]).item()) == 0
        for key in keys
    )


def model_hash_record(state: Mapping[str, Any]) -> Dict[str, Any]:
    if not state:
        raise ProbeAuditError("checkpoint has an empty model state")
    groups = _adapter_key_groups(state)
    base_keys = sorted(set(str(key) for key in state).difference(groups["all"]))
    result: Dict[str, Any] = {
        "model_state_keys": len(state),
        "base_state_keys": len(base_keys),
        "adapter_state_keys": len(groups["all"]),
        "base_model_sha256": tensor_state_sha256(state, base_keys),
    }
    if groups["all"]:
        if groups["unknown"] or not groups["rank"] or not groups["confidence"]:
            raise ProbeAuditError(
                "adapter checkpoint has incomplete or unknown branch keys: "
                f"rank={len(groups['rank'])}, confidence={len(groups['confidence'])}, "
                f"unknown={list(groups['unknown'])[:8]}"
            )
        rank_final = [
            ADAPTER_PREFIX + "rank_output.weight",
            ADAPTER_PREFIX + "rank_output.bias",
        ]
        confidence_final = [
            ADAPTER_PREFIX + "confidence_gate.4.weight",
            ADAPTER_PREFIX + "confidence_gate.4.bias",
        ]
        for key in rank_final + confidence_final:
            if key not in state:
                raise ProbeAuditError(f"adapter checkpoint is missing final layer {key}")
        result.update(
            {
                "adapter_sha256": tensor_state_sha256(state, groups["all"]),
                "rank_sha256": tensor_state_sha256(state, groups["rank"]),
                "confidence_sha256": tensor_state_sha256(
                    state, groups["confidence"]
                ),
                "rank_final_zero": _all_zero(state, rank_final),
                "confidence_final_zero": _all_zero(state, confidence_final),
            }
        )
    return result


def checkpoint_record(path: Path, *, include_file_hash: bool = True) -> Dict[str, Any]:
    payload = load_checkpoint(path)
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise ProbeAuditError(f"checkpoint has no non-empty model mapping: {path}")
    args = checkpoint_args(payload)
    record: Dict[str, Any] = {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "epoch": int(payload.get("epoch", -1)),
        "iteration": int(payload.get("iteration", 0) or 0),
        "epoch_finished": payload.get("epoch_finished"),
        "checkpoint_reason": payload.get("checkpoint_reason"),
        "has_criterion": isinstance(payload.get("criterion"), Mapping),
        "has_optimizer": isinstance(payload.get("optimizer"), Mapping),
        "has_lr_scheduler": isinstance(payload.get("lr_scheduler"), Mapping),
        "has_scaler": isinstance(payload.get("scaler"), Mapping),
        "has_rng_state": isinstance(payload.get("rng_state"), Mapping),
        "has_epoch_rng_state": isinstance(payload.get("epoch_rng_state"), Mapping),
        "checkpoint_args": {
            key: args.get(key)
            for key in (
                "config_file",
                "datasets",
                "world_size",
                "batch_size",
                "distributed",
                "amp",
                "max_train_iters",
                "pretrain_model_path",
                "resume",
                "stage_b_gdino_adapter_train_mode",
                "stage_b_gdino_tn_scope",
                "stage_b_gdino_gate_pool_temperature",
                "stage_b_gdino_gate_topk",
            )
        },
    }
    if include_file_hash:
        record["sha256"] = sha256_file(path)
    record.update(model_hash_record(state))
    del payload
    return record


def _load_cfg(path: Path):
    from util.slconfig import SLConfig

    return SLConfig.fromfile(str(path))


def _expected_source_records(phase: str) -> Sequence[Dict[str, Any]]:
    spec = PHASE_SPECS[phase]
    records = []
    for source in spec["sources"]:
        path = resolve_path(source["path"])
        record = file_record(path)
        rows = count_nonempty_lines(path)
        if rows != int(source["rows"]):
            raise ProbeAuditError(
                f"{phase} source row drift for {path}: expected {source['rows']}, got {rows}"
            )
        if record["sha256"] != source["sha256"]:
            raise ProbeAuditError(
                f"{phase} source hash drift for {path}: expected {source['sha256']}, "
                f"got {record['sha256']}"
            )
        record.update(
            {
                "rows": rows,
                "dataset_mode": source["dataset_mode"],
                "mix_weight": source["mix_weight"],
            }
        )
        records.append(record)
    return records


def validate_phase_static(phase: str) -> Dict[str, Any]:
    if phase not in PHASE_SPECS:
        raise ProbeAuditError(f"unknown phase {phase!r}")
    spec = PHASE_SPECS[phase]
    config_path = resolve_path(spec["config"])
    datasets_path = resolve_path(spec["datasets"])
    cfg = _load_cfg(config_path)
    expected_cfg = {
        "stage_b_gdino_score_adapter": True,
        "stage_b_gdino_adapter_train_mode": spec["mode"],
        "stage_b_gdino_tn_scope": spec["scope"],
        "patch_only": False,
        "stage_b": False,
        "enable_patch_branch": False,
        "batch_size": 4,
        "epochs": 1,
        "skip_eval": True,
        "find_unused_params": False,
        "data_aug_hflip_prob": 0.0,
        "stage_b_gdino_gate_pool_temperature": 0.01,
        "stage_b_gdino_gate_topk": 3,
        "stage_b_gdino_confidence_objective": "detached_recent_q05_trust",
        "stage_b_gdino_fpr_temperature": 0.1,
        "stage_b_gdino_fpr_margin": 0.0,
        "stage_b_gdino_paired_margin": 0.05,
        "stage_b_gdino_paired_margin_weight": spec["paired_margin_weight"],
        "stage_b_gdino_positive_trust_margin": 0.02,
        "stage_b_gdino_positive_trust_weight": 1.0,
    }
    if phase == "rank":
        expected_cfg.update(
            {
                "stage_b_gdino_rank_weight": 1.0,
                "stage_b_gdino_confidence_weight": 0.0,
                "stage_b_gdino_paired_margin_weight": 0.0,
                "stage_b_gdino_queue_size": 0,
                "stage_b_gdino_queue_min_count": 0,
                "stage_b_gdino_rank_lr": 3.0e-5,
            }
        )
    else:
        expected_cfg.update(
            {
                "stage_b_gdino_rank_weight": 0.0,
                "stage_b_gdino_confidence_weight": 1.0,
                "stage_b_gdino_paired_margin_weight": 0.25,
                "stage_b_gdino_gate_lr": 3.0e-4,
                "stage_b_gdino_queue_size": spec["queue_size"],
                "stage_b_gdino_queue_min_count": spec["queue_min_count"],
            }
        )
    for key, expected in expected_cfg.items():
        observed = getattr(cfg, key, None)
        if isinstance(expected, float):
            matches = isinstance(observed, (int, float)) and math.isclose(
                float(observed), expected, rel_tol=0.0, abs_tol=1e-12
            )
        else:
            matches = observed == expected
        if not matches:
            raise ProbeAuditError(
                f"{phase} config mismatch for {key}: expected {expected!r}, got {observed!r}"
            )

    dataset_meta = read_json(datasets_path)
    train = dataset_meta.get("train")
    if not isinstance(train, list) or len(train) != len(spec["sources"]):
        raise ProbeAuditError(
            f"{phase} dataset config must contain exactly {len(spec['sources'])} train sources"
        )
    sources = _expected_source_records(phase)
    for index, (entry, source, source_record) in enumerate(
        zip(train, spec["sources"], sources)
    ):
        if not isinstance(entry, Mapping):
            raise ProbeAuditError(f"{phase} dataset entry {index} is not a mapping")
        observed_path = entry.get("anno")
        if not observed_path or resolve_path(str(observed_path)) != Path(source_record["path"]):
            raise ProbeAuditError(f"{phase} dataset entry {index} points to the wrong annotation")
        for key in ("dataset_mode", "mix_weight"):
            observed = entry.get(key)
            expected = source[key]
            if observed != expected:
                raise ProbeAuditError(
                    f"{phase} dataset entry {index} {key}: expected {expected!r}, got {observed!r}"
                )
    if dataset_meta.get("val") != []:
        raise ProbeAuditError(f"{phase} probe dataset must have an empty val list")
    if phase == "confidence":
        entry = train[0]
        required = {
            "source": "stage_b_gdino_adapter_benchmark_dataft_alltn",
            "neg_episode_prob": 0.0,
            "require_benchmark_dataft_alltn": True,
            "stage_b_gdino_adapter_no_support": True,
        }
        for key, expected in required.items():
            if entry.get(key) != expected:
                raise ProbeAuditError(
                    f"confidence dataset requires {key}={expected!r}, got {entry.get(key)!r}"
                )
        data_audit = read_json(
            resolve_path("data/ablations/stageb_gdino_adapter_dataft_20260711/audit.json")
        )
        if (
            data_audit.get("schema") != "stage-b-gdino-adapter-dataft-pairs-v1"
            or data_audit.get("tn_scope") != "benchmark_dataft_alltn"
            or data_audit.get("rows") != 60000
            or data_audit.get("output_sha256") != spec["sources"][0]["sha256"]
        ):
            raise ProbeAuditError("confidence pair-data audit does not match the fixed 60k source")

    try:
        config_paths = config_import_chain(config_path, root=REPO_ROOT)
        code_paths = local_python_dependency_paths(
            TRAIN_CODE_ENTRIES,
            root=REPO_ROOT,
            include=TRAIN_CODE_INCLUDE,
        )
    except DependencyAuditError as error:
        raise ProbeAuditError(str(error)) from error
    return {
        "phase": phase,
        "train_mode": spec["mode"],
        "tn_scope": spec["scope"],
        "config": file_record(config_path),
        "config_import_chain": [file_record(path) for path in config_paths],
        "resolved_config": expected_cfg,
        "datasets": file_record(datasets_path),
        "sources": list(sources),
        "total_source_rows": sum(int(row["rows"]) for row in sources),
        "code": [file_record(path) for path in code_paths],
        "orchestration": [
            file_record(resolve_path(path)) for path in ORCHESTRATION_PATHS
        ],
    }


def _validate_fixed_baseline(path: Path) -> Dict[str, Any]:
    from tools.stageb_fixed_protocol_audit import (
        ProtocolError,
        _completed_stageb_baseline_checkpoint_record,
    )

    try:
        record = _completed_stageb_baseline_checkpoint_record(path)
    except ProtocolError as error:
        raise ProbeAuditError(f"fixed baseline validation failed: {error}") from error
    completion_path = path.parent / "protocol_train_complete.json"
    completion = read_json(completion_path)
    authoritative = completion.get("authoritative_checkpoint")
    if not isinstance(authoritative, Mapping):
        raise ProbeAuditError(
            f"fixed baseline completion audit has no authoritative checkpoint: {completion_path}"
        )
    if authoritative.get("sha256") != record.get("sha256"):
        raise ProbeAuditError(
            "fixed baseline checkpoint does not match protocol_train_complete.json"
        )
    record["protocol_train_complete"] = file_record(completion_path)
    record.update(checkpoint_record(path))
    if int(record.get("adapter_state_keys", 0)) != 0:
        raise ProbeAuditError("rank phase must start from a pure baseline without adapter keys")
    return record


def _validate_rank_initial(path: Path, audit_path: Path) -> Dict[str, Any]:
    verified = _verify_milestone_checkpoint(path, audit_path)
    if (
        verified.get("phase") != "rank"
        or verified.get("iteration") not in RANK_MILESTONES
    ):
        raise ProbeAuditError("confidence phase must initialize from an audited rank milestone")
    return dict(verified["checkpoint"])


def _stable_preflight_payload(
    *,
    phase: str,
    initial_checkpoint: Path,
    initial_audit: Path | None,
    world_size: int,
    per_gpu_batch: int,
) -> Dict[str, Any]:
    if world_size != 2 or per_gpu_batch != 4:
        raise ProbeAuditError(
            "adapter probes require two DDP ranks with per-GPU batch 4 (global batch 8)"
        )
    static = validate_phase_static(phase)
    if phase == "rank":
        initial = _validate_fixed_baseline(initial_checkpoint)
        initialization = "fixed_pure_stageb_dataft_to_rank_pretrain_model_path"
    else:
        if initial_audit is None:
            raise ProbeAuditError("confidence preflight requires --initial-audit")
        initial = _validate_rank_initial(initial_checkpoint, initial_audit)
        initialization = "selected_rank_to_confidence_pretrain_model_path"
    return {
        "schema": SCHEMA,
        "kind": "phase_preflight",
        "phase": phase,
        "static": static,
        "initial_checkpoint": initial,
        "initial_audit": file_record(initial_audit) if initial_audit else None,
        "launch": {
            "world_size": world_size,
            "per_gpu_batch": per_gpu_batch,
            "global_batch": world_size * per_gpu_batch,
            "milestones": list(MILESTONES_BY_PHASE[phase]),
            "initialization": initialization,
            "first_invocation": "pretrain_model_path",
            "later_invocations": "resume",
        },
    }


def _preflight_equivalent(existing: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    return dict(existing) == dict(current)


def _verify_current_preflight(preflight_path: Path) -> Dict[str, Any]:
    preflight = read_json(preflight_path)
    phase = str(preflight.get("phase", ""))
    initial = preflight.get("initial_checkpoint")
    launch = preflight.get("launch")
    initial_audit = preflight.get("initial_audit")
    if (
        phase not in PHASE_SPECS
        or not isinstance(initial, Mapping)
        or not isinstance(launch, Mapping)
        or not initial.get("path")
    ):
        raise ProbeAuditError("phase preflight is missing required lineage fields")
    current = _stable_preflight_payload(
        phase=phase,
        initial_checkpoint=resolve_path(str(initial["path"])),
        initial_audit=(
            resolve_path(str(initial_audit["path"]))
            if isinstance(initial_audit, Mapping) and initial_audit.get("path")
            else None
        ),
        world_size=int(launch.get("world_size", -1)),
        per_gpu_batch=int(launch.get("per_gpu_batch", -1)),
    )
    if not _preflight_equivalent(preflight, current):
        raise ProbeAuditError(
            "phase config/data/code/initial lineage drifted since preflight"
        )
    return preflight


def _cmd_static(args: argparse.Namespace) -> None:
    phases = [args.phase] if args.phase != "all" else ["rank", "confidence"]
    payload = {
        "schema": SCHEMA,
        "kind": "static_probe_inputs",
        "milestones": {
            phase: list(MILESTONES_BY_PHASE[phase]) for phase in phases
        },
        "world_size": 2,
        "per_gpu_batch": 4,
        "global_batch": 8,
        "phases": {phase: validate_phase_static(phase) for phase in phases},
    }
    if args.output:
        write_json(resolve_path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))


def _cmd_phase_preflight(args: argparse.Namespace) -> None:
    output = resolve_path(args.output)
    output_dir = resolve_path(args.output_dir)
    initial_audit = resolve_path(args.initial_audit) if args.initial_audit else None
    current = _stable_preflight_payload(
        phase=args.phase,
        initial_checkpoint=resolve_path(args.initial_checkpoint),
        initial_audit=initial_audit,
        world_size=int(args.world_size),
        per_gpu_batch=int(args.per_gpu_batch),
    )
    if args.continue_run:
        if not output.is_file():
            raise ProbeAuditError(f"--continue-run requires an existing preflight: {output}")
        existing = read_json(output)
        if not _preflight_equivalent(existing, current):
            raise ProbeAuditError(
                "phase preflight drifted since the original launch; use a fresh output directory"
            )
        print(f"[OK] unchanged {args.phase} preflight: {output}")
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ProbeAuditError(
            f"fresh {args.phase} output directory is not empty: {output_dir}; "
            "use --continue-run only for an audited continuation"
        )
    if output.exists():
        raise ProbeAuditError(f"refusing to overwrite phase preflight: {output}")
    write_json(output, current)
    print(f"[OK] wrote {args.phase} phase preflight: {output}")


def _resolve_checkpoint_arg_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return resolve_path(str(value))


def _scalar_code(state: Mapping[str, Any], key: str) -> int:
    value = state.get(key)
    if not torch.is_tensor(value) or value.numel() != 1:
        raise ProbeAuditError(f"criterion state is missing scalar {key}")
    return int(value.detach().reshape(-1)[0].item())


def _scalar_float(state: Mapping[str, Any], key: str) -> float:
    value = state.get(key)
    if not torch.is_tensor(value) or value.numel() != 1:
        raise ProbeAuditError(f"criterion state is missing scalar {key}")
    result = float(value.detach().reshape(-1)[0].item())
    if not math.isfinite(result):
        raise ProbeAuditError(f"criterion state scalar {key} is not finite")
    return result


def _validate_criterion_protocol(
    phase: str,
    criterion: Mapping[str, Any],
) -> None:
    spec = PHASE_SPECS[phase]
    integer_contract = {
        "criterion_confidence_objective_code": spec["confidence_objective_code"],
        "criterion_queue_size": spec["queue_size"],
        "criterion_queue_min_count": spec["queue_min_count"],
    }
    for key, expected in integer_contract.items():
        observed = _scalar_code(criterion, key)
        if observed != int(expected):
            raise ProbeAuditError(
                f"{phase} criterion {key}: expected {expected}, got {observed}"
            )
    float_contract = {
        "criterion_positive_trust_margin": spec[
            "criterion_positive_trust_margin"
        ],
        "criterion_positive_trust_weight": spec[
            "criterion_positive_trust_weight"
        ],
    }
    for key, expected in float_contract.items():
        observed = _scalar_float(criterion, key)
        if not math.isclose(observed, float(expected), rel_tol=0.0, abs_tol=1e-6):
            raise ProbeAuditError(
                f"{phase} criterion {key}: expected {expected}, got {observed}"
            )

    queue_size = int(spec["queue_size"])
    for key in ("fpr_positive_queue", "fpr_negative_queue"):
        value = criterion.get(key)
        if (
            not torch.is_tensor(value)
            or value.dim() != 1
            or int(value.numel()) != queue_size
            or not bool(torch.isfinite(value.detach()).all().item())
        ):
            raise ProbeAuditError(
                f"{phase} criterion {key} must be a finite length-{queue_size} tensor"
            )
    count = _scalar_code(criterion, "fpr_queue_count")
    pointer = _scalar_code(criterion, "fpr_queue_ptr")
    if not 0 <= count <= queue_size:
        raise ProbeAuditError(f"{phase} criterion queue count is out of range: {count}")
    if queue_size == 0:
        if count != 0 or pointer != 0:
            raise ProbeAuditError("rank criterion must keep its disabled queue empty")
    else:
        if not 0 <= pointer < queue_size:
            raise ProbeAuditError(
                f"{phase} criterion queue pointer is out of range: {pointer}"
            )
        if count < queue_size and pointer != count:
            raise ProbeAuditError(
                f"{phase} partially-filled queue pointer/count mismatch: {pointer}/{count}"
            )
        if count < int(spec["queue_min_count"]):
            raise ProbeAuditError(
                f"{phase} milestone queue is not warm: {count} < {spec['queue_min_count']}"
            )


def _validate_training_checkpoint_common(
    *,
    phase: str,
    checkpoint: Path,
    preflight: Mapping[str, Any],
    expected_target: int,
    source_checkpoint: Path | None,
    require_exact_iteration: bool,
) -> Dict[str, Any]:
    milestones = MILESTONES_BY_PHASE[phase]
    if expected_target not in milestones:
        raise ProbeAuditError(
            f"{phase} expected target must be one of {milestones}"
        )
    spec = PHASE_SPECS[phase]
    payload = load_checkpoint(checkpoint)
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise ProbeAuditError("training checkpoint has no model state")
    args = checkpoint_args(payload)
    iteration = int(payload.get("iteration", 0) or 0)
    if require_exact_iteration:
        if iteration != expected_target:
            raise ProbeAuditError(
                f"{phase} milestone expected iteration {expected_target}, got {iteration}"
            )
        if payload.get("checkpoint_reason") != "max_train_iters":
            raise ProbeAuditError(
                f"{phase} milestone must have reason=max_train_iters, got "
                f"{payload.get('checkpoint_reason')!r}"
            )
    elif not 0 < iteration <= expected_target:
        raise ProbeAuditError(
            f"{phase} live checkpoint iteration must be in [1,{expected_target}], got {iteration}"
        )
    if int(payload.get("epoch", -1)) != 0 or payload.get("epoch_finished") is not False:
        raise ProbeAuditError(
            f"{phase} probe checkpoint must be a mid-epoch epoch=0 checkpoint"
        )
    for key in ("criterion", "optimizer", "lr_scheduler", "scaler", "rng_state", "epoch_rng_state"):
        if not isinstance(payload.get(key), Mapping):
            raise ProbeAuditError(f"{phase} checkpoint is missing resumable {key} state")

    static = preflight.get("static")
    if not isinstance(static, Mapping):
        raise ProbeAuditError("phase preflight has no static section")
    expected_paths = {
        "config_file": Path(static["config"]["path"]),
        "datasets": Path(static["datasets"]["path"]),
    }
    for key, expected in expected_paths.items():
        observed = _resolve_checkpoint_arg_path(args.get(key))
        if observed != expected:
            raise ProbeAuditError(
                f"{phase} checkpoint {key} mismatch: expected {expected}, got {observed}"
            )
    expected_args = {
        "world_size": 2,
        "batch_size": 4,
        "distributed": True,
        "amp": True,
        "max_train_iters": expected_target,
        "data_aug_hflip_prob": 0.0,
        "stage_b_gdino_gate_pool_temperature": 0.01,
        "stage_b_gdino_gate_topk": 3,
        "stage_b_gdino_confidence_objective": "detached_recent_q05_trust",
        "stage_b_gdino_fpr_temperature": 0.1,
        "stage_b_gdino_fpr_margin": 0.0,
        "stage_b_gdino_paired_margin": 0.05,
        "stage_b_gdino_paired_margin_weight": spec["paired_margin_weight"],
        "stage_b_gdino_positive_trust_margin": 0.02,
        "stage_b_gdino_positive_trust_weight": 1.0,
        "stage_b_gdino_queue_size": spec["queue_size"],
        "stage_b_gdino_queue_min_count": spec["queue_min_count"],
        "stage_b_gdino_adapter_train_mode": spec["mode"],
        "stage_b_gdino_tn_scope": spec["scope"],
    }
    for key, expected in expected_args.items():
        observed = args.get(key)
        if isinstance(expected, float):
            matches = isinstance(observed, (int, float)) and math.isclose(
                float(observed), expected, rel_tol=0.0, abs_tol=1e-12
            )
        else:
            matches = observed == expected
        if not matches:
            raise ProbeAuditError(
                f"{phase} checkpoint arg {key}: expected {expected!r}, got {observed!r}"
            )

    pretrain = _resolve_checkpoint_arg_path(args.get("pretrain_model_path"))
    resume = _resolve_checkpoint_arg_path(args.get("resume"))
    if source_checkpoint is not None:
        source_checkpoint = source_checkpoint.resolve()
        if (pretrain == source_checkpoint) == (resume == source_checkpoint):
            raise ProbeAuditError(
                "checkpoint lineage must use exactly one of --pretrain_model_path or --resume "
                f"for source {source_checkpoint}"
            )
        initialization_mode = "pretrain_model_path" if pretrain else "resume"
    else:
        if (pretrain is None) == (resume is None):
            raise ProbeAuditError(
                "checkpoint must record exactly one initialization path"
            )
        initialization_mode = "pretrain_model_path" if pretrain else "resume"

    criterion = payload["criterion"]
    if _scalar_code(criterion, "criterion_train_mode_code") != spec["mode_code"]:
        raise ProbeAuditError(f"{phase} criterion train-mode code mismatch")
    if _scalar_code(criterion, "criterion_scope_code") != spec["scope_code"]:
        raise ProbeAuditError(f"{phase} criterion scope code mismatch")
    _validate_criterion_protocol(phase, criterion)

    optimizer = payload["optimizer"]
    groups = optimizer.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1:
        raise ProbeAuditError(f"{phase} optimizer must contain exactly one parameter group")
    group = groups[0]
    if group.get("stage_b_gdino_branch") != spec["optimizer_branch"]:
        raise ProbeAuditError(f"{phase} optimizer owns the wrong adapter branch")
    observed_lr = float(group.get("lr", math.nan))
    if not math.isclose(
        observed_lr, float(spec["learning_rate"]), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ProbeAuditError(
            f"{phase} optimizer LR mismatch: expected {spec['learning_rate']}, got {observed_lr}"
        )

    record = checkpoint_record(checkpoint)
    initial = preflight.get("initial_checkpoint")
    if not isinstance(initial, Mapping):
        raise ProbeAuditError("phase preflight has no initial checkpoint record")
    if record["base_model_sha256"] != initial.get("base_model_sha256"):
        raise ProbeAuditError(f"{phase} changed the frozen pure-GDINO base parameters")
    if phase == "confidence":
        if record.get("rank_sha256") != initial.get("rank_sha256"):
            raise ProbeAuditError("confidence phase changed rank parameters")
    del payload
    return {
        "record": record,
        "iteration": iteration,
        "initialization_mode": initialization_mode,
    }


def _expected_previous_iteration(phase: str, iteration: int) -> int | None:
    milestones = MILESTONES_BY_PHASE.get(phase)
    if milestones is None:
        raise ProbeAuditError(f"unsupported milestone phase {phase!r}")
    if iteration not in milestones:
        raise ProbeAuditError(
            f"{phase} milestone iteration must be one of {milestones}"
        )
    index = milestones.index(iteration)
    return None if index == 0 else milestones[index - 1]


def _validate_previous_audit(
    phase: str,
    path: Path | None,
    iteration: int,
    *,
    preflight_path: Path,
    replay_stack: set[Path] | None = None,
) -> Dict[str, Any] | None:
    expected_iteration = _expected_previous_iteration(phase, iteration)
    if expected_iteration is None:
        if path is not None:
            raise ProbeAuditError("first milestone must not have a previous audit")
        return None
    if path is None:
        raise ProbeAuditError(
            f"milestone {iteration} requires adjacent previous milestone "
            f"{expected_iteration}"
        )
    return _replay_milestone_audit(
        path,
        expected_phase=phase,
        expected_iteration=expected_iteration,
        expected_preflight_path=preflight_path,
        replay_stack=replay_stack,
    )


def _validate_branch_isolation(
    *,
    phase: str,
    record: Mapping[str, Any],
    initial: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
) -> None:
    if phase == "rank":
        if record.get("rank_final_zero") is True:
            raise ProbeAuditError("rank milestone still has an all-zero rank output layer")
        if record.get("confidence_final_zero") is not True:
            raise ProbeAuditError(
                "rank phase must keep the zero-init confidence gate exactly zero"
            )
        if previous is not None:
            previous_record = previous.get("checkpoint")
            if not isinstance(previous_record, Mapping):
                raise ProbeAuditError("previous milestone audit has no checkpoint record")
            if record.get("confidence_sha256") != previous_record.get("confidence_sha256"):
                raise ProbeAuditError("rank phase changed confidence/gate parameters")
    else:
        if record.get("rank_sha256") != initial.get("rank_sha256"):
            raise ProbeAuditError("confidence phase changed rank parameters")
        if record.get("confidence_sha256") == initial.get("confidence_sha256"):
            raise ProbeAuditError("confidence milestone did not change confidence/gate parameters")
        if record.get("confidence_final_zero") is True:
            raise ProbeAuditError("confidence milestone still has an all-zero gate output layer")


def _same_file_identity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left.get(key) == right.get(key) for key in ("path", "size_bytes", "sha256"))


def _cmd_segment_lineage(args: argparse.Namespace) -> None:
    phase = args.phase
    target = int(args.expected_target)
    preflight_path = resolve_path(args.preflight)
    source = resolve_path(args.source_checkpoint)
    preflight = read_json(preflight_path)
    if preflight.get("phase") != phase:
        raise ProbeAuditError("segment lineage phase does not match preflight")
    source_record = file_record(source)
    previous_path = resolve_path(args.previous_audit) if args.previous_audit else None
    previous = _validate_previous_audit(
        phase,
        previous_path,
        target,
        preflight_path=preflight_path,
    )
    recovery_path = (
        resolve_path(args.recovery_inspection) if args.recovery_inspection else None
    )
    ancestry = ""
    if recovery_path is not None:
        if args.initialization_mode != "resume":
            raise ProbeAuditError("recovery segment must use --resume")
        inspection = read_json(recovery_path)
        inspected_checkpoint = inspection.get("checkpoint")
        if (
            inspection.get("schema") != SCHEMA
            or inspection.get("kind") != "live_checkpoint_inspection"
            or inspection.get("phase") != phase
            or int(inspection.get("expected_target", -1)) != target
            or not isinstance(inspected_checkpoint, Mapping)
            or source_record.get("sha256") != inspected_checkpoint.get("sha256")
            or source_record.get("size_bytes") != inspected_checkpoint.get("size_bytes")
        ):
            raise ProbeAuditError(
                "recovery source does not match the audited live checkpoint"
            )
        ancestry = "audited_live_recovery"
    elif previous_path is not None:
        if args.initialization_mode != "resume":
            raise ProbeAuditError("post-milestone segment must use --resume")
        assert previous is not None
        previous_checkpoint = previous.get("checkpoint")
        if (
            previous.get("schema") != SCHEMA
            or previous.get("kind") != "milestone_checkpoint"
            or previous.get("phase") != phase
            or not isinstance(previous_checkpoint, Mapping)
            or not _same_file_identity(source_record, previous_checkpoint)
        ):
            raise ProbeAuditError(
                "segment source is not the expected previous milestone checkpoint"
            )
        ancestry = "previous_milestone"
    else:
        initial = preflight.get("initial_checkpoint")
        if (
            target != MILESTONES_BY_PHASE[phase][0]
            or args.initialization_mode != "pretrain"
            or not isinstance(initial, Mapping)
            or not _same_file_identity(source_record, initial)
        ):
            raise ProbeAuditError(
                "first segment must pretrain from the exact phase initial checkpoint"
            )
        ancestry = "phase_initial"
    payload = {
        "schema": SCHEMA,
        "kind": "segment_lineage",
        "phase": phase,
        "expected_target": target,
        "initialization_mode": args.initialization_mode,
        "ancestry": ancestry,
        "source_checkpoint": source_record,
        "preflight": file_record(preflight_path),
        "previous_audit": file_record(previous_path) if previous_path else None,
        "recovery_inspection": file_record(recovery_path) if recovery_path else None,
    }
    output = resolve_path(args.output)
    if output.exists():
        raise ProbeAuditError(f"refusing to overwrite segment lineage: {output}")
    write_json(output, payload)
    print(f"[OK] recorded {phase} segment ancestry for target {target}: {output}")


def _cmd_inspect(args: argparse.Namespace) -> None:
    preflight_path = resolve_path(args.preflight)
    preflight = read_json(preflight_path)
    if preflight.get("phase") != args.phase:
        raise ProbeAuditError("checkpoint phase does not match preflight")
    lineage_path = resolve_path(args.segment_lineage)
    lineage = read_json(lineage_path)
    source_record = lineage.get("source_checkpoint")
    if (
        lineage.get("schema") != SCHEMA
        or lineage.get("kind") != "segment_lineage"
        or lineage.get("phase") != args.phase
        or int(lineage.get("expected_target", -1)) != int(args.expected_target)
        or not isinstance(source_record, Mapping)
    ):
        raise ProbeAuditError("live checkpoint has no matching current-segment lineage")
    source = resolve_path(str(source_record.get("path", "")))
    if not _same_file_identity(file_record(source), source_record):
        raise ProbeAuditError("current-segment source checkpoint drifted")
    previous_path = (
        resolve_path(args.previous_audit) if args.previous_audit else None
    )
    _validate_segment_lineage_chain(
        lineage_path=lineage_path,
        phase=args.phase,
        target=int(args.expected_target),
        preflight_path=preflight_path,
        previous_path=previous_path,
        source=source,
    )
    result = _validate_training_checkpoint_common(
        phase=args.phase,
        checkpoint=resolve_path(args.checkpoint),
        preflight=preflight,
        expected_target=int(args.expected_target),
        source_checkpoint=source,
        require_exact_iteration=False,
    )
    previous = _validate_previous_audit(
        args.phase,
        previous_path,
        int(args.expected_target),
        preflight_path=preflight_path,
    )
    _validate_branch_isolation(
        phase=args.phase,
        record=result["record"],
        initial=preflight["initial_checkpoint"],
        previous=previous,
    )
    payload = {
        "schema": SCHEMA,
        "kind": "live_checkpoint_inspection",
        "phase": args.phase,
        "expected_target": int(args.expected_target),
        "iteration": result["iteration"],
        "initialization_mode": result["initialization_mode"],
        "segment_lineage": file_record(lineage_path),
        "checkpoint": result["record"],
    }
    if args.output:
        write_json(resolve_path(args.output), payload)
    if args.print_iteration:
        print(result["iteration"])
    else:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))


def _cmd_metadata(args: argparse.Namespace) -> None:
    checkpoint = resolve_path(args.checkpoint)
    payload = load_checkpoint(checkpoint)
    args_dict = checkpoint_args(payload)
    pretrain = args_dict.get("pretrain_model_path")
    resume = args_dict.get("resume")
    lineage_sources = [value for value in (pretrain, resume) if value not in (None, "")]
    metadata = {
        "iteration": int(payload.get("iteration", 0) or 0),
        "epoch": int(payload.get("epoch", -1)),
        "epoch_finished": payload.get("epoch_finished"),
        "checkpoint_reason": payload.get("checkpoint_reason"),
        "max_train_iters": args_dict.get("max_train_iters"),
        "train_mode": args_dict.get("stage_b_gdino_adapter_train_mode"),
        "source_checkpoint": (
            str(resolve_path(str(lineage_sources[0])))
            if len(lineage_sources) == 1
            else None
        ),
    }
    del payload
    if args.field:
        value = metadata[args.field]
        if value is None:
            raise ProbeAuditError(f"checkpoint metadata field is null: {args.field}")
        print(str(value).lower() if isinstance(value, bool) else value)
    else:
        print(json.dumps(metadata, sort_keys=True, ensure_ascii=True))


def _validate_segment_lineage_chain(
    *,
    lineage_path: Path,
    phase: str,
    target: int,
    preflight_path: Path,
    previous_path: Path | None,
    source: Path,
    milestone_replay_stack: set[Path] | None = None,
    lineage_stack: set[Path] | None = None,
) -> Dict[str, Any]:
    lineage_path = lineage_path.resolve()
    if lineage_stack is None:
        lineage_stack = set()
    if lineage_path in lineage_stack:
        raise ProbeAuditError(f"segment lineage cycle detected: {lineage_path}")
    lineage_stack.add(lineage_path)
    lineage = read_json(lineage_path)
    source_record = lineage.get("source_checkpoint")
    if (
        lineage.get("schema") != SCHEMA
        or lineage.get("kind") != "segment_lineage"
        or lineage.get("phase") != phase
        or int(lineage.get("expected_target", -1)) != target
        or not isinstance(source_record, Mapping)
        or not _same_file_identity(file_record(source), source_record)
        or lineage.get("preflight") != file_record(preflight_path)
        or lineage.get("previous_audit")
        != (file_record(previous_path) if previous_path else None)
    ):
        raise ProbeAuditError("segment lineage does not match milestone ancestry")
    ancestry = lineage.get("ancestry")
    mode = lineage.get("initialization_mode")
    recovery_record = lineage.get("recovery_inspection")
    if ancestry == "phase_initial":
        preflight = read_json(preflight_path)
        initial = preflight.get("initial_checkpoint")
        if (
            target != MILESTONES_BY_PHASE[phase][0]
            or mode != "pretrain"
            or previous_path is not None
            or recovery_record is not None
            or not isinstance(initial, Mapping)
            or not _same_file_identity(source_record, initial)
        ):
            raise ProbeAuditError("initial segment lineage is malformed")
    elif ancestry == "previous_milestone":
        if mode != "resume" or previous_path is None or recovery_record is not None:
            raise ProbeAuditError("previous-milestone segment lineage is malformed")
        previous = _validate_previous_audit(
            phase,
            previous_path,
            target,
            preflight_path=preflight_path,
            replay_stack=milestone_replay_stack,
        )
        assert previous is not None
        previous_checkpoint = previous.get("checkpoint")
        if not isinstance(previous_checkpoint, Mapping) or not _same_file_identity(
            source_record, previous_checkpoint
        ):
            raise ProbeAuditError("segment did not resume from the previous milestone")
    elif ancestry == "audited_live_recovery":
        if mode != "resume" or not isinstance(recovery_record, Mapping):
            raise ProbeAuditError("recovery segment lineage is malformed")
        recovery_path = resolve_path(str(recovery_record.get("path", "")))
        if file_record(recovery_path) != recovery_record:
            raise ProbeAuditError("recovery inspection sidecar drifted")
        inspection = read_json(recovery_path)
        inspected_checkpoint = inspection.get("checkpoint")
        prior_lineage_record = inspection.get("segment_lineage")
        if (
            inspection.get("schema") != SCHEMA
            or inspection.get("kind") != "live_checkpoint_inspection"
            or inspection.get("phase") != phase
            or int(inspection.get("expected_target", -1)) != target
            or not isinstance(inspected_checkpoint, Mapping)
            or source_record.get("sha256") != inspected_checkpoint.get("sha256")
            or source_record.get("size_bytes") != inspected_checkpoint.get("size_bytes")
            or not isinstance(prior_lineage_record, Mapping)
        ):
            raise ProbeAuditError("recovery inspection does not seal the recovery checkpoint")
        prior_lineage_path = resolve_path(str(prior_lineage_record.get("path", "")))
        if file_record(prior_lineage_path) != prior_lineage_record:
            raise ProbeAuditError("prior segment lineage drifted after live inspection")
        prior_lineage = read_json(prior_lineage_path)
        prior_source_record = prior_lineage.get("source_checkpoint")
        if not isinstance(prior_source_record, Mapping):
            raise ProbeAuditError("prior segment lineage has no source checkpoint")
        prior_source = resolve_path(str(prior_source_record.get("path", "")))
        _validate_segment_lineage_chain(
            lineage_path=prior_lineage_path,
            phase=phase,
            target=target,
            preflight_path=preflight_path,
            previous_path=previous_path,
            source=prior_source,
            milestone_replay_stack=milestone_replay_stack,
            lineage_stack=lineage_stack,
        )
        preflight = read_json(preflight_path)
        recovered = _validate_training_checkpoint_common(
            phase=phase,
            checkpoint=source,
            preflight=preflight,
            expected_target=target,
            source_checkpoint=prior_source,
            require_exact_iteration=False,
        )
        previous = _validate_previous_audit(
            phase,
            previous_path,
            target,
            preflight_path=preflight_path,
            replay_stack=milestone_replay_stack,
        )
        _validate_branch_isolation(
            phase=phase,
            record=recovered["record"],
            initial=preflight["initial_checkpoint"],
            previous=previous,
        )
    else:
        raise ProbeAuditError(f"unsupported segment ancestry {ancestry!r}")
    return lineage


def _milestone_payload(
    *,
    phase: str,
    checkpoint: Path,
    preflight_path: Path,
    iteration: int,
    source: Path,
    previous_path: Path | None,
    segment_lineage_path: Path,
    replay_stack: set[Path] | None = None,
) -> Dict[str, Any]:
    preflight = read_json(preflight_path)
    if preflight.get("phase") != phase:
        raise ProbeAuditError("milestone phase does not match preflight")
    result = _validate_training_checkpoint_common(
        phase=phase,
        checkpoint=checkpoint,
        preflight=preflight,
        expected_target=iteration,
        source_checkpoint=source,
        require_exact_iteration=True,
    )
    previous = _validate_previous_audit(
        phase,
        previous_path,
        iteration,
        preflight_path=preflight_path,
        replay_stack=replay_stack,
    )
    _validate_branch_isolation(
        phase=phase,
        record=result["record"],
        initial=preflight["initial_checkpoint"],
        previous=previous,
    )
    _validate_segment_lineage_chain(
        lineage_path=segment_lineage_path,
        phase=phase,
        target=iteration,
        preflight_path=preflight_path,
        previous_path=previous_path,
        source=source,
        milestone_replay_stack=replay_stack,
    )
    return {
        "schema": SCHEMA,
        "kind": "milestone_checkpoint",
        "phase": phase,
        "iteration": iteration,
        "global_batch": int(preflight["launch"]["global_batch"]),
        "initialization_mode": result["initialization_mode"],
        "source_checkpoint": file_record(source),
        "preflight": file_record(preflight_path),
        "previous_audit": file_record(previous_path) if previous_path else None,
        "segment_lineage": file_record(segment_lineage_path),
        "checkpoint": result["record"],
    }


def _cmd_milestone(args: argparse.Namespace) -> None:
    iteration = int(args.expected_iteration)
    preflight_path = resolve_path(args.preflight)
    source = resolve_path(args.source_checkpoint)
    previous_path = resolve_path(args.previous_audit) if args.previous_audit else None
    segment_lineage_path = resolve_path(args.segment_lineage)
    payload = _milestone_payload(
        phase=args.phase,
        checkpoint=resolve_path(args.checkpoint),
        preflight_path=preflight_path,
        iteration=iteration,
        source=source,
        previous_path=previous_path,
        segment_lineage_path=segment_lineage_path,
    )
    output = resolve_path(args.output)
    if output.exists() and not args.verify_only:
        raise ProbeAuditError(f"refusing to overwrite milestone audit: {output}")
    if args.verify_only:
        existing = read_json(output)
        if existing != payload:
            raise ProbeAuditError(f"milestone audit drifted: {output}")
        print(f"[OK] unchanged {args.phase} milestone {iteration}: {output}")
        return
    write_json(output, payload)
    print(f"[OK] audited {args.phase} milestone {iteration}: {output}")


def _replay_milestone_audit(
    audit_path: Path,
    *,
    expected_phase: str | None = None,
    expected_iteration: int | None = None,
    expected_preflight_path: Path | None = None,
    expected_checkpoint_path: Path | None = None,
    replay_stack: set[Path] | None = None,
) -> Dict[str, Any]:
    audit_path = resolve_path(audit_path)
    if replay_stack is None:
        replay_stack = set()
    if audit_path in replay_stack:
        raise ProbeAuditError(f"previous milestone audit cycle detected: {audit_path}")
    replay_stack.add(audit_path)
    try:
        audit = read_json(audit_path)
        if audit.get("schema") != SCHEMA or audit.get("kind") != "milestone_checkpoint":
            raise ProbeAuditError(f"invalid previous milestone audit: {audit_path}")
        phase = str(audit.get("phase", ""))
        if phase not in PHASE_SPECS:
            raise ProbeAuditError(f"milestone audit has unsupported phase {phase!r}")
        iteration = int(audit.get("iteration", -1))
        _expected_previous_iteration(phase, iteration)
        if expected_phase is not None and phase != expected_phase:
            raise ProbeAuditError(
                f"previous milestone phase mismatch: expected {expected_phase}, got {phase}"
            )
        if expected_iteration is not None and iteration != expected_iteration:
            raise ProbeAuditError(
                "previous milestone is not adjacent: "
                f"expected {expected_iteration}, got {iteration}"
            )

        checkpoint_record_value = audit.get("checkpoint")
        preflight_record = audit.get("preflight")
        source_record = audit.get("source_checkpoint")
        segment_record = audit.get("segment_lineage")
        previous_record = audit.get("previous_audit")
        if not all(
            isinstance(value, Mapping)
            for value in (
                checkpoint_record_value,
                preflight_record,
                source_record,
                segment_record,
            )
        ):
            raise ProbeAuditError("milestone audit is missing checkpoint/preflight/source lineage")

        checkpoint_path = resolve_path(str(checkpoint_record_value.get("path", "")))
        preflight_path = resolve_path(str(preflight_record.get("path", "")))
        source_path = resolve_path(str(source_record.get("path", "")))
        segment_path = resolve_path(str(segment_record.get("path", "")))
        if not _same_file_identity(file_record(checkpoint_path), checkpoint_record_value):
            raise ProbeAuditError("milestone checkpoint file drifted from its audit")
        for label, path, record in (
            ("preflight", preflight_path, preflight_record),
            ("source checkpoint", source_path, source_record),
            ("segment lineage", segment_path, segment_record),
        ):
            if file_record(path) != record:
                raise ProbeAuditError(f"milestone {label} file drifted from its audit")
        if expected_checkpoint_path is not None and checkpoint_path != resolve_path(
            expected_checkpoint_path
        ):
            raise ProbeAuditError("evaluation checkpoint path does not match milestone audit")
        if expected_preflight_path is not None and preflight_path != resolve_path(
            expected_preflight_path
        ):
            raise ProbeAuditError("milestone chain switched phase preflight")

        expected_previous = _expected_previous_iteration(phase, iteration)
        if expected_previous is None:
            if previous_record is not None:
                raise ProbeAuditError("first milestone must not have a previous audit")
            previous_path = None
        else:
            if not isinstance(previous_record, Mapping):
                raise ProbeAuditError(
                    f"milestone {iteration} is disconnected from previous {expected_previous}"
                )
            previous_path = resolve_path(str(previous_record.get("path", "")))
            if file_record(previous_path) != previous_record:
                raise ProbeAuditError("previous milestone sidecar drifted")

        recomputed = _milestone_payload(
            phase=phase,
            checkpoint=checkpoint_path,
            preflight_path=preflight_path,
            iteration=iteration,
            source=source_path,
            previous_path=previous_path,
            segment_lineage_path=segment_path,
            replay_stack=replay_stack,
        )
        if recomputed != audit:
            raise ProbeAuditError(f"milestone audit payload drifted: {audit_path}")
        return recomputed
    finally:
        replay_stack.remove(audit_path)


def _verify_milestone_checkpoint(
    checkpoint: Path,
    audit_path: Path,
) -> Dict[str, Any]:
    checkpoint = resolve_path(checkpoint)
    audit_path = resolve_path(audit_path)
    audit = _replay_milestone_audit(
        audit_path,
        expected_checkpoint_path=checkpoint,
    )
    phase = str(audit["phase"])
    preflight_record = audit.get("preflight")
    assert isinstance(preflight_record, Mapping)
    preflight_path = resolve_path(str(preflight_record.get("path", "")))
    current_preflight = _verify_current_preflight(preflight_path)
    return {
        "schema": SCHEMA,
        "kind": "evaluation_checkpoint_verified",
        "phase": phase,
        "iteration": int(audit["iteration"]),
        "checkpoint": audit["checkpoint"],
        "config": current_preflight["static"]["config"],
        "datasets": current_preflight["static"]["datasets"],
        "train_mode": PHASE_SPECS[phase]["mode"],
        "tn_scope": PHASE_SPECS[phase]["scope"],
        "audit": file_record(audit_path),
    }


def _cmd_verify_evaluation(args: argparse.Namespace) -> None:
    checkpoint = resolve_path(args.checkpoint)
    result = _verify_milestone_checkpoint(
        checkpoint,
        resolve_path(args.audit),
    )
    if args.output:
        write_json(resolve_path(args.output), result)
    print(f"[OK] verified {result['phase']} evaluation checkpoint: {checkpoint}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    static = subparsers.add_parser("static", help="Audit immutable configs and data")
    static.add_argument("--phase", choices=("rank", "confidence", "all"), default="all")
    static.add_argument("--output")
    static.set_defaults(func=_cmd_static)

    preflight = subparsers.add_parser("phase-preflight")
    preflight.add_argument("--phase", choices=("rank", "confidence"), required=True)
    preflight.add_argument("--initial-checkpoint", required=True)
    preflight.add_argument("--initial-audit")
    preflight.add_argument("--output-dir", required=True)
    preflight.add_argument("--world-size", type=int, required=True)
    preflight.add_argument("--per-gpu-batch", type=int, required=True)
    preflight.add_argument("--output", required=True)
    preflight.add_argument("--continue-run", action="store_true")
    preflight.set_defaults(func=_cmd_phase_preflight)

    segment = subparsers.add_parser("segment-lineage")
    segment.add_argument("--phase", choices=("rank", "confidence"), required=True)
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

    inspect = subparsers.add_parser("inspect", help="Validate a resumable live checkpoint")
    inspect.add_argument("--phase", choices=("rank", "confidence"), required=True)
    inspect.add_argument("--checkpoint", required=True)
    inspect.add_argument("--preflight", required=True)
    inspect.add_argument("--expected-target", type=int, required=True)
    inspect.add_argument("--segment-lineage", required=True)
    inspect.add_argument("--previous-audit")
    inspect.add_argument("--output")
    inspect.add_argument("--print-iteration", action="store_true")
    inspect.set_defaults(func=_cmd_inspect)

    metadata = subparsers.add_parser("metadata", help="Read lightweight checkpoint metadata")
    metadata.add_argument("--checkpoint", required=True)
    metadata.add_argument(
        "--field",
        choices=(
            "iteration",
            "epoch",
            "epoch_finished",
            "checkpoint_reason",
            "max_train_iters",
            "train_mode",
            "source_checkpoint",
        ),
    )
    metadata.set_defaults(func=_cmd_metadata)

    milestone = subparsers.add_parser("milestone", help="Audit an exact phase milestone")
    milestone.add_argument("--phase", choices=("rank", "confidence"), required=True)
    milestone.add_argument("--checkpoint", required=True)
    milestone.add_argument("--preflight", required=True)
    milestone.add_argument("--expected-iteration", type=int, required=True)
    milestone.add_argument("--source-checkpoint", required=True)
    milestone.add_argument("--previous-audit")
    milestone.add_argument("--segment-lineage", required=True)
    milestone.add_argument("--output", required=True)
    milestone.add_argument("--verify-only", action="store_true")
    milestone.set_defaults(func=_cmd_milestone)

    verify_evaluation = subparsers.add_parser(
        "verify-evaluation", help="Replay a milestone audit before formal evaluation"
    )
    verify_evaluation.add_argument("--checkpoint", required=True)
    verify_evaluation.add_argument("--audit", required=True)
    verify_evaluation.add_argument("--output")
    verify_evaluation.set_defaults(func=_cmd_verify_evaluation)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        args.func(args)
    except ProbeAuditError as error:
        raise SystemExit(f"[FAIL] {error}") from error


if __name__ == "__main__":
    main()
