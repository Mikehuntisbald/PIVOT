#!/usr/bin/env python3
"""Seal the two-trunk MM-GDINO ownership 2x2 before GPU forwards."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from tools.mmgdino_e5_ownership import MMGDinoE5ResponsibilityOwners
from tools.mmgdino_e6_ownership_2x2 import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    E5_REFERENCE,
    EVAL_CONFIG,
    EVAL_CONFIG_SHA256,
    EXPERIMENT_ROOT,
    FORMAL_SEEDS,
    IMAGE_ROOT,
    MMDET_COMMIT,
    MMDET_ROOT,
    OWNERS,
    PREREGISTRATION,
    REC_NONINFERIORITY_MARGIN,
    REF_INPUTS,
    ROOT,
    SCHEDULE_RECEIPT,
    TRUNK_SPECS,
    eval_cache_path,
    owner_output_dir,
    training_cache_path,
)
from tools.responsibility_isolation_cache import file_sha256


SCHEMA = "arrow.mmgdino_e6_ownership_2x2.preregistration/v1"


class PreregistrationError(RuntimeError):
    pass


def _record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    result = {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }
    if rows is not None:
        with resolved.open("r", encoding="utf-8") as handle:
            actual = sum(1 for _ in handle)
        if actual != rows:
            raise PreregistrationError(
                f"row count drift for {resolved}: expected {rows}, got {actual}"
            )
        result["rows"] = actual
    return result


def _atomic_json(value: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _git() -> dict[str, str]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    if status:
        raise PreregistrationError(
            "preregistration requires a clean committed worktree"
        )
    return {"commit": head, "status": "clean"}


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
        text=True, capture_output=True,
    ).stdout.strip()


def _state_signature(state: Mapping[str, torch.Tensor]) -> str:
    rows = [
        [name, str(state[name].dtype), list(state[name].shape)]
        for name in sorted(state)
    ]
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _checkpoint_audit() -> tuple[dict[str, Any], str]:
    audits = {}
    common_signature = None
    for trunk_id, spec in TRUNK_SPECS.items():
        if file_sha256(spec.checkpoint) != spec.checkpoint_sha256:
            raise PreregistrationError(f"{trunk_id} checkpoint SHA drifted")
        try:
            checkpoint = torch.load(
                spec.checkpoint, map_location="cpu", mmap=True,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(spec.checkpoint, map_location="cpu")
        meta = checkpoint.get("meta", {})
        if (
            meta.get("epoch") != spec.expected_epoch
            or meta.get("iter") != spec.expected_iter
            or meta.get("experiment_name") != spec.expected_experiment_name
        ):
            raise PreregistrationError(f"{trunk_id} checkpoint metadata drifted")
        state = checkpoint.get("state_dict")
        if not isinstance(state, Mapping) or len(state) != 908:
            raise PreregistrationError(f"{trunk_id} state-dict schema drifted")
        signature = _state_signature(state)
        if common_signature is None:
            common_signature = signature
        elif signature != common_signature:
            raise PreregistrationError("e6 trunks do not share tensor schema")
        if any(
            not bool(torch.isfinite(tensor).all().item())
            for tensor in state.values()
        ):
            raise PreregistrationError(f"{trunk_id} contains non-finite tensors")
        audits[trunk_id] = {
            **spec.as_dict(),
            "checkpoint": _record(spec.checkpoint),
            "meta": {
                "epoch": meta.get("epoch"), "iter": meta.get("iter"),
                "seed": meta.get("seed"),
                "experiment_name": meta.get("experiment_name"),
                "time": meta.get("time"),
            },
            "state_dict_tensor_count": len(state),
            "state_dict_parameter_count": sum(
                int(tensor.numel()) for tensor in state.values()
            ),
            "state_schema_sha256": signature,
            "all_tensors_finite": True,
        }
        del checkpoint, state
    assert common_signature is not None
    return audits, common_signature


def build(output: Path = PREREGISTRATION) -> dict[str, Any]:
    if output.exists():
        raise PreregistrationError("preregistration output already exists")
    if EXPERIMENT_ROOT.exists():
        raise PreregistrationError("experiment output root must not exist")
    if _git_head(MMDET_ROOT) != MMDET_COMMIT:
        raise PreregistrationError("MMDetection commit drifted")
    if file_sha256(EVAL_CONFIG) != EVAL_CONFIG_SHA256:
        raise PreregistrationError("MM-GDINO eval config drifted")
    if not IMAGE_ROOT.is_dir():
        raise PreregistrationError("bound COCO image root is absent")

    schedules = json.loads(SCHEDULE_RECEIPT.read_text(encoding="utf-8"))
    if schedules.get("status") != "complete_before_owner_training":
        raise PreregistrationError("e5 schedule receipt status drifted")
    schedule_records = {}
    for seed in FORMAL_SEEDS:
        source = schedules["outputs"][str(seed)]
        schedule_records[str(seed)] = {
            name: _record(Path(source[name]["path"]))
            for name in ("rank", "d3", "schedule")
        }
        for name in ("rank", "d3", "schedule"):
            if schedule_records[str(seed)][name]["sha256"] != source[name]["sha256"]:
                raise PreregistrationError(f"seed{seed} {name} schedule drifted")

    checkpoints, state_schema = _checkpoint_audit()
    formal_targets = [
        owner_output_dir(trunk, owner, seed)
        for trunk in TRUNK_SPECS for owner in OWNERS for seed in FORMAL_SEEDS
    ]
    cache_targets = [
        training_cache_path(trunk, seed)
        for trunk in TRUNK_SPECS for seed in FORMAL_SEEDS
    ] + [
        eval_cache_path(trunk, surface)
        for trunk in TRUNK_SPECS for surface in REF_INPUTS
    ]
    existing = [str(path) for path in (*formal_targets, *cache_targets) if path.exists()]
    if existing:
        raise PreregistrationError(f"formal outputs exist before lock: {existing}")

    architecture = {
        owner: MMGDinoE5ResponsibilityOwners(
            ownership=owner
        ).architecture_report().as_dict()
        for owner in OWNERS
    }
    code_paths = {
        "contract": ROOT / "tools/mmgdino_e6_ownership_2x2.py",
        "preregistration_builder": Path(__file__),
        "runner": ROOT / "tools/run_mmgdino_e6_ownership_2x2.py",
        "aggregator": ROOT / "tools/aggregate_mmgdino_e6_ownership_2x2.py",
        "owners": ROOT / "tools/mmgdino_e5_ownership.py",
        "trainer": ROOT / "tools/train_mmgdino_e5_ownership.py",
        "training_extractor": ROOT / "tools/extract_mmgdino_responsibility_cache.py",
        "evaluation_extractor": ROOT / "tools/extract_mmgdino_e5_eval_cache.py",
        "cache_evaluator": ROOT / "tools/eval_mmgdino_e5_ownership_cache.py",
    }
    payload = {
        "schema": SCHEMA,
        "status": "locked_before_any_e6_owner_gpu_forward",
        "locked_at": dt.datetime.now(
            dt.timezone(dt.timedelta(hours=8))
        ).isoformat(),
        "git": _git(),
        "research_question": (
            "Does TN-aware trunk adaptation turn near-orthogonal shared "
            "ranking/abstention gradients into a negative-tail regime where "
            "isolation protects REC?"
        ),
        "trunks": checkpoints,
        "shared_state_schema_sha256": state_schema,
        "matrix": {
            "trunks": list(TRUNK_SPECS),
            "owners": list(OWNERS),
            "seeds": list(FORMAL_SEEDS),
            "formal_trajectories": len(TRUNK_SPECS) * len(OWNERS) * len(FORMAL_SEEDS),
            "architecture": architecture,
            "excluded_arm": "shared_128",
        },
        "training": {
            "updates": 150,
            "rank_updates": 100,
            "confidence_updates": 50,
            "interleave": "rank,confidence,rank",
            "rank_batch_size": 32,
            "confidence_batch_size": 8,
            "rank_learning_rate": 3e-5,
            "confidence_learning_rate": 1e-4,
            "two_task_specific_adamw_states": True,
            "weight_decay": 0.0,
            "clip_norm": 0.1,
            "precision": "fp32 deterministic",
            "milestones_audit_only": [25, 50, 100, 150],
            "selected_update": 150,
        },
        "schedule_contract": {
            "receipt": _record(SCHEDULE_RECEIPT),
            "per_seed": schedule_records,
            "same_identity_and_batch_order_across_trunks_and_owners": True,
            "checkpoint_or_metric_dependent_selection": False,
        },
        "evaluation": {
            surface: {
                "mode": spec["mode"],
                "input": _record(spec["path"], rows=spec["rows"]),
            }
            for surface, spec in REF_INPUTS.items()
        },
        "statistics": {
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "cluster": "image_id; TestA and TestB sampled as separate strata",
            "same_draw_across_both_trunks_both_owners_and_three_seeds": True,
            "fpr95_recomputes_each_model_seed_positive_q05_per_replicate": True,
            "rec_noninferiority_margin": REC_NONINFERIORITY_MARGIN,
            "planned_within_trunk_contrasts": [
                "e6_posctrl:isolated_128-shared_wide",
                "e6_tn10:isolated_128-shared_wide",
            ],
            "planned_cross_trunk_contrast": (
                "(isolated_128-shared_wide)_e6_tn10 minus "
                "(isolated_128-shared_wide)_e6_posctrl"
            ),
            "holm_family_size": 2,
            "gradient_tail": {
                "primary_description": "pooled fixed milestone probes",
                "metrics": ["P(cosine<0)", "q05", "minimum", "mean"],
                "status": "mechanism_descriptive_not_iid_expression_inference",
            },
        },
        "e5_reference": {
            "result": _record(E5_REFERENCE),
            "reuse_only_no_new_forward": True,
        },
        "runtime_assets": {
            "mmdet_root": str(MMDET_ROOT.resolve(strict=True)),
            "mmdet_commit": MMDET_COMMIT,
            "eval_config": _record(EVAL_CONFIG),
            "image_root": str(IMAGE_ROOT.resolve(strict=True)),
        },
        "code": {name: _record(path) for name, path in code_paths.items()},
        "formal_output_targets_absent": [str(path) for path in formal_targets],
        "cache_output_targets_absent": [str(path) for path in cache_targets],
        "prohibitions": [
            "do not run Shared-128",
            "do not change seeds, sample identities, batch order, update count, losses, learning rates, optimizer states, or weight decay by trunk or owner",
            "do not select a milestone, threshold, margin, or checkpoint from e6 results",
            "do not stop after one seed or one trunk",
            "do not interpret milestone gradient probes as expression-IID samples",
        ],
    }
    _atomic_json(payload, output)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PREREGISTRATION)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(json.dumps(build(args.output.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
