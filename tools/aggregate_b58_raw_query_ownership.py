#!/usr/bin/env python3
"""Aggregate B58 raw-query owners and the matched parent-to-B58 causal axis."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import tools.aggregate_mmgdino_pretrain_ownership as mature
from tools.b58_raw_query_ownership import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    B58_CAPACITY_REFERENCE,
    E5_REFERENCE,
    EXPERIMENT_ROOT,
    FORMAL_SEEDS,
    OWNERS,
    PARENT_AGGREGATE,
    PARENT_EVALUATION_ROOT,
    REC_NONINFERIORITY_MARGIN,
    REF_INPUTS,
    TEST5_SURFACES,
    TESTAB_SURFACES,
    TRUNK_ID,
)


SCHEMA = "arrow.b58_raw_query_ownership.aggregate/v1"
SHARED, ISOLATED = OWNERS
PARENT_TRUNK_ID = "original_parent"


class AggregateError(mature.AggregateError):
    pass


@contextlib.contextmanager
def _context():
    replacements = {
        "BOOTSTRAP_REPLICATES": BOOTSTRAP_REPLICATES,
        "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
        "B58_REFERENCE": B58_CAPACITY_REFERENCE,
        "E5_REFERENCE": E5_REFERENCE,
        "FORMAL_SEEDS": FORMAL_SEEDS,
        "OWNERS": OWNERS,
        "SHARED": SHARED,
        "ISOLATED": ISOLATED,
        "REC_NONINFERIORITY_MARGIN": REC_NONINFERIORITY_MARGIN,
        "REF_INPUTS": REF_INPUTS,
        "TEST5_SURFACES": TEST5_SURFACES,
        "TESTAB_SURFACES": TESTAB_SURFACES,
        "TRUNK_ID": TRUNK_ID,
        "SCHEMA": SCHEMA,
    }
    previous = {name: getattr(mature, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(mature, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(mature, name, value)


@contextlib.contextmanager
def _record_trunk(trunk_id: str):
    previous = mature.TRUNK_ID
    try:
        mature.TRUNK_ID = trunk_id
        yield
    finally:
        mature.TRUNK_ID = previous


def _load_records(root: Path, trunk_id: str):
    with _record_trunk(trunk_id):
        images, refs = mature._load_ref(root)
        tn_images, tn = mature._load_tn(root)
    return images, refs, tn_images, tn


def _effect(
    multiplicity: np.ndarray,
    surfaces: Sequence[str],
    seed: int,
    counts: Mapping[str, np.ndarray],
    hits: Mapping[str, Mapping[tuple[str, int | None], np.ndarray]],
) -> float:
    isolated = mature._draw_ref_metric(
        multiplicity, surfaces, ISOLATED, seed, counts, hits
    )
    shared = mature._draw_ref_metric(
        multiplicity, surfaces, SHARED, seed, counts, hits
    )
    return isolated - shared


def _same_head_axis(
    *, evaluation_root: Path, b58_payload: Mapping[str, Any]
) -> dict[str, Any]:
    b_images, b_refs, b_tn_images, b_tn = _load_records(
        evaluation_root, TRUNK_ID
    )
    p_images, p_refs, p_tn_images, p_tn = _load_records(
        PARENT_EVALUATION_ROOT, PARENT_TRUNK_ID
    )
    if b_images != p_images:
        raise AggregateError("parent/B58 Ref image identities differ")
    if b_tn_images != p_tn_images:
        raise AggregateError("parent/B58 Strict image identities differ")
    b_image_ids, b_counts, b_hits = mature._image_accounting(b_images, b_refs)
    p_image_ids, p_counts, p_hits = mature._image_accounting(p_images, p_refs)
    if b_image_ids != p_image_ids:
        raise AggregateError("parent/B58 global Ref clusters differ")
    tn_groups: dict[str, list[int]] = {}
    for index, image in enumerate(b_tn_images):
        tn_groups.setdefault(image, []).append(index)
    ordered_tn_groups = [
        np.asarray(tn_groups[key], dtype=np.int64) for key in sorted(tn_groups)
    ]
    draws = {"test5": [], "testab": [], "fpr95_reduction": []}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for _ in range(BOOTSTRAP_REPLICATES):
        multiplicity = np.bincount(
            rng.integers(0, len(b_image_ids), size=len(b_image_ids)),
            minlength=len(b_image_ids),
        ).astype(np.float64)
        sampled_groups = rng.integers(
            0, len(ordered_tn_groups), size=len(ordered_tn_groups)
        )
        tn_indices = np.concatenate(
            [ordered_tn_groups[index] for index in sampled_groups]
        )
        seed_test5, seed_testab, seed_fpr = [], [], []
        for seed in FORMAL_SEEDS:
            b_test5 = _effect(
                multiplicity, TEST5_SURFACES, seed, b_counts, b_hits
            )
            p_test5 = _effect(
                multiplicity, TEST5_SURFACES, seed, p_counts, p_hits
            )
            b_testab = _effect(
                multiplicity, TESTAB_SURFACES, seed, b_counts, b_hits
            )
            p_testab = _effect(
                multiplicity, TESTAB_SURFACES, seed, p_counts, p_hits
            )
            seed_test5.append(b_test5 - p_test5)
            seed_testab.append(b_testab - p_testab)
            b_sp, b_sn = b_tn[(SHARED, seed)]
            b_ip, b_in = b_tn[(ISOLATED, seed)]
            p_sp, p_sn = p_tn[(SHARED, seed)]
            p_ip, p_in = p_tn[(ISOLATED, seed)]
            b_reduction = mature._exact_fpr(
                b_sp[tn_indices], b_sn[tn_indices]
            ) - mature._exact_fpr(b_ip[tn_indices], b_in[tn_indices])
            p_reduction = mature._exact_fpr(
                p_sp[tn_indices], p_sn[tn_indices]
            ) - mature._exact_fpr(p_ip[tn_indices], p_in[tn_indices])
            seed_fpr.append(b_reduction - p_reduction)
        draws["test5"].append(float(np.mean(seed_test5)))
        draws["testab"].append(float(np.mean(seed_testab)))
        draws["fpr95_reduction"].append(float(np.mean(seed_fpr)))

    parent = json.loads(PARENT_AGGREGATE.read_text(encoding="utf-8"))
    p_contrast = parent["isolated_minus_shared_wide"]
    b_contrast = b58_payload["isolated_minus_shared_wide"]

    def comparison(key: str, draw_key: str) -> dict[str, Any]:
        b_value = float(b_contrast[key])
        p_value = float(p_contrast[key])
        return {
            "parent_effect": p_value,
            "b58_effect": b_value,
            "b58_minus_parent_difference_in_differences": b_value - p_value,
            "ci95": mature._ci(draws[draw_key]),
        }

    p_native = parent["point_metrics"]["native"]
    b_native = b58_payload["point_metrics"]["native"]
    return {
        "matched_contract": {
            "same_owner_architecture_and_initialization": True,
            "same_100k_raw_query_heads": True,
            "same_two_task_specific_adam_states": True,
            "same_zero_weight_decay": True,
            "same_u150_schedules_losses_seeds_and_data_order": True,
            "same_full_expression_mean_native_score": True,
            "same_test5_strict2031_evaluator": True,
            "same_effective_938_tensor_architecture": True,
            "trunk_weight_change_only": "727 changed and 211 unchanged tensors",
        },
        "native_parent_to_b58": {
            "test5_gain": (
                float(b_native["test5_micro_p1"])
                - float(p_native["test5_micro_p1"])
            ),
            "testab_gain": (
                float(b_native["testab_micro_p1"])
                - float(p_native["testab_micro_p1"])
            ),
            "fpr95_reduction": (
                float(p_native["strict2031_fpr95"])
                - float(b_native["strict2031_fpr95"])
            ),
        },
        "ownership_effect_difference_in_differences": {
            "test5": comparison("test5_gain", "test5"),
            "testab": comparison("testab_gain", "testab"),
            "fpr95_reduction": comparison(
                "fpr95_gain", "fpr95_reduction"
            ),
        },
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "same_image_draw_across_parent_b58_owners_and_seeds": True,
            "fpr95_recomputes_each_trunk_owner_seed_positive_q05": True,
        },
    }


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def aggregate(
    *, evaluation_root: Path, formal_root: Path, output: Path
) -> dict[str, Any]:
    if output.exists():
        raise AggregateError("aggregate output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".b58_base_aggregate.", suffix=".json", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        with _context():
            payload = mature.aggregate(
                evaluation_root=evaluation_root,
                formal_root=formal_root,
                output=temporary,
            )
        payload["schema"] = SCHEMA
        payload["same_head_parent_to_b58"] = _same_head_axis(
            evaluation_root=evaluation_root, b58_payload=payload
        )
        _atomic_json(payload, output)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def main() -> None:
    value = aggregate(
        evaluation_root=EXPERIMENT_ROOT / "evaluation",
        formal_root=EXPERIMENT_ROOT / "formal",
        output=EXPERIMENT_ROOT / "aggregate.json",
    )
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["AggregateError", "SCHEMA", "_context", "aggregate"]
