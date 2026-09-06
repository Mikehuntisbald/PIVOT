#!/usr/bin/env python3
"""Two-process scheduling of the unchanged, separable per-localizer analysis.

The frozen formal input loader validates the full two-localizer population.
Each worker calls the frozen analyze_readout with all three seeds and the same
image draws. Only localizer blocks are merged; every other field must agree
exactly. No cross-localizer contrast exists in the frozen study. This wrapper
is not an alternative metric implementation and cannot resume partial draws.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tools.analyze_confidence_readout import ANALYSIS_DEPENDENCIES, LOCALIZERS, _digest, load_manifest
from tools.confidence_readout_metrics import (
    CELLS, SCHEMA, SEEDS, _validate, _with_cross_readouts, analyze_readout,
)
from tools.grounding_confidence_ordering import _draw_cluster_weights

ITERATIONS, BOOTSTRAP_SEED = 5000, 20260911


def code_hashes():
    return {name: _digest(ROOT / "tools" / name) for name in ANALYSIS_DEPENDENCIES}


def full_contract(runs, iterations, seed):
    """Cross-localizer validation happens BEFORE dispatch or reuse acceptance."""
    expanded, cross = _with_cross_readouts(runs)
    aligned, lookup, strata, arms = _validate(expanded, SEEDS)
    rng, digest = np.random.Generator(np.random.PCG64(seed)), hashlib.sha256()
    for _ in range(iterations):
        draw = _draw_cluster_weights(rng, strata, len(lookup))
        digest.update(draw.astype("<u4").tobytes())
    populations = {}
    for localizer, seeds in aligned.items():
        first = next(iter(seeds.values()))
        populations[localizer] = {
            "records": len(first), "images": len(lookup),
            "C": sum(r["kind"] == "positive" and r["correct"] for r in first),
            "W": sum(r["kind"] == "positive" and not r["correct"] for r in first),
            "N": sum(r["kind"] != "positive" for r in first),
        }
    return {"localizers": tuple(sorted(aligned)), "arms": set(arms), "populations": populations,
            "cross_readouts": cross, "strata": {k: len(v) for k, v in strata.items()},
            "draws_sha256": digest.hexdigest(), "iterations": iterations, "seed": seed}


def validate_block(result, localizer, contract):
    if (result.get("schema") != SCHEMA or result.get("matched_cells") != list(CELLS)
            or result.get("primary_metric") != "mixed_augrc"
            or set(result.get("localizers", {})) != {localizer}
            or result.get("cross_readout_scores") != contract["cross_readouts"]):
        raise ValueError("localizer block schema/matched/diagnostic scope mismatch")
    expected = {
        "iterations": contract["iterations"], "seed": contract["seed"], "rng": "PCG64",
        "unit": "image_cluster", "required_seeds": list(SEEDS), "strata": contract["strata"],
        "draws_sha256": contract["draws_sha256"], "same_draw_all_localizers_heads_seeds": True,
        "q05_recomputed_each_draw": True, "fixed_threshold_fit": False,
        "observed_mixture_varies_with_image_draw": True,
        "fixed_priors_class_renormalized_each_draw": True,
    }
    if any(result.get("bootstrap", {}).get(k) != v for k, v in expected.items()):
        raise ValueError("localizer block does not use the full-input shared image draws")
    block = result["localizers"][localizer]
    if (block.get("population") != contract["populations"][localizer]
            or set(block.get("per_seed", {})) != set(SEEDS)
            or set(block.get("summary", {})) != contract["arms"]
            or block.get("conditional_counts") is None
            or set(block.get("winner_geometry", {})) != set(SEEDS)):
        raise ValueError("localizer block population/seed/score/conditional scope mismatch")
    for arm in contract["arms"]:
        value = block["summary"][arm]["mixed_augrc"]
        interval = value.get("ci95")
        if (not isinstance(value.get("mean"), (int, float)) or not np.isfinite(value["mean"])
                or not isinstance(interval, list) or len(interval) != 2
                or not np.isfinite(interval).all() or interval[0] > interval[1]
                or value.get("undefined_replicates") != 0):
            raise ValueError("completed block is missing a primary score estimate/interval")


def validate_reuse(result, source, bindings, contract, current_code):
    """Only a completed FineCops MM stage with byte-identical inputs is reusable."""
    localizer = "mmgdino_positive"
    if source.get("surface") != "finecops_val":
        raise ValueError("the MM stage is reusable only for FineCops validation")
    receipt = result.get("receipt", {})
    if (receipt.get("stage_mm_only") is not True
            or receipt.get("formal_requested_configuration") is not False
            or receipt.get("study_final_receipt") is not False
            or receipt.get("surface") != source["surface"]
            or receipt.get("protocol_sha256") != source["protocol_sha256"]
            or receipt.get("records") != {localizer: bindings[localizer]}
            or receipt.get("code_sha256") != current_code
            or receipt.get("model_forward") is not False
            or receipt.get("checkpoint_selection") is not False
            or receipt.get("threshold_fitting") is not False):
        raise ValueError("MM stage is not a completed same-protocol, same-record/stat/code replay")
    validate_block(result, localizer, contract)
    return {k: v for k, v in result.items() if k != "receipt"}


def merge_blocks(parts, contract):
    if set(parts) != set(contract["localizers"]):
        raise ValueError("all expected localizer blocks are required")
    merged, top = {}, None
    for localizer in sorted(parts):
        result = parts[localizer]
        validate_block(result, localizer, contract)
        if "receipt" in result:
            raise ValueError("block receipts must be validated separately, never merged silently")
        other = {k: v for k, v in result.items() if k != "localizers"}
        if top is None:
            top = other
        elif top != other:
            raise ValueError("non-localizer result metadata differs; merging is invalid")
        merged[localizer] = result["localizers"][localizer]
    return {**top, "localizers": merged}


def _compute_one(localizer, runs, iterations, seed, emit_progress):
    def progress(done, total):
        if emit_progress and (done % 100 == 0 or done == total):
            print(json.dumps({"localizer": localizer, "bootstrap": done, "total": total}), flush=True)
    return analyze_readout({localizer: runs}, iterations=iterations, seed=seed,
                           required_seeds=SEEDS, conditionals=True, progress=progress)


def run_parts(runs, contract, *, reusable=None, workers=2, emit_progress=False):
    """Internal scheduling helper; caller validates reuse provenance first."""
    if workers not in (1, 2):
        raise ValueError("one or two CPU worker processes supported")
    parts = dict(reusable or {})
    if not set(parts).issubset(runs):
        raise ValueError("foreign reuse block")
    for localizer, block in parts.items():
        validate_block(block, localizer, contract)
    pending = [loc for loc in sorted(runs) if loc not in parts]
    if pending:
        # Spawn, not fork: do not inherit a parent's initialized BLAS runtime.
        with ProcessPoolExecutor(max_workers=min(workers, len(pending)),
                                 mp_context=multiprocessing.get_context("spawn")) as executor:
            jobs = {executor.submit(_compute_one, loc, runs[loc], contract["iterations"],
                                    contract["seed"], emit_progress): loc for loc in pending}
            for job in as_completed(jobs):
                parts[jobs[job]] = job.result()
    return merge_blocks(parts, contract)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--reuse-mm-stage", type=Path)
    parser.add_argument("--reuse-mm-stage-sha256")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("append-only full analysis output already exists")
    if (args.reuse_mm_stage is None) != (args.reuse_mm_stage_sha256 is None):
        raise ValueError("reuse requires both the sealed stage path and its expected SHA-256")
    input_digest, code = _digest(args.input), code_hashes()
    wrapper = {"path": str(Path(__file__).resolve()), "sha256": _digest(Path(__file__))}
    source, runs, bindings = load_manifest(args.input)  # Full two-localizer loader, never stage mode.
    if set(runs) != set(LOCALIZERS):
        raise ValueError("complete formal two-localizer input required")
    contract = full_contract(runs, ITERATIONS, BOOTSTRAP_SEED)
    reusable, reuse_binding = {}, None
    if args.reuse_mm_stage is not None:
        if _digest(args.reuse_mm_stage) != args.reuse_mm_stage_sha256:
            raise ValueError("sealed MM stage analysis SHA drift")
        original = json.loads(args.reuse_mm_stage.read_text())
        reusable["mmgdino_positive"] = validate_reuse(original, source, bindings, contract, code)
        reuse_binding = {"path": str(args.reuse_mm_stage.resolve()), "sha256": args.reuse_mm_stage_sha256,
                         "original_receipt": original["receipt"]}
    result = run_parts(runs, contract, reusable=reusable, workers=args.workers, emit_progress=True)
    if _digest(args.input) != input_digest or code_hashes() != code or _digest(Path(__file__)) != wrapper["sha256"]:
        raise ValueError("input or implementation changed during parallel evaluation")
    # Check all bound data/statistics files again, without any model or refitting.
    for localizer, seeds in source["runs"].items():
        for seed, record in seeds.items():
            for binding in (record, source["sirc_statistics"][localizer][seed]):
                path = Path(binding["path"])
                if not path.is_absolute():
                    path = args.input.parent / path
                if _digest(path) != binding["sha256"]:
                    raise ValueError("record or training-statistics file changed during evaluation")
    if reuse_binding is not None and _digest(args.reuse_mm_stage) != reuse_binding["sha256"]:
        raise ValueError("reused stage artifact changed during evaluation")
    result["receipt"] = {
        "protocol_sha256": source["protocol_sha256"], "input_sha256": input_digest,
        "records": bindings, "created_utc": datetime.now(timezone.utc).isoformat(),
        "surface": source["surface"], "stage_mm_only": False, "code_sha256": code,
        "formal_requested_configuration": True, "study_final_receipt": False,
        "model_forward": False, "checkpoint_selection": False, "threshold_fitting": False,
        "parallel_execution": {
            "schema": "arrow.confidence_readout.parallel_execution/v1", "wrapper": wrapper,
            "workers_requested": args.workers, "process_start_method": "spawn",
            "computed_localizers": sorted(set(runs) - set(reusable)),
            "reused_mm_stage": reuse_binding, "partial_bootstrap_resume": False,
            "full_population_validated_before_split": True,
            "all_other_result_metadata_exactly_equal": True,
            "shared_draws_sha256": contract["draws_sha256"],
            "python": platform.python_version(), "numpy": np.__version__,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
