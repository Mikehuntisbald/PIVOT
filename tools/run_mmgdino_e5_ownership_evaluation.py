#!/usr/bin/env python3
"""Execute the frozen RefCOCO/D3/Strict2031 ownership evaluation matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_mmgdino_e5_ownership_cache import ROUTES, evaluate_cache
from tools.mmgdino_e5_ownership import OWNERSHIP_MODES
from tools.responsibility_isolation_cache import file_sha256
from tools.train_mmgdino_e5_ownership import FORMAL_SEEDS


EXPERIMENT_ROOT = ROOT / "outputs/mmgdino_e5_ownership_transfer_20260821"
PREREG = ROOT / "paper/data/mmgdino_e5_ownership_evaluation_preregistration.json"
STRICT_PATH_AMENDMENT = ROOT / "paper/data/mmgdino_e5_ownership_strict_path_amendment.json"
VAL_SCHEMA_AMENDMENT = ROOT / "paper/data/mmgdino_e5_ownership_val_schema_amendment.json"
PRIMARY_REF_SURFACES = ("refcoco_testA", "refcoco_testB")
REF_SURFACES = ("refcoco_val",) + PRIMARY_REF_SURFACES


class EvaluationRunError(RuntimeError):
    pass


def _checkpoint(route: str, seed: int) -> Path:
    return EXPERIMENT_ROOT / f"formal/{route}/seed{seed}/checkpoint_u150.pt"


def _output(surface: str, route: str, seed: int | None) -> Path:
    suffix = "native" if route == "native" else f"{route}/seed{seed}"
    return EXPERIMENT_ROOT / f"evaluation/{surface}/{suffix}"


def _cache(surface: str) -> Path:
    return EXPERIMENT_ROOT / f"evaluation_caches/{surface}.pt"


def _validate_prereg(*, allow_val_amendment: bool = False) -> dict[str, Any]:
    value = json.loads(PREREG.read_text(encoding="utf-8"))
    if value.get("status") != "locked_after_u150_and_before_eval_cache_extraction":
        raise EvaluationRunError("evaluation preregistration status drifted")
    for name, relative in (
        ("eval_cache_extractor", "tools/extract_mmgdino_e5_eval_cache.py"),
        ("cache_evaluator", "tools/eval_mmgdino_e5_ownership_cache.py"),
        ("bootstrap", "tools/aggregate_mmgdino_e5_ownership.py"),
        ("score_adapter", "models/GroundingDINO/stage_b_gdino_score_adapter.py"),
    ):
        actual = file_sha256(ROOT / relative)
        if value["code"][name]["sha256"] != actual:
            covered = False
            if name == "eval_cache_extractor":
                amendment = json.loads(
                    STRICT_PATH_AMENDMENT.read_text(encoding="utf-8")
                )
                covered = (
                    amendment.get("status")
                    == "locked_before_strict2031_model_forward"
                    and amendment.get("change", {}).get(
                        "new_eval_cache_extractor_sha256"
                    )
                    == actual
                )
            if allow_val_amendment and name in (
                "eval_cache_extractor", "cache_evaluator"
            ):
                amendment = json.loads(
                    VAL_SCHEMA_AMENDMENT.read_text(encoding="utf-8")
                )
                expected_key = (
                    "tools/extract_mmgdino_e5_eval_cache.py"
                    if name == "eval_cache_extractor"
                    else "tools/eval_mmgdino_e5_ownership_cache.py"
                )
                covered = covered or (
                    amendment.get("status")
                    == "locked_before_refcoco_val_rerun"
                    and amendment.get("code_after_amendment", {}).get(expected_key)
                    == actual
                )
            if not covered:
                raise EvaluationRunError(
                    f"preregistered code SHA drift is not covered: {name}"
                )
    for route in OWNERSHIP_MODES:
        for seed in FORMAL_SEEDS:
            if value["checkpoints"][route][str(seed)]["checkpoint"]["sha256"] != file_sha256(_checkpoint(route, seed)):
                raise EvaluationRunError(
                    f"preregistered checkpoint drifted: {route}/seed{seed}"
                )
    return value


def _is_complete(path: Path) -> bool:
    summary = path / "summary.json"
    records = path / "records.jsonl"
    if not summary.is_file() or not records.is_file():
        return False
    value = json.loads(summary.read_text(encoding="utf-8"))
    return (
        value.get("status") == "complete"
        and value.get("records", {}).get("sha256") == file_sha256(records)
    )


def status() -> dict[str, Any]:
    _validate_prereg(allow_val_amendment=True)
    rows = []
    for surface in (*REF_SURFACES, "d3_calibration", "strict2031"):
        route_values = ("native",) if surface == "d3_calibration" else ROUTES
        # Native calibration is intentionally not evaluated.
        if surface == "d3_calibration":
            route_values = OWNERSHIP_MODES
        for route in route_values:
            seeds = (None,) if route == "native" else FORMAL_SEEDS
            for seed in seeds:
                output = _output(surface, route, seed)
                rows.append(
                    {
                        "surface": surface,
                        "route": route,
                        "seed": seed,
                        "cache_ready": _cache(surface).is_file(),
                        "complete": _is_complete(output),
                    }
                )
    return {
        "schema": "arrow.mmgdino_e5_ownership.evaluation_status/v1",
        "rows": rows,
        "complete": sum(row["complete"] for row in rows),
        "total": len(rows),
    }


def _run_one(
    *,
    surface: str,
    route: str,
    seed: int | None,
    device: str,
    fixed_threshold: float | None = None,
) -> dict[str, Any]:
    output = _output(surface, route, seed)
    if _is_complete(output):
        return json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if output.exists():
        raise EvaluationRunError(f"partial evaluation output exists: {output}")
    return evaluate_cache(
        cache_path=_cache(surface),
        route=route,
        surface=surface,
        output_dir=output,
        checkpoint_path=(None if route == "native" else _checkpoint(route, int(seed))),
        fixed_threshold=fixed_threshold,
        device=device,
        batch_size=32,
    )


def run(*, device: str, include_val: bool = False) -> dict[str, Any]:
    _validate_prereg(allow_val_amendment=include_val)
    ref_surfaces = REF_SURFACES if include_val else PRIMARY_REF_SURFACES
    for surface in (*ref_surfaces, "d3_calibration", "strict2031"):
        if not _cache(surface).is_file():
            raise EvaluationRunError(f"evaluation cache is missing: {surface}")
    completed = []
    thresholds: dict[str, float] = {}
    for route in OWNERSHIP_MODES:
        for seed in FORMAL_SEEDS:
            summary = _run_one(
                surface="d3_calibration",
                route=route,
                seed=seed,
                device=device,
            )
            threshold = float(summary["metrics"]["domain_q05"])
            thresholds[f"{route}:{seed}"] = threshold
            completed.append(f"d3_calibration/{route}/seed{seed}")
    for surface in ref_surfaces:
        _run_one(surface=surface, route="native", seed=None, device=device)
        completed.append(f"{surface}/native")
        for route in OWNERSHIP_MODES:
            for seed in FORMAL_SEEDS:
                _run_one(surface=surface, route=route, seed=seed, device=device)
                completed.append(f"{surface}/{route}/seed{seed}")
    _run_one(surface="strict2031", route="native", seed=None, device=device)
    completed.append("strict2031/native")
    for route in OWNERSHIP_MODES:
        for seed in FORMAL_SEEDS:
            _run_one(
                surface="strict2031",
                route=route,
                seed=seed,
                device=device,
                fixed_threshold=thresholds[f"{route}:{seed}"],
            )
            completed.append(f"strict2031/{route}/seed{seed}")
    return {"completed": completed, "thresholds": thresholds, "status": status()}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "run", "reconcile"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--include-val", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    result = (
        status()
        if args.action in ("status", "reconcile")
        else run(device=args.device, include_val=args.include_val)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
