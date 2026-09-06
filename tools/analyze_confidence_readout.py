#!/usr/bin/env python3
"""Run a locked record-only v6 analysis; no detector, optimizer, or fitting.

Input JSON: {"schema":"arrow.confidence_readout_analysis_input/v1",
 "protocol":{"path":"protocol.json","sha256":"<64 hex>"},
 "protocol_sha256":"<same hex>", "surface":"finecops_val",
 "expected_population":{"records":18455,"images":...,"positive":9426,"no_target":9029},
 "sirc_statistics":{"localizer":{"17":{"path":"stats/17.json","sha256":"..."},...}},
 "runs": {"localizer": {"17":
 {"path":"rows.jsonl", "sha256":"<64 hex>"}, "42":..., "73":...}}}.
Relative record paths resolve relative to the input file. Combination scores
are rechecked against the bound training-only statistics, not refitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from tools.confidence_readout_metrics import CELLS, SEEDS, analyze_readout, combine_scores

LOCALIZERS = ("mmgdino_positive", "mdetr_r101_refcoco_ema")
ANALYSIS_DEPENDENCIES = (
    "confidence_readout_metrics.py", "analyze_confidence_readout.py",
    "grounding_emission_audit.py", "grounding_generalized_risk_audit.py",
    "grounding_prevalence_audit.py", "grounding_confidence_ordering.py",
)
SURFACES = {
    "finecops_val": {"records": 18455, "positive": 9426, "no_target": 9029},
    "gref_full": {"records": 20684, "images": 1500, "positive": 11563, "no_target": 9121},
    "gref_finecops_train_val_source_disjoint": {
        "records": 17564, "images": 1277, "positive": 9848, "no_target": 7716,
    },
}


def _digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _bound_json(entry, base):
    target = Path(entry["path"])
    if not target.is_absolute():
        target = base / target
    if _digest(target) != entry["sha256"]:
        raise ValueError(f"bound JSON SHA drift: {target.name}")
    return json.loads(target.read_text())


def _check_population(rows, expected, surface):
    got = {"records": len(rows), "images": len({r["cluster_id"] for r in rows}),
           "positive": sum(r["kind"] == "positive" for r in rows),
           "no_target": sum(r["kind"] != "positive" for r in rows)}
    if got != expected:
        raise ValueError(f"locked population drift: expected {expected}, got {got}")
    strata = {r["stratum"] for r in rows}
    if surface.startswith("gref"):
        if strata not in ({"testA", "testB"}, {"A", "B"}):
            raise ValueError("gRef requires TestA/TestB bootstrap strata")
        if any(r["kind"] not in ("positive", "no_target") or r.get("parent_positive_id") is not None for r in rows):
            raise ValueError("gRef is single/no-target only, without edited-expression parents")
    elif any(r["kind"] not in ("positive", "text") for r in rows):
        raise ValueError("FineCops val is positive/negative-text only")


def load_manifest(path, stage_mm_only=False):
    source = json.loads(path.read_text())
    if source.get("schema") != "arrow.confidence_readout_analysis_input/v1":
        raise ValueError("versioned readout analysis input required")
    if not re.fullmatch(r"[0-9a-f]{64}", source.get("protocol_sha256", "")):
        raise ValueError("bound protocol SHA required before real-record analysis")
    binding = source.get("protocol", {})
    if binding.get("sha256") != source["protocol_sha256"]:
        raise ValueError("protocol file binding must match protocol_sha256")
    protocol = _bound_json(binding, path.parent)
    if protocol.get("schema") != "arrow.confidence_readout.study_protocol/v1":
        raise ValueError("locked readout study protocol required")
    if source.get("analysis_code_lock") is not None:
        lock = _bound_json(source["analysis_code_lock"], path.parent)
        for filename in ANALYSIS_DEPENDENCIES:
            entry = lock.get("code", {}).get("tools/" + filename)
            if not isinstance(entry, dict) or entry.get("sha256") != _digest(Path(__file__).parent / filename):
                raise ValueError("analysis code differs from pre-evaluation implementation lock")
    surface = source.get("surface")
    if surface not in SURFACES or surface not in protocol.get("evaluation", {}).get("surfaces", []):
        raise ValueError("known locked evaluation surface required")
    expected = source.get("expected_population")
    if (not isinstance(expected, dict) or set(expected) != {"records", "images", "positive", "no_target"}
            or any(type(v) is not int or v < 1 for v in expected.values())
            or any(expected.get(k) != v for k, v in SURFACES[surface].items())):
        raise ValueError("official locked surface population required")
    wanted = {"mmgdino_positive"} if stage_mm_only else set(LOCALIZERS)
    if set(source.get("runs", {})) != wanted:
        raise ValueError("full analysis needs both localizers; staged MM-only needs --stage-mm-only")
    if set(source.get("sirc_statistics", {})) != wanted:
        raise ValueError("per-localizer/seed training-statistics bindings required")
    runs, bindings = {}, {}
    for localizer, seeds in source["runs"].items():
        if set(seeds) != set(SEEDS) or set(source["sirc_statistics"][localizer]) != set(SEEDS):
            raise ValueError("all three seeds and their training statistics are required")
        runs[localizer], bindings[localizer] = {}, {}
        for seed, entry in seeds.items():
            record_path = Path(entry["path"])
            if not record_path.is_absolute():
                record_path = path.parent / record_path
            digest = _digest(record_path)
            if digest != entry["sha256"]:
                raise ValueError(f"record SHA drift: {localizer}/{seed}")
            with record_path.open() as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            _check_population(rows, expected, surface)
            stats_binding = source["sirc_statistics"][localizer][seed]
            stats = _bound_json(stats_binding, path.parent)
            if stats.get("count") != 83341 or not re.fullmatch(r"[0-9a-f]{64}", stats.get("inputs_sha256", "")):
                raise ValueError("statistics must bind all 83341 unique training positives")
            joint = combine_scores([r["native_score"] for r in rows],
                                   [r["scores"][CELLS[0]] for r in rows], stats)
            for name, values in joint.items():
                if any(r["scores"].get(name) != float(values[i]) for i, r in enumerate(rows)):
                    raise ValueError("record combination scores do not match bound train statistics")
            if not all(r.get("readout_diagnostics") for r in rows):
                raise ValueError("full matched and cross-readout diagnostics required")
            runs[localizer][seed] = rows
            bindings[localizer][seed] = {"sha256": digest, "rows": len(rows),
                                        "sirc_statistics_sha256": stats_binding["sha256"]}
    return source, runs, bindings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--no-conditionals", action="store_true", help="diagnostic run, not full v6 analysis")
    parser.add_argument("--stage-mm-only", action="store_true", help="staged first-localizer mechanism analysis; never final v6")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("append-only analysis output already exists")
    source, runs, bindings = load_manifest(args.input, stage_mm_only=args.stage_mm_only)
    result = analyze_readout(runs, iterations=args.iterations, seed=args.seed,
                             conditionals=not args.no_conditionals,
                             progress=lambda n, total: print(f"bootstrap {n}/{total}", flush=True))
    result["receipt"] = {
        "protocol_sha256": source["protocol_sha256"], "input_sha256": _digest(args.input),
        "records": bindings, "created_utc": datetime.now(timezone.utc).isoformat(),
        "surface": source["surface"], "stage_mm_only": args.stage_mm_only,
        "code_sha256": {name: _digest(Path(__file__).parent / name) for name in ANALYSIS_DEPENDENCIES},
        "formal_requested_configuration": args.iterations == 5000 and args.seed == 20260911 and not args.no_conditionals and not args.stage_mm_only,
        "study_final_receipt": False,
        "model_forward": False, "checkpoint_selection": False, "threshold_fitting": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
