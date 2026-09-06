#!/usr/bin/env python3
"""Lock metric code or build hash-bound inputs from completed readout records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.seal_confidence_readout_heads import LOCALIZERS, SEEDS, bind, verify, publish_json

CODE = ("tools/confidence_readout_metrics.py", "tools/analyze_confidence_readout.py",
        "tools/grounding_emission_audit.py", "tools/grounding_generalized_risk_audit.py",
        "tools/grounding_prevalence_audit.py", "tools/grounding_confidence_ordering.py")


def write_new(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("existing analysis input differs")
        return
    publish_json(path, value)


def run(args):
    protocol = bind(args.protocol)
    study = json.loads(args.protocol.read_text())
    root = args.protocol.resolve().parent
    lockpath = root / "analysis_code_lock.json"
    if args.command == "lock":
        if (root / "analysis").exists():
            raise ValueError("analysis output directory predates code freeze")
        write_new(lockpath, {"schema": "arrow.confidence_readout.analysis_code_lock/v1",
                            "protocol": protocol, "code": {name: bind(ROOT/name) for name in CODE}})
        print(json.dumps(bind(lockpath), indent=2))
        return
    lock = json.loads(lockpath.read_text())
    if lock["protocol"] != protocol:
        raise ValueError("analysis protocol drift")
    for name in CODE:
        verify(lock["code"][name])
    if args.surface not in study["evaluation"]["surfaces"]:
        raise ValueError("unplanned evaluation surface")
    if args.stage_mm_only and args.surface != "finecops_val":
        raise ValueError("only val may be inspected at the first-localizer stage")
    localizers = LOCALIZERS[:1] if args.stage_mm_only else LOCALIZERS
    if not args.stage_mm_only:
        sealed = json.loads((root / "all_heads_sealed.json").read_text())
        if (sealed.get("status") != "complete" or sealed.get("trajectories") != 18
                or sealed.get("study_protocol") != protocol):
            raise ValueError("formal analysis needs all eighteen new heads")
    inputs = root / "analysis_inputs"
    runs, statistics, postflights = {}, {}, {}
    populations, identities, subsets = [], [], []
    source_surface = "finecops_val" if args.surface == "finecops_val" else "gref_full"
    for loc in localizers:
        records_root = root / "evaluation" / loc / source_surface
        postpath = records_root / "postflight.json"
        post = json.loads(postpath.read_text())
        if (post.get("schema") != "arrow.confidence_readout.cache_evaluation_postflight/v1"
                or post.get("status") != "complete" or post.get("mode") != "evaluation"
                or post.get("localizer") != loc or set(post.get("records", {})) != set(SEEDS)
                or post.get("native_boxes_invariant") is not True or post.get("optimizer_updates") != 0
                or (loc == LOCALIZERS[0] and post.get("legacy_global_bitwise_parity") is not True)):
            raise ValueError("completed/parity-checked record evaluation required")
        verify(post["design"])
        design = json.loads(Path(post["design"]["path"]).read_text())
        if design["study_protocol"] != protocol or design.get("localizer") != loc or design.get("mode") != "evaluation":
            raise ValueError("record protocol mismatch")
        verify(design["checkpoint_panel"])
        panel = json.loads(Path(design["checkpoint_panel"]["path"]).read_text())
        if panel.get("localizer") != loc or panel.get("study_protocol") != protocol or set(panel.get("seeds", {})) != set(SEEDS):
            raise ValueError("record checkpoint panel belongs to another localizer")
        verify(design["training_statistics"])
        stats = json.loads(Path(design["training_statistics"]["path"]).read_text())
        if stats["study_protocol"] != protocol or stats["unique_train_positive_count"] != 83341 or stats.get("localizer") != loc:
            raise ValueError("training-only statistic provenance drift")
        runs[loc], statistics[loc] = {}, {}
        postflights[loc] = bind(postpath)
        for seed in SEEDS:
            record_binding = post["records"][seed]
            verify(record_binding)
            with Path(record_binding["path"]).open() as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
            if args.surface.endswith("source_disjoint"):
                rows = [r for r in rows if r["finecops_train_val_source_disjoint"]]
                target = inputs / "source_disjoint" / loc / f"seed{seed}.jsonl"
                subsets.append((loc, seed, target, rows))
            population = {"records": len(rows), "images": len({r["cluster_id"] for r in rows}),
                          "positive": sum(r["kind"] == "positive" for r in rows),
                          "no_target": sum(r["kind"] != "positive" for r in rows)}
            populations.append(population)
            identities.append([(r["sample_id"], r["cluster_id"], r["stratum"], r["kind"]) for r in rows])
            runs[loc][seed] = record_binding
            statistics[loc][seed] = stats["statistics"][seed]
            verify(statistics[loc][seed])
    if any(p != populations[0] for p in populations) or any(i != identities[0] for i in identities):
        raise ValueError("localizer/seed sample universe or ordering differs")
    # Validate every source first; an interrupted preparation can then verify
    # already published identical subsets, never adopt differing bytes.
    for loc, seed, target, rows in subsets:
        payload = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
        if target.exists():
            if target.read_text() != payload:
                raise ValueError("existing source-disjoint subset differs")
        else:
            import os
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + ".partial")
            if temp.exists():
                if temp.read_text() != payload:
                    raise ValueError("incomplete subset needs explicit recovery; preserved")
            else:
                with temp.open("x") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            os.link(temp, target)
            temp.unlink()
        runs[loc][seed] = bind(target)
    source = {"schema": "arrow.confidence_readout_analysis_input/v1", "protocol": protocol,
              "protocol_sha256": protocol["sha256"], "surface": args.surface,
              "expected_population": populations[0], "analysis_code_lock": bind(lockpath),
              "runs": runs, "sirc_statistics": statistics, "source_postflights": postflights,
              "stage_mm_only": args.stage_mm_only}
    suffix = "_mm_stage" if args.stage_mm_only else ""
    target = inputs / (args.surface + suffix + ".json")
    write_new(target, source)
    print(json.dumps(bind(target), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("lock", "inputs"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--surface", choices=("finecops_val", "gref_full", "gref_finecops_train_val_source_disjoint"))
    parser.add_argument("--stage-mm-only", action="store_true")
    run(parser.parse_args())
