#!/usr/bin/env python3
"""Portable v8 build gate: verify sealed bytes, never rewrite absolute lineage.

The original experiment/render scripts remain byte-for-byte frozen. This public
wrapper rebinds their *locations* to the current checkout while retaining every
source/output SHA and all semantic checks in the original result loader.
"""
from __future__ import annotations
import json
from pathlib import Path
import build_coverage_v8_assets as v8


def source_hashes(bindings):
    return {key:value["sha256"] for key,value in bindings.items()}


def verify_assets(directory, generator, sources):
    directory=Path(directory).resolve()
    receipt=json.loads((directory/"receipt.json").read_text())
    expected=receipt.get("generator_sha256",receipt.get("generator",{}).get("sha256"))
    if expected!=v8.old.sha(generator):raise ValueError("frozen renderer SHA drift")
    if source_hashes(receipt["sources"])!=source_hashes(sources):
        raise ValueError("sealed source SHA/key drift")
    for name,digest in receipt["outputs"].items():
        target=(directory/name).resolve()
        if not target.is_relative_to(directory):raise ValueError("asset escapes bundle directory")
        if v8.old.sha(target)!=digest:raise ValueError("sealed output SHA drift: "+name)
    return len(receipt["outputs"])


def verify():
    previous,coverage,sources=v8.load_sources()
    # load_sources verifies complete seeds/populations, training process exits,
    # protocol/code SHAs, Native reference parity and exact state-risk identities.
    _,old_sources=v8.v7.load_sources()
    paper=v8.PAPER
    v7_sources={**old_sources,"legacy_renderer":v8.bind(v8.old.__file__),
                "legacy_assets":v8.bind(paper/"generated/readout_v6/receipt.json")}
    groups=(
        ("readout_v6",v8.old.__file__,old_sources),
        ("evidence_v7_r2",v8.v7.__file__,v7_sources),
        ("evidence_v7_seed_r1",v8.seed_display.__file__,old_sources),
        ("coverage_v8_r2",v8.__file__,sources),
    )
    counts={name:verify_assets(paper/"generated"/name,generator,bindings)
            for name,generator,bindings in groups}
    facts=json.loads((paper/"generated/coverage_v8_r2/risk_decomposition.json").read_text())
    if facts!=v8.decompositions(coverage):raise ValueError("point decomposition drift")
    seed=json.loads((paper/"generated/evidence_v7_seed_r1/seed_effects.json").read_text())
    if seed!=v8.seed_display.seed_effects(previous):raise ValueError("paired seed display drift")
    return {"schema":"arrow.paper.portable_bundle_check/v1","status":"passed",
            "assets":counts,"sealed_artifacts_rewritten":False,"source_location_rebased":True,
            "model_scoring":False,"bootstrap_recomputed":False}


if __name__=="__main__":print(json.dumps(verify(),sort_keys=True))
