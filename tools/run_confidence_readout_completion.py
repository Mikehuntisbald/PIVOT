#!/usr/bin/env python3
"""Complete the frozen v6 evaluation after all heads, without metric selection.

This supervisor waits for all eighteen heads, scores train-only combinations and
validation, extracts one gRef query cache per localizer, and launches the three
prescribed image-cluster analyses. No FineCops Test command is available here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_confidence_readout_mdetr_stage import execute, write
from tools.seal_confidence_readout_heads import LOCALIZERS, bind, seal, verify


def complete_post(path, protocol, localizer, mode, cache, panel):
    if not path.exists():
        return False
    post = json.loads(path.read_text())
    expected_count = 83341 if mode == "train_statistics" else (20684 if "gref" in str(path.parent) else 18455)
    if (post.get("schema") != "arrow.confidence_readout.cache_evaluation_postflight/v1"
            or post.get("status") != "complete" or post.get("mode") != mode
            or post.get("localizer") != localizer or set(post.get("records", {})) != {"17", "42", "73"}
            or post.get("rows_per_seed") != expected_count or post.get("native_boxes_invariant") is not True
            or post.get("optimizer_updates") != 0
            or (mode == "evaluation" and localizer == LOCALIZERS[0] and post.get("legacy_global_bitwise_parity") is not True)):
        raise ValueError("existing postflight identity or completeness differs")
    verify(post["design"])
    design = json.loads(Path(post["design"]["path"]).read_text())
    if (design.get("study_protocol") != protocol or design.get("localizer") != localizer
            or design.get("mode") != mode or design.get("cache") != bind(cache)
            or design.get("checkpoint_panel") != bind(panel)):
        raise ValueError("existing evaluation belongs to another study")
    for record in post.get("records", {}).values():
        verify(record)
    stats_binding = post.get("training_statistics") if mode == "train_statistics" else design.get("training_statistics")
    if not isinstance(stats_binding, dict):
        raise ValueError("missing training-statistics provenance")
    verify(stats_binding)
    stats = json.loads(Path(stats_binding["path"]).read_text())
    if (stats.get("schema") != "arrow.confidence_readout.training_statistics_panel/v1"
            or stats.get("localizer") != localizer or stats.get("study_protocol") != protocol
            or stats.get("unique_train_positive_count") != 83341):
        raise ValueError("existing training-statistics identity differs")
    return True


def validate_analysis(path, source_path, protocol, surface):
    result = json.loads(path.read_text())
    receipt = result.get("receipt", {})
    source = json.loads(source_path.read_text())
    if (result.get("schema") != "arrow.confidence_readout_metrics/v1"
            or receipt.get("formal_requested_configuration") is not True
            or receipt.get("protocol_sha256") != protocol["sha256"]
            or receipt.get("surface") != surface or receipt.get("input_sha256") != bind(source_path)["sha256"]
            or receipt.get("stage_mm_only") is not False or receipt.get("model_forward") is not False
            or receipt.get("checkpoint_selection") is not False or receipt.get("threshold_fitting") is not False):
        raise ValueError("requested full analysis identity missing")
    verify(source["analysis_code_lock"])
    lock = json.loads(Path(source["analysis_code_lock"]["path"]).read_text())
    for binding in lock["code"].values():
        verify(binding)
    expected_code = {Path(name).name: value["sha256"] for name, value in lock["code"].items()}
    if receipt.get("code_sha256") != expected_code:
        raise ValueError("analysis metric implementation differs from lock")
    expected_records = {}
    for loc, seeds in source["runs"].items():
        expected_records[loc] = {}
        for seed, binding in seeds.items():
            verify(binding)
            verify(source["sirc_statistics"][loc][seed])
            expected_records[loc][seed] = {"sha256": binding["sha256"],
                "rows": source["expected_population"]["records"],
                "sirc_statistics_sha256": source["sirc_statistics"][loc][seed]["sha256"]}
    if receipt.get("records") != expected_records:
        raise ValueError("analysis records/statistics differ from frozen input")
    wanted_bootstrap = {"iterations": 5000, "seed": 20260911, "rng": "PCG64", "unit": "image_cluster",
        "required_seeds": ["17", "42", "73"], "same_draw_all_localizers_heads_seeds": True,
        "q05_recomputed_each_draw": True, "fixed_threshold_fit": False}
    if any(result.get("bootstrap", {}).get(k) != v for k, v in wanted_bootstrap.items()):
        raise ValueError("analysis bootstrap contract differs")
    if set(result.get("localizers", {})) != set(LOCALIZERS):
        raise ValueError("analysis localizer matrix incomplete")
    for local in result["localizers"].values():
        if set(local.get("per_seed", {})) != {"17", "42", "73"}:
            raise ValueError("analysis seed matrix incomplete")


def run(args):
    protocol = bind(args.protocol)
    root = args.protocol.resolve().parent
    study = json.loads(args.protocol.read_text())
    codepaths = ("tools/run_confidence_readout_completion.py", "tools/run_confidence_readout_mdetr_stage.py",
        "tools/seal_confidence_readout_heads.py", "tools/evaluate_confidence_readout_cache.py",
        "tools/extract_confidence_readout_gref_cache.py", "tools/prepare_confidence_readout_analysis.py")
    launchroot = root / "launches" / f"completion_{time.time_ns()}"
    launchroot.mkdir(parents=True)
    launch = {"schema": "arrow.confidence_readout.completion_launch/v1", "protocol": protocol,
        "pid": os.getpid(), "code": {name: bind(ROOT / name) for name in codepaths},
        "head_python": str(args.head_python.absolute()), "mdetr_python": str(args.mdetr_python.absolute()),
        "status": "waiting_all_18_heads", "checkpoint_selection": False, "threshold_fitting": False}
    if args.supersedes is not None:
        launch["supersedes"] = bind(args.supersedes)
        launch["restart_reason"] = "infrastructure-only atomic publication/resume/receipt hardening; no model or objective change"
    write(launchroot / "launch.json", launch)
    write(launchroot / "state.json", launch)
    print(json.dumps({"launch": str(launchroot), "status": launch["status"]}), flush=True)
    while not (root / "all_heads_sealed.json").exists():
        time.sleep(10)
    saved = json.loads((root / "all_heads_sealed.json").read_text())
    if seal(args.protocol) != saved:
        raise ValueError("all-head gate differs from independently rechecked endpoints")
    headpy, mdpy = str(args.head_python.absolute()), str(args.mdetr_python.absolute())
    preparation_path = args.mdetr_preparation.resolve() if args.mdetr_preparation is not None else root / "mdetr_preparation.json"
    preparation = json.loads(preparation_path.read_text())
    md_cache_root = Path(preparation.get("cache_root", root / "cache/mdetr_r101_refcoco_ema"))
    evaluator = str(ROOT / "tools/evaluate_confidence_readout_cache.py")
    cachepaths = {LOCALIZERS[0]: {s: study["localizers"][LOCALIZERS[0]][s+"_cache"]["path"] for s in ("train", "val")},
                  LOCALIZERS[1]: {s: str(md_cache_root / s / "manifest.json") for s in ("train", "val")}}
    try:
        for loc in LOCALIZERS:
            panel = root / "panels" / (loc + "_val.json")
            value = seal(args.protocol, localizer=loc, surface="val")
            if panel.exists():
                if json.loads(panel.read_text()) != value:
                    raise ValueError("fixed validation panel drift")
            else:
                panel.parent.mkdir(parents=True, exist_ok=True)
                with panel.open("x") as stream:
                    json.dump(value, stream, indent=2, sort_keys=True)
                    stream.write("\n")
            for mode, surface, split in (("train_statistics", "train_statistics", "train"), ("evaluation", "finecops_val", "val")):
                output = root / "evaluation" / loc / surface
                launch["status"] = f"{loc}_{surface}"
                write(launchroot / "state.json", launch)
                if complete_post(output / "postflight.json", protocol, loc, mode, cachepaths[loc][split], panel):
                    continue
                command = [headpy, evaluator, "--study-protocol", protocol["path"], "--localizer", loc,
                    "--cache", cachepaths[loc][split], "--mode", mode, "--checkpoint-panel", str(panel),
                    "--output", str(output), "--device", "cuda:0"]
                if mode == "evaluation":
                    command += ["--training-statistics", str(root / "evaluation" / loc / "train_statistics/training_statistics.json")]
                execute([command], launchroot / (loc + "_" + surface))
        for loc in LOCALIZERS:
            py = headpy if loc == LOCALIZERS[0] else mdpy
            cache = root / "cache" / loc / "gref_testab"
            common = ["--protocol", protocol["path"], "--protocol-sha256", protocol["sha256"],
                "--all-heads-sealed", str(root / "all_heads_sealed.json"), "--localizer", loc,
                "--output", str(cache), "--val-cache", cachepaths[loc]["val"]]
            if loc == LOCALIZERS[1]:
                common += ["--mdetr-preparation", str(preparation_path)]
            extractor = str(ROOT / "tools/extract_confidence_readout_gref_cache.py")
            launch["status"] = f"{loc}_gref_cache"
            write(launchroot / "state.json", launch)
            if not (cache / "manifest.json").exists():
                commands = [[py, extractor, "extract", *common, "--worker-index", str(gpu), "--device", f"cuda:{gpu}"] for gpu in range(4)]
                execute(commands, launchroot / (loc + "_gref_cache"))
                execute([[py, extractor, "finalize", *common]], launchroot / (loc + "_gref_finalize"))
            manifest = json.loads((cache / "manifest.json").read_text())
            if (manifest.get("status") != "complete" or manifest.get("split") != "gref_testab"
                    or manifest.get("localizer") != loc or manifest.get("all_heads_sealed") != bind(root / "all_heads_sealed.json")):
                raise ValueError("existing gRef cache identity mismatch")
            panel = root / "panels" / (loc + "_gref.json")
            value = seal(args.protocol, localizer=loc, surface="gref")
            if panel.exists():
                if json.loads(panel.read_text()) != value:
                    raise ValueError("gRef checkpoint panel drift")
            else:
                with panel.open("x") as stream:
                    json.dump(value, stream, indent=2, sort_keys=True)
                    stream.write("\n")
            output = root / "evaluation" / loc / "gref_full"
            if not complete_post(output / "postflight.json", protocol, loc, "evaluation", cache / "manifest.json", panel):
                command = [headpy, evaluator, "--study-protocol", protocol["path"], "--localizer", loc,
                    "--cache", str(cache / "manifest.json"), "--mode", "evaluation", "--checkpoint-panel", str(panel),
                    "--training-statistics", str(root / "evaluation" / loc / "train_statistics/training_statistics.json"),
                    "--output", str(output), "--device", "cuda:0"]
                execute([command], launchroot / (loc + "_gref_records"))
        launch["status"] = "three_surface_image_bootstrap"
        write(launchroot / "state.json", launch)
        commands = []
        analyses = {}
        for surface in study["evaluation"]["surfaces"]:
            source = root / "analysis_inputs" / (surface + ".json")
            if not source.exists():
                execute([[headpy, str(ROOT / "tools/prepare_confidence_readout_analysis.py"), "inputs",
                    "--protocol", protocol["path"], "--surface", surface]], launchroot / (surface + "_input"))
            output = root / "analysis" / (surface + ".json")
            analyses[surface] = output
            if not output.exists():
                commands.append([headpy, "-m", "tools.analyze_confidence_readout", "--input", str(source), "--output", str(output)])
        if commands:
            execute(commands, launchroot / "bootstrap")
        for surface, output in analyses.items():
            validate_analysis(output, root / "analysis_inputs" / (surface + ".json"), protocol, surface)
        for record in (*study["finecops_test_state"].values(), *launch["code"].values()):
            verify(record)
        final = {"schema": "arrow.confidence_readout.experimental_completion/v1", "status": "complete",
            "study_protocol": protocol, "all_heads_sealed": bind(root / "all_heads_sealed.json"),
            "analyses": {k: bind(v) for k, v in analyses.items()}, "launcher": bind(launchroot / "launch.json"),
            "new_heads": 18, "new_finecops_test_forwards": 0, "trunk_updates": 0,
            "checkpoint_selection": False, "threshold_fitting": False,
            "paper_build_complete": False, "scientific_interpretation_requires_effects_and_intervals": True}
        with (root / "experimental_completion.json").open("x") as stream:
            json.dump(final, stream, indent=2, sort_keys=True)
            stream.write("\n")
        launch.update(status="complete", final_receipt=bind(root / "experimental_completion.json"))
        write(launchroot / "state.json", launch)
        print(json.dumps(final, indent=2), flush=True)
    except Exception as exc:
        launch.update(status="failed_preserved_for_audit", error=repr(exc))
        write(launchroot / "state.json", launch)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--head-python", type=Path, required=True)
    parser.add_argument("--mdetr-python", type=Path, required=True)
    parser.add_argument("--mdetr-preparation", type=Path)
    parser.add_argument("--supersedes", type=Path)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.detach:
        logfile = args.protocol.resolve().parent / "launches" / f"completion_supervisor_{time.time_ns()}.log"
        with logfile.open("x") as stream:
            process = subprocess.Popen([sys.executable, __file__, *[a for a in sys.argv[1:] if a != "--detach"]],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"pid": process.pid, "log": str(logfile)}))
    else:
        run(args)
