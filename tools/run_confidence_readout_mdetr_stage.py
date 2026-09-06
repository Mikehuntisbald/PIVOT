#!/usr/bin/env python3
"""Fixed four-GPU MDETR cache extraction followed by three-seed head training.

Only train/val are accepted. This stage cannot open gRef or FineCops Test.
Prerequisites are explicit passed smoke receipts and the locked study, not
metrics. Commands and exit codes survive a disconnected SSH client.
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
from tools.seal_confidence_readout_heads import bind, check_postflight, verify


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def execute(commands, directory):
    directory.mkdir(parents=True, exist_ok=False)
    processes, handles = [], []
    for index, command in enumerate(commands):
        handle = (directory / f"worker{index}.log").open("x")
        handles.append(handle)
        processes.append(subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT))
    launch = {"commands": commands, "pids": [p.pid for p in processes], "start_time": time.time()}
    write(directory / "launch.json", launch)
    while any(p.poll() is None for p in processes):
        if any(p.poll() not in (None, 0) for p in processes):
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            break
        time.sleep(5)
    exits = [p.wait() for p in processes]
    for handle in handles:
        handle.close()
    launch.update(returncodes=exits, end_time=time.time(), logs=[bind(directory / f"worker{i}.log") for i in range(len(commands))])
    write(directory / "terminal.json", launch)
    if any(exits):
        raise RuntimeError("cache/head stage failed; partial artifacts preserved")


def run(args):
    protocol_binding = bind(args.protocol)
    protocol = json.loads(args.protocol.read_text())
    root = args.protocol.resolve().parent
    smoke = json.loads(args.smoke.read_text())
    if (protocol.get("schema") != "arrow.confidence_readout.study_protocol/v1"
            or smoke.get("status") != "passed" or not smoke.get("repeat_bitwise_parity")
            or not smoke.get("raw_postprocess_bitwise_parity") or not smoke.get("preprocess_bitwise_parity")
            or smoke["runtime"].get("state_key") != "model_ema"):
        raise ValueError("fixed scientific protocol and EMA runtime preflight required")
    for key in ("runtime_code", "extractor_code", "fixture"):
        verify(smoke[key])
    runtime = smoke["runtime"]
    verify(runtime["checkpoint"])
    verify(runtime["text_assets"])
    checkpoint = runtime["checkpoint"]
    text_assets = str(Path(runtime["text_assets"]["path"]).parent)
    prep = args.preparation.resolve() if args.preparation is not None else root / "mdetr_preparation.json"
    cache_root = args.cache_root.resolve() if args.cache_root is not None else root / "cache/mdetr_r101_refcoco_ema"
    extractor = args.extractor.resolve(strict=True)
    if bind(extractor) != smoke["extractor_code"]:
        raise ValueError("selected extractor differs from runtime preflight")
    preparation = {"schema": "arrow.confidence_readout.mdetr_preparation/v1", "study_protocol": protocol_binding,
        "smoke": bind(args.smoke), "runtime": runtime, "extractor": smoke["extractor_code"],
        "runtime_environment": bind(args.runtime_python.parent.parent / "runtime_receipt.json"),
        "runtime_python": str(args.runtime_python.absolute()), "head_python": str(args.head_python.absolute()),
        "source_caches": {s: protocol["localizers"]["mmgdino_positive"][f"{s}_cache"] for s in ("train", "val")},
        "upstream": str(args.upstream.resolve()), "cache_root": str(cache_root), "cache_extractor": bind(extractor),
        "new_test_forwards": 0, "gref_forwards": 0,
        "extractor_batch": 1, "worker_count": 4, "shard_size": 128, "cache_dtype": "FP16 features, FP32 scores/boxes"}
    if args.supersedes is not None:
        preparation["supersedes"] = bind(args.supersedes)
        preparation["revision_reason"] = "negative annotation reference boxes are not study GT; no target-label, head, or detector change"
    if prep.exists():
        if json.loads(prep.read_text()) != preparation:
            raise ValueError("immutable MDETR preparation changed")
    else:
        with prep.open("x") as stream:
            json.dump(preparation, stream, indent=2, sort_keys=True)
            stream.write("\n")
    if args.prepare_only:
        print(json.dumps({"preparation": bind(prep), "status": "prepared_no_forward"}, indent=2), flush=True)
        return
    launchroot = root / "launches" / f"mdetr_cache_heads_{time.time_ns()}"
    launchroot.mkdir(parents=True)
    state = {"schema": "arrow.confidence_readout.mdetr_stage/v1", "preparation": bind(prep),
             "launcher": bind(__file__), "pid": os.getpid(), "status": "waiting_mm_heads", "completed_stages": []}
    write(launchroot / "state.json", state)
    print(json.dumps({"launch": str(launchroot), "status": state["status"]}), flush=True)
    while True:
        posts = [root / "heads/mmgdino_positive" / f"seed{s}" / "postflight.json" for s in (17, 42, 73)]
        if all(p.exists() for p in posts):
            for seed, path in zip((17, 42, 73), posts):
                check_postflight(path, protocol_binding, "mmgdino_positive", seed)
            break
        time.sleep(10)
    # No metrics are consulted before committing the second-model matrix.
    common = ["--upstream", str(args.upstream.resolve()), "--checkpoint", checkpoint["path"],
        "--checkpoint-sha256", checkpoint["sha256"], "--text-assets", text_assets,
        "--protocol", str(args.protocol.resolve()), "--protocol-sha256", protocol_binding["sha256"],
        "--smoke-receipt", str(args.smoke.resolve()), "--worker-count", "4", "--shard-size", "128"]
    for split in ("train", "val"):
        output = cache_root / split
        manifest = output / "manifest.json"
        source = preparation["source_caches"][split]
        verify(source)
        state["status"] = f"extracting_{split}"
        write(launchroot / "state.json", state)
        if not manifest.exists():
            commands = [[str(args.runtime_python.absolute()), str(extractor),
                "extract", *common, "--source-manifest", source["path"], "--output", str(output),
                "--worker-index", str(gpu), "--device", f"cuda:{gpu}"] for gpu in range(4)]
            execute(commands, launchroot / f"extract_{split}")
            command = [str(args.runtime_python.absolute()), str(extractor),
                "finalize", "--source-manifest", source["path"], "--output", str(output), "--worker-count", "4"]
            execute([command], launchroot / f"finalize_{split}")
        loaded = json.loads(manifest.read_text())
        if (loaded.get("status") != "complete" or loaded.get("binding", {}).get("protocol") != protocol_binding
                or loaded["model"]["checkpoint"] != checkpoint or loaded["binding"]["smoke"] != bind(args.smoke)):
            raise ValueError("completed cache identity drift")
        state["completed_stages"].append({"stage": split, "manifest": bind(manifest)})
        write(launchroot / "state.json", state)
    state["status"] = "training_mdetr_heads"
    write(launchroot / "state.json", state)
    command = [str(args.head_python.absolute()), str(ROOT / "tools/run_confidence_readout_stage.py"),
        "--protocol", str(args.protocol.resolve()), "--localizer", "mdetr_r101_refcoco_ema",
        "--train-cache", str(cache_root / "train/manifest.json"),
        "--val-cache", str(cache_root / "val/manifest.json")]
    execute([command], launchroot / "train_heads")
    command = [str(args.head_python.absolute()), str(ROOT / "tools/seal_confidence_readout_heads.py"),
        "--protocol", str(args.protocol.resolve()), "--output", str(root / "all_heads_sealed.json")]
    execute([command], launchroot / "seal_all_heads")
    state.update(status="complete", all_heads_sealed=bind(root / "all_heads_sealed.json"))
    write(launchroot / "state.json", state)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--head-python", type=Path, required=True)
    parser.add_argument("--extractor", type=Path, default=ROOT / "tools/extract_mdetr_readout_cache.py")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--supersedes", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.detach:
        logs = args.protocol.resolve().parent / "launches"
        logs.mkdir(parents=True, exist_ok=True)
        logpath = logs / f"mdetr_supervisor_{time.time_ns()}.log"
        with logpath.open("x") as stream:
            process = subprocess.Popen([sys.executable, __file__, *[x for x in sys.argv[1:] if x != "--detach"]],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"pid": process.pid, "log": str(logpath)}))
    else:
        run(args)
