#!/usr/bin/env python3
"""Supervise three fixed seed jobs; persist launch and terminal exit receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def bind(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(args):
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("schema") != "arrow.confidence_coverage.study_protocol/v1":
        raise ValueError("study protocol identity mismatch")
    if len(set(args.gpus)) != 3:
        raise ValueError("three independent GPUs required")
    root = args.protocol.resolve().parent
    launch = root / "launches" / f"{args.localizer}_{time.time_ns()}"
    launch.mkdir(parents=True)
    commands = {}
    for seed, gpu in zip((17, 42, 73), args.gpus):
        # Resolving a venv symlink would silently select the base interpreter.
        commands[str(seed)] = [str(Path(sys.executable).absolute()), str(ROOT / "tools/train_confidence_coverage_heads.py"),
            "--study-protocol", str(args.protocol.resolve()), "--localizer", args.localizer,
            "--train-cache", str(args.train_cache.resolve()), "--val-cache", str(args.val_cache.resolve()),
            "--seed", str(seed), "--device", f"cuda:{gpu}",
            "--output", str(root / "heads" / args.localizer / f"seed{seed}")]
        if args.resume:
            commands[str(seed)].append("--resume")
    receipt = {"schema": "arrow.confidence_coverage.stage_launch/v1", "protocol": bind(args.protocol),
        "localizer": args.localizer, "commands": commands, "status": "launching", "pid": os.getpid(),
        "launcher": bind(__file__), "start_time_unix": time.time(), "training_metrics_selection": False}
    write(launch / "launch.json", receipt)
    processes, logs = {}, {}
    for seed, command in commands.items():
        logs[seed] = (launch / f"seed{seed}.log").open("x")
        processes[seed] = subprocess.Popen(command, cwd=ROOT, stdout=logs[seed], stderr=subprocess.STDOUT)
    receipt.update(status="running", workers={s: p.pid for s, p in processes.items()})
    write(launch / "launch.json", receipt)
    print(json.dumps({"launch": str(launch), "workers": receipt["workers"]}), flush=True)
    while any(p.poll() is None for p in processes.values()):
        time.sleep(5)
    terminal = {}
    for seed, process in processes.items():
        logs[seed].close()
        postflight = root / "heads" / args.localizer / f"seed{seed}" / "postflight.json"
        item = {"returncode": process.returncode, "log": bind(launch / f"seed{seed}.log")}
        if process.returncode == 0 and postflight.exists():
            post = json.loads(postflight.read_text())
            if post.get("status") == "complete" and post.get("updates_per_head") == 12575:
                checkpoint = post["checkpoint"]
                if bind(checkpoint["path"]) == checkpoint:
                    item["postflight"] = bind(postflight)
        terminal[seed] = item
    passed = all(x["returncode"] == 0 and "postflight" in x for x in terminal.values())
    receipt.update(status="complete" if passed else "failed_preserved_for_audit",
                   finish_time_unix=time.time(), terminal=terminal)
    write(launch / "terminal.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    if not passed:
        return 1
    evaluator = [sys.executable, str(ROOT/"tools/evaluate_confidence_coverage.py")]
    for command in ("seal", "evaluate", "evaluate"):
        if command == "seal":
            argv = evaluator + [command, "--protocol", str(args.protocol)]
        else:
            surface = "finecops_val" if not (root/"evaluation/finecops_val/postflight.json").exists() else "gref_full"
            argv = evaluator + [command, "--protocol", str(args.protocol), "--surface", surface, "--device", "cuda:3"]
        code = subprocess.call(argv, cwd=ROOT)
        if code: return code
    jobs=[]
    for surface in ("finecops_val","gref_full","gref_source_disjoint"):
        log=(launch/(surface+".bootstrap.log")).open("x")
        argv=evaluator+["analyze","--protocol",str(args.protocol),"--surface",surface]
        jobs.append((surface,subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT),log,argv))
    statuses={}
    for surface, process, log, argv in jobs:
        statuses[surface]={"returncode":process.wait(),"command":argv};log.close()
        if statuses[surface]["returncode"]==0:
            statuses[surface]["result"]=bind(root/"analysis"/(surface+".json"))
    complete=all(v["returncode"]==0 and "result" in v for v in statuses.values())
    write(root/"completion.json",{"schema":"arrow.confidence_coverage.completion/v1",
        "status":"complete" if complete else "failed_preserved","protocol":bind(args.protocol),
        "training_terminal":bind(launch/"terminal.json"),"analyses":statuses,"new_heads":12,
        "detector_forwards":0,"finecops_test_access":False})
    return 0 if complete else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--localizer", choices=("mmgdino_positive",), required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=3, default=[0, 1, 2])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.detach:
        logs = args.protocol.resolve().parent / "launches"
        logs.mkdir(parents=True, exist_ok=True)
        logpath = logs / f"supervisor_{args.localizer}_{time.time_ns()}.log"
        with logpath.open("x") as stream:
            process = subprocess.Popen([sys.executable, __file__, *[x for x in sys.argv[1:] if x != "--detach"]],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"supervisor_pid": process.pid, "log": str(logpath)}))
    else:
        sys.exit(run(args))
