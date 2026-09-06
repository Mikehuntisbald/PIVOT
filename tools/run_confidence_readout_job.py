#!/usr/bin/env python3
"""Detached exact-command runner with immutable launch and terminal records."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write(path, payload):
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run(args):
    job = args.directory.resolve()
    job.mkdir(parents=True, exist_ok=False)
    if not args.command:
        raise ValueError("explicit argv required")
    command = args.command[1:] if args.command[0] == "--" else args.command
    with (job / "output.log").open("x") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        launch = {"schema": "arrow.confidence_readout.job/v1", "command": command,
                  "pid": process.pid, "supervisor_pid": os.getpid(), "started_unix": time.time()}
        write(job / "launch.json", launch)
        code = process.wait()
    write(job / "terminal.json", {**launch, "returncode": code, "finished_unix": time.time()})
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.detach:
        args.directory.parent.mkdir(parents=True, exist_ok=True)
        logfile = args.directory.with_suffix(".supervisor.log")
        with logfile.open("x") as stream:
            process = subprocess.Popen([sys.executable, __file__, *[a for a in sys.argv[1:] if a != "--detach"]],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"pid": process.pid, "directory": str(args.directory), "log": str(logfile)}))
    else:
        sys.exit(run(args))
