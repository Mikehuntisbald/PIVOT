#!/usr/bin/env python3
"""Bind complete fixed endpoints before evaluation; never select by metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

LOCALIZERS = ("mmgdino_positive", "mdetr_r101_refcoco_ema")
SEEDS = ("17", "42", "73")


def bind(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def verify(binding):
    if bind(binding["path"])["sha256"] != binding["sha256"]:
        raise ValueError("bound artifact changed: " + binding["path"])


def publish_json(path, value):
    """Publish a complete JSON atomically, without overwriting a prior seal."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("append-only artifact exists: " + str(path))
    temp = path.with_name(path.name + ".partial")
    with temp.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # Atomic create-if-absent: rename() would silently replace a racing seal.
    os.link(temp, path)
    temp.unlink()


def check_postflight(path, protocol, localizer, seed):
    post = json.loads(Path(path).read_text())
    expected = [f"{readout}__{target}" for readout in (
        ("native_selected",) if localizer == LOCALIZERS[0] else ("global_max", "native_selected"))
        for target in ("exists", "emit")]
    if (post.get("schema") != "arrow.confidence_readout.training_postflight/v1"
            or post.get("status") != "complete" or post.get("seed") != int(seed)
            or post.get("localizer") != localizer or set(post.get("arms", [])) != set(expected)
            or post.get("updates_per_head") != 12575 or post.get("epochs") != 5
            or post.get("no_optimizer_skips") is not True or post.get("no_amp") is not True
            or post.get("validation_model_evaluations") != 0 or post.get("test_forwards") != 0
            or post.get("gref_forwards") != 0):
        raise ValueError("incomplete/altered formal head endpoint")
    for key in ("checkpoint", "design", "initial_state"):
        verify(post[key])
    design = json.loads(Path(post["design"]["path"]).read_text())
    if design["study_protocol"] != protocol:
        raise ValueError("head does not belong to fixed study")
    if len(post.get("history", [])) != 5:
        raise ValueError("missing epoch audit history")
    for epoch, row in enumerate(post["history"], 1):
        if (row.get("epoch") != epoch or row.get("updates") != epoch * 2515
                or row.get("amp_skips") != 0 or row.get("nonfinite") != 0):
            raise ValueError("training health/update drift")
    if set(post.get("ownership", {})) != set(expected):
        raise ValueError("missing head ownership")
    for owner in post["ownership"].values():
        if owner.get("confidence_tensors") != 8 or owner.get("confidence_parameters") != 50179:
            raise ValueError("head capacity drift")
    return post


def terminal_for_head(root, localizer, seed, postpath):
    """A valid checkpoint with a killed/failed process is not a completed run."""
    wanted_output = (root / "heads" / localizer / f"seed{seed}").resolve()
    for path in sorted((root / "launches").glob("*/terminal.json")):
        value = json.loads(path.read_text())
        nested = value.get("terminal", {}).get(str(seed), {})
        if nested.get("returncode") == 0 and nested.get("postflight") == bind(postpath):
            return bind(path)
        command = value.get("command", [])
        if (value.get("returncode") != 0 or not isinstance(command, list)
                or not any(str(p).endswith("train_confidence_readout_heads.py") for p in command)):
            continue
        try:
            actual_seed = command[command.index("--seed") + 1]
            actual_loc = command[command.index("--localizer") + 1]
            output = Path(command[command.index("--output") + 1])
            if not output.is_absolute():
                output = root.parent.parent / output
            if actual_seed == str(seed) and actual_loc == localizer and output.resolve() == wanted_output:
                return bind(path)
        except (ValueError, IndexError):
            continue
    raise ValueError(f"no successful terminal exit for {localizer}/seed{seed}")


def seal(protocol_path, *, localizer=None, surface="val"):
    protocol = bind(protocol_path)
    root = Path(protocol["path"]).parent
    study = json.loads(Path(protocol["path"]).read_text())
    if study.get("schema") != "arrow.confidence_readout.study_protocol/v1":
        raise ValueError("study schema mismatch")
    if localizer:
        seeds = {}
        for seed in SEEDS:
            postpath = root / "heads" / localizer / f"seed{seed}" / "postflight.json"
            post = check_postflight(postpath, protocol, localizer, seed)
            entry = {"readout": post["checkpoint"], "postflight": bind(postpath),
                     "execution": terminal_for_head(root, localizer, seed, postpath)}
            if localizer == LOCALIZERS[0]:
                old = study["localizers"][localizer]["reused_global_heads"][seed]
                entry["legacy_global"] = old["checkpoint"]
                if surface == "val":
                    entry["legacy_records"] = old["records"]
                else:
                    entry["legacy_records"] = bind(root.parent / "arrow_gref_fixed_targets_20260905/all_records.json")
                    if entry["legacy_records"]["sha256"] != "b0eed5b50e71665929d4c5eed81733f61473e752fafb6b74b500179c243bac0c":
                        raise ValueError("sealed old gRef records drift")
                for record in (entry["legacy_global"], entry["legacy_records"]):
                    verify(record)
            seeds[seed] = entry
        return {"schema": "arrow.confidence_readout.checkpoint_panel/v1", "localizer": localizer,
                "study_protocol": protocol, "surface": surface, "seeds": seeds}
    postflights, executions, count = {}, {}, 0
    for loc in LOCALIZERS:
        postflights[loc] = {}
        executions[loc] = {}
        for seed in SEEDS:
            path = root / "heads" / loc / f"seed{seed}" / "postflight.json"
            post = check_postflight(path, protocol, loc, seed)
            count += len(post["arms"])
            postflights[loc][seed] = bind(path)
            executions[loc][seed] = terminal_for_head(root, loc, seed, path)
    if count != 18:
        raise ValueError("all eighteen new heads are mandatory")
    return {"schema": "arrow.confidence_readout.all_heads_sealed/v1", "status": "complete",
            "study_protocol": protocol, "trajectories": count, "postflights": postflights, "executions": executions,
            "metric_selection": False, "sealer": bind(__file__)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--localizer", choices=LOCALIZERS)
    parser.add_argument("--surface", choices=("val", "gref"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = seal(args.protocol, localizer=args.localizer, surface=args.surface)
    publish_json(args.output, result)
    print(json.dumps(bind(args.output), indent=2))
