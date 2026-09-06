#!/usr/bin/env python3
"""Train one locked seed of matched positive-coverage targets; never score validation.

Only confidence parameters update. The historical initialization and complete
rank/confidence event generator are retained, including skipped rank RNG draws.
All validation access is population/provenance checking, never model scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.confidence_readout import (
    LOCALIZERS, native_labels,
    readout_scores,
)
from tools.confidence_coverage import (load_readout_cache, make_readout_heads, parse_arm,
    training_arms, make_streams, COVERAGES, SCHEMA)
from tools.finecops_fixed_rank_targets import target_loss
from tools.train_finecops_bce_l2_heads import (
    _atomic_json, _atomic_torch, _deterministic_algorithms, _epoch_schedule,
    _stack_rows, _tensor_state_sha256, _validate_population, file_sha256,
)

RECIPE = {"epochs": 5, "updates_per_head": 12575, "pair_batch": 32,
          "lr": 1e-4, "wd": 0, "clip": .1, "logit_l2": .001,
          "dtype": "deterministic_fp32"}
REQUIRED_CODE = (
    "tools/confidence_coverage.py", "tools/train_confidence_coverage_heads.py",
    "tools/confidence_readout.py", "tools/train_confidence_readout_heads.py",
    "tools/finecops_fixed_rank_targets.py", "tools/train_finecops_bce_l2_heads.py",
    "tools/b32a1_heads.py", "tools/mmgdino_e5_ownership.py",
    "tools/responsibility_isolation_cache.py", "tools/b32a1_objectives.py",
    "tools/b32a1_metrics.py", "tools/finecops_bce_l2_control.py",
)


def bind(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": file_sha256(path)}


def verify(binding):
    if bind(binding["path"])["sha256"] != binding["sha256"]:
        raise ValueError("artifact hash drift: " + str(binding["path"]))


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_protocol(path, localizer, seed, train_cache, val_cache):
    protocol = json.loads(Path(path).read_text())
    if (protocol.get("schema") != SCHEMA or protocol.get("seeds") != [17, 42, 73]
            or seed not in protocol["seeds"] or localizer not in LOCALIZERS):
        raise ValueError("study identity drift")
    if any(protocol.get("training", {}).get(k) != v for k, v in RECIPE.items()):
        raise ValueError("fixed recipe drift")
    if (protocol.get("arms") != list(training_arms(localizer))
            or protocol.get("readout") != "global_max"
            or protocol.get("new_heads") != 12
            or protocol.get("positive_sampling", {}).get("positive_presentations_per_head") != 402255):
        raise ValueError("coverage/readout/budget intervention drift")
    code = protocol.get("code", {})
    for name in REQUIRED_CODE:
        if name not in code or code[name]["sha256"] != file_sha256(ROOT / name):
            raise ValueError("required training code not locked: " + name)
        verify(code[name])
    local = protocol.get("localizers", {}).get(localizer)
    if not isinstance(local, dict):
        raise ValueError("localizer missing from protocol")
    caches = {"train_cache": bind(train_cache), "val_cache": bind(val_cache)}
    for key, artifact in caches.items():
        if key in local:
            if artifact != local[key]:
                raise ValueError("protocol cache binding drift")
        elif local.get("cache_status") != "pending_before_head_training":
            raise ValueError("unbound cache is not prospectively declared")
    return protocol, caches


def parameter_audit(model, optimizer, initial_rank):
    active = tuple(p for p in model.parameters() if p.requires_grad)
    confidence = model.task_parameters("confidence")
    owned = tuple(p for group in optimizer.param_groups for p in group["params"])
    if (len(active) != 8 or sum(p.numel() for p in active) != 50179
            or {id(p) for p in active} != {id(p) for p in confidence}
            or len(owned) != len(active) or {id(p) for p in owned} != {id(p) for p in active}):
        raise ValueError("optimizer ownership/capacity drift")
    if any(group["lr"] != 1e-4 or group["weight_decay"] != 0 for group in optimizer.param_groups):
        raise ValueError("optimizer LR or zero-WD contract drift")
    frozen = {k: p.detach().cpu() for k, p in model.named_task_parameters("rank")}
    if set(frozen) != set(initial_rank) or any(not torch.equal(v, initial_rank[k]) for k, v in frozen.items()):
        raise ValueError("frozen rank tensor drift")
    if any(p.grad is not None or p.requires_grad for p in model.task_parameters("rank")):
        raise ValueError("frozen rank autograd connection")
    return {"confidence_tensors": len(active), "confidence_parameters": sum(p.numel() for p in active),
            "frozen_rank_sha256": _tensor_state_sha256(frozen),
            "confidence_sha256": _tensor_state_sha256(dict(model.named_task_parameters("confidence")))}


def train_step(model, optimizer, features, native, mask, selected, correct, *, target, readout, positive_count):
    optimizer.zero_grad(set_to_none=True)
    scores = readout_scores(model, features, native, mask, selected)[readout]
    loss = target_loss(scores[:positive_count], scores[positive_count:], correct, target)
    loss.backward()
    active = model.task_parameters("confidence")
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in active):
        raise ValueError("nonfinite or missing confidence gradient")
    if any(p.grad is not None for p in model.task_parameters("rank")):
        raise ValueError("rank received confidence gradient")
    norm = torch.nn.utils.clip_grad_norm_(active, .1)
    if not torch.isfinite(norm) or not torch.isfinite(loss):
        raise ValueError("nonfinite training health")
    optimizer.step()
    return float(loss.detach()), float(norm.detach())


def rng_state(device):
    return {"python": random.getstate(), "numpy_legacy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_device": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(value, device):
    random.setstate(value["python"])
    np.random.set_state(value["numpy_legacy"])
    torch.set_rng_state(value["torch_cpu"])
    if device.type == "cuda":
        if value["torch_device"] is None:
            raise ValueError("CUDA RNG state missing")
        torch.cuda.set_rng_state(value["torch_device"], device)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def run(args):
    protocol, caches = validate_protocol(args.study_protocol, args.localizer, args.seed,
                                         args.train_cache, args.val_cache)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lockpath = root / "design_lock.json"
    if (root / "postflight.json").exists():
        raise ValueError("completed head run is immutable")
    if not args.resume and any(root.iterdir()):
        raise ValueError("fresh head output directory must be empty")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
        raise ValueError("CUBLAS deterministic workspace contract drift")
    device = torch.device(args.device)
    torch.set_num_threads(2)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    # Match the sealed FP32 route; no autocast, scaler, optimizer skip or TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train, tm = load_readout_cache(args.train_cache, split="train", localizer=args.localizer)
    val, vm = load_readout_cache(args.val_cache, split="val", localizer=args.localizer)
    trunk = tm["model"]["checkpoint"]
    if trunk != vm["model"]["checkpoint"]:
        raise ValueError("train/val frozen localizer mismatch")
    verify(trunk)
    local = protocol["localizers"][args.localizer]
    if "checkpoint" in local and local["checkpoint"] != trunk:
        raise ValueError("localizer checkpoint differs from study lock")
    rank_rows, pairs = _validate_population(train, val)
    if len(rank_rows) != 83341 or len(pairs) != 80451:
        raise ValueError("formal source counts drift")
    # Validation is used only for identity/population checks, never labels/scores.
    population = {"train_positive": len(rank_rows), "train_pairs": len(pairs),
                  "val_positive": sum(r["kind"] == "positive" for r in val),
                  "val_text": sum(r["kind"] == "text" for r in val),
                  "train_order_sha256": canonical_sha([r["sample_id"] for r in train]),
                  "val_order_sha256": canonical_sha([r["sample_id"] for r in val])}
    pools, streams = make_streams(rank_rows, args.seed)
    population["positive_sampling"] = {c: {
        "pool_size": len(pools[c]), "pool_ids_sha256": canonical_sha([r["sample_id"] for r in pools[c]]),
        "stream_sha256": canonical_sha(streams[c].tolist()),
        "draws": len(streams[c]), "min_presentations": int(np.bincount(streams[c]).min()),
        "max_presentations": int(np.bincount(streams[c]).max())} for c in COVERAGES}
    labels, selected_indices = native_labels(train, args.localizer)
    population.update({"train_native_correct": sum(v is True for v in labels.values()),
                       "labels_sha256": canonical_sha(labels),
                       "native_selected_index_sha256": canonical_sha(selected_indices)})
    design = {"schema": "arrow.confidence_coverage.training_lock/v1",
              "study_protocol": bind(args.study_protocol), "seed": args.seed,
              "localizer": args.localizer, "arms": list(training_arms(args.localizer)),
              "training": RECIPE, "caches": caches, "checkpoint": trunk, "population": population,
              "code": {k: protocol["code"][k] for k in REQUIRED_CODE},
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                  "numpy": np.__version__, "cuda": torch.version.cuda, "device": str(device),
                  "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  "torch_num_threads": torch.get_num_threads(),
                  "cublas_workspace": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                  "matmul_allow_tf32": False, "cudnn_allow_tf32": False},
              "validation_model_evaluations": 0, "test_forwards": 0, "gref_forwards": 0}
    if lockpath.exists():
        if not args.resume or json.loads(lockpath.read_text()) != design:
            raise ValueError("resume requires exact training lock")
    else:
        if args.resume or any(root.iterdir()):
            raise ValueError("cannot resume or adopt an unlocked run")
        _atomic_json(lockpath, design)
    lock_binding = bind(lockpath)
    started = time.monotonic()
    with _deterministic_algorithms():
        models = make_readout_heads(args.seed, device, args.localizer)
        initial_rank = {k: p.detach().cpu().clone()
                        for k, p in next(iter(models.values())).named_task_parameters("rank")}
        opts = {arm: torch.optim.AdamW(model.task_parameters("confidence"), lr=1e-4,
                                       weight_decay=0., foreach=False) for arm, model in models.items()}
        initial = {arm: cpu_state(model) for arm, model in models.items()}
        initial_hashes = {arm: _tensor_state_sha256(value) for arm, value in initial.items()}
        initialpath = root / "initial_state.pt"
        if initialpath.exists():
            saved = torch.load(initialpath, map_location="cpu", weights_only=False)
            if saved.get("design") != lock_binding or saved.get("model_hashes") != initial_hashes:
                raise ValueError("initialization identity drift")
        else:
            _atomic_torch(initialpath, {"schema": "arrow.confidence_coverage.initial/v1", "design": lock_binding,
                                      "models": initial, "model_hashes": initial_hashes,
                                      "rng": rng_state(device)})
        start_epoch, updates, history = 0, 0, []
        for epoch in range(1, 6):
            marker = root / f"epoch{epoch}.json"
            if not marker.exists():
                if any((root / f"epoch{later}.json").exists() for later in range(epoch+1, 6)):
                    raise ValueError("resume epoch markers are not contiguous")
                break
            if not args.resume:
                raise ValueError("existing epoch requires explicit resume")
            receipt = json.loads(marker.read_text())
            verify(receipt["checkpoint"])
            ck = torch.load(receipt["checkpoint"]["path"], map_location="cpu", weights_only=False)
            if (ck.get("schema") != "arrow.confidence_coverage.head_checkpoint/v1"
                    or ck["design"] != lock_binding or ck["seed"] != args.seed
                    or ck["epoch"] != epoch or set(ck["models"]) != set(models)
                    or ck["initial_hashes"] != initial_hashes or ck["updates"] != epoch * 2515):
                raise ValueError("checkpoint resume identity drift")
            for arm in models:
                models[arm].load_state_dict(ck["models"][arm], strict=True)
                opts[arm].load_state_dict(ck["optimizers"][arm])
                parameter_audit(models[arm], opts[arm], initial_rank)
            restore_rng(ck["rng"], device)
            start_epoch, updates, history = epoch, ck["updates"], ck["history"]
        for epoch in range(start_epoch + 1, 6):
            events, schedule = _epoch_schedule(seed=args.seed, epoch=epoch,
                                               rank_count=len(rank_rows), confidence_count=len(pairs))
            if schedule["confidence_batches"] != 2515:
                raise ValueError("confidence schedule count drift")
            for model in models.values():
                model.train()
            loss_sums = dict.fromkeys(models, 0.)
            max_grad = dict.fromkeys(models, 0.)
            min_free = None
            batch_sizes = {}
            for task, indices in events:
                if task != "confidence":
                    continue
                batch = [pairs[int(i)] for i in indices]
                neg = [p["negative"] for p in batch]
                start = (epoch-1)*80451 + sum(k*v for k,v in batch_sizes.items())
                for coverage in COVERAGES:
                    pos = [pools[coverage][int(i)] for i in streams[coverage][start:start+len(neg)]]
                    rows = pos + neg
                    features, native, mask = _stack_rows(rows, device)
                    selected = torch.tensor([selected_indices[r["sample_id"]] for r in rows],
                                            device=device, dtype=torch.int64)
                    correct = torch.tensor([labels[r["sample_id"]] for r in pos], device=device, dtype=torch.bool)
                    for target in ("exists", "emit"):
                        arm = f"{coverage}__{target}"
                        loss, grad = train_step(models[arm], opts[arm], features, native, mask, selected, correct,
                                                target=target, readout="global_max", positive_count=len(pos))
                        loss_sums[arm] += loss
                        max_grad[arm] = max(max_grad[arm], grad)
                updates += 1
                batch_sizes[len(pos)] = batch_sizes.get(len(pos), 0) + 1
                if device.type == "cuda":
                    free = torch.cuda.mem_get_info(device)[0]
                    min_free = free if min_free is None else min(min_free, free)
                if updates % 100 == 0:
                    print(f"[COVERAGE] seed={args.seed} localizer={args.localizer} epoch={epoch} "
                          f"updates={updates}/12575 elapsed={time.monotonic()-started:.1f}s", flush=True)
            audit = {arm: parameter_audit(models[arm], opts[arm], initial_rank) for arm in models}
            if batch_sizes != {32: 2514, 3: 1} or updates != epoch * 2515:
                raise ValueError("formal final-batch or update count drift")
            for optimizer in opts.values():
                if len(optimizer.state) != 8 or {int(s["step"]) for s in optimizer.state.values()} != {updates}:
                    raise ValueError("optimizer update counter drift")
            history.append({"epoch": epoch, "schedule": schedule, "loss_sums": loss_sums,
                            "max_preclip_grad_norm": max_grad, "pair_batch_sizes": batch_sizes,
                            "updates": updates, "ownership": audit, "amp_skips": 0, "nonfinite": 0,
                            "minimum_free_device_bytes": min_free,
                            "maximum_allocated_device_bytes": torch.cuda.max_memory_allocated(device)
                            if device.type == "cuda" else None})
            ckpath = root / f"checkpoint_epoch{epoch}.pt"
            if ckpath.exists():
                recovery = root / "uncommitted"
                recovery.mkdir(exist_ok=True)
                destination = recovery / (ckpath.name + "." + file_sha256(ckpath))
                if destination.exists():
                    raise ValueError("uncommitted checkpoint recovery collision")
                ckpath.rename(destination)
            _atomic_torch(ckpath, {"schema": "arrow.confidence_coverage.head_checkpoint/v1",
                "design": lock_binding, "seed": args.seed, "localizer": args.localizer,
                "epoch": epoch, "updates": updates, "history": history, "initial_hashes": initial_hashes,
                "models": {arm: cpu_state(model) for arm, model in models.items()},
                "optimizers": {arm: optimizer.state_dict() for arm, optimizer in opts.items()},
                "rng": rng_state(device)})
            _atomic_json(root / f"epoch{epoch}.json", {"checkpoint": bind(ckpath), "ownership": audit})
        if updates != 12575 or len(history) != 5:
            raise ValueError("formal endpoint incomplete")
        endpoint = {arm: parameter_audit(models[arm], opts[arm], initial_rank) for arm in models}
    for artifact in (*design["code"].values(), *caches.values(), trunk, design["study_protocol"]):
        verify(artifact)
    _atomic_json(root / "postflight.json", {"schema": "arrow.confidence_coverage.training_postflight/v1",
        "status": "complete", "design": lock_binding, "checkpoint": bind(root / "checkpoint_epoch5.pt"),
        "initial_state": bind(initialpath), "initial_hashes": initial_hashes, "seed": args.seed,
        "localizer": args.localizer, "arms": list(models), "updates_per_head": updates,
        "epochs": 5, "history": history, "ownership": endpoint,
        "initialization_factory": "tools.finecops_fixed_rank_targets.make_heads",
        "schedule_factory": "legacy negative schedule unchanged; positive stream cycles independent",
        "no_optimizer_skips": True, "no_amp": True, "frozen_native_rank": "bitwise unchanged",
        "validation_model_evaluations": 0, "test_forwards": 0, "gref_forwards": 0})
    print(f"[COVERAGE] COMPLETE seed={args.seed} localizer={args.localizer}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-protocol", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--localizer", choices=LOCALIZERS, required=True)
    parser.add_argument("--seed", type=int, choices=(17, 42, 73), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
