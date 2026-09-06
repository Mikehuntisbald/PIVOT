"""Coverage-only positive streams; immutable v6 heads, loss and negative order."""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np
import torch

from tools.confidence_readout import GLOBAL, MMGDINO
from tools.finecops_fixed_rank_targets import make_heads
from tools.train_finecops_bce_l2_heads import (
    CACHE_ROW_SCHEMA, CACHE_SHARD_SCHEMA, _epoch_schedule, _load_manifest, file_sha256,
)

COVERAGES = ("l1_uniform", "all_uniform")
ARMS = tuple(f"{c}__{t}" for c in COVERAGES for t in ("exists", "emit"))
SEEDS = (17, 42, 73)
PAIR_COUNT = 80451
TOTAL = 5 * PAIR_COUNT
SCHEMA = "arrow.confidence_coverage.study_protocol/v1"


def training_arms(localizer):
    if localizer != MMGDINO:
        raise ValueError("coverage intervention fixes the MM-GDINO positive trunk")
    return ARMS


def parse_arm(arm):
    if arm not in ARMS:
        raise ValueError("unknown coverage arm")
    return GLOBAL, arm.split("__")[1]


def make_readout_heads(seed, device, localizer):
    training_arms(localizer)
    originals = make_heads(seed, device)
    return {arm: copy.deepcopy(originals[parse_arm(arm)[1]]) for arm in ARMS}


def positive_stream(pool_size, draws, seed):
    """Independent RNG; balanced cycles, never an epoch-prefix resampling bias."""
    if type(pool_size) is not int or pool_size < 1 or type(draws) is not int or draws < 1:
        raise ValueError("positive integer pool and draw counts required")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20260912, seed])))
    chunks = [rng.permutation(pool_size) for _ in range((draws + pool_size - 1)//pool_size)]
    return np.concatenate(chunks)[:draws]


def make_streams(positive, seed):
    ids = [r["sample_id"] for r in positive]
    if len(set(ids)) != len(ids) or any(r["kind"] != "positive" for r in positive):
        raise ValueError("unique positive training rows required")
    if Counter(r["level"] for r in positive) != {1: 54015, 2: 25282, 3: 4044}:
        raise ValueError("positive difficulty population drift")
    pools = {c: sorted((r for r in positive if c == "all_uniform" or r["level"] == 1),
                       key=lambda r: r["sample_id"]) for c in COVERAGES}
    streams = {c: positive_stream(len(pool), TOTAL, seed) for c, pool in pools.items()}
    return pools, streams


def epoch_batches(seed, epoch):
    if seed not in SEEDS or epoch not in range(1, 6):
        raise ValueError("fixed seed/epoch required")
    events, receipt = _epoch_schedule(seed=seed, epoch=epoch, rank_count=83341,
                                      confidence_count=PAIR_COUNT)
    return [indices for task, indices in events if task == "confidence"], receipt


def load_readout_cache(path, *, split, localizer):
    """Read-only mmap keeps the three workers' 75GB feature pages shareable.

    Tensor bytes/schema/order are unchanged. No fallback to an unverified cache.
    The legacy loader remains unmodified; tests compare both loading paths.
    """
    if localizer != MMGDINO or split not in ("train", "val"):
        raise ValueError("only the frozen MM train/val cache is allowed")
    path = Path(path).resolve(strict=True)
    manifest = _load_manifest(path, split=split)
    rows = []
    for i, item in enumerate(manifest["shards"]):
        shard_path = Path(item["path"])
        if file_sha256(shard_path) != item["sha256"]:
            raise ValueError("cache shard SHA drift")
        shard = torch.load(shard_path, map_location="cpu", weights_only=False, mmap=True)
        if (shard.get("schema") != CACHE_SHARD_SCHEMA or shard.get("split") != split
                or shard.get("start") != len(rows) or len(shard.get("rows", [])) != item["rows"]):
            raise ValueError("cache shard identity/order drift")
        for row in shard["rows"]:
            if (row.get("schema") != CACHE_ROW_SCHEMA
                    or row["query_features"].shape != (900, 256)
                    or row["query_features"].dtype != torch.float16
                    or row["native_score"].shape != (900,)
                    or row["native_score"].dtype != torch.float32
                    or row["boxes"].shape != (900, 4)
                    or row["candidate_mask"].dtype != torch.bool
                    or row["candidate_mask"].shape != (900,)):
                raise ValueError("frozen cache tensor contract drift")
            rows.append(row)
        if i % 100 == 0:
            print(f"[COVERAGE] mmap {split} shard={i+1}/{len(manifest['shards'])}", flush=True)
    if len(rows) != manifest["records"]:
        raise ValueError("cache record count drift")
    return rows, manifest
