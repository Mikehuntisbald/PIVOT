#!/usr/bin/env python3
"""Train the capacity-controlled MM-GDINO e5 ownership transfer heads.

The detector never executes here.  A versioned frozen-candidate cache and an
explicit batched schedule are the only model/data inputs.  Shared arms use two
independent task-specific AdamW optimizer states over the same trunk; the
isolated arm uses two optimizers over disjoint trunks.  Weight decay is fixed
to zero in every formal arm, removing repeated-decay exposure as a confound.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.GroundingDINO.stage_b_gdino_score_adapter import (
    baseline_preserving_top1_rank_loss,
    detached_recent_q05_trust_surrogate,
)
from tools.mmgdino_e5_ownership import (
    MMGDinoE5ResponsibilityOwners,
    OWNERSHIP_ISOLATED_128,
    OWNERSHIP_MODES,
    task_gradient_connection_report,
)
from tools.responsibility_isolation_cache import (
    CACHE_TASK_CONFIDENCE_PAIR,
    CACHE_TASK_RANK,
    file_sha256,
    load_cached_candidate_shard,
    normalized_cxcywh_iou,
)


FORMAL_SEEDS = (17, 42, 73)
FORMAL_UPDATES = 150
FORMAL_RANK_UPDATES = 100
FORMAL_CONFIDENCE_UPDATES = 50
FORMAL_RANK_BATCH_SIZE = 32
FORMAL_CONFIDENCE_BATCH_SIZE = 8
FORMAL_MILESTONES = (25, 50, 100, 150)
FORMAL_PROBE_BATCHES = 8

SCHEDULE_SCHEMA = "arrow.mmgdino_e5_ownership.schedule/v1"
CHECKPOINT_SCHEMA = "arrow.mmgdino_e5_ownership.checkpoint/v1"
RECEIPT_SCHEMA = "arrow.mmgdino_e5_ownership.training_receipt/v1"
TRAINING_CONTRACT = "arrow.mmgdino_e5_ownership.capacity_controlled/v1"
RANK_OBJECTIVE = "baseline_preserving_top1_residual/v1"
REJECTION_OBJECTIVE = "detached_recent_q05_total_trust/v1"


class FormalOwnershipError(RuntimeError):
    """Raised when the formal ownership experiment drifts or becomes invalid."""


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _tensor_mapping_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        header = json.dumps(
            [name, str(value.dtype), list(value.shape)], separators=(",", ":")
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = value.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_torch_save(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(dict(value), temporary_name)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _require_identifier(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise FormalOwnershipError(f"{name} must be a trimmed nonempty string")
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FormalOwnershipError(f"{name} must be a lowercase SHA-256")
    return value


def validate_schedule(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the explicit U150 batch schedule."""
    if not isinstance(value, Mapping):
        raise FormalOwnershipError("schedule must be a mapping")
    expected_fields = {
        "schema",
        "seed",
        "source",
        "rank_batch_size",
        "confidence_batch_size",
        "updates",
    }
    if set(value) != expected_fields:
        raise FormalOwnershipError(
            "schedule fields drifted: "
            f"missing={sorted(expected_fields - set(value))}, "
            f"extra={sorted(set(value) - expected_fields)}"
        )
    if value["schema"] != SCHEDULE_SCHEMA:
        raise FormalOwnershipError("schedule schema drifted")
    seed = value["seed"]
    if seed not in FORMAL_SEEDS:
        raise FormalOwnershipError(f"schedule seed must be one of {FORMAL_SEEDS}")
    if value["rank_batch_size"] != FORMAL_RANK_BATCH_SIZE:
        raise FormalOwnershipError("rank batch size must be 32")
    if value["confidence_batch_size"] != FORMAL_CONFIDENCE_BATCH_SIZE:
        raise FormalOwnershipError("confidence batch size must be 8")
    source = value["source"]
    if not isinstance(source, Mapping):
        raise FormalOwnershipError("schedule source must be a mapping")
    for name in ("rank_jsonl_sha256", "d3_jsonl_sha256"):
        _require_sha256(source.get(name), name=f"source.{name}")
    updates = value["updates"]
    if not isinstance(updates, Sequence) or isinstance(updates, (str, bytes)):
        raise FormalOwnershipError("schedule updates must be a sequence")
    if len(updates) != FORMAL_UPDATES:
        raise FormalOwnershipError("formal schedule must contain 150 updates")
    rank_count = 0
    confidence_count = 0
    for index, update in enumerate(updates, start=1):
        if not isinstance(update, Mapping) or set(update) != {
            "update",
            "task",
            "identities",
        }:
            raise FormalOwnershipError(f"schedule update {index} fields drifted")
        if update["update"] != index:
            raise FormalOwnershipError("schedule update indices must be consecutive")
        expected_task = (
            CACHE_TASK_CONFIDENCE_PAIR if (index - 1) % 3 == 1 else CACHE_TASK_RANK
        )
        if update["task"] != expected_task:
            raise FormalOwnershipError(
                f"schedule update {index} must be {expected_task!r}"
            )
        identities = update["identities"]
        expected_batch = (
            FORMAL_CONFIDENCE_BATCH_SIZE
            if expected_task == CACHE_TASK_CONFIDENCE_PAIR
            else FORMAL_RANK_BATCH_SIZE
        )
        if (
            not isinstance(identities, Sequence)
            or isinstance(identities, (str, bytes))
            or len(identities) != expected_batch
            or any(not isinstance(item, str) or not item for item in identities)
            or len(set(identities)) != len(identities)
        ):
            raise FormalOwnershipError(
                f"schedule update {index} identities must contain "
                f"{expected_batch} unique strings"
            )
        rank_count += int(expected_task == CACHE_TASK_RANK)
        confidence_count += int(expected_task == CACHE_TASK_CONFIDENCE_PAIR)
    if (rank_count, confidence_count) != (
        FORMAL_RANK_UPDATES,
        FORMAL_CONFIDENCE_UPDATES,
    ):
        raise FormalOwnershipError("formal schedule must be R100+C50")
    return dict(value)


def load_schedule(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path).resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    return validate_schedule(payload), file_sha256(source)


def _cache_indices(shard: Mapping[str, Any]) -> tuple[dict[str, Mapping], dict[str, dict[str, Mapping]]]:
    rank: dict[str, Mapping] = {}
    pairs: dict[str, dict[str, Mapping]] = {}
    for row in shard["rows"]:
        if row["task"] == CACHE_TASK_RANK:
            rank[row["sample_id"]] = row
        elif row["task"] == CACHE_TASK_CONFIDENCE_PAIR:
            pairs.setdefault(row["pair_id"], {})[row["pair_role"]] = row
    if any(set(roles) != {"positive", "negative"} for roles in pairs.values()):
        raise FormalOwnershipError("cache contains an incomplete confidence pair")
    return rank, pairs


def validate_schedule_cache_coverage(
    schedule: Mapping[str, Any], shard: Mapping[str, Any]
) -> None:
    rank, pairs = _cache_indices(shard)
    required_rank = {
        identity
        for update in schedule["updates"]
        if update["task"] == CACHE_TASK_RANK
        for identity in update["identities"]
    }
    required_pairs = {
        identity
        for update in schedule["updates"]
        if update["task"] == CACHE_TASK_CONFIDENCE_PAIR
        for identity in update["identities"]
    }
    if required_rank != set(rank):
        raise FormalOwnershipError(
            "rank cache must equal the scheduled identity set: "
            f"missing={len(required_rank - set(rank))}, "
            f"extra={len(set(rank) - required_rank)}"
        )
    if required_pairs != set(pairs):
        raise FormalOwnershipError(
            "confidence cache must equal the scheduled pair set: "
            f"missing={len(required_pairs - set(pairs))}, "
            f"extra={len(set(pairs) - required_pairs)}"
        )


def _stack_rows(rows: Sequence[Mapping[str, Any]], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.stack([row["query_features"] for row in rows]).to(device=device),
        torch.stack([row["native_score"] for row in rows]).to(device=device),
        torch.stack([row["candidate_mask"] for row in rows]).to(device=device),
    )


def _rank_loss(
    module: MMGDinoE5ResponsibilityOwners,
    rows: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> tuple[Tensor, dict[str, float]]:
    features, native, mask = _stack_rows(rows, device)
    output = module(features, native, mask)
    candidate_iou = torch.stack(
        [
            normalized_cxcywh_iou(row["boxes"], row["gt_boxes"]).amax(dim=1)
            for row in rows
        ]
    ).to(device=device)
    result = baseline_preserving_top1_rank_loss(
        output["rank_score"],
        output["native_score"],
        output["rank_residual"],
        candidate_iou,
        iou_threshold=0.5,
        fix_margin=0.05,
        preserve_margin=0.02,
        temperature=0.1,
        residual_weight=1e-3,
    )
    if not bool(torch.isfinite(result.loss).item()):
        raise FormalOwnershipError("rank objective became non-finite")
    return result.loss, {
        "loss": float(result.loss.detach()),
        "fix_loss": float(result.fix_loss.detach()),
        "preserve_loss": float(result.preserve_loss.detach()),
        "residual_loss": float(result.residual_loss.detach()),
        "base_correct": float(result.base_correct.detach()),
        "adapted_correct": float(result.adapted_correct.detach()),
        "wrong_fixed": float(result.wrong_fixed.detach()),
        "correct_regressed": float(result.correct_regressed.detach()),
    }


@dataclass
class D3QueueState:
    values: Tensor
    count: int = 0
    cursor: int = 0

    @classmethod
    def empty(cls, *, size: int = 512) -> "D3QueueState":
        return cls(torch.zeros(size, dtype=torch.float32), 0, 0)

    def history(self, *, min_count: int = 256, device: torch.device) -> Tensor | None:
        if self.count < min_count:
            return None
        if self.count < int(self.values.numel()):
            value = self.values[: self.count]
        else:
            value = torch.cat((self.values[self.cursor :], self.values[: self.cursor]))
        return value.detach().to(device=device)

    def append(self, values: Tensor) -> None:
        payload = values.detach().cpu().float().reshape(-1)
        if payload.numel() == 0 or not bool(torch.isfinite(payload).all().item()):
            raise FormalOwnershipError("D3 queue payload must be nonempty and finite")
        size = int(self.values.numel())
        for value in payload:
            self.values[self.cursor] = value
            self.cursor = (self.cursor + 1) % size
            self.count = min(size, self.count + 1)

    def state_dict(self) -> dict[str, Any]:
        return {
            "values": self.values.clone(),
            "count": self.count,
            "cursor": self.cursor,
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        values = value.get("values")
        count = value.get("count")
        cursor = value.get("cursor")
        if (
            not torch.is_tensor(values)
            or values.dtype != torch.float32
            or tuple(values.shape) != tuple(self.values.shape)
            or not isinstance(count, int)
            or not 0 <= count <= int(self.values.numel())
            or not isinstance(cursor, int)
            or not 0 <= cursor < int(self.values.numel())
        ):
            raise FormalOwnershipError("D3 queue checkpoint drifted")
        self.values.copy_(values.cpu())
        self.count = count
        self.cursor = cursor


def _confidence_loss(
    module: MMGDinoE5ResponsibilityOwners,
    pairs: Sequence[Mapping[str, Mapping[str, Any]]],
    queue: D3QueueState,
    *,
    device: torch.device,
) -> tuple[Tensor, dict[str, float], Tensor]:
    positive_rows = [pair["positive"] for pair in pairs]
    negative_rows = [pair["negative"] for pair in pairs]
    positive_inputs = _stack_rows(positive_rows, device)
    negative_inputs = _stack_rows(negative_rows, device)
    positive_output = module(*positive_inputs)
    negative_output = module(*negative_inputs)
    positive_max = positive_output["confidence_score"].masked_fill(
        ~positive_output["candidate_mask"], -torch.inf
    ).max(dim=1).values
    negative_max = negative_output["confidence_score"].masked_fill(
        ~negative_output["candidate_mask"], -torch.inf
    ).max(dim=1).values
    result = detached_recent_q05_trust_surrogate(
        positive_output["confidence_score"],
        negative_output["confidence_score"],
        positive_max,
        negative_max,
        positive_candidate_mask=positive_output["candidate_mask"],
        negative_candidate_mask=negative_output["candidate_mask"],
        positive_history=queue.history(device=device),
        temperature=0.1,
        margin=0.0,
        target_tpr=0.95,
        positive_trust_margin=0.02,
        positive_trust_weight=1.0,
        paired_margin_weight=0.0,
        paired_margin=0.0,
        positive_score_trust=True,
    )
    if not bool(torch.isfinite(result.loss).item()):
        raise FormalOwnershipError("D3 objective became non-finite")
    return result.loss, {
        "loss": float(result.loss.detach()),
        "negative_loss": float(result.negative_loss.detach()),
        "positive_trust_loss": float(result.positive_trust_loss.detach()),
        "positive_threshold": float(result.positive_threshold.detach()),
        "current_positive_threshold": float(
            result.current_positive_threshold.detach()
        ),
        "exact_fpr": float(result.exact_fpr.detach()),
        "current_exact_fpr": float(result.current_exact_fpr.detach()),
        "positive_mean": float(result.positive_global_score.detach().mean()),
        "negative_mean": float(result.negative_global_score.detach().mean()),
        "queue_count_before": float(queue.count),
    }, result.local_positive_global_score.detach()


def _unique_parameters(parameters: Iterable[torch.nn.Parameter]) -> tuple[torch.nn.Parameter, ...]:
    result: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for parameter in parameters:
        if id(parameter) not in seen:
            result.append(parameter)
            seen.add(id(parameter))
    return tuple(result)


def _optimizer_state_tensor_count(optimizer: torch.optim.Optimizer) -> int:
    return sum(
        int(torch.is_tensor(value))
        for state in optimizer.state.values()
        for value in state.values()
    )


@contextlib.contextmanager
def _deterministic_algorithms():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def _gradient_probe(
    module: MMGDinoE5ResponsibilityOwners,
    rank_batches: Sequence[Sequence[Mapping[str, Any]]],
    confidence_batches: Sequence[Sequence[Mapping[str, Mapping[str, Any]]]],
    queue: D3QueueState,
    *,
    device: torch.device,
) -> dict[str, Any]:
    if module.ownership == OWNERSHIP_ISOLATED_128:
        rank_rows = rank_batches[0]
        confidence_pairs = confidence_batches[0]
        rank_loss, _ = _rank_loss(module, rank_rows, device=device)
        confidence_loss, _, _ = _confidence_loss(
            module, confidence_pairs, queue, device=device
        )
        topology = task_gradient_connection_report(
            module, rank_loss, confidence_loss
        )
        return {
            "structurally_isolated": True,
            "cross_task_parameter_count": len(
                topology["cross_task_parameter_names"]
            ),
            "probe_count": 0,
            "cosine_mean": None,
            "sign_conflict_fraction_mean": None,
        }
    shared = module.shared_parameters()
    if not shared:
        raise FormalOwnershipError("shared probe has no shared parameters")
    cosines: list[float] = []
    sign_conflicts: list[float] = []
    for rank_rows, confidence_pairs in zip(rank_batches, confidence_batches):
        rank_loss, _ = _rank_loss(module, rank_rows, device=device)
        confidence_loss, _, _ = _confidence_loss(
            module, confidence_pairs, queue, device=device
        )
        rank_grad = torch.autograd.grad(
            rank_loss, shared, retain_graph=False, allow_unused=False
        )
        confidence_grad = torch.autograd.grad(
            confidence_loss, shared, retain_graph=False, allow_unused=False
        )
        rank_flat = torch.cat([value.detach().float().reshape(-1) for value in rank_grad])
        confidence_flat = torch.cat(
            [value.detach().float().reshape(-1) for value in confidence_grad]
        )
        rank_norm = torch.linalg.vector_norm(rank_flat)
        confidence_norm = torch.linalg.vector_norm(confidence_flat)
        if not bool((rank_norm > 0).item()) or not bool((confidence_norm > 0).item()):
            raise FormalOwnershipError("gradient probe encountered a zero task gradient")
        cosine = torch.dot(rank_flat, confidence_flat) / (
            rank_norm * confidence_norm
        )
        nonzero = (rank_flat != 0) & (confidence_flat != 0)
        if not bool(nonzero.any().item()):
            raise FormalOwnershipError("gradient probe has no jointly nonzero elements")
        conflict = (torch.sign(rank_flat[nonzero]) != torch.sign(confidence_flat[nonzero])).float().mean()
        cosines.append(float(cosine))
        sign_conflicts.append(float(conflict))
    return {
        "structurally_isolated": False,
        "cross_task_parameter_count": len(shared),
        "probe_count": len(cosines),
        "cosines": cosines,
        "sign_conflict_fractions": sign_conflicts,
        "cosine_mean": float(np.mean(cosines)),
        "sign_conflict_fraction_mean": float(np.mean(sign_conflicts)),
    }


@dataclass(frozen=True)
class FormalConfig:
    ownership: str
    seed: int
    device: str = "cuda"
    updates: int = FORMAL_UPDATES
    rank_learning_rate: float = 3e-5
    confidence_learning_rate: float = 1e-4
    weight_decay: float = 0.0
    clip_norm: float = 0.1

    def validate(self) -> "FormalConfig":
        if self.ownership not in OWNERSHIP_MODES:
            raise FormalOwnershipError(f"ownership must be one of {OWNERSHIP_MODES}")
        if self.seed not in FORMAL_SEEDS:
            raise FormalOwnershipError(f"seed must be one of {FORMAL_SEEDS}")
        if self.updates != FORMAL_UPDATES:
            raise FormalOwnershipError("formal runs must stop at U150")
        if self.device not in ("cpu", "cuda"):
            raise FormalOwnershipError("device must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise FormalOwnershipError("CUDA was requested but unavailable")
        if self.weight_decay != 0.0:
            raise FormalOwnershipError("formal owner weight_decay must be zero")
        for name, value in (
            ("rank_learning_rate", self.rank_learning_rate),
            ("confidence_learning_rate", self.confidence_learning_rate),
            ("clip_norm", self.clip_norm),
        ):
            if not math.isfinite(value) or value <= 0:
                raise FormalOwnershipError(f"{name} must be finite and positive")
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "ownership": self.ownership,
            "seed": self.seed,
            "device": self.device,
            "updates": self.updates,
            "rank_learning_rate": self.rank_learning_rate,
            "confidence_learning_rate": self.confidence_learning_rate,
            "weight_decay": self.weight_decay,
            "clip_norm": self.clip_norm,
            "rank_batch_size": FORMAL_RANK_BATCH_SIZE,
            "confidence_batch_size": FORMAL_CONFIDENCE_BATCH_SIZE,
            "rank_updates": FORMAL_RANK_UPDATES,
            "confidence_updates": FORMAL_CONFIDENCE_UPDATES,
            "training_contract": TRAINING_CONTRACT,
            "rank_objective": RANK_OBJECTIVE,
            "rejection_objective": REJECTION_OBJECTIVE,
        }


def _checkpoint_load(path: Path, device: torch.device) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def run_formal_training(
    *,
    cache_path: str | Path,
    schedule_path: str | Path,
    output_dir: str | Path,
    config: FormalConfig,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Run one fixed U150 arm/seed trajectory and return its final receipt."""
    if config.device == "cuda":
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace is None:
            # Must be set before the first owner GEMM.  CUDA availability
            # checks do not create a cuBLAS handle, so setting it here remains
            # early enough while keeping the CLI self-contained.
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        elif workspace not in (":4096:8", ":16:8"):
            raise FormalOwnershipError(
                "deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
                "or :16:8"
            )
    config = config.validate()
    cache_path = Path(cache_path).resolve(strict=True)
    schedule_path = Path(schedule_path).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "training_receipt.json"
    if receipt_path.exists():
        raise FormalOwnershipError("training receipt already exists")
    schedule, schedule_file_sha = load_schedule(schedule_path)
    if schedule["seed"] != config.seed:
        raise FormalOwnershipError("schedule seed and run seed differ")
    cache_file_before = file_sha256(cache_path)
    shard = load_cached_candidate_shard(cache_path)
    validate_schedule_cache_coverage(schedule, shard)
    rank_index, pair_index = _cache_indices(shard)
    device = torch.device(config.device)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    module = MMGDinoE5ResponsibilityOwners(
        ownership=config.ownership
    ).to(device=device)
    architecture = module.architecture_report().as_dict()
    rank_parameters = _unique_parameters(module.task_parameters("rank"))
    confidence_parameters = _unique_parameters(module.task_parameters("confidence"))
    rank_optimizer = torch.optim.AdamW(
        rank_parameters,
        lr=config.rank_learning_rate,
        weight_decay=0.0,
        foreach=False,
    )
    confidence_optimizer = torch.optim.AdamW(
        confidence_parameters,
        lr=config.confidence_learning_rate,
        weight_decay=0.0,
        foreach=False,
    )
    if any(group["weight_decay"] != 0.0 for optimizer in (rank_optimizer, confidence_optimizer) for group in optimizer.param_groups):
        raise FormalOwnershipError("optimizer weight decay drifted from zero")
    queue = D3QueueState.empty(size=512)
    completed = 0
    update_history: list[dict[str, Any]] = []
    gradient_probes: dict[str, Any] = {}
    resume_record = None

    if resume_from is not None:
        resume_path = Path(resume_from).resolve(strict=True)
        checkpoint = _checkpoint_load(resume_path, device)
        if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
            raise FormalOwnershipError("resume checkpoint schema drifted")
        if checkpoint.get("config") != config.as_dict():
            raise FormalOwnershipError("resume config differs from formal run")
        if checkpoint.get("cache_sha256") != cache_file_before:
            raise FormalOwnershipError("resume cache SHA drifted")
        if checkpoint.get("schedule_sha256") != schedule_file_sha:
            raise FormalOwnershipError("resume schedule SHA drifted")
        module.load_state_dict(checkpoint["model_state_dict"], strict=True)
        rank_optimizer.load_state_dict(checkpoint["rank_optimizer_state_dict"])
        confidence_optimizer.load_state_dict(
            checkpoint["confidence_optimizer_state_dict"]
        )
        queue.load_state_dict(checkpoint["d3_queue_state_dict"])
        completed = int(checkpoint["completed_updates"])
        update_history = list(checkpoint["update_history"])
        gradient_probes = dict(checkpoint["gradient_probes"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        np.random.set_state(checkpoint["numpy_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(
                [value.cpu() for value in checkpoint["cuda_rng_state_all"]]
            )
        resume_record = {
            "path": str(resume_path),
            "sha256": file_sha256(resume_path),
            "completed_updates": completed,
        }

    rank_probe_batches = [
        [rank_index[identity] for identity in update["identities"]]
        for update in schedule["updates"]
        if update["task"] == CACHE_TASK_RANK
    ][:FORMAL_PROBE_BATCHES]
    confidence_probe_batches = [
        [pair_index[identity] for identity in update["identities"]]
        for update in schedule["updates"]
        if update["task"] == CACHE_TASK_CONFIDENCE_PAIR
    ][:FORMAL_PROBE_BATCHES]

    module.train()
    with _deterministic_algorithms():
        for update in schedule["updates"][completed:]:
            module.zero_grad(set_to_none=True)
            if update["task"] == CACHE_TASK_RANK:
                rows = [rank_index[identity] for identity in update["identities"]]
                loss, metrics = _rank_loss(module, rows, device=device)
                optimizer = rank_optimizer
                active_parameters = rank_parameters
            else:
                pairs = [pair_index[identity] for identity in update["identities"]]
                loss, metrics, positive_payload = _confidence_loss(
                    module, pairs, queue, device=device
                )
                optimizer = confidence_optimizer
                active_parameters = confidence_parameters
            loss.backward()
            gradients = [value.grad for value in active_parameters]
            if not gradients or any(value is None for value in gradients):
                raise FormalOwnershipError(
                    f"active optimizer lost a gradient at update {update['update']}"
                )
            if not all(bool(torch.isfinite(value).all().item()) for value in gradients):
                raise FormalOwnershipError("non-finite task gradient")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                active_parameters, config.clip_norm
            )
            if not bool(torch.isfinite(gradient_norm).item()):
                raise FormalOwnershipError("non-finite clipped gradient norm")
            optimizer.step()
            if update["task"] == CACHE_TASK_CONFIDENCE_PAIR:
                queue.append(positive_payload)
                metrics["queue_count_after"] = float(queue.count)
            completed = int(update["update"])
            update_history.append(
                {
                    "update": completed,
                    "task": update["task"],
                    "batch_identity_sha256": _canonical_json_sha256(
                        update["identities"]
                    ),
                    "gradient_norm_before_clip": float(gradient_norm.detach()),
                    **metrics,
                }
            )
            if completed in FORMAL_MILESTONES:
                module.zero_grad(set_to_none=True)
                gradient_probes[str(completed)] = _gradient_probe(
                    module,
                    rank_probe_batches,
                    confidence_probe_batches,
                    queue,
                    device=device,
                )
                module.zero_grad(set_to_none=True)
                checkpoint_path = output_dir / f"checkpoint_u{completed:03d}.pt"
                if checkpoint_path.exists():
                    raise FormalOwnershipError(
                        f"milestone checkpoint already exists: {checkpoint_path}"
                    )
                checkpoint_payload = {
                    "schema": CHECKPOINT_SCHEMA,
                    "config": config.as_dict(),
                    "architecture": architecture,
                    "cache_sha256": cache_file_before,
                    "schedule_sha256": schedule_file_sha,
                    "completed_updates": completed,
                    "update_history": update_history,
                    "gradient_probes": gradient_probes,
                    "model_state_dict": module.state_dict(),
                    "rank_optimizer_state_dict": rank_optimizer.state_dict(),
                    "confidence_optimizer_state_dict": confidence_optimizer.state_dict(),
                    "d3_queue_state_dict": queue.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "numpy_rng_state": np.random.get_state(),
                    "python_rng_state": random.getstate(),
                    "cuda_rng_state_all": (
                        torch.cuda.get_rng_state_all()
                        if device.type == "cuda"
                        else []
                    ),
                }
                _atomic_torch_save(checkpoint_payload, checkpoint_path)

    if completed != FORMAL_UPDATES or len(update_history) != FORMAL_UPDATES:
        raise FormalOwnershipError("formal run did not complete U150")
    cache_file_after = file_sha256(cache_path)
    if cache_file_after != cache_file_before:
        raise FormalOwnershipError("frozen cache changed during training")
    final_checkpoint = output_dir / "checkpoint_u150.pt"
    if not final_checkpoint.is_file():
        raise FormalOwnershipError("final milestone checkpoint is missing")
    final_payload = _checkpoint_load(final_checkpoint, torch.device("cpu"))
    model_sha = _tensor_mapping_sha256(final_payload["model_state_dict"])
    topology = {
        "shared_parameter_count": architecture["shared_parameter_count"],
        "structurally_isolated": config.ownership == OWNERSHIP_ISOLATED_128,
        "rank_parameter_names": [
            name for name, _ in module.named_task_parameters("rank")
        ],
        "confidence_parameter_names": [
            name for name, _ in module.named_task_parameters("confidence")
        ],
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "complete",
        "config": config.as_dict(),
        "architecture": architecture,
        "inputs": {
            "cache": {"path": str(cache_path), "sha256": cache_file_before},
            "schedule": {
                "path": str(schedule_path),
                "sha256": schedule_file_sha,
                "semantic_sha256": _canonical_json_sha256(schedule),
            },
        },
        "updates": {
            "total": completed,
            "rank": FORMAL_RANK_UPDATES,
            "confidence": FORMAL_CONFIDENCE_UPDATES,
            "nonfinite": 0,
            "amp_skips": 0,
        },
        "optimizers": {
            "task_specific_states": True,
            "rank_learning_rate": config.rank_learning_rate,
            "confidence_learning_rate": config.confidence_learning_rate,
            "weight_decay": 0.0,
            "rank_state_tensor_count": _optimizer_state_tensor_count(
                rank_optimizer
            ),
            "confidence_state_tensor_count": _optimizer_state_tensor_count(
                confidence_optimizer
            ),
        },
        "ownership": topology,
        "d3_queue": {
            "size": int(queue.values.numel()),
            "count": queue.count,
            "cursor": queue.cursor,
            "warmup_min_count": 256,
        },
        "gradient_probes": gradient_probes,
        "checkpoint": {
            "path": str(final_checkpoint),
            "sha256": file_sha256(final_checkpoint),
            "model_state_sha256": model_sha,
            "resume_from": resume_record,
        },
        "cache_unchanged": True,
        "runtime": {
            "torch_version": torch.__version__,
            "device": str(device),
            "deterministic_algorithms": True,
            "cublas_workspace_config": (
                os.environ.get("CUBLAS_WORKSPACE_CONFIG")
                if device.type == "cuda"
                else None
            ),
        },
    }
    _atomic_json(receipt, receipt_path)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ownership", choices=OWNERSHIP_MODES, required=True)
    parser.add_argument("--seed", choices=FORMAL_SEEDS, type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume-from", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    receipt = run_formal_training(
        cache_path=args.cache,
        schedule_path=args.schedule,
        output_dir=args.output_dir,
        config=FormalConfig(
            ownership=args.ownership,
            seed=args.seed,
            device=args.device,
        ),
        resume_from=args.resume_from,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "CHECKPOINT_SCHEMA",
    "D3QueueState",
    "FORMAL_CONFIDENCE_BATCH_SIZE",
    "FORMAL_CONFIDENCE_UPDATES",
    "FORMAL_MILESTONES",
    "FORMAL_RANK_BATCH_SIZE",
    "FORMAL_RANK_UPDATES",
    "FORMAL_SEEDS",
    "FORMAL_UPDATES",
    "FormalConfig",
    "FormalOwnershipError",
    "RECEIPT_SCHEMA",
    "REJECTION_OBJECTIVE",
    "RANK_OBJECTIVE",
    "SCHEDULE_SCHEMA",
    "load_schedule",
    "run_formal_training",
    "validate_schedule",
    "validate_schedule_cache_coverage",
]
