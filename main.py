#CUDA_VISIBLE_DEVICES=0 python main.py -c config/cfg_patch_stage_a_emb.py --datasets config/datasets_patch_stage_a_raw_local.json --output_dir outputs/stageA_emb --pretrain_model_path weights/groundingdino_swint_ogc.pth --num_workers 8 --amp# --resume outputs/stageA_emb/checkpoint.pth
# CUDA_VISIBLE_DEVICES=0 python main.py -c config/cfg_patch_stage_a.py --datasets config/datasets_patch_stage_a_coco2017_local.json --output_dir outputs/stageA_patch --pretrain_model_path weights/groundingdino_swint_ogc.pth --num_workers 8 --amp --resume outputs/stageA_patch/checkpoint.pth 
# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import csv
import datetime
import hashlib
import json
import math
import pickle
import random
import signal
import time
from pathlib import Path
from typing import Any, Mapping, Optional
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from util.get_param_dicts import get_param_dict, match_name_keywords
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import  BestMetricHolder
import util.misc as utils

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import GracefulTrainingExit, evaluate, train_one_epoch

from groundingdino.util.utils import clean_state_dict


class WeightedDistributedSampler(torch.utils.data.Sampler):
    """Distributed weighted sampling with equal per-rank iteration counts."""

    def __init__(
        self,
        weights,
        *,
        num_replicas: int,
        rank: int,
        replacement: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank={rank} is invalid for num_replicas={num_replicas}")
        weights = torch.as_tensor(weights, dtype=torch.double)
        if weights.numel() <= 0:
            raise ValueError("weights must be non-empty")
        if not torch.isfinite(weights).all() or float(weights.sum().item()) <= 0.0:
            raise ValueError("weights must be finite and have positive sum")

        self.weights = weights
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.epoch = 0
        if drop_last and len(self.weights) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.weights) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.weights) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            self.replacement,
            generator=g,
        ).tolist()
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


STAGE_B_DATA_DRIVEN_SAMPLING_CONTRACT = "deterministic_epoch_ledger_v1"
_STAGE_B_DATA_DRIVEN_MAX_SEED = 2**63 - 1


def _stage_b_data_driven_epoch_seed(base_seed: int, epoch: int) -> int:
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise ValueError("data-driven sampling seeds must be exact integers")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("data-driven sampling epoch must be a non-negative integer")
    if not 0 <= base_seed < _STAGE_B_DATA_DRIVEN_MAX_SEED:
        raise ValueError(
            "data-driven sampling seeds must be in [0, 2**63 - 1)"
        )
    return int((base_seed + epoch) % _STAGE_B_DATA_DRIVEN_MAX_SEED)


class DeterministicEpochSampler(torch.utils.data.Sampler):
    """Materialize one model-RNG-independent sample-index ledger per epoch."""

    def __init__(
        self,
        dataset_size: int,
        *,
        seed: int,
        weights=None,
        num_samples: Optional[int] = None,
        replacement: bool = False,
    ):
        if isinstance(dataset_size, bool) or not isinstance(dataset_size, int):
            raise ValueError("dataset_size must be an exact integer")
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        _stage_b_data_driven_epoch_seed(seed, 0)
        self.dataset_size = int(dataset_size)
        self.seed = seed
        if num_samples is not None and (
            isinstance(num_samples, bool) or not isinstance(num_samples, int)
        ):
            raise ValueError("num_samples must be an exact integer")
        self.num_samples = self.dataset_size if num_samples is None else num_samples
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.replacement = bool(replacement)
        self.weights = None
        self.weights_sha256 = None
        if weights is not None:
            normalized = torch.as_tensor(weights, dtype=torch.double).detach().cpu()
            if normalized.dim() != 1 or int(normalized.numel()) != self.dataset_size:
                raise ValueError("sampling weights must match the dataset length")
            if not bool(torch.isfinite(normalized).all().item()) or bool(
                (normalized < 0).any().item()
            ) or float(normalized.sum().item()) <= 0.0:
                raise ValueError(
                    "sampling weights must be finite, non-negative, and have positive sum"
                )
            self.weights = normalized.contiguous().clone()
            weights_bytes = self.weights.numpy().astype("<f8", copy=False).tobytes()
            self.weights_sha256 = hashlib.sha256(weights_bytes).hexdigest()
        if not self.replacement and self.num_samples > self.dataset_size:
            raise ValueError(
                "sampling without replacement cannot exceed the dataset length"
            )
        self.epoch = 0
        self._cached_epoch = None
        self._cached_ledger = None

    def _ledger_for_epoch(self, epoch: int):
        epoch = int(epoch)
        if self._cached_epoch == epoch and self._cached_ledger is not None:
            return self._cached_ledger
        generator = torch.Generator()
        generator.manual_seed(_stage_b_data_driven_epoch_seed(self.seed, epoch))
        if self.weights is not None:
            indices = torch.multinomial(
                self.weights,
                self.num_samples,
                self.replacement,
                generator=generator,
            )
        elif self.replacement:
            indices = torch.randint(
                self.dataset_size,
                (self.num_samples,),
                generator=generator,
            )
        else:
            indices = torch.randperm(
                self.dataset_size, generator=generator
            )[: self.num_samples]
        ledger = tuple(int(index) for index in indices.tolist())
        self._cached_epoch = epoch
        self._cached_ledger = ledger
        return ledger

    def __iter__(self):
        return iter(self._ledger_for_epoch(self.epoch))

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        _stage_b_data_driven_epoch_seed(self.seed, epoch)
        self.epoch = epoch

    def ledger_state(self, epoch: Optional[int] = None) -> dict[str, Any]:
        ledger_epoch = self.epoch if epoch is None else epoch
        _stage_b_data_driven_epoch_seed(self.seed, ledger_epoch)
        ledger = self._ledger_for_epoch(ledger_epoch)
        ledger_bytes = np.asarray(ledger, dtype="<i8").tobytes()
        return {
            "schema": STAGE_B_DATA_DRIVEN_SAMPLING_CONTRACT,
            "epoch": ledger_epoch,
            "sampler_seed": self.seed,
            "sampler_epoch_seed": _stage_b_data_driven_epoch_seed(
                self.seed, ledger_epoch
            ),
            "dataset_size": self.dataset_size,
            "num_samples": self.num_samples,
            "replacement": self.replacement,
            "weighted": self.weights is not None,
            "weights_sha256": self.weights_sha256,
            "ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        }


def _prepare_stage_b_data_driven_epoch_sampling(
    sampler: DeterministicEpochSampler,
    loader_generator: torch.Generator,
    *,
    epoch: int,
    sampler_seed: int,
    loader_seed: int,
) -> dict[str, Any]:
    if not isinstance(sampler, DeterministicEpochSampler):
        raise RuntimeError(
            "deterministic_epoch_ledger_v1 requires DeterministicEpochSampler"
        )
    if sampler.seed != sampler_seed:
        raise RuntimeError("data-driven sampler seed drifted after construction")
    if not isinstance(loader_generator, torch.Generator):
        raise RuntimeError(
            "deterministic_epoch_ledger_v1 requires a dedicated DataLoader generator"
        )
    _stage_b_data_driven_epoch_seed(sampler_seed, epoch)
    loader_epoch_seed = _stage_b_data_driven_epoch_seed(loader_seed, epoch)
    sampler.set_epoch(epoch)
    loader_generator.manual_seed(loader_epoch_seed)

    # This also covers num_workers=0, where dataset transforms execute in the
    # training process instead of worker-local RNG streams.
    random.seed(loader_epoch_seed)
    np.random.seed(loader_epoch_seed % (2**32))
    torch.manual_seed(loader_epoch_seed)

    state = sampler.ledger_state(epoch)
    state.update(
        {
            "loader_seed": loader_seed,
            "loader_epoch_seed": loader_epoch_seed,
            "persistent_workers": False,
        }
    )
    return state


def _stage_b_data_driven_epoch_checkpoint_due(
    args, completed_optimizer_updates: int
) -> bool:
    interval = getattr(args, "stage_b_data_driven_epoch_checkpoint_interval", 1)
    if (
        isinstance(interval, bool)
        or not isinstance(interval, int)
        or interval <= 0
    ):
        raise ValueError(
            "stage_b_data_driven_epoch_checkpoint_interval must be a positive integer"
        )
    if (
        isinstance(completed_optimizer_updates, bool)
        or not isinstance(completed_optimizer_updates, int)
        or completed_optimizer_updates < 0
    ):
        raise ValueError("completed optimizer updates must be a non-negative integer")
    return (
        completed_optimizer_updates > 0
        and completed_optimizer_updates % interval == 0
    )


def _validate_stage_b_data_driven_sampling_resume_state(
    sampler: DeterministicEpochSampler,
    checkpoint: Mapping[str, Any],
    *,
    loader_seed: int,
) -> dict[str, Any]:
    observed = checkpoint.get("stage_b_data_driven_sampling_state")
    if not isinstance(observed, Mapping):
        raise RuntimeError(
            "deterministic data-driven resume checkpoint is missing its sampling state"
        )
    checkpoint_epoch = checkpoint.get("epoch")
    if isinstance(checkpoint_epoch, bool) or not isinstance(checkpoint_epoch, int):
        raise RuntimeError("deterministic data-driven checkpoint epoch is invalid")
    expected = sampler.ledger_state(checkpoint_epoch)
    expected.update(
        {
            "loader_seed": loader_seed,
            "loader_epoch_seed": _stage_b_data_driven_epoch_seed(
                loader_seed, checkpoint_epoch
            ),
            "persistent_workers": False,
        }
    )
    if dict(observed) != expected:
        raise RuntimeError(
            "deterministic data-driven sampling ledger drifted across resume: "
            f"checkpoint={dict(observed)}, expected={expected}"
        )
    return expected


def _validate_stage_b_data_driven_eval_update_gate(
    args,
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_label: str,
) -> Optional[int]:
    expected = getattr(
        args, "stage_b_data_driven_eval_expected_optimizer_updates", 0
    )
    if isinstance(expected, bool) or not isinstance(expected, int):
        raise ValueError(
            "stage_b_data_driven_eval_expected_optimizer_updates must be an exact integer"
        )
    if expected < 0:
        raise ValueError(
            "stage_b_data_driven_eval_expected_optimizer_updates cannot be negative"
        )
    if expected == 0:
        return None
    if not bool(getattr(args, "eval", False)):
        raise RuntimeError(
            "the data-driven exact-update gate is valid only for evaluation"
        )
    observed = checkpoint.get("optimizer_updates")
    if isinstance(observed, bool) or not isinstance(observed, int) or observed != expected:
        raise RuntimeError(
            f"{checkpoint_label}: expected exactly {expected} successful optimizer "
            f"updates, observed {observed!r}"
        )
    if checkpoint.get("checkpoint_reason") != "max_train_iters":
        raise RuntimeError(
            f"{checkpoint_label}: exact-update evaluation requires a "
            "max_train_iters terminal checkpoint"
        )
    saved_args = checkpoint.get("args")
    if not isinstance(saved_args, Mapping) or saved_args.get("max_train_iters") != expected:
        raise RuntimeError(
            f"{checkpoint_label}: saved max_train_iters does not bind the exact "
            f"evaluation target {expected}"
        )
    return expected


def _torch_load_compat(path: str, *, map_location: str = "cpu"):
    """
    PyTorch >= 2.6 defaults `torch.load(..., weights_only=True)`, which can fail on older
    training checkpoints that include non-tensor objects (e.g. argparse.Namespace).
    """
    import torch as _torch

    try:
        return _torch.load(path, map_location=map_location)
    except Exception as e:
        msg = str(e)
        if "Weights only load failed" not in msg and "weights_only" not in msg:
            raise
        # Allowlist argparse.Namespace for safe weights-only loading (our checkpoints store `args`).
        try:
            from torch import serialization as _serialization  # type: ignore

            _serialization.add_safe_globals([argparse.Namespace])  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            return _torch.load(path, map_location=map_location)
        except Exception:
            # Fall back to full unpickling (unsafe for untrusted files).
            return _torch.load(path, map_location=map_location, weights_only=False)


def _make_grad_scaler(enabled: bool, init_scale: Optional[float] = None):
    scaler_kwargs = {"enabled": enabled}
    if init_scale is not None and float(init_scale) > 0:
        scaler_kwargs["init_scale"] = float(init_scale)
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "GradScaler"):
        try:
            return amp_mod.GradScaler("cuda", **scaler_kwargs)
        except TypeError:
            try:
                return amp_mod.GradScaler(device_type="cuda", **scaler_kwargs)
            except TypeError:
                pass
    return torch.cuda.amp.GradScaler(**scaler_kwargs)


def _maybe_sync_stage_b_v7_verifier_from_text_branch(model, state_dict, logger) -> None:
    verifier = getattr(model, "stage_b_verifier", None)
    if verifier is None or not hasattr(verifier, "load_from_text_branch"):
        return
    if any(str(k).startswith("stage_b_verifier.") for k in state_dict.keys()):
        return
    verifier.load_from_text_branch(model)
    verifier.freeze_bert()
    if logger is not None:
        logger.info("Initialized stage_b_verifier text branch from loaded GroundingDINO text branch.")


def _maybe_sync_stage_b_v11_scorer_from_decoder(model, state_dict, logger) -> None:
    scorer = getattr(model, "stage_b_fixed_text_scorer", None)
    if scorer is None:
        return
    if any(
        str(key).startswith("stage_b_fixed_text_scorer.")
        for key in state_dict.keys()
    ):
        return
    if hasattr(scorer, "load_from_groundingdino"):
        scorer.load_from_groundingdino(model)
        source_description = "complete GroundingDINO text/image transformer"
    elif hasattr(scorer, "load_from_decoder"):
        scorer.load_from_decoder(model.transformer.decoder)
        source_description = "GroundingDINO decoder"
    else:
        return
    if logger is not None:
        logger.info(
            "Initialized stage_b_fixed_text_scorer from the loaded frozen "
            f"{source_description}."
        )


_STAGE_B_V15_SCORER_INIT_AUDIT_SCHEMA = "stage_b_v15_scorer_init/v1"
_STAGE_B_DATA_DRIVEN_PROVENANCE_SCHEMA = (
    "pivot.stageb.data_driven_training_provenance/v1"
)
_STAGE_B_DATA_DRIVEN_DD1H_FORMAL_SCOPE = "formal_fresh_a1_u5020_v1"
_STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_SCOPE = (
    "formal_fresh_a0_new_head_3epoch_v1"
)
_STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_CONTRACT = (
    "sealed_new_head_d0_d1_3epoch_v1"
)
_STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_BINDING_SCHEMA = (
    "pivot.stageb.data_driven.new_head_formal_runtime_binding/v1"
)
_STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_EXPECTED_UPDATES = 12357
_STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_TRAIN_ROWS = 263661
_STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_SOURCE_MANIFESTS = (
    "refcoco_stageb_phrase_v1.jsonl",
    "refcocoplus_stageb_phrase_v1.jsonl",
    "refcocog_stageb_phrase_v1.jsonl",
)
_STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_VARIANTS = {
    "DD0": {
        "variant_id": "DD0-NEWHEAD-DATA",
        "category_complete": False,
        "dataset_variant": "dd0_ordinary_primary",
        "receipt_variant": "d0_ordinary_primary",
        "manifest_sha256": (
            "d992677bb297248d4cb516ef25e8d34686462428c846a949ca5f6423734b5f32",
            "e108f4b80001b0de171e40a973f74945989917561aede11aa55b370af4b1cab1",
            "bb3b9bcc2bd4e546b5ec18d1ed2cf792bf041851fe87f502b0188ed045254082",
        ),
    },
    "DD1": {
        "variant_id": "DD1-NEWHEAD-DATA",
        "category_complete": True,
        "dataset_variant": "dd1_category_complete",
        "receipt_variant": "d1_category_complete",
        "manifest_sha256": (
            "defa48cd85659c689734ba717e94baf0416c5391c77df80a7fc4cbc8f1202cc4",
            "7b5f4540ba565e692417a6f15ed2460f4a856e35521e284d4b5db6056b03332c",
            "b45ee9494b05a57aa13a0fce075f3fdbda7ab580cce46fa90bce31fee6a10dfa",
        ),
    },
}
_STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_A0 = {
    "path": (
        "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
        "data_driven_initializers/fair_v2_seed42/"
        "checkpoint_dd_a0_absolute_v2_init.pth"
    ),
    "sha256": "c2c4ba71656054d3afc3d219ca2f6d56839396d6258bac0201878566b1937034",
}
_STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_PAIR_RECEIPT = {
    "path": (
        "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
        "data_driven_initializers/fair_v2_seed42/a0_a1_v2_pair_receipt.json"
    ),
    "sha256": "e304d2e8439f5714facf1b510795ba3a9874ec456433110afeef91f2d1dc7d8d",
}
_STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_PARTITION_RECEIPT = {
    "path": (
        "/media/haoyi/T9/pivot/data/ablations/"
        "stageb_data_driven_new_head_partition_20260723/receipt.json"
    ),
    "sha256": "56de31d883ed137f3f9332c34de846839d82c0724120f42a49c5c1c302f38506",
    "schema": "pivot.stageb.data_driven.new_head_partition_receipt/v1",
    "canonical_payload_sha256": (
        "351126ffe7f1a1f99b5085c2533126b6c91d13056bd329adc294d98716190270"
    ),
}
_STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_SUPPORT_RECEIPT = {
    "path": (
        "/media/haoyi/T9/pivot/data/ablations/"
        "stageb_data_driven_support_partition_20260723/receipt.json"
    ),
    "sha256": "a0e6632182bc7c01ac6e6997b15f1f96e0fbb0bf6dd9d1e3fd8485ad39a6da62",
    "schema": "pivot.stageb.data_driven.support_partition_receipt/v1",
    "canonical_payload_sha256": (
        "e7e10de38eb0eaf2a08fb3943834c3283751e356a8fb132602c4b30d8b96db3c"
    ),
}
_STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_RUNTIME_SUPPORT = {
    "path": (
        "/media/haoyi/T9/pivot/data/ablations/"
        "stageb_data_driven_support_partition_20260723/filtered_support.tsv"
    ),
    "sha256": "a3c7dc02e1159ebac5196ccb2c53da1e1bd7e2c2b0322159efcf4178a53a1d37",
    "size_bytes": 35880333,
    "rows": 158599,
    "class_count": 2018,
    "required_training_classes": 78,
}
_STAGE_B_DATA_DRIVEN_NEW_HEAD_LR_SELECTION_RECEIPT_PATH = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_data_driven_new_head_lr_selection_20260723/selection_receipt.json"
)
_STAGE_B_DATA_DRIVEN_NEW_HEAD_LR_SELECTION_RECEIPT_SCHEMA = (
    "pivot.stageb.data_driven.new_head_lr_selection_receipt/v1"
)
_STAGE_B_DATA_DRIVEN_NEW_HEAD_LR_CANDIDATES = [3e-5, 1e-4, 3e-4]
_STAGE_B_DATA_DRIVEN_PAIRTOP1_VARIANT = "DD1-PairTop1"
_STAGE_B_DATA_DRIVEN_PAIRTOP1_HARDGAP3_VARIANT = (
    "DD1-PairTop1-HardGap3"
)
_STAGE_B_DATA_DRIVEN_ASSIGNMENT_VARIANTS = {
    _STAGE_B_DATA_DRIVEN_PAIRTOP1_VARIANT: 0.0,
    _STAGE_B_DATA_DRIVEN_PAIRTOP1_HARDGAP3_VARIANT: 1.0,
}
_STAGE_B_DATA_DRIVEN_ASSIGNMENT_FULL_SCOPE = (
    "official_assignment_full_321327_v1"
)
_STAGE_B_DATA_DRIVEN_ASSIGNMENT_OVERFIT_SCOPE = (
    "official_assignment_overfit64_u500_v1"
)
_STAGE_B_DATA_DRIVEN_ROLE_ROUTED_VARIANT = "DD1-RoleRouted-Clean"
_STAGE_B_DATA_DRIVEN_ROLE_ROUTED_SCOPE = (
    "official_assignment_clean_train_263661_v1"
)
_STAGE_B_DATA_DRIVEN_ROLE_ROUTED_INITIALIZER_RECEIPT_SCHEMA = (
    "pivot.stageb.data_driven.role_routed_initializer_receipt/v1"
)
_STAGE_B_DATA_DRIVEN_PATCH_RESIDUAL_INITIALIZER_RECEIPT_SCHEMA = (
    "pivot.stageb.data_driven.patch_residual_initializer_receipt/v1"
)
_STAGE_B_DATA_DRIVEN_PATCH_TOPK_INITIALIZER_RECEIPT_SCHEMA = (
    "pivot.stageb.data_driven.patch_residual_initializer_receipt/v2"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_b_data_driven_file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"data-driven provenance input is not a file: {resolved}")
    before = resolved.stat()
    digest = _sha256_file(resolved)
    after = resolved.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise RuntimeError(
            f"data-driven provenance input changed while hashing: {resolved}"
        )
    return {
        "path": str(resolved),
        "size_bytes": int(after.st_size),
        "sha256": digest,
    }


def _stage_b_data_driven_code_paths() -> list[Path]:
    root = Path(__file__).resolve().parent
    paths = {root / "main.py", root / "engine.py"}
    for relative_root in ("models", "datasets", "util", "groundingdino"):
        paths.update((root / relative_root).rglob("*.py"))
        paths.update((root / relative_root).rglob("*.so"))
    return sorted(path.resolve(strict=True) for path in paths if path.is_file())


def _expand_stage_b_data_driven_path(value: str) -> Path:
    media_user = os.environ.get("MEDIA_USER", "haoyi")
    t9_root = os.environ.get("T9_ROOT", f"/media/{media_user}/T9")
    replacements = {
        "MEDIA_USER": media_user,
        "T9_ROOT": t9_root,
        "DATA_ROOT": os.environ.get("DATA_ROOT", f"{t9_root}/data"),
        "GDINO_ROOT": os.environ.get("GDINO_ROOT", f"{t9_root}/gdino"),
    }
    expanded = str(value)
    for key, replacement in replacements.items():
        expanded = expanded.replace(f"${{{key}}}", replacement)
        expanded = expanded.replace(f"${key}", replacement)
    return Path(os.path.expandvars(os.path.expanduser(expanded)))


def _stage_b_data_driven_dataset_asset_paths(dataset_path: Path) -> list[Path]:
    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"could not parse data-driven dataset config {dataset_path}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise RuntimeError("data-driven dataset config must be a JSON mapping")
    rows = payload.get("train")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("data-driven dataset config requires non-empty train rows")
    paths: set[Path] = set()
    new_head_support_bindings: set[tuple[Path, str, Path]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise RuntimeError(
                f"data-driven dataset train row {index} must be a mapping"
            )
        for key in (
            "anno",
            "stage_b_data_driven_receipt",
            "stage_b_data_driven_support_receipt",
            "canonical_classes_json",
            "support_patch_tsv",
            "support_patch_class_map_json",
        ):
            value = str(row.get(key, "") or "").strip()
            if value:
                paths.add(
                    _expand_stage_b_data_driven_path(value).resolve(strict=True)
                )
        is_new_head_row = bool(
            row.get("stage_b_data_driven_partition") == "train"
            and (
                row.get("stage_b_data_driven_variant")
                in {"dd0_ordinary_primary", "dd1_category_complete"}
                or (
                    row.get("stage_b_data_driven_variant")
                    == "dd1_official_assignment_pair"
                    and bool(
                        str(
                            row.get(
                                "stage_b_data_driven_support_receipt", ""
                            )
                            or ""
                        ).strip()
                    )
                )
            )
        )
        if is_new_head_row:
            support_receipt_value = str(
                row.get("stage_b_data_driven_support_receipt", "") or ""
            ).strip()
            support_receipt_sha = str(
                row.get("stage_b_data_driven_support_receipt_sha256", "") or ""
            ).strip()
            support_tsv_value = str(
                row.get("support_patch_tsv", "") or ""
            ).strip()
            if not all(
                (support_receipt_value, support_receipt_sha, support_tsv_value)
            ):
                raise RuntimeError(
                    "new-head dataset row has incomplete support provenance: "
                    f"index={index}"
                )
            if str(row.get("patch_bank_cache_path", "") or "").strip():
                raise RuntimeError(
                    "new-head dataset row must not retain a support cache path: "
                    f"index={index}"
                )
            new_head_support_bindings.add(
                (
                    _expand_stage_b_data_driven_path(
                        support_receipt_value
                    ).resolve(strict=True),
                    support_receipt_sha,
                    _expand_stage_b_data_driven_path(
                        support_tsv_value
                    ).resolve(strict=True),
                )
            )
        support_tsv = str(row.get("support_patch_tsv", "") or "").strip()
        if support_tsv and bool(row.get("patch_bank_cache", True)):
            cache_value = str(row.get("patch_bank_cache_path", "") or "").strip()
            if cache_value:
                cache_path = _expand_stage_b_data_driven_path(cache_value)
            else:
                tsv_path = _expand_stage_b_data_driven_path(support_tsv)
                bucket = str(row.get("support_patch_bucket", "") or "all")
                mode = (
                    "emb"
                    if bool(row.get("support_patch_use_embedding", False))
                    else "img"
                )
                cache_path = Path(f"{tsv_path}.bank.{bucket}.{mode}.pkl")
            paths.add(cache_path.resolve(strict=True))
    if len(new_head_support_bindings) > 1:
        raise RuntimeError(
            "new-head dataset train rows do not share one support receipt/TSV"
        )
    return sorted(paths)


def _stage_b_data_driven_support_pool_content_records(
    asset_paths: list[Path],
) -> list[dict[str, Any]]:
    cache_paths = sorted(
        path for path in asset_paths
        if ".bank." in path.name and path.suffix == ".pkl"
    )
    records = []
    for cache_path in cache_paths:
        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
        except (OSError, pickle.PickleError) as error:
            raise RuntimeError(
                f"could not load support-pool cache {cache_path}: {error}"
            ) from error
        bank = payload.get("bank") if isinstance(payload, Mapping) else None
        if not isinstance(bank, Mapping) or not bank:
            raise RuntimeError(f"support-pool cache has no bank: {cache_path}")
        normalized_keys = []
        for raw_key in bank:
            try:
                normalized_keys.append((int(raw_key), raw_key))
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"support-pool cache has a non-integer class key: {raw_key!r}"
                ) from error
        if len({key for key, _raw in normalized_keys}) != len(normalized_keys):
            raise RuntimeError("support-pool cache has duplicate normalized classes")
        digest = hashlib.sha256()
        file_count = 0
        total_size = 0
        for class_id, raw_key in sorted(normalized_keys):
            candidates = bank[raw_key]
            if not isinstance(candidates, list) or not candidates:
                raise RuntimeError(
                    f"support-pool class {class_id} has no ordered candidates"
                )
            for candidate_index, raw_path in enumerate(candidates):
                if not isinstance(raw_path, str) or not raw_path:
                    raise RuntimeError(
                        f"support-pool class {class_id} has an invalid candidate path"
                    )
                image_path = Path(raw_path).expanduser().resolve(strict=True)
                if not image_path.is_file():
                    raise RuntimeError(
                        f"support-pool candidate is not a file: {image_path}"
                    )
                before = image_path.stat()
                image_sha = _sha256_file(image_path)
                after = image_path.stat()
                before_identity = (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                )
                after_identity = (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                )
                if before_identity != after_identity:
                    raise RuntimeError(
                        f"support-pool image changed while hashing: {image_path}"
                    )
                header = json.dumps(
                    [
                        int(class_id),
                        int(candidate_index),
                        str(image_path),
                        int(after.st_size),
                        image_sha,
                    ],
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("ascii")
                digest.update(len(header).to_bytes(8, "little"))
                digest.update(header)
                file_count += 1
                total_size += int(after.st_size)
        records.append(
            {
                "cache_path": str(cache_path.resolve(strict=True)),
                "class_count": len(normalized_keys),
                "file_count": file_count,
                "total_size_bytes": total_size,
                "ordered_content_sha256": digest.hexdigest(),
            }
        )
    if not records:
        tsv_paths = sorted(path for path in asset_paths if path.suffix == ".tsv")
        if len(tsv_paths) != 1:
            raise RuntimeError(
                "data-driven direct support requires exactly one TSV when cache is disabled"
            )
        tsv_path = tsv_paths[0]
        bank: dict[int, list[str]] = {}
        with tsv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames or not {"class_id", "path"}.issubset(
                reader.fieldnames
            ):
                raise RuntimeError(
                    f"direct support TSV has no class_id/path columns: {tsv_path}"
                )
            for row_index, row in enumerate(reader, start=2):
                try:
                    class_id = int(row["class_id"])
                except (TypeError, ValueError) as error:
                    raise RuntimeError(
                        f"direct support TSV row {row_index} has invalid class_id"
                    ) from error
                raw_path = str(row.get("path", "") or "").strip()
                if not raw_path:
                    raise RuntimeError(
                        f"direct support TSV row {row_index} has no image path"
                    )
                image_path = Path(raw_path)
                if not image_path.is_absolute():
                    image_path = tsv_path.parent / image_path
                bank.setdefault(class_id, []).append(
                    str(image_path.expanduser().resolve(strict=True))
                )
        digest = hashlib.sha256()
        file_count = 0
        total_size = 0
        for class_id, candidates in sorted(bank.items()):
            if not candidates:
                raise RuntimeError(f"direct support class {class_id} is empty")
            for candidate_index, raw_path in enumerate(candidates):
                image_path = Path(raw_path).resolve(strict=True)
                before = image_path.stat()
                image_sha = _sha256_file(image_path)
                after = image_path.stat()
                before_identity = (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                )
                after_identity = (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                )
                if before_identity != after_identity:
                    raise RuntimeError(
                        f"direct support image changed while hashing: {image_path}"
                    )
                header = json.dumps(
                    [
                        class_id,
                        candidate_index,
                        str(image_path),
                        int(after.st_size),
                        image_sha,
                    ],
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("ascii")
                digest.update(len(header).to_bytes(8, "little"))
                digest.update(header)
                file_count += 1
                total_size += int(after.st_size)
        records.append(
            {
                "support_tsv_path": str(tsv_path.resolve(strict=True)),
                "class_count": len(bank),
                "file_count": file_count,
                "total_size_bytes": total_size,
                "ordered_content_sha256": digest.hexdigest(),
            }
        )
    if not records:
        raise RuntimeError("data-driven provenance found no support-pool cache")
    return records


def _stage_b_data_driven_software_record() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cudnn": torch.backends.cudnn.version(),
    }


def _build_stage_b_data_driven_training_provenance(
    args, *, dataset_path: Path
) -> dict[str, Any]:
    required_allocator_env = str(
        getattr(args, "stage_b_data_driven_required_allocator_env", "") or ""
    ).strip()
    required_allocator_conf = str(
        getattr(args, "stage_b_data_driven_required_allocator_conf", "") or ""
    ).strip()
    if required_allocator_env or required_allocator_conf:
        if not required_allocator_env or not required_allocator_conf:
            raise RuntimeError(
                "data-driven allocator contract requires both env name and value"
            )
        observed = os.environ.get(required_allocator_env)
        if not bool(getattr(args, "eval", False)) and observed != required_allocator_conf:
            raise RuntimeError(
                "data-driven training allocator contract drifted: "
                f"{required_allocator_env}={observed!r}, "
                f"expected={required_allocator_conf!r}"
            )
    image_roots = set()
    dataset_payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    for row in dataset_payload.get("train", []):
        if not isinstance(row, Mapping):
            continue
        for key in ("support_patch_image_root", "coco_image_root", "lvis_image_root"):
            value = str(row.get(key, "") or "").strip()
            if value:
                image_roots.add(str(_expand_stage_b_data_driven_path(value).resolve()))
    asset_paths = _stage_b_data_driven_dataset_asset_paths(dataset_path)
    return {
        "schema": _STAGE_B_DATA_DRIVEN_PROVENANCE_SCHEMA,
        "code_files": [
            _stage_b_data_driven_file_record(path)
            for path in _stage_b_data_driven_code_paths()
        ],
        "dataset_asset_files": [
            _stage_b_data_driven_file_record(path)
            for path in asset_paths
        ],
        "support_patch_pool_content": (
            _stage_b_data_driven_support_pool_content_records(asset_paths)
        ),
        "allocator_environment": {
            name: os.environ.get(name)
            for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
        },
        "required_allocator": {
            "environment_variable": required_allocator_env,
            "value": required_allocator_conf,
        },
        "software": _stage_b_data_driven_software_record(),
        "unhashed_image_roots": sorted(image_roots),
        "image_content_note": (
            "Annotation/support-bank manifests and every cached support candidate "
            "image are content-hashed; source-image trees are path-bound only."
        ),
    }


def _validate_stage_b_data_driven_confidence_handoff_provenance(
    args, initializer_contract: Mapping[str, Any]
) -> None:
    source = initializer_contract.get("source_training_provenance")
    current = getattr(args, "stage_b_data_driven_training_provenance", None)
    if not isinstance(source, Mapping) or not isinstance(current, Mapping):
        raise RuntimeError("data-driven confidence handoff has no provenance")
    exact_shared_fields = (
        "schema",
        "code_files",
        "support_patch_pool_content",
        "allocator_environment",
        "required_allocator",
        "software",
    )
    drift = {
        key: (source.get(key), current.get(key))
        for key in exact_shared_fields
        if source.get(key) != current.get(key)
    }
    if drift:
        raise RuntimeError(
            "data-driven confidence handoff crossed code/support/runtime "
            f"provenance: {sorted(drift)}"
        )
    source_assets = {
        record.get("path"): dict(record)
        for record in source.get("dataset_asset_files", [])
        if isinstance(record, Mapping) and isinstance(record.get("path"), str)
    }
    current_assets = {
        record.get("path"): dict(record)
        for record in current.get("dataset_asset_files", [])
        if isinstance(record, Mapping) and isinstance(record.get("path"), str)
    }
    shared_suffixes = (
        "canonical_classes_with_aliases.json",
        "emb_index_from_quality.tsv",
        "emb_index_from_quality.tsv.bank.clean.img.pkl",
    )
    for suffix in shared_suffixes:
        matches = [
            path for path in source_assets
            if str(path).endswith(suffix)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"DD1 provenance does not bind one shared asset ending {suffix!r}"
            )
        path = matches[0]
        if current_assets.get(path) != source_assets[path]:
            raise RuntimeError(
                f"confidence phase shared asset drifted: {path}"
            )


def _validate_stage_b_data_driven_dd1h_fresh_training_contract(
    args,
    *,
    config_path: Path,
    base_path: Path,
    variant_id: str = "DD1-H",
) -> None:
    label = str(variant_id)
    if bool(getattr(args, "eval", False)):
        return
    if str(getattr(args, "resume", "") or "").strip():
        raise RuntimeError(
            f"{label} training is fresh-start only; --resume is forbidden, including "
            "resume from a memory probe"
        )
    pretrain_value = str(getattr(args, "pretrain_model_path", "") or "").strip()
    if not pretrain_value:
        raise RuntimeError(
            f"{label} training requires the A1 initializer via --pretrain_model_path"
        )
    pretrain_path = Path(pretrain_value).expanduser().resolve(strict=True)
    if pretrain_path != base_path:
        raise RuntimeError(
            f"{label} --pretrain_model_path differs from its canonical A1 initializer"
        )

    scope = str(
        getattr(args, "stage_b_data_driven_execution_scope", "") or ""
    ).strip()
    max_train_iters = getattr(args, "max_train_iters", None)
    if max_train_iters == 5020 and scope != _STAGE_B_DATA_DRIVEN_DD1H_FORMAL_SCOPE:
        raise RuntimeError(
            f"{label} U5020 must use the sealed fresh-start formal config"
        )
    if not scope:
        return
    if scope != _STAGE_B_DATA_DRIVEN_DD1H_FORMAL_SCOPE:
        raise RuntimeError(f"unknown {label} execution scope: {scope!r}")
    if getattr(args, "stage_b_data_driven_formal_fresh_start", None) is not True:
        raise RuntimeError(
            f"{label} formal run must declare exact fresh-start=True"
        )
    expected_updates = getattr(
        args, "stage_b_data_driven_formal_expected_optimizer_updates", None
    )
    if (
        not isinstance(expected_updates, int)
        or isinstance(expected_updates, bool)
        or expected_updates != 5020
        or max_train_iters != expected_updates
    ):
        raise RuntimeError(
            f"{label} formal run requires exactly max_train_iters=5020"
        )
    expected_config_value = str(
        getattr(args, "stage_b_data_driven_formal_config_path", "") or ""
    ).strip()
    expected_output_value = str(
        getattr(args, "stage_b_data_driven_formal_output_dir", "") or ""
    ).strip()
    if not expected_config_value or not expected_output_value:
        raise RuntimeError(f"{label} formal config/output binding is incomplete")
    if config_path != Path(expected_config_value).expanduser().resolve(strict=True):
        raise RuntimeError(f"{label} formal training config path drifted")
    output_path = Path(str(args.output_dir)).expanduser().resolve()
    if output_path != Path(expected_output_value).expanduser().resolve():
        raise RuntimeError(f"{label} formal output directory drifted")

    required_runtime = {
        "seed": 42,
        "batch_size": 64,
        "epochs": 1,
        "max_train_iters": 5020,
        "iter_checkpoint_interval": 500,
        "num_workers": 4,
        "prefetch_factor": 1,
        "gradient_accumulation_steps": 1,
        "amp": True,
        "save_log": True,
        "world_size": 1,
        "distributed": False,
    }
    drifted = {
        key: (getattr(args, key, None), expected)
        for key, expected in required_runtime.items()
        if getattr(args, key, None) != expected
    }
    if drifted:
        raise RuntimeError(
            f"{label} formal runtime differs from the predeclared fair-v2 run: "
            f"{drifted}"
        )


def _validate_stage_b_data_driven_new_head_formal_training_contract(
    args,
    *,
    config_path: Path,
    dataset_path: Path,
    base_path: Path,
    observed_base_sha: str,
    pair_path: Optional[Path],
    observed_pair_sha: Optional[str],
) -> Optional[dict[str, Any]]:
    scope = str(
        getattr(args, "stage_b_data_driven_execution_scope", "") or ""
    ).strip()
    if scope != _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_SCOPE:
        return None

    def _matches_exact_type(observed: Any, expected: Any) -> bool:
        return type(observed) is type(expected) and observed == expected

    def _load_json_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"could not parse new-head formal {label}: {error}") from error
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"new-head formal {label} must be a JSON mapping")
        return payload

    experiment_id = str(
        getattr(args, "stage_b_data_driven_experiment_id", "") or ""
    ).strip()
    variant_contract = _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_VARIANTS.get(
        experiment_id
    )
    if variant_contract is None:
        raise RuntimeError(
            "new-head formal training requires experiment_id DD0 or DD1"
        )

    required_runtime = {
        "eval": False,
        "stage_b_data_driven_score": True,
        "stage_b_data_driven_variant_id": variant_contract["variant_id"],
        "stage_b_data_driven_category_complete": variant_contract[
            "category_complete"
        ],
        "stage_b_data_driven_new_head_formal_contract": (
            _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_CONTRACT
        ),
        "stage_b_data_driven_formal_fresh_start": True,
        "stage_b_data_driven_formal_expected_optimizer_updates": (
            _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_EXPECTED_UPDATES
        ),
        "stage_b_data_driven_train_mode": "rank_patch_only",
        "stage_b_data_driven_confidence_trained": False,
        "stage_b_data_driven_rank_architecture": "absolute_token",
        "stage_b_data_driven_rank_supervision": "all_nonpositive_negative_v1",
        "stage_b_data_driven_strict_sample_identity": True,
        "stage_b_data_driven_rank_weight": 1.0,
        "stage_b_data_driven_patch_weight": 1.0,
        "stage_b_data_driven_patch_lr": 3e-4,
        "stage_b_data_driven_sampling_contract": (
            STAGE_B_DATA_DRIVEN_SAMPLING_CONTRACT
        ),
        "stage_b_data_driven_sampler_seed": 42,
        "stage_b_data_driven_loader_seed": 1042,
        "stage_b_data_driven_grad_clip_contract": "per_optimizer_branch_v1",
        "stage_b_data_driven_required_allocator_env": (
            "PYTORCH_CUDA_ALLOC_CONF"
        ),
        "stage_b_data_driven_required_allocator_conf": (
            "expandable_segments:True"
        ),
        "stage_b": False,
        "stage_b_gdino_score_adapter": False,
        "stage_b_u0_patch_rank": False,
        "stage_b_v7": False,
        "stage_b_v11_fixed_text": False,
        "stage_b_legacy_global_gate": False,
        "patch_only": False,
        "enable_patch_branch": True,
        "patch_gate_with_text": False,
        "stage_b_data_driven_category_gate": False,
        "seed": 42,
        "batch_size": 64,
        "epochs": 3,
        "gradient_accumulation_steps": 1,
        "amp": True,
        "amp_init_scale": 8192.0,
        "fix_size": True,
        "strong_aug": False,
        "data_aug_hflip_prob": 0.0,
        "aux_loss": False,
        "use_checkpoint": False,
        "use_transformer_ckpt": False,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "clip_max_norm": 0.1,
        "lr_drop": 100,
        "onecyclelr": False,
        "multi_step_lr": False,
        "save_checkpoint_interval": 1,
        "skip_eval": True,
        "use_coco_eval": False,
        "max_train_iters": 0,
        "iter_checkpoint_interval": 0,
        "num_workers": 4,
        "prefetch_factor": 1,
        "pin_memory": True,
        "persistent_workers": False,
        "save_log": True,
        "world_size": 1,
        "distributed": False,
    }
    runtime_drift = {
        key: (getattr(args, key, None), expected)
        for key, expected in required_runtime.items()
        if not _matches_exact_type(getattr(args, key, None), expected)
    }
    if runtime_drift:
        raise RuntimeError(
            "new-head formal runtime/model contract drifted: "
            f"{runtime_drift}"
        )

    if str(getattr(args, "resume", "") or "").strip():
        raise RuntimeError(
            "new-head formal training is fresh-start only; --resume is forbidden"
        )
    pretrain_value = str(
        getattr(args, "pretrain_model_path", "") or ""
    ).strip()
    if not pretrain_value:
        raise RuntimeError(
            "new-head formal training requires A0 via --pretrain_model_path"
        )

    config_path = config_path.expanduser().resolve(strict=True)
    dataset_path = dataset_path.expanduser().resolve(strict=True)
    base_path = base_path.expanduser().resolve(strict=True)
    expected_config_value = str(
        getattr(args, "stage_b_data_driven_formal_config_path", "") or ""
    ).strip()
    expected_dataset_value = str(
        getattr(
            args, "stage_b_data_driven_new_head_dataset_config_path", ""
        )
        or ""
    ).strip()
    expected_dataset_sha = str(
        getattr(
            args, "stage_b_data_driven_new_head_dataset_config_sha256", ""
        )
        or ""
    ).strip()
    expected_output_value = str(
        getattr(args, "stage_b_data_driven_formal_output_dir", "") or ""
    ).strip()
    if not all(
        (expected_config_value, expected_dataset_value, expected_output_value)
    ) or not _is_stage_b_data_driven_sha256(expected_dataset_sha):
        raise RuntimeError(
            "new-head formal config/dataset/output binding is incomplete"
        )
    if config_path != _expand_stage_b_data_driven_path(
        expected_config_value
    ).resolve(strict=True):
        raise RuntimeError("new-head formal training config path drifted")
    if dataset_path != _expand_stage_b_data_driven_path(
        expected_dataset_value
    ).resolve(strict=True):
        raise RuntimeError("new-head formal dataset config path drifted")
    observed_dataset_sha = _sha256_file(dataset_path)
    if observed_dataset_sha != expected_dataset_sha:
        raise RuntimeError("new-head formal dataset config SHA drifted")
    output_path = Path(str(getattr(args, "output_dir", "") or "")).expanduser().resolve()
    expected_output_path = _expand_stage_b_data_driven_path(
        expected_output_value
    ).resolve()
    if output_path != expected_output_path:
        raise RuntimeError("new-head formal output directory drifted")

    expected_a0_path = Path(
        _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_A0["path"]
    ).resolve(strict=True)
    if not (
        base_path == expected_a0_path
        and observed_base_sha
        == _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_A0["sha256"]
        and str(
            getattr(args, "stage_b_data_driven_base_initializer_sha256", "")
            or ""
        ).strip()
        == _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_A0["sha256"]
        and Path(
            str(
                getattr(args, "stage_b_data_driven_base_initializer_path", "")
                or ""
            )
        ).expanduser().resolve(strict=True)
        == expected_a0_path
        and Path(pretrain_value).expanduser().resolve(strict=True)
        == expected_a0_path
    ):
        raise RuntimeError("new-head formal A0 initializer binding drifted")

    expected_pair_path = Path(
        _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_PAIR_RECEIPT["path"]
    ).resolve(strict=True)
    pair_value = str(
        getattr(args, "stage_b_data_driven_initializer_pair_receipt_path", "")
        or ""
    ).strip()
    pair_sha = str(
        getattr(args, "stage_b_data_driven_initializer_pair_receipt_sha256", "")
        or ""
    ).strip()
    if not (
        pair_path is not None
        and pair_path.expanduser().resolve(strict=True) == expected_pair_path
        and observed_pair_sha
        == _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_PAIR_RECEIPT["sha256"]
        and pair_value
        and Path(pair_value).expanduser().resolve(strict=True)
        == expected_pair_path
        and pair_sha
        == _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_PAIR_RECEIPT["sha256"]
    ):
        raise RuntimeError("new-head formal A0/A1 pair receipt binding drifted")

    lr_selection_value = str(
        getattr(
            args,
            "stage_b_data_driven_new_head_lr_selection_receipt_path",
            "",
        )
        or ""
    ).strip()
    lr_selection_sha = str(
        getattr(
            args,
            "stage_b_data_driven_new_head_lr_selection_receipt_sha256",
            "",
        )
        or ""
    ).strip()
    if not lr_selection_value or not _is_stage_b_data_driven_sha256(
        lr_selection_sha
    ):
        raise RuntimeError(
            "new-head formal LR selection receipt binding is incomplete"
        )
    lr_selection_path = _expand_stage_b_data_driven_path(
        lr_selection_value
    ).resolve(strict=True)
    expected_lr_selection_path = Path(
        _STAGE_B_DATA_DRIVEN_NEW_HEAD_LR_SELECTION_RECEIPT_PATH
    ).resolve(strict=True)
    if not (
        lr_selection_path == expected_lr_selection_path
        and _sha256_file(lr_selection_path) == lr_selection_sha
    ):
        raise RuntimeError("new-head formal LR selection receipt binding drifted")
    lr_selection_receipt = _load_json_mapping(
        lr_selection_path, label="LR selection receipt"
    )
    selected_rank_lr = lr_selection_receipt.get("selected_rank_lr")
    observed_rank_lr = getattr(args, "stage_b_data_driven_rank_lr", None)
    if not (
        lr_selection_receipt.get("schema")
        == _STAGE_B_DATA_DRIVEN_NEW_HEAD_LR_SELECTION_RECEIPT_SCHEMA
        and lr_selection_receipt.get("status") == "passed"
        and lr_selection_receipt.get("candidate_rank_lrs")
        == _STAGE_B_DATA_DRIVEN_NEW_HEAD_LR_CANDIDATES
        and lr_selection_receipt.get("optimizer_updates_per_candidate") == 1000
        and lr_selection_receipt.get("selection_partition") == "dev_screen"
        and lr_selection_receipt.get("selection_metric")
        == "macro_ref3_acc50"
        and lr_selection_receipt.get("secondary_selection_metric")
        == "macro_ref3_mean_listwise_nll"
        and type(selected_rank_lr) is float
        and selected_rank_lr in _STAGE_B_DATA_DRIVEN_NEW_HEAD_LR_CANDIDATES
        and type(observed_rank_lr) is float
        and observed_rank_lr in _STAGE_B_DATA_DRIVEN_NEW_HEAD_LR_CANDIDATES
        and selected_rank_lr == observed_rank_lr
    ):
        raise RuntimeError("new-head formal LR selection receipt semantics drifted")

    def _resolve_sealed_receipt(
        *, path_key: str, sha_key: str, sealed: Mapping[str, Any], label: str
    ) -> Path:
        value = str(getattr(args, path_key, "") or "").strip()
        configured_sha = str(getattr(args, sha_key, "") or "").strip()
        if not value or configured_sha != sealed["sha256"]:
            raise RuntimeError(
                f"new-head formal {label} receipt binding is incomplete or drifted"
            )
        resolved = _expand_stage_b_data_driven_path(value).resolve(strict=True)
        expected = Path(str(sealed["path"])).resolve(strict=True)
        if resolved != expected or _sha256_file(resolved) != sealed["sha256"]:
            raise RuntimeError(f"new-head formal {label} receipt binding drifted")
        return resolved

    partition_path = _resolve_sealed_receipt(
        path_key="stage_b_data_driven_new_head_partition_receipt_path",
        sha_key="stage_b_data_driven_new_head_partition_receipt_sha256",
        sealed=_STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_PARTITION_RECEIPT,
        label="partition",
    )
    support_path = _resolve_sealed_receipt(
        path_key="stage_b_data_driven_new_head_support_receipt_path",
        sha_key="stage_b_data_driven_new_head_support_receipt_sha256",
        sealed=_STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_SUPPORT_RECEIPT,
        label="support",
    )
    partition_receipt = _load_json_mapping(
        partition_path, label="partition receipt"
    )
    support_receipt = _load_json_mapping(support_path, label="support receipt")

    partition_train = (
        partition_receipt.get("partition_summary", {}).get("train")
        if isinstance(partition_receipt.get("partition_summary"), Mapping)
        else None
    )
    partition_invariants = partition_receipt.get("invariants")
    if not (
        partition_receipt.get("schema")
        == _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_PARTITION_RECEIPT["schema"]
        and partition_receipt.get("canonical_payload_sha256")
        == _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_PARTITION_RECEIPT[
            "canonical_payload_sha256"
        ]
        and partition_receipt.get("source_manifest_order")
        == list(_STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_SOURCE_MANIFESTS)
        and isinstance(partition_train, Mapping)
        and partition_train.get("rows")
        == _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_TRAIN_ROWS
        and isinstance(partition_invariants, Mapping)
        and partition_invariants
        and all(value is True for value in partition_invariants.values())
    ):
        raise RuntimeError("new-head formal partition receipt semantics drifted")

    runtime_support_record = (
        support_receipt.get("outputs", {}).get("runtime_support_tsv")
        if isinstance(support_receipt.get("outputs"), Mapping)
        else None
    )
    runtime_bank = support_receipt.get("runtime_bank")
    coverage = support_receipt.get("training_class_coverage")
    support_invariants = support_receipt.get("invariants")
    expected_runtime_support_record = {
        key: _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_RUNTIME_SUPPORT[key]
        for key in ("path", "sha256", "size_bytes", "rows")
    }
    if not (
        support_receipt.get("schema")
        == _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_SUPPORT_RECEIPT["schema"]
        and support_receipt.get("canonical_payload_sha256")
        == _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_SUPPORT_RECEIPT[
            "canonical_payload_sha256"
        ]
        and runtime_support_record == expected_runtime_support_record
        and isinstance(runtime_bank, Mapping)
        and runtime_bank.get("candidate_rows")
        == _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_RUNTIME_SUPPORT["rows"]
        and runtime_bank.get("class_count")
        == _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_RUNTIME_SUPPORT["class_count"]
        and isinstance(coverage, Mapping)
        and coverage.get("required_class_count")
        == _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_RUNTIME_SUPPORT[
            "required_training_classes"
        ]
        and coverage.get("covered_class_count")
        == _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_RUNTIME_SUPPORT[
            "required_training_classes"
        ]
        and coverage.get("missing_class_ids") == []
        and isinstance(support_invariants, Mapping)
        and support_invariants
        and all(value is True for value in support_invariants.values())
    ):
        raise RuntimeError("new-head formal support receipt semantics drifted")
    runtime_support_path = Path(
        _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_RUNTIME_SUPPORT["path"]
    ).resolve(strict=True)
    if not (
        runtime_support_path.stat().st_size
        == _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_RUNTIME_SUPPORT["size_bytes"]
        and _sha256_file(runtime_support_path)
        == _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_RUNTIME_SUPPORT["sha256"]
    ):
        raise RuntimeError("new-head formal runtime support TSV drifted")

    dataset_payload = _load_json_mapping(dataset_path, label="dataset config")
    train_rows = dataset_payload.get("train")
    if not (
        isinstance(train_rows, list)
        and len(train_rows)
        == len(_STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_SOURCE_MANIFESTS)
        and dataset_payload.get("val") == []
    ):
        raise RuntimeError("new-head formal dataset topology drifted")
    variant_outputs = partition_receipt.get("outputs", {}).get(
        variant_contract["receipt_variant"], {}
    )
    receipt_train_records = (
        variant_outputs.get("train")
        if isinstance(variant_outputs, Mapping)
        else None
    )
    if not isinstance(receipt_train_records, Mapping):
        raise RuntimeError("new-head formal partition has no variant train records")

    manifest_bindings = []
    total_rows = 0
    for index, (row, manifest_name, manifest_sha) in enumerate(
        zip(
            train_rows,
            _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_SOURCE_MANIFESTS,
            variant_contract["manifest_sha256"],
        )
    ):
        record = receipt_train_records.get(manifest_name)
        if not isinstance(row, Mapping) or not isinstance(record, Mapping):
            raise RuntimeError(
                f"new-head formal dataset row {index} binding is incomplete"
            )
        exact_row_fields = {
            "dataset_mode": "patch_episode",
            "root": "/",
            "stage_b_data_driven_variant": variant_contract["dataset_variant"],
            "stage_b_data_driven_partition": "train",
            "stage_b_data_driven_manifest_sha256": manifest_sha,
            "stage_b_data_driven_receipt_sha256": (
                _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_PARTITION_RECEIPT["sha256"]
            ),
            "stage_b_data_driven_support_receipt_sha256": (
                _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_SUPPORT_RECEIPT["sha256"]
            ),
            "patch_bank_cache": False,
            "patch_bank_cache_write": False,
            "support_patch_use_embedding": False,
            "support_patch_max_per_class": 200,
            "mix_weight": 2.0,
        }
        row_drift = {
            key: (row.get(key), expected)
            for key, expected in exact_row_fields.items()
            if not _matches_exact_type(row.get(key), expected)
        }
        if row_drift or str(row.get("patch_bank_cache_path", "") or "").strip():
            raise RuntimeError(
                f"new-head formal dataset row {index} contract drifted: {row_drift}"
            )
        annotation_path = _expand_stage_b_data_driven_path(
            str(row.get("anno", "") or "")
        ).resolve(strict=True)
        record_path = _expand_stage_b_data_driven_path(
            str(record.get("path", "") or "")
        ).resolve(strict=True)
        row_partition_path = _expand_stage_b_data_driven_path(
            str(row.get("stage_b_data_driven_receipt", "") or "")
        ).resolve(strict=True)
        row_support_path = _expand_stage_b_data_driven_path(
            str(row.get("stage_b_data_driven_support_receipt", "") or "")
        ).resolve(strict=True)
        row_runtime_support_path = _expand_stage_b_data_driven_path(
            str(row.get("support_patch_tsv", "") or "")
        ).resolve(strict=True)
        if not (
            annotation_path.name == manifest_name
            and annotation_path == record_path
            and annotation_path.stat().st_size == record.get("size_bytes")
            and record.get("sha256") == manifest_sha
            and type(record.get("rows")) is int
            and record["rows"] > 0
            and row_partition_path == partition_path
            and row_support_path == support_path
            and row_runtime_support_path == runtime_support_path
        ):
            raise RuntimeError(
                f"new-head formal dataset row {index} asset binding drifted"
            )
        total_rows += record["rows"]
        manifest_bindings.append(
            {
                "path": str(annotation_path),
                "sha256": manifest_sha,
                "rows": record["rows"],
            }
        )
    if total_rows != _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_TRAIN_ROWS:
        raise RuntimeError("new-head formal train-row budget drifted")

    steps_per_epoch = total_rows // required_runtime["batch_size"]
    dropped_rows_per_epoch = total_rows % required_runtime["batch_size"]
    expected_updates = steps_per_epoch * required_runtime["epochs"]
    if expected_updates != _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_EXPECTED_UPDATES:
        raise RuntimeError("new-head formal optimizer-update budget drifted")
    allocator_value = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if allocator_value != "expandable_segments:True":
        raise RuntimeError(
            "new-head formal allocator environment drifted: "
            f"PYTORCH_CUDA_ALLOC_CONF={allocator_value!r}"
        )

    binding = {
        "schema": _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_BINDING_SCHEMA,
        "scope": scope,
        "contract": _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_CONTRACT,
        "experiment_id": experiment_id,
        "variant_id": variant_contract["variant_id"],
        "category_complete": variant_contract["category_complete"],
        "config_file": {
            "path": str(config_path),
            "sha256": _sha256_file(config_path),
        },
        "dataset_config": {
            "path": str(dataset_path),
            "sha256": observed_dataset_sha,
        },
        "output_dir": str(expected_output_path),
        "initializer": {
            "path": str(expected_a0_path),
            "sha256": observed_base_sha,
        },
        "initializer_pair_receipt": {
            "path": str(expected_pair_path),
            "sha256": observed_pair_sha,
        },
        "lr_selection_receipt": {
            "path": str(lr_selection_path),
            "sha256": lr_selection_sha,
            "schema": lr_selection_receipt["schema"],
            "selected_rank_lr": selected_rank_lr,
        },
        "partition_receipt": {
            "path": str(partition_path),
            "sha256": _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_PARTITION_RECEIPT[
                "sha256"
            ],
            "canonical_payload_sha256": partition_receipt[
                "canonical_payload_sha256"
            ],
        },
        "support_receipt": {
            "path": str(support_path),
            "sha256": _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_SUPPORT_RECEIPT[
                "sha256"
            ],
            "canonical_payload_sha256": support_receipt[
                "canonical_payload_sha256"
            ],
        },
        "runtime_support_tsv": dict(expected_runtime_support_record),
        "manifests": manifest_bindings,
        "training_budget": {
            "train_rows_per_epoch": total_rows,
            "batch_size": required_runtime["batch_size"],
            "drop_last": True,
            "steps_per_epoch": steps_per_epoch,
            "dropped_rows_per_epoch": dropped_rows_per_epoch,
            "epochs": required_runtime["epochs"],
            "expected_optimizer_updates": expected_updates,
            "max_train_iters": required_runtime["max_train_iters"],
        },
        "optimizer_contract": {
            "rank_lr": observed_rank_lr,
            "selected_rank_lr": selected_rank_lr,
            "patch_lr": required_runtime["stage_b_data_driven_patch_lr"],
            "weight_decay": required_runtime["weight_decay"],
            "clip_max_norm": required_runtime["clip_max_norm"],
            "lr_drop": required_runtime["lr_drop"],
            "onecyclelr": required_runtime["onecyclelr"],
            "multi_step_lr": required_runtime["multi_step_lr"],
        },
        "runtime": {
            "seed": required_runtime["seed"],
            "sampler_seed": required_runtime[
                "stage_b_data_driven_sampler_seed"
            ],
            "loader_seed": required_runtime[
                "stage_b_data_driven_loader_seed"
            ],
            "gradient_accumulation_steps": required_runtime[
                "gradient_accumulation_steps"
            ],
            "amp": required_runtime["amp"],
            "num_workers": required_runtime["num_workers"],
            "prefetch_factor": required_runtime["prefetch_factor"],
            "pin_memory": required_runtime["pin_memory"],
            "persistent_workers": required_runtime["persistent_workers"],
            "allocator": {
                "environment_variable": "PYTORCH_CUDA_ALLOC_CONF",
                "value": allocator_value,
            },
        },
    }
    args.stage_b_data_driven_new_head_formal_binding = binding
    return binding


def _validate_stage_b_data_driven_assignment_training_contract(
    args,
    *,
    base_path: Path,
    variant_id: str,
    dataset_path: Optional[Path] = None,
) -> None:
    variant_id = str(variant_id)
    if variant_id not in _STAGE_B_DATA_DRIVEN_ASSIGNMENT_VARIANTS:
        raise RuntimeError(f"unknown official-assignment variant: {variant_id!r}")
    expected_deployment_weight = _STAGE_B_DATA_DRIVEN_ASSIGNMENT_VARIANTS[
        variant_id
    ]
    required = {
        "stage_b_data_driven_experiment_id": "DD1",
        "stage_b_data_driven_train_mode": "rank_patch_only",
        "stage_b_data_driven_category_complete": True,
        "stage_b_data_driven_confidence_trained": False,
        "stage_b_data_driven_rank_architecture": "relational_v1",
        "stage_b_data_driven_rank_supervision": (
            "official_same_image_same_category_assignment_v1"
        ),
        "stage_b_data_driven_strict_sample_identity": True,
        "stage_b_data_driven_rank_negative_iou_threshold": 0.3,
        "stage_b_data_driven_category_gate_max_gap": 3.0,
        "stage_b_data_driven_patch_score_clip": 5.0,
        "stage_b_data_driven_rank_weight": 0.0,
        "stage_b_data_driven_patch_weight": 1.0,
        "stage_b_data_driven_assignment_weight": 1.0,
        "stage_b_data_driven_deployment_weight": expected_deployment_weight,
        "stage_b_data_driven_no_teacher_contract": (
            "b58_only_random_independent_heads_v1"
        ),
        "enable_patch_branch": True,
        "fix_size": True,
        "strong_aug": False,
        "data_aug_hflip_prob": 0.0,
    }
    drifted = {
        key: (getattr(args, key, None), expected)
        for key, expected in required.items()
        if getattr(args, key, None) != expected
    }
    if drifted:
        raise RuntimeError(
            f"{variant_id} official-assignment contract drifted: {drifted}"
        )
    forbidden_routes = {
        "stage_b": False,
        "stage_b_gdino_score_adapter": False,
        "stage_b_u0_patch_rank": False,
        "stage_b_v7": False,
        "stage_b_v11_fixed_text": False,
        "stage_b_legacy_global_gate": False,
    }
    enabled_forbidden_routes = {
        key: getattr(args, key, None)
        for key, expected in forbidden_routes.items()
        if getattr(args, key, None) is not expected
    }
    if enabled_forbidden_routes:
        raise RuntimeError(
            f"{variant_id} teacher/legacy score routes are not disabled: "
            f"{enabled_forbidden_routes}"
        )

    if dataset_path is None:
        dataset_value = str(getattr(args, "datasets", "") or "").strip()
        if not dataset_value:
            raise RuntimeError(f"{variant_id} requires an exact dataset config")
        dataset_path = Path(dataset_value).expanduser().resolve(strict=True)
    _validate_stage_b_data_driven_assignment_dataset_contract(
        args, dataset_path=dataset_path
    )

    if bool(getattr(args, "eval", False)):
        return
    if str(getattr(args, "resume", "") or "").strip():
        raise RuntimeError(f"{variant_id} training is fresh-start only")
    pretrain_value = str(
        getattr(args, "pretrain_model_path", "") or ""
    ).strip()
    if not pretrain_value:
        raise RuntimeError(
            f"{variant_id} training requires the canonical A1 initializer"
        )
    if Path(pretrain_value).expanduser().resolve(strict=True) != base_path:
        raise RuntimeError(
            f"{variant_id} pretrain path differs from the canonical A1 initializer"
        )
    pair_receipt_path = str(
        getattr(args, "stage_b_data_driven_initializer_pair_receipt_path", "")
        or ""
    ).strip()
    pair_receipt_sha = str(
        getattr(args, "stage_b_data_driven_initializer_pair_receipt_sha256", "")
        or ""
    ).strip()
    if not pair_receipt_path or not _is_stage_b_data_driven_sha256(
        pair_receipt_sha
    ):
        raise RuntimeError(
            f"{variant_id} requires the sealed fresh A1 initializer pair receipt"
        )

    scope = str(
        getattr(args, "stage_b_data_driven_assignment_dataset_scope", "") or ""
    ).strip()
    if scope == _STAGE_B_DATA_DRIVEN_ASSIGNMENT_OVERFIT_SCOPE:
        sealed_runtime = {
            "stage_b_data_driven_pairtop1_u500_expected_max_train_iters": 500,
            "stage_b_data_driven_pairtop1_u500_expected_num_workers": 0,
            "stage_b_data_driven_pairtop1_u500_expected_pin_memory": False,
            "stage_b_data_driven_pairtop1_u500_expected_iter_checkpoint_interval": 500,
            "stage_b_data_driven_pairtop1_u500_expected_save_checkpoint_interval": 500,
        }
        sealed_runtime_drift = {
            key: (getattr(args, key, None), expected)
            for key, expected in sealed_runtime.items()
            if getattr(args, key, None) != expected
        }
        if sealed_runtime_drift:
            raise RuntimeError(
                f"{variant_id} Overfit64/U500 sealed runtime drifted: "
                f"{sealed_runtime_drift}"
            )
        expected_runtime = {
            "seed": 42,
            "batch_size": 64,
            "epochs": 500,
            "lr_drop": 500,
            "stage_b_data_driven_epoch_checkpoint_interval": 500,
            "max_train_iters": 500,
            "iter_checkpoint_interval": 500,
            "save_checkpoint_interval": 500,
            "num_workers": 0,
            "prefetch_factor": 1,
            "pin_memory": False,
            "persistent_workers": False,
            "gradient_accumulation_steps": 1,
            "amp": True,
            "save_log": True,
            "world_size": 1,
            "distributed": False,
        }
        runtime_drift = {
            key: (getattr(args, key, None), expected)
            for key, expected in expected_runtime.items()
            if getattr(args, key, None) != expected
        }
        if runtime_drift:
            raise RuntimeError(
                f"{variant_id} Overfit64/U500 runtime drifted: {runtime_drift}"
            )

    if int(getattr(args, "max_train_iters", -1) or -1) == 5020:
        formal_scope = str(
            getattr(args, "stage_b_data_driven_execution_scope", "") or ""
        ).strip()
        if (
            variant_id != _STAGE_B_DATA_DRIVEN_PAIRTOP1_HARDGAP3_VARIANT
            or scope != _STAGE_B_DATA_DRIVEN_ASSIGNMENT_FULL_SCOPE
            or formal_scope != _STAGE_B_DATA_DRIVEN_DD1H_FORMAL_SCOPE
        ):
            raise RuntimeError(
                f"{variant_id} U5020 is not authorized until its paired U50 "
                "causal probe and formal evidence are sealed"
            )


def _validate_stage_b_data_driven_role_routed_training_contract(
    args,
    *,
    base_path: Path,
    dataset_path: Path,
) -> None:
    required = {
        "stage_b_data_driven_variant_id": _STAGE_B_DATA_DRIVEN_ROLE_ROUTED_VARIANT,
        "stage_b_data_driven_experiment_id": "DD1",
        "stage_b_data_driven_train_mode": "rank_patch_only",
        "stage_b_data_driven_category_complete": True,
        "stage_b_data_driven_confidence_trained": False,
        "stage_b_data_driven_rank_architecture": "absolute_token",
        "stage_b_data_driven_rank_supervision": (
            "role_routed_official_assignment_all_exclusive_nonowned_v2"
        ),
        "stage_b_data_driven_strict_sample_identity": True,
        "stage_b_data_driven_rank_negative_iou_threshold": 0.3,
        "stage_b_data_driven_positive_iou_threshold": 0.5,
        "stage_b_data_driven_patch_negative_iou_threshold": 0.3,
        "stage_b_data_driven_category_gate_max_gap": 3.0,
        "stage_b_data_driven_category_gate_boundary_margin": 0.25,
        "stage_b_data_driven_patch_active_unsafe_auxiliary_weight": 1.0,
        "stage_b_data_driven_patch_dense_category_focal_weight": 0.0,
        "stage_b_data_driven_patch_dense_category_focal_alpha": 0.25,
        "stage_b_data_driven_patch_dense_category_focal_gamma": 2.0,
        "stage_b_data_driven_patch_dense_category_focal_negative_weight": 1.0,
        "stage_b_data_driven_patch_score_clip": 5.0,
        "stage_b_data_driven_rank_margin": 0.1,
        "stage_b_data_driven_rank_weight": 0.0,
        "stage_b_data_driven_patch_weight": 1.0,
        "stage_b_data_driven_assignment_weight": 1.0,
        "stage_b_data_driven_deployment_weight": 0.0,
        "stage_b_data_driven_assignment_dataset_scope": (
            _STAGE_B_DATA_DRIVEN_ROLE_ROUTED_SCOPE
        ),
        "stage_b_data_driven_no_teacher_contract": (
            "clean_dd1_u1000_stageb_data_only_model_warm_start_v1"
        ),
        "stage_b_data_driven_role_fresh_optimizer": True,
        "stage_b_data_driven_role_expected_max_train_iters": 1000,
        "stage_b_data_driven_role_expected_iter_checkpoint_interval": 1000,
        "stage_b_data_driven_role_expected_amp": True,
        "stage_b_data_driven_role_expected_seed": 42,
        "stage_b_data_driven_role_expected_prefetch_factor": 1,
        "stage_b_data_driven_role_expected_gradient_accumulation_steps": 1,
        "stage_b_data_driven_sampler_seed": 42,
        "stage_b_data_driven_loader_seed": 1042,
        "stage_b_data_driven_grad_clip_contract": "per_optimizer_branch_v1",
        "stage_b_data_driven_rank_lr": 3e-4,
        "weight_decay": 1e-4,
        "clip_max_norm": 0.1,
        "batch_size": 64,
        "epochs": 1,
        "enable_patch_branch": True,
        "patch_gate_with_text": False,
        "fix_size": True,
        "strong_aug": False,
        "data_aug_hflip_prob": 0.0,
    }
    drifted = {
        key: (getattr(args, key, None), expected)
        for key, expected in required.items()
        if getattr(args, key, None) != expected
    }
    if drifted:
        raise RuntimeError(f"clean role-routed contract drifted: {drifted}")
    patch_gradient_contract = (
        getattr(args, "stage_b_data_driven_patch_row_balance_contract", None),
        getattr(
            args,
            "stage_b_data_driven_patch_drop_positive_anchor_gradient_policy",
            None,
        ),
    )
    allowed_patch_gradient_contracts = {
        (
            "gate_barrier_role_exclusive_plus_allnegative_active_severity_"
            "zero_sum_no_raw_focal_v9",
            "global_max_positive_v1",
        ),
        (
            "gate_barrier_role_exclusive_plus_allnegative_active_severity_"
            "instance_balanced_zero_sum_no_raw_focal_v10",
            "reachable_instance_best_mean_straight_through_v1",
        ),
    }
    if patch_gradient_contract not in allowed_patch_gradient_contracts:
        raise RuntimeError(
            "clean role-routed patch gradient contract drifted: "
            f"{patch_gradient_contract}"
        )
    patch_training_surface = (
        bool(getattr(args, "stage_b_data_driven_patch_residual", False)),
        getattr(args, "stage_b_data_driven_patch_training_surface", None),
        getattr(args, "stage_b_data_driven_initializer_contract", None),
        getattr(args, "stage_b_data_driven_patch_lr", None),
    )
    allowed_patch_training_surfaces = {
        (
            False,
            "base_projection_8tensor_v1",
            "clean_dd1_u1000_model_only_role_routed_v1",
            1e-4,
        ),
        (
            True,
            "residual_only_3tensor_v1",
            "clean_dd1_u1000_model_only_patch_residual128_v1",
            3e-4,
        ),
        (
            True,
            "residual_only_3tensor_v1",
            "clean_dd1_u1000_model_only_patch_residual128_raw_centered_v1",
            3e-4,
        ),
        (
            True,
            "residual_only_6tensor_topk_semantic_v1",
            "clean_dd1_u1000_model_only_patch_topksemantic128_ctx16_v1",
            3e-4,
        ),
    }
    if patch_training_surface not in allowed_patch_training_surfaces:
        raise RuntimeError(
            "clean role-routed patch training surface drifted: "
            f"{patch_training_surface}"
        )
    patch_residual = patch_training_surface[0]
    if patch_residual:
        residual_architecture = (
            getattr(args, "stage_b_data_driven_patch_residual_contract", None),
            getattr(args, "stage_b_data_driven_patch_residual_hidden_dim", None),
            getattr(args, "stage_b_data_driven_patch_residual_context_dim", None),
            getattr(args, "stage_b_data_driven_patch_residual_context_topk", None),
            getattr(args, "stage_b_data_driven_patch_residual_limit", None),
            getattr(args, "stage_b_data_driven_patch_residual_init_seed", None),
            bool(
                getattr(
                    args,
                    "stage_b_data_driven_patch_residual_center_raw",
                    False,
                )
            ),
            getattr(
                args,
                "stage_b_data_driven_patch_residual_source_initializer_sha256",
                None,
            ),
        )
        allowed_residual_architectures = {
            (
                "detached_qp_mlp128_tanh025_v1",
                128,
                None,
                None,
                0.25,
                42,
                False,
                "5ae688008cf56130c69c152197911fa61fecb6a24956f425fdd5a7ac42e97bd1",
            ),
            (
                "detached_qp_mlp128_query_raw_centered_tanh025_v2",
                128,
                None,
                None,
                0.25,
                42,
                True,
                "5ae688008cf56130c69c152197911fa61fecb6a24956f425fdd5a7ac42e97bd1",
            ),
            (
                "detached_qp_base_topk10_semantic_context16_"
                "query_raw_centered_tanh025_v3",
                128,
                16,
                10,
                0.25,
                42,
                True,
                "5ae688008cf56130c69c152197911fa61fecb6a24956f425fdd5a7ac42e97bd1",
            ),
        }
        if residual_architecture not in allowed_residual_architectures:
            raise RuntimeError(
                "clean role-routed patch residual architecture drifted: "
                f"{residual_architecture}"
            )
    expected_num_workers = getattr(
        args, "stage_b_data_driven_role_expected_num_workers", None
    )
    if (
        isinstance(expected_num_workers, bool)
        or not isinstance(expected_num_workers, int)
        or expected_num_workers not in {0, 2, 4, 8}
    ):
        raise RuntimeError(
            "clean role-routed expected worker count must be one of {0, 2, 4, 8}: "
            f"{expected_num_workers!r}"
        )
    expected_pin_memory = getattr(
        args, "stage_b_data_driven_role_expected_pin_memory", None
    )
    if expected_pin_memory is not None:
        if not isinstance(expected_pin_memory, bool):
            raise RuntimeError(
                "clean role-routed expected pin-memory value must be boolean: "
                f"{expected_pin_memory!r}"
            )
        if getattr(args, "pin_memory", None) is not expected_pin_memory:
            raise RuntimeError(
                "clean role-routed CLI pin-memory runtime drifted: "
                f"{getattr(args, 'pin_memory', None)!r} != {expected_pin_memory!r}"
            )
    required_runtime = {
        "max_train_iters": 1000,
        "iter_checkpoint_interval": 1000,
        "amp": True,
        "seed": 42,
        "num_workers": expected_num_workers,
        "prefetch_factor": 1,
        "gradient_accumulation_steps": 1,
    }
    runtime_drift = {
        key: (getattr(args, key, None), expected)
        for key, expected in required_runtime.items()
        if getattr(args, key, None) != expected
    }
    if runtime_drift:
        raise RuntimeError(
            f"clean role-routed CLI runtime drifted: {runtime_drift}"
        )
    forbidden_routes = (
        "stage_b",
        "stage_b_gdino_score_adapter",
        "stage_b_u0_patch_rank",
        "stage_b_v7",
        "stage_b_v11_fixed_text",
        "stage_b_legacy_global_gate",
    )
    enabled = {
        key: getattr(args, key, None)
        for key in forbidden_routes
        if getattr(args, key, None) is not False
    }
    if enabled:
        raise RuntimeError(
            f"clean role-routed teacher/legacy routes are not disabled: {enabled}"
        )
    if bool(getattr(args, "eval", False)):
        raise RuntimeError("clean role-routed training variant is training-only")
    if str(getattr(args, "resume", "") or "").strip():
        raise RuntimeError(
            "clean role-routed training requires a fresh optimizer, not --resume"
        )
    pretrain = str(getattr(args, "pretrain_model_path", "") or "").strip()
    if not pretrain or Path(pretrain).expanduser().resolve(strict=True) != base_path:
        raise RuntimeError(
            "clean role-routed pretrain path differs from its model-only initializer"
        )

    expected_dataset_value = str(
        getattr(
            args, "stage_b_data_driven_assignment_dataset_config_path", ""
        )
        or ""
    ).strip()
    expected_dataset_sha = str(
        getattr(
            args, "stage_b_data_driven_assignment_dataset_config_sha256", ""
        )
        or ""
    ).strip()
    if not expected_dataset_value or not _is_stage_b_data_driven_sha256(
        expected_dataset_sha
    ):
        raise RuntimeError("clean role-routed dataset path/SHA binding is incomplete")
    expected_dataset_path = _expand_stage_b_data_driven_path(
        expected_dataset_value
    ).resolve(strict=True)
    if dataset_path.resolve(strict=True) != expected_dataset_path:
        raise RuntimeError("clean role-routed dataset config path drifted")
    if _sha256_file(expected_dataset_path) != expected_dataset_sha:
        raise RuntimeError("clean role-routed dataset config SHA drifted")
    try:
        dataset_payload = json.loads(
            expected_dataset_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"could not parse clean role-routed dataset config: {error}"
        ) from error
    train_rows = (
        dataset_payload.get("train")
        if isinstance(dataset_payload, Mapping)
        else None
    )
    expected_names = list(_STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_SOURCE_MANIFESTS)
    if not (
        isinstance(train_rows, list)
        and len(train_rows) == 3
        and dataset_payload.get("val") == []
        and [Path(str(row.get("anno", ""))).name for row in train_rows] == expected_names
        and all(row.get("lazy_jsonl") is True for row in train_rows)
    ):
        raise RuntimeError(
            "clean role-routed dataset row order/coverage/lazy-loading drifted"
        )
    from datasets.patch_episode import _validate_data_driven_ref_dataset_binding

    for index, row in enumerate(train_rows):
        try:
            observed_variant = _validate_data_driven_ref_dataset_binding(
                args, dict(row), image_set="train"
            )
        except (OSError, ValueError, RuntimeError) as error:
            raise RuntimeError(
                f"clean role-routed dataset row {index} failed lineage validation: "
                f"{error}"
            ) from error
        if observed_variant != "dd1_official_assignment_pair":
            raise RuntimeError(
                f"clean role-routed dataset row {index} variant drifted"
            )

    expected_receipt_value = str(
        getattr(args, "stage_b_data_driven_assignment_receipt_path", "") or ""
    ).strip()
    expected_receipt_sha = str(
        getattr(args, "stage_b_data_driven_assignment_receipt_sha256", "") or ""
    ).strip()
    manifest_shas = getattr(
        args, "stage_b_data_driven_assignment_manifest_sha256", None
    )
    if not (
        expected_receipt_value
        and _is_stage_b_data_driven_sha256(expected_receipt_sha)
        and isinstance(manifest_shas, Mapping)
        and list(manifest_shas) == expected_names
        and all(_is_stage_b_data_driven_sha256(value) for value in manifest_shas.values())
        and getattr(args, "stage_b_data_driven_assignment_expected_rows", None)
        == 263661
        and getattr(
            args, "stage_b_data_driven_assignment_expected_valid_rows", None
        )
        == 224723
    ):
        raise RuntimeError("clean role-routed manifest bindings are incomplete")
    receipt_path = _expand_stage_b_data_driven_path(
        expected_receipt_value
    ).resolve(strict=True)
    if _sha256_file(receipt_path) != expected_receipt_sha:
        raise RuntimeError("clean role-routed assignment receipt SHA drifted")
    if any(
        row.get("stage_b_data_driven_receipt_sha256") != expected_receipt_sha
        or row.get("stage_b_data_driven_manifest_sha256")
        != manifest_shas[expected_names[index]]
        for index, row in enumerate(train_rows)
    ):
        raise RuntimeError("clean role-routed dataset/argument hashes disagree")

    initializer_receipt_value = str(
        getattr(
            args, "stage_b_data_driven_role_initializer_receipt_path", ""
        )
        or ""
    ).strip()
    initializer_receipt_sha = str(
        getattr(
            args, "stage_b_data_driven_role_initializer_receipt_sha256", ""
        )
        or ""
    ).strip()
    if not initializer_receipt_value or not _is_stage_b_data_driven_sha256(
        initializer_receipt_sha
    ):
        raise RuntimeError(
            "clean role-routed initializer receipt path/SHA binding is incomplete"
        )
    initializer_receipt_path = _expand_stage_b_data_driven_path(
        initializer_receipt_value
    ).resolve(strict=True)
    if _sha256_file(initializer_receipt_path) != initializer_receipt_sha:
        raise RuntimeError("clean role-routed initializer receipt SHA drifted")
    try:
        initializer_receipt = json.loads(
            initializer_receipt_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"could not parse clean role-routed initializer receipt: {error}"
        ) from error
    canonical_sha = initializer_receipt.get("canonical_payload_sha256")
    canonical_payload = dict(initializer_receipt)
    canonical_payload.pop("canonical_payload_sha256", None)
    try:
        observed_canonical_sha = hashlib.sha256(
            json.dumps(
                canonical_payload,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise RuntimeError(
            "clean role-routed initializer receipt canonical payload is invalid"
        ) from error
    checkpoint_record = initializer_receipt.get("checkpoint")
    source_record = initializer_receipt.get("source_checkpoint")
    source_a0 = initializer_receipt.get("source_a0_initializer")
    invariants = initializer_receipt.get("invariants")
    expected_source_sha = str(
        getattr(
            args,
            "stage_b_data_driven_role_initializer_source_checkpoint_sha256",
            "",
        )
        or ""
    ).strip()
    expected_a0_sha = str(
        getattr(
            args, "stage_b_data_driven_role_initializer_a0_sha256", ""
        )
        or ""
    ).strip()
    if not (
        isinstance(initializer_receipt, Mapping)
        and initializer_receipt.get("schema")
        == (
            (
                _STAGE_B_DATA_DRIVEN_PATCH_TOPK_INITIALIZER_RECEIPT_SCHEMA
                if patch_training_surface[1]
                == "residual_only_6tensor_topk_semantic_v1"
                else _STAGE_B_DATA_DRIVEN_PATCH_RESIDUAL_INITIALIZER_RECEIPT_SCHEMA
            )
            if patch_residual
            else _STAGE_B_DATA_DRIVEN_ROLE_ROUTED_INITIALIZER_RECEIPT_SCHEMA
        )
        and canonical_sha == observed_canonical_sha
        and isinstance(checkpoint_record, Mapping)
        and checkpoint_record.get("sha256")
        == getattr(args, "stage_b_data_driven_base_initializer_sha256", None)
        and _expand_stage_b_data_driven_path(
            str(checkpoint_record.get("path", ""))
        ).resolve(strict=True)
        == base_path
        and isinstance(source_record, Mapping)
        and source_record.get("sha256") == expected_source_sha
        and isinstance(source_a0, Mapping)
        and source_a0.get("sha256") == expected_a0_sha
        and isinstance(invariants, Mapping)
        and invariants
        and all(value is True for value in invariants.values())
        and (
            not patch_residual
            or initializer_receipt.get(
                "source_role_routed_initializer", {}
            ).get("sha256")
            == getattr(
                args,
                "stage_b_data_driven_patch_residual_source_initializer_sha256",
                None,
            )
        )
    ):
        raise RuntimeError("clean role-routed initializer receipt contract drifted")
    args.stage_b_data_driven_role_initializer_receipt = {
        "path": str(initializer_receipt_path),
        "sha256": initializer_receipt_sha,
        "schema": initializer_receipt["schema"],
        "source_checkpoint_sha256": expected_source_sha,
        "source_a0_initializer_sha256": expected_a0_sha,
    }


def _is_stage_b_data_driven_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_stage_b_data_driven_assignment_dataset_contract(
    args, *, dataset_path: Path
) -> None:
    scope = str(
        getattr(args, "stage_b_data_driven_assignment_dataset_scope", "") or ""
    ).strip()
    if scope not in {
        _STAGE_B_DATA_DRIVEN_ASSIGNMENT_FULL_SCOPE,
        _STAGE_B_DATA_DRIVEN_ASSIGNMENT_OVERFIT_SCOPE,
    }:
        raise RuntimeError(
            f"unknown official-assignment dataset scope: {scope!r}"
        )

    expected_dataset_value = str(
        getattr(
            args, "stage_b_data_driven_assignment_dataset_config_path", ""
        )
        or ""
    ).strip()
    expected_dataset_sha = str(
        getattr(
            args, "stage_b_data_driven_assignment_dataset_config_sha256", ""
        )
        or ""
    ).strip()
    if not expected_dataset_value or not _is_stage_b_data_driven_sha256(
        expected_dataset_sha
    ):
        raise RuntimeError(
            "official-assignment dataset config path/SHA binding is incomplete"
        )
    expected_dataset_path = _expand_stage_b_data_driven_path(
        expected_dataset_value
    ).resolve(strict=True)
    resolved_dataset_path = dataset_path.expanduser().resolve(strict=True)
    if resolved_dataset_path != expected_dataset_path:
        raise RuntimeError("official-assignment dataset config path drifted")
    if _sha256_file(resolved_dataset_path) != expected_dataset_sha:
        raise RuntimeError("official-assignment dataset config SHA drifted")
    try:
        dataset_payload = json.loads(
            resolved_dataset_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"could not parse official-assignment dataset config: {error}"
        ) from error
    train_rows = (
        dataset_payload.get("train")
        if isinstance(dataset_payload, Mapping)
        else None
    )
    if not isinstance(train_rows, list) or not train_rows:
        raise RuntimeError(
            "official-assignment dataset config requires non-empty train rows"
        )
    if dataset_payload.get("val") != []:
        raise RuntimeError(
            "official-assignment training dataset config must have val=[]"
        )

    expected_receipt_value = str(
        getattr(args, "stage_b_data_driven_assignment_receipt_path", "") or ""
    ).strip()
    expected_receipt_sha = str(
        getattr(args, "stage_b_data_driven_assignment_receipt_sha256", "") or ""
    ).strip()
    if not expected_receipt_value or not _is_stage_b_data_driven_sha256(
        expected_receipt_sha
    ):
        raise RuntimeError(
            "official-assignment receipt path/SHA binding is incomplete"
        )
    receipt_path = _expand_stage_b_data_driven_path(
        expected_receipt_value
    ).resolve(strict=True)
    if _sha256_file(receipt_path) != expected_receipt_sha:
        raise RuntimeError("official-assignment receipt SHA drifted")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"could not parse official-assignment receipt: {error}"
        ) from error
    if not isinstance(receipt, Mapping):
        raise RuntimeError("official-assignment receipt must be a JSON mapping")

    raw_manifest_shas = getattr(
        args, "stage_b_data_driven_assignment_manifest_sha256", None
    )
    if not isinstance(raw_manifest_shas, Mapping) or not raw_manifest_shas:
        raise RuntimeError(
            "official-assignment manifest SHA binding is incomplete"
        )
    expected_manifest_shas = {
        str(name): str(sha) for name, sha in raw_manifest_shas.items()
    }
    if any(
        not name or not _is_stage_b_data_driven_sha256(sha)
        for name, sha in expected_manifest_shas.items()
    ):
        raise RuntimeError("official-assignment manifest SHA binding is malformed")

    observed_names = []
    observed_paths = {}
    for index, row in enumerate(train_rows):
        if not isinstance(row, Mapping):
            raise RuntimeError(
                f"official-assignment train row {index} is not a mapping"
            )
        if (
            row.get("dataset_mode") != "patch_episode"
            or row.get("stage_b_data_driven_variant")
            != "dd1_official_assignment_pair"
            or row.get("neg_episode_prob") != 0.0
            or row.get("tn_balance_sampling") is not False
        ):
            raise RuntimeError(
                f"official-assignment train row {index} contract drifted"
            )
        row_receipt = _expand_stage_b_data_driven_path(
            str(row.get("stage_b_data_driven_receipt", ""))
        ).resolve(strict=True)
        if (
            row_receipt != receipt_path
            or row.get("stage_b_data_driven_receipt_sha256")
            != expected_receipt_sha
        ):
            raise RuntimeError(
                f"official-assignment train row {index} receipt drifted"
            )
        anno_path = _expand_stage_b_data_driven_path(
            str(row.get("anno", ""))
        ).resolve(strict=True)
        name = anno_path.name
        observed_names.append(name)
        observed_paths[name] = anno_path
        if row.get("stage_b_data_driven_manifest_sha256") != (
            expected_manifest_shas.get(name)
        ):
            raise RuntimeError(
                f"official-assignment train row {index} manifest SHA drifted"
            )
    if observed_names != list(expected_manifest_shas):
        raise RuntimeError(
            "official-assignment manifest order/coverage drifted: "
            f"expected={list(expected_manifest_shas)}, observed={observed_names}"
        )

    expected_rows = getattr(
        args, "stage_b_data_driven_assignment_expected_rows", None
    )
    expected_valid_rows = getattr(
        args, "stage_b_data_driven_assignment_expected_valid_rows", None
    )
    if scope == _STAGE_B_DATA_DRIVEN_ASSIGNMENT_FULL_SCOPE:
        invariants = receipt.get("invariants")
        selection = receipt.get("selection_contract")
        if (
            receipt.get("schema")
            != "pivot.stageb.data_driven.official_assignment_pair_receipt/v1"
            or receipt.get("row_schema")
            != "pivot.stageb.data_driven.official_assignment_pair/v1"
            or receipt.get("rows") != expected_rows
            or receipt.get("valid_rows") != expected_valid_rows
            or receipt.get("unique_identities") != expected_rows
            or receipt.get("manifest_order") != observed_names
            or not isinstance(invariants, Mapping)
            or any(value is not True for value in invariants.values())
            or not isinstance(selection, Mapping)
            or selection.get("model_score_free") is not True
            or selection.get("same_image_and_category_only") is not True
            or selection.get("same_manifest_only") is not True
            or selection.get("max_target_iou_exclusive") != 0.3
        ):
            raise RuntimeError("official-assignment full receipt contract drifted")
        forbidden_inputs = set(selection.get("forbidden_inputs") or [])
        if forbidden_inputs != {
            "teacher_scores",
            "teacher_logits",
            "model_scores",
            "model_logits",
            "checkpoint_outputs",
        }:
            raise RuntimeError(
                "official-assignment full receipt model-input prohibition drifted"
            )
        receipt_manifests = receipt.get("manifests")
        if not isinstance(receipt_manifests, Mapping):
            raise RuntimeError(
                "official-assignment full receipt has no manifest records"
            )
        for name, expected_sha in expected_manifest_shas.items():
            manifest = receipt_manifests.get(name)
            output = (
                manifest.get("output") if isinstance(manifest, Mapping) else None
            )
            if (
                    not isinstance(output, Mapping)
                    or output.get("sha256") != expected_sha
                    or _expand_stage_b_data_driven_path(
                        str(output.get("path", ""))
                    ).resolve(strict=True)
                    != observed_paths[name]
            ):
                raise RuntimeError(
                    f"official-assignment full receipt manifest {name!r} drifted"
                )
    else:
        invariants = receipt.get("invariants")
        selection = receipt.get("selection_contract")
        support = receipt.get("support")
        heldout = receipt.get("heldout")
        upstream = receipt.get("upstream_assignment_receipt")
        upstream_category = receipt.get("upstream_category_complete_receipt")
        source_counts = {
            "refcoco_stageb_phrase_v1.jsonl": 22,
            "refcocoplus_stageb_phrase_v1.jsonl": 21,
            "refcocog_stageb_phrase_v1.jsonl": 21,
        }
        if (
            receipt.get("schema")
            != "pivot.stageb.data_driven.assignment_overfit64_receipt/v1"
            or receipt.get("row_schema")
            != "pivot.stageb.data_driven.official_assignment_pair/v1"
            or receipt.get("rows") != expected_rows
            or expected_rows != 64
            or expected_valid_rows != 64
            or receipt.get("invalid_rows") != 0
            or receipt.get("output_manifest") != "overfit64.jsonl"
            or observed_names != ["overfit64.jsonl"]
            or receipt.get("unique_images") != 64
            or receipt.get("unique_annotation_endpoints") != 128
            or receipt.get("unique_unordered_annotation_edges") != 64
            or receipt.get("source_manifest_order") != list(source_counts)
            or receipt.get("source_counts") != source_counts
            or not isinstance(receipt.get("members"), list)
            or len(receipt["members"]) != 64
            or not isinstance(invariants, Mapping)
            or any(value is not True for value in invariants.values())
            or not isinstance(selection, Mapping)
            or selection.get("model_score_free") is not True
            or selection.get("all_eight_official_ref_splits_are_heldout") is not True
            or selection.get("external_clean_support_required") is not True
            or selection.get("target_crop_fallback_allowed") is not False
            or selection.get("runtime_support_candidates_per_selected_class") != 1
            or selection.get("runtime_support_source")
            != "overfit64_support_clean.tsv"
        ):
            raise RuntimeError("official-assignment Overfit64 receipt contract drifted")
        if set(selection.get("forbidden_inputs") or []) != {
            "teacher_scores",
            "teacher_logits",
            "model_scores",
            "model_logits",
            "checkpoint_outputs",
        }:
            raise RuntimeError(
                "official-assignment Overfit64 model-input prohibition drifted"
            )
        output = receipt.get("output")
        only_name = observed_names[0]
        if (
            not isinstance(output, Mapping)
            or output.get("sha256") != expected_manifest_shas[only_name]
            or _expand_stage_b_data_driven_path(
                str(output.get("path", ""))
            ).resolve(strict=True)
            != observed_paths[only_name]
        ):
            raise RuntimeError("official-assignment Overfit64 output binding drifted")

        support_path_value = str(
            getattr(
                args,
                "stage_b_data_driven_assignment_overfit_support_tsv_path",
                "",
            )
            or ""
        ).strip()
        support_sha = str(
            getattr(
                args,
                "stage_b_data_driven_assignment_overfit_support_tsv_sha256",
                "",
            )
            or ""
        ).strip()
        if not support_path_value or not _is_stage_b_data_driven_sha256(support_sha):
            raise RuntimeError("official-assignment Overfit64 support binding is incomplete")
        support_path = _expand_stage_b_data_driven_path(
            support_path_value
        ).resolve(strict=True)
        support_record = support.get("mini_support_tsv") if isinstance(support, Mapping) else None
        dataset_row = train_rows[0]
        if (
            _sha256_file(support_path) != support_sha
            or not isinstance(support_record, Mapping)
            or support_record.get("sha256") != support_sha
            or _expand_stage_b_data_driven_path(
                str(support_record.get("path", ""))
            ).resolve(strict=True)
            != support_path
            or support.get("mini_support_rows") != 25
            or support.get("mini_support_candidates_per_class") != 1
            or support.get("target_crop_fallback_allowed") is not False
            or _expand_stage_b_data_driven_path(
                str(dataset_row.get("support_patch_tsv", ""))
            ).resolve(strict=True)
            != support_path
            or dataset_row.get("support_patch_bucket") != "clean"
            or dataset_row.get("support_patch_use_embedding") is not False
            or dataset_row.get("support_patch_max_per_class") != 1
            or dataset_row.get("patch_bank_cache") is not False
            or dataset_row.get("patch_bank_cache_write") is not False
        ):
            raise RuntimeError("official-assignment Overfit64 support contract drifted")

        expected_member_stream = str(
            getattr(
                args,
                "stage_b_data_driven_assignment_overfit_member_stream_sha256",
                "",
            )
            or ""
        ).strip()
        expected_heldout_sha = str(
            getattr(
                args,
                "stage_b_data_driven_assignment_overfit_heldout_sha256",
                "",
            )
            or ""
        ).strip()
        if (
            not _is_stage_b_data_driven_sha256(expected_member_stream)
            or receipt.get("ordered_member_pair_id_stream_sha256")
            != expected_member_stream
            or not _is_stage_b_data_driven_sha256(expected_heldout_sha)
            or not isinstance(heldout, Mapping)
            or heldout.get("rows") != 57457
            or heldout.get("unique_images") != 6549
            or heldout.get("sorted_image_id_json_sha256") != expected_heldout_sha
            or not isinstance(upstream, Mapping)
            or upstream.get("sha256")
            != "7b9ce1c911a2e1f0b67464243df8290fc2baf0786a2a3b131ddc57a6a6d2ddaa"
            or not isinstance(upstream_category, Mapping)
            or upstream_category.get("sha256")
            != "fab09c61a8f53f05d75eedff25039a843ff27cb2d491d6c6576fe2b1e8aedd74"
        ):
            raise RuntimeError("official-assignment Overfit64 lineage contract drifted")

        witnesses = support.get("selected_class_witnesses")
        if not isinstance(witnesses, list) or len(witnesses) != 25:
            raise RuntimeError("official-assignment Overfit64 support witnesses drifted")
        witness_classes = set()
        for witness in witnesses:
            image = witness.get("image") if isinstance(witness, Mapping) else None
            class_id = witness.get("class_id") if isinstance(witness, Mapping) else None
            if (
                not isinstance(class_id, int)
                or class_id in witness_classes
                or not isinstance(image, Mapping)
                or not _is_stage_b_data_driven_sha256(image.get("sha256"))
            ):
                raise RuntimeError("official-assignment Overfit64 support witnesses drifted")
            image_path = _expand_stage_b_data_driven_path(
                str(image.get("path", ""))
            ).resolve(strict=True)
            if _sha256_file(image_path) != image["sha256"]:
                raise RuntimeError("official-assignment Overfit64 witness image drifted")
            witness_classes.add(class_id)

    args.stage_b_data_driven_assignment_dataset_binding = {
        "scope": scope,
        "dataset_config": {
            "path": str(resolved_dataset_path),
            "sha256": expected_dataset_sha,
        },
        "receipt": {
            "path": str(receipt_path),
            "sha256": expected_receipt_sha,
            "schema": receipt["schema"],
        },
        "manifests": [
            {
                "path": str(observed_paths[name]),
                "sha256": expected_manifest_shas[name],
            }
            for name in observed_names
        ],
        "rows": expected_rows,
        "valid_rows": expected_valid_rows,
    }


def _stage_b_data_driven_manifest_sha256(records: Any) -> str:
    payload = json.dumps(
        records,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _validate_stage_b_data_driven_formal_evidence_payloads(
    *,
    variant_id: str,
    probe_payload: Any,
    gate_payload: Any,
    expected_output: str,
) -> str:
    contracts = {
        "DD1-H": {
            "probe_schema": (
                "pivot.stageb.data_driven.dd1_h_strict_probe_receipt/v2"
            ),
            "gate_schema": "pivot.stageb.data_driven.dd1_h_formal_gate/v1",
            "rank_supervision_contract_id": 2,
        },
        "DD1-HC": {
            "probe_schema": (
                "pivot.stageb.data_driven.dd1_hc_gap3_coverage_probe_receipt/v1"
            ),
            "gate_schema": "pivot.stageb.data_driven.dd1_hc_formal_gate/v1",
            "rank_supervision_contract_id": 3,
        },
        _STAGE_B_DATA_DRIVEN_PAIRTOP1_HARDGAP3_VARIANT: {
            "probe_schema": (
                "pivot.stageb.data_driven.pairtop1_hardgap3_probe_receipt/v1"
            ),
            "gate_schema": (
                "pivot.stageb.data_driven.pairtop1_hardgap3_formal_gate/v1"
            ),
            "rank_supervision_contract_id": 4,
        },
    }
    if variant_id not in contracts:
        raise RuntimeError(f"unsupported formal evidence variant: {variant_id!r}")
    contract = contracts[variant_id]
    checkpoint = (
        probe_payload.get("checkpoint")
        if isinstance(probe_payload, Mapping)
        else None
    )
    criterion = (
        probe_payload.get("criterion")
        if isinstance(probe_payload, Mapping)
        else None
    )
    source = (
        probe_payload.get("source")
        if isinstance(probe_payload, Mapping)
        else None
    )
    code_manifest_sha256 = (
        source.get("training_code_files_manifest_sha256")
        if isinstance(source, Mapping)
        else None
    )
    if (
        not isinstance(probe_payload, Mapping)
        or probe_payload.get("schema") != contract["probe_schema"]
        or probe_payload.get("status") != "passed"
        or probe_payload.get("scope")
        != "memory_and_protocol_probe_only_do_not_resume_into_formal"
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("optimizer_updates") != 50
        or checkpoint.get("iteration") != 50
        or checkpoint.get("checkpoint_reason") != "max_train_iters"
        or not isinstance(criterion, Mapping)
        or criterion.get("rank_supervision_contract_id")
        != contract["rank_supervision_contract_id"]
        or not isinstance(code_manifest_sha256, str)
        or len(code_manifest_sha256) != 64
    ):
        raise RuntimeError(f"{variant_id} strict U50 probe receipt drifted")
    if variant_id == "DD1-HC":
        causal = probe_payload.get("causal_audit")
        invariants = probe_payload.get("invariants")
        if (
            not isinstance(causal, Mapping)
            or causal.get("all_non_rank_model_tensors_bitwise_equal_to_dd1_h")
            is not True
            or causal.get("patch_optimizer_state_bitwise_equal_to_dd1_h")
            is not True
            or not isinstance(invariants, Mapping)
            or invariants.get("gap3_coverage_mask_uses_detached_patch_scores")
            is not True
            or invariants.get("probe_checkpoint_is_forbidden_as_formal_resume_source")
            is not True
        ):
            raise RuntimeError("DD1-HC strict U50 causal audit drifted")
    if variant_id == _STAGE_B_DATA_DRIVEN_PAIRTOP1_HARDGAP3_VARIANT:
        data = probe_payload.get("data")
        fresh_start = probe_payload.get("fresh_start")
        causal = probe_payload.get("causal_audit")
        invariants = probe_payload.get("invariants")
        if (
            criterion.get("criterion_contract_version") != 4
            or criterion.get("assignment_weight") != 1.0
            or criterion.get("deployment_weight") != 1.0
            or not isinstance(data, Mapping)
            or data.get("scope") != _STAGE_B_DATA_DRIVEN_ASSIGNMENT_FULL_SCOPE
            or data.get("rows") != 321327
            or not isinstance(fresh_start, Mapping)
            or fresh_start.get("resume") != ""
            or not isinstance(causal, Mapping)
            or causal.get("all_non_rank_model_tensors_bitwise_equal_to_pairtop1")
            is not True
            or causal.get("patch_optimizer_state_bitwise_equal_to_pairtop1")
            is not True
            or not isinstance(invariants, Mapping)
            or invariants.get("fifty_of_fifty_optimizer_updates_succeeded")
            is not True
            or invariants.get("amp_step_skips_zero") is not True
            or invariants.get("all_model_and_optimizer_tensors_are_finite")
            is not True
            or invariants.get("no_teacher_logits_weights_or_loss_targets_are_used")
            is not True
            or invariants.get("probe_checkpoint_is_forbidden_as_formal_resume_source")
            is not True
        ):
            raise RuntimeError(
                "DD1-PairTop1-HardGap3 strict U50 causal audit drifted"
            )

    training = (
        gate_payload.get("training")
        if isinstance(gate_payload, Mapping)
        else None
    )
    headline_evaluation = (
        gate_payload.get("headline_evaluation")
        if isinstance(gate_payload, Mapping)
        else None
    )
    headline_gate = (
        gate_payload.get("headline_gate")
        if isinstance(gate_payload, Mapping)
        else None
    )
    if (
        not isinstance(gate_payload, Mapping)
        or gate_payload.get("schema") != contract["gate_schema"]
        or gate_payload.get("status") != "sealed_before_training"
        or not isinstance(training, Mapping)
        or training.get("variant_id") != variant_id
        or training.get("output_dir") != expected_output
        or training.get("optimizer_updates") != 5020
        or not isinstance(headline_evaluation, Mapping)
        or headline_evaluation.get("score_route")
        != "patch_category_gate_then_full_text_rank"
        or headline_evaluation.get("category_gate_max_gap") != 3.0
        or not isinstance(headline_gate, Mapping)
        or headline_gate.get("minimum_correct") != 3943
        or headline_gate.get("total") != 4896
    ):
        raise RuntimeError(f"{variant_id} formal gate contract drifted")
    return code_manifest_sha256


_STAGE_B_NATIVE_PATCH_D1_U500_SCOPE = "native_patch_category_d1_u500_v1"
_STAGE_B_NATIVE_PATCH_D2_U500_SCOPE = "native_patch_category_d2_u500_v1"
_STAGE_B_NATIVE_PATCH_D3_U200_SCOPE = "native_patch_category_d3_u200_v1"
_STAGE_B_NATIVE_PATCH_D4_U200_SCOPE = "native_patch_category_d4_u200_v1"
_STAGE_B_NATIVE_PATCH_D5_U100_SCOPE = "native_patch_category_d5_u100_v1"
_STAGE_B_NATIVE_PATCH_D6_U100_SCOPE = "native_patch_category_d6_u100_v1"
_STAGE_B_NATIVE_PATCH_D7_U100_SCOPE = "native_patch_category_d7_u100_v1"
_STAGE_B_NATIVE_PATCH_D8_U100_SCOPE = "native_patch_category_d8_u100_v1"
_STAGE_B_NATIVE_PATCH_D9_U100_SCOPE = "native_patch_category_d9_u100_v1"
_STAGE_B_NATIVE_PATCH_TRAINABLE_PREFIXES = (
    "patch_encoder.input_proj.",
    "patch_encoder.norm.",
    "query_proj_for_patch.",
)


def _bind_stage_b_native_patch_runtime_inputs(args) -> None:
    scope = str(
        getattr(args, "stage_b_native_patch_execution_scope", "") or ""
    ).strip()
    enabled = bool(getattr(args, "stage_b_native_patch_category", False))
    if not enabled:
        if scope:
            raise RuntimeError(
                "native patch-category execution scope requires its model mode"
            )
        return
    if not scope:
        return
    if scope not in {
        _STAGE_B_NATIVE_PATCH_D1_U500_SCOPE,
        _STAGE_B_NATIVE_PATCH_D2_U500_SCOPE,
        _STAGE_B_NATIVE_PATCH_D3_U200_SCOPE,
        _STAGE_B_NATIVE_PATCH_D4_U200_SCOPE,
        _STAGE_B_NATIVE_PATCH_D5_U100_SCOPE,
        _STAGE_B_NATIVE_PATCH_D6_U100_SCOPE,
        _STAGE_B_NATIVE_PATCH_D7_U100_SCOPE,
        _STAGE_B_NATIVE_PATCH_D8_U100_SCOPE,
        _STAGE_B_NATIVE_PATCH_D9_U100_SCOPE,
    }:
        raise RuntimeError(
            f"unknown native patch-category execution scope: {scope!r}"
        )
    d2_scope = scope == _STAGE_B_NATIVE_PATCH_D2_U500_SCOPE
    d3_scope = scope == _STAGE_B_NATIVE_PATCH_D3_U200_SCOPE
    d4_scope = scope == _STAGE_B_NATIVE_PATCH_D4_U200_SCOPE
    d5_scope = scope == _STAGE_B_NATIVE_PATCH_D5_U100_SCOPE
    d6_scope = scope == _STAGE_B_NATIVE_PATCH_D6_U100_SCOPE
    d7_scope = scope == _STAGE_B_NATIVE_PATCH_D7_U100_SCOPE
    d8_scope = scope == _STAGE_B_NATIVE_PATCH_D8_U100_SCOPE
    d9_scope = scope == _STAGE_B_NATIVE_PATCH_D9_U100_SCOPE
    continuation_scope = (
        d2_scope
        or d3_scope
        or d4_scope
        or d5_scope
        or d6_scope
        or d7_scope
        or d8_scope
        or d9_scope
    )
    scope_label = (
        "D9 U100"
        if d9_scope
        else "D8 U100"
        if d8_scope
        else "D7 U100"
        if d7_scope
        else "D6 U100"
        if d6_scope
        else "D5 U100"
        if d5_scope
        else "D4 U200"
        if d4_scope
        else "D3 U200"
        if d3_scope
        else "D2 U500"
        if d2_scope
        else "D1 U500"
    )
    if bool(getattr(args, "eval", False)):
        raise RuntimeError(
            f"native patch-category {scope_label} scope is training-only"
        )
    resume_requested = str(getattr(args, "resume", "") or "").strip()
    if resume_requested and not continuation_scope:
        raise RuntimeError(
            f"native patch-category {scope_label} must start with a fresh optimizer"
        )
    if d2_scope and not (
        getattr(args, "stage_b_native_patch_contract_version", None) == 2
        and str(
            getattr(args, "stage_b_native_patch_objective", "") or ""
        ).strip().lower()
        == "d2_gate_aligned"
    ):
        raise RuntimeError("native patch-category D2 objective contract drifted")
    if d2_scope:
        expected_d2_objective = {
            "lr": 1e-4,
            "stage_b_native_patch_lr": 1e-4,
            "stage_b_native_patch_positive_iou_threshold": 0.5,
            "stage_b_native_patch_negative_iou_threshold": 0.3,
            "stage_b_native_patch_d2_weight": 1.0,
            "stage_b_native_patch_gate_max_gap": 3.0,
            "stage_b_native_patch_score_clip": 5.0,
            "stage_b_native_patch_d2_keep_gap": 2.75,
            "stage_b_native_patch_d2_drop_gap": 3.25,
            "stage_b_native_patch_d2_temperature": 0.25,
            "stage_b_native_patch_d2_native_hard_negatives": 16,
            "stage_b_native_patch_d2_patch_hard_negatives": 4,
            "stage_b_native_patch_d2_keep_weight": 2.0,
            "stage_b_native_patch_d2_drop_weight": 1.0,
            "stage_b_native_patch_d2_coverage_weight": 0.25,
        }
        drifted_d2_objective = {
            key: (getattr(args, key, None), expected)
            for key, expected in expected_d2_objective.items()
            if type(getattr(args, key, None)) is not type(expected)
            or getattr(args, key, None) != expected
        }
        if drifted_d2_objective:
            raise RuntimeError(
                "native patch-category D2 gate-aligned objective drifted: "
                f"{drifted_d2_objective}"
            )
    if d3_scope and not (
        getattr(args, "stage_b_native_patch_contract_version", None) == 3
        and str(
            getattr(args, "stage_b_native_patch_objective", "") or ""
        ).strip().lower()
        == "d3_critical_winner"
    ):
        raise RuntimeError("native patch-category D3 objective contract drifted")
    if d3_scope:
        expected_d3_objective = {
            "lr": 5e-5,
            "stage_b_native_patch_lr": 5e-5,
            "amp_init_scale": 16.0,
            "stage_b_native_patch_positive_iou_threshold": 0.5,
            "stage_b_native_patch_negative_iou_threshold": 0.3,
            "stage_b_native_patch_d3_weight": 1.0,
            "stage_b_native_patch_gate_max_gap": 3.0,
            "stage_b_native_patch_score_clip": 5.0,
            "stage_b_native_patch_d3_keep_gap": 2.75,
            "stage_b_native_patch_d3_separation_gap": 3.25,
            "stage_b_native_patch_d3_temperature": 0.25,
            "stage_b_native_patch_d3_critical_weight": 2.0,
            "stage_b_native_patch_d3_critical_keep_weight": 1.0,
            "stage_b_native_patch_d3_positive_keep_weight": 1.0,
        }
        drifted_d3_objective = {
            key: (getattr(args, key, None), expected)
            for key, expected in expected_d3_objective.items()
            if type(getattr(args, key, None)) is not type(expected)
            or getattr(args, key, None) != expected
        }
        if drifted_d3_objective:
            raise RuntimeError(
                "native patch-category D3 critical-winner objective drifted: "
                f"{drifted_d3_objective}"
            )
    if d4_scope and not (
        getattr(args, "stage_b_native_patch_contract_version", None) == 4
        and str(
            getattr(args, "stage_b_native_patch_objective", "") or ""
        ).strip().lower()
        == "d4_positive_protected_critical_winner"
    ):
        raise RuntimeError("native patch-category D4 objective contract drifted")
    if d4_scope:
        expected_d4_objective = {
            "lr": 5e-5,
            "stage_b_native_patch_lr": 5e-5,
            "amp_init_scale": 8.0,
            "stage_b_native_patch_positive_iou_threshold": 0.5,
            "stage_b_native_patch_negative_iou_threshold": 0.3,
            "stage_b_native_patch_d4_weight": 1.0,
            "stage_b_native_patch_gate_max_gap": 3.0,
            "stage_b_native_patch_score_clip": 5.0,
            "stage_b_native_patch_d4_keep_gap": 2.75,
            "stage_b_native_patch_d4_separation_gap": 3.25,
            "stage_b_native_patch_d4_temperature": 0.25,
            "stage_b_native_patch_d4_critical_weight": 2.0,
            "stage_b_native_patch_d4_critical_keep_weight": 1.0,
            "stage_b_native_patch_d4_positive_keep_weight": 32.0,
        }
        drifted_d4_objective = {
            key: (getattr(args, key, None), expected)
            for key, expected in expected_d4_objective.items()
            if type(getattr(args, key, None)) is not type(expected)
            or getattr(args, key, None) != expected
        }
        if drifted_d4_objective:
            raise RuntimeError(
                "native patch-category D4 positive-protected critical-winner "
                f"objective drifted: {drifted_d4_objective}"
            )
    if d5_scope and not (
        getattr(args, "stage_b_native_patch_contract_version", None) == 5
        and str(
            getattr(args, "stage_b_native_patch_objective", "") or ""
        ).strip().lower()
        == "d5_active_tail_positive_barrier"
    ):
        raise RuntimeError("native patch-category D5 objective contract drifted")
    if d5_scope:
        expected_d5_objective = {
            "lr": 5e-5,
            "stage_b_native_patch_lr": 5e-5,
            "amp_init_scale": 8.0,
            "stage_b_native_patch_positive_iou_threshold": 0.5,
            "stage_b_native_patch_negative_iou_threshold": 0.3,
            "stage_b_native_patch_d5_weight": 1.0,
            "stage_b_native_patch_gate_max_gap": 3.0,
            "stage_b_native_patch_score_clip": 5.0,
            "stage_b_native_patch_d5_keep_gap": 2.75,
            "stage_b_native_patch_d5_separation_gap": 3.25,
            "stage_b_native_patch_d5_temperature": 0.25,
            "stage_b_native_patch_d5_critical_weight": 2.0,
            "stage_b_native_patch_d5_critical_keep_weight": 1.0,
            "stage_b_native_patch_d5_active_gap": 2.0,
            "stage_b_native_patch_d5_target_gap": 2.5,
            "stage_b_native_patch_d5_positive_barrier_weight": 2.0,
        }
        drifted_d5_objective = {
            key: (getattr(args, key, None), expected)
            for key, expected in expected_d5_objective.items()
            if type(getattr(args, key, None)) is not type(expected)
            or getattr(args, key, None) != expected
        }
        if drifted_d5_objective:
            raise RuntimeError(
                "native patch-category D5 active-tail positive-barrier "
                f"objective drifted: {drifted_d5_objective}"
            )
    if d6_scope and not (
        getattr(args, "stage_b_native_patch_contract_version", None) == 6
        and str(
            getattr(args, "stage_b_native_patch_objective", "") or ""
        ).strip().lower()
        == "d6_direct_deployment_gap"
    ):
        raise RuntimeError("native patch-category D6 objective contract drifted")
    if d6_scope:
        expected_d6_objective = {
            "lr": 5e-5,
            "stage_b_native_patch_lr": 5e-5,
            "amp_init_scale": 8.0,
            "save_checkpoint_interval": 100,
            "stage_b_native_patch_positive_iou_threshold": 0.5,
            "stage_b_native_patch_negative_iou_threshold": 0.3,
            "stage_b_native_patch_d6_weight": 1.0,
            "stage_b_native_patch_gate_max_gap": 3.0,
            "stage_b_native_patch_score_clip": 5.0,
            "stage_b_native_patch_d6_keep_gap": 2.75,
            "stage_b_native_patch_d6_drop_gap": 3.25,
            "stage_b_native_patch_d6_drop_active_gap": 3.75,
            "stage_b_native_patch_d6_temperature": 0.25,
            "stage_b_native_patch_d6_drop_weight": 2.0,
            "stage_b_native_patch_d6_critical_keep_weight": 1.0,
            "stage_b_native_patch_d6_positive_active_gap": 2.0,
            "stage_b_native_patch_d6_positive_target_gap": 2.5,
            "stage_b_native_patch_d6_positive_barrier_weight": 2.0,
        }
        drifted_d6_objective = {
            key: (getattr(args, key, None), expected)
            for key, expected in expected_d6_objective.items()
            if type(getattr(args, key, None)) is not type(expected)
            or getattr(args, key, None) != expected
        }
        if drifted_d6_objective:
            raise RuntimeError(
                "native patch-category D6 direct deployment-gap objective drifted: "
                f"{drifted_d6_objective}"
            )
    if d7_scope and not (
        getattr(args, "stage_b_native_patch_contract_version", None) == 7
        and str(
            getattr(args, "stage_b_native_patch_objective", "") or ""
        ).strip().lower()
        == "d7_all_state_positive_anchor"
    ):
        raise RuntimeError("native patch-category D7 objective contract drifted")
    if d7_scope:
        expected_d7_objective = {
            "lr": 5e-5,
            "stage_b_native_patch_lr": 5e-5,
            "amp_init_scale": 8.0,
            "save_checkpoint_interval": 100,
            "stage_b_native_patch_positive_iou_threshold": 0.5,
            "stage_b_native_patch_negative_iou_threshold": 0.3,
            "stage_b_native_patch_d7_weight": 1.0,
            "stage_b_native_patch_gate_max_gap": 3.0,
            "stage_b_native_patch_score_clip": 5.0,
            "stage_b_native_patch_d7_keep_gap": 2.75,
            "stage_b_native_patch_d7_drop_gap": 3.25,
            "stage_b_native_patch_d7_drop_active_gap": 3.75,
            "stage_b_native_patch_d7_temperature": 0.25,
            "stage_b_native_patch_d7_drop_weight": 2.0,
            "stage_b_native_patch_d7_critical_keep_weight": 1.0,
            "stage_b_native_patch_d7_positive_active_gap": 2.0,
            "stage_b_native_patch_d7_positive_target_gap": 2.5,
            "stage_b_native_patch_d7_positive_barrier_weight": 2.0,
            "stage_b_native_patch_d7_anchor_active_gap": 2.0,
            "stage_b_native_patch_d7_anchor_target_gap": 2.5,
            "stage_b_native_patch_d7_anchor_weight": 2.0,
        }
        drifted_d7_objective = {
            key: (getattr(args, key, None), expected)
            for key, expected in expected_d7_objective.items()
            if type(getattr(args, key, None)) is not type(expected)
            or getattr(args, key, None) != expected
        }
        if drifted_d7_objective:
            raise RuntimeError(
                "native patch-category D7 all-state positive-anchor objective "
                f"drifted: {drifted_d7_objective}"
            )
    if d8_scope and not (
        getattr(args, "stage_b_native_patch_contract_version", None) == 8
        and str(
            getattr(args, "stage_b_native_patch_objective", "") or ""
        ).strip().lower()
        == "d8_state_class_macro_anchor"
    ):
        raise RuntimeError("native patch-category D8 objective contract drifted")
    if d8_scope:
        expected_d8_objective = {
            "lr": 5e-5,
            "stage_b_native_patch_lr": 5e-5,
            "amp_init_scale": 8.0,
            "save_checkpoint_interval": 100,
            "stage_b_native_patch_positive_iou_threshold": 0.5,
            "stage_b_native_patch_negative_iou_threshold": 0.3,
            "stage_b_native_patch_d8_weight": 1.0,
            "stage_b_native_patch_gate_max_gap": 3.0,
            "stage_b_native_patch_score_clip": 5.0,
            "stage_b_native_patch_d8_keep_gap": 2.75,
            "stage_b_native_patch_d8_drop_gap": 3.25,
            "stage_b_native_patch_d8_drop_active_gap": 3.75,
            "stage_b_native_patch_d8_temperature": 0.25,
            "stage_b_native_patch_d8_drop_weight": 2.0,
            "stage_b_native_patch_d8_critical_keep_weight": 1.0,
            "stage_b_native_patch_d8_positive_active_gap": 2.0,
            "stage_b_native_patch_d8_positive_target_gap": 2.5,
            "stage_b_native_patch_d8_positive_barrier_weight": 2.0,
            "stage_b_native_patch_d8_anchor_active_gap": 2.0,
            "stage_b_native_patch_d8_anchor_target_gap": 2.5,
            "stage_b_native_patch_d8_anchor_negative_weight": 1.0,
            "stage_b_native_patch_d8_anchor_neutral_weight": 2.0,
            "stage_b_native_patch_d8_anchor_positive_weight": 4.0,
        }
        drifted_d8_objective = {
            key: (getattr(args, key, None), expected)
            for key, expected in expected_d8_objective.items()
            if type(getattr(args, key, None)) is not type(expected)
            or getattr(args, key, None) != expected
        }
        if drifted_d8_objective:
            raise RuntimeError(
                "native patch-category D8 state-class macro-anchor objective "
                f"drifted: {drifted_d8_objective}"
            )
    if d9_scope and not (
        getattr(args, "stage_b_native_patch_contract_version", None) == 9
        and str(
            getattr(args, "stage_b_native_patch_objective", "") or ""
        ).strip().lower()
        == "d9_loss_gradient_localized"
    ):
        raise RuntimeError("native patch-category D9 objective contract drifted")
    if d9_scope:
        expected_d9_objective = {
            "lr": 5e-5,
            "stage_b_native_patch_lr": 5e-5,
            "amp_init_scale": 8.0,
            "save_checkpoint_interval": 100,
            "stage_b_native_patch_positive_iou_threshold": 0.5,
            "stage_b_native_patch_negative_iou_threshold": 0.3,
            "stage_b_native_patch_d8_weight": 1.0,
            "stage_b_native_patch_gate_max_gap": 3.0,
            "stage_b_native_patch_score_clip": 5.0,
            "stage_b_native_patch_d8_keep_gap": 2.75,
            "stage_b_native_patch_d8_drop_gap": 3.25,
            "stage_b_native_patch_d8_drop_active_gap": 3.75,
            "stage_b_native_patch_d8_temperature": 0.25,
            "stage_b_native_patch_d8_drop_weight": 2.0,
            "stage_b_native_patch_d8_critical_keep_weight": 1.0,
            "stage_b_native_patch_d8_positive_active_gap": 2.0,
            "stage_b_native_patch_d8_positive_target_gap": 2.5,
            "stage_b_native_patch_d8_positive_barrier_weight": 2.0,
            "stage_b_native_patch_d8_anchor_active_gap": 2.0,
            "stage_b_native_patch_d8_anchor_target_gap": 2.5,
            "stage_b_native_patch_d8_anchor_negative_weight": 1.0,
            "stage_b_native_patch_d8_anchor_neutral_weight": 2.0,
            "stage_b_native_patch_d8_anchor_positive_weight": 4.0,
            "stage_b_native_patch_d9_detach_row_stats": True,
        }
        drifted_d9_objective = {
            key: (getattr(args, key, None), expected)
            for key, expected in expected_d9_objective.items()
            if type(getattr(args, key, None)) is not type(expected)
            or getattr(args, key, None) != expected
        }
        if drifted_d9_objective:
            raise RuntimeError(
                "native patch-category D9 loss-gradient-localized objective "
                f"drifted: {drifted_d9_objective}"
            )

    config_path = Path(args.config_file).expanduser().resolve(strict=True)
    dataset_path = Path(args.datasets).expanduser().resolve(strict=True)
    initializer_path = Path(
        str(getattr(args, "pretrain_model_path", "") or "")
    ).expanduser().resolve(strict=True)
    expected_paths = {
        "config": getattr(args, "stage_b_native_patch_formal_config_path", ""),
        "dataset": getattr(args, "stage_b_native_patch_dataset_config_path", ""),
        "initializer": getattr(
            args, "stage_b_native_patch_initializer_path", ""
        ),
    }
    observed_paths = {
        "config": config_path,
        "dataset": dataset_path,
        "initializer": initializer_path,
    }
    for label, expected_value in expected_paths.items():
        expected = Path(str(expected_value or "")).expanduser().resolve(
            strict=True
        )
        if observed_paths[label] != expected:
            raise RuntimeError(
                f"native patch-category {scope_label} {label} path drifted"
            )

    expected_dataset_sha = str(
        getattr(args, "stage_b_native_patch_dataset_config_sha256", "") or ""
    ).strip()
    observed_dataset_sha = _sha256_file(dataset_path)
    if (
        len(expected_dataset_sha) != 64
        or observed_dataset_sha != expected_dataset_sha
    ):
        raise RuntimeError(
            f"native patch-category {scope_label} dataset config SHA256 drifted"
        )
    continuation_base_initializer_record = None
    if continuation_scope:
        continuation_label = (
            "D9"
            if d9_scope
            else "D8"
            if d8_scope
            else "D7"
            if d7_scope
            else "D6"
            if d6_scope
            else "D5"
            if d5_scope
            else "D4"
            if d4_scope
            else "D3"
            if d3_scope
            else "D2"
        )
        expected_source_sha = str(
            getattr(args, "stage_b_native_patch_initializer_sha256", "") or ""
        ).strip()
        if (
            len(expected_source_sha) != 64
            or _sha256_file(initializer_path) != expected_source_sha
        ):
            raise RuntimeError(
                f"native patch-category {continuation_label} D1-source "
                "checkpoint SHA256 drifted"
        )
        base_path_key = (
            "stage_b_native_patch_d9_base_initializer_path"
            if d9_scope
            else "stage_b_native_patch_d8_base_initializer_path"
            if d8_scope
            else "stage_b_native_patch_d7_base_initializer_path"
            if d7_scope
            else "stage_b_native_patch_d6_base_initializer_path"
            if d6_scope
            else "stage_b_native_patch_d5_base_initializer_path"
            if d5_scope
            else "stage_b_native_patch_d4_base_initializer_path"
            if d4_scope
            else "stage_b_native_patch_d3_base_initializer_path"
            if d3_scope
            else "stage_b_native_patch_d2_base_initializer_path"
        )
        base_sha_key = (
            "stage_b_native_patch_d9_base_initializer_sha256"
            if d9_scope
            else "stage_b_native_patch_d8_base_initializer_sha256"
            if d8_scope
            else "stage_b_native_patch_d7_base_initializer_sha256"
            if d7_scope
            else "stage_b_native_patch_d6_base_initializer_sha256"
            if d6_scope
            else "stage_b_native_patch_d5_base_initializer_sha256"
            if d5_scope
            else "stage_b_native_patch_d4_base_initializer_sha256"
            if d4_scope
            else "stage_b_native_patch_d3_base_initializer_sha256"
            if d3_scope
            else "stage_b_native_patch_d2_base_initializer_sha256"
        )
        continuation_base_initializer_path = Path(
            str(
                getattr(args, base_path_key, "")
                or ""
            )
        ).expanduser().resolve(strict=True)
        expected_base_sha = str(
            getattr(args, base_sha_key, "")
            or ""
        ).strip()
        if (
            continuation_base_initializer_path == initializer_path
            or len(expected_base_sha) != 64
            or _sha256_file(continuation_base_initializer_path)
            != expected_base_sha
        ):
            raise RuntimeError(
                f"native patch-category {continuation_label} b58-only "
                "initializer binding drifted"
            )
        continuation_base_initializer_record = _stage_b_data_driven_file_record(
            continuation_base_initializer_path
        )
    expected_output = Path(
        str(getattr(args, "stage_b_native_patch_formal_output_dir", "") or "")
    ).expanduser().resolve()
    if Path(args.output_dir).expanduser().resolve() != expected_output:
        raise RuntimeError(
            f"native patch-category {scope_label} output directory drifted"
        )
    if resume_requested:
        observed_resume = Path(resume_requested).expanduser().resolve(strict=True)
        expected_resume = expected_output / "checkpoint_iter.pth"
        if observed_resume != expected_resume:
            raise RuntimeError(
                f"native patch-category {scope_label} resume must use the exact formal "
                f"checkpoint: {expected_resume}"
            )

    short_continuation_scope = (
        d3_scope
        or d4_scope
        or d5_scope
        or d6_scope
        or d7_scope
        or d8_scope
        or d9_scope
    )
    expected_max_train_iters = (
        100
        if d5_scope or d6_scope or d7_scope or d8_scope or d9_scope
        else (200 if d3_scope or d4_scope else 500)
    )
    expected_iter_checkpoint_interval = (
        50 if short_continuation_scope else (100 if d2_scope else 500)
    )
    required_runtime = {
        "seed": 42,
        "batch_size": 36,
        "epochs": 250,
        "max_train_iters": expected_max_train_iters,
        # Continuations checkpoint frequently so a host interruption cannot
        # erase the trajectory. Saving does not advance the optimizer or RNG.
        "iter_checkpoint_interval": expected_iter_checkpoint_interval,
        "num_workers": 8,
        "prefetch_factor": 1,
        "pin_memory": None,
        "persistent_workers": False,
        "gradient_accumulation_steps": 2,
        "amp": True,
        "world_size": 1,
        "distributed": False,
    }
    drifted = {
        key: (getattr(args, key, None), expected)
        for key, expected in required_runtime.items()
        if type(getattr(args, key, None)) is not type(expected)
        or getattr(args, key, None) != expected
    }
    if drifted:
        raise RuntimeError(
            f"native patch-category {scope_label} runtime drifted: "
            f"{drifted}"
        )
    required_formal_values = {
        "stage_b_native_patch_expected_max_train_iters": expected_max_train_iters,
        "stage_b_native_patch_expected_gradient_accumulation_steps": 2,
        "stage_b_native_patch_expected_num_workers": 8,
        "stage_b_native_patch_expected_seed": 42,
        "stage_b_data_driven_sampling_contract": "deterministic_epoch_ledger_v1",
        "stage_b_data_driven_sampler_seed": 43 if short_continuation_scope else 42,
        "stage_b_data_driven_loader_seed": 1043 if short_continuation_scope else 1042,
    }
    drifted_formal_values = {
        key: (getattr(args, key, None), expected)
        for key, expected in required_formal_values.items()
        if type(getattr(args, key, None)) is not type(expected)
        or getattr(args, key, None) != expected
    }
    if drifted_formal_values:
        raise RuntimeError(
            f"native patch-category {scope_label} formal contract drifted: "
            f"{drifted_formal_values}"
        )
    required_allocator_env = str(
        getattr(args, "stage_b_data_driven_required_allocator_env", "") or ""
    ).strip()
    required_allocator_conf = str(
        getattr(args, "stage_b_data_driven_required_allocator_conf", "") or ""
    ).strip()
    if (
        required_allocator_env != "PYTORCH_CUDA_ALLOC_CONF"
        or required_allocator_conf != "expandable_segments:True"
        or os.environ.get(required_allocator_env) != required_allocator_conf
    ):
        raise RuntimeError(
            f"native patch-category {scope_label} allocator contract drifted"
        )
    args.stage_b_native_patch_runtime_binding = {
        "schema": (
            "pivot.stageb.native_patch_category_d9_runtime/v1"
            if d9_scope
            else "pivot.stageb.native_patch_category_d8_runtime/v1"
            if d8_scope
            else "pivot.stageb.native_patch_category_d7_runtime/v1"
            if d7_scope
            else "pivot.stageb.native_patch_category_d6_runtime/v1"
            if d6_scope
            else "pivot.stageb.native_patch_category_d5_runtime/v1"
            if d5_scope
            else "pivot.stageb.native_patch_category_d4_runtime/v1"
            if d4_scope
            else "pivot.stageb.native_patch_category_d3_runtime/v1"
            if d3_scope
            else "pivot.stageb.native_patch_category_d2_runtime/v1"
            if d2_scope
            else "pivot.stageb.native_patch_category_d1_runtime/v1"
        ),
        "scope": scope,
        "config": _stage_b_data_driven_file_record(config_path),
        "dataset_config": _stage_b_data_driven_file_record(dataset_path),
        "initializer": _stage_b_data_driven_file_record(initializer_path),
        **(
            {"d9_base_initializer": continuation_base_initializer_record}
            if d9_scope
            else {"d8_base_initializer": continuation_base_initializer_record}
            if d8_scope
            else {"d7_base_initializer": continuation_base_initializer_record}
            if d7_scope
            else {"d6_base_initializer": continuation_base_initializer_record}
            if d6_scope
            else {"d5_base_initializer": continuation_base_initializer_record}
            if d5_scope
            else {"d4_base_initializer": continuation_base_initializer_record}
            if d4_scope
            else {"d3_base_initializer": continuation_base_initializer_record}
            if d3_scope
            else {"d2_base_initializer": continuation_base_initializer_record}
        ),
        "output_dir": str(expected_output),
        "runtime": required_runtime,
        "allocator": {
            "environment_variable": required_allocator_env,
            "value": required_allocator_conf,
        },
    }


def _validate_stage_b_native_patch_d2_resume_checkpoint(args, checkpoint) -> None:
    scope = str(
        getattr(args, "stage_b_native_patch_execution_scope", "") or ""
    ).strip()
    d3_scope = scope == _STAGE_B_NATIVE_PATCH_D3_U200_SCOPE
    d4_scope = scope == _STAGE_B_NATIVE_PATCH_D4_U200_SCOPE
    d5_scope = scope == _STAGE_B_NATIVE_PATCH_D5_U100_SCOPE
    d6_scope = scope == _STAGE_B_NATIVE_PATCH_D6_U100_SCOPE
    d7_scope = scope == _STAGE_B_NATIVE_PATCH_D7_U100_SCOPE
    d8_scope = scope == _STAGE_B_NATIVE_PATCH_D8_U100_SCOPE
    d9_scope = scope == _STAGE_B_NATIVE_PATCH_D9_U100_SCOPE
    if scope not in {
        "",
        _STAGE_B_NATIVE_PATCH_D2_U500_SCOPE,
        _STAGE_B_NATIVE_PATCH_D3_U200_SCOPE,
        _STAGE_B_NATIVE_PATCH_D4_U200_SCOPE,
        _STAGE_B_NATIVE_PATCH_D5_U100_SCOPE,
        _STAGE_B_NATIVE_PATCH_D6_U100_SCOPE,
        _STAGE_B_NATIVE_PATCH_D7_U100_SCOPE,
        _STAGE_B_NATIVE_PATCH_D8_U100_SCOPE,
        _STAGE_B_NATIVE_PATCH_D9_U100_SCOPE,
    }:
        raise RuntimeError(
            "native patch-category strict resume requires a D2-D9 "
            "formal scope"
        )
    contract_label = (
        "D9"
        if d9_scope
        else "D8"
        if d8_scope
        else "D7"
        if d7_scope
        else "D6"
        if d6_scope
        else "D5"
        if d5_scope
        else "D4"
        if d4_scope
        else "D3"
        if d3_scope
        else "D2"
    )
    expected_optimizer_updates = (
        100
        if d5_scope or d6_scope or d7_scope or d8_scope or d9_scope
        else (200 if d3_scope or d4_scope else 500)
    )
    required_keys = {
        "model",
        "criterion",
        "optimizer",
        "lr_scheduler",
        "scaler",
        "epoch",
        "iteration",
        "optimizer_updates",
        "epoch_finished",
        "rng_state",
        "epoch_rng_state",
        "args",
        "checkpoint_reason",
        "stage_b_data_driven_sampling_state",
    }
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(
            f"native patch-category {contract_label} resume payload must be a mapping"
        )
    missing = sorted(required_keys.difference(checkpoint))
    if missing:
        raise RuntimeError(
            f"native patch-category {contract_label} resume is missing complete "
            "training state: "
            f"{missing}"
        )

    epoch = checkpoint["epoch"]
    iteration = checkpoint["iteration"]
    optimizer_updates = checkpoint["optimizer_updates"]
    if (
        type(epoch) is not int
        or epoch != 0
        or type(iteration) is not int
        or type(optimizer_updates) is not int
        or not 0 < optimizer_updates < expected_optimizer_updates
        or iteration != 2 * optimizer_updates
        or checkpoint["epoch_finished"] is not False
        or checkpoint["checkpoint_reason"] not in {"interval", "signal"}
    ):
        raise RuntimeError(
            f"native patch-category {contract_label} resume is not an exact unfinished "
            "optimizer-boundary checkpoint"
        )
    for key in (
        "criterion",
        "optimizer",
        "lr_scheduler",
        "scaler",
        "rng_state",
        "epoch_rng_state",
        "stage_b_data_driven_sampling_state",
    ):
        if not isinstance(checkpoint[key], Mapping):
            raise RuntimeError(
                f"native patch-category {contract_label} resume has invalid {key} state"
            )

    saved_args = checkpoint["args"]
    if not isinstance(saved_args, Mapping):
        raise RuntimeError(
            f"native patch-category {contract_label} resume requires its complete saved args"
        )
    contract_keys = (
        "stage_b_native_patch_category",
        "stage_b_native_patch_contract_version",
        "stage_b_native_patch_objective",
        "stage_b_native_patch_execution_scope",
        "stage_b_native_patch_formal_config_path",
        "stage_b_native_patch_formal_output_dir",
        "stage_b_native_patch_initializer_path",
        "stage_b_native_patch_initializer_sha256",
        "stage_b_native_patch_d2_base_initializer_path",
        "stage_b_native_patch_d2_base_initializer_sha256",
        "stage_b_native_patch_d3_base_initializer_path",
        "stage_b_native_patch_d3_base_initializer_sha256",
        "stage_b_native_patch_d4_base_initializer_path",
        "stage_b_native_patch_d4_base_initializer_sha256",
        "stage_b_native_patch_d5_base_initializer_path",
        "stage_b_native_patch_d5_base_initializer_sha256",
        "stage_b_native_patch_d6_base_initializer_path",
        "stage_b_native_patch_d6_base_initializer_sha256",
        "stage_b_native_patch_d7_base_initializer_path",
        "stage_b_native_patch_d7_base_initializer_sha256",
        "stage_b_native_patch_d8_base_initializer_path",
        "stage_b_native_patch_d8_base_initializer_sha256",
        "stage_b_native_patch_d9_base_initializer_path",
        "stage_b_native_patch_d9_base_initializer_sha256",
        "stage_b_native_patch_dataset_config_path",
        "stage_b_native_patch_dataset_config_sha256",
        "stage_b_native_patch_dataset_manifest_sha256",
        "stage_b_native_patch_dataset_manifest_sha256_by_source",
        "stage_b_native_patch_dataset_receipt_sha256",
        "stage_b_native_patch_dataset_receipt_canonical_sha256",
        "stage_b_native_patch_gate_max_gap",
        "stage_b_native_patch_score_clip",
        "stage_b_native_patch_positive_iou_threshold",
        "stage_b_native_patch_negative_iou_threshold",
        "stage_b_native_patch_d2_weight",
        "stage_b_native_patch_d2_keep_gap",
        "stage_b_native_patch_d2_drop_gap",
        "stage_b_native_patch_d2_temperature",
        "stage_b_native_patch_d2_native_hard_negatives",
        "stage_b_native_patch_d2_patch_hard_negatives",
        "stage_b_native_patch_d2_keep_weight",
        "stage_b_native_patch_d2_drop_weight",
        "stage_b_native_patch_d2_coverage_weight",
        "stage_b_native_patch_d3_weight",
        "stage_b_native_patch_d3_keep_gap",
        "stage_b_native_patch_d3_separation_gap",
        "stage_b_native_patch_d3_temperature",
        "stage_b_native_patch_d3_critical_weight",
        "stage_b_native_patch_d3_critical_keep_weight",
        "stage_b_native_patch_d3_positive_keep_weight",
        "stage_b_native_patch_d4_weight",
        "stage_b_native_patch_d4_keep_gap",
        "stage_b_native_patch_d4_separation_gap",
        "stage_b_native_patch_d4_temperature",
        "stage_b_native_patch_d4_critical_weight",
        "stage_b_native_patch_d4_critical_keep_weight",
        "stage_b_native_patch_d4_positive_keep_weight",
        "stage_b_native_patch_d5_weight",
        "stage_b_native_patch_d5_keep_gap",
        "stage_b_native_patch_d5_separation_gap",
        "stage_b_native_patch_d5_temperature",
        "stage_b_native_patch_d5_critical_weight",
        "stage_b_native_patch_d5_critical_keep_weight",
        "stage_b_native_patch_d5_active_gap",
        "stage_b_native_patch_d5_target_gap",
        "stage_b_native_patch_d5_positive_barrier_weight",
        "stage_b_native_patch_d6_weight",
        "stage_b_native_patch_d6_keep_gap",
        "stage_b_native_patch_d6_drop_gap",
        "stage_b_native_patch_d6_drop_active_gap",
        "stage_b_native_patch_d6_temperature",
        "stage_b_native_patch_d6_drop_weight",
        "stage_b_native_patch_d6_critical_keep_weight",
        "stage_b_native_patch_d6_positive_active_gap",
        "stage_b_native_patch_d6_positive_target_gap",
        "stage_b_native_patch_d6_positive_barrier_weight",
        "stage_b_native_patch_d7_weight",
        "stage_b_native_patch_d7_keep_gap",
        "stage_b_native_patch_d7_drop_gap",
        "stage_b_native_patch_d7_drop_active_gap",
        "stage_b_native_patch_d7_temperature",
        "stage_b_native_patch_d7_drop_weight",
        "stage_b_native_patch_d7_critical_keep_weight",
        "stage_b_native_patch_d7_positive_active_gap",
        "stage_b_native_patch_d7_positive_target_gap",
        "stage_b_native_patch_d7_positive_barrier_weight",
        "stage_b_native_patch_d7_anchor_active_gap",
        "stage_b_native_patch_d7_anchor_target_gap",
        "stage_b_native_patch_d7_anchor_weight",
        "stage_b_native_patch_d8_weight",
        "stage_b_native_patch_d8_keep_gap",
        "stage_b_native_patch_d8_drop_gap",
        "stage_b_native_patch_d8_drop_active_gap",
        "stage_b_native_patch_d8_temperature",
        "stage_b_native_patch_d8_drop_weight",
        "stage_b_native_patch_d8_critical_keep_weight",
        "stage_b_native_patch_d8_positive_active_gap",
        "stage_b_native_patch_d8_positive_target_gap",
        "stage_b_native_patch_d8_positive_barrier_weight",
        "stage_b_native_patch_d8_anchor_active_gap",
        "stage_b_native_patch_d8_anchor_target_gap",
        "stage_b_native_patch_d8_anchor_negative_weight",
        "stage_b_native_patch_d8_anchor_neutral_weight",
        "stage_b_native_patch_d8_anchor_positive_weight",
        "stage_b_native_patch_d9_detach_row_stats",
        "stage_b_data_driven_sampling_contract",
        "stage_b_data_driven_sampler_seed",
        "stage_b_data_driven_loader_seed",
        "stage_b_data_driven_required_allocator_env",
        "stage_b_data_driven_required_allocator_conf",
        "stage_b_native_patch_runtime_binding",
        "config_file",
        "datasets",
        "output_dir",
        "pretrain_model_path",
        "seed",
        "batch_size",
        "epochs",
        "max_train_iters",
        "iter_checkpoint_interval",
        "num_workers",
        "prefetch_factor",
        "pin_memory",
        "persistent_workers",
        "gradient_accumulation_steps",
        "amp",
        "amp_init_scale",
        "world_size",
        "distributed",
        "lr",
        "stage_b_native_patch_lr",
        "weight_decay",
        "clip_max_norm",
        "save_checkpoint_interval",
    )
    drifted = {
        key: (saved_args.get(key), getattr(args, key, None))
        for key in contract_keys
        if saved_args.get(key) != getattr(args, key, None)
    }
    if drifted:
        raise RuntimeError(
            f"native patch-category {contract_label} resume crossed its experiment contract: "
            f"{drifted}"
        )

    def model_state(payload, *, label: str):
        value = payload.get("model") if isinstance(payload, Mapping) else None
        if not isinstance(value, Mapping):
            raise RuntimeError(f"{label} is missing its model state")
        return clean_state_dict(value)

    resume_state = model_state(
        checkpoint, label=f"{contract_label} resume checkpoint"
    )
    initializer_path = Path(
        str(getattr(args, "stage_b_native_patch_initializer_path", "") or "")
    ).expanduser().resolve(strict=True)
    initializer_payload = _torch_load_compat(
        str(initializer_path), map_location="cpu"
    )
    initializer_state = model_state(
        initializer_payload, label=f"{contract_label} D1-source checkpoint"
    )
    if set(resume_state) != set(initializer_state):
        raise RuntimeError(
            f"native patch-category {contract_label} resume model coverage differs "
            "from its D1 source"
        )
    trainable_keys = {
        key
        for key in initializer_state
        if key.startswith(_STAGE_B_NATIVE_PATCH_TRAINABLE_PREFIXES)
    }
    if len(trainable_keys) != 8:
        raise RuntimeError(
            f"native patch-category {contract_label} resume source does not expose "
            "exactly eight "
            "projection tensors"
        )
    changed = set()
    for key, source_tensor in initializer_state.items():
        observed_tensor = resume_state[key]
        if (
            not torch.is_tensor(source_tensor)
            or not torch.is_tensor(observed_tensor)
            or source_tensor.shape != observed_tensor.shape
            or source_tensor.dtype != observed_tensor.dtype
        ):
            raise RuntimeError(
                f"native patch-category {contract_label} resume tensor layout drifted at {key}"
            )
        if (observed_tensor.is_floating_point() or observed_tensor.is_complex()) and not (
            torch.isfinite(observed_tensor).all()
        ):
            raise RuntimeError(
                f"native patch-category {contract_label} resume contains non-finite tensor {key}"
            )
        if not torch.equal(source_tensor, observed_tensor):
            changed.add(key)
    if changed != trainable_keys:
        raise RuntimeError(
            f"native patch-category {contract_label} resume changed tensors outside "
            "or short of its "
            f"eight-tensor surface: {sorted(changed.symmetric_difference(trainable_keys))}"
        )


def _bind_stage_b_data_driven_runtime_inputs(args) -> None:
    execution_scope = str(
        getattr(args, "stage_b_data_driven_execution_scope", "") or ""
    ).strip()
    if not bool(getattr(args, "stage_b_data_driven_score", False)):
        if execution_scope == _STAGE_B_DATA_DRIVEN_NEW_HEAD_FORMAL_SCOPE:
            raise RuntimeError(
                "new-head formal scope requires stage_b_data_driven_score=True"
            )
        return
    from tools.stageb_dependency_audit import config_import_chain

    config_path = Path(args.config_file).resolve(strict=True)
    dataset_path = Path(args.datasets).resolve(strict=True)
    chain = config_import_chain(config_path, root=Path(__file__).resolve().parent)
    args.stage_b_data_driven_config_import_chain = [
        {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path.resolve(strict=True)),
        }
        for path in chain
    ]
    args.stage_b_data_driven_dataset_config = {
        "path": str(dataset_path),
        "sha256": _sha256_file(dataset_path),
    }
    variant_id = str(
        getattr(args, "stage_b_data_driven_variant_id", "") or ""
    ).strip()
    rank_supervision = str(
        getattr(
            args,
            "stage_b_data_driven_rank_supervision",
            "all_nonpositive_negative_v1",
        )
    ).strip().lower()
    if variant_id in {"DD1-H", "DD1-HC"}:
        expected_rank_supervision = {
            "DD1-H": "primary_vs_same_category_aux_v1",
            "DD1-HC": (
                "primary_vs_same_category_aux_plus_gap3_coverage_v1"
            ),
        }[variant_id]
        if (
            rank_supervision != expected_rank_supervision
            or str(
                getattr(args, "stage_b_data_driven_experiment_id", "")
            )
            != "DD1"
            or not bool(
                getattr(args, "stage_b_data_driven_category_complete", False)
            )
        ):
            raise RuntimeError(
                f"{variant_id} requires DD1 category-complete rank supervision "
                f"{expected_rank_supervision!r}"
            )
        if getattr(
            args, "stage_b_data_driven_strict_sample_identity", None
        ) is not True:
            raise RuntimeError(
                f"{variant_id} requires exact strict sample identity"
            )
        if variant_id == "DD1-HC" and (
            float(
                getattr(
                    args,
                    "stage_b_data_driven_category_gate_max_gap",
                    float("nan"),
                )
            )
            != 3.0
            or float(
                getattr(
                    args,
                    "stage_b_data_driven_patch_score_clip",
                    float("nan"),
                )
            )
            != 5.0
        ):
            raise RuntimeError("DD1-HC requires the fixed Gap3/clip5 contract")
        if variant_id == "DD1-H":
            evidence_specs = {
                "control_checkpoint": (
                    "stage_b_data_driven_control_checkpoint_path",
                    "stage_b_data_driven_control_checkpoint_sha256",
                ),
                "control_resolved_args": (
                    "stage_b_data_driven_control_resolved_args_path",
                    "stage_b_data_driven_control_resolved_args_sha256",
                ),
                "control_rank_summary": (
                    "stage_b_data_driven_control_rank_summary_path",
                    "stage_b_data_driven_control_rank_summary_sha256",
                ),
                "control_gap3_summary": (
                    "stage_b_data_driven_control_gap3_summary_path",
                    "stage_b_data_driven_control_gap3_summary_sha256",
                ),
                "control_source_snapshot": (
                    "stage_b_data_driven_control_source_snapshot_path",
                    "stage_b_data_driven_control_source_snapshot_sha256",
                ),
                "control_source_snapshot_supplement": (
                    "stage_b_data_driven_control_source_snapshot_supplement_path",
                    "stage_b_data_driven_control_source_snapshot_supplement_sha256",
                ),
                "control_source_snapshot_supplement_receipt": (
                    "stage_b_data_driven_control_source_snapshot_supplement_receipt_path",
                    "stage_b_data_driven_control_source_snapshot_supplement_receipt_sha256",
                ),
            }
        else:
            evidence_specs = {
                "dd1_h_checkpoint": (
                    "stage_b_data_driven_hc_control_checkpoint_path",
                    "stage_b_data_driven_hc_control_checkpoint_sha256",
                ),
                "dd1_h_formal_result": (
                    "stage_b_data_driven_hc_control_result_path",
                    "stage_b_data_driven_hc_control_result_sha256",
                ),
                "dd1_h_rank_diagnostic_summary": (
                    "stage_b_data_driven_hc_rank_diagnostic_path",
                    "stage_b_data_driven_hc_rank_diagnostic_sha256",
                ),
                "dd1_h_query_diagnostic_analysis": (
                    "stage_b_data_driven_hc_query_diagnostic_path",
                    "stage_b_data_driven_hc_query_diagnostic_sha256",
                ),
                "pre_hc_source_snapshot": (
                    "stage_b_data_driven_hc_source_snapshot_path",
                    "stage_b_data_driven_hc_source_snapshot_sha256",
                ),
            }
        control_evidence = {}
        for label, (path_key, sha_key) in evidence_specs.items():
            path_value = str(getattr(args, path_key, "") or "").strip()
            expected_sha = str(getattr(args, sha_key, "") or "").strip()
            if not path_value or len(expected_sha) != 64:
                raise RuntimeError(
                    f"{variant_id} requires an exact {label} path/SHA binding"
                )
            evidence_path = Path(path_value).resolve(strict=True)
            observed_sha = _sha256_file(evidence_path)
            if observed_sha != expected_sha:
                raise RuntimeError(f"{variant_id} {label} SHA drifted")
            control_evidence[label] = {
                "path": str(evidence_path),
                "sha256": observed_sha,
            }
        args.stage_b_data_driven_control_evidence = control_evidence
    base_path_value = str(
        getattr(args, "stage_b_data_driven_base_initializer_path", "") or ""
    ).strip()
    base_sha = str(
        getattr(args, "stage_b_data_driven_base_initializer_sha256", "") or ""
    ).strip()
    if not base_path_value or len(base_sha) != 64:
        raise RuntimeError(
            "data-driven training requires an exact canonical base initializer"
        )
    base_path = Path(base_path_value).resolve(strict=True)
    observed_base_sha = _sha256_file(base_path)
    if observed_base_sha != base_sha:
        raise RuntimeError("data-driven canonical base initializer SHA drifted")
    args.stage_b_data_driven_base_initializer = {
        "path": str(base_path),
        "sha256": observed_base_sha,
    }
    if variant_id in _STAGE_B_DATA_DRIVEN_ASSIGNMENT_VARIANTS:
        _validate_stage_b_data_driven_assignment_training_contract(
            args,
            base_path=base_path,
            variant_id=variant_id,
            dataset_path=dataset_path,
        )
    if variant_id == _STAGE_B_DATA_DRIVEN_ROLE_ROUTED_VARIANT:
        _validate_stage_b_data_driven_role_routed_training_contract(
            args,
            base_path=base_path,
            dataset_path=dataset_path,
        )
    if variant_id in {
        "DD1-H",
        "DD1-HC",
        _STAGE_B_DATA_DRIVEN_PAIRTOP1_HARDGAP3_VARIANT,
    }:
        _validate_stage_b_data_driven_dd1h_fresh_training_contract(
            args,
            config_path=config_path,
            base_path=base_path,
            variant_id=variant_id,
        )
        execution_scope = str(
            getattr(args, "stage_b_data_driven_execution_scope", "") or ""
        ).strip()
        if execution_scope == _STAGE_B_DATA_DRIVEN_DD1H_FORMAL_SCOPE:
            formal_specs = {
                "metadata_preflight": (
                    "stage_b_data_driven_formal_preflight_path",
                    "stage_b_data_driven_formal_preflight_sha256",
                ),
                "strict_u50_probe_receipt": (
                    "stage_b_data_driven_formal_probe_receipt_path",
                    "stage_b_data_driven_formal_probe_receipt_sha256",
                ),
                "formal_gate_contract": (
                    "stage_b_data_driven_formal_gate_contract_path",
                    "stage_b_data_driven_formal_gate_contract_sha256",
                ),
            }
            formal_evidence = {}
            for label, (path_key, sha_key) in formal_specs.items():
                path_value = str(getattr(args, path_key, "") or "").strip()
                expected_sha = str(getattr(args, sha_key, "") or "").strip()
                if not path_value or len(expected_sha) != 64:
                    raise RuntimeError(
                        f"{variant_id} formal run requires exact {label} evidence"
                    )
                evidence_path = Path(path_value).resolve(strict=True)
                observed_sha = _sha256_file(evidence_path)
                if observed_sha != expected_sha:
                    raise RuntimeError(
                        f"{variant_id} formal {label} SHA drifted"
                    )
                formal_evidence[label] = {
                    "path": str(evidence_path),
                    "sha256": observed_sha,
                }
            probe_payload = json.loads(
                Path(formal_evidence["strict_u50_probe_receipt"]["path"])
                .read_text(encoding="utf-8")
            )
            gate_payload = json.loads(
                Path(formal_evidence["formal_gate_contract"]["path"])
                .read_text(encoding="utf-8")
            )
            expected_output = str(
                getattr(args, "stage_b_data_driven_formal_output_dir", "")
            )
            args.stage_b_data_driven_formal_probe_code_manifest_sha256 = (
                _validate_stage_b_data_driven_formal_evidence_payloads(
                    variant_id=variant_id,
                    probe_payload=probe_payload,
                    gate_payload=gate_payload,
                    expected_output=expected_output,
                )
            )
            args.stage_b_data_driven_formal_evidence = formal_evidence
    pair_path: Optional[Path] = None
    observed_pair_sha: Optional[str] = None
    pair_path_value = str(
        getattr(
            args,
            "stage_b_data_driven_initializer_pair_receipt_path",
            "",
        )
        or ""
    ).strip()
    pair_sha = str(
        getattr(
            args,
            "stage_b_data_driven_initializer_pair_receipt_sha256",
            "",
        )
        or ""
    ).strip()
    if pair_path_value or pair_sha:
        if not pair_path_value or len(pair_sha) != 64:
            raise RuntimeError(
                "data-driven initializer pairing requires an exact receipt path/SHA"
            )
        pair_path = Path(pair_path_value).resolve(strict=True)
        observed_pair_sha = _sha256_file(pair_path)
        if observed_pair_sha != pair_sha:
            raise RuntimeError("data-driven initializer pair receipt SHA drifted")
        try:
            pair_payload = json.loads(pair_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"could not parse data-driven initializer pair receipt: {error}"
            ) from error
        if not isinstance(pair_payload, Mapping) or pair_payload.get(
            "schema"
        ) != "pivot.stageb.data_driven_initializer_pair/v1" or pair_payload.get(
            "status"
        ) != "passed":
            raise RuntimeError("data-driven initializer pair receipt is not passed")
        paired_shas = {
            record.get("sha256")
            for record in (
                pair_payload.get("absolute_initializer"),
                pair_payload.get("relational_initializer"),
            )
            if isinstance(record, Mapping)
        }
        if observed_base_sha not in paired_shas:
            raise RuntimeError(
                "data-driven base initializer is not a member of its pair receipt"
            )
        pair_invariants = pair_payload.get("invariants")
        required_pair_invariants = {
            "b58_is_only_tensor_checkpoint_source",
            "all_common_non_rank_non_contract_tensors_bitwise_equal",
            "patch_and_confidence_initialization_bitwise_equal",
            "rank_subtree_is_the_only_parameterized_architecture_intervention",
            "no_teacher_u1000_u5020_or_old_initializer_tensor_source",
        }
        if not isinstance(pair_invariants, Mapping) or any(
            pair_invariants.get(key) is not True
            for key in required_pair_invariants
        ):
            raise RuntimeError(
                "data-driven initializer pair receipt invariants drifted"
            )
        args.stage_b_data_driven_initializer_pair_receipt = {
            "path": str(pair_path),
            "sha256": observed_pair_sha,
            "schema": pair_payload["schema"],
            "common_tensor_sha256": pair_payload[
                "common_non_rank_non_contract"
            ]["tensor_sha256"],
        }
    _validate_stage_b_data_driven_new_head_formal_training_contract(
        args,
        config_path=config_path,
        dataset_path=dataset_path,
        base_path=base_path,
        observed_base_sha=observed_base_sha,
        pair_path=pair_path,
        observed_pair_sha=observed_pair_sha,
    )
    args.stage_b_data_driven_training_provenance = (
        _build_stage_b_data_driven_training_provenance(
            args, dataset_path=dataset_path
        )
    )
    expected_probe_code_sha = str(
        getattr(
            args,
            "stage_b_data_driven_formal_probe_code_manifest_sha256",
            "",
        )
        or ""
    ).strip()
    if expected_probe_code_sha:
        observed_probe_code_sha = _stage_b_data_driven_manifest_sha256(
            args.stage_b_data_driven_training_provenance["code_files"]
        )
        if observed_probe_code_sha != expected_probe_code_sha:
            raise RuntimeError(
                "formal training code closure drifted from its sealed U50 probe"
            )
        args.stage_b_data_driven_formal_observed_code_manifest_sha256 = (
            observed_probe_code_sha
        )
    if str(
        getattr(args, "stage_b_data_driven_train_mode", "") or ""
    ).strip().lower() == "confidence_pair":
        if int(getattr(args, "gradient_accumulation_steps", 1)) != 1:
            raise RuntimeError(
                "data-driven confidence q05 training requires "
                "gradient_accumulation_steps=1"
            )
        expected_path_value = str(
            getattr(
                args,
                "stage_b_data_driven_confidence_dataset_config_path",
                "",
            )
            or ""
        ).strip()
        expected_sha = str(
            getattr(
                args,
                "stage_b_data_driven_confidence_dataset_config_sha256",
                "",
            )
            or ""
        ).strip()
        if not expected_path_value or len(expected_sha) != 64:
            raise RuntimeError(
                "data-driven confidence requires an exact paired dataset path/SHA"
            )
        expected_path = Path(expected_path_value).resolve(strict=True)
        if dataset_path != expected_path or _sha256_file(dataset_path) != expected_sha:
            raise RuntimeError(
                "data-driven confidence dataset config differs from its paired "
                "DD2/DD3 contract"
            )


def _stage_b_v15_scorer_init_request(args) -> str:
    return str(
        getattr(args, "stage_b_v15_scorer_init_checkpoint", "") or ""
    ).strip()


def _validate_stage_b_confidence_rank_evidence_contract(
    args,
    *,
    revision: str,
) -> str:
    contract = str(
        getattr(
            args,
            "stage_b_dense_duty_confidence_rank_evidence_contract",
            "off_v1",
        )
        or ""
    ).strip().lower()
    if revision == "word_veto_gated_pool_rank_evidence_v11":
        if contract != "zero_init_rank_logit_scale_v1":
            raise RuntimeError(
                "rank-evidence v11 requires the exact zero-initialized "
                "rank-logit scale contract"
            )
    elif revision == "word_veto_gated_pool_rank_affine_v12":
        if contract != "zero_init_rank_logit_affine_v2":
            raise RuntimeError(
                "rank-affine v12 requires the exact zero-initialized "
                "rank-logit affine contract"
            )
    elif revision == "word_veto_gated_pool_gate_margin_v13":
        if contract != "zero_init_rank_logit_gate_margin_scale_v3":
            raise RuntimeError(
                "gate-margin v13 requires the exact zero-initialized "
                "rank-logit gate-margin contract"
            )
    elif revision == "word_veto_gated_pool_carrier_slope_v14":
        if contract != "zero_init_carrier_token_rank_slope_v4":
            raise RuntimeError(
                "carrier-slope v14 requires the exact zero-initialized "
                "carrier-token rank-slope contract"
            )
    elif revision in {
        "word_veto_gated_pool_carrier_affine_v15",
        "word_veto_gated_pool_tail_ste_v16",
        "word_veto_gated_pool_tail_carrier_v17",
        "word_veto_gated_pool_tail_paired_v18",
    }:
        if contract != "zero_init_carrier_token_rank_affine_v5":
            raise RuntimeError(
                "carrier-affine confidence requires the exact zero-initialized "
                "carrier-token rank-affine contract"
            )
    elif revision in {
        "word_veto_gated_pool_tail_paired_rank_channel_v19",
        "word_veto_gated_pool_tail_paired_signed_rank_pool_v20",
        "word_veto_continuous_conditional_residual_v21",
        "word_veto_continuous_monotone_depth_v22",
        "word_veto_token_conditioned_monotone_depth_v23",
        "word_veto_complementary_trust_veto_v24",
        "word_veto_ungated_monotone_tail_veto_v25",
        "word_veto_floor_gated_monotone_tail_veto_v26",
        "word_veto_independent_absolute_confidence_v27",
        "word_veto_cross_attention_absolute_confidence_v28",
        "word_veto_candidate_absolute_confidence_v29",
        "word_veto_candidate_patch_invariant_confidence_v30",
        "word_veto_candidate_normalized_confidence_v31",
        "word_veto_candidate_asymmetric_confidence_v32",
        "word_veto_candidate_set_attention_confidence_v33",
        "word_veto_candidate_asymmetric_deployed_routing_v43",
        "word_veto_candidate_split_tail_aligned_v45",
        "word_veto_candidate_split_positive_tail_v46",
        "word_veto_candidate_split_boundary_routing_v47",
        "word_veto_candidate_split_fpr_active_set_v48",
        "word_veto_candidate_split_global_trust_veto_v49",
        "word_veto_candidate_split_strong_boundary_routing_v50",
        "word_veto_candidate_split_independent_deployed_router_v51",
        "word_veto_candidate_sample_calibrator_split_v52",
        "word_veto_rank_full_expression_global_absolute_v53",
        "word_veto_rank_full_expression_global_absolute_exact_residual_v54",
        "word_veto_rank_full_expression_global_independent_absolute_v55",
        "word_veto_rank_full_expression_deployment_owned_global_v56",
        "word_veto_rank_full_expression_deployed_global_balanced_absolute_v57",
        "word_veto_rank_full_expression_deployment_owned_global_stable_fpr95_active_set_v58",
        "word_veto_rank_full_expression_deployment_owned_query_global_v59",
        "word_veto_rank_full_expression_deployment_owned_query_veto_v60",
    }:
        if contract != (
            "zero_init_carrier_token_rank_affine_sparse_rank_channel_v6"
        ):
            raise RuntimeError(
                "sparse rank-channel v19/v20 requires the exact zero-initialized "
                "query-token mismatch contract"
            )
    elif contract != "off_v1":
        raise RuntimeError(
            "pre-v11 confidence revisions forbid the rank-evidence residual"
        )
    residual_gain = float(
        getattr(
            args,
            "stage_b_dense_duty_confidence_residual_parameterization_gain",
            1.0,
        )
    )
    expected_gain = 0.25 / 0.03
    if revision in {
        "word_veto_gated_pool_gate_margin_v13",
        "word_veto_gated_pool_carrier_slope_v14",
        "word_veto_gated_pool_carrier_affine_v15",
        "word_veto_gated_pool_tail_ste_v16",
        "word_veto_gated_pool_tail_carrier_v17",
        "word_veto_gated_pool_tail_paired_v18",
        "word_veto_gated_pool_tail_paired_rank_channel_v19",
        "word_veto_gated_pool_tail_paired_signed_rank_pool_v20",
        "word_veto_continuous_conditional_residual_v21",
        "word_veto_continuous_monotone_depth_v22",
        "word_veto_token_conditioned_monotone_depth_v23",
        "word_veto_complementary_trust_veto_v24",
        "word_veto_ungated_monotone_tail_veto_v25",
        "word_veto_floor_gated_monotone_tail_veto_v26",
        "word_veto_independent_absolute_confidence_v27",
        "word_veto_cross_attention_absolute_confidence_v28",
        "word_veto_candidate_absolute_confidence_v29",
        "word_veto_candidate_patch_invariant_confidence_v30",
        "word_veto_candidate_normalized_confidence_v31",
        "word_veto_candidate_asymmetric_confidence_v32",
        "word_veto_candidate_set_attention_confidence_v33",
        "word_veto_candidate_asymmetric_deployed_routing_v43",
        "word_veto_candidate_split_tail_aligned_v45",
        "word_veto_candidate_split_positive_tail_v46",
        "word_veto_candidate_split_boundary_routing_v47",
        "word_veto_candidate_split_fpr_active_set_v48",
        "word_veto_candidate_split_global_trust_veto_v49",
        "word_veto_candidate_split_strong_boundary_routing_v50",
        "word_veto_candidate_split_independent_deployed_router_v51",
        "word_veto_candidate_sample_calibrator_split_v52",
        "word_veto_rank_full_expression_global_absolute_v53",
        "word_veto_rank_full_expression_global_absolute_exact_residual_v54",
        "word_veto_rank_full_expression_global_independent_absolute_v55",
        "word_veto_rank_full_expression_deployment_owned_global_v56",
        "word_veto_rank_full_expression_deployed_global_balanced_absolute_v57",
        "word_veto_rank_full_expression_deployment_owned_global_stable_fpr95_active_set_v58",
        "word_veto_rank_full_expression_deployment_owned_query_global_v59",
        "word_veto_rank_full_expression_deployment_owned_query_veto_v60",
    }:
        if not math.isclose(
            residual_gain, expected_gain, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise RuntimeError(
                "conditioned carrier revisions require residual gain=0.25/0.03"
            )
    elif not math.isclose(residual_gain, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            "pre-v13 confidence revisions require unit residual parameterization"
        )
    gate_gradient_contract = str(
        getattr(
            args,
            "stage_b_dense_duty_confidence_gate_gradient_contract",
            "hard_detached_v1",
        )
        or ""
    ).strip().lower()
    if revision in {
        "word_veto_continuous_monotone_depth_v22",
        "word_veto_token_conditioned_monotone_depth_v23",
        "word_veto_complementary_trust_veto_v24",
        "word_veto_ungated_monotone_tail_veto_v25",
        "word_veto_floor_gated_monotone_tail_veto_v26",
        "word_veto_independent_absolute_confidence_v27",
        "word_veto_cross_attention_absolute_confidence_v28",
        "word_veto_candidate_absolute_confidence_v29",
        "word_veto_candidate_patch_invariant_confidence_v30",
        "word_veto_candidate_normalized_confidence_v31",
        "word_veto_candidate_asymmetric_confidence_v32",
        "word_veto_candidate_set_attention_confidence_v33",
        "word_veto_candidate_asymmetric_deployed_routing_v43",
        "word_veto_candidate_split_tail_aligned_v45",
        "word_veto_candidate_split_positive_tail_v46",
        "word_veto_candidate_split_boundary_routing_v47",
        "word_veto_candidate_split_fpr_active_set_v48",
        "word_veto_candidate_split_global_trust_veto_v49",
        "word_veto_candidate_split_strong_boundary_routing_v50",
        "word_veto_candidate_split_independent_deployed_router_v51",
        "word_veto_candidate_sample_calibrator_split_v52",
        "word_veto_rank_full_expression_global_absolute_v53",
        "word_veto_rank_full_expression_global_absolute_exact_residual_v54",
        "word_veto_rank_full_expression_global_independent_absolute_v55",
        "word_veto_rank_full_expression_deployment_owned_global_v56",
        "word_veto_rank_full_expression_deployed_global_balanced_absolute_v57",
        "word_veto_rank_full_expression_deployment_owned_global_stable_fpr95_active_set_v58",
        "word_veto_rank_full_expression_deployment_owned_query_global_v59",
        "word_veto_rank_full_expression_deployment_owned_query_veto_v60",
    }:
        expected_gate_contract = {
            "word_veto_complementary_trust_veto_v24": (
                "continuous_sigmoid_complementary_trust_veto_v5"
            ),
            "word_veto_ungated_monotone_tail_veto_v25": (
                "token_conditioned_ungated_monotone_depth_v6"
            ),
            "word_veto_floor_gated_monotone_tail_veto_v26": (
                "token_conditioned_floor_gated_monotone_depth_v7"
            ),
            "word_veto_independent_absolute_confidence_v27": (
                "token_conditioned_independent_absolute_logit_v8"
            ),
            "word_veto_cross_attention_absolute_confidence_v28": (
                "cross_attention_independent_absolute_logit_v9"
            ),
            "word_veto_candidate_absolute_confidence_v29": (
                "candidate_cross_attention_independent_absolute_logit_v10"
            ),
            "word_veto_candidate_patch_invariant_confidence_v30": (
                "candidate_patch_invariant_monotone_veto_absolute_logit_v11"
            ),
            "word_veto_candidate_normalized_confidence_v31": (
                "candidate_normalized_patch_amplified_monotone_veto_absolute_logit_v12"
            ),
            "word_veto_candidate_asymmetric_confidence_v32": (
                "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13"
            ),
            "word_veto_candidate_asymmetric_deployed_routing_v43": (
                "candidate_raw_patch_asymmetric_deployed_routing_st_v15"
            ),
            "word_veto_candidate_split_tail_aligned_v45": (
                "candidate_raw_patch_asymmetric_deployed_routing_st_v15"
            ),
            "word_veto_candidate_split_positive_tail_v46": (
                "candidate_raw_patch_asymmetric_deployed_routing_st_v15"
            ),
            "word_veto_candidate_split_boundary_routing_v47": (
                "candidate_raw_patch_asymmetric_deployed_routing_st_v15"
            ),
            "word_veto_candidate_split_strong_boundary_routing_v50": (
                "candidate_raw_patch_asymmetric_deployed_routing_st_v15"
            ),
            "word_veto_candidate_split_independent_deployed_router_v51": (
                "candidate_raw_patch_asymmetric_deployed_routing_st_v15"
            ),
            "word_veto_candidate_sample_calibrator_split_v52": (
                "candidate_raw_patch_asymmetric_deployed_routing_st_v15"
            ),
            "word_veto_rank_full_expression_global_absolute_v53": (
                "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13"
            ),
            "word_veto_rank_full_expression_global_absolute_exact_residual_v54": (
                "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13"
            ),
            "word_veto_rank_full_expression_global_independent_absolute_v55": (
                "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13"
            ),
            "word_veto_rank_full_expression_deployment_owned_global_v56": (
                "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13"
            ),
            "word_veto_rank_full_expression_deployed_global_balanced_absolute_v57": (
                "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13"
            ),
            "word_veto_rank_full_expression_deployment_owned_global_stable_fpr95_active_set_v58": (
                "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13"
            ),
            "word_veto_rank_full_expression_deployment_owned_query_global_v59": (
                "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13"
            ),
            "word_veto_rank_full_expression_deployment_owned_query_veto_v60": (
                "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13"
            ),
            "word_veto_candidate_split_fpr_active_set_v48": (
                "candidate_raw_patch_asymmetric_deployed_routing_st_v15"
            ),
            "word_veto_candidate_split_global_trust_veto_v49": (
                "candidate_raw_patch_asymmetric_deployed_routing_st_v15"
            ),
            "word_veto_candidate_set_attention_confidence_v33": (
                "candidate_set_attention_asymmetric_monotone_veto_absolute_logit_v14"
            ),
        }.get(revision, "continuous_sigmoid_monotone_depth_v4")
        if gate_gradient_contract != expected_gate_contract:
            raise RuntimeError(
                "continuous monotone-depth confidence requires its exact "
                "straight-through nonnegative-depth contract"
            )
    elif revision == "word_veto_continuous_conditional_residual_v21":
        if gate_gradient_contract != "continuous_sigmoid_v3":
            raise RuntimeError(
                "continuous conditional-residual v21 requires differentiable "
                "sigmoid modifier gates"
            )
    elif revision == "word_veto_gated_pool_tail_ste_v16":
        if gate_gradient_contract != "hard_forward_soft_backward_v2":
            raise RuntimeError(
                "tail-STE v16 requires hard-forward soft-backward carrier gates"
            )
    elif gate_gradient_contract != "hard_detached_v1":
        raise RuntimeError(
            "pre-v16 confidence revisions require detached hard gate gradients"
        )
    return contract


def _stage_b_target_iou_carrier_pair_admission_contract(args) -> str:
    token_edit_query_scope = str(
        getattr(args, "stage_b_v21_token_edit_query_scope", "target_iou_v1")
    ).strip().lower()
    if token_edit_query_scope != "target_iou_v1":
        raise RuntimeError(
            "target-IoU carrier-pair admission requires target_iou_v1"
        )
    carrier_pair_gradient_contract = str(
        getattr(
            args,
            "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
            "bidirectional_v1",
        )
    ).strip().lower()
    if carrier_pair_gradient_contract == "tn_only_positive_detached_v2":
        return (
            "u400_word_veto_candidate_tn_only_carrier_pair_"
            "confidence_strict1607_v42"
        )
    if carrier_pair_gradient_contract == "bidirectional_v1":
        return (
            "u400_word_veto_candidate_gate_zero_offset_"
            "confidence_strict1607_v39"
        )
    raise RuntimeError(
        "target-IoU carrier-pair admission has an unknown gradient contract"
    )


def _stage_b_candidate_asymmetric_formal_admission_contract(args) -> str:
    """Resolve V32's admission lazily so unrelated revisions stay independent."""
    positive_gradient_contract = str(
        getattr(
            args,
            "stage_b_v15_tail_queue_positive_gradient_contract",
            "mean_translation_v1",
        )
    ).strip()
    tail_contracts = {
        "exact_batch_lower_tail_st_v2": (
            "u400_word_veto_candidate_q05_confidence_strict1607_v34"
        ),
        "mean_plus_exact_lower_tail_st_v3": (
            "u400_word_veto_candidate_tail_balanced_confidence_strict1607_v35"
        ),
        "mean_plus_quarter_exact_lower_tail_st_v4": (
            "u400_word_veto_candidate_tail_quarter_confidence_strict1607_v36"
        ),
        "bounded_mean_plus_sixteenth_exact_lower_tail_st_v5": (
            "u400_word_veto_candidate_tail_bounded_confidence_strict1607_v37"
        ),
    }
    if positive_gradient_contract in tail_contracts:
        return tail_contracts[positive_gradient_contract]
    if positive_gradient_contract != (
        "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
    ):
        return "u400_word_veto_candidate_asymmetric_confidence_strict1607_v32"

    veto_gate_offset = float(
        getattr(
            args,
            "stage_b_dense_duty_confidence_veto_gate_offset",
            -1.0,
        )
    )
    if veto_gate_offset != 0.0:
        return "u400_word_veto_candidate_tail_elementwise_confidence_strict1607_v38"

    token_edit_query_scope = str(
        getattr(args, "stage_b_v21_token_edit_query_scope", "target_iou_v1")
    ).strip().lower()
    if token_edit_query_scope == (
        "target_iou_union_detached_role_complete_confidence_base_argmax_v3"
    ):
        return (
            "u400_word_veto_candidate_role_complete_carrier_"
            "confidence_strict1607_v41"
        )
    if token_edit_query_scope == (
        "target_iou_union_detached_final_confidence_base_argmax_v2"
    ):
        return "u400_word_veto_candidate_hardest_edit_confidence_strict1607_v40"
    return _stage_b_target_iou_carrier_pair_admission_contract(args)


def _bind_stage_b_confidence_probe_admission(args) -> Optional[dict[str, Any]]:
    if not bool(getattr(args, "stage_b_dense_duty", False)) or str(
        getattr(args, "stage_b_v22_score_ownership", "")
    ).strip() != "rank_tower_stopgrad_token_adapter_two_phase":
        return None
    aggregation = str(
        getattr(
            args,
            "stage_b_dense_duty_confidence_phrase_aggregation",
            "",
        )
    ).strip().lower()
    revision = str(
        getattr(args, "stage_b_dense_duty_confidence_revision", "")
    ).strip()
    positive_gradient_contract = str(
        getattr(
            args,
            "stage_b_v15_tail_queue_positive_gradient_contract",
            "mean_translation_v1",
        )
    ).strip()
    token_edit_query_scope = str(
        getattr(args, "stage_b_v21_token_edit_query_scope", "target_iou_v1")
    ).strip().lower()
    carrier_pair_gradient_contract = str(
        getattr(
            args,
            "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
            "bidirectional_v1",
        )
    ).strip().lower()
    if (
        aggregation == "trace_activated_word_veto_product_v1"
        and revision == "word_veto_net_trust_v1"
    ):
        formal_contract = "u300_word_veto_strict1607_v1"
        from tools import (
            run_stageb_confidence_adapter_veto_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_penalty_v2"
        and revision == "word_veto_raw_gate_margin_v3"
    ):
        formal_contract = "u300_word_veto_gate_strict1607_v3"
        from tools import (
            run_stageb_confidence_adapter_veto_gate_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_absolute_cap_v4"
        and revision == "word_veto_coverage_absolute_cap_v4"
    ):
        formal_contract = "u300_word_veto_absolute_cap_strict1607_v4"
        from tools import (
            run_stageb_confidence_adapter_veto_cap_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision == "word_veto_gated_pool_absolute_cap_v5"
    ):
        formal_contract = "u300_word_veto_gated_pool_absolute_cap_strict1607_v5"
        from tools import (
            run_stageb_confidence_adapter_veto_gated_pool_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision == "word_veto_gated_pool_calibrated_v6"
    ):
        formal_contract = "u300_word_veto_gated_pool_calibrated_strict1607_v6"
        from tools import (
            run_stageb_confidence_adapter_veto_gated_pool_calibrated_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision == "word_veto_gated_pool_carrier_balanced_v7"
    ):
        formal_contract = (
            "u300_word_veto_gated_pool_carrier_balanced_strict1607_v7"
        )
        from tools import (
            run_stageb_confidence_adapter_veto_gated_pool_carrier_balanced_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision == "word_veto_gated_pool_carrier_quarter_v8"
    ):
        formal_contract = (
            "u300_word_veto_gated_pool_carrier_quarter_strict1607_v8"
        )
        from tools import (
            run_stageb_confidence_adapter_veto_gated_pool_carrier_quarter_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision == "word_veto_gated_pool_carrier_pair_v9"
    ):
        formal_contract = (
            "u300_word_veto_gated_pool_carrier_pair_strict1607_v9"
        )
        from tools import (
            run_stageb_confidence_adapter_veto_gated_pool_carrier_pair_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision == "word_veto_gated_pool_dual_carrier_pair_v10"
    ):
        formal_contract = (
            "u300_word_veto_gated_pool_dual_carrier_pair_strict1607_v10"
        )
        from tools import (
            run_stageb_confidence_adapter_veto_gated_pool_dual_carrier_pair_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision == "word_veto_gated_pool_rank_evidence_v11"
    ):
        formal_contract = (
            "u300_word_veto_gated_pool_rank_evidence_strict1607_v11"
        )
        from tools import (
            run_stageb_confidence_adapter_veto_gated_pool_rank_evidence_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision == "word_veto_gated_pool_rank_affine_v12"
    ):
        formal_contract = (
            "u300_word_veto_gated_pool_rank_affine_strict1607_v12"
        )
        from tools import (
            run_stageb_confidence_adapter_veto_gated_pool_rank_affine_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision == "word_veto_gated_pool_gate_margin_v13"
    ):
        formal_contract = (
            "u300_word_veto_gated_pool_gate_margin_strict1607_v13"
        )
        from tools import (
            run_stageb_confidence_adapter_veto_gated_pool_gate_margin_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision == "word_veto_gated_pool_carrier_slope_v14"
    ):
        formal_contract = (
            "u300_word_veto_gated_pool_carrier_slope_strict1607_v14"
        )
        from tools import (
            run_stageb_confidence_adapter_veto_gated_pool_carrier_slope_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision == "word_veto_gated_pool_carrier_affine_v15"
    ):
        formal_contract = (
            "u300_word_veto_gated_pool_carrier_affine_strict1607_v15"
        )
        from tools import (
            run_stageb_confidence_adapter_veto_gated_pool_carrier_affine_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision == "word_veto_gated_pool_tail_ste_v16"
    ):
        formal_contract = (
            "u300_word_veto_gated_pool_tail_ste_strict1607_v16"
        )
        from tools import (
            run_stageb_confidence_adapter_veto_gated_pool_tail_ste_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision == "word_veto_gated_pool_tail_carrier_v17"
    ):
        formal_contract = (
            "u300_word_veto_gated_pool_tail_carrier_strict1607_v17"
        )
        from tools import (
            run_stageb_confidence_adapter_veto_gated_pool_tail_carrier_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision == "word_veto_gated_pool_tail_paired_v18"
    ):
        formal_contract = (
            "u300_word_veto_gated_pool_tail_paired_strict1607_v18"
        )
        from tools import (
            run_stageb_confidence_adapter_veto_gated_pool_tail_paired_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision == "word_veto_candidate_asymmetric_confidence_v32"
        and positive_gradient_contract == "exact_batch_lower_tail_st_v2"
    ):
        formal_contract = (
            "u400_word_veto_candidate_q05_confidence_strict1607_v34"
        )
        from tools import (
            run_stageb_confidence_adapter_candidate_q05_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision == "word_veto_candidate_asymmetric_confidence_v32"
        and positive_gradient_contract == "mean_plus_exact_lower_tail_st_v3"
    ):
        formal_contract = (
            "u400_word_veto_candidate_tail_balanced_confidence_strict1607_v35"
        )
        from tools import (
            run_stageb_confidence_adapter_candidate_tail_balanced_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision == "word_veto_candidate_asymmetric_confidence_v32"
        and positive_gradient_contract == "mean_plus_quarter_exact_lower_tail_st_v4"
    ):
        formal_contract = (
            "u400_word_veto_candidate_tail_quarter_confidence_strict1607_v36"
        )
        from tools import (
            run_stageb_confidence_adapter_candidate_tail_quarter_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision == "word_veto_candidate_asymmetric_confidence_v32"
        and positive_gradient_contract
        == "bounded_mean_plus_sixteenth_exact_lower_tail_st_v5"
    ):
        formal_contract = (
            "u400_word_veto_candidate_tail_bounded_confidence_strict1607_v37"
        )
        from tools import (
            run_stageb_confidence_adapter_candidate_tail_bounded_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision == "word_veto_candidate_sample_calibrator_split_v52"
        and positive_gradient_contract
        == "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
        and str(
            getattr(
                args,
                "stage_b_dense_duty_confidence_head_gradient_contract",
                "",
            )
        ).strip().lower()
        == "split_token_veto_candidate_absolute_sample_calibrator_v6"
        and str(
            getattr(
                args,
                "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
                "",
            )
        ).strip().lower()
        == "balanced_top_quarter_cvar_v2"
        and str(
            getattr(
                args,
                "stage_b_v15_tail_queue_positive_trust_reduction_contract",
                "",
            )
        ).strip().lower()
        == "top_quarter_cvar_v2"
        and str(
            getattr(
                args,
                "stage_b_v15_tail_queue_negative_reduction_contract",
                "all_mean_v1",
            )
        ).strip().lower()
        == "all_mean_v1"
        and str(
            getattr(
                args,
                "stage_b_dense_duty_positive_trust_contract",
                "",
            )
        ).strip().lower()
        == "absolute_global_confidence_logit_v2"
        and float(
            getattr(
                args,
                "stage_b_dense_duty_deployed_veto_positive_max",
                -1.0,
            )
        )
        == 0.1
        and float(
            getattr(
                args,
                "stage_b_dense_duty_deployed_veto_tn_min",
                -1.0,
            )
        )
        == 0.9
        and float(
            getattr(
                args,
                "stage_b_dense_duty_confidence_veto_gate_offset",
                -1.0,
            )
        )
        == 0.0
        and token_edit_query_scope == "target_iou_v1"
        and carrier_pair_gradient_contract == "bidirectional_v1"
        and float(
            getattr(
                args,
                "stage_b_dense_duty_deployed_veto_routing_weight",
                -1.0,
            )
        )
        == 0.0
    ):
        formal_contract = (
            "u400_word_veto_candidate_sample_calibrator_"
            "confidence_strict1607_v52"
        )
        from tools import (
            run_stageb_confidence_adapter_candidate_sample_calibrator_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision
        in {
            "word_veto_rank_full_expression_global_absolute_v53",
            "word_veto_rank_full_expression_global_absolute_exact_residual_v54",
            "word_veto_rank_full_expression_global_independent_absolute_v55",
            "word_veto_rank_full_expression_deployment_owned_global_v56",
            "word_veto_rank_full_expression_deployed_global_balanced_absolute_v57",
            "word_veto_rank_full_expression_deployment_owned_global_stable_fpr95_active_set_v58",
            "word_veto_rank_full_expression_deployment_owned_query_global_v59",
            "word_veto_rank_full_expression_deployment_owned_query_veto_v60",
        }
        and positive_gradient_contract
        == "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
        and str(
            getattr(
                args,
                "stage_b_dense_duty_confidence_head_gradient_contract",
                "",
            )
        ).strip().lower()
        == {
            "word_veto_rank_full_expression_global_absolute_v53": (
                "split_token_veto_fulltext_global_absolute_v7"
            ),
            "word_veto_rank_full_expression_global_absolute_exact_residual_v54": (
                "split_token_veto_fulltext_global_absolute_v7"
            ),
            "word_veto_rank_full_expression_global_independent_absolute_v55": (
                "split_token_veto_local_candidate_global_absolute_v8"
            ),
            "word_veto_rank_full_expression_deployment_owned_global_v56": (
                "split_token_veto_deployment_owned_global_absolute_v9"
            ),
            "word_veto_rank_full_expression_deployed_global_balanced_absolute_v57": (
                "split_token_veto_deployment_owned_global_absolute_v9"
            ),
            "word_veto_rank_full_expression_deployment_owned_global_stable_fpr95_active_set_v58": (
                "split_token_veto_deployment_owned_global_absolute_v9"
            ),
            "word_veto_rank_full_expression_deployment_owned_query_global_v59": (
                "split_token_veto_deployment_owned_query_global_absolute_v10"
            ),
            "word_veto_rank_full_expression_deployment_owned_query_veto_v60": (
                "split_token_veto_deployment_owned_query_veto_global_absolute_v11"
            ),
        }[revision]
        and str(
            getattr(
                args,
                "stage_b_dense_duty_confidence_pool_feature_contract",
                "",
            )
        ).strip().lower()
        == {
            "word_veto_rank_full_expression_global_absolute_v53": (
                "detached_rank_full_expression_candidate_residual_global_pool_v10"
            ),
            "word_veto_rank_full_expression_global_absolute_exact_residual_v54": (
                "detached_rank_full_expression_candidate_residual_global_pool_"
                "exact_rank_max_reference_v11"
            ),
            "word_veto_rank_full_expression_global_independent_absolute_v55": (
                "detached_rank_full_expression_local_candidate_"
                "frozen_rank_global_pool_v12"
            ),
            "word_veto_rank_full_expression_deployment_owned_global_v56": (
                "detached_rank_full_expression_deployment_owned_global_pool_v13"
            ),
            "word_veto_rank_full_expression_deployed_global_balanced_absolute_v57": (
                "detached_rank_full_expression_deployment_owned_global_pool_v13"
            ),
            "word_veto_rank_full_expression_deployment_owned_global_stable_fpr95_active_set_v58": (
                "detached_rank_full_expression_deployment_owned_global_pool_v13"
            ),
            "word_veto_rank_full_expression_deployment_owned_query_global_v59": (
                "detached_rank_full_expression_monotone_query_"
                "deployment_owned_global_pool_v14"
            ),
            "word_veto_rank_full_expression_deployment_owned_query_veto_v60": (
                "detached_rank_full_expression_token_conditioned_query_veto_"
                "deployment_owned_global_pool_v15"
            ),
        }[revision]
        and str(
            getattr(
                args,
                "stage_b_dense_duty_confidence_gate_gradient_contract",
                "",
            )
        ).strip().lower()
        == "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13"
        and str(
            getattr(
                args,
                "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
                "",
            )
        ).strip().lower()
        == "balanced_top_quarter_cvar_v2"
        and str(
            getattr(
                args,
                "stage_b_v15_tail_queue_positive_trust_reduction_contract",
                "",
            )
        ).strip().lower()
        == "top_quarter_cvar_v2"
        and str(
            getattr(
                args,
                "stage_b_v15_tail_queue_negative_reduction_contract",
                "all_mean_v1",
            )
        ).strip().lower()
        == {
            "word_veto_rank_full_expression_deployment_owned_global_stable_fpr95_active_set_v58": "exact_fpr95_active_set_all_count_mean_v2",
        }.get(revision, "all_mean_v1")
        and str(
            getattr(
                args,
                "stage_b_dense_duty_positive_trust_contract",
                "",
            )
        ).strip().lower()
        == {
            "word_veto_rank_full_expression_global_absolute_v53": (
                "absolute_global_confidence_logit_v2"
            ),
            "word_veto_rank_full_expression_global_absolute_exact_residual_v54": (
                "exact_frozen_rank_max_confidence_delta_v3"
            ),
            "word_veto_rank_full_expression_global_independent_absolute_v55": (
                "absolute_global_pool_logit_v4"
            ),
            "word_veto_rank_full_expression_deployment_owned_global_v56": (
                "absolute_global_pool_logit_v4"
            ),
            "word_veto_rank_full_expression_deployed_global_balanced_absolute_v57": (
                "absolute_global_pool_logit_v4"
            ),
            "word_veto_rank_full_expression_deployment_owned_global_stable_fpr95_active_set_v58": (
                "absolute_global_pool_logit_v4"
            ),
            "word_veto_rank_full_expression_deployment_owned_query_global_v59": (
                "absolute_global_confidence_logit_v2"
            ),
            "word_veto_rank_full_expression_deployment_owned_query_veto_v60": (
                "absolute_global_confidence_logit_v2"
            ),
        }[revision]
        and float(
            getattr(
                args,
                "stage_b_dense_duty_deployed_veto_positive_max",
                -1.0,
            )
        )
        == 0.1
        and float(
            getattr(
                args,
                "stage_b_dense_duty_deployed_veto_tn_min",
                -1.0,
            )
        )
        == 0.9
        and float(
            getattr(
                args,
                "stage_b_dense_duty_confidence_veto_gate_offset",
                -1.0,
            )
        )
        == 0.0
        and token_edit_query_scope == "target_iou_v1"
        and carrier_pair_gradient_contract == "bidirectional_v1"
        and float(
            getattr(
                args,
                "stage_b_dense_duty_deployed_veto_routing_weight",
                -1.0,
            )
        )
        == 0.0
        and (
            revision
            not in {
                "word_veto_rank_full_expression_deployment_owned_global_v56",
                "word_veto_rank_full_expression_deployed_global_balanced_absolute_v57",
                "word_veto_rank_full_expression_deployment_owned_global_stable_fpr95_active_set_v58",
            }
            or float(
                getattr(args, "stage_b_v14_local_absolute_weight", -1.0)
            )
            == 0.0
        )
        and (
            revision
            != "word_veto_rank_full_expression_deployed_global_balanced_absolute_v57"
            or (
                float(
                    getattr(
                        args,
                        "stage_b_dense_duty_deployed_global_absolute_weight",
                        -1.0,
                    )
                )
                == 1.0
                and float(
                    getattr(
                        args,
                        "stage_b_dense_duty_deployed_global_absolute_gamma",
                        -1.0,
                    )
                )
                == 1.0
            )
        )
        and (
            revision != "word_veto_rank_full_expression_deployment_owned_global_stable_fpr95_active_set_v58"
            or float(
                getattr(
                    args,
                    "stage_b_dense_duty_deployed_global_absolute_weight",
                    -1.0,
                )
            )
            == 0.0
        )
    ):
        if revision.endswith("_v60"):
            formal_contract = (
                "u400_word_veto_rank_full_expression_deployment_owned_query_"
                "veto_confidence_strict1607_v60"
            )
            from tools import (
                run_stageb_confidence_adapter_deployment_owned_query_veto_probe_evaluation as promotion,
            )
        elif revision.endswith("_v59"):
            formal_contract = (
                "u400_word_veto_rank_full_expression_deployment_owned_query_"
                "global_confidence_strict1607_v59"
            )
            from tools import (
                run_stageb_confidence_adapter_deployment_owned_query_global_probe_evaluation as promotion,
            )
        elif revision.endswith("_v58"):
            formal_contract = (
                "u400_word_veto_rank_full_expression_deployment_owned_global_"
                "stable_fpr95_active_set_confidence_strict1607_v58"
            )
            from tools import (
                run_stageb_confidence_adapter_deployment_owned_global_stable_fpr95_active_set_probe_evaluation as promotion,
            )
        elif revision.endswith("_v57"):
            formal_contract = (
                "u400_word_veto_rank_full_expression_deployed_global_"
                "balanced_absolute_confidence_strict1607_v57"
            )
            from tools import (
                run_stageb_confidence_adapter_deployed_global_balanced_absolute_probe_evaluation as promotion,
            )
        elif revision.endswith("_v56"):
            formal_contract = (
                "u400_word_veto_rank_full_expression_deployment_owned_global_"
                "confidence_strict1607_v56"
            )
            from tools import (
                run_stageb_confidence_adapter_deployment_owned_global_probe_evaluation as promotion,
            )
        elif revision.endswith("_v55"):
            formal_contract = (
                "u400_word_veto_rank_full_expression_global_independent_absolute_"
                "confidence_strict1607_v55"
            )
            from tools import (
                run_stageb_confidence_adapter_fulltext_global_independent_absolute_probe_evaluation as promotion,
            )
        elif revision.endswith("_v54"):
            formal_contract = (
                "u400_word_veto_rank_full_expression_global_absolute_"
                "exact_residual_confidence_strict1607_v54"
            )
            from tools import (
                run_stageb_confidence_adapter_fulltext_global_absolute_exact_residual_probe_evaluation as promotion,
            )
        else:
            formal_contract = (
                "u400_word_veto_rank_full_expression_global_absolute_"
                "confidence_strict1607_v53"
            )
            from tools import (
                run_stageb_confidence_adapter_fulltext_global_absolute_probe_evaluation as promotion,
            )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision
        == "word_veto_candidate_split_independent_deployed_router_v51"
        and positive_gradient_contract
        == "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
        and str(
            getattr(
                args,
                "stage_b_dense_duty_confidence_head_gradient_contract",
                "",
            )
        ).strip().lower()
        == "split_token_veto_deployed_router_global_absolute_v5"
        and str(
            getattr(
                args,
                "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
                "",
            )
        ).strip().lower()
        == "balanced_top_quarter_cvar_v2"
        and str(
            getattr(
                args,
                "stage_b_v15_tail_queue_positive_trust_reduction_contract",
                "",
            )
        ).strip().lower()
        == "top_quarter_cvar_v2"
        and str(
            getattr(
                args,
                "stage_b_v15_tail_queue_negative_reduction_contract",
                "all_mean_v1",
            )
        ).strip().lower()
        == "all_mean_v1"
        and str(
            getattr(
                args,
                "stage_b_dense_duty_positive_trust_contract",
                "",
            )
        ).strip().lower()
        == "absolute_global_confidence_logit_v2"
        and float(
            getattr(
                args,
                "stage_b_dense_duty_deployed_veto_positive_max",
                -1.0,
            )
        )
        == 0.1
        and float(
            getattr(
                args,
                "stage_b_dense_duty_deployed_veto_tn_min",
                -1.0,
            )
        )
        == 0.9
        and float(
            getattr(
                args,
                "stage_b_dense_duty_confidence_veto_gate_offset",
                -1.0,
            )
        )
        == 0.0
        and token_edit_query_scope == "target_iou_v1"
        and carrier_pair_gradient_contract == "bidirectional_v1"
        and float(
            getattr(
                args,
                "stage_b_dense_duty_deployed_veto_routing_weight",
                0.0,
            )
        )
        == 0.1
    ):
        formal_contract = (
            "u400_word_veto_candidate_split_independent_deployed_router_"
            "confidence_strict1607_v51"
        )
        from tools import (
            run_stageb_confidence_adapter_candidate_split_independent_deployed_router_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision
        == "word_veto_candidate_split_strong_boundary_routing_v50"
        and positive_gradient_contract
        == "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
        and str(
            getattr(
                args,
                "stage_b_dense_duty_confidence_head_gradient_contract",
                "",
            )
        ).strip().lower()
        == "split_token_veto_global_absolute_v2"
        and str(
            getattr(
                args,
                "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
                "",
            )
        ).strip().lower()
        == "balanced_top_quarter_cvar_v2"
        and str(
            getattr(
                args,
                "stage_b_v15_tail_queue_positive_trust_reduction_contract",
                "",
            )
        ).strip().lower()
        == "top_quarter_cvar_v2"
        and str(
            getattr(
                args,
                "stage_b_v15_tail_queue_negative_reduction_contract",
                "all_mean_v1",
            )
        ).strip().lower()
        == "all_mean_v1"
        and str(
            getattr(
                args,
                "stage_b_dense_duty_positive_trust_contract",
                "",
            )
        ).strip().lower()
        == "absolute_global_confidence_logit_v2"
        and float(
            getattr(
                args,
                "stage_b_dense_duty_deployed_veto_positive_max",
                -1.0,
            )
        )
        == 0.1
        and float(
            getattr(
                args,
                "stage_b_dense_duty_deployed_veto_tn_min",
                -1.0,
            )
        )
        == 0.9
        and float(
            getattr(
                args,
                "stage_b_dense_duty_confidence_veto_gate_offset",
                -1.0,
            )
        )
        == 0.0
        and token_edit_query_scope == "target_iou_v1"
        and carrier_pair_gradient_contract == "bidirectional_v1"
        and float(
            getattr(
                args,
                "stage_b_dense_duty_deployed_veto_routing_weight",
                0.0,
            )
        )
        == 0.25
    ):
        formal_contract = (
            "u400_word_veto_candidate_split_strong_boundary_routing_"
            "confidence_strict1607_v50"
        )
        from tools import (
            run_stageb_confidence_adapter_candidate_split_strong_boundary_routing_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision == "word_veto_candidate_split_global_trust_veto_v49"
        and positive_gradient_contract
        == "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
        and str(
            getattr(
                args,
                "stage_b_dense_duty_confidence_head_gradient_contract",
                "",
            )
        ).strip().lower()
        == "split_token_veto_global_trust_veto_v4"
        and str(
            getattr(
                args,
                "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
                "",
            )
        ).strip().lower()
        == "balanced_top_quarter_cvar_v2"
        and str(
            getattr(
                args,
                "stage_b_v15_tail_queue_positive_trust_reduction_contract",
                "",
            )
        ).strip().lower()
        == "top_quarter_cvar_v2"
        and str(
            getattr(
                args,
                "stage_b_v15_tail_queue_negative_reduction_contract",
                "",
            )
        ).strip().lower()
        == "all_mean_v1"
        and float(
            getattr(
                args,
                "stage_b_dense_duty_confidence_veto_gate_offset",
                -1.0,
            )
        )
        == 0.0
        and token_edit_query_scope == "target_iou_v1"
        and carrier_pair_gradient_contract == "bidirectional_v1"
        and float(
            getattr(
                args,
                "stage_b_dense_duty_deployed_veto_routing_weight",
                0.0,
            )
        )
        == 0.1
    ):
        formal_contract = (
            "u400_word_veto_candidate_split_global_trust_veto_confidence_"
            "strict1607_v49"
        )
        from tools import (
            run_stageb_confidence_adapter_candidate_split_global_trust_veto_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision == "word_veto_candidate_asymmetric_deployed_routing_v43"
        and positive_gradient_contract
        == "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
        and float(
            getattr(
                args,
                "stage_b_dense_duty_confidence_veto_gate_offset",
                -1.0,
            )
        )
        == 0.0
        and token_edit_query_scope == "target_iou_v1"
        and carrier_pair_gradient_contract == "bidirectional_v1"
        and float(
            getattr(
                args,
                "stage_b_dense_duty_deployed_veto_routing_weight",
                0.0,
            )
        )
        == 0.1
    ):
        formal_contract = (
            "u400_word_veto_candidate_deployed_routing_confidence_"
            "strict1607_v43"
        )
        from tools import (
            run_stageb_confidence_adapter_candidate_deployed_routing_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision == "word_veto_candidate_asymmetric_confidence_v32"
        and positive_gradient_contract
        == "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
        and float(
            getattr(
                args,
                "stage_b_dense_duty_confidence_veto_gate_offset",
                -1.0,
            )
        )
        == 0.0
        and token_edit_query_scope
        == "target_iou_union_detached_role_complete_confidence_base_argmax_v3"
    ):
        formal_contract = (
            "u400_word_veto_candidate_role_complete_carrier_"
            "confidence_strict1607_v41"
        )
        from tools import (
            run_stageb_confidence_adapter_candidate_role_complete_carrier_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision == "word_veto_candidate_asymmetric_confidence_v32"
        and positive_gradient_contract
        == "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
        and float(
            getattr(
                args,
                "stage_b_dense_duty_confidence_veto_gate_offset",
                -1.0,
            )
        )
        == 0.0
        and token_edit_query_scope
        == "target_iou_union_detached_final_confidence_base_argmax_v2"
    ):
        formal_contract = (
            "u400_word_veto_candidate_hardest_edit_confidence_strict1607_v40"
        )
        from tools import (
            run_stageb_confidence_adapter_candidate_hardest_edit_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision == "word_veto_candidate_asymmetric_confidence_v32"
        and positive_gradient_contract
        == "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
        and float(
            getattr(
                args,
                "stage_b_dense_duty_confidence_veto_gate_offset",
                -1.0,
            )
        )
        == 0.0
        and token_edit_query_scope == "target_iou_v1"
        and carrier_pair_gradient_contract == "tn_only_positive_detached_v2"
    ):
        formal_contract = _stage_b_target_iou_carrier_pair_admission_contract(
            args
        )
        from tools import (
            run_stageb_confidence_adapter_candidate_tn_only_carrier_pair_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision == "word_veto_candidate_asymmetric_confidence_v32"
        and positive_gradient_contract
        == "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
        and float(
            getattr(
                args,
                "stage_b_dense_duty_confidence_veto_gate_offset",
                -1.0,
            )
        )
        == 0.0
        and token_edit_query_scope == "target_iou_v1"
        and carrier_pair_gradient_contract == "bidirectional_v1"
    ):
        formal_contract = (
            "u400_word_veto_candidate_gate_zero_offset_confidence_strict1607_v39"
        )
        from tools import (
            run_stageb_confidence_adapter_candidate_gate_zero_offset_probe_evaluation as promotion,
        )
    elif (
        aggregation == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and revision == "word_veto_candidate_asymmetric_confidence_v32"
        and positive_gradient_contract
        == "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
        and float(
            getattr(
                args,
                "stage_b_dense_duty_confidence_veto_gate_offset",
                -1.0,
            )
        )
        == 0.02
    ):
        formal_contract = (
            "u400_word_veto_candidate_tail_elementwise_confidence_strict1607_v38"
        )
        from tools import (
            run_stageb_confidence_adapter_candidate_tail_elementwise_probe_evaluation as promotion,
        )
    else:
        return None
    scope = str(
        getattr(args, "stage_b_dense_duty_execution_scope", "")
    ).strip().lower()
    contract = str(
        getattr(
            args,
            "stage_b_dense_duty_confidence_probe_admission_contract",
            "",
        )
    ).strip()
    report_value = str(
        getattr(
            args,
            "stage_b_dense_duty_confidence_probe_admission_report",
            "",
        )
        or ""
    ).strip()
    if scope == "probe":
        if contract != "disabled_for_probe_v1" or report_value:
            raise RuntimeError(
                "word-veto probe must disable the formal-promotion admission gate"
            )
        return None
    if scope != "formal" or contract != formal_contract:
        raise RuntimeError(
            "formal word-veto confidence requires the U300 strict1607 "
            "promotion contract"
        )
    if not report_value:
        raise RuntimeError(
            "formal word-veto confidence lacks its promotion report path"
        )
    report = Path(report_value).expanduser().resolve(strict=True)
    if report != promotion.REPORT.resolve(strict=True):
        raise RuntimeError(
            "formal word-veto confidence points at a noncanonical promotion report"
        )
    audit = promotion.verify_admission_report(report)
    setattr(
        args,
        "stage_b_dense_duty_confidence_probe_admission_audit",
        audit,
    )
    return audit


def _validate_stage_b_dense_duty_args(args) -> None:
    if not bool(getattr(args, "stage_b_dense_duty", False)):
        return
    phase = str(getattr(args, "stage_b_dense_duty_phase", "")).strip().lower()
    if phase not in {"rank", "confidence"}:
        raise RuntimeError(
            "stage_b_dense_duty_phase must be exactly 'rank' or 'confidence'"
        )
    if type(getattr(args, "stage_b_dense_duty_no_stageb_teacher", None)) is not bool or not getattr(
        args, "stage_b_dense_duty_no_stageb_teacher"
    ):
        raise RuntimeError(
            "dense-duty Stage B requires the explicit no-Stage-B-teacher contract"
        )
    if int(getattr(args, "stage_b_v11_num_layers", 0)) != 6:
        raise RuntimeError("dense-duty Stage B requires the complete six-layer decoder")
    if int(getattr(args, "stage_b_v11_candidate_topk", 0)) != 50:
        raise RuntimeError("dense-duty Stage B requires fixed patch Top-50 candidates")
    if bool(getattr(args, "stage_b_v15_patch_rank_fusion", True)):
        raise RuntimeError("dense-duty Stage B forbids additive patch/text rank fusion")
    if getattr(args, "stage_b_v15_exclude_canonical_from_score", None) is not True:
        raise RuntimeError(
            "dense-duty Stage B requires canonical exclusion so patch scores "
            "own category-only ranking"
        )
    if str(getattr(args, "stage_b_v21_token_objective", "")).strip() not in {
        "edit_bce",
        "edit_focal",
        "edit_bce_group_balanced",
    }:
        raise RuntimeError(
            "dense-duty Stage B requires an edit-aware token objective"
        )
    if getattr(args, "stage_b_v21_allow_legacy_token_diff_fallback", None) is not False:
        raise RuntimeError(
            "dense-duty Stage B forbids legacy full-pair token-diff supervision"
        )
    token_edit_query_scope = str(
        getattr(args, "stage_b_v21_token_edit_query_scope", "target_iou_v1")
    ).strip().lower()
    if token_edit_query_scope not in {
        "target_iou_v1",
        "target_iou_union_detached_final_confidence_base_argmax_v2",
        "target_iou_union_detached_role_complete_confidence_base_argmax_v3",
        "candidate_complete_trace_v4",
    }:
        raise RuntimeError(
            "dense-duty Stage B has an unknown changed-token query scope"
        )
    if getattr(args, "stage_b_dense_duty_allow_incidental_trace_edits", None) is not False:
        raise RuntimeError(
            "formal dense-duty token roles require exact single-edit reconstruction"
        )
    if str(getattr(args, "stage_b_dense_duty_token_role_source", "")).strip() != (
        "exact_direct_trace_v1"
    ):
        raise RuntimeError(
            "dense-duty Stage B requires the exact direct-trace token-role source"
        )
    score_ownership = str(
        getattr(args, "stage_b_v22_score_ownership", "")
    ).strip()
    if score_ownership not in {
        "independent_decoders_two_phase",
        "rank_tower_stopgrad_token_adapter_two_phase",
    }:
        raise RuntimeError("dense-duty Stage B requires an audited two-phase score owner")
    adapter_contract = (
        score_ownership == "rank_tower_stopgrad_token_adapter_two_phase"
    )
    if adapter_contract:
        if phase != "confidence":
            raise RuntimeError(
                "the CVPR confidence-adapter contract reuses a completed rank phase"
            )
        if int(getattr(args, "stage_b_dense_duty_confidence_adapter_dim", 0)) <= 0:
            raise RuntimeError("confidence adapter requires a positive bottleneck width")
        if (
            isinstance(
                getattr(args, "stage_b_dense_duty_confidence_init_seed", None), bool
            )
            or int(
                getattr(args, "stage_b_dense_duty_confidence_init_seed", -1)
            )
            < 0
        ):
            raise RuntimeError("confidence adapter requires a deterministic init seed")
        if str(
            getattr(args, "stage_b_dense_duty_confidence_token_contract", "")
        ).strip() != "detached_rank_token_minus_zero_init_residual_v1":
            raise RuntimeError(
                "confidence adapter requires detached rank-token logits minus a "
                "zero-initialized residual"
            )
        confidence_revision = str(
            getattr(args, "stage_b_dense_duty_confidence_revision", "")
        ).strip()
        head_gradient_contract = str(
            getattr(
                args,
                "stage_b_dense_duty_confidence_head_gradient_contract",
                "shared_token_veto_global_absolute_v1",
            )
        ).strip().lower()
        if head_gradient_contract not in {
            "shared_token_veto_global_absolute_v1",
            "split_token_veto_global_absolute_v2",
            "split_token_veto_global_absolute_joint_clip_v3",
            "split_token_veto_global_trust_veto_v4",
            "split_token_veto_deployed_router_global_absolute_v5",
            "split_token_veto_candidate_absolute_sample_calibrator_v6",
            "split_token_veto_fulltext_global_absolute_v7",
            "split_token_veto_local_candidate_global_absolute_v8",
            "split_token_veto_deployment_owned_global_absolute_v9",
            "split_token_veto_deployment_owned_query_global_absolute_v10",
            "split_token_veto_deployment_owned_query_veto_global_absolute_v11",
        }:
            raise RuntimeError(
                "confidence adapter has an unknown head-gradient contract"
            )
        if (
            head_gradient_contract
            == "split_token_veto_candidate_absolute_sample_calibrator_v6"
            and confidence_revision
            != "word_veto_candidate_sample_calibrator_split_v52"
        ):
            raise RuntimeError(
                "candidate-absolute/sample-calibrator heads require the exact "
                "v52 surface"
            )
        if (
            head_gradient_contract
            == "split_token_veto_fulltext_global_absolute_v7"
            and confidence_revision
            not in {
                "word_veto_rank_full_expression_global_absolute_v53",
                "word_veto_rank_full_expression_global_absolute_exact_residual_v54",
            }
        ):
            raise RuntimeError(
                "full-expression global-absolute v7 heads require an exact v53/v54 surface"
            )
        if (
            head_gradient_contract
            == "split_token_veto_local_candidate_global_absolute_v8"
            and confidence_revision
            != "word_veto_rank_full_expression_global_independent_absolute_v55"
        ):
            raise RuntimeError(
                "local-candidate/global-absolute v8 heads require the exact v55 surface"
            )
        if (
            head_gradient_contract
            == "split_token_veto_deployment_owned_global_absolute_v9"
            and confidence_revision
            not in {
                "word_veto_rank_full_expression_deployment_owned_global_v56",
                "word_veto_rank_full_expression_deployed_global_balanced_absolute_v57",
                "word_veto_rank_full_expression_deployment_owned_global_stable_fpr95_active_set_v58",
            }
        ):
            raise RuntimeError(
                "deployment-owned global-absolute v9 heads require the exact "
                "V56/V57/V58 surface"
            )
        if (
            head_gradient_contract
            == "split_token_veto_deployment_owned_query_global_absolute_v10"
            and confidence_revision
            != "word_veto_rank_full_expression_deployment_owned_query_global_v59"
        ):
            raise RuntimeError(
                "deployment-owned query-global v10 heads require the exact V59 surface"
            )
        if (
            head_gradient_contract
            == "split_token_veto_deployment_owned_query_veto_global_absolute_v11"
            and confidence_revision
            != "word_veto_rank_full_expression_deployment_owned_query_veto_v60"
        ):
            raise RuntimeError(
                "deployment-owned query-veto v11 heads require the exact V60 surface"
            )
        if (
            head_gradient_contract
            == "split_token_veto_global_trust_veto_v4"
            and confidence_revision
            != "word_veto_candidate_split_global_trust_veto_v49"
        ):
            raise RuntimeError(
                "split global trust/veto heads require the exact v49 surface"
            )
        if (
            head_gradient_contract
            == "split_token_veto_deployed_router_global_absolute_v5"
            and confidence_revision
            != "word_veto_candidate_split_independent_deployed_router_v51"
        ):
            raise RuntimeError(
                "independent deployed-router heads require the exact v51 surface"
            )
        if (
            head_gradient_contract
            == "split_token_veto_global_absolute_v2"
            and confidence_revision
            not in {
                "word_veto_candidate_asymmetric_deployed_routing_v43",
                "word_veto_candidate_split_positive_tail_v46",
                "word_veto_candidate_split_boundary_routing_v47",
                "word_veto_candidate_split_fpr_active_set_v48",
                "word_veto_candidate_split_strong_boundary_routing_v50",
            }
        ):
            raise RuntimeError(
                "split token-veto/global-absolute heads require an exact "
                "declared candidate deployed-routing surface"
            )
        if (
            head_gradient_contract
            == "split_token_veto_global_absolute_joint_clip_v3"
            and confidence_revision
            == "word_veto_candidate_split_boundary_routing_v47"
        ):
            raise RuntimeError(
                "v47 boundary routing requires split-v2 heads, not joint-clipped "
                "token-veto/global-absolute heads"
            )
        if (
            head_gradient_contract
            == "split_token_veto_global_absolute_joint_clip_v3"
            and confidence_revision != "word_veto_candidate_split_tail_aligned_v45"
        ):
            raise RuntimeError(
                "joint-clipped split token-veto/global-absolute heads require "
                "the exact v45 tail-aligned surface"
            )
        if head_gradient_contract in {
            "split_token_veto_global_absolute_v2",
            "split_token_veto_global_absolute_joint_clip_v3",
            "split_token_veto_global_trust_veto_v4",
            "split_token_veto_deployed_router_global_absolute_v5",
            "split_token_veto_candidate_absolute_sample_calibrator_v6",
            "split_token_veto_fulltext_global_absolute_v7",
        }:
            split_clip_max_norm = float(getattr(args, "clip_max_norm", 0.0))
            if (
                not math.isfinite(split_clip_max_norm)
                or split_clip_max_norm <= 0.0
            ):
                raise RuntimeError(
                    "split token-veto/global-absolute heads require a finite "
                    "positive clip_max_norm"
                )
        deployed_routing_weight = float(
            getattr(
                args,
                "stage_b_dense_duty_deployed_veto_routing_weight",
                0.0,
            )
            or 0.0
        )
        deployed_positive_max = float(
            getattr(
                args,
                "stage_b_dense_duty_deployed_veto_positive_max",
                0.1,
            )
        )
        deployed_tn_min = float(
            getattr(
                args,
                "stage_b_dense_duty_deployed_veto_tn_min",
                0.9,
            )
        )
        deployed_routing_reduction = str(
            getattr(
                args,
                "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
                "balanced_mean_v1",
            )
        ).strip().lower()
        positive_trust_reduction = str(
            getattr(
                args,
                "stage_b_v15_tail_queue_positive_trust_reduction_contract",
                "mean_v1",
            )
        ).strip().lower()
        negative_reduction = str(
            getattr(
                args,
                "stage_b_v15_tail_queue_negative_reduction_contract",
                "all_mean_v1",
            )
        ).strip().lower()
        if confidence_revision == (
            "word_veto_candidate_asymmetric_deployed_routing_v43"
        ):
            if (
                deployed_routing_weight != 0.1
                or deployed_positive_max != 0.1
                or deployed_tn_min != 0.9
                or token_edit_query_scope != "target_iou_v1"
                or deployed_routing_reduction != "balanced_mean_v1"
                or positive_trust_reduction != "mean_v1"
                or negative_reduction != "all_mean_v1"
            ):
                raise RuntimeError(
                    "v43 deployed routing requires weight=0.1, positive_max=0.1, "
                    "tn_min=0.9, target_iou_v1, and mean reductions"
                )
        elif confidence_revision == "word_veto_candidate_split_tail_aligned_v45":
            if (
                head_gradient_contract
                != "split_token_veto_global_absolute_joint_clip_v3"
                or deployed_routing_weight != 1.0
                or deployed_positive_max != 0.1
                or deployed_tn_min != 0.9
                or token_edit_query_scope != "target_iou_v1"
                or deployed_routing_reduction
                != "balanced_top_quarter_cvar_v2"
                or positive_trust_reduction != "top_quarter_cvar_v2"
                or negative_reduction != "all_mean_v1"
                or str(
                    getattr(
                        args,
                        "stage_b_dense_duty_positive_trust_contract",
                        "",
                    )
                ).strip().lower()
                != "absolute_global_confidence_logit_v2"
            ):
                raise RuntimeError(
                    "v45 tail alignment requires joint-clipped split heads, "
                    "routing weight=1, 0.1/0.9 margins, target_iou_v1, "
                    "top-quarter reductions, and absolute-global trust"
                )
        elif confidence_revision == "word_veto_candidate_split_positive_tail_v46":
            if (
                head_gradient_contract != "split_token_veto_global_absolute_v2"
                or deployed_routing_weight != 0.1
                or deployed_positive_max != 0.1
                or deployed_tn_min != 0.9
                or token_edit_query_scope != "target_iou_v1"
                or deployed_routing_reduction != "balanced_mean_v1"
                or positive_trust_reduction != "top_quarter_cvar_v2"
                or negative_reduction != "all_mean_v1"
                or str(
                    getattr(
                        args,
                        "stage_b_dense_duty_positive_trust_contract",
                        "",
                    )
                ).strip().lower()
                != "absolute_global_confidence_logit_v2"
            ):
                raise RuntimeError(
                    "v46 positive-tail alignment requires split-v2 heads, "
                    "routing weight=0.1, 0.1/0.9 margins, target_iou_v1, "
                    "balanced-mean routing, top-quarter positive trust, and "
                    "absolute-global trust"
                )
        elif confidence_revision == "word_veto_candidate_split_boundary_routing_v47":
            if (
                head_gradient_contract != "split_token_veto_global_absolute_v2"
                or deployed_routing_weight != 0.1
                or deployed_positive_max != 0.1
                or deployed_tn_min != 0.9
                or token_edit_query_scope != "target_iou_v1"
                or deployed_routing_reduction
                != "balanced_top_quarter_cvar_v2"
                or positive_trust_reduction != "top_quarter_cvar_v2"
                or negative_reduction != "all_mean_v1"
                or str(
                    getattr(
                        args,
                        "stage_b_dense_duty_positive_trust_contract",
                        "",
                    )
                ).strip().lower()
                != "absolute_global_confidence_logit_v2"
            ):
                raise RuntimeError(
                    "v47 boundary routing requires split-v2 heads, routing "
                    "weight=0.1, 0.1/0.9 margins, target_iou_v1, top-quarter "
                    "routing and positive trust, and absolute-global trust"
                )
        elif confidence_revision == (
            "word_veto_candidate_sample_calibrator_split_v52"
        ):
            if (
                head_gradient_contract
                != "split_token_veto_candidate_absolute_sample_calibrator_v6"
                or deployed_routing_weight != 0.0
                or deployed_positive_max != 0.1
                or deployed_tn_min != 0.9
                or token_edit_query_scope != "target_iou_v1"
                or deployed_routing_reduction
                != "balanced_top_quarter_cvar_v2"
                or positive_trust_reduction != "top_quarter_cvar_v2"
                or negative_reduction != "all_mean_v1"
                or str(
                    getattr(
                        args,
                        "stage_b_dense_duty_positive_trust_contract",
                        "",
                    )
                ).strip().lower()
                != "absolute_global_confidence_logit_v2"
            ):
                raise RuntimeError(
                    "v52 candidate/sample calibration requires routing weight=0, "
                    "the V51 tail/trust surface, and its split-v6 three-owner head"
                )
        elif confidence_revision == (
            "word_veto_rank_full_expression_global_absolute_v53"
        ):
            if (
                head_gradient_contract
                != "split_token_veto_fulltext_global_absolute_v7"
                or deployed_routing_weight != 0.0
                or deployed_positive_max != 0.1
                or deployed_tn_min != 0.9
                or token_edit_query_scope != "target_iou_v1"
                or deployed_routing_reduction
                != "balanced_top_quarter_cvar_v2"
                or positive_trust_reduction != "top_quarter_cvar_v2"
                or negative_reduction != "all_mean_v1"
                or str(
                    getattr(
                        args,
                        "stage_b_dense_duty_positive_trust_contract",
                        "",
                    )
                ).strip().lower()
                != "absolute_global_confidence_logit_v2"
            ):
                raise RuntimeError(
                    "v53 full-expression confidence requires routing weight=0, "
                    "the V52 tail/trust surface, and its split-v7 two-owner head"
                )
        elif confidence_revision == (
            "word_veto_rank_full_expression_global_absolute_exact_residual_v54"
        ):
            if (
                head_gradient_contract
                != "split_token_veto_fulltext_global_absolute_v7"
                or deployed_routing_weight != 0.0
                or deployed_positive_max != 0.1
                or deployed_tn_min != 0.9
                or token_edit_query_scope != "target_iou_v1"
                or deployed_routing_reduction
                != "balanced_top_quarter_cvar_v2"
                or positive_trust_reduction != "top_quarter_cvar_v2"
                or negative_reduction != "all_mean_v1"
                or str(
                    getattr(
                        args,
                        "stage_b_dense_duty_positive_trust_contract",
                        "",
                    )
                ).strip().lower()
                != "exact_frozen_rank_max_confidence_delta_v3"
            ):
                raise RuntimeError(
                    "v54 exact-residual confidence requires the V53 two-owner "
                    "surface with exact frozen-rank-max positive trust"
                )
        elif confidence_revision == (
            "word_veto_rank_full_expression_global_independent_absolute_v55"
        ):
            if (
                head_gradient_contract
                != "split_token_veto_local_candidate_global_absolute_v8"
                or deployed_routing_weight != 0.0
                or deployed_positive_max != 0.1
                or deployed_tn_min != 0.9
                or token_edit_query_scope != "target_iou_v1"
                or deployed_routing_reduction
                != "balanced_top_quarter_cvar_v2"
                or positive_trust_reduction != "top_quarter_cvar_v2"
                or negative_reduction != "all_mean_v1"
                or str(
                    getattr(
                        args,
                        "stage_b_dense_duty_positive_trust_contract",
                        "",
                    )
                ).strip().lower()
                != "absolute_global_pool_logit_v4"
            ):
                raise RuntimeError(
                    "v55 independent confidence requires the split-v8 local/global "
                    "surface and exact deployed pool-absolute positive trust"
                )
        elif confidence_revision == (
            "word_veto_rank_full_expression_deployment_owned_global_v56"
        ):
            if (
                head_gradient_contract
                != "split_token_veto_deployment_owned_global_absolute_v9"
                or deployed_routing_weight != 0.0
                or deployed_positive_max != 0.1
                or deployed_tn_min != 0.9
                or token_edit_query_scope != "target_iou_v1"
                or deployed_routing_reduction
                != "balanced_top_quarter_cvar_v2"
                or positive_trust_reduction != "top_quarter_cvar_v2"
                or negative_reduction != "all_mean_v1"
                or float(
                    getattr(args, "stage_b_v14_local_absolute_weight", -1.0)
                )
                != 0.0
                or int(getattr(args, "stage_b_v11_trainable_params_min", -1))
                != 468_164
                or int(getattr(args, "stage_b_v11_trainable_params_max", -1))
                != 468_164
                or str(
                    getattr(
                        args,
                        "stage_b_dense_duty_positive_trust_contract",
                        "",
                    )
                ).strip().lower()
                != "absolute_global_pool_logit_v4"
            ):
                raise RuntimeError(
                    "v56 deployment-owned confidence requires local absolute "
                    "weight=0, exactly 468164 active parameters, the split-v9 "
                    "owner surface, and exact deployed pool-absolute trust"
                )
        elif confidence_revision == (
            "word_veto_rank_full_expression_deployed_global_balanced_absolute_v57"
        ):
            if (
                head_gradient_contract
                != "split_token_veto_deployment_owned_global_absolute_v9"
                or deployed_routing_weight != 0.0
                or deployed_positive_max != 0.1
                or deployed_tn_min != 0.9
                or token_edit_query_scope != "target_iou_v1"
                or deployed_routing_reduction
                != "balanced_top_quarter_cvar_v2"
                or positive_trust_reduction != "top_quarter_cvar_v2"
                or negative_reduction != "all_mean_v1"
                or float(
                    getattr(args, "stage_b_v14_local_absolute_weight", -1.0)
                )
                != 0.0
                or float(
                    getattr(
                        args,
                        "stage_b_dense_duty_deployed_global_absolute_weight",
                        -1.0,
                    )
                )
                != 1.0
                or float(
                    getattr(
                        args,
                        "stage_b_dense_duty_deployed_global_absolute_gamma",
                        -1.0,
                    )
                )
                != 1.0
                or int(getattr(args, "stage_b_v11_trainable_params_min", -1))
                != 468_164
                or int(getattr(args, "stage_b_v11_trainable_params_max", -1))
                != 468_164
                or str(
                    getattr(
                        args,
                        "stage_b_dense_duty_positive_trust_contract",
                        "",
                    )
                ).strip().lower()
                != "absolute_global_pool_logit_v4"
            ):
                raise RuntimeError(
                    "v57 deployed-global balanced absolute confidence requires "
                    "the unchanged V56 two-owner surface, local candidate "
                    "weight=0, and deployed-global focal BCE weight/gamma=1"
                )
        elif confidence_revision == (
            "word_veto_rank_full_expression_deployment_owned_global_stable_fpr95_active_set_v58"
        ):
            if (
                head_gradient_contract
                != "split_token_veto_deployment_owned_global_absolute_v9"
                or deployed_routing_weight != 0.0
                or deployed_positive_max != 0.1
                or deployed_tn_min != 0.9
                or token_edit_query_scope != "target_iou_v1"
                or deployed_routing_reduction
                != "balanced_top_quarter_cvar_v2"
                or positive_trust_reduction != "top_quarter_cvar_v2"
                or negative_reduction
                != "exact_fpr95_active_set_all_count_mean_v2"
                or float(
                    getattr(args, "stage_b_v14_local_absolute_weight", -1.0)
                )
                != 0.0
                or float(
                    getattr(
                        args,
                        "stage_b_dense_duty_deployed_global_absolute_weight",
                        -1.0,
                    )
                )
                != 0.0
                or int(getattr(args, "stage_b_v11_trainable_params_min", -1))
                != 468_164
                or int(getattr(args, "stage_b_v11_trainable_params_max", -1))
                != 468_164
                or str(
                    getattr(
                        args,
                        "stage_b_dense_duty_positive_trust_contract",
                        "",
                    )
                ).strip().lower()
                != "absolute_global_pool_logit_v4"
            ):
                raise RuntimeError(
                    "v58 deployment-owned stable FPR95 active-set confidence "
                    "requires the unchanged V56 two-owner surface, no candidate "
                    "or deployed BCE, and all-count-normalized exact active-set "
                    "reduction"
                )
        elif confidence_revision == (
            "word_veto_rank_full_expression_deployment_owned_query_global_v59"
        ):
            if (
                head_gradient_contract
                != "split_token_veto_deployment_owned_query_global_absolute_v10"
                or deployed_routing_weight != 0.0
                or deployed_positive_max != 0.1
                or deployed_tn_min != 0.9
                or token_edit_query_scope != "target_iou_v1"
                or deployed_routing_reduction
                != "balanced_top_quarter_cvar_v2"
                or positive_trust_reduction != "top_quarter_cvar_v2"
                or negative_reduction != "all_mean_v1"
                or float(
                    getattr(args, "stage_b_v14_local_absolute_weight", -1.0)
                )
                != 0.0
                or float(
                    getattr(
                        args,
                        "stage_b_dense_duty_deployed_global_absolute_weight",
                        -1.0,
                    )
                )
                != 0.0
                or int(getattr(args, "stage_b_v11_trainable_params_min", -1))
                != 534_725
                or int(getattr(args, "stage_b_v11_trainable_params_max", -1))
                != 534_725
                or str(
                    getattr(
                        args,
                        "stage_b_dense_duty_positive_trust_contract",
                        "",
                    )
                ).strip().lower()
                != "absolute_global_confidence_logit_v2"
            ):
                raise RuntimeError(
                    "v59 deployment-owned query-global confidence requires the "
                    "split-v10 owner, all-TN/q05 deployed objectives, local and "
                    "deployed focal weights=0, and exactly 534725 active parameters"
                )
        elif confidence_revision == (
            "word_veto_rank_full_expression_deployment_owned_query_veto_v60"
        ):
            full_decoder_verifier = bool(
                getattr(
                    args,
                    "stage_b_dense_duty_confidence_full_decoder_verifier",
                    False,
                )
            )
            veto_only_patch_softmin = bool(
                getattr(
                    args,
                    "stage_b_dense_duty_confidence_veto_only_patch_softmin",
                    False,
                )
            )
            candidate_trace_contract = str(
                getattr(
                    args,
                    "stage_b_dense_duty_confidence_candidate_trace_contract",
                    "off_v1",
                )
            ).strip().lower()
            candidate_trace_contracts = {
                "off_v1": (
                    "target_iou_v1",
                    "rank_cloned_full_decoder_patch_softmin_veto_v2",
                    "full_decoder_token_entailment_patch_weighted_"
                    "existential_veto_v62",
                    25_530_881,
                ),
                "candidate_complete_free_head_coverage_v1": (
                    "candidate_complete_trace_v4",
                    "rank_cloned_full_decoder_candidate_complete_free_head_v3",
                    "candidate_complete_trace_free_head_coverage_c1",
                    25_530_881,
                ),
                "candidate_complete_monotone_token_entailment_v2": (
                    "candidate_complete_trace_v4",
                    "rank_cloned_full_decoder_candidate_complete_monotone_v4",
                    "candidate_complete_trace_monotone_token_entailment_c2",
                    25_464_320,
                ),
            }
            if candidate_trace_contract not in candidate_trace_contracts:
                raise RuntimeError(
                    "v60 query-veto confidence has an unknown candidate trace contract"
                )
            (
                expected_token_scope,
                expected_patch_softmin_capacity,
                expected_patch_softmin_variant,
                expected_patch_softmin_trainable,
            ) = candidate_trace_contracts[candidate_trace_contract]
            expected_trainable = (
                expected_patch_softmin_trainable
                if veto_only_patch_softmin
                else 25_664_258
                if full_decoder_verifier
                else 534_725
            )
            capacity_contract = str(
                getattr(
                    args,
                    "stage_b_dense_duty_confidence_capacity_contract",
                    "lightweight_adapter_v1",
                )
            ).strip().lower()
            confidence_variant = str(
                getattr(args, "stage_b_dense_duty_confidence_variant", "")
            ).strip().lower()
            candidate_depth_contract = (
                float(
                    getattr(
                        args,
                        "stage_b_dense_duty_candidate_depth_all_weight",
                        -1.0,
                    )
                ),
                float(
                    getattr(
                        args,
                        "stage_b_dense_duty_candidate_depth_escape_weight",
                        -1.0,
                    )
                ),
                float(
                    getattr(
                        args,
                        "stage_b_dense_duty_candidate_depth_positive_weight",
                        -1.0,
                    )
                ),
                float(
                    getattr(
                        args,
                        "stage_b_dense_duty_candidate_depth_tn_margin",
                        -1.0,
                    )
                ),
                float(
                    getattr(
                        args,
                        "stage_b_dense_duty_candidate_depth_escape_margin",
                        -1.0,
                    )
                ),
                float(
                    getattr(
                        args,
                        "stage_b_dense_duty_candidate_depth_positive_max",
                        -1.0,
                    )
                ),
                float(
                    getattr(
                        args,
                        "stage_b_dense_duty_candidate_depth_temperature",
                        -1.0,
                    )
                ),
            )
            if (
                head_gradient_contract
                != "split_token_veto_deployment_owned_query_veto_global_absolute_v11"
                or deployed_routing_weight != 0.0
                or deployed_positive_max != 0.1
                or deployed_tn_min != 0.9
                or token_edit_query_scope != expected_token_scope
                or deployed_routing_reduction
                != "balanced_top_quarter_cvar_v2"
                or positive_trust_reduction != "top_quarter_cvar_v2"
                or negative_reduction != "all_mean_v1"
                or float(
                    getattr(args, "stage_b_v14_local_absolute_weight", -1.0)
                )
                != 0.0
                or float(
                    getattr(
                        args,
                        "stage_b_dense_duty_deployed_global_absolute_weight",
                        -1.0,
                    )
                )
                != 0.0
                or int(getattr(args, "stage_b_v11_trainable_params_min", -1))
                != expected_trainable
                or int(getattr(args, "stage_b_v11_trainable_params_max", -1))
                != expected_trainable
                or (
                    veto_only_patch_softmin
                    and capacity_contract
                    != expected_patch_softmin_capacity
                )
                or (
                    veto_only_patch_softmin
                    and confidence_variant != expected_patch_softmin_variant
                )
                or (
                    full_decoder_verifier
                    and not veto_only_patch_softmin
                    and capacity_contract
                    != "rank_cloned_full_decoder_6layer_256d_v1"
                )
                or (
                    not full_decoder_verifier
                    and capacity_contract != "lightweight_adapter_v1"
                )
                or str(
                    getattr(
                        args,
                        "stage_b_dense_duty_positive_trust_contract",
                        "",
                    )
                ).strip().lower()
                != "absolute_global_confidence_logit_v2"
                or (
                    candidate_trace_contract != "off_v1"
                    and (
                        not full_decoder_verifier
                        or not veto_only_patch_softmin
                        or str(
                            getattr(args, "stage_b_v21_token_objective", "")
                        ).strip().lower()
                        != "edit_bce_group_balanced"
                        or candidate_depth_contract
                        != (1.0, 1.0, 1.0, 0.5, 0.5, 0.05, 0.1)
                        or float(
                            getattr(
                                args,
                                "stage_b_dense_duty_confidence_token_depth_base_scale",
                                0.0,
                            )
                        )
                        <= 0.0
                    )
                )
                or (
                    candidate_trace_contract
                    == "candidate_complete_monotone_token_entailment_v2"
                    and (
                        float(
                            getattr(
                                args,
                                "stage_b_dense_duty_raw_veto_gate_weight",
                                -1.0,
                            )
                        )
                        != 0.0
                        or float(
                            getattr(
                                args,
                                "stage_b_dense_duty_raw_veto_carrier_pair_weight",
                                -1.0,
                            )
                        )
                        != 0.0
                    )
                )
            ):
                raise RuntimeError(
                    "v60 deployment-owned query-veto confidence requires the "
                    "split-v11 owner, all-TN/q05 deployed objectives, local and "
                    "deployed focal weights=0, and the declared exact lightweight "
                    "or full-decoder capacity surface"
                )
        elif confidence_revision == (
            "word_veto_candidate_split_independent_deployed_router_v51"
        ):
            if (
                head_gradient_contract
                != "split_token_veto_deployed_router_global_absolute_v5"
                or deployed_routing_weight != 0.1
                or deployed_positive_max != 0.1
                or deployed_tn_min != 0.9
                or token_edit_query_scope != "target_iou_v1"
                or deployed_routing_reduction
                != "balanced_top_quarter_cvar_v2"
                or positive_trust_reduction != "top_quarter_cvar_v2"
                or negative_reduction != "all_mean_v1"
                or str(
                    getattr(
                        args,
                        "stage_b_dense_duty_positive_trust_contract",
                        "",
                    )
                ).strip().lower()
                != "absolute_global_confidence_logit_v2"
            ):
                raise RuntimeError(
                    "v51 independent deployed routing requires the exact V47 "
                    "loss/reduction surface, routing weight=0.1, and its split-v5 "
                    "three-owner head"
                )
        elif confidence_revision == (
            "word_veto_candidate_split_strong_boundary_routing_v50"
        ):
            if (
                head_gradient_contract != "split_token_veto_global_absolute_v2"
                or deployed_routing_weight != 0.25
                or deployed_positive_max != 0.1
                or deployed_tn_min != 0.9
                or token_edit_query_scope != "target_iou_v1"
                or deployed_routing_reduction
                != "balanced_top_quarter_cvar_v2"
                or positive_trust_reduction != "top_quarter_cvar_v2"
                or negative_reduction != "all_mean_v1"
                or str(
                    getattr(
                        args,
                        "stage_b_dense_duty_positive_trust_contract",
                        "",
                    )
                ).strip().lower()
                != "absolute_global_confidence_logit_v2"
            ):
                raise RuntimeError(
                    "v50 strong boundary routing requires the exact V47 "
                    "surface with routing weight=0.25"
                )
        elif confidence_revision == (
            "word_veto_candidate_split_fpr_active_set_v48"
        ):
            if (
                head_gradient_contract != "split_token_veto_global_absolute_v2"
                or deployed_routing_weight != 0.1
                or deployed_positive_max != 0.1
                or deployed_tn_min != 0.9
                or token_edit_query_scope != "target_iou_v1"
                or deployed_routing_reduction
                != "balanced_top_quarter_cvar_v2"
                or positive_trust_reduction != "top_quarter_cvar_v2"
                or negative_reduction
                != "exact_fpr95_active_set_mean_v1"
                or str(
                    getattr(
                        args,
                        "stage_b_dense_duty_positive_trust_contract",
                        "",
                    )
                ).strip().lower()
                != "absolute_global_confidence_logit_v2"
            ):
                raise RuntimeError(
                    "v48 FPR active set requires the exact V47 split-v2 "
                    "routing/trust surface and exact-q05 active-set negative "
                    "reduction"
                )
        elif confidence_revision == (
            "word_veto_candidate_split_global_trust_veto_v49"
        ):
            if (
                head_gradient_contract
                != "split_token_veto_global_trust_veto_v4"
                or deployed_routing_weight != 0.1
                or deployed_positive_max != 0.1
                or deployed_tn_min != 0.9
                or token_edit_query_scope != "target_iou_v1"
                or deployed_routing_reduction
                != "balanced_top_quarter_cvar_v2"
                or positive_trust_reduction != "top_quarter_cvar_v2"
                or negative_reduction != "all_mean_v1"
                or str(
                    getattr(
                        args,
                        "stage_b_dense_duty_positive_trust_contract",
                        "",
                    )
                ).strip().lower()
                != "absolute_global_confidence_logit_v2"
            ):
                raise RuntimeError(
                    "v49 global trust/veto split requires the exact V47 "
                    "routing/trust/reduction surface and its split-v4 head"
                )
        elif (
            deployed_routing_weight != 0.0
            or deployed_routing_reduction != "balanced_mean_v1"
            or positive_trust_reduction != "mean_v1"
            or negative_reduction != "all_mean_v1"
        ):
            raise RuntimeError(
                "deployed veto routing and hard-tail reductions are restricted "
                "to their declared revisions"
            )
        if token_edit_query_scope in {
            "target_iou_union_detached_final_confidence_base_argmax_v2",
            "target_iou_union_detached_role_complete_confidence_base_argmax_v3",
        }:
            exact_carrier_surface = (
                str(getattr(args, "stage_b_v21_token_objective", "")).strip()
                == "edit_bce"
                and str(
                    getattr(
                        args,
                        "stage_b_dense_duty_confidence_phrase_aggregation",
                        "",
                    )
                ).strip().lower()
                == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                and confidence_revision
                == "word_veto_candidate_asymmetric_confidence_v32"
                and str(
                    getattr(
                        args,
                        "stage_b_v15_tail_queue_positive_gradient_contract",
                        "",
                    )
                ).strip()
                == "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
                and float(
                    getattr(
                        args,
                        "stage_b_dense_duty_confidence_veto_gate_offset",
                        -1.0,
                    )
                )
                == 0.0
                and getattr(args, "stage_b_v14_global_tn_all_candidates", None)
                is True
                and str(
                    getattr(
                        args,
                        "stage_b_dense_duty_confidence_tn_scope",
                        "",
                    )
                ).strip().lower()
                == "direct_trace_valid_v1"
            )
            if not exact_carrier_surface:
                carrier_version = (
                    "v41"
                    if token_edit_query_scope
                    == "target_iou_union_detached_role_complete_"
                    "confidence_base_argmax_v3"
                    else "v40"
                )
                raise RuntimeError(
                    "global-carrier token supervision is restricted to the exact "
                    f"{carrier_version} candidate-confidence training surface"
                )
        if confidence_revision == (
            "word_veto_rank_full_expression_deployment_owned_query_veto_v60"
        ):
            expected_pool_feature_contract = (
                "detached_rank_full_expression_token_conditioned_query_veto_"
                "deployment_owned_global_pool_v15"
            )
        elif confidence_revision == (
            "word_veto_rank_full_expression_deployment_owned_query_global_v59"
        ):
            expected_pool_feature_contract = (
                "detached_rank_full_expression_monotone_query_"
                "deployment_owned_global_pool_v14"
            )
        elif confidence_revision in {
            "word_veto_rank_full_expression_deployment_owned_global_v56",
            "word_veto_rank_full_expression_deployed_global_balanced_absolute_v57",
            "word_veto_rank_full_expression_deployment_owned_global_stable_fpr95_active_set_v58",
        }:
            expected_pool_feature_contract = (
                "detached_rank_full_expression_deployment_owned_global_pool_v13"
            )
        elif confidence_revision == (
            "word_veto_rank_full_expression_global_independent_absolute_v55"
        ):
            expected_pool_feature_contract = (
                "detached_rank_full_expression_local_candidate_"
                "frozen_rank_global_pool_v12"
            )
        elif confidence_revision == (
            "word_veto_rank_full_expression_global_absolute_exact_residual_v54"
        ):
            expected_pool_feature_contract = (
                "detached_rank_full_expression_candidate_residual_global_pool_"
                "exact_rank_max_reference_v11"
            )
        elif confidence_revision == (
            "word_veto_rank_full_expression_global_absolute_v53"
        ):
            expected_pool_feature_contract = (
                "detached_rank_full_expression_candidate_residual_"
                "global_pool_v10"
            )
        elif confidence_revision == "word_veto_candidate_set_attention_confidence_v33":
            expected_pool_feature_contract = (
                "detached_candidate_set_attention_absolute_"
                "asymmetric_veto_logits_v9"
            )
        elif confidence_revision in {
            "word_veto_candidate_asymmetric_confidence_v32",
            "word_veto_candidate_asymmetric_deployed_routing_v43",
            "word_veto_candidate_split_tail_aligned_v45",
            "word_veto_candidate_split_positive_tail_v46",
            "word_veto_candidate_split_boundary_routing_v47",
            "word_veto_candidate_split_fpr_active_set_v48",
            "word_veto_candidate_split_global_trust_veto_v49",
            "word_veto_candidate_split_strong_boundary_routing_v50",
            "word_veto_candidate_split_independent_deployed_router_v51",
            "word_veto_candidate_sample_calibrator_split_v52",
        }:
            expected_pool_feature_contract = (
                "detached_candidate_absolute_raw_patch_"
                "asymmetric_veto_logits_v8"
            )
        elif confidence_revision == "word_veto_candidate_normalized_confidence_v31":
            expected_pool_feature_contract = (
                "detached_candidate_absolute_normalized_patch_"
                "amplified_veto_logits_v7"
            )
        elif confidence_revision == "word_veto_candidate_patch_invariant_confidence_v30":
            expected_pool_feature_contract = (
                "detached_candidate_absolute_patch_invariant_"
                "monotone_veto_logits_v6"
            )
        elif confidence_revision == "word_veto_candidate_absolute_confidence_v29":
            expected_pool_feature_contract = (
                "detached_query_modifier_cross_attention_"
                "candidate_absolute_logits_v5"
            )
        elif confidence_revision == "word_veto_cross_attention_absolute_confidence_v28":
            expected_pool_feature_contract = (
                "detached_rank_query_modifier_cross_attention_plus_"
                "patch_statistics_absolute_v4"
            )
        elif confidence_revision in {
            "word_veto_token_conditioned_monotone_depth_v23",
            "word_veto_complementary_trust_veto_v24",
            "word_veto_ungated_monotone_tail_veto_v25",
            "word_veto_floor_gated_monotone_tail_veto_v26",
            "word_veto_independent_absolute_confidence_v27",
        }:
            expected_pool_feature_contract = (
                "detached_rank_query_token_context_plus_patch_statistics_monotone_v3"
            )
        elif confidence_revision in {
            "word_veto_gated_pool_tail_paired_signed_rank_pool_v20",
            "word_veto_continuous_conditional_residual_v21",
            "word_veto_continuous_monotone_depth_v22",
        }:
            expected_pool_feature_contract = (
                "detached_rank_query_plus_patch_statistics_signed_residual_v2"
            )
        else:
            expected_pool_feature_contract = "patch_statistics_only_v1"
        if str(
            getattr(
                args,
                "stage_b_dense_duty_confidence_pool_feature_contract",
                "",
            )
        ).strip() != expected_pool_feature_contract:
            raise RuntimeError(
                "confidence adapter pool-feature contract does not match its "
                "declared revision"
            )
        _validate_stage_b_confidence_rank_evidence_contract(
            args,
            revision=confidence_revision,
        )
        phrase_aggregation = str(
            getattr(
                args,
                "stage_b_dense_duty_confidence_phrase_aggregation",
                "legacy_prob_mean_add_v1",
            )
        ).strip().lower()
        if phrase_aggregation not in {
            "legacy_prob_mean_add_v1",
            "trace_activated_word_veto_product_v1",
            "trace_activated_word_veto_penalty_v2",
            "trace_activated_word_veto_absolute_cap_v4",
            "trace_activated_word_veto_gated_pool_absolute_cap_v5",
        }:
            raise RuntimeError(
                "confidence adapter has an unknown phrase aggregation contract"
            )
        trust_contract = str(
            getattr(
                args,
                "stage_b_dense_duty_positive_trust_contract",
                "pool_residual_v1",
            )
        ).strip().lower()
        if trust_contract not in {
            "pool_residual_v1",
            "net_total_confidence_delta_v1",
            "absolute_global_confidence_logit_v2",
            "exact_frozen_rank_max_confidence_delta_v3",
            "absolute_global_pool_logit_v4",
        }:
            raise RuntimeError(
                "confidence adapter has an unknown positive-trust contract"
            )
        confidence_tn_scope = str(
            getattr(
                args,
                "stage_b_dense_duty_confidence_tn_scope",
                "all_verified_v1",
            )
        ).strip().lower()
        if confidence_tn_scope not in {
            "all_verified_v1",
            "direct_trace_valid_v1",
        }:
            raise RuntimeError("confidence adapter has an unknown TN scope")
        if phrase_aggregation in {
            "trace_activated_word_veto_product_v1",
            "trace_activated_word_veto_penalty_v2",
            "trace_activated_word_veto_absolute_cap_v4",
            "trace_activated_word_veto_gated_pool_absolute_cap_v5",
        }:
            if confidence_revision in {
                "word_veto_rank_full_expression_deployment_owned_query_global_v59",
                "word_veto_rank_full_expression_deployment_owned_query_veto_v60",
            }:
                expected_trust_contract = "absolute_global_confidence_logit_v2"
            elif confidence_revision in {
                "word_veto_rank_full_expression_global_independent_absolute_v55",
                "word_veto_rank_full_expression_deployment_owned_global_v56",
                "word_veto_rank_full_expression_deployed_global_balanced_absolute_v57",
                "word_veto_rank_full_expression_deployment_owned_global_stable_fpr95_active_set_v58",
            }:
                expected_trust_contract = "absolute_global_pool_logit_v4"
            elif confidence_revision == (
                "word_veto_rank_full_expression_global_absolute_exact_residual_v54"
            ):
                expected_trust_contract = (
                    "exact_frozen_rank_max_confidence_delta_v3"
                )
            elif confidence_revision in {
                    "word_veto_independent_absolute_confidence_v27",
                    "word_veto_cross_attention_absolute_confidence_v28",
                    "word_veto_candidate_absolute_confidence_v29",
                    "word_veto_candidate_patch_invariant_confidence_v30",
                    "word_veto_candidate_normalized_confidence_v31",
                    "word_veto_candidate_asymmetric_confidence_v32",
                    "word_veto_candidate_set_attention_confidence_v33",
                    "word_veto_candidate_asymmetric_deployed_routing_v43",
                    "word_veto_candidate_split_tail_aligned_v45",
                    "word_veto_candidate_split_positive_tail_v46",
                    "word_veto_candidate_split_boundary_routing_v47",
                    "word_veto_candidate_split_fpr_active_set_v48",
                    "word_veto_candidate_split_global_trust_veto_v49",
                    "word_veto_candidate_split_strong_boundary_routing_v50",
                    "word_veto_candidate_split_independent_deployed_router_v51",
                    "word_veto_candidate_sample_calibrator_split_v52",
                    "word_veto_rank_full_expression_global_absolute_v53",
                }:
                expected_trust_contract = "absolute_global_confidence_logit_v2"
            else:
                expected_trust_contract = "net_total_confidence_delta_v1"
            if trust_contract != expected_trust_contract:
                raise RuntimeError(
                    "word-veto confidence requires its revision-matched positive "
                    "trust score"
                )
            if confidence_tn_scope != "direct_trace_valid_v1":
                raise RuntimeError(
                    "word-veto confidence requires directly traceable TN rows"
                )
            for field in (
                "stage_b_dense_duty_confidence_word_softmin_temperature",
                "stage_b_dense_duty_confidence_veto_gate_scale",
            ):
                value = float(getattr(args, field, 0.0))
                if not math.isfinite(value) or value <= 0.0:
                    raise RuntimeError(f"word-veto confidence requires positive {field}")
            raw_veto_weight = float(
                getattr(args, "stage_b_dense_duty_raw_veto_gate_weight", 0.0)
                or 0.0
            )
            raw_veto_revision = confidence_revision
            monotone_candidate_trace_raw_veto_disabled = (
                raw_veto_revision
                == "word_veto_rank_full_expression_deployment_owned_query_veto_v60"
                and str(
                    getattr(
                        args,
                        "stage_b_dense_duty_confidence_candidate_trace_contract",
                        "off_v1",
                    )
                ).strip().lower()
                == "candidate_complete_monotone_token_entailment_v2"
            )
            raw_veto_revisions = {
                "word_veto_raw_gate_margin_v3": (
                    "trace_activated_word_veto_penalty_v2"
                ),
                "word_veto_coverage_absolute_cap_v4": (
                    "trace_activated_word_veto_absolute_cap_v4"
                ),
                "word_veto_gated_pool_absolute_cap_v5": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_gated_pool_calibrated_v6": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_gated_pool_carrier_balanced_v7": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_gated_pool_carrier_quarter_v8": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_gated_pool_carrier_pair_v9": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_gated_pool_dual_carrier_pair_v10": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_gated_pool_rank_evidence_v11": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_gated_pool_rank_affine_v12": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_gated_pool_gate_margin_v13": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_gated_pool_carrier_slope_v14": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_gated_pool_carrier_affine_v15": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_gated_pool_tail_ste_v16": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_gated_pool_tail_carrier_v17": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_gated_pool_tail_paired_v18": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_gated_pool_tail_paired_rank_channel_v19": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_gated_pool_tail_paired_signed_rank_pool_v20": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_continuous_conditional_residual_v21": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_continuous_monotone_depth_v22": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_token_conditioned_monotone_depth_v23": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_complementary_trust_veto_v24": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_ungated_monotone_tail_veto_v25": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_floor_gated_monotone_tail_veto_v26": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_independent_absolute_confidence_v27": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_cross_attention_absolute_confidence_v28": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_candidate_absolute_confidence_v29": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_candidate_patch_invariant_confidence_v30": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_candidate_normalized_confidence_v31": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_candidate_asymmetric_confidence_v32": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_candidate_asymmetric_deployed_routing_v43": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_candidate_split_tail_aligned_v45": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_candidate_split_positive_tail_v46": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_candidate_split_boundary_routing_v47": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_candidate_split_strong_boundary_routing_v50": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_candidate_split_independent_deployed_router_v51": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_candidate_sample_calibrator_split_v52": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_rank_full_expression_global_absolute_v53": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_rank_full_expression_global_absolute_exact_residual_v54": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_rank_full_expression_global_independent_absolute_v55": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_rank_full_expression_deployment_owned_global_v56": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_rank_full_expression_deployed_global_balanced_absolute_v57": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_rank_full_expression_deployment_owned_global_stable_fpr95_active_set_v58": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_rank_full_expression_deployment_owned_query_global_v59": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_rank_full_expression_deployment_owned_query_veto_v60": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_candidate_split_fpr_active_set_v48": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_candidate_split_global_trust_veto_v49": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
                "word_veto_candidate_set_attention_confidence_v33": (
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
                ),
            }
            if not monotone_candidate_trace_raw_veto_disabled and (
                raw_veto_weight > 0.0 or raw_veto_revision in raw_veto_revisions
            ):
                if (
                    raw_veto_revision not in raw_veto_revisions
                    or phrase_aggregation != raw_veto_revisions[raw_veto_revision]
                    or not math.isfinite(raw_veto_weight)
                    or raw_veto_weight <= 0.0
                ):
                    raise RuntimeError(
                        "raw-gated word veto requires a matching revision, positive "
                        "gate weight, and phrase aggregation"
                    )
                for field in (
                    "stage_b_dense_duty_raw_veto_positive_margin",
                    "stage_b_dense_duty_raw_veto_tn_margin",
                ):
                    value = float(getattr(args, field, 0.0))
                    if not math.isfinite(value) or value <= 0.0:
                        raise RuntimeError(
                            f"raw-gated word veto requires positive {field}"
                        )
                if raw_veto_revision in {
                    "word_veto_coverage_absolute_cap_v4",
                    "word_veto_gated_pool_absolute_cap_v5",
                    "word_veto_gated_pool_calibrated_v6",
                    "word_veto_gated_pool_carrier_balanced_v7",
                    "word_veto_gated_pool_carrier_quarter_v8",
                    "word_veto_gated_pool_carrier_pair_v9",
                    "word_veto_gated_pool_dual_carrier_pair_v10",
                    "word_veto_gated_pool_rank_evidence_v11",
                    "word_veto_gated_pool_rank_affine_v12",
                    "word_veto_gated_pool_gate_margin_v13",
                    "word_veto_gated_pool_carrier_slope_v14",
                    "word_veto_gated_pool_carrier_affine_v15",
                    "word_veto_gated_pool_tail_ste_v16",
                    "word_veto_gated_pool_tail_carrier_v17",
                    "word_veto_gated_pool_tail_paired_v18",
                    "word_veto_gated_pool_tail_paired_rank_channel_v19",
                    "word_veto_gated_pool_tail_paired_signed_rank_pool_v20",
                    "word_veto_continuous_conditional_residual_v21",
                    "word_veto_continuous_monotone_depth_v22",
                    "word_veto_token_conditioned_monotone_depth_v23",
                    "word_veto_complementary_trust_veto_v24",
                    "word_veto_ungated_monotone_tail_veto_v25",
                    "word_veto_floor_gated_monotone_tail_veto_v26",
                    "word_veto_independent_absolute_confidence_v27",
                    "word_veto_cross_attention_absolute_confidence_v28",
                    "word_veto_candidate_absolute_confidence_v29",
                    "word_veto_candidate_patch_invariant_confidence_v30",
                    "word_veto_candidate_normalized_confidence_v31",
                    "word_veto_candidate_asymmetric_confidence_v32",
                    "word_veto_candidate_set_attention_confidence_v33",
                    "word_veto_candidate_asymmetric_deployed_routing_v43",
                    "word_veto_candidate_split_tail_aligned_v45",
                    "word_veto_candidate_split_positive_tail_v46",
                    "word_veto_candidate_split_boundary_routing_v47",
                    "word_veto_candidate_split_fpr_active_set_v48",
                    "word_veto_candidate_split_global_trust_veto_v49",
                    "word_veto_candidate_split_strong_boundary_routing_v50",
                    "word_veto_candidate_split_independent_deployed_router_v51",
                    "word_veto_candidate_sample_calibrator_split_v52",
                    "word_veto_rank_full_expression_global_absolute_v53",
                    "word_veto_rank_full_expression_global_absolute_exact_residual_v54",
                    "word_veto_rank_full_expression_global_independent_absolute_v55",
                    "word_veto_rank_full_expression_deployment_owned_global_v56",
                    "word_veto_rank_full_expression_deployed_global_balanced_absolute_v57",
                    "word_veto_rank_full_expression_deployment_owned_global_stable_fpr95_active_set_v58",
                    "word_veto_rank_full_expression_deployment_owned_query_global_v59",
                    "word_veto_rank_full_expression_deployment_owned_query_veto_v60",
                }:
                    carrier_balanced_revisions = {
                        "word_veto_gated_pool_carrier_balanced_v7": 0.5,
                        "word_veto_gated_pool_carrier_quarter_v8": 0.25,
                        "word_veto_gated_pool_carrier_pair_v9": 0.25,
                        "word_veto_gated_pool_dual_carrier_pair_v10": 0.25,
                        "word_veto_gated_pool_rank_evidence_v11": 0.25,
                        "word_veto_gated_pool_rank_affine_v12": 0.25,
                        "word_veto_gated_pool_gate_margin_v13": 0.25,
                        "word_veto_gated_pool_carrier_slope_v14": 0.25,
                        "word_veto_gated_pool_carrier_affine_v15": 0.25,
                        "word_veto_gated_pool_tail_ste_v16": 0.25,
                        "word_veto_gated_pool_tail_carrier_v17": 0.25,
                        "word_veto_gated_pool_tail_paired_v18": 0.25,
                        "word_veto_gated_pool_tail_paired_rank_channel_v19": 0.25,
                        "word_veto_gated_pool_tail_paired_signed_rank_pool_v20": 0.25,
                        "word_veto_continuous_conditional_residual_v21": 0.25,
                        "word_veto_continuous_monotone_depth_v22": 0.25,
                        "word_veto_token_conditioned_monotone_depth_v23": 0.25,
                        "word_veto_complementary_trust_veto_v24": 0.25,
                        "word_veto_ungated_monotone_tail_veto_v25": 0.25,
                        "word_veto_floor_gated_monotone_tail_veto_v26": 0.25,
                        "word_veto_independent_absolute_confidence_v27": 0.25,
                        "word_veto_cross_attention_absolute_confidence_v28": 0.25,
                        "word_veto_candidate_absolute_confidence_v29": 0.25,
                        "word_veto_candidate_patch_invariant_confidence_v30": 0.25,
                        "word_veto_candidate_normalized_confidence_v31": 0.25,
                        "word_veto_candidate_asymmetric_confidence_v32": 0.25,
                        "word_veto_candidate_set_attention_confidence_v33": 0.25,
                        "word_veto_candidate_asymmetric_deployed_routing_v43": 0.25,
                        "word_veto_candidate_split_tail_aligned_v45": 0.25,
                        "word_veto_candidate_split_positive_tail_v46": 0.25,
                        "word_veto_candidate_split_boundary_routing_v47": 0.25,
                        "word_veto_candidate_split_fpr_active_set_v48": 0.25,
                        "word_veto_candidate_split_global_trust_veto_v49": 0.25,
                        "word_veto_candidate_split_strong_boundary_routing_v50": 0.25,
                        "word_veto_candidate_split_independent_deployed_router_v51": 0.25,
                        "word_veto_candidate_sample_calibrator_split_v52": 0.25,
                        "word_veto_rank_full_expression_global_absolute_v53": 0.25,
                        "word_veto_rank_full_expression_global_absolute_exact_residual_v54": 0.25,
                        "word_veto_rank_full_expression_global_independent_absolute_v55": 0.25,
                        "word_veto_rank_full_expression_deployment_owned_global_v56": 0.25,
                        "word_veto_rank_full_expression_deployed_global_balanced_absolute_v57": 0.25,
                        "word_veto_rank_full_expression_deployment_owned_global_stable_fpr95_active_set_v58": 0.25,
                        "word_veto_rank_full_expression_deployment_owned_query_global_v59": 0.25,
                        "word_veto_rank_full_expression_deployment_owned_query_veto_v60": 0.25,
                    }
                    if raw_veto_revision == (
                        "word_veto_gated_pool_dual_carrier_pair_v10"
                    ):
                        expected_raw_veto_scope = (
                            "tn_all_admitted_dual_carrier_balanced_paired_v5"
                        )
                    elif raw_veto_revision in {
                        "word_veto_gated_pool_tail_paired_v18",
                        "word_veto_gated_pool_tail_paired_rank_channel_v19",
                        "word_veto_gated_pool_tail_paired_signed_rank_pool_v20",
                        "word_veto_continuous_conditional_residual_v21",
                        "word_veto_continuous_monotone_depth_v22",
                        "word_veto_token_conditioned_monotone_depth_v23",
                        "word_veto_complementary_trust_veto_v24",
                        "word_veto_ungated_monotone_tail_veto_v25",
                        "word_veto_floor_gated_monotone_tail_veto_v26",
                        "word_veto_independent_absolute_confidence_v27",
                        "word_veto_cross_attention_absolute_confidence_v28",
                        "word_veto_candidate_absolute_confidence_v29",
                        "word_veto_candidate_patch_invariant_confidence_v30",
                        "word_veto_candidate_normalized_confidence_v31",
                        "word_veto_candidate_asymmetric_confidence_v32",
                        "word_veto_candidate_set_attention_confidence_v33",
                        "word_veto_candidate_asymmetric_deployed_routing_v43",
                        "word_veto_candidate_split_tail_aligned_v45",
                        "word_veto_candidate_split_positive_tail_v46",
                        "word_veto_candidate_split_boundary_routing_v47",
                        "word_veto_candidate_split_fpr_active_set_v48",
                        "word_veto_candidate_split_global_trust_veto_v49",
                        "word_veto_candidate_split_strong_boundary_routing_v50",
                        "word_veto_candidate_split_independent_deployed_router_v51",
                        "word_veto_candidate_sample_calibrator_split_v52",
                        "word_veto_rank_full_expression_global_absolute_v53",
                        "word_veto_rank_full_expression_global_absolute_exact_residual_v54",
                        "word_veto_rank_full_expression_global_independent_absolute_v55",
                        "word_veto_rank_full_expression_deployment_owned_global_v56",
                        "word_veto_rank_full_expression_deployed_global_balanced_absolute_v57",
                        "word_veto_rank_full_expression_deployment_owned_global_stable_fpr95_active_set_v58",
                        "word_veto_rank_full_expression_deployment_owned_query_global_v59",
                        "word_veto_rank_full_expression_deployment_owned_query_veto_v60",
                    }:
                        expected_raw_veto_scope = (
                            "tn_all_admitted_tail_weighted_carrier_tail_paired_v7"
                        )
                    elif raw_veto_revision == (
                        "word_veto_gated_pool_tail_carrier_v17"
                    ):
                        expected_raw_veto_scope = (
                            "tn_all_admitted_tail_weighted_carrier_"
                            "positive_carrier_paired_v6"
                        )
                    elif raw_veto_revision in {
                        "word_veto_gated_pool_carrier_pair_v9",
                        "word_veto_gated_pool_rank_evidence_v11",
                        "word_veto_gated_pool_rank_affine_v12",
                        "word_veto_gated_pool_gate_margin_v13",
                        "word_veto_gated_pool_carrier_slope_v14",
                        "word_veto_gated_pool_carrier_affine_v15",
                        "word_veto_gated_pool_tail_ste_v16",
                    }:
                        expected_raw_veto_scope = (
                            "tn_all_admitted_carrier_balanced_"
                            "positive_carrier_paired_v4"
                        )
                    elif raw_veto_revision in carrier_balanced_revisions:
                        expected_raw_veto_scope = (
                            "tn_all_admitted_carrier_balanced_positive_carrier_v3"
                        )
                    else:
                        expected_raw_veto_scope = (
                            "tn_all_admitted_positive_carrier_v2"
                        )
                    if str(
                        getattr(
                            args,
                            "stage_b_dense_duty_raw_veto_query_scope",
                            "",
                        )
                    ).strip() != expected_raw_veto_scope:
                        raise RuntimeError(
                            "absolute-cap veto raw supervision scope does not match "
                            "its confidence revision"
                        )
                    if raw_veto_revision in carrier_balanced_revisions:
                        carrier_balance = float(
                            getattr(
                                args,
                                "stage_b_dense_duty_raw_veto_tn_carrier_balance",
                                -1.0,
                            )
                        )
                        expected_carrier_balance = carrier_balanced_revisions[
                            raw_veto_revision
                        ]
                        if carrier_balance != expected_carrier_balance:
                            raise RuntimeError(
                                "carrier-balanced veto requires the exact "
                                f"{expected_carrier_balance} all-query/carrier "
                                "balance for its revision"
                            )
                        if str(
                            getattr(
                                args,
                                "stage_b_dense_duty_confidence_carrier_selector_contract",
                                "",
                            )
                        ).strip() != (
                            "final_layer_reference_argmax_exact_eligible_v1"
                        ):
                            raise RuntimeError(
                                "carrier-balanced veto requires the exact inference "
                                "carrier selector contract"
                            )
                        pair_weight = float(
                            getattr(
                                args,
                                "stage_b_dense_duty_raw_veto_carrier_pair_weight",
                                0.0,
                            )
                            or 0.0
                        )
                        pair_margin = float(
                            getattr(
                                args,
                                "stage_b_dense_duty_raw_veto_carrier_pair_margin",
                                0.0,
                            )
                            or 0.0
                        )
                        paired_revisions = {
                            "word_veto_gated_pool_carrier_pair_v9",
                            "word_veto_gated_pool_dual_carrier_pair_v10",
                            "word_veto_gated_pool_rank_evidence_v11",
                            "word_veto_gated_pool_rank_affine_v12",
                            "word_veto_gated_pool_gate_margin_v13",
                            "word_veto_gated_pool_carrier_slope_v14",
                            "word_veto_gated_pool_carrier_affine_v15",
                            "word_veto_gated_pool_tail_ste_v16",
                            "word_veto_gated_pool_tail_carrier_v17",
                            "word_veto_gated_pool_tail_paired_v18",
                            "word_veto_gated_pool_tail_paired_rank_channel_v19",
                            "word_veto_gated_pool_tail_paired_signed_rank_pool_v20",
                            "word_veto_continuous_conditional_residual_v21",
                            "word_veto_continuous_monotone_depth_v22",
                            "word_veto_token_conditioned_monotone_depth_v23",
                            "word_veto_complementary_trust_veto_v24",
                            "word_veto_ungated_monotone_tail_veto_v25",
                            "word_veto_floor_gated_monotone_tail_veto_v26",
                            "word_veto_independent_absolute_confidence_v27",
                            "word_veto_cross_attention_absolute_confidence_v28",
                            "word_veto_candidate_absolute_confidence_v29",
                            "word_veto_candidate_patch_invariant_confidence_v30",
                            "word_veto_candidate_normalized_confidence_v31",
                            "word_veto_candidate_asymmetric_confidence_v32",
                            "word_veto_candidate_set_attention_confidence_v33",
                            "word_veto_candidate_asymmetric_deployed_routing_v43",
                            "word_veto_candidate_split_tail_aligned_v45",
                            "word_veto_candidate_split_positive_tail_v46",
                            "word_veto_candidate_split_boundary_routing_v47",
                            "word_veto_candidate_split_fpr_active_set_v48",
                            "word_veto_candidate_split_global_trust_veto_v49",
                            "word_veto_candidate_split_strong_boundary_routing_v50",
                            "word_veto_candidate_split_independent_deployed_router_v51",
                            "word_veto_candidate_sample_calibrator_split_v52",
                            "word_veto_rank_full_expression_global_absolute_v53",
                            "word_veto_rank_full_expression_global_absolute_exact_residual_v54",
                            "word_veto_rank_full_expression_global_independent_absolute_v55",
                            "word_veto_rank_full_expression_deployment_owned_global_v56",
                            "word_veto_rank_full_expression_deployed_global_balanced_absolute_v57",
                            "word_veto_rank_full_expression_deployment_owned_global_stable_fpr95_active_set_v58",
                            "word_veto_rank_full_expression_deployment_owned_query_global_v59",
                            "word_veto_rank_full_expression_deployment_owned_query_veto_v60",
                        }
                        if raw_veto_revision in paired_revisions:
                            if pair_weight != 0.25 or pair_margin != 0.25:
                                raise RuntimeError(
                                    "paired-carrier veto requires exact pair "
                                    "weight=0.25 and margin=0.25"
                                )
                        elif pair_weight != 0.0 or pair_margin != 0.0:
                            raise RuntimeError(
                                "pre-v9 carrier revisions forbid paired-carrier "
                                "supervision"
                            )
                        if raw_veto_revision in {
                            "word_veto_gated_pool_tail_carrier_v17",
                            "word_veto_gated_pool_tail_paired_v18",
                            "word_veto_gated_pool_tail_paired_rank_channel_v19",
                            "word_veto_gated_pool_tail_paired_signed_rank_pool_v20",
                            "word_veto_continuous_conditional_residual_v21",
                            "word_veto_continuous_monotone_depth_v22",
                            "word_veto_token_conditioned_monotone_depth_v23",
                            "word_veto_complementary_trust_veto_v24",
                            "word_veto_ungated_monotone_tail_veto_v25",
                            "word_veto_floor_gated_monotone_tail_veto_v26",
                            "word_veto_independent_absolute_confidence_v27",
                            "word_veto_cross_attention_absolute_confidence_v28",
                            "word_veto_candidate_absolute_confidence_v29",
                            "word_veto_candidate_patch_invariant_confidence_v30",
                            "word_veto_candidate_normalized_confidence_v31",
                            "word_veto_candidate_asymmetric_confidence_v32",
                            "word_veto_candidate_set_attention_confidence_v33",
                            "word_veto_candidate_asymmetric_deployed_routing_v43",
                            "word_veto_candidate_split_tail_aligned_v45",
                            "word_veto_candidate_split_positive_tail_v46",
                            "word_veto_candidate_split_boundary_routing_v47",
                            "word_veto_candidate_split_fpr_active_set_v48",
                            "word_veto_candidate_split_global_trust_veto_v49",
                            "word_veto_candidate_split_strong_boundary_routing_v50",
                            "word_veto_candidate_split_independent_deployed_router_v51",
                            "word_veto_candidate_sample_calibrator_split_v52",
                            "word_veto_rank_full_expression_global_absolute_v53",
                            "word_veto_rank_full_expression_global_absolute_exact_residual_v54",
                            "word_veto_rank_full_expression_global_independent_absolute_v55",
                            "word_veto_rank_full_expression_deployment_owned_global_v56",
                            "word_veto_rank_full_expression_deployed_global_balanced_absolute_v57",
                            "word_veto_rank_full_expression_deployment_owned_global_stable_fpr95_active_set_v58",
                            "word_veto_rank_full_expression_deployment_owned_query_global_v59",
                            "word_veto_rank_full_expression_deployment_owned_query_veto_v60",
                        }:
                            tail_quantile = float(
                                getattr(
                                    args,
                                    "stage_b_dense_duty_raw_veto_tail_quantile",
                                    -1.0,
                                )
                            )
                            tail_temperature = float(
                                getattr(
                                    args,
                                    "stage_b_dense_duty_raw_veto_tail_temperature",
                                    -1.0,
                                )
                            )
                            tail_min_count = int(
                                getattr(
                                    args,
                                    "stage_b_dense_duty_raw_veto_tail_min_count",
                                    -1,
                                )
                            )
                            if (
                                tail_quantile != 0.95
                                or tail_temperature != 0.1
                                or tail_min_count != 256
                            ):
                                raise RuntimeError(
                                    "tail-focused confidence requires q=0.95, tau=0.1, "
                                    "and min_count=256"
                                )
                        positive_carrier_balance = float(
                            getattr(
                                args,
                                "stage_b_dense_duty_raw_veto_positive_carrier_balance",
                                0.0,
                            )
                            or 0.0
                        )
                        if raw_veto_revision == (
                            "word_veto_gated_pool_dual_carrier_pair_v10"
                        ):
                            if positive_carrier_balance != 0.25:
                                raise RuntimeError(
                                    "dual-carrier veto requires exact positive "
                                    "carrier balance=0.25"
                                )
                        elif positive_carrier_balance != 0.0:
                            raise RuntimeError(
                                "pre-v10 carrier revisions forbid positive carrier "
                                "balancing"
                            )
                    pair_gradient_contract = str(
                        getattr(
                            args,
                            "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
                            "bidirectional_v1",
                        )
                    ).strip().lower()
                    if pair_gradient_contract not in {
                        "bidirectional_v1",
                        "tn_only_positive_detached_v2",
                    }:
                        raise RuntimeError(
                            "raw-veto carrier-pair gradient contract is invalid"
                        )
                    if pair_gradient_contract == "tn_only_positive_detached_v2":
                        if (
                            raw_veto_revision
                            != "word_veto_candidate_asymmetric_confidence_v32"
                            or str(
                                getattr(
                                    args,
                                    "stage_b_dense_duty_raw_veto_query_scope",
                                    "",
                                )
                            ).strip()
                            != "tn_all_admitted_tail_weighted_carrier_tail_paired_v7"
                            or float(
                                getattr(
                                    args,
                                    "stage_b_dense_duty_raw_veto_carrier_pair_weight",
                                    0.0,
                                )
                            )
                            != 0.25
                            or str(
                                getattr(
                                    args,
                                    "stage_b_v21_token_edit_query_scope",
                                    "target_iou_v1",
                                )
                            ).strip()
                            != "target_iou_v1"
                        ):
                            raise RuntimeError(
                                "TN-only carrier-pair gradients require the exact "
                                "v39 candidate-asymmetric tail-paired contract"
                            )
                    gate_offset = float(
                        getattr(
                            args,
                            "stage_b_dense_duty_confidence_veto_gate_offset",
                            -1.0,
                        )
                    )
                    gate_scale = float(
                        getattr(
                            args,
                            "stage_b_dense_duty_confidence_veto_gate_scale",
                            -1.0,
                        )
                    )
                    if (
                        not math.isfinite(gate_offset)
                        or gate_offset < 0.0
                        or not math.isfinite(gate_scale)
                        or gate_scale <= 0.0
                        or gate_offset + gate_scale
                        > float(
                            getattr(
                                args,
                                "stage_b_dense_duty_raw_veto_tn_margin",
                                0.0,
                            )
                        )
                        + 1e-8
                    ):
                        raise RuntimeError(
                            "absolute-cap gate dead-zone/ramp must reach one by the "
                            "TN raw margin"
                        )
                    coverage_offset = float(
                        getattr(
                            args,
                            "stage_b_dense_duty_confidence_veto_coverage_offset",
                            -1.0,
                        )
                    )
                    coverage_ramp = float(
                        getattr(
                            args,
                            "stage_b_dense_duty_confidence_veto_coverage_ramp",
                            -1.0,
                        )
                    )
                    cap_temperature = float(
                        getattr(
                            args,
                            "stage_b_dense_duty_confidence_veto_cap_temperature",
                            -1.0,
                        )
                    )
                    cap_ceiling = float(
                        getattr(
                            args,
                            "stage_b_dense_duty_confidence_veto_cap_initial_ceiling",
                            0.0,
                        )
                    )
                    if (
                        not math.isfinite(coverage_offset)
                        or not 0.0 <= coverage_offset < 1.0
                        or not math.isfinite(coverage_ramp)
                        or coverage_ramp <= 0.0
                        or coverage_offset + coverage_ramp > 1.0
                        or not math.isfinite(cap_temperature)
                        or cap_temperature <= 0.0
                        or not math.isfinite(cap_ceiling)
                        or cap_ceiling >= 0.0
                    ):
                        raise RuntimeError(
                            "absolute-cap coverage and ceiling contract is invalid"
                        )
            global_negative_weight = float(
                getattr(args, "stage_b_v11_global_tn_negative_weight", 0.0)
            )
            global_tail_weight = float(
                getattr(args, "stage_b_v11_global_tn_tail_weight", 0.0)
            )
            if global_negative_weight <= 0.0 or global_tail_weight != 0.0:
                raise RuntimeError(
                    "word-veto confidence requires one nonzero global-TN scalar "
                    "loss and disables the duplicate candidate-tail term"
                )
            if getattr(args, "stage_b_v15_tail_queue_global_scores", None) is not True:
                raise RuntimeError(
                    "word-veto confidence requires expression-global FPR queue scores"
                )
            if str(
                getattr(args, "stage_b_v15_tail_queue_objective", "")
            ).strip().lower() != "fpr95":
                raise RuntimeError(
                    "word-veto confidence requires the exact FPR95 queue objective"
                )
            execution_scope = str(
                getattr(args, "stage_b_dense_duty_execution_scope", "")
            ).strip().lower()
            admission_contract = str(
                getattr(
                    args,
                    "stage_b_dense_duty_confidence_probe_admission_contract",
                    "",
                )
            ).strip()
            admission_report = str(
                getattr(
                    args,
                    "stage_b_dense_duty_confidence_probe_admission_report",
                    "",
                )
                or ""
            ).strip()
            admission_audit = getattr(
                args,
                "stage_b_dense_duty_confidence_probe_admission_audit",
                None,
            )
            formal_admission_contract = {
                "word_veto_net_trust_v1": "u300_word_veto_strict1607_v1",
                "word_veto_raw_gate_margin_v3": (
                    "u300_word_veto_gate_strict1607_v3"
                ),
                "word_veto_coverage_absolute_cap_v4": (
                    "u300_word_veto_absolute_cap_strict1607_v4"
                ),
                "word_veto_gated_pool_absolute_cap_v5": (
                    "u300_word_veto_gated_pool_absolute_cap_strict1607_v5"
                ),
                "word_veto_gated_pool_calibrated_v6": (
                    "u300_word_veto_gated_pool_calibrated_strict1607_v6"
                ),
                "word_veto_gated_pool_carrier_balanced_v7": (
                    "u300_word_veto_gated_pool_carrier_balanced_strict1607_v7"
                ),
                "word_veto_gated_pool_carrier_quarter_v8": (
                    "u300_word_veto_gated_pool_carrier_quarter_strict1607_v8"
                ),
                "word_veto_gated_pool_carrier_pair_v9": (
                    "u300_word_veto_gated_pool_carrier_pair_strict1607_v9"
                ),
                "word_veto_gated_pool_dual_carrier_pair_v10": (
                    "u300_word_veto_gated_pool_dual_carrier_pair_strict1607_v10"
                ),
                "word_veto_gated_pool_rank_evidence_v11": (
                    "u300_word_veto_gated_pool_rank_evidence_strict1607_v11"
                ),
                "word_veto_gated_pool_rank_affine_v12": (
                    "u300_word_veto_gated_pool_rank_affine_strict1607_v12"
                ),
                "word_veto_gated_pool_gate_margin_v13": (
                    "u300_word_veto_gated_pool_gate_margin_strict1607_v13"
                ),
                "word_veto_gated_pool_carrier_slope_v14": (
                    "u300_word_veto_gated_pool_carrier_slope_strict1607_v14"
                ),
                "word_veto_gated_pool_carrier_affine_v15": (
                    "u300_word_veto_gated_pool_carrier_affine_strict1607_v15"
                ),
                "word_veto_gated_pool_tail_ste_v16": (
                    "u300_word_veto_gated_pool_tail_ste_strict1607_v16"
                ),
                "word_veto_gated_pool_tail_carrier_v17": (
                    "u300_word_veto_gated_pool_tail_carrier_strict1607_v17"
                ),
                "word_veto_gated_pool_tail_paired_v18": (
                    "u300_word_veto_gated_pool_tail_paired_strict1607_v18"
                ),
                "word_veto_gated_pool_tail_paired_rank_channel_v19": (
                    "u300_word_veto_gated_pool_tail_paired_rank_channel_strict1607_v19"
                ),
                "word_veto_gated_pool_tail_paired_signed_rank_pool_v20": (
                    "u300_word_veto_gated_pool_tail_paired_signed_rank_pool_strict1607_v20"
                ),
                "word_veto_continuous_conditional_residual_v21": (
                    "u300_word_veto_continuous_conditional_residual_strict1607_v21"
                ),
                "word_veto_continuous_monotone_depth_v22": (
                    "u300_word_veto_continuous_monotone_depth_strict1607_v22"
                ),
                "word_veto_token_conditioned_monotone_depth_v23": (
                    "u300_word_veto_token_conditioned_monotone_depth_strict1607_v23"
                ),
                "word_veto_complementary_trust_veto_v24": (
                    "u300_word_veto_complementary_trust_veto_strict1607_v24"
                ),
                "word_veto_ungated_monotone_tail_veto_v25": (
                    "u300_word_veto_ungated_monotone_tail_veto_strict1607_v25"
                ),
                "word_veto_floor_gated_monotone_tail_veto_v26": (
                    "u300_word_veto_floor_gated_monotone_tail_veto_strict1607_v26"
                ),
                "word_veto_independent_absolute_confidence_v27": (
                    "u300_word_veto_independent_absolute_confidence_strict1607_v27"
                ),
                "word_veto_cross_attention_absolute_confidence_v28": (
                    "u300_word_veto_cross_attention_absolute_confidence_strict1607_v28"
                ),
                "word_veto_candidate_absolute_confidence_v29": (
                    "u300_word_veto_candidate_absolute_confidence_strict1607_v29"
                ),
                "word_veto_candidate_patch_invariant_confidence_v30": (
                    "u400_word_veto_candidate_patch_invariant_confidence_strict1607_v30"
                ),
                "word_veto_candidate_normalized_confidence_v31": (
                    "u400_word_veto_candidate_normalized_confidence_strict1607_v31"
                ),
                "word_veto_candidate_asymmetric_deployed_routing_v43": (
                    "u400_word_veto_candidate_deployed_routing_confidence_"
                    "strict1607_v43"
                ),
                "word_veto_candidate_split_tail_aligned_v45": (
                    "u400_word_veto_candidate_split_tail_aligned_confidence_"
                    "strict1607_v45"
                ),
                "word_veto_candidate_split_positive_tail_v46": (
                    "u400_word_veto_candidate_split_positive_tail_confidence_"
                    "strict1607_v46"
                ),
                "word_veto_candidate_split_boundary_routing_v47": (
                    "u400_word_veto_candidate_split_boundary_routing_confidence_"
                    "strict1607_v47"
                ),
                "word_veto_candidate_split_strong_boundary_routing_v50": (
                    "u400_word_veto_candidate_split_strong_boundary_routing_"
                    "confidence_strict1607_v50"
                ),
                "word_veto_candidate_split_independent_deployed_router_v51": (
                    "u400_word_veto_candidate_split_independent_deployed_router_"
                    "confidence_strict1607_v51"
                ),
                "word_veto_candidate_sample_calibrator_split_v52": (
                    "u400_word_veto_candidate_sample_calibrator_"
                    "confidence_strict1607_v52"
                ),
                "word_veto_rank_full_expression_global_absolute_v53": (
                    "u400_word_veto_rank_full_expression_global_absolute_"
                    "confidence_strict1607_v53"
                ),
                "word_veto_rank_full_expression_global_absolute_exact_residual_v54": (
                    "u400_word_veto_rank_full_expression_global_absolute_"
                    "exact_residual_confidence_strict1607_v54"
                ),
                "word_veto_rank_full_expression_global_independent_absolute_v55": (
                    "u400_word_veto_rank_full_expression_global_independent_absolute_"
                    "confidence_strict1607_v55"
                ),
                "word_veto_rank_full_expression_deployment_owned_global_v56": (
                    "u400_word_veto_rank_full_expression_deployment_owned_global_"
                    "confidence_strict1607_v56"
                ),
                "word_veto_rank_full_expression_deployed_global_balanced_absolute_v57": (
                    "u400_word_veto_rank_full_expression_deployed_global_"
                    "balanced_absolute_confidence_strict1607_v57"
                ),
                "word_veto_rank_full_expression_deployment_owned_global_stable_fpr95_active_set_v58": (
                    "u400_word_veto_rank_full_expression_deployment_owned_global_"
                    "stable_fpr95_active_set_confidence_strict1607_v58"
                ),
                "word_veto_rank_full_expression_deployment_owned_query_global_v59": (
                    "u400_word_veto_rank_full_expression_deployment_owned_query_"
                    "global_confidence_strict1607_v59"
                ),
                "word_veto_rank_full_expression_deployment_owned_query_veto_v60": (
                    "u400_word_veto_rank_full_expression_deployment_owned_query_"
                    "veto_confidence_strict1607_v60"
                ),
                "word_veto_candidate_split_fpr_active_set_v48": (
                    "u400_word_veto_candidate_split_fpr_active_set_confidence_"
                    "strict1607_v48"
                ),
                "word_veto_candidate_split_global_trust_veto_v49": (
                    "u400_word_veto_candidate_split_global_trust_veto_confidence_"
                    "strict1607_v49"
                ),
                # V32's carrier-pair resolver is intentionally lazy. Evaluating
                # it while selecting V60 used to reject candidate-complete scopes.
                "word_veto_candidate_asymmetric_confidence_v32": None,
                "word_veto_candidate_set_attention_confidence_v33": (
                    "u400_word_veto_candidate_set_attention_confidence_strict1607_v33"
                ),
            }[raw_veto_revision]
            if formal_admission_contract is None:
                formal_admission_contract = (
                    _stage_b_candidate_asymmetric_formal_admission_contract(args)
                )
            if execution_scope == "probe":
                if (
                    admission_contract != "disabled_for_probe_v1"
                    or admission_report
                    or admission_audit is not None
                ):
                    raise RuntimeError(
                        "word-veto probe crossed the formal promotion boundary"
                    )
            elif (
                execution_scope != "formal"
                or admission_contract != formal_admission_contract
                or not admission_report
                or not isinstance(admission_audit, Mapping)
                or admission_audit.get("status") != "verified"
                or admission_audit.get("decision")
                != "admit_to_formal_training"
                or admission_audit.get("formal_training_admitted") is not True
            ):
                raise RuntimeError(
                    "formal word-veto confidence lacks a verified U300 "
                    "strict1607 promotion audit"
                )
    if getattr(args, "finetune_ignore", None):
        raise RuntimeError("dense-duty Stage B forbids finetune_ignore")

    def bound_path(path_field: str, sha_field: str, label: str) -> tuple[Path, str]:
        path_value = str(getattr(args, path_field, "") or "").strip()
        expected_sha = str(getattr(args, sha_field, "") or "").strip()
        if not path_value or len(expected_sha) != 64:
            raise RuntimeError(f"dense-duty Stage B requires exact {label} path/SHA")
        path = Path(path_value).expanduser().resolve(strict=True)
        observed_sha = _sha256_file(path)
        if observed_sha != expected_sha:
            raise RuntimeError(
                f"dense-duty {label} SHA drifted: expected={expected_sha}, "
                f"observed={observed_sha}"
            )
        return path, observed_sha

    base_path, base_sha = bound_path(
        "stage_b_dense_duty_base_checkpoint_path",
        "stage_b_dense_duty_base_checkpoint_sha256",
        "Stage-A base",
    )
    text_path, text_sha = bound_path(
        "stage_b_dense_duty_text_checkpoint_path",
        "stage_b_dense_duty_text_checkpoint_sha256",
        "OGC text",
    )
    dataset_path, dataset_sha = bound_path(
        "stage_b_dense_duty_dataset_config_path",
        "stage_b_dense_duty_dataset_config_sha256",
        f"{phase} dataset",
    )
    tn_path, tn_sha = bound_path(
        "stage_b_dense_duty_tn_manifest_path",
        "stage_b_dense_duty_tn_manifest_sha256",
        "traceable TN manifest",
    )
    trace_audit_path, trace_audit_sha = bound_path(
        "stage_b_dense_duty_trace_audit_path",
        "stage_b_dense_duty_trace_audit_sha256",
        "direct-trace audit receipt",
    )
    try:
        with trace_audit_path.open("r", encoding="ascii") as handle:
            trace_audit = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("dense-duty direct-trace receipt is unreadable") from error
    if not isinstance(trace_audit, dict) or trace_audit.get("schema") != (
        "pivot.stageb.dense_duty_direct_trace_audit/v1"
    ):
        raise RuntimeError("dense-duty direct-trace receipt schema is invalid")
    trace_source = trace_audit.get("source")
    if not isinstance(trace_source, dict) or (
        trace_source.get("sha256") != tn_sha
        or trace_source.get("rows")
        != int(getattr(args, "stage_b_dense_duty_trace_total_rows", -1))
        or trace_source.get("scope") != "image_global_topk_verified"
        or trace_source.get("global_tn_verified") is not True
    ):
        raise RuntimeError("dense-duty direct-trace receipt source contract drifted")
    trace_algorithm = trace_audit.get("algorithm")
    if not isinstance(trace_algorithm, dict) or (
        trace_algorithm.get("allow_incidental_trace_edits") is not False
        or trace_algorithm.get("token_role_source") != "exact_direct_trace_v1"
        or trace_algorithm.get("canonical_tokens_excluded_from_roles") is not True
        or trace_algorithm.get("legacy_full_pair_diff_used_for_token_roles") is not False
    ):
        raise RuntimeError("dense-duty direct-trace algorithm contract drifted")
    expected_trace_counts = {
        "lexical_exact_valid_rows": int(
            getattr(args, "stage_b_dense_duty_trace_lexical_valid_rows", -1)
        ),
        "direct_token_valid_rows": int(
            getattr(args, "stage_b_dense_duty_trace_direct_token_valid_rows", -1)
        ),
        "direct_token_invalid_rows": int(
            getattr(args, "stage_b_dense_duty_trace_direct_token_invalid_rows", -1)
        ),
        "no_unique_exact_reconstruction": int(
            getattr(
                args,
                "stage_b_dense_duty_trace_no_unique_reconstruction_rows",
                -1,
            )
        ),
        "no_target_side_changed_token": int(
            getattr(args, "stage_b_dense_duty_trace_deletion_only_rows", -1)
        ),
        "canonical_score_surface_rejections": int(
            getattr(
                args,
                "stage_b_dense_duty_trace_canonical_surface_rejections",
                -1,
            )
        ),
    }
    trace_counts = trace_audit.get("counts")
    if (
        not isinstance(trace_counts, dict)
        or any(value < 0 for value in expected_trace_counts.values())
        or any(
            trace_counts.get(key) != value
            for key, value in expected_trace_counts.items()
        )
    ):
        raise RuntimeError("dense-duty direct-trace receipt counts drifted")
    if (
        expected_trace_counts["direct_token_valid_rows"]
        + expected_trace_counts["direct_token_invalid_rows"]
        != int(getattr(args, "stage_b_dense_duty_trace_total_rows", -1))
    ):
        raise RuntimeError("dense-duty direct-trace valid/invalid counts do not close")
    candidate_trace_contract = str(
        getattr(
            args,
            "stage_b_dense_duty_confidence_candidate_trace_contract",
            "off_v1",
        )
    ).strip().lower()
    if candidate_trace_contract != "off_v1":
        candidate_provenance = trace_audit.get("candidate_trace_provenance")
        scope_rows = (
            candidate_provenance.get("scope_rows")
            if isinstance(candidate_provenance, dict)
            else None
        )
        required_scopes = (
            "expression_only",
            "global_word_absent",
            "candidate_verified",
        )
        total_rows = int(
            getattr(args, "stage_b_dense_duty_trace_total_rows", -1)
        )
        if (
            not isinstance(candidate_provenance, dict)
            or candidate_provenance.get("contract")
            != "fail_closed_candidate_complete_trace_v1"
            or not isinstance(scope_rows, dict)
            or any(type(scope_rows.get(name)) is not int for name in required_scopes)
            or any(scope_rows[name] < 0 for name in required_scopes)
            or sum(scope_rows[name] for name in required_scopes) != total_rows
            or candidate_provenance.get(
                "expression_level_depth_supervision_rows"
            )
            != total_rows
            or candidate_provenance.get("global_word_absent_verified_rows")
            != scope_rows["global_word_absent"]
            or candidate_provenance.get("candidate_verified_rows")
            != scope_rows["candidate_verified"]
            or candidate_provenance.get("token_broadcast_capable_rows")
            != scope_rows["global_word_absent"]
            + scope_rows["candidate_verified"]
        ):
            raise RuntimeError(
                "candidate-complete trace provenance receipt is invalid"
            )
    source_closure = getattr(args, "stage_b_dense_duty_source_closure", None)
    receipt_code = trace_audit.get("code_source_closure")
    if not (
        isinstance(source_closure, dict)
        and isinstance(source_closure.get("code"), dict)
        and isinstance(receipt_code, dict)
        and receipt_code.get("sha256") == source_closure["code"].get("sha256")
    ):
        raise RuntimeError("dense-duty direct-trace receipt code binding drifted")
    canonical_contract = trace_audit.get("canonical_classes")
    if not isinstance(canonical_contract, dict):
        raise RuntimeError("dense-duty direct-trace canonical contract is missing")
    canonical_path = Path(str(canonical_contract.get("path", ""))).expanduser().resolve(
        strict=True
    )
    if _sha256_file(canonical_path) != canonical_contract.get("sha256"):
        raise RuntimeError("dense-duty direct-trace canonical classes drifted")
    tokenizer_contract = trace_audit.get("tokenizer")
    tokenizer_path = Path(str(getattr(args, "text_encoder_type", ""))).expanduser().resolve(
        strict=True
    )
    if not isinstance(tokenizer_contract, dict) or tokenizer_path != Path(
        str(tokenizer_contract.get("path", ""))
    ).expanduser().resolve(strict=True):
        raise RuntimeError("dense-duty direct-trace tokenizer path drifted")
    for record in tokenizer_contract.get("files", ()):
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            raise RuntimeError("dense-duty direct-trace tokenizer receipt is malformed")
        tokenizer_file = tokenizer_path / record["name"]
        if (
            not tokenizer_file.is_file()
            or int(tokenizer_file.stat().st_size) != record.get("size_bytes")
            or _sha256_file(tokenizer_file) != record.get("sha256")
        ):
            raise RuntimeError(
                f"dense-duty direct-trace tokenizer file drifted: {record['name']}"
            )
    runtime_dataset = Path(str(args.datasets)).expanduser().resolve(strict=True)
    if runtime_dataset != dataset_path:
        raise RuntimeError(
            "dense-duty runtime --datasets differs from its exact phase contract"
        )

    forbidden = list(
        getattr(args, "stage_b_dense_duty_forbidden_checkpoint_sha256", ()) or ()
    )
    if not forbidden or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for value in forbidden
    ):
        raise RuntimeError("dense-duty forbidden-checkpoint SHA list is invalid")
    if base_sha in forbidden or text_sha in forbidden:
        raise RuntimeError("dense-duty initializer is a forbidden Stage-B checkpoint")

    requested_scorer = _stage_b_v15_scorer_init_request(args)
    source_path = None
    source_sha = None
    if getattr(args, "resume", ""):
        source_path = Path(str(args.resume)).expanduser().resolve(strict=True)
        source_sha = _sha256_file(source_path)
    elif phase == "rank":
        if not getattr(args, "pretrain_model_path", ""):
            raise RuntimeError("dense-duty rank requires --pretrain_model_path")
        source_path = Path(str(args.pretrain_model_path)).expanduser().resolve(strict=True)
        source_sha = _sha256_file(source_path)
        if source_path != base_path or source_sha != base_sha:
            raise RuntimeError("dense-duty rank must initialize from exact Stage-A0006")
        scorer_path = Path(requested_scorer).expanduser().resolve(strict=True)
        if scorer_path != text_path:
            raise RuntimeError("dense-duty rank must initialize both text towers from exact OGC")
    else:
        if requested_scorer:
            raise RuntimeError(
                "dense-duty confidence must preserve the loaded rank checkpoint and "
                "must not reapply a scorer initializer"
            )
        if not getattr(args, "pretrain_model_path", ""):
            raise RuntimeError(
                "dense-duty confidence requires the completed rank checkpoint via "
                "--pretrain_model_path"
            )
        source_path = Path(str(args.pretrain_model_path)).expanduser().resolve(strict=True)
        source_sha = _sha256_file(source_path)
        if adapter_contract:
            expected_source_path = Path(
                str(
                    getattr(
                        args,
                        "stage_b_dense_duty_rank_source_checkpoint_path",
                        "",
                    )
                )
            ).expanduser().resolve(strict=True)
            expected_source_sha = str(
                getattr(
                    args,
                    "stage_b_dense_duty_rank_source_checkpoint_sha256",
                    "",
                )
            )
            if source_path != expected_source_path or source_sha != expected_source_sha:
                raise RuntimeError(
                    "confidence adapter must initialize from the exact selected U6551 rank"
                )
            if (
                int(
                    getattr(
                        args,
                        "stage_b_dense_duty_rank_source_optimizer_updates",
                        0,
                    )
                )
                != 6551
                or str(
                    getattr(
                        args,
                        "stage_b_dense_duty_rank_source_checkpoint_reason",
                        "",
                    )
                )
                != "signal"
            ):
                raise RuntimeError("confidence adapter rank-source progress contract drifted")
    if source_sha in forbidden:
        raise RuntimeError("dense-duty runtime source is the forbidden B58 checkpoint")

    scope = str(
        getattr(args, "stage_b_dense_duty_execution_scope", "formal") or ""
    ).strip().lower()
    if scope not in {"formal", "probe"}:
        raise RuntimeError("dense-duty execution scope must be 'formal' or 'probe'")
    if scope == "formal":
        if (
            type(getattr(args, "world_size", None)) is not int
            or getattr(args, "world_size") != 1
            or getattr(args, "distributed", None) is not False
        ):
            raise RuntimeError(
                "formal dense-duty training requires a single process so its "
                "checkpoint contains the complete runtime RNG state"
            )
        expected_updates = int(
            getattr(
                args,
                f"stage_b_dense_duty_{phase}_expected_optimizer_updates",
                0,
            )
        )
        if int(getattr(args, "max_train_iters", 0) or 0) != expected_updates:
            raise RuntimeError(
                f"formal dense-duty {phase} requires exactly "
                f"max_train_iters={expected_updates}"
            )
        forward_pack_factor = int(
            getattr(args, "stage_b_dense_duty_forward_pack_factor", 1) or 1
        )
        logical_loss_batch_size = int(
            getattr(args, "stage_b_dense_duty_logical_loss_batch_size", 0) or 0
        )
        expected_forward_batch_size = int(
            getattr(
                args,
                "stage_b_dense_duty_expected_forward_batch_size",
                int(getattr(args, "batch_size", 0)),
            )
        )
        if forward_pack_factor < 1:
            raise RuntimeError("formal dense-duty forward pack must be positive")
        if forward_pack_factor > 1:
            if phase != "confidence" or str(
                getattr(args, "stage_b_v22_score_ownership", "")
            ).strip() != "rank_tower_stopgrad_token_adapter_two_phase":
                raise RuntimeError(
                    "formal packed forward is restricted to confidence-adapter training"
                )
            if logical_loss_batch_size != int(getattr(args, "batch_size", 0)):
                raise RuntimeError(
                    "formal packed forward must preserve batch_size as its "
                    "logical loss batch"
                )
            observed_forward_batch_size = (
                logical_loss_batch_size * forward_pack_factor
            )
            if expected_forward_batch_size != observed_forward_batch_size:
                raise RuntimeError(
                    "formal packed forward batch differs from its measured "
                    f"contract: expected={expected_forward_batch_size}, "
                    f"observed={observed_forward_batch_size}"
                )
            expected_logical_batches = int(
                getattr(
                    args,
                    "stage_b_dense_duty_expected_logical_batches_per_epoch",
                    0,
                )
            )
            expected_physical_forwards = int(
                getattr(
                    args,
                    "stage_b_dense_duty_expected_physical_forwards_per_epoch",
                    0,
                )
            )
            observed_logical_batches = int(
                getattr(args, "stage_b_dense_duty_trace_total_rows", 0)
            ) // int(getattr(args, "batch_size", 0))
            observed_physical_forwards = math.ceil(
                observed_logical_batches / forward_pack_factor
            )
            if (
                expected_logical_batches != observed_logical_batches
                or expected_physical_forwards != observed_physical_forwards
            ):
                raise RuntimeError(
                    "formal packed epoch geometry differs from its measured "
                    "contract: "
                    f"expected=({expected_logical_batches}, "
                    f"{expected_physical_forwards}), observed=("
                    f"{observed_logical_batches}, {observed_physical_forwards})"
                )
        effective_batch = (
            int(getattr(args, "batch_size", 0))
            * forward_pack_factor
            * int(getattr(args, "gradient_accumulation_steps", 1))
        )
        if effective_batch != 64:
            raise RuntimeError(
                "formal dense-duty training requires effective batch 64, got "
                f"{effective_batch}"
            )
        expected_runtime = {
            "batch_size": int(
                getattr(
                    args,
                    "stage_b_dense_duty_expected_physical_batch_size",
                    0,
                )
            ),
            "gradient_accumulation_steps": int(
                getattr(
                    args,
                    "stage_b_dense_duty_expected_gradient_accumulation_steps",
                    0,
                )
            ),
            "stage_b_v11_expression_microbatch": int(
                getattr(
                    args,
                    "stage_b_dense_duty_expected_expression_microbatch",
                    0,
                )
            ),
        }
        observed_runtime = {
            key: int(getattr(args, key, 0)) for key in expected_runtime
        }
        if any(value <= 0 for value in expected_runtime.values()):
            raise RuntimeError(
                "formal dense-duty training lacks a positive measured runtime "
                "contract"
            )
        if observed_runtime != expected_runtime:
            raise RuntimeError(
                "formal dense-duty runtime differs from the measured stable "
                f"configuration: expected={expected_runtime}, "
                f"observed={observed_runtime}"
            )

    args.stage_b_dense_duty_lineage_audit = {
        "schema": "pivot.stageb.dense_duty_lineage/v1",
        "phase": phase,
        "execution_scope": scope,
        "no_stage_b_teacher": True,
        "base_checkpoint": {"path": str(base_path), "sha256": base_sha},
        "text_checkpoint": {"path": str(text_path), "sha256": text_sha},
        "dataset_config": {"path": str(dataset_path), "sha256": dataset_sha},
        "tn_manifest": {"path": str(tn_path), "sha256": tn_sha},
        "direct_trace_audit": {
            "path": str(trace_audit_path),
            "sha256": trace_audit_sha,
            "direct_token_valid_rows": expected_trace_counts[
                "direct_token_valid_rows"
            ],
        },
        "runtime_source": {"path": str(source_path), "sha256": source_sha},
        "forbidden_checkpoint_sha256": forbidden,
    }


def _validate_stage_b_v15_scorer_init_args(args) -> None:
    requested = _stage_b_v15_scorer_init_request(args)
    if not requested:
        return
    if not bool(getattr(args, "stage_b_v11_fixed_text", False)):
        raise RuntimeError(
            "stage_b_v15_scorer_init_checkpoint requires "
            "stage_b_v11_fixed_text=True"
        )
    explicit_ownership = str(
        getattr(args, "stage_b_v22_score_ownership", "") or ""
    ).strip()
    if not bool(getattr(args, "stage_b_v15_decoupled_confidence", False)) and not explicit_ownership:
        raise RuntimeError(
            "stage_b_v15_scorer_init_checkpoint requires "
            "stage_b_v15_decoupled_confidence=True or an explicit "
            "stage_b_v22_score_ownership contract"
        )
    if not getattr(args, "resume", "") and not getattr(
        args, "pretrain_model_path", ""
    ):
        raise RuntimeError(
            "stage_b_v15_scorer_init_checkpoint requires the Stage-A base to be "
            "loaded explicitly with --pretrain_model_path"
        )


def _validate_stage_b_v15_stage_a_pretrain_state(
    args,
    state_dict,
    *,
    checkpoint_payload: Optional[Mapping[str, Any]] = None,
) -> None:
    if bool(getattr(args, "stage_b_dense_duty", False)):
        phase = str(getattr(args, "stage_b_dense_duty_phase", "")).strip().lower()
        if not isinstance(checkpoint_payload, Mapping):
            raise RuntimeError("dense-duty initializer payload must be a mapping")
        saved_args = checkpoint_payload.get("args", {})
        if isinstance(saved_args, argparse.Namespace):
            saved_args = vars(saved_args)
        if not isinstance(saved_args, Mapping):
            saved_args = {}
        scorer_keys = sorted(
            str(key)
            for key in state_dict
            if str(key).startswith("stage_b_fixed_text_scorer.")
        )
        if phase == "rank":
            forbidden_stage_b = sorted(
                str(key)
                for key in state_dict
                if str(key).startswith("stage_b_")
            )
            if forbidden_stage_b:
                raise RuntimeError(
                    "dense-duty rank base is not a scorer-free Stage-A checkpoint: "
                    f"{forbidden_stage_b[:8]}"
                )
            if saved_args.get("patch_only") is not True:
                raise RuntimeError("dense-duty rank base must record patch_only=True")
        else:
            if not scorer_keys:
                raise RuntimeError(
                    "dense-duty confidence initializer has no trained dense scorer"
                )
            from util.stage_b_dense_duty_audit import (
                SOURCE_CLOSURE_ARG,
                validate_source_closure,
                validate_current_source_closure,
            )

            adapter_contract = str(
                getattr(args, "stage_b_v22_score_ownership", "")
            ).strip() == "rank_tower_stopgrad_token_adapter_two_phase"
            if adapter_contract:
                # The selected rank was trained under the sealed v1 source.
                # Its closure is validated internally; bitwise state migration
                # below is the bridge to the new v2 architecture.
                validate_source_closure(saved_args.get(SOURCE_CLOSURE_ARG))
            else:
                validate_current_source_closure(
                    saved_args.get(SOURCE_CLOSURE_ARG),
                    getattr(args, SOURCE_CLOSURE_ARG, None),
                    compare_config=False,
                )
            required_saved = {
                "stage_b_dense_duty": True,
                "stage_b_dense_duty_phase": "rank",
                "stage_b_dense_duty_no_stageb_teacher": True,
                "stage_b_v22_score_ownership": "independent_decoders_two_phase",
                "stage_b_dense_duty_base_checkpoint_sha256": getattr(
                    args, "stage_b_dense_duty_base_checkpoint_sha256"
                ),
                "stage_b_dense_duty_text_checkpoint_sha256": getattr(
                    args, "stage_b_dense_duty_text_checkpoint_sha256"
                ),
                "stage_b_dense_duty_dataset_config_sha256": getattr(
                    args, "stage_b_dense_duty_rank_dataset_config_sha256"
                ),
            }
            drift = {
                key: (saved_args.get(key), expected)
                for key, expected in required_saved.items()
                if saved_args.get(key) != expected
            }
            if drift:
                raise RuntimeError(
                    f"dense-duty confidence rank lineage drifted: {drift}"
                )
            expected_updates = int(
                getattr(
                    args,
                    (
                        "stage_b_dense_duty_rank_source_optimizer_updates"
                        if adapter_contract
                        else "stage_b_dense_duty_rank_expected_optimizer_updates"
                    ),
                )
            )
            execution_scope = str(
                getattr(args, "stage_b_dense_duty_execution_scope", "formal")
            ).strip().lower()
            if execution_scope == "formal" and adapter_contract:
                expected_reason = str(
                    getattr(
                        args,
                        "stage_b_dense_duty_rank_source_checkpoint_reason",
                    )
                )
                if (
                    checkpoint_payload.get("optimizer_updates") != expected_updates
                    or checkpoint_payload.get("checkpoint_reason") != expected_reason
                ):
                    raise RuntimeError(
                        "confidence adapter requires the exact selected U6551 "
                        "optimizer-boundary checkpoint"
                    )
            elif execution_scope == "formal":
                if (
                    checkpoint_payload.get("optimizer_updates") != expected_updates
                    or checkpoint_payload.get("checkpoint_reason") != "max_train_iters"
                    or saved_args.get("max_train_iters") != expected_updates
                ):
                    raise RuntimeError(
                        "dense-duty confidence requires the terminal formal rank "
                        f"checkpoint at exactly {expected_updates} updates"
                    )
            elif not isinstance(checkpoint_payload.get("optimizer_updates"), int) or int(
                checkpoint_payload.get("optimizer_updates")
            ) <= 0:
                raise RuntimeError(
                    "dense-duty confidence probe requires a rank checkpoint with "
                    "at least one successful optimizer update"
                )
            scorer_audit = saved_args.get("stage_b_v15_scorer_init_audit")
            if (
                not isinstance(scorer_audit, Mapping)
                or scorer_audit.get("source_sha256")
                != getattr(args, "stage_b_dense_duty_text_checkpoint_sha256")
            ):
                raise RuntimeError(
                    "dense-duty confidence initializer lacks exact OGC scorer lineage"
                )
            from util.stage_b_dense_duty_audit import (
                audit_checkpoint_payload,
                validate_confidence_adapter_rank_source_audit,
            )

            rank_checkpoint_audit = audit_checkpoint_payload(checkpoint_payload)
            if rank_checkpoint_audit.get("phase") != "rank":
                raise RuntimeError(
                    "dense-duty confidence initializer did not pass the rank "
                    "ownership-transition audit"
                )
            if adapter_contract:
                rank_checkpoint_audit = (
                    validate_confidence_adapter_rank_source_audit(
                        rank_checkpoint_audit,
                        args,
                    )
                )
            setattr(
                args,
                "stage_b_dense_duty_rank_source_checkpoint_audit",
                rank_checkpoint_audit,
            )
            setattr(args, "stage_b_v15_scorer_init_audit", dict(scorer_audit))
    if not _stage_b_v15_scorer_init_request(args):
        return
    scorer_keys = sorted(
        str(key)
        for key in state_dict
        if str(key).startswith("stage_b_fixed_text_scorer.")
    )
    if scorer_keys:
        raise RuntimeError(
            "Stage-B v15 scorer warm-start requires --pretrain_model_path to be "
            "a scorer-free Stage-A checkpoint; found scorer state such as "
            f"{scorer_keys[:8]}"
        )


def _validate_stage_b_v15_scorer_init_audit(
    audit: Any,
    *,
    scorer,
    checkpoint_label: str,
) -> dict:
    if not isinstance(audit, Mapping):
        raise RuntimeError(
            f"{checkpoint_label}: scorer warm-start audit must be a mapping"
        )
    audit = dict(audit)
    required = {
        "schema",
        "status",
        "requested_source_path",
        "resolved_source_path",
        "source_sha256",
        "source_size_bytes",
        "source_decoder_num_layers",
        "selected_source_layer_indices",
        "loaded_num_layers",
        "loaded_tensor_count",
        "loaded_components",
    }
    missing = sorted(required.difference(audit))
    if missing:
        raise RuntimeError(
            f"{checkpoint_label}: scorer warm-start audit is missing {missing}"
        )
    if audit["schema"] != _STAGE_B_V15_SCORER_INIT_AUDIT_SCHEMA:
        raise RuntimeError(
            f"{checkpoint_label}: unsupported scorer warm-start audit schema "
            f"{audit['schema']!r}"
        )
    if audit["status"] != "applied":
        raise RuntimeError(
            f"{checkpoint_label}: scorer warm-start status must be 'applied'"
        )
    source_sha256 = str(audit["source_sha256"])
    if len(source_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in source_sha256
    ):
        raise RuntimeError(
            f"{checkpoint_label}: invalid scorer source SHA-256"
        )
    resolved_source_path = Path(str(audit["resolved_source_path"]))
    if not resolved_source_path.is_absolute():
        raise RuntimeError(
            f"{checkpoint_label}: scorer source path must be absolute"
        )
    loaded_num_layers = int(audit["loaded_num_layers"])
    scorer_num_layers = int(
        getattr(
            scorer,
            "num_layers",
            getattr(getattr(scorer, "rank_tower", None), "num_layers", 0),
        )
    )
    if loaded_num_layers != scorer_num_layers:
        raise RuntimeError(
            f"{checkpoint_label}: scorer warm-start loaded_num_layers="
            f"{loaded_num_layers} does not match current scorer layers="
            f"{scorer_num_layers}"
        )
    selected_layers = list(audit["selected_source_layer_indices"])
    if (
        len(selected_layers) != loaded_num_layers
        or selected_layers != list(range(selected_layers[0], selected_layers[0] + loaded_num_layers))
    ):
        raise RuntimeError(
            f"{checkpoint_label}: invalid selected scorer source layers "
            f"{selected_layers}"
        )
    if int(audit["source_decoder_num_layers"]) < loaded_num_layers:
        raise RuntimeError(
            f"{checkpoint_label}: source decoder layer count is inconsistent"
        )
    if int(audit["source_size_bytes"]) <= 0 or int(
        audit["loaded_tensor_count"]
    ) <= 0:
        raise RuntimeError(
            f"{checkpoint_label}: scorer warm-start audit has invalid counts"
        )
    expected_components = list(
        getattr(
            scorer,
            "warmstart_components",
            [
                "decoder.layers[-N:]",
                "decoder.ref_point_head",
                "decoder.norm",
            ],
        )
    )
    if list(audit["loaded_components"]) != expected_components:
        raise RuntimeError(
            f"{checkpoint_label}: scorer warm-start components are not exact"
        )
    return audit


def _write_stage_b_v15_scorer_init_audit(args, audit: Mapping[str, Any]) -> None:
    if int(getattr(args, "rank", 0)) != 0 or not getattr(args, "output_dir", ""):
        return
    output = Path(args.output_dir) / "stage_b_v15_scorer_init_audit.json"
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(audit), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output)


def _atomic_torch_save_on_master(payload: Any, output: Path) -> None:
    """Publish a checkpoint only after its complete bytes reach the same disk."""
    if not utils.is_main_process():
        return
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def _apply_stage_b_v15_scorer_init(model, args, logger) -> Optional[dict]:
    requested = _stage_b_v15_scorer_init_request(args)
    if not requested:
        return None
    scorer = getattr(model, "stage_b_fixed_text_scorer", None)
    if scorer is None or not hasattr(
        scorer, "load_from_full_text_checkpoint_state"
    ):
        raise RuntimeError(
            "Stage-B v15 scorer warm-start requested, but the model has no "
            "compatible stage_b_fixed_text_scorer"
        )

    source_path = Path(os.path.expandvars(requested)).expanduser().resolve(strict=True)
    stat_before = source_path.stat()
    source_sha256 = _sha256_file(source_path)
    stat_after_hash = source_path.stat()
    stable_identity = (
        stat_before.st_dev,
        stat_before.st_ino,
        stat_before.st_size,
        stat_before.st_mtime_ns,
    )
    if stable_identity != (
        stat_after_hash.st_dev,
        stat_after_hash.st_ino,
        stat_after_hash.st_size,
        stat_after_hash.st_mtime_ns,
    ):
        raise RuntimeError(
            f"Stage-B v15 scorer source changed while hashing: {source_path}"
        )

    source_checkpoint = _torch_load_compat(str(source_path), map_location="cpu")
    stat_after_load = source_path.stat()
    if stable_identity != (
        stat_after_load.st_dev,
        stat_after_load.st_ino,
        stat_after_load.st_size,
        stat_after_load.st_mtime_ns,
    ):
        raise RuntimeError(
            f"Stage-B v15 scorer source changed while loading: {source_path}"
        )
    if not isinstance(source_checkpoint, Mapping):
        raise TypeError("Stage-B v15 scorer source checkpoint must be a mapping")
    if "model" not in source_checkpoint or not isinstance(
        source_checkpoint["model"], Mapping
    ):
        raise ValueError(
            "Stage-B v15 scorer source checkpoint must contain a mapping at 'model'"
        )
    source_model_state = clean_state_dict(source_checkpoint["model"])
    load_audit = scorer.load_from_full_text_checkpoint_state(
        source_model_state,
        checkpoint_label=f"Stage-B v15 scorer source {source_path}",
    )
    audit = {
        "schema": _STAGE_B_V15_SCORER_INIT_AUDIT_SCHEMA,
        "status": "applied",
        "requested_source_path": requested,
        "resolved_source_path": str(source_path),
        "source_sha256": source_sha256,
        "source_size_bytes": int(stat_before.st_size),
        **load_audit,
    }
    audit = _validate_stage_b_v15_scorer_init_audit(
        audit,
        scorer=scorer,
        checkpoint_label="new Stage-B v15 scorer initialization",
    )
    setattr(args, "stage_b_v15_scorer_init_audit", audit)
    _write_stage_b_v15_scorer_init_audit(args, audit)
    if logger is not None:
        logger.info(
            "Applied scorer-only Stage-B v15 warm-start; Stage-A candidate path "
            "was not modified:\n" + json.dumps(audit, indent=2, sort_keys=True)
        )
    return audit


def _restore_stage_b_v15_scorer_init_audit_for_resume(
    model,
    args,
    checkpoint: Mapping[str, Any],
    logger,
) -> Optional[dict]:
    checkpoint_args = checkpoint.get("args", {})
    if isinstance(checkpoint_args, argparse.Namespace):
        checkpoint_args = vars(checkpoint_args)
    if not isinstance(checkpoint_args, Mapping):
        checkpoint_args = {}
    recorded = checkpoint_args.get("stage_b_v15_scorer_init_audit")
    requested = _stage_b_v15_scorer_init_request(args)
    if recorded is None:
        if requested:
            raise RuntimeError(
                "Stage-B v15 --resume checkpoint has no scorer warm-start audit; "
                "remove stage_b_v15_scorer_init_checkpoint or start a new run "
                "with --pretrain_model_path"
            )
        return None

    scorer = getattr(model, "stage_b_fixed_text_scorer", None)
    if scorer is None:
        raise RuntimeError(
            "Stage-B v15 --resume checkpoint records a scorer warm-start, but "
            "the current model has no stage_b_fixed_text_scorer"
        )
    audit = _validate_stage_b_v15_scorer_init_audit(
        recorded,
        scorer=scorer,
        checkpoint_label="Stage-B v15 --resume checkpoint",
    )
    if requested:
        requested_resolved = str(
            Path(os.path.expandvars(requested)).expanduser().resolve(strict=False)
        )
        if requested_resolved != audit["resolved_source_path"]:
            raise RuntimeError(
                "Stage-B v15 scorer init path on --resume does not match the "
                f"recorded source: requested={requested_resolved}, "
                f"recorded={audit['resolved_source_path']}"
            )
    setattr(args, "stage_b_v15_scorer_init_audit", audit)
    _write_stage_b_v15_scorer_init_audit(args, audit)
    if logger is not None:
        logger.info(
            "Restored Stage-B v15 scorer warm-start provenance from --resume. "
            "The source checkpoint was not reopened and initialization was not "
            "reapplied:\n" + json.dumps(audit, indent=2, sort_keys=True)
        )
    return audit


def _audit_stage_b_v11_trainable_parameters(
    model,
    *,
    minimum: int = 5_000_000,
    maximum: int = 7_000_000,
) -> int:
    scorer = getattr(model, "stage_b_fixed_text_scorer", None)
    if scorer is None:
        raise RuntimeError(
            "stage_b_v11_fixed_text=True but stage_b_fixed_text_scorer is missing"
        )
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith("stage_b_fixed_text_scorer.")
    ]
    if unexpected:
        raise RuntimeError(
            "Stage B v11 must freeze Stage A/backbone/patch/bbox; unexpected "
            f"trainable parameters: {unexpected[:20]}"
        )
    scorer_trainable = sum(
        parameter.numel()
        for parameter in scorer.parameters()
        if parameter.requires_grad
    )
    total_trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if total_trainable != scorer_trainable or not int(minimum) <= total_trainable <= int(maximum):
        raise RuntimeError(
            "Stage B v11 trainable-parameter audit failed: "
            f"scorer={scorer_trainable:,}, total={total_trainable:,}, "
            f"expected range=[{int(minimum):,}, {int(maximum):,}]"
        )
    return total_trainable


def _freeze_and_audit_stage_b_dense_duty(
    model,
    *,
    phase: str,
    minimum: int,
    maximum: int,
) -> int:
    scorer = getattr(model, "stage_b_fixed_text_scorer", None)
    if scorer is None or not bool(getattr(scorer, "is_dense_duty", False)):
        raise RuntimeError(
            "stage_b_dense_duty=True but the dense-duty scorer is missing"
        )
    phase = str(phase).strip().lower()
    if phase not in {"rank", "confidence"}:
        raise RuntimeError(
            "dense-duty training phase must be exactly 'rank' or 'confidence'"
        )
    if not all(
        hasattr(scorer, name)
        for name in ("rank_parameters", "confidence_parameters", "set_phase")
    ):
        raise RuntimeError("dense-duty scorer lacks its parameter ownership API")

    scorer.set_phase(phase)
    scorer.train(True)
    rank_parameters = tuple(scorer.rank_parameters())
    confidence_parameters = tuple(scorer.confidence_parameters())
    rank_ids = {id(parameter) for parameter in rank_parameters}
    confidence_ids = {id(parameter) for parameter in confidence_parameters}
    if not rank_ids or not confidence_ids or rank_ids & confidence_ids:
        raise RuntimeError(
            "dense-duty rank/confidence ownership is empty or shares parameters"
        )
    confidence_rank_adaptation = tuple(
        scorer.confidence_rank_adaptation_parameters()
        if hasattr(scorer, "confidence_rank_adaptation_parameters")
        else ()
    )
    confidence_rank_adaptation_ids = {
        id(parameter) for parameter in confidence_rank_adaptation
    }
    if not confidence_rank_adaptation_ids.issubset(rank_ids):
        raise RuntimeError(
            "dense-duty confidence rank adaptation must be a subset of rank ownership"
        )
    active_ids = (
        rank_ids
        if phase == "rank"
        else confidence_ids | confidence_rank_adaptation_ids
    )

    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in active_ids)
    observed_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if observed_ids != active_ids:
        missing = len(active_ids.difference(observed_ids))
        unexpected = len(observed_ids.difference(active_ids))
        raise RuntimeError(
            "dense-duty phase ownership changed while freezing: "
            f"missing={missing}, unexpected={unexpected}"
        )
    unexpected_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith("stage_b_fixed_text_scorer.")
    ]
    if unexpected_names:
        raise RuntimeError(
            "dense-duty Stage B must freeze the complete Stage-A path; "
            f"unexpected trainable parameters: {unexpected_names[:20]}"
        )

    total_trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if not int(minimum) <= total_trainable <= int(maximum):
        raise RuntimeError(
            "dense-duty trainable-parameter audit failed: "
            f"phase={phase}, total={total_trainable:,}, "
            f"expected range=[{int(minimum):,}, {int(maximum):,}]"
        )
    return total_trainable


def _prepare_stage_b_dense_duty_state_fingerprint(
    model,
    args,
    logger,
    *,
    resume_checkpoint: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    if not bool(getattr(args, "stage_b_dense_duty", False)) or bool(args.eval):
        return None
    from util.stage_b_dense_duty_audit import (
        FINGERPRINT_ARG,
        SOURCE_CLOSURE_ARG,
        TRAINING_CONTRACT_ARG,
        build_training_contract,
        fingerprint_model,
        fingerprint_state,
        validate_initial_fingerprint,
        validate_confidence_adapter_rank_source_audit,
        validate_rank_handoff_audit,
        validate_resume_training_contract,
        validate_source_closure,
        write_json_atomic,
    )

    phase = str(getattr(args, "stage_b_dense_duty_phase", "")).strip().lower()
    if resume_checkpoint is None:
        training_contract = build_training_contract(args)
        initial = fingerprint_model(model, phase=phase)
        source = "fresh_phase_initializer"
    else:
        saved_args = resume_checkpoint.get("args", {})
        if isinstance(saved_args, argparse.Namespace):
            saved_args = vars(saved_args)
        if not isinstance(saved_args, Mapping):
            raise RuntimeError(
                "dense-duty resume checkpoint has no saved argument mapping"
            )
        training_contract = validate_resume_training_contract(args, saved_args)
        initial = validate_initial_fingerprint(
            saved_args.get(FINGERPRINT_ARG), expected_phase=phase
        )
        current = fingerprint_state(
            model.state_dict(),
            active_parameter_names=initial["active_parameter_names"],
            phase=phase,
        )
        if current["frozen"] != initial["frozen"]:
            raise RuntimeError(
                "dense-duty resume checkpoint changed state outside its phase owner"
            )
        completed_updates = resume_checkpoint.get("optimizer_updates", 0)
        if (
            isinstance(completed_updates, bool)
            or not isinstance(completed_updates, int)
            or completed_updates < 0
        ):
            raise RuntimeError(
                "dense-duty resume checkpoint has invalid optimizer_updates"
            )
        if (
            completed_updates > 0
            and current["active"]["sha256"] == initial["active"]["sha256"]
        ):
            raise RuntimeError(
                "dense-duty resume checkpoint reports updates but its active state "
                "is identical to initialization"
            )
        saved_runtime_audit = saved_args.get(
            "stage_b_dense_duty_runtime_audit"
        )
        if isinstance(saved_runtime_audit, Mapping):
            setattr(
                args,
                "stage_b_dense_duty_runtime_audit",
                dict(saved_runtime_audit),
            )
        if phase == "confidence":
            execution_scope = str(
                getattr(args, "stage_b_dense_duty_execution_scope", "formal")
                or ""
            ).strip().lower()
            adapter_contract = str(
                getattr(args, "stage_b_v22_score_ownership", "")
            ).strip() == "rank_tower_stopgrad_token_adapter_two_phase"
            if adapter_contract:
                from util.stage_b_confidence_adapter_migration import (
                    validate_confidence_adapter_migration_audit,
                )

                migration_audit = validate_confidence_adapter_migration_audit(
                    saved_args.get(
                        "stage_b_dense_duty_confidence_adapter_migration_audit"
                    ),
                    source_checkpoint_sha256=str(
                        getattr(
                            args,
                            "stage_b_dense_duty_rank_source_checkpoint_sha256",
                        )
                    ),
                    source_optimizer_updates=int(
                        getattr(
                            args,
                            "stage_b_dense_duty_rank_source_optimizer_updates",
                        )
                    ),
                    source_checkpoint_reason=str(
                        getattr(
                            args,
                            "stage_b_dense_duty_rank_source_checkpoint_reason",
                        )
                    ),
                    rank_sha256=str(
                        getattr(args, "stage_b_dense_duty_rank_source_rank_sha256")
                    ),
                    transferred_sha256=str(
                        getattr(
                            args,
                            "stage_b_dense_duty_rank_source_transferred_sha256",
                        )
                    ),
                )
                setattr(
                    args,
                    "stage_b_dense_duty_confidence_adapter_migration_audit",
                    migration_audit,
                )
                restored_rank_audit = (
                    validate_confidence_adapter_rank_source_audit(
                        saved_args.get(
                            "stage_b_dense_duty_rank_source_checkpoint_audit"
                        ),
                        args,
                    )
                )
            else:
                restored_rank_audit = validate_rank_handoff_audit(
                    saved_args.get(
                        "stage_b_dense_duty_rank_source_checkpoint_audit"
                    ),
                    execution_scope=execution_scope,
                    rank_dataset_sha256=str(
                        getattr(
                            args,
                            "stage_b_dense_duty_rank_dataset_config_sha256",
                            "",
                        )
                    ),
                    required_optimizer_updates=(
                        int(
                            getattr(
                                args,
                                "stage_b_dense_duty_rank_expected_optimizer_updates",
                            )
                        )
                        if execution_scope == "formal"
                        else None
                    ),
                    code_source_sha256=validate_source_closure(
                        getattr(args, SOURCE_CLOSURE_ARG)
                    )["code"]["sha256"],
                )
            setattr(
                args,
                "stage_b_dense_duty_rank_source_checkpoint_audit",
                restored_rank_audit,
            )
        source = "restored_phase_initializer"

    setattr(args, FINGERPRINT_ARG, initial)
    setattr(args, TRAINING_CONTRACT_ARG, training_contract)
    if int(getattr(args, "rank", 0)) == 0 and getattr(args, "output_dir", ""):
        write_json_atomic(
            Path(args.output_dir)
            / "stage_b_dense_duty_initial_state_fingerprint.json",
            initial,
        )
        write_json_atomic(
            Path(args.output_dir) / "stage_b_dense_duty_training_contract.json",
            training_contract,
        )
    if logger is not None:
        logger.info(
            "Sealed dense-duty initial state fingerprint: "
            f"phase={phase}, source={source}, "
            f"active={initial['active']['sha256']}, "
            f"frozen={initial['frozen']['sha256']}."
        )
    return initial


def _stage_b_gdino_adapter_train_mode(value) -> str:
    from models.GroundingDINO.stage_b_gdino_score_adapter import (
        stage_b_gdino_adapter_train_mode_code,
    )

    mode = str(value).strip()
    stage_b_gdino_adapter_train_mode_code(mode)
    return mode


def _freeze_and_audit_stage_b_gdino_adapter(
    model,
    train_mode: str = "joint",
) -> int:
    adapter = getattr(model, "stage_b_gdino_score_adapter", None)
    if adapter is None:
        raise RuntimeError(
            "stage_b_gdino_score_adapter=True but the adapter module is missing"
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    train_mode = _stage_b_gdino_adapter_train_mode(train_mode)
    active_parameters = []
    if train_mode in {"rank_only", "joint"}:
        active_parameters.extend(adapter.rank_parameters())
    if train_mode in {"confidence_only", "joint"}:
        active_parameters.extend(adapter.gate_parameters())
    for parameter in active_parameters:
        parameter.requires_grad_(True)
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith("stage_b_gdino_score_adapter.")
    ]
    if unexpected:
        raise RuntimeError(
            "pure-GDINO adapter must freeze the full base model; unexpected "
            f"trainable parameters: {unexpected[:20]}"
        )
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    expected = sum(parameter.numel() for parameter in active_parameters)
    if trainable <= 0 or trainable != expected:
        raise RuntimeError(
            f"GDINO adapter trainable audit failed: trainable={trainable}, expected={expected}"
        )
    return trainable


def _freeze_and_audit_stage_b_u0_patch_rank(model) -> int:
    adapter = getattr(model, "stage_b_u0_patch_rank_adapter", None)
    patch_encoder = getattr(model, "patch_encoder", None)
    query_projection = getattr(model, "query_proj_for_patch", None)
    if adapter is None or patch_encoder is None or query_projection is None:
        raise RuntimeError("Stage-B U0 patch-rank modules are incomplete")
    if getattr(patch_encoder, "backbone", None) is not model.backbone:
        raise RuntimeError("Stage-B U0 must share one frozen image/patch backbone")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    active_parameters = list(adapter.trainable_parameters())
    active_parameters.extend(patch_encoder.input_proj.parameters())
    active_parameters.extend(patch_encoder.norm.parameters())
    active_parameters.extend(query_projection.parameters())
    if model.patch_logit_scale is None:
        raise RuntimeError("Stage-B U0 is missing patch_logit_scale")
    for parameter in active_parameters:
        parameter.requires_grad_(True)
    allowed_prefixes = (
        "stage_b_u0_patch_rank_adapter.",
        "patch_encoder.input_proj.",
        "patch_encoder.norm.",
        "query_proj_for_patch.",
    )
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith(allowed_prefixes)
    ]
    if unexpected:
        raise RuntimeError(
            "Stage-B U0 exposed parameters outside its patch-rank surface: "
            f"{unexpected[:20]}"
        )
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise RuntimeError("Stage-B U0 must keep the shared b58 backbone frozen")
    if any(
        parameter.requires_grad
        for parameter in model.stage_b_gdino_score_adapter.parameters()
    ):
        raise RuntimeError("Stage-B U0 must keep sealed R100/P50 parameters frozen")
    trainable_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    expected_ids = {id(parameter) for parameter in active_parameters}
    if not trainable_ids or trainable_ids != expected_ids:
        raise RuntimeError(
            "Stage-B U0 trainable-parameter audit failed: "
            f"trainable={len(trainable_ids)}, expected={len(expected_ids)}"
        )
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _freeze_and_audit_stage_b_u0_gate_aligned_d10(model) -> int:
    """Expose only the eight D9 patch-projection tensors for D10."""
    adapter = getattr(model, "stage_b_u0_patch_rank_adapter", None)
    score_adapter = getattr(model, "stage_b_gdino_score_adapter", None)
    if adapter is None or score_adapter is None:
        raise RuntimeError("D10 requires the frozen U0 and R100/P50 adapters")
    active = _stage_b_native_patch_category_parameters(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in active:
        parameter.requires_grad_(True)
    observed = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    expected_ids = {id(parameter) for parameter in active}
    if len(observed) != 8 or {id(value) for value in observed.values()} != expected_ids:
        raise RuntimeError("D10 did not expose exactly eight patch-projection tensors")
    allowed_prefixes = (
        "patch_encoder.input_proj.",
        "patch_encoder.norm.",
        "query_proj_for_patch.",
    )
    unexpected = [name for name in observed if not name.startswith(allowed_prefixes)]
    if unexpected:
        raise RuntimeError(f"D10 exposed unexpected trainable tensors: {unexpected[:20]}")
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise RuntimeError("D10 must keep the shared b58 backbone frozen")
    if any(parameter.requires_grad for parameter in adapter.parameters()):
        raise RuntimeError("D10 must keep the U0 residual adapter frozen")
    if any(parameter.requires_grad for parameter in score_adapter.parameters()):
        raise RuntimeError("D10 must keep R100/P50 frozen")
    patch_logit_scale = getattr(model, "patch_logit_scale", None)
    if not isinstance(patch_logit_scale, torch.nn.Parameter):
        raise RuntimeError("D10 requires patch_logit_scale")
    if patch_logit_scale.requires_grad:
        raise RuntimeError("D10 patch_logit_scale must remain frozen")
    return sum(parameter.numel() for parameter in active)


def _freeze_and_audit_stage_b_u0_gate_aligned_d11(model) -> int:
    """Expose only R100's rank-output weight for surgical D11 tuning."""
    u0_adapter = getattr(model, "stage_b_u0_patch_rank_adapter", None)
    score_adapter = getattr(model, "stage_b_gdino_score_adapter", None)
    if u0_adapter is None or score_adapter is None:
        raise RuntimeError("D11 requires the frozen U0 and R100/P50 adapters")
    active = (score_adapter.rank_output.weight,)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in active:
        parameter.requires_grad_(True)
    observed = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    expected_ids = {id(parameter) for parameter in active}
    if len(observed) != 1 or {id(value) for value in observed.values()} != expected_ids:
        raise RuntimeError("D11 did not expose exactly one rank-output tensor")
    unexpected = [
        name
        for name in observed
        if not name.startswith("stage_b_gdino_score_adapter.rank_output.")
    ]
    if unexpected:
        raise RuntimeError(f"D11 exposed unexpected trainable tensors: {unexpected}")
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise RuntimeError("D11 must keep the shared b58 backbone frozen")
    if any(parameter.requires_grad for parameter in u0_adapter.parameters()):
        raise RuntimeError("D11 must keep U0 frozen")
    if any(
        parameter.requires_grad
        for parameter in score_adapter.gate_parameters()
    ):
        raise RuntimeError("D11 must keep P50 confidence frozen")
    patch_modules = (
        getattr(model, "patch_encoder", None),
        getattr(model, "query_proj_for_patch", None),
    )
    if any(
        parameter.requires_grad
        for module in patch_modules
        if module is not None
        for parameter in module.parameters()
    ):
        raise RuntimeError("D11 must keep the D9 patch gate frozen")
    return sum(parameter.numel() for parameter in active)


def _freeze_and_audit_stage_b_u0_gate_aligned_d12(model) -> int:
    """Expose only the zero-initialized D12 conditional rank residual."""
    d12 = getattr(model, "stage_b_u0_gate_aligned_rank_residual", None)
    u0_adapter = getattr(model, "stage_b_u0_patch_rank_adapter", None)
    score_adapter = getattr(model, "stage_b_gdino_score_adapter", None)
    if d12 is None or u0_adapter is None or score_adapter is None:
        raise RuntimeError("D12 requires its residual plus frozen U0/R100/P50")
    active = tuple(d12.trainable_parameters())
    if len(active) != 7 or len({id(parameter) for parameter in active}) != 7:
        raise RuntimeError("D12 requires exactly seven residual tensors")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in active:
        parameter.requires_grad_(True)
    observed = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    expected_ids = {id(parameter) for parameter in active}
    if {id(value) for value in observed.values()} != expected_ids:
        raise RuntimeError("D12 trainable tensors differ from its residual module")
    unexpected = [
        name
        for name in observed
        if not name.startswith("stage_b_u0_gate_aligned_rank_residual.")
    ]
    if unexpected:
        raise RuntimeError(f"D12 exposed unexpected tensors: {unexpected[:20]}")
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise RuntimeError("D12 must keep b58/backbone frozen")
    if any(parameter.requires_grad for parameter in u0_adapter.parameters()):
        raise RuntimeError("D12 must keep U0 frozen")
    if any(parameter.requires_grad for parameter in score_adapter.parameters()):
        raise RuntimeError("D12 must keep R100/P50 frozen")
    return sum(parameter.numel() for parameter in active)


def _freeze_and_audit_stage_b_u0_gate_aligned_d13(model) -> int:
    """Expose only D13's independent patch-category residual."""
    d13 = getattr(model, "stage_b_u0_gate_aligned_patch_residual", None)
    u0_adapter = getattr(model, "stage_b_u0_patch_rank_adapter", None)
    score_adapter = getattr(model, "stage_b_gdino_score_adapter", None)
    patch_encoder = getattr(model, "patch_encoder", None)
    query_projection = getattr(model, "query_proj_for_patch", None)
    if any(
        module is None
        for module in (
            d13,
            u0_adapter,
            score_adapter,
            patch_encoder,
            query_projection,
        )
    ):
        raise RuntimeError("D13 requires its residual plus frozen D9/U0/R100/P50")
    active = tuple(d13.trainable_parameters())
    if len(active) != 7 or len({id(parameter) for parameter in active}) != 7:
        raise RuntimeError("D13 requires exactly seven residual tensors")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in active:
        parameter.requires_grad_(True)
    observed = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    expected_ids = {id(parameter) for parameter in active}
    if {id(value) for value in observed.values()} != expected_ids:
        raise RuntimeError("D13 trainable tensors differ from its residual module")
    unexpected = [
        name
        for name in observed
        if not name.startswith("stage_b_u0_gate_aligned_patch_residual.")
    ]
    if unexpected:
        raise RuntimeError(f"D13 exposed unexpected tensors: {unexpected[:20]}")
    frozen_modules = (
        model.backbone,
        patch_encoder,
        query_projection,
        u0_adapter,
        score_adapter,
    )
    if any(
        parameter.requires_grad
        for module in frozen_modules
        for parameter in module.parameters()
    ):
        raise RuntimeError("D13 must freeze b58/D9/U0/R100/P50")
    return sum(parameter.numel() for parameter in active)


def _stage_b_native_patch_category_parameters(model):
    patch_encoder = getattr(model, "patch_encoder", None)
    query_projection = getattr(model, "query_proj_for_patch", None)
    if patch_encoder is None or query_projection is None:
        raise RuntimeError("native patch-category projection modules are incomplete")
    if getattr(patch_encoder, "backbone", None) is not model.backbone:
        raise RuntimeError(
            "native patch-category training must share the frozen b58 backbone"
        )
    active = tuple(
        list(patch_encoder.input_proj.parameters())
        + list(patch_encoder.norm.parameters())
        + list(query_projection.parameters())
    )
    if len(active) != 8 or len({id(parameter) for parameter in active}) != 8:
        raise RuntimeError(
            "native patch-category training requires exactly eight projection tensors"
        )
    return active


def _freeze_and_audit_stage_b_native_patch_category(model) -> int:
    active = _stage_b_native_patch_category_parameters(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in active:
        parameter.requires_grad_(True)

    expected_ids = {id(parameter) for parameter in active}
    observed = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if {id(parameter) for parameter in observed.values()} != expected_ids:
        raise RuntimeError(
            "native patch-category trainable parameters differ from the declared surface"
        )
    allowed_prefixes = (
        "patch_encoder.input_proj.",
        "patch_encoder.norm.",
        "query_proj_for_patch.",
    )
    unexpected = [
        name for name in observed if not name.startswith(allowed_prefixes)
    ]
    if unexpected or len(observed) != 8:
        raise RuntimeError(
            "native patch-category training exposed unexpected tensors: "
            f"{unexpected[:20]}"
        )
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise RuntimeError("native patch-category training must freeze the b58 backbone")
    patch_logit_scale = getattr(model, "patch_logit_scale", None)
    if not isinstance(patch_logit_scale, torch.nn.Parameter):
        raise RuntimeError("native patch-category training requires patch_logit_scale")
    if patch_logit_scale.requires_grad:
        raise RuntimeError(
            "native patch-category patch_logit_scale is deployment-inert and must freeze"
        )
    forbidden_modules = (
        getattr(model, "stage_b_gdino_score_adapter", None),
        getattr(model, "stage_b_u0_patch_rank_adapter", None),
        getattr(model, "stage_b_data_driven_score_heads", None),
    )
    if any(module is not None for module in forbidden_modules):
        raise RuntimeError(
            "native patch-category training cannot contain teacher or data-driven adapters"
        )
    return sum(parameter.numel() for parameter in active)


def _stage_b_native_patch_category_optimizer_groups(model, *, lr: float):
    lr = float(lr)
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError("stage_b_native_patch_lr must be finite and positive")
    parameters = [
        parameter
        for parameter in _stage_b_native_patch_category_parameters(model)
        if parameter.requires_grad
    ]
    if len(parameters) != 8:
        raise RuntimeError(
            "native patch-category optimizer requires exactly eight trainable tensors"
        )
    return [
        {
            "params": parameters,
            "lr": lr,
            "stage_b_native_patch_branch": "patch_category",
        }
    ]


def _stage_b_u0_gate_aligned_d10_optimizer_groups(model, *, lr: float):
    groups = _stage_b_native_patch_category_optimizer_groups(model, lr=lr)
    groups[0].pop("stage_b_native_patch_branch")
    groups[0]["stage_b_u0_d10_branch"] = "patch_projection"
    return groups


def _stage_b_u0_gate_aligned_d11_optimizer_groups(model, *, lr: float):
    lr = float(lr)
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError("stage_b_u0_d11_rank_lr must be finite and positive")
    adapter = getattr(model, "stage_b_gdino_score_adapter", None)
    if adapter is None:
        raise RuntimeError("D11 requires the R100/P50 score adapter")
    parameters = [adapter.rank_output.weight]
    if not parameters[0].requires_grad or adapter.rank_output.bias.requires_grad:
        raise RuntimeError("D11 optimizer requires only the rank-output weight")
    return [
        {
            "params": parameters,
            "lr": lr,
            "stage_b_u0_d11_branch": "r100_rank_output",
        }
    ]


def _stage_b_u0_gate_aligned_d12_optimizer_groups(model, *, lr: float):
    lr = float(lr)
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError("stage_b_u0_d12_rank_lr must be finite and positive")
    d12 = getattr(model, "stage_b_u0_gate_aligned_rank_residual", None)
    if d12 is None:
        raise RuntimeError("D12 residual module is missing")
    parameters = [
        parameter for parameter in d12.trainable_parameters() if parameter.requires_grad
    ]
    if len(parameters) != 7:
        raise RuntimeError("D12 optimizer requires exactly seven trainable tensors")
    return [
        {
            "params": parameters,
            "lr": lr,
            "stage_b_u0_d12_branch": "conditional_rank_residual",
        }
    ]


def _stage_b_u0_gate_aligned_d13_optimizer_groups(model, *, lr: float):
    lr = float(lr)
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError("stage_b_u0_d13_patch_lr must be finite and positive")
    d13 = getattr(model, "stage_b_u0_gate_aligned_patch_residual", None)
    if d13 is None:
        raise RuntimeError("D13 residual module is missing")
    parameters = [
        parameter for parameter in d13.trainable_parameters() if parameter.requires_grad
    ]
    if len(parameters) != 7:
        raise RuntimeError("D13 optimizer requires exactly seven trainable tensors")
    return [
        {
            "params": parameters,
            "lr": lr,
            "stage_b_u0_d13_branch": "conditional_patch_residual",
        }
    ]


def _stage_b_data_driven_train_mode(value) -> str:
    from models.GroundingDINO.stage_b_data_driven_score import (
        normalize_data_driven_train_mode,
    )

    return normalize_data_driven_train_mode(value)


def _stage_b_data_driven_parameter_groups(model, train_mode: str):
    heads = getattr(model, "stage_b_data_driven_score_heads", None)
    if heads is None:
        raise RuntimeError("stage_b_data_driven_score=True but score heads are missing")
    train_mode = _stage_b_data_driven_train_mode(train_mode)
    rank = list(heads.rank_parameters())
    confidence = list(heads.confidence_parameters())
    patch_encoder = getattr(model, "patch_encoder", None)
    query_projection = getattr(model, "query_proj_for_patch", None)
    if patch_encoder is None or query_projection is None:
        raise RuntimeError("data-driven patch projection modules are incomplete")
    patch_residual = getattr(
        model, "stage_b_data_driven_patch_residual", None
    )
    if patch_residual is None:
        patch = list(patch_encoder.input_proj.parameters())
        patch.extend(patch_encoder.norm.parameters())
        patch.extend(query_projection.parameters())
    else:
        patch = list(patch_residual.trainable_parameters())
        patch_architecture = patch_residual.architecture()
        if (
            len(patch)
            != int(patch_architecture.get("trainable_tensors", -1))
            or sum(parameter.numel() for parameter in patch)
            != int(patch_architecture.get("trainable_parameters", -1))
        ):
            raise RuntimeError(
                "data-driven patch residual surface differs from its architecture"
            )
    patch_logit_scale = getattr(model, "patch_logit_scale", None)
    if not isinstance(patch_logit_scale, torch.nn.Parameter):
        raise RuntimeError("data-driven patch scoring requires patch_logit_scale")
    # Deployment standardizes patch logits per row, so any positive global
    # scale is exactly decision-inert.  It must never be an optimizer shortcut.
    groups = {
        "rank": tuple(rank),
        "confidence": tuple(confidence),
        "patch": tuple(patch),
    }
    ids = {name: {id(parameter) for parameter in values} for name, values in groups.items()}
    for left, right in (("rank", "confidence"), ("rank", "patch"), ("confidence", "patch")):
        if ids[left] & ids[right]:
            raise RuntimeError(
                f"data-driven parameter branches overlap: {left}/{right}"
            )
    if train_mode == "rank_patch_only":
        active = groups["rank"] + groups["patch"]
    else:
        active = groups["confidence"]
    if not active:
        raise RuntimeError("data-driven train mode has no active parameters")
    return groups, active


def _freeze_and_audit_stage_b_data_driven(model, train_mode: str) -> int:
    patch_encoder = getattr(model, "patch_encoder", None)
    if patch_encoder is None or getattr(patch_encoder, "backbone", None) is not model.backbone:
        raise RuntimeError(
            "data-driven scorer must share the frozen b58 image/patch backbone"
        )
    groups, active = _stage_b_data_driven_parameter_groups(model, train_mode)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in active:
        parameter.requires_grad_(True)
    active_ids = {id(parameter) for parameter in active}
    observed_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    if observed_ids != active_ids:
        raise RuntimeError(
            "data-driven trainable parameter audit differs from its declared branches"
        )
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise RuntimeError("data-driven training must keep the b58 backbone frozen")
    patch_logit_scale = getattr(model, "patch_logit_scale", None)
    if (
        not isinstance(patch_logit_scale, torch.nn.Parameter)
        or patch_logit_scale.requires_grad
    ):
        raise RuntimeError(
            "data-driven patch_logit_scale is deployment-inert and must freeze"
        )
    patch_residual = getattr(
        model, "stage_b_data_driven_patch_residual", None
    )
    if patch_residual is not None:
        frozen_patch_modules = (
            patch_encoder.input_proj,
            patch_encoder.norm,
            getattr(model, "query_proj_for_patch"),
        )
        if any(
            parameter.requires_grad
            for module in frozen_patch_modules
            for parameter in module.parameters()
        ):
            raise RuntimeError(
                "patch residual training must freeze the base patch scorer"
            )
    inactive = (
        groups["confidence"]
        if _stage_b_data_driven_train_mode(train_mode) == "rank_patch_only"
        else groups["rank"] + groups["patch"]
    )
    if any(parameter.requires_grad for parameter in inactive):
        raise RuntimeError("an inactive data-driven branch remains trainable")
    return sum(parameter.numel() for parameter in active)


def _stage_b_data_driven_optimizer_groups(
    model,
    *,
    train_mode: str,
    rank_lr: float,
    confidence_lr: float,
    patch_lr: float,
):
    groups, _active = _stage_b_data_driven_parameter_groups(model, train_mode)
    mode = _stage_b_data_driven_train_mode(train_mode)
    result = []
    if mode == "rank_patch_only":
        for name, lr in (("rank", rank_lr), ("patch", patch_lr)):
            lr = float(lr)
            if not math.isfinite(lr) or lr <= 0.0:
                raise ValueError(f"data-driven {name} LR must be finite and positive")
            parameters = [
                parameter for parameter in groups[name] if parameter.requires_grad
            ]
            if not parameters:
                raise RuntimeError(f"data-driven {name} optimizer group is empty")
            result.append(
                {
                    "params": parameters,
                    "lr": lr,
                    "stage_b_data_driven_branch": name,
                }
            )
    else:
        confidence_lr = float(confidence_lr)
        if not math.isfinite(confidence_lr) or confidence_lr <= 0.0:
            raise ValueError("data-driven confidence LR must be finite and positive")
        parameters = [
            parameter
            for parameter in groups["confidence"]
            if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError("data-driven confidence optimizer group is empty")
        result.append(
            {
                "params": parameters,
                "lr": confidence_lr,
                "stage_b_data_driven_branch": "confidence",
            }
        )
    return result


def _stage_b_u0_patch_rank_optimizer_groups(
    model,
    *,
    residual_lr: float,
    patch_projection_lr: float,
    direct_patch_gain_lr: Optional[float] = None,
):
    adapter = getattr(model, "stage_b_u0_patch_rank_adapter", None)
    if adapter is None:
        raise RuntimeError("missing Stage-B U0 patch-rank adapter")
    residual_lr = float(residual_lr)
    patch_projection_lr = float(patch_projection_lr)
    if not math.isfinite(residual_lr) or residual_lr <= 0.0:
        raise ValueError("stage_b_u0_patch_rank_lr must be finite and positive")
    if not math.isfinite(patch_projection_lr) or patch_projection_lr <= 0.0:
        raise ValueError(
            "stage_b_u0_patch_projection_lr must be finite and positive"
        )
    direct_patch_gain = getattr(adapter, "direct_patch_gain", None)
    residual = [
        parameter
        for parameter in adapter.trainable_parameters()
        if parameter.requires_grad and parameter is not direct_patch_gain
    ]
    residual_ids = {id(parameter) for parameter in residual}
    patch_projection = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
        and id(parameter) not in residual_ids
        and parameter is not direct_patch_gain
    ]
    patch_ids = {id(parameter) for parameter in patch_projection}
    trainable_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    gain_parameters = []
    if direct_patch_gain is not None and direct_patch_gain.requires_grad:
        if direct_patch_gain_lr is None:
            raise ValueError("stage_b_u1_direct_patch_gain_lr is required")
        direct_patch_gain_lr = float(direct_patch_gain_lr)
        if not math.isfinite(direct_patch_gain_lr) or direct_patch_gain_lr <= 0.0:
            raise ValueError(
                "stage_b_u1_direct_patch_gain_lr must be finite and positive"
            )
        gain_parameters = [direct_patch_gain]
    gain_ids = {id(parameter) for parameter in gain_parameters}
    if not residual_ids or not patch_ids or residual_ids & patch_ids:
        raise RuntimeError("Stage-B U0 optimizer branches are empty or overlap")
    if residual_ids & gain_ids or patch_ids & gain_ids:
        raise RuntimeError("Stage-B U1 direct gain optimizer branch overlaps")
    if residual_ids | patch_ids | gain_ids != trainable_ids:
        raise RuntimeError("Stage-B U0 optimizer groups do not cover trainable parameters")
    groups = [
        {
            "params": residual,
            "lr": residual_lr,
            "stage_b_u0_branch": "patch_rank_residual",
        },
        {
            "params": patch_projection,
            "lr": patch_projection_lr,
            "stage_b_u0_branch": "patch_projection",
        },
    ]
    if gain_parameters:
        groups.insert(
            1,
            {
                "params": gain_parameters,
                "lr": direct_patch_gain_lr,
                "weight_decay": 0.0,
                "stage_b_u0_branch": "direct_patch_gain",
            },
        )
    return groups


def _stage_b_gdino_adapter_optimizer_groups(
    model,
    *,
    rank_lr: float,
    gate_lr: float,
    train_mode: str = "joint",
):
    adapter = getattr(model, "stage_b_gdino_score_adapter", None)
    if adapter is None:
        raise RuntimeError("missing stage_b_gdino_score_adapter")
    train_mode = _stage_b_gdino_adapter_train_mode(train_mode)
    rank_enabled = train_mode in {"rank_only", "joint"}
    gate_enabled = train_mode in {"confidence_only", "joint"}
    rank_lr = float(rank_lr)
    gate_lr = float(gate_lr)
    if rank_enabled and (not math.isfinite(rank_lr) or rank_lr <= 0.0):
        raise ValueError("stage_b_gdino_rank_lr must be finite and positive")
    if gate_enabled and (not math.isfinite(gate_lr) or gate_lr <= 0.0):
        raise ValueError("stage_b_gdino_gate_lr must be finite and positive")
    rank_parameters = [
        parameter for parameter in adapter.rank_parameters() if parameter.requires_grad
    ]
    gate_parameters = [
        parameter for parameter in adapter.gate_parameters() if parameter.requires_grad
    ]
    rank_ids = {id(parameter) for parameter in rank_parameters}
    gate_ids = {id(parameter) for parameter in gate_parameters}
    trainable_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if (rank_enabled and not rank_ids) or (gate_enabled and not gate_ids) or rank_ids & gate_ids:
        raise RuntimeError("GDINO adapter rank/gate optimizer branches are not disjoint")
    if rank_ids | gate_ids != trainable_ids:
        raise RuntimeError("GDINO adapter optimizer groups do not cover trainable parameters exactly")
    groups = []
    if rank_enabled:
        groups.append({
            "params": rank_parameters,
            "lr": rank_lr,
            "stage_b_gdino_branch": "rank",
        })
    if gate_enabled:
        groups.append({
            "params": gate_parameters,
            "lr": gate_lr,
            "stage_b_gdino_branch": "confidence",
        })
    return groups


def _isolate_stage_b_v15_validity_optimizer_group(
    param_dicts,
    model,
    *,
    validity_lr: float,
):
    validity_lr = float(validity_lr)
    if not math.isfinite(validity_lr) or validity_lr <= 0.0:
        raise ValueError(
            f"stage_b_v15_validity_lr must be finite and positive, got {validity_lr}"
        )

    trainable = {
        id(parameter): (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    validity = {
        parameter_id: (name, parameter)
        for parameter_id, (name, parameter) in trainable.items()
        if name.startswith("stage_b_fixed_text_scorer.validity_head.")
    }
    if not validity:
        raise RuntimeError(
            "stage_b_v15_validity_lr is set but no trainable "
            "stage_b_fixed_text_scorer.validity_head parameters were found"
        )

    def _audit_coverage(groups, *, label: str) -> None:
        counts = {}
        unknown = []
        for group_idx, group in enumerate(groups):
            for parameter in group.get("params", []):
                parameter_id = id(parameter)
                counts[parameter_id] = counts.get(parameter_id, 0) + 1
                if parameter_id not in trainable:
                    unknown.append(group_idx)
        missing = [name for pid, (name, _) in trainable.items() if counts.get(pid, 0) == 0]
        duplicate = [
            name for pid, (name, _) in trainable.items() if counts.get(pid, 0) > 1
        ]
        if missing or duplicate or unknown:
            raise RuntimeError(
                f"Stage B v15 optimizer parameter coverage failed {label}: "
                f"missing={missing[:20]}, duplicate={duplicate[:20]}, "
                f"unknown_group_indices={unknown[:20]}"
            )

    _audit_coverage(param_dicts, label="before validity split")

    validity_ids = set(validity)
    split_groups = []
    validity_groups = []
    for group in param_dicts:
        source_params = list(group.get("params", []))
        rank_params = [p for p in source_params if id(p) not in validity_ids]
        validity_params = [p for p in source_params if id(p) in validity_ids]

        rank_group = dict(group)
        rank_group["params"] = rank_params
        rank_group.pop("stage_b_v15_validity_group", None)
        split_groups.append(rank_group)

        if validity_params:
            validity_group = dict(group)
            validity_group["params"] = validity_params
            validity_group["lr"] = validity_lr
            validity_group["stage_b_v15_validity_group"] = True
            validity_groups.append(validity_group)

    if len(validity_groups) != 1:
        raise RuntimeError(
            "Stage B v15 validity parameters must originate from exactly one "
            f"optimizer group so its options remain unambiguous; got {len(validity_groups)}"
        )
    split_groups.extend(validity_groups)
    _audit_coverage(split_groups, label="after validity split")
    return split_groups


def _isolate_stage_b_dense_duty_rank_adaptation_optimizer_group(
    param_dicts,
    model,
    *,
    adaptation_lr: float,
):
    """Give the explicitly unfrozen rank-decoder suffix one conservative LR."""
    adaptation_lr = float(adaptation_lr)
    if not math.isfinite(adaptation_lr) or adaptation_lr <= 0.0:
        raise ValueError("rank-adaptation learning rate must be finite and positive")
    scorer = getattr(model, "stage_b_fixed_text_scorer", None)
    provider = getattr(scorer, "confidence_rank_adaptation_parameters", None)
    if not callable(provider):
        raise RuntimeError("rank-adaptation optimizer requires explicit ownership")
    adaptation_ids = {id(parameter) for parameter in provider()}
    if not adaptation_ids:
        raise RuntimeError("rank-adaptation learning rate set with an empty owner")

    split_groups = []
    adaptation_parameters = []
    seen = set()
    for group in param_dicts:
        source = list(group.get("params", ()))
        retained = [parameter for parameter in source if id(parameter) not in adaptation_ids]
        selected = [parameter for parameter in source if id(parameter) in adaptation_ids]
        if retained:
            retained_group = dict(group)
            retained_group["params"] = retained
            split_groups.append(retained_group)
        for parameter in selected:
            parameter_id = id(parameter)
            if parameter_id in seen:
                raise RuntimeError("rank-adaptation optimizer ownership is duplicated")
            seen.add(parameter_id)
            adaptation_parameters.append(parameter)
    if seen != adaptation_ids:
        raise RuntimeError(
            "rank-adaptation optimizer does not cover the exact decoder suffix"
        )
    split_groups.append(
        {
            "params": adaptation_parameters,
            "lr": adaptation_lr,
            "stage_b_dense_duty_rank_adaptation_group": True,
        }
    )
    return split_groups


def _audit_stage_b_v11_optimizer_group_lrs(
    param_dicts,
    *,
    base_lr: float,
    validity_lr: Optional[float] = None,
) -> float:
    base_lr = float(base_lr)
    expected_validity_lr = (
        None if validity_lr is None else float(validity_lr)
    )
    validity_group_count = 0
    group_lrs = []
    for group in param_dicts:
        group_lr = float(group.get("lr", base_lr))
        group_lrs.append(group_lr)
        if bool(group.get("stage_b_v15_validity_group", False)):
            validity_group_count += 1
            if expected_validity_lr is None or not math.isclose(
                group_lr,
                expected_validity_lr,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    "Stage B v15 validity optimizer group must use the explicit "
                    f"stage_b_v15_validity_lr={expected_validity_lr}, got {group_lr}"
                )
            continue
        if group_lr > base_lr + max(1e-12, abs(base_lr) * 1e-9):
            raise RuntimeError(
                "Stage B v11 optimizer group learning rate exceeds the base rate: "
                f"group_lr={group_lr}, groups={group_lrs}, base_lr={base_lr}. "
                "Only the explicit Stage B v15 validity group may exceed base LR. "
                "Note that lr_linear_proj_mult is an absolute learning rate in "
                "this repository."
            )

    expected_validity_groups = 1 if expected_validity_lr is not None else 0
    if validity_group_count != expected_validity_groups:
        raise RuntimeError(
            "Stage B v15 validity optimizer group count mismatch: "
            f"expected={expected_validity_groups}, got={validity_group_count}"
        )
    maximum = max(group_lrs, default=base_lr)
    return maximum


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument("--datasets", type=str, required=True, help='path to datasets json')
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument(
        '--prefetch_factor',
        default=1,
        type=int,
        help='DataLoader batches prefetched per worker when num_workers > 0; lower values use fewer shared-memory file descriptors',
    )
    parser.add_argument(
        '--pin_memory',
        dest='pin_memory',
        action='store_true',
        default=None,
        help='enable DataLoader pin_memory; default is enabled for CUDA devices',
    )
    parser.add_argument(
        '--no_pin_memory',
        '--no-pin-memory',
        dest='pin_memory',
        action='store_false',
        help='disable DataLoader pin_memory',
    )
    parser.add_argument(
        '--persistent_workers',
        dest='persistent_workers',
        action='store_true',
        default=None,
        help='keep DataLoader workers alive between epochs; default is enabled when num_workers > 0',
    )
    parser.add_argument(
        '--no_persistent_workers',
        '--no-persistent-workers',
        dest='persistent_workers',
        action='store_false',
        help='disable persistent DataLoader workers',
    )
    parser.add_argument(
        '--mp_sharing_strategy',
        default=os.environ.get("TORCH_MP_SHARING_STRATEGY", "file_system"),
        choices=("file_system", "file_descriptor", "none"),
        help='torch multiprocessing CPU tensor sharing strategy; file_system avoids one fd per shared storage',
    )
    parser.add_argument(
        '--min_nofile',
        default=_env_int("GDINO_MIN_NOFILE", 65536),
        type=int,
        help='try to raise the process open-file soft limit to at least this value; 0 disables',
    )
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')
    parser.add_argument(
        '--iter_checkpoint_interval',
        default=0,
        type=int,
        help='save output_dir/checkpoint_iter.pth every N successful optimizer updates; 0 disables periodic saves',
    )
    parser.add_argument(
        '--max_train_iters',
        default=0,
        type=int,
        help='stop training after N successful optimizer updates by writing output_dir/checkpoint_iter.pth; 0 disables',
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        '--gradient-accumulation-steps',
        default=1,
        type=int,
        help='number of DataLoader micro-batches per optimizer update; default 1 preserves legacy behavior',
    )

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--local-rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


def _capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


STAGE_B_V25_TRAINING_STATE_LAYOUT_SCHEMA = (
    "pivot.stageb.v25_training_state_layout/v1"
)
STAGE_B_V25_TRAINING_STATE_LAYOUT_FILENAME = "training_state_layout.json"
STAGE_B_V25_TRAINING_STATE_LAYOUT_ARG = (
    "stage_b_v25_training_state_layout_sha256"
)
STAGE_B_V25_TRAINABLE_PARAMETER_COUNT = 94
STAGE_B_V25_OPTIMIZER_GROUP_COUNT = 4
_STAGE_B_V25_OPTIMIZER_PARAMETER_NAMES_KEY = (
    "stage_b_v25_parameter_names"
)


def _stage_b_v25_canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _stage_b_v25_semantic_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("semantic_sha256", None)
    return hashlib.sha256(_stage_b_v25_canonical_json_bytes(payload)).hexdigest()


def _stage_b_v25_class_name(value: Any) -> str:
    cls = value.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _stage_b_v25_key_contract(value: Any) -> dict[str, Any]:
    if value is None or isinstance(value, (bool, int, str)):
        return {"kind": type(value).__name__, "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError("Stage-B v25 layout cannot encode a non-finite key")
        return {"kind": "float", "value": value}
    if isinstance(value, np.generic):
        return _stage_b_v25_key_contract(value.item())
    raise RuntimeError(
        "Stage-B v25 layout encountered an unsupported mapping key type: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _stage_b_v25_value_schema(value: Any) -> dict[str, Any]:
    if torch.is_tensor(value):
        return {
            "kind": "tensor",
            "dtype": str(value.dtype),
            "layout": str(value.layout),
            "shape": [int(dimension) for dimension in value.shape],
            "numel": int(value.numel()),
        }
    if isinstance(value, np.ndarray):
        return {
            "kind": "numpy.ndarray",
            "dtype": str(value.dtype),
            "shape": [int(dimension) for dimension in value.shape],
            "size": int(value.size),
        }
    if isinstance(value, np.generic):
        return {"kind": f"numpy.{value.dtype}"}
    if isinstance(value, Mapping):
        return {
            "kind": "mapping",
            "entries": [
                {
                    "key": _stage_b_v25_key_contract(key),
                    "value": _stage_b_v25_value_schema(nested),
                }
                for key, nested in value.items()
            ],
        }
    if isinstance(value, (tuple, list)):
        item_schemas = [_stage_b_v25_value_schema(item) for item in value]
        sequence_schema: dict[str, Any] = {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "length": len(item_schemas),
        }
        if item_schemas and all(item == item_schemas[0] for item in item_schemas):
            sequence_schema["homogeneous_item"] = item_schemas[0]
        else:
            sequence_schema["items"] = item_schemas
        return sequence_schema
    if value is None:
        return {"kind": "NoneType"}
    if isinstance(value, bool):
        return {"kind": "bool"}
    if isinstance(value, int):
        return {"kind": "int"}
    if isinstance(value, float):
        return {"kind": "float"}
    if isinstance(value, str):
        return {"kind": "str"}
    raise RuntimeError(
        "Stage-B v25 layout encountered an unsupported state value type: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _stage_b_v25_value_contract(value: Any) -> dict[str, Any]:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return {"kind": type(value).__name__, "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError(
                "Stage-B v25 layout cannot encode a non-finite static value"
            )
        return {"kind": "float", "value": value}
    if torch.is_tensor(value):
        if value.numel() > 64:
            raise RuntimeError(
                "Stage-B v25 static state unexpectedly contains a large tensor"
            )
        return {
            **_stage_b_v25_value_schema(value),
            "values": value.detach().cpu().reshape(-1).tolist(),
        }
    if isinstance(value, np.ndarray):
        if value.size > 64:
            raise RuntimeError(
                "Stage-B v25 static state unexpectedly contains a large array"
            )
        return {
            **_stage_b_v25_value_schema(value),
            "values": value.reshape(-1).tolist(),
        }
    if isinstance(value, Mapping):
        return {
            "kind": "mapping",
            "entries": [
                {
                    "key": _stage_b_v25_key_contract(key),
                    "value": _stage_b_v25_value_contract(nested),
                }
                for key, nested in value.items()
            ],
        }
    if isinstance(value, (tuple, list)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": [_stage_b_v25_value_contract(item) for item in value],
        }
    raise RuntimeError(
        "Stage-B v25 layout encountered an unsupported static value type: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _stage_b_v25_ordered_state_schema(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(state, Mapping):
        raise RuntimeError("Stage-B v25 state_dict must be a mapping")
    return [
        {"name": str(name), "value": _stage_b_v25_value_schema(value)}
        for name, value in state.items()
    ]


def _stage_b_v25_optimizer_static_options(group: Mapping[str, Any]) -> dict[str, Any]:
    ignored = {
        "params",
        "lr",
        "initial_lr",
        _STAGE_B_V25_OPTIMIZER_PARAMETER_NAMES_KEY,
    }
    options = {
        str(key): _stage_b_v25_value_contract(value)
        for key, value in group.items()
        if key not in ignored
    }
    initial_lr = group.get("initial_lr", group.get("lr"))
    if initial_lr is None:
        raise RuntimeError("Stage-B v25 optimizer group has no initial learning rate")
    options["initial_lr"] = _stage_b_v25_value_contract(initial_lr)
    return dict(sorted(options.items()))


def _build_stage_b_v25_training_state_layout(
    model: torch.nn.Module,
    criterion: Any,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    scaler: Any,
    *,
    rng_state: Optional[Mapping[str, Any]] = None,
    expected_trainable_parameter_count: int = (
        STAGE_B_V25_TRAINABLE_PARAMETER_COUNT
    ),
    expected_optimizer_group_count: int = STAGE_B_V25_OPTIMIZER_GROUP_COUNT,
) -> dict[str, Any]:
    named_parameters = list(model.named_parameters())
    trainable_parameters = [
        {
            "name": name,
            "shape": [int(dimension) for dimension in parameter.shape],
            "numel": int(parameter.numel()),
        }
        for name, parameter in named_parameters
        if parameter.requires_grad
    ]
    if len(trainable_parameters) != int(expected_trainable_parameter_count):
        raise RuntimeError(
            "Stage-B v25 trainable parameter layout drifted: "
            f"expected {expected_trainable_parameter_count}, "
            f"got {len(trainable_parameters)}"
        )
    if len(optimizer.param_groups) != int(expected_optimizer_group_count):
        raise RuntimeError(
            "Stage-B v25 optimizer group layout drifted: "
            f"expected {expected_optimizer_group_count}, "
            f"got {len(optimizer.param_groups)}"
        )

    trainable_by_id = {
        id(parameter): name
        for name, parameter in named_parameters
        if parameter.requires_grad
    }
    optimizer_parameter_ids: list[int] = []
    optimizer_groups = []
    for group_index, group in enumerate(optimizer.param_groups):
        group_parameters = list(group.get("params", []))
        names = []
        for parameter in group_parameters:
            parameter_id = id(parameter)
            if parameter_id not in trainable_by_id:
                raise RuntimeError(
                    "Stage-B v25 optimizer contains a frozen or unknown parameter "
                    f"in group {group_index}"
                )
            optimizer_parameter_ids.append(parameter_id)
            names.append(trainable_by_id[parameter_id])
        group[_STAGE_B_V25_OPTIMIZER_PARAMETER_NAMES_KEY] = list(names)
        optimizer_groups.append(
            {
                "index": group_index,
                "parameter_names": names,
                "static_options": _stage_b_v25_optimizer_static_options(group),
            }
        )

    expected_ids = set(trainable_by_id)
    observed_ids = set(optimizer_parameter_ids)
    duplicate_ids = len(optimizer_parameter_ids) != len(observed_ids)
    if duplicate_ids or observed_ids != expected_ids:
        missing = [
            trainable_by_id[parameter_id]
            for parameter_id in expected_ids - observed_ids
        ]
        raise RuntimeError(
            "Stage-B v25 optimizer must cover every trainable parameter exactly "
            f"once; missing={missing[:20]}, duplicate={duplicate_ids}"
        )

    criterion_state = criterion.state_dict()
    scheduler_state = lr_scheduler.state_dict()
    scaler_state = scaler.state_dict()
    captured_rng_state = dict(rng_state or _capture_rng_state())
    layout = {
        "schema": STAGE_B_V25_TRAINING_STATE_LAYOUT_SCHEMA,
        "strict_resume": True,
        "model": {
            "class": _stage_b_v25_class_name(model),
            "ordered_state_schema": _stage_b_v25_ordered_state_schema(
                model.state_dict()
            ),
        },
        "criterion": {
            "class": _stage_b_v25_class_name(criterion),
            "ordered_state_schema": _stage_b_v25_ordered_state_schema(
                criterion_state
            ),
        },
        "trainable_parameters": trainable_parameters,
        "optimizer": {
            "class": _stage_b_v25_class_name(optimizer),
            "group_count": len(optimizer_groups),
            "groups": optimizer_groups,
        },
        "lr_scheduler": {
            "class": _stage_b_v25_class_name(lr_scheduler),
            "state_schema": _stage_b_v25_value_schema(scheduler_state),
            "initial_state_contract": _stage_b_v25_value_contract(
                scheduler_state
            ),
        },
        "scaler": {
            "class": _stage_b_v25_class_name(scaler),
            "state_schema": _stage_b_v25_value_schema(scaler_state),
            "initial_state_contract": _stage_b_v25_value_contract(scaler_state),
        },
        "rng_state": {
            "state_schema": _stage_b_v25_value_schema(captured_rng_state),
        },
        "epoch_rng_state": {
            "state_schema": _stage_b_v25_value_schema(captured_rng_state),
        },
    }
    layout["semantic_sha256"] = _stage_b_v25_semantic_sha256(layout)
    return layout


def _stage_b_v25_layout_file_bytes(layout: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            layout,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _read_stage_b_v25_training_state_layout(path: Path) -> dict[str, Any]:
    try:
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise RuntimeError(
            f"Stage-B v25 training-state sidecar is unavailable: {path}"
        ) from exc
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise RuntimeError(
            "Stage-B v25 training-state sidecar changed while being read"
        )
    try:
        layout = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Stage-B v25 training-state sidecar is not valid JSON"
        ) from exc
    if not isinstance(layout, dict):
        raise RuntimeError("Stage-B v25 training-state sidecar must be an object")
    if layout.get("schema") != STAGE_B_V25_TRAINING_STATE_LAYOUT_SCHEMA:
        raise RuntimeError("Stage-B v25 training-state sidecar schema drifted")
    semantic_sha256 = layout.get("semantic_sha256")
    if (
        not isinstance(semantic_sha256, str)
        or len(semantic_sha256) != 64
        or semantic_sha256 != _stage_b_v25_semantic_sha256(layout)
    ):
        raise RuntimeError(
            "Stage-B v25 training-state sidecar semantic digest is invalid"
        )
    if raw != _stage_b_v25_layout_file_bytes(layout):
        raise RuntimeError(
            "Stage-B v25 training-state sidecar bytes are not canonical"
        )
    return layout


def _write_stage_b_v25_training_state_layout(
    output_dir: Path,
    layout: Mapping[str, Any],
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / STAGE_B_V25_TRAINING_STATE_LAYOUT_FILENAME
    expected = _stage_b_v25_layout_file_bytes(layout)
    try:
        with path.open("xb") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        existing = _read_stage_b_v25_training_state_layout(path)
        if existing != dict(layout):
            raise RuntimeError(
                "refusing to overwrite a different Stage-B v25 "
                "training-state sidecar"
            )
    persisted = _read_stage_b_v25_training_state_layout(path)
    if persisted != dict(layout):
        raise RuntimeError("Stage-B v25 training-state sidecar write drifted")
    return path


def _stage_b_v25_require_state_schema(
    label: str,
    state: Any,
    expected_schema: Any,
) -> None:
    if not isinstance(state, Mapping):
        raise RuntimeError(f"Stage-B v25 checkpoint {label} state must be a mapping")
    observed_schema = _stage_b_v25_ordered_state_schema(state)
    if observed_schema == expected_schema:
        return
    expected_names = [
        entry.get("name") for entry in expected_schema
        if isinstance(entry, Mapping)
    ] if isinstance(expected_schema, list) else []
    observed_names = [entry["name"] for entry in observed_schema]
    missing = [name for name in expected_names if name not in observed_names]
    extra = [name for name in observed_names if name not in expected_names]
    raise RuntimeError(
        f"Stage-B v25 checkpoint {label} ordered schema drifted; "
        f"missing={missing[:20]}, extra={extra[:20]}"
    )


def _stage_b_v25_validate_checkpoint_optimizer(
    checkpoint_optimizer: Any,
    optimizer: torch.optim.Optimizer,
    layout: Mapping[str, Any],
) -> None:
    if not isinstance(checkpoint_optimizer, Mapping):
        raise RuntimeError("Stage-B v25 checkpoint optimizer must be a mapping")
    checkpoint_groups = checkpoint_optimizer.get("param_groups")
    checkpoint_state = checkpoint_optimizer.get("state")
    expected_groups = layout.get("groups")
    current_groups = optimizer.state_dict().get("param_groups")
    if (
        not isinstance(checkpoint_groups, list)
        or not isinstance(checkpoint_state, Mapping)
        or not isinstance(expected_groups, list)
        or not isinstance(current_groups, list)
        or len(checkpoint_groups) != len(expected_groups)
        or len(current_groups) != len(expected_groups)
    ):
        raise RuntimeError("Stage-B v25 checkpoint optimizer group count drifted")

    checkpoint_parameter_ids = []
    for index, (checkpoint_group, current_group, expected_group) in enumerate(
        zip(checkpoint_groups, current_groups, expected_groups)
    ):
        if not all(
            isinstance(group, Mapping)
            for group in (checkpoint_group, current_group, expected_group)
        ):
            raise RuntimeError(
                f"Stage-B v25 checkpoint optimizer group {index} is invalid"
            )
        expected_names = expected_group.get("parameter_names")
        if (
            checkpoint_group.get(_STAGE_B_V25_OPTIMIZER_PARAMETER_NAMES_KEY)
            != expected_names
            or current_group.get(_STAGE_B_V25_OPTIMIZER_PARAMETER_NAMES_KEY)
            != expected_names
        ):
            raise RuntimeError(
                f"Stage-B v25 checkpoint optimizer group {index} parameter "
                "name layout drifted"
            )
        checkpoint_ids = checkpoint_group.get("params")
        current_ids = current_group.get("params")
        if checkpoint_ids != current_ids or not isinstance(checkpoint_ids, list):
            raise RuntimeError(
                f"Stage-B v25 checkpoint optimizer group {index} parameter "
                "order drifted"
            )
        if (
            _stage_b_v25_optimizer_static_options(checkpoint_group)
            != expected_group.get("static_options")
        ):
            raise RuntimeError(
                f"Stage-B v25 checkpoint optimizer group {index} static "
                "options drifted"
            )
        checkpoint_parameter_ids.extend(checkpoint_ids)

    if (
        len(checkpoint_parameter_ids) != len(set(checkpoint_parameter_ids))
        or set(checkpoint_state) != set(checkpoint_parameter_ids)
        or any(
            not isinstance(checkpoint_state[parameter_id], Mapping)
            or not checkpoint_state[parameter_id]
            for parameter_id in checkpoint_parameter_ids
        )
    ):
        raise RuntimeError(
            "Stage-B v25 checkpoint optimizer state does not cover every "
            "trainable parameter exactly once"
        )


def _validate_stage_b_v25_resume_checkpoint(
    checkpoint: Any,
    *,
    checkpoint_path: Path,
    current_layout: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> str:
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("Stage-B v25 resume checkpoint must be a mapping")
    required_keys = {
        "model",
        "criterion",
        "optimizer",
        "lr_scheduler",
        "scaler",
        "epoch",
        "iteration",
        "optimizer_updates",
        "epoch_finished",
        "rng_state",
        "epoch_rng_state",
        "args",
    }
    missing_keys = sorted(required_keys - set(checkpoint))
    if missing_keys:
        raise RuntimeError(
            "Stage-B v25 resume checkpoint is incomplete; missing="
            f"{missing_keys}"
        )
    if not isinstance(checkpoint.get("args"), Mapping):
        raise RuntimeError("Stage-B v25 checkpoint args must be a plain mapping")
    if (
        type(checkpoint.get("epoch")) is not int
        or type(checkpoint.get("iteration")) is not int
        or type(checkpoint.get("optimizer_updates")) is not int
        or type(checkpoint.get("epoch_finished")) is not bool
    ):
        raise RuntimeError("Stage-B v25 checkpoint cursor metadata is invalid")

    sidecar_path = (
        Path(checkpoint_path).expanduser().resolve(strict=True).parent
        / STAGE_B_V25_TRAINING_STATE_LAYOUT_FILENAME
    )
    sidecar_before = _read_stage_b_v25_training_state_layout(sidecar_path)
    current_digest = current_layout.get("semantic_sha256")
    if (
        current_layout.get("schema") != STAGE_B_V25_TRAINING_STATE_LAYOUT_SCHEMA
        or current_digest != _stage_b_v25_semantic_sha256(current_layout)
        or sidecar_before != dict(current_layout)
    ):
        raise RuntimeError(
            "Stage-B v25 runtime training-state layout differs from its sidecar"
        )
    checkpoint_digest = checkpoint["args"].get(
        STAGE_B_V25_TRAINING_STATE_LAYOUT_ARG
    )
    if checkpoint_digest != current_digest:
        raise RuntimeError(
            "Stage-B v25 checkpoint training-state layout digest drifted"
        )

    _stage_b_v25_require_state_schema(
        "model",
        checkpoint.get("model"),
        sidecar_before["model"]["ordered_state_schema"],
    )
    _stage_b_v25_require_state_schema(
        "criterion",
        checkpoint.get("criterion"),
        sidecar_before["criterion"]["ordered_state_schema"],
    )
    _stage_b_v25_validate_checkpoint_optimizer(
        checkpoint.get("optimizer"),
        optimizer,
        sidecar_before["optimizer"],
    )
    if (
        _stage_b_v25_value_schema(checkpoint.get("lr_scheduler"))
        != sidecar_before["lr_scheduler"]["state_schema"]
    ):
        raise RuntimeError("Stage-B v25 checkpoint scheduler schema drifted")
    if (
        _stage_b_v25_value_schema(checkpoint.get("scaler"))
        != sidecar_before["scaler"]["state_schema"]
    ):
        raise RuntimeError("Stage-B v25 checkpoint scaler schema drifted")
    for rng_key in ("rng_state", "epoch_rng_state"):
        if not isinstance(checkpoint.get(rng_key), Mapping):
            raise RuntimeError(
                f"Stage-B v25 checkpoint {rng_key} state must be a mapping"
            )
        if (
            _stage_b_v25_value_schema(checkpoint[rng_key])
            != sidecar_before[rng_key]["state_schema"]
        ):
            raise RuntimeError(
                f"Stage-B v25 checkpoint {rng_key} schema drifted"
            )

    sidecar_after = _read_stage_b_v25_training_state_layout(sidecar_path)
    if sidecar_after != sidecar_before:
        raise RuntimeError(
            "Stage-B v25 training-state sidecar changed during resume validation"
        )
    return str(current_digest)


def _restore_stage_b_v25_rng_state(rng_state: Mapping[str, Any]) -> None:
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["cuda"])


def _install_signal_checkpoint_handlers(args):
    def _handler(signum, _frame):
        if getattr(args, "_stop_requested", False):
            raise KeyboardInterrupt(f"Received signal {signum} twice.")
        args._stop_requested = True
        args._stop_signal = int(signum)
        print(
            f"Received signal {signum}; will save checkpoint_iter.pth after the current optimizer-update boundary.",
            flush=True,
        )

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass


def _configure_torch_multiprocessing(args, logger):
    strategy = str(getattr(args, "mp_sharing_strategy", "file_system") or "none")
    if strategy == "none":
        return
    try:
        available = torch.multiprocessing.get_all_sharing_strategies()
        if strategy not in available:
            logger.warning(
                f"Requested mp_sharing_strategy={strategy!r}, but available strategies are {sorted(available)}."
            )
            return
        torch.multiprocessing.set_sharing_strategy(strategy)
        logger.info(f"torch multiprocessing sharing strategy: {torch.multiprocessing.get_sharing_strategy()}")
    except Exception as e:
        logger.warning(f"Failed to set torch multiprocessing sharing strategy to {strategy!r}: {e}")


def _get_nofile_limit():
    try:
        import resource

        return resource.getrlimit(resource.RLIMIT_NOFILE)
    except Exception:
        return None


def _raise_nofile_limit(args, logger):
    minimum = int(getattr(args, "min_nofile", 0) or 0)
    if minimum <= 0:
        return
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(max(int(soft), minimum), int(hard))
        if target > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            logger.info(f"Raised RLIMIT_NOFILE soft limit from {soft} to {target} (hard={hard}).")
        elif soft < minimum:
            logger.warning(
                f"RLIMIT_NOFILE soft/hard is {soft}/{hard}; cannot raise to requested minimum {minimum}."
            )
    except Exception as e:
        logger.warning(f"Failed to adjust RLIMIT_NOFILE: {e}")


def _resolve_dataloader_runtime(args):
    num_workers = int(args.num_workers)
    if num_workers < 0:
        raise ValueError(f"--num_workers must be >= 0, got {num_workers}.")
    prefetch_factor = int(getattr(args, "prefetch_factor", 1))
    if prefetch_factor < 1:
        raise ValueError(f"--prefetch_factor must be >= 1, got {prefetch_factor}.")
    pin_memory_arg = getattr(args, "pin_memory", None)
    pin_memory = str(args.device).startswith("cuda") if pin_memory_arg is None else bool(pin_memory_arg)
    persistent_arg = getattr(args, "persistent_workers", None)
    persistent_workers = num_workers > 0 if persistent_arg is None else bool(persistent_arg)
    return {
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }


def _stage_b_training_forward_batches_per_epoch(args, logical_batches: int) -> int:
    logical_batches = int(logical_batches)
    if logical_batches < 0:
        raise ValueError("logical DataLoader length must be non-negative")
    if not bool(getattr(args, "stage_b_dense_duty", False)):
        return logical_batches
    pack_factor = int(
        getattr(args, "stage_b_dense_duty_forward_pack_factor", 1) or 1
    )
    if pack_factor < 1:
        raise ValueError("dense-duty forward pack factor must be positive")
    return int(math.ceil(logical_batches / pack_factor))


def _trainable_param_summary(model: torch.nn.Module):
    trainable = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    by_module = {}
    for name, count in trainable.items():
        root = name.split(".", 1)[0]
        by_module[root] = by_module.get(root, 0) + int(count)
    return trainable, by_module


def main(args):
    

    utils.setup_distributed(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    # Some flags exist both in argparse and config; allow config to override a small safe subset.
    allow_cfg_override = {"fix_size", "persistent_workers"}
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        elif k in allow_cfg_override:
            setattr(args, k, v)
            if args.rank == 0:
                print(f"[WARN] Config overrides argparse key: {k}={v}")
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, 'debug', None):
        args.debug = False
    args.gradient_accumulation_steps = int(
        getattr(args, "gradient_accumulation_steps", 1) or 1
    )
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be >= 1")
    _bind_stage_b_native_patch_runtime_inputs(args)
    _bind_stage_b_data_driven_runtime_inputs(args)
    skip_eval = bool(getattr(args, "skip_eval", False)) and (not args.eval)
    if bool(getattr(args, "stage_b_dense_duty", False)):
        from util.stage_b_dense_duty_audit import (
            SOURCE_CLOSURE_ARG,
            build_source_closure,
            validate_formal_invocation,
        )

        validate_formal_invocation(args, repo_root=Path(__file__).resolve().parent)
        setattr(
            args,
            SOURCE_CLOSURE_ARG,
            build_source_closure(
                Path(args.config_file), repo_root=Path(__file__).resolve().parent
            ),
        )
    _bind_stage_b_confidence_probe_admission(args)
    _validate_stage_b_dense_duty_args(args)
    _validate_stage_b_v15_scorer_init_args(args)

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")

    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    _raise_nofile_limit(args, logger)
    _configure_torch_multiprocessing(args, logger)
    nofile_limit = _get_nofile_limit()
    if nofile_limit is not None:
        logger.info(f"RLIMIT_NOFILE soft/hard: {nofile_limit[0]}/{nofile_limit[1]}")
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))

    with open(args.datasets) as f:
        dataset_meta = json.load(f)
    if args.use_coco_eval and (args.eval or not skip_eval):
        args.coco_val_path = dataset_meta["val"][0]["anno"]

    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    logger.debug("build model ... ...")
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)
    if isinstance(criterion, torch.nn.Module):
        criterion.to(device)
    logger.debug("build model, done.")
    patch_only = bool(getattr(args, "patch_only", False))
    if patch_only and args.eval:
        raise ValueError("patch_only training does not support --eval (postprocessors/eval prompt are text-based).")


    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    logger.info("params before freezing:\n"+json.dumps({n: p.numel() for n, p in model_without_ddp.named_parameters() if p.requires_grad}, indent=2))

    # freeze some layers BEFORE building optimizer param groups
    if args.freeze_keywords is not None:
        for name, parameter in model_without_ddp.named_parameters():
            for keyword in args.freeze_keywords:
                if keyword in name:
                    parameter.requires_grad_(False)
                    break

    # Optional: unfreeze last N decoder layers (useful for patch-only adaptation).
    unfreeze_n = int(getattr(args, "unfreeze_decoder_last_n_layers", 0) or 0)
    if unfreeze_n <= 0 and bool(getattr(args, "unfreeze_decoder_last_layer", False)):
        # Backward compatibility with older configs.
        unfreeze_n = 1
    if unfreeze_n > 0:
        try:
            decoder = model_without_ddp.transformer.decoder
            layers = list(getattr(decoder, "layers", []))
            if not layers:
                raise RuntimeError("transformer.decoder.layers is empty or missing.")
            n = min(int(unfreeze_n), len(layers))
            for layer in layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad_(True)
            logger.info(f"unfreeze_decoder_last_n_layers={unfreeze_n}: transformer.decoder.layers[-{n}:] are trainable.")
        except Exception as e:
            logger.warning(f"unfreeze_decoder_last_n_layers={unfreeze_n} but failed to unfreeze decoder layers: {e}")

    only_train_keywords = getattr(args, "only_train_keywords", None)
    if only_train_keywords:
        if isinstance(only_train_keywords, str):
            only_train_keywords = [only_train_keywords]
        only_train_exclude_keywords = getattr(args, "only_train_exclude_keywords", None)
        if isinstance(only_train_exclude_keywords, str):
            only_train_exclude_keywords = [only_train_exclude_keywords]
        only_train_exclude_keywords = list(only_train_exclude_keywords or [])
        for _, parameter in model_without_ddp.named_parameters():
            parameter.requires_grad_(False)
        for name, parameter in model_without_ddp.named_parameters():
            if match_name_keywords(name, only_train_keywords) and not match_name_keywords(name, only_train_exclude_keywords):
                parameter.requires_grad_(True)

        unexpected = [
            name
            for name, parameter in model_without_ddp.named_parameters()
            if parameter.requires_grad
            and (
                not match_name_keywords(name, only_train_keywords)
                or match_name_keywords(name, only_train_exclude_keywords)
            )
        ]
        if unexpected:
            raise RuntimeError(f"Unexpected trainable parameters outside only_train_keywords: {unexpected[:20]}")

    trainable_params, trainable_modules = _trainable_param_summary(model_without_ddp)
    if bool(getattr(args, "stage_b_native_patch_category", False)):
        total_trainable = _freeze_and_audit_stage_b_native_patch_category(
            model_without_ddp
        )
        logger.info(
            "Stage-B native patch-category audit passed: frozen b58 full-text "
            "ranking and shared backbone, exactly eight trainable patch-projection "
            f"tensors ({total_trainable:,} parameters)."
        )
        trainable_params, trainable_modules = _trainable_param_summary(
            model_without_ddp
        )
    elif bool(getattr(args, "stage_b_data_driven_score", False)):
        data_driven_mode = _stage_b_data_driven_train_mode(
            getattr(args, "stage_b_data_driven_train_mode", "rank_patch_only")
        )
        total_trainable = _freeze_and_audit_stage_b_data_driven(
            model_without_ddp, data_driven_mode
        )
        logger.info(
            "Stage-B data-driven audit passed: frozen b58 with disjoint "
            f"rank-confidence score heads, train_mode={data_driven_mode}, "
            f"trainable parameters={total_trainable:,}."
        )
        trainable_params, trainable_modules = _trainable_param_summary(
            model_without_ddp
        )
    elif bool(getattr(args, "stage_b_u0_gate_aligned_d13", False)):
        total_trainable = _freeze_and_audit_stage_b_u0_gate_aligned_d13(
            model_without_ddp
        )
        logger.info(
            "Stage-B D13 audit passed: frozen b58/D9/R100/P50/U0, only the "
            f"patch-category residual is trainable ({total_trainable:,} parameters)."
        )
        trainable_params, trainable_modules = _trainable_param_summary(
            model_without_ddp
        )
    elif bool(getattr(args, "stage_b_u0_gate_aligned_d12", False)):
        total_trainable = _freeze_and_audit_stage_b_u0_gate_aligned_d12(
            model_without_ddp
        )
        logger.info(
            "Stage-B D12 audit passed: frozen b58/D9/R100/P50/U0, only the "
            f"conditional rank residual is trainable ({total_trainable:,} parameters)."
        )
        trainable_params, trainable_modules = _trainable_param_summary(
            model_without_ddp
        )
    elif bool(getattr(args, "stage_b_u0_gate_aligned_d11", False)):
        total_trainable = _freeze_and_audit_stage_b_u0_gate_aligned_d11(
            model_without_ddp
        )
        logger.info(
            "Stage-B D11 audit passed: frozen b58/D9/P50/U0 and R100 "
            "representation, exactly one trainable R100 rank-output weight "
            f"({total_trainable:,} parameters)."
        )
        trainable_params, trainable_modules = _trainable_param_summary(
            model_without_ddp
        )
    elif bool(getattr(args, "stage_b_u0_gate_aligned_d10", False)):
        total_trainable = _freeze_and_audit_stage_b_u0_gate_aligned_d10(
            model_without_ddp
        )
        logger.info(
            "Stage-B D10 audit passed: frozen b58/R100/P50/U0 residual, "
            "exactly eight trainable patch-projection tensors "
            f"({total_trainable:,} parameters)."
        )
        trainable_params, trainable_modules = _trainable_param_summary(
            model_without_ddp
        )
    elif bool(getattr(args, "stage_b_u0_patch_rank", False)):
        total_trainable = _freeze_and_audit_stage_b_u0_patch_rank(
            model_without_ddp
        )
        logger.info(
            "Stage-B U0 audit passed: frozen b58/R100/P50 and shared backbone, "
            f"patch-rank trainable parameters={total_trainable:,}."
        )
        trainable_params, trainable_modules = _trainable_param_summary(
            model_without_ddp
        )
    elif bool(getattr(args, "stage_b_gdino_score_adapter", False)):
        gdino_adapter_train_mode = _stage_b_gdino_adapter_train_mode(
            getattr(args, "stage_b_gdino_adapter_train_mode", "joint")
        )
        total_trainable = _freeze_and_audit_stage_b_gdino_adapter(
            model_without_ddp,
            train_mode=gdino_adapter_train_mode,
        )
        logger.info(
            "Stage-B pure-GDINO adapter audit passed: frozen deterministic base, "
            f"train_mode={gdino_adapter_train_mode}, "
            f"adapter-only trainable parameters={total_trainable:,}."
        )
        trainable_params, trainable_modules = _trainable_param_summary(
            model_without_ddp
        )
    elif bool(getattr(args, "stage_b_dense_duty", False)):
        phase = str(getattr(args, "stage_b_dense_duty_phase", "")).strip()
        minimum = int(getattr(args, "stage_b_v11_trainable_params_min", 1))
        maximum = int(
            getattr(args, "stage_b_v11_trainable_params_max", 100_000_000)
        )
        total_trainable = _freeze_and_audit_stage_b_dense_duty(
            model_without_ddp,
            phase=phase,
            minimum=minimum,
            maximum=maximum,
        )
        logger.info(
            "Dense-duty Stage B ownership audit passed: frozen Stage-A and "
            f"parameter-disjoint phase={phase}, trainable parameters="
            f"{total_trainable:,}."
        )
        trainable_params, trainable_modules = _trainable_param_summary(
            model_without_ddp
        )
    elif bool(getattr(args, "stage_b_v11_fixed_text", False)):
        minimum = int(getattr(args, "stage_b_v11_trainable_params_min", 5_000_000))
        maximum = int(getattr(args, "stage_b_v11_trainable_params_max", 7_000_000))
        total_trainable = _audit_stage_b_v11_trainable_parameters(
            model_without_ddp, minimum=minimum, maximum=maximum
        )
        logger.info(
            "Stage B v11 trainable audit passed: only "
            f"stage_b_fixed_text_scorer ({total_trainable:,} parameters)."
        )
        trainable_params, trainable_modules = _trainable_param_summary(model_without_ddp)
    elif bool(getattr(args, "stage_b_v7", False)) and getattr(model_without_ddp, "stage_b_verifier", None) is not None:
        model_without_ddp.stage_b_verifier.freeze_bert()
        trainable_params, trainable_modules = _trainable_param_summary(model_without_ddp)
    logger.info("params after freezing:\n" + json.dumps(trainable_params, indent=2))
    logger.info("trainable module summary:\n" + json.dumps(trainable_modules, indent=2))
    if only_train_keywords and (not trainable_params):
        raise RuntimeError("No trainable parameters remain after applying only_train_keywords.")

    # DDP snapshots the trainable parameter set when it builds reducer buckets.
    # Complete every requires_grad decision and trainable-parameter audit first.
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=args.find_unused_params,
        )
        if not args.find_unused_params:
            model._set_static_graph()
        model_without_ddp = model.module

    if bool(getattr(args, "stage_b_native_patch_category", False)):
        param_dicts = _stage_b_native_patch_category_optimizer_groups(
            model_without_ddp,
            lr=float(getattr(args, "stage_b_native_patch_lr", args.lr)),
        )
    elif bool(getattr(args, "stage_b_data_driven_score", False)):
        param_dicts = _stage_b_data_driven_optimizer_groups(
            model_without_ddp,
            train_mode=str(
                getattr(
                    args,
                    "stage_b_data_driven_train_mode",
                    "rank_patch_only",
                )
            ),
            rank_lr=float(
                getattr(args, "stage_b_data_driven_rank_lr", args.lr)
            ),
            confidence_lr=float(
                getattr(args, "stage_b_data_driven_confidence_lr", args.lr)
            ),
            patch_lr=float(
                getattr(args, "stage_b_data_driven_patch_lr", args.lr)
            ),
        )
    elif bool(getattr(args, "stage_b_u0_gate_aligned_d13", False)):
        param_dicts = _stage_b_u0_gate_aligned_d13_optimizer_groups(
            model_without_ddp,
            lr=float(getattr(args, "stage_b_u0_d13_patch_lr", args.lr)),
        )
    elif bool(getattr(args, "stage_b_u0_gate_aligned_d12", False)):
        param_dicts = _stage_b_u0_gate_aligned_d12_optimizer_groups(
            model_without_ddp,
            lr=float(getattr(args, "stage_b_u0_d12_rank_lr", args.lr)),
        )
    elif bool(getattr(args, "stage_b_u0_gate_aligned_d11", False)):
        param_dicts = _stage_b_u0_gate_aligned_d11_optimizer_groups(
            model_without_ddp,
            lr=float(getattr(args, "stage_b_u0_d11_rank_lr", args.lr)),
        )
    elif bool(getattr(args, "stage_b_u0_gate_aligned_d10", False)):
        param_dicts = _stage_b_u0_gate_aligned_d10_optimizer_groups(
            model_without_ddp,
            lr=float(getattr(args, "stage_b_u0_d10_patch_lr", args.lr)),
        )
    elif bool(getattr(args, "stage_b_u0_patch_rank", False)):
        param_dicts = _stage_b_u0_patch_rank_optimizer_groups(
            model_without_ddp,
            residual_lr=float(
                getattr(args, "stage_b_u0_patch_rank_lr", args.lr)
            ),
            patch_projection_lr=float(
                getattr(args, "stage_b_u0_patch_projection_lr", args.lr)
            ),
            direct_patch_gain_lr=getattr(
                args, "stage_b_u1_direct_patch_gain_lr", None
            ),
        )
    elif bool(getattr(args, "stage_b_gdino_score_adapter", False)):
        param_dicts = _stage_b_gdino_adapter_optimizer_groups(
            model_without_ddp,
            rank_lr=float(getattr(args, "stage_b_gdino_rank_lr", args.lr)),
            gate_lr=float(getattr(args, "stage_b_gdino_gate_lr", args.lr)),
            train_mode=str(
                getattr(args, "stage_b_gdino_adapter_train_mode", "joint")
            ),
        )
    else:
        param_dicts = get_param_dict(args, model_without_ddp)
    rank_adaptation_last_n = int(
        getattr(
            args,
            "stage_b_dense_duty_confidence_rank_decoder_unfreeze_last_n",
            0,
        )
    )
    if rank_adaptation_last_n > 0:
        param_dicts = _isolate_stage_b_dense_duty_rank_adaptation_optimizer_group(
            param_dicts,
            model_without_ddp,
            adaptation_lr=float(
                getattr(
                    args,
                    "stage_b_dense_duty_confidence_rank_decoder_lr",
                    args.lr,
                )
            ),
        )
    validity_lr = getattr(args, "stage_b_v15_validity_lr", None)
    if validity_lr is not None:
        score_ownership = str(
            getattr(args, "stage_b_v22_score_ownership", "") or ""
        ).strip()
        if (
            not bool(getattr(args, "stage_b_v15_decoupled_confidence", False))
            and score_ownership != "shared_trunk_two_heads"
        ):
            raise RuntimeError(
                "stage_b_v15_validity_lr requires "
                "stage_b_v15_decoupled_confidence=True or "
                "stage_b_v22_score_ownership='shared_trunk_two_heads'"
            )
        param_dicts = _isolate_stage_b_v15_validity_optimizer_group(
            param_dicts,
            model_without_ddp,
            validity_lr=float(validity_lr),
        )
    if bool(getattr(args, "stage_b_v11_fixed_text", False)):
        maximum_group_lr = _audit_stage_b_v11_optimizer_group_lrs(
            param_dicts,
            base_lr=args.lr,
            validity_lr=validity_lr,
        )
        logger.info(
            "Stage B v11 optimizer LR audit passed: "
            f"max_group_lr={maximum_group_lr:g}, base_lr={float(args.lr):g}, "
            f"validity_lr={validity_lr}."
        )

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    logger.debug("build dataset ... ...")
    dataset_train_list = None
    train_mix_weights = None
    if not args.eval:
        num_of_dataset_train = len(dataset_meta["train"])
        if num_of_dataset_train == 1:
            dataset_train = build_dataset(image_set='train', args=args, datasetinfo=dataset_meta["train"][0])
        else:
            from torch.utils.data import ConcatDataset
            dataset_train_list = []
            train_mix_weights = []
            for idx in range(len(dataset_meta["train"])):
                datasetinfo = dataset_meta["train"][idx]
                dataset_train_list.append(build_dataset(image_set='train', args=args, datasetinfo=datasetinfo))
                train_mix_weights.append(float(datasetinfo.get("mix_weight", 1.0)))
            dataset_train = ConcatDataset(dataset_train_list)
        logger.debug("build dataset, done.")
        logger.debug(f'number of training dataset: {num_of_dataset_train}, samples: {len(dataset_train)}')

    dataset_val = None
    if (not patch_only) and (args.eval or not skip_eval):
        dataset_val = build_dataset(image_set='val', args=args, datasetinfo=dataset_meta["val"][0])

    data_driven_sampling_contract = str(
        getattr(args, "stage_b_data_driven_sampling_contract", "") or ""
    ).strip()
    if data_driven_sampling_contract not in {
        "",
        STAGE_B_DATA_DRIVEN_SAMPLING_CONTRACT,
    }:
        raise ValueError(
            "stage_b_data_driven_sampling_contract must be empty or "
            f"{STAGE_B_DATA_DRIVEN_SAMPLING_CONTRACT!r}"
        )
    fair_data_driven_sampling = bool(
        not args.eval
        and (
            getattr(args, "stage_b_data_driven_score", False)
            or getattr(args, "stage_b_native_patch_category", False)
        )
        and data_driven_sampling_contract
        == STAGE_B_DATA_DRIVEN_SAMPLING_CONTRACT
    )
    if data_driven_sampling_contract and not bool(
        getattr(args, "stage_b_data_driven_score", False)
        or getattr(args, "stage_b_native_patch_category", False)
    ):
        raise ValueError(
            "the deterministic data-driven sampling contract requires "
            "a data-driven or native patch-category training mode"
        )
    if fair_data_driven_sampling and args.distributed:
        raise ValueError(
            "deterministic_epoch_ledger_v1 currently supports only non-DDP "
            "training; use a matched one-process protocol"
        )
    data_driven_sampler_seed = None
    data_driven_loader_seed = None
    data_loader_train_generator = None
    if fair_data_driven_sampling:
        data_driven_sampler_seed = getattr(
            args, "stage_b_data_driven_sampler_seed", None
        )
        data_driven_loader_seed = getattr(
            args, "stage_b_data_driven_loader_seed", None
        )
        _stage_b_data_driven_epoch_seed(data_driven_sampler_seed, 0)
        _stage_b_data_driven_epoch_seed(data_driven_loader_seed, 0)
        data_loader_train_generator = torch.Generator()
        data_loader_train_generator.manual_seed(data_driven_loader_seed)
        logger.info(
            "Enabled Stage-B data-driven deterministic sampling: "
            f"contract={data_driven_sampling_contract}, "
            f"sampler_seed={data_driven_sampler_seed}, "
            f"loader_seed={data_driven_loader_seed}."
        )

    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False) if dataset_val is not None else None
        if not args.eval:
            has_dataset_sample_weights = (
                dataset_train_list is not None
                and any(getattr(ds, "sample_weights", None) is not None for ds in dataset_train_list)
            )
            has_explicit_mix_weights = train_mix_weights is not None and any(
                abs(w - 1.0) > 1e-12 for w in train_mix_weights
            )
            if train_mix_weights is not None and (has_explicit_mix_weights or has_dataset_sample_weights):
                sample_weights = []
                for ds, mix_weight in zip(dataset_train_list, train_mix_weights):
                    ds_len = max(1, len(ds))
                    if has_explicit_mix_weights:
                        base_weight = float(mix_weight) / float(ds_len)
                    else:
                        # Preserve DistributedSampler's length-proportional dataset mix when all mix weights are 1.
                        base_weight = 1.0
                    ds_sample_weights = getattr(ds, "sample_weights", None)
                    if ds_sample_weights is not None and len(ds_sample_weights) == len(ds):
                        sample_weights.extend([base_weight * float(w) for w in ds_sample_weights])
                    else:
                        sample_weights.extend([base_weight] * len(ds))
                sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)
                sampler_train = WeightedDistributedSampler(
                    weights=sample_weights,
                    num_replicas=args.world_size,
                    rank=args.rank,
                    replacement=True,
                    seed=args.seed,
                )
                expected = []
                total_mix = sum(train_mix_weights) if has_explicit_mix_weights else sum(len(ds) for ds in dataset_train_list)
                for idx, (ds, mix_weight) in enumerate(zip(dataset_train_list, train_mix_weights)):
                    expected_fraction = (
                        (float(mix_weight) / float(total_mix))
                        if has_explicit_mix_weights and total_mix > 0
                        else (float(len(ds)) / float(total_mix) if total_mix > 0 else 0.0)
                    )
                    expected.append(
                        {
                            "dataset_idx": idx,
                            "len": len(ds),
                            "mix_weight": float(mix_weight),
                            "expected_fraction": expected_fraction,
                            "tn_balance_stats": getattr(ds, "tn_balance_stats", None),
                            "native_patch_category_sampling_stats": getattr(
                                ds,
                                "native_patch_category_sampling_stats",
                                None,
                            ),
                        }
                    )
                logger.info(
                    "using distributed mix_weight weighted sampling:\n"
                    + json.dumps(expected, indent=2)
                )
            elif getattr(dataset_train, "sample_weights", None) is not None:
                sample_weights = torch.as_tensor(getattr(dataset_train, "sample_weights"), dtype=torch.double)
                sampler_train = WeightedDistributedSampler(
                    weights=sample_weights,
                    num_replicas=args.world_size,
                    rank=args.rank,
                    replacement=True,
                    seed=args.seed,
                )
                logger.info(
                    "using distributed dataset-level weighted sampling:\n"
                    + json.dumps(
                        {
                            "tn_balance_stats": getattr(
                                dataset_train, "tn_balance_stats", None
                            ),
                            "native_patch_category_sampling_stats": getattr(
                                dataset_train,
                                "native_patch_category_sampling_stats",
                                None,
                            ),
                        },
                        indent=2,
                    )
                )
            else:
                sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val) if dataset_val is not None else None
        if not args.eval:
            has_dataset_sample_weights = (
                dataset_train_list is not None
                and any(getattr(ds, "sample_weights", None) is not None for ds in dataset_train_list)
            )
            has_explicit_mix_weights = train_mix_weights is not None and any(
                abs(w - 1.0) > 1e-12 for w in train_mix_weights
            )
            if train_mix_weights is not None and (has_explicit_mix_weights or has_dataset_sample_weights):
                sample_weights = []
                for ds, mix_weight in zip(dataset_train_list, train_mix_weights):
                    ds_len = max(1, len(ds))
                    if has_explicit_mix_weights:
                        base_weight = float(mix_weight) / float(ds_len)
                    else:
                        # Preserve RandomSampler's length-proportional dataset mix when all mix weights are 1.
                        base_weight = 1.0
                    ds_sample_weights = getattr(ds, "sample_weights", None)
                    if ds_sample_weights is not None and len(ds_sample_weights) == len(ds):
                        sample_weights.extend([base_weight * float(w) for w in ds_sample_weights])
                    else:
                        sample_weights.extend([base_weight] * len(ds))
                sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)
                if fair_data_driven_sampling:
                    sampler_train = DeterministicEpochSampler(
                        len(dataset_train),
                        seed=data_driven_sampler_seed,
                        weights=sample_weights,
                        num_samples=len(dataset_train),
                        replacement=True,
                    )
                else:
                    sampler_train = torch.utils.data.WeightedRandomSampler(
                        weights=sample_weights,
                        num_samples=len(dataset_train),
                        replacement=True,
                    )
                expected = []
                total_mix = sum(train_mix_weights) if has_explicit_mix_weights else sum(len(ds) for ds in dataset_train_list)
                for idx, (ds, mix_weight) in enumerate(zip(dataset_train_list, train_mix_weights)):
                    expected_fraction = (
                        (float(mix_weight) / float(total_mix))
                        if has_explicit_mix_weights and total_mix > 0
                        else (float(len(ds)) / float(total_mix) if total_mix > 0 else 0.0)
                    )
                    expected.append(
                        {
                            "dataset_idx": idx,
                            "len": len(ds),
                            "mix_weight": float(mix_weight),
                            "expected_fraction": expected_fraction,
                            "tn_balance_stats": getattr(ds, "tn_balance_stats", None),
                            "native_patch_category_sampling_stats": getattr(
                                ds,
                                "native_patch_category_sampling_stats",
                                None,
                            ),
                        }
                    )
                logger.info("using mix_weight weighted sampling:\n" + json.dumps(expected, indent=2))
            elif getattr(dataset_train, "sample_weights", None) is not None:
                sample_weights = torch.as_tensor(getattr(dataset_train, "sample_weights"), dtype=torch.double)
                if fair_data_driven_sampling:
                    sampler_train = DeterministicEpochSampler(
                        len(dataset_train),
                        seed=data_driven_sampler_seed,
                        weights=sample_weights,
                        num_samples=len(dataset_train),
                        replacement=True,
                    )
                else:
                    sampler_train = torch.utils.data.WeightedRandomSampler(
                        weights=sample_weights,
                        num_samples=len(dataset_train),
                        replacement=True,
                    )
                logger.info(
                    "using dataset-level weighted sampling:\n"
                    + json.dumps(
                        {
                            "tn_balance_stats": getattr(
                                dataset_train, "tn_balance_stats", None
                            ),
                            "native_patch_category_sampling_stats": getattr(
                                dataset_train,
                                "native_patch_category_sampling_stats",
                                None,
                            ),
                        },
                        indent=2,
                    )
                )
            else:
                if fair_data_driven_sampling:
                    sampler_train = DeterministicEpochSampler(
                        len(dataset_train),
                        seed=data_driven_sampler_seed,
                        replacement=False,
                    )
                else:
                    sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if not args.eval:
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)
        def _worker_init_fn(_worker_id: int):
            # Avoid CPU thread oversubscription when using many DataLoader workers.
            try:
                torch.set_num_threads(1)
            except Exception:
                pass
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass

        dataloader_runtime = _resolve_dataloader_runtime(args)
        if fair_data_driven_sampling and dataloader_runtime["persistent_workers"]:
            raise ValueError(
                "deterministic_epoch_ledger_v1 requires persistent_workers=False "
                "so worker RNG streams can be reconstructed on mid-epoch resume"
            )
        logger.info("DataLoader runtime settings: " + json.dumps(dataloader_runtime, indent=2))
        dl_train_kwargs = dict(
            batch_sampler=batch_sampler_train,
            collate_fn=utils.collate_fn,
            num_workers=dataloader_runtime["num_workers"],
        )
        dl_train_kwargs["pin_memory"] = dataloader_runtime["pin_memory"]
        if fair_data_driven_sampling:
            dl_train_kwargs["generator"] = data_loader_train_generator
        if dataloader_runtime["num_workers"] > 0:
            dl_train_kwargs["persistent_workers"] = dataloader_runtime["persistent_workers"]
            dl_train_kwargs["prefetch_factor"] = dataloader_runtime["prefetch_factor"]
            dl_train_kwargs["worker_init_fn"] = _worker_init_fn
        data_loader_train = DataLoader(dataset_train, **dl_train_kwargs)
        if bool(getattr(args, "stage_b_dense_duty", False)) and int(
            getattr(args, "stage_b_dense_duty_forward_pack_factor", 1) or 1
        ) > 1:
            observed_logical_batches = len(data_loader_train)
            observed_physical_forwards = (
                _stage_b_training_forward_batches_per_epoch(
                    args, observed_logical_batches
                )
            )
            expected_logical_batches = int(
                getattr(
                    args,
                    "stage_b_dense_duty_expected_logical_batches_per_epoch",
                    0,
                )
            )
            expected_physical_forwards = int(
                getattr(
                    args,
                    "stage_b_dense_duty_expected_physical_forwards_per_epoch",
                    0,
                )
            )
            if (
                observed_logical_batches != expected_logical_batches
                or observed_physical_forwards != expected_physical_forwards
            ):
                raise RuntimeError(
                    "packed dense-duty DataLoader epoch geometry drifted: "
                    f"expected=({expected_logical_batches}, "
                    f"{expected_physical_forwards}), observed=("
                    f"{observed_logical_batches}, {observed_physical_forwards})"
                )

    data_loader_val = None
    if dataset_val is not None:
        dataloader_runtime = _resolve_dataloader_runtime(args)
        logger.info("Validation DataLoader runtime settings: " + json.dumps(dataloader_runtime, indent=2))
        dl_val_kwargs = dict(
            batch_size=4,
            sampler=sampler_val,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=dataloader_runtime["num_workers"],
        )
        dl_val_kwargs["pin_memory"] = dataloader_runtime["pin_memory"]
        if dataloader_runtime["num_workers"] > 0:
            dl_val_kwargs["persistent_workers"] = dataloader_runtime["persistent_workers"]
            dl_val_kwargs["prefetch_factor"] = dataloader_runtime["prefetch_factor"]
            dl_val_kwargs["worker_init_fn"] = _worker_init_fn
        data_loader_val = DataLoader(dataset_val, **dl_val_kwargs)

    if args.onecyclelr:
        optimizer_steps_per_epoch = math.ceil(
            _stage_b_training_forward_batches_per_epoch(
                args, len(data_loader_train)
            )
            / args.gradient_accumulation_steps
        )
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            steps_per_epoch=optimizer_steps_per_epoch,
            epochs=args.epochs,
            pct_start=0.2,
        )
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    training_forward_batches_per_epoch = (
        _stage_b_training_forward_batches_per_epoch(
            args, len(data_loader_train)
        )
    )

    scaler = _make_grad_scaler(
        enabled=args.amp,
        init_scale=getattr(args, "amp_init_scale", None),
    )
    resume_iter = 0
    resume_optimizer_updates = 0
    resume_epoch_rng_state = None
    resume_runtime_rng_state = None

    base_ds = get_coco_api_from_dataset(dataset_val) if dataset_val is not None else None

    if args.frozen_weights is not None:
        checkpoint = _torch_load_compat(args.frozen_weights, map_location="cpu")
        model_without_ddp.detr.load_state_dict(clean_state_dict(checkpoint['model']),strict=False)

    output_dir = Path(args.output_dir)
    stage_b_v25_strict_resume = bool(
        getattr(args, "stage_b_v25_strict_resume", False)
    )
    stage_b_native_patch_strict_resume = bool(
        getattr(args, "stage_b_native_patch_category", False)
        and str(
            getattr(args, "stage_b_native_patch_execution_scope", "") or ""
        ).strip()
        in {
            _STAGE_B_NATIVE_PATCH_D2_U500_SCOPE,
            _STAGE_B_NATIVE_PATCH_D3_U200_SCOPE,
            _STAGE_B_NATIVE_PATCH_D4_U200_SCOPE,
            _STAGE_B_NATIVE_PATCH_D5_U100_SCOPE,
            _STAGE_B_NATIVE_PATCH_D6_U100_SCOPE,
            _STAGE_B_NATIVE_PATCH_D7_U100_SCOPE,
            _STAGE_B_NATIVE_PATCH_D8_U100_SCOPE,
            _STAGE_B_NATIVE_PATCH_D9_U100_SCOPE,
        }
        and str(getattr(args, "resume", "") or "").strip()
        and not bool(args.eval)
    )
    stage_b_data_driven_strict_resume = bool(
        getattr(args, "stage_b_data_driven_score", False)
    ) and not bool(args.eval)
    stage_b_data_driven_strict_resume = (
        stage_b_data_driven_strict_resume
        or stage_b_native_patch_strict_resume
    )
    stage_b_dense_duty_strict_resume = bool(
        getattr(args, "stage_b_dense_duty", False)
        and str(getattr(args, "resume", "") or "").strip()
        and not bool(args.eval)
    )
    stage_b_v25_training_state_layout = None
    if stage_b_v25_strict_resume:
        stage_b_v25_training_state_layout = (
            _build_stage_b_v25_training_state_layout(
                model_without_ddp,
                criterion,
                optimizer,
                lr_scheduler,
                scaler,
            )
        )
        if not args.resume:
            if utils.is_main_process():
                _write_stage_b_v25_training_state_layout(
                    output_dir,
                    stage_b_v25_training_state_layout,
                )
            if utils.is_dist_avail_and_initialized():
                torch.distributed.barrier()
            persisted_layout = _read_stage_b_v25_training_state_layout(
                output_dir / STAGE_B_V25_TRAINING_STATE_LAYOUT_FILENAME
            )
            if persisted_layout != stage_b_v25_training_state_layout:
                raise RuntimeError(
                    "Stage-B v25 persisted training-state layout drifted"
                )
            setattr(
                args,
                STAGE_B_V25_TRAINING_STATE_LAYOUT_ARG,
                stage_b_v25_training_state_layout["semantic_sha256"],
            )
            logger.info(
                "Sealed Stage-B v25 training-state layout: "
                f"{persisted_layout['semantic_sha256']}"
            )
    auto_resume_checkpoint = output_dir / 'checkpoint.pth'
    if (not args.resume) and auto_resume_checkpoint.exists():
        logger.info(
            f"Found existing checkpoint at {auto_resume_checkpoint}, but auto-resume is disabled. "
            "Pass --resume explicitly to restore model/optimizer/scheduler from it."
        )
    if args.resume:
        logger.info(f"Loading resume checkpoint from {args.resume}")
        if stage_b_dense_duty_strict_resume and args.resume.startswith('https'):
            raise RuntimeError(
                "dense-duty strict resume requires the local atomic "
                "checkpoint_iter.pth snapshot"
            )
        if stage_b_v25_strict_resume and args.resume.startswith('https'):
            raise RuntimeError(
                "Stage-B v25 strict resume requires a local checkpoint with "
                "its adjacent training_state_layout.json sidecar"
            )
        dense_resume_path = None
        if stage_b_dense_duty_strict_resume:
            from util.stage_b_dense_duty_audit import (
                validate_strict_resume_checkpoint_path,
            )

            dense_resume_path = validate_strict_resume_checkpoint_path(
                args, Path(args.resume)
            )
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = _torch_load_compat(args.resume, map_location="cpu")
        if stage_b_dense_duty_strict_resume:
            from util.stage_b_dense_duty_audit import (
                validate_strict_resume_checkpoint_payload,
            )

            dense_resume_audit = validate_strict_resume_checkpoint_payload(
                checkpoint,
                args,
                checkpoint_path=dense_resume_path,
            )
            if dense_resume_audit["rank_handoff"] is not None:
                handoff_field = (
                    "stage_b_dense_duty_confidence_adapter_migration_audit"
                    if str(
                        getattr(args, "stage_b_v22_score_ownership", "")
                    ).strip()
                    == "rank_tower_stopgrad_token_adapter_two_phase"
                    else "stage_b_dense_duty_rank_source_checkpoint_audit"
                )
                setattr(
                    args,
                    handoff_field,
                    dense_resume_audit["rank_handoff"],
                )
            logger.info(
                "Validated dense-duty strict same-phase resume: "
                f"phase={dense_resume_audit['phase']}, "
                f"epoch={dense_resume_audit['epoch']}, "
                f"iteration={dense_resume_audit['iteration']}, "
                f"optimizer_updates={dense_resume_audit['optimizer_updates']}, "
                f"reason={dense_resume_audit['checkpoint_reason']}"
            )
        if stage_b_native_patch_strict_resume:
            _validate_stage_b_native_patch_d2_resume_checkpoint(args, checkpoint)
        if bool(getattr(args, "stage_b_data_driven_score", False)):
            _validate_stage_b_data_driven_eval_update_gate(
                args,
                checkpoint,
                checkpoint_label=f"Stage-B data-driven --resume {args.resume}",
            )
        if stage_b_v25_strict_resume:
            if stage_b_v25_training_state_layout is None:
                raise AssertionError("Stage-B v25 runtime layout was not built")
            layout_digest = _validate_stage_b_v25_resume_checkpoint(
                checkpoint,
                checkpoint_path=Path(args.resume),
                current_layout=stage_b_v25_training_state_layout,
                optimizer=optimizer,
            )
            setattr(args, STAGE_B_V25_TRAINING_STATE_LAYOUT_ARG, layout_digest)
            logger.info(
                "Validated Stage-B v25 strict resume training-state layout: "
                f"{layout_digest}"
            )
        if bool(getattr(args, "stage_b_v15_decoupled_confidence", False)) or bool(
            str(getattr(args, "stage_b_v22_score_ownership", "") or "").strip()
        ):
            _restore_stage_b_v15_scorer_init_audit_for_resume(
                model_without_ddp,
                args,
                checkpoint,
                logger,
            )
        resume_state = (
            checkpoint['model']
            if stage_b_v25_strict_resume or stage_b_dense_duty_strict_resume
            else clean_state_dict(checkpoint['model'])
        )
        if bool(getattr(args, "stage_b_data_driven_score", False)):
            from models.GroundingDINO.stage_b_data_driven_score import (
                validate_data_driven_trained_checkpoint_payload,
                validate_stage_b_data_driven_score_checkpoint,
            )

            validate_stage_b_data_driven_score_checkpoint(
                model_without_ddp,
                resume_state,
                checkpoint_label=(
                    f"Stage-B data-driven --resume checkpoint {args.resume}"
                ),
            )
            if args.eval:
                experiment_id = str(
                    getattr(args, "stage_b_data_driven_experiment_id", "")
                )
                validate_data_driven_trained_checkpoint_payload(
                    model_without_ddp,
                    checkpoint,
                    checkpoint_label=(
                        f"Stage-B data-driven --resume checkpoint {args.resume}"
                    ),
                    expected_experiment_id=experiment_id,
                    expected_confidence_trained=bool(
                        getattr(
                            args,
                            "stage_b_data_driven_confidence_trained",
                            False,
                        )
                    ),
                    expected_variant_id=(
                        str(
                            getattr(
                                args,
                                "stage_b_data_driven_variant_id",
                                "",
                            )
                        ).strip()
                        or None
                    ),
                    expected_rank_supervision=str(
                        getattr(
                            args,
                            "stage_b_data_driven_rank_supervision",
                            "all_nonpositive_negative_v1",
                        )
                    ),
                    expected_rank_negative_iou_threshold=float(
                        getattr(
                            args,
                            "stage_b_data_driven_rank_negative_iou_threshold",
                            0.3,
                        )
                    ),
                    expected_assignment_weight=(
                        float(
                            getattr(
                                args,
                                "stage_b_data_driven_assignment_weight",
                            )
                        )
                        if hasattr(
                            args, "stage_b_data_driven_assignment_weight"
                        )
                        else None
                    ),
                    expected_deployment_weight=(
                        float(
                            getattr(
                                args,
                                "stage_b_data_driven_deployment_weight",
                            )
                        )
                        if hasattr(
                            args, "stage_b_data_driven_deployment_weight"
                        )
                        else None
                    ),
                    expected_token_weight=(
                        float(
                            getattr(
                                args, "stage_b_data_driven_token_weight", 0.0
                            )
                        )
                        if experiment_id in {"DD2", "DD3"}
                        else None
                    ),
                    expected_confidence_initializer_sha256=(
                        str(
                            getattr(
                                args,
                                "stage_b_data_driven_confidence_initializer_sha256",
                                "",
                            )
                        )
                        if experiment_id in {"DD2", "DD3"}
                        else None
                    ),
                    expected_optimizer_updates=(
                        int(
                            getattr(
                                args,
                                "stage_b_data_driven_eval_expected_optimizer_updates",
                                0,
                            )
                        )
                        or None
                    ),
                    allow_legacy_criterion_contract=False,
                )
            saved_args = checkpoint.get("args")
            if not isinstance(saved_args, dict):
                raise RuntimeError(
                    "Stage-B data-driven resume requires its complete saved args"
                )
            resume_contract_keys = (
                "stage_b_data_driven_experiment_id",
                "stage_b_data_driven_variant_id",
                "stage_b_data_driven_train_mode",
                "stage_b_data_driven_category_complete",
                "stage_b_data_driven_confidence_trained",
                "stage_b_data_driven_rank_supervision",
                "stage_b_data_driven_strict_sample_identity",
                "stage_b_data_driven_control_checkpoint_path",
                "stage_b_data_driven_control_checkpoint_sha256",
                "stage_b_data_driven_control_resolved_args_path",
                "stage_b_data_driven_control_resolved_args_sha256",
                "stage_b_data_driven_control_rank_summary_path",
                "stage_b_data_driven_control_rank_summary_sha256",
                "stage_b_data_driven_control_gap3_summary_path",
                "stage_b_data_driven_control_gap3_summary_sha256",
                "stage_b_data_driven_control_source_snapshot_path",
                "stage_b_data_driven_control_source_snapshot_sha256",
                "stage_b_data_driven_control_source_snapshot_supplement_path",
                "stage_b_data_driven_control_source_snapshot_supplement_sha256",
                "stage_b_data_driven_control_source_snapshot_supplement_receipt_path",
                "stage_b_data_driven_control_source_snapshot_supplement_receipt_sha256",
                "stage_b_data_driven_control_evidence",
                "stage_b_data_driven_execution_scope",
                "stage_b_data_driven_formal_fresh_start",
                "stage_b_data_driven_formal_expected_optimizer_updates",
                "stage_b_data_driven_formal_config_path",
                "stage_b_data_driven_formal_output_dir",
                "stage_b_data_driven_formal_preflight_path",
                "stage_b_data_driven_formal_preflight_sha256",
                "stage_b_data_driven_formal_probe_receipt_path",
                "stage_b_data_driven_formal_probe_receipt_sha256",
                "stage_b_data_driven_formal_gate_contract_path",
                "stage_b_data_driven_formal_gate_contract_sha256",
                "stage_b_data_driven_formal_evidence",
                "stage_b_data_driven_base_initializer_path",
                "stage_b_data_driven_base_initializer_sha256",
                "stage_b_data_driven_base_initializer",
                "stage_b_data_driven_initializer_pair_receipt_path",
                "stage_b_data_driven_initializer_pair_receipt_sha256",
                "stage_b_data_driven_initializer_pair_receipt",
                "stage_b_data_driven_confidence_initializer_scope",
                "stage_b_data_driven_confidence_initializer_min_dd1_updates",
                "stage_b_data_driven_confidence_initializer_sha256",
                "stage_b_data_driven_confidence_dataset_config_path",
                "stage_b_data_driven_confidence_dataset_config_sha256",
                "stage_b_data_driven_config_import_chain",
                "stage_b_data_driven_dataset_config",
                "stage_b_data_driven_training_provenance",
                "stage_b_data_driven_required_allocator_env",
                "stage_b_data_driven_required_allocator_conf",
                "stage_b_data_driven_rank_architecture",
                "stage_b_data_driven_rank_dim",
                "stage_b_data_driven_rank_num_heads",
                "stage_b_data_driven_rank_image_level_policy",
                "stage_b_data_driven_rank_image_levels",
                "stage_b_data_driven_rank_image_pool_size",
                "stage_b_data_driven_rank_image_pool_policy",
                "stage_b_data_driven_rank_box_fourier_bands",
                "stage_b_data_driven_rank_ffn_dim",
                "stage_b_data_driven_rank_dropout",
                "stage_b_data_driven_head_init_seed",
                "stage_b_data_driven_confidence_dim",
                "stage_b_data_driven_token_temperature",
                "stage_b_data_driven_gate_hidden_dim",
                "stage_b_data_driven_gate_pool_temperature",
                "stage_b_data_driven_gate_topk",
                "stage_b_data_driven_category_gate",
                "stage_b_data_driven_category_gate_max_gap",
                "stage_b_data_driven_category_gate_boundary_margin",
                "stage_b_data_driven_patch_active_unsafe_auxiliary_weight",
                "stage_b_data_driven_patch_dense_category_focal_weight",
                "stage_b_data_driven_patch_dense_category_focal_alpha",
                "stage_b_data_driven_patch_dense_category_focal_gamma",
                "stage_b_data_driven_patch_dense_category_focal_negative_weight",
                "stage_b_data_driven_patch_drop_positive_anchor_gradient_policy",
                "stage_b_data_driven_patch_residual",
                "stage_b_data_driven_patch_training_surface",
                "stage_b_data_driven_patch_residual_contract",
                "stage_b_data_driven_patch_residual_hidden_dim",
                "stage_b_data_driven_patch_residual_context_dim",
                "stage_b_data_driven_patch_residual_context_topk",
                "stage_b_data_driven_patch_residual_limit",
                "stage_b_data_driven_patch_residual_init_seed",
                "stage_b_data_driven_patch_residual_center_raw",
                "stage_b_data_driven_patch_residual_source_initializer_sha256",
                "stage_b_data_driven_patch_score_clip",
                "stage_b_data_driven_rank_weight",
                "stage_b_data_driven_patch_weight",
                "stage_b_data_driven_assignment_weight",
                "stage_b_data_driven_deployment_weight",
                "stage_b_data_driven_confidence_weight",
                "stage_b_data_driven_token_weight",
                "stage_b_data_driven_shared_token_weight",
                "stage_b_data_driven_positive_iou_threshold",
                "stage_b_data_driven_rank_negative_iou_threshold",
                "stage_b_data_driven_patch_negative_iou_threshold",
                "stage_b_data_driven_temperature",
                "stage_b_data_driven_rank_margin",
                "stage_b_data_driven_category_margin",
                "stage_b_data_driven_fpr_temperature",
                "stage_b_data_driven_fpr_margin",
                "stage_b_data_driven_target_tpr",
                "stage_b_data_driven_positive_queue_size",
                "stage_b_data_driven_rank_lr",
                "stage_b_data_driven_patch_lr",
                "stage_b_data_driven_confidence_lr",
                "stage_b_data_driven_sampling_contract",
                "stage_b_data_driven_sampler_seed",
                "stage_b_data_driven_loader_seed",
                "stage_b_data_driven_grad_clip_contract",
                "config_file",
                "datasets",
                "options",
                "seed",
                "batch_size",
                "num_workers",
                "prefetch_factor",
                "pin_memory",
                "persistent_workers",
                "gradient_accumulation_steps",
                "amp",
                "amp_init_scale",
                "world_size",
                "distributed",
                "find_unused_params",
                "weight_decay",
                "clip_max_norm",
                "epochs",
                "lr_drop",
                "onecyclelr",
                "multi_step_lr",
                "fix_size",
                "data_aug_hflip_prob",
                "max_train_iters",
                "pretrain_model_path",
            )
            drifted = {
                key: (saved_args.get(key), getattr(args, key, None))
                for key in resume_contract_keys
                if saved_args.get(key) != getattr(args, key, None)
            }
            if drifted:
                raise RuntimeError(
                    "Stage-B data-driven resume crossed an experiment contract: "
                    f"{drifted}"
                )
        elif bool(getattr(args, "stage_b_gdino_score_adapter", False)):
            from models.GroundingDINO.stage_b_gdino_score_adapter import (
                validate_stage_b_gdino_score_adapter_checkpoint,
            )

            validate_stage_b_gdino_score_adapter_checkpoint(
                model_without_ddp,
                resume_state,
                checkpoint_label=f"Stage-B GDINO adapter --resume checkpoint {args.resume}",
            )
            if bool(getattr(args, "stage_b_u0_patch_rank", False)):
                from models.GroundingDINO.stage_b_u0_patch_rank import (
                    validate_stage_b_u0_patch_rank_checkpoint,
                )

                validate_stage_b_u0_patch_rank_checkpoint(
                    model_without_ddp,
                    resume_state,
                    checkpoint_label=f"Stage-B U0 --resume checkpoint {args.resume}",
                )
        elif bool(getattr(args, "stage_b_v11_fixed_text", False)):
            from models.GroundingDINO.stage_b_fixed_text_scorer import (
                validate_stage_b_fixed_text_scorer_checkpoint,
            )

            validate_stage_b_fixed_text_scorer_checkpoint(
                model_without_ddp,
                resume_state,
                checkpoint_label=f"Stage B v11 --resume checkpoint {args.resume}",
            )
        load_output = model_without_ddp.load_state_dict(
            resume_state,
            strict=(
                stage_b_v25_strict_resume
                or stage_b_dense_duty_strict_resume
                or bool(getattr(args, "stage_b_u0_patch_rank", False))
                or bool(getattr(args, "stage_b_data_driven_score", False))
                or bool(getattr(args, "stage_b_native_patch_category", False))
            ),
        )
        if bool(getattr(args, "stage_b_v11_fixed_text", False)):
            _maybe_sync_stage_b_v11_scorer_from_decoder(
                model_without_ddp, resume_state, logger
            )
        elif bool(getattr(args, "stage_b_v7", False)):
            _maybe_sync_stage_b_v7_verifier_from_text_branch(model_without_ddp, resume_state, logger)
        logger.info(f"Loaded resume model state: {load_output}")
        criterion_state = checkpoint.get('criterion', None)
        requires_criterion_state = int(
            getattr(args, "stage_b_v14_tail_queue_size", 0) or 0
        ) > 0 or (
            bool(getattr(args, "stage_b_gdino_score_adapter", False))
            or bool(getattr(args, "stage_b_data_driven_score", False))
            or bool(getattr(args, "stage_b_native_patch_category", False))
        )
        if criterion_state is not None:
            criterion.load_state_dict(criterion_state, strict=True)
            logger.info("Restored criterion state from resume checkpoint.")
        elif requires_criterion_state:
            raise RuntimeError(
                "Stage-B --resume checkpoint is missing required criterion state; "
                "initialize a new phase with --pretrain_model_path or use a "
                "complete same-phase training checkpoint."
            )


        
        if stage_b_data_driven_strict_resume:
            required_training_keys = {
                "model",
                "criterion",
                "optimizer",
                "lr_scheduler",
                "scaler",
                "epoch",
                "iteration",
                "optimizer_updates",
                "epoch_finished",
                "rng_state",
                "epoch_rng_state",
                "args",
            }
            missing_training_keys = sorted(
                required_training_keys.difference(checkpoint)
            )
            if missing_training_keys:
                raise RuntimeError(
                    "data-driven strict resume is missing complete training state: "
                    f"{missing_training_keys}"
                )
            if not isinstance(checkpoint["rng_state"], Mapping) or not isinstance(
                checkpoint["epoch_rng_state"], Mapping
            ):
                raise RuntimeError(
                    "data-driven strict resume requires runtime and epoch RNG mappings"
                )
            for integer_key in ("epoch", "iteration", "optimizer_updates"):
                value = checkpoint[integer_key]
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise RuntimeError(
                        f"data-driven strict resume has invalid {integer_key}={value!r}"
                    )
            if type(checkpoint["epoch_finished"]) is not bool:
                raise RuntimeError(
                    "data-driven strict resume epoch_finished must be an exact bool"
                )
            if fair_data_driven_sampling:
                sampling_state = _validate_stage_b_data_driven_sampling_resume_state(
                    sampler_train,
                    checkpoint,
                    loader_seed=data_driven_loader_seed,
                )
                logger.info(
                    "Validated deterministic data-driven resume sampling ledger: "
                    f"epoch={sampling_state['epoch']}, "
                    f"sha256={sampling_state['ledger_sha256']}."
                )

        has_complete_training_state = (
            not args.eval
            and 'optimizer' in checkpoint
            and 'lr_scheduler' in checkpoint
            and 'epoch' in checkpoint
        )
        if has_complete_training_state:
            saved_args = checkpoint.get('args', {})
            saved_accumulation_steps = int(
                saved_args.get('gradient_accumulation_steps', 1)
                if isinstance(saved_args, Mapping)
                else 1
            )
            if saved_accumulation_steps != args.gradient_accumulation_steps:
                raise RuntimeError(
                    "Cannot resume with a different gradient accumulation contract: "
                    f"checkpoint={saved_accumulation_steps}, "
                    f"requested={args.gradient_accumulation_steps}."
                )
            if stage_b_v25_strict_resume:
                saved_amp = saved_args.get('amp')
                saved_batch_size = saved_args.get('batch_size')
                if type(saved_amp) is not bool or saved_amp != bool(args.amp):
                    raise RuntimeError(
                        "Stage-B v25 strict resume cannot change AMP mode"
                    )
                if (
                    type(saved_batch_size) is not int
                    or saved_batch_size != int(args.batch_size)
                ):
                    raise RuntimeError(
                        "Stage-B v25 strict resume cannot change physical batch size"
                    )
            if args.gradient_accumulation_steps > 1:
                saved_amp = bool(saved_args.get('amp', False))
                if saved_amp != bool(args.amp):
                    raise RuntimeError(
                        "Cannot change AMP mode when resuming accumulated training"
                    )
                saved_batch_size = int(saved_args.get('batch_size', -1))
                if saved_batch_size != int(args.batch_size):
                    raise RuntimeError(
                        "Cannot change physical batch size when resuming accumulated training"
                    )
                if args.amp and 'scaler' not in checkpoint:
                    raise RuntimeError(
                        "AMP accumulation resume checkpoint is missing GradScaler state"
                    )
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                restored_scaler = False
                if (
                    stage_b_v25_strict_resume
                    or stage_b_data_driven_strict_resume
                    or stage_b_dense_duty_strict_resume
                    or 'scaler' in checkpoint
                ):
                    scaler.load_state_dict(checkpoint['scaler'])
                    restored_scaler = True
                ckpt_epoch = int(checkpoint['epoch'])
                ckpt_iter = int(checkpoint.get('iteration', 0) or 0)
                ckpt_optimizer_updates = checkpoint.get('optimizer_updates', None)
                if ckpt_optimizer_updates is None:
                    if saved_accumulation_steps != 1:
                        raise RuntimeError(
                            "Accumulated-training checkpoint is missing optimizer_updates"
                        )
                    checkpoint_epoch_finished = bool(
                        checkpoint.get(
                            'epoch_finished', 'iteration' not in checkpoint
                        )
                    )
                    completed_epochs = ckpt_epoch + (
                        1 if checkpoint_epoch_finished else 0
                    )
                    ckpt_optimizer_updates = (
                        completed_epochs * training_forward_batches_per_epoch
                        + ckpt_iter
                    )
                ckpt_optimizer_updates = int(ckpt_optimizer_updates)
                epoch_finished = bool(checkpoint.get('epoch_finished', 'iteration' not in checkpoint))
                if (
                    stage_b_v25_strict_resume
                    or stage_b_data_driven_strict_resume
                    or stage_b_dense_duty_strict_resume
                ):
                    resume_epoch_rng_state = checkpoint['epoch_rng_state']
                    resume_runtime_rng_state = checkpoint['rng_state']
                logger.info(
                    "Restored resume training state: "
                    f"epoch={ckpt_epoch}, iteration={ckpt_iter}, "
                    f"optimizer_updates={ckpt_optimizer_updates}, "
                    f"epoch_finished={epoch_finished}, scaler_restored={restored_scaler}"
                )
                if (
                    stage_b_dense_duty_strict_resume
                    and not epoch_finished
                    and not (
                        0
                        < ckpt_iter
                        < training_forward_batches_per_epoch
                    )
                ):
                    raise RuntimeError(
                        "dense-duty strict resume iteration is outside the current "
                        "epoch DataLoader boundary"
                    )
                if (
                    (not epoch_finished)
                    and ckpt_iter > 0
                    and ckpt_iter < training_forward_batches_per_epoch
                ):
                    if (
                        args.gradient_accumulation_steps > 1
                        and (ckpt_iter % args.gradient_accumulation_steps) != 0
                    ):
                        raise RuntimeError(
                            "Accumulated-training checkpoint is not at an optimizer-step boundary"
                        )
                    args.start_epoch = ckpt_epoch
                    resume_iter = ckpt_iter
                    resume_optimizer_updates = ckpt_optimizer_updates
                    if not (
                        stage_b_v25_strict_resume
                        or stage_b_data_driven_strict_resume
                        or stage_b_dense_duty_strict_resume
                    ):
                        resume_epoch_rng_state = checkpoint.get('epoch_rng_state', None)
                        resume_runtime_rng_state = checkpoint.get('rng_state', None)
                    logger.info(
                        f"Resuming mid-epoch from epoch={ckpt_epoch}, "
                        f"next_iter={resume_iter}/"
                        f"{training_forward_batches_per_epoch}, "
                        f"optimizer_updates={resume_optimizer_updates}"
                    )
                else:
                    args.start_epoch = ckpt_epoch + 1
                    resume_optimizer_updates = ckpt_optimizer_updates
                    logger.info(
                        f"Resuming from next epoch: checkpoint_epoch={ckpt_epoch}, "
                        f"start_epoch={args.start_epoch}"
                    )
                    if (
                        stage_b_v25_strict_resume
                        or stage_b_data_driven_strict_resume
                        or stage_b_dense_duty_strict_resume
                    ):
                        _restore_stage_b_v25_rng_state(resume_runtime_rng_state)
            except Exception as e:
                if stage_b_v25_strict_resume or stage_b_dense_duty_strict_resume:
                    raise RuntimeError(
                        "strict resume could not restore the complete "
                        "optimizer/scheduler/scaler/RNG state"
                    ) from e
                if args.gradient_accumulation_steps != 1:
                    raise RuntimeError(
                        "Gradient-accumulation resume requires an intact optimizer-step "
                        "boundary checkpoint"
                    ) from e
                if bool(getattr(args, "stage_b_gdino_score_adapter", False)) or bool(
                    getattr(args, "stage_b_data_driven_score", False)
                ) or bool(
                    getattr(args, "stage_b_native_patch_category", False)
                ):
                    raise RuntimeError(
                        "Stage-B adapter/data-driven resume requires complete optimizer/"
                        "scheduler/scaler state"
                    ) from e
                logger.warning(
                    f"Failed to restore optimizer/scheduler state from resume checkpoint; "
                    f"continuing with fresh optimizer state. Error: {e}"
                )
        elif not args.eval:
            if stage_b_dense_duty_strict_resume:
                raise RuntimeError(
                    "dense-duty strict resume checkpoint must include complete "
                    "optimizer, scheduler, scaler, criterion, and RNG state"
                )
            if args.gradient_accumulation_steps != 1:
                raise RuntimeError(
                    "--resume with gradient accumulation requires complete optimizer, "
                    "scheduler, epoch, scaler, and optimizer-update state"
                )
            if bool(getattr(args, "stage_b_gdino_score_adapter", False)) or bool(
                getattr(args, "stage_b_data_driven_score", False)
            ) or bool(
                getattr(args, "stage_b_native_patch_category", False)
            ):
                raise RuntimeError(
                    "Stage-B adapter/data-driven resume checkpoint must include optimizer, "
                    "lr_scheduler, and epoch state"
                )
            logger.info(
                "Resume checkpoint did not include optimizer/lr_scheduler/epoch; "
                "loaded model weights only and will use fresh training state."
            )

    if (not args.resume) and args.pretrain_model_path:
        checkpoint_payload = _torch_load_compat(
            args.pretrain_model_path, map_location="cpu"
        )
        if bool(getattr(args, "stage_b_native_patch_category", False)):
            from tools.build_stageb_native_patch_category_initializer import (
                EXPECTED_B58_SHA256,
                _safe_load_checkpoint,
                extract_b58_source_state,
                validate_native_patch_category_initializer_payload,
            )

            checkpoint_label = (
                "Stage-B native patch-category --pretrain_model_path "
                f"{args.pretrain_model_path}"
            )
            expected_initializer_sha = str(
                getattr(args, "stage_b_native_patch_initializer_sha256", "")
                or ""
            ).strip()
            observed_initializer_sha = _sha256_file(
                Path(args.pretrain_model_path).resolve(strict=True)
            )
            if (
                len(expected_initializer_sha) != 64
                or observed_initializer_sha != expected_initializer_sha
            ):
                raise RuntimeError(
                    "native patch-category initializer SHA256 mismatch: "
                    f"expected={expected_initializer_sha!r}, "
                    f"observed={observed_initializer_sha}"
                )
            b58_path = Path(
                str(getattr(args, "stage_b_native_patch_b58_path", "") or "")
            ).resolve(strict=True)
            observed_b58_sha = _sha256_file(b58_path)
            if observed_b58_sha != EXPECTED_B58_SHA256:
                raise RuntimeError(
                    "native patch-category b58 source SHA256 mismatch: "
                    f"expected={EXPECTED_B58_SHA256}, observed={observed_b58_sha}"
                )
            b58_payload = _safe_load_checkpoint(
                b58_path, label="native patch-category b58 source"
            )
            b58_state = extract_b58_source_state(b58_payload)
            native_patch_objective = str(
                getattr(args, "stage_b_native_patch_objective", "d1_raw_margin")
                or ""
            ).strip().lower()
            if native_patch_objective == "d1_raw_margin":
                validate_native_patch_category_initializer_payload(
                    model_without_ddp,
                    checkpoint_payload,
                    checkpoint_label=checkpoint_label,
                    expected_b58_state=b58_state,
                )
            elif native_patch_objective in {
                "d2_gate_aligned",
                "d3_critical_winner",
                "d4_positive_protected_critical_winner",
                "d5_active_tail_positive_barrier",
                "d6_direct_deployment_gap",
                "d7_all_state_positive_anchor",
                "d8_state_class_macro_anchor",
                "d9_loss_gradient_localized",
            }:
                from tools.stageb_native_patch_category_d2_contract import (
                    audit_d2_source_transition,
                )

                d3_objective = native_patch_objective == "d3_critical_winner"
                d4_objective = (
                    native_patch_objective
                    == "d4_positive_protected_critical_winner"
                )
                d5_objective = (
                    native_patch_objective == "d5_active_tail_positive_barrier"
                )
                d6_objective = (
                    native_patch_objective == "d6_direct_deployment_gap"
                )
                d7_objective = (
                    native_patch_objective == "d7_all_state_positive_anchor"
                )
                d8_objective = (
                    native_patch_objective == "d8_state_class_macro_anchor"
                )
                d9_objective = (
                    native_patch_objective == "d9_loss_gradient_localized"
                )
                continuation_label = (
                    "D9"
                    if d9_objective
                    else "D8"
                    if d8_objective
                    else "D7"
                    if d7_objective
                    else "D6"
                    if d6_objective
                    else "D5"
                    if d5_objective
                    else "D4"
                    if d4_objective
                    else "D3"
                    if d3_objective
                    else "D2"
                )
                base_path_key = (
                    "stage_b_native_patch_d9_base_initializer_path"
                    if d9_objective
                    else "stage_b_native_patch_d8_base_initializer_path"
                    if d8_objective
                    else "stage_b_native_patch_d7_base_initializer_path"
                    if d7_objective
                    else "stage_b_native_patch_d6_base_initializer_path"
                    if d6_objective
                    else "stage_b_native_patch_d5_base_initializer_path"
                    if d5_objective
                    else "stage_b_native_patch_d4_base_initializer_path"
                    if d4_objective
                    else "stage_b_native_patch_d3_base_initializer_path"
                    if d3_objective
                    else "stage_b_native_patch_d2_base_initializer_path"
                )
                base_sha_key = (
                    "stage_b_native_patch_d9_base_initializer_sha256"
                    if d9_objective
                    else "stage_b_native_patch_d8_base_initializer_sha256"
                    if d8_objective
                    else "stage_b_native_patch_d7_base_initializer_sha256"
                    if d7_objective
                    else "stage_b_native_patch_d6_base_initializer_sha256"
                    if d6_objective
                    else "stage_b_native_patch_d5_base_initializer_sha256"
                    if d5_objective
                    else "stage_b_native_patch_d4_base_initializer_sha256"
                    if d4_objective
                    else "stage_b_native_patch_d3_base_initializer_sha256"
                    if d3_objective
                    else "stage_b_native_patch_d2_base_initializer_sha256"
                )
                continuation_base_initializer_path = Path(
                    str(
                        getattr(args, base_path_key, "")
                        or ""
                    )
                ).resolve(strict=True)
                expected_base_initializer_sha = str(
                    getattr(args, base_sha_key, "")
                    or ""
                ).strip()
                observed_base_initializer_sha = _sha256_file(
                    continuation_base_initializer_path
                )
                if (
                    len(expected_base_initializer_sha) != 64
                    or observed_base_initializer_sha
                    != expected_base_initializer_sha
                ):
                    raise RuntimeError(
                        f"native patch-category {continuation_label} base "
                        "initializer SHA256 mismatch"
                    )
                continuation_base_initializer_payload = _safe_load_checkpoint(
                    continuation_base_initializer_path,
                    label=(
                        f"native patch-category {continuation_label} "
                        "b58-only initializer"
                    ),
                )
                validate_native_patch_category_initializer_payload(
                    model_without_ddp,
                    continuation_base_initializer_payload,
                    checkpoint_label=(
                        f"native patch-category {continuation_label} "
                        "b58-only initializer"
                    ),
                    expected_b58_state=b58_state,
                )
                source_audit = audit_d2_source_transition(
                    model_without_ddp,
                    continuation_base_initializer_payload,
                    checkpoint_payload,
                    expected_optimizer_updates=500,
                )
                if (
                    d3_objective
                    or d4_objective
                    or d5_objective
                    or d6_objective
                    or d7_objective
                    or d8_objective
                    or d9_objective
                ):
                    source_audit = dict(source_audit)
                    source_audit.update(
                        {
                            "schema": (
                                "pivot.stageb.native_patch_category_d9_source_audit/v1"
                                if d9_objective
                                else "pivot.stageb.native_patch_category_d8_source_audit/v1"
                                if d8_objective
                                else "pivot.stageb.native_patch_category_d7_source_audit/v1"
                                if d7_objective
                                else "pivot.stageb.native_patch_category_d6_source_audit/v1"
                                if d6_objective
                                else "pivot.stageb.native_patch_category_d5_source_audit/v1"
                                if d5_objective
                                else "pivot.stageb.native_patch_category_d4_source_audit/v1"
                                if d4_objective
                                else "pivot.stageb.native_patch_category_d3_source_audit/v1"
                            ),
                            "objective": native_patch_objective,
                            "source_scope": _STAGE_B_NATIVE_PATCH_D1_U500_SCOPE,
                            "source_validator": "audit_d2_source_transition",
                        }
                    )
                    if d9_objective:
                        args.stage_b_native_patch_d9_source_audit = source_audit
                    elif d8_objective:
                        args.stage_b_native_patch_d8_source_audit = source_audit
                    elif d7_objective:
                        args.stage_b_native_patch_d7_source_audit = source_audit
                    elif d6_objective:
                        args.stage_b_native_patch_d6_source_audit = source_audit
                    elif d5_objective:
                        args.stage_b_native_patch_d5_source_audit = source_audit
                    elif d4_objective:
                        args.stage_b_native_patch_d4_source_audit = source_audit
                    else:
                        args.stage_b_native_patch_d3_source_audit = source_audit
                else:
                    args.stage_b_native_patch_d2_source_audit = source_audit
                del continuation_base_initializer_payload
            else:
                raise RuntimeError(
                    "unknown native patch-category training objective: "
                    f"{native_patch_objective!r}"
                )
            del b58_state, b58_payload
        elif bool(getattr(args, "stage_b_data_driven_score", False)):
            _validate_stage_b_data_driven_eval_update_gate(
                args,
                checkpoint_payload,
                checkpoint_label=(
                    "Stage-B data-driven --pretrain_model_path "
                    f"{args.pretrain_model_path}"
                ),
            )
            from models.GroundingDINO.stage_b_data_driven_score import (
                normalize_data_driven_rank_architecture,
                normalize_data_driven_train_mode,
                validate_data_driven_confidence_initializer_payload,
                validate_data_driven_initializer_payload,
                validate_data_driven_relational_initializer_payload,
                validate_data_driven_role_routed_initializer_payload,
            )

            checkpoint_label = (
                "Stage-B data-driven --pretrain_model_path "
                f"{args.pretrain_model_path}"
            )
            data_driven_mode = normalize_data_driven_train_mode(
                getattr(
                    args, "stage_b_data_driven_train_mode", "rank_patch_only"
                )
            )
            if data_driven_mode == "rank_patch_only":
                expected_base_sha = str(
                    getattr(
                        args,
                        "stage_b_data_driven_base_initializer_sha256",
                        "",
                    )
                    or ""
                ).strip()
                observed_base_sha = _sha256_file(
                    Path(args.pretrain_model_path).resolve(strict=True)
                )
                if (
                    len(expected_base_sha) != 64
                    or any(
                        char not in "0123456789abcdef"
                        for char in expected_base_sha
                    )
                    or observed_base_sha != expected_base_sha
                ):
                    raise RuntimeError(
                        "data-driven DD0/DD1 base initializer SHA256 mismatch: "
                        f"expected={expected_base_sha!r}, "
                        f"observed={observed_base_sha}"
                    )
                rank_architecture = normalize_data_driven_rank_architecture(
                    getattr(
                        args,
                        "stage_b_data_driven_rank_architecture",
                        "absolute_token",
                    )
                )
                variant_id = str(
                    getattr(args, "stage_b_data_driven_variant_id", "") or ""
                ).strip()
                if variant_id == _STAGE_B_DATA_DRIVEN_ROLE_ROUTED_VARIANT:
                    validate_data_driven_role_routed_initializer_payload(
                        model_without_ddp,
                        checkpoint_payload,
                        checkpoint_label=checkpoint_label,
                        expected_source_checkpoint_sha256=str(
                            getattr(
                                args,
                                "stage_b_data_driven_role_initializer_source_checkpoint_sha256",
                                "",
                            )
                            or ""
                        ).strip(),
                        expected_a0_initializer_sha256=str(
                            getattr(
                                args,
                                "stage_b_data_driven_role_initializer_a0_sha256",
                                "",
                            )
                            or ""
                        ).strip(),
                        expected_source_optimizer_updates=int(
                            getattr(
                                args,
                                "stage_b_data_driven_role_initializer_source_optimizer_updates",
                                0,
                            )
                            or 0
                        ),
                    )
                else:
                    validator = (
                        validate_data_driven_relational_initializer_payload
                        if rank_architecture == "relational_v1"
                        else validate_data_driven_initializer_payload
                    )
                    validator(
                        model_without_ddp,
                        checkpoint_payload,
                        checkpoint_label=checkpoint_label,
                    )
            else:
                expected_scope = str(
                    getattr(
                        args,
                        "stage_b_data_driven_confidence_initializer_scope",
                        "",
                    )
                    or ""
                ).strip()
                expected_initializer_sha = str(
                    getattr(
                        args,
                        "stage_b_data_driven_confidence_initializer_sha256",
                        "",
                    )
                    or ""
                ).strip()
                minimum_source_updates = int(
                    getattr(
                        args,
                        "stage_b_data_driven_confidence_initializer_min_dd1_updates",
                        0,
                    )
                    or 0
                )
                if (
                    expected_scope not in {"smoke", "formal"}
                    or len(expected_initializer_sha) != 64
                    or any(
                        char not in "0123456789abcdef"
                        for char in expected_initializer_sha
                    )
                    or minimum_source_updates <= 0
                ):
                    raise RuntimeError(
                        "data-driven confidence phase requires an exact initializer "
                        "scope, SHA256, and positive DD1 update floor"
                    )
                observed_initializer_sha = _sha256_file(
                    Path(args.pretrain_model_path).resolve(strict=True)
                )
                if observed_initializer_sha != expected_initializer_sha:
                    raise RuntimeError(
                        "data-driven confidence initializer SHA256 mismatch: "
                        f"expected={expected_initializer_sha}, "
                        f"observed={observed_initializer_sha}"
                    )
                validate_data_driven_confidence_initializer_payload(
                    model_without_ddp,
                    checkpoint_payload,
                    checkpoint_label=checkpoint_label,
                    expected_scope=expected_scope,
                    minimum_source_optimizer_updates=minimum_source_updates,
                )
                _validate_stage_b_data_driven_confidence_handoff_provenance(
                    args,
                    checkpoint_payload["data_driven_confidence_initializer"],
                )
        elif bool(getattr(args, "stage_b_u0_patch_rank", False)):
            if bool(getattr(args, "stage_b_u0_gate_aligned_d10", False)) or bool(
                getattr(args, "stage_b_u0_gate_aligned_d11", False)
            ) or bool(
                getattr(args, "stage_b_u0_gate_aligned_d12", False)
            ) or bool(
                getattr(args, "stage_b_u0_gate_aligned_d13", False)
            ):
                from tools.build_stageb_data_only_composite import (
                    validate_data_only_composite_payload,
                )

                if bool(getattr(args, "stage_b_u0_gate_aligned_d13", False)):
                    initializer_name = "stage_b_u0_d13_initializer_sha256"
                    phase_name = "D13"
                elif bool(getattr(args, "stage_b_u0_gate_aligned_d12", False)):
                    initializer_name = "stage_b_u0_d12_initializer_sha256"
                    phase_name = "D12"
                elif bool(getattr(args, "stage_b_u0_gate_aligned_d11", False)):
                    initializer_name = "stage_b_u0_d11_initializer_sha256"
                    phase_name = "D11"
                else:
                    initializer_name = "stage_b_u0_d10_initializer_sha256"
                    phase_name = "D10"
                expected_sha = str(
                    getattr(args, initializer_name, "") or ""
                ).strip()
                observed_sha = _sha256_file(
                    Path(args.pretrain_model_path).resolve(strict=True)
                )
                if (
                    len(expected_sha) != 64
                    or any(char not in "0123456789abcdef" for char in expected_sha)
                    or observed_sha != expected_sha
                ):
                    raise RuntimeError(
                        f"{phase_name} data-only composite initializer SHA256 mismatch: "
                        f"expected={expected_sha!r}, observed={observed_sha}"
                    )
                composite_expected = model_without_ddp
                if phase_name in {"D12", "D13"}:
                    residual_prefix = (
                        "stage_b_u0_gate_aligned_rank_residual."
                        if phase_name == "D12"
                        else "stage_b_u0_gate_aligned_patch_residual."
                    )
                    composite_expected = {
                        key: value
                        for key, value in model_without_ddp.state_dict().items()
                        if not str(key).startswith(residual_prefix)
                    }
                validate_data_only_composite_payload(
                    composite_expected,
                    checkpoint_payload,
                    checkpoint_label=(
                        f"Stage-B {phase_name} --pretrain_model_path "
                        f"{args.pretrain_model_path}"
                    ),
                )
            elif bool(getattr(args, "stage_b_u1_direct_patch_skip", False)):
                from models.GroundingDINO.stage_b_u0_patch_rank import (
                    validate_stage_b_u1_direct_patch_initializer_payload,
                )

                validate_stage_b_u1_direct_patch_initializer_payload(
                    model_without_ddp,
                    checkpoint_payload,
                    checkpoint_label=(
                        "Stage-B U1 --pretrain_model_path "
                        f"{args.pretrain_model_path}"
                    ),
                )
            else:
                from models.GroundingDINO.stage_b_u0_patch_rank import (
                    validate_stage_b_u0_initializer_payload,
                )

                validate_stage_b_u0_initializer_payload(
                    model_without_ddp,
                    checkpoint_payload,
                    checkpoint_label=(
                        "Stage-B U0 --pretrain_model_path "
                        f"{args.pretrain_model_path}"
                    ),
                )
        checkpoint = checkpoint_payload["model"]
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        if (
            bool(getattr(args, "stage_b_u0_patch_rank", False))
            or bool(getattr(args, "stage_b_data_driven_score", False))
            or bool(getattr(args, "stage_b_native_patch_category", False))
        ) and _ignorekeywordlist:
            raise ValueError(
                "Stage-B U0/data-driven/native-patch initializers forbid finetune_ignore"
            )
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})
        _validate_stage_b_v15_stage_a_pretrain_state(
            args,
            _tmp_st,
            checkpoint_payload=checkpoint_payload,
        )
        dense_confidence_adapter_init = bool(
            getattr(args, "stage_b_dense_duty", False)
            and str(getattr(args, "stage_b_dense_duty_phase", "")).strip()
            == "confidence"
            and str(getattr(args, "stage_b_v22_score_ownership", "")).strip()
            == "rank_tower_stopgrad_token_adapter_two_phase"
        )
        if dense_confidence_adapter_init:
            from util.stage_b_confidence_adapter_migration import (
                migrate_legacy_rank_to_confidence_adapter,
                validate_confidence_adapter_migration_audit,
            )
            from util.stage_b_dense_duty_audit import write_json_atomic

            _tmp_st, migration_audit = migrate_legacy_rank_to_confidence_adapter(
                model_without_ddp,
                _tmp_st,
                checkpoint_label=(
                    "CVPR confidence-adapter rank source "
                    f"{args.pretrain_model_path}"
                ),
                source_checkpoint_sha256=str(
                    getattr(
                        args,
                        "stage_b_dense_duty_rank_source_checkpoint_sha256",
                    )
                ),
                source_optimizer_updates=int(checkpoint_payload["optimizer_updates"]),
                source_checkpoint_reason=str(checkpoint_payload["checkpoint_reason"]),
                expected_rank_sha256=str(
                    getattr(args, "stage_b_dense_duty_rank_source_rank_sha256")
                ),
                expected_transferred_sha256=str(
                    getattr(
                        args,
                        "stage_b_dense_duty_rank_source_transferred_sha256",
                    )
                ),
            )
            migration_audit = validate_confidence_adapter_migration_audit(
                migration_audit,
                source_checkpoint_sha256=str(
                    getattr(
                        args,
                        "stage_b_dense_duty_rank_source_checkpoint_sha256",
                    )
                ),
                source_optimizer_updates=int(checkpoint_payload["optimizer_updates"]),
                source_checkpoint_reason=str(checkpoint_payload["checkpoint_reason"]),
                rank_sha256=str(
                    getattr(args, "stage_b_dense_duty_rank_source_rank_sha256")
                ),
                transferred_sha256=str(
                    getattr(
                        args,
                        "stage_b_dense_duty_rank_source_transferred_sha256",
                    )
                ),
            )
            setattr(
                args,
                "stage_b_dense_duty_confidence_adapter_migration_audit",
                migration_audit,
            )
            if int(getattr(args, "rank", 0)) == 0 and getattr(
                args, "output_dir", ""
            ):
                write_json_atomic(
                    Path(args.output_dir)
                    / "stage_b_confidence_adapter_migration_audit.json",
                    migration_audit,
                )
            logger.info(
                "Applied strict U6551 rank-to-confidence-adapter migration:\n"
                + json.dumps(migration_audit, indent=2, sort_keys=True)
            )

        if bool(getattr(args, "stage_b_gdino_score_adapter", False)) and any(
            str(key).startswith("stage_b_gdino_score_adapter.")
            for key in _tmp_st
        ):
            from models.GroundingDINO.stage_b_gdino_score_adapter import (
                validate_stage_b_gdino_score_adapter_checkpoint,
            )

            validate_stage_b_gdino_score_adapter_checkpoint(
                model_without_ddp,
                _tmp_st,
                checkpoint_label=(
                    f"Stage-B GDINO adapter --pretrain_model_path {args.pretrain_model_path}"
                ),
            )
            if bool(getattr(args, "stage_b_u0_patch_rank", False)):
                from models.GroundingDINO.stage_b_u0_patch_rank import (
                    validate_stage_b_u0_patch_rank_checkpoint,
                )

                validate_stage_b_u0_patch_rank_checkpoint(
                    model_without_ddp,
                    _tmp_st,
                    checkpoint_label=(
                        f"Stage-B U0 --pretrain_model_path {args.pretrain_model_path}"
                    ),
                )
        elif (
            bool(getattr(args, "stage_b_v11_fixed_text", False))
            and bool(
                str(
                    getattr(args, "stage_b_v22_score_ownership", "") or ""
                ).strip()
            )
            and any(
                str(key).startswith("stage_b_fixed_text_scorer.")
                for key in _tmp_st
            )
        ):
            from models.GroundingDINO.stage_b_fixed_text_scorer import (
                validate_stage_b_fixed_text_scorer_checkpoint,
            )

            validate_stage_b_fixed_text_scorer_checkpoint(
                model_without_ddp,
                _tmp_st,
                checkpoint_label=(
                    "Stage-B fixed scorer --pretrain_model_path "
                    f"{args.pretrain_model_path}"
                ),
            )

        if bool(getattr(args, "stage_b_u0_gate_aligned_d12", False)):
            d12_prefix = "stage_b_u0_gate_aligned_rank_residual."
            if any(str(key).startswith(d12_prefix) for key in _tmp_st):
                raise RuntimeError(
                    "D12 must start from the sealed pre-residual composite"
                )
            runtime_state = model_without_ddp.state_dict()
            d12_state = {
                key: value.detach().clone()
                for key, value in runtime_state.items()
                if str(key).startswith(d12_prefix)
            }
            if not d12_state:
                raise RuntimeError("D12 runtime residual state is missing")
            output_weight = d12_state.get(
                d12_prefix + "output.weight"
            )
            if not torch.is_tensor(output_weight) or not torch.equal(
                output_weight, torch.zeros_like(output_weight)
            ):
                raise RuntimeError("D12 initializer output must be exactly zero")
            _tmp_st.update(d12_state)

        if bool(getattr(args, "stage_b_u0_gate_aligned_d13", False)):
            d13_prefix = "stage_b_u0_gate_aligned_patch_residual."
            if any(str(key).startswith(d13_prefix) for key in _tmp_st):
                raise RuntimeError(
                    "D13 must start from the sealed pre-residual composite"
                )
            runtime_state = model_without_ddp.state_dict()
            d13_state = {
                key: value.detach().clone()
                for key, value in runtime_state.items()
                if str(key).startswith(d13_prefix)
            }
            if not d13_state:
                raise RuntimeError("D13 runtime residual state is missing")
            output_weight = d13_state.get(d13_prefix + "output.weight")
            if not torch.is_tensor(output_weight) or not torch.equal(
                output_weight, torch.zeros_like(output_weight)
            ):
                raise RuntimeError("D13 initializer output must be exactly zero")
            _tmp_st.update(d13_state)

        _load_output = model_without_ddp.load_state_dict(
            _tmp_st,
            strict=(
                bool(getattr(args, "stage_b_u0_patch_rank", False))
                or bool(getattr(args, "stage_b_data_driven_score", False))
                or bool(getattr(args, "stage_b_native_patch_category", False))
                or dense_confidence_adapter_init
            ),
        )
        if bool(getattr(args, "stage_b_v11_fixed_text", False)):
            _maybe_sync_stage_b_v11_scorer_from_decoder(
                model_without_ddp, _tmp_st, logger
            )
            _apply_stage_b_v15_scorer_init(
                model_without_ddp,
                args,
                logger,
            )
        elif bool(getattr(args, "stage_b_v7", False)):
            _maybe_sync_stage_b_v7_verifier_from_text_branch(model_without_ddp, _tmp_st, logger)
        if bool(getattr(args, "stage_b_dense_duty", False)):
            logger.info(
                "Loaded dense-duty initializer state: "
                f"missing_keys={len(_load_output.missing_keys)}, "
                f"unexpected_keys={len(_load_output.unexpected_keys)}. "
                "Missing scorer keys are initialized by the separately audited "
                "dense text checkpoint."
            )
        else:
            logger.info(str(_load_output))

    _prepare_stage_b_dense_duty_state_fingerprint(
        model_without_ddp,
        args,
        logger,
        resume_checkpoint=(checkpoint if args.resume else None),
    )

 
    
    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return
    
 
    
    print("Start training")
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=False) if (not patch_only and not skip_eval) else None
    _install_signal_checkpoint_handlers(args)

    current_epoch_rng_state = None
    current_data_driven_sampling_state = None
    completed_optimizer_updates = resume_optimizer_updates
    coco_evaluator = None

    def _checkpoint_payload(
        epoch,
        *,
        iteration=0,
        optimizer_updates=0,
        epoch_finished=True,
        reason=None,
    ):
        payload = {
            'model': model_without_ddp.state_dict(),
            'criterion': criterion.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': int(epoch),
            'iteration': int(iteration),
            'optimizer_updates': int(optimizer_updates),
            'epoch_finished': bool(epoch_finished),
            'rng_state': _capture_rng_state(),
            'epoch_rng_state': current_epoch_rng_state,
            # Store as plain dict to stay compatible with `weights_only=True` safe loading.
            'args': vars(args),
        }
        if fair_data_driven_sampling:
            if (
                not isinstance(current_data_driven_sampling_state, Mapping)
                or current_data_driven_sampling_state.get("epoch") != int(epoch)
            ):
                raise RuntimeError(
                    "cannot checkpoint deterministic data-driven training without "
                    "the exact current-epoch sampling ledger"
                )
            payload["stage_b_data_driven_sampling_state"] = dict(
                current_data_driven_sampling_state
            )
        if reason is not None:
            payload['checkpoint_reason'] = str(reason)
        return payload

    def _save_iter_checkpoint(
        *,
        epoch,
        iteration,
        optimizer_updates,
        scaler=None,
        epoch_finished=False,
        reason=None,
    ):
        if not args.output_dir:
            return
        checkpoint_path = output_dir / 'checkpoint_iter.pth'
        _atomic_torch_save_on_master(
            _checkpoint_payload(
                epoch,
                iteration=iteration,
                optimizer_updates=optimizer_updates,
                epoch_finished=epoch_finished,
                reason=reason,
            ),
            checkpoint_path,
        )
        msg = (
            f"Saved iteration checkpoint to {checkpoint_path} "
            f"(epoch={epoch}, next_iter={iteration}, "
            f"optimizer_updates={optimizer_updates}, reason={reason})."
        )
        logger.info(msg) if args.save_log else print(msg, flush=True)

    try:
        for epoch in range(args.start_epoch, args.epochs):
            epoch_start_time = time.time()
            if args.distributed:
                sampler_train.set_epoch(epoch)
            if fair_data_driven_sampling:
                current_data_driven_sampling_state = (
                    _prepare_stage_b_data_driven_epoch_sampling(
                        sampler_train,
                        data_loader_train_generator,
                        epoch=epoch,
                        sampler_seed=data_driven_sampler_seed,
                        loader_seed=data_driven_loader_seed,
                    )
                )
                setattr(
                    args,
                    "stage_b_data_driven_sampling_ledger_epoch",
                    int(epoch),
                )
                setattr(
                    args,
                    "stage_b_data_driven_sampling_ledger_sha256",
                    current_data_driven_sampling_state["ledger_sha256"],
                )
                setattr(
                    args,
                    "stage_b_data_driven_sampling_ledger_num_samples",
                    current_data_driven_sampling_state["num_samples"],
                )
                logger.info(
                    "Stage-B data-driven sample-index ledger: "
                    + json.dumps(
                        current_data_driven_sampling_state,
                        sort_keys=True,
                    )
                )

            this_start_iter = resume_iter if epoch == args.start_epoch else 0
            if this_start_iter > 0 and resume_epoch_rng_state is not None:
                current_epoch_rng_state = resume_epoch_rng_state
            else:
                current_epoch_rng_state = _capture_rng_state()

            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer, device, epoch,
                args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler,
                args=args, logger=(logger if args.save_log else None), scaler=scaler,
                start_iter=this_start_iter,
                start_optimizer_updates=completed_optimizer_updates,
                epoch_rng_state=current_epoch_rng_state,
                runtime_rng_state=(resume_runtime_rng_state if this_start_iter > 0 else None),
                iter_checkpoint_fn=_save_iter_checkpoint)
            completed_optimizer_updates = int(
                train_stats.get("optimizer_updates", 0)
            )
            resume_iter = 0
            resume_epoch_rng_state = None
            resume_runtime_rng_state = None
            epoch_stop_requested = bool(
                getattr(args, "_stop_requested", False)
            )
            if utils.is_dist_avail_and_initialized():
                stop_flag = torch.as_tensor(
                    [1 if epoch_stop_requested else 0],
                    dtype=torch.int32,
                    device=device,
                )
                torch.distributed.all_reduce(
                    stop_flag, op=torch.distributed.ReduceOp.MAX
                )
                epoch_stop_requested = bool(int(stop_flag.item()) > 0)
            if epoch_stop_requested:
                args._stop_requested = True
            if not args.onecyclelr:
                lr_scheduler.step()
            if epoch_stop_requested:
                _save_iter_checkpoint(
                    epoch=epoch,
                    iteration=0,
                    optimizer_updates=completed_optimizer_updates,
                    scaler=scaler,
                    epoch_finished=True,
                    reason="signal_after_epoch",
                )
                return
            iter_interval = int(getattr(args, "iter_checkpoint_interval", 0) or 0)
            if (
                iter_interval > 0
                and completed_optimizer_updates > 0
                and (completed_optimizer_updates % iter_interval) == 0
            ):
                _save_iter_checkpoint(
                    epoch=epoch,
                    iteration=0,
                    optimizer_updates=completed_optimizer_updates,
                    scaler=scaler,
                    epoch_finished=True,
                    reason="interval_epoch",
                )
            if args.output_dir and _stage_b_data_driven_epoch_checkpoint_due(
                args, completed_optimizer_updates
            ):
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    weights = _checkpoint_payload(
                        epoch,
                        iteration=0,
                        optimizer_updates=completed_optimizer_updates,
                        epoch_finished=True,
                        reason="epoch",
                    )

                    utils.save_on_master(weights, checkpoint_path)
                
            if not patch_only and not skip_eval:
                # eval
                test_stats, coco_evaluator = evaluate(
                    model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                    wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
                )
                map_regular = test_stats['coco_eval_bbox'][0]
                _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
                if _isbest:
                    checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'criterion': criterion.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'scaler': scaler.state_dict(),
                        'epoch': epoch,
                        'iteration': 0,
                        'optimizer_updates': completed_optimizer_updates,
                        'epoch_finished': True,
                        'args': vars(args),
                    }, checkpoint_path)
                log_stats = {
                    **{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                }
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}}


            try:
                log_stats.update({'now_time': str(datetime.datetime.now())})
            except:
                pass

            epoch_time = time.time() - epoch_start_time
            epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
            log_stats['epoch_time'] = epoch_time_str

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if (not patch_only) and coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                       output_dir / "eval" / name)
    except GracefulTrainingExit as e:
        msg = str(e) or "Training stopped after writing iteration checkpoint."
        logger.info(msg) if args.save_log else print(msg)
        return
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # remove the copied files.
    copyfilelist = vars(args).get('copyfilelist')
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
