#!/usr/bin/env python3
"""Train one exposure-matched interleaved U2-v5 ownership row."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import util.misc as utils
from datasets import build_dataset
from engine import GracefulTrainingExit, train_one_epoch
from main import build_model_main, get_args_parser
from models.GroundingDINO.stage_b_gdino_score_adapter import StageBGDINOScoreAdapterCriterion
from models.GroundingDINO.stage_b_u0_patch_rank import stage_b_u0_tensor_state_sha256
from tools.stageb_u2v4_legacy_training_contract import (
    AUXILIARY_RESIDUAL_KEYS,
    SURFACE_PARAMETER_KEYS,
)
from tools.stageb_u2v5_ablation_registry import get_row
from util.slconfig import SLConfig


SCHEMA = "pivot.stageb.u2v5_ownership_checkpoint/v1"


class OwnershipError(RuntimeError):
    pass


class _NoopScheduler:
    def step(self) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {"schema": "noop"}


class _OwnershipCriterion(torch.nn.Module):
    def __init__(self, admission, confidence) -> None:
        super().__init__()
        self.admission = admission
        self.confidence = confidence
        self.weight_dict = {
            **dict(admission.weight_dict),
            **dict(confidence.weight_dict),
        }

    def forward(self, outputs, targets):
        if "stage_b_gdino_tn_outputs" in outputs:
            return self.confidence(outputs, targets)
        if "stage_b_u0_rank_score" in outputs:
            return self.admission(outputs, targets)
        raise OwnershipError("ownership criterion cannot identify task outputs")

    def commit_tail_queue(self, succeeded: bool) -> None:
        self.confidence.commit_tail_queue(succeeded)

    def defer_tail_queue_payload(self) -> None:
        self.confidence.defer_tail_queue_payload()


class _TaggedSchedule:
    def __init__(self, admission_loader, confidence_loader, *, admission_updates: int, confidence_updates: int) -> None:
        self.admission_loader = admission_loader
        self.confidence_loader = confidence_loader
        self.admission_updates = int(admission_updates)
        self.confidence_updates = int(confidence_updates)
        if self.admission_updates <= 0 or not 0 <= self.confidence_updates <= self.admission_updates:
            raise OwnershipError("invalid ownership exposure counts")

    def __len__(self) -> int:
        return self.admission_updates + self.confidence_updates

    @staticmethod
    def _tag(batch, task: str):
        samples, targets = batch
        tagged = []
        for target in targets:
            value = dict(target)
            value["u2v5_ownership_task"] = task
            if task == "confidence":
                value["table_b_id"] = "D3"
                value["table_b_audit_sha256"] = (
                    "7d74d541529a3e9abfbe84b192f2d0d3608d291bf46d19263c7c06a6ccb2291d"
                )
                value["tn_scope"] = "proposal_covered_verified"
            tagged.append(value)
        return samples, tagged

    @staticmethod
    def _next(iterator, loader):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    def __iter__(self):
        admission_iter = iter(self.admission_loader)
        confidence_iter = iter(self.confidence_loader)
        confidence_positions = {
            int(math.floor((index + 1) * self.admission_updates / self.confidence_updates))
            for index in range(self.confidence_updates)
        } if self.confidence_updates else set()
        for admission_index in range(self.admission_updates):
            batch, admission_iter = self._next(admission_iter, self.admission_loader)
            yield self._tag(batch, "admission")
            if (admission_index + 1) in confidence_positions:
                batch, confidence_iter = self._next(confidence_iter, self.confidence_loader)
                yield self._tag(batch, "confidence")


def _load_args(row, seed: int, output: Path) -> SimpleNamespace:
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args(
        [
            "--config_file", row.config,
            "--datasets", "config/datasets_stageb_u2_category_complete_three_ref.json",
            "--output_dir", str(output),
            "--pretrain_model_path", str(
                ROOT / "outputs/u2v5_leakage_clean_anchor_20260817/initializer/checkpoint_clean_init.pth"
            ),
            "--num_workers", "4", "--prefetch_factor", "1", "--amp",
            "--seed", str(seed), "--max_train_iters", str(
                int(os.environ.get("PIVOT_OWNERSHIP_ADMISSION_UPDATES", "100"))
                + int(os.environ.get("PIVOT_OWNERSHIP_CONFIDENCE_UPDATES", "50"))
            ),
            "--iter_checkpoint_interval", str(
                int(os.environ.get("PIVOT_OWNERSHIP_ADMISSION_UPDATES", "100"))
                + int(os.environ.get("PIVOT_OWNERSHIP_CONFIDENCE_UPDATES", "50"))
            ),
        ]
    )
    cfg = SLConfig.fromfile(args.config_file)
    for key, value in cfg._cfg_dict.to_dict().items():
        if not hasattr(args, key) or key in {"fix_size", "persistent_workers"}:
            setattr(args, key, value)
        else:
            raise OwnershipError(f"config/arg key collision: {key}")
    args.distributed = False
    args.rank = 0
    args.local_rank = 0
    args.world_size = 1
    args.gpu = 0
    args.device = "cuda"
    args.batch_size = 56
    args.gradient_accumulation_steps = 1
    args.stage_b_u2v5_ownership = True
    args.stage_b_gdino_tn_scope = "proposal_covered_verified"
    args.stage_b_gdino_confidence_objective = "detached_recent_q05_proposal_covered"
    args.stage_b_gdino_queue_size = 512
    args.stage_b_gdino_queue_min_count = 256
    args.stage_b_gdino_positive_trust_weight = 1.0
    args.stage_b_gdino_positive_trust_margin = 0.02
    args.stage_b_gdino_paired_margin_weight = 0.0
    args.stage_b_gdino_adapter_train_mode = "joint"
    args.stage_b_v19_table_b_id = "D3"
    args.stage_b_v19_table_b_audit_sha256 = "7d74d541529a3e9abfbe84b192f2d0d3608d291bf46d19263c7c06a6ccb2291d"
    args.stage_b_u2v5_clean_confidence = False
    args.stage_b_u2v4_legacy_training_replay = False
    args.skip_eval = True
    return args


def _datasets(path: Path, args) -> list:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [build_dataset(image_set="train", args=args, datasetinfo=entry) for entry in payload["train"]]


def _loader(datasets: list, *, batch_size: int, weighted_equal: bool, seed: int):
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    generator = torch.Generator().manual_seed(seed)
    if weighted_equal and len(datasets) > 1:
        weights = []
        for source in datasets:
            weights.extend([1.0 / len(source)] * len(source))
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(dataset), replacement=True, generator=generator,
        )
    else:
        sampler = RandomSampler(dataset, generator=generator)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        collate_fn=utils.collate_fn,
        num_workers=4,
        prefetch_factor=1,
        pin_memory=True,
        persistent_workers=True,
    )


def _confidence_criterion(args) -> StageBGDINOScoreAdapterCriterion:
    return StageBGDINOScoreAdapterCriterion(
        tn_scope=args.stage_b_gdino_tn_scope,
        train_mode="confidence_only",
        confidence_objective=args.stage_b_gdino_confidence_objective,
        positive_iou_threshold=float(args.stage_b_gdino_positive_iou_threshold),
        negative_iou_threshold=float(args.stage_b_gdino_negative_iou_threshold),
        listwise_temperature=float(args.stage_b_gdino_listwise_temperature),
        rank_fix_margin=float(args.stage_b_gdino_rank_fix_margin),
        rank_preserve_margin=float(args.stage_b_gdino_rank_preserve_margin),
        rank_residual_weight=float(args.stage_b_gdino_rank_residual_weight),
        rank_weight=0.0,
        confidence_weight=1.0,
        fpr_temperature=float(args.stage_b_gdino_fpr_temperature),
        fpr_margin=float(args.stage_b_gdino_fpr_margin),
        paired_margin_weight=0.0,
        paired_margin=float(args.stage_b_gdino_paired_margin),
        positive_trust_margin=float(args.stage_b_gdino_positive_trust_margin),
        positive_trust_weight=float(args.stage_b_gdino_positive_trust_weight),
        queue_size=int(args.stage_b_gdino_queue_size),
        queue_min_count=int(args.stage_b_gdino_queue_min_count),
    )


def _set_trainable(model, row_id: str) -> list[str]:
    named = dict(model.named_parameters())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    keys = list(SURFACE_PARAMETER_KEYS) + list(AUXILIARY_RESIDUAL_KEYS)
    adapter = model.stage_b_gdino_score_adapter
    rank_names = [name for name, parameter in named.items() if any(parameter is item for item in adapter.rank_parameters())]
    gate_names = [name for name, parameter in named.items() if any(parameter is item for item in adapter.gate_parameters())]
    confidence_gate_names = [name for name in gate_names if ".confidence_gate." in name]
    if row_id == "O0":
        keys += rank_names
    elif row_id == "O1":
        keys += rank_names + confidence_gate_names
    elif row_id == "O2":
        keys += gate_names
    else:
        raise OwnershipError(f"unsupported ownership row {row_id}")
    if len(keys) != len(set(keys)) or set(keys) - set(named):
        raise OwnershipError("ownership trainable key construction drifted")
    for name in keys:
        named[name].requires_grad_(True)
    return sorted(keys)


def _optimizer(model, trainable_keys: list[str], args):
    named = dict(model.named_parameters())
    groups = []
    definitions = (
        ("admission_auxiliary", AUXILIARY_RESIDUAL_KEYS, 3e-4),
        ("admission_surface", SURFACE_PARAMETER_KEYS, 3e-4),
        ("shared_rank", tuple(name for name in trainable_keys if ".rank_" in name), 3e-5),
        ("confidence", tuple(name for name in trainable_keys if ".confidence_" in name), 3e-4),
    )
    for branch, keys, lr in definitions:
        parameters = [named[name] for name in keys if name in trainable_keys]
        if parameters:
            groups.append({"params": parameters, "lr": lr, "u2v5_owner": branch})
    covered = {id(parameter) for group in groups for parameter in group["params"]}
    expected = {id(named[name]) for name in trainable_keys}
    if covered != expected:
        raise OwnershipError("ownership optimizer does not cover trainable keys")
    return torch.optim.AdamW(groups, lr=3e-4, weight_decay=float(args.weight_decay))


def _sanitize_args(args) -> dict[str, Any]:
    return {
        key: value for key, value in vars(args).items()
        if not key.startswith("_u2v5_") and not torch.is_tensor(value)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-id", required=True, choices=["O0", "O1", "O2"])
    parser.add_argument("--seed", required=True, type=int, choices=[17, 42, 73])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initializer", required=True)
    cli = parser.parse_args()
    row = get_row(cli.row_id)
    output = Path(cli.output_dir).resolve()
    if output.exists():
        raise OwnershipError(f"ownership output must be fresh: {output}")
    output.mkdir(parents=True)
    args = _load_args(row, cli.seed, output)
    admission_updates = int(os.environ.get("PIVOT_OWNERSHIP_ADMISSION_UPDATES", "100"))
    confidence_updates = int(os.environ.get("PIVOT_OWNERSHIP_CONFIDENCE_UPDATES", "50"))
    device = torch.device("cuda:0")
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    random.seed(cli.seed)
    model, admission_criterion, _ = build_model_main(args)
    initializer = Path(cli.initializer).resolve(strict=True)
    payload = torch.load(initializer, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    confidence_criterion = _confidence_criterion(args).to(device)
    criterion = _OwnershipCriterion(admission_criterion.to(device), confidence_criterion).to(device)
    trainable_keys = _set_trainable(model, row.row_id)
    optimizer = _optimizer(model, trainable_keys, args)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=8192.0)
    admission_sets = _datasets(ROOT / "config/datasets_stageb_u2_category_complete_three_ref.json", args)
    confidence_sets = _datasets(ROOT / "config/datasets_stageb_u2v5_clean_confidence_d3.json", args)
    schedule = _TaggedSchedule(
        _loader(admission_sets, batch_size=56, weighted_equal=True, seed=cli.seed),
        _loader(confidence_sets, batch_size=8, weighted_equal=False, seed=cli.seed + 1000),
        admission_updates=admission_updates,
        confidence_updates=confidence_updates,
    )
    frozen_keys = sorted(set(payload["model"]) - set(trainable_keys))

    def save_checkpoint(**kwargs) -> None:
        runtime = dict(getattr(args, "stage_b_u2v5_ownership_runtime_audit", {}))
        gradient = dict(getattr(args, "stage_b_u2v5_ownership_gradient_audit", {}))
        contract = {
            "schema": SCHEMA,
            "row": row.payload(),
            "trainable_keys": trainable_keys,
            "frozen_keys": frozen_keys,
            "frozen_tensor_sha256": stage_b_u0_tensor_state_sha256(
                model.state_dict(), frozen_keys
            ),
            "initializer_frozen_tensor_sha256": stage_b_u0_tensor_state_sha256(
                payload["model"], frozen_keys
            ),
            "exposure": {"admission": admission_updates, "confidence": confidence_updates},
            "runtime_audit": runtime,
            "gradient_audit": gradient,
            "c100_confidence_imported": False,
        }
        if contract["frozen_tensor_sha256"] != contract["initializer_frozen_tensor_sha256"]:
            raise OwnershipError("ownership training changed a frozen tensor")
        torch.save(
            {
                "model": model.state_dict(),
                "criterion": criterion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": _NoopScheduler().state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": 0,
                "iteration": int(kwargs.get("iteration", len(schedule))),
                "optimizer_updates": int(kwargs.get("optimizer_updates", len(schedule))),
                "epoch_finished": False,
                "args": _sanitize_args(args),
                "u2v5_ownership": contract,
                "checkpoint_reason": kwargs.get("reason", "max_train_iters"),
            },
            output / "checkpoint_iter.pth",
        )

    try:
        train_one_epoch(
            model, criterion, schedule, optimizer, device, 0,
            max_norm=0.1, lr_scheduler=_NoopScheduler(), args=args,
            logger=None, scaler=scaler, iter_checkpoint_fn=save_checkpoint,
        )
    except GracefulTrainingExit:
        pass
    checkpoint = output / "checkpoint_iter.pth"
    if not checkpoint.is_file():
        save_checkpoint(iteration=len(schedule), optimizer_updates=len(schedule), reason="schedule_complete")
    result = torch.load(checkpoint, map_location="cpu", weights_only=False)
    runtime = result["u2v5_ownership"]["runtime_audit"]
    if runtime.get("task_successful_steps") != {"admission": admission_updates, "confidence": confidence_updates}:
        raise OwnershipError("ownership exposure count drifted")
    (output / "ownership_receipt.json").write_text(
        json.dumps(result["u2v5_ownership"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if "--output-dir" in sys.argv:
            index = sys.argv.index("--output-dir") + 1
            if index < len(sys.argv):
                output = Path(sys.argv[index]).resolve()
                output.mkdir(parents=True, exist_ok=True)
                (output / "failure_traceback.txt").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
        raise
