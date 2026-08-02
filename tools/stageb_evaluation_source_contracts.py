#!/usr/bin/env python3
"""Dependency-light source contracts for Stage-B paper evaluation.

This module deliberately uses only the Python standard library.  Validation
and matrix queues can therefore seal source identities without importing the
headline release machinery or any training stack.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE_FAMILY_TOKEN = "token"
SOURCE_FAMILY_PAPER = "paper"
SOURCE_FAMILY_HISTORICAL_BASELINE = "historical_baseline"
SOURCE_FAMILIES = (
    SOURCE_FAMILY_TOKEN,
    SOURCE_FAMILY_PAPER,
    SOURCE_FAMILY_HISTORICAL_BASELINE,
)

BASELINE_ID = "gdino_stageb_data_ft_b58"
BASELINE_TRAIN_SEED = 42
_BASELINE_TRAIN_ROOT = Path(
    "/media/haoyi/T9/gdino/outputs/"
    "gdino_ft_stageb_from_gdino_ft_e1_with_tn_bs19_nopatchbranch"
)
FIXED_BASELINE: Mapping[str, Any] = {
    "id": BASELINE_ID,
    "train_seed": BASELINE_TRAIN_SEED,
    "role": "fixed_historical_checkpoint",
    "config": str((_BASELINE_TRAIN_ROOT / "config_cfg.py").resolve(strict=False)),
    "config_sha256": (
        "f0dc568c6f35225176712618d5f3449b253478ef32a2b65fa9e089da1ad8a05f"
    ),
    "checkpoint": str(
        (_BASELINE_TRAIN_ROOT / "checkpoint0001.pth").resolve(strict=False)
    ),
    "checkpoint_sha256": (
        "b58e5209dc07dbffb2e5ed3d792e0db70c3306ec2ed24389693b4aeeebab1157"
    ),
}

DEFAULT_PYTHON = Path("/home/haoyi/miniconda/envs/gdino5090/bin/python")
DEFAULT_DATA_ROOT = Path("/media/haoyi/T9/data")
CANONICAL_RUNTIME: Mapping[str, Any] = {
    "python": str(DEFAULT_PYTHON.resolve(strict=False)),
    "data_root": str(DEFAULT_DATA_ROOT.resolve(strict=False)),
    "device": "cuda:0",
    "batch_size": 16,
    "num_workers": 4,
    "amp": True,
    "log_every": 50,
    "eval_seed": 42,
    "max_ref_batches": 0,
    "max_tn_batches": 0,
}

VALIDATION_ROOT = REPO_ROOT / "outputs/paper_cvpr_v1/evaluations/headline_selection"
FORMAL_TRAINING_ROOT = (
    REPO_ROOT / "outputs/paper_cvpr_v1/headline_main_compute_matched"
)
FORMAL_TRAINING_SEEDS = (17, 42, 73)


class EvaluationSourceContractError(RuntimeError):
    """Raised when a source differs from its predeclared evaluation contract."""


@dataclass(frozen=True)
class FormalPaperRunContract:
    """Immutable formal-training semantics needed by the evaluator."""

    id: str
    config: str
    runner: str
    output_root: str
    dataset: str
    architecture_objective: str
    compute_contract: str
    phase_ids: tuple[str, ...]
    batch_size: int
    optimizer_updates: int
    contributing_phase_updates: tuple[tuple[str, int], ...]
    successful_update_batch_slots: int
    final_phase_updates: int
    iter_checkpoint_interval: int
    seeds: tuple[int, ...]
    token_objective: str | None = None
    token_objective_scope: str | None = None
    predicate_pair_rank_weight: float | None = None
    matrix_validation_only: bool = False
    headline: bool = False

    def canonical_training_root(self, seed: int) -> Path:
        if seed not in self.seeds:
            raise EvaluationSourceContractError(
                f"unexpected {self.id} training seed {seed}"
            )
        return (
            REPO_ROOT / self.output_root / self.id / f"seed{seed}"
        ).resolve(strict=False)

    def expected_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "row_id": self.id,
            "config": self.config,
            "architecture_objective": self.architecture_objective,
            "compute_contract": self.compute_contract,
            "datasets": self.dataset,
        }
        if self.token_objective is not None:
            row["token_objective"] = self.token_objective
        if self.token_objective_scope is not None:
            row["token_objective_scope"] = self.token_objective_scope
        if self.predicate_pair_rank_weight is not None:
            row["predicate_pair_rank_weight"] = self.predicate_pair_rank_weight
        return row

    def expected_budget(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "optimizer_updates": self.optimizer_updates,
            "contributing_phase_updates": dict(self.contributing_phase_updates),
            "successful_update_batch_slots": self.successful_update_batch_slots,
        }

    @property
    def dedicated_queue_order(self) -> tuple[int, ...]:
        return self.seeds

    @property
    def dedicated_queue_run_ids(self) -> tuple[str, ...]:
        return tuple(f"{self.id}:{seed}" for seed in self.dedicated_queue_order)


M0_CONTRACT = FormalPaperRunContract(
    id="M0",
    config="config/ablations/cfg_stageb_v25_m0_compute_matched.py",
    runner="tools/run_stageb_headline_m0.py",
    output_root="outputs/paper_cvpr_v1/headline_main_compute_matched",
    dataset="config/datasets_stageb_v21_single_edit_train.json",
    architecture_objective="S2F",
    compute_contract="b58_successful_update_batch_slot_matched",
    phase_ids=("joint",),
    batch_size=40,
    optimizer_updates=23532,
    contributing_phase_updates=(("joint", 23532),),
    successful_update_batch_slots=941280,
    final_phase_updates=23532,
    iter_checkpoint_interval=500,
    seeds=FORMAL_TRAINING_SEEDS,
    matrix_validation_only=False,
    headline=True,
)

M0N_CONTRACT = FormalPaperRunContract(
    id="M0N",
    config="config/ablations/cfg_stageb_v25_m0n_compute_matched_allneg_bce.py",
    runner="tools/run_stageb_headline_m0.py",
    output_root="outputs/paper_cvpr_v1/headline_main_compute_matched",
    dataset="config/datasets_stageb_v21_single_edit_train.json",
    architecture_objective="S2F",
    compute_contract="b58_successful_update_batch_slot_matched",
    phase_ids=("joint",),
    batch_size=40,
    optimizer_updates=23532,
    contributing_phase_updates=(("joint", 23532),),
    successful_update_batch_slots=941280,
    final_phase_updates=23532,
    iter_checkpoint_interval=500,
    seeds=FORMAL_TRAINING_SEEDS,
    token_objective="targetlocal_allneg_bce",
    token_objective_scope="target_local_positive_and_all_negative_token_logits",
    predicate_pair_rank_weight=1.0,
    matrix_validation_only=True,
    headline=False,
)

FORMAL_PAPER_RUN_CONTRACTS: Mapping[str, FormalPaperRunContract] = {
    contract.id: contract for contract in (M0_CONTRACT, M0N_CONTRACT)
}


def formal_paper_run_contract(
    contract_id: str,
) -> FormalPaperRunContract | None:
    return FORMAL_PAPER_RUN_CONTRACTS.get(contract_id)


def source_family_for_kind(kind: str) -> str:
    if kind == "pivot_token_ablation_training_run":
        return SOURCE_FAMILY_TOKEN
    if kind in {
        "pivot_paper_training_run",
        "pivot_paper_training_run_rank_diagnostic",
    }:
        return SOURCE_FAMILY_PAPER
    if kind == "historical_pure_gdino_explicit":
        return SOURCE_FAMILY_HISTORICAL_BASELINE
    raise EvaluationSourceContractError(f"unknown evaluation source kind: {kind!r}")


def sha256_file(path: Path) -> str:
    path = path.expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    if not path.is_file():
        raise EvaluationSourceContractError(f"source artifact is not a file: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def canonical_validation_root(role: str, seed: int | None = None) -> Path:
    if role == "baseline":
        if seed not in (None, BASELINE_TRAIN_SEED):
            raise EvaluationSourceContractError(
                "fixed b58 validation has only seed42"
            )
        return VALIDATION_ROOT / BASELINE_ID / "fixed"
    contract = M0_CONTRACT
    if role == "candidate" and seed in contract.seeds:
        return VALIDATION_ROOT / contract.id / f"seed{int(seed)}"
    raise EvaluationSourceContractError("unknown headline validation instance")


def validate_fixed_baseline_identity(
    *,
    evaluation_id: str,
    config: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    config = config.expanduser().resolve(strict=True)
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    expected = FIXED_BASELINE
    if (
        evaluation_id != expected["id"]
        or config != Path(str(expected["config"])).resolve(strict=True)
        or checkpoint != Path(str(expected["checkpoint"])).resolve(strict=True)
        or checkpoint_sha256 != expected["checkpoint_sha256"]
        or sha256_file(config) != expected["config_sha256"]
        or sha256_file(checkpoint) != expected["checkpoint_sha256"]
    ):
        raise EvaluationSourceContractError(
            "source is not the fixed b58 historical checkpoint/config identity"
        )
    return {
        "id": expected["id"],
        "train_seed": int(expected["train_seed"]),
        "role": expected["role"],
        "config": file_record(config),
        "checkpoint": file_record(checkpoint),
    }


def runtime_projection(runtime: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in CANONICAL_RUNTIME if key not in runtime]
    if missing:
        raise EvaluationSourceContractError(
            f"evaluation runtime lacks fields {missing}"
        )
    projection = {key: runtime[key] for key in CANONICAL_RUNTIME}
    projection["python"] = str(
        Path(str(projection["python"])).expanduser().resolve(strict=False)
    )
    projection["data_root"] = str(
        Path(str(projection["data_root"])).expanduser().resolve(strict=False)
    )
    return projection


def validate_canonical_runtime(runtime: Mapping[str, Any]) -> dict[str, Any]:
    projection = runtime_projection(runtime)
    expected = dict(CANONICAL_RUNTIME)
    expected["python"] = str(Path(str(expected["python"])).resolve(strict=True))
    expected["data_root"] = str(
        Path(str(expected["data_root"])).resolve(strict=True)
    )
    if projection != expected:
        raise EvaluationSourceContractError(
            f"headline evaluation runtime drifted: expected {expected}, got {projection}"
        )
    return projection
